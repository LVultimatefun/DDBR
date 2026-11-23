"""Camera-role-adaptive PromptBuilder with S2 backbone & Few-Shot injection."""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from ..config import ConfigManager
from ..io.media import select_tile_mode
from ..models.annotation import MacroStage, QcSegment
from ..models.taxonomy import FewShotItem, LabelRegistry, PromptBundle, TaskSpec

_IF_BLOCK_RE = re.compile(
    r"\{%\s*if\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*%\}(.*?)\{%\s*endif\s*%\}",
    re.DOTALL,
)
_VAR_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


def render_template(template: str, ctx: dict[str, Any]) -> str:
    """Render minimal Jinja2 subset (`{% if var %}...{% endif %}` and `{{ var }}`)."""

    def _repl_if(match: re.Match[str]) -> str:
        key = match.group(1)
        body = match.group(2)
        val = ctx.get(key)
        return body if val else ""

    rendered = _IF_BLOCK_RE.sub(_repl_if, template)

    def _repl_var(match: re.Match[str]) -> str:
        key = match.group(1)
        val = ctx.get(key, "")
        return "" if val is None else str(val)

    rendered = _VAR_RE.sub(_repl_var, rendered)
    return rendered.strip()


def describe_camera_layout(camera_roles: dict[str, str] | None) -> str:
    """Return camera-role-adaptive 2x2 / 1x2 layout description for VLM prompts."""
    roles = camera_roles or {}
    mode = select_tile_mode(roles)
    if mode == "wrist_2x2":
        return (
            "2x2 grid composite (top TIME bar): "
            "top-left L-WRIST (left wrist closeup), top-right R-WRIST (right wrist closeup), "
            "bottom-left MAIN (main camera overview), bottom-right MAIN-ZOOM (zoomed workspace crop)."
        )
    if mode == "torso_head_2x2":
        top_role = (
            "TORSO (front view) + TORSO-ZOOM"
            if "front" in roles
            else "WIDE (wide-angle head view) + WIDE-ZOOM"
            if ("wide" in roles or "wide_right" in roles)
            else "AUX (auxiliary view) + AUX-ZOOM"
        )
        return (
            f"2x2 grid composite (top TIME bar): top row shows {top_role}; "
            "bottom row shows HEAD (main overview) and HEAD-ZOOM (zoomed workspace crop)."
        )
    return (
        "1x2 single-camera composite (top TIME bar): "
        "left shows MAIN (main camera overview), right shows MAIN-CENTER-ZOOM (zoomed central workspace crop)."
    )


build_camera_layout_desc = describe_camera_layout


def format_objects_desc(task: TaskSpec) -> str:
    if not task.grounding_objects:
        return (
            "- target_object: primary manipulated objects or containers in the scene (open-vocabulary, 1-4 key objects)"
        )
    lines = []
    for obj in task.grounding_objects:
        desc = obj.desc.strip() or obj.name
        static_tag = " [static container]" if obj.is_static else ""
        lines.append(f"- {obj.id}: {obj.name} ({desc}{static_tag})")
    return "\n".join(lines)


def format_macro_backbone(macro_stages: Sequence[MacroStage]) -> str:
    if not macro_stages:
        return "(No macro stage backbone provided; evaluate strictly using Precision-First criteria.)"
    lines = []
    for st in macro_stages:
        summary = f" -- {st.summary}" if st.summary else ""
        lines.append(
            f"- {st.stage_id} [{st.t_start:.2f}s..{st.t_end:.2f}s] {st.label}{summary}"
        )
    return "\n".join(lines)


def format_labels_desc(labels: LabelRegistry) -> str:
    inactive_ids = {lbl.id for lbl in labels.labels if lbl.status != "active"}

    def _sanitize_inactive(text: str) -> str:
        out = text
        for bad_id in inactive_ids:
            out = re.sub(rf"\(\s*{re.escape(bad_id)}\s*\)", "", out)
            out = re.sub(rf"\b{re.escape(bad_id)}\b", "", out)
        return out.strip()

    lines: list[str] = []
    for lbl in labels.labels:
        if lbl.status != "active":
            continue
        cues = "; ".join(_sanitize_inactive(c) for c in lbl.vlm_semantics.visual_cues[:2])
        counter = "; ".join(
            _sanitize_inactive(c) for c in lbl.vlm_semantics.counter_examples[:2]
        )
        line = (
            f"- {lbl.id} ({lbl.name} | category={lbl.category} | "
            f"default_role={lbl.default_training_role} | default_policy={lbl.default_splice_policy}): "
            f"{_sanitize_inactive(lbl.vlm_semantics.definition)}"
        )
        if cues:
            line += f" Visual cues: {cues}."
        if counter:
            line += f" Counter-examples: {counter}."
        lines.append(line)
    return "\n".join(lines)


def _estimate_tokens(text: str) -> int:
    # Conservative token estimate (~2 chars per token for dense JSON/text)
    return max(1, (len(text) + 1) // 2)


def _select_few_shots(
    few_shots: Sequence[FewShotItem] | None,
    *,
    task_id: str | None = None,
    top_k: int = 4,
    max_tokens: int = 4000,
) -> list[FewShotItem]:
    if not few_shots or top_k <= 0 or max_tokens <= 0:
        return []
    enabled = [fs for fs in few_shots if fs.enabled]

    def _priority(fs: FewShotItem) -> tuple[int, float, str]:
        if task_id and (fs.scope == f"task:{task_id}" or fs.task_id == task_id):
            tier = 0
        elif fs.scope.startswith("label:"):
            tier = 1
        elif fs.scope == "global":
            tier = 2
        else:
            tier = 3
        return (tier, -float(fs.created_at), fs.example_id)

    ordered = sorted(enabled, key=_priority)
    selected: list[FewShotItem] = []
    used_tokens = 0
    for fs in ordered:
        entry = _format_single_few_shot(fs)
        tok = _estimate_tokens(entry)
        if used_tokens + tok > max_tokens:
            continue
        selected.append(fs)
        used_tokens += tok
        if len(selected) >= top_k:
            break
    return selected


def _format_single_few_shot(fs: FewShotItem) -> str:
    wrong_s = json.dumps(fs.vlm_wrong_json, ensure_ascii=False)
    right_s = json.dumps(fs.human_right_json, ensure_ascii=False)
    return (
        f"- [{fs.error_type}] ({fs.episode_id} {fs.t_start:.2f}s-{fs.t_end:.2f}s): "
        f"wrong={wrong_s} -> correct={right_s}; rationale: {fs.rationale}"
    )


def format_few_shots_text(
    few_shots: Sequence[FewShotItem] | None,
    *,
    task_id: str | None = None,
    top_k: int = 4,
    max_tokens: int = 4000,
) -> str:
    chosen = _select_few_shots(
        few_shots, task_id=task_id, top_k=top_k, max_tokens=max_tokens
    )
    return "\n".join(_format_single_few_shot(fs) for fs in chosen)


class PromptBuilder:
    """Renders stage-specific system/user prompts from ConfigManager."""

    def __init__(
        self,
        config: ConfigManager | None = None,
        *,
        cfg: ConfigManager | None = None,
        prompts: PromptBundle | None = None,
        labels: LabelRegistry | None = None,
        settings: dict[str, Any] | None = None,
    ) -> None:
        cm = config or cfg
        if cm is None and (prompts is None or labels is None):
            raise ValueError("PromptBuilder requires ConfigManager or prompts+labels")
        self._cm = cm
        self._prompts_override = prompts
        self._labels_override = labels
        self._settings_override = settings

    @property
    def prompts(self) -> PromptBundle:
        if self._prompts_override is not None:
            return self._prompts_override
        assert self._cm is not None
        return self._cm.prompts

    @prompts.setter
    def prompts(self, value: PromptBundle) -> None:
        self._prompts_override = value

    @property
    def labels(self) -> LabelRegistry:
        if self._labels_override is not None:
            return self._labels_override
        assert self._cm is not None
        return self._cm.labels

    @labels.setter
    def labels(self, value: LabelRegistry) -> None:
        self._labels_override = value

    @property
    def settings(self) -> dict[str, Any]:
        if self._settings_override is not None:
            return self._settings_override
        if self._cm is not None:
            return self._cm.settings
        return {}

    @settings.setter
    def settings(self, value: dict[str, Any]) -> None:
        self._settings_override = value

    def build_s1(
        self,
        *,
        task: TaskSpec | None = None,
        task_spec: TaskSpec | None = None,
        camera_roles: dict[str, str] | None = None,
    ) -> tuple[str, str]:
        t = task or task_spec
        assert t is not None
        tpl = self.prompts.s1_grounding
        main_cam = (camera_roles or {}).get("main", "main")
        ctx = {
            "task_id": t.task_id,
            "task_goal": t.goal or t.description or t.title or t.task_id,
            "camera_layout_desc": f"Main HD camera frames (camera={main_cam})",
            "objects_desc": format_objects_desc(t),
            "task_prompt": t.task_prompt.strip() or "Ground all visibly active target objects and containers.",
        }
        return (
            render_template(tpl.system_template, ctx),
            render_template(tpl.user_template, ctx),
        )

    def build_s2(
        self,
        *,
        task: TaskSpec | None = None,
        task_spec: TaskSpec | None = None,
        camera_roles: dict[str, str] | None = None,
        duration_s: float = 0.0,
    ) -> tuple[str, str]:
        t = task or task_spec
        assert t is not None
        tpl = self.prompts.s2_macro
        steps = "\n".join(f"- {s}" for s in t.steps) if t.steps else "- Segment stages according to observed task progression"
        ctx = {
            "task_id": t.task_id,
            "duration_s": round(float(duration_s), 2),
            "task_goal": t.goal or t.description or t.title or t.task_id,
            "expected_steps": steps,
            "camera_layout_desc": describe_camera_layout(camera_roles),
            "task_prompt": t.task_prompt.strip() or "Segment macro stages and dual-arm timelines strictly from visual evidence.",
        }
        return (
            render_template(tpl.system_template, ctx),
            render_template(tpl.user_template, ctx),
        )

    def build_s3(
        self,
        *,
        task: TaskSpec | None = None,
        task_spec: TaskSpec | None = None,
        camera_roles: dict[str, str] | None = None,
        macro_stages: Sequence[MacroStage] = (),
        window_start_s: float = 0.0,
        window_end_s: float = 0.0,
        duration_s: float = 0.0,
        few_shots: Sequence[FewShotItem] | None = None,
        few_shot_items: Sequence[FewShotItem] | None = None,
    ) -> tuple[str, str]:
        t = task or task_spec
        assert t is not None
        tpl = self.prompts.s3_discover
        s3_cfg = self.settings.get("vlm", {}).get("s3_discover", {})
        top_k = int(s3_cfg.get("few_shot_top_k", 4))
        max_tok = int(s3_cfg.get("few_shot_max_tokens", 4000))
        fs_list = few_shots if few_shots is not None else few_shot_items
        fs_text = format_few_shots_text(
            fs_list, task_id=t.task_id, top_k=top_k, max_tokens=max_tok
        )
        ctx = {
            "task_id": t.task_id,
            "window_start_s": round(float(window_start_s), 2),
            "window_end_s": round(float(window_end_s), 2),
            "duration_s": round(float(duration_s), 2),
            "camera_layout_desc": describe_camera_layout(camera_roles),
            "macro_backbone_text": format_macro_backbone(macro_stages),
            "labels_desc": format_labels_desc(self.labels),
            "few_shot_text": fs_text,
        }
        return (
            render_template(tpl.system_template, ctx),
            render_template(tpl.user_template, ctx),
        )

    def build_s4a(
        self,
        *,
        task: TaskSpec | None = None,
        task_spec: TaskSpec | None = None,
        duration_s: float = 0.0,
        macro_stages: Sequence[MacroStage] = (),
        raw_candidates: Sequence[dict[str, Any] | QcSegment] = (),
    ) -> tuple[str, str]:
        t = task or task_spec
        assert t is not None
        tpl = self.prompts.s4_verify
        cands_list: list[dict[str, Any]] = []
        for c in raw_candidates:
            if isinstance(c, QcSegment):
                cands_list.append(
                    {
                        "seg_id": c.seg_id,
                        "t_start": round(c.t_start, 3),
                        "t_end": round(c.t_end, 3),
                        "label_id": c.label_id,
                        "affected_arm": c.affected_arm,
                        "splice_policy": c.splice_policy,
                        "visual_evidence": c.visual_evidence.model_dump(),
                        "confidence": c.confidence,
                        "unknown_note": c.unknown_note,
                    }
                )
            else:
                cands_list.append(dict(c))
        ctx = {
            "task_id": t.task_id,
            "duration_s": round(float(duration_s), 2),
            "macro_backbone_text": format_macro_backbone(macro_stages),
            "raw_candidates_json": json.dumps(cands_list, ensure_ascii=False),
        }
        sys_t = tpl.extra_templates.get("merge_system") or tpl.system_template
        usr_t = tpl.extra_templates.get("merge_user") or tpl.user_template
        return render_template(sys_t, ctx), render_template(usr_t, ctx)

    def build_s4b(
        self,
        *,
        task: TaskSpec | None = None,
        task_spec: TaskSpec | None = None,
        camera_roles: dict[str, str] | None = None,
        macro_stages: Sequence[MacroStage] = (),
        candidate: QcSegment,
        window_start_s: float,
        window_end_s: float,
        few_shots: Sequence[FewShotItem] | None = None,
    ) -> tuple[str, str]:
        t = task or task_spec
        assert t is not None
        tpl = self.prompts.s4_verify
        s3_cfg = self.settings.get("vlm", {}).get("s3_discover", {})
        top_k = int(s3_cfg.get("few_shot_top_k", 4))
        max_tok = int(s3_cfg.get("few_shot_max_tokens", 4000))
        fs_text = format_few_shots_text(
            few_shots, task_id=t.task_id, top_k=top_k, max_tokens=max_tok
        )
        ctx = {
            "task_id": t.task_id,
            "seg_id": candidate.seg_id,
            "t_start": round(float(candidate.t_start), 2),
            "t_end": round(float(candidate.t_end), 2),
            "label_id": candidate.label_id,
            "window_start_s": round(float(window_start_s), 2),
            "window_end_s": round(float(window_end_s), 2),
            "camera_layout_desc": describe_camera_layout(camera_roles),
            "macro_backbone_text": format_macro_backbone(macro_stages),
            "few_shot_text": fs_text,
        }
        return (
            render_template(tpl.system_template, ctx),
            render_template(tpl.user_template, ctx),
        )

    def build_s5(
        self,
        *,
        task: TaskSpec | None = None,
        task_spec: TaskSpec | None = None,
        camera_roles: dict[str, str] | None = None,
        duration_s: float = 0.0,
    ) -> tuple[str, str]:
        t = task or task_spec
        assert t is not None
        tpl = self.prompts.s5_success
        criteria = (
            "\n".join(f"- {c}" for c in t.criteria)
            if t.criteria
            else "- Target object reaches the intended goal state or placement."
        )
        ctx = {
            "task_id": t.task_id,
            "duration_s": round(float(duration_s), 2),
            "task_goal": t.goal or t.description or t.title or t.task_id,
            "criteria_list": criteria,
            "camera_layout_desc": describe_camera_layout(camera_roles),
            "task_prompt": t.task_prompt.strip() or "Verify each completion criterion strictly from visual evidence.",
        }
        return (
            render_template(tpl.system_template, ctx),
            render_template(tpl.user_template, ctx),
        )

    def build_s5_control(
        self,
        *,
        task: TaskSpec | None = None,
        task_spec: TaskSpec | None = None,
        cut_s: float = 0.0,
    ) -> tuple[str, str]:
        t = task or task_spec
        assert t is not None
        tpl = self.prompts.s5_success
        criteria = (
            "\n".join(f"- {c}" for c in t.criteria)
            if t.criteria
            else "- Target object reaches the intended goal state or placement."
        )
        ctx = {
            "task_id": t.task_id,
            "cut_s": round(float(cut_s), 2),
            "criteria_list": criteria,
        }
        sys_t = tpl.extra_templates.get("control_system") or tpl.system_template
        usr_t = tpl.extra_templates.get("control_user") or tpl.user_template
        return render_template(sys_t, ctx), render_template(usr_t, ctx)

    render_s1 = build_s1
    render_s2 = build_s2
    render_s3 = build_s3
    render_s4a = build_s4a
    render_s4b = build_s4b
    render_s5 = build_s5
    render_s5_control = build_s5_control
