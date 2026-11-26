"""Camera-role-adaptive PromptBuilder with S2 backbone & Few-Shot injection."""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

import numpy as np

from ..config import ConfigManager
from ..io.base import KinematicsTrack
from ..io.media import select_tile_mode
from ..models.annotation import GroundingResult, MacroStage, QcSegment
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
        instr = f" | instruction: {st.instruction.strip()}" if getattr(st, "instruction", "").strip() else ""
        summary = f" -- {st.summary}" if st.summary else ""
        lines.append(
            f"- {st.stage_id} [{st.t_start:.2f}s..{st.t_end:.2f}s] {st.label}{instr}{summary}"
        )
    return "\n".join(lines)


def format_grounding_summary(
    grounding: GroundingResult | None,
    *,
    max_chars: int = 1600,
    max_transitions_per_obj: int = 8,
) -> str:
    """Format a compact S1 2D object state & motion transition summary for S2 / subtask inference."""
    if grounding is None or not grounding.object_tracks_2d:
        return ""

    lines: list[str] = []
    for oid, track in grounding.object_tracks_2d.items():
        kfs = sorted(track.keyframes, key=lambda k: k.time_s)
        if not kfs:
            continue

        # Compress consecutive keyframes sharing the same (visible, normalized state)
        runs: list[dict[str, Any]] = []
        for kf in kfs:
            st = (kf.state or ("visible" if kf.visible else "occluded")).strip()
            cx = 0.5 * (kf.bbox_norm_1000[0] + kf.bbox_norm_1000[2]) if kf.visible else None
            cy = 0.5 * (kf.bbox_norm_1000[1] + kf.bbox_norm_1000[3]) if kf.visible else None
            key = (bool(kf.visible), st.lower())
            if runs and runs[-1]["key"] == key:
                runs[-1]["t_end"] = kf.time_s
                if cx is not None and cy is not None:
                    runs[-1]["end_xy"] = (cx, cy)
            else:
                runs.append(
                    {
                        "key": key,
                        "visible": bool(kf.visible),
                        "state": st,
                        "t_start": kf.time_s,
                        "t_end": kf.time_s,
                        "start_xy": (cx, cy) if cx is not None and cy is not None else None,
                        "end_xy": (cx, cy) if cx is not None and cy is not None else None,
                    }
                )

        # Keep visible spans plus any meaningful occlusion transitions between visible spans
        parts: list[str] = []
        for r_idx, r in enumerate(runs):
            if not r["visible"]:
                has_vis_before = any(prev["visible"] for prev in runs[:r_idx])
                has_vis_after = any(nxt["visible"] for nxt in runs[r_idx + 1 :])
                if not (has_vis_before and has_vis_after):
                    continue
                t_span = (
                    f"{r['t_start']:.2f}s"
                    if abs(r["t_end"] - r["t_start"]) < 1e-3
                    else f"{r['t_start']:.2f}..{r['t_end']:.2f}s"
                )
                parts.append(f"{t_span} [not visible]")
                continue

            t_span = (
                f"{r['t_start']:.2f}s"
                if abs(r["t_end"] - r["t_start"]) < 1e-3
                else f"{r['t_start']:.2f}..{r['t_end']:.2f}s"
            )
            motion_note = ""
            if not track.is_static_container and r["start_xy"] and r["end_xy"]:
                dx = r["end_xy"][0] - r["start_xy"][0]
                dy = r["end_xy"][1] - r["start_xy"][1]
                if (dx * dx + dy * dy) ** 0.5 >= 45.0:
                    motion_note = f", moved dx={dx:+.0f}, dy={dy:+.0f}"
            parts.append(f"{t_span} [{r['state']}{motion_note}]")

        if not parts:
            continue
        if len(parts) > max_transitions_per_obj:
            half = max(1, max_transitions_per_obj // 2)
            parts = parts[:half] + ["..."] + parts[-(max_transitions_per_obj - half) :]
        disp_name = track.name.strip() if track.name and track.name.strip() != oid else ""
        header = f"{oid} ({disp_name})" if disp_name else oid
        if track.is_static_container:
            header += " [static]"
        lines.append(f"- {header}: " + " -> ".join(parts))

    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[: max(0, max_chars - 3)].rstrip() + "..."
    return text


def format_qc_segments_summary(
    qc_segments: Sequence[QcSegment],
    duration_s: float,
) -> str:
    """Format verified QC segments highlighting cut_clean exclusions, recovery spans, and kept intervals."""
    dur = max(0.01, float(duration_s))
    ordered = sorted(qc_segments, key=lambda s: (s.t_start, s.t_end))
    cut_intervals: list[tuple[float, float]] = []
    seg_lines: list[str] = []

    for seg in ordered:
        ts = max(0.0, min(dur, float(seg.t_start)))
        te = max(ts, min(dur, float(seg.t_end)))
        ev_txt = seg.visual_evidence.as_text().strip()
        ev_suffix = f" | evidence: {ev_txt[:140]}" if ev_txt else ""
        if seg.splice_policy == "cut_clean":
            cut_intervals.append((ts, te))
            seg_lines.append(
                f"- [{ts:.2f}s..{te:.2f}s] CUT_CLEAN EXCLUDED ({seg.label_id}, role={seg.training_role}): "
                f"excluded idle/cut span; align subtask boundaries to this cut{ev_suffix}"
            )
        elif seg.training_role == "recovery":
            seg_lines.append(
                f"- [{ts:.2f}s..{te:.2f}s] RECOVERY ({seg.label_id}, policy={seg.splice_policy}, side={seg.affected_arm}): "
                f"retry/recovery span{ev_suffix}"
            )
        elif seg.splice_policy == "mask_loss":
            seg_lines.append(
                f"- [{ts:.2f}s..{te:.2f}s] MASK_LOSS ({seg.label_id}, role={seg.training_role}, side={seg.affected_arm}){ev_suffix}"
            )
        else:
            seg_lines.append(
                f"- [{ts:.2f}s..{te:.2f}s] KEEP_POSITIVE ({seg.label_id}, role={seg.training_role}){ev_suffix}"
            )

    # Merge cut_clean spans to compute retained active execution intervals
    merged_cuts: list[list[float]] = []
    for cs, ce in sorted(cut_intervals):
        if ce <= cs + 1e-4:
            continue
        if merged_cuts and cs <= merged_cuts[-1][1] + 1e-3:
            merged_cuts[-1][1] = max(merged_cuts[-1][1], ce)
        else:
            merged_cuts.append([cs, ce])

    kept: list[tuple[float, float]] = []
    cursor = 0.0
    for cs, ce in merged_cuts:
        if cs > cursor + 1e-3:
            kept.append((cursor, cs))
        cursor = max(cursor, ce)
    if cursor < dur - 1e-3:
        kept.append((cursor, dur))

    parts: list[str] = []
    if kept:
        kept_str = ", ".join(f"[{a:.2f}s..{b:.2f}s]" for a, b in kept)
        parts.append(f"Active retained execution spans (outside cut_clean): {kept_str}")
    if seg_lines:
        parts.append("Verified QC segments:")
        parts.extend(seg_lines)
    else:
        parts.append(f"No QC anomalies or cuts flagged; full episode [0.00s..{dur:.2f}s] is retained.")
    return "\n".join(parts)


def _sanitize_inactive_labels(text: str, labels: LabelRegistry) -> str:
    inactive_ids = {lbl.id for lbl in labels.labels if lbl.status != "active"}
    if not inactive_ids:
        return text
    out = text
    for bad_id in inactive_ids:
        out = re.sub(rf"\(\s*{re.escape(bad_id)}\s*\)", "", out)
        out = re.sub(rf"\b{re.escape(bad_id)}\b", "", out)
    return out.strip()


def format_labels_desc(labels: LabelRegistry) -> str:
    lines: list[str] = []
    for lbl in labels.labels:
        if lbl.status != "active":
            continue
        cues = "; ".join(
            _sanitize_inactive_labels(c, labels) for c in lbl.vlm_semantics.visual_cues[:2]
        )
        counter = "; ".join(
            _sanitize_inactive_labels(c, labels)
            for c in lbl.vlm_semantics.counter_examples[:2]
        )
        line = (
            f"- {lbl.id} ({lbl.name} | category={lbl.category} | "
            f"default_role={lbl.default_training_role} | default_policy={lbl.default_splice_policy}): "
            f"{_sanitize_inactive_labels(lbl.vlm_semantics.definition, labels)}"
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


def _compute_smoothed_velocities(
    ts: np.ndarray,
    pts: np.ndarray,
    *,
    smooth_s: float = 0.3,
) -> np.ndarray:
    """Compute ~0.3s time-averaged 3D velocity (m/s) per valid sample."""
    n = len(ts)
    if n < 2:
        return np.zeros((n, 3), dtype=np.float64)
    half_s = 0.5 * max(0.05, float(smooth_s))
    lo_idx = np.clip(np.searchsorted(ts, ts - half_s, side="left"), 0, n - 1)
    hi_idx = np.clip(np.searchsorted(ts, ts + half_s, side="right") - 1, 0, n - 1)
    same = hi_idx <= lo_idx
    if np.any(same):
        hi_idx = np.where(same & (hi_idx < n - 1), hi_idx + 1, hi_idx)
        same2 = hi_idx <= lo_idx
        lo_idx = np.where(same2 & (lo_idx > 0), lo_idx - 1, lo_idx)
    dt = ts[hi_idx] - ts[lo_idx]
    vel = np.zeros((n, 3), dtype=np.float64)
    good = dt > 1e-4
    vel[good] = (pts[hi_idx[good]] - pts[lo_idx[good]]) / dt[good, None]
    return vel


def _summarize_arm_bucket(
    t_b: np.ndarray,
    pts_b: np.ndarray,
    vel_b: np.ndarray,
    *,
    min_speed_cm_s: float = 1.5,
    rev_dot_thresh: float = -0.25,
    debounce_s: float = 0.25,
) -> tuple[float, float, float, int, float, str]:
    diffs = np.diff(pts_b, axis=0)
    path_cm = float(np.sum(np.linalg.norm(diffs, axis=1))) * 100.0
    net_cm = float(np.linalg.norm(pts_b[-1] - pts_b[0])) * 100.0
    backtrack_ratio = path_cm / max(net_cm, 0.5)

    speeds_cm_s = np.linalg.norm(vel_b, axis=1) * 100.0
    low_speed_pct = float(np.mean(speeds_cm_s < min_speed_cm_s)) * 100.0

    reversals = 0
    ref_dir: np.ndarray | None = None
    last_rev_t = -1e9
    for idx in range(len(t_b)):
        spd = float(speeds_cm_s[idx])
        if spd < min_speed_cm_s:
            continue
        u = vel_b[idx] / (spd / 100.0)
        if ref_dir is None:
            ref_dir = u
            continue
        if float(np.dot(ref_dir, u)) < rev_dot_thresh and (
            float(t_b[idx]) - last_rev_t
        ) >= debounce_s:
            reversals += 1
            last_rev_t = float(t_b[idx])
            ref_dir = u

    # Purely descriptive kinematic tag (never a QC label or rule detector)
    if path_cm < 1.5 and low_speed_pct >= 80.0:
        tag = "[low-speed stall]"
    elif (
        (reversals >= 2 and backtrack_ratio >= 2.2)
        or (path_cm >= 4.0 and backtrack_ratio >= 3.0)
        or (reversals >= 3 and path_cm >= 3.0)
    ):
        tag = "[high-backtrack / low-net-progress]"
    elif net_cm >= 3.0 and backtrack_ratio <= 1.6 and reversals == 0:
        tag = "[steady unidirectional progress]"
    elif low_speed_pct >= 75.0 and net_cm < 2.0:
        tag = "[low-speed stall]"
    else:
        tag = "[slow fine-adjust / turn]"
    return path_cm, net_cm, backtrack_ratio, reversals, low_speed_pct, tag


def format_window_kinematics_summary(
    kinematics: KinematicsTrack | None,
    window_start_s: float,
    window_end_s: float,
    *,
    slice_s: float = 2.0,
) -> str:
    """Format objective 30Hz EEF kinematics statistics per slice for VLM cross-verification.

    Purely factual telemetry summary -- NEVER generates QC segments or rule detections.
    Returns "" gracefully when kinematics is unavailable or has insufficient valid samples.
    """
    if kinematics is None:
        return ""
    eef_pos = getattr(kinematics, "eef_pos", None)
    timestamps_s = getattr(kinematics, "timestamps_s", None)
    if eef_pos is None or timestamps_s is None:
        return ""

    pos = np.asarray(eef_pos, dtype=np.float64)
    ts = np.asarray(timestamps_s, dtype=np.float64)
    if pos.ndim != 3 or ts.ndim != 1 or pos.shape[0] != len(ts) or pos.shape[0] < 2:
        return ""
    if pos.shape[1] < 1 or pos.shape[2] < 3:
        return ""

    w0 = float(window_start_s)
    w1 = float(window_end_s)
    if w1 <= w0 + 1e-6:
        return ""

    raw_valid = getattr(kinematics, "valid_mask", None)
    if raw_valid is not None and len(raw_valid) == len(ts):
        base_valid = np.asarray(raw_valid, dtype=bool) & np.isfinite(ts)
    else:
        base_valid = np.isfinite(ts)

    n_arms = min(2, pos.shape[1])
    arm_names = ("L", "R")[:n_arms]
    pad_s = 0.35

    # Precompute ~0.3s smoothed velocity per arm on valid local slice [w0-pad, w1+pad]
    arm_series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for arm_idx, arm_name in enumerate(arm_names):
        arm_ok = base_valid & np.all(np.isfinite(pos[:, arm_idx, :3]), axis=1)
        loc_mask = arm_ok & (ts >= w0 - pad_s) & (ts <= w1 + pad_s)
        t_loc = ts[loc_mask]
        if len(t_loc) < 2:
            continue
        p_loc = pos[loc_mask, arm_idx, :3]
        if np.any(np.diff(t_loc) < 0.0):
            order = np.argsort(t_loc, kind="mergesort")
            t_loc = t_loc[order]
            p_loc = p_loc[order]
        v_loc = _compute_smoothed_velocities(t_loc, p_loc, smooth_s=0.3)
        arm_series[arm_name] = (t_loc, p_loc, v_loc)

    if not arm_series:
        return ""

    step = max(0.5, float(slice_s))
    buckets: list[tuple[float, float]] = []
    cur = w0
    while cur < w1 - 1e-6:
        nxt = min(w1, cur + step)
        if buckets and (w1 - cur) < 0.5 and (w1 - w0) > step:
            prev_s, _ = buckets[-1]
            buckets[-1] = (prev_s, w1)
            break
        buckets.append((cur, nxt))
        cur = nxt

    lines: list[str] = []
    for b0, b1 in buckets:
        parts: list[str] = []
        for arm_name in arm_names:
            series = arm_series.get(arm_name)
            if series is None:
                continue
            t_loc, p_loc, v_loc = series
            b_mask = (t_loc >= b0 - 1e-6) & (t_loc <= b1 + 1e-6)
            if int(np.sum(b_mask)) < 2:
                continue
            path_cm, net_cm, ratio, revs, low_pct, tag = _summarize_arm_bucket(
                t_loc[b_mask], p_loc[b_mask], v_loc[b_mask]
            )
            parts.append(
                f"{arm_name}: path={path_cm:.1f}cm, net={net_cm:.1f}cm, "
                f"backtrack={ratio:.1f}x, reversals={revs}, low_speed={low_pct:.0f}% {tag}"
            )
        if parts:
            lines.append(f"- [{b0:.2f}s..{b1:.2f}s] " + " | ".join(parts))

    if not lines:
        return ""

    header = (
        "Note: The following 30Hz objective end-effector kinematics telemetry is for cross-checking "
        "with visual frames only; NEVER apply mechanically without visual confirmation. "
        "Intentional cyclic task motions (such as wiping a whiteboard or stirring) are normal "
        "even with a high backtrack ratio."
    )
    return "\n".join([header, *lines])


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
            _sanitize_inactive_labels(render_template(tpl.system_template, ctx), self.labels),
            render_template(tpl.user_template, ctx),
        )

    def build_s1_discovery(
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
            "camera_layout_desc": f"Main HD camera overview frames (camera={main_cam})",
            "task_prompt": t.task_prompt.strip()
            or "Identify the 1-4 primary physical objects manipulated by robot arms/grippers or human hands plus the target receptacle.",
        }
        sys_t = tpl.extra_templates.get("discover_system") or (
            "You are an embodied manipulation visual grounding expert (2D Grounding canonical vocabulary discovery). "
            "Inspect the overview keyframes and select 1 to 4 task-relevant physical objects prioritized by manipulator/hand interaction and target container/receptacle. "
            'Output strict JSON only: {"canonical_objects": [{"id": "snake_case_id", "name": str, "desc": str, "is_static": bool}]}.'
        )
        usr_t = tpl.extra_templates.get("discover_user") or (
            "Task: {{ task_id }} -- {{ task_goal }}\n"
            "Camera view: {{ camera_layout_desc }}\n"
            "Task guidance: {{ task_prompt }}\n"
            "Identify 1-4 canonical physical objects with stable snake_case ids."
        )
        return render_template(sys_t, ctx), render_template(usr_t, ctx)

    def build_s2(
        self,
        *,
        task: TaskSpec | None = None,
        task_spec: TaskSpec | None = None,
        camera_roles: dict[str, str] | None = None,
        duration_s: float = 0.0,
        grounding: GroundingResult | None = None,
        grounding_summary: str | None = None,
    ) -> tuple[str, str]:
        t = task or task_spec
        assert t is not None
        tpl = self.prompts.s2_macro
        steps = "\n".join(f"- {s}" for s in t.steps) if t.steps else "- Segment stages according to observed task progression"
        g_text = (
            grounding_summary.strip()
            if isinstance(grounding_summary, str)
            else format_grounding_summary(grounding)
        )
        ctx = {
            "task_id": t.task_id,
            "duration_s": round(float(duration_s), 2),
            "task_goal": t.goal or t.description or t.title or t.task_id,
            "expected_steps": steps,
            "camera_layout_desc": describe_camera_layout(camera_roles),
            "grounding_summary_text": g_text,
            "task_prompt": t.task_prompt.strip()
            or "Segment macro subtask stages with VLA instructions and dual-manipulator timelines strictly from visual evidence.",
        }
        sys_rendered = render_template(tpl.system_template, ctx)
        usr_rendered = render_template(tpl.user_template, ctx)
        if g_text and "grounding_summary_text" not in tpl.user_template:
            usr_rendered = (
                f"{usr_rendered}\n\n[Stage 1 Grounding Object State Summary]:\n{g_text}"
            )
        return sys_rendered, usr_rendered

    def build_infer_subtasks(
        self,
        *,
        task: TaskSpec | None = None,
        task_spec: TaskSpec | None = None,
        camera_roles: dict[str, str] | None = None,
        duration_s: float = 0.0,
        qc_segments: Sequence[QcSegment] = (),
        macro_stages: Sequence[MacroStage] = (),
        grounding: GroundingResult | None = None,
        grounding_summary: str | None = None,
        custom_hint: str | None = None,
    ) -> tuple[str, str]:
        t = task or task_spec
        assert t is not None
        tpl = self.prompts.s2_macro
        g_text = (
            grounding_summary.strip()
            if isinstance(grounding_summary, str)
            else format_grounding_summary(grounding)
        )
        existing_text = format_macro_backbone(macro_stages) if macro_stages else ""
        qc_text = format_qc_segments_summary(qc_segments, duration_s)
        hint_text = (custom_hint or "").strip()
        ctx = {
            "task_id": t.task_id,
            "duration_s": round(float(duration_s), 2),
            "task_goal": t.goal or t.description or t.title or t.task_id,
            "camera_layout_desc": describe_camera_layout(camera_roles),
            "qc_segments_text": qc_text,
            "existing_stages_text": existing_text,
            "grounding_summary_text": g_text,
            "custom_hint": hint_text,
        }
        sys_t = tpl.extra_templates.get("infer_subtasks_system") or (
            "You are an embodied manipulation subtask alignment expert (supporting both robot arms/grippers and human hands). "
            "Align contiguous macro_stages covering 0.0s to total duration to the verified QC segmentation (respecting cut_clean exclusions and recovery spans) and provide concise imperative VLA instructions. "
            'Output strict JSON only: {"macro_stages": [{"stage_id": "S1", "t_start": float, "t_end": float, "label": str, "label_en": str, "instruction": str, "summary": str}]} (leave arm_timelines omitted).'
        )
        usr_t = tpl.extra_templates.get("infer_subtasks_user") or (
            "Task: {{ task_id }} (duration {{ duration_s }}s)\n"
            "Goal: {{ task_goal }}\n"
            "Camera layout: {{ camera_layout_desc }}\n"
            "[Verified QC Segmentation Skeleton]:\n{{ qc_segments_text }}\n"
            "{% if existing_stages_text %}\n[Current Subtask Stages Draft]:\n{{ existing_stages_text }}\n{% endif %}"
            "{% if grounding_summary_text %}\n[Stage 1 Grounding Object State Summary]:\n{{ grounding_summary_text }}\n{% endif %}"
            "{% if custom_hint %}\n[Reviewer Guidance]:\n{{ custom_hint }}\n{% endif %}"
            "\nAlign contiguous macro_stages covering 0.0s to {{ duration_s }}s with imperative VLA instructions."
        )
        return render_template(sys_t, ctx), render_template(usr_t, ctx)

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
        kinematics_summary: str = "",
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
            "task_goal": (t.goal or "").strip(),
            "task_prompt": (t.task_prompt or "").strip(),
            "window_start_s": round(float(window_start_s), 2),
            "window_end_s": round(float(window_end_s), 2),
            "duration_s": round(float(duration_s), 2),
            "camera_layout_desc": describe_camera_layout(camera_roles),
            "macro_backbone_text": format_macro_backbone(macro_stages),
            "labels_desc": format_labels_desc(self.labels),
            "few_shot_text": fs_text,
            "kinematics_summary": kinematics_summary.strip(),
        }
        return (
            _sanitize_inactive_labels(render_template(tpl.system_template, ctx), self.labels),
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
        kinematics_summary: str = "",
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
            "task_goal": (t.goal or "").strip(),
            "task_prompt": (t.task_prompt or "").strip(),
            "seg_id": candidate.seg_id,
            "t_start": round(float(candidate.t_start), 2),
            "t_end": round(float(candidate.t_end), 2),
            "label_id": candidate.label_id,
            "window_start_s": round(float(window_start_s), 2),
            "window_end_s": round(float(window_end_s), 2),
            "camera_layout_desc": describe_camera_layout(camera_roles),
            "macro_backbone_text": format_macro_backbone(macro_stages),
            "labels_desc": format_labels_desc(self.labels),
            "few_shot_text": fs_text,
            "kinematics_summary": kinematics_summary.strip(),
        }
        return (
            _sanitize_inactive_labels(render_template(tpl.system_template, ctx), self.labels),
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
            _sanitize_inactive_labels(render_template(tpl.system_template, ctx), self.labels),
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
            "task_goal": (t.goal or "").strip(),
            "task_prompt": (t.task_prompt or "").strip(),
            "cut_s": round(float(cut_s), 2),
            "criteria_list": criteria,
        }
        sys_t = tpl.extra_templates.get("control_system") or tpl.system_template
        usr_t = tpl.extra_templates.get("control_user") or tpl.user_template
        return render_template(sys_t, ctx), render_template(usr_t, ctx)

    render_s1 = build_s1
    render_s1_discovery = build_s1_discovery
    render_s2 = build_s2
    render_infer_subtasks = build_infer_subtasks
    render_s3 = build_s3
    render_s4a = build_s4a
    render_s4b = build_s4b
    render_s5 = build_s5
    render_s5_control = build_s5_control
