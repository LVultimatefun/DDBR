"""Stage 2: Macro task backbone stages and gapless dual-arm state timelines."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import numpy as np

from ..config import ConfigManager
from ..io.base import EpisodeView
from ..io.media import compose_2x2_tile
from ..models.annotation import (
    ArmSpan,
    ArmState,
    ArmTimelines,
    GroundingResult,
    MacroStage,
    QcSegment,
)
from ..models.taxonomy import TaskSpec
from .client import BudgetTracker, VlmClient, check_client_cancelled
from .prompt_builder import (
    PromptBuilder,
    format_grounding_summary,
    format_macro_backbone,
    format_qc_segments_summary,
)

STAGE_PALETTE_RGB: list[list[int]] = [
    [59, 130, 246],   # blue
    [16, 185, 129],   # emerald
    [245, 158, 11],   # amber
    [139, 92, 246],   # violet
    [14, 165, 233],   # sky
    [236, 72, 153],   # pink
]

_VALID_ARM_STATES: set[str] = {"idle", "work", "hold", "anom", "mask", "good"}


def _resolve_s2_cfg(cfg: ConfigManager | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(cfg, ConfigManager):
        return dict(cfg.settings.get("vlm", {}).get("s2_macro", {}))
    if isinstance(cfg, dict):
        if "vlm" in cfg and isinstance(cfg["vlm"], dict):
            return dict(cfg["vlm"].get("s2_macro", {}))
        if "s2_macro" in cfg and isinstance(cfg["s2_macro"], dict):
            return dict(cfg["s2_macro"])
        return dict(cfg)
    return {}


def _plan_s2_times(
    duration_s: float,
    interval_s: float,
    max_frames: int,
    long_thresh_s: float,
    long_max_frames: int,
) -> list[float]:
    dur = max(0.05, float(duration_s))
    cap = int(long_max_frames) if dur > float(long_thresh_s) else int(max_frames)
    cap = max(2, cap)
    iv = max(0.25, float(interval_s))
    n_est = int(np.floor(dur / iv)) + 1
    if n_est > cap or n_est < 2:
        n_frames = min(cap, max(2, n_est))
        return sorted({round(float(t), 3) for t in np.linspace(0.0, dur, n_frames)})
    times = [round(min(dur, i * iv), 3) for i in range(n_est)]
    if times[-1] < dur - 0.05:
        times.append(round(dur, 3))
    return sorted(set(times))


def normalize_macro_stages(
    raw_stages: Any, duration_s: float
) -> list[MacroStage]:
    dur = max(0.01, float(duration_s))
    parsed: list[dict[str, Any]] = []
    if isinstance(raw_stages, list):
        for idx, item in enumerate(raw_stages):
            if not isinstance(item, dict):
                continue
            try:
                ts = float(item.get("t_start", 0.0))
                te = float(item.get("t_end", dur))
            except (TypeError, ValueError):
                continue
            ts = max(0.0, min(dur, ts))
            te = max(0.0, min(dur, te))
            if te <= ts + 1e-4:
                continue
            label = str(
                item.get("label") or item.get("label_zh") or item.get("label_en") or f"Stage {idx+1}"
            ).strip()
            if not label:
                label = f"Stage {idx+1}"
            color = item.get("color_rgb")
            if (
                not isinstance(color, list)
                or len(color) != 3
                or any(not isinstance(c, (int, float)) or c < 0 or c > 255 for c in color)
            ):
                color = STAGE_PALETTE_RGB[idx % len(STAGE_PALETTE_RGB)]
            else:
                color = [int(c) for c in color]
            instruction = str(
                item.get("instruction") or item.get("subtask_instruction") or ""
            ).strip()
            parsed.append(
                {
                    "stage_id": str(item.get("stage_id") or f"S{idx+1}"),
                    "t_start": ts,
                    "t_end": te,
                    "label": label,
                    "label_en": str(item.get("label_en") or ""),
                    "instruction": instruction,
                    "summary": str(item.get("summary") or ""),
                    "color_rgb": color,
                }
            )

    if not parsed:
        return [
            MacroStage(
                stage_id="S1",
                t_start=0.0,
                t_end=round(dur, 4),
                label="Full Episode Execution",
                label_en="full_episode",
                instruction="",
                summary="Default macro stage",
                color_rgb=STAGE_PALETTE_RGB[0],
            )
        ]

    parsed.sort(key=lambda d: (d["t_start"], d["t_end"]))
    out: list[MacroStage] = []
    cursor = 0.0
    for idx, d in enumerate(parsed):
        s = cursor
        e = max(s, float(d["t_end"]))
        if idx == len(parsed) - 1:
            e = dur
        if e <= s + 1e-4:
            continue
        out.append(
            MacroStage(
                stage_id=f"S{len(out) + 1}",
                t_start=round(s, 4),
                t_end=round(e, 4),
                label=d["label"],
                label_en=d["label_en"],
                instruction=d["instruction"],
                summary=d["summary"],
                color_rgb=d["color_rgb"],
            )
        )
        cursor = e

    if not out:
        return [
            MacroStage(
                stage_id="S1",
                t_start=0.0,
                t_end=round(dur, 4),
                label="Full Episode Execution",
                label_en="full_episode",
                instruction="",
                summary="Default macro stage",
                color_rgb=STAGE_PALETTE_RGB[0],
            )
        ]
    if out[-1].t_end < dur - 1e-4:
        out[-1] = out[-1].model_copy(update={"t_end": round(dur, 4)})
    return out


def normalize_contiguous_timeline(
    raw_spans: Any, duration_s: float
) -> list[ArmSpan]:
    dur = max(0.01, float(duration_s))
    parsed: list[tuple[float, float, ArmState, str]] = []
    if isinstance(raw_spans, list):
        for item in raw_spans:
            if not isinstance(item, dict):
                continue
            try:
                ts = float(item.get("t_start", 0.0))
                te = float(item.get("t_end", dur))
            except (TypeError, ValueError):
                continue
            ts = max(0.0, min(dur, ts))
            te = max(0.0, min(dur, te))
            if te <= ts + 1e-4:
                continue
            st_raw = str(item.get("state") or "idle").strip().lower()
            st: ArmState = st_raw if st_raw in _VALID_ARM_STATES else "idle"  # type: ignore[assignment]
            desc = str(item.get("desc") or "")
            parsed.append((ts, te, st, desc))

    if not parsed:
        return [ArmSpan(t_start=0.0, t_end=round(dur, 4), state="idle", desc="Default idle")]

    parsed.sort(key=lambda x: (x[0], x[1]))
    out: list[ArmSpan] = []
    cursor = 0.0
    for ts, te, st, desc in parsed:
        if ts > cursor + 1e-3:
            out.append(
                ArmSpan(
                    t_start=round(cursor, 4),
                    t_end=round(ts, 4),
                    state="idle",
                    desc="Transition / idle",
                )
            )
            cursor = ts
        s = cursor
        e = max(s, te)
        if e <= s + 1e-4:
            continue
        out.append(ArmSpan(t_start=round(s, 4), t_end=round(e, 4), state=st, desc=desc))
        cursor = e

    if cursor < dur - 1e-3:
        out.append(
            ArmSpan(
                t_start=round(cursor, 4),
                t_end=round(dur, 4),
                state="idle",
                desc="Trailing idle",
            )
        )
    elif out and out[-1].t_end < dur - 1e-4:
        out[-1] = out[-1].model_copy(update={"t_end": round(dur, 4)})
    return out


def run_stage2_macro(
    view: EpisodeView | None = None,
    *,
    episode_view: EpisodeView | None = None,
    task: TaskSpec | None = None,
    task_spec: TaskSpec | None = None,
    client: VlmClient | None = None,
    vlm_client: VlmClient | None = None,
    cfg: ConfigManager | dict[str, Any] | None = None,
    config: ConfigManager | dict[str, Any] | None = None,
    prompt_builder: PromptBuilder | None = None,
    budget: BudgetTracker | None = None,
    grounding: GroundingResult | None = None,
) -> tuple[list[MacroStage], ArmTimelines]:
    """Execute Stage 2 macro task backbone and dual-arm timelines."""
    ev = view or episode_view
    cl = client or vlm_client
    cm = cfg if isinstance(cfg, ConfigManager) else (config if isinstance(config, ConfigManager) else None)
    t_spec = task or task_spec or (cm.get_task(ev.meta.task_id) if cm and ev else TaskSpec(task_id="_default"))
    assert ev is not None and cl is not None

    s2_cfg = _resolve_s2_cfg(cfg or config)
    times_s = _plan_s2_times(
        ev.duration_s,
        float(s2_cfg.get("interval_s", 2.0)),
        int(s2_cfg.get("max_frames", 48)),
        float(s2_cfg.get("long_episode_threshold_s", 96.0)),
        int(s2_cfg.get("long_episode_max_frames", 64)),
    )
    cell_w = int(s2_cfg.get("cell_w", 320))
    cell_h = int(s2_cfg.get("cell_h", 180))
    quality = int(s2_cfg.get("quality", 70))
    thinking = bool(s2_cfg.get("thinking", False))
    thinking_budget = int(s2_cfg.get("thinking_budget", 1536))
    max_tokens = int(s2_cfg.get("max_tokens", 4096))

    check_client_cancelled(cl)
    roles = ev.camera_roles

    def _build_tile(t: float) -> np.ndarray:
        check_client_cancelled(cl)
        return compose_2x2_tile(
            ev.media, roles, t, cell_w=cell_w, cell_h=cell_h, include_header=True
        )

    with ThreadPoolExecutor(max_workers=min(6, max(1, len(times_s)))) as pool:
        tiles = list(pool.map(_build_tile, times_s))

    check_client_cancelled(cl)

    pb = prompt_builder or (PromptBuilder(cm) if cm is not None else None)
    if pb is not None:
        sys_p, usr_p = pb.build_s2(
            task=t_spec,
            camera_roles=roles,
            duration_s=ev.duration_s,
            grounding=grounding,
        )
    else:
        sys_p = (
            "You are an embodied manipulation task segmentation expert (supporting both robot arms/grippers and human hands). "
            'Output strict JSON only: {"macro_stages": [{"stage_id": "S1", "t_start": float, "t_end": float, "label": str, "label_en": str, "instruction": str, "summary": str}], "arm_timelines": {"left": [...], "right": [...]}}.'
        )
        g_txt = format_grounding_summary(grounding)
        usr_p = f"Task: {t_spec.task_id} (duration {ev.duration_s:.2f}s)"
        if g_txt:
            usr_p += f"\n[Stage 1 Grounding Object State Summary]:\n{g_txt}"

    resp = cl.chat_json(
        stage="s2_macro",
        system_prompt=sys_p,
        user_prompt=usr_p,
        images=tiles,
        timestamps_s=times_s,
        thinking=thinking,
        thinking_budget=thinking_budget,
        max_tokens=max_tokens,
        jpeg_quality=quality,
        budget=budget,
    )

    macro_stages = normalize_macro_stages(resp.get("macro_stages"), ev.duration_s)
    arms_raw = resp.get("arm_timelines") if isinstance(resp.get("arm_timelines"), dict) else {}
    left_tl = normalize_contiguous_timeline(arms_raw.get("left"), ev.duration_s)
    right_tl = normalize_contiguous_timeline(arms_raw.get("right"), ev.duration_s)
    return macro_stages, ArmTimelines(left=left_tl, right=right_tl)


def _plan_infer_subtask_times(
    duration_s: float,
    qc_segments: Sequence[QcSegment],
    interval_s: float,
    max_frames: int,
    long_thresh_s: float,
    long_max_frames: int,
) -> list[float]:
    dur = max(0.05, float(duration_s))
    cap = int(long_max_frames) if dur > float(long_thresh_s) else int(max_frames)
    cap = max(2, cap)

    # Start with regular S2 grid, then blend in key QC cut/recovery boundary timestamps
    base = _plan_s2_times(dur, interval_s, max_frames, long_thresh_s, long_max_frames)
    anchors: list[float] = [0.0, round(dur, 3)]
    for seg in qc_segments:
        if seg.splice_policy == "cut_clean" or seg.training_role == "recovery":
            ts = round(max(0.0, min(dur, float(seg.t_start))), 3)
            te = round(max(0.0, min(dur, float(seg.t_end))), 3)
            anchors.append(ts)
            anchors.append(te)

    merged = sorted(set(base) | set(anchors))
    if len(merged) <= cap:
        return merged

    # Preserve boundary anchors first, then fill evenly from base grid up to cap
    dedup_anchors: list[float] = []
    for t in sorted(set(anchors)):
        if not dedup_anchors or (t - dedup_anchors[-1]) >= 0.15:
            dedup_anchors.append(t)
    if len(dedup_anchors) >= cap:
        idxs = np.linspace(0, len(dedup_anchors) - 1, cap)
        return sorted({dedup_anchors[int(round(float(i)))] for i in idxs})

    chosen = set(dedup_anchors)
    remaining = [t for t in base if all(abs(t - a) >= 0.15 for a in chosen)]
    need = cap - len(chosen)
    if remaining and need > 0:
        if len(remaining) <= need:
            chosen.update(remaining)
        else:
            idxs = np.linspace(0, len(remaining) - 1, need)
            for i in idxs:
                chosen.add(remaining[int(round(float(i)))])
    return sorted(chosen)


def infer_subtasks_from_segments(
    view: EpisodeView | None = None,
    *,
    episode_view: EpisodeView | None = None,
    qc_segments: Sequence[QcSegment] = (),
    macro_stages: Sequence[MacroStage] = (),
    grounding: GroundingResult | None = None,
    custom_hint: str | None = None,
    task: TaskSpec | None = None,
    task_spec: TaskSpec | None = None,
    client: VlmClient | None = None,
    vlm_client: VlmClient | None = None,
    cfg: ConfigManager | dict[str, Any] | None = None,
    config: ConfigManager | dict[str, Any] | None = None,
    prompt_builder: PromptBuilder | None = None,
    budget: BudgetTracker | None = None,
) -> list[MacroStage]:
    """Re-infer aligned Subtask stages & VLA instructions from verified QC segments + grounding + frames."""
    ev = view or episode_view
    cl = client or vlm_client
    cm = cfg if isinstance(cfg, ConfigManager) else (config if isinstance(config, ConfigManager) else None)
    t_spec = task or task_spec or (cm.get_task(ev.meta.task_id) if cm and ev else TaskSpec(task_id="_default"))
    assert ev is not None and cl is not None

    s2_cfg = _resolve_s2_cfg(cfg or config)
    times_s = _plan_infer_subtask_times(
        ev.duration_s,
        qc_segments,
        float(s2_cfg.get("interval_s", 2.0)),
        int(s2_cfg.get("max_frames", 48)),
        float(s2_cfg.get("long_episode_threshold_s", 96.0)),
        int(s2_cfg.get("long_episode_max_frames", 64)),
    )
    cell_w = int(s2_cfg.get("cell_w", 320))
    cell_h = int(s2_cfg.get("cell_h", 180))
    quality = int(s2_cfg.get("quality", 70))
    thinking = bool(s2_cfg.get("thinking", False))
    thinking_budget = int(s2_cfg.get("thinking_budget", 1536))
    max_tokens = int(s2_cfg.get("max_tokens", 4096))

    check_client_cancelled(cl)
    roles = ev.camera_roles

    def _build_tile(t: float) -> np.ndarray:
        check_client_cancelled(cl)
        return compose_2x2_tile(
            ev.media, roles, t, cell_w=cell_w, cell_h=cell_h, include_header=True
        )

    with ThreadPoolExecutor(max_workers=min(6, max(1, len(times_s)))) as pool:
        tiles = list(pool.map(_build_tile, times_s))

    check_client_cancelled(cl)

    pb = prompt_builder or (PromptBuilder(cm) if cm is not None else None)
    if pb is not None:
        sys_p, usr_p = pb.build_infer_subtasks(
            task=t_spec,
            camera_roles=roles,
            duration_s=ev.duration_s,
            qc_segments=qc_segments,
            macro_stages=macro_stages,
            grounding=grounding,
            custom_hint=custom_hint,
        )
    else:
        sys_p = (
            "You are an embodied manipulation subtask alignment expert (supporting both robot arms/grippers and human hands). "
            'Output strict JSON only: {"macro_stages": [{"stage_id": "S1", "t_start": float, "t_end": float, "label": str, "label_en": str, "instruction": str, "summary": str}]} (leave arm_timelines omitted).'
        )
        parts = [
            f"Task: {t_spec.task_id} (duration {ev.duration_s:.2f}s)",
            "[Verified QC Segmentation Skeleton]:",
            format_qc_segments_summary(qc_segments, ev.duration_s),
        ]
        if macro_stages:
            parts.append("[Current Subtask Stages Draft]:")
            parts.append(format_macro_backbone(macro_stages))
        g_txt = format_grounding_summary(grounding)
        if g_txt:
            parts.append("[Stage 1 Grounding Object State Summary]:")
            parts.append(g_txt)
        if custom_hint and custom_hint.strip():
            parts.append("[Reviewer Guidance]:")
            parts.append(custom_hint.strip())
        usr_p = "\n".join(parts)

    resp = cl.chat_json(
        stage="s2_macro",
        system_prompt=sys_p,
        user_prompt=usr_p,
        images=tiles,
        timestamps_s=times_s,
        thinking=thinking,
        thinking_budget=thinking_budget,
        max_tokens=max_tokens,
        jpeg_quality=quality,
        budget=budget,
    )
    return normalize_macro_stages(resp.get("macro_stages"), ev.duration_s)


run_s2 = run_stage2_macro
