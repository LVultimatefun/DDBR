"""Stage 3: Precision-First sliding-window QC discovery with S2 protection & Few-Shot injection."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import numpy as np

from ..config import ConfigManager
from ..io.base import EpisodeView
from ..io.media import compose_2x2_tile
from ..models.annotation import (
    MacroStage,
    MaskSide,
    QcSegment,
    SplicePolicy,
    TrainingRole,
    VisualEvidence,
)
from ..models.taxonomy import FewShotItem, LabelRegistry, TaskSpec, VisualLabelDef
from .client import BudgetExceeded, BudgetTracker, VlmClient
from .prompt_builder import PromptBuilder

_FORBIDDEN_LEGACY = {"long_pause", "back_and_forth"}


def _resolve_s3_cfg(cfg: ConfigManager | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(cfg, ConfigManager):
        return dict(cfg.settings.get("vlm", {}).get("s3_discover", {}))
    if isinstance(cfg, dict):
        if "vlm" in cfg and isinstance(cfg["vlm"], dict):
            return dict(cfg["vlm"].get("s3_discover", {}))
        if "s3_discover" in cfg and isinstance(cfg["s3_discover"], dict):
            return dict(cfg["s3_discover"])
        return dict(cfg)
    return {}


def _plan_windows(
    duration_s: float, window_s: float, stride_s: float
) -> list[tuple[float, float]]:
    dur = max(0.05, float(duration_s))
    ws = max(1.0, float(window_s))
    st = max(0.5, float(stride_s))
    if dur <= ws + 1e-4:
        return [(0.0, round(dur, 3))]
    windows: list[tuple[float, float]] = []
    cur = 0.0
    while cur < dur - 1e-4:
        end = min(dur, cur + ws)
        windows.append((round(cur, 3), round(end, 3)))
        if end >= dur - 1e-4:
            break
        cur += st
    return windows


def _plan_window_times(w0: float, w1: float, step_s: float) -> list[float]:
    span = max(0.02, w1 - w0)
    step = max(0.2, float(step_s))
    n = max(2, int(np.floor(span / step)) + 1)
    if span < step:
        n = 2
    return sorted({round(float(t), 3) for t in np.linspace(w0, w1, n)})


def normalize_triplet_fields(
    *,
    label_def: VisualLabelDef | None,
    raw_role: str | None,
    raw_policy: str | None,
    raw_arm: str | None,
    raw_mask_side: str | None,
) -> tuple[TrainingRole, SplicePolicy, MaskSide, MaskSide | None]:
    arm: MaskSide = (
        raw_arm if raw_arm in {"left", "right", "both"} else "both"  # type: ignore[assignment]
    )
    policy: SplicePolicy
    if raw_policy in {"keep_positive", "mask_loss", "cut_clean"}:
        policy = raw_policy  # type: ignore[assignment]
    elif label_def is not None:
        policy = label_def.default_splice_policy
    else:
        policy = "mask_loss"

    role: TrainingRole
    if raw_role in {"necessary", "recovery", "badcase", "terminal_fail"}:
        role = raw_role  # type: ignore[assignment]
    elif label_def is not None:
        role = label_def.default_training_role
    else:
        role = "badcase"

    mask_side: MaskSide | None
    if policy == "keep_positive":
        mask_side = None
        if role not in {"necessary", "recovery"}:
            role = (
                label_def.default_training_role
                if (label_def and label_def.default_training_role in {"necessary", "recovery"})
                else "necessary"
            )
    elif policy == "cut_clean":
        mask_side = None
        if role not in {"badcase", "terminal_fail"}:
            role = (
                label_def.default_training_role
                if (label_def and label_def.default_training_role in {"badcase", "terminal_fail"})
                else "badcase"
            )
    else:  # mask_loss
        mask_side = (
            raw_mask_side
            if raw_mask_side in {"left", "right", "both"}
            else arm
        )  # type: ignore[assignment]
        if role not in {"badcase", "terminal_fail", "recovery"}:
            role = (
                label_def.default_training_role
                if (label_def and label_def.default_training_role in {"badcase", "terminal_fail", "recovery"})
                else "badcase"
            )

    return role, policy, arm, mask_side


def _parse_visual_evidence(raw: Any) -> VisualEvidence:
    if isinstance(raw, VisualEvidence):
        return raw
    if isinstance(raw, dict):
        return VisualEvidence(
            initial_state=str(raw.get("initial_state") or ""),
            process_change=str(raw.get("process_change") or ""),
            end_state=str(raw.get("end_state") or ""),
            confidence_note=str(raw.get("confidence_note") or ""),
        )
    if isinstance(raw, str):
        return VisualEvidence(process_change=raw)
    return VisualEvidence()


def run_stage3_discover(
    view: EpisodeView | None = None,
    *,
    episode_view: EpisodeView | None = None,
    task: TaskSpec | None = None,
    task_spec: TaskSpec | None = None,
    macro_stages: Sequence[MacroStage] = (),
    few_shots: Sequence[FewShotItem] | None = None,
    few_shot_items: Sequence[FewShotItem] | None = None,
    client: VlmClient | None = None,
    vlm_client: VlmClient | None = None,
    cfg: ConfigManager | dict[str, Any] | None = None,
    config: ConfigManager | dict[str, Any] | None = None,
    prompt_builder: PromptBuilder | None = None,
    labels: LabelRegistry | None = None,
    budget: BudgetTracker | None = None,
) -> list[QcSegment]:
    """Execute Stage 3 sliding-window QC discovery with S2 backbone protection & Few-Shot."""
    ev = view or episode_view
    cl = client or vlm_client
    cm = cfg if isinstance(cfg, ConfigManager) else (config if isinstance(config, ConfigManager) else None)
    t_spec = task or task_spec or (cm.get_task(ev.meta.task_id) if cm and ev else TaskSpec(task_id="_default"))
    reg = labels or (cm.labels if cm else LabelRegistry())
    fs_list = few_shots if few_shots is not None else few_shot_items
    assert ev is not None and cl is not None

    s3_cfg = _resolve_s3_cfg(cfg or config)
    window_s = float(s3_cfg.get("window_s", 16.0))
    stride_s = float(s3_cfg.get("window_stride_s", 12.0))
    step_s = float(s3_cfg.get("window_step_s", 0.8))
    cell_w = int(s3_cfg.get("cell_w", 360))
    cell_h = int(s3_cfg.get("cell_h", 202))
    quality = int(s3_cfg.get("quality", 72))
    thinking = bool(s3_cfg.get("thinking", False))
    thinking_budget = int(s3_cfg.get("thinking_budget", 1536))
    max_tokens = int(s3_cfg.get("max_tokens", 3200))

    windows = _plan_windows(ev.duration_s, window_s, stride_s)
    roles = ev.camera_roles
    pb = prompt_builder or (PromptBuilder(cm) if cm is not None else None)
    min_span = 0.4 if ev.duration_s >= 1.5 else 0.1

    def _run_window(w_idx: int, w0: float, w1: float) -> tuple[int, list[dict[str, Any]], Exception | None]:
        t_list = _plan_window_times(w0, w1, step_s)
        tiles = [
            compose_2x2_tile(
                ev.media, roles, t, cell_w=cell_w, cell_h=cell_h, include_header=True
            )
            for t in t_list
        ]
        if pb is not None:
            sys_p, usr_p = pb.build_s3(
                task=t_spec,
                camera_roles=roles,
                macro_stages=macro_stages,
                window_start_s=w0,
                window_end_s=w1,
                duration_s=ev.duration_s,
                few_shots=fs_list,
            )
        else:
            sys_p = "You are a high-precision embodied trajectory QC expert following Precision-First principles."
            usr_p = (
                f"Task: {t_spec.task_id} | Window: [{w0:.2f}s, {w1:.2f}s] / total {ev.duration_s:.2f}s\n"
                "[Active Pure-Visual QC Label Registry]:\n- boundary_idle"
            )
        try:
            resp = cl.chat_json(
                stage="s3_discover",
                system_prompt=sys_p,
                user_prompt=usr_p,
                images=tiles,
                timestamps_s=t_list,
                thinking=thinking,
                thinking_budget=thinking_budget,
                max_tokens=max_tokens,
                jpeg_quality=quality,
                budget=budget,
            )
            segs = resp.get("segments")
            return w_idx, segs if isinstance(segs, list) else [], None
        except BudgetExceeded:
            raise
        except Exception as exc:
            return w_idx, [], exc

    win_results: list[tuple[int, list[dict[str, Any]]]] = []
    errors: list[Exception] = []
    with ThreadPoolExecutor(max_workers=min(6, max(1, len(windows)))) as pool:
        futs = [pool.submit(_run_window, i, w0, w1) for i, (w0, w1) in enumerate(windows)]
        for fut in futs:
            w_idx, segs, err = fut.result()
            if err is not None:
                errors.append(err)
            else:
                win_results.append((w_idx, segs))

    if errors:
        raise errors[0]

    win_results.sort(key=lambda x: x[0])
    out: list[QcSegment] = []
    for w_idx, raw_list in win_results:
        w0, w1 = windows[w_idx]
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            raw_lbl = str(
                item.get("label_id") or item.get("type") or item.get("segment_type") or ""
            ).strip()
            if not raw_lbl or raw_lbl.lower() in _FORBIDDEN_LEGACY:
                continue
            try:
                ts = float(item.get("t_start", item.get("start_s", 0.0)))
                te = float(item.get("t_end", item.get("end_s", 0.0)))
            except (TypeError, ValueError):
                continue
            ts = max(0.0, max(w0, min(ev.duration_s, ts)))
            te = min(ev.duration_s, min(w1, te))
            eff_min = (
                min(min_span, 0.1)
                if (
                    raw_lbl.lower() == "boundary_idle"
                    and (ts <= 0.15 + 1e-6 or (te + 1e-6) >= max(0.0, ev.duration_s - 0.15))
                )
                else min_span
            )
            if round(te - ts, 6) < eff_min:
                continue

            lbl_def = reg.get(raw_lbl)
            unknown_note = str(item.get("unknown_note") or "")
            suggested: str | None = item.get("suggested_label")
            if lbl_def is None or lbl_def.status == "retired":
                eff_lbl = "unknown_anomaly"
                if not unknown_note:
                    unknown_note = f"Unrecognized label {raw_lbl!r}"
                if not suggested:
                    suggested = raw_lbl
                lbl_def = reg.get("unknown_anomaly")
            else:
                eff_lbl = lbl_def.id

            role, policy, arm, mask_side = normalize_triplet_fields(
                label_def=lbl_def,
                raw_role=item.get("training_role"),
                raw_policy=item.get("splice_policy"),
                raw_arm=item.get("affected_arm") or item.get("hand"),
                raw_mask_side=item.get("mask_side"),
            )
            try:
                conf = float(item.get("confidence", 0.85))
            except (TypeError, ValueError):
                conf = 0.85
            conf = max(0.0, min(1.0, conf))

            out.append(
                QcSegment(
                    seg_id=f"s3_w{w_idx}_{len(out)+1}",
                    t_start=round(ts, 3),
                    t_end=round(te, 3),
                    label_id=eff_lbl,
                    name=lbl_def.name if lbl_def else eff_lbl,
                    training_role=role,
                    splice_policy=policy,
                    mask_side=mask_side,
                    affected_arm=arm,
                    visual_evidence=_parse_visual_evidence(item.get("visual_evidence")),
                    vlm_raw_policy=policy,
                    is_unknown=(eff_lbl == "unknown_anomaly"),
                    unknown_note=unknown_note,
                    suggested_label=suggested,
                    confidence=conf,
                )
            )

    out.sort(key=lambda s: (s.t_start, s.t_end))
    return out


run_s3 = run_stage3_discover
