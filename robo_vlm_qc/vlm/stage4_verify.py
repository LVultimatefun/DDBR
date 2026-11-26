"""Stage 4A global consolidation/merge + Stage 4B 0.5s dense Start-vs-End CoV refinement."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import numpy as np

from ..config import ConfigManager
from ..io.base import EpisodeView
from ..io.media import compose_2x2_tile
from ..models.annotation import (
    GuardrailMetrics,
    GuardrailReport,
    MacroStage,
    QcSegment,
    VisualEvidence,
)
from ..models.taxonomy import FewShotItem, LabelRegistry, TaskSpec
from .client import (
    BudgetExceeded,
    BudgetTracker,
    VlmBlindVisionError,
    VlmCancelledError,
    VlmCircuitOpenError,
    VlmClient,
    VlmError,
    check_client_cancelled,
)
from .prompt_builder import PromptBuilder, format_window_kinematics_summary
from .stage3_discover import _parse_visual_evidence, normalize_triplet_fields

_FORBIDDEN_LEGACY = {"long_pause", "back_and_forth"}


def _resolve_s4_cfg(cfg: ConfigManager | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(cfg, ConfigManager):
        return dict(cfg.settings.get("vlm", {}).get("s4_verify", {}))
    if isinstance(cfg, dict):
        if "vlm" in cfg and isinstance(cfg["vlm"], dict):
            return dict(cfg["vlm"].get("s4_verify", {}))
        if "s4_verify" in cfg and isinstance(cfg["s4_verify"], dict):
            return dict(cfg["s4_verify"])
        return dict(cfg)
    return {}


def _resolve_edge_idle_kin_skip(cfg: ConfigManager | dict[str, Any] | None) -> bool:
    if isinstance(cfg, ConfigManager):
        s4 = cfg.settings.get("vlm", {}).get("s4_verify", {})
        if isinstance(s4, dict) and "boundary_idle_kin_skip_s4b" in s4:
            return bool(s4["boundary_idle_kin_skip_s4b"])
        return True
    if isinstance(cfg, dict):
        if "boundary_idle_kin_skip_s4b" in cfg:
            return bool(cfg["boundary_idle_kin_skip_s4b"])
        s4_top = cfg.get("s4_verify")
        if isinstance(s4_top, dict) and "boundary_idle_kin_skip_s4b" in s4_top:
            return bool(s4_top["boundary_idle_kin_skip_s4b"])
        vlm = cfg.get("vlm", {})
        if isinstance(vlm, dict):
            s4 = vlm.get("s4_verify", {})
            if isinstance(s4, dict) and "boundary_idle_kin_skip_s4b" in s4:
                return bool(s4["boundary_idle_kin_skip_s4b"])
    return True


def _resolve_boundary_edge_tol_s(cfg: ConfigManager | dict[str, Any] | None) -> float:
    if isinstance(cfg, ConfigManager):
        sg = cfg.settings.get("guardrail", {}).get("splice_gate", {})
        if isinstance(sg, dict) and "boundary_edge_tol_s" in sg:
            return float(sg["boundary_edge_tol_s"])
    elif isinstance(cfg, dict):
        gr = cfg.get("guardrail", {})
        if isinstance(gr, dict):
            sg = gr.get("splice_gate", {})
            if isinstance(sg, dict) and "boundary_edge_tol_s" in sg:
                return float(sg["boundary_edge_tol_s"])
    return 0.5


def _needs_s4a_vlm_merge(
    ordered: Sequence[QcSegment], *, gap_threshold_s: float = 2.0
) -> bool:
    """Return False when <=1 candidate or all sorted candidates are separated by > gap_threshold_s."""
    if len(ordered) <= 1:
        return False
    max_end = float(ordered[0].t_end)
    for seg in ordered[1:]:
        if float(seg.t_start) - max_end <= gap_threshold_s + 1e-6:
            return True
        if float(seg.t_end) > max_end:
            max_end = float(seg.t_end)
    return False


def _try_kinematics_verify_edge_boundary_idle(
    cand: QcSegment,
    *,
    view: EpisodeView,
    duration_s: float,
    edge_tol_s: float = 0.5,
    max_p95_speed_mps: float = 0.03,
    max_net_disp_m: float = 0.02,
    max_hand_delta_rad: float = 0.15,
    max_torso_delta_rad: float = 0.05,
    max_chassis_disp_m: float = 0.02,
) -> QcSegment | None:
    """Verify edge `boundary_idle` purely via 30Hz kinematics stillness, skipping S4B VLM call."""
    if cand.label_id != "boundary_idle":
        return None
    kin = view.kinematics
    if kin is None or kin.eef_pos is None or kin.timestamps_s is None:
        return None

    dur = max(0.05, float(duration_s))
    near_start = float(cand.t_start) <= edge_tol_s + 1e-6
    near_end = (float(cand.t_end) + 1e-6) >= max(0.0, dur - edge_tol_s)
    if not (near_start or near_end):
        return None

    snap_start = 0.0 if near_start else round(float(cand.t_start), 3)
    snap_end = round(dur, 3) if near_end else round(float(cand.t_end), 3)
    if snap_end <= snap_start + 0.05:
        return None

    ts = np.asarray(kin.timestamps_s, dtype=np.float64)
    pos = np.asarray(kin.eef_pos, dtype=np.float64)
    if ts.ndim != 1 or len(ts) < 2:
        return None
    if pos.ndim == 2 and pos.shape[1] >= 3:
        pos = pos[:, None, :3]
    if pos.ndim != 3 or pos.shape[0] != len(ts) or pos.shape[1] < 1 or pos.shape[2] < 3:
        return None

    raw_vm = getattr(kin, "valid_mask", None)
    if raw_vm is not None and len(raw_vm) == len(ts):
        valid_mask = np.asarray(raw_vm, dtype=bool)
    else:
        valid_mask = np.ones(len(ts), dtype=bool)

    span_mask = (ts >= snap_start - 1e-6) & (ts <= snap_end + 1e-6) & np.isfinite(ts) & valid_mask
    idx = np.flatnonzero(span_mask)
    if len(idx) < 2:
        return None

    t_seg = ts[idx]
    order = np.argsort(t_seg, kind="mergesort")
    idx = idx[order]
    t_seg = t_seg[order]
    dt_all = np.diff(t_seg)
    if not np.any(dt_all > 1e-6):
        return None

    max_net_disp = 0.0
    any_valid_arm = False
    for a in range(pos.shape[1]):
        pts_a = pos[idx, a, :3]
        ok_a = np.all(np.isfinite(pts_a), axis=1)
        if int(np.sum(ok_a)) < 2:
            continue
        t_a = t_seg[ok_a]
        p_a = pts_a[ok_a]
        dt_a = np.diff(t_a)
        good_dt = dt_a > 1e-6
        if not np.any(good_dt):
            continue
        any_valid_arm = True
        step_d = np.linalg.norm(np.diff(p_a, axis=0), axis=1)
        speeds = step_d[good_dt] / dt_a[good_dt]
        p95_spd = float(np.percentile(speeds, 95))
        net_disp = float(np.linalg.norm(p_a[-1] - p_a[0]))
        max_exc = float(np.linalg.norm(p_a - p_a[0][None, :], axis=1).max())
        if p95_spd >= max_p95_speed_mps or net_disp >= max_net_disp_m or max_exc >= max_net_disp_m:
            return None
        if net_disp > max_net_disp:
            max_net_disp = net_disp

    if not any_valid_arm:
        return None

    max_hand_delta = 0.0
    if kin.hand_q is not None:
        hq = np.asarray(kin.hand_q, dtype=np.float64)
        if hq.shape[0] == len(ts) and hq.size > 0:
            hq_seg = hq[idx].reshape(len(idx), -1)
            ok_h = np.all(np.isfinite(hq_seg), axis=1)
            if int(np.sum(ok_h)) >= 2:
                hq_valid = hq_seg[ok_h]
                max_hand_delta = float(np.max(np.ptp(hq_valid, axis=0)))
                if max_hand_delta >= max_hand_delta_rad:
                    return None

    max_torso_delta = 0.0
    if kin.torso_q is not None:
        tq = np.asarray(kin.torso_q, dtype=np.float64)
        if tq.shape[0] == len(ts) and tq.size > 0:
            tq_seg = tq[idx].reshape(len(idx), -1)
            ok_t = np.all(np.isfinite(tq_seg), axis=1)
            if int(np.sum(ok_t)) >= 2:
                tq_valid = tq_seg[ok_t]
                max_torso_delta = float(np.max(np.ptp(tq_valid, axis=0)))
                if max_torso_delta >= max_torso_delta_rad:
                    return None

    max_chassis_disp = 0.0
    if kin.chassis_pose is not None:
        ch = np.asarray(kin.chassis_pose, dtype=np.float64)
        if ch.ndim >= 2 and ch.shape[0] == len(ts) and ch.shape[1] >= 2:
            ch_seg = ch[idx, :2]
            ok_c = np.all(np.isfinite(ch_seg), axis=1)
            if int(np.sum(ok_c)) >= 2:
                ch_valid = ch_seg[ok_c]
                max_chassis_disp = float(
                    np.linalg.norm(ch_valid - ch_valid[0][None, :], axis=1).max()
                )
                if max_chassis_disp >= max_chassis_disp_m:
                    return None

    new_ev = cand.visual_evidence.model_copy(
        update={"confidence_note": "kinematics_verified_edge_idle"}
    )
    gr = GuardrailReport(
        gate_type="splice_gate",
        passed=True,
        downgraded=False,
        metrics=GuardrailMetrics(
            eef_pos_m=round(max_net_disp, 5),
            hand_rad=round(max_hand_delta, 5),
            torso_rad=round(max_torso_delta, 5),
            chassis_pos_m=round(max_chassis_disp, 5),
        ),
        opt_t_start=snap_start,
        opt_t_end=snap_end,
    )
    return cand.model_copy(
        update={
            "t_start": snap_start,
            "t_end": snap_end,
            "training_role": "badcase",
            "splice_policy": "cut_clean",
            "mask_side": None,
            "vlm_raw_policy": "cut_clean",
            "visual_evidence": new_ev,
            "guardrail": gr,
        }
    )


def _is_short_edge_boundary_idle(
    seg: QcSegment,
    *,
    duration_s: float | None = None,
    edge_tol_s: float = 0.15,
    min_edge_len_s: float = 0.1,
) -> bool:
    if seg.label_id != "boundary_idle":
        return False
    if round(seg.t_end - seg.t_start, 6) < min_edge_len_s:
        return False
    near_start = seg.t_start <= edge_tol_s + 1e-6
    near_end = duration_s is not None and (seg.t_end + 1e-6) >= max(0.0, duration_s - edge_tol_s)
    return near_start or near_end


def _deterministic_merge_raw_segments(
    candidates: Sequence[QcSegment],
    *,
    min_len_s: float = 0.35,
    duration_s: float | None = None,
    min_edge_len_s: float = 0.1,
) -> list[QcSegment]:
    """Deterministic overlap dedup fallback preserving distinct non-overlapping retries."""
    ordered = sorted(candidates, key=lambda s: (s.t_start, s.t_end))
    merged: list[QcSegment] = []
    for seg in ordered:
        eff_min = (
            min(min_len_s, min_edge_len_s)
            if _is_short_edge_boundary_idle(
                seg, duration_s=duration_s, min_edge_len_s=min_edge_len_s
            )
            else min_len_s
        )
        if round(seg.t_end - seg.t_start, 6) < eff_min:
            continue
        if not merged:
            merged.append(seg)
            continue
        prev = merged[-1]
        inter = max(0.0, min(prev.t_end, seg.t_end) - max(prev.t_start, seg.t_start))
        union = max(prev.t_end, seg.t_end) - min(prev.t_start, seg.t_start)
        iou = inter / union if union > 0 else 0.0
        same_label = prev.label_id == seg.label_id
        compat_arm = (
            prev.affected_arm == seg.affected_arm
            or prev.affected_arm == "both"
            or seg.affected_arm == "both"
        )
        if same_label and compat_arm and (inter >= 0.15 or iou >= 0.25):
            new_arm = (
                "both"
                if prev.affected_arm != seg.affected_arm
                else prev.affected_arm
            )
            new_ev = (
                prev.visual_evidence
                if len(prev.visual_evidence.as_text()) >= len(seg.visual_evidence.as_text())
                else seg.visual_evidence
            )
            merged[-1] = prev.model_copy(
                update={
                    "t_start": min(prev.t_start, seg.t_start),
                    "t_end": max(prev.t_end, seg.t_end),
                    "affected_arm": new_arm,
                    "visual_evidence": new_ev,
                    "confidence": max(prev.confidence, seg.confidence),
                }
            )
        else:
            merged.append(seg)
    return merged


def _build_segment_from_dict(
    item: dict[str, Any],
    *,
    seg_id: str,
    duration_s: float,
    reg: LabelRegistry,
    fallback_seg: QcSegment | None = None,
) -> QcSegment | None:
    raw_lbl = str(
        item.get("label_id")
        or item.get("type")
        or item.get("segment_type")
        or (fallback_seg.label_id if fallback_seg else "")
    ).strip()
    if not raw_lbl or raw_lbl.lower() in _FORBIDDEN_LEGACY:
        return None
    try:
        ts = float(
            item.get("t_start", item.get("start_s", fallback_seg.t_start if fallback_seg else 0.0))
        )
        te = float(
            item.get("t_end", item.get("end_s", fallback_seg.t_end if fallback_seg else 0.0))
        )
    except (TypeError, ValueError):
        return None
    ts = max(0.0, min(duration_s, ts))
    te = max(0.0, min(duration_s, te))
    if te <= ts + 0.05:
        return None

    lbl_def = reg.get(raw_lbl)
    unknown_note = str(item.get("unknown_note") or (fallback_seg.unknown_note if fallback_seg else ""))
    suggested = item.get("suggested_label") or (fallback_seg.suggested_label if fallback_seg else None)
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
        raw_arm=item.get("affected_arm") or item.get("hand") or (fallback_seg.affected_arm if fallback_seg else None),
        raw_mask_side=item.get("mask_side"),
    )
    ev = _parse_visual_evidence(item.get("visual_evidence"))
    if not ev.as_text() and fallback_seg is not None:
        ev = fallback_seg.visual_evidence
    try:
        conf = float(item.get("confidence", fallback_seg.confidence if fallback_seg else 0.88))
    except (TypeError, ValueError):
        conf = 0.88
    conf = max(0.0, min(1.0, conf))

    return QcSegment(
        seg_id=seg_id,
        t_start=round(ts, 3),
        t_end=round(te, 3),
        label_id=eff_lbl,
        name=lbl_def.name if lbl_def else eff_lbl,
        training_role=role,
        splice_policy=policy,
        mask_side=mask_side,
        affected_arm=arm,
        visual_evidence=ev,
        vlm_raw_policy=policy,
        is_unknown=(eff_lbl == "unknown_anomaly"),
        unknown_note=unknown_note,
        suggested_label=suggested,
        confidence=conf,
    )


def run_stage4a_merge(
    view: EpisodeView | None = None,
    *,
    episode_view: EpisodeView | None = None,
    candidates: Sequence[QcSegment] | None = None,
    segments: Sequence[QcSegment] | None = None,
    raw_segments: Sequence[QcSegment] | None = None,
    task: TaskSpec | None = None,
    task_spec: TaskSpec | None = None,
    macro_stages: Sequence[MacroStage] = (),
    client: VlmClient | None = None,
    vlm_client: VlmClient | None = None,
    cfg: ConfigManager | dict[str, Any] | None = None,
    config: ConfigManager | dict[str, Any] | None = None,
    prompt_builder: PromptBuilder | None = None,
    labels: LabelRegistry | None = None,
    budget: BudgetTracker | None = None,
    model_override: str | None = None,
) -> list[QcSegment]:
    """Stage 4A global deduplication & merge with short-circuit and deterministic overlap fallback."""
    ev = view or episode_view
    cl = client or vlm_client
    raw = list(candidates if candidates is not None else (segments if segments is not None else (raw_segments or [])))
    if not raw:
        return []

    check_client_cancelled(cl)
    ordered_raw = sorted(raw, key=lambda s: (s.t_start, s.t_end))
    if not _needs_s4a_vlm_merge(ordered_raw, gap_threshold_s=2.0):
        return list(ordered_raw)

    cm = cfg if isinstance(cfg, ConfigManager) else (config if isinstance(config, ConfigManager) else None)
    t_spec = task or task_spec or (cm.get_task(ev.meta.task_id) if cm and ev else TaskSpec(task_id="_default"))
    reg = labels or (cm.labels if cm else LabelRegistry())
    dur = float(ev.duration_s) if ev is not None else max(s.t_end for s in raw)
    assert cl is not None

    s4_cfg = _resolve_s4_cfg(cfg or config)
    thinking = bool(s4_cfg.get("thinking", False))
    merge_budget = int(s4_cfg.get("merge_thinking_budget", 1536))
    merge_max_tokens = int(s4_cfg.get("merge_max_tokens", 6144))

    pb = prompt_builder or (PromptBuilder(cm) if cm is not None else None)
    if pb is not None:
        sys_p, usr_p = pb.build_s4a(
            task=t_spec, duration_s=dur, macro_stages=macro_stages, raw_candidates=raw
        )
    else:
        import json

        sys_p = 'You are an embodied QC global segment consolidation expert (Stage 4A). Output strict JSON only: {"merged_segments": [...]}.'
        usr_p = (
            f"Task: {t_spec.task_id} (duration {dur:.2f}s)\n"
            f"Stage 3 raw candidates: {json.dumps([s.model_dump() for s in raw], ensure_ascii=False)}"
        )

    check_client_cancelled(cl)
    try:
        chat_kwargs: dict[str, Any] = {
            "stage": "s4a_merge",
            "system_prompt": sys_p,
            "user_prompt": usr_p,
            "thinking": thinking,
            "thinking_budget": merge_budget,
            "max_tokens": merge_max_tokens,
            "budget": budget,
        }
        if model_override is not None:
            chat_kwargs["model_override"] = model_override
        resp = cl.chat_json(**chat_kwargs)
        merged_raw = resp["merged_segments"] if "merged_segments" in resp else resp.get("segments")
        if not isinstance(merged_raw, list):
            raise ValueError("Missing merged_segments list in S4A response")
        out: list[QcSegment] = []
        for idx, item in enumerate(merged_raw):
            if not isinstance(item, dict):
                continue
            seg = _build_segment_from_dict(
                item,
                seg_id=f"s4a_{idx+1}",
                duration_s=dur,
                reg=reg,
            )
            if seg is not None:
                out.append(seg)
        out.sort(key=lambda s: (s.t_start, s.t_end))
        return out
    except (
        BudgetExceeded,
        VlmCancelledError,
        VlmCircuitOpenError,
        VlmBlindVisionError,
    ):
        raise
    except Exception:
        check_client_cancelled(cl)
        min_len = 0.35 if dur >= 1.5 else 0.1
        return _deterministic_merge_raw_segments(raw, min_len_s=min_len, duration_s=dur)


merge_candidates = run_stage4a_merge


def _resolve_overlaps(
    segments: list[QcSegment],
    *,
    min_len_s: float = 0.3,
    duration_s: float | None = None,
    min_edge_len_s: float = 0.1,
) -> list[QcSegment]:
    if not segments:
        return []
    ordered = sorted(segments, key=lambda s: (s.t_start, s.t_end))
    non_overlap: list[QcSegment] = []
    for cur in ordered:
        cur_min = (
            min(min_len_s, min_edge_len_s)
            if _is_short_edge_boundary_idle(
                cur, duration_s=duration_s, min_edge_len_s=min_edge_len_s
            )
            else min_len_s
        )
        if round(cur.t_end - cur.t_start, 6) < cur_min:
            continue
        if not non_overlap:
            non_overlap.append(cur)
            continue
        prev = non_overlap[-1]
        if cur.t_start < prev.t_end:
            prev_min = (
                min(min_len_s, min_edge_len_s)
                if _is_short_edge_boundary_idle(
                    prev, duration_s=duration_s, min_edge_len_s=min_edge_len_s
                )
                else min_len_s
            )
            mid = round(0.5 * (prev.t_end + cur.t_start), 3)
            new_prev_end = min(prev.t_end, max(round(prev.t_start + prev_min, 3), mid))
            new_cur_start = new_prev_end
            if round(cur.t_end - new_cur_start, 6) < cur_min:
                continue
            non_overlap[-1] = prev.model_copy(update={"t_end": new_prev_end})
            cur = cur.model_copy(update={"t_start": new_cur_start})
        non_overlap.append(cur)
    return non_overlap


def run_stage4b_verify(
    view: EpisodeView | None = None,
    *,
    episode_view: EpisodeView | None = None,
    candidates: Sequence[QcSegment] | None = None,
    segments: Sequence[QcSegment] | None = None,
    raw_segments: Sequence[QcSegment] | None = None,
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
    model_override: str | None = None,
) -> list[QcSegment]:
    """Stage 4B 0.5s dense Start-vs-End CoV verification & boundary refinement."""
    ev = view or episode_view
    cl = client or vlm_client
    cands = list(
        candidates
        if candidates is not None
        else (segments if segments is not None else (raw_segments or []))
    )
    if not cands:
        return []

    cm = cfg if isinstance(cfg, ConfigManager) else (config if isinstance(config, ConfigManager) else None)
    t_spec = task or task_spec or (cm.get_task(ev.meta.task_id) if cm and ev else TaskSpec(task_id="_default"))
    reg = labels or (cm.labels if cm else LabelRegistry())
    fs_list = few_shots if few_shots is not None else few_shot_items
    assert ev is not None and cl is not None

    s4_cfg = _resolve_s4_cfg(cfg or config)
    margin_s = float(s4_cfg.get("refine_margin_s", 1.5))
    step_s = float(s4_cfg.get("refine_step_s", 0.5))
    max_frames = int(s4_cfg.get("refine_max_frames", 18))
    cell_w = int(s4_cfg.get("cell_w", 384))
    cell_h = int(s4_cfg.get("cell_h", 216))
    quality = int(s4_cfg.get("quality", 76))
    thinking = bool(s4_cfg.get("thinking", False))
    refine_budget = int(s4_cfg.get("refine_thinking_budget", 1280))
    refine_max_tokens = int(s4_cfg.get("refine_max_tokens", 1536))

    dur = float(ev.duration_s)
    roles = ev.camera_roles
    pb = prompt_builder or (PromptBuilder(cm) if cm is not None else None)
    edge_idle_kin_skip = _resolve_edge_idle_kin_skip(cfg or config)
    edge_tol_s = _resolve_boundary_edge_tol_s(cfg or config)

    def _verify_one(idx: int, cand: QcSegment) -> QcSegment | None:
        check_client_cancelled(cl)
        if edge_idle_kin_skip and ev.kinematics is not None:
            kin_verified = _try_kinematics_verify_edge_boundary_idle(
                cand,
                view=ev,
                duration_s=dur,
                edge_tol_s=edge_tol_s,
            )
            if kin_verified is not None:
                return kin_verified
        w0 = round(max(0.0, cand.t_start - margin_s), 3)
        w1 = round(min(dur, cand.t_end + margin_s), 3)
        span = max(0.05, w1 - w0)
        n_frames = min(max(2, max_frames), max(2, int(np.floor(span / max(0.1, step_s))) + 1))
        t_list = sorted({round(float(t), 3) for t in np.linspace(w0, w1, n_frames)})
        tiles: list[np.ndarray] = []
        for t in t_list:
            check_client_cancelled(cl)
            tiles.append(
                compose_2x2_tile(
                    ev.media, roles, t, cell_w=cell_w, cell_h=cell_h, include_header=True
                )
            )
        check_client_cancelled(cl)
        kin_summary = format_window_kinematics_summary(ev.kinematics, w0, w1)
        if pb is not None:
            sys_p, usr_p = pb.build_s4b(
                task=t_spec,
                camera_roles=roles,
                macro_stages=macro_stages,
                candidate=cand,
                window_start_s=w0,
                window_end_s=w1,
                few_shots=fs_list,
                kinematics_summary=kin_summary,
            )
        else:
            sys_p = "You are an embodied QC segment verification expert (Stage 4B dense start-vs-end boundary refinement)."
            usr_p = (
                f"Task: {t_spec.task_id} | Candidate {cand.seg_id}: "
                f"[{cand.t_start:.2f}s, {cand.t_end:.2f}s], initial label={cand.label_id}\n"
                f"Dense refinement window: [{w0:.2f}s, {w1:.2f}s]\n"
            )
            if kin_summary:
                usr_p += f"[Local Window 30Hz Objective Kinematics Telemetry]:\n{kin_summary}\n"
            usr_p += 'Output JSON: {"keep_segment": bool, ...}'
        try:
            chat_kwargs: dict[str, Any] = {
                "stage": "s4b_verify",
                "system_prompt": sys_p,
                "user_prompt": usr_p,
                "images": tiles,
                "timestamps_s": t_list,
                "thinking": thinking,
                "thinking_budget": refine_budget,
                "max_tokens": refine_max_tokens,
                "jpeg_quality": quality,
                "budget": budget,
            }
            if model_override is not None:
                chat_kwargs["model_override"] = model_override
            resp = cl.chat_json(**chat_kwargs)
        except (
            BudgetExceeded,
            VlmCancelledError,
            VlmCircuitOpenError,
            VlmBlindVisionError,
            VlmError,
        ):
            raise
        except Exception:
            check_client_cancelled(cl)
            return cand

        keep_val = resp.get("keep_segment", resp.get("confirmed", True))
        if keep_val is False:
            return None

        refined = _build_segment_from_dict(
            resp,
            seg_id=cand.seg_id,
            duration_s=dur,
            reg=reg,
            fallback_seg=cand,
        )
        return refined if refined is not None else cand

    verified: list[QcSegment] = []
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(cands)))) as pool:
        results = list(pool.map(lambda p: _verify_one(p[0], p[1]), enumerate(cands)))
    for res in results:
        if res is not None:
            verified.append(res)

    min_len = 0.3 if dur >= 1.5 else 0.1
    resolved = _resolve_overlaps(verified, min_len_s=min_len, duration_s=dur)
    # Assign clean sequential seg_ids
    return [
        seg.model_copy(update={"seg_id": f"seg_{idx+1:03d}"})
        for idx, seg in enumerate(resolved)
    ]


def run_stage4_verify(
    view: EpisodeView | None = None,
    *,
    episode_view: EpisodeView | None = None,
    candidates: Sequence[QcSegment] | None = None,
    segments: Sequence[QcSegment] | None = None,
    raw_segments: Sequence[QcSegment] | None = None,
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
    model_override: str | None = None,
) -> list[QcSegment]:
    """Run Stage 4A merge followed by Stage 4B dense Start-vs-End CoV refinement."""
    cands = list(
        candidates
        if candidates is not None
        else (segments if segments is not None else (raw_segments or []))
    )
    if not cands:
        return []
    merged = run_stage4a_merge(
        view=view,
        episode_view=episode_view,
        candidates=cands,
        task=task,
        task_spec=task_spec,
        macro_stages=macro_stages,
        client=client,
        vlm_client=vlm_client,
        cfg=cfg,
        config=config,
        prompt_builder=prompt_builder,
        labels=labels,
        budget=budget,
        model_override=model_override,
    )
    if not merged:
        return []
    check_client_cancelled(client or vlm_client)
    return run_stage4b_verify(
        view=view,
        episode_view=episode_view,
        candidates=merged,
        task=task,
        task_spec=task_spec,
        macro_stages=macro_stages,
        few_shots=few_shots,
        few_shot_items=few_shot_items,
        client=client,
        vlm_client=vlm_client,
        cfg=cfg,
        config=config,
        prompt_builder=prompt_builder,
        labels=labels,
        budget=budget,
        model_override=model_override,
    )


run_s4 = run_stage4_verify
