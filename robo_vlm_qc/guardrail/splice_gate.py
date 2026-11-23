"""Pose-closure 2D grid splice optimizer + bimanual coupling guardrail."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from ..config import ConfigManager
from ..io.base import KinematicsTrack
from ..models.annotation import GuardrailMetrics, GuardrailReport, QcSegment


def _extract_splice_cfg(cfg: ConfigManager | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(cfg, ConfigManager):
        return dict(cfg.settings.get("guardrail", {}).get("splice_gate", {}))
    if isinstance(cfg, dict):
        if "guardrail" in cfg and isinstance(cfg["guardrail"], dict):
            return dict(cfg["guardrail"].get("splice_gate", {}))
        if "splice_gate" in cfg and isinstance(cfg["splice_gate"], dict):
            return dict(cfg["splice_gate"])
        return dict(cfg)
    return {}


def _rotvec_geodesic_mat(
    rot_i: np.ndarray, rot_j: np.ndarray
) -> np.ndarray:
    """Compute max SO(3) geodesic angle (rad) across arms between (S, arms, 3) and (E, arms, 3)."""
    S, n_arms, _ = rot_i.shape
    E = rot_j.shape[0]
    if n_arms == 0 or S == 0 or E == 0:
        return np.zeros((S, E), dtype=np.float64)
    fin_i = np.isfinite(rot_i).all(axis=-1)
    fin_j = np.isfinite(rot_j).all(axis=-1)
    safe_i = np.where(fin_i[:, :, None], rot_i, 0.0)
    safe_j = np.where(fin_j[:, :, None], rot_j, 0.0)
    q_i = np.empty((S, n_arms, 4), dtype=np.float64)
    q_j = np.empty((E, n_arms, 4), dtype=np.float64)
    for a in range(n_arms):
        q_i[:, a, :] = Rotation.from_rotvec(safe_i[:, a, :]).as_quat()
        q_j[:, a, :] = Rotation.from_rotvec(safe_j[:, a, :]).as_quat()
    # q_i: (S, 1, n_arms, 4), q_j: (1, E, n_arms, 4)
    dots = np.abs(np.sum(q_i[:, None, :, :] * q_j[None, :, :, :], axis=-1))
    dots = np.clip(dots, 0.0, 1.0)
    angles = 2.0 * np.arccos(dots)
    fin_pair = fin_i[:, None, :] & fin_j[None, :, :]
    angles = np.where(fin_pair, angles, np.nan)
    return angles.max(axis=-1)


def _angle_diff_abs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    return np.abs(np.arctan2(np.sin(d), np.cos(d)))


def apply_splice_gate(
    segments: Sequence[QcSegment] | None = None,
    *,
    qc_segments: Sequence[QcSegment] | None = None,
    kinematics: KinematicsTrack | None = None,
    kin: KinematicsTrack | None = None,
    cfg: ConfigManager | dict[str, Any] | None = None,
    config: ConfigManager | dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
    duration_s: float | None = None,
) -> list[QcSegment]:
    """Verify cut_clean candidates via 2D pose-closure grid search + bimanual coupling check."""
    segs = list(segments if segments is not None else (qc_segments or []))
    k = kinematics or kin
    if k is None:
        return segs

    g_cfg = _extract_splice_cfg(cfg or config or settings)
    margin_s = float(g_cfg.get("search_margin_s", 1.0))
    edge_tol_s = float(g_cfg.get("boundary_edge_tol_s", 0.5))
    max_pos = float(g_cfg.get("max_pos_jump_m", 0.02))
    max_rot = float(g_cfg.get("max_rot_jump_rad", 0.10))
    max_hand = float(g_cfg.get("max_hand_jump_rad", 0.30))
    max_torso = float(g_cfg.get("max_torso_jump_rad", 0.05))
    max_ch_pos = float(g_cfg.get("max_chassis_pos_jump_m", 0.02))
    max_ch_yaw = float(g_cfg.get("max_chassis_yaw_jump_rad", 0.05))
    max_other_arm = float(g_cfg.get("max_other_arm_disp_m", 0.03))

    ts = np.asarray(k.timestamps_s, dtype=np.float64)
    T = len(ts)
    if T < 4:
        return segs
    dur = float(duration_s) if duration_s is not None else float(ts[-1])
    dt_med = float(np.median(np.diff(ts))) if T > 1 else (1.0 / 30.0)
    fps = 1.0 / max(1e-4, dt_med)
    m_frames = max(1, int(round(margin_s * fps)))

    eef_pos = np.asarray(k.eef_pos, dtype=np.float64) if k.eef_pos is not None else None
    eef_rot = np.asarray(k.eef_rotvec, dtype=np.float64) if k.eef_rotvec is not None else None
    hand_q = np.asarray(k.hand_q, dtype=np.float64) if k.hand_q is not None else None
    torso_q = np.asarray(k.torso_q, dtype=np.float64) if k.torso_q is not None else None
    chassis = np.asarray(k.chassis_pose, dtype=np.float64) if k.chassis_pose is not None else None
    raw_vm = getattr(k, "valid_mask", None)
    if raw_vm is not None and len(raw_vm) == T:
        valid_mask = np.asarray(raw_vm, dtype=bool)
    else:
        valid_mask = np.ones(T, dtype=bool)

    out: list[QcSegment] = []
    for seg in segs:
        if seg.splice_policy != "cut_clean":
            out.append(seg)
            continue

        # Boundary idle at trajectory edges passes directly
        if seg.label_id == "boundary_idle" and (
            seg.t_start <= edge_tol_s + 1e-6 or (seg.t_end + 1e-6) >= max(0.0, dur - edge_tol_s)
        ):
            out.append(
                seg.model_copy(
                    update={
                        "guardrail": GuardrailReport(
                            gate_type="splice_gate",
                            passed=True,
                            downgraded=False,
                            metrics=seg.guardrail.metrics,
                            opt_t_start=seg.t_start,
                            opt_t_end=seg.t_end,
                        )
                    }
                )
            )
            continue

        start_i = int(np.clip(np.searchsorted(ts, seg.t_start, side="left"), 0, T - 1))
        end_i = int(np.clip(np.searchsorted(ts, seg.t_end, side="right"), start_i + 1, T))

        if start_i == 0:
            s_cands = np.array([0], dtype=np.int64)
        else:
            s_hi = min(T - 3, start_i)
            s_lo = min(s_hi, max(0, start_i - m_frames))
            s_cands = np.arange(s_lo, s_hi + 1, dtype=np.int64)

        if end_i >= T:
            e_cands = np.array([T], dtype=np.int64)
        else:
            e_lo = max(3, end_i)
            e_hi = min(T, end_i + m_frames)
            e_cands = np.arange(e_lo, max(e_lo, e_hi) + 1, dtype=np.int64)

        valid = (e_cands[None, :] - s_cands[:, None]) > 2
        if not valid.any():
            s_fb = min(start_i, T - 3)
            e_fb = min(T, max(s_fb + 3, end_i))
            s_cands = np.array([s_fb], dtype=np.int64)
            e_cands = np.array([e_fb], dtype=np.int64)
            valid = np.ones((1, 1), dtype=bool)

        i_idx = np.maximum(0, s_cands - 1)
        j_idx = np.minimum(T - 1, e_cands)
        S, E = len(s_cands), len(e_cands)

        p_mat = (
            np.linalg.norm(
                eef_pos[j_idx][None, :, :, :] - eef_pos[i_idx][:, None, :, :],
                axis=-1,
            ).max(axis=-1)
            if (eef_pos is not None and eef_pos.ndim == 3 and eef_pos.shape[1] > 0)
            else np.zeros((S, E), dtype=np.float64)
        )
        r_mat = (
            _rotvec_geodesic_mat(eef_rot[i_idx], eef_rot[j_idx])
            if (eef_rot is not None and eef_rot.ndim == 3 and eef_rot.shape[1] > 0)
            else np.zeros((S, E), dtype=np.float64)
        )
        h_mat = (
            np.abs(hand_q[j_idx][None, :, :] - hand_q[i_idx][:, None, :]).max(axis=-1)
            if (hand_q is not None and hand_q.shape[1] > 0)
            else np.zeros((S, E), dtype=np.float64)
        )
        t_mat = (
            np.abs(torso_q[j_idx][None, :, :] - torso_q[i_idx][:, None, :]).max(axis=-1)
            if (torso_q is not None and torso_q.shape[1] > 0)
            else np.zeros((S, E), dtype=np.float64)
        )
        if chassis is not None and chassis.shape[1] >= 2:
            cp_mat = np.linalg.norm(
                chassis[j_idx, :2][None, :, :] - chassis[i_idx, :2][:, None, :],
                axis=-1,
            )
            cy_mat = (
                _angle_diff_abs(chassis[i_idx, 2][:, None], chassis[j_idx, 2][None, :])
                if chassis.shape[1] >= 3
                else np.zeros((S, E), dtype=np.float64)
            )
        else:
            cp_mat = np.zeros((S, E), dtype=np.float64)
            cy_mat = np.zeros((S, E), dtype=np.float64)

        pair_mask_valid = valid_mask[i_idx][:, None] & valid_mask[j_idx][None, :]
        finite_mats = (
            np.isfinite(p_mat)
            & np.isfinite(r_mat)
            & np.isfinite(h_mat)
            & np.isfinite(t_mat)
            & np.isfinite(cp_mat)
            & np.isfinite(cy_mat)
        )
        cand_valid = valid & pair_mask_valid & finite_mats

        if cand_valid.any():
            feasible = (
                cand_valid
                & (p_mat <= max_pos)
                & (r_mat <= max_rot)
                & (h_mat <= max_hand)
                & (t_mat <= max_torso)
                & (cp_mat <= max_ch_pos)
                & (cy_mat <= max_ch_yaw)
            )

            shift_pen = (
                1e-4
                * (np.abs(s_cands[:, None] - start_i) + np.abs(e_cands[None, :] - end_i))
                / max(1.0, fps)
            )
            cost = (
                (p_mat / max(1e-6, max_pos))
                + 0.35 * (r_mat / max(1e-6, max_rot))
                + 0.25 * (h_mat / max(1e-6, max_hand))
                + 0.20 * (t_mat / max(1e-6, max_torso))
                + 0.20 * (cp_mat / max(1e-6, max_ch_pos))
                + 0.20 * (cy_mat / max(1e-6, max_ch_yaw))
                + shift_pen
                - np.where(feasible, 10.0, 0.0)
            )
            cost = np.where(cand_valid, cost, math.inf)
            flat_best = int(np.argmin(cost))
            bs_idx, be_idx = divmod(flat_best, E)
            opt_s = int(s_cands[bs_idx])
            opt_e = int(e_cands[be_idx])
            is_feasible = bool(feasible[bs_idx, be_idx])

            p_val = float(p_mat[bs_idx, be_idx])
            r_val = float(r_mat[bs_idx, be_idx])
            h_val = float(h_mat[bs_idx, be_idx])
            t_val = float(t_mat[bs_idx, be_idx])
            cp_val = float(cp_mat[bs_idx, be_idx])
            cy_val = float(cy_mat[bs_idx, be_idx])
            all_invalid = False
        else:
            opt_s = min(start_i, T - 3)
            opt_e = min(T, max(opt_s + 3, end_i))
            is_feasible = False
            p_val = r_val = h_val = t_val = cp_val = cy_val = 0.0
            all_invalid = True

        # Bimanual coupling check on [opt_s, opt_e)
        other_disp = 0.0
        coupled = False
        if not all_invalid and eef_pos is not None and eef_pos.shape[1] >= 2:
            sl_e = min(T, max(opt_s + 1, opt_e))
            pos_seg = eef_pos[opt_s:sl_e]
            seg_vm = valid_mask[opt_s:sl_e]
            if len(pos_seg) >= 2:
                if seg.affected_arm == "left":
                    act_idx = 0
                elif seg.affected_arm == "right":
                    act_idx = 1
                else:
                    arm_paths: list[float] = []
                    for a in range(pos_seg.shape[1]):
                        arm_fin = np.isfinite(pos_seg[:, a, :]).all(axis=-1) & seg_vm
                        arm_pts = pos_seg[arm_fin, a, :]
                        if len(arm_pts) >= 2:
                            arm_paths.append(
                                float(np.linalg.norm(np.diff(arm_pts, axis=0), axis=-1).sum())
                            )
                        else:
                            arm_paths.append(0.0)
                    act_idx = int(np.argmax(arm_paths))
                other_seg = pos_seg[:, 1 - act_idx, :]
                fin_other = np.isfinite(other_seg).all(axis=-1) & seg_vm
                other_clean = other_seg[fin_other]
                if len(other_clean) >= 2:
                    net_disp = float(np.linalg.norm(other_clean[-1] - other_clean[0]))
                    max_exc = float(
                        np.linalg.norm(other_clean - other_clean[0][None, :], axis=-1).max()
                    )
                    other_disp = max(net_disp, max_exc)
                    if not math.isfinite(other_disp):
                        other_disp = 0.0
                    coupled = other_disp >= max_other_arm

        opt_t0 = round(float(ts[opt_s]), 3)
        opt_t1 = round(dur if opt_e >= T else float(ts[opt_e]), 3)
        if opt_t1 <= opt_t0:
            cand_t1 = round(min(dur, opt_t0 + max(0.1, seg.t_end - seg.t_start)), 3)
            if cand_t1 > opt_t0:
                opt_t1 = cand_t1
            elif opt_t1 > 0.0:
                opt_t0 = round(max(0.0, opt_t1 - max(0.05, min(0.1, opt_t1))), 3)

        def _fin(v: float) -> float:
            return round(v, 5) if math.isfinite(v) else 0.0

        metrics = GuardrailMetrics(
            eef_pos_m=_fin(p_val),
            eef_rot_rad=_fin(r_val),
            hand_rad=_fin(h_val),
            torso_rad=_fin(t_val),
            chassis_pos_m=_fin(cp_val),
            chassis_yaw_rad=_fin(cy_val),
            other_arm_disp_m=_fin(other_disp),
            bimanual_coupled=coupled,
            visual_ssim=seg.guardrail.metrics.visual_ssim,
        )

        if is_feasible and not coupled:
            out.append(
                seg.model_copy(
                    update={
                        "t_start": opt_t0,
                        "t_end": opt_t1,
                        "guardrail": GuardrailReport(
                            gate_type="splice_gate",
                            passed=True,
                            downgraded=False,
                            metrics=metrics,
                            opt_t_start=opt_t0,
                            opt_t_end=opt_t1,
                        ),
                    }
                )
            )
        else:
            reasons: list[str] = []
            if p_val > max_pos:
                reasons.append(f"eef_pos_m={p_val:.4f}m>{max_pos:.3f}m")
            if r_val > max_rot:
                reasons.append(f"eef_rot_rad={r_val:.4f}rad>{max_rot:.3f}rad")
            if h_val > max_hand:
                reasons.append(f"hand_rad={h_val:.4f}rad>{max_hand:.3f}rad")
            if t_val > max_torso:
                reasons.append(f"torso_rad={t_val:.4f}rad>{max_torso:.3f}rad")
            if cp_val > max_ch_pos:
                reasons.append(f"chassis_pos_m={cp_val:.4f}m>{max_ch_pos:.3f}m")
            if cy_val > max_ch_yaw:
                reasons.append(f"chassis_yaw_rad={cy_val:.4f}rad>{max_ch_yaw:.3f}rad")
            if coupled:
                reasons.append(
                    f"bimanual_coupled(other_arm_disp_m={other_disp:.4f}m>={max_other_arm:.3f}m)"
                )
            out.append(
                seg.model_copy(
                    update={
                        "splice_policy": "mask_loss",
                        "mask_side": seg.affected_arm,
                        "guardrail": GuardrailReport(
                            gate_type="splice_gate",
                            passed=False,
                            downgraded=True,
                            downgrade_reason="; ".join(reasons) or "pose_closure_failed",
                            metrics=metrics,
                            opt_t_start=opt_t0,
                            opt_t_end=opt_t1,
                        ),
                    }
                )
            )

    return out


run_splice_gate = apply_splice_gate
check_splice_gate = apply_splice_gate


class SpliceGate:
    """OOP wrapper around `apply_splice_gate`."""

    def __init__(
        self,
        cfg: ConfigManager | dict[str, Any] | None = None,
        *,
        config: ConfigManager | dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg or config or settings

    def apply(
        self,
        segments: Sequence[QcSegment] | None = None,
        *,
        qc_segments: Sequence[QcSegment] | None = None,
        kinematics: KinematicsTrack | None = None,
        kin: KinematicsTrack | None = None,
        duration_s: float | None = None,
    ) -> list[QcSegment]:
        return apply_splice_gate(
            segments=segments,
            qc_segments=qc_segments,
            kinematics=kinematics,
            kin=kin,
            cfg=self.cfg,
            duration_s=duration_s,
        )

    check = apply
    verify = apply
