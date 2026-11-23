"""Mild anti-hallucination verifier: overrides idle labels contradicted by >15cm monotonic motion."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..config import ConfigManager
from ..io.base import KinematicsTrack
from ..models.annotation import GuardrailReport, QcSegment

_IDLE_LABELS = frozenset({"mid_idle_freeze", "boundary_idle"})


def _extract_hallu_cfg(cfg: ConfigManager | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(cfg, ConfigManager):
        return dict(cfg.settings.get("guardrail", {}).get("hallucination", {}))
    if isinstance(cfg, dict):
        if "guardrail" in cfg and isinstance(cfg["guardrail"], dict):
            return dict(cfg["guardrail"].get("hallucination", {}))
        if "hallucination" in cfg and isinstance(cfg["hallucination"], dict):
            return dict(cfg["hallucination"])
        return dict(cfg)
    return {}


def apply_hallucination_check(
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
    """Override idle_freeze/boundary_idle -> keep_positive if all arms move monotonically > idle_arm_disp_m."""
    segs = list(segments if segments is not None else (qc_segments or []))
    k = kinematics or kin
    h_cfg = _extract_hallu_cfg(cfg or config or settings)
    if not bool(h_cfg.get("enabled", True)):
        return segs
    if k is None or k.eef_pos is None:
        return segs

    disp_thresh = float(h_cfg.get("idle_arm_disp_m", 0.15))
    ts = np.asarray(k.timestamps_s, dtype=np.float64)
    eef_pos = np.asarray(k.eef_pos, dtype=np.float64)
    T = len(ts)
    if T < 2 or eef_pos.ndim != 3 or eef_pos.shape[1] < 1:
        return segs
    raw_vm = getattr(k, "valid_mask", None)
    valid_mask = (
        np.asarray(raw_vm, dtype=bool)
        if raw_vm is not None and len(raw_vm) == T
        else np.ones(T, dtype=bool)
    )

    out: list[QcSegment] = []
    for seg in segs:
        if seg.label_id not in _IDLE_LABELS or seg.splice_policy == "keep_positive":
            out.append(seg)
            continue

        i0 = int(np.clip(np.searchsorted(ts, seg.t_start, side="left"), 0, T - 1))
        i1 = int(np.clip(np.searchsorted(ts, seg.t_end, side="right"), i0 + 1, T))
        pos_seg = eef_pos[i0:i1]
        seg_vm = valid_mask[i0:i1]
        if len(pos_seg) < 2:
            out.append(seg)
            continue

        n_arms = pos_seg.shape[1]
        arm_nets: list[float] = []
        arm_mono: list[bool] = []
        for a in range(n_arms):
            arm_raw = pos_seg[:, a, :]
            arm_ok = np.isfinite(arm_raw).all(axis=-1) & seg_vm
            traj = arm_raw[arm_ok]
            if len(traj) < 2:
                arm_nets.append(0.0)
                arm_mono.append(False)
                continue
            net = float(np.linalg.norm(traj[-1] - traj[0]))
            path = float(np.linalg.norm(np.diff(traj, axis=0), axis=-1).sum())
            arm_nets.append(net)
            arm_mono.append(net > disp_thresh and net >= 0.5 * max(path, 1e-9))

        if arm_mono and all(arm_mono):
            min_disp = min(arm_nets)
            new_role = seg.training_role if seg.training_role in {"necessary", "recovery"} else "necessary"
            out.append(
                seg.model_copy(
                    update={
                        "training_role": new_role,
                        "splice_policy": "keep_positive",
                        "mask_side": None,
                        "guardrail": GuardrailReport(
                            gate_type="hallucination",
                            passed=True,
                            downgraded=False,
                            guardrail_override=True,
                            override_reason=(
                                f"Monotonic dual-arm motion ({min_disp:.3f}m > {disp_thresh:.2f}m) "
                                f"contradicts {seg.label_id}"
                            ),
                            metrics=seg.guardrail.metrics,
                        ),
                    }
                )
            )
        else:
            out.append(seg)

    return out


run_hallucination_guardrail = apply_hallucination_check
verify_hallucination = apply_hallucination_check


class HallucinationGuardrail:
    """OOP wrapper around `apply_hallucination_check`."""

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
        return apply_hallucination_check(
            segments=segments,
            qc_segments=qc_segments,
            kinematics=kinematics,
            kin=kin,
            cfg=self.cfg,
            duration_s=duration_s,
        )

    check = apply
    verify = apply
