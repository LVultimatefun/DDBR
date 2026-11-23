"""Pure-video fallback visual gate (SSIM + L1 diff) when KinematicsTrack is None."""

from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np

from ..config import ConfigManager
from ..io.base import MediaSource
from ..models.annotation import GuardrailMetrics, GuardrailReport, QcSegment


def _extract_visual_cfg(
    cfg: ConfigManager | dict[str, Any] | None
) -> tuple[dict[str, Any], float]:
    edge_tol = 0.5
    if isinstance(cfg, ConfigManager):
        g = cfg.settings.get("guardrail", {})
        edge_tol = float(g.get("splice_gate", {}).get("boundary_edge_tol_s", 0.5))
        return dict(g.get("visual_gate", {})), edge_tol
    if isinstance(cfg, dict):
        if "guardrail" in cfg and isinstance(cfg["guardrail"], dict):
            g = cfg["guardrail"]
            edge_tol = float(g.get("splice_gate", {}).get("boundary_edge_tol_s", 0.5))
            return dict(g.get("visual_gate", {})), edge_tol
        if "visual_gate" in cfg and isinstance(cfg["visual_gate"], dict):
            edge_tol = float(cfg.get("splice_gate", {}).get("boundary_edge_tol_s", 0.5))
            return dict(cfg["visual_gate"]), edge_tol
        return dict(cfg), edge_tol
    return {}, edge_tol


def compute_ssim_and_l1(img_a: np.ndarray, img_b: np.ndarray) -> tuple[float, float]:
    """Compute structural similarity (SSIM) and normalized L1 diff in [0, 1]."""
    a = np.asarray(img_a, dtype=np.float64)
    b = np.asarray(img_b, dtype=np.float64)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)

    l1_diff = float(np.mean(np.abs(a - b)) / 255.0)

    ga = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64) if a.ndim == 3 else a
    gb = cv2.cvtColor(b.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64) if b.ndim == 3 else b

    h, w = ga.shape[:2]
    k = min(7, h if h % 2 == 1 else max(1, h - 1), w if w % 2 == 1 else max(1, w - 1))
    if k < 3:
        mu_a = float(np.mean(ga))
        mu_b = float(np.mean(gb))
        var_a = float(np.var(ga))
        var_b = float(np.var(gb))
        cov = float(np.mean((ga - mu_a) * (gb - mu_b)))
        c1 = (0.01 * 255.0) ** 2
        c2 = (0.03 * 255.0) ** 2
        ssim = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
            (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
        )
        return float(max(-1.0, min(1.0, ssim))), l1_diff

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    sigma = 1.5
    mu1 = cv2.GaussianBlur(ga, (k, k), sigma)
    mu2 = cv2.GaussianBlur(gb, (k, k), sigma)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = np.maximum(0.0, cv2.GaussianBlur(ga * ga, (k, k), sigma) - mu1_sq)
    sigma2_sq = np.maximum(0.0, cv2.GaussianBlur(gb * gb, (k, k), sigma) - mu2_sq)
    sigma12 = cv2.GaussianBlur(ga * gb, (k, k), sigma) - mu1_mu2

    num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = num / np.maximum(den, 1e-12)
    ssim = float(np.mean(ssim_map))
    return float(max(-1.0, min(1.0, ssim))), l1_diff


def evaluate_visual_segment(
    seg: QcSegment,
    *,
    media: MediaSource | None,
    cfg: ConfigManager | dict[str, Any] | None = None,
    config: ConfigManager | dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
    duration_s: float | None = None,
) -> tuple[QcSegment, float | None]:
    """Evaluate a single segment with pure-video visual gate, returning (updated_seg, l1_diff)."""
    if media is None or not media.cameras or seg.splice_policy != "cut_clean":
        return seg, None

    v_cfg, edge_tol_s = _extract_visual_cfg(cfg or config or settings)
    min_ssim = float(v_cfg.get("min_crop_ssim", 0.85))
    max_l1 = float(v_cfg.get("max_l1_diff", 0.12))
    dur = float(duration_s) if duration_s is not None else float(media.duration_s)

    if seg.label_id == "boundary_idle" and (
        seg.t_start <= edge_tol_s + 1e-6 or (seg.t_end + 1e-6) >= max(0.0, dur - edge_tol_s)
    ):
        return (
            seg.model_copy(
                update={
                    "guardrail": GuardrailReport(
                        gate_type="visual_gate",
                        passed=True,
                        downgraded=False,
                        metrics=seg.guardrail.metrics,
                        opt_t_start=seg.t_start,
                        opt_t_end=seg.t_end,
                    )
                }
            ),
            None,
        )

    main_cam = next(
        (c.name for c in media.cameras if c.role == "main"),
        media.cameras[0].name,
    )
    fr_a, fr_b = media.read_frames_batch(main_cam, [seg.t_start, seg.t_end])
    ssim, l1_diff = compute_ssim_and_l1(fr_a, fr_b)
    metrics = GuardrailMetrics(
        **{
            **seg.guardrail.metrics.model_dump(),
            "visual_ssim": round(ssim, 4),
        }
    )
    if ssim >= min_ssim and l1_diff <= max_l1:
        updated = seg.model_copy(
            update={
                "guardrail": GuardrailReport(
                    gate_type="visual_gate",
                    passed=True,
                    downgraded=False,
                    metrics=metrics,
                    opt_t_start=seg.t_start,
                    opt_t_end=seg.t_end,
                )
            }
        )
    else:
        updated = seg.model_copy(
            update={
                "splice_policy": "mask_loss",
                "mask_side": seg.affected_arm,
                "guardrail": GuardrailReport(
                    gate_type="visual_gate",
                    passed=False,
                    downgraded=True,
                    downgrade_reason=(
                        f"visual_gate: ssim={ssim:.3f} (min={min_ssim:.2f}), "
                        f"l1_diff={l1_diff:.3f} (max={max_l1:.2f})"
                    ),
                    metrics=metrics,
                    opt_t_start=seg.t_start,
                    opt_t_end=seg.t_end,
                ),
            }
        )
    return updated, float(l1_diff)


def apply_visual_gate(
    segments: Sequence[QcSegment] | None = None,
    *,
    qc_segments: Sequence[QcSegment] | None = None,
    media: MediaSource | None = None,
    cfg: ConfigManager | dict[str, Any] | None = None,
    config: ConfigManager | dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
    duration_s: float | None = None,
) -> list[QcSegment]:
    """Pure-video splice gate using start-vs-end SSIM and normalized L1 diff."""
    segs = list(segments if segments is not None else (qc_segments or []))
    if media is None or not media.cameras:
        return segs

    eff_cfg = cfg or config or settings
    dur = float(duration_s) if duration_s is not None else float(media.duration_s)

    out: list[QcSegment] = []
    for seg in segs:
        if seg.splice_policy != "cut_clean":
            out.append(seg)
            continue
        updated, _ = evaluate_visual_segment(
            seg,
            media=media,
            cfg=eff_cfg,
            duration_s=dur,
        )
        out.append(updated)
    return out


run_visual_gate = apply_visual_gate
check_visual_gate = apply_visual_gate


class VisualGate:
    """OOP wrapper around `apply_visual_gate`."""

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
        media: MediaSource | None = None,
        duration_s: float | None = None,
    ) -> list[QcSegment]:
        return apply_visual_gate(
            segments=segments,
            qc_segments=qc_segments,
            media=media,
            cfg=self.cfg,
            duration_s=duration_s,
        )

    check = apply
    verify = apply
