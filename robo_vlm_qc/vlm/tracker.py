"""BAFT: Bidirectional Anchor-Fused Optical-Flow + Multi-Scale NCC Local 2D Tracker.

Pure OpenCV / NumPy / SciPy frame-level 2D object tracker designed to densify sparse
Stage-1 VLM semantic anchor boxes with ZERO internet downloads, low peak RAM (<35 MB
via chunked interval-streaming decode), cooperative cancellation, camera ego-motion
detection, temporal VLM anchor outlier rejection, and occlusion awareness.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import cv2
import numpy as np
from scipy.signal import savgol_filter

from ..io.base import EpisodeView
from ..models.annotation import GroundingKeyframe, GroundingTrack

# Dense frame provenance flags (aligned with GroundingTrack & StudioPlayer.tsx)
FLAG_INVISIBLE = 0
FLAG_TRACKED = 1
FLAG_VLM_ANCHOR = 2
FLAG_STATIC_LOCKED = 3


@dataclass
class TrackingConfig:
    """Configuration knobs for BAFT local frame-level 2D tracking."""

    enabled: bool = True
    track_fps: float = 15.0
    short_native_fps_max_s: float = 15.0
    track_width: int = 320
    flow_preset: str = "ultrafast"
    ncc_stride: int = 2
    ncc_min_score: float = 0.55
    fb_max_err_px: float = 2.5
    occlusion_conf_thresh: float = 0.25
    occlusion_min_frames: int = 4
    smooth_window: int = 5
    camera_motion_threshold_norm: float = 25.0
    outlier_jump_norm: float = 250.0
    adaptive_reanchor_max_calls: int = 0
    adaptive_disagree_ratio: float = 0.65

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "TrackingConfig":
        if not isinstance(raw, dict):
            return cls()
        return cls(
            enabled=bool(raw.get("enabled", True)),
            track_fps=max(1.0, float(raw.get("track_fps", 15.0))),
            short_native_fps_max_s=max(0.0, float(raw.get("short_native_fps_max_s", 15.0))),
            track_width=max(64, int(raw.get("track_width", 320))),
            flow_preset=str(raw.get("flow_preset", "ultrafast")).strip().lower(),
            ncc_stride=max(1, int(raw.get("ncc_stride", 2))),
            ncc_min_score=float(np.clip(float(raw.get("ncc_min_score", 0.55)), 0.0, 0.99)),
            fb_max_err_px=max(0.25, float(raw.get("fb_max_err_px", 2.5))),
            occlusion_conf_thresh=float(
                np.clip(float(raw.get("occlusion_conf_thresh", 0.25)), 0.0, 0.95)
            ),
            occlusion_min_frames=max(1, int(raw.get("occlusion_min_frames", 4))),
            smooth_window=max(1, int(raw.get("smooth_window", 5))),
            camera_motion_threshold_norm=max(
                1.0, float(raw.get("camera_motion_threshold_norm", 25.0))
            ),
            outlier_jump_norm=max(20.0, float(raw.get("outlier_jump_norm", 250.0))),
            adaptive_reanchor_max_calls=max(
                0, int(raw.get("adaptive_reanchor_max_calls", 0))
            ),
            adaptive_disagree_ratio=float(
                np.clip(float(raw.get("adaptive_disagree_ratio", 0.65)), 0.05, 1.0)
            ),
        )


@dataclass
class IntervalDisagreement:
    """Tracks bidirectional forward/backward disagreement on an anchor interval."""

    object_id: str
    a0_idx: int
    a1_idx: int
    dense_i0: int
    dense_i1: int
    t0: float
    t1: float
    disagree_ratio: float

    @property
    def gap_s(self) -> float:
        return max(0.0, self.t1 - self.t0)

    @property
    def mid_dense_idx(self) -> int:
        return (self.dense_i0 + self.dense_i1) // 2

    @property
    def mid_t(self) -> float:
        return round(0.5 * (self.t0 + self.t1), 3)


@dataclass
class BaftTrackingOutput:
    """Result returned by BAFT densification for Stage 1 grounding."""

    track_fps: float
    dense_times_s: list[float]
    camera_motion_detected: bool
    tracks: dict[str, GroundingTrack]
    interval_reports: list[IntervalDisagreement]


def _noop() -> None:
    return None


def _even_dim(val: float, min_val: int = 32) -> int:
    iv = int(round(val))
    iv = max(min_val, (iv // 2) * 2)
    return iv


def _create_dis_flow(preset: str) -> Any:
    p = preset.strip().lower()
    if p == "medium":
        code = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    elif p == "fast":
        code = cv2.DISOPTICAL_FLOW_PRESET_FAST
    else:
        code = cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST
    return cv2.DISOpticalFlow_create(code)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-9 else 0.0


def box_center_dist(a: Sequence[float], b: Sequence[float]) -> float:
    acx = 0.5 * (float(a[0]) + float(a[2]))
    acy = 0.5 * (float(a[1]) + float(a[3]))
    bcx = 0.5 * (float(b[0]) + float(b[2]))
    bcy = 0.5 * (float(b[1]) + float(b[3]))
    return math.hypot(acx - bcx, acy - bcy)


def _norm_to_px(box_norm: Sequence[float], w: int, h: int) -> np.ndarray:
    return np.array(
        [
            float(box_norm[0]) * w / 1000.0,
            float(box_norm[1]) * h / 1000.0,
            float(box_norm[2]) * w / 1000.0,
            float(box_norm[3]) * h / 1000.0,
        ],
        dtype=np.float64,
    )


def _px_to_norm_int(box_px: Sequence[float], w: int, h: int) -> list[int]:
    if w <= 0 or h <= 0:
        return [0, 0, 0, 0]
    x1 = int(round(max(0.0, min(999.0, float(box_px[0]) * 1000.0 / w))))
    y1 = int(round(max(0.0, min(999.0, float(box_px[1]) * 1000.0 / h))))
    x2 = int(round(max(float(x1 + 1), min(1000.0, float(box_px[2]) * 1000.0 / w))))
    y2 = int(round(max(float(y1 + 1), min(1000.0, float(box_px[3]) * 1000.0 / h))))
    return [x1, y1, x2, y2]


def _clamp_box_px(box: np.ndarray, w: int, h: int) -> np.ndarray:
    x1 = float(np.clip(box[0], 0.0, max(0.0, w - 2.0)))
    y1 = float(np.clip(box[1], 0.0, max(0.0, h - 2.0)))
    x2 = float(np.clip(box[2], x1 + 2.0, float(w)))
    y2 = float(np.clip(box[3], y1 + 2.0, float(h)))
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def plan_dense_times(
    duration_s: float,
    native_fps: float,
    anchor_times_s: Sequence[float],
    cfg: TrackingConfig,
) -> tuple[list[float], float, list[int]]:
    """Build sorted dense sample timestamps aligned exactly with all VLM anchors."""
    dur = max(0.01, float(duration_s))
    nat = float(native_fps) if native_fps and native_fps > 0.0 else 30.0
    if dur <= cfg.short_native_fps_max_s:
        eff_fps = nat
    else:
        eff_fps = min(nat, max(1.0, float(cfg.track_fps)))

    n_steps = max(1, int(round(dur * eff_fps)))
    raw_grid = [round(min(dur, i / eff_fps), 3) for i in range(n_steps + 1)]
    grid: list[float] = []
    for t in raw_grid:
        if not grid or t > grid[-1] + 1e-4:
            grid.append(t)
    if not grid:
        grid = [0.0]

    # Snap nearest unpinned grid points to exact anchor timestamps (or insert if needed)
    clean_anchors = [round(max(0.0, min(dur, float(t))), 3) for t in anchor_times_s]
    unique_anchors = sorted(set(clean_anchors))
    pinned_indices: set[int] = set()
    snap_tol = 0.55 / max(1.0, eff_fps)

    for ta in unique_anchors:
        pos = bisect.bisect_left(grid, ta)
        # Already present?
        if pos < len(grid) and abs(grid[pos] - ta) <= 1e-4:
            pinned_indices.add(pos)
            continue
        if pos > 0 and abs(grid[pos - 1] - ta) <= 1e-4:
            pinned_indices.add(pos - 1)
            continue

        # Check nearest unpinned candidate to snap
        cands: list[tuple[float, int]] = []
        if pos < len(grid) and pos not in pinned_indices:
            cands.append((abs(grid[pos] - ta), pos))
        if pos > 0 and (pos - 1) not in pinned_indices:
            cands.append((abs(grid[pos - 1] - ta), pos - 1))
        cands.sort()

        snapped = False
        for dist, cand_i in cands:
            if dist > snap_tol:
                continue
            lo_ok = cand_i == 0 or grid[cand_i - 1] < ta - 1e-4
            hi_ok = cand_i == len(grid) - 1 or ta < grid[cand_i + 1] - 1e-4
            if lo_ok and hi_ok:
                grid[cand_i] = ta
                pinned_indices.add(cand_i)
                snapped = True
                break
        if not snapped:
            ins = bisect.bisect_left(grid, ta)
            grid.insert(ins, ta)
            pinned_indices = {i + 1 if i >= ins else i for i in pinned_indices}
            pinned_indices.add(ins)

    # Map each input anchor timestamp to its exact dense index
    anchor_indices: list[int] = []
    for ta in clean_anchors:
        pos = bisect.bisect_left(grid, ta)
        if pos >= len(grid):
            pos = len(grid) - 1
        elif pos > 0 and abs(grid[pos - 1] - ta) < abs(grid[pos] - ta):
            pos = pos - 1
        anchor_indices.append(pos)

    return grid, round(eff_fps, 3), anchor_indices


def _to_track_gray(rgb: np.ndarray, eff_w: int, eff_h: int) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    h, w = gray.shape[:2]
    if w != eff_w or h != eff_h:
        interp = cv2.INTER_AREA if (eff_w < w or eff_h < h) else cv2.INTER_LINEAR
        gray = cv2.resize(gray, (eff_w, eff_h), interpolation=interp)
    return np.ascontiguousarray(gray, dtype=np.uint8)


def _decode_span_gray(
    view: EpisodeView,
    camera: str,
    times_s: Sequence[float],
    target_w: int,
    check_cancelled: Callable[[], None],
    *,
    chunk_size: int = 64,
    reuse_first: np.ndarray | None = None,
    dims: tuple[int, int] | None = None,
) -> tuple[list[np.ndarray], int, int]:
    """Chunked streaming decode -> downsampled grayscale frames (peak RAM bounded)."""
    n = len(times_s)
    if n == 0:
        w0, h0 = dims if dims is not None else (target_w, _even_dim(target_w * 9 / 16))
        return [], w0, h0

    out: list[np.ndarray] = []
    start_idx = 0
    eff_w, eff_h = dims if dims is not None else (0, 0)
    if reuse_first is not None:
        out.append(reuse_first)
        eff_h, eff_w = int(reuse_first.shape[0]), int(reuse_first.shape[1])
        start_idx = 1

    for offset in range(start_idx, n, chunk_size):
        check_cancelled()
        chunk_t = list(times_s[offset : offset + chunk_size])
        rgb_chunk = view.media.read_frames_batch(camera, chunk_t)
        check_cancelled()
        if not rgb_chunk:
            continue
        if eff_w <= 0 or eff_h <= 0:
            orig_h, orig_w = rgb_chunk[0].shape[:2]
            eff_w = _even_dim(min(orig_w, target_w), 32)
            eff_h = _even_dim(orig_h * eff_w / float(max(1, orig_w)), 32)
        for rgb in rgb_chunk:
            out.append(_to_track_gray(rgb, eff_w, eff_h))
        del rgb_chunk

    if eff_w <= 0 or eff_h <= 0:
        eff_w, eff_h = target_w, _even_dim(target_w * 9 / 16)
    return out, eff_w, eff_h


def _compute_span_flows(
    grays: Sequence[np.ndarray],
    dis: Any,
    check_cancelled: Callable[[], None],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Compute forward and backward DIS optical flow once per adjacent pair."""
    n = len(grays)
    use_f16 = n > 128
    flow_fwd: list[np.ndarray] = []
    flow_bwd: list[np.ndarray] = []
    for i in range(n - 1):
        if i % 16 == 0:
            check_cancelled()
        g0 = np.ascontiguousarray(grays[i], dtype=np.uint8)
        g1 = np.ascontiguousarray(grays[i + 1], dtype=np.uint8)
        ff = dis.calc(g0, g1, None)
        fb = dis.calc(g1, g0, None)
        if use_f16:
            flow_fwd.append(ff.astype(np.float16))
            flow_bwd.append(fb.astype(np.float16))
        else:
            flow_fwd.append(ff)
            flow_bwd.append(fb)
    return flow_fwd, flow_bwd


def _estimate_bg_ego_motion_step(
    ff_raw: np.ndarray,
    masked_boxes_px: Sequence[np.ndarray],
) -> tuple[float, float, float]:
    """Estimate coherent background translation (dx, dy) & pixel agreement ratio."""
    h, w = ff_raw.shape[:2]
    # Subsample by 2 for speed
    ff = ff_raw[::2, ::2].astype(np.float32, copy=False)
    sh, sw = ff.shape[:2]
    bg_mask = np.ones((sh, sw), dtype=bool)
    for box in masked_boxes_px:
        x1 = max(0, int(math.floor((float(box[0]) - 4.0) / 2.0)))
        y1 = max(0, int(math.floor((float(box[1]) - 4.0) / 2.0)))
        x2 = min(sw, int(math.ceil((float(box[2]) + 4.0) / 2.0)))
        y2 = min(sh, int(math.ceil((float(box[3]) + 4.0) / 2.0)))
        if x2 > x1 and y2 > y1:
            bg_mask[y1:y2, x1:x2] = False

    pts = ff[bg_mask]
    if pts.shape[0] < 32:
        pts = ff.reshape(-1, 2)
    if pts.shape[0] == 0:
        return 0.0, 0.0, 0.0

    med = np.median(pts, axis=0)
    dx, dy = float(med[0]), float(med[1])
    diffs = np.linalg.norm(pts - med[None, :], axis=-1)
    agree = float(np.mean(diffs <= 1.5))
    if agree < 0.50:
        return 0.0, 0.0, agree
    return dx, dy, agree


def _step_box_with_flow(
    box: np.ndarray,
    flow_step: np.ndarray,
    flow_rev: np.ndarray,
    bg_dx: float,
    bg_dy: float,
    anchor_wh: tuple[float, float],
    fb_max_err_px: float,
    w: int,
    h: int,
) -> tuple[np.ndarray, float]:
    """Propagate box 1 frame using inner-70%-core DIS flow filtered by FB consistency."""
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    bw = max(2.0, x2 - x1)
    bh = max(2.0, y2 - y1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    mx = 0.15 * bw
    my = 0.15 * bh
    cx1 = max(0, min(w - 2, int(round(x1 + mx))))
    cy1 = max(0, min(h - 2, int(round(y1 + my))))
    cx2 = max(cx1 + 2, min(w, int(round(x2 - mx))))
    cy2 = max(cy1 + 2, min(h, int(round(y2 - my))))

    ff_core = flow_step[cy1:cy2, cx1:cx2].astype(np.float32, copy=False)
    ys, xs = np.mgrid[cy1:cy2, cx1:cx2]
    tx = np.clip(np.round(xs + ff_core[..., 0]).astype(np.int32), 0, w - 1)
    ty = np.clip(np.round(ys + ff_core[..., 1]).astype(np.int32), 0, h - 1)
    fb_at_t = flow_rev[ty, tx].astype(np.float32, copy=False)
    fb_err = np.linalg.norm(ff_core + fb_at_t, axis=-1)

    good = fb_err <= fb_max_err_px
    n_good = int(np.count_nonzero(good))
    good_ratio = n_good / float(max(1, good.size))

    step_s = 1.0
    if good_ratio >= 0.15 and n_good >= 4:
        good_f = ff_core[good]
        dx = float(np.median(good_f[:, 0]))
        dy = float(np.median(good_f[:, 1]))
        med_err = float(np.median(fb_err[good]))
        flow_conf = float(
            np.clip(
                good_ratio * math.exp(-0.35 * med_err / max(0.5, fb_max_err_px)),
                0.0,
                1.0,
            )
        )
        # Modest clamped scale estimation on well-supported boxes
        if n_good >= 16 and bw >= 12.0 and bh >= 12.0:
            gx = xs[good].astype(np.float32)
            gy = ys[good].astype(np.float32)
            mx0 = float(np.mean(gx))
            my0 = float(np.mean(gy))
            r0 = np.hypot(gx - mx0, gy - my0)
            valid_r = r0 > 2.0
            if int(np.count_nonzero(valid_r)) >= 8:
                gx1 = gx[valid_r] + good_f[valid_r, 0]
                gy1 = gy[valid_r] + good_f[valid_r, 1]
                r1 = np.hypot(gx1 - (mx0 + dx), gy1 - (my0 + dy))
                raw_s = float(np.median(r1 / r0[valid_r]))
                if math.isfinite(raw_s) and 0.5 < raw_s < 2.0:
                    step_s = float(np.clip(1.0 + 0.35 * (raw_s - 1.0), 0.96, 1.04))
    else:
        dx, dy = bg_dx, bg_dy
        flow_conf = 0.10

    aw, ah = anchor_wh
    new_w = float(np.clip(bw * step_s, max(4.0, 0.65 * aw), min(float(w), 1.55 * aw)))
    new_h = float(np.clip(bh * step_s, max(4.0, 0.65 * ah), min(float(h), 1.55 * ah)))
    ncx = cx + dx
    ncy = cy + dy
    new_box = _clamp_box_px(
        np.array(
            [
                ncx - 0.5 * new_w,
                ncy - 0.5 * new_h,
                ncx + 0.5 * new_w,
                ncy + 0.5 * new_h,
            ],
            dtype=np.float64,
        ),
        w,
        h,
    )
    return new_box, flow_conf


def _extract_template(
    gray: np.ndarray, box: np.ndarray
) -> tuple[np.ndarray | None, tuple[int, int, int, int]]:
    """Extract grayscale template patch, adding a 3px silhouette collar if low-texture."""
    h, w = gray.shape[:2]
    x1 = max(0, min(w - 2, int(round(float(box[0])))))
    y1 = max(0, min(h - 2, int(round(float(box[1])))))
    x2 = max(x1 + 2, min(w, int(round(float(box[2])))))
    y2 = max(y1 + 2, min(h, int(round(float(box[3])))))
    patch = gray[y1:y2, x1:x2]
    if patch.size < 16 or patch.shape[0] < 4 or patch.shape[1] < 4:
        return None, (0, 0, 0, 0)

    # Low-texture object (e.g. uniform white cup or solid synthetic box): include 3px border
    if float(np.std(patch)) < 6.0:
        px1 = max(0, x1 - 3)
        py1 = max(0, y1 - 3)
        px2 = min(w, x2 + 3)
        py2 = min(h, y2 + 3)
        padded = gray[py1:py2, px1:px2]
        if padded.shape[0] >= 6 and padded.shape[1] >= 6:
            return (
                padded.astype(np.float32),
                (x1 - px1, y1 - py1, px2 - x2, py2 - y2),
            )
    return patch.astype(np.float32), (0, 0, 0, 0)


def _match_template_best(
    gray: np.ndarray,
    pred_box: np.ndarray,
    tmpl: np.ndarray,
    collar: tuple[int, int, int, int],
    scales: Sequence[float],
) -> tuple[float, np.ndarray | None]:
    """Bounded local multi-scale cv2.matchTemplate(TM_CCOEFF_NORMED)."""
    h, w = gray.shape[:2]
    th0, tw0 = tmpl.shape[:2]
    if th0 < 4 or tw0 < 4 or float(np.std(tmpl)) < 1.0:
        return 0.0, None

    pl, pt, pr, pb = collar
    bw = max(4.0, float(pred_box[2] - pred_box[0]))
    bh = max(4.0, float(pred_box[3] - pred_box[1]))
    cx = 0.5 * (float(pred_box[0]) + float(pred_box[2]))
    cy = 0.5 * (float(pred_box[1]) + float(pred_box[3]))

    pad = max(10, int(round(0.35 * max(bw, bh))))
    rx1 = max(0, int(round(pred_box[0] - pl - pad)))
    ry1 = max(0, int(round(pred_box[1] - pt - pad)))
    rx2 = min(w, int(round(pred_box[2] + pr + pad)))
    ry2 = min(h, int(round(pred_box[3] + pb + pad)))
    roi = gray[ry1:ry2, rx1:rx2]
    rh, rw = roi.shape[:2]
    if rh < 6 or rw < 6:
        return 0.0, None
    roi_f = roi.astype(np.float32)
    if float(np.std(roi_f)) < 1.0:
        return 0.0, None

    best_score = -1.0
    best_box: np.ndarray | None = None
    gate_dist = 0.45 * max(bw, bh)

    for s in scales:
        tw = max(4, int(round(tw0 * s)))
        th = max(4, int(round(th0 * s)))
        if tw >= rw or th >= rh:
            continue
        if tw == tw0 and th == th0:
            t_cur = tmpl
        else:
            t_cur = cv2.resize(
                tmpl,
                (tw, th),
                interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR,
            )
        if float(np.std(t_cur)) < 1.0:
            continue
        res = cv2.matchTemplate(roi_f, t_cur, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if not math.isfinite(max_val):
            continue
        mx, my = int(max_loc[0]), int(max_loc[1])
        # Undo collar offset in scaled coordinates
        spl = pl * s
        spt = pt * s
        spr = pr * s
        spb = pb * s
        bx1 = rx1 + mx + spl
        by1 = ry1 + my + spt
        bx2 = rx1 + mx + tw - spr
        by2 = ry1 + my + th - spb
        if bx2 <= bx1 + 2.0 or by2 <= by1 + 2.0:
            continue
        bcx = 0.5 * (bx1 + bx2)
        bcy = 0.5 * (by1 + by2)
        if math.hypot(bcx - cx, bcy - cy) > gate_dist:
            continue
        if max_val > best_score:
            best_score = float(max_val)
            best_box = _clamp_box_px(np.array([bx1, by1, bx2, by2], dtype=np.float64), w, h)

    return max(0.0, best_score), best_box


def _rollout_one_direction(
    grays: Sequence[np.ndarray],
    flow_step_list: Sequence[np.ndarray],
    flow_rev_list: Sequence[np.ndarray],
    bg_steps: Sequence[tuple[float, float]],
    init_box_px: np.ndarray,
    cfg: TrackingConfig,
    w: int,
    h: int,
    *,
    stop_on_low_conf: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run single-direction flow + periodic multi-scale NCC rollout over L frames.

    Returns (boxes (L, 4), confs (L,), ncc_scores (L,)).
    """
    L = len(grays)
    boxes = np.zeros((L, 4), dtype=np.float64)
    confs = np.zeros(L, dtype=np.float64)
    ncc_scores = np.ones(L, dtype=np.float64)

    cur_box = _clamp_box_px(init_box_px, w, h)
    aw = max(4.0, float(cur_box[2] - cur_box[0]))
    ah = max(4.0, float(cur_box[3] - cur_box[1]))
    boxes[0] = cur_box
    confs[0] = 1.0
    ncc_scores[0] = 1.0

    tmpl_anchor, collar = _extract_template(grays[0], cur_box)
    tmpl_ema = tmpl_anchor.copy() if tmpl_anchor is not None else None
    last_ncc = 1.0
    scales_multi = (0.94, 1.0, 1.06) if min(aw, ah) >= 16.0 else (1.0,)

    for i in range(L - 1):
        bg_dx, bg_dy = bg_steps[i]
        pred_box, flow_conf = _step_box_with_flow(
            cur_box,
            flow_step_list[i],
            flow_rev_list[i],
            bg_dx,
            bg_dy,
            (aw, ah),
            cfg.fb_max_err_px,
            w,
            h,
        )

        step_conf = flow_conf
        step_ncc = last_ncc
        run_ncc = tmpl_anchor is not None and (
            ((i + 1) % cfg.ncc_stride == 0) or (i + 1 == L - 1) or flow_conf < 0.35
        )
        if run_ncc and tmpl_anchor is not None:
            sc_a, box_a = _match_template_best(
                grays[i + 1], pred_box, tmpl_anchor, collar, scales_multi
            )
            best_sc, best_box = sc_a, box_a
            if tmpl_ema is not None:
                sc_e, box_e = _match_template_best(
                    grays[i + 1], pred_box, tmpl_ema, collar, (1.0,)
                )
                # Slight anchor preference (+0.03) prevents online drift
                if sc_e > sc_a + 0.03 and box_e is not None:
                    best_sc, best_box = sc_e, box_e

            step_ncc = best_sc
            last_ncc = best_sc
            if best_box is not None and best_sc >= cfg.ncc_min_score:
                strength = (best_sc - cfg.ncc_min_score) / max(
                    1e-3, 1.0 - cfg.ncc_min_score
                )
                alpha = min(0.35, max(0.15, strength * 0.35))
                if best_sc >= 0.80 and flow_conf < 0.40:
                    alpha = 0.55
                pred_box = _clamp_box_px(
                    (1.0 - alpha) * pred_box + alpha * best_box, w, h
                )
                step_conf = max(flow_conf, 0.40 * flow_conf + 0.60 * best_sc)
                # Slow EMA online appearance update when confident
                if step_conf > 0.65 and tmpl_ema is not None:
                    cur_patch, _ = _extract_template(grays[i + 1], pred_box)
                    if cur_patch is not None:
                        if cur_patch.shape != tmpl_ema.shape:
                            cur_patch = cv2.resize(
                                cur_patch,
                                (tmpl_ema.shape[1], tmpl_ema.shape[0]),
                                interpolation=cv2.INTER_LINEAR,
                            )
                        tmpl_ema = 0.85 * tmpl_ema + 0.15 * cur_patch
            elif best_sc < 0.30:
                # Severe appearance mismatch -> strong occlusion / out-of-FOV indicator
                step_conf = min(step_conf, max(0.0, best_sc))
            else:
                step_conf = min(
                    step_conf, max(0.05, 0.50 * flow_conf + 0.50 * best_sc)
                )
        elif last_ncc < 0.30:
            # Carry occlusion drop cleanly across non-NCC frames
            step_conf = min(step_conf, max(0.0, last_ncc))

        cur_box = pred_box
        boxes[i + 1] = cur_box
        confs[i + 1] = float(np.clip(step_conf, 0.0, 1.0))
        ncc_scores[i + 1] = float(np.clip(step_ncc, 0.0, 1.0))
        if stop_on_low_conf and (confs[i + 1] < 0.55 or ncc_scores[i + 1] < 0.50):
            # Mark remaining frames as zero-confidence invisible
            confs[i + 1 :] = 0.0
            ncc_scores[i + 1 :] = 0.0
            break

    return boxes, confs, ncc_scores


def track_interval_bidirectional(
    grays: Sequence[np.ndarray],
    flow_fwd: Sequence[np.ndarray],
    flow_bwd: Sequence[np.ndarray],
    bg_steps: Sequence[tuple[float, float]],
    box_start_px: np.ndarray,
    box_end_px: np.ndarray,
    cfg: TrackingConfig,
    w: int,
    h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Bidirectional forward-backward BAFT tracking & confidence-weighted fusion.

    Returns (fused_boxes (L, 4), flags (L,), confs (L,), disagree_ratio).
    """
    L = len(grays)
    if L <= 1:
        b = _clamp_box_px(box_start_px, w, h)
        return (
            b[None, :],
            np.array([FLAG_VLM_ANCHOR], dtype=np.int32),
            np.array([1.0], dtype=np.float64),
            0.0,
        )

    fwd_boxes, fwd_confs, _ = _rollout_one_direction(
        grays, flow_fwd, flow_bwd, bg_steps, box_start_px, cfg, w, h
    )

    # Reverse views for backward rollout (L-1 -> 0)
    rev_grays = list(reversed(grays))
    rev_flow_step = list(reversed(flow_bwd))
    rev_flow_back = list(reversed(flow_fwd))
    rev_bg = [(-dx, -dy) for dx, dy in reversed(bg_steps)]

    bwd_boxes_rev, bwd_confs_rev, _ = _rollout_one_direction(
        rev_grays, rev_flow_step, rev_flow_back, rev_bg, box_end_px, cfg, w, h
    )
    bwd_boxes = bwd_boxes_rev[::-1]
    bwd_confs = bwd_confs_rev[::-1]

    fused = np.zeros((L, 4), dtype=np.float64)
    flags = np.full(L, FLAG_TRACKED, dtype=np.int32)
    confs = np.zeros(L, dtype=np.float64)
    both_low = np.zeros(L, dtype=bool)

    fused[0] = _clamp_box_px(box_start_px, w, h)
    flags[0] = FLAG_VLM_ANCHOR
    confs[0] = 1.0

    fused[L - 1] = _clamp_box_px(box_end_px, w, h)
    flags[L - 1] = FLAG_VLM_ANCHOR
    confs[L - 1] = 1.0

    disagrees: list[float] = []
    mid_lo = max(1, L // 4)
    mid_hi = min(L - 1, max(mid_lo + 1, (3 * L) // 4 + 1))

    for i in range(1, L - 1):
        u = i / float(L - 1)
        cf = float(fwd_confs[i])
        cb = float(bwd_confs[i])
        wf = ((1.0 - u) ** 1.5) * (0.15 + cf)
        wb = (u**1.5) * (0.15 + cb)
        denom = max(1e-6, wf + wb)
        iou_i = box_iou(fwd_boxes[i], bwd_boxes[i])
        if iou_i < 0.20:
            # Disjoint objects: never linearly average across empty air.
            fwd_ok = cf >= cfg.occlusion_conf_thresh
            bwd_ok = cb >= cfg.occlusion_conf_thresh
            if not fwd_ok and not bwd_ok:
                both_low[i] = True
                flags[i] = FLAG_INVISIBLE
                fused[i] = 0.0
                confs[i] = 0.0
            else:
                if fwd_ok != bwd_ok:
                    if fwd_ok:
                        win_box, win_conf = fwd_boxes[i], cf
                    else:
                        win_box, win_conf = bwd_boxes[i], cb
                elif abs(cf - cb) >= 0.15:
                    if cf > cb:
                        win_box, win_conf = fwd_boxes[i], cf
                    else:
                        win_box, win_conf = bwd_boxes[i], cb
                elif abs(wf - wb) <= 0.08 * (wf + wb):
                    flags[i] = FLAG_INVISIBLE
                    fused[i] = 0.0
                    confs[i] = 0.0
                    if mid_lo <= i < mid_hi:
                        disagrees.append(1.0 - iou_i)
                    continue
                else:
                    if wf > wb:
                        win_box, win_conf = fwd_boxes[i], cf
                    else:
                        win_box, win_conf = bwd_boxes[i], cb
                agree = max(0.10, float(iou_i))
                fused[i] = _clamp_box_px(win_box, w, h)
                confs[i] = float(np.clip(win_conf * agree, 0.0, 1.0))
        else:
            agree = float(np.clip(0.35 + 0.65 * (iou_i / 0.65), 0.35, 1.0))
            fused[i] = _clamp_box_px((wf * fwd_boxes[i] + wb * bwd_boxes[i]) / denom, w, h)
            base_conf = (wf * cf + wb * cb) / denom
            confs[i] = float(np.clip(base_conf * agree, 0.0, 1.0))
            if cf < cfg.occlusion_conf_thresh and cb < cfg.occlusion_conf_thresh:
                both_low[i] = True
        if mid_lo <= i < mid_hi:
            disagrees.append(1.0 - iou_i)

    # Occlusion hysteresis: only hide when BOTH directions fail for >= occlusion_min_frames
    i = 1
    while i < L - 1:
        if not both_low[i]:
            i += 1
            continue
        j = i
        while j < L - 1 and both_low[j]:
            j += 1
        if (j - i) >= cfg.occlusion_min_frames:
            flags[i:j] = FLAG_INVISIBLE
            fused[i:j] = 0.0
            confs[i:j] = 0.0
        i = j

    disagree_ratio = float(np.percentile(disagrees, 75)) if disagrees else 0.0
    return fused, flags, confs, disagree_ratio


def _track_interval_onesided(
    grays: Sequence[np.ndarray],
    flow_fwd: Sequence[np.ndarray],
    flow_bwd: Sequence[np.ndarray],
    bg_steps: Sequence[tuple[float, float]],
    anchor_box_px: np.ndarray,
    *,
    from_start: bool,
    cfg: TrackingConfig,
    w: int,
    h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate one-sidedly across a visible <-> invisible anchor boundary."""
    L = len(grays)
    if L <= 1:
        b = _clamp_box_px(anchor_box_px, w, h)
        return (
            b[None, :],
            np.array([FLAG_VLM_ANCHOR], dtype=np.int32),
            np.array([1.0], dtype=np.float64),
        )

    if from_start:
        boxes, confs, nccs = _rollout_one_direction(
            grays,
            flow_fwd,
            flow_bwd,
            bg_steps,
            anchor_box_px,
            cfg,
            w,
            h,
            stop_on_low_conf=True,
        )
        flags = np.zeros(L, dtype=np.int32)
        flags[0] = FLAG_VLM_ANCHOR
        for i in range(1, L - 1):
            if confs[i] >= 0.55 and nccs[i] >= 0.50:
                flags[i] = FLAG_TRACKED
            else:
                boxes[i:] = 0.0
                confs[i:] = 0.0
                flags[i:] = FLAG_INVISIBLE
                break
        # Endpoint is the invisible anchor
        boxes[L - 1] = 0.0
        confs[L - 1] = 0.0
        flags[L - 1] = FLAG_INVISIBLE
        return boxes, flags, confs

    # from_end (backward from L-1 -> 0; index 0 is invisible anchor)
    rev_grays = list(reversed(grays))
    rev_flow_step = list(reversed(flow_bwd))
    rev_flow_back = list(reversed(flow_fwd))
    rev_bg = [(-dx, -dy) for dx, dy in reversed(bg_steps)]
    boxes_r, confs_r, nccs_r = _rollout_one_direction(
        rev_grays,
        rev_flow_step,
        rev_flow_back,
        rev_bg,
        anchor_box_px,
        cfg,
        w,
        h,
        stop_on_low_conf=True,
    )
    flags_r = np.zeros(L, dtype=np.int32)
    flags_r[0] = FLAG_VLM_ANCHOR
    for i in range(1, L - 1):
        if confs_r[i] >= 0.55 and nccs_r[i] >= 0.50:
            flags_r[i] = FLAG_TRACKED
        else:
            boxes_r[i:] = 0.0
            confs_r[i:] = 0.0
            flags_r[i:] = FLAG_INVISIBLE
            break
    boxes_r[L - 1] = 0.0
    confs_r[L - 1] = 0.0
    flags_r[L - 1] = FLAG_INVISIBLE
    return boxes_r[::-1], flags_r[::-1], confs_r[::-1]


def _smooth_dense_track(
    boxes_px: np.ndarray,
    flags: np.ndarray,
    smooth_window: int,
    w: int,
    h: int,
) -> np.ndarray:
    """Apply zero-phase Savitzky-Golay smoothing on contiguous visible runs, softly pinning anchors."""
    n = len(flags)
    out = boxes_px.copy()
    win = int(smooth_window)
    if win < 5:
        return out
    if win % 2 == 0:
        win += 1

    i = 0
    while i < n:
        if flags[i] == FLAG_INVISIBLE:
            i += 1
            continue
        j = i + 1
        while (
            j < n
            and flags[j] != FLAG_INVISIBLE
            and box_iou(boxes_px[j - 1], boxes_px[j]) >= 0.20
        ):
            j += 1
        run_len = j - i
        if run_len >= win:
            seg = boxes_px[i:j]
            cx = 0.5 * (seg[:, 0] + seg[:, 2])
            cy = 0.5 * (seg[:, 1] + seg[:, 3])
            bw = np.maximum(2.0, seg[:, 2] - seg[:, 0])
            bh = np.maximum(2.0, seg[:, 3] - seg[:, 1])

            s_cx = savgol_filter(cx, window_length=win, polyorder=2)
            s_cy = savgol_filter(cy, window_length=win, polyorder=2)
            s_bw = np.maximum(2.0, savgol_filter(bw, window_length=win, polyorder=2))
            s_bh = np.maximum(2.0, savgol_filter(bh, window_length=win, polyorder=2))

            sm = np.stack(
                [
                    s_cx - 0.5 * s_bw,
                    s_cy - 0.5 * s_bh,
                    s_cx + 0.5 * s_bw,
                    s_cy + 0.5 * s_bh,
                ],
                axis=1,
            )
            for k in range(run_len):
                idx = i + k
                if flags[idx] == FLAG_VLM_ANCHOR:
                    blended = 0.8 * boxes_px[idx] + 0.2 * sm[k]
                else:
                    blended = sm[k]
                out[idx] = _clamp_box_px(blended, w, h)
        i = j
    return out


def _find_candidate_outliers(
    keyframes: Sequence[GroundingKeyframe],
    outlier_jump_norm: float,
) -> dict[int, tuple[int, int]]:
    """Return mapping {anchor_k: (prev_vis_k, next_vis_k)} for geometric jump candidates."""
    vis_indices = [i for i, kf in enumerate(keyframes) if kf.visible]
    cands: dict[int, tuple[int, int]] = {}
    for pos in range(1, len(vis_indices) - 1):
        k_prev = vis_indices[pos - 1]
        k_cur = vis_indices[pos]
        k_next = vis_indices[pos + 1]
        b_prev = keyframes[k_prev].bbox_norm_1000
        b_cur = keyframes[k_cur].bbox_norm_1000
        b_next = keyframes[k_next].bbox_norm_1000
        d_prev = box_center_dist(b_cur, b_prev)
        d_next = box_center_dist(b_cur, b_next)
        d_bridge = box_center_dist(b_prev, b_next)
        if d_prev > outlier_jump_norm and d_next > outlier_jump_norm and d_bridge < 120.0:
            cands[k_cur] = (k_prev, k_next)
    return cands


def clamp_anchor_scale_outliers(
    kfs: Sequence[GroundingKeyframe],
    *,
    ref_w: int,
    ref_h: int,
    max_scale_dev: float = 0.35,
) -> list[GroundingKeyframe]:
    """Clamp anchor width/height outliers (> max_scale_dev from inlier median) preserving center."""
    vis_kfs = [k for k in kfs if k.visible]
    if len(vis_kfs) < 3:
        return list(kfs)

    ws = np.array(
        [max(1.0, float(k.bbox_norm_1000[2] - k.bbox_norm_1000[0])) for k in vis_kfs],
        dtype=np.float64,
    )
    hs = np.array(
        [max(1.0, float(k.bbox_norm_1000[3] - k.bbox_norm_1000[1])) for k in vis_kfs],
        dtype=np.float64,
    )
    w0 = float(np.median(ws))
    h0 = float(np.median(hs))

    dev = max(0.05, float(max_scale_dev))
    inlier_mask = (np.abs(ws - w0) <= dev * w0) & (np.abs(hs - h0) <= dev * h0)
    if np.any(inlier_mask):
        med_w = float(np.median(ws[inlier_mask]))
        med_h = float(np.median(hs[inlier_mask]))
    else:
        med_w, med_h = w0, h0

    lo_w = max(1.0, (1.0 - dev) * med_w)
    hi_w = min(1000.0, (1.0 + dev) * med_w)
    lo_h = max(1.0, (1.0 - dev) * med_h)
    hi_h = min(1000.0, (1.0 + dev) * med_h)

    lo_w_int = min(1000, math.ceil(lo_w))
    hi_w_int = max(1, math.floor(hi_w))
    lo_h_int = min(1000, math.ceil(lo_h))
    hi_h_int = max(1, math.floor(hi_h))

    out: list[GroundingKeyframe] = []
    for k in kfs:
        if not k.visible:
            out.append(k)
            continue
        x1, y1, x2, y2 = (float(v) for v in k.bbox_norm_1000)
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        if lo_w <= bw <= hi_w and lo_h <= bh <= hi_h:
            out.append(k)
            continue

        nw = float(np.clip(bw, lo_w, hi_w))
        nh = float(np.clip(bh, lo_h, hi_h))
        target_w = int(round(nw))
        if lo_w_int <= hi_w_int:
            target_w = max(lo_w_int, min(hi_w_int, target_w))
        target_w = max(1, min(1000, target_w))

        target_h = int(round(nh))
        if lo_h_int <= hi_h_int:
            target_h = max(lo_h_int, min(hi_h_int, target_h))
        target_h = max(1, min(1000, target_h))

        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        nx1 = max(0, min(1000 - target_w, int(round(cx - 0.5 * target_w))))
        ny1 = max(0, min(1000 - target_h, int(round(cy - 0.5 * target_h))))
        nx2 = nx1 + target_w
        ny2 = ny1 + target_h
        bbox_px = [
            int(round(nx1 * ref_w / 1000.0)),
            int(round(ny1 * ref_h / 1000.0)),
            int(round(nx2 * ref_w / 1000.0)),
            int(round(ny2 * ref_h / 1000.0)),
        ]
        out.append(
            k.model_copy(
                update={
                    "bbox_norm_1000": [nx1, ny1, nx2, ny2],
                    "bbox_px": bbox_px,
                }
            )
        )
    return out


def stabilize_static_keyframes(
    kfs: Sequence[GroundingKeyframe],
    *,
    ref_w: int,
    ref_h: int,
    static_min_frames: int,
    jitter_tol_px: float,
    default_state: str = "stationary",
) -> tuple[list[GroundingKeyframe], list[int] | None]:
    """Inlier-refined median box stabilization for static objects on fixed cameras.

    Returns (updated_keyframes, median_norm_box_or_None).
    """
    vis_kfs = [k for k in kfs if k.visible]
    if not vis_kfs:
        return list(kfs), None

    arr_norm = np.asarray([k.bbox_norm_1000 for k in vis_kfs], dtype=np.float64)
    arr_px = np.asarray([k.bbox_px for k in vis_kfs], dtype=np.float64)
    init_med_px = np.median(arr_px, axis=0)
    inlier_mask = (
        np.max(np.abs(arr_px - init_med_px[None, :]), axis=-1) <= jitter_tol_px
    )
    med_src = arr_norm[inlier_mask] if np.any(inlier_mask) else arr_norm
    med = np.median(med_src, axis=0)
    mx1 = int(round(max(0.0, min(999.0, float(med[0])))))
    my1 = int(round(max(0.0, min(999.0, float(med[1])))))
    mx2 = int(round(max(float(mx1 + 1), min(1000.0, float(med[2])))))
    my2 = int(round(max(float(my1 + 1), min(1000.0, float(med[3])))))
    med_norm = [mx1, my1, mx2, my2]
    med_px = [
        int(round(mx1 * ref_w / 1000.0)),
        int(round(my1 * ref_h / 1000.0)),
        int(round(mx2 * ref_w / 1000.0)),
        int(round(my2 * ref_h / 1000.0)),
    ]

    if len(vis_kfs) >= static_min_frames:
        locked_kfs = [
            k.model_copy(
                update={
                    "visible": True,
                    "bbox_norm_1000": list(med_norm),
                    "bbox_px": list(med_px),
                    "state": k.state or default_state,
                }
            )
            for k in kfs
        ]
        return locked_kfs, med_norm

    return list(kfs), med_norm


def run_baft_tracking(
    *,
    view: EpisodeView,
    camera: str,
    anchor_times_s: Sequence[float],
    raw_tracks: dict[str, GroundingTrack],
    ref_w: int,
    ref_h: int,
    static_min_frames: int,
    jitter_tol_px: float,
    static_defaults: dict[str, str],
    cfg: TrackingConfig,
    check_cancelled: Callable[[], None] | None = None,
) -> BaftTrackingOutput:
    """Densify sparse Stage 1 VLM anchor tracks into smooth frame-level 2D tracks."""
    chk = check_cancelled or _noop
    chk()

    dense_times, eff_fps, anchor_dense_idx = plan_dense_times(
        duration_s=view.duration_s,
        native_fps=view.fps,
        anchor_times_s=anchor_times_s,
        cfg=cfg,
    )
    n_dense = len(dense_times)
    n_anchors = len(anchor_times_s)

    if not raw_tracks or n_dense == 0 or n_anchors == 0:
        return BaftTrackingOutput(
            track_fps=eff_fps,
            dense_times_s=dense_times,
            camera_motion_detected=False,
            tracks=dict(raw_tracks),
            interval_reports=[],
        )

    # Clamp anchor scale outliers (>35% width/height breathing) before interval tracking
    work_tracks: dict[str, GroundingTrack] = {
        oid: tr.model_copy(
            update={
                "keyframes": clamp_anchor_scale_outliers(
                    tr.keyframes, ref_w=ref_w, ref_h=ref_h
                )
            }
        )
        for oid, tr in raw_tracks.items()
    }

    # Identify candidate VLM instance-switch outliers per object BEFORE streaming intervals
    cand_outliers: dict[str, dict[int, tuple[int, int]]] = {
        oid: _find_candidate_outliers(tr.keyframes, cfg.outlier_jump_norm)
        for oid, tr in work_tracks.items()
    }
    # Merge adjacent anchor intervals whenever any object has a candidate outlier on the middle anchor
    merged_boundary_anchors: set[int] = set()
    for c_map in cand_outliers.values():
        merged_boundary_anchors.update(c_map.keys())

    # Build contiguous anchor-index spans [a_start, a_end] to decode & flow-compute ONCE
    spans: list[tuple[int, int]] = []
    a_cur = 0
    while a_cur < max(1, n_anchors - 1):
        a_next = min(n_anchors - 1, a_cur + 1)
        while a_next < n_anchors - 1 and a_next in merged_boundary_anchors:
            a_next += 1
        spans.append((a_cur, a_next))
        a_cur = a_next
    if n_anchors == 1:
        spans = [(0, 0)]

    # Per-object dense buffers (stored in tracking pixel coords)
    obj_boxes_px: dict[str, np.ndarray] = {
        oid: np.zeros((n_dense, 4), dtype=np.float64) for oid in work_tracks
    }
    obj_flags: dict[str, np.ndarray] = {
        oid: np.zeros(n_dense, dtype=np.int32) for oid in work_tracks
    }
    obj_confs: dict[str, np.ndarray] = {
        oid: np.zeros(n_dense, dtype=np.float64) for oid in work_tracks
    }

    dis = _create_dis_flow(cfg.flow_preset)
    eff_w = 0
    eff_h = 0
    dims: tuple[int, int] | None = None
    last_gray: np.ndarray | None = None
    last_dense_idx = -1

    # Global camera ego-motion accumulator (in tracking pixel coords)
    cum_dx = 0.0
    cum_dy = 0.0
    min_cx = 0.0
    max_cx = 0.0
    min_cy = 0.0
    max_cy = 0.0

    interval_reports: list[IntervalDisagreement] = []

    for a_start, a_end in spans:
        chk()
        d_start = anchor_dense_idx[a_start]
        d_end = anchor_dense_idx[a_end]
        span_times = dense_times[d_start : d_end + 1]
        reuse = last_gray if (last_gray is not None and d_start == last_dense_idx) else None
        grays, eff_w, eff_h = _decode_span_gray(
            view,
            camera,
            span_times,
            cfg.track_width,
            chk,
            chunk_size=64,
            reuse_first=reuse,
            dims=dims,
        )
        dims = (eff_w, eff_h)
        if grays:
            last_gray = grays[-1]
            last_dense_idx = d_end

        if len(grays) <= 1:
            # Single-frame span: pin visible anchors directly
            for oid, tr in work_tracks.items():
                kf = tr.keyframes[a_start]
                if kf.visible:
                    obj_boxes_px[oid][d_start] = _norm_to_px(
                        kf.bbox_norm_1000, eff_w, eff_h
                    )
                    obj_flags[oid][d_start] = FLAG_VLM_ANCHOR
                    obj_confs[oid][d_start] = 1.0
            continue

        flow_fwd, flow_bwd = _compute_span_flows(grays, dis, chk)

        # Estimate background camera ego-motion once per pair in this span
        span_anchor_boxes: list[np.ndarray] = []
        for tr in work_tracks.values():
            for ak in range(a_start, a_end + 1):
                if ak < len(tr.keyframes) and tr.keyframes[ak].visible:
                    span_anchor_boxes.append(
                        _norm_to_px(tr.keyframes[ak].bbox_norm_1000, eff_w, eff_h)
                    )

        bg_steps: list[tuple[float, float]] = []
        for p_idx in range(len(flow_fwd)):
            bdx, bdy, _ = _estimate_bg_ego_motion_step(
                flow_fwd[p_idx], span_anchor_boxes
            )
            bg_steps.append((bdx, bdy))
            # Dead-zone filter rejects sub-pixel JPEG/sensor jitter while catching real pan/tilt
            if math.hypot(bdx, bdy) >= 0.30:
                cum_dx += bdx
                cum_dy += bdy
                if cum_dx < min_cx:
                    min_cx = cum_dx
                if cum_dx > max_cx:
                    max_cx = cum_dx
                if cum_dy < min_cy:
                    min_cy = cum_dy
                if cum_dy > max_cy:
                    max_cy = cum_dy

        # Track each object across its sub-intervals inside [a_start .. a_end]
        for oid, tr in work_tracks.items():
            chk()
            c_map = cand_outliers.get(oid, {})

            # Validate any candidate outlier inside (a_start, a_end) using forward optical flow
            dropped_anchors: set[int] = set()
            for ak in range(a_start + 1, a_end):
                if ak not in c_map:
                    continue
                k_prev, k_next = c_map[ak]
                if k_prev < a_start or k_next > a_end:
                    continue
                rel0 = anchor_dense_idx[k_prev] - d_start
                rel1 = anchor_dense_idx[k_next] - d_start
                if rel1 <= rel0:
                    continue
                b_prev_px = _norm_to_px(
                    tr.keyframes[k_prev].bbox_norm_1000, eff_w, eff_h
                )
                b_next_px = _norm_to_px(
                    tr.keyframes[k_next].bbox_norm_1000, eff_w, eff_h
                )
                fwd_b, _, _ = _rollout_one_direction(
                    grays[rel0 : rel1 + 1],
                    flow_fwd[rel0:rel1],
                    flow_bwd[rel0:rel1],
                    bg_steps[rel0:rel1],
                    b_prev_px,
                    cfg,
                    eff_w,
                    eff_h,
                )
                if box_iou(fwd_b[-1], b_next_px) > 0.35:
                    dropped_anchors.add(ak)

            active_anchors = [
                ak for ak in range(a_start, a_end + 1) if ak not in dropped_anchors
            ]
            for s_i in range(len(active_anchors) - 1):
                k0 = active_anchors[s_i]
                k1 = active_anchors[s_i + 1]
                i0 = anchor_dense_idx[k0]
                i1 = anchor_dense_idx[k1]
                r0 = i0 - d_start
                r1 = i1 - d_start
                if r1 < r0:
                    continue

                kf0 = tr.keyframes[k0]
                kf1 = tr.keyframes[k1]
                sub_grays = grays[r0 : r1 + 1]
                sub_ff = flow_fwd[r0:r1]
                sub_fb = flow_bwd[r0:r1]
                sub_bg = bg_steps[r0:r1]

                if kf0.visible and kf1.visible:
                    b0_px = _norm_to_px(kf0.bbox_norm_1000, eff_w, eff_h)
                    b1_px = _norm_to_px(kf1.bbox_norm_1000, eff_w, eff_h)
                    f_boxes, f_flags, f_confs, disagree = track_interval_bidirectional(
                        sub_grays,
                        sub_ff,
                        sub_fb,
                        sub_bg,
                        b0_px,
                        b1_px,
                        cfg,
                        eff_w,
                        eff_h,
                    )
                    obj_boxes_px[oid][i0 : i1 + 1] = f_boxes
                    obj_flags[oid][i0 : i1 + 1] = f_flags
                    obj_confs[oid][i0 : i1 + 1] = f_confs
                    interval_reports.append(
                        IntervalDisagreement(
                            object_id=oid,
                            a0_idx=k0,
                            a1_idx=k1,
                            dense_i0=i0,
                            dense_i1=i1,
                            t0=dense_times[i0],
                            t1=dense_times[i1],
                            disagree_ratio=disagree,
                        )
                    )
                elif kf0.visible and not kf1.visible:
                    b0_px = _norm_to_px(kf0.bbox_norm_1000, eff_w, eff_h)
                    f_boxes, f_flags, f_confs = _track_interval_onesided(
                        sub_grays,
                        sub_ff,
                        sub_fb,
                        sub_bg,
                        b0_px,
                        from_start=True,
                        cfg=cfg,
                        w=eff_w,
                        h=eff_h,
                    )
                    obj_boxes_px[oid][i0 : i1 + 1] = f_boxes
                    obj_flags[oid][i0 : i1 + 1] = f_flags
                    obj_confs[oid][i0 : i1 + 1] = f_confs
                elif not kf0.visible and kf1.visible:
                    b1_px = _norm_to_px(kf1.bbox_norm_1000, eff_w, eff_h)
                    f_boxes, f_flags, f_confs = _track_interval_onesided(
                        sub_grays,
                        sub_ff,
                        sub_fb,
                        sub_bg,
                        b1_px,
                        from_start=False,
                        cfg=cfg,
                        w=eff_w,
                        h=eff_h,
                    )
                    obj_boxes_px[oid][i0 : i1 + 1] = f_boxes
                    obj_flags[oid][i0 : i1 + 1] = f_flags
                    obj_confs[oid][i0 : i1 + 1] = f_confs
                else:
                    obj_boxes_px[oid][i0 : i1 + 1] = 0.0
                    obj_flags[oid][i0 : i1 + 1] = FLAG_INVISIBLE
                    obj_confs[oid][i0 : i1 + 1] = 0.0

        # Free span frames & flows before moving to the next interval
        del grays, flow_fwd, flow_bwd

    # Handle any leading frames before anchor_dense_idx[0] or trailing frames after anchor_dense_idx[-1]
    first_d = anchor_dense_idx[0]
    last_d = anchor_dense_idx[-1]
    if first_d > 0 and dims is not None:
        chk()
        g_lead, eff_w, eff_h = _decode_span_gray(
            view, camera, dense_times[: first_d + 1], cfg.track_width, chk, dims=dims
        )
        if len(g_lead) > 1:
            ff_l, fb_l = _compute_span_flows(g_lead, dis, chk)
            bg_l = [(0.0, 0.0)] * len(ff_l)
            for oid, tr in work_tracks.items():
                kf0 = tr.keyframes[0]
                if kf0.visible:
                    b0_px = _norm_to_px(kf0.bbox_norm_1000, eff_w, eff_h)
                    rev_b, rev_c, rev_n = _rollout_one_direction(
                        list(reversed(g_lead)),
                        list(reversed(fb_l)),
                        list(reversed(ff_l)),
                        bg_l,
                        b0_px,
                        cfg,
                        eff_w,
                        eff_h,
                        stop_on_low_conf=True,
                    )
                    boxes_l = rev_b[::-1]
                    confs_l = rev_c[::-1]
                    nccs_l = rev_n[::-1]
                    for idx in range(first_d):
                        if confs_l[idx] >= 0.55 and nccs_l[idx] >= 0.50:
                            obj_boxes_px[oid][idx] = boxes_l[idx]
                            obj_flags[oid][idx] = FLAG_TRACKED
                            obj_confs[oid][idx] = confs_l[idx]
    if last_d < n_dense - 1 and dims is not None:
        chk()
        g_tail, eff_w, eff_h = _decode_span_gray(
            view, camera, dense_times[last_d:], cfg.track_width, chk, dims=dims
        )
        if len(g_tail) > 1:
            ff_t, fb_t = _compute_span_flows(g_tail, dis, chk)
            bg_t = [(0.0, 0.0)] * len(ff_t)
            for oid, tr in work_tracks.items():
                kfn = tr.keyframes[-1]
                if kfn.visible:
                    bn_px = _norm_to_px(kfn.bbox_norm_1000, eff_w, eff_h)
                    fwd_b, fwd_c, fwd_n = _rollout_one_direction(
                        g_tail,
                        ff_t,
                        fb_t,
                        bg_t,
                        bn_px,
                        cfg,
                        eff_w,
                        eff_h,
                        stop_on_low_conf=True,
                    )
                    for off in range(1, len(g_tail)):
                        idx = last_d + off
                        if fwd_c[off] >= 0.55 and fwd_n[off] >= 0.50:
                            obj_boxes_px[oid][idx] = fwd_b[off]
                            obj_flags[oid][idx] = FLAG_TRACKED
                            obj_confs[oid][idx] = fwd_c[off]

    # Global camera ego-motion verdict
    drift_x_norm = (max_cx - min_cx) * 1000.0 / float(max(1, eff_w))
    drift_y_norm = (max_cy - min_cy) * 1000.0 / float(max(1, eff_h))
    max_drift_norm = math.hypot(drift_x_norm, drift_y_norm)
    camera_motion_detected = max_drift_norm >= cfg.camera_motion_threshold_norm

    # Finalize tracks: Smart Static-vs-Moving policy + zero-phase smoothing
    final_tracks: dict[str, GroundingTrack] = {}
    for oid, tr in work_tracks.items():
        if tr.is_static_container and not camera_motion_detected:
            def_state = static_defaults.get(oid, "stationary")
            locked_kfs, med_norm = stabilize_static_keyframes(
                tr.keyframes,
                ref_w=ref_w,
                ref_h=ref_h,
                static_min_frames=static_min_frames,
                jitter_tol_px=jitter_tol_px,
                default_state=def_state,
            )
            if med_norm is not None:
                final_tracks[oid] = tr.model_copy(
                    update={
                        "keyframes": locked_kfs,
                        "tracking_mode": "static_locked",
                        "dense_boxes_norm_1000": [list(med_norm) for _ in range(n_dense)],
                        "dense_flags": [FLAG_STATIC_LOCKED] * n_dense,
                        "dense_conf_pct": [100] * n_dense,
                    }
                )
                continue

        # Moving object OR static container observed by a moving camera -> BAFT dense track
        smoothed_px = _smooth_dense_track(
            obj_boxes_px[oid], obj_flags[oid], cfg.smooth_window, eff_w, eff_h
        )
        dense_boxes: list[list[int]] = []
        dense_flags: list[int] = []
        dense_confs: list[int] = []
        for idx in range(n_dense):
            fl = int(obj_flags[oid][idx])
            if fl == FLAG_INVISIBLE:
                dense_boxes.append([0, 0, 0, 0])
                dense_flags.append(FLAG_INVISIBLE)
                dense_confs.append(0)
            else:
                dense_boxes.append(_px_to_norm_int(smoothed_px[idx], eff_w, eff_h))
                dense_flags.append(fl)
                dense_confs.append(
                    int(round(max(0.0, min(100.0, float(obj_confs[oid][idx]) * 100.0))))
                )

        final_tracks[oid] = tr.model_copy(
            update={
                "tracking_mode": "bidir_flow_ncc",
                "dense_boxes_norm_1000": dense_boxes,
                "dense_flags": dense_flags,
                "dense_conf_pct": dense_confs,
            }
        )

    return BaftTrackingOutput(
        track_fps=eff_fps,
        dense_times_s=dense_times,
        camera_motion_detected=camera_motion_detected,
        tracks=final_tracks,
        interval_reports=interval_reports,
    )


def retrack_split_intervals(
    *,
    view: EpisodeView,
    camera: str,
    baft_out: BaftTrackingOutput,
    split_specs: Sequence[tuple[int, int, int]],
    mid_keyframes: dict[int, dict[str, GroundingKeyframe]],
    cfg: TrackingConfig,
    check_cancelled: Callable[[], None] | None = None,
) -> BaftTrackingOutput:
    """Re-track ONLY intervals split by adaptive VLM re-anchoring (O(interval) decode).

    `split_specs`: list of `(dense_i0, dense_mid, dense_i1)` intervals.
    `mid_keyframes`: mapping `dense_mid -> {object_id: GroundingKeyframe}`.
    """
    chk = check_cancelled or _noop
    if not split_specs:
        return baft_out

    dense_times = baft_out.dense_times_s
    dis = _create_dis_flow(cfg.flow_preset)
    updated_tracks: dict[str, GroundingTrack] = {}
    boxes_buf: dict[str, list[list[int]]] = {}
    flags_buf: dict[str, list[int]] = {}
    confs_buf: dict[str, list[int]] = {}
    kfs_buf: dict[str, list[GroundingKeyframe]] = {}

    for oid, tr in baft_out.tracks.items():
        boxes_buf[oid] = [list(b) for b in tr.dense_boxes_norm_1000]
        flags_buf[oid] = list(tr.dense_flags)
        confs_buf[oid] = list(tr.dense_conf_pct)
        kfs_buf[oid] = list(tr.keyframes)

    dims: tuple[int, int] | None = None
    for d0, d_mid, d1 in split_specs:
        chk()
        if not (0 <= d0 < d_mid < d1 < len(dense_times)):
            continue
        span_times = dense_times[d0 : d1 + 1]
        grays, eff_w, eff_h = _decode_span_gray(
            view, camera, span_times, cfg.track_width, chk, chunk_size=64, dims=dims
        )
        dims = (eff_w, eff_h)
        if len(grays) <= 2:
            continue
        flow_fwd, flow_bwd = _compute_span_flows(grays, dis, chk)

        # Anchor boxes for background masking in this span
        mask_boxes: list[np.ndarray] = []
        for oid, tr in baft_out.tracks.items():
            for d_idx in (d0, d1):
                if flags_buf[oid][d_idx] != FLAG_INVISIBLE:
                    mask_boxes.append(_norm_to_px(boxes_buf[oid][d_idx], eff_w, eff_h))
            mk = mid_keyframes.get(d_mid, {}).get(oid)
            if mk is not None and mk.visible:
                mask_boxes.append(_norm_to_px(mk.bbox_norm_1000, eff_w, eff_h))

        bg_steps: list[tuple[float, float]] = [
            _estimate_bg_ego_motion_step(ff, mask_boxes)[:2] for ff in flow_fwd
        ]
        rel_mid = d_mid - d0

        for oid, tr in baft_out.tracks.items():
            chk()
            if tr.tracking_mode == "static_locked":
                continue
            mk = mid_keyframes.get(d_mid, {}).get(oid)
            if mk is not None:
                # Insert midpoint VLM keyframe into sparse keyframes sorted by time_s
                if not any(abs(k.time_s - mk.time_s) <= 1e-4 for k in kfs_buf[oid]):
                    kfs_buf[oid].append(mk)
                    kfs_buf[oid].sort(key=lambda k: k.time_s)

            vis0 = flags_buf[oid][d0] != FLAG_INVISIBLE
            vis1 = flags_buf[oid][d1] != FLAG_INVISIBLE
            vis_m = mk is not None and mk.visible
            b0_px = _norm_to_px(boxes_buf[oid][d0], eff_w, eff_h) if vis0 else None
            b1_px = _norm_to_px(boxes_buf[oid][d1], eff_w, eff_h) if vis1 else None
            bm_px = _norm_to_px(mk.bbox_norm_1000, eff_w, eff_h) if vis_m and mk is not None else None

            L = d1 - d0 + 1
            seg_boxes = np.zeros((L, 4), dtype=np.float64)
            seg_flags = np.zeros(L, dtype=np.int32)
            seg_confs = np.zeros(L, dtype=np.float64)

            # Left half: [0 .. rel_mid]
            if vis0 and vis_m and b0_px is not None and bm_px is not None:
                fb, fl, fc, _ = track_interval_bidirectional(
                    grays[: rel_mid + 1],
                    flow_fwd[:rel_mid],
                    flow_bwd[:rel_mid],
                    bg_steps[:rel_mid],
                    b0_px,
                    bm_px,
                    cfg,
                    eff_w,
                    eff_h,
                )
                seg_boxes[: rel_mid + 1] = fb
                seg_flags[: rel_mid + 1] = fl
                seg_confs[: rel_mid + 1] = fc
            elif vis0 and not vis_m and b0_px is not None:
                fb, fl, fc = _track_interval_onesided(
                    grays[: rel_mid + 1],
                    flow_fwd[:rel_mid],
                    flow_bwd[:rel_mid],
                    bg_steps[:rel_mid],
                    b0_px,
                    from_start=True,
                    cfg=cfg,
                    w=eff_w,
                    h=eff_h,
                )
                seg_boxes[: rel_mid + 1] = fb
                seg_flags[: rel_mid + 1] = fl
                seg_confs[: rel_mid + 1] = fc
            elif not vis0 and vis_m and bm_px is not None:
                fb, fl, fc = _track_interval_onesided(
                    grays[: rel_mid + 1],
                    flow_fwd[:rel_mid],
                    flow_bwd[:rel_mid],
                    bg_steps[:rel_mid],
                    bm_px,
                    from_start=False,
                    cfg=cfg,
                    w=eff_w,
                    h=eff_h,
                )
                seg_boxes[: rel_mid + 1] = fb
                seg_flags[: rel_mid + 1] = fl
                seg_confs[: rel_mid + 1] = fc

            # Right half: [rel_mid .. L-1]
            if vis_m and vis1 and bm_px is not None and b1_px is not None:
                fb, fl, fc, _ = track_interval_bidirectional(
                    grays[rel_mid:],
                    flow_fwd[rel_mid:],
                    flow_bwd[rel_mid:],
                    bg_steps[rel_mid:],
                    bm_px,
                    b1_px,
                    cfg,
                    eff_w,
                    eff_h,
                )
                seg_boxes[rel_mid:] = fb
                seg_flags[rel_mid:] = fl
                seg_confs[rel_mid:] = fc
            elif vis_m and not vis1 and bm_px is not None:
                fb, fl, fc = _track_interval_onesided(
                    grays[rel_mid:],
                    flow_fwd[rel_mid:],
                    flow_bwd[rel_mid:],
                    bg_steps[rel_mid:],
                    bm_px,
                    from_start=True,
                    cfg=cfg,
                    w=eff_w,
                    h=eff_h,
                )
                seg_boxes[rel_mid:] = fb
                seg_flags[rel_mid:] = fl
                seg_confs[rel_mid:] = fc
            elif not vis_m and vis1 and b1_px is not None:
                fb, fl, fc = _track_interval_onesided(
                    grays[rel_mid:],
                    flow_fwd[rel_mid:],
                    flow_bwd[rel_mid:],
                    bg_steps[rel_mid:],
                    b1_px,
                    from_start=False,
                    cfg=cfg,
                    w=eff_w,
                    h=eff_h,
                )
                seg_boxes[rel_mid:] = fb
                seg_flags[rel_mid:] = fl
                seg_confs[rel_mid:] = fc

            sm_px = _smooth_dense_track(seg_boxes, seg_flags, cfg.smooth_window, eff_w, eff_h)
            for off in range(L):
                g_idx = d0 + off
                fl_val = int(seg_flags[off])
                if fl_val == FLAG_INVISIBLE:
                    boxes_buf[oid][g_idx] = [0, 0, 0, 0]
                    flags_buf[oid][g_idx] = FLAG_INVISIBLE
                    confs_buf[oid][g_idx] = 0
                else:
                    boxes_buf[oid][g_idx] = _px_to_norm_int(sm_px[off], eff_w, eff_h)
                    flags_buf[oid][g_idx] = fl_val
                    confs_buf[oid][g_idx] = int(
                        round(max(0.0, min(100.0, float(seg_confs[off]) * 100.0)))
                    )

        del grays, flow_fwd, flow_bwd

    for oid, tr in baft_out.tracks.items():
        updated_tracks[oid] = tr.model_copy(
            update={
                "keyframes": kfs_buf[oid],
                "dense_boxes_norm_1000": boxes_buf[oid],
                "dense_flags": flags_buf[oid],
                "dense_conf_pct": confs_buf[oid],
            }
        )

    return BaftTrackingOutput(
        track_fps=baft_out.track_fps,
        dense_times_s=baft_out.dense_times_s,
        camera_motion_detected=baft_out.camera_motion_detected,
        tracks=updated_tracks,
        interval_reports=baft_out.interval_reports,
    )
