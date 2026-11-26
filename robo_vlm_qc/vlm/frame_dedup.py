"""Fast deterministic static-frame deduplication using downsampled grayscale MAD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameDedupResult:
    """Result of static-frame deduplication over a chronological frame sequence."""

    kept_indices: list[int]
    skipped_ranges: list[tuple[float, float, int]]
    kept_count: int
    dropped_count: int


def _to_gray_thumb(
    img: np.ndarray,
    *,
    crop_top_px: int = 0,
    thumb_w: int = 96,
    thumb_h: int = 54,
) -> np.ndarray:
    arr = np.asarray(img)
    if crop_top_px > 0 and arr.ndim >= 2 and arr.shape[0] > crop_top_px + 4:
        arr = arr[crop_top_px:]
    if arr.ndim == 3 and arr.shape[2] >= 3:
        gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
    elif arr.ndim == 3:
        gray = arr[:, :, 0]
    else:
        gray = arr
    if gray.shape[0] != thumb_h or gray.shape[1] != thumb_w:
        gray = cv2.resize(gray, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32)


def _frame_mad(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def deduplicate_frames(
    images: Sequence[np.ndarray],
    timestamps_s: Sequence[float],
    *,
    mad_threshold: float = 4.0,
    max_gap_s: float = 6.0,
    min_keep: int = 2,
    crop_top_px: int = 0,
    thumb_w: int = 96,
    thumb_h: int = 54,
    pinned_indices: Sequence[int] | None = None,
) -> FrameDedupResult:
    """Drop visually redundant static intermediate frames while preserving timing bounds.

    Computes fast deterministic downsampled grayscale MAD (0..255 scale). Always keeps
    the first and last frame, respects ``max_gap_s`` between consecutive kept frames, and
    guarantees at least ``min(len, max(1, min_keep))`` kept frames.
    """
    n = min(len(images), len(timestamps_s))
    if n == 0:
        return FrameDedupResult(
            kept_indices=[],
            skipped_ranges=[],
            kept_count=0,
            dropped_count=0,
        )

    eff_min_keep = min(n, max(1, int(min_keep)))
    eff_thresh = float(mad_threshold)
    eff_max_gap = float(max_gap_s)

    if n <= eff_min_keep or eff_thresh <= 0.0:
        kept = list(range(n))
        return FrameDedupResult(
            kept_indices=kept,
            skipped_ranges=[],
            kept_count=n,
            dropped_count=0,
        )

    thumbs = [
        _to_gray_thumb(
            images[i],
            crop_top_px=crop_top_px,
            thumb_w=thumb_w,
            thumb_h=thumb_h,
        )
        for i in range(n)
    ]
    ts = [float(timestamps_s[i]) for i in range(n)]
    pinned: set[int] = {0, n - 1}
    if pinned_indices:
        for idx in pinned_indices:
            if 0 <= int(idx) < n:
                pinned.add(int(idx))

    kept: list[int] = [0]
    last_kept = 0
    for i in range(1, n - 1):
        if i in pinned:
            kept.append(i)
            last_kept = i
            continue

        mad = _frame_mad(thumbs[i], thumbs[last_kept])
        gap_cur = ts[i] - ts[last_kept]
        gap_next = ts[i + 1] - ts[last_kept]

        must_keep_gap = eff_max_gap > 0.0 and (
            gap_cur >= eff_max_gap - 1e-6 or gap_next > eff_max_gap + 1e-6
        )
        if mad > eff_thresh or must_keep_gap:
            kept.append(i)
            last_kept = i

    if kept[-1] != n - 1:
        kept.append(n - 1)

    # Guarantee min_keep by iteratively splitting the largest temporal/index gap
    if len(kept) < eff_min_keep:
        kept_set = set(kept)
        while len(kept_set) < eff_min_keep:
            ordered = sorted(kept_set)
            best_idx: int | None = None
            best_score = (-1.0, -1)
            for a, b in zip(ordered, ordered[1:]):
                if b - a <= 1:
                    continue
                mid = (a + b) // 2
                if mid in kept_set:
                    continue
                score = (ts[b] - ts[a], b - a)
                if score > best_score:
                    best_score = score
                    best_idx = mid
            if best_idx is None:
                for cand in range(n):
                    if cand not in kept_set:
                        best_idx = cand
                        break
            if best_idx is None:
                break
            kept_set.add(best_idx)
        kept = sorted(kept_set)

    skipped_ranges: list[tuple[float, float, int]] = []
    for k_prev, k_next in zip(kept, kept[1:]):
        dropped_here = k_next - k_prev - 1
        if dropped_here <= 0:
            continue
        start_s = round(ts[k_prev], 3)
        end_is_static = _frame_mad(thumbs[k_next], thumbs[k_prev]) <= eff_thresh
        end_s = round(ts[k_next] if end_is_static else ts[k_next - 1], 3)
        skipped_ranges.append((start_s, end_s, dropped_here))

    kept_count = len(kept)
    dropped_count = n - kept_count
    return FrameDedupResult(
        kept_indices=kept,
        skipped_ranges=skipped_ranges,
        kept_count=kept_count,
        dropped_count=dropped_count,
    )


def format_skipped_ranges_note(
    skipped_ranges: Sequence[tuple[float, float, int] | tuple[float, float]],
) -> str:
    """Format a concise English prompt note describing omitted static frame spans."""
    if not skipped_ranges:
        return ""
    parts: list[str] = []
    total_dropped = 0
    for item in skipped_ranges:
        if len(item) >= 3:
            s_val, e_val, cnt = float(item[0]), float(item[1]), int(item[2])
            total_dropped += max(0, cnt)
            noun = "frame" if cnt == 1 else "frames"
            parts.append(f"[{s_val:.2f}s..{e_val:.2f}s] ({cnt} {noun} omitted)")
        elif len(item) == 2:
            s_val, e_val = float(item[0]), float(item[1])
            parts.append(f"[{s_val:.2f}s..{e_val:.2f}s]")
    if not parts:
        return ""
    spans_text = ", ".join(parts)
    suffix = f" ({total_dropped} static frame{'s' if total_dropped != 1 else ''} omitted in total)" if total_dropped > 0 else ""
    return (
        f"[Static-Frame Deduplication Note]: Visually unchanged intermediate frames were omitted{suffix}; "
        f"the scene remained visually identical to the preceding shown frame during: {spans_text}. "
        "Retain exact idle/static durations from the frame [t=XX.XXs] timestamps."
    )
