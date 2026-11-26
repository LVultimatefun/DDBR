"""Stage 1: HD main-camera 2D grounding + BAFT local frame-level 2D tracking."""

from __future__ import annotations

import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from ..config import ConfigManager
from ..io.base import EpisodeView
from ..models.annotation import GroundingKeyframe, GroundingResult, GroundingTrack
from ..models.taxonomy import GroundingObjectSpec, TaskSpec
from .client import (
    BudgetExceeded,
    BudgetTracker,
    VlmBlindVisionError,
    VlmCancelledError,
    VlmClient,
    check_client_cancelled,
)
from .prompt_builder import PromptBuilder
from .tracker import (
    TrackingConfig,
    _patch_texture_stats,
    box_center_dist,
    detect_camera_motion_direct,
    retrack_split_intervals,
    run_baft_tracking,
    stabilize_static_keyframes,
)

STATIC_OBJECT_DEFAULTS: dict[str, str] = {
    "bin": "stationary",
    "tray": "stationary",
    "container": "stationary",
    "rack": "stationary",
    "bowl": "stationary",
    "table_zone": "stationary",
    "target_bin": "stationary",
    "source_tray": "stationary",
    "target_tray": "stationary",
}

_REF_BOX_PALETTE_RGB: tuple[tuple[int, int, int], ...] = (
    (255, 140, 0),
    (56, 189, 248),
    (34, 197, 94),
    (244, 63, 94),
    (168, 85, 247),
    (250, 204, 21),
)

_NON_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify_object_id(raw: str, *, fallback_idx: int = 1) -> str:
    """Convert an arbitrary object name into a clean snake_case GroundingObjectSpec id."""
    slug = _NON_SLUG_RE.sub("_", str(raw or "").strip().lower()).strip("_")
    if not slug:
        slug = f"object_{max(1, int(fallback_idx))}"
    if not slug[0].isalnum():
        slug = f"obj_{slug}"
    return slug[:48]


def normalize_custom_objects(custom_objects: Sequence[str] | None) -> list[str]:
    """Return deduplicated, stripped custom object names preserving input order."""
    if not custom_objects:
        return []
    out: list[str] = []
    seen_low: set[str] = set()
    for item in custom_objects:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if not cleaned:
            continue
        low = cleaned.lower()
        if low in seen_low:
            continue
        seen_low.add(low)
        out.append(cleaned)
    return out


def build_custom_grounding_objects(
    custom_objects: Sequence[str] | None,
) -> list[GroundingObjectSpec]:
    """Convert user-specified object names into slugified GroundingObjectSpec entries."""
    names = normalize_custom_objects(custom_objects)
    specs: list[GroundingObjectSpec] = []
    used_ids: set[str] = set()
    for idx, name in enumerate(names, start=1):
        base_id = slugify_object_id(name, fallback_idx=idx)
        oid = base_id
        suffix = 2
        while oid in used_ids:
            oid = f"{base_id}_{suffix}"
            suffix += 1
        used_ids.add(oid)
        specs.append(
            GroundingObjectSpec(
                id=oid,
                name=name,
                desc=f"User-specified target object: {name}",
                tag_pos="bottom" if oid in STATIC_OBJECT_DEFAULTS else "top",
                is_static=oid in STATIC_OBJECT_DEFAULTS,
            )
        )
    return specs


def _select_discovery_indices(n_frames: int, max_frames: int = 5) -> list[int]:
    """Pick 4-6 (bounded by n_frames) evenly spaced frame indices for Pass-0 object discovery."""
    if n_frames <= 0:
        return []
    target = min(n_frames, max(2, min(6, int(max_frames))))
    raw = np.linspace(0, n_frames - 1, target)
    seen: set[int] = set()
    indices: list[int] = []
    for v in raw:
        idx = int(round(float(v)))
        idx = max(0, min(n_frames - 1, idx))
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
    return indices


def _parse_canonical_objects_response(
    resp: Any,
    *,
    max_objects: int = 4,
) -> list[GroundingObjectSpec]:
    """Parse Pass-0 canonical object discovery JSON cleanly; return [] on legacy/mock payloads."""
    if not isinstance(resp, dict):
        return []

    raw_list: Any = resp.get("canonical_objects")
    if not isinstance(raw_list, list):
        # Accept list-form "objects" only (NEVER dict-form legacy {"objects": {oid: {visible, bbox}}})
        cand_objs = resp.get("objects")
        if isinstance(cand_objs, list):
            raw_list = cand_objs
        else:
            return []

    specs: list[GroundingObjectSpec] = []
    used_ids: set[str] = set()
    cap = max(1, min(6, int(max_objects)))

    for idx, item in enumerate(raw_list, start=1):
        if len(specs) >= cap:
            break
        if isinstance(item, str):
            name = item.strip()
            if not name:
                continue
            oid = slugify_object_id(name, fallback_idx=idx)
            desc = f"Discovered task-relevant object: {name}"
            is_static = oid in STATIC_OBJECT_DEFAULTS
            tag_pos = "bottom" if is_static else "top"
        elif isinstance(item, dict):
            # Guard against accidental frame-bbox dicts
            if ("bbox" in item or "bbox_norm_1000" in item) and "name" not in item and "id" not in item:
                continue
            raw_id = str(item.get("id") or item.get("object_id") or "").strip()
            raw_name = str(item.get("name") or item.get("label") or raw_id).strip()
            if not raw_id and not raw_name:
                continue
            oid = slugify_object_id(raw_id or raw_name, fallback_idx=idx)
            name = raw_name or oid.replace("_", " ").title()
            desc = str(item.get("desc") or item.get("description") or "").strip() or name
            raw_static = item.get("is_static")
            if raw_static is None:
                raw_static = item.get("is_static_container")
            is_static = bool(raw_static) if isinstance(raw_static, bool) else (oid in STATIC_OBJECT_DEFAULTS)
            raw_tp = str(item.get("tag_pos") or "").strip().lower()
            tag_pos = raw_tp if raw_tp in ("top", "bottom") else ("bottom" if is_static else "top")
        else:
            continue

        if oid in used_ids:
            continue
        used_ids.add(oid)
        specs.append(
            GroundingObjectSpec(
                id=oid,
                name=name,
                desc=desc,
                tag_pos=tag_pos,  # type: ignore[arg-type]
                is_static=is_static,
            )
        )
    return specs


def _resolve_s1_cfg(cfg: ConfigManager | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(cfg, ConfigManager):
        return dict(cfg.settings.get("vlm", {}).get("s1_grounding", {}))
    if isinstance(cfg, dict):
        if "vlm" in cfg and isinstance(cfg["vlm"], dict):
            return dict(cfg["vlm"].get("s1_grounding", {}))
        if "s1_grounding" in cfg and isinstance(cfg["s1_grounding"], dict):
            return dict(cfg["s1_grounding"])
        return dict(cfg)
    return {}


def _resolve_main_camera_ref_size(
    ev: EpisodeView, main_cam: str, s1_cfg: dict[str, Any]
) -> tuple[int, int]:
    """Use actual main-camera CameraSpec resolution when available; fall back to config."""
    for cam_list in (getattr(ev.media, "cameras", None), getattr(ev.meta, "cameras", None)):
        if not cam_list:
            continue
        for c in cam_list:
            if getattr(c, "name", None) == main_cam:
                cw = int(getattr(c, "width", 0) or 0)
                ch = int(getattr(c, "height", 0) or 0)
                if cw > 0 and ch > 0:
                    return cw, ch
    return int(s1_cfg.get("ref_width", 1280)), int(s1_cfg.get("ref_height", 720))


def _plan_keyframe_times(
    duration_s: float,
    interval_s: float,
    max_keyframes: int,
    hard_max_keyframes: int,
) -> list[float]:
    dur = max(0.05, float(duration_s))
    iv = max(0.25, float(interval_s))
    cap = min(max(1, int(max_keyframes)), max(1, int(hard_max_keyframes)))
    n_est = int(np.floor(dur / iv)) + 1
    if n_est > cap:
        iv = dur / max(1, cap - 1)
    times = [round(min(dur, i * iv), 3) for i in range(cap)]
    dedup: list[float] = []
    for t in times:
        if not dedup or t > dedup[-1] + 1e-4:
            dedup.append(t)
        if t >= dur - 1e-4:
            break
    if dur > 0.2 and dedup and dedup[-1] < dur - 0.15 and len(dedup) < cap:
        dedup.append(round(dur, 3))
    return dedup or [0.0]


def _resize_to_height(rgb: np.ndarray, target_h: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    if h <= 0 or w <= 0 or h == target_h:
        return rgb
    new_w = max(2, int(round(w * (target_h / float(h)))))
    interp = cv2.INTER_AREA if target_h < h else cv2.INTER_LINEAR
    return cv2.resize(rgb, (new_w, int(target_h)), interpolation=interp)


def _normalize_bbox(
    raw_box: Any,
    *,
    img_w: int,
    img_h: int,
    ref_w: int,
    ref_h: int,
    min_box_px: int,
) -> tuple[bool, list[int], list[int]]:
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
        return False, [0, 0, 0, 0], [0, 0, 0, 0]
    try:
        vals = [float(v) for v in raw_box]
    except (TypeError, ValueError):
        return False, [0, 0, 0, 0], [0, 0, 0, 0]
    if not all(np.isfinite(vals)):
        return False, [0, 0, 0, 0], [0, 0, 0, 0]

    max_v = max(abs(v) for v in vals)
    is_float_unit = any(isinstance(v, float) for v in raw_box) and max_v <= 1.05
    if (max_v <= 1.05 and any(0.0 < abs(v) < 1.0 for v in vals)) or is_float_unit:
        # [0, 1] normalized floats (including [0.0, 0.0, 1.0, 1.0] full-frame)
        nx1, ny1, nx2, ny2 = [v * 1000.0 for v in vals]
    elif max_v > 1000.0:
        # Pixel coordinates (accept sent image dims, camera ref dims, or standard 1280x720)
        mx_x = max(vals[0], vals[2])
        mx_y = max(vals[1], vals[3])
        if mx_x <= img_w + 2:
            denom_w = float(max(1, img_w))
        elif mx_x <= ref_w + 2:
            denom_w = float(max(1, ref_w))
        else:
            denom_w = float(max(1280, ref_w, img_w))
        if mx_y <= img_h + 2:
            denom_h = float(max(1, img_h))
        elif mx_y <= ref_h + 2:
            denom_h = float(max(1, ref_h))
        else:
            denom_h = float(max(720, ref_h, img_h))
        nx1 = vals[0] * 1000.0 / denom_w
        ny1 = vals[1] * 1000.0 / denom_h
        nx2 = vals[2] * 1000.0 / denom_w
        ny2 = vals[3] * 1000.0 / denom_h
    else:
        nx1, ny1, nx2, ny2 = vals

    x1 = int(round(max(0.0, min(1000.0, min(nx1, nx2)))))
    x2 = int(round(max(0.0, min(1000.0, max(nx1, nx2)))))
    y1 = int(round(max(0.0, min(1000.0, min(ny1, ny2)))))
    y2 = int(round(max(0.0, min(1000.0, max(ny1, ny2)))))

    px1 = int(round(x1 * ref_w / 1000.0))
    py1 = int(round(y1 * ref_h / 1000.0))
    px2 = int(round(x2 * ref_w / 1000.0))
    py2 = int(round(y2 * ref_h / 1000.0))

    # Scale min_box_px gracefully on low-resolution cameras while preserving exact 1280x720 semantics
    eff_min_w = max(2, int(round(min_box_px * min(1.0, ref_w / 1280.0))))
    eff_min_h = max(2, int(round(min_box_px * min(1.0, ref_h / 720.0))))
    if (px2 - px1) < eff_min_w or (py2 - py1) < eff_min_h or x2 <= x1 or y2 <= y1:
        return False, [0, 0, 0, 0], [0, 0, 0, 0]

    return True, [x1, y1, x2, y2], [px1, py1, px2, py2]


def _entry_to_keyframe(
    entry: Any,
    *,
    row_idx: int,
    t_val: float,
    img_w: int,
    img_h: int,
    ref_w: int,
    ref_h: int,
    min_box_px: int,
) -> GroundingKeyframe:
    valid_status = {"neutral", "warn", "error", "good"}
    if isinstance(entry, dict) and bool(entry.get("visible", True)):
        ok, b_norm, b_px = _normalize_bbox(
            entry.get("bbox") or entry.get("bbox_norm_1000"),
            img_w=img_w,
            img_h=img_h,
            ref_w=ref_w,
            ref_h=ref_h,
            min_box_px=min_box_px,
        )
        state = str(entry.get("state") or "")
        sk_raw = str(entry.get("status_kind") or "neutral")
        sk = sk_raw if sk_raw in valid_status else "neutral"
        return GroundingKeyframe(
            row=row_idx,
            time_s=round(t_val, 3),
            visible=ok,
            bbox_norm_1000=b_norm if ok else [0, 0, 0, 0],
            bbox_px=b_px if ok else [0, 0, 0, 0],
            state=state if ok else "",
            status_kind=sk if ok else "neutral",  # type: ignore[arg-type]
        )
    return GroundingKeyframe(
        row=row_idx,
        time_s=round(t_val, 3),
        visible=False,
        bbox_norm_1000=[0, 0, 0, 0],
        bbox_px=[0, 0, 0, 0],
        state="",
        status_kind="neutral",
    )


def _partition_batches(
    times_s: Sequence[float],
    frames: Sequence[np.ndarray],
    batch_size: int,
) -> list[list[tuple[int, float, np.ndarray]]]:
    """Partition keyframes into balanced temporal batches.

    When ``batch_size <= 1``, emit 1-element batches preserving the legacy
    single-frame path. Otherwise use balanced ``divmod`` splitting so tail
    batches never degrade to an isolated 1-frame batch when ``len > 1``.
    """
    n = min(len(times_s), len(frames))
    if n == 0:
        return []
    if batch_size <= 1 or n <= 1:
        return [[(i, float(times_s[i]), frames[i])] for i in range(n)]

    n_batches = math.ceil(n / int(batch_size))
    q, r = divmod(n, n_batches)
    batches: list[list[tuple[int, float, np.ndarray]]] = []
    pos = 0
    for b in range(n_batches):
        sz = q + (1 if b < r else 0)
        batches.append(
            [(i, float(times_s[i]), frames[i]) for i in range(pos, pos + sz)]
        )
        pos += sz
    return batches


def _parse_optional_int(val: Any) -> int | None:
    if isinstance(val, bool) or val is None:
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val) if math.isfinite(val) and val.is_integer() else None
    if isinstance(val, str):
        s = val.strip()
        if s.lstrip("+-").isdigit():
            try:
                return int(s)
            except ValueError:
                return None
    return None


def _box_norm_to_gray_px(box_norm: Sequence[int], gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    return np.array(
        [
            float(box_norm[0]) * w / 1000.0,
            float(box_norm[1]) * h / 1000.0,
            float(box_norm[2]) * w / 1000.0,
            float(box_norm[3]) * h / 1000.0,
        ],
        dtype=np.float64,
    )


def _is_blank_box_on_textured_frame(box_norm: Sequence[int], gray: np.ndarray) -> bool:
    """Return True if box_norm lands on a flat/blank region of an otherwise textured frame."""
    if float(np.std(gray)) < 8.0:
        return False
    std_v, grad_v = _patch_texture_stats(gray, _box_norm_to_gray_px(box_norm, gray))
    return std_v < 5.5 and grad_v < 3.5


def _is_batch0_phantom_hallucination(
    batch: Sequence[tuple[int, float, np.ndarray]],
    parsed_b0: dict[int, dict[str, Any]],
    *,
    is_fixed_camera: bool,
    spec_by_id: dict[str, Any],
    img_w: int,
    img_h: int,
    ref_w: int,
    ref_h: int,
    min_box_px: int,
    fps: float,
) -> bool:
    """Physical-causality sanity guard detecting blind-node Batch-0 hallucinations.

    Triggers when (on a textured fixed-camera clip):
    1. A dynamic object supposedly teleports >180 norm units inside Batch 0 while the actual
       video pixels at both its start and end boxes barely changed (<12% pixels with |diff|>15); OR
    2. All detected static containers in Batch 0 land on flat/blank regions (std<5.5 & grad<3.5).
    """
    if not is_fixed_camera or len(batch) < 2:
        return False
    grays = [
        cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY) if fr.ndim == 3 else fr
        for _, _, fr in batch
    ]
    if float(np.std(grays[0])) < 8.0:
        return False

    oids: list[str] = list(spec_by_id.keys())
    for g_idx, _, _ in batch:
        for oid in parsed_b0.get(g_idx, {}):
            if isinstance(oid, str) and oid and oid not in oids:
                oids.append(oid)

    static_seen = 0
    static_blank = 0
    for oid in oids:
        spec = spec_by_id.get(oid)
        is_static = bool(spec.is_static) if spec else (oid in STATIC_OBJECT_DEFAULTS)
        vis_locs: list[tuple[int, list[int]]] = []
        for loc, (g_idx, t_val, _) in enumerate(batch):
            kf = _entry_to_keyframe(
                parsed_b0.get(g_idx, {}).get(oid),
                row_idx=max(0, int(round(t_val * fps))),
                t_val=t_val,
                img_w=img_w,
                img_h=img_h,
                ref_w=ref_w,
                ref_h=ref_h,
                min_box_px=min_box_px,
            )
            if kf.visible:
                vis_locs.append((loc, kf.bbox_norm_1000))

        if not vis_locs:
            continue
        if is_static:
            static_seen += 1
            loc0, b0_box = vis_locs[0]
            if _is_blank_box_on_textured_frame(b0_box, grays[loc0]):
                static_blank += 1
        elif len(vis_locs) >= 2:
            loc_a, box_a = vis_locs[0]
            loc_b, box_b = vis_locs[-1]
            if box_center_dist(box_a, box_b) > 180.0:
                ga = grays[loc_a].astype(np.int16)
                gb = grays[loc_b].astype(np.int16)
                h, w = ga.shape[:2]

                def _change_ratio(b_norm: list[int]) -> float:
                    x1 = max(0, min(w - 2, int(round(b_norm[0] * w / 1000.0))))
                    y1 = max(0, min(h - 2, int(round(b_norm[1] * h / 1000.0))))
                    x2 = max(x1 + 2, min(w, int(round(b_norm[2] * w / 1000.0))))
                    y2 = max(y1 + 2, min(h, int(round(b_norm[3] * h / 1000.0))))
                    diff = np.abs(ga[y1:y2, x1:x2] - gb[y1:y2, x1:x2])
                    return float(np.mean(diff > 15)) if diff.size else 1.0

                if _change_ratio(box_a) < 0.12 and _change_ratio(box_b) < 0.12:
                    return True

    if static_seen > 0 and static_blank == static_seen:
        return True
    return False


def _select_static_consensus_box(
    kfs: Sequence[GroundingKeyframe],
    batch_id_of_idx: Sequence[int],
    frame_grays: Sequence[np.ndarray],
    *,
    ref_w: int,
    ref_h: int,
    jitter_tol_px: float,
    untrusted_batches: set[int] | None = None,
) -> tuple[list[int], list[int]] | None:
    """Select fixed-camera static container box via cross-batch consensus clustering.

    Prevents a single bad/poisoned batch (including Batch 0) from overriding multi-batch
    agreement, filters out blank-tabletop hallucinations, and breaks ties in favor of
    trusted earlier batches (Batch 0) and stronger visual texture.
    """
    untrusted = untrusted_batches or set()
    vis_entries: list[tuple[int, int, GroundingKeyframe, float, bool]] = []
    for idx, kf in enumerate(kfs):
        if not kf.visible:
            continue
        b_id = batch_id_of_idx[idx] if idx < len(batch_id_of_idx) else 0
        grad_v = 0.0
        is_blank = False
        if idx < len(frame_grays):
            gray = frame_grays[idx]
            if float(np.std(gray)) >= 8.0:
                std_v, grad_v = _patch_texture_stats(
                    gray, _box_norm_to_gray_px(kf.bbox_norm_1000, gray)
                )
                is_blank = std_v < 5.5 and grad_v < 3.5
        vis_entries.append((idx, b_id, kf, grad_v, is_blank))

    if not vis_entries:
        return None

    non_blank = [e for e in vis_entries if not e[4]]
    pool = non_blank if non_blank else vis_entries

    by_batch: dict[int, list[tuple[int, GroundingKeyframe, float]]] = {}
    for idx, b_id, kf, grad_v, _ in pool:
        by_batch.setdefault(b_id, []).append((idx, kf, grad_v))

    batch_meds_px: dict[int, np.ndarray] = {}
    batch_grads: dict[int, float] = {}
    for b_id, items in by_batch.items():
        arr_px = np.asarray([kf.bbox_px for _, kf, _ in items], dtype=np.float64)
        batch_meds_px[b_id] = np.median(arr_px, axis=0)
        batch_grads[b_id] = float(np.mean([g for _, _, g in items]))

    batch_ids = sorted(by_batch.keys())
    best_score: tuple[float, int, int, float] | None = None
    best_anchor_px: np.ndarray | None = None
    best_cluster_batches: list[int] = []

    for anchor_b in batch_ids:
        a_px = batch_meds_px[anchor_b]
        cluster_b = [
            b_id
            for b_id in batch_ids
            if float(np.max(np.abs(batch_meds_px[b_id] - a_px))) <= jitter_tol_px
        ]
        trusted_weight = sum(0.25 if b_id in untrusted else 1.0 for b_id in cluster_b)
        support_frames = sum(
            1
            for b_id in cluster_b
            for _, kf, _ in by_batch[b_id]
            if float(np.max(np.abs(np.asarray(kf.bbox_px, dtype=np.float64) - a_px)))
            <= jitter_tol_px
        )
        has_trusted_b0 = 1 if (0 in cluster_b and 0 not in untrusted) else 0
        mean_grad = float(np.mean([batch_grads[b_id] for b_id in cluster_b]))
        score = (round(trusted_weight, 4), support_frames, has_trusted_b0, round(mean_grad, 3))
        if best_score is None or score > best_score:
            best_score = score
            best_anchor_px = a_px
            best_cluster_batches = cluster_b

    assert best_anchor_px is not None
    inlier_norms: list[list[int]] = []
    for b_id in best_cluster_batches:
        for _, kf, _ in by_batch[b_id]:
            if (
                float(np.max(np.abs(np.asarray(kf.bbox_px, dtype=np.float64) - best_anchor_px)))
                <= jitter_tol_px
            ):
                inlier_norms.append(kf.bbox_norm_1000)
    if not inlier_norms:
        inlier_norms = [kf.bbox_norm_1000 for b_id in best_cluster_batches for _, kf, _ in by_batch[b_id]]

    med = np.median(np.asarray(inlier_norms, dtype=np.float64), axis=0)
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
    return med_norm, med_px


def _draw_reference_boxes(
    frame_rgb: np.ndarray,
    boxes_by_oid: dict[str, list[int]],
) -> np.ndarray:
    """Draw colored 2D anchor boxes and obj_id labels onto a copy of frame_rgb."""
    canvas = np.ascontiguousarray(frame_rgb.copy())
    h, w = canvas.shape[:2]
    if h < 4 or w < 4 or not boxes_by_oid:
        return canvas

    # Subtle top-left badge identifying this frame as reference-only anchor context
    badge = "REF ANCHOR"
    (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(canvas, (4, 4), (min(w - 1, 14 + bw), min(h - 1, 12 + bh)), (18, 24, 38), -1)
    cv2.putText(
        canvas,
        badge,
        (9, min(h - 3, 9 + bh)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (250, 204, 21),
        1,
        cv2.LINE_AA,
    )

    for idx, (oid, box_norm) in enumerate(boxes_by_oid.items()):
        if len(box_norm) != 4:
            continue
        x1 = int(round(max(0.0, min(float(w - 2), box_norm[0] * w / 1000.0))))
        y1 = int(round(max(0.0, min(float(h - 2), box_norm[1] * h / 1000.0))))
        x2 = int(round(max(float(x1 + 1), min(float(w - 1), box_norm[2] * w / 1000.0))))
        y2 = int(round(max(float(y1 + 1), min(float(h - 1), box_norm[3] * h / 1000.0))))
        color = _REF_BOX_PALETTE_RGB[idx % len(_REF_BOX_PALETTE_RGB)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        label = str(oid)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx1 = max(0, min(w - tw - 8, x1))
        ty1 = y1 - th - 6 if y1 >= th + 8 else min(h - th - 6, y1 + 2)
        ty1 = max(0, ty1)
        tx2 = min(w - 1, tx1 + tw + 8)
        ty2 = min(h - 1, ty1 + th + 6)
        cv2.rectangle(canvas, (tx1, ty1), (tx2, ty2), color, -1)
        luma = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_rgb = (15, 15, 15) if luma > 150.0 else (255, 255, 255)
        cv2.putText(
            canvas,
            label,
            (tx1 + 4, max(th, ty2 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_rgb,
            1,
            cv2.LINE_AA,
        )
    return canvas


def _extract_entry_time_s(entry: dict[str, Any]) -> float | None:
    raw_t = entry.get("time_s")
    if raw_t is None:
        raw_t = entry.get("timestamp_s")
    if raw_t is None:
        raw_t = entry.get("t")
    if raw_t is None:
        return None
    try:
        t_val = float(raw_t)
    except (TypeError, ValueError):
        return None
    return t_val if math.isfinite(t_val) else None


def _is_explicit_ref_entry(
    entry: dict[str, Any],
    *,
    ref_time_s: float | None,
    batch_min_t: float,
) -> bool:
    for key in ("label", "frame_label", "frame_id", "id"):
        val = entry.get(key)
        if isinstance(val, str) and "ref" in val.lower():
            return True
    raw_fi = entry.get("frame_index")
    if raw_fi is None:
        raw_fi = entry.get("index")
    fi = _parse_optional_int(raw_fi)
    if fi is not None and fi < 0:
        return True
    if ref_time_s is not None:
        t_val = _extract_entry_time_s(entry)
        if t_val is not None:
            ref_gap = batch_min_t - ref_time_s
            if ref_gap > 1e-4:
                ref_tol = min(0.25, 0.45 * ref_gap)
                if abs(t_val - ref_time_s) <= ref_tol and abs(t_val - ref_time_s) < abs(
                    t_val - batch_min_t
                ):
                    return True
    return False


def _parse_batch_response(
    resp: Any,
    batch: Sequence[tuple[int, float, np.ndarray]],
    *,
    has_ref_prefix: bool = False,
    ref_time_s: float | None = None,
) -> dict[int, dict[str, Any]]:
    """Parse batched multi-frame or legacy single-frame S1 JSON into {global_idx: objects}.

    When ``has_ref_prefix=True``, tolerantly skips any mistakenly emitted ``ref_frame_0``
    entry and detects +1 index shifts so entries strictly map back to ``0..len(batch)-1``.
    """
    out: dict[int, dict[str, Any]] = {g_idx: {} for g_idx, _, _ in batch}
    if not batch or not isinstance(resp, dict):
        return out

    frames_list = resp.get("frames")
    if not isinstance(frames_list, list):
        frames_list = resp.get("keyframes")

    if isinstance(frames_list, list):
        b_len = len(batch)
        claimed: set[int] = set()
        matched_entries: set[int] = set()
        idx_by_global = {g_idx: loc for loc, (g_idx, _, _) in enumerate(batch)}
        min_gap = min(
            (abs(batch[k + 1][1] - batch[k][1]) for k in range(b_len - 1)),
            default=0.0,
        )
        tol = max(0.25, 0.49 * min_gap) if (b_len > 1 and min_gap > 0.0) else 0.5

        dict_entries = [e for e in frames_list if isinstance(e, dict)]
        dropped_ref = False
        is_shifted_plus_1 = False

        if has_ref_prefix:
            batch_min_t = min(t_val for _, t_val, _ in batch)
            filtered: list[dict[str, Any]] = []
            for entry in dict_entries:
                if _is_explicit_ref_entry(
                    entry, ref_time_s=ref_time_s, batch_min_t=batch_min_t
                ):
                    dropped_ref = True
                    continue
                filtered.append(entry)

            # Fallback: if model emitted b_len + 1 entries without label/timestamp on ref_frame_0
            if not dropped_ref and len(filtered) == b_len + 1:
                first_t = _extract_entry_time_s(filtered[0])
                if first_t is None or abs(first_t - batch[0][1]) > tol:
                    filtered = filtered[1:]
                    dropped_ref = True

            dict_entries = filtered

            # Detect +1 frame_index shift (e.g. 1..b_len corresponding to frame_0..frame_{b_len-1})
            pairs_fi_t: list[tuple[int, float]] = []
            all_fis: list[int] = []
            for entry in dict_entries:
                raw_fi = entry.get("frame_index")
                if raw_fi is None:
                    raw_fi = entry.get("index")
                fi = _parse_optional_int(raw_fi)
                if fi is None:
                    continue
                all_fis.append(fi)
                t_val = _extract_entry_time_s(entry)
                if t_val is not None:
                    pairs_fi_t.append((fi, t_val))

            if pairs_fi_t:
                score_0 = sum(
                    1
                    for fi, tv in pairs_fi_t
                    if 0 <= fi < b_len and abs(batch[fi][1] - tv) <= tol
                )
                score_1 = sum(
                    1
                    for fi, tv in pairs_fi_t
                    if 1 <= fi <= b_len and abs(batch[fi - 1][1] - tv) <= tol
                )
                if score_1 > score_0:
                    is_shifted_plus_1 = True
            if (
                not is_shifted_plus_1
                and all_fis
                and (not pairs_fi_t)
                and 0 not in all_fis
                and all(1 <= fi <= b_len for fi in all_fis)
                and (b_len in all_fis or dropped_ref)
                and not (set(all_fis) <= set(idx_by_global.keys()) and not dropped_ref)
            ):
                is_shifted_plus_1 = True

        # Pass 1: explicit frame_index / index
        for e_i, entry in enumerate(dict_entries):
            raw_fi = entry.get("frame_index")
            if raw_fi is None:
                raw_fi = entry.get("index")
            fi = _parse_optional_int(raw_fi)
            if fi is None:
                continue
            eff_fi = fi - 1 if is_shifted_plus_1 else fi
            target_loc: int | None = None
            if 0 <= eff_fi < b_len and eff_fi not in claimed:
                target_loc = eff_fi
            elif (
                not is_shifted_plus_1
                and fi in idx_by_global
                and idx_by_global[fi] not in claimed
            ):
                target_loc = idx_by_global[fi]
            if target_loc is not None:
                claimed.add(target_loc)
                matched_entries.add(e_i)
                objs = entry.get("objects")
                out[batch[target_loc][0]] = objs if isinstance(objs, dict) else {}

        # Pass 2: closest unclaimed time_s / timestamp_s within tolerance
        for e_i, entry in enumerate(dict_entries):
            if e_i in matched_entries:
                continue
            raw_fi = entry.get("frame_index")
            if raw_fi is None:
                raw_fi = entry.get("index")
            if _parse_optional_int(raw_fi) is not None:
                continue
            t_val = _extract_entry_time_s(entry)
            if t_val is None:
                continue
            best_loc: int | None = None
            best_diff = float("inf")
            for loc in range(b_len):
                if loc in claimed:
                    continue
                diff = abs(batch[loc][1] - t_val)
                if diff <= tol and diff < best_diff:
                    best_diff = diff
                    best_loc = loc
            if best_loc is not None:
                claimed.add(best_loc)
                matched_entries.add(e_i)
                objs = entry.get("objects")
                out[batch[best_loc][0]] = objs if isinstance(objs, dict) else {}

        # Pass 3: positional fallback for entries without explicit out-of-range index
        for e_i, entry in enumerate(dict_entries):
            if e_i in matched_entries:
                continue
            raw_fi = entry.get("frame_index")
            if raw_fi is None:
                raw_fi = entry.get("index")
            if _parse_optional_int(raw_fi) is not None:
                continue
            next_loc = next((loc for loc in range(b_len) if loc not in claimed), None)
            if next_loc is None:
                break
            claimed.add(next_loc)
            matched_entries.add(e_i)
            objs = entry.get("objects")
            out[batch[next_loc][0]] = objs if isinstance(objs, dict) else {}

        return out

    # Backward-compatible single-frame / legacy mock {"objects": {...}} fallback
    objs = resp.get("objects")
    if isinstance(objs, dict):
        for g_idx, _, _ in batch:
            out[g_idx] = dict(objs)
    return out


def run_stage1_grounding(
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
    cancel_event: threading.Event | None = None,
    custom_objects: Sequence[str] | None = None,
) -> GroundingResult:
    """Execute Stage 1 HD 2D grounding + optional BAFT local frame-level 2D tracking."""
    ev = view or episode_view
    cl = client or vlm_client
    cm = cfg if isinstance(cfg, ConfigManager) else (config if isinstance(config, ConfigManager) else None)
    t_spec = task or task_spec or (cm.get_task(ev.meta.task_id) if cm and ev else TaskSpec(task_id="_default"))
    assert ev is not None and cl is not None

    def _check_cancelled() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise VlmCancelledError("Stage 1 grounding cancelled")
        check_client_cancelled(cl)

    s1_cfg = _resolve_s1_cfg(cfg or config)
    tracking_cfg = TrackingConfig.from_dict(s1_cfg.get("tracking"))

    main_cam = ev.camera_roles.get("main") or (ev.media.cameras[0].name if ev.media.cameras else "")
    ref_w, ref_h = _resolve_main_camera_ref_size(ev, main_cam, s1_cfg)

    if not bool(s1_cfg.get("enabled", True)):
        return GroundingResult(
            schema_version=1,
            camera_name=main_cam or None,
            ref_size=[ref_w, ref_h],
            keyframes_s=[],
            object_tracks_2d={},
        )

    if not main_cam:
        return GroundingResult(
            schema_version=1,
            camera_name=None,
            ref_size=[ref_w, ref_h],
            keyframes_s=[],
            object_tracks_2d={},
        )

    max_kf_cfg = max(1, int(s1_cfg.get("max_keyframes", 36)))
    hard_max_kf = max(1, int(s1_cfg.get("hard_max_keyframes", 72)))
    total_cap = min(max_kf_cfg, hard_max_kf)

    # When adaptive re-anchoring is enabled, reserve budget inside total_cap so
    # total VLM calls strictly NEVER exceed max_keyframes.
    reserve = (
        min(tracking_cfg.adaptive_reanchor_max_calls, max(0, total_cap - 2))
        if (tracking_cfg.enabled and tracking_cfg.adaptive_reanchor_max_calls > 0)
        else 0
    )
    init_cap = max(1, total_cap - reserve) if total_cap > 1 else 1

    times_s = _plan_keyframe_times(
        ev.duration_s,
        float(s1_cfg.get("interval_s", 2.5)),
        init_cap,
        hard_max_kf,
    )
    ego_h = int(s1_cfg.get("ego_height", 540))
    jpeg_q = int(s1_cfg.get("jpeg_quality", 80))
    min_box_px = int(s1_cfg.get("min_box_px", 12))
    thinking = bool(s1_cfg.get("thinking", False))
    thinking_budget = int(s1_cfg.get("thinking_budget", 768))
    max_tokens = int(s1_cfg.get("max_tokens", 2400))
    batch_size = max(1, int(s1_cfg.get("batch_size", 4)))
    batch_concurrency = max(1, int(s1_cfg.get("batch_concurrency", 3)))
    reference_anchor_enabled = bool(s1_cfg.get("reference_anchor_enabled", True))
    pass0_image_detail = s1_cfg.get("pass0_image_detail", "low")
    s1_image_detail = s1_cfg.get("image_detail", "high")
    merge_pass0_short_episode = bool(s1_cfg.get("merge_pass0_short_episode", True))
    static_min_frames = int(s1_cfg.get("static_container_min_frames", 5))
    jitter_tol_px = float(s1_cfg.get("static_jitter_tol_px_1280", 85.0)) * (ref_w / 1280.0)

    pb = prompt_builder or (PromptBuilder(cm) if cm is not None else None)

    _check_cancelled()
    raw_frames = ev.media.read_frames_batch(main_cam, times_s)
    _check_cancelled()
    resized_frames = [_resize_to_height(fr, ego_h) for fr in raw_frames]
    img_h, img_w = resized_frames[0].shape[:2] if resized_frames else (ref_h, ref_w)
    del raw_frames

    # Direct ~30ms fixed-vs-moving camera check BEFORE VLM calls
    is_fixed_camera = not detect_camera_motion_direct(
        ev,
        main_cam,
        sample_times_s=None,
        target_w=min(320, tracking_cfg.track_width),
        motion_threshold_norm=tracking_cfg.camera_motion_threshold_norm,
        check_cancelled=_check_cancelled,
    )

    def _has_any_grounding_box(parsed: dict[int, dict[str, Any]]) -> bool:
        for objs in parsed.values():
            if not isinstance(objs, dict):
                continue
            for entry in objs.values():
                if isinstance(entry, dict) and (
                    "bbox" in entry or "bbox_norm_1000" in entry or "visible" in entry
                ):
                    return True
        return False

    # Lock canonical grounding object vocabulary:
    # 1) explicit custom_objects override, else
    # 2) preconfigured task.grounding_objects, else
    # 3) Short-episode merged Pass-0 + Batch-0 (when enabled and len(keyframes) <= batch_size), else
    # 4) Pass-0 lightweight canonical discovery on 4-6 evenly spaced overview frames
    effective_task = t_spec
    merged_short_objs: dict[int, dict[str, Any]] | None = None
    custom_specs = build_custom_grounding_objects(custom_objects)
    needs_pass0 = (
        not custom_specs
        and not t_spec.grounding_objects
        and len(resized_frames) > 1
        and bool(s1_cfg.get("canonical_discovery_enabled", True))
    )

    if custom_specs:
        effective_task = t_spec.model_copy(update={"grounding_objects": custom_specs})
    elif needs_pass0:
        can_merge_short = (
            merge_pass0_short_episode
            and batch_size > 1
            and len(resized_frames) <= batch_size
        )
        run_separate_pass0 = True
        if can_merge_short:
            _check_cancelled()
            if pb is not None:
                comb_sys, comb_usr = pb.build_s1_discover_and_ground(
                    task=t_spec, camera_roles=ev.camera_roles
                )
            else:
                comb_sys = (
                    "You are an embodied manipulation visual grounding expert "
                    "(2D Grounding canonical vocabulary discovery and batched keyframe grounding, "
                    "supporting both robot arms/grippers and human hands). "
                    "Select 1 to 4 task-relevant physical objects and ground them in every frame with normalized [0, 1000] bboxes. "
                    'Output strict JSON only: {"canonical_objects": [{"id": "snake_case_id", "name": str, "desc": str, "is_static": bool}], '
                    '"frames": [{"frame_index": 0, "time_s": 0.0, "objects": {obj_id: {"visible": bool, "bbox": [x1,y1,x2,y2], "state": str, "status_kind": "neutral|warn|error|good"}}}]}.'
                )
                comb_usr = (
                    f"Task: {t_spec.task_id}\n"
                    "Identify 1 to 4 canonical physical objects and output their 2D bounding boxes for every keyframe in a single JSON response."
                )
            short_batch = [
                (i, float(times_s[i]), resized_frames[i])
                for i in range(len(resized_frames))
            ]
            try:
                comb_resp = cl.chat_json(
                    stage="s1_grounding",
                    system_prompt=comb_sys,
                    user_prompt=comb_usr,
                    images=resized_frames,
                    timestamps_s=times_s,
                    labels=[f"frame_{i}" for i in range(len(resized_frames))],
                    thinking=thinking,
                    thinking_budget=thinking_budget,
                    max_tokens=max(max_tokens, 280 * len(resized_frames) + 400),
                    jpeg_quality=jpeg_q,
                    image_detail=s1_image_detail,
                    budget=budget,
                )
                _check_cancelled()
                discovered_specs = _parse_canonical_objects_response(
                    comb_resp,
                    max_objects=int(s1_cfg.get("canonical_discovery_max_objects", 4)),
                )
                if discovered_specs:
                    effective_task = t_spec.model_copy(
                        update={"grounding_objects": discovered_specs}
                    )
                parsed_comb = _parse_batch_response(comb_resp, short_batch)
                if _has_any_grounding_box(parsed_comb):
                    merged_short_objs = parsed_comb
                    run_separate_pass0 = False
                elif discovered_specs:
                    # Canonical objects discovered, boxes missing -> fall back directly to Batch-0 grounding
                    run_separate_pass0 = False
            except (BudgetExceeded, VlmCancelledError):
                raise
            except Exception:
                # Fall back seamlessly to separate Pass-0 + Batch-0
                _check_cancelled()

        if run_separate_pass0:
            disc_indices = _select_discovery_indices(
                len(resized_frames),
                max_frames=int(s1_cfg.get("canonical_discovery_frames", 5)),
            )
            if disc_indices:
                _check_cancelled()
                if pb is not None:
                    disc_sys, disc_usr = pb.build_s1_discovery(
                        task=t_spec, camera_roles=ev.camera_roles
                    )
                else:
                    disc_sys = (
                        "You are an embodied manipulation visual grounding expert (2D Grounding canonical vocabulary discovery). "
                        "Inspect the overview keyframes and select 1 to 4 task-relevant physical objects prioritized by manipulator/hand interaction and target container/receptacle. "
                        'Output strict JSON only: {"canonical_objects": [{"id": "snake_case_id", "name": str, "desc": str, "is_static": bool}]}.'
                    )
                    disc_usr = (
                        f"Task: {t_spec.task_id}\n"
                        "Identify 1 to 4 canonical physical objects actively manipulated by robot arms/grippers or human hands plus target receptacle."
                    )
                disc_frames = [resized_frames[i] for i in disc_indices]
                disc_times = [times_s[i] for i in disc_indices]
                try:
                    disc_resp = cl.chat_json(
                        stage="s1_grounding",
                        system_prompt=disc_sys,
                        user_prompt=disc_usr,
                        images=disc_frames,
                        timestamps_s=disc_times,
                        labels=[f"overview_{i}" for i in range(len(disc_frames))],
                        thinking=False,
                        thinking_budget=0,
                        max_tokens=min(800, max_tokens),
                        jpeg_quality=jpeg_q,
                        image_detail=pass0_image_detail,
                        budget=budget,
                    )
                    _check_cancelled()
                    discovered_specs = _parse_canonical_objects_response(
                        disc_resp,
                        max_objects=int(s1_cfg.get("canonical_discovery_max_objects", 4)),
                    )
                    if discovered_specs:
                        effective_task = t_spec.model_copy(
                            update={"grounding_objects": discovered_specs}
                        )
                except (BudgetExceeded, VlmCancelledError):
                    raise
                except Exception:
                    # Fall back seamlessly to open-vocab grounding if Pass-0 returns unexpected/failed response
                    _check_cancelled()

    if pb is not None:
        sys_p, usr_p = pb.build_s1(task=effective_task, camera_roles=ev.camera_roles)
    else:
        if batch_size > 1:
            sys_p = (
                "You are an embodied manipulation 2D visual grounding expert (2D Grounding). "
                "Output strict JSON with normalized [0, 1000] integer bounding boxes [x1, y1, x2, y2] for all keyframes in this batch: "
                '{"frames": [{"frame_index": 0, "time_s": 0.0, "objects": {obj_id: {"visible": bool, "bbox": [x1,y1,x2,y2], "state": str, "status_kind": "neutral|warn|error|good"}}}]}. '
                "Maintain target identity lock and consistent semantic box boundaries across frames."
            )
        else:
            sys_p = (
                "You are an embodied manipulation 2D visual grounding expert (2D Grounding). "
                'Output strict JSON only: {"objects": {obj_id: {"visible": bool, "bbox": [x1,y1,x2,y2], "state": str, "status_kind": "neutral|warn|error|good"}}}.'
            )
        obj_lines = "\n".join(
            f"- {g.id}: {g.name} ({g.desc})" for g in effective_task.grounding_objects
        ) or "- target_object: Target Object (primary manipulated object)"
        usr_p = f"Task: {effective_task.task_id}\nTarget objects to ground:\n{obj_lines}"

    spec_by_id = {g.id: g for g in effective_task.grounding_objects}
    fps = float(ev.fps) if ev.fps > 0 else 30.0

    def _call_batch(
        batch: Sequence[tuple[int, float, np.ndarray]],
        *,
        ref_img: np.ndarray | None = None,
        ref_t: float | None = None,
        ref_hint: str = "",
    ) -> tuple[dict[int, dict[str, Any]], Exception | None]:
        _check_cancelled()
        batch_times = [t_val for _, t_val, _ in batch]
        batch_frames = [fr for _, _, fr in batch]
        has_ref = ref_img is not None and ref_t is not None and batch_size > 1
        eff_max_tokens = (
            max(max_tokens, 280 * len(batch_frames)) if batch_size > 1 else max_tokens
        )
        eff_user_prompt = f"{usr_p}\n\n{ref_hint}" if (has_ref and ref_hint) else usr_p
        chat_kw: dict[str, Any] = {
            "stage": "s1_grounding",
            "system_prompt": sys_p,
            "user_prompt": eff_user_prompt,
            "images": ([ref_img, *batch_frames] if has_ref and ref_img is not None else batch_frames),
            "timestamps_s": (
                [float(ref_t), *batch_times] if has_ref and ref_t is not None else batch_times
            ),
            "thinking": thinking,
            "thinking_budget": thinking_budget,
            "max_tokens": eff_max_tokens,
            "jpeg_quality": jpeg_q,
            "image_detail": s1_image_detail,
            "budget": budget,
        }
        if batch_size > 1:
            frame_labels = [f"frame_{i}" for i in range(len(batch_frames))]
            chat_kw["labels"] = ["ref_frame_0", *frame_labels] if has_ref else frame_labels
        try:
            resp = cl.chat_json(**chat_kw)
            _check_cancelled()
            return (
                _parse_batch_response(
                    resp,
                    batch,
                    has_ref_prefix=has_ref,
                    ref_time_s=ref_t if has_ref else None,
                ),
                None,
            )
        except (BudgetExceeded, VlmCancelledError):
            raise
        except Exception as exc:
            _check_cancelled()
            return {}, exc

    batches = _partition_batches(times_s, resized_frames, batch_size)
    batch_id_of_idx: list[int] = [0] * len(times_s)
    for b_idx, batch in enumerate(batches):
        for g_idx, _, _ in batch:
            if 0 <= g_idx < len(batch_id_of_idx):
                batch_id_of_idx[g_idx] = b_idx

    frame_grays_small: list[np.ndarray] = []
    for fr in resized_frames:
        gray = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY) if fr.ndim == 3 else fr
        gh, gw = gray.shape[:2]
        if gw > 320 and gh > 0:
            nw = 320
            nh = max(32, int(round(gh * nw / float(gw))))
            gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
        frame_grays_small.append(gray)

    per_frame_objs: dict[int, dict[str, Any]] = {}
    errors: list[Exception] = []
    untrusted_batches: set[int] = set()
    ref_annotated_img: np.ndarray | None = None
    ref_t_val: float | None = None
    ref_hint_base = ""

    if merged_short_objs is not None:
        per_frame_objs.update(merged_short_objs)
    elif batch_size > 1 and len(batches) > 1 and reference_anchor_enabled:
        # Phase 1: execute Batch 0 alone to extract soft spatial reference anchors
        b0 = batches[0]
        parsed_b0, err_b0 = _call_batch(b0)
        if err_b0 is not None:
            raise err_b0

        # Physical-causality check on Batch 0: retry once if blind-node phantom motion detected
        if _is_batch0_phantom_hallucination(
            b0,
            parsed_b0,
            is_fixed_camera=is_fixed_camera,
            spec_by_id=spec_by_id,
            img_w=img_w,
            img_h=img_h,
            ref_w=ref_w,
            ref_h=ref_h,
            min_box_px=min_box_px,
            fps=fps,
        ):
            parsed_retry, err_retry = _call_batch(b0)
            if err_retry is None and not _is_batch0_phantom_hallucination(
                b0,
                parsed_retry,
                is_fixed_camera=is_fixed_camera,
                spec_by_id=spec_by_id,
                img_w=img_w,
                img_h=img_h,
                ref_w=ref_w,
                ref_h=ref_h,
                min_box_px=min_box_px,
                fps=fps,
            ):
                parsed_b0 = parsed_retry
            else:
                if err_retry is None:
                    parsed_b0 = parsed_retry
                untrusted_batches.add(0)

        per_frame_objs.update(parsed_b0)

        # Build soft visual reference ONLY if Batch 0 passed physical-causality sanity
        if 0 not in untrusted_batches:
            b0_oids: list[str] = [g.id for g in effective_task.grounding_objects]
            for g_idx, _, _ in b0:
                for oid in per_frame_objs.get(g_idx, {}):
                    if isinstance(oid, str) and oid and oid not in b0_oids:
                        b0_oids.append(oid)

            b0_static_ref_norm: dict[str, list[int]] = {}
            b0_vis_by_loc: list[dict[str, list[int]]] = [{} for _ in range(len(b0))]
            for oid in b0_oids:
                kfs_b0: list[GroundingKeyframe] = []
                for loc, (g_idx, t_val, _) in enumerate(b0):
                    kf = _entry_to_keyframe(
                        per_frame_objs.get(g_idx, {}).get(oid),
                        row_idx=max(0, int(round(t_val * fps))),
                        t_val=t_val,
                        img_w=img_w,
                        img_h=img_h,
                        ref_w=ref_w,
                        ref_h=ref_h,
                        min_box_px=min_box_px,
                    )
                    kfs_b0.append(kf)
                    if kf.visible and not _is_blank_box_on_textured_frame(
                        kf.bbox_norm_1000, frame_grays_small[g_idx]
                    ):
                        b0_vis_by_loc[loc][oid] = list(kf.bbox_norm_1000)

                spec = spec_by_id.get(oid)
                is_static = bool(spec.is_static) if spec else (oid in STATIC_OBJECT_DEFAULTS)
                if is_fixed_camera and is_static:
                    vis_kfs_b0 = [
                        k
                        for loc, k in enumerate(kfs_b0)
                        if k.visible
                        and not _is_blank_box_on_textured_frame(
                            k.bbox_norm_1000, frame_grays_small[b0[loc][0]]
                        )
                    ]
                    b0_stable_min = min(2, len(b0))
                    if len(vis_kfs_b0) >= b0_stable_min:
                        arr_px = np.asarray([k.bbox_px for k in vis_kfs_b0], dtype=np.float64)
                        init_med_px = np.median(arr_px, axis=0)
                        inliers = int(
                            np.count_nonzero(
                                np.max(np.abs(arr_px - init_med_px[None, :]), axis=-1)
                                <= jitter_tol_px
                            )
                        )
                        if inliers >= b0_stable_min:
                            _, med_norm = stabilize_static_keyframes(
                                vis_kfs_b0,
                                ref_w=ref_w,
                                ref_h=ref_h,
                                static_min_frames=b0_stable_min,
                                jitter_tol_px=jitter_tol_px,
                                default_state=STATIC_OBJECT_DEFAULTS.get(oid, "stationary"),
                            )
                            if med_norm is not None:
                                b0_static_ref_norm[oid] = list(med_norm)

            # Pick Batch 0's first visible frame (prefer t=0 / loc=0) as reference base image
            ref_loc: int | None = None
            if b0_vis_by_loc and b0_vis_by_loc[0]:
                ref_loc = 0
            else:
                for loc, vis_map in enumerate(b0_vis_by_loc):
                    if vis_map:
                        ref_loc = loc
                        break
            if ref_loc is None and b0_static_ref_norm and b0:
                ref_loc = 0

            if ref_loc is not None:
                ref_boxes: dict[str, list[int]] = {}
                for oid in b0_oids:
                    if oid in b0_static_ref_norm:
                        ref_boxes[oid] = list(b0_static_ref_norm[oid])
                    elif oid in b0_vis_by_loc[ref_loc]:
                        ref_boxes[oid] = list(b0_vis_by_loc[ref_loc][oid])
                if ref_boxes:
                    ref_t_val = float(b0[ref_loc][1])
                    ref_annotated_img = _draw_reference_boxes(b0[ref_loc][2], ref_boxes)
                    hint_lines = [
                        f"[Reference Anchor Frame (ref_frame_0, t={ref_t_val:.2f}s): use ONLY for target identity recognition and initial spatial reference; NEVER output an entry for ref_frame_0 in the frames array; if any reference box differs from actual pixels in a target frame, strictly follow the target frame's real pixel location rather than copying drifted coordinates]"
                    ]
                    for oid, box in ref_boxes.items():
                        spec = spec_by_id.get(oid)
                        name = spec.name if spec else oid
                        if oid in b0_static_ref_norm:
                            hint_lines.append(
                                f"- {oid} ({name}, stationary container reference): bbox={box} (verify against real image pixels; do not drift onto blank tabletop or jump to another container after occlusion)"
                            )
                        else:
                            hint_lines.append(
                                f"- {oid} ({name}, initial reference): bbox={box} (keep target identity locked on the same physical object and fit actual pixels tightly)"
                            )
                    ref_hint_base = "\n".join(hint_lines)

        # Phase 2: execute Batch 1..N with paced concurrency + quiet recovery on blind-node exhaustion
        rem_batches = batches[1:]
        workers = min(batch_concurrency, max(1, len(rem_batches)))

        def _submit_relay(
            batch: Sequence[tuple[int, float, np.ndarray]],
        ) -> tuple[dict[int, dict[str, Any]], Exception | None]:
            if ref_annotated_img is not None and ref_t_val is not None:
                batch_hint = (
                    f"{ref_hint_base}\n"
                    f"Output ONLY the {len(batch)} target frames in this batch (frame_0..frame_{len(batch) - 1}), "
                    "with frame_index starting at 0 strictly matching frame_0; do NOT include ref_frame_0."
                )
                return _call_batch(
                    batch,
                    ref_img=ref_annotated_img,
                    ref_t=ref_t_val,
                    ref_hint=batch_hint,
                )
            return _call_batch(batch)

        blind_retry_batches: list[Sequence[tuple[int, float, np.ndarray]]] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [(batch, pool.submit(_submit_relay, batch)) for batch in rem_batches]
            for batch, fut in futs:
                parsed_map, err = fut.result()
                if err is not None:
                    if isinstance(err, VlmBlindVisionError):
                        blind_retry_batches.append(batch)
                    else:
                        errors.append(err)
                else:
                    per_frame_objs.update(parsed_map)

        # Quiet sequential recovery for any batch that exhausted blind-node retries under burst load
        if not errors and blind_retry_batches:
            for batch in blind_retry_batches:
                _check_cancelled()
                parsed_map, err = _submit_relay(batch)
                if err is not None:
                    errors.append(err)
                else:
                    per_frame_objs.update(parsed_map)
    else:
        workers = min(batch_concurrency, max(1, len(batches)))
        blind_retry_batches_plain: list[Sequence[tuple[int, float, np.ndarray]]] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [(batch, pool.submit(_call_batch, batch)) for batch in batches]
            for batch, fut in futs:
                parsed_map, err = fut.result()
                if err is not None:
                    if isinstance(err, VlmBlindVisionError):
                        blind_retry_batches_plain.append(batch)
                    else:
                        errors.append(err)
                else:
                    per_frame_objs.update(parsed_map)

        if not errors and blind_retry_batches_plain:
            for batch in blind_retry_batches_plain:
                _check_cancelled()
                parsed_map, err = _call_batch(batch)
                if err is not None:
                    errors.append(err)
                else:
                    per_frame_objs.update(parsed_map)

    del resized_frames, batches
    if errors:
        raise errors[0]

    _check_cancelled()

    # Collect all seen object IDs + configured/discovered/custom object metadata
    all_oids: list[str] = [g.id for g in effective_task.grounding_objects]
    for idx in sorted(per_frame_objs):
        for oid in per_frame_objs[idx]:
            if isinstance(oid, str) and oid and oid not in all_oids:
                all_oids.append(oid)

    raw_tracks: dict[str, GroundingTrack] = {}

    for oid in all_oids:
        spec = spec_by_id.get(oid)
        is_static = bool(spec.is_static) if spec else (oid in STATIC_OBJECT_DEFAULTS)
        kfs: list[GroundingKeyframe] = []
        any_visible = False

        for idx, t_val in enumerate(times_s):
            row_idx = max(0, int(round(t_val * fps)))
            kf = _entry_to_keyframe(
                per_frame_objs.get(idx, {}).get(oid),
                row_idx=row_idx,
                t_val=t_val,
                img_w=img_w,
                img_h=img_h,
                ref_w=ref_w,
                ref_h=ref_h,
                min_box_px=min_box_px,
            )
            if kf.visible:
                any_visible = True
            kfs.append(kf)

        if not any_visible:
            continue

        # Cross-batch consensus lock for fixed-camera static containers (zero single-batch poisoning)
        if is_fixed_camera and is_static:
            vis_count = sum(1 for k in kfs if k.visible)
            consensus = _select_static_consensus_box(
                kfs,
                batch_id_of_idx,
                frame_grays_small,
                ref_w=ref_w,
                ref_h=ref_h,
                jitter_tol_px=jitter_tol_px,
                untrusted_batches=untrusted_batches,
            )
            if consensus is not None and (vis_count >= static_min_frames or tracking_cfg.enabled):
                med_norm, med_px = consensus
                def_state = STATIC_OBJECT_DEFAULTS.get(oid, "stationary")
                kfs = [
                    k.model_copy(
                        update={
                            "visible": True,
                            "bbox_norm_1000": list(med_norm),
                            "bbox_px": list(med_px),
                            "state": k.state or def_state,
                        }
                    )
                    for k in kfs
                ]

        raw_tracks[oid] = GroundingTrack(
            object_id=oid,
            name=spec.name if spec else oid,
            tag_pos=spec.tag_pos if spec else "top",
            is_static_container=is_static,
            keyframes=kfs,
        )

    del frame_grays_small

    final_times_s = [round(t, 3) for t in times_s]

    # Fast path: tracking disabled or no detected tracks -> schema_version=1 sparse output
    if not tracking_cfg.enabled or not raw_tracks or not final_times_s:
        sparse_tracks: dict[str, GroundingTrack] = {}
        for oid, tr in raw_tracks.items():
            if tr.is_static_container and is_fixed_camera:
                locked_kfs, _ = stabilize_static_keyframes(
                    tr.keyframes,
                    ref_w=ref_w,
                    ref_h=ref_h,
                    static_min_frames=static_min_frames,
                    jitter_tol_px=jitter_tol_px,
                    default_state=STATIC_OBJECT_DEFAULTS.get(oid, "stationary"),
                )
                sparse_tracks[oid] = tr.model_copy(update={"keyframes": locked_kfs})
            else:
                sparse_tracks[oid] = tr
        return GroundingResult(
            schema_version=1,
            camera_name=main_cam,
            ref_size=[ref_w, ref_h],
            keyframes_s=final_times_s,
            camera_motion_detected=not is_fixed_camera,
            object_tracks_2d=sparse_tracks,
        )

    # BAFT local frame-level 2D tracking (ZERO extra VLM calls by default)
    baft_out = run_baft_tracking(
        view=ev,
        camera=main_cam,
        anchor_times_s=final_times_s,
        raw_tracks=raw_tracks,
        ref_w=ref_w,
        ref_h=ref_h,
        static_min_frames=static_min_frames,
        jitter_tol_px=jitter_tol_px,
        static_defaults=STATIC_OBJECT_DEFAULTS,
        cfg=tracking_cfg,
        check_cancelled=_check_cancelled,
        camera_motion_override=(not is_fixed_camera),
    )

    # Optional bounded adaptive VLM re-anchoring on high bidirectional disagreement
    remaining_budget = min(
        tracking_cfg.adaptive_reanchor_max_calls,
        max(0, total_cap - len(final_times_s)),
    )
    if remaining_budget > 0 and baft_out.interval_reports:
        candidates = [
            rep
            for rep in baft_out.interval_reports
            if rep.gap_s >= 2.0
            and rep.disagree_ratio > tracking_cfg.adaptive_disagree_ratio
            and baft_out.tracks[rep.object_id].tracking_mode != "static_locked"
            and (rep.dense_i1 - rep.dense_i0) >= 4
        ]
        candidates.sort(key=lambda r: r.disagree_ratio, reverse=True)

        chosen_splits: list[tuple[int, int, int]] = []
        chosen_times: list[float] = []
        for rep in candidates:
            if len(chosen_splits) >= remaining_budget:
                break
            d_mid = rep.mid_dense_idx
            t_mid = baft_out.dense_times_s[d_mid]
            if any(abs(t_mid - et) < 0.5 for et in final_times_s):
                continue
            if any(abs(t_mid - ct) < 0.5 for ct in chosen_times):
                continue
            chosen_splits.append((rep.dense_i0, d_mid, rep.dense_i1))
            chosen_times.append(t_mid)

        if chosen_splits:
            _check_cancelled()
            re_raw_frames = ev.media.read_frames_batch(main_cam, chosen_times)
            _check_cancelled()
            re_resized = [_resize_to_height(fr, ego_h) for fr in re_raw_frames]
            del re_raw_frames

            re_batches = _partition_batches(chosen_times, re_resized, batch_size)
            re_objs: dict[int, dict[str, Any]] = {}
            re_errors: list[Exception] = []
            re_workers = min(8, max(1, len(re_batches)))
            with ThreadPoolExecutor(max_workers=re_workers) as pool:
                futs = [pool.submit(_call_batch, batch) for batch in re_batches]
                for fut in futs:
                    parsed_map, err = fut.result()
                    if err is not None:
                        re_errors.append(err)
                    else:
                        re_objs.update(parsed_map)
            del re_resized, re_batches
            if re_errors:
                raise re_errors[0]

            mid_keyframes: dict[int, dict[str, GroundingKeyframe]] = {}
            for i, (_d0, d_mid, _d1) in enumerate(chosen_splits):
                t_mid = chosen_times[i]
                row_idx = max(0, int(round(t_mid * fps)))
                obj_map = re_objs.get(i, {})
                mid_keyframes[d_mid] = {
                    oid: _entry_to_keyframe(
                        obj_map.get(oid),
                        row_idx=row_idx,
                        t_val=t_mid,
                        img_w=img_w,
                        img_h=img_h,
                        ref_w=ref_w,
                        ref_h=ref_h,
                        min_box_px=min_box_px,
                    )
                    for oid in baft_out.tracks
                }

            baft_out = retrack_split_intervals(
                view=ev,
                camera=main_cam,
                baft_out=baft_out,
                split_specs=chosen_splits,
                mid_keyframes=mid_keyframes,
                cfg=tracking_cfg,
                check_cancelled=_check_cancelled,
            )
            final_times_s = sorted({*final_times_s, *(round(t, 3) for t in chosen_times)})

    return GroundingResult(
        schema_version=2,
        camera_name=main_cam,
        ref_size=[ref_w, ref_h],
        keyframes_s=final_times_s,
        track_fps=baft_out.track_fps,
        dense_times_s=baft_out.dense_times_s,
        camera_motion_detected=baft_out.camera_motion_detected,
        object_tracks_2d=baft_out.tracks,
    )


run_s1 = run_stage1_grounding
