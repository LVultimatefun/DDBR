"""Stage 1: HD main-camera 2D grounding + BAFT local frame-level 2D tracking."""

from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from ..config import ConfigManager
from ..io.base import EpisodeView
from ..models.annotation import GroundingKeyframe, GroundingResult, GroundingTrack
from ..models.taxonomy import TaskSpec
from .client import (
    BudgetExceeded,
    BudgetTracker,
    VlmCancelledError,
    VlmClient,
    check_client_cancelled,
)
from .prompt_builder import PromptBuilder
from .tracker import (
    TrackingConfig,
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


def _parse_batch_response(
    resp: Any,
    batch: Sequence[tuple[int, float, np.ndarray]],
) -> dict[int, dict[str, Any]]:
    """Parse batched multi-frame or legacy single-frame S1 JSON into {global_idx: objects}."""
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

        # Pass 1: explicit frame_index / index
        for e_i, entry in enumerate(frames_list):
            if not isinstance(entry, dict):
                continue
            raw_fi = entry.get("frame_index")
            if raw_fi is None:
                raw_fi = entry.get("index")
            fi = _parse_optional_int(raw_fi)
            if fi is None:
                continue
            target_loc: int | None = None
            if 0 <= fi < b_len and fi not in claimed:
                target_loc = fi
            elif fi in idx_by_global and idx_by_global[fi] not in claimed:
                target_loc = idx_by_global[fi]
            if target_loc is not None:
                claimed.add(target_loc)
                matched_entries.add(e_i)
                objs = entry.get("objects")
                out[batch[target_loc][0]] = objs if isinstance(objs, dict) else {}

        # Pass 2: closest unclaimed time_s / timestamp_s within tolerance
        min_gap = min(
            (abs(batch[k + 1][1] - batch[k][1]) for k in range(b_len - 1)),
            default=0.0,
        )
        tol = max(0.25, 0.49 * min_gap) if (b_len > 1 and min_gap > 0.0) else 0.5
        for e_i, entry in enumerate(frames_list):
            if e_i in matched_entries or not isinstance(entry, dict):
                continue
            raw_fi = entry.get("frame_index")
            if raw_fi is None:
                raw_fi = entry.get("index")
            if _parse_optional_int(raw_fi) is not None:
                continue
            raw_t = entry.get("time_s")
            if raw_t is None:
                raw_t = entry.get("timestamp_s")
            if raw_t is None:
                raw_t = entry.get("t")
            if raw_t is None:
                continue
            try:
                t_val = float(raw_t)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(t_val):
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
        for e_i, entry in enumerate(frames_list):
            if e_i in matched_entries or not isinstance(entry, dict):
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
    batch_size = max(1, int(s1_cfg.get("batch_size", 8)))
    static_min_frames = int(s1_cfg.get("static_container_min_frames", 5))
    jitter_tol_px = float(s1_cfg.get("static_jitter_tol_px_1280", 85.0)) * (ref_w / 1280.0)

    pb = prompt_builder or (PromptBuilder(cm) if cm is not None else None)
    if pb is not None:
        sys_p, usr_p = pb.build_s1(task=t_spec, camera_roles=ev.camera_roles)
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
            f"- {g.id}: {g.name} ({g.desc})" for g in t_spec.grounding_objects
        ) or "- target_object: Target Object (primary manipulated object)"
        usr_p = f"Task: {t_spec.task_id}\nTarget objects to ground:\n{obj_lines}"

    _check_cancelled()
    raw_frames = ev.media.read_frames_batch(main_cam, times_s)
    _check_cancelled()
    resized_frames = [_resize_to_height(fr, ego_h) for fr in raw_frames]
    img_h, img_w = resized_frames[0].shape[:2] if resized_frames else (ref_h, ref_w)
    del raw_frames

    def _call_batch(
        batch: Sequence[tuple[int, float, np.ndarray]],
    ) -> tuple[dict[int, dict[str, Any]], Exception | None]:
        _check_cancelled()
        batch_times = [t_val for _, t_val, _ in batch]
        batch_frames = [fr for _, _, fr in batch]
        eff_max_tokens = (
            max(max_tokens, 280 * len(batch_frames)) if batch_size > 1 else max_tokens
        )
        chat_kw: dict[str, Any] = {
            "stage": "s1_grounding",
            "system_prompt": sys_p,
            "user_prompt": usr_p,
            "images": batch_frames,
            "timestamps_s": batch_times,
            "thinking": thinking,
            "thinking_budget": thinking_budget,
            "max_tokens": eff_max_tokens,
            "jpeg_quality": jpeg_q,
            "budget": budget,
        }
        if batch_size > 1:
            chat_kw["labels"] = [f"frame_{i}" for i in range(len(batch_frames))]
        try:
            resp = cl.chat_json(**chat_kw)
            _check_cancelled()
            return _parse_batch_response(resp, batch), None
        except (BudgetExceeded, VlmCancelledError):
            raise
        except Exception as exc:
            _check_cancelled()
            return {}, exc

    batches = _partition_batches(times_s, resized_frames, batch_size)
    per_frame_objs: dict[int, dict[str, Any]] = {}
    errors: list[Exception] = []
    workers = min(8, max(1, len(batches)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_call_batch, batch) for batch in batches]
        for fut in futs:
            parsed_map, err = fut.result()
            if err is not None:
                errors.append(err)
            else:
                per_frame_objs.update(parsed_map)

    del resized_frames, batches
    if errors:
        raise errors[0]

    _check_cancelled()

    # Collect all seen object IDs + configured object metadata
    spec_by_id = {g.id: g for g in t_spec.grounding_objects}
    all_oids: list[str] = [g.id for g in t_spec.grounding_objects]
    for idx in sorted(per_frame_objs):
        for oid in per_frame_objs[idx]:
            if isinstance(oid, str) and oid and oid not in all_oids:
                all_oids.append(oid)

    raw_tracks: dict[str, GroundingTrack] = {}
    fps = float(ev.fps) if ev.fps > 0 else 30.0

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

        raw_tracks[oid] = GroundingTrack(
            object_id=oid,
            name=spec.name if spec else oid,
            tag_pos=spec.tag_pos if spec else "top",
            is_static_container=is_static,
            keyframes=kfs,
        )

    final_times_s = [round(t, 3) for t in times_s]

    # Fast path: tracking disabled or no detected tracks -> schema_version=1 sparse output
    if not tracking_cfg.enabled or not raw_tracks or not final_times_s:
        sparse_tracks: dict[str, GroundingTrack] = {}
        for oid, tr in raw_tracks.items():
            if tr.is_static_container:
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
