"""Stage 1: HD main-camera 2D grounding with static container median stabilization."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np

from ..config import ConfigManager
from ..io.base import EpisodeView
from ..models.annotation import GroundingKeyframe, GroundingResult, GroundingTrack
from ..models.taxonomy import TaskSpec
from .client import BudgetExceeded, BudgetTracker, VlmClient, VlmError
from .prompt_builder import PromptBuilder

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
        # Pixel coordinates
        denom_w = float(max(1, img_w if max(vals[0], vals[2]) <= img_w + 2 else ref_w))
        denom_h = float(max(1, img_h if max(vals[1], vals[3]) <= img_h + 2 else ref_h))
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

    if (px2 - px1) < min_box_px or (py2 - py1) < min_box_px or x2 <= x1 or y2 <= y1:
        return False, [0, 0, 0, 0], [0, 0, 0, 0]

    return True, [x1, y1, x2, y2], [px1, py1, px2, py2]


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
) -> GroundingResult:
    """Execute Stage 1 HD 2D grounding with static container median stabilization."""
    ev = view or episode_view
    cl = client or vlm_client
    cm = cfg if isinstance(cfg, ConfigManager) else (config if isinstance(config, ConfigManager) else None)
    t_spec = task or task_spec or (cm.get_task(ev.meta.task_id) if cm and ev else TaskSpec(task_id="_default"))
    assert ev is not None and cl is not None

    s1_cfg = _resolve_s1_cfg(cfg or config)
    ref_w = int(s1_cfg.get("ref_width", 1280))
    ref_h = int(s1_cfg.get("ref_height", 720))

    if not bool(s1_cfg.get("enabled", True)):
        return GroundingResult(ref_size=[ref_w, ref_h], keyframes_s=[], object_tracks_2d={})

    main_cam = ev.camera_roles.get("main") or (ev.media.cameras[0].name if ev.media.cameras else "")
    if not main_cam:
        return GroundingResult(ref_size=[ref_w, ref_h], keyframes_s=[], object_tracks_2d={})

    times_s = _plan_keyframe_times(
        ev.duration_s,
        float(s1_cfg.get("interval_s", 2.5)),
        int(s1_cfg.get("max_keyframes", 36)),
        int(s1_cfg.get("hard_max_keyframes", 72)),
    )
    ego_h = int(s1_cfg.get("ego_height", 540))
    jpeg_q = int(s1_cfg.get("jpeg_quality", 80))
    min_box_px = int(s1_cfg.get("min_box_px", 12))
    thinking = bool(s1_cfg.get("thinking", False))
    thinking_budget = int(s1_cfg.get("thinking_budget", 768))
    max_tokens = int(s1_cfg.get("max_tokens", 800))
    static_min_frames = int(s1_cfg.get("static_container_min_frames", 5))
    jitter_tol_px = float(s1_cfg.get("static_jitter_tol_px_1280", 85.0)) * (ref_w / 1280.0)

    pb = prompt_builder or (PromptBuilder(cm) if cm is not None else None)
    if pb is not None:
        sys_p, usr_p = pb.build_s1(task=t_spec, camera_roles=ev.camera_roles)
    else:
        sys_p = (
            "You are an embodied manipulation 2D visual grounding expert. "
            'Output strict JSON only: {"objects": {obj_id: {"visible": bool, "bbox": [x1,y1,x2,y2], "state": str, "status_kind": "neutral|warn|error|good"}}}.'
        )
        obj_lines = "\n".join(
            f"- {g.id}: {g.name} ({g.desc})" for g in t_spec.grounding_objects
        ) or "- target_object: Target Object (primary manipulated object)"
        usr_p = f"Task: {t_spec.task_id}\nTarget objects to ground:\n{obj_lines}"

    raw_frames = ev.media.read_frames_batch(main_cam, times_s)
    resized_frames = [_resize_to_height(fr, ego_h) for fr in raw_frames]
    img_h, img_w = resized_frames[0].shape[:2] if resized_frames else (ref_h, ref_w)

    def _call_one(idx: int) -> tuple[int, dict[str, Any] | None, Exception | None]:
        t_val = times_s[idx]
        fr = resized_frames[idx]
        try:
            resp = cl.chat_json(
                stage="s1_grounding",
                system_prompt=sys_p,
                user_prompt=usr_p,
                images=[fr],
                timestamps_s=[t_val],
                thinking=thinking,
                thinking_budget=thinking_budget,
                max_tokens=max_tokens,
                jpeg_quality=jpeg_q,
                budget=budget,
            )
            return idx, resp, None
        except BudgetExceeded:
            raise
        except Exception as exc:
            return idx, None, exc

    per_frame_objs: dict[int, dict[str, Any]] = {}
    errors: list[Exception] = []
    workers = min(8, max(1, len(times_s)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_call_one, i) for i in range(len(times_s))]
        for fut in futs:
            idx, resp, err = fut.result()
            if err is not None:
                errors.append(err)
            elif resp is not None:
                objs = resp.get("objects")
                per_frame_objs[idx] = objs if isinstance(objs, dict) else {}

    if errors:
        raise errors[0]

    # Collect all seen object IDs + configured object metadata
    spec_by_id = {g.id: g for g in t_spec.grounding_objects}
    all_oids: list[str] = [g.id for g in t_spec.grounding_objects]
    for idx in sorted(per_frame_objs):
        for oid in per_frame_objs[idx]:
            if isinstance(oid, str) and oid and oid not in all_oids:
                all_oids.append(oid)

    tracks: dict[str, GroundingTrack] = {}
    fps = float(ev.fps) if ev.fps > 0 else 30.0
    valid_status = {"neutral", "warn", "error", "good"}

    for oid in all_oids:
        spec = spec_by_id.get(oid)
        is_static = bool(spec.is_static) if spec else (oid in STATIC_OBJECT_DEFAULTS)
        kfs: list[GroundingKeyframe] = []
        any_visible = False

        for idx, t_val in enumerate(times_s):
            row_idx = max(0, int(round(t_val * fps)))
            entry = per_frame_objs.get(idx, {}).get(oid)
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
                if ok:
                    any_visible = True
                kfs.append(
                    GroundingKeyframe(
                        row=row_idx,
                        time_s=round(t_val, 3),
                        visible=ok,
                        bbox_norm_1000=b_norm if ok else [0, 0, 0, 0],
                        bbox_px=b_px if ok else [0, 0, 0, 0],
                        state=state if ok else "",
                        status_kind=sk if ok else "neutral",  # type: ignore[arg-type]
                    )
                )
            else:
                kfs.append(
                    GroundingKeyframe(
                        row=row_idx,
                        time_s=round(t_val, 3),
                        visible=False,
                        bbox_norm_1000=[0, 0, 0, 0],
                        bbox_px=[0, 0, 0, 0],
                        state="",
                        status_kind="neutral",
                    )
                )

        if not any_visible and spec is None:
            continue
        if not any_visible and spec is not None and not per_frame_objs:
            continue
        if not any_visible:
            # Skip unpopulated tracks when VLM explicitly returned empty detections
            continue

        # Static container median stabilization (inlier-refined via static_jitter_tol_px_1280)
        if is_static:
            vis_kfs = [k for k in kfs if k.visible]
            if len(vis_kfs) >= static_min_frames:
                arr_norm = np.asarray([k.bbox_norm_1000 for k in vis_kfs], dtype=np.float64)
                arr_px = np.asarray([k.bbox_px for k in vis_kfs], dtype=np.float64)
                init_med_px = np.median(arr_px, axis=0)
                inlier_mask = (
                    np.max(np.abs(arr_px - init_med_px[None, :]), axis=-1) <= jitter_tol_px
                )
                med_src = arr_norm[inlier_mask] if np.any(inlier_mask) else arr_norm
                med = np.median(med_src, axis=0)
                mx1 = int(round(max(0.0, min(999.0, med[0]))))
                my1 = int(round(max(0.0, min(999.0, med[1]))))
                mx2 = int(round(max(mx1 + 1.0, min(1000.0, med[2]))))
                my2 = int(round(max(my1 + 1.0, min(1000.0, med[3]))))
                med_norm = [mx1, my1, mx2, my2]
                med_px = [
                    int(round(mx1 * ref_w / 1000.0)),
                    int(round(my1 * ref_h / 1000.0)),
                    int(round(mx2 * ref_w / 1000.0)),
                    int(round(my2 * ref_h / 1000.0)),
                ]
                def_state = STATIC_OBJECT_DEFAULTS.get(oid, "stationary")
                # Anchor all keyframes of the static container to the inlier-refined median box
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

        tracks[oid] = GroundingTrack(
            object_id=oid,
            name=spec.name if spec else oid,
            tag_pos=spec.tag_pos if spec else "top",
            is_static_container=is_static,
            keyframes=kfs,
        )

    return GroundingResult(
        schema_version=1,
        ref_size=[ref_w, ref_h],
        keyframes_s=[round(t, 3) for t in times_s],
        object_tracks_2d=tracks,
    )


run_s1 = run_stage1_grounding
