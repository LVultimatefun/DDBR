"""Comprehensive unit & integration tests for Stage 1 BAFT frame-level 2D tracker & viz_mp4 fixes."""

from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from robo_vlm_qc.config import ConfigManager
from robo_vlm_qc.export.viz_mp4 import (
    _HudCompositor,
    _interp_track_box,
    _project_norm_box_for_crop,
)
from robo_vlm_qc.io import auto_detect, get_adapter
from robo_vlm_qc.io.base import EpisodeView
from robo_vlm_qc.models.annotation import (
    ArmTimelines,
    GroundingKeyframe,
    GroundingResult,
    GroundingTrack,
    SuccessVerdict,
)
from robo_vlm_qc.models.episode import CameraSpec, EpisodeMeta, KinematicsSpec
from robo_vlm_qc.models.review import EpisodeStatus, ResolvedEpisode
from robo_vlm_qc.models.taxonomy import GroundingObjectSpec, TaskSpec
from robo_vlm_qc.store import Repository
from robo_vlm_qc.vlm.client import BudgetTracker, VlmCancelledError
from robo_vlm_qc.vlm.dag import VlmDag
from robo_vlm_qc.vlm.stage1_grounding import run_stage1_grounding
from robo_vlm_qc.vlm.tracker import (
    FLAG_INVISIBLE,
    FLAG_STATIC_LOCKED,
    FLAG_TRACKED,
    FLAG_VLM_ANCHOR,
    TrackingConfig,
    box_center_dist,
    box_iou,
    clamp_anchor_scale_outliers,
    run_baft_tracking,
    track_interval_bidirectional,
)


class _DeterministicVlmClient:
    """In-memory deterministic VlmClient double supporting cancellation checks."""

    def __init__(
        self,
        handler: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.handler = handler
        self.calls: list[dict[str, Any]] = []
        self.budget = BudgetTracker()
        self._cancel_event = cancel_event or threading.Event()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def model_for_stage(self, stage: str, override: str | None = None) -> str:
        return override or "fake-gpt-4o"

    def chat_json(self, **kwargs: Any) -> dict[str, Any]:
        if self.is_cancelled():
            raise VlmCancelledError("FakeClient cancelled")
        self.calls.append(kwargs)
        if kwargs.get("budget") is not None:
            kwargs["budget"].record(
                stage=kwargs.get("stage", "s1_grounding"),
                model="fake-gpt-4o",
                prompt_tokens=40,
                completion_tokens=20,
                total_tokens=60,
            )
        return self.handler(kwargs)


def _make_textured_patch(seed: int, h: int, w: int, base_bgr: tuple[int, int, int]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.integers(-45, 46, size=(h, w, 3), dtype=np.int16)
    patch = np.clip(np.asarray(base_bgr, dtype=np.int16)[None, None, :] + noise, 10, 245).astype(np.uint8)
    # High-contrast geometric internal markings for rich optical-flow & NCC features
    cv2.rectangle(patch, (3, 3), (w - 4, h - 4), (250, 250, 250), 2)
    cv2.line(patch, (4, 4), (w - 5, h - 5), (20, 20, 20), 2)
    cv2.circle(patch, (w // 2, h // 2), max(4, min(w, h) // 4), (255, 220, 40), -1)
    return patch


def _make_textured_bg(seed: int, h: int, w: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    low = rng.integers(35, 115, size=(max(8, h // 4), max(8, w // 4), 3), dtype=np.uint8)
    bg = cv2.resize(low, (w, h), interpolation=cv2.INTER_LINEAR)
    # Subtle static tabletop grid lines
    for x in range(0, w, 40):
        cv2.line(bg, (x, 0), (x, h - 1), (75, 80, 88), 1)
    for y in range(0, h, 40):
        cv2.line(bg, (0, y), (w - 1, y), (75, 80, 88), 1)
    return bg


def _px_to_norm_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> list[int]:
    return [
        int(round(x1 * 1000.0 / w)),
        int(round(y1 * 1000.0 / h)),
        int(round(x2 * 1000.0 / w)),
        int(round(y2 * 1000.0 / h)),
    ]


def _write_mp4(path: Path, frames_bgr: list[np.ndarray], fps: float = 30.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    assert writer.isOpened()
    try:
        for fr in frames_bgr:
            writer.write(fr)
    finally:
        writer.release()


def _coffee_task() -> TaskSpec:
    return TaskSpec(
        task_id="coffee_barista",
        grounding_objects=[
            GroundingObjectSpec(id="cup", name="Cup", desc="Moving cup", is_static=False),
            GroundingObjectSpec(
                id="coffee_machine",
                name="Coffee Machine",
                desc="Countertop coffee machine",
                is_static=True,
            ),
        ],
    )


def test_baft_tracks_curved_moving_object_with_minimal_vlm_anchors_and_locks_static(
    tmp_path: Path,
) -> None:
    w, h, fps, dur_s = 640, 360, 30.0, 3.0
    n_frames = int(round(dur_s * fps))
    bg = _make_textured_bg(101, h, w)
    cup_patch = _make_textured_patch(202, 72, 60, (60, 140, 230))
    machine_patch = _make_textured_patch(303, 120, 110, (160, 110, 70))

    # Static machine at [460, 80, 570, 200]
    m_x1, m_y1, m_x2, m_y2 = 460, 80, 570, 200
    gt_cup_norm: dict[int, list[int]] = {}
    frames: list[np.ndarray] = []

    def _cup_pos_at_t(t_s: float) -> tuple[int, int, int, int]:
        u = max(0.0, min(1.0, t_s / dur_s))
        # Curved lift arc: linear interpolation between endpoints would miss the apex by ~115px!
        cx = int(round(100 + 280 * u))
        cy = int(round(260 - 145 * math.sin(math.pi * u)))
        x1 = cx - 30
        y1 = cy - 36
        return x1, y1, x1 + 60, y1 + 72

    for i in range(n_frames):
        fr = bg.copy()
        fr[m_y1:m_y2, m_x1:m_x2] = machine_patch
        cx1, cy1, cx2, cy2 = _cup_pos_at_t(i / fps)
        fr[cy1:cy2, cx1:cx2] = cup_patch
        gt_cup_norm[i] = _px_to_norm_box(cx1, cy1, cx2, cy2, w, h)
        frames.append(fr)

    ds_dir = tmp_path / "curved_ds" / "ep_0001"
    _write_mp4(ds_dir / "main.mp4", frames, fps=fps)
    handle = auto_detect(tmp_path / "curved_ds", adapter_name="video_dir")[0]

    # Only 3 VLM anchors: t=0.0, t=1.5, t=3.0 (plus tiny jitter on static machine)
    m_jitter = [(0, 0), (4, -3), (-3, 4)]
    call_idx = {"n": 0}

    def _vlm_handler(kw: dict[str, Any]) -> dict[str, Any]:
        t_val = float(kw["timestamps_s"][0])
        cx1, cy1, cx2, cy2 = _cup_pos_at_t(t_val)
        jx, jy = m_jitter[call_idx["n"] % len(m_jitter)]
        call_idx["n"] += 1
        return {
            "objects": {
                "cup": {
                    "visible": True,
                    "bbox": _px_to_norm_box(cx1, cy1, cx2, cy2, w, h),
                    "state": "lifting" if 0.5 < t_val < 2.5 else "table",
                    "status_kind": "neutral",
                },
                "coffee_machine": {
                    "visible": True,
                    "bbox": _px_to_norm_box(m_x1 + jx, m_y1 + jy, m_x2 + jx, m_y2 + jy, w, h),
                    "state": "ready",
                    "status_kind": "good",
                },
            }
        }

    cfg = ConfigManager()
    cfg.settings["vlm"]["s1_grounding"]["batch_size"] = 1
    cfg.settings["vlm"]["s1_grounding"]["interval_s"] = 1.5
    cfg.settings["vlm"]["s1_grounding"]["max_keyframes"] = 3
    cfg.settings["vlm"]["s1_grounding"]["static_container_min_frames"] = 3

    client = _DeterministicVlmClient(_vlm_handler)
    with get_adapter("video_dir").open(handle) as view:
        res = run_stage1_grounding(
            view=view,
            task=_coffee_task(),
            client=client,  # type: ignore[arg-type]
            cfg=cfg,
        )

    # Strictly minimal VLM calls: ONLY 3 anchors, ZERO extra VLM calls
    assert len(client.calls) == 3
    assert res.schema_version == 2
    assert res.camera_name == "main"
    assert res.ref_size == [w, h]
    assert res.camera_motion_detected is False
    assert res.track_fps == pytest.approx(30.0)
    assert len(res.dense_times_s) >= 85

    cup = res.object_tracks_2d["cup"]
    assert cup.tracking_mode == "bidir_flow_ncc"
    assert len(cup.dense_boxes_norm_1000) == len(res.dense_times_s)
    assert cup.dense_flags.count(FLAG_VLM_ANCHOR) == 3

    ious: list[float] = []
    center_errs: list[float] = []
    for idx, t_s in enumerate(res.dense_times_s):
        if cup.dense_flags[idx] != FLAG_TRACKED:
            continue
        gt_b = _px_to_norm_box(*_cup_pos_at_t(t_s), w, h)
        pred_b = cup.dense_boxes_norm_1000[idx]
        ious.append(box_iou(pred_b, gt_b))
        center_errs.append(box_center_dist(pred_b, gt_b))

    assert len(ious) >= 80
    assert float(np.mean(ious)) > 0.80, f"Expected mean IoU > 0.80, got {np.mean(ious):.3f}"
    assert float(np.mean(center_errs)) < 20.0, f"Expected mean center err < 20, got {np.mean(center_errs):.2f}"

    # Static container on fixed camera stays rock-solid locked
    cm = res.object_tracks_2d["coffee_machine"]
    assert cm.tracking_mode == "static_locked"
    assert set(cm.dense_flags) == {FLAG_STATIC_LOCKED}
    assert set(cm.dense_conf_pct) == {100}
    assert len({tuple(b) for b in cm.dense_boxes_norm_1000}) == 1
    assert len({tuple(k.bbox_norm_1000) for k in cm.keyframes if k.visible}) == 1


def test_baft_moving_camera_tracks_static_container_instead_of_freezing(
    tmp_path: Path,
) -> None:
    w, h, fps, dur_s = 640, 360, 30.0, 2.5
    n_frames = int(round(dur_s * fps))
    # Large textured canvas that pans by 140px across 2.5s (simulating humanoid head pan!)
    big_w = w + 180
    big_bg = _make_textured_bg(404, h, big_w)
    machine_patch = _make_textured_patch(505, 130, 120, (170, 90, 60))
    # Place static machine in world coords at x=[260..380], y=[90..220]
    world_mx1, world_my1, world_mx2, world_my2 = 260, 90, 380, 220
    big_bg[world_my1:world_my2, world_mx1:world_mx2] = machine_patch

    def _cam_offset_x(t_s: float) -> int:
        u = max(0.0, min(1.0, t_s / dur_s))
        return int(round(140 * u))

    def _machine_box_at_t(t_s: float) -> tuple[int, int, int, int]:
        ox = _cam_offset_x(t_s)
        return world_mx1 - ox, world_my1, world_mx2 - ox, world_my2

    frames: list[np.ndarray] = []
    for i in range(n_frames):
        ox = _cam_offset_x(i / fps)
        frames.append(np.ascontiguousarray(big_bg[:, ox : ox + w]))

    ds_dir = tmp_path / "pan_ds" / "ep_pan"
    _write_mp4(ds_dir / "main.mp4", frames, fps=fps)
    handle = auto_detect(tmp_path / "pan_ds", adapter_name="video_dir")[0]

    def _vlm_handler(kw: dict[str, Any]) -> dict[str, Any]:
        t_val = float(kw["timestamps_s"][0])
        mx1, my1, mx2, my2 = _machine_box_at_t(t_val)
        return {
            "objects": {
                "coffee_machine": {
                    "visible": True,
                    "bbox": _px_to_norm_box(mx1, my1, mx2, my2, w, h),
                    "state": "visible",
                    "status_kind": "neutral",
                }
            }
        }

    cfg = ConfigManager()
    cfg.settings["vlm"]["s1_grounding"]["batch_size"] = 1
    cfg.settings["vlm"]["s1_grounding"]["interval_s"] = 1.25
    cfg.settings["vlm"]["s1_grounding"]["max_keyframes"] = 3
    cfg.settings["vlm"]["s1_grounding"]["static_container_min_frames"] = 2

    client = _DeterministicVlmClient(_vlm_handler)
    with get_adapter("video_dir").open(handle) as view:
        res = run_stage1_grounding(
            view=view,
            task=_coffee_task(),
            client=client,  # type: ignore[arg-type]
            cfg=cfg,
        )

    assert len(client.calls) == 3
    assert res.camera_motion_detected is True
    cm = res.object_tracks_2d["coffee_machine"]
    assert cm.is_static_container is True
    # Must NOT freeze to 1 median box when head camera is panning!
    assert cm.tracking_mode == "bidir_flow_ncc"
    assert len({tuple(k.bbox_norm_1000) for k in cm.keyframes if k.visible}) >= 3

    ious = [
        box_iou(cm.dense_boxes_norm_1000[i], _px_to_norm_box(*_machine_box_at_t(t), w, h))
        for i, t in enumerate(res.dense_times_s)
        if cm.dense_flags[i] == FLAG_TRACKED
    ]
    assert ious and float(np.mean(ious)) > 0.80, f"Moving-camera static track IoU too low: {np.mean(ious):.3f}"


def test_baft_occlusion_gap_emits_flag_zero_and_empty_box(tmp_path: Path) -> None:
    w, h, fps, dur_s = 640, 360, 30.0, 2.5
    n_frames = int(round(dur_s * fps))
    bg = _make_textured_bg(606, h, w)
    cup_patch = _make_textured_patch(707, 70, 60, (50, 180, 240))
    # High-contrast occluder (dark hand/arm moving across cup)
    occ_patch = np.full((180, 180, 3), (18, 20, 22), dtype=np.uint8)

    def _cup_pos(t_s: float) -> tuple[int, int, int, int]:
        u = max(0.0, min(1.0, t_s / dur_s))
        x1 = int(round(180 + 140 * u))
        y1 = 140
        return x1, y1, x1 + 60, y1 + 70

    frames: list[np.ndarray] = []
    for i in range(n_frames):
        t_s = i / fps
        fr = bg.copy()
        cx1, cy1, cx2, cy2 = _cup_pos(t_s)
        fr[cy1:cy2, cx1:cx2] = cup_patch
        # Completely occlude cup during [0.80s, 1.70s]
        if 0.80 <= t_s <= 1.70:
            ox1 = max(0, cx1 - 60)
            oy1 = max(0, cy1 - 55)
            ox2 = min(w, ox1 + 180)
            oy2 = min(h, oy1 + 180)
            fr[oy1:oy2, ox1:ox2] = occ_patch[: oy2 - oy1, : ox2 - ox1]
        frames.append(fr)

    ds_dir = tmp_path / "occ_ds" / "ep_occ"
    _write_mp4(ds_dir / "main.mp4", frames, fps=fps)
    handle = auto_detect(tmp_path / "occ_ds", adapter_name="video_dir")[0]

    # Only 2 anchors at t=0.0 and t=2.5 (both visible)
    def _vlm_handler(kw: dict[str, Any]) -> dict[str, Any]:
        t_val = float(kw["timestamps_s"][0])
        return {
            "objects": {
                "cup": {
                    "visible": True,
                    "bbox": _px_to_norm_box(*_cup_pos(t_val), w, h),
                    "state": "moving",
                    "status_kind": "neutral",
                }
            }
        }

    cfg = ConfigManager()
    cfg.settings["vlm"]["s1_grounding"]["batch_size"] = 1
    cfg.settings["vlm"]["s1_grounding"]["interval_s"] = 2.5
    cfg.settings["vlm"]["s1_grounding"]["max_keyframes"] = 2

    client = _DeterministicVlmClient(_vlm_handler)
    with get_adapter("video_dir").open(handle) as view:
        res = run_stage1_grounding(
            view=view,
            task=_coffee_task(),
            client=client,  # type: ignore[arg-type]
            cfg=cfg,
        )

    assert len(client.calls) == 2
    cup = res.object_tracks_2d["cup"]
    mid_occ_indices = [
        i for i, t in enumerate(res.dense_times_s) if 0.95 <= t <= 1.55
    ]
    assert mid_occ_indices
    occ_flags = [cup.dense_flags[i] for i in mid_occ_indices]
    assert FLAG_INVISIBLE in occ_flags, f"Expected occlusion flag=0 inside occluded span, got {occ_flags}"
    for i, fl in enumerate(cup.dense_flags):
        if fl == FLAG_INVISIBLE:
            assert cup.dense_boxes_norm_1000[i] == [0, 0, 0, 0]
            assert cup.dense_conf_pct[i] == 0
    # Endpoints remain anchored & visible
    assert cup.dense_flags[0] == FLAG_VLM_ANCHOR
    assert cup.dense_flags[-1] == FLAG_VLM_ANCHOR


def test_baft_temporal_outlier_anchor_filter_bridges_instance_switch(
    tmp_path: Path,
) -> None:
    w, h, fps, dur_s = 640, 360, 30.0, 3.0
    n_frames = int(round(dur_s * fps))
    bg = _make_textured_bg(808, h, w)
    cup_a = _make_textured_patch(909, 70, 60, (50, 160, 235))
    cup_distractor = _make_textured_patch(909, 70, 60, (50, 160, 235))

    def _cup_a_pos(t_s: float) -> tuple[int, int, int, int]:
        u = max(0.0, min(1.0, t_s / dur_s))
        x1 = int(round(110 + 40 * u))
        y1 = 145
        return x1, y1, x1 + 60, y1 + 70

    frames: list[np.ndarray] = []
    for i in range(n_frames):
        fr = bg.copy()
        cx1, cy1, cx2, cy2 = _cup_a_pos(i / fps)
        fr[cy1:cy2, cx1:cx2] = cup_a
        # Identical distractor cup sitting on far right
        fr[145:215, 490:550] = cup_distractor
        frames.append(fr)

    ds_dir = tmp_path / "outlier_ds" / "ep_out"
    _write_mp4(ds_dir / "main.mp4", frames, fps=fps)
    handle = auto_detect(tmp_path / "outlier_ds", adapter_name="video_dir")[0]

    # Middle anchor at t=1.5s erroneously jumps to the far-right distractor cup!
    def _vlm_handler(kw: dict[str, Any]) -> dict[str, Any]:
        t_val = float(kw["timestamps_s"][0])
        if 1.0 < t_val < 2.0:
            box = _px_to_norm_box(490, 145, 550, 215, w, h)
        else:
            box = _px_to_norm_box(*_cup_a_pos(t_val), w, h)
        return {
            "objects": {
                "cup": {"visible": True, "bbox": box, "state": "on_table", "status_kind": "neutral"}
            }
        }

    cfg = ConfigManager()
    cfg.settings["vlm"]["s1_grounding"]["batch_size"] = 1
    cfg.settings["vlm"]["s1_grounding"]["interval_s"] = 1.5
    cfg.settings["vlm"]["s1_grounding"]["max_keyframes"] = 3

    client = _DeterministicVlmClient(_vlm_handler)
    with get_adapter("video_dir").open(handle) as view:
        res = run_stage1_grounding(
            view=view,
            task=_coffee_task(),
            client=client,  # type: ignore[arg-type]
            cfg=cfg,
        )

    cup = res.object_tracks_2d["cup"]
    mid_idx = min(range(len(res.dense_times_s)), key=lambda i: abs(res.dense_times_s[i] - 1.5))
    mid_box = cup.dense_boxes_norm_1000[mid_idx]
    mid_cx = 0.5 * (mid_box[0] + mid_box[2])
    # Must stay bridged on the left cup (~240 norm), NOT teleported to distractor (~812 norm)!
    assert mid_cx < 400.0, f"Instance-switch outlier was not filtered; mid_box={mid_box}"
    assert cup.dense_flags[mid_idx] == FLAG_TRACKED


def test_baft_cooperative_cancel_aborts_during_tracking_and_dag_rollback(
    tmp_path: Path,
) -> None:
    ds_dir = tmp_path / "cancel_ds" / "ep_c"
    frames = [_make_textured_bg(12 + i, 180, 320) for i in range(90)]
    _write_mp4(ds_dir / "main.mp4", frames, fps=30.0)
    handle = auto_detect(tmp_path / "cancel_ds", adapter_name="video_dir")[0]

    cancel_ev = threading.Event()

    def _vlm_handler(kw: dict[str, Any]) -> dict[str, Any]:
        return {
            "objects": {
                "cup": {
                    "visible": True,
                    "bbox": [200, 200, 400, 450],
                    "state": "ok",
                    "status_kind": "neutral",
                }
            }
        }

    cfg = ConfigManager()
    cfg.settings["vlm"]["s1_grounding"]["batch_size"] = 1
    cfg.settings["vlm"]["s1_grounding"]["interval_s"] = 1.5
    cfg.settings["vlm"]["s1_grounding"]["max_keyframes"] = 3

    # 1. Direct stage1 cancellation fired during BAFT chunked decode (after VLM anchors finish)
    with get_adapter("video_dir").open(handle) as view:
        orig_batch = view.media.read_frames_batch
        batch_calls = {"n": 0}

        def _cancelling_batch(cam: str, times: list[float]) -> list[np.ndarray]:
            batch_calls["n"] += 1
            if batch_calls["n"] >= 2:
                # Fires right when BAFT tracker starts decoding dense frames
                cancel_ev.set()
            return orig_batch(cam, times)

        view.media.read_frames_batch = _cancelling_batch  # type: ignore[method-assign]
        client = _DeterministicVlmClient(_vlm_handler, cancel_event=cancel_ev)
        with pytest.raises(VlmCancelledError):
            run_stage1_grounding(
                view=view,
                task=_coffee_task(),
                client=client,  # type: ignore[arg-type]
                cfg=cfg,
                cancel_event=cancel_ev,
            )

    # 2. VlmDag rollback when cancel_event triggers inside BAFT tracking
    ws = tmp_path / "ws_cancel"
    cfg_ws = ConfigManager(workspace_dir=ws)
    cfg_ws.settings["vlm"]["s1_grounding"]["interval_s"] = 1.5
    cfg_ws.settings["vlm"]["s1_grounding"]["max_keyframes"] = 3
    repo = Repository(ws)
    try:
        cancel_ev2 = threading.Event()
        with get_adapter("video_dir").open(handle) as view2:
            repo.ingest_episode(view2.meta)
            orig_batch2 = view2.media.read_frames_batch
            b2 = {"n": 0}

            def _cancelling_batch2(cam: str, times: list[float]) -> list[np.ndarray]:
                b2["n"] += 1
                if b2["n"] >= 2:
                    cancel_ev2.set()
                return orig_batch2(cam, times)

            view2.media.read_frames_batch = _cancelling_batch2  # type: ignore[method-assign]
            dag = VlmDag(
                workspace=ws,
                config=cfg_ws,
                client=_DeterministicVlmClient(_vlm_handler),  # type: ignore[arg-type]
                repo=repo,
            )
            with pytest.raises(VlmCancelledError):
                dag.run_episode(view2, cancel_event=cancel_ev2)

            row = repo.get_episode_row(view2.meta.episode_id)
            assert row["status"] == EpisodeStatus.INGESTED.value
            assert row["active_run_id"] is None
    finally:
        repo.close()


def test_baft_adaptive_reanchor_respects_cap_and_default_zero_extra_calls(
    tmp_path: Path,
) -> None:
    w, h, fps, dur_s = 320, 180, 15.0, 5.0
    n_frames = int(round(dur_s * fps))
    bg = _make_textured_bg(314, h, w)
    cup = _make_textured_patch(271, 44, 38, (60, 150, 230))

    # Abrupt midpoint teleport at t=2.5s causes forward & backward rollouts to disagree strongly
    def _pos(t_s: float) -> tuple[int, int, int, int]:
        if t_s < 2.5:
            return 30, 60, 68, 104
        return 220, 70, 258, 114

    frames = []
    for i in range(n_frames):
        fr = bg.copy()
        x1, y1, x2, y2 = _pos(i / fps)
        fr[y1:y2, x1:x2] = cup
        frames.append(fr)

    ds_dir = tmp_path / "adapt_ds" / "ep_adapt"
    _write_mp4(ds_dir / "main.mp4", frames, fps=fps)
    handle = auto_detect(tmp_path / "adapt_ds", adapter_name="video_dir")[0]

    def _vlm_handler(kw: dict[str, Any]) -> dict[str, Any]:
        t_val = float(kw["timestamps_s"][0])
        return {
            "objects": {
                "cup": {
                    "visible": True,
                    "bbox": _px_to_norm_box(*_pos(t_val), w, h),
                    "state": "moved",
                    "status_kind": "neutral",
                }
            }
        }

    # Case A: default adaptive_reanchor_max_calls = 0 -> strictly initial anchors only (2 calls)
    cfg0 = ConfigManager()
    cfg0.settings["vlm"]["s1_grounding"]["batch_size"] = 1
    cfg0.settings["vlm"]["s1_grounding"]["interval_s"] = 5.0
    cfg0.settings["vlm"]["s1_grounding"]["max_keyframes"] = 2
    cfg0.settings["vlm"]["s1_grounding"]["tracking"]["adaptive_reanchor_max_calls"] = 0
    cl0 = _DeterministicVlmClient(_vlm_handler)
    with get_adapter("video_dir").open(handle) as view:
        res0 = run_stage1_grounding(view=view, task=_coffee_task(), client=cl0, cfg=cfg0)  # type: ignore[arg-type]
    assert len(cl0.calls) == 2
    assert len(res0.keyframes_s) == 2

    # Case B: adaptive_reanchor_max_calls = 2 with max_keyframes = 4 -> triggers re-anchor but <= max_keyframes!
    cfg1 = ConfigManager()
    cfg1.settings["vlm"]["s1_grounding"]["batch_size"] = 1
    cfg1.settings["vlm"]["s1_grounding"]["interval_s"] = 5.0
    cfg1.settings["vlm"]["s1_grounding"]["max_keyframes"] = 4
    cfg1.settings["vlm"]["s1_grounding"]["tracking"]["adaptive_reanchor_max_calls"] = 2
    cfg1.settings["vlm"]["s1_grounding"]["tracking"]["adaptive_disagree_ratio"] = 0.25
    cl1 = _DeterministicVlmClient(_vlm_handler)
    with get_adapter("video_dir").open(handle) as view:
        res1 = run_stage1_grounding(view=view, task=_coffee_task(), client=cl1, cfg=cfg1)  # type: ignore[arg-type]

    assert 3 <= len(cl1.calls) <= 4
    assert len(res1.keyframes_s) == len(cl1.calls)
    assert len(res1.object_tracks_2d["cup"].keyframes) == len(cl1.calls)
    assert res1.keyframes_s == sorted(res1.keyframes_s)


def test_grounding_schema_backward_compat_validators_and_tracking_disabled(
    tmp_path: Path,
) -> None:
    # 1. Legacy v1 sparse JSON roundtrips cleanly without new optional fields
    legacy_payload = {
        "schema_version": 1,
        "ref_size": [1280, 720],
        "keyframes_s": [0.0, 2.5],
        "object_tracks_2d": {
            "cup": {
                "object_id": "cup",
                "name": "Cup",
                "tag_pos": "top",
                "is_static_container": False,
                "keyframes": [
                    {"row": 0, "time_s": 0.0, "visible": True, "bbox_norm_1000": [100, 200, 300, 400]},
                    {"row": 75, "time_s": 2.5, "visible": True, "bbox_norm_1000": [150, 210, 350, 410]},
                ],
            }
        },
    }
    parsed = GroundingResult.model_validate(legacy_payload)
    assert parsed.schema_version == 1
    assert parsed.camera_name is None
    assert parsed.dense_times_s == []
    assert parsed.object_tracks_2d["cup"].tracking_mode == "sparse_vlm"

    # 2. Validators reject malformed dense tracks
    with pytest.raises(ValidationError):
        GroundingTrack(
            object_id="cup",
            dense_boxes_norm_1000=[[100, 100, 200, 200]],
            dense_flags=[0],  # flag=0 requires [0,0,0,0]
        )
    with pytest.raises(ValidationError):
        GroundingResult(
            dense_times_s=[0.0, 0.1],
            object_tracks_2d={
                "cup": GroundingTrack(
                    object_id="cup",
                    dense_boxes_norm_1000=[[100, 100, 200, 200]],  # len 1 != 2
                    dense_flags=[1],
                )
            },
        )
    with pytest.raises(ValidationError):
        GroundingTrack(
            object_id="cup",
            dense_boxes_norm_1000=[[100, 100, 1050, 200]],  # > 1000
            dense_flags=[1],
        )

    # 3. tracking.enabled = false produces schema_version=1 sparse result with camera_name
    ds_dir = tmp_path / "off_ds" / "ep_off"
    _write_mp4(ds_dir / "main.mp4", [_make_textured_bg(i, 240, 320) for i in range(30)], fps=30.0)
    handle = auto_detect(tmp_path / "off_ds", adapter_name="video_dir")[0]
    cfg = ConfigManager()
    cfg.settings["vlm"]["s1_grounding"]["tracking"]["enabled"] = False
    cl = _DeterministicVlmClient(
        lambda _kw: {"objects": {"cup": {"visible": True, "bbox": [200, 200, 400, 400]}}}
    )
    with get_adapter("video_dir").open(handle) as view:
        res = run_stage1_grounding(view=view, task=_coffee_task(), client=cl, cfg=cfg)  # type: ignore[arg-type]
    assert res.schema_version == 1
    assert res.camera_name == "main"
    assert res.ref_size == [320, 240]
    assert res.dense_times_s == []
    assert res.object_tracks_2d["cup"].tracking_mode == "sparse_vlm"


def test_viz_mp4_non_16_9_crop_projection_dense_interp_and_camera_guard() -> None:
    # 1. Exact 4:3 (640x480) -> 16:9 (1280x720) center-crop projection math
    # Center crop removes 60px top and 60px bottom (crop_h=360, y0=60, scale=2.0).
    # Box norm [250, 250, 750, 750] => src px [160, 120, 480, 360]
    # => cropped px [160, 60, 480, 300] => target 1280x720 px [320, 120, 960, 600]!
    proj_4_3 = _project_norm_box_for_crop([250, 250, 750, 750], 640, 480, 1280, 720)
    assert proj_4_3 == [320, 120, 960, 600]

    # Box wholly inside the cropped-out top band (y=[0..100] norm => src y=[0..48] < y0=60) returns None
    assert _project_norm_box_for_crop([200, 0, 400, 100], 640, 480, 1280, 720) is None

    # 16:9 identity projection
    assert _project_norm_box_for_crop([100, 200, 500, 600], 1280, 720, 1280, 720) == [
        128,
        144,
        640,
        432,
    ]

    # 2. Dense interpolation, occlusion hiding (flag==0), and anchor metadata lookup
    tr = GroundingTrack(
        object_id="cup",
        name="Cup",
        tracking_mode="bidir_flow_ncc",
        keyframes=[
            GroundingKeyframe(row=0, time_s=0.0, visible=True, bbox_norm_1000=[100, 250, 200, 750], state="start", status_kind="neutral"),
            GroundingKeyframe(row=30, time_s=1.0, visible=True, bbox_norm_1000=[500, 250, 600, 750], state="done", status_kind="good"),
        ],
        dense_boxes_norm_1000=[
            [100, 250, 200, 750],
            [300, 250, 400, 750],
            [0, 0, 0, 0],
            [500, 250, 600, 750],
        ],
        dense_flags=[2, 1, 0, 2],
        dense_conf_pct=[100, 90, 0, 100],
    )
    gr = GroundingResult(
        schema_version=2,
        camera_name="main",
        ref_size=[640, 480],
        keyframes_s=[0.0, 1.0],
        track_fps=10.0,
        dense_times_s=[0.0, 0.2, 0.6, 1.0],
        object_tracks_2d={"cup": tr},
    )

    # Midpoint t=0.1 linearly interpolates [100..200] & [300..400] -> [200, 250, 300, 750]
    # On 640x480 -> 1280x720 center crop: x=[256, 384], y=[120, 600]
    mid = _interp_track_box(tr, 0.1, 1280, 720, grounding=gr, src_w=640, src_h=480)
    assert mid is not None
    assert mid[0] == [256, 120, 384, 600]
    assert mid[1] == "start" and mid[2] == "neutral"

    # Occluded timestamp t=0.6 (flag==0) returns None
    assert _interp_track_box(tr, 0.6, 1280, 720, grounding=gr, src_w=640, src_h=480) is None

    # 3. _HudCompositor camera guard: mismatched camera_name skips drawing boxes
    class _DummyMedia:
        cameras = [CameraSpec(name="main", role="main", width=640, height=480, fps=30.0)]
        duration_s = 1.0
        fps = 30.0

        def read_frame_rgb(self, camera: str, t_s: float) -> np.ndarray:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        def read_frames_batch(self, camera: str, times_s: list[float]) -> list[np.ndarray]:
            return [np.zeros((480, 640, 3), dtype=np.uint8) for _ in times_s]

    meta = EpisodeMeta(
        episode_id="ds/ep",
        dataset="ds",
        stem="ep",
        task_id="coffee_barista",
        adapter_type="video_dir",
        source_uri="/tmp/x",
        source_mtime=1.0,
        episode_fingerprint="fp_test_viz_crop",
        duration_s=1.0,
        fps=30.0,
        frame_count=30,
        cameras=[CameraSpec(name="main", role="main", width=640, height=480, fps=30.0)],
        camera_roles={"main": "main"},
        kinematics=KinematicsSpec(has_kinematics=False),
    )

    view = EpisodeView(meta=meta, media=_DummyMedia())  # type: ignore[arg-type]
    black_4_3 = np.zeros((480, 640, 3), dtype=np.uint8)

    # Matching camera draws bbox at cropped y=120 (canvas y=60+120=180), NOT old uncropped y=180 (canvas y=240)
    res_match = ResolvedEpisode(
        episode_id="ds/ep",
        meta=meta,
        status=EpisodeStatus.VLM_READY,
        qc_segments=[],
        macro_stages=[],
        grounding=gr,
        arm_timelines=ArmTimelines(),
        raw_vlm_segments=[],
        success=SuccessVerdict(verdict="success", passed=True, confidence=0.9),
    )
    fr_match = _HudCompositor(res_match, view).render_frame_bgr(0, 30, 0.1, black_4_3)
    main_pane = fr_match[60:780, 16:1296]
    # Top edge of drawn box at y=120, x=256..384 must be non-black
    assert int(main_pane[120, 320].sum()) > 0
    # Old buggy uncropped y=180 interior (x=320) should still be black!
    assert int(main_pane[180, 320].sum()) == 0

    # Mismatched camera_name="torso_rgbd" must NOT draw boxes onto "main"
    gr_other = gr.model_copy(update={"camera_name": "torso_rgbd"})
    res_mismatch = res_match.model_copy(update={"grounding": gr_other})
    fr_mismatch = _HudCompositor(res_mismatch, view).render_frame_bgr(0, 30, 0.1, black_4_3)
    assert int(fr_mismatch[60:780, 16:1296].sum()) == 0


def test_track_interval_bidirectional_disjoint_no_midpoint_averaging() -> None:
    L, h, w = 9, 120, 160
    bg = cv2.cvtColor(_make_textured_bg(1111, h, w), cv2.COLOR_BGR2GRAY)
    patch_left = cv2.cvtColor(_make_textured_patch(2222, 30, 30, (60, 160, 220)), cv2.COLOR_BGR2GRAY)
    patch_right = cv2.cvtColor(_make_textured_patch(3333, 30, 30, (200, 100, 70)), cv2.COLOR_BGR2GRAY)

    # Two far-apart static textured patches: left at x=[10..40], right at x=[110..140] (IoU=0.0)
    frame = bg.copy()
    frame[40:70, 10:40] = patch_left
    frame[40:70, 110:140] = patch_right
    grays = [frame.copy() for _ in range(L)]
    flow_fwd = [np.zeros((h, w, 2), dtype=np.float32) for _ in range(L - 1)]
    flow_bwd = [np.zeros((h, w, 2), dtype=np.float32) for _ in range(L - 1)]
    bg_steps = [(0.0, 0.0) for _ in range(L - 1)]

    box_start_px = np.array([10.0, 40.0, 40.0, 70.0], dtype=np.float64)
    box_end_px = np.array([110.0, 40.0, 140.0, 70.0], dtype=np.float64)
    cfg = TrackingConfig()

    fused, flags, confs, disagree_ratio = track_interval_bidirectional(
        grays, flow_fwd, flow_bwd, bg_steps, box_start_px, box_end_px, cfg, w, h
    )

    assert flags[0] == FLAG_VLM_ANCHOR and flags[-1] == FLAG_VLM_ANCHOR
    # Midpoint (i=4, u=0.5) has equal weights & IoU < 0.20 -> must hide cleanly
    assert flags[4] == FLAG_INVISIBLE
    assert np.all(fused[4] == 0.0)
    assert confs[4] == 0.0

    for i in range(1, L - 1):
        # Confidence must NEVER report ~0.99 on disjoint tracks
        assert confs[i] < 0.20, f"Disjoint frame {i} reported inflated conf={confs[i]:.3f}"
        if flags[i] != FLAG_INVISIBLE:
            cx = 0.5 * (fused[i][0] + fused[i][2])
            # Never linearly averaged into empty air between x=40 and x=110
            assert not (45.0 < cx < 105.0), f"Frame {i} averaged into empty gap cx={cx:.1f}"
            assert (
                box_iou(fused[i], box_start_px) > 0.7
                or box_iou(fused[i], box_end_px) > 0.7
            )
    assert disagree_ratio > 0.8


def test_track_interval_bidirectional_asymmetric_confidence_picks_stronger_side() -> None:
    L, h, w = 9, 120, 160
    bg = cv2.cvtColor(_make_textured_bg(4444, h, w), cv2.COLOR_BGR2GRAY)
    patch_left = cv2.cvtColor(_make_textured_patch(5555, 30, 30, (70, 180, 210)), cv2.COLOR_BGR2GRAY)
    patch_right = cv2.cvtColor(_make_textured_patch(6666, 30, 30, (210, 90, 80)), cv2.COLOR_BGR2GRAY)

    # Right patch persists cleanly on all frames; left patch vanishes after frame 0 -> low fwd_conf
    grays: list[np.ndarray] = []
    for t in range(L):
        fr = bg.copy()
        if t == 0:
            fr[40:70, 10:40] = patch_left
        else:
            fr[40:70, 10:40] = 128  # blank uniform patch -> NCC drops sharply
        fr[40:70, 110:140] = patch_right
        grays.append(fr)

    flow_fwd = [np.zeros((h, w, 2), dtype=np.float32) for _ in range(L - 1)]
    flow_bwd = [np.zeros((h, w, 2), dtype=np.float32) for _ in range(L - 1)]
    bg_steps = [(0.0, 0.0) for _ in range(L - 1)]

    box_start_px = np.array([10.0, 40.0, 40.0, 70.0], dtype=np.float64)
    box_end_px = np.array([110.0, 40.0, 140.0, 70.0], dtype=np.float64)
    cfg = TrackingConfig(ncc_stride=1)

    fused, flags, confs, _ = track_interval_bidirectional(
        grays, flow_fwd, flow_bwd, bg_steps, box_start_px, box_end_px, cfg, w, h
    )

    # Even near start (i=1, 2), bwd side has much higher confidence so it wins over low-conf fwd
    for i in range(1, L - 1):
        assert flags[i] == FLAG_TRACKED
        assert box_iou(fused[i], box_end_px) > 0.7, f"Expected bwd box at i={i}, got {fused[i]}"
        assert 0.0 < confs[i] < 0.20, f"Expected down-weighted conf < 0.20 at i={i}, got {confs[i]}"


def test_clamp_anchor_scale_outliers_clamps_gt_35pct_preserving_center() -> None:
    def _kf(row: int, t: float, box: list[int] | None, vis: bool = True) -> GroundingKeyframe:
        b = box or [0, 0, 0, 0]
        return GroundingKeyframe(
            row=row,
            time_s=t,
            visible=vis,
            bbox_norm_1000=b,
            bbox_px=[
                int(round(b[0] * 1280 / 1000.0)),
                int(round(b[1] * 720 / 1000.0)),
                int(round(b[2] * 1280 / 1000.0)),
                int(round(b[3] * 720 / 1000.0)),
            ],
            state="ok" if vis else "",
            status_kind="neutral",
        )

    # 4 inliers of 100x100 + 1 oversized outlier 220x220 centered at (500, 400) + 1 invisible
    kfs = [
        _kf(0, 0.0, [100, 100, 200, 200]),
        _kf(15, 0.5, [120, 110, 220, 210]),
        _kf(30, 1.0, [390, 290, 610, 510]),  # 220x220 (+120%!) centered at (500, 400)
        _kf(45, 1.5, None, vis=False),
        _kf(60, 2.0, [140, 120, 240, 220]),
        _kf(75, 2.5, [160, 130, 260, 230]),
    ]
    clamped = clamp_anchor_scale_outliers(kfs, ref_w=1280, ref_h=720, max_scale_dev=0.35)
    assert len(clamped) == len(kfs)
    # Inliers and invisible frame remain unchanged
    assert clamped[0].bbox_norm_1000 == [100, 100, 200, 200]
    assert clamped[3].visible is False and clamped[3].bbox_norm_1000 == [0, 0, 0, 0]
    # Oversized outlier clamped to <= 135x135 while preserving center (500, 400)
    ob = clamped[2].bbox_norm_1000
    ow = ob[2] - ob[0]
    oh = ob[3] - ob[1]
    assert 65 <= ow <= 135 and 65 <= oh <= 135
    assert abs(0.5 * (ob[0] + ob[2]) - 500.0) <= 1.0
    assert abs(0.5 * (ob[1] + ob[3]) - 400.0) <= 1.0
    assert clamped[2].bbox_px == [
        int(round(ob[0] * 1280 / 1000.0)),
        int(round(ob[1] * 720 / 1000.0)),
        int(round(ob[2] * 1280 / 1000.0)),
        int(round(ob[3] * 720 / 1000.0)),
    ]

    # Undersized outlier 40x40 (-60%) centered at (500, 400) -> clamped up to >= 65x65
    kfs_small = [
        _kf(0, 0.0, [100, 100, 200, 200]),
        _kf(15, 0.5, [120, 110, 220, 210]),
        _kf(30, 1.0, [480, 380, 520, 420]),  # 40x40 centered at (500, 400)
        _kf(45, 1.5, [140, 120, 240, 220]),
        _kf(60, 2.0, [160, 130, 260, 230]),
    ]
    clamped_s = clamp_anchor_scale_outliers(kfs_small, ref_w=1280, ref_h=720, max_scale_dev=0.35)
    sb = clamped_s[2].bbox_norm_1000
    assert 65 <= (sb[2] - sb[0]) <= 135 and 65 <= (sb[3] - sb[1]) <= 135
    assert abs(0.5 * (sb[0] + sb[2]) - 500.0) <= 1.0
    assert abs(0.5 * (sb[1] + sb[3]) - 400.0) <= 1.0


def test_baft_clamps_vlm_anchor_scale_breathing_outlier(tmp_path: Path) -> None:
    w, h, fps, dur_s = 640, 360, 30.0, 2.0
    n_frames = int(round(dur_s * fps))
    bg = _make_textured_bg(7777, h, w)
    cup_patch = _make_textured_patch(8888, 72, 64, (50, 160, 230))

    def _cup_px(t_s: float) -> tuple[int, int, int, int]:
        u = max(0.0, min(1.0, t_s / dur_s))
        x1 = int(round(120 + 180 * u))
        y1 = 140
        return x1, y1, x1 + 64, y1 + 72

    frames = []
    for i in range(n_frames):
        fr = bg.copy()
        x1, y1, x2, y2 = _cup_px(i / fps)
        fr[y1:y2, x1:x2] = cup_patch
        frames.append(fr)

    ds_dir = tmp_path / "breath_ds" / "ep_breath"
    _write_mp4(ds_dir / "main.mp4", frames, fps=fps)
    handle = auto_detect(tmp_path / "breath_ds", adapter_name="video_dir")[0]

    # Normal cup size in norm coords is ~100x200; at t=1.0s inject a 2.2x breathing box!
    def _vlm_handler(kw: dict[str, Any]) -> dict[str, Any]:
        ts = [float(t) for t in kw["timestamps_s"]]
        frame_entries: list[dict[str, Any]] = []
        for idx, t_val in enumerate(ts):
            x1, y1, x2, y2 = _cup_px(t_val)
            b = _px_to_norm_box(x1, y1, x2, y2, w, h)
            if abs(t_val - 1.0) < 0.15:
                cx = 0.5 * (b[0] + b[2])
                cy = 0.5 * (b[1] + b[3])
                bw = (b[2] - b[0]) * 2.2
                bh = (b[3] - b[1]) * 2.2
                b = [
                    int(round(cx - 0.5 * bw)),
                    int(round(cy - 0.5 * bh)),
                    int(round(cx + 0.5 * bw)),
                    int(round(cy + 0.5 * bh)),
                ]
            frame_entries.append(
                {
                    "frame_index": idx,
                    "time_s": t_val,
                    "objects": {
                        "cup": {
                            "visible": True,
                            "bbox": b,
                            "state": "moving",
                            "status_kind": "neutral",
                        }
                    },
                }
            )
        return {"frames": frame_entries}

    cfg = ConfigManager()
    cfg.settings["vlm"]["s1_grounding"]["interval_s"] = 0.5
    cfg.settings["vlm"]["s1_grounding"]["max_keyframes"] = 5

    client = _DeterministicVlmClient(_vlm_handler)
    with get_adapter("video_dir").open(handle) as view:
        res = run_stage1_grounding(
            view=view,
            task=TaskSpec(
                task_id="cup_only",
                grounding_objects=[
                    GroundingObjectSpec(id="cup", name="Cup", desc="Moving cup", is_static=False)
                ],
            ),
            client=client,  # type: ignore[arg-type]
            cfg=cfg,
        )

    assert len(client.calls) == 1
    cup = res.object_tracks_2d["cup"]
    mid_kf = min(cup.keyframes, key=lambda k: abs(k.time_s - 1.0))
    norm_w = cup.keyframes[0].bbox_norm_1000[2] - cup.keyframes[0].bbox_norm_1000[0]
    norm_h = cup.keyframes[0].bbox_norm_1000[3] - cup.keyframes[0].bbox_norm_1000[1]
    mid_w = mid_kf.bbox_norm_1000[2] - mid_kf.bbox_norm_1000[0]
    mid_h = mid_kf.bbox_norm_1000[3] - mid_kf.bbox_norm_1000[1]
    assert mid_w <= int(math.ceil(1.35 * norm_w)) + 1
    assert mid_h <= int(math.ceil(1.35 * norm_h)) + 1

    # Center at t=1.0s preserved
    gt_mid = _px_to_norm_box(*_cup_px(1.0), w, h)
    assert abs(0.5 * (mid_kf.bbox_norm_1000[0] + mid_kf.bbox_norm_1000[2]) - 0.5 * (gt_mid[0] + gt_mid[2])) <= 3.0
    assert abs(0.5 * (mid_kf.bbox_norm_1000[1] + mid_kf.bbox_norm_1000[3]) - 0.5 * (gt_mid[1] + gt_mid[3])) <= 3.0

    # Dense track at t=1.0s is also scale-clamped
    d_mid = min(range(len(res.dense_times_s)), key=lambda i: abs(res.dense_times_s[i] - 1.0))
    db = cup.dense_boxes_norm_1000[d_mid]
    assert (db[2] - db[0]) <= int(math.ceil(1.40 * norm_w))
    assert (db[3] - db[1]) <= int(math.ceil(1.40 * norm_h))


def test_run_baft_tracking_disjoint_even_and_asymmetric_no_smooth_smearing() -> None:
    """Reproducers A (L=6), B (L=16), and C (L=9 asymmetric): _smooth_dense_track must
    split contiguous runs across disjoint jumps (IoU < 0.20) so Savitzky-Golay never
    smears dense_boxes_norm_1000 into the empty midpoint gap or pulls anchors."""
    w, h, fps = 320, 180, 15.0
    bg = _make_textured_bg(901, h, w)
    patch_left = _make_textured_patch(902, 36, 36, (60, 140, 230))
    patch_right = _make_textured_patch(903, 36, 36, (200, 90, 80))

    left_norm = [62, 389, 175, 589]  # px [20, 70, 56, 106], cx=118.5
    right_norm = [688, 389, 800, 589]  # px [220, 70, 256, 106], cx=744.0

    class _MemMedia:
        def __init__(self, frames_rgb: list[np.ndarray]) -> None:
            self._frames = frames_rgb
            self.fps = fps
            self.duration_s = (len(frames_rgb) - 1) / fps
            self.cameras = [CameraSpec(name="main", role="main", width=w, height=h, fps=fps)]

        def read_frames_batch(self, camera: str, times_s: list[float]) -> list[np.ndarray]:
            return [
                self._frames[max(0, min(len(self._frames) - 1, int(round(t * self.fps))))]
                for t in times_s
            ]

    def _run_disjoint_case(
        name: str,
        n_frames: int,
        occlude_right: bool = False,
        cfg_override: TrackingConfig | None = None,
    ) -> GroundingTrack:
        frames: list[np.ndarray] = []
        for i in range(n_frames):
            fr = bg.copy()
            fr[70:106, 20:56] = patch_left
            fr[70:106, 220:256] = patch_right
            if occlude_right and 0 < i < n_frames - 1:
                # Replace right patch with uniform gray so bwd conf drops >= 0.15 below fwd conf
                fr[70:106, 220:256] = 128
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))

        dur_s = (n_frames - 1) / fps
        anchors = [0.0, round(dur_s, 3)]
        meta = EpisodeMeta(
            episode_id=f"ds/{name}",
            dataset="ds",
            stem=name,
            task_id="t",
            adapter_type="video_dir",
            source_uri="/tmp/disjoint",
            source_mtime=1.0,
            episode_fingerprint=f"fp_{name}",
            duration_s=dur_s,
            fps=fps,
            frame_count=n_frames,
            cameras=[CameraSpec(name="main", role="main", width=w, height=h, fps=fps)],
            camera_roles={"main": "main"},
            kinematics=KinematicsSpec(has_kinematics=False),
        )
        view = EpisodeView(meta=meta, media=_MemMedia(frames))  # type: ignore[arg-type]
        tr = GroundingTrack(
            object_id="cup",
            name="Cup",
            keyframes=[
                GroundingKeyframe(
                    row=0,
                    time_s=anchors[0],
                    visible=True,
                    bbox_norm_1000=list(left_norm),
                    bbox_px=[20, 70, 56, 106],
                ),
                GroundingKeyframe(
                    row=n_frames - 1,
                    time_s=anchors[1],
                    visible=True,
                    bbox_norm_1000=list(right_norm),
                    bbox_px=[220, 70, 256, 106],
                ),
            ],
        )
        out = run_baft_tracking(
            view=view,
            camera="main",
            anchor_times_s=anchors,
            raw_tracks={"cup": tr},
            ref_w=w,
            ref_h=h,
            static_min_frames=5,
            jitter_tol_px=6.0,
            static_defaults={},
            cfg=cfg_override or TrackingConfig(),
        )
        return out.tracks["cup"]

    # Reproducer A (L=6) & Reproducer B (L=16): symmetric even-length intervals (no FLAG_INVISIBLE emitted)
    for label, n_fr in [("repro_a_L6", 6), ("repro_b_L16", 16)]:
        tr_res = _run_disjoint_case(label, n_fr)
        assert len(tr_res.dense_boxes_norm_1000) == n_fr
        for idx, (box, fl, conf) in enumerate(
            zip(tr_res.dense_boxes_norm_1000, tr_res.dense_flags, tr_res.dense_conf_pct)
        ):
            if 0 < idx < n_fr - 1:
                assert conf < 20, f"{label} interior frame {idx} has inflated conf={conf}"
            if fl != FLAG_INVISIBLE:
                cx = 0.5 * (box[0] + box[2])
                assert not (250.0 < cx < 610.0), (
                    f"{label} frame {idx} smeared into empty midpoint gap cx={cx:.1f}"
                )
                assert box_iou(box, left_norm) > 0.85 or box_iou(box, right_norm) > 0.85, (
                    f"{label} frame {idx} distorted away from object sides: {box}"
                )

    # Reproducer C (L=9): asymmetric confidence (all interior frames pick left; jump occurs right before anchor 8)
    tr_asym = _run_disjoint_case(
        "repro_c_L9_asym",
        9,
        occlude_right=True,
        cfg_override=TrackingConfig(ncc_stride=1),
    )
    assert len(tr_asym.dense_boxes_norm_1000) == 9
    assert all(fl != FLAG_INVISIBLE for fl in tr_asym.dense_flags)
    for idx in range(1, 8):
        assert tr_asym.dense_conf_pct[idx] < 20
        box = tr_asym.dense_boxes_norm_1000[idx]
        cx = 0.5 * (box[0] + box[2])
        assert not (250.0 < cx < 610.0), f"repro_c frame {idx} smeared into gap cx={cx:.1f}"
        assert box_iou(box, left_norm) > 0.85
    # End anchor (index 8) stays cleanly locked on right patch without being pulled toward left
    assert box_iou(tr_asym.dense_boxes_norm_1000[8], right_norm) > 0.90


def test_track_interval_bidirectional_disjoint_prefers_supported_over_below_thresh() -> None:
    """When IoU < 0.20 and one side is below occlusion_conf_thresh while the other is >= thresh,
    prefer the supported side even when abs(cf - cb) < 0.15 and wf > wb."""
    L, h, w = 11, 120, 160
    bg = cv2.cvtColor(_make_textured_bg(911, h, w), cv2.COLOR_BGR2GRAY)
    patch_left = cv2.cvtColor(_make_textured_patch(912, 30, 30, (70, 180, 210)), cv2.COLOR_BGR2GRAY)
    patch_right = cv2.cvtColor(_make_textured_patch(913, 30, 30, (210, 90, 80)), cv2.COLOR_BGR2GRAY)

    # Occlude 60% of left patch so cf drops to ~0.77 (< thresh=0.85), while occluding 25% of
    # right patch so cb drops to ~0.88 (>= thresh=0.85), giving abs(cf - cb) ~ 0.11 < 0.15!
    grays: list[np.ndarray] = []
    for t in range(L):
        fr = bg.copy()
        if t == 0:
            fr[40:70, 10:40] = patch_left
        else:
            p_left_occ = patch_left.copy()
            p_left_occ[:18, :] = 128
            fr[40:70, 10:40] = p_left_occ
        if t == L - 1:
            fr[40:70, 110:140] = patch_right
        else:
            p_right_occ = patch_right.copy()
            p_right_occ[:8, :] = 128
            fr[40:70, 110:140] = p_right_occ
        grays.append(fr)

    flow_fwd = [np.zeros((h, w, 2), dtype=np.float32) for _ in range(L - 1)]
    flow_bwd = [np.zeros((h, w, 2), dtype=np.float32) for _ in range(L - 1)]
    bg_steps = [(0.0, 0.0) for _ in range(L - 1)]

    box_start_px = np.array([10.0, 40.0, 40.0, 70.0], dtype=np.float64)
    box_end_px = np.array([110.0, 40.0, 140.0, 70.0], dtype=np.float64)
    cfg = TrackingConfig(ncc_stride=1, ncc_min_score=0.90, occlusion_conf_thresh=0.85)

    fused, flags, confs, _ = track_interval_bidirectional(
        grays, flow_fwd, flow_bwd, bg_steps, box_start_px, box_end_px, cfg, w, h
    )
    # Even at i=2 (u=0.2, where wf >> wb and abs(cf - cb) < 0.15), supported right box wins
    assert flags[2] == FLAG_TRACKED
    assert box_iou(fused[2], box_end_px) > 0.7
    assert 0.0 < confs[2] < 0.20


def test_baft_suppresses_moderate_and_multiframe_anchor_jitter_via_flow_and_ncc(
    tmp_path: Path,
) -> None:
    """Moderate (<250 norm) and 2-frame consecutive VLM anchor jitter must be suppressed
    using optical-flow prediction residual + NCC confidence instead of rubber-banding."""
    w, h, fps, dur_s = 640, 360, 15.0, 4.0
    n_frames = int(round(dur_s * fps))
    bg = _make_textured_bg(951, h, w)
    cup_patch = _make_textured_patch(952, 70, 60, (55, 165, 235))

    def _gt_cup_px(t_s: float) -> tuple[int, int, int, int]:
        u = max(0.0, min(1.0, t_s / dur_s))
        x1 = int(round(100 + 260 * u))
        y1 = int(round(145 + 15 * math.sin(math.pi * u)))
        return x1, y1, x1 + 60, y1 + 70

    frames: list[np.ndarray] = []
    for i in range(n_frames):
        fr = bg.copy()
        x1, y1, x2, y2 = _gt_cup_px(i / fps)
        fr[y1:y2, x1:x2] = cup_patch
        frames.append(fr)

    ds_dir = tmp_path / "jitter_ds" / "ep_jitter"
    _write_mp4(ds_dir / "main.mp4", frames, fps=fps)
    handle = auto_detect(tmp_path / "jitter_ds", adapter_name="video_dir")[0]

    # 5 anchors at t=0.0, 1.0, 2.0, 3.0, 4.0s.
    # Inject a 2-frame moderate jitter burst at t=1.0s (+85 norm X, -50 norm Y) and t=2.0s (-80 norm X, +55 norm Y)
    # -- BOTH are well below the legacy 250-norm outlier_jump_norm threshold!
    def _jitter_vlm(kw: dict[str, Any]) -> dict[str, Any]:
        ts = [float(t) for t in kw["timestamps_s"]]
        entries: list[dict[str, Any]] = []
        for idx, t_val in enumerate(ts):
            gt = _px_to_norm_box(*_gt_cup_px(t_val), w, h)
            if abs(t_val - 1.0) < 0.15:
                box = [gt[0] + 85, max(0, gt[1] - 50), gt[2] + 85, max(10, gt[3] - 50)]
            elif abs(t_val - 2.0) < 0.15:
                box = [max(0, gt[0] - 80), gt[1] + 55, max(10, gt[2] - 80), gt[3] + 55]
            else:
                box = gt
            entries.append(
                {
                    "frame_index": idx,
                    "time_s": t_val,
                    "objects": {
                        "cup": {
                            "visible": True,
                            "bbox": box,
                            "state": "moving",
                            "status_kind": "neutral",
                        }
                    },
                }
            )
        return {"frames": entries}

    cfg = ConfigManager()
    cfg.settings["vlm"]["s1_grounding"]["interval_s"] = 1.0
    cfg.settings["vlm"]["s1_grounding"]["max_keyframes"] = 5

    client = _DeterministicVlmClient(_jitter_vlm)
    with get_adapter("video_dir").open(handle) as view:
        res = run_stage1_grounding(
            view=view,
            task=TaskSpec(
                task_id="cup_jitter",
                grounding_objects=[
                    GroundingObjectSpec(id="cup", name="Cup", desc="Moving cup", is_static=False)
                ],
            ),
            client=client,  # type: ignore[arg-type]
            cfg=cfg,
        )

    cup = res.object_tracks_2d["cup"]
    # Evaluate accuracy right around the two jittered anchor timestamps (1.0s and 2.0s)
    for target_t in (1.0, 2.0):
        idx = min(range(len(res.dense_times_s)), key=lambda i: abs(res.dense_times_s[i] - target_t))
        gt_b = _px_to_norm_box(*_gt_cup_px(res.dense_times_s[idx]), w, h)
        pred_b = cup.dense_boxes_norm_1000[idx]
        err = box_center_dist(pred_b, gt_b)
        assert err < 35.0, (
            f"Jitter burst at t={target_t}s was not suppressed (err={err:.1f}, pred={pred_b}, gt={gt_b})"
        )
        assert box_iou(pred_b, gt_b) > 0.60


def test_smooth_dense_track_adaptive_anchor_weighting_prevents_rubber_banding() -> None:
    """_smooth_dense_track must adaptively down-weight noisy VLM anchors when local optical
    flow is smooth & consistent, while firmly preserving anchors that agree with flow."""
    from robo_vlm_qc.vlm.tracker import _smooth_dense_track

    w, h = 320, 180
    n = 15
    # Smooth linear trajectory moving +2px/frame in x, constant 40x40 box
    gt_boxes = np.array(
        [[40.0 + 2.0 * i, 60.0, 80.0 + 2.0 * i, 100.0] for i in range(n)],
        dtype=np.float64,
    )
    flags = np.full(n, FLAG_TRACKED, dtype=np.int32)
    flags[0] = FLAG_VLM_ANCHOR
    flags[7] = FLAG_VLM_ANCHOR
    flags[n - 1] = FLAG_VLM_ANCHOR
    confs = np.full(n, 0.92, dtype=np.float64)

    # Case 1: Noisy interior anchor at i=7 jumps +18px in x and +14px in width/height
    # (still overlapping IoU > 0.20 so it stays inside the same contiguous run)
    noisy_boxes = gt_boxes.copy()
    noisy_boxes[7] = np.array(
        [gt_boxes[7, 0] + 18.0, gt_boxes[7, 1] - 7.0, gt_boxes[7, 2] + 32.0, gt_boxes[7, 3] + 7.0],
        dtype=np.float64,
    )
    flow_info = {7: (box_iou(noisy_boxes[7], gt_boxes[7]), 0.92, 0.90, gt_boxes[7].copy())}

    sm_noisy = _smooth_dense_track(
        noisy_boxes,
        flags,
        smooth_window=5,
        w=w,
        h=h,
        confs=confs,
        anchor_flow_info=flow_info,
    )
    # Old hardcoded 0.8*anchor + 0.2*sm would keep ~80% of the +25px center shift (>18px error)!
    # Adaptive flow-agreement weighting must suppress the jump strongly (< 8px error).
    err_noisy = box_center_dist(sm_noisy[7], gt_boxes[7])
    assert err_noisy < 8.5, f"Adaptive weighting failed to suppress anchor jump: err={err_noisy:.2f}px"
    # Scale jump (+14px width) is also suppressed well below old 0.8*54 + 0.2*40 = 51.2px
    sm_w = float(sm_noisy[7, 2] - sm_noisy[7, 0])
    assert sm_w < 46.0, f"Adaptive weighting failed to suppress scale breathing: w={sm_w:.2f}px"

    # Case 2: Clean interior anchor agreeing closely with optical flow stays firmly pinned
    clean_boxes = gt_boxes.copy()
    clean_boxes[7] = gt_boxes[7] + np.array([1.0, 0.0, 1.0, 0.0])
    flow_info_clean = {7: (box_iou(clean_boxes[7], gt_boxes[7]), 0.95, 0.95, gt_boxes[7].copy())}
    sm_clean = _smooth_dense_track(
        clean_boxes,
        flags,
        smooth_window=5,
        w=w,
        h=h,
        confs=confs,
        anchor_flow_info=flow_info_clean,
    )
    assert box_center_dist(sm_clean[7], clean_boxes[7]) < 1.0
