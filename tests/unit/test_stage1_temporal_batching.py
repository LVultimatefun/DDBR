"""Offline unit & integration tests for Stage 1 temporal multi-frame grounding."""

from __future__ import annotations

import threading
from typing import Any, Callable

import numpy as np
import pytest

from robo_vlm_qc.config import ConfigManager
from robo_vlm_qc.io.base import EpisodeView
from robo_vlm_qc.models.episode import CameraSpec, EpisodeMeta, KinematicsSpec
from robo_vlm_qc.models.taxonomy import GroundingObjectSpec, TaskSpec
from robo_vlm_qc.vlm.client import BudgetTracker, VlmCancelledError, VlmError
from robo_vlm_qc.vlm.prompt_builder import PromptBuilder
from robo_vlm_qc.vlm.stage1_grounding import (
    _parse_batch_response,
    _partition_batches,
    run_stage1_grounding,
)


class _FakeMediaSource:
    """Lightweight in-memory MediaSource avoiding MP4 encoding overhead for 138s tests."""

    def __init__(self, *, duration_s: float, fps: float = 30.0, width: int = 1280, height: int = 720) -> None:
        self.duration_s = float(duration_s)
        self.fps = float(fps)
        self.cameras = [
            CameraSpec(name="ego", role="main", width=width, height=height, fps=self.fps)
        ]
        self._w = width
        self._h = height

    def read_frame_rgb(self, camera: str, t_s: float) -> np.ndarray:
        return np.zeros((self._h, self._w, 3), dtype=np.uint8)

    def read_frames_batch(self, camera: str, times_s: list[float]) -> list[np.ndarray]:
        return [np.zeros((self._h, self._w, 3), dtype=np.uint8) for _ in times_s]


def _make_synthetic_view(duration_s: float, fps: float = 30.0) -> EpisodeView:
    meta = EpisodeMeta(
        episode_id="ds/ep_synth",
        dataset="ds",
        stem="ep_synth",
        task_id="coffee_barista",
        adapter_type="video_dir",
        source_uri="/tmp/synth",
        source_mtime=1.0,
        episode_fingerprint=f"fp_{duration_s}",
        duration_s=float(duration_s),
        fps=float(fps),
        frame_count=max(1, int(round(duration_s * fps))),
        cameras=[CameraSpec(name="ego", role="main", width=1280, height=720, fps=fps)],
        camera_roles={"main": "ego"},
        kinematics=KinematicsSpec(has_kinematics=False),
    )
    return EpisodeView(meta=meta, media=_FakeMediaSource(duration_s=duration_s, fps=fps))  # type: ignore[arg-type]


class _FakeClient:
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
        self._lock = threading.Lock()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def model_for_stage(self, stage: str, override: str | None = None) -> str:
        return override or "fake-gpt-4o"

    def chat_json(self, **kwargs: Any) -> dict[str, Any]:
        if self.is_cancelled():
            raise VlmCancelledError("cancelled before call")
        with self._lock:
            self.calls.append(kwargs)
        return self.handler(kwargs)


def _coffee_task() -> TaskSpec:
    return TaskSpec(
        task_id="coffee_barista",
        goal="Place the cup beneath the coffee machine to brew coffee",
        grounding_objects=[
            GroundingObjectSpec(
                id="cup",
                name="Cup",
                desc="Cup grasped and moved by the robotic hand",
                is_static=False,
            ),
            GroundingObjectSpec(
                id="coffee_machine",
                name="Coffee Machine",
                desc="Countertop coffee machine body",
                is_static=True,
            ),
        ],
    )


def test_s1_138s_episode_temporal_batching_call_reduction() -> None:
    cfg = ConfigManager()
    # Disable local optical-flow tracking here so 138s test executes in <0.1s
    cfg.settings["vlm"]["s1_grounding"]["tracking"]["enabled"] = False

    def _batched_handler(kw: dict[str, Any]) -> dict[str, Any]:
        ts = list(kw["timestamps_s"])
        return {
            "frames": [
                {
                    "frame_index": idx,
                    "time_s": float(t),
                    "objects": {
                        "cup": {
                            "visible": True,
                            "bbox": [100 + idx * 10, 200, 220 + idx * 10, 340],
                            "state": "moving",
                            "status_kind": "neutral",
                        },
                        "coffee_machine": {
                            "visible": True,
                            "bbox": [450, 150, 720, 520],
                            "state": "stationary",
                            "status_kind": "good",
                        },
                    },
                }
                for idx, t in enumerate(ts)
            ]
        }

    client = _FakeClient(_batched_handler)
    view = _make_synthetic_view(duration_s=138.0)
    res = run_stage1_grounding(
        view=view,
        task=_coffee_task(),
        client=client,  # type: ignore[arg-type]
        cfg=cfg,
    )

    # 32 keyframes / batch_size=8 => strictly 4 calls (~88.9% reduction vs 36 calls)
    assert 2 <= len(client.calls) <= 4
    assert len(client.calls) == 4
    for call in client.calls:
        imgs = call["images"]
        ts = call["timestamps_s"]
        assert 6 <= len(imgs) <= 9
        assert len(imgs) == len(ts)
        assert call["labels"] == [f"frame_{i}" for i in range(len(imgs))]
        assert call["max_tokens"] >= 2400
        assert call["max_tokens"] >= 280 * len(imgs)

    assert len(res.keyframes_s) == 32
    assert res.keyframes_s == sorted(res.keyframes_s)
    assert res.keyframes_s[0] == 0.0
    assert res.keyframes_s[-1] == pytest.approx(138.0)
    assert len(res.object_tracks_2d["cup"].keyframes) == 32
    assert all(k.visible for k in res.object_tracks_2d["cup"].keyframes)


def test_s1_partition_batches_avoids_single_frame_tail() -> None:
    dummy = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(32)]
    times32 = [float(i) for i in range(32)]

    b32 = _partition_batches(times32, dummy, batch_size=8)
    assert [len(b) for b in b32] == [8, 8, 8, 8]

    # 9 frames with batch_size=8 must split as [5, 4] (never [8, 1])
    b9 = _partition_batches(times32[:9], dummy[:9], batch_size=8)
    assert [len(b) for b in b9] == [5, 4]

    b10 = _partition_batches(times32[:10], dummy[:10], batch_size=8)
    assert [len(b) for b in b10] == [5, 5]

    b6 = _partition_batches(times32[:6], dummy[:6], batch_size=8)
    assert [len(b) for b in b6] == [6]

    # Legacy single-frame path when batch_size == 1
    b1 = _partition_batches(times32[:9], dummy[:9], batch_size=1)
    assert [len(b) for b in b1] == [1] * 9


def test_s1_prompt_temporal_consistency_and_builder_compat() -> None:
    cfg = ConfigManager()
    pb = PromptBuilder(cfg)
    task = _coffee_task()

    # Preserve existing build_s1(task=..., camera_roles=...) signature
    sys_p, usr_p = pb.build_s1(task=task, camera_roles={"main": "ego"})
    sys_p2, usr_p2 = pb.build_s1(task_spec=task)
    assert sys_p and usr_p and sys_p2 and usr_p2

    # Stage classifier keywords for MockVlmServer / E2E tests must remain intact
    assert "2D Grounding" in sys_p
    assert "bbox" in sys_p and "[0, 1000]" in sys_p
    assert "Target objects to ground" in usr_p

    # Batched multi-frame JSON schema + temporal identity lock + consistent semantic boundary
    assert '"frames"' in sys_p and "frame_index" in sys_p and "time_s" in sys_p
    assert "Target Identity Lock" in sys_p
    assert "Consistent Box Boundary" in sys_p
    assert "breathing" in sys_p.lower() or "fluctuat" in sys_p.lower() or "expanding" in sys_p.lower()


def test_s1_multiframe_json_parsing_variants_and_missing_frames() -> None:
    dummy = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(4)]
    batch = [(10, 0.0, dummy[0]), (11, 2.5, dummy[1]), (12, 5.0, dummy[2]), (13, 7.5, dummy[3])]

    # 1. Out-of-order "frames" via frame_index + missing index 2 -> treated as invisible/empty
    resp_ooo = {
        "frames": [
            {"frame_index": 3, "objects": {"cup": {"visible": True, "bbox": [300, 300, 400, 400]}}},
            {"frame_index": 0, "objects": {"cup": {"visible": True, "bbox": [100, 100, 200, 200]}}},
            {"frame_index": 1, "objects": {"cup": {"visible": True, "bbox": [200, 200, 300, 300]}}},
        ]
    }
    parsed = _parse_batch_response(resp_ooo, batch)
    assert parsed[10]["cup"]["bbox"] == [100, 100, 200, 200]
    assert parsed[11]["cup"]["bbox"] == [200, 200, 300, 300]
    assert parsed[12] == {}
    assert parsed[13]["cup"]["bbox"] == [300, 300, 400, 400]

    # 2. "keyframes" alias matched by rounded time_s
    resp_kf = {
        "keyframes": [
            {"time_s": 5.02, "objects": {"cup": {"visible": True, "bbox": [0.2, 0.2, 0.4, 0.4]}}},
            {"time_s": 0.0, "objects": {"cup": {"visible": True, "bbox": [100, 100, 200, 200]}}},
        ]
    }
    parsed_kf = _parse_batch_response(resp_kf, batch)
    assert parsed_kf[10]["cup"]["bbox"] == [100, 100, 200, 200]
    assert parsed_kf[12]["cup"]["bbox"] == [0.2, 0.2, 0.4, 0.4]
    assert parsed_kf[11] == {} and parsed_kf[13] == {}

    # 2b. Duplicate/out-of-range explicit frame_index with valid time_s must NOT steal a different frame in Pass 2
    resp_dup_fi = {
        "frames": [
            {"frame_index": 0, "time_s": 0.0, "objects": {"cup": {"visible": True, "bbox": [100, 100, 200, 200]}}},
            {"frame_index": 0, "time_s": 2.5, "objects": {"cup": {"visible": True, "bbox": [999, 999, 999, 999]}}},
            {"frame_index": 99, "time_s": 5.0, "objects": {"cup": {"visible": True, "bbox": [888, 888, 888, 888]}}},
        ]
    }
    parsed_dup = _parse_batch_response(resp_dup_fi, batch)
    assert parsed_dup[10]["cup"]["bbox"] == [100, 100, 200, 200]
    assert parsed_dup[11] == {}
    assert parsed_dup[12] == {}

    # 2c. Legacy {"objects": ...} broadcast creates isolated per-frame dicts
    parsed_legacy = _parse_batch_response({"objects": {"cup": {"visible": True, "bbox": [10, 10, 50, 50]}}}, batch)
    assert parsed_legacy[10] is not parsed_legacy[11]

    # 3. End-to-end missing frame yields visible=False safely
    cfg = ConfigManager()
    cfg.settings["vlm"]["s1_grounding"]["tracking"]["enabled"] = False
    client = _FakeClient(lambda _kw: resp_ooo)
    view = _make_synthetic_view(duration_s=7.5)
    res = run_stage1_grounding(view=view, task=_coffee_task(), client=client, cfg=cfg)  # type: ignore[arg-type]
    cup_kfs = res.object_tracks_2d["cup"].keyframes
    assert len(cup_kfs) == 4
    assert [k.visible for k in cup_kfs] == [True, True, False, True]
    assert cup_kfs[2].bbox_norm_1000 == [0, 0, 0, 0]


def test_s1_backward_compat_batch_size_1_and_legacy_objects_mock() -> None:
    # 1. batch_size=1 emits 1 image per call and omits batch labels
    cfg1 = ConfigManager()
    cfg1.settings["vlm"]["s1_grounding"]["batch_size"] = 1
    cfg1.settings["vlm"]["s1_grounding"]["max_tokens"] = 800
    cfg1.settings["vlm"]["s1_grounding"]["tracking"]["enabled"] = False

    def _legacy_single(kw: dict[str, Any]) -> dict[str, Any]:
        assert len(kw["images"]) == 1
        assert len(kw["timestamps_s"]) == 1
        assert "labels" not in kw
        assert kw["max_tokens"] == 800
        return {
            "objects": {
                "cup": {"visible": True, "bbox": [150, 250, 300, 420], "state": "ok"}
            }
        }

    cl1 = _FakeClient(_legacy_single)
    res1 = run_stage1_grounding(
        view=_make_synthetic_view(duration_s=20.0),
        task=_coffee_task(),
        client=cl1,  # type: ignore[arg-type]
        cfg=cfg1,
    )
    assert len(cl1.calls) == 9
    assert len(res1.object_tracks_2d["cup"].keyframes) == 9

    # 2. batch_size=8 with legacy top-level {"objects": ...} mock broadcasts cleanly
    cfg8 = ConfigManager()
    cfg8.settings["vlm"]["s1_grounding"]["batch_size"] = 8
    cfg8.settings["vlm"]["s1_grounding"]["tracking"]["enabled"] = False
    cl8 = _FakeClient(
        lambda _kw: {
            "objects": {"cup": {"visible": True, "bbox": [200, 200, 350, 380], "state": "ok"}}
        }
    )
    res8 = run_stage1_grounding(
        view=_make_synthetic_view(duration_s=20.0),
        task=_coffee_task(),
        client=cl8,  # type: ignore[arg-type]
        cfg=cfg8,
    )
    assert len(cl8.calls) == 2
    assert len(res8.object_tracks_2d["cup"].keyframes) == 9
    assert all(k.visible and k.bbox_norm_1000 == [200, 200, 350, 380] for k in res8.object_tracks_2d["cup"].keyframes)


def test_s1_batched_cancellation_and_partial_error_fail_fast() -> None:
    # 1. Cooperative cancel_event triggered during batch execution raises VlmCancelledError
    cfg = ConfigManager()
    cancel_ev = threading.Event()
    call_n = {"n": 0}

    def _cancel_mid(kw: dict[str, Any]) -> dict[str, Any]:
        call_n["n"] += 1
        cancel_ev.set()
        return {"objects": {"cup": {"visible": True, "bbox": [100, 100, 200, 200]}}}

    cl_cancel = _FakeClient(_cancel_mid, cancel_event=cancel_ev)
    with pytest.raises(VlmCancelledError):
        run_stage1_grounding(
            view=_make_synthetic_view(duration_s=20.0),
            task=_coffee_task(),
            client=cl_cancel,  # type: ignore[arg-type]
            cfg=cfg,
            cancel_event=cancel_ev,
        )

    # 2. Partial batch VlmError fails fast rather than returning truncated keyframes
    err_n = {"n": 0}
    lock = threading.Lock()

    def _fail_second(kw: dict[str, Any]) -> dict[str, Any]:
        with lock:
            err_n["n"] += 1
            cur = err_n["n"]
        if cur == 2:
            raise VlmError("Batch 2 failed")
        return {"objects": {"cup": {"visible": True, "bbox": [100, 100, 200, 200]}}}

    cl_err = _FakeClient(_fail_second)
    with pytest.raises(VlmError, match="Batch 2 failed"):
        run_stage1_grounding(
            view=_make_synthetic_view(duration_s=20.0),
            task=_coffee_task(),
            client=cl_err,  # type: ignore[arg-type]
            cfg=cfg,
        )
