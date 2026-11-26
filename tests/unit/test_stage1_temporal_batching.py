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
from robo_vlm_qc.vlm.client import BudgetTracker, VlmBlindVisionError, VlmCancelledError, VlmError
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
                desc="Cup grasped and moved by the manipulator",
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
    # 1. Default batch_size=4 + reference_anchor_enabled=True (anchor-relay path)
    cfg = ConfigManager()
    cfg.settings["vlm"]["s1_grounding"]["batch_size"] = 4
    cfg.settings["vlm"]["s1_grounding"]["reference_anchor_enabled"] = True
    # Disable local optical-flow tracking here so 138s test executes in <0.1s
    cfg.settings["vlm"]["s1_grounding"]["tracking"]["enabled"] = False

    def _relay_handler(kw: dict[str, Any]) -> dict[str, Any]:
        ts = list(kw["timestamps_s"])
        labels = list(kw.get("labels") or [])
        has_ref = bool(labels and labels[0] == "ref_frame_0")
        target_ts = ts[1:] if has_ref else ts
        # Deliberately drift coffee_machine in middle batches (t in [35s, 75s]) while majority agree
        t0_b = float(target_ts[0]) if target_ts else 0.0
        cm_box = [820, 700, 960, 920] if (35.0 <= t0_b <= 75.0) else [450, 150, 720, 520]
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
                            "bbox": cm_box,
                            "state": "stationary",
                            "status_kind": "good",
                        },
                    },
                }
                for idx, t in enumerate(target_ts)
            ]
        }

    client = _FakeClient(_relay_handler)
    view = _make_synthetic_view(duration_s=138.0)
    res = run_stage1_grounding(
        view=view,
        task=_coffee_task(),
        client=client,  # type: ignore[arg-type]
        cfg=cfg,
    )

    # 32 keyframes / batch_size=4 => 8 calls (Batch 0 first, then Batch 1..7 with ref_frame_0)
    assert len(client.calls) == 8
    first_call = client.calls[0]
    assert len(first_call["images"]) == 4
    assert first_call["labels"] == ["frame_0", "frame_1", "frame_2", "frame_3"]
    assert first_call["timestamps_s"][0] == 0.0

    for call in client.calls[1:]:
        imgs = call["images"]
        ts = call["timestamps_s"]
        assert len(imgs) == 5  # ref_frame_0 + 4 target frames
        assert len(imgs) == len(ts)
        assert call["labels"] == ["ref_frame_0", "frame_0", "frame_1", "frame_2", "frame_3"]
        assert ts[0] == 0.0
        # Reference image has colored anchor boxes & labels drawn onto it
        assert int(imgs[0].sum()) > 0
        assert "ref_frame_0" in call["user_prompt"]
        assert "coffee_machine" in call["user_prompt"]
        assert call["max_tokens"] >= 2400

    assert len(res.keyframes_s) == 32
    assert res.keyframes_s == sorted(res.keyframes_s)
    assert res.keyframes_s[0] == 0.0
    assert res.keyframes_s[-1] == pytest.approx(138.0)
    assert len(res.object_tracks_2d["cup"].keyframes) == 32
    assert all(k.visible for k in res.object_tracks_2d["cup"].keyframes)
    # Fixed-camera static consensus lock prevents Batch 1..7 drift on coffee_machine
    cm_kfs = res.object_tracks_2d["coffee_machine"].keyframes
    assert len(cm_kfs) == 32
    assert all(k.visible and k.bbox_norm_1000 == [450, 150, 720, 520] for k in cm_kfs)

    # 2. Explicit parallel batching: batch_size=8 + reference_anchor_enabled=False => 4 calls, no ref_frame_0
    cfg8 = ConfigManager()
    cfg8.settings["vlm"]["s1_grounding"]["batch_size"] = 8
    cfg8.settings["vlm"]["s1_grounding"]["reference_anchor_enabled"] = False
    cfg8.settings["vlm"]["s1_grounding"]["tracking"]["enabled"] = False
    client8 = _FakeClient(_relay_handler)
    res8 = run_stage1_grounding(
        view=view,
        task=_coffee_task(),
        client=client8,  # type: ignore[arg-type]
        cfg=cfg8,
    )
    assert len(client8.calls) == 4
    for call in client8.calls:
        imgs = call["images"]
        ts = call["timestamps_s"]
        assert 6 <= len(imgs) <= 9
        assert len(imgs) == len(ts)
        assert call["labels"] == [f"frame_{i}" for i in range(len(imgs))]
        assert call["max_tokens"] >= 2400
        assert call["max_tokens"] >= 280 * len(imgs)
    assert len(res8.keyframes_s) == 32
    assert all(k.visible for k in res8.object_tracks_2d["cup"].keyframes)


def test_s1_partition_batches_avoids_single_frame_tail() -> None:
    dummy = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(32)]
    times32 = [float(i) for i in range(32)]

    b32_4 = _partition_batches(times32, dummy, batch_size=4)
    assert [len(b) for b in b32_4] == [4] * 8

    # 5 frames with batch_size=4 must split as [3, 2] (never [4, 1])
    b5_4 = _partition_batches(times32[:5], dummy[:5], batch_size=4)
    assert [len(b) for b in b5_4] == [3, 2]

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

    # Embodiment-agnostic wording preserved
    assert "human hand" in sys_p.lower()
    assert "robot arm" in sys_p.lower()

    # Batched multi-frame JSON schema + temporal identity lock + consistent semantic boundary + ref anchoring
    assert '"frames"' in sys_p and "frame_index" in sys_p and "time_s" in sys_p
    assert "Target Identity Lock" in sys_p
    assert "Consistent Box Boundary" in sys_p
    assert "breathing" in sys_p.lower() or "fluctuat" in sys_p.lower() or "expanding" in sys_p.lower()
    assert "ref_frame_0" in sys_p
    assert cfg.prompts.s1_grounding.version == "2.2.0"


def test_s1_multiframe_json_parsing_variants_and_missing_frames() -> None:
    dummy = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(4)]
    batch = [(10, 10.0, dummy[0]), (11, 12.5, dummy[1]), (12, 15.0, dummy[2]), (13, 17.5, dummy[3])]

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
            {"time_s": 15.02, "objects": {"cup": {"visible": True, "bbox": [0.2, 0.2, 0.4, 0.4]}}},
            {"time_s": 10.0, "objects": {"cup": {"visible": True, "bbox": [100, 100, 200, 200]}}},
        ]
    }
    parsed_kf = _parse_batch_response(resp_kf, batch)
    assert parsed_kf[10]["cup"]["bbox"] == [100, 100, 200, 200]
    assert parsed_kf[12]["cup"]["bbox"] == [0.2, 0.2, 0.4, 0.4]
    assert parsed_kf[11] == {} and parsed_kf[13] == {}

    # 2b. Duplicate/out-of-range explicit frame_index with valid time_s must NOT steal a different frame in Pass 2
    resp_dup_fi = {
        "frames": [
            {"frame_index": 0, "time_s": 10.0, "objects": {"cup": {"visible": True, "bbox": [100, 100, 200, 200]}}},
            {"frame_index": 0, "time_s": 12.5, "objects": {"cup": {"visible": True, "bbox": [999, 999, 999, 999]}}},
            {"frame_index": 99, "time_s": 15.0, "objects": {"cup": {"visible": True, "bbox": [888, 888, 888, 888]}}},
        ]
    }
    parsed_dup = _parse_batch_response(resp_dup_fi, batch)
    assert parsed_dup[10]["cup"]["bbox"] == [100, 100, 200, 200]
    assert parsed_dup[11] == {}
    assert parsed_dup[12] == {}

    # 2c. Legacy {"objects": ...} broadcast creates isolated per-frame dicts
    parsed_legacy = _parse_batch_response({"objects": {"cup": {"visible": True, "bbox": [10, 10, 50, 50]}}}, batch)
    assert parsed_legacy[10] is not parsed_legacy[11]

    # 2d. Reference-anchor prefix tolerance: skips accidental ref_frame_0 entry & corrects +1 index shift
    resp_with_ref_and_shift = {
        "frames": [
            {"frame_index": 0, "time_s": 0.0, "objects": {"cup": {"visible": True, "bbox": [999, 999, 999, 999]}}},
            {"frame_index": 1, "time_s": 10.0, "objects": {"cup": {"visible": True, "bbox": [100, 100, 200, 200]}}},
            {"frame_index": 2, "time_s": 12.5, "objects": {"cup": {"visible": True, "bbox": [200, 200, 300, 300]}}},
            {"frame_index": 3, "time_s": 15.0, "objects": {"cup": {"visible": True, "bbox": [300, 300, 400, 400]}}},
            {"frame_index": 4, "time_s": 17.5, "objects": {"cup": {"visible": True, "bbox": [400, 400, 500, 500]}}},
        ]
    }
    parsed_ref1 = _parse_batch_response(
        resp_with_ref_and_shift, batch, has_ref_prefix=True, ref_time_s=0.0
    )
    assert parsed_ref1[10]["cup"]["bbox"] == [100, 100, 200, 200]
    assert parsed_ref1[11]["cup"]["bbox"] == [200, 200, 300, 300]
    assert parsed_ref1[12]["cup"]["bbox"] == [300, 300, 400, 400]
    assert parsed_ref1[13]["cup"]["bbox"] == [400, 400, 500, 500]

    # 2e. Reference-anchor prefix skipped by model, but frame_index still +1 shifted (1..4) without timestamps
    resp_shifted_only = {
        "frames": [
            {"frame_index": 1, "objects": {"cup": {"visible": True, "bbox": [110, 110, 210, 210]}}},
            {"frame_index": 2, "objects": {"cup": {"visible": True, "bbox": [210, 210, 310, 310]}}},
            {"frame_index": 3, "objects": {"cup": {"visible": True, "bbox": [310, 310, 410, 410]}}},
            {"frame_index": 4, "objects": {"cup": {"visible": True, "bbox": [410, 410, 510, 510]}}},
        ]
    }
    parsed_ref2 = _parse_batch_response(
        resp_shifted_only, batch, has_ref_prefix=True, ref_time_s=0.0
    )
    assert parsed_ref2[10]["cup"]["bbox"] == [110, 110, 210, 210]
    assert parsed_ref2[13]["cup"]["bbox"] == [410, 410, 510, 510]

    # 2f. Reference-anchor explicit label="ref_frame_0" + compliant 0..3 indices
    resp_ref_label = {
        "frames": [
            {"label": "ref_frame_0", "frame_index": 0, "objects": {"cup": {"visible": True, "bbox": [999, 999, 999, 999]}}},
            {"frame_index": 0, "objects": {"cup": {"visible": True, "bbox": [120, 120, 220, 220]}}},
            {"frame_index": 1, "objects": {"cup": {"visible": True, "bbox": [220, 220, 320, 320]}}},
            {"frame_index": 2, "objects": {"cup": {"visible": True, "bbox": [320, 320, 420, 420]}}},
            {"frame_index": 3, "objects": {"cup": {"visible": True, "bbox": [420, 420, 520, 520]}}},
        ]
    }
    parsed_ref3 = _parse_batch_response(
        resp_ref_label, batch, has_ref_prefix=True, ref_time_s=0.0
    )
    assert parsed_ref3[10]["cup"]["bbox"] == [120, 120, 220, 220]
    assert parsed_ref3[13]["cup"]["bbox"] == [420, 420, 520, 520]

    # 2g. Reference-anchor entry present AND one real frame missing -> keeps missing slot empty
    resp_ref_missing = {
        "frames": [
            {"frame_index": 0, "time_s": 0.0, "objects": {"cup": {"visible": True, "bbox": [999, 999, 999, 999]}}},
            {"frame_index": 1, "time_s": 10.0, "objects": {"cup": {"visible": True, "bbox": [100, 100, 200, 200]}}},
            {"frame_index": 2, "time_s": 12.5, "objects": {"cup": {"visible": True, "bbox": [200, 200, 300, 300]}}},
            {"frame_index": 4, "time_s": 17.5, "objects": {"cup": {"visible": True, "bbox": [400, 400, 500, 500]}}},
        ]
    }
    parsed_ref4 = _parse_batch_response(
        resp_ref_missing, batch, has_ref_prefix=True, ref_time_s=0.0
    )
    assert parsed_ref4[10]["cup"]["bbox"] == [100, 100, 200, 200]
    assert parsed_ref4[11]["cup"]["bbox"] == [200, 200, 300, 300]
    assert parsed_ref4[12] == {}
    assert parsed_ref4[13]["cup"]["bbox"] == [400, 400, 500, 500]

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


def test_s1_cross_batch_static_consensus_defeats_poisoned_batch0_and_phantom_retry() -> None:
    # 1. Poisoned Batch 0 static container MUST NOT override consensus of Batches 1..N
    cfg = ConfigManager()
    cfg.settings["vlm"]["s1_grounding"]["batch_size"] = 4
    cfg.settings["vlm"]["s1_grounding"]["reference_anchor_enabled"] = True
    cfg.settings["vlm"]["s1_grounding"]["tracking"]["enabled"] = False

    def _poisoned_b0_handler(kw: dict[str, Any]) -> dict[str, Any]:
        ts = list(kw["timestamps_s"])
        labels = list(kw.get("labels") or [])
        has_ref = bool(labels and labels[0] == "ref_frame_0")
        target_ts = ts[1:] if has_ref else ts
        # Batch 0 returns hallucinated coordinates [182, 458, 412, 736]; Batch 1..N return true [397, 228, 554, 483]
        cm_box = [397, 228, 554, 483] if has_ref else [182, 458, 412, 736]
        return {
            "frames": [
                {
                    "frame_index": idx,
                    "time_s": float(t),
                    "objects": {
                        "coffee_machine": {"visible": True, "bbox": cm_box, "state": "stationary"},
                        "cup": {"visible": True, "bbox": [390, 320, 415, 420], "state": "ok"},
                    },
                }
                for idx, t in enumerate(target_ts)
            ]
        }

    cl = _FakeClient(_poisoned_b0_handler)
    res = run_stage1_grounding(
        view=_make_synthetic_view(duration_s=60.0),
        task=_coffee_task(),
        client=cl,  # type: ignore[arg-type]
        cfg=cfg,
    )
    cm_kfs = res.object_tracks_2d["coffee_machine"].keyframes
    assert all(k.visible and k.bbox_norm_1000 == [397, 228, 554, 483] for k in cm_kfs)

    # 2. Textured static fixed-camera clip: Batch 0 phantom fly-across triggers automatic retry
    rng = np.random.default_rng(707)
    static_rgb = rng.integers(30, 220, size=(360, 640, 3), dtype=np.uint8)

    class _StaticTexturedMedia:
        duration_s = 20.0
        fps = 30.0
        cameras = [CameraSpec(name="ego", role="main", width=640, height=360, fps=30.0)]

        def read_frames_batch(self, camera: str, times_s: list[float]) -> list[np.ndarray]:
            return [static_rgb.copy() for _ in times_s]

    meta = EpisodeMeta(
        episode_id="ds/ep_phantom",
        dataset="ds",
        stem="ep_phantom",
        task_id="coffee_barista",
        adapter_type="video_dir",
        source_uri="/tmp/phantom",
        source_mtime=1.0,
        episode_fingerprint="fp_phantom",
        duration_s=20.0,
        fps=30.0,
        frame_count=600,
        cameras=[CameraSpec(name="ego", role="main", width=640, height=360, fps=30.0)],
        camera_roles={"main": "ego"},
        kinematics=KinematicsSpec(has_kinematics=False),
    )
    view_tex = EpisodeView(meta=meta, media=_StaticTexturedMedia())  # type: ignore[arg-type]

    b0_attempts = {"n": 0}

    def _phantom_then_good(kw: dict[str, Any]) -> dict[str, Any]:
        ts = list(kw["timestamps_s"])
        labels = list(kw.get("labels") or [])
        has_ref = bool(labels and labels[0] == "ref_frame_0")
        target_ts = ts[1:] if has_ref else ts
        if not has_ref:
            b0_attempts["n"] += 1
            if b0_attempts["n"] == 1:
                # Blind-node hallucination: cup teleports x=150 -> x=750 on completely static pixels!
                return {
                    "frames": [
                        {
                            "frame_index": idx,
                            "time_s": float(t),
                            "objects": {
                                "coffee_machine": {"visible": True, "bbox": [180, 450, 410, 740]},
                                "cup": {
                                    "visible": True,
                                    "bbox": [150 + idx * 300, 300, 210 + idx * 300, 420],
                                },
                            },
                        }
                        for idx, t in enumerate(target_ts)
                    ]
                }
        return {
            "frames": [
                {
                    "frame_index": idx,
                    "time_s": float(t),
                    "objects": {
                        "coffee_machine": {"visible": True, "bbox": [397, 228, 554, 483]},
                        "cup": {"visible": True, "bbox": [392, 320, 415, 420]},
                    },
                }
                for idx, t in enumerate(target_ts)
            ]
        }

    cl_ph = _FakeClient(_phantom_then_good)
    res_ph = run_stage1_grounding(
        view=view_tex,
        task=_coffee_task(),
        client=cl_ph,  # type: ignore[arg-type]
        cfg=cfg,
    )
    assert b0_attempts["n"] == 2  # Batch 0 auto-retried once after phantom-motion detection!
    assert all(
        k.visible and k.bbox_norm_1000 == [397, 228, 554, 483]
        for k in res_ph.object_tracks_2d["coffee_machine"].keyframes
    )
    assert all(
        k.visible and k.bbox_norm_1000 == [392, 320, 415, 420]
        for k in res_ph.object_tracks_2d["cup"].keyframes
    )

    # 3. Transient VlmBlindVisionError during parallel wave recovers cleanly via quiet sequential retry
    blind_once = {"done": False}
    lock = threading.Lock()

    def _blind_then_ok(kw: dict[str, Any]) -> dict[str, Any]:
        ts = list(kw["timestamps_s"])
        labels = list(kw.get("labels") or [])
        has_ref = bool(labels and labels[0] == "ref_frame_0")
        target_ts = ts[1:] if has_ref else ts
        if has_ref:
            with lock:
                if not blind_once["done"]:
                    blind_once["done"] = True
                    raise VlmBlindVisionError("simulated burst blind exhaustion")
        return {
            "frames": [
                {
                    "frame_index": idx,
                    "time_s": float(t),
                    "objects": {
                        "coffee_machine": {"visible": True, "bbox": [397, 228, 554, 483]},
                        "cup": {"visible": True, "bbox": [392, 320, 415, 420]},
                    },
                }
                for idx, t in enumerate(target_ts)
            ]
        }

    cl_b = _FakeClient(_blind_then_ok)
    res_b = run_stage1_grounding(
        view=_make_synthetic_view(duration_s=20.0),
        task=_coffee_task(),
        client=cl_b,  # type: ignore[arg-type]
        cfg=cfg,
    )
    assert blind_once["done"] is True
    assert len(res_b.object_tracks_2d["cup"].keyframes) == 9
    assert all(k.visible for k in res_b.object_tracks_2d["cup"].keyframes)
