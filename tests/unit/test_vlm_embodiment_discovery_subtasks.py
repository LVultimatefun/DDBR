"""Unit tests for embodiment-agnostic prompts, MacroStage.instruction, S1 Pass-0 discovery + custom_objects, and S1->S2 context/subtask inference."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from robo_vlm_qc.config import ConfigManager
from robo_vlm_qc.io import auto_detect, get_adapter
from robo_vlm_qc.io.base import EpisodeView
from robo_vlm_qc.models.annotation import (
    GroundingKeyframe,
    GroundingResult,
    GroundingTrack,
    MacroStage,
    QcSegment,
    VisualEvidence,
)
from robo_vlm_qc.models.episode import CameraSpec, EpisodeMeta, KinematicsSpec
from robo_vlm_qc.models.taxonomy import TaskSpec
from robo_vlm_qc.store import Repository
from robo_vlm_qc.vlm.client import BudgetTracker, VlmCancelledError
from robo_vlm_qc.vlm.dag import VlmDag
from robo_vlm_qc.vlm.prompt_builder import (
    PromptBuilder,
    format_grounding_summary,
    format_qc_segments_summary,
)
from robo_vlm_qc.vlm.stage1_grounding import (
    build_custom_grounding_objects,
    run_stage1_grounding,
    slugify_object_id,
)
from robo_vlm_qc.vlm.stage2_macro import (
    infer_subtasks_from_segments,
    normalize_macro_stages,
    run_stage2_macro,
)


class _MemMedia:
    def __init__(self, duration_s: float = 20.0, fps: float = 15.0, w: int = 320, h: int = 180) -> None:
        self.duration_s = duration_s
        self.fps = fps
        self.cameras = [CameraSpec(name="main", role="main", width=w, height=h, fps=fps)]
        self._w = w
        self._h = h

    def read_frame_rgb(self, camera: str, t_s: float) -> np.ndarray:
        return np.full((self._h, self._w, 3), 80, dtype=np.uint8)

    def read_frames_batch(self, camera: str, times_s: list[float]) -> list[np.ndarray]:
        return [np.full((self._h, self._w, 3), 80, dtype=np.uint8) for _ in times_s]


def _make_view(duration_s: float = 20.0, task_id: str = "_default") -> EpisodeView:
    meta = EpisodeMeta(
        episode_id="ego_ds/ep_001",
        dataset="ego_ds",
        stem="ep_001",
        task_id=task_id,
        adapter_type="video_dir",
        source_uri="/tmp/ego_ep",
        source_mtime=1.0,
        episode_fingerprint=f"fp_ego_{duration_s}_{task_id}",
        duration_s=duration_s,
        fps=15.0,
        frame_count=max(1, int(round(duration_s * 15.0))),
        cameras=[CameraSpec(name="main", role="main", width=320, height=180, fps=15.0)],
        camera_roles={"main": "main"},
        kinematics=KinematicsSpec(has_kinematics=False),
    )
    return EpisodeView(meta=meta, media=_MemMedia(duration_s=duration_s))  # type: ignore[arg-type]


class _FakeVlm:
    def __init__(self, handler: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self.handler = handler
        self.calls: list[dict[str, Any]] = []
        self.budget = BudgetTracker()
        self._lock = threading.Lock()

    def is_cancelled(self) -> bool:
        return False

    def model_for_stage(self, stage: str, override: str | None = None) -> str:
        return override or "fake-gpt-4o"

    def chat_json(self, **kwargs: Any) -> dict[str, Any]:
        with self._lock:
            self.calls.append(kwargs)
        if kwargs.get("budget") is not None:
            kwargs["budget"].record(
                stage=kwargs.get("stage", "unknown"),
                model="fake-gpt-4o",
                prompt_tokens=20,
                completion_tokens=15,
                total_tokens=35,
            )
        return self.handler(kwargs)


def test_embodiment_agnostic_default_task_and_all_prompts() -> None:
    cfg = ConfigManager()
    default_task = cfg.get_task("_default")
    joined_task = " ".join(
        [default_task.goal, *default_task.criteria, *default_task.steps, default_task.task_prompt]
    ).lower()
    assert "human hand" in joined_task
    assert "robot arm" in joined_task

    pb = PromptBuilder(cfg)
    s1_sys, s1_usr = pb.build_s1(task=default_task, camera_roles={"main": "main"})
    s1d_sys, s1d_usr = pb.build_s1_discovery(task=default_task, camera_roles={"main": "main"})
    s2_sys, s2_usr = pb.build_s2(task=default_task, camera_roles={"main": "main"}, duration_s=10.0)
    s3_sys, _ = pb.build_s3(task=default_task, camera_roles={"main": "main"}, duration_s=10.0)
    s4a_sys, _ = pb.build_s4a(task=default_task, duration_s=10.0)
    s4b_sys, _ = pb.build_s4b(
        task=default_task,
        camera_roles={"main": "main"},
        candidate=QcSegment(
            seg_id="c1",
            t_start=0.0,
            t_end=1.0,
            label_id="boundary_idle",
            training_role="badcase",
            splice_policy="cut_clean",
            vlm_raw_policy="cut_clean",
        ),
        window_start_s=0.0,
        window_end_s=2.0,
    )
    s5_sys, _ = pb.build_s5(task=default_task, camera_roles={"main": "main"}, duration_s=10.0)

    for text in (s1_sys, s1d_sys, s2_sys, s3_sys, s4a_sys, s4b_sys, s5_sys):
        low = text.lower()
        assert "human hand" in low, f"Missing human hand compatibility in: {text[:160]}"
        assert "robot arm" in low, f"Missing robot arm compatibility in: {text[:160]}"

    # Preserve stage-classifier & schema anchors
    assert "2D Grounding" in s1_sys and "2D Grounding" in s1d_sys
    assert "Target objects to ground" in s1_usr
    assert "canonical_objects" in s1d_sys and s1d_usr
    assert "macro_stages" in s2_sys and "arm_timelines" in s2_sys and '"instruction"' in s2_sys
    assert "Precision-First" in s3_sys and "unknown_note" in s3_sys
    assert "Stage 4A" in s4a_sys and "merged_segments" in s4a_sys
    assert "Stage 4B" in s4b_sys
    assert "satisfied_at_s" in s5_sys


def test_macro_stage_instruction_backward_compat_and_normalization() -> None:
    # Default empty string when omitted (backward compatibility)
    st_legacy = MacroStage.model_validate(
        {"stage_id": "S1", "t_start": 0.0, "t_end": 2.5, "label": "Pick up mug"}
    )
    assert st_legacy.instruction == ""

    # Explicit instruction preserved & extra="forbid" still active
    st_with_instr = MacroStage(
        stage_id="S1",
        t_start=0.0,
        t_end=2.5,
        label="Pick up mug",
        instruction="Pick up the green mug from the counter",
    )
    assert st_with_instr.instruction == "Pick up the green mug from the counter"
    with pytest.raises(ValidationError):
        MacroStage.model_validate(
            {
                "stage_id": "S1",
                "t_start": 0.0,
                "t_end": 2.5,
                "label": "Pick up mug",
                "unexpected_extra": 123,
            }
        )

    # normalize_macro_stages parses instruction & falls back gracefully when omitted/None
    norm = normalize_macro_stages(
        [
            {
                "stage_id": "S1",
                "t_start": 0.0,
                "t_end": 2.0,
                "label": "Grasp mug",
                "instruction": "  Pick up the green mug  ",
            },
            {
                "stage_id": "S2",
                "t_start": 2.0,
                "t_end": 5.0,
                "label": "Place in rack",
                "instruction": None,
            },
            {
                "stage_id": "S3",
                "t_start": 5.0,
                "t_end": 8.0,
                "label": "Release",
                "subtask_instruction": "Release the mug inside the dish rack",
            },
        ],
        duration_s=8.0,
    )
    assert [s.instruction for s in norm] == [
        "Pick up the green mug",
        "",
        "Release the mug inside the dish rack",
    ]


def test_s1_pass0_canonical_discovery_locks_ids_across_batches_and_falls_back_gracefully() -> None:
    cfg = ConfigManager()
    cfg.settings["vlm"]["s1_grounding"]["tracking"]["enabled"] = False
    cfg.settings["vlm"]["s1_grounding"]["batch_size"] = 4
    cfg.settings["vlm"]["s1_grounding"]["interval_s"] = 2.5

    # Case 1: Real Pass-0 discovery locks 2 canonical objects and injects them into ALL S1 batches
    batch_user_prompts: list[str] = []

    def _pass0_handler(kw: dict[str, Any]) -> dict[str, Any]:
        if "canonical_objects" in kw["system_prompt"]:
            assert 4 <= len(kw["images"]) <= 6
            return {
                "canonical_objects": [
                    {
                        "id": "green_mug",
                        "name": "Green Mug",
                        "desc": "Green ceramic mug grasped by hand",
                        "is_static": False,
                    },
                    {
                        "id": "dish_rack",
                        "name": "Dish Rack",
                        "desc": "Countertop drying rack",
                        "is_static": True,
                    },
                ]
            }
        batch_user_prompts.append(kw["user_prompt"])
        ts = list(kw["timestamps_s"])
        return {
            "frames": [
                {
                    "frame_index": i,
                    "time_s": float(t),
                    "objects": {
                        "green_mug": {
                            "visible": True,
                            "bbox": [120 + i * 10, 200, 240 + i * 10, 360],
                            "state": "held" if t >= 5.0 else "on_counter",
                            "status_kind": "neutral",
                        },
                        "dish_rack": {
                            "visible": True,
                            "bbox": [600, 250, 880, 560],
                            "state": "stationary",
                            "status_kind": "good",
                        },
                    },
                }
                for i, t in enumerate(ts)
            ]
        }

    cl = _FakeVlm(_pass0_handler)
    res = run_stage1_grounding(
        view=_make_view(duration_s=20.0, task_id="_default"),
        task=cfg.get_task("_default"),
        client=cl,  # type: ignore[arg-type]
        cfg=cfg,
    )
    # 9 keyframes with batch_size=4 => 3 batches + 1 Pass-0 discovery = 4 calls total
    assert len(cl.calls) == 4
    assert len(batch_user_prompts) == 3
    for prompt in batch_user_prompts:
        assert "green_mug" in prompt
        assert "dish_rack" in prompt
    assert set(res.object_tracks_2d.keys()) == {"green_mug", "dish_rack"}
    assert res.object_tracks_2d["green_mug"].name == "Green Mug"
    assert res.object_tracks_2d["dish_rack"].is_static_container is True

    # Case 2: Legacy mock returning {"objects": {...}} on Pass-0 falls back seamlessly
    cl_legacy = _FakeVlm(
        lambda _kw: {
            "objects": {
                "target_object": {"visible": True, "bbox": [100, 100, 250, 280], "state": "ok"}
            }
        }
    )
    res_legacy = run_stage1_grounding(
        view=_make_view(duration_s=20.0, task_id="_default"),
        task=cfg.get_task("_default"),
        client=cl_legacy,  # type: ignore[arg-type]
        cfg=cfg,
    )
    assert "target_object" in res_legacy.object_tracks_2d
    assert len(res_legacy.object_tracks_2d["target_object"].keyframes) == 9


def test_s1_custom_objects_slugify_override_and_dag_cache_differentiation(tmp_path: Path) -> None:
    assert slugify_object_id("Green Mug!") == "green_mug"
    assert slugify_object_id("  Dish-Rack #2 ") == "dish_rack_2"
    specs = build_custom_grounding_objects(["Green Mug", "  ", "green mug", "Dish Rack"])
    assert [(s.id, s.name) for s in specs] == [
        ("green_mug", "Green Mug"),
        ("dish_rack", "Dish Rack"),
    ]

    # Build a short real mp4 so VlmDag can run end-to-end
    ds_dir = tmp_path / "data" / "custom_ds" / "ep_0000"
    ds_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(ds_dir / "main.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (160, 120))
    for i in range(45):
        writer.write(np.full((120, 160, 3), (40 + i % 30, 80, 110), dtype=np.uint8))
    writer.release()
    handle = auto_detect(tmp_path / "data" / "custom_ds", adapter_name="video_dir")[0]

    ws = tmp_path / "ws"
    cfg = ConfigManager(workspace_dir=ws)
    cfg.settings["vlm"]["s1_grounding"]["tracking"]["enabled"] = False
    repo = Repository(ws)
    s1_prompts_seen: list[str] = []
    s2_prompts_seen: list[str] = []

    def _dag_handler(kw: dict[str, Any]) -> dict[str, Any]:
        st = kw["stage"]
        if st == "s1_grounding":
            s1_prompts_seen.append(kw["user_prompt"])
            if "green_mug" in kw["user_prompt"]:
                return {
                    "objects": {
                        "green_mug": {
                            "visible": True,
                            "bbox": [100, 150, 260, 350],
                            "state": "grasped_by_hand",
                            "status_kind": "good",
                        }
                    }
                }
            return {
                "objects": {
                    "target_object": {
                        "visible": True,
                        "bbox": [200, 200, 400, 420],
                        "state": "on_table",
                        "status_kind": "neutral",
                    }
                }
            }
        if st == "s2_macro":
            s2_prompts_seen.append(kw["user_prompt"])
            return {
                "macro_stages": [
                    {
                        "stage_id": "S1",
                        "t_start": 0.0,
                        "t_end": 3.0,
                        "label": "Grasp mug",
                        "instruction": "Pick up the green mug",
                    }
                ],
                "arm_timelines": {},
            }
        if st == "s3_discover":
            return {"segments": []}
        if st.startswith("s5"):
            return {"verdict": "failure", "confidence": 0.8, "criteria": []}
        return {"merged_segments": []}

    try:
        with get_adapter("video_dir").open(handle) as view:
            repo.ingest_episode(view.meta)
            cl = _FakeVlm(_dag_handler)
            dag = VlmDag(workspace=ws, config=cfg, client=cl, repo=repo)  # type: ignore[arg-type]

            # 1. Default run -> S1 runs, S2 receives S1 grounding context summary
            art_default = dag.run_episode(view)
            assert "target_object" in art_default.grounding.object_tracks_2d
            assert art_default.macro_stages[0].instruction == "Pick up the green mug"
            assert any("target_object" in p and "on_table" in p for p in s2_prompts_seen)

            # 2. Run with custom_objects -> DIFFERENT S1 cache key, Pass-0 skipped, S2/S5 hit cache
            s1_prompts_seen.clear()
            s2_prompts_seen.clear()
            art_custom = dag.run_episode(view, custom_objects=["Green Mug", "Dish Rack"])
            assert art_custom.stage_refs.s1_grounding != art_default.stage_refs.s1_grounding
            assert art_custom.stage_refs.s2_macro == art_default.stage_refs.s2_macro
            assert "green_mug" in art_custom.grounding.object_tracks_2d
            assert art_custom.grounding.object_tracks_2d["green_mug"].name == "Green Mug"
            assert len(s1_prompts_seen) == 1  # No Pass-0 discovery when custom_objects provided
            assert "green_mug" in s1_prompts_seen[0] and "dish_rack" in s1_prompts_seen[0]
            assert s2_prompts_seen == []  # S2 cache reused cleanly!

            # 3. Repeating the exact same custom_objects hits S1 cache (0 new S1 calls)
            s1_prompts_seen.clear()
            art_custom_hit = dag.run_episode(view, custom_objects=["Green Mug", "Dish Rack"])
            assert art_custom_hit.stage_refs.s1_grounding == art_custom.stage_refs.s1_grounding
            assert s1_prompts_seen == []
    finally:
        repo.close()


def test_s1_to_s2_grounding_summary_and_infer_subtasks_from_segments() -> None:
    cfg = ConfigManager()
    grounding = GroundingResult(
        schema_version=1,
        keyframes_s=[0.0, 2.5, 5.0],
        object_tracks_2d={
            "green_mug": GroundingTrack(
                object_id="green_mug",
                name="Green Mug",
                keyframes=[
                    GroundingKeyframe(row=0, time_s=0.0, visible=True, bbox_norm_1000=[100, 400, 200, 550], state="on counter"),
                    GroundingKeyframe(row=38, time_s=2.5, visible=True, bbox_norm_1000=[350, 250, 450, 400], state="held by hand"),
                    GroundingKeyframe(row=75, time_s=5.0, visible=True, bbox_norm_1000=[650, 380, 750, 530], state="in dish rack"),
                ],
            )
        },
    )
    summary = format_grounding_summary(grounding)
    assert "green_mug (Green Mug)" in summary
    assert "on counter" in summary and "held by hand" in summary and "in dish rack" in summary

    qc_segs = [
        QcSegment(
            seg_id="cut_0",
            t_start=0.0,
            t_end=1.2,
            label_id="boundary_idle",
            training_role="badcase",
            splice_policy="cut_clean",
            vlm_raw_policy="cut_clean",
        ),
        QcSegment(
            seg_id="rec_1",
            t_start=2.0,
            t_end=2.8,
            label_id="missed_grasp_retry",
            training_role="recovery",
            splice_policy="mask_loss",
            mask_side="right",
            affected_arm="right",
            visual_evidence=VisualEvidence(process_change="Hand missed mug handle and re-grasped"),
            vlm_raw_policy="mask_loss",
        ),
    ]
    qc_summary = format_qc_segments_summary(qc_segs, duration_s=5.0)
    assert "CUT_CLEAN EXCLUDED" in qc_summary
    assert "RECOVERY" in qc_summary
    assert "[1.20s..5.00s]" in qc_summary
    assert "unknown_note" not in qc_summary

    seen_calls: list[dict[str, Any]] = []

    def _subtask_handler(kw: dict[str, Any]) -> dict[str, Any]:
        seen_calls.append(kw)
        return {
            "macro_stages": [
                {
                    "stage_id": "S1",
                    "t_start": 0.0,
                    "t_end": 1.2,
                    "label": "Initial idle",
                    "label_en": "initial_idle",
                    "instruction": "Wait before starting",
                    "summary": "Excluded leading idle",
                },
                {
                    "stage_id": "S2",
                    "t_start": 1.2,
                    "t_end": 3.4,
                    "label": "Pick up green mug",
                    "label_en": "pick_up_mug",
                    "instruction": "Pick up the green mug from the counter",
                    "summary": "Grasp and lift green mug",
                },
                {
                    "stage_id": "S3",
                    "t_start": 3.4,
                    "t_end": 5.0,
                    "label": "Place in dish rack",
                    "label_en": "place_in_rack",
                    "instruction": "Place the green mug into the dish rack",
                    "summary": "Place and release mug",
                },
            ]
        }

    cl = _FakeVlm(_subtask_handler)
    stages = infer_subtasks_from_segments(
        view=_make_view(duration_s=5.0),
        qc_segments=qc_segs,
        macro_stages=[MacroStage(stage_id="S1", t_start=0.0, t_end=5.0, label="Old draft")],
        grounding=grounding,
        custom_hint="Keep the grasp retry attached to the pick stage",
        client=cl,  # type: ignore[arg-type]
        cfg=cfg,
    )
    assert len(seen_calls) == 1
    up = seen_calls[0]["user_prompt"]
    assert "CUT_CLEAN EXCLUDED" in up
    assert "RECOVERY" in up
    assert "green_mug (Green Mug)" in up
    assert "Old draft" in up
    assert "Keep the grasp retry attached to the pick stage" in up
    # Boundary timestamp 1.2s from cut_clean was sampled into frames
    assert 1.2 in seen_calls[0]["timestamps_s"]
    assert len(stages) == 3
    assert stages[1].t_start == 1.2
    assert stages[1].instruction == "Pick up the green mug from the counter"
    assert stages[2].instruction == "Place the green mug into the dish rack"

    # Also verify direct run_stage2_macro injects grounding summary when provided
    s2_calls: list[dict[str, Any]] = []
    cl2 = _FakeVlm(lambda kw: (s2_calls.append(kw) or {"macro_stages": [], "arm_timelines": {}}))
    run_stage2_macro(
        view=_make_view(duration_s=5.0),
        client=cl2,  # type: ignore[arg-type]
        cfg=cfg,
        grounding=grounding,
    )
    assert "green_mug (Green Mug)" in s2_calls[0]["user_prompt"]
