"""Unit and integration tests for PromptBuilder, Stages 1-5, and VlmDag."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from robo_vlm_qc.config import ConfigManager
from robo_vlm_qc.io import auto_detect, get_adapter
from robo_vlm_qc.models.annotation import MacroStage, QcSegment
from robo_vlm_qc.models.review import EpisodeStatus
from robo_vlm_qc.models.taxonomy import FewShotItem
from robo_vlm_qc.store import Repository
from robo_vlm_qc.vlm.client import VlmError
from robo_vlm_qc.vlm.dag import VlmDag
from robo_vlm_qc.vlm.prompt_builder import PromptBuilder, describe_camera_layout
from robo_vlm_qc.vlm.stage1_grounding import run_stage1_grounding
from robo_vlm_qc.vlm.stage2_macro import normalize_contiguous_timeline, normalize_macro_stages
from robo_vlm_qc.vlm.stage4_verify import (
    _deterministic_merge_raw_segments,
    run_stage4a_merge,
    run_stage4b_verify,
)
from robo_vlm_qc.vlm.stage5_success import run_stage5_success


class _FakeClient:
    """In-memory deterministic VlmClient double for stage unit tests."""

    def __init__(self, handler: Any) -> None:
        self.handler = handler
        self.calls: list[dict[str, Any]] = []
        from robo_vlm_qc.vlm.client import BudgetTracker

        self.budget = BudgetTracker()

    def model_for_stage(self, stage: str, override: str | None = None) -> str:
        return override or "fake-gpt-4o"

    def chat_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if kwargs.get("budget") is not None:
            kwargs["budget"].record(
                stage=kwargs.get("stage", "unknown"),
                model="fake-gpt-4o",
                prompt_tokens=50,
                completion_tokens=30,
                total_tokens=80,
            )
        return self.handler(kwargs)


def _make_video_ep(root: Path, duration_s: float = 4.0) -> Path:
    ep_dir = root / "pick_place_ds" / "ep_0000"
    ep_dir.mkdir(parents=True, exist_ok=True)
    n = int(round(duration_s * 30.0))
    for cam in ("main", "left_wrist", "right_wrist"):
        w = cv2.VideoWriter(str(ep_dir / f"{cam}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
        for i in range(n):
            fr = np.full((240, 320, 3), (30 + (i * 2) % 60, 60, 120), dtype=np.uint8)
            w.write(fr)
        w.release()
    return root / "pick_place_ds"


def test_prompt_builder_layouts_inactive_labels_and_few_shot_truncation() -> None:
    cfg = ConfigManager()
    assert "L-WRIST" in describe_camera_layout({"main": "ego", "left_wrist": "lw"})
    assert "TORSO" in describe_camera_layout({"main": "head_rgbd", "front": "torso_rgbd"})
    assert "MAIN-CENTER-ZOOM" in describe_camera_layout({"main": "main"})

    pb = PromptBuilder(cfg)
    raw = cfg.labels.model_dump(mode="json")
    for lbl in raw["labels"]:
        if lbl["id"] == "hesitation_oscillation":
            lbl["status"] = "retired"
    from robo_vlm_qc.models.taxonomy import LabelRegistry

    pb_retired = PromptBuilder(cfg, labels=LabelRegistry.model_validate(raw))
    _, usr_s3 = pb_retired.build_s3(
        task=cfg.get_task("pick_and_place"),
        camera_roles={"main": "head_rgbd", "front": "torso_rgbd"},
        macro_stages=[MacroStage(stage_id="S1", t_start=0.0, t_end=4.0, label="Grasp object")],
        window_start_s=0.0,
        window_end_s=4.0,
        duration_s=4.0,
    )
    assert "hesitation_oscillation" not in usr_s3
    assert "Grasp object" in usr_s3

    # Token truncation check
    fs_items = [
        FewShotItem(
            example_id=f"fs_{i}",
            scope="global",
            error_type="fp_overflag",
            episode_id="ds/ep",
            t_start=0.5,
            t_end=1.0,
            vlm_wrong_json={"label_id": "mid_idle_freeze"},
            human_right_json={"splice_policy": "keep_positive"},
            rationale=f"MARKER_{i}_" + ("a" * 300),
            created_at=float(100 + i),
        )
        for i in range(4)
    ]
    cfg.settings["vlm"]["s3_discover"]["few_shot_max_tokens"] = 180
    _, usr_fs = pb.build_s3(
        task=cfg.get_task("pick_and_place"),
        camera_roles={"main": "main"},
        macro_stages=[],
        window_start_s=0.0,
        window_end_s=4.0,
        duration_s=4.0,
        few_shots=fs_items,
    )
    included = sum(1 for i in range(4) if f"MARKER_{i}_" in usr_fs)
    assert 0 <= included < 4


def test_stage1_normalization_and_static_container_stabilization(tmp_path: Path) -> None:
    ds_dir = _make_video_ep(tmp_path, duration_s=12.0)
    handle = auto_detect(ds_dir, adapter_name="video_dir")[0]
    cfg = ConfigManager()

    idx_box = {"n": 0}
    boxes = [
        [0.40, 0.20, 0.60, 0.50],         # [0,1] float format
        [410, 205, 610, 505],             # [0,1000] int
        [512, 144, 768, 360],             # [0,1000] int
        [1050, 220, 1200, 500],           # >1000 pixel coords
        [402, 202, 602, 502],
        [398, 198, 598, 498],
    ]

    def _handler(_kw: dict[str, Any]) -> dict[str, Any]:
        i = idx_box["n"]
        idx_box["n"] += 1
        b = boxes[i % len(boxes)]
        return {
            "objects": {
                "bin": {"visible": True, "bbox": b, "state": "ok", "status_kind": "neutral"},
                "tiny_noise": {"visible": True, "bbox": [100, 100, 104, 104], "state": "noise", "status_kind": "neutral"},
            }
        }

    with get_adapter("video_dir").open(handle) as view:
        res = run_stage1_grounding(
            view=view,
            task=cfg.get_task("pick_and_place"),
            client=_FakeClient(_handler),  # type: ignore[arg-type]
            cfg=cfg,
        )
    assert "tiny_noise" not in res.object_tracks_2d
    container = res.object_tracks_2d["bin"]
    assert container.is_static_container is True
    vis_boxes = {tuple(k.bbox_norm_1000) for k in container.keyframes if k.visible}
    assert len(vis_boxes) == 1


def test_stage2_normalization_fallbacks() -> None:
    stages = normalize_macro_stages([], 5.0)
    assert len(stages) == 1 and stages[0].t_start == 0.0 and stages[0].t_end == 5.0

    stages2 = normalize_macro_stages(
        [{"stage_id": "X", "t_start": -1.0, "t_end": 2.0, "label": "Stage A"},
         {"stage_id": "Y", "t_start": 3.0, "t_end": 99.0, "label": "Stage B"}],
        5.0,
    )
    assert stages2[0].t_start == 0.0 and stages2[-1].t_end == 5.0
    assert stages2[0].t_end == stages2[1].t_start

    tl = normalize_contiguous_timeline([{"t_start": 1.0, "t_end": 3.0, "state": "work"}], 5.0)
    assert tl[0].t_start == 0.0 and tl[0].state == "idle"
    assert tl[1].state == "work"
    assert tl[-1].t_end == 5.0 and tl[-1].state == "idle"


def test_stage5_anti_fp_control_withdrawal(tmp_path: Path) -> None:
    ds_dir = _make_video_ep(tmp_path, duration_s=4.0)
    handle = auto_detect(ds_dir, adapter_name="video_dir")[0]
    cfg = ConfigManager()

    def _handler(kw: dict[str, Any]) -> dict[str, Any]:
        return {
            "verdict": "success",
            "confidence": 0.95,
            "criteria": [{"criterion": "done", "met": True, "satisfied_at_s": 0.5, "evidence": "ok"}],
            "summary": "claimed success",
        }

    with get_adapter("video_dir").open(handle) as view:
        verdict = run_stage5_success(
            view=view,
            task=cfg.get_task("pick_and_place"),
            client=_FakeClient(_handler),  # type: ignore[arg-type]
            cfg=cfg,
        )
    assert verdict.verdict == "unsure"
    assert verdict.passed is None
    assert verdict.confidence == 0.0
    assert verdict.control_check is not None and verdict.control_check.passed is False


def test_stage4_deterministic_fallback_and_overlap_split(tmp_path: Path) -> None:
    ds_dir = _make_video_ep(tmp_path, duration_s=4.0)
    handle = auto_detect(ds_dir, adapter_name="video_dir")[0]
    cfg = ConfigManager()

    cands = [
        QcSegment(seg_id="c1", t_start=0.5, t_end=1.3, label_id="missed_grasp_retry", training_role="recovery", splice_policy="mask_loss", mask_side="left", affected_arm="left", vlm_raw_policy="mask_loss"),
        QcSegment(seg_id="c2", t_start=0.7, t_end=1.5, label_id="missed_grasp_retry", training_role="recovery", splice_policy="mask_loss", mask_side="left", affected_arm="left", vlm_raw_policy="mask_loss"),
        QcSegment(seg_id="c3", t_start=2.5, t_end=3.2, label_id="missed_grasp_retry", training_role="recovery", splice_policy="mask_loss", mask_side="left", affected_arm="left", vlm_raw_policy="mask_loss"),
    ]
    det = _deterministic_merge_raw_segments(cands)
    assert len(det) == 2

    def _fail_s4a(_kw: dict[str, Any]) -> dict[str, Any]:
        raise VlmError("simulated S4A error")

    with get_adapter("video_dir").open(handle) as view:
        merged = run_stage4a_merge(
            view=view, candidates=cands, client=_FakeClient(_fail_s4a), cfg=cfg  # type: ignore[arg-type]
        )
        assert len(merged) == 2

        def _s4b_handler(kw: dict[str, Any]) -> dict[str, Any]:
            up = kw["user_prompt"]
            if "c1" in up:
                return {"keep_segment": False}
            if "c2" in up:
                return {
                    "keep_segment": True,
                    "t_start": 1.0,
                    "t_end": 2.2,
                    "label_id": "missed_grasp_retry",
                    "affected_arm": "left",
                    "splice_policy": "mask_loss",
                    "visual_evidence": {"initial_state": "a", "process_change": "b", "end_state": "c"},
                }
            return {
                "keep_segment": True,
                "t_start": 1.8,
                "t_end": 3.0,
                "label_id": "placement_redo",
                "affected_arm": "right",
                "splice_policy": "mask_loss",
                "visual_evidence": {"initial_state": "d", "process_change": "e", "end_state": "f"},
            }

        three = [
            cands[0],
            cands[1].model_copy(update={"seg_id": "c2", "t_start": 1.0, "t_end": 2.0}),
            cands[2],
        ]
        verified = run_stage4b_verify(
            view=view, candidates=three, client=_FakeClient(_s4b_handler), cfg=cfg  # type: ignore[arg-type]
        )
        assert len(verified) == 2
        assert verified[0].t_end == verified[1].t_start == 2.0


def test_dag_few_shot_persistence_and_failure_rollback(tmp_path: Path) -> None:
    ds_dir = _make_video_ep(tmp_path / "data", duration_s=2.0)
    handle = auto_detect(ds_dir, adapter_name="video_dir")[0]
    ws = tmp_path / "ws"
    cfg = ConfigManager(workspace_dir=ws)
    repo = Repository(ws)
    try:
        with get_adapter("video_dir").open(handle) as view:
            repo.ingest_episode(view.meta)
            repo.save_few_shot(
                FewShotItem(
                    example_id="fs_repo_1",
                    scope="task:pick_and_place",
                    task_id="pick_and_place",
                    error_type="fp_overflag",
                    episode_id=view.meta.episode_id,
                    t_start=0.2,
                    t_end=0.6,
                    rationale="Keep slow grasp",
                    created_at=time.time(),
                )
            )
            assert len(repo.list_few_shots(task_id="pick_and_place")) == 1

            def _fail_s3(kw: dict[str, Any]) -> dict[str, Any]:
                st = kw["stage"]
                if st == "s1_grounding":
                    return {"objects": {}}
                if st == "s2_macro":
                    return {"macro_stages": [], "arm_timelines": {}}
                if st.startswith("s5"):
                    return {"verdict": "failure", "confidence": 0.9, "criteria": []}
                raise VlmError("boom in S3")

            dag = VlmDag(workspace=ws, config=cfg, client=_FakeClient(_fail_s3), repo=repo)  # type: ignore[arg-type]
            with pytest.raises(VlmError):
                dag.run_episode(view)

            row = repo.get_episode_row(view.meta.episode_id)
            assert row["status"] == EpisodeStatus.INGESTED.value
            assert row["active_run_id"] is None
    finally:
        repo.close()
