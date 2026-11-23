"""Unit tests for robo_vlm_qc.lab.diagnosis and robo_vlm_qc.lab.few_shot."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from robo_vlm_qc.io import auto_detect, get_adapter
from robo_vlm_qc.lab.diagnosis import DiagnosisEngine, diagnose_workspace
from robo_vlm_qc.lab.few_shot import FewShotBook, create_few_shot_example, retrieve_few_shots
from robo_vlm_qc.models.annotation import (
    ArmTimelines,
    ConfigFingerprintSnapshot,
    CriterionResult,
    GroundingResult,
    MacroStage,
    QcSegment,
    SuccessVerdict,
    VlmRunArtifact,
)
from robo_vlm_qc.models.review import ReviewWritePayload, WatchCoverage
from robo_vlm_qc.models.taxonomy import FewShotItem
from robo_vlm_qc.store import NotFoundError, Repository


def _make_video_dataset(root: Path, ds_name: str = "pick_diag", duration_s: float = 10.0) -> Path:
    ds_dir = root / ds_name
    ds_dir.mkdir(parents=True, exist_ok=True)
    mp4 = ds_dir / "ep_000.mp4"
    fps = 10.0
    n_frames = int(round(duration_s * fps))
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    assert writer.isOpened()
    try:
        for i in range(n_frames):
            frame = np.full((120, 160, 3), (i * 5) % 255, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return ds_dir


def _ingest(ws: Path, ds_dir: Path, task_id: str = "pick_and_place") -> str:
    handles = auto_detect(ds_dir)
    assert handles
    with get_adapter(handles[0].adapter_type).open(handles[0]) as view:
        meta = view.meta.model_copy(update={"task_id": task_id})
    repo = Repository(ws)
    try:
        ep_id, _ = repo.ingest_episode(meta)
        return ep_id
    finally:
        repo.close()


def _make_run(
    ep_id: str,
    run_id: str,
    duration_s: float,
    segments: list[QcSegment],
    *,
    passed: bool = True,
    macro_stages: list[MacroStage] | None = None,
) -> VlmRunArtifact:
    stages = macro_stages or [
        MacroStage(stage_id="S1", t_start=0.0, t_end=duration_s * 0.5, label="Grasp object", label_en="grasp"),
        MacroStage(stage_id="S2", t_start=duration_s * 0.5, t_end=duration_s, label="Place object", label_en="place"),
    ]
    return VlmRunArtifact(
        run_id=run_id,
        episode_id=ep_id,
        created_at=time.time(),
        config_fingerprint=ConfigFingerprintSnapshot(
            model_routing={"s1_grounding": "gpt-4o"},
            prompt_bundle_version="1.0.0",
            label_taxonomy_version="v1",
            task_spec_version="1.0.0",
        ),
        grounding=GroundingResult(),
        macro_stages=stages,
        arm_timelines=ArmTimelines(),
        qc_segments=segments,
        success=SuccessVerdict(
            passed=passed,
            verdict="success" if passed else "failure",
            confidence=0.9,
            criteria=[CriterionResult(criterion="Goal satisfied", met=passed)],
        ),
    )


def _seg(
    seg_id: str,
    t_start: float,
    t_end: float,
    label_id: str = "missed_grasp_retry",
    role: str = "recovery",
    policy: str = "mask_loss",
    mask_side: str | None = "left",
) -> QcSegment:
    return QcSegment(
        seg_id=seg_id,
        t_start=t_start,
        t_end=t_end,
        label_id=label_id,
        training_role=role,  # type: ignore[arg-type]
        splice_policy=policy,  # type: ignore[arg-type]
        mask_side=mask_side,  # type: ignore[arg-type]
        affected_arm="left" if mask_side is None else mask_side,  # type: ignore[arg-type]
        vlm_raw_policy=policy,  # type: ignore[arg-type]
    )


def test_diagnosis_all_six_buckets_and_clean_and_empty(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ds_dir = _make_video_dataset(tmp_path / "data", "pick_diag", duration_s=10.0)
    ep_id = _ingest(ws, ds_dir)

    empty_rep = diagnose_workspace(ws)
    assert empty_rep.episodes_evaluated == 0
    assert empty_rep.total_errors == 0
    assert empty_rep.errors == []

    repo = Repository(ws)
    try:
        vlm_segs = [
            _seg("v_fp", 0.5, 1.3, "hesitation_oscillation", "badcase", "mask_loss", "left"),
            _seg("v_pflip", 2.0, 3.0, "mid_idle_freeze", "badcase", "cut_clean", None),
            _seg("v_bnd", 4.0, 5.0, "missed_grasp_retry", "recovery", "mask_loss", "left"),
        ]
        gt_segs = [
            _seg("g_pflip", 2.0, 3.0, "bimanual_assist_hold", "necessary", "keep_positive", None),
            _seg("g_bnd", 4.0, 5.8, "missed_grasp_retry", "recovery", "mask_loss", "left"),
            _seg("g_fn", 7.0, 8.2, "collision_or_knock", "badcase", "mask_loss", "right"),
        ]
        run = _make_run(ep_id, "r1", 10.0, vlm_segs, passed=True)
        repo.save_vlm_run_artifact(run)
        repo.save_review(
            ep_id,
            "lead_u",
            "lead",
            ReviewWritePayload(
                base_run_id="r1",
                action="freeze_gt",
                qc_segments=gt_segs,
                success=SuccessVerdict(passed=False, verdict="failure", confidence=1.0),
                macro_stages=[
                    MacroStage(stage_id="S1", t_start=0.0, t_end=3.0, label="Custom stage", label_en="custom"),
                ],
                watch_coverage=WatchCoverage(
                    watched_intervals=[(0.0, 10.0)],
                    watched_s=10.0,
                    duration_s=10.0,
                    coverage_ratio=1.0,
                    confirmed=True,
                ),
            ),
        )

        rep = diagnose_workspace(repo, episode_ids=[ep_id], top_n=4)
        assert rep.episodes_evaluated == 1
        assert rep.total_errors == 6
        assert set(rep.counts_by_type.keys()) == {
            "fp_overflag",
            "fn_missed",
            "policy_flip",
            "boundary",
            "success_flip",
            "stage",
        }
        assert len(rep.top_errors) == 4
        assert len(rep.clusters) == 6

        eng_rep = DiagnosisEngine(repo).diagnose(top_n=2)
        assert len(eng_rep.top_errors) == 2
    finally:
        repo.close()


def test_diagnosis_greedy_best_iou_pairing_and_zero_diff(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ds_dir = _make_video_dataset(tmp_path / "data", "pick_iou", duration_s=4.0)
    ep_id = _ingest(ws, ds_dir)

    repo = Repository(ws)
    try:
        v0 = _seg("v0", 1.0, 2.0, "missed_grasp_retry", "recovery", "mask_loss", "left")
        gt_weak = _seg("gt_weak", 0.6, 1.1, "boundary_idle", "badcase", "cut_clean", None)
        gt_strong = _seg("gt_strong", 1.02, 1.98, "missed_grasp_retry", "recovery", "mask_loss", "left")
        run = _make_run(ep_id, "r_iou", 4.0, [v0], passed=True)
        repo.save_vlm_run_artifact(run)
        repo.save_review(
            ep_id,
            "lead_u",
            "lead",
            ReviewWritePayload(
                base_run_id="r_iou",
                action="approve",
                qc_segments=[gt_weak, gt_strong],
                success=run.success,
                macro_stages=run.macro_stages,
                watch_coverage=WatchCoverage(
                    watched_intervals=[(0.0, 4.0)], watched_s=4.0, duration_s=4.0, coverage_ratio=1.0, confirmed=True
                ),
            ),
        )
        rep = diagnose_workspace(repo)
        etypes = [e.error_type for e in rep.errors]
        assert etypes == ["fn_missed"]
        assert rep.errors[0].gt_seg_id == "gt_weak"

        repo.save_review(
            ep_id,
            "lead_u",
            "lead",
            ReviewWritePayload(
                base_run_id="r_iou",
                action="freeze_gt",
                qc_segments=[v0],
                success=run.success,
                macro_stages=run.macro_stages,
                watch_coverage=WatchCoverage(
                    watched_intervals=[(0.0, 4.0)], watched_s=4.0, duration_s=4.0, coverage_ratio=1.0, confirmed=True
                ),
            ),
        )
        clean_rep = diagnose_workspace(repo)
        assert clean_rep.total_errors == 0
        assert clean_rep.errors == []
        assert clean_rep.counts_by_type == {}
    finally:
        repo.close()


def test_few_shot_create_tile_validation_and_scoped_retrieval(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ds_dir = _make_video_dataset(tmp_path / "data", "pick_fs", duration_s=4.0)
    ep_id = _ingest(ws, ds_dir)

    repo = Repository(ws)
    try:
        with pytest.raises(ValueError, match="t_end"):
            create_few_shot_example(
                repo,
                episode_id=ep_id,
                t_start=2.0,
                t_end=1.0,
                rationale="invalid span",
            )

        with pytest.raises(NotFoundError):
            create_few_shot_example(
                repo,
                episode_id="no_ds/no_ep",
                t_start=0.5,
                t_end=1.5,
                rationale="missing episode",
            )

        item = create_few_shot_example(
            repo,
            episode_id=ep_id,
            t_start=0.5,
            t_end=1.5,
            scope="task:pick_and_place",
            task_id="pick_and_place",
            label_id="missed_grasp_retry",
            error_type="fp_overflag",
            vlm_wrong_json={"label_id": "missed_grasp_retry"},
            human_right_json={"splice_policy": "keep_positive"},
            rationale="Normal smooth approach",
        )
        assert item.tile_image_path is not None
        tile_path = ws / item.tile_image_path
        assert tile_path.is_file() and tile_path.stat().st_size > 100
        img = cv2.imread(str(tile_path))
        assert img is not None and img.shape[0] >= 120 and img.shape[1] >= 160

        repo.save_few_shot(
            FewShotItem(
                example_id="fs_global_1",
                scope="global",
                error_type="fp_overflag",
                episode_id=ep_id,
                t_start=0.1,
                t_end=0.5,
                rationale="Global guidance",
                enabled=True,
                created_at=100.0,
            )
        )
        repo.save_few_shot(
            FewShotItem(
                example_id="fs_handover_1",
                scope="task:bimanual_handover",
                task_id="bimanual_handover",
                error_type="fn_missed",
                episode_id=ep_id,
                t_start=0.1,
                t_end=0.5,
                rationale="Handover only rule",
                enabled=True,
                created_at=200.0,
            )
        )
        repo.save_few_shot(
            FewShotItem(
                example_id="fs_lbl_1",
                scope="label:collision_or_knock",
                label_id="collision_or_knock",
                error_type="boundary",
                episode_id=ep_id,
                t_start=0.1,
                t_end=0.5,
                rationale="Collision boundary",
                enabled=True,
                created_at=150.0,
            )
        )
        repo.save_few_shot(
            FewShotItem(
                example_id="fs_disabled_1",
                scope="global",
                error_type="fp_overflag",
                episode_id=ep_id,
                t_start=0.1,
                t_end=0.5,
                rationale="Should never appear",
                enabled=False,
                created_at=999.0,
            )
        )

        got = retrieve_few_shots(
            repo,
            task_id="pick_and_place",
            label_ids=["collision_or_knock"],
            top_k=10,
            max_tokens=5000,
        )
        got_ids = [g.example_id for g in got]
        assert item.example_id in got_ids
        assert "fs_lbl_1" in got_ids
        assert "fs_global_1" in got_ids
        assert "fs_handover_1" not in got_ids
        assert "fs_disabled_1" not in got_ids
        assert got_ids.index(item.example_id) < got_ids.index("fs_lbl_1") < got_ids.index("fs_global_1")

        assert retrieve_few_shots(repo, task_id="pick_and_place", top_k=0) == []
        assert retrieve_few_shots(repo, task_id="pick_and_place", top_k=5, max_tokens=1) == []

        book = FewShotBook(ws)
        assert len(book.retrieve(task_id="pick_and_place", top_k=1)) == 1

        with sqlite3.connect(str(ws / "robo_vlm_qc.db")) as conn:
            cnt = conn.execute("SELECT COUNT(*) FROM few_shot_examples").fetchone()[0]
            assert cnt == 5
    finally:
        repo.close()
