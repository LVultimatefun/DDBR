"""Unit tests for SQLite WAL store, claim leases, immutability, gt_frozen, and ResolvedEpisode."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from robo_vlm_qc.models import (
    ArmSpan,
    ArmTimelines,
    CameraSpec,
    ConfigFingerprintSnapshot,
    EpisodeMeta,
    EpisodeStatus,
    GroundingKeyframe,
    GroundingResult,
    GroundingTrack,
    KinematicsSpec,
    MacroStage,
    QcSegment,
    ReviewWritePayload,
    SuccessVerdict,
    VisualEvidence,
    VlmRunArtifact,
    WatchCoverage,
)
from robo_vlm_qc.store import (
    AuthorizationError,
    ClaimConflictError,
    GtFrozenError,
    Repository,
    WatchCoverageError,
)


def _make_meta(ep_id: str = "ds/ep_001", mtime: float = 100.0) -> EpisodeMeta:
    ds, stem = ep_id.split("/", 1)
    return EpisodeMeta(
        episode_id=ep_id,
        dataset=ds,
        stem=stem,
        task_id="pick_and_place",
        adapter_type="h5_raw_v3",
        source_uri=f"/readonly/{ds}/{stem}",
        source_mtime=mtime,
        episode_fingerprint=f"fp_{ep_id}_{mtime}",
        duration_s=20.0,
        fps=30.0,
        frame_count=600,
        cameras=[
            CameraSpec(name="ego", role="main", width=1280, height=720, fps=30.0),
        ],
        camera_roles={"main": "ego"},
        kinematics=KinematicsSpec(has_kinematics=True),
    )


def _make_vlm_run(
    run_id: str = "run_001",
    ep_id: str = "ds/ep_001",
    *,
    is_sandbox: bool = False,
) -> VlmRunArtifact:
    return VlmRunArtifact(
        run_id=run_id,
        episode_id=ep_id,
        created_at=time.time(),
        is_eval_sandbox=is_sandbox,
        config_fingerprint=ConfigFingerprintSnapshot(
            model_routing={"s1_grounding": "gpt-4o"},
            prompt_bundle_version="1.0.0",
            label_taxonomy_version="1",
            task_spec_version="1.0.0",
        ),
        grounding=GroundingResult(
            keyframes_s=[0.0],
            object_tracks_2d={
                "bin": GroundingTrack(
                    object_id="bin",
                    name="Target Bin",
                    is_static_container=True,
                    keyframes=[
                        GroundingKeyframe(
                            row=0,
                            time_s=0.0,
                            visible=True,
                            bbox_norm_1000=[100, 100, 300, 400],
                        )
                    ],
                )
            },
        ),
        macro_stages=[
            MacroStage(
                stage_id="S1",
                t_start=0.0,
                t_end=20.0,
                label="Pick and place object",
                label_en="pick_and_place",
            )
        ],
        arm_timelines=ArmTimelines(
            left=[ArmSpan(t_start=0.0, t_end=20.0, state="work", desc="pick")],
            right=[ArmSpan(t_start=0.0, t_end=20.0, state="hold", desc="hold")],
        ),
        qc_segments=[
            QcSegment(
                seg_id="seg_1",
                t_start=2.0,
                t_end=4.0,
                label_id="boundary_idle",
                training_role="badcase",
                splice_policy="cut_clean",
                vlm_raw_policy="cut_clean",
                visual_evidence=VisualEvidence(
                    initial_state="idle",
                    process_change="still idle",
                    end_state="idle",
                ),
            )
        ],
        success=SuccessVerdict(passed=True, verdict="success", confidence=0.95),
    )


def test_wal_mode_ingest_idempotency_and_immutability(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "ws")
    try:
        mode = repo.db.connect().execute("PRAGMA journal_mode;").fetchone()[0]
        assert str(mode).lower() == "wal"

        meta = _make_meta()
        assert repo.ingest_episode(meta) == ("ds/ep_001", "inserted")
        assert repo.ingest_episode(meta) == ("ds/ep_001", "skipped")
        meta2 = _make_meta(mtime=105.0)
        assert repo.ingest_episode(meta2) == ("ds/ep_001", "updated")

        run = _make_vlm_run()
        repo.save_vlm_run_artifact(run)
        with pytest.raises(FileExistsError):
            repo.save_vlm_run_artifact(run)

        row = repo.get_episode_row("ds/ep_001")
        assert row["status"] == EpisodeStatus.VLM_READY.value
        assert row["active_run_id"] == "run_001"
    finally:
        repo.close()


def test_concurrent_claim_lease_expiry_and_release(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "ws")
    try:
        repo.ingest_episode(_make_meta())
        repo.save_vlm_run_artifact(_make_vlm_run())

        winners: list[str] = []
        conflicts: list[str] = []
        lock = threading.Lock()
        barrier = threading.Barrier(4)

        def _try_claim(user: str) -> None:
            barrier.wait()
            try:
                repo.claim_episode("ds/ep_001", user, ttl_s=120.0, now=1000.0)
                with lock:
                    winners.append(user)
            except ClaimConflictError:
                with lock:
                    conflicts.append(user)

        threads = [threading.Thread(target=_try_claim, args=(f"user_{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(winners) == 1
        assert len(conflicts) == 3
        winner = winners[0]

        repo.renew_claim("ds/ep_001", winner, ttl_s=60.0, now=1050.0)
        with pytest.raises(ClaimConflictError):
            repo.renew_claim("ds/ep_001", "intruder", ttl_s=60.0, now=1050.0)

        claim2 = repo.claim_episode("ds/ep_001", "user_b", ttl_s=120.0, now=1115.0)
        assert claim2["claimed_by"] == "user_b"

        reaped = repo.reap_expired_claims(now=1250.0)
        assert reaped == 1
        assert repo.get_episode_row("ds/ep_001")["status"] == EpisodeStatus.VLM_READY.value
    finally:
        repo.close()


def test_review_diff_resolved_overlay_and_gt_frozen_isolation(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "ws")
    try:
        repo.ingest_episode(_make_meta())
        vlm_run = _make_vlm_run("run_001")
        repo.save_vlm_run_artifact(vlm_run)

        low_watch_payload = ReviewWritePayload(
            base_run_id="run_001",
            action="submit_review",
            qc_segments=[],
            success=SuccessVerdict(passed=False, verdict="failure", confidence=1.0),
            macro_stages=vlm_run.macro_stages,
            watch_coverage=WatchCoverage(
                watched_intervals=[(0.0, 5.0)],
                watched_s=5.0,
                duration_s=20.0,
                coverage_ratio=0.25,
                confirmed=False,
            ),
        )
        with pytest.raises(WatchCoverageError):
            repo.save_review("ds/ep_001", "rev_a", "reviewer", low_watch_payload)

        with pytest.raises(AuthorizationError):
            repo.save_review(
                "ds/ep_001",
                "rev_a",
                "reviewer",
                low_watch_payload.model_copy(
                    update={
                        "action": "freeze_gt",
                        "watch_coverage": WatchCoverage(
                            watched_intervals=[(0.0, 20.0)],
                            watched_s=20.0,
                            duration_s=20.0,
                            coverage_ratio=1.0,
                            confirmed=True,
                        ),
                    }
                ),
            )

        fn_seg = QcSegment(
            seg_id="seg_hum",
            t_start=8.0,
            t_end=10.5,
            label_id="object_slip_recovery",
            training_role="recovery",
            splice_policy="mask_loss",
            mask_side="left",
            vlm_raw_policy="mask_loss",
        )
        good_payload = ReviewWritePayload(
            base_run_id="run_001",
            action="submit_review",
            qc_segments=[fn_seg],
            success=SuccessVerdict(passed=False, verdict="failure", confidence=1.0),
            macro_stages=[
                MacroStage(stage_id="S1", t_start=0.0, t_end=20.0, label="Pick and place revised")
            ],
            watch_coverage=WatchCoverage(
                watched_intervals=[(0.0, 20.0)],
                watched_s=20.0,
                duration_s=20.0,
                coverage_ratio=1.0,
                confirmed=True,
            ),
            notes="deleted FP idle, added slip recovery",
        )
        rev = repo.save_review("ds/ep_001", "rev_a", "reviewer", good_payload)
        assert rev.diff_summary.fp_deleted == 1
        assert rev.diff_summary.fn_added == 1
        assert rev.diff_summary.success_flipped is True
        assert rev.diff_summary.macro_stage_edits == 1

        resolved = repo.resolve_episode("ds/ep_001")
        assert resolved.is_human_reviewed is True
        assert resolved.qc_segments == [fn_seg]
        assert resolved.success.passed is False
        assert resolved.macro_stages[0].label == "Pick and place revised"
        assert resolved.grounding == vlm_run.grounding
        assert resolved.arm_timelines == vlm_run.arm_timelines
        assert resolved.raw_vlm_segments == vlm_run.qc_segments

        repo.save_review(
            "ds/ep_001",
            "lead_1",
            "lead",
            good_payload.model_copy(update={"action": "freeze_gt"}),
        )
        assert repo.get_episode_row("ds/ep_001")["status"] == EpisodeStatus.GT_FROZEN.value

        with pytest.raises(GtFrozenError):
            repo.claim_episode("ds/ep_001", "lead_1")
        with pytest.raises(GtFrozenError):
            repo.save_review("ds/ep_001", "lead_1", "lead", good_payload)
        with pytest.raises(GtFrozenError):
            repo.save_vlm_run_artifact(_make_vlm_run("run_002", is_sandbox=False))
        with pytest.raises(GtFrozenError):
            repo.ingest_episode(_make_meta(mtime=999.0))

        repo.save_vlm_run_artifact(_make_vlm_run("run_eval_1", is_sandbox=True))
        ep_after = repo.get_episode_row("ds/ep_001")
        assert ep_after["status"] == EpisodeStatus.GT_FROZEN.value
        assert ep_after["active_run_id"] == "run_001"

        orphan_path = (
            repo.workspace / "artifacts" / "vlm_runs" / "ds" / "ep_001" / "run_002.json"
        )
        assert not orphan_path.exists()
    finally:
        repo.close()


def test_revert_status_for_release_deterministic_tie_breaker(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "ws")
    try:
        repo.ingest_episode(_make_meta())
        vlm_run = _make_vlm_run("run_001")
        repo.save_vlm_run_artifact(vlm_run)

        wc = WatchCoverage(
            watched_intervals=[(0.0, 20.0)],
            watched_s=20.0,
            duration_s=20.0,
            coverage_ratio=1.0,
            confirmed=True,
        )
        base_payload = ReviewWritePayload(
            base_run_id="run_001",
            action="submit_review",
            qc_segments=[],
            success=SuccessVerdict(passed=True, verdict="success", confidence=1.0),
            macro_stages=vlm_run.macro_stages,
            watch_coverage=wc,
        )
        repo.save_review("ds/ep_001", "rev_a", "reviewer", base_payload, now=2000.0)
        repo.save_review(
            "ds/ep_001",
            "rev_a",
            "reviewer",
            base_payload.model_copy(
                update={"action": "flag", "flagged_reason": "need lead check"}
            ),
            now=2000.0,
        )
        assert repo.get_episode_row("ds/ep_001")["status"] == EpisodeStatus.FLAGGED.value

        repo.claim_episode("ds/ep_001", "lead_1", now=2001.0)
        reverted = repo.release_claim("ds/ep_001", "lead_1", now=2002.0)
        assert reverted == EpisodeStatus.FLAGGED.value
    finally:
        repo.close()
