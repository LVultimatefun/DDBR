"""Unit tests for Repository extensions (list_reviews, jobs CRUD, guardrail_downgraded, API resolve)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from robo_vlm_qc.models import (
    ArmTimelines,
    CameraSpec,
    ConfigFingerprintSnapshot,
    EpisodeMeta,
    EpisodeStatus,
    GroundingResult,
    GuardrailReport,
    KinematicsSpec,
    MacroStage,
    QcSegment,
    ReviewWritePayload,
    SuccessVerdict,
    VlmRunArtifact,
    WatchCoverage,
)
from robo_vlm_qc.store import NotFoundError, Repository


def _meta(ep_id: str) -> EpisodeMeta:
    ds, stem = ep_id.split("/", 1)
    return EpisodeMeta(
        episode_id=ep_id,
        dataset=ds,
        stem=stem,
        task_id="pick_and_place",
        adapter_type="video_dir",
        source_uri=f"/readonly/{ds}/{stem}.mp4",
        source_mtime=100.0,
        episode_fingerprint=f"fp_{ep_id}",
        duration_s=5.0,
        fps=30.0,
        frame_count=150,
        cameras=[CameraSpec(name="main", role="main", width=320, height=240, fps=30.0)],
        camera_roles={"main": "main"},
        kinematics=KinematicsSpec(has_kinematics=False),
    )


def _run(run_id: str, ep_id: str, *, downgraded: bool = False) -> VlmRunArtifact:
    segs = []
    if downgraded:
        segs.append(
            QcSegment(
                seg_id="seg_dg",
                t_start=1.0,
                t_end=2.0,
                label_id="mid_idle_freeze",
                training_role="badcase",
                splice_policy="mask_loss",
                mask_side="both",
                vlm_raw_policy="cut_clean",
                guardrail=GuardrailReport(
                    gate_type="splice_gate",
                    passed=False,
                    downgraded=True,
                    downgrade_reason="pose jump > 2cm",
                ),
            )
        )
    return VlmRunArtifact(
        run_id=run_id,
        episode_id=ep_id,
        created_at=time.time(),
        config_fingerprint=ConfigFingerprintSnapshot(
            model_routing={"s1_grounding": "gpt-4o"},
            prompt_bundle_version="1.0.0",
            label_taxonomy_version="1",
            task_spec_version="1.0.0",
        ),
        grounding=GroundingResult(),
        macro_stages=[MacroStage(stage_id="S1", t_start=0.0, t_end=5.0, label="Operation")],
        arm_timelines=ArmTimelines(),
        qc_segments=segs,
        success=SuccessVerdict(passed=True, verdict="success", confidence=0.95),
    )


def test_list_reviews_chronological_and_resolve_for_api(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "ws")
    try:
        repo.ingest_episode(_meta("ds/ep_ingested"))

        with pytest.raises(NotFoundError):
            repo.resolve_episode("ds/ep_ingested")

        api_res = repo.resolve_episode_for_api("ds/ep_ingested")
        assert api_res.status == EpisodeStatus.INGESTED
        assert api_res.active_run_id is None
        assert api_res.qc_segments == []

        run = _run("run_1", "ds/ep_ingested", downgraded=True)
        repo.save_vlm_run_artifact(run)

        wc = WatchCoverage(duration_s=5.0, watched_s=5.0, coverage_ratio=1.0, confirmed=True)
        for i, action in enumerate(("save_draft", "submit_review", "approve")):
            role = "lead" if action == "approve" else "reviewer"
            author = "lead_u" if role == "lead" else "rev_u"
            payload = ReviewWritePayload(
                base_run_id="run_1",
                action=action,
                qc_segments=[],
                success=run.success,
                macro_stages=run.macro_stages,
                watch_coverage=wc,
                notes=f"step_{i}",
            )
            repo.save_review("ds/ep_ingested", author, role, payload, now=1000.0 + i)

        revs = repo.list_reviews("ds/ep_ingested")
        assert [r.notes for r in revs] == ["step_0", "step_1", "step_2"]
        assert [r.action for r in revs] == ["save_draft", "submit_review", "approve"]

        with pytest.raises(NotFoundError):
            repo.list_reviews("ds/missing")
    finally:
        repo.close()


def test_guardrail_downgraded_filter_and_jobs_crud(tmp_path: Path) -> None:
    repo = Repository(tmp_path / "ws")
    try:
        repo.ingest_episode(_meta("ds/ep_a"))
        repo.ingest_episode(_meta("ds/ep_b"))
        repo.save_vlm_run_artifact(_run("run_a", "ds/ep_a", downgraded=True))
        repo.save_vlm_run_artifact(_run("run_b", "ds/ep_b", downgraded=False))

        dg_rows = repo.list_episodes(guardrail_downgraded=True)
        assert [r["episode_id"] for r in dg_rows] == ["ds/ep_a"]
        assert repo.count_episodes(guardrail_downgraded=True) == 1

        clean_rows = repo.list_episodes(guardrail_downgraded=False)
        assert [r["episode_id"] for r in clean_rows] == ["ds/ep_b"]

        j1 = repo.create_job("batch_vlm", {"episode_ids": ["ds/ep_a"]}, progress_total=1, now=10.0)
        j2 = repo.create_job("render_mp4", {"episode_ids": ["ds/ep_b"]}, progress_total=1, now=20.0)
        assert repo.get_job(j1["job_id"])["status"] == "queued"

        listed = repo.list_jobs()
        assert [j["job_id"] for j in listed] == [j2["job_id"], j1["job_id"]]

        upd = repo.update_job(j1["job_id"], status="running", progress_done=1, summary={"ok": True})
        assert upd["status"] == "running"
        assert upd["progress_done"] == 1
        assert upd["summary"] == {"ok": True}

        cancelled = repo.cancel_job(j2["job_id"])
        assert cancelled["status"] == "cancelled"

        with pytest.raises(NotFoundError):
            repo.cancel_job("job_missing")
    finally:
        repo.close()
