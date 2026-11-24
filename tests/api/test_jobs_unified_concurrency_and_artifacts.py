"""Tests for unified async jobs, concurrent batch_vlm, artifact download, partial rerun, and run_gt_eval progress."""

from __future__ import annotations

import io
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

from robo_vlm_qc.config import ConfigManager
from robo_vlm_qc.lab.eval_runner import run_gt_eval
from robo_vlm_qc.models.annotation import SuccessVerdict
from robo_vlm_qc.models.review import EpisodeStatus, ReviewWritePayload, WatchCoverage
from robo_vlm_qc.store import Repository
from tests.api.conftest import (
    LEAD_HEADERS,
    REV_HEADERS,
    create_video_episode_file,
    ingest_dir,
    make_client,
    seed_vlm_run,
    write_workspace_config,
)
from tests.e2e.conftest import MockVlmServer
from tests.e2e.helpers import make_synthetic_h5_raw_v3


def _poll_job_until_done(
    client: Any, job_id: str, timeout_s: float = 20.0
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        r = client.get(f"/api/jobs/{job_id}", headers=LEAD_HEADERS)
        assert r.status_code == 200
        job = r.json()
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(0.04)
    raise AssertionError(f"Job {job_id} did not finish within {timeout_s}s")


def test_batch_vlm_4way_concurrency_sse_and_db_cleanup(
    api_env: dict[str, Any]
) -> None:
    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]
    mock_vlm: MockVlmServer = api_env["mock_vlm"]
    ds_dir = api_env["data_root"] / "ds_conc"

    for i in range(4):
        make_synthetic_h5_raw_v3(ds_dir, name=f"ep_{i:02d}.h5", duration_s=2.5, stationary=True)

    repo = Repository(ws)
    try:
        ep_ids = sorted(ingest_dir(repo, cfg_dir, ws, ds_dir, dataset="ds_conc"))
    finally:
        repo.close()

    # Add 0.2s latency on S1 so we can measure true concurrent in-flight execution across 4 episodes
    inflight_lock = threading.Lock()
    inflight = 0
    max_inflight = 0

    def _delayed_s1(_rec: dict[str, Any]) -> dict[str, Any]:
        nonlocal inflight, max_inflight
        with inflight_lock:
            inflight += 1
            if inflight > max_inflight:
                max_inflight = inflight
        try:
            time.sleep(0.22)
            return {
                "objects": {
                    "target_object": {
                        "visible": True,
                        "bbox": [100, 100, 300, 300],
                        "state": "ok",
                        "status_kind": "neutral",
                    }
                }
            }
        finally:
            with inflight_lock:
                inflight -= 1

    mock_vlm.set_stage_response("s1", _delayed_s1)

    with make_client(ws, cfg_dir) as client:
        seq_start = client.app.state.ctx.sse.latest_seq()  # type: ignore[attr-defined]
        t0 = time.perf_counter()
        r_post = client.post(
            "/api/jobs",
            headers=LEAD_HEADERS,
            json={
                "job_type": "batch_vlm",
                "episode_ids": ep_ids,
                "concurrency": 4,
            },
        )
        assert r_post.status_code == 201
        job_id = r_post.json()["job_id"]
        done_job = _poll_job_until_done(client, job_id)
        elapsed = time.perf_counter() - t0

        assert done_job["status"] == "completed", done_job
        assert done_job["progress_done"] == 4
        assert done_job["progress_total"] == 4
        assert len(done_job["summary"]["results"]) == 4
        assert done_job["summary"]["errors"] == []
        # True parallelism verified
        assert max_inflight >= 2
        assert elapsed < 3.0

        # SSE published incremental job_update events
        _, events = client.app.state.ctx.sse.since(seq_start)  # type: ignore[attr-defined]
        prog_vals = [
            ev["job"]["progress_done"]
            for ev in events
            if ev.get("type") == "job_update" and ev["job"]["job_id"] == job_id
        ]
        assert 4 in prog_vals

        # Worker thread SQLite connections were closed cleanly (only main/server thread connections remain)
        conns_left = len(client.app.state.ctx.repo.db._conns)  # type: ignore[attr-defined]
        assert conns_left <= 3


def test_batch_vlm_s3_s4_partial_rerun_and_gt_frozen_safety(
    api_env: dict[str, Any]
) -> None:
    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]
    mock_vlm: MockVlmServer = api_env["mock_vlm"]
    ds_dir = api_env["data_root"] / "ds_partial"

    make_synthetic_h5_raw_v3(ds_dir, name="ep_frozen.h5", duration_s=3.0, stationary=True)
    make_synthetic_h5_raw_v3(ds_dir, name="ep_open.h5", duration_s=3.0, stationary=True)

    repo = Repository(ws)
    try:
        ep_frozen = ingest_dir(repo, cfg_dir, ws, ds_dir / "ep_frozen.h5", dataset="ds_partial")[0]
        ep_open = ingest_dir(repo, cfg_dir, ws, ds_dir / "ep_open.h5", dataset="ds_partial")[0]
    finally:
        repo.close()

    with make_client(ws, cfg_dir) as client:
        # 1. Warm S1-S5 on both episodes
        r1 = client.post(
            "/api/jobs",
            headers=LEAD_HEADERS,
            json={
                "job_type": "batch_vlm",
                "episode_ids": [ep_frozen, ep_open],
                "concurrency": 2,
            },
        )
        assert r1.status_code == 201
        j1 = _poll_job_until_done(client, r1.json()["job_id"])
        assert j1["status"] == "completed"

    # Freeze ep_frozen as gt_frozen
    repo2 = Repository(ws)
    try:
        frozen_run_before = repo2.get_episode_row(ep_frozen)["active_run_id"]
        assert frozen_run_before is not None
        repo2.claim_episode(ep_frozen, "lead_1")
        rev_art = repo2.save_review(
            ep_frozen,
            author="lead_1",
            role="lead",
            payload=ReviewWritePayload(
                base_run_id=frozen_run_before,
                action="freeze_gt",
                qc_segments=[],
                success=SuccessVerdict(
                    verdict="success", passed=True, confidence=0.95, summary="gt"
                ),
                watch_coverage=WatchCoverage(
                    watched_intervals=[[0.0, 3.0]],
                    watched_s=3.0,
                    duration_s=3.0,
                    coverage_ratio=1.0,
                ),
            ),
            min_watch_coverage=0.95,
        )
    finally:
        repo2.close()

    mock_vlm.reset()
    with make_client(ws, cfg_dir) as client:
        # 2. Mixed batch with stage="S3/S4" -> ep_frozen fails cleanly in errors without mutating GT,
        # while ep_open runs ONLY S3/S4A/S4B (reusing S1/S2/S5 cache!)
        r2 = client.post(
            "/api/jobs",
            headers=LEAD_HEADERS,
            json={
                "job_type": "batch_vlm",
                "episode_ids": [ep_frozen, ep_open],
                "stage": "S3/S4",
                "concurrency": 2,
            },
        )
        assert r2.status_code == 201
        j2 = _poll_job_until_done(client, r2.json()["job_id"])
        assert j2["status"] == "failed"
        assert len(j2["summary"]["results"]) == 1
        assert j2["summary"]["results"][0]["episode_id"] == ep_open
        assert len(j2["summary"]["errors"]) == 1
        assert j2["summary"]["errors"][0]["episode_id"] == ep_frozen
        assert "gt_frozen" in j2["summary"]["errors"][0]["error"]

        stages_called = {req["stage"] for req in mock_vlm.requests}
        assert stages_called.issubset({"S3", "S4A", "S4B"})
        assert "S1" not in stages_called
        assert "S2" not in stages_called
        assert "S5" not in stages_called

        # 2b. Partial rerun with stage="S3/S4" & fresh=True STILL reuses S1/S2/S5 cache!
        mock_vlm.reset()
        r2b = client.post(
            "/api/jobs",
            headers=LEAD_HEADERS,
            json={
                "job_type": "batch_vlm",
                "episode_ids": [ep_open],
                "stage": "S3/S4",
                "fresh": True,
            },
        )
        assert r2b.status_code == 201
        j2b = _poll_job_until_done(client, r2b.json()["job_id"])
        assert j2b["status"] == "completed"
        stages_2b = {req["stage"] for req in mock_vlm.requests}
        assert stages_2b.issubset({"S3", "S4A", "S4B"})
        assert not ({"S1", "S2", "S5"} & stages_2b)

        # 2c. Backward-compat: stage="all" with fresh=False reuses StageCache (0 VLM calls)
        mock_vlm.reset()
        r2c = client.post(
            "/api/jobs",
            headers=LEAD_HEADERS,
            json={
                "job_type": "batch_vlm",
                "episode_ids": [ep_open],
                "stage": "all",
                "fresh": False,
            },
        )
        assert r2c.status_code == 201
        j2c = _poll_job_until_done(client, r2c.json()["job_id"])
        assert j2c["status"] == "completed"
        assert mock_vlm.requests == []

        # 3. Sandbox rerun on gt_frozen succeeds without mutating status/active_run_id/active_rev_id
        mock_vlm.reset()
        r3 = client.post(
            "/api/jobs",
            headers=LEAD_HEADERS,
            json={
                "job_type": "batch_vlm",
                "episode_ids": [ep_frozen],
                "stage": "S1-S5",
                "is_eval_sandbox": True,
            },
        )
        assert r3.status_code == 201
        j3 = _poll_job_until_done(client, r3.json()["job_id"])
        assert j3["status"] == "completed"
        stages_s1_s5 = {req["stage"] for req in mock_vlm.requests}
        assert {"S1", "S2", "S3", "S5"}.issubset(stages_s1_s5)

    repo3 = Repository(ws)
    try:
        row_frozen = repo3.get_episode_row(ep_frozen)
        assert row_frozen["status"] == EpisodeStatus.GT_FROZEN.value
        assert row_frozen["active_run_id"] == frozen_run_before
        assert row_frozen["active_rev_id"] == rev_art.rev_id
    finally:
        repo3.close()


def test_unified_jobs_render_mp4_export_eval_gt_ingest_and_artifact_download(
    api_env: dict[str, Any], tmp_path: Path
) -> None:
    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]
    allowed_root = tmp_path / "allowed_ingest"
    forbidden_root = tmp_path / "forbidden_ingest"
    allowed_root.mkdir(parents=True)
    forbidden_root.mkdir(parents=True)

    make_synthetic_h5_raw_v3(
        allowed_root / "ds_ing", name="ep_ing_1.h5", duration_s=2.5, stationary=True
    )
    make_synthetic_h5_raw_v3(
        forbidden_root / "ds_sec", name="ep_sec.h5", duration_s=2.5, stationary=True
    )
    # Symlink inside allowed_root pointing outside allowed_root
    (allowed_root / "evil_link.h5").symlink_to(forbidden_root / "ds_sec" / "ep_sec.h5")

    with make_client(ws, cfg_dir, allowed_data_roots=[allowed_root]) as client:
        # Backward compat: batch_vlm & render_mp4 with episode_ids=[] short-circuit immediately
        for jt in ("batch_vlm", "render_mp4"):
            r_empty = client.post(
                "/api/jobs",
                headers=LEAD_HEADERS,
                json={"job_type": jt, "episode_ids": []},
            )
            assert r_empty.status_code == 201
            assert r_empty.json()["status"] == "completed"
            assert r_empty.json()["progress_total"] == 0

        # 1. Async ingest job (empty episode_ids MUST execute, and symlink escape is blocked)
        assert (
            client.post(
                "/api/jobs",
                headers=LEAD_HEADERS,
                json={
                    "job_type": "ingest",
                    "ingest_params": {"data_path": str(forbidden_root)},
                },
            ).status_code
            == 403
        )
        r_ing = client.post(
            "/api/jobs",
            headers=LEAD_HEADERS,
            json={
                "job_type": "ingest",
                "ingest_params": {"data_path": str(allowed_root / "ds_ing"), "dataset": "ds_ing"},
            },
        )
        assert r_ing.status_code == 201
        j_ing = _poll_job_until_done(client, r_ing.json()["job_id"])
        assert j_ing["status"] == "completed", j_ing
        assert j_ing["summary"]["inserted"] == 1
        ep_id = j_ing["summary"]["results"][0]["episode_id"]

        # Scanning allowed_root directly blocks escaping symlink evil_link.h5 -> forbidden_root
        r_ing_esc = client.post(
            "/api/jobs",
            headers=LEAD_HEADERS,
            json={
                "job_type": "ingest",
                "ingest_params": {"path": str(allowed_root)},
            },
        )
        assert r_ing_esc.status_code == 201
        j_ing_esc = _poll_job_until_done(client, r_ing_esc.json()["job_id"])
        assert any(
            "escapes allowed_data_roots" in e["error"] for e in j_ing_esc["summary"]["errors"]
        )
        assert not any("ep_sec" in r["episode_id"] for r in j_ing_esc["summary"]["results"])

    # Seed VLM run + freeze GT for export & eval_gt
    repo = Repository(ws)
    try:
        cfg = ConfigManager(config_dir=cfg_dir, workspace_dir=ws)
        seed_vlm_run(repo, cfg, ep_id, run_id="run_exp_1", segments=[])
        repo.claim_episode(ep_id, "lead_1")
        repo.save_review(
            ep_id,
            author="lead_1",
            role="lead",
            payload=ReviewWritePayload(
                base_run_id="run_exp_1",
                action="freeze_gt",
                qc_segments=[],
                success=SuccessVerdict(
                    verdict="success", passed=True, confidence=0.95, summary="gt"
                ),
                watch_coverage=WatchCoverage(
                    watched_intervals=[[0.0, 2.5]],
                    watched_s=2.5,
                    duration_s=2.5,
                    coverage_ratio=1.0,
                ),
            ),
            min_watch_coverage=0.95,
        )
    finally:
        repo.close()

    with make_client(ws, cfg_dir, allowed_data_roots=[allowed_root]) as client:
        # 2. render_mp4 + GET /api/jobs/{job_id}/artifact
        r_mp4 = client.post(
            "/api/jobs",
            headers=LEAD_HEADERS,
            json={"job_type": "render_mp4", "episode_ids": [ep_id]},
        )
        assert r_mp4.status_code == 201
        j_mp4 = _poll_job_until_done(client, r_mp4.json()["job_id"])
        assert j_mp4["status"] == "completed", j_mp4
        assert len(j_mp4["summary"]["artifacts"]) == 1
        assert j_mp4["summary"]["artifacts"][0]["kind"] == "mp4"

        assert (
            client.get(
                f"/api/jobs/{j_mp4['job_id']}/artifact?index=0", headers=REV_HEADERS
            ).status_code
            == 403
        )
        dl_mp4 = client.get(
            f"/api/jobs/{j_mp4['job_id']}/artifact?index=0", headers=LEAD_HEADERS
        )
        assert dl_mp4.status_code == 200
        assert dl_mp4.headers["content-type"].startswith("video/mp4")
        assert len(dl_mp4.content) > 100
        # Query-param ?token= auth also works for direct browser <a download> links
        dl_mp4_qs = client.get(
            f"/api/jobs/{j_mp4['job_id']}/artifact?index=0&token=lead-tok"
        )
        assert dl_mp4_qs.status_code == 200
        assert len(dl_mp4_qs.content) == len(dl_mp4.content)
        assert (
            client.get(
                f"/api/jobs/{j_mp4['job_id']}/artifact?index=9", headers=LEAD_HEADERS
            ).status_code
            == 404
        )

        # 3. Async export job + artifact ZIP download (with empty episode_ids=[])
        r_exp = client.post(
            "/api/jobs",
            headers=LEAD_HEADERS,
            json={
                "job_type": "export",
                "episode_ids": [],
                "export_params": {"format": "lerobot_v3", "recipe": "bc_clean"},
            },
        )
        assert r_exp.status_code == 201
        j_exp = _poll_job_until_done(client, r_exp.json()["job_id"])
        assert j_exp["status"] == "completed", j_exp
        assert j_exp["summary"]["export_id"].startswith("exp_")
        assert j_exp["summary"]["artifacts"][0]["kind"] == "export"

        dl_zip = client.get(
            f"/api/jobs/{j_exp['job_id']}/artifact?index=0", headers=LEAD_HEADERS
        )
        assert dl_zip.status_code == 200
        assert dl_zip.headers["content-type"] == "application/zip"
        with zipfile.ZipFile(io.BytesIO(dl_zip.content)) as zf:
            assert any(name.endswith("qc_export_manifest.json") for name in zf.namelist())

        # Sync POST /api/exports still works identically
        r_sync_exp = client.post(
            "/api/exports",
            headers=LEAD_HEADERS,
            json={"format": "lerobot_v3", "recipe": "bc_clean"},
        )
        assert r_sync_exp.status_code == 200
        assert r_sync_exp.json()["export_id"].startswith("exp_")

        # 4. Async eval_gt job (with empty episode_ids=[]) + sync POST /api/lab/eval compat
        r_eval = client.post(
            "/api/jobs",
            headers=LEAD_HEADERS,
            json={
                "job_type": "eval_gt",
                "episode_ids": [],
                "eval_params": {"workers": 2, "few_shot": False},
            },
        )
        assert r_eval.status_code == 201
        j_eval = _poll_job_until_done(client, r_eval.json()["job_id"])
        assert j_eval["status"] == "completed", j_eval
        assert j_eval["summary"]["eval_run_id"].startswith("eval_")
        assert j_eval["summary"]["eval_id"] == j_eval["summary"]["eval_run_id"]
        assert j_eval["summary"]["eval_report"]["eval_id"] == j_eval["summary"]["eval_run_id"]
        assert j_eval["summary"]["metrics"]["episodes_evaluated"] == 1

        r_sync_eval = client.post(
            "/api/lab/eval",
            headers=LEAD_HEADERS,
            json={"workers": 1, "few_shot": False},
        )
        assert r_sync_eval.status_code == 200
        assert r_sync_eval.json()["metrics"]["episodes_evaluated"] == 1

        # 5. Path traversal containment check on /artifact (tampered summary path -> 403 before 404)
        tampered_job = client.app.state.ctx.repo.create_job(  # type: ignore[attr-defined]
            "render_mp4",
            {},
            progress_total=1,
            status="completed",
        )
        client.app.state.ctx.repo.update_job(  # type: ignore[attr-defined]
            tampered_job["job_id"],
            status="completed",
            summary={
                "artifacts": [
                    {
                        "name": "passwd",
                        "path": "../../etc/passwd",
                        "kind": "mp4",
                        "export_id": None,
                        "episode_id": None,
                    }
                ]
            },
        )
        assert (
            client.get(
                f"/api/jobs/{tampered_job['job_id']}/artifact?index=0",
                headers=LEAD_HEADERS,
            ).status_code
            == 403
        )


def test_run_gt_eval_on_progress_unit(api_env: dict[str, Any]) -> None:
    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]
    ds_dir = api_env["data_root"] / "ds_eval_prog"
    for i in range(2):
        create_video_episode_file(ds_dir, f"ep_ev_{i}.mp4", duration_s=1.5, fps=10.0)

    repo = Repository(ws)
    try:
        ep_ids = sorted(ingest_dir(repo, cfg_dir, ws, ds_dir, dataset="ds_eval_prog"))
        cfg = ConfigManager(config_dir=cfg_dir, workspace_dir=ws)
        for idx, ep_id in enumerate(ep_ids):
            seed_vlm_run(repo, cfg, ep_id, run_id=f"r_ev_{idx}", segments=[])
            repo.claim_episode(ep_id, "lead_1")
            repo.save_review(
                ep_id,
                author="lead_1",
                role="lead",
                payload=ReviewWritePayload(
                    base_run_id=f"r_ev_{idx}",
                    action="freeze_gt",
                    qc_segments=[],
                    success=SuccessVerdict(
                        verdict="success", passed=True, confidence=0.95, summary="gt"
                    ),
                    watch_coverage=WatchCoverage(
                        watched_intervals=[[0.0, 1.5]],
                        watched_s=1.5,
                        duration_s=1.5,
                        coverage_ratio=1.0,
                    ),
                ),
                min_watch_coverage=0.95,
            )

        events: list[tuple[int, int, str]] = []
        rep2 = run_gt_eval(
            repo,
            config=cfg,
            config_dir=cfg_dir,
            few_shot=False,
            workers=2,
            persist_report=False,
            on_progress=lambda d, t, eid, _info: events.append((d, t, eid)),
        )
        assert rep2.metrics.episodes_evaluated == 2
        assert [e[0] for e in events] == [1, 2]
        assert all(e[1] == 2 for e in events)
        assert {e[2] for e in events} == set(ep_ids)
        assert len(repo.db._conns) <= 2

        # Backward compat with on_progress=None & workers=1
        rep1 = run_gt_eval(
            repo,
            config=cfg,
            config_dir=cfg_dir,
            few_shot=False,
            workers=1,
            persist_report=False,
        )
        assert rep1.metrics.episodes_evaluated == 2
    finally:
        repo.close()


def test_batch_vlm_s1_only_rerun_refreshes_grounding_and_reuses_s2_s5_cache(
    api_env: dict[str, Any],
) -> None:
    """POST /api/jobs with stage='s1_grounding' or 's1' refreshes S1 (including v1->v2) with 0 S2-S5 calls & preserves protected status."""
    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]
    mock_vlm: MockVlmServer = api_env["mock_vlm"]
    ds_dir = api_env["data_root"] / "ds_s1_only"

    make_synthetic_h5_raw_v3(ds_dir, name="ep_open.h5", duration_s=2.5, stationary=True)
    make_synthetic_h5_raw_v3(ds_dir, name="ep_rev.h5", duration_s=2.5, stationary=True)

    repo = Repository(ws)
    try:
        ep_open = ingest_dir(repo, cfg_dir, ws, ds_dir / "ep_open.h5", dataset="ds_s1_only")[0]
        ep_rev = ingest_dir(repo, cfg_dir, ws, ds_dir / "ep_rev.h5", dataset="ds_s1_only")[0]
    finally:
        repo.close()

    # Start with tracking disabled (schema_version=1) and initial target_object bbox [100, 100, 220, 220]
    write_workspace_config(
        ws,
        extra_overrides={"vlm": {"s1_grounding": {"tracking": {"enabled": False}}}},
    )
    mock_vlm.set_stage_response(
        "s1",
        {
            "objects": {
                "target_object": {
                    "visible": True,
                    "bbox": [100, 100, 220, 220],
                    "state": "initial",
                    "status_kind": "neutral",
                }
            }
        },
    )

    with make_client(ws, cfg_dir) as client:
        r_warm = client.post(
            "/api/jobs",
            headers=LEAD_HEADERS,
            json={"job_type": "batch_vlm", "episode_ids": [ep_open, ep_rev], "stage": "all"},
        )
        assert r_warm.status_code == 201
        j_warm = _poll_job_until_done(client, r_warm.json()["job_id"])
        assert j_warm["status"] == "completed", j_warm

        ep_open_v1 = client.get(f"/api/episodes/{ep_open}", headers=LEAD_HEADERS).json()
        assert ep_open_v1["grounding"]["schema_version"] == 1
        assert ep_open_v1["grounding"]["object_tracks_2d"]["target_object"]["keyframes"][0]["bbox_norm_1000"] == [
            100,
            100,
            220,
            220,
        ]
        run_id_v1 = ep_open_v1["active_run_id"]
        macro_v1 = ep_open_v1["macro_stages"]
        segs_v1 = ep_open_v1["qc_segments"]
        succ_v1 = ep_open_v1["success"]

    repo_chk = Repository(ws)
    try:
        art_v1 = repo_chk.get_vlm_run(run_id_v1)
        # Put ep_rev into protected 'reviewed' state
        rev_run_before = repo_chk.get_episode_row(ep_rev)["active_run_id"]
        assert rev_run_before is not None
        repo_chk.claim_episode(ep_rev, "reviewer_1")
        repo_chk.save_review(
            ep_rev,
            author="reviewer_1",
            role="reviewer",
            payload=ReviewWritePayload(
                base_run_id=rev_run_before,
                action="submit_review",
                qc_segments=[],
                success=SuccessVerdict(verdict="success", passed=True, confidence=0.95, summary="reviewed"),
                watch_coverage=WatchCoverage(
                    watched_intervals=[[0.0, 2.5]],
                    watched_s=2.5,
                    duration_s=2.5,
                    coverage_ratio=1.0,
                ),
            ),
            min_watch_coverage=0.95,
        )
    finally:
        repo_chk.close()

    # Enable dense BAFT tracking (schema_version=2) and update S1 bbox to [320, 240, 480, 420]
    write_workspace_config(
        ws,
        extra_overrides={"vlm": {"s1_grounding": {"tracking": {"enabled": True}}}},
    )
    mock_vlm.reset()
    mock_vlm.set_stage_response(
        "s1",
        {
            "objects": {
                "target_object": {
                    "visible": True,
                    "bbox": [320, 240, 480, 420],
                    "state": "updated_v2",
                    "status_kind": "good",
                }
            }
        },
    )

    with make_client(ws, cfg_dir) as client:
        r_s1 = client.post(
            "/api/jobs",
            headers=LEAD_HEADERS,
            json={
                "job_type": "batch_vlm",
                "episode_ids": [ep_open],
                "stage": "s1_grounding",
                "fresh": False,
            },
        )
        assert r_s1.status_code == 201
        j_s1 = _poll_job_until_done(client, r_s1.json()["job_id"])
        assert j_s1["status"] == "completed", j_s1

        called_stages = {req["stage"] for req in mock_vlm.requests}
        assert called_stages == {"S1"}, f"Expected only S1 calls, got {called_stages}"

        ep_open_v2 = client.get(f"/api/episodes/{ep_open}", headers=LEAD_HEADERS).json()
        run_id_v2 = ep_open_v2["active_run_id"]
        assert run_id_v2 != run_id_v1
        assert ep_open_v2["grounding"]["schema_version"] == 2
        assert len(ep_open_v2["grounding"]["dense_times_s"]) > 0
        obj_kf0 = ep_open_v2["grounding"]["object_tracks_2d"]["target_object"]["keyframes"][0]["bbox_norm_1000"]
        assert obj_kf0 == [320, 240, 480, 420]
        assert ep_open_v2["macro_stages"] == macro_v1
        assert ep_open_v2["qc_segments"] == segs_v1
        assert ep_open_v2["success"] == succ_v1

        # Second S1-only rerun using short alias stage='s1' and fresh=True still executes S1 only
        mock_vlm.reset()
        mock_vlm.set_stage_response(
            "s1",
            {
                "objects": {
                    "target_object": {
                        "visible": True,
                        "bbox": [500, 300, 660, 480],
                        "state": "updated_alias",
                        "status_kind": "good",
                    }
                }
            },
        )
        r_s1_alias = client.post(
            "/api/jobs",
            headers=LEAD_HEADERS,
            json={
                "job_type": "batch_vlm",
                "episode_ids": [ep_open, ep_rev],
                "stage": "s1",
                "fresh": True,
            },
        )
        assert r_s1_alias.status_code == 201
        j_s1_alias = _poll_job_until_done(client, r_s1_alias.json()["job_id"])
        assert j_s1_alias["status"] == "completed", j_s1_alias

        called_alias = {req["stage"] for req in mock_vlm.requests}
        assert called_alias == {"S1"}, f"Expected only S1 calls, got {called_alias}"

        ep_open_v3 = client.get(f"/api/episodes/{ep_open}", headers=LEAD_HEADERS).json()
        run_id_v3 = ep_open_v3["active_run_id"]
        assert run_id_v3 != run_id_v2
        assert ep_open_v3["grounding"]["schema_version"] == 2
        assert ep_open_v3["grounding"]["object_tracks_2d"]["target_object"]["keyframes"][0]["bbox_norm_1000"] == [
            500,
            300,
            660,
            480,
        ]
        assert ep_open_v3["macro_stages"] == macro_v1
        assert ep_open_v3["qc_segments"] == segs_v1
        assert ep_open_v3["success"] == succ_v1

    repo_after = Repository(ws)
    try:
        art_v2 = repo_after.get_vlm_run(run_id_v2)
        art_v3 = repo_after.get_vlm_run(run_id_v3)
        for art in (art_v2, art_v3):
            assert art.stage_refs.s2_macro == art_v1.stage_refs.s2_macro
            assert art.stage_refs.s3_discover == art_v1.stage_refs.s3_discover
            assert art.stage_refs.s4_verify == art_v1.stage_refs.s4_verify
            assert art.stage_refs.s5_success == art_v1.stage_refs.s5_success
        # Protected 'reviewed' episode retains its status and active_run_id
        row_rev = repo_after.get_episode_row(ep_rev)
        assert row_rev["status"] == EpisodeStatus.REVIEWED.value
        assert row_rev["active_run_id"] == rev_run_before
    finally:
        repo_after.close()
