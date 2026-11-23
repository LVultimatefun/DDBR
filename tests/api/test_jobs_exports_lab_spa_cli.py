"""Integration tests for jobs/SSE, exports/download, lab RBAC, SPA fallback, and serve CLI."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from robo_vlm_qc.api import create_app
from robo_vlm_qc.io import auto_detect, get_adapter
from robo_vlm_qc.models import (
    ArmTimelines,
    ConfigFingerprintSnapshot,
    EpisodeStatus,
    GroundingResult,
    MacroStage,
    SuccessVerdict,
    VlmRunArtifact,
)
from robo_vlm_qc.store import Repository


def _seed_workspace(tmp_path: Path) -> tuple[Path, str, str]:
    ws = tmp_path / "ws"
    ds_dir = tmp_path / "data" / "demo_ds"
    ds_dir.mkdir(parents=True, exist_ok=True)
    mp4 = ds_dir / "ep_001.mp4"
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (160, 120))
    assert writer.isOpened()
    try:
        for i in range(30):
            writer.write(np.full((120, 160, 3), (i * 8) % 255, dtype=np.uint8))
    finally:
        writer.release()

    h = auto_detect(ds_dir)[0]
    with get_adapter(h.adapter_type).open(h) as view:
        meta = view.meta

    repo = Repository(ws)
    try:
        repo.ingest_episode(meta)
        run = VlmRunArtifact(
            run_id="run_seed_1",
            episode_id=meta.episode_id,
            created_at=time.time(),
            config_fingerprint=ConfigFingerprintSnapshot(
                model_routing={"s1_grounding": "gpt-4o"},
                prompt_bundle_version="1.0.0",
                label_taxonomy_version="1",
                task_spec_version="1.0.0",
            ),
            grounding=GroundingResult(),
            macro_stages=[MacroStage(stage_id="S1", t_start=0.0, t_end=1.0, label="Stage 1")],
            arm_timelines=ArmTimelines(),
            qc_segments=[],
            success=SuccessVerdict(passed=True, verdict="success", confidence=0.95),
        )
        repo.save_vlm_run_artifact(run)
    finally:
        repo.close()
    return ws, meta.episode_id, "run_seed_1"


def test_jobs_cancel_sse_and_global_events(tmp_path: Path) -> None:
    ws, ep_id, _ = _seed_workspace(tmp_path)
    repo = Repository(ws)
    try:
        repo.create_job(
            "batch_vlm",
            {"episode_ids": [ep_id]},
            progress_total=1,
            job_id="j_pre_done",
            status="completed",
        )
        repo.create_job(
            "batch_vlm",
            {"episode_ids": [ep_id]},
            progress_total=3,
            job_id="j_pre_q",
            status="queued",
        )
    finally:
        repo.close()

    client = TestClient(create_app(ws, reviewer_token="r-tok", lead_token="l-tok"))
    rev_h = {"X-Auth-Token": "r-tok"}
    lead_h = {"X-Auth-Token": "l-tok"}

    # Reviewer denied on POST /api/jobs and cancel
    assert (
        client.post(
            "/api/jobs", json={"job_type": "batch_vlm", "episode_ids": []}, headers=rev_h
        ).status_code
        == 403
    )
    assert client.post("/api/jobs/j_pre_q/cancel", headers=rev_h).status_code == 403

    # Empty episode_ids creates immediate completed job
    r_empty = client.post(
        "/api/jobs", json={"job_type": "batch_vlm", "episode_ids": []}, headers=lead_h
    )
    assert r_empty.status_code in (200, 201)
    assert r_empty.json()["status"] == "completed"
    assert r_empty.json()["progress_total"] == 0

    # Invalid stage -> 422
    assert (
        client.post(
            "/api/jobs",
            json={"job_type": "batch_vlm", "episode_ids": [ep_id], "stage": "s99"},
            headers=lead_h,
        ).status_code
        == 422
    )

    # Cancel queued job
    r_cancel = client.post("/api/jobs/j_pre_q/cancel", headers=lead_h)
    assert r_cancel.status_code == 200
    assert r_cancel.json()["status"] == "cancelled"

    # SSE on terminal job emits initial frame and closes cleanly
    with client.stream("GET", "/api/jobs/j_pre_done/events?token=l-tok") as sse:
        assert sse.status_code == 200
        assert "text/event-stream" in sse.headers["content-type"]
        lines = [ln for ln in sse.iter_lines() if ln]
        assert any("j_pre_done" in ln for ln in lines)

    # Global SSE emits connected frame immediately
    with client.stream("GET", "/api/events?token=l-tok") as g_sse:
        assert g_sse.status_code == 200
        first = next(g_sse.iter_lines(), None)
        assert first is not None and "connected" in first

    # Render MP4 job executes end-to-end
    r_render = client.post(
        "/api/jobs",
        json={"job_type": "render_mp4", "episode_ids": [ep_id]},
        headers=lead_h,
    )
    assert r_render.status_code in (200, 201, 202)
    jid = r_render.json()["job_id"]
    deadline = time.time() + 30.0
    st = ""
    while time.time() < deadline:
        jd = client.get(f"/api/jobs/{jid}", headers=lead_h).json()
        st = jd["status"]
        if st in ("completed", "failed"):
            break
        time.sleep(0.1)
    assert st == "completed"
    assert list((ws / "exports" / "viz_mp4").glob("*.mp4"))


def test_exports_dry_run_sync_export_and_zip_download(tmp_path: Path) -> None:
    ws, ep_id, _ = _seed_workspace(tmp_path)
    repo = Repository(ws)
    try:
        repo.transition_status(ep_id, EpisodeStatus.APPROVED)
    finally:
        repo.close()

    client = TestClient(create_app(ws, reviewer_token="r-tok", lead_token="l-tok"))
    lead_h = {"X-Auth-Token": "l-tok"}

    r_dry = client.post(
        "/api/exports/dry-run",
        json={"format": "zero_copy_index", "recipe": "bc_clean"},
        headers=lead_h,
    )
    assert r_dry.status_code == 200
    assert r_dry.json()["episodes_included"] == 1
    assert not (ws / "exports").exists() or list((ws / "exports").iterdir()) == []

    r_exp = client.post(
        "/api/exports",
        json={"format": "zero_copy_index", "recipe": "bc_clean"},
        headers=lead_h,
    )
    assert r_exp.status_code == 200
    exp = r_exp.json()
    eid = exp["export_id"]
    assert exp["episodes_included"] == 1

    r_list = client.get("/api/exports", headers=lead_h)
    assert r_list.status_code == 200
    assert [it["export_id"] for it in r_list.json()["items"]] == [eid]

    r_dl = client.get(f"/api/exports/{eid}/download", headers=lead_h)
    assert r_dl.status_code == 200
    assert "application/zip" in r_dl.headers["content-type"]
    with zipfile.ZipFile(io.BytesIO(r_dl.content), "r") as zf:
        names = zf.namelist()
        assert "qc_index.parquet" in names
        assert "manifest.jsonl" in names

    # Nonexistent episode_ids must return 404 (not 500) on both dry-run and export trigger
    r_missing_dry = client.post(
        "/api/exports/dry-run",
        json={"format": "zero_copy_index", "recipe": "bc_clean", "episode_ids": ["no_ds/no_ep"]},
        headers=lead_h,
    )
    assert r_missing_dry.status_code == 404
    r_missing_exp = client.post(
        "/api/exports",
        json={"format": "zero_copy_index", "recipe": "bc_clean", "episode_ids": ["no_ds/no_ep"]},
        headers=lead_h,
    )
    assert r_missing_exp.status_code == 404


def test_lab_subset_spa_fallback_and_cli_serve_validation(tmp_path: Path) -> None:
    ws, _, _ = _seed_workspace(tmp_path)
    fake_dist = tmp_path / "web_dist"
    (fake_dist / "assets").mkdir(parents=True)
    (fake_dist / "index.html").write_text(
        "<html><body><div id='root'></div></body></html>", encoding="utf-8"
    )
    (fake_dist / "assets" / "app.js").write_text("console.log('ok');", encoding="utf-8")

    client = TestClient(
        create_app(ws, reviewer_token="r-tok", lead_token="l-tok", web_dist_dir=fake_dist)
    )
    rev_h = {"X-Auth-Token": "r-tok"}
    lead_h = {"X-Auth-Token": "l-tok"}

    # Lab RBAC
    assert client.get("/api/lab/labels", headers=rev_h).status_code == 403
    assert client.get("/api/lab/diagnosis", headers=rev_h).status_code == 403
    r_lbl = client.get("/api/lab/labels", headers=lead_h)
    assert r_lbl.status_code == 200
    assert len(r_lbl.json()["labels"]) >= 11
    r_diag = client.get("/api/lab/diagnosis", headers=lead_h)
    assert r_diag.status_code == 200
    diag_data = r_diag.json()
    assert "episodes_evaluated" in diag_data
    assert "total_errors" in diag_data
    assert "counts_by_type" in diag_data

    # SPA vs API 404 separation
    r_root = client.get("/")
    assert r_root.status_code == 200
    assert "text/html" in r_root.headers["content-type"]
    assert client.get("/review/demo_ds/ep_001").status_code == 200
    assert client.get("/assets/app.js").status_code == 200

    r_api_404 = client.get("/api/no_such_route", headers=lead_h)
    assert r_api_404.status_code == 404
    assert "application/json" in r_api_404.headers["content-type"]

    # CLI serve port, host default, --cors-origin, and auth validation
    clean_env = os.environ.copy()
    clean_env.pop("ROBO_VLM_QC_REVIEWER_TOKEN", None)
    clean_env.pop("ROBO_VLM_QC_LEAD_TOKEN", None)

    proc_help = subprocess.run(
        [sys.executable, "-m", "robo_vlm_qc.cli", "serve", "--help"],
        env=clean_env,
        capture_output=True,
        text=True,
        timeout=15.0,
        check=False,
    )
    assert proc_help.returncode == 0
    assert "127.0.0.1" in proc_help.stdout
    assert "--cors-origin" in proc_help.stdout

    proc_noauth_flag = subprocess.run(
        [sys.executable, "-m", "robo_vlm_qc.cli", "-w", str(ws), "serve"],
        env=clean_env,
        capture_output=True,
        text=True,
        timeout=15.0,
        check=False,
    )
    assert proc_noauth_flag.returncode == 2

    for bad_port in ("0", "-1", "70000"):
        proc = subprocess.run(
            [sys.executable, "-m", "robo_vlm_qc.cli", "-w", str(ws), "serve", "--port", bad_port],
            env=clean_env,
            capture_output=True,
            text=True,
            timeout=15.0,
            check=False,
        )
        assert proc.returncode != 0
