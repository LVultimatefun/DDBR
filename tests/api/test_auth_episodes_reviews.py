"""Integration tests for FastAPI auth, CORS, episodes, frame/preview, claims & reviews."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from robo_vlm_qc.api import create_app
from robo_vlm_qc.config import default_config_dir
from robo_vlm_qc.io import auto_detect, get_adapter
from robo_vlm_qc.models import (
    ArmTimelines,
    ConfigFingerprintSnapshot,
    GroundingResult,
    GuardrailReport,
    MacroStage,
    QcSegment,
    SuccessVerdict,
    VlmRunArtifact,
)
from robo_vlm_qc.store import Repository


def _write_video_dir(root: Path, name: str = "ep_001.mp4", n_frames: int = 30) -> Path:
    ds_dir = root / "demo_ds"
    ds_dir.mkdir(parents=True, exist_ok=True)
    mp4 = ds_dir / name
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (160, 120))
    assert writer.isOpened()
    try:
        for i in range(n_frames):
            img = np.full((120, 160, 3), (i * 8) % 255, dtype=np.uint8)
            writer.write(img)
    finally:
        writer.release()
    return ds_dir


def _seed_workspace(tmp_path: Path, *, downgraded: bool = False) -> tuple[Path, str, str]:
    ws = tmp_path / "ws"
    ds_dir = _write_video_dir(tmp_path / "data")
    handles = auto_detect(ds_dir)
    assert len(handles) == 1
    ad = get_adapter(handles[0].adapter_type)
    with ad.open(handles[0]) as view:
        meta = view.meta

    repo = Repository(ws)
    try:
        repo.ingest_episode(meta)
        segs = []
        if downgraded:
            segs.append(
                QcSegment(
                    seg_id="seg_dg",
                    t_start=0.2,
                    t_end=0.6,
                    label_id="mid_idle_freeze",
                    training_role="badcase",
                    splice_policy="mask_loss",
                    mask_side="both",
                    vlm_raw_policy="cut_clean",
                    guardrail=GuardrailReport(
                        gate_type="splice_gate",
                        passed=False,
                        downgraded=True,
                        downgrade_reason="pose jump",
                    ),
                )
            )
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
            qc_segments=segs,
            success=SuccessVerdict(passed=True, verdict="success", confidence=0.95),
        )
        repo.save_vlm_run_artifact(run)
    finally:
        repo.close()
    return ws, meta.episode_id, "run_seed_1"


def test_auth_me_shared_labels_and_no_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ROBO_VLM_QC_REVIEWER_TOKEN", raising=False)
    monkeypatch.delenv("ROBO_VLM_QC_LEAD_TOKEN", raising=False)

    ws, _, _ = _seed_workspace(tmp_path)
    app = create_app(ws, reviewer_token="r-tok", lead_token="l-tok")
    client = TestClient(app)

    assert client.get("/api/auth/me").status_code == 401
    assert client.get("/api/auth/me", headers={"X-Auth-Token": "bad"}).status_code == 401

    me_rev = client.get("/api/auth/me", headers={"X-Auth-Token": "r-tok", "X-User": "alice"})
    assert me_rev.status_code == 200
    assert me_rev.json() == {"role": "reviewer", "user": "alice"}

    me_lead = client.get("/api/auth/me?token=l-tok")
    assert me_lead.status_code == 200
    assert me_lead.json() == {"role": "lead", "user": "lead"}

    # Shared studio /api/labels works for Reviewer while /api/lab/labels is Lead-only
    assert client.get("/api/labels", headers={"X-Auth-Token": "r-tok"}).status_code == 200
    assert client.get("/api/lab/labels", headers={"X-Auth-Token": "r-tok"}).status_code == 403
    assert client.get("/api/lab/labels", headers={"X-Auth-Token": "l-tok"}).status_code == 200

    # no_auth mode grants lead access without token
    app_noauth = create_app(ws, no_auth=True)
    c_noauth = TestClient(app_noauth)
    assert c_noauth.get("/api/auth/me").json()["role"] == "lead"

    # Partial token config must NOT silently fall back to default "lead-token" / "rev-token"
    c_rev_only = TestClient(create_app(ws, reviewer_token="custom-rev-only"))
    assert (
        c_rev_only.get("/api/auth/me", headers={"X-Auth-Token": "custom-rev-only"}).status_code
        == 200
    )
    assert c_rev_only.get("/api/auth/me", headers={"X-Auth-Token": "lead-token"}).status_code == 401
    c_lead_only = TestClient(create_app(ws, lead_token="custom-lead-only"))
    assert (
        c_lead_only.get("/api/auth/me", headers={"X-Auth-Token": "custom-lead-only"}).status_code
        == 200
    )
    assert c_lead_only.get("/api/auth/me", headers={"X-Auth-Token": "rev-token"}).status_code == 401


def test_create_app_requires_tokens_when_auth_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ROBO_VLM_QC_REVIEWER_TOKEN", raising=False)
    monkeypatch.delenv("ROBO_VLM_QC_LEAD_TOKEN", raising=False)

    ws, _, _ = _seed_workspace(tmp_path)
    cfg_dir = default_config_dir()

    # No explicit tokens, no env vars, and no_auth=False MUST fail closed with ValueError
    with pytest.raises(ValueError):
        create_app(workspace=ws, config_dir=cfg_dir, no_auth=False)

    with pytest.raises(ValueError):
        create_app(workspace=ws, config_dir=cfg_dir)

    # Supplying tokens via env vars satisfies auth requirement
    monkeypatch.setenv("ROBO_VLM_QC_REVIEWER_TOKEN", "env-rev-tok")
    monkeypatch.setenv("ROBO_VLM_QC_LEAD_TOKEN", "env-lead-tok")
    c_env = TestClient(create_app(workspace=ws, config_dir=cfg_dir, no_auth=False))
    assert c_env.get("/api/auth/me", headers={"X-Auth-Token": "env-rev-tok"}).json()["role"] == "reviewer"
    assert c_env.get("/api/auth/me", headers={"X-Auth-Token": "env-lead-tok"}).json()["role"] == "lead"


def test_cors_origins_parameter_and_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ROBO_VLM_QC_CORS_ORIGINS", raising=False)
    ws, _, _ = _seed_workspace(tmp_path)
    cfg_dir = default_config_dir()

    # 1. Explicit cors_origins parameter allows configured origins and rejects disallowed origins
    app_explicit = create_app(
        workspace=ws,
        config_dir=cfg_dir,
        reviewer_token="r-tok",
        lead_token="l-tok",
        cors_origins=["https://studio.example.com", "http://localhost:3000"],
    )
    c_explicit = TestClient(app_explicit)

    r_ok = c_explicit.get("/api/health", headers={"Origin": "https://studio.example.com"})
    assert r_ok.status_code == 200
    assert r_ok.headers.get("access-control-allow-origin") == "https://studio.example.com"
    assert r_ok.headers.get("access-control-allow-credentials") == "true"

    r_pre_ok = c_explicit.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert r_pre_ok.status_code == 200
    assert r_pre_ok.headers.get("access-control-allow-origin") == "http://localhost:3000"

    r_deny = c_explicit.get("/api/health", headers={"Origin": "https://evil.example.com"})
    assert r_deny.status_code == 200
    assert "access-control-allow-origin" not in r_deny.headers

    r_pre_deny = c_explicit.options(
        "/api/health",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert r_pre_deny.status_code == 400
    assert "access-control-allow-origin" not in r_pre_deny.headers

    # 2. ROBO_VLM_QC_CORS_ORIGINS env var configures allowed origins when cors_origins is omitted
    monkeypatch.setenv(
        "ROBO_VLM_QC_CORS_ORIGINS",
        "https://env-a.example.com, https://env-b.example.com",
    )
    app_env = create_app(
        workspace=ws,
        config_dir=cfg_dir,
        reviewer_token="r-tok",
        lead_token="l-tok",
    )
    c_env = TestClient(app_env)

    for allowed in ("https://env-a.example.com", "https://env-b.example.com"):
        r_env_ok = c_env.get("/api/health", headers={"Origin": allowed})
        assert r_env_ok.status_code == 200
        assert r_env_ok.headers.get("access-control-allow-origin") == allowed

    r_env_deny = c_env.get("/api/health", headers={"Origin": "https://other.example.com"})
    assert r_env_deny.status_code == 200
    assert "access-control-allow-origin" not in r_env_deny.headers

    r_env_pre_deny = c_env.options(
        "/api/health",
        headers={
            "Origin": "https://other.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert r_env_pre_deny.status_code == 400
    assert "access-control-allow-origin" not in r_env_pre_deny.headers

    # 3. Default localhost origins when neither param nor env var is set
    monkeypatch.delenv("ROBO_VLM_QC_CORS_ORIGINS", raising=False)
    app_default = create_app(
        workspace=ws,
        config_dir=cfg_dir,
        reviewer_token="r-tok",
        lead_token="l-tok",
    )
    c_default = TestClient(app_default)
    r_local = c_default.get("/api/health", headers={"Origin": "http://127.0.0.1:5173"})
    assert r_local.status_code == 200
    assert r_local.headers.get("access-control-allow-origin") == "http://127.0.0.1:5173"
    r_ext = c_default.get("/api/health", headers={"Origin": "https://untrusted.example.com"})
    assert "access-control-allow-origin" not in r_ext.headers


def test_episodes_detail_frame_preview_and_ingested_fallback(tmp_path: Path) -> None:
    ws, ep_id, _ = _seed_workspace(tmp_path, downgraded=True)

    # Add an ingested-only episode (no VLM run yet)
    ds2 = _write_video_dir(tmp_path / "data2", name="ep_ing.mp4", n_frames=15)
    h2 = auto_detect(ds2)[0]
    with get_adapter(h2.adapter_type).open(h2) as v2:
        meta2 = v2.meta.model_copy(update={"dataset": "ds_ing", "episode_id": "ds_ing/ep_ing"})
    repo = Repository(ws)
    try:
        repo.ingest_episode(meta2)
    finally:
        repo.close()

    client = TestClient(create_app(ws, reviewer_token="r-tok", lead_token="l-tok"))
    hdr = {"X-Auth-Token": "r-tok"}

    # Filter guardrail_downgraded
    r_dg = client.get("/api/episodes?guardrail_downgraded=true", headers=hdr)
    assert r_dg.status_code == 200
    assert [it["episode_id"] for it in r_dg.json()["items"]] == [ep_id]

    # Detail for ingested episode succeeds cleanly via resolve_episode_for_api
    r_ing = client.get(f"/api/episodes/{meta2.episode_id}", headers=hdr)
    assert r_ing.status_code == 200
    assert r_ing.json()["status"] == "ingested"
    assert r_ing.json()["active_run_id"] is None

    # Frame read (clamped & invalid camera)
    r_fr = client.get(f"/api/episodes/{ep_id}/frame?camera=main&t_s=99.0", headers=hdr)
    assert r_fr.status_code == 200
    assert r_fr.headers["content-type"].startswith("image/jpeg")
    assert r_fr.content[:2] == b"\xff\xd8"

    r_bad_cam = client.get(f"/api/episodes/{ep_id}/frame?camera=nope&t_s=0.1", headers=hdr)
    assert r_bad_cam.status_code == 400

    # Preview MP4
    r_prev = client.get(f"/api/episodes/{ep_id}/preview?token=r-tok")
    assert r_prev.status_code in (200, 206)
    assert "video" in r_prev.headers["content-type"]


def test_claim_lease_and_review_validation_and_gt_frozen_409(tmp_path: Path) -> None:
    ws, ep_id, run_id = _seed_workspace(tmp_path)
    client = TestClient(create_app(ws, reviewer_token="r-tok", lead_token="l-tok"))
    rev_h = {"X-Auth-Token": "r-tok", "X-User": "alice"}
    lead_h = {"X-Auth-Token": "l-tok", "X-User": "lead1"}

    assert (
        client.post(f"/api/episodes/{ep_id}/claim", json={"user": "alice"}, headers=rev_h).status_code
        == 200
    )
    assert (
        client.post(f"/api/episodes/{ep_id}/claim", json={"user": "bob"}, headers=rev_h).status_code
        == 409
    )

    base_body = {
        "base_run_id": run_id,
        "action": "submit_review",
        "qc_segments": [],
        "success": {"passed": True, "verdict": "success", "confidence": 0.9, "criteria": []},
        "macro_stages": [{"stage_id": "S1", "t_start": 0.0, "t_end": 1.0, "label": "Stage 1"}],
        "watch_coverage": {
            "watched_intervals": [[0.0, 1.0]],
            "watched_s": 1.0,
            "duration_s": 1.0,
            "coverage_ratio": 1.0,
            "confirmed": True,
        },
    }

    # author key forbidden by extra="forbid"
    with_author = {**base_body, "author": "alice"}
    assert (
        client.post(f"/api/episodes/{ep_id}/reviews", json=with_author, headers=rev_h).status_code
        == 422
    )

    # Claimer mismatch (bob tries to submit while claimed by alice)
    assert (
        client.post(
            f"/api/episodes/{ep_id}/reviews",
            json=base_body,
            headers={"X-Auth-Token": "r-tok", "X-User": "bob"},
        ).status_code
        == 409
    )

    # Low watch coverage -> 422
    low_cov = {
        **base_body,
        "watch_coverage": {
            "watched_intervals": [[0.0, 0.1]],
            "watched_s": 0.1,
            "duration_s": 1.0,
            "coverage_ratio": 0.1,
            "confirmed": False,
        },
    }
    assert (
        client.post(f"/api/episodes/{ep_id}/reviews", json=low_cov, headers=rev_h).status_code
        == 422
    )

    # Reviewer submit succeeds
    assert (
        client.post(f"/api/episodes/{ep_id}/reviews", json=base_body, headers=rev_h).status_code
        == 200
    )

    # Lead freezes GT
    freeze_body = {**base_body, "action": "freeze_gt"}
    assert (
        client.post(f"/api/episodes/{ep_id}/reviews", json=freeze_body, headers=lead_h).status_code
        == 200
    )

    # Subsequent mutations on gt_frozen return 409 (NOT 423), including release
    r_mut = client.post(
        f"/api/episodes/{ep_id}/reviews",
        json={**base_body, "action": "save_draft"},
        headers=lead_h,
    )
    assert r_mut.status_code == 409
    r_claim = client.post(f"/api/episodes/{ep_id}/claim", json={"user": "lead1"}, headers=lead_h)
    assert r_claim.status_code == 409
    r_release = client.post(
        f"/api/episodes/{ep_id}/release", json={"user": "lead1"}, headers=lead_h
    )
    assert r_release.status_code == 409
    _, sse_events = client.app.state.ctx.sse.since(0)
    assert not any(ev.get("type") == "episode_released" for ev in sse_events)

    # Reviews history returns chronological items
    r_hist = client.get(f"/api/episodes/{ep_id}/reviews", headers=lead_h)
    assert r_hist.status_code == 200
    assert [it["action"] for it in r_hist.json()["items"]] == ["submit_review", "freeze_gt"]
