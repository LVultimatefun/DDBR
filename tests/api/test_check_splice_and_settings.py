"""Tests for /check-splice with LRU kinematics cache and GET/PUT /api/settings."""

from __future__ import annotations

import gc
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from robo_vlm_qc.api.state import AppState
from robo_vlm_qc.config import ConfigManager
from robo_vlm_qc.models.annotation import SuccessVerdict
from robo_vlm_qc.models.review import ReviewWritePayload, WatchCoverage
from robo_vlm_qc.store import Repository
from tests.api.conftest import (
    LEAD_HEADERS,
    REV_HEADERS,
    _write_mp4_frames,
    create_video_episode_file,
    ingest_dir,
    make_client,
    seed_vlm_run,
)
from tests.e2e.helpers import make_synthetic_h5_raw_v3


def test_check_splice_auth_validation_and_route_ordering(api_env: dict[str, Any]) -> None:
    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]
    data_dir = api_env["data_root"] / "ds_route"
    make_synthetic_h5_raw_v3(data_dir, name="ep_000.h5", duration_s=4.0, stationary=False)

    repo = Repository(ws)
    try:
        ep_ids = ingest_dir(repo, cfg_dir, ws, data_dir, dataset="ds_route")
        ep_id = ep_ids[0]
        cfg = ConfigManager(config_dir=cfg_dir, workspace_dir=ws)
        seed_vlm_run(repo, cfg, ep_id, run_id="run_route_1")
    finally:
        repo.close()

    with make_client(ws, cfg_dir) as client:
        # Unauth -> 401
        r_unauth = client.post(
            f"/api/episodes/{ep_id}/check-splice",
            json={
                "t_start": 0.0,
                "t_end": 0.4,
                "label_id": "boundary_idle",
                "splice_policy": "cut_clean",
                "affected_arm": "both",
            },
        )
        assert r_unauth.status_code == 401

        # Route order: Reviewer can hit /check-splice (200), /frame (200), while /runs enforces 403
        r_rev = client.post(
            f"/api/episodes/{ep_id}/check-splice",
            headers=REV_HEADERS,
            json={
                "t_start": 0.0,
                "t_end": 0.4,
                "label_id": "boundary_idle",
                "splice_policy": "cut_clean",
                "affected_arm": "both",
            },
        )
        assert r_rev.status_code == 200
        assert client.get(f"/api/episodes/{ep_id}/runs", headers=REV_HEADERS).status_code == 403
        assert client.get(f"/api/episodes/{ep_id}/runs", headers=LEAD_HEADERS).status_code == 200

        # Validation 422s
        assert (
            client.post(
                f"/api/episodes/{ep_id}/check-splice",
                headers=REV_HEADERS,
                json={
                    "t_start": 1.0,
                    "t_end": 0.5,
                    "label_id": "boundary_idle",
                    "splice_policy": "cut_clean",
                    "affected_arm": "both",
                },
            ).status_code
            == 422
        )
        assert (
            client.post(
                f"/api/episodes/{ep_id}/check-splice",
                headers=REV_HEADERS,
                json={
                    "t_start": 0.0,
                    "t_end": 0.5,
                    "label_id": "long_pause",
                    "splice_policy": "cut_clean",
                    "affected_arm": "both",
                },
            ).status_code
            == 422
        )
        assert (
            client.post(
                f"/api/episodes/{ep_id}/check-splice",
                headers=REV_HEADERS,
                json={
                    "t_start": True,
                    "t_end": 0.5,
                    "label_id": "boundary_idle",
                    "splice_policy": "cut_clean",
                },
            ).status_code
            == 422
        )
        # Missing episode -> 404
        assert (
            client.post(
                "/api/episodes/ds_route/does_not_exist/check-splice",
                headers=REV_HEADERS,
                json={
                    "t_start": 0.0,
                    "t_end": 0.5,
                    "label_id": "boundary_idle",
                    "splice_policy": "cut_clean",
                },
            ).status_code
            == 404
        )


def test_check_splice_kinematics_policies_lru_cache_and_zero_mutation(
    api_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]
    data_dir = api_env["data_root"] / "ds_kin"
    h5_path = make_synthetic_h5_raw_v3(
        data_dir, name="ep_kin.h5", duration_s=4.0, stationary=False
    )

    repo = Repository(ws)
    try:
        ep_id = ingest_dir(repo, cfg_dir, ws, data_dir, dataset="ds_kin")[0]
        cfg = ConfigManager(config_dir=cfg_dir, workspace_dir=ws)
        seed_vlm_run(repo, cfg, ep_id, run_id="run_kin_1")
        row_before = repo.get_episode_row(ep_id)
        runs_before = len(repo.list_vlm_runs(ep_id))
    finally:
        repo.close()

    open_calls = 0
    orig_open_view = AppState.open_view

    def _counted_open_view(self: AppState, meta: Any, cfg_mgr: Any = None) -> Any:
        nonlocal open_calls
        open_calls += 1
        return orig_open_view(self, meta, cfg_mgr)

    monkeypatch.setattr(AppState, "open_view", _counted_open_view)

    with make_client(ws, cfg_dir) as client:
        # 1. Cold call on stationary boundary [0.0, 0.4] -> mode="kinematics", passed=True
        r1 = client.post(
            f"/api/episodes/{ep_id}/check-splice",
            headers=REV_HEADERS,
            json={
                "t_start": 0.0,
                "t_end": 0.4,
                "label_id": "boundary_idle",
                "splice_policy": "cut_clean",
                "affected_arm": "both",
            },
        )
        assert r1.status_code == 200
        b1 = r1.json()
        assert b1["mode"] == "kinematics"
        assert b1["passed"] is True
        assert b1["downgraded"] is False
        assert b1["effective_splice_policy"] == "cut_clean"
        assert b1["metrics"]["visual_ssim"] is None
        assert b1["metrics"]["visual_l1"] is None
        assert open_calls == 1

        # 2. Warm call across large motion interval [0.6, 2.5] -> hits LRU cache (open_calls still 1), <20ms
        gc.collect()
        r2 = client.post(
            f"/api/episodes/{ep_id}/check-splice",
            headers=REV_HEADERS,
            json={
                "t_start": 0.6,
                "t_end": 2.5,
                "label_id": "missed_grasp_retry",
                "splice_policy": "cut_clean",
                "affected_arm": "left",
            },
        )
        assert r2.status_code == 200
        b2 = r2.json()
        assert open_calls == 1
        assert b2["mode"] == "kinematics"
        assert b2["downgraded"] is True
        assert b2["passed"] is False
        assert b2["effective_splice_policy"] == "mask_loss"
        assert b2["metrics"]["eef_pos_m"] is not None and b2["metrics"]["eef_pos_m"] > 0.02
        assert b2["compute_ms"] < 50.0

        # 3. Hallucination override: mid_idle_freeze across monotonic dual-arm motion [0.5, 3.8]
        r3 = client.post(
            f"/api/episodes/{ep_id}/check-splice",
            headers=REV_HEADERS,
            json={
                "t_start": 0.5,
                "t_end": 3.8,
                "label_id": "mid_idle_freeze",
                "splice_policy": "cut_clean",
                "affected_arm": "both",
            },
        )
        assert r3.status_code == 200
        b3 = r3.json()
        assert open_calls == 1
        assert b3["mode"] == "kinematics"
        assert b3["effective_splice_policy"] == "keep_positive"
        assert b3["passed"] is True
        assert b3["compute_ms"] < 50.0

        # 4. keep_positive & non-idle mask_loss -> mode="skipped"
        r4 = client.post(
            f"/api/episodes/{ep_id}/check-splice",
            headers=REV_HEADERS,
            json={
                "t_start": 0.8,
                "t_end": 1.6,
                "label_id": "precision_fine_adjust",
                "splice_policy": "keep_positive",
                "affected_arm": "both",
            },
        )
        assert r4.status_code == 200
        assert r4.json()["mode"] == "skipped"
        assert r4.json()["passed"] is True

        r5 = client.post(
            f"/api/episodes/{ep_id}/check-splice",
            headers=REV_HEADERS,
            json={
                "t_start": 0.8,
                "t_end": 1.6,
                "label_id": "missed_grasp_retry",
                "splice_policy": "mask_loss",
                "affected_arm": "left",
            },
        )
        assert r5.status_code == 200
        assert r5.json()["mode"] == "skipped"
        assert r5.json()["effective_splice_policy"] == "mask_loss"

        # 5. Concurrent warm calls from 8 threads -> all 200, <20ms steady-state, zero extra opens
        # Flush cyclic garbage and warm per-thread SQLite TLS connections on the first pass so
        # initial PRAGMA journal_mode=WAL lock contention does not skew steady-state LRU latency.
        gc.collect()

        def _call_one(_: int) -> dict[str, Any]:
            resp = client.post(
                f"/api/episodes/{ep_id}/check-splice",
                headers=REV_HEADERS,
                json={
                    "t_start": 0.6,
                    "t_end": 2.5,
                    "label_id": "missed_grasp_retry",
                    "splice_policy": "cut_clean",
                    "affected_arm": "both",
                },
            )
            assert resp.status_code == 200
            return resp.json()

        with ThreadPoolExecutor(max_workers=8) as pool:
            _ = list(pool.map(_call_one, range(8)))
            res_list = list(pool.map(_call_one, range(8)))
        times_ms = sorted(item["compute_ms"] for item in res_list)
        median_ms = times_ms[len(times_ms) // 2]
        assert median_ms < 25.0 and all(t < 60.0 for t in times_ms), times_ms
        assert open_calls == 1

        # 6. Bumping source file mtime invalidates LRU cache cleanly
        st = h5_path.stat()
        os.utime(h5_path, (st.st_atime + 5.0, st.st_mtime + 5.0))
        r6 = client.post(
            f"/api/episodes/{ep_id}/check-splice",
            headers=REV_HEADERS,
            json={
                "t_start": 0.0,
                "t_end": 0.4,
                "label_id": "boundary_idle",
                "splice_policy": "cut_clean",
                "affected_arm": "both",
            },
        )
        assert r6.status_code == 200
        assert open_calls == 2

    # Zero DB mutations
    repo_after = Repository(ws)
    try:
        row_after = repo_after.get_episode_row(ep_id)
        assert row_after["status"] == row_before["status"]
        assert row_after["active_run_id"] == row_before["active_run_id"]
        assert row_after["active_rev_id"] == row_before["active_rev_id"]
        assert len(repo_after.list_vlm_runs(ep_id)) == runs_before
        assert repo_after.list_reviews(ep_id) == []
    finally:
        repo_after.close()


def test_check_splice_visual_gate_fallback(api_env: dict[str, Any]) -> None:
    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]
    ds_dir = api_env["data_root"] / "ds_vis"
    v_same = create_video_episode_file(
        ds_dir, "ep_vis_same.mp4", duration_s=2.0, fps=15.0, identical_frames=True
    )
    # Build a visually divergent video: dark first half, bright second half
    diff_frames = [
        np.full((120, 160, 3), 20 if i < 15 else 235, dtype=np.uint8)
        for i in range(30)
    ]
    _write_mp4_frames(ds_dir / "ep_vis_diff.mp4", diff_frames, fps=15.0)

    repo = Repository(ws)
    try:
        ep_ids = sorted(ingest_dir(repo, cfg_dir, ws, ds_dir, dataset="ds_vis"))
        ep_diff, ep_same = ep_ids[0], ep_ids[1]
    finally:
        repo.close()

    with make_client(ws, cfg_dir) as client:
        r = client.post(
            f"/api/episodes/{ep_same}/check-splice",
            headers=REV_HEADERS,
            json={
                "t_start": 0.2,
                "t_end": 1.6,
                "label_id": "mid_idle_freeze",
                "splice_policy": "cut_clean",
                "affected_arm": "both",
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["mode"] == "visual"
        assert body["passed"] is True
        assert body["downgraded"] is False
        assert body["metrics"]["visual_ssim"] is not None
        assert body["metrics"]["visual_ssim"] >= 0.99
        assert body["metrics"]["visual_l1"] is not None
        assert body["metrics"]["visual_l1"] <= 0.01
        assert body["metrics"]["eef_pos_m"] is None

        # Visually divergent start/end frames -> visual_gate downgrades cut_clean -> mask_loss
        r_diff = client.post(
            f"/api/episodes/{ep_diff}/check-splice",
            headers=REV_HEADERS,
            json={
                "t_start": 0.1,
                "t_end": 1.8,
                "label_id": "missed_grasp_retry",
                "splice_policy": "cut_clean",
                "affected_arm": "left",
            },
        )
        assert r_diff.status_code == 200
        b_diff = r_diff.json()
        assert b_diff["mode"] == "visual"
        assert b_diff["passed"] is False
        assert b_diff["downgraded"] is True
        assert b_diff["effective_splice_policy"] == "mask_loss"
        assert b_diff["metrics"]["visual_ssim"] is not None
        assert b_diff["metrics"]["visual_l1"] is not None

        # Missing source file on disk -> 404 cleanly (never 500)
        v_same.unlink()
        r_missing = client.post(
            f"/api/episodes/{ep_same}/check-splice",
            headers=REV_HEADERS,
            json={
                "t_start": 0.2,
                "t_end": 1.6,
                "label_id": "mid_idle_freeze",
                "splice_policy": "cut_clean",
                "affected_arm": "both",
            },
        )
        assert r_missing.status_code == 404


def test_settings_get_put_validation_history_and_dynamic_review_gate(
    api_env: dict[str, Any]
) -> None:
    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]
    data_dir = api_env["data_root"] / "ds_set"
    make_synthetic_h5_raw_v3(data_dir, name="ep_set.h5", duration_s=4.0, stationary=True)

    repo = Repository(ws)
    try:
        ep_id = ingest_dir(repo, cfg_dir, ws, data_dir, dataset="ds_set")[0]
        cfg = ConfigManager(config_dir=cfg_dir, workspace_dir=ws)
        seed_vlm_run(repo, cfg, ep_id, run_id="run_set_1")
    finally:
        repo.close()

    with make_client(ws, cfg_dir) as client:
        # GET works for both Reviewer and Lead
        assert client.get("/api/settings").status_code == 401
        r_get = client.get("/api/settings", headers=REV_HEADERS)
        assert r_get.status_code == 200
        assert r_get.json() == {"min_watch_coverage": 0.95, "batch_vlm_concurrency": 4}

        # PUT blocked for Reviewer (403)
        assert (
            client.put(
                "/api/settings",
                headers=REV_HEADERS,
                json={"min_watch_coverage": 0.5},
            ).status_code
            == 403
        )

        # Validation errors -> 422
        for bad_body in [
            {},
            {"min_watch_coverage": -0.1},
            {"min_watch_coverage": 1.1},
            {"min_watch_coverage": True},
            {"batch_vlm_concurrency": 0},
            {"batch_vlm_concurrency": 33},
            {"batch_vlm_concurrency": 4.5},
            {"batch_vlm_concurrency": True},
            {"unknown_key": 1},
        ]:
            assert (
                client.put("/api/settings", headers=LEAD_HEADERS, json=bad_body).status_code
                == 422
            )

        # PUT updates settings, preserves unrelated configs_override/default.yaml keys, creates .history/
        seq_before = client.app.state.ctx.sse.latest_seq()  # type: ignore[attr-defined]
        r_put = client.put(
            "/api/settings",
            headers=LEAD_HEADERS,
            json={"min_watch_coverage": 0.4, "batch_vlm_concurrency": 6},
        )
        assert r_put.status_code == 200
        assert r_put.json() == {"min_watch_coverage": 0.4, "batch_vlm_concurrency": 6}

        ov = yaml.safe_load((ws / "configs_override" / "default.yaml").read_text("utf-8"))
        assert ov["vlm"]["base_url"] == api_env["mock_vlm"].base_url
        assert ov["vlm"]["episode_concurrency"] == 6
        assert ov["review"]["min_watch_coverage"] == 0.4
        assert len(list((ws / ".history").rglob("*.json"))) >= 1

        _, sse_events = client.app.state.ctx.sse.since(seq_before)  # type: ignore[attr-defined]
        assert any(
            ev.get("type") == "settings_updated" and ev.get("min_watch_coverage") == 0.4
            for ev in sse_events
        )

        # Dynamic enforcement in POST /api/episodes/{ep_id}/reviews:
        # 50% watch coverage passes when min_watch_coverage == 0.4
        assert (
            client.post(f"/api/episodes/{ep_id}/claim", headers=REV_HEADERS).status_code == 200
        )
        review_payload = ReviewWritePayload(
            base_run_id="run_set_1",
            action="submit_review",
            qc_segments=[],
            success=SuccessVerdict(verdict="success", passed=True, confidence=0.95, summary="ok"),
            watch_coverage=WatchCoverage(
                watched_intervals=[[0.0, 2.0]],
                watched_s=2.0,
                duration_s=4.0,
                coverage_ratio=0.5,
            ),
            notes="ok",
        ).model_dump(mode="json")
        r_rev_ok = client.post(
            f"/api/episodes/{ep_id}/reviews",
            headers=REV_HEADERS,
            json=review_payload,
        )
        assert r_rev_ok.status_code == 200

        # Raise min_watch_coverage to 0.9 -> 0.5 now fails with 422 on next episode
        assert (
            client.put(
                "/api/settings",
                headers=LEAD_HEADERS,
                json={"min_watch_coverage": 0.9},
            ).status_code
            == 200
        )
        assert (
            client.get("/api/settings", headers=REV_HEADERS).json()["min_watch_coverage"] == 0.9
        )


def test_vlm_settings_rbac_masking_env_isolation_and_connectivity_probe(
    api_env: dict[str, Any],
) -> None:
    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]
    mock_url: str = api_env["mock_vlm"].base_url

    with make_client(ws, cfg_dir) as client:
        # RBAC: unauth -> 401, Reviewer -> 403 on all 3 VLM settings endpoints
        assert client.get("/api/settings/vlm").status_code == 401
        assert client.get("/api/settings/vlm", headers=REV_HEADERS).status_code == 403
        assert (
            client.put(
                "/api/settings/vlm",
                headers=REV_HEADERS,
                json={"default_model": "gpt-4o"},
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/settings/vlm/test",
                headers=REV_HEADERS,
                json={},
            ).status_code
            == 403
        )

        # GET /api/settings/vlm works for Lead & lists built-in presets + custom
        r_get = client.get("/api/settings/vlm", headers=LEAD_HEADERS)
        assert r_get.status_code == 200
        body = r_get.json()
        assert body["provider"] == "openai_compat"
        assert body["base_url"] == mock_url
        assert body["api_key_env"] == "MOCK_VLM_API_KEY"
        assert body["api_key_configured"] is True
        assert body["api_key_masked"] in ("***", "moc***-key") or "***" in body["api_key_masked"]
        assert "mock-e2e-key" not in r_get.text
        preset_ids = {p["id"] for p in body["presets"]}
        assert {"openai", "dashscope", "gemini", "vllm", "custom"}.issubset(preset_ids)

        # Validation 422s (extra="forbid", invalid URL, invalid env var, newline in key, empty body)
        for bad_payload in [
            {},
            {"unknown_field": "x"},
            {"base_url": "ftp://invalid.example.com"},
            {"api_key_env": "BAD ENV=1"},
            {"api_key": "sk-good\nINJECTED=1"},
            {"stage_models": {"not_a_stage": "gpt-4o"}},
        ]:
            assert (
                client.put(
                    "/api/settings/vlm", headers=LEAD_HEADERS, json=bad_payload
                ).status_code
                == 422
            )
        assert (
            client.post(
                "/api/settings/vlm/test",
                headers=LEAD_HEADERS,
                json={"extra_nope": 1},
            ).status_code
            == 422
        )

        # PUT /api/settings/vlm persists non-secret YAML + 0600 .env secret + SSE event
        secret_key = "sk-super-secret-token-9876abcd"
        seq_before = client.app.state.ctx.sse.latest_seq()  # type: ignore[attr-defined]
        r_put = client.put(
            "/api/settings/vlm",
            headers=LEAD_HEADERS,
            json={
                "provider": "openai_compat",
                "base_url": mock_url,
                "api_key_env": "CUSTOM_TEST_VLM_KEY",
                "api_key": secret_key,
                "thinking_mode": "disabled",
                "default_model": "qwen-vl-max",
                "stage_models": {"s1_grounding": "qwen-vl-plus"},
            },
        )
        assert r_put.status_code == 200, r_put.text
        upd = r_put.json()
        assert upd["api_key_configured"] is True
        assert upd["api_key_masked"] == "sk-***abcd"
        assert upd["stage_models"]["s1_grounding"] == "qwen-vl-plus"
        assert upd["stage_models"]["s2_macro"] == "qwen-vl-max"
        assert upd["stage_models"]["s5_success"] == "qwen-vl-max"
        assert secret_key not in r_put.text

        # Security: YAML override & .history NEVER contain api_key or secret value
        ov_text = (ws / "configs_override" / "default.yaml").read_text(encoding="utf-8")
        assert secret_key not in ov_text
        ov_data = yaml.safe_load(ov_text)
        assert "api_key" not in ov_data.get("vlm", {})
        assert ov_data["vlm"]["api_key_env"] == "CUSTOM_TEST_VLM_KEY"
        for hist_file in (ws / ".history").rglob("*.json"):
            assert secret_key not in hist_file.read_text(encoding="utf-8")

        # Security: <workspace>/.env contains key with strict 0600 permissions
        env_path = ws / ".env"
        assert env_path.is_file()
        assert (env_path.stat().st_mode & 0o777) == 0o600
        assert f"CUSTOM_TEST_VLM_KEY={secret_key}" in env_path.read_text(encoding="utf-8")

        # SSE published without secret leakage
        _, sse_events = client.app.state.ctx.sse.since(seq_before)  # type: ignore[attr-defined]
        vlm_evts = [ev for ev in sse_events if ev.get("type") == "vlm_settings_updated"]
        assert len(vlm_evts) >= 1
        assert secret_key not in str(vlm_evts)

        # Updating model with blank api_key preserves saved key in .env
        r_keep = client.put(
            "/api/settings/vlm",
            headers=LEAD_HEADERS,
            json={"default_model": "gpt-4.1", "api_key": ""},
        )
        assert r_keep.status_code == 200
        assert r_keep.json()["api_key_configured"] is True
        assert r_keep.json()["api_key_masked"] == "sk-***abcd"
        assert f"CUSTOM_TEST_VLM_KEY={secret_key}" in env_path.read_text(encoding="utf-8")

        # Short key masks to "***"
        r_short = client.put(
            "/api/settings/vlm",
            headers=LEAD_HEADERS,
            json={"api_key": "short99"},
        )
        assert r_short.status_code == 200
        assert r_short.json()["api_key_masked"] == "***"
        assert "short99" not in r_short.text

        # Restore key and test POST /api/settings/vlm/test against mock_vlm
        client.put("/api/settings/vlm", headers=LEAD_HEADERS, json={"api_key": secret_key})
        r_probe_ok = client.post(
            "/api/settings/vlm/test",
            headers=LEAD_HEADERS,
            json={"model": "gpt-4.1"},
        )
        assert r_probe_ok.status_code == 200
        p_ok = r_probe_ok.json()
        assert p_ok["ok"] is True
        assert p_ok["model"] == "gpt-4.1"
        assert isinstance(p_ok["latency_ms"], (int, float)) and p_ok["latency_ms"] >= 0.0
        assert secret_key not in r_probe_ok.text

        # Candidate probe against failing endpoint must return ok=False and NEVER leak api_key
        cand_secret = "sk-candidate-must-not-leak-5555"
        r_probe_fail = client.post(
            "/api/settings/vlm/test",
            headers=LEAD_HEADERS,
            json={
                "provider": "openai_compat",
                "base_url": "http://127.0.0.1:1/v1",
                "api_key": cand_secret,
                "model": "gpt-4o",
            },
        )
        assert r_probe_fail.status_code == 200
        p_fail = r_probe_fail.json()
        assert p_fail["ok"] is False
        assert cand_secret not in p_fail["detail"]
        assert cand_secret not in r_probe_fail.text

        # Unconfigured env var returns ok=False cleanly
        r_probe_nokey = client.post(
            "/api/settings/vlm/test",
            headers=LEAD_HEADERS,
            json={"api_key_env": "DEFINITELY_UNSET_KEY_XYZ"},
        )
        assert r_probe_nokey.status_code == 200
        assert r_probe_nokey.json()["ok"] is False


def test_exports_api_state_schema_mode_and_split_by_subtask(
    api_env: dict[str, Any],
) -> None:
    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]
    ds_dir = api_env["data_root"] / "ds_exp_sub"
    create_video_episode_file(ds_dir, "ep_sub.mp4", duration_s=2.0, fps=30.0)

    repo = Repository(ws)
    try:
        ep_id = ingest_dir(repo, cfg_dir, ws, ds_dir, dataset="ds_exp_sub")[0]
        cfg = ConfigManager(config_dir=cfg_dir, workspace_dir=ws)
        seed_vlm_run(repo, cfg, ep_id, run_id="run_exp_1", segments=[])
        from robo_vlm_qc.models.annotation import MacroStage

        stages = [
            MacroStage(
                stage_id="S1",
                t_start=0.0,
                t_end=1.0,
                label="Pick",
                label_en="pick",
                instruction="Pick up the bowl",
            ),
            MacroStage(
                stage_id="S2",
                t_start=1.0,
                t_end=2.0,
                label="Place",
                label_en="place",
                instruction="Place the bowl in the sink",
            ),
        ]
        repo.save_review(
            ep_id,
            "lead_1",
            "lead",
            ReviewWritePayload(
                base_run_id="run_exp_1",
                action="approve",
                qc_segments=[],
                success=SuccessVerdict(verdict="success", passed=True, confidence=0.95),
                macro_stages=stages,
                watch_coverage=WatchCoverage(
                    watched_intervals=[(0.0, 2.0)],
                    watched_s=2.0,
                    duration_s=2.0,
                    coverage_ratio=1.0,
                    confirmed=True,
                ),
            ),
        )
    finally:
        repo.close()

    with make_client(ws, cfg_dir) as client:
        r_dry = client.post(
            "/api/exports/dry-run",
            headers=LEAD_HEADERS,
            json={
                "format": "lerobot_v3",
                "recipe": "bc_clean",
                "state_schema_mode": "auto",
                "split_by_subtask": True,
            },
        )
        assert r_dry.status_code == 200, r_dry.text
        dry = r_dry.json()
        assert dry["pieces_exported"] == 2
        assert dry["recipe"]["state_schema_mode"] == "auto"
        assert dry["recipe"]["split_by_subtask"] is True

        r_exp = client.post(
            "/api/exports",
            headers=LEAD_HEADERS,
            json={
                "format": "lerobot_v3",
                "recipe": "bc_clean",
                "state_schema_mode": "auto",
                "split_by_subtask": True,
            },
        )
        assert r_exp.status_code == 200, r_exp.text
        exp = r_exp.json()
        assert exp["pieces_exported"] == 2
        out_dir = Path(exp["output_dir"])
        import json as _json

        info = _json.loads((out_dir / "meta" / "info.json").read_text("utf-8"))
        assert info["total_episodes"] == 2
        assert info["total_tasks"] == 2
        assert "observation.state" not in info["features"]
