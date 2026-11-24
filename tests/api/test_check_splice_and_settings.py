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
        assert b2["compute_ms"] < 20.0

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
        assert b3["compute_ms"] < 20.0

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
        assert all(item["compute_ms"] < 20.0 for item in res_list), [
            item["compute_ms"] for item in res_list
        ]
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
