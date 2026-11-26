"""Tests for system performance, cost/token telemetry APIs, PreviewCache LRU GC, CLI doctor/gc, and zero-key demo."""

from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from pydantic import ValidationError

from robo_vlm_qc.api.app import SSESafeGZipMiddleware, create_app
from robo_vlm_qc.api.routers import episodes as episodes_router
from robo_vlm_qc.cli import main as cli_main
from robo_vlm_qc.config import ConfigManager
from robo_vlm_qc.io import get_adapter
from robo_vlm_qc.io.media import PreviewCache
from robo_vlm_qc.models.annotation import (
    ArmTimelines,
    ConfigFingerprintSnapshot,
    MacroStage,
    StageUsage,
    SuccessVerdict,
    UsageSummary,
    VlmRunArtifact,
)
from robo_vlm_qc.models.review import EpisodeStatus, ResolvedEpisode
from robo_vlm_qc.store import Repository
from robo_vlm_qc.store.db import DatabasePool

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_SCRIPT = REPO_ROOT / "scripts" / "generate_demo_dataset.py"


def _load_api_conftest() -> Any:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    conftest_path = REPO_ROOT / "tests" / "api" / "conftest.py"
    spec = importlib.util.spec_from_file_location("_ddbr_api_conftest", conftest_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_api_conftest = _load_api_conftest()
LEAD_HEADERS = _api_conftest.LEAD_HEADERS
REV_HEADERS = _api_conftest.REV_HEADERS
api_env = _api_conftest.api_env
api_mock_vlm = _api_conftest.api_mock_vlm
create_video_episode_file = _api_conftest.create_video_episode_file
ingest_dir = _api_conftest.ingest_dir
make_client = _api_conftest.make_client

__all__ = ["api_env", "api_mock_vlm"]


def _poll_job_done(client: TestClient, job_id: str, timeout_s: float = 30.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        r = client.get(f"/api/jobs/{job_id}", headers=LEAD_HEADERS)
        assert r.status_code == 200, r.text
        job = r.json()
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"Job {job_id} did not finish within {timeout_s}s")


def test_db_composite_indexes_and_idempotent_vlm_runs_migration(tmp_path: Path) -> None:
    ws_new = tmp_path / "ws_new"
    repo_new = Repository(ws_new)
    try:
        conn = repo_new.db.connect()
        indexes = {
            str(r["name"])
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
        assert "idx_episodes_task_status_updated" in indexes
        assert "idx_episodes_dataset_status" in indexes
        assert "idx_jobs_type_status_created" in indexes

        cols = {
            str(r["name"])
            for r in conn.execute("PRAGMA table_info(vlm_runs)").fetchall()
        }
        assert {"cached_tokens", "cost_usd"}.issubset(cols)
    finally:
        repo_new.close()

    # Legacy DB without cached_tokens / cost_usd upgrades idempotently
    legacy_db = tmp_path / "ws_legacy" / "robo_vlm_qc.db"
    legacy_db.parent.mkdir(parents=True, exist_ok=True)
    raw_conn = sqlite3.connect(str(legacy_db))
    try:
        raw_conn.executescript(
            """
            CREATE TABLE episodes (
                episode_id TEXT PRIMARY KEY,
                dataset TEXT NOT NULL,
                task_id TEXT NOT NULL,
                adapter_type TEXT NOT NULL,
                source_uri TEXT NOT NULL,
                source_mtime REAL NOT NULL,
                duration_s REAL NOT NULL,
                fps REAL NOT NULL,
                has_kinematics INTEGER NOT NULL,
                camera_roles_json TEXT NOT NULL,
                meta_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL,
                active_run_id TEXT,
                active_rev_id TEXT,
                claimed_by TEXT,
                claim_expires_at REAL,
                resolved_passed INTEGER,
                cut_ratio REAL,
                mask_ratio REAL,
                badcase_count INTEGER DEFAULT 0,
                unknown_count INTEGER DEFAULT 0,
                updated_at REAL NOT NULL
            );
            CREATE TABLE vlm_runs (
                run_id TEXT PRIMARY KEY,
                episode_id TEXT NOT NULL,
                is_eval_sandbox INTEGER NOT NULL DEFAULT 0,
                model_routing_json TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                label_version TEXT NOT NULL,
                task_spec_version TEXT NOT NULL,
                few_shot_pack_id TEXT,
                artifact_path TEXT NOT NULL,
                status TEXT NOT NULL,
                tokens_total INTEGER DEFAULT 0,
                elapsed_s REAL DEFAULT 0,
                error_message TEXT,
                created_at REAL NOT NULL
            );
            """
        )
        raw_conn.commit()
    finally:
        raw_conn.close()

    pool = DatabasePool(legacy_db)
    try:
        cols_after = {
            str(r["name"])
            for r in pool.connect().execute("PRAGMA table_info(vlm_runs)").fetchall()
        }
        assert "cached_tokens" in cols_after
        assert "cost_usd" in cols_after
        # Second initialize() call must remain strictly idempotent
        pool.initialize()
        cols_after_2 = {
            str(r["name"])
            for r in pool.connect().execute("PRAGMA table_info(vlm_runs)").fetchall()
        }
        assert "cached_tokens" in cols_after_2
        assert "cost_usd" in cols_after_2
    finally:
        pool.close_all()


def test_episode_and_run_token_cost_telemetry_and_usage_payload(api_env: dict[str, Any]) -> None:
    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]
    ds_dir = api_env["data_root"] / "ds_tel"
    create_video_episode_file(ds_dir, "ep_no_run.mp4", duration_s=2.0, fps=15.0)
    create_video_episode_file(ds_dir, "ep_with_run.mp4", duration_s=2.0, fps=15.0)

    repo = Repository(ws)
    try:
        ep_ids = sorted(ingest_dir(repo, cfg_dir, ws, ds_dir, dataset="ds_tel"))
        ep_no_run, ep_with_run = ep_ids[0], ep_ids[1]
        cfg = ConfigManager(config_dir=cfg_dir, workspace_dir=ws)

        has_stage_cached = "cached_tokens" in StageUsage.model_fields
        has_summary_cached = "total_cached_tokens" in UsageSummary.model_fields

        def _mk_run(run_id: str, total_tok: int, cached_tok: int, cost: float, elapsed: float) -> VlmRunArtifact:
            st_kwargs: dict[str, Any] = {
                "stage": "s3_discover",
                "model": "gpt-4o",
                "prompt_tokens": int(total_tok * 0.8),
                "completion_tokens": total_tok - int(total_tok * 0.8),
                "total_tokens": total_tok,
                "elapsed_s": elapsed,
                "estimated_cost_usd": cost,
            }
            if has_stage_cached:
                st_kwargs["cached_tokens"] = cached_tok
            u_kwargs: dict[str, Any] = {
                "stages": [StageUsage(**st_kwargs)],
                "total_tokens": total_tok,
                "total_elapsed_s": elapsed,
                "total_cost_usd": cost,
            }
            if has_summary_cached:
                u_kwargs["total_cached_tokens"] = cached_tok
            return VlmRunArtifact(
                run_id=run_id,
                episode_id=ep_with_run,
                created_at=time.time(),
                config_fingerprint=ConfigFingerprintSnapshot(
                    config_sha256=cfg.fingerprints.config_sha256,
                    model_routing={"s3_discover": "gpt-4o"},
                    prompt_bundle_version="p1",
                    label_taxonomy_version="1",
                    task_spec_version="1.0.0",
                ),
                macro_stages=[
                    MacroStage(stage_id="S1", t_start=0.0, t_end=2.0, label="Op", label_en="op")
                ],
                arm_timelines=ArmTimelines(),
                qc_segments=[],
                success=SuccessVerdict(passed=True, verdict="success", confidence=0.95),
                usage=UsageSummary(**u_kwargs),
            )

        repo.save_vlm_run_artifact(_mk_run("run_old", 1200, 300, 0.0125, 1.4))
        time.sleep(0.01)
        repo.save_vlm_run_artifact(_mk_run("run_new", 2400, 900, 0.0215, 2.3))
    finally:
        repo.close()

    with make_client(ws, cfg_dir) as client:
        r_list = client.get("/api/episodes", headers=REV_HEADERS)
        assert r_list.status_code == 200
        by_id = {it["episode_id"]: it for it in r_list.json()["items"]}

        assert by_id[ep_no_run]["tokens_total"] is None
        assert by_id[ep_no_run]["cached_tokens"] is None
        assert by_id[ep_no_run]["estimated_cost_usd"] is None

        assert by_id[ep_with_run]["tokens_total"] == 2400
        assert by_id[ep_with_run]["estimated_cost_usd"] == pytest.approx(0.0215)
        assert by_id[ep_with_run]["cached_tokens"] == 900

        # GET /api/episodes/{ep_id}/runs includes cached_tokens, estimated_cost_usd, elapsed_s
        r_runs = client.get(f"/api/episodes/{ep_with_run}/runs", headers=LEAD_HEADERS)
        assert r_runs.status_code == 200
        runs = r_runs.json()["runs"]
        assert [r["run_id"] for r in runs] == ["run_new", "run_old"]
        assert runs[0]["tokens_total"] == 2400
        assert runs[0]["estimated_cost_usd"] == pytest.approx(0.0215)
        assert runs[0]["elapsed_s"] == pytest.approx(2.3)
        assert runs[0]["cached_tokens"] == 900

        # GET /api/episodes/{ep_id} exposes usage without altering strict ResolvedEpisode schema
        r_det_none = client.get(f"/api/episodes/{ep_no_run}", headers=REV_HEADERS)
        assert r_det_none.status_code == 200
        assert r_det_none.json()["usage"] is None

        r_det = client.get(f"/api/episodes/{ep_with_run}", headers=REV_HEADERS)
        assert r_det.status_code == 200
        det = r_det.json()
        assert det["usage"] is not None
        assert det["usage"]["total_tokens"] == 2400
        assert det["usage"]["total_cached_tokens"] == 900
        assert det["usage"]["total_cost_usd"] == pytest.approx(0.0215)

        # Historical run_id returns historical run usage
        r_hist = client.get(
            f"/api/episodes/{ep_with_run}?run_id=run_old", headers=LEAD_HEADERS
        )
        assert r_hist.status_code == 200
        assert r_hist.json()["usage"]["total_tokens"] == 1200
        assert r_hist.json()["usage"]["total_cost_usd"] == pytest.approx(0.0125)

    # Confirm ResolvedEpisode itself stays extra="forbid" (did not add usage field to strict model)
    assert "usage" not in ResolvedEpisode.model_fields
    with pytest.raises(ValidationError):
        ResolvedEpisode.model_validate({"episode_id": "x", "usage": {}})


def test_vlm_settings_cost_tiers_contract(api_env: dict[str, Any]) -> None:
    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]
    with make_client(ws, cfg_dir) as client:
        r = client.get("/api/settings/vlm", headers=LEAD_HEADERS)
        assert r.status_code == 200
        body = r.json()
        tiers = body.get("cost_tiers")
        assert isinstance(tiers, list)
        assert [t["id"] for t in tiers] == ["fast", "balanced", "high_precision"]
        expected_stages = {
            "s1_grounding",
            "s2_macro",
            "s3_discover",
            "s4_verify",
            "s5_success",
        }
        for tier in tiers:
            assert tier["label"]
            assert tier["description"]
            assert tier["relative_cost_hint"]
            by_prov = tier["stage_models_by_provider"]
            assert set(by_prov.keys()) == {"openai_compat", "gemini", "dashscope"}
            for prov_models in by_prov.values():
                assert set(prov_models.keys()) == expected_stages
                assert all(isinstance(v, str) and v.strip() for v in prov_models.values())

        balanced = next(t for t in tiers if t["id"] == "balanced")
        # Balanced tier uses heterogeneous routing (cheaper S1/S2/S5, flagship S3/S4)
        oai_bal = balanced["stage_models_by_provider"]["openai_compat"]
        assert len(set(oai_bal.values())) >= 2


def test_jobs_estimate_endpoint_and_batch_vlm_plus_streaming_ingest_telemetry(
    api_env: dict[str, Any],
) -> None:
    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]
    data_root: Path = api_env["data_root"]
    ds_dir = data_root / "ds_stream_ingest"
    create_video_episode_file(ds_dir, "ep_s1.mp4", duration_s=2.0, fps=15.0)
    create_video_episode_file(ds_dir, "ep_s2.mp4", duration_s=2.0, fps=15.0)

    with make_client(ws, cfg_dir, allowed_data_roots=[data_root]) as client:
        # 1. Streaming ingest progress (#25)
        seq_before = client.app.state.ctx.sse.latest_seq()  # type: ignore[attr-defined]
        r_ing = client.post(
            "/api/jobs",
            headers=LEAD_HEADERS,
            json={
                "job_type": "ingest",
                "ingest_params": {"data_path": str(ds_dir), "dataset": "ds_stream_ingest"},
            },
        )
        assert r_ing.status_code == 201, r_ing.text
        ing_job = _poll_job_done(client, r_ing.json()["job_id"])
        assert ing_job["status"] == "completed"
        assert ing_job["summary"]["discovered"] == 2
        assert ing_job["summary"]["inserted"] == 2

        _, events = client.app.state.ctx.sse.since(seq_before)  # type: ignore[attr-defined]
        ing_updates = [
            ev["job"]
            for ev in events
            if ev.get("type") == "job_update"
            and ev.get("job", {}).get("job_id") == ing_job["job_id"]
            and isinstance(ev.get("job", {}).get("summary"), dict)
            and ev["job"]["summary"].get("current_episode_id") is not None
        ]
        assert len(ing_updates) >= 2
        seen_eps = {u["summary"]["current_episode_id"] for u in ing_updates}
        assert seen_eps == {"ds_stream_ingest/ep_s1", "ds_stream_ingest/ep_s2"}

        # 2. POST /api/jobs/estimate (RBAC, 422 invalid stage, 404 missing ep, 200 valid)
        ep_ids = ["ds_stream_ingest/ep_s1", "ds_stream_ingest/ep_s2"]
        assert client.post("/api/jobs/estimate", json={"episode_ids": ep_ids}).status_code == 401
        assert (
            client.post(
                "/api/jobs/estimate", headers=REV_HEADERS, json={"episode_ids": ep_ids}
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/jobs/estimate",
                headers=LEAD_HEADERS,
                json={"episode_ids": ep_ids, "stage": "bogus_stage"},
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/jobs/estimate",
                headers=LEAD_HEADERS,
                json={"episode_ids": ["ds_stream_ingest/does_not_exist"]},
            ).status_code
            == 404
        )
        assert (
            client.post(
                "/api/jobs/estimate",
                headers=LEAD_HEADERS,
                json={"episode_ids": ep_ids, "unexpected": True},
            ).status_code
            == 422
        )

        r_est = client.post(
            "/api/jobs/estimate",
            headers=LEAD_HEADERS,
            json={"episode_ids": ep_ids, "stage": "all", "fresh": True},
        )
        assert r_est.status_code == 200, r_est.text
        est = r_est.json()
        assert est["episode_count"] == 2
        assert est["total_duration_s"] > 0
        assert est["estimated_prompt_tokens"] > 0
        assert est["estimated_completion_tokens"] > 0
        assert est["estimated_total_tokens"] > 0
        assert est["estimated_cost_usd"] > 0.0
        assert isinstance(est["per_stage"], list) and len(est["per_stage"]) == 5
        assert isinstance(est["token_saving_active"], dict)

        r_est_s34 = client.post(
            "/api/jobs/estimate",
            headers=LEAD_HEADERS,
            json={"episode_ids": ep_ids, "stage": "s3_s4", "fresh": False},
        )
        assert r_est_s34.status_code == 200, r_est_s34.text
        assert r_est_s34.json()["stage"] == "s3_s4"

        # 3. batch_vlm running + final cost/token telemetry
        seq_vlm = client.app.state.ctx.sse.latest_seq()  # type: ignore[attr-defined]
        r_vlm = client.post(
            "/api/jobs",
            headers=LEAD_HEADERS,
            json={"job_type": "batch_vlm", "episode_ids": ep_ids, "concurrency": 1},
        )
        assert r_vlm.status_code == 201, r_vlm.text
        vlm_job = _poll_job_done(client, r_vlm.json()["job_id"])
        assert vlm_job["status"] == "completed", vlm_job
        summary = vlm_job["summary"]
        assert summary["tokens_total"] > 0
        assert summary["cached_tokens"] >= 0
        assert summary["estimated_cost_usd"] >= 0.0
        assert len(summary["results"]) == 2
        for res_item in summary["results"]:
            assert res_item["tokens_total"] > 0
            assert "cached_tokens" in res_item
            assert "estimated_cost_usd" in res_item
            assert "elapsed_s" in res_item and res_item["elapsed_s"] >= 0.0

        _, vlm_events = client.app.state.ctx.sse.since(seq_vlm)  # type: ignore[attr-defined]
        progress_summaries = [
            ev["job"]["summary"]
            for ev in vlm_events
            if ev.get("type") == "job_update"
            and ev.get("job", {}).get("job_id") == vlm_job["job_id"]
            and ev["job"].get("progress_done", 0) >= 1
            and isinstance(ev["job"].get("summary"), dict)
        ]
        assert len(progress_summaries) >= 2
        assert all(
            "tokens_total" in ps and "cached_tokens" in ps and "estimated_cost_usd" in ps
            for ps in progress_summaries
        )


def test_sse_safe_gzip_and_media_etag_304(
    api_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]
    ds_dir = api_env["data_root"] / "ds_media_cache"
    create_video_episode_file(ds_dir, "ep_m1.mp4", duration_s=2.0, fps=15.0)

    repo = Repository(ws)
    try:
        ep_id = ingest_dir(repo, cfg_dir, ws, ds_dir, dataset="ds_media_cache")[0]
    finally:
        repo.close()

    with make_client(ws, cfg_dir) as client:
        # Large JSON (>=1024 B) compresses with gzip
        r_lbl = client.get(
            "/api/labels",
            headers={**REV_HEADERS, "Accept-Encoding": "gzip"},
        )
        assert r_lbl.status_code == 200
        assert r_lbl.headers.get("content-encoding") == "gzip"
        assert "accept-encoding" in r_lbl.headers.get("vary", "").lower()
        assert len(r_lbl.json()["labels"]) > 0

        # Small JSON (<1024 B) is NOT gzipped
        r_health = client.get("/api/health", headers={"Accept-Encoding": "gzip"})
        assert r_health.status_code == 200
        assert r_health.headers.get("content-encoding") is None

        # /api/episodes/{ep_id}/frame ETag + Cache-Control + 304 BEFORE frame decode
        decode_calls = 0
        orig_decode = episodes_router._decode_jpeg_frame_sync

        def _counted_decode(*args: Any, **kwargs: Any) -> bytes:
            nonlocal decode_calls
            decode_calls += 1
            return orig_decode(*args, **kwargs)

        monkeypatch.setattr(episodes_router, "_decode_jpeg_frame_sync", _counted_decode)

        r_f1 = client.get(
            f"/api/episodes/{ep_id}/frame?t_s=0.5",
            headers={**REV_HEADERS, "Accept-Encoding": "gzip"},
        )
        assert r_f1.status_code == 200
        assert decode_calls == 1
        assert r_f1.headers.get("content-encoding") is None
        etag_f = r_f1.headers.get("etag")
        assert etag_f and etag_f.startswith('"') and etag_f.endswith('"')
        assert r_f1.headers.get("cache-control") == "private, max-age=3600"

        # 304 Not Modified must skip frame decoding entirely!
        r_f2 = client.get(
            f"/api/episodes/{ep_id}/frame?t_s=0.5",
            headers={**REV_HEADERS, "If-None-Match": etag_f},
        )
        assert r_f2.status_code == 304
        assert r_f2.content == b""
        assert r_f2.headers.get("etag") == etag_f
        assert r_f2.headers.get("cache-control") == "private, max-age=3600"
        assert decode_calls == 1

        # Different timestamp -> different ETag & decodes again
        r_f3 = client.get(
            f"/api/episodes/{ep_id}/frame?t_s=1.0",
            headers={**REV_HEADERS, "If-None-Match": etag_f},
        )
        assert r_f3.status_code == 200
        assert r_f3.headers.get("etag") != etag_f
        assert decode_calls == 2

        # /api/episodes/{ep_id}/preview ETag + Cache-Control + 304 + non-gzipped
        r_p1 = client.get(
            f"/api/episodes/{ep_id}/preview",
            headers={**REV_HEADERS, "Accept-Encoding": "gzip"},
        )
        assert r_p1.status_code == 200
        assert r_p1.headers.get("content-encoding") is None
        etag_p = r_p1.headers.get("etag")
        assert etag_p and etag_p.startswith('"') and etag_p.endswith('"')
        assert r_p1.headers.get("cache-control") == "private, max-age=86400"

        r_p2 = client.get(
            f"/api/episodes/{ep_id}/preview",
            headers={**REV_HEADERS, "If-None-Match": etag_p},
        )
        assert r_p2.status_code == 304
        assert r_p2.headers.get("etag") == etag_p
        assert r_p2.headers.get("cache-control") == "private, max-age=86400"

    # Dedicated ASGI check: text/event-stream is NEVER gzipped even on a large non-/events path
    mini = FastAPI()
    mini.add_middleware(SSESafeGZipMiddleware, minimum_size=1024)

    @mini.get("/custom-stream")
    async def _custom_sse() -> StreamingResponse:
        async def _gen():
            yield f"data: {'x' * 2048}\n\n"
            yield "data: done\n\n"

        return StreamingResponse(_gen(), media_type="text/event-stream")

    @mini.get("/api/events")
    async def _events_sse() -> StreamingResponse:
        async def _gen():
            yield f"data: {'y' * 2048}\n\n"

        return StreamingResponse(_gen(), media_type="text/event-stream")

    with TestClient(mini) as mc:
        for path in ("/custom-stream", "/api/events"):
            resp = mc.get(path, headers={"Accept-Encoding": "gzip"})
            assert resp.status_code == 200
            assert resp.headers.get("content-encoding") is None
            assert resp.text.startswith("data: ")


def test_preview_cache_lru_eviction_and_gc_cli(tmp_path: Path) -> None:
    ws = tmp_path / "ws_gc"
    cache = PreviewCache(ws, max_cache_gb=10.0)

    f1 = cache.preview_path("ds1", "ep1", "main")
    f2 = cache.preview_path("ds1", "ep2", "main")
    f3 = cache.preview_path("ds2", "ep3", "main")
    stale_partial = f1.with_suffix(".partial.mp4")

    for p in (f1, f2, f3, stale_partial):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"0" * 4096)

    # Set explicit deterministic mtimes & atimes: f1 oldest, f2 middle, f3 newest
    base_t = time.time() - 1000.0
    os.utime(f1, (base_t + 10.0, base_t + 10.0))
    os.utime(f2, (base_t + 20.0, base_t + 20.0))
    os.utime(f3, (base_t + 30.0, base_t + 30.0))

    # touch_access updates atime while preserving st_mtime_ns
    f1_mtime_before = f1.stat().st_mtime_ns
    cache.touch_access(f1)
    assert f1.stat().st_mtime_ns == f1_mtime_before
    assert f1.stat().st_atime > base_t + 100.0  # f1 is now MOST recently used!

    stats = cache.cache_stats()
    assert stats["file_count"] == 3
    assert stats["partial_count"] == 1
    assert stats["preview_bytes"] == 3 * 4096
    assert stats["partial_bytes"] == 4096

    # Dry-run evicts nothing on disk, plans removal of stale partial + f2 (now oldest atime)
    dry = cache.evict_lru(max_bytes=8192, dry_run=True)
    assert dry["dry_run"] is True
    assert dry["stale_partials_removed"] == 1
    assert dry["evicted_count"] == 1
    assert str(f2.resolve()) in dry["evicted_files"]
    assert f2.is_file() and stale_partial.is_file()

    # Apply LRU eviction protecting f2 -> evicts f3 instead (next oldest), removes empty ds2/ep3 dir
    applied = cache.evict_lru(max_bytes=8192, protect_paths={f2})
    assert applied["stale_partials_removed"] == 1
    assert not stale_partial.exists()
    assert f1.is_file() and f2.is_file()
    assert not f3.exists()
    assert applied["empty_dirs_removed"] >= 1
    assert not (cache.previews_root / "ds2").exists()

    # Auto-eviction on get_or_build when max_cache_gb is tight
    ds_dir = tmp_path / "vid_auto"
    create_video_episode_file(ds_dir, "ep_a.mp4", duration_s=1.0, fps=10.0)
    ad = get_adapter("video_dir")
    handles = ad.discover(ds_dir)
    tight_cache = PreviewCache(ws, max_cache_gb=0.000001)  # ~1 KB cap
    with ad.open(handles[0]) as view:
        built = tight_cache.get_or_build(
            view.media,
            dataset="ds_auto",
            stem="ep_a",
            camera="main",
            source_mtime=0.0,
        )
    # Newly built preview is protected; older f1 & f2 were automatically evicted!
    assert built.is_file()
    assert not f1.exists() and not f2.exists()

    # CLI robo-vlm-qc gc (--dry-run, --clear-all, --json)
    runner = CliRunner()
    res_dry = runner.invoke(
        cli_main, ["-w", str(ws), "gc", "--clear-all", "--dry-run", "--json"]
    )
    assert res_dry.exit_code == 0, res_dry.output
    dry_cli = json.loads(res_dry.output)
    assert dry_cli["ok"] is True
    assert dry_cli["evicted_count"] == 1
    assert built.is_file()

    res_clr = runner.invoke(cli_main, ["-w", str(ws), "gc", "--clear-all", "--json"])
    assert res_clr.exit_code == 0, res_clr.output
    clr_cli = json.loads(res_clr.output)
    assert clr_cli["ok"] is True
    assert clr_cli["after_files"] == 0
    assert not built.exists()

    # Config wiring in create_app()
    cfg_dir = REPO_ROOT / "configs"
    ws_cfg = tmp_path / "ws_app_cfg"
    (ws_cfg / "configs_override").mkdir(parents=True, exist_ok=True)
    (ws_cfg / "configs_override" / "default.yaml").write_text(
        yaml.safe_dump({"io": {"preview_cache": {"max_cache_gb": 4.25}}}),
        encoding="utf-8",
    )
    app = create_app(workspace=ws_cfg, config_dir=cfg_dir, no_auth=True)
    assert app.state.ctx.preview_cache.max_cache_gb == pytest.approx(4.25)
    app.state.ctx.repo.close()


def test_cli_doctor_diagnostics_masking_and_failure_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ws = tmp_path / "ws_doc"
    ws.mkdir(parents=True, exist_ok=True)

    # Seed valid SQLite WAL DB
    repo = Repository(ws)
    repo.close()

    runner = CliRunner()
    res_ok = runner.invoke(cli_main, ["-w", str(ws), "doctor", "--json"])
    assert res_ok.exit_code == 0, res_ok.output
    doc = json.loads(res_ok.output)
    assert doc["ok"] is True
    check_names = {c["name"] for c in doc["checks"]}
    assert {
        "python",
        "opencv",
        "ffmpeg",
        "ffprobe",
        "nvenc",
        "workspace",
        "sqlite_db",
        "web_dist",
        "vlm_config",
    }.issubset(check_names)
    assert doc["workspace_info"]["writable"] is True
    assert doc["workspace_info"]["free_disk_gb"] > 0
    assert doc["workspace_info"]["journal_mode"] == "wal"
    assert doc["workspace_info"]["episode_count"] == 0

    # Secret API key in <ws>/.env MUST be masked and NEVER leaked in plaintext
    secret = "sk-doctor-secret-value-99887766"
    (ws / ".env").write_text(f"OPENAI_API_KEY={secret}\n", encoding="utf-8")
    res_key = runner.invoke(cli_main, ["-w", str(ws), "doctor", "--json"])
    assert res_key.exit_code == 0, res_key.output
    assert secret not in res_key.output
    doc_key = json.loads(res_key.output)
    assert doc_key["vlm"]["api_key_configured"] is True
    assert doc_key["vlm"]["api_key_masked"] == "sk-***7766"

    # Critical failure: workspace path is a regular file -> exit 1
    bad_ws = tmp_path / "not_a_dir"
    bad_ws.write_text("file", encoding="utf-8")
    res_bad = runner.invoke(cli_main, ["-w", str(bad_ws), "doctor", "--json"])
    assert res_bad.exit_code == 1
    doc_bad = json.loads(res_bad.output)
    assert doc_bad["ok"] is False


def test_generate_demo_dataset_with_sample_run_zero_api_key_studio(tmp_path: Path) -> None:
    demo_out = tmp_path / "demo_data"
    ws = tmp_path / "demo_ws"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.pop("OPENAI_API_KEY", None)

    cmd = [
        sys.executable,
        str(DEMO_SCRIPT),
        "--out",
        str(demo_out),
        "--duration-s",
        "1.5",
        "--fps",
        "15",
        "--width",
        "160",
        "--height",
        "120",
        "--with-sample-run",
        "--workspace",
        str(ws),
        "--json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=90, check=False)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    summary = json.loads(proc.stdout)
    assert len(summary["episodes"]) == 3
    assert "sample_run" in summary
    assert len(summary["sample_run"]["seeded_runs"]) == 3

    # Re-running with --with-sample-run into the same workspace is cleanly idempotent
    proc2 = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=90, check=False)
    assert proc2.returncode == 0, f"stdout={proc2.stdout}\nstderr={proc2.stderr}"

    # Verify via Studio FastAPI app with zero auth & zero API key
    app = create_app(workspace=ws, config_dir=REPO_ROOT / "configs", no_auth=True)
    with TestClient(app) as client:
        r_list = client.get("/api/episodes")
        assert r_list.status_code == 200
        items = r_list.json()["items"]
        assert len(items) == 3
        assert all(it["status"] == EpisodeStatus.VLM_READY.value for it in items)
        assert all(it["tokens_total"] and it["tokens_total"] > 0 for it in items)
        assert all(it["cached_tokens"] and it["cached_tokens"] > 0 for it in items)
        assert all(it["estimated_cost_usd"] and it["estimated_cost_usd"] > 0 for it in items)

        # Inspect showcase episode ep_0002 (boundary_idle cut_clean + missed_grasp_retry downgraded to mask_loss)
        r_ep2 = client.get("/api/episodes/demo_pick_and_place/ep_0002")
        assert r_ep2.status_code == 200
        ep2 = r_ep2.json()
        assert ep2["status"] == "vlm_ready"
        assert ep2["guardrail_downgraded"] is True
        assert ep2["usage"] is not None
        assert ep2["usage"]["total_cached_tokens"] > 0
        assert ep2["usage"]["total_cost_usd"] > 0
        assert len(ep2["grounding"]["dense_times_s"]) >= 5
        assert "target_object" in ep2["grounding"]["object_tracks_2d"]
        assert len(ep2["macro_stages"]) >= 3
        assert len(ep2["arm_timelines"]["right"]) >= 2

        segs_by_label = {s["label_id"]: s for s in ep2["qc_segments"]}
        assert "boundary_idle" in segs_by_label
        assert segs_by_label["boundary_idle"]["splice_policy"] == "cut_clean"
        assert "missed_grasp_retry" in segs_by_label
        retry_seg = segs_by_label["missed_grasp_retry"]
        assert retry_seg["vlm_raw_policy"] == "cut_clean"
        assert retry_seg["splice_policy"] == "mask_loss"
        assert retry_seg["guardrail"]["downgraded"] is True
