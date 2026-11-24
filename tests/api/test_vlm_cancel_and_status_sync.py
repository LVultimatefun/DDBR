"""Regression tests for mid-flight VLM cancellation, episode status rollback/sync, SSE events, and no_auth."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from robo_vlm_qc.api import create_app
from robo_vlm_qc.config import ConfigManager
from robo_vlm_qc.io import auto_detect, get_adapter
from robo_vlm_qc.models.annotation import QcSegment, SuccessVerdict
from robo_vlm_qc.models.review import EpisodeStatus, ReviewWritePayload, WatchCoverage
from robo_vlm_qc.store import Repository
from robo_vlm_qc.vlm.client import VlmCancelledError, VlmClient, VlmError
from robo_vlm_qc.vlm.dag import VlmDag
from robo_vlm_qc.vlm.stage4_verify import run_stage4a_merge, run_stage4b_verify
from tests.api.conftest import (
    LEAD_HEADERS,
    create_video_episode_file,
    ingest_dir,
    make_client,
    seed_vlm_run,
)
from tests.e2e.conftest import MockVlmServer
from tests.e2e.helpers import make_synthetic_h5_raw_v3


class _SlowHttpServer:
    """Minimal HTTP server that blocks on POST until released or timeout."""

    def __init__(self, *, sleep_s: float = 5.0) -> None:
        self.sleep_s = sleep_s
        self.started = threading.Event()
        self.release = threading.Event()
        self.request_count = 0
        self._lock = threading.Lock()
        srv = self

        class _H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A002
                return

            def do_POST(self) -> None:  # noqa: N802
                n = int(self.headers.get("Content-Length", "0"))
                if n > 0:
                    self.rfile.read(n)
                with srv._lock:
                    srv.request_count += 1
                srv.started.set()
                srv.release.wait(timeout=srv.sleep_s)
                payload = json.dumps(
                    {
                        "choices": [{"message": {"content": '{"ok": true}'}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    }
                ).encode("utf-8")
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except OSError:
                    pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _H)
        self.base_url = f"http://127.0.0.1:{self._httpd.server_address[1]}/v1"
        self._t = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._t.start()

    def stop(self) -> None:
        self.release.set()
        self._httpd.shutdown()
        self._httpd.server_close()
        self._t.join(timeout=2.0)


def test_vlm_client_cancel_aborts_inflight_http_immediately() -> None:
    srv = _SlowHttpServer(sleep_s=5.0)
    try:
        ev = threading.Event()
        client = VlmClient.from_config(
            {
                "base_url": srv.base_url,
                "retries": 3,
                "retry_min_wait_s": 0.5,
                "timeout_s": 10.0,
            },
            cancel_event=ev,
        )
        client._explicit_api_key = "test-key"
        raised: list[BaseException] = []

        def _worker() -> None:
            try:
                client.chat_json(stage="s1_grounding", user_prompt="slow call")
            except BaseException as exc:  # noqa: BLE001
                raised.append(exc)

        t0 = time.monotonic()
        th = threading.Thread(target=_worker)
        th.start()
        assert srv.started.wait(timeout=3.0), "HTTP request never reached mock server"

        cancel_t0 = time.monotonic()
        client.cancel()
        th.join(timeout=2.0)
        cancel_elapsed = time.monotonic() - cancel_t0
        total_elapsed = time.monotonic() - t0

        assert not th.is_alive(), "VlmClient thread hung after cancel()"
        assert len(raised) == 1 and isinstance(raised[0], VlmCancelledError)
        assert cancel_elapsed < 1.0, f"cancel() took too long: {cancel_elapsed:.3f}s"
        assert total_elapsed < 2.5
        assert srv.request_count == 1, "Cancelled request must not be retried"

        # Pre-cancelled client refuses new calls without making any HTTP request
        with pytest.raises(VlmCancelledError):
            client.chat_json(stage="s2_macro", user_prompt="should not send")
        assert srv.request_count == 1
    finally:
        srv.stop()


def test_stage4_propagates_vlm_cancelled_error_without_fallback(tmp_path: Path) -> None:
    ds_dir = tmp_path / "ds_s4"
    create_video_episode_file(ds_dir, "ep_s4.mp4", duration_s=2.0, fps=10.0)
    handle = auto_detect(ds_dir, adapter_name="video_dir")[0]
    cfg = ConfigManager()
    cands = [
        QcSegment(
            seg_id="c1",
            t_start=0.2,
            t_end=1.2,
            label_id="mid_idle_freeze",
            training_role="badcase",
            splice_policy="mask_loss",
            mask_side="both",
            affected_arm="both",
            vlm_raw_policy="mask_loss",
        )
    ]

    class _CancelOnCallClient:
        def __init__(self) -> None:
            self.cancelled = False

        def is_cancelled(self) -> bool:
            return self.cancelled

        def chat_json(self, **_kwargs: Any) -> dict[str, Any]:
            self.cancelled = True
            raise VlmCancelledError("cancelled inside stage 4")

    with get_adapter("video_dir").open(handle) as view:
        with pytest.raises(VlmCancelledError):
            run_stage4a_merge(
                view=view,
                candidates=cands,
                client=_CancelOnCallClient(),  # type: ignore[arg-type]
                cfg=cfg,
            )
        with pytest.raises(VlmCancelledError):
            run_stage4b_verify(
                view=view,
                candidates=cands,
                client=_CancelOnCallClient(),  # type: ignore[arg-type]
                cfg=cfg,
            )


def test_dag_cancel_rollback_and_repository_restore_atomic(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ds_dir = tmp_path / "data" / "ds_dag"
    create_video_episode_file(ds_dir, "ep_ing.mp4", duration_s=2.0, fps=10.0)
    create_video_episode_file(ds_dir, "ep_rdy.mp4", duration_s=2.0, fps=10.0)
    create_video_episode_file(ds_dir, "ep_prot.mp4", duration_s=2.0, fps=10.0)

    cfg = ConfigManager(workspace_dir=ws)
    repo = Repository(ws)
    try:
        ep_ids = sorted(ingest_dir(repo, cfg.config_dir, ws, ds_dir, dataset="ds_dag"))
        ep_ing, ep_prot, ep_rdy = ep_ids[0], ep_ids[1], ep_ids[2]
        seed_vlm_run(repo, cfg, ep_rdy, run_id="run_existing_rdy")
        seed_vlm_run(repo, cfg, ep_prot, run_id="run_existing_prot")
        repo.claim_episode(ep_prot, "reviewer_a")

        # 1. queue_vlm_episodes only transitions ingested & vlm_ready; leaves claimed untouched
        queued = repo.queue_vlm_episodes([ep_ing, ep_rdy, ep_prot])
        assert dict(queued) == {
            ep_ing: EpisodeStatus.INGESTED,
            ep_rdy: EpisodeStatus.VLM_READY,
        }
        assert repo.get_episode_row(ep_ing)["status"] == EpisodeStatus.VLM_QUEUED.value
        assert repo.get_episode_row(ep_rdy)["status"] == EpisodeStatus.VLM_QUEUED.value
        assert repo.get_episode_row(ep_prot)["status"] == EpisodeStatus.CLAIMED.value

        # 2. VlmDag rollback when prev_status was VLM_QUEUED restores cleanly to INGESTED (no active_run_id)
        #    and to VLM_READY (has active_run_id) -- NEVER left stuck in VLM_QUEUED!
        cancel_ev = threading.Event()
        emitted: list[tuple[str, EpisodeStatus]] = []

        class _CancelAfterS1Client:
            def __init__(self, ev: threading.Event) -> None:
                from robo_vlm_qc.vlm.client import BudgetTracker

                self.budget = BudgetTracker()
                self._ev = ev

            def is_cancelled(self) -> bool:
                return self._ev.is_set()

            def model_for_stage(self, stage: str, override: str | None = None) -> str:
                return override or "fake-model"

            def chat_json(self, **kwargs: Any) -> dict[str, Any]:
                if kwargs.get("stage") == "s1_grounding":
                    self._ev.set()
                    return {"objects": {}}
                raise AssertionError(f"Should not reach stage {kwargs.get('stage')} after cancel")

        h_ing = [h for h in auto_detect(ds_dir) if h.stem == "ep_ing"][0]
        with get_adapter(h_ing.adapter_type).open(h_ing) as view_ing:
            view_ing.meta = repo.get_episode_meta(ep_ing)
            dag_ing = VlmDag(
                workspace=ws,
                config=cfg,
                client=_CancelAfterS1Client(cancel_ev),  # type: ignore[arg-type]
                repo=repo,
            )
            with pytest.raises(VlmCancelledError):
                dag_ing.run_episode(
                    view_ing,
                    cancel_event=cancel_ev,
                    on_status_change=lambda eid, st: emitted.append((eid, st)),
                )
        row_ing = repo.get_episode_row(ep_ing)
        assert row_ing["status"] == EpisodeStatus.INGESTED.value
        assert row_ing["active_run_id"] is None
        assert emitted == [
            (ep_ing, EpisodeStatus.VLM_RUNNING),
            (ep_ing, EpisodeStatus.INGESTED),
        ]

        # 3. Same for ep_rdy (currently VLM_QUEUED, has active_run_id="run_existing_rdy"):
        #    even on non-cancel VlmError, rollback restores VLM_READY (not VLM_QUEUED!)
        emitted_rdy: list[tuple[str, EpisodeStatus]] = []

        class _FailS1Client:
            def __init__(self) -> None:
                from robo_vlm_qc.vlm.client import BudgetTracker

                self.budget = BudgetTracker()

            def model_for_stage(self, stage: str, override: str | None = None) -> str:
                return override or "fake-model"

            def chat_json(self, **_kwargs: Any) -> dict[str, Any]:
                raise VlmError("simulated failure")

        h_rdy = [h for h in auto_detect(ds_dir) if h.stem == "ep_rdy"][0]
        with get_adapter(h_rdy.adapter_type).open(h_rdy) as view_rdy:
            view_rdy.meta = repo.get_episode_meta(ep_rdy)
            dag_rdy = VlmDag(
                workspace=ws,
                config=cfg,
                client=_FailS1Client(),  # type: ignore[arg-type]
                repo=repo,
            )
            with pytest.raises(VlmError):
                dag_rdy.run_episode(
                    view_rdy,
                    fresh=True,
                    on_status_change=lambda eid, st: emitted_rdy.append((eid, st)),
                )
        row_rdy = repo.get_episode_row(ep_rdy)
        assert row_rdy["status"] == EpisodeStatus.VLM_READY.value
        assert row_rdy["active_run_id"] == "run_existing_rdy"
        assert emitted_rdy == [
            (ep_rdy, EpisodeStatus.VLM_RUNNING),
            (ep_rdy, EpisodeStatus.VLM_READY),
        ]

        # 4. restore_vlm_episodes_on_cancel with fallback (prev_statuses=None) & protected safety
        repo.transition_status(ep_ing, EpisodeStatus.VLM_QUEUED)
        repo.transition_status(ep_rdy, EpisodeStatus.VLM_RUNNING)
        restored = repo.restore_vlm_episodes_on_cancel([ep_ing, ep_rdy, ep_prot, "missing/ep"])
        assert restored == [
            (ep_ing, EpisodeStatus.INGESTED),
            (ep_rdy, EpisodeStatus.VLM_READY),
        ]
        assert repo.get_episode_row(ep_prot)["status"] == EpisodeStatus.CLAIMED.value

        # 5. update_job must never overwrite a cancelled job back to running/completed
        j = repo.create_job("batch_vlm", {"episode_ids": [ep_ing]}, progress_total=1)
        repo.cancel_job(j["job_id"])
        after_upd = repo.update_job(j["job_id"], status="running", progress_done=1)
        assert after_upd["status"] == "cancelled"
        assert repo.get_job(j["job_id"])["status"] == "cancelled"
    finally:
        repo.close()


def test_api_cancel_midflight_restores_ingested_and_vlm_ready_and_emits_sse(
    api_env: dict[str, Any],
) -> None:
    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]
    mock_vlm: MockVlmServer = api_env["mock_vlm"]
    ds_dir = api_env["data_root"] / "ds_cancel_api"

    make_synthetic_h5_raw_v3(ds_dir, name="ep_01_ing.h5", duration_s=2.0, stationary=True)
    make_synthetic_h5_raw_v3(ds_dir, name="ep_02_rdy.h5", duration_s=2.0, stationary=True)
    make_synthetic_h5_raw_v3(ds_dir, name="ep_03_claimed.h5", duration_s=2.0, stationary=True)
    make_synthetic_h5_raw_v3(ds_dir, name="ep_04_frozen.h5", duration_s=2.0, stationary=True)

    repo = Repository(ws)
    try:
        ep_ids = sorted(ingest_dir(repo, cfg_dir, ws, ds_dir, dataset="ds_cancel_api"))
        ep_ing, ep_rdy, ep_claimed, ep_frozen = ep_ids
        cfg = ConfigManager(config_dir=cfg_dir, workspace_dir=ws)

        seed_vlm_run(repo, cfg, ep_rdy, run_id="run_keep_rdy")
        seed_vlm_run(repo, cfg, ep_claimed, run_id="run_keep_claimed")
        repo.claim_episode(ep_claimed, "reviewer_1")

        seed_vlm_run(repo, cfg, ep_frozen, run_id="run_keep_frozen")
        repo.claim_episode(ep_frozen, "lead_1")
        repo.save_review(
            ep_frozen,
            author="lead_1",
            role="lead",
            payload=ReviewWritePayload(
                base_run_id="run_keep_frozen",
                action="freeze_gt",
                qc_segments=[],
                success=SuccessVerdict(verdict="success", passed=True, confidence=0.95),
                watch_coverage=WatchCoverage(
                    watched_intervals=[[0.0, 2.0]],
                    watched_s=2.0,
                    duration_s=2.0,
                    coverage_ratio=1.0,
                ),
            ),
            min_watch_coverage=0.95,
        )
    finally:
        repo.close()

    # Block S1 in-flight on MockVlmServer until test releases or client aborts socket
    s1_entered = threading.Event()
    s1_release = threading.Event()

    def _blocking_s1(_rec: dict[str, Any]) -> dict[str, Any]:
        s1_entered.set()
        s1_release.wait(timeout=6.0)
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

    mock_vlm.set_stage_response("s1", _blocking_s1)

    try:
        with make_client(ws, cfg_dir) as client:
            sse_start = client.app.state.ctx.sse.latest_seq()  # type: ignore[attr-defined]

            # Launch single-worker batch with ingested + vlm_ready + protected episodes
            # concurrency=1 ensures ep_ing is running in S1 while ep_rdy is still vlm_queued!
            r_post = client.post(
                "/api/jobs",
                headers=LEAD_HEADERS,
                json={
                    "job_type": "batch_vlm",
                    "episode_ids": [ep_ing, ep_rdy, ep_claimed, ep_frozen],
                    "fresh": True,
                    "concurrency": 1,
                },
            )
            assert r_post.status_code == 201
            job_data = r_post.json()
            job_id = job_data["job_id"]
            # Persisted params must NOT be polluted with internal tracking keys
            assert set(job_data["params"].keys()) >= {"job_type", "episode_ids"}
            assert "prev_statuses" not in job_data["params"]

            # Wait until S1 HTTP request is actively in-flight for ep_ing
            assert s1_entered.wait(timeout=5.0), "Worker never entered S1"

            # Verify live DB/API statuses BEFORE cancel:
            # ep_ing is vlm_running, ep_rdy is vlm_queued, protected ones untouched!
            assert client.get(f"/api/episodes/{ep_ing}", headers=LEAD_HEADERS).json()["status"] == "vlm_running"
            assert client.get(f"/api/episodes/{ep_rdy}", headers=LEAD_HEADERS).json()["status"] == "vlm_queued"
            assert client.get(f"/api/episodes/{ep_claimed}", headers=LEAD_HEADERS).json()["status"] == "claimed"
            assert client.get(f"/api/episodes/{ep_frozen}", headers=LEAD_HEADERS).json()["status"] == "gt_frozen"

            # Click Cancel mid-flight!
            t_cancel = time.monotonic()
            r_cancel = client.post(f"/api/jobs/{job_id}/cancel", headers=LEAD_HEADERS)
            assert r_cancel.status_code == 200
            assert r_cancel.json()["status"] == "cancelled"
            assert (time.monotonic() - t_cancel) < 1.5

            # IMMEDIATELY after cancel returns, both episodes must already be restored!
            det_ing = client.get(f"/api/episodes/{ep_ing}", headers=LEAD_HEADERS).json()
            det_rdy = client.get(f"/api/episodes/{ep_rdy}", headers=LEAD_HEADERS).json()
            det_claimed = client.get(f"/api/episodes/{ep_claimed}", headers=LEAD_HEADERS).json()
            det_frozen = client.get(f"/api/episodes/{ep_frozen}", headers=LEAD_HEADERS).json()

            assert det_ing["status"] == "ingested"
            assert det_ing["active_run_id"] is None
            assert det_rdy["status"] == "vlm_ready"
            assert det_rdy["active_run_id"] == "run_keep_rdy"
            assert det_claimed["status"] == "claimed"
            assert det_frozen["status"] == "gt_frozen"

            # Wait briefly for background worker thread to finish exiting cleanly, and verify
            # status NEVER flips to vlm_ready afterwards and AppState control is cleaned up
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if not client.app.state.ctx.has_job_control(job_id):  # type: ignore[attr-defined]
                    break
                time.sleep(0.05)
            assert not client.app.state.ctx.has_job_control(job_id)  # type: ignore[attr-defined]

            # Re-check after worker exit: still cancelled, still ingested / vlm_ready, no new runs
            assert client.get(f"/api/jobs/{job_id}", headers=LEAD_HEADERS).json()["status"] == "cancelled"
            assert client.get(f"/api/episodes/{ep_ing}", headers=LEAD_HEADERS).json()["status"] == "ingested"
            assert client.get(f"/api/episodes/{ep_ing}", headers=LEAD_HEADERS).json()["active_run_id"] is None
            assert client.get(f"/api/episodes/{ep_rdy}", headers=LEAD_HEADERS).json()["status"] == "vlm_ready"
            assert client.get(f"/api/episodes/{ep_rdy}", headers=LEAD_HEADERS).json()["active_run_id"] == "run_keep_rdy"

            # Verify SSE stream emitted episode_updated transitions AND cancelled job_update
            _, events = client.app.state.ctx.sse.since(sse_start)  # type: ignore[attr-defined]
            ep_events = [
                (ev["episode_id"], ev["status"])
                for ev in events
                if ev.get("type") == "episode_updated"
            ]
            assert (ep_ing, "vlm_queued") in ep_events
            assert (ep_ing, "vlm_running") in ep_events
            assert (ep_ing, "ingested") in ep_events
            assert (ep_rdy, "vlm_queued") in ep_events
            assert (ep_rdy, "vlm_ready") in ep_events
            assert not any(eid in (ep_claimed, ep_frozen) for eid, _ in ep_events)

            job_statuses = [
                ev["job"]["status"]
                for ev in events
                if ev.get("type") == "job_update" and ev["job"]["job_id"] == job_id
            ]
            assert "cancelled" in job_statuses
            assert job_statuses[-1] == "cancelled"
    finally:
        s1_release.set()


def test_api_cancel_midflight_rerun_on_vlm_ready_preserves_active_run(
    api_env: dict[str, Any],
) -> None:
    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]
    mock_vlm: MockVlmServer = api_env["mock_vlm"]
    ds_dir = api_env["data_root"] / "ds_rerun_cancel"

    make_synthetic_h5_raw_v3(ds_dir, name="ep_rerun.h5", duration_s=2.0, stationary=True)
    repo = Repository(ws)
    try:
        ep_id = ingest_dir(repo, cfg_dir, ws, ds_dir, dataset="ds_rerun_cancel")[0]
        cfg = ConfigManager(config_dir=cfg_dir, workspace_dir=ws)
        seed_vlm_run(repo, cfg, ep_id, run_id="run_original_v1")
    finally:
        repo.close()

    s1_entered = threading.Event()
    s1_release = threading.Event()

    def _blocking_s1(_rec: dict[str, Any]) -> dict[str, Any]:
        s1_entered.set()
        s1_release.wait(timeout=6.0)
        return {"objects": {}}

    mock_vlm.set_stage_response("s1", _blocking_s1)

    try:
        with make_client(ws, cfg_dir) as client:
            r_post = client.post(
                "/api/jobs",
                headers=LEAD_HEADERS,
                json={"job_type": "batch_vlm", "episode_ids": [ep_id], "stage": "S1-S5"},
            )
            assert r_post.status_code == 201
            job_id = r_post.json()["job_id"]
            assert s1_entered.wait(timeout=5.0)
            assert client.get(f"/api/episodes/{ep_id}", headers=LEAD_HEADERS).json()["status"] == "vlm_running"

            r_cancel = client.post(f"/api/jobs/{job_id}/cancel", headers=LEAD_HEADERS)
            assert r_cancel.status_code == 200
            assert r_cancel.json()["status"] == "cancelled"

            det = client.get(f"/api/episodes/{ep_id}", headers=LEAD_HEADERS).json()
            assert det["status"] == "vlm_ready"
            assert det["active_run_id"] == "run_original_v1"

            # Wait for worker cleanup and assert active_run_id is still preserved
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and client.app.state.ctx.has_job_control(job_id):  # type: ignore[attr-defined]
                time.sleep(0.05)
            assert not client.app.state.ctx.has_job_control(job_id)  # type: ignore[attr-defined]
            det_after = client.get(f"/api/episodes/{ep_id}", headers=LEAD_HEADERS).json()
            assert det_after["status"] == "vlm_ready"
            assert det_after["active_run_id"] == "run_original_v1"
    finally:
        s1_release.set()


def test_api_cancel_queued_before_worker_starts_and_no_auth_flag(
    api_env: dict[str, Any],
) -> None:
    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]
    ds_dir = api_env["data_root"] / "ds_q_cancel"
    create_video_episode_file(ds_dir, "ep_q.mp4", duration_s=1.5, fps=10.0)

    repo = Repository(ws)
    try:
        ep_id = ingest_dir(repo, cfg_dir, ws, ds_dir, dataset="ds_q_cancel")[0]
    finally:
        repo.close()

    # Test --no-auth auto-detection boolean in /api/auth/me
    app_noauth = create_app(workspace=ws, config_dir=cfg_dir, no_auth=True)
    with TestClient(app_noauth) as client:
        r_me = client.get("/api/auth/me")
        assert r_me.status_code == 200
        assert r_me.json() == {"role": "lead", "user": "lead", "no_auth": True}

        # Occupy all 4 executor threads momentarily so batch_vlm stays queued before cancel
        gate = threading.Event()
        entered_lock = threading.Lock()
        entered_n = 0
        all_occupied = threading.Event()

        def _hold_worker() -> None:
            nonlocal entered_n
            with entered_lock:
                entered_n += 1
                if entered_n >= 4:
                    all_occupied.set()
            gate.wait(timeout=5.0)

        futs = [client.app.state.ctx.executor.submit(_hold_worker) for _ in range(4)]  # type: ignore[attr-defined]
        assert all_occupied.wait(timeout=2.0)
        try:
            sse_seq = client.app.state.ctx.sse.latest_seq()  # type: ignore[attr-defined]
            r_post = client.post(
                "/api/jobs",
                json={"job_type": "batch_vlm", "episode_ids": [ep_id]},
            )
            assert r_post.status_code == 201
            job_id = r_post.json()["job_id"]
            assert r_post.json()["status"] == "queued"

            # Episode immediately shows vlm_queued even while worker hasn't started
            assert client.get(f"/api/episodes/{ep_id}").json()["status"] == "vlm_queued"

            # Cancel while still queued -> immediate revert to ingested
            r_cancel = client.post(f"/api/jobs/{job_id}/cancel")
            assert r_cancel.status_code == 200
            assert r_cancel.json()["status"] == "cancelled"
            assert client.get(f"/api/episodes/{ep_id}").json()["status"] == "ingested"
        finally:
            gate.set()
            for f in futs:
                f.result(timeout=5.0)

        # Wait for queued worker to wake up, see cancelled, and clean up without touching status
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and client.app.state.ctx.has_job_control(job_id):  # type: ignore[attr-defined]
            time.sleep(0.05)
        assert not client.app.state.ctx.has_job_control(job_id)  # type: ignore[attr-defined]
        assert client.get(f"/api/jobs/{job_id}").json()["status"] == "cancelled"
        assert client.get(f"/api/episodes/{ep_id}").json()["status"] == "ingested"

        _, evs = client.app.state.ctx.sse.since(sse_seq)  # type: ignore[attr-defined]
        ep_evs = [(e["episode_id"], e["status"]) for e in evs if e.get("type") == "episode_updated"]
        assert ep_evs == [(ep_id, "vlm_queued"), (ep_id, "ingested")]


def test_duplicate_concurrent_vlm_conflict_and_review_blocked_while_running(
    api_env: dict[str, Any],
) -> None:
    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]
    mock_vlm: MockVlmServer = api_env["mock_vlm"]
    ds_dir = api_env["data_root"] / "ds_conflict"
    make_synthetic_h5_raw_v3(ds_dir, name="ep_dup.h5", duration_s=2.0, stationary=True)

    repo = Repository(ws)
    try:
        ep_id = ingest_dir(repo, cfg_dir, ws, ds_dir, dataset="ds_conflict")[0]
        cfg = ConfigManager(config_dir=cfg_dir, workspace_dir=ws)
        seed_vlm_run(repo, cfg, ep_id, run_id="run_seed_conflict")
    finally:
        repo.close()

    s1_entered = threading.Event()
    s1_release = threading.Event()

    def _blocking_s1(_rec: dict[str, Any]) -> dict[str, Any]:
        s1_entered.set()
        s1_release.wait(timeout=6.0)
        return {"objects": {}}

    mock_vlm.set_stage_response("s1", _blocking_s1)

    try:
        with make_client(ws, cfg_dir) as client:
            # Duplicate episode_ids are deduplicated preserving order -> progress_total == 1
            r1 = client.post(
                "/api/jobs",
                headers=LEAD_HEADERS,
                json={
                    "job_type": "batch_vlm",
                    "episode_ids": [ep_id, ep_id],
                    "fresh": True,
                },
            )
            assert r1.status_code == 201
            j1 = r1.json()
            assert j1["progress_total"] == 1
            assert j1["params"]["episode_ids"] == [ep_id]

            assert s1_entered.wait(timeout=5.0)
            assert client.get(f"/api/episodes/{ep_id}", headers=LEAD_HEADERS).json()["status"] == "vlm_running"

            # Second concurrent non-sandbox batch_vlm on same episode -> 409 Conflict
            r_dup = client.post(
                "/api/jobs",
                headers=LEAD_HEADERS,
                json={"job_type": "batch_vlm", "episode_ids": [ep_id], "fresh": True},
            )
            assert r_dup.status_code == 409
            assert ep_id in r_dup.json()["detail"]

            # Submitting a review while episode is vlm_running -> 409 Conflict
            r_rev = client.post(
                f"/api/episodes/{ep_id}/reviews",
                headers=LEAD_HEADERS,
                json={
                    "base_run_id": "run_seed_conflict",
                    "action": "save_draft",
                    "qc_segments": [],
                    "success": {"passed": True, "verdict": "success", "confidence": 0.9, "criteria": []},
                    "macro_stages": [],
                    "watch_coverage": {
                        "watched_intervals": [[0.0, 2.0]],
                        "watched_s": 2.0,
                        "duration_s": 2.0,
                        "coverage_ratio": 1.0,
                        "confirmed": True,
                    },
                },
            )
            assert r_rev.status_code == 409

            # Cancel job 1 -> immediately allows starting a fresh job 2 on the same episode
            r_c1 = client.post(f"/api/jobs/{j1['job_id']}/cancel", headers=LEAD_HEADERS)
            assert r_c1.status_code == 200
            assert client.get(f"/api/episodes/{ep_id}", headers=LEAD_HEADERS).json()["status"] == "vlm_ready"

            s1_release.set()
            r2 = client.post(
                "/api/jobs",
                headers=LEAD_HEADERS,
                json={"job_type": "batch_vlm", "episode_ids": [ep_id], "fresh": True},
            )
            assert r2.status_code == 201
            j2_id = r2.json()["job_id"]
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                st = client.get(f"/api/jobs/{j2_id}", headers=LEAD_HEADERS).json()["status"]
                if st in ("completed", "failed", "cancelled"):
                    break
                time.sleep(0.05)
            assert client.get(f"/api/jobs/{j2_id}", headers=LEAD_HEADERS).json()["status"] == "completed"
    finally:
        s1_release.set()


def test_eval_gt_cancel_aborts_inflight_vlm_quickly(
    api_env: dict[str, Any],
) -> None:
    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]
    mock_vlm: MockVlmServer = api_env["mock_vlm"]
    ds_dir = api_env["data_root"] / "ds_eval_cancel"
    create_video_episode_file(ds_dir, "ep_ev_a.mp4", duration_s=1.5, fps=10.0)
    create_video_episode_file(ds_dir, "ep_ev_b.mp4", duration_s=1.5, fps=10.0)

    repo = Repository(ws)
    try:
        ep_ids = sorted(ingest_dir(repo, cfg_dir, ws, ds_dir, dataset="ds_eval_cancel"))
        cfg = ConfigManager(config_dir=cfg_dir, workspace_dir=ws)
        for i, eid in enumerate(ep_ids):
            seed_vlm_run(repo, cfg, eid, run_id=f"r_eval_{i}", segments=[])
            repo.claim_episode(eid, "lead_1")
            repo.save_review(
                eid,
                author="lead_1",
                role="lead",
                payload=ReviewWritePayload(
                    base_run_id=f"r_eval_{i}",
                    action="freeze_gt",
                    qc_segments=[],
                    success=SuccessVerdict(verdict="success", passed=True, confidence=0.95),
                    watch_coverage=WatchCoverage(
                        watched_intervals=[[0.0, 1.5]],
                        watched_s=1.5,
                        duration_s=1.5,
                        coverage_ratio=1.0,
                    ),
                ),
                min_watch_coverage=0.95,
            )
    finally:
        repo.close()

    s1_entered = threading.Event()
    s1_release = threading.Event()

    def _slow_eval_s1(_rec: dict[str, Any]) -> dict[str, Any]:
        s1_entered.set()
        s1_release.wait(timeout=6.0)
        return {"objects": {}}

    mock_vlm.set_stage_response("s1", _slow_eval_s1)

    try:
        with make_client(ws, cfg_dir) as client:
            r_post = client.post(
                "/api/jobs",
                headers=LEAD_HEADERS,
                json={
                    "job_type": "eval_gt",
                    "episode_ids": ep_ids,
                    "eval_params": {"workers": 2, "few_shot": False},
                },
            )
            assert r_post.status_code == 201
            jid = r_post.json()["job_id"]
            assert s1_entered.wait(timeout=5.0)

            t0 = time.monotonic()
            r_cancel = client.post(f"/api/jobs/{jid}/cancel", headers=LEAD_HEADERS)
            assert r_cancel.status_code == 200
            assert r_cancel.json()["status"] == "cancelled"

            # Worker & ThreadPoolExecutor must exit quickly without waiting 6s!
            deadline = time.monotonic() + 2.5
            while time.monotonic() < deadline and client.app.state.ctx.has_job_control(jid):  # type: ignore[attr-defined]
                time.sleep(0.05)
            elapsed = time.monotonic() - t0
            assert not client.app.state.ctx.has_job_control(jid), "eval_gt worker hung after cancel"  # type: ignore[attr-defined]
            assert elapsed < 2.5, f"eval_gt cancel took too long: {elapsed:.2f}s"
            assert client.get(f"/api/jobs/{jid}", headers=LEAD_HEADERS).json()["status"] == "cancelled"
    finally:
        s1_release.set()


def test_export_download_containment_and_jobs_true_total(
    api_env: dict[str, Any], tmp_path: Path
) -> None:
    import io
    import zipfile

    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]
    ds_dir = api_env["data_root"] / "ds_export_sec"
    create_video_episode_file(ds_dir, "ep_exp.mp4", duration_s=1.5, fps=10.0)

    repo = Repository(ws)
    try:
        ep_id = ingest_dir(repo, cfg_dir, ws, ds_dir, dataset="ds_export_sec")[0]
        cfg = ConfigManager(config_dir=cfg_dir, workspace_dir=ws)
        seed_vlm_run(repo, cfg, ep_id, run_id="r_exp_sec", segments=[])
        repo.transition_status(ep_id, EpisodeStatus.APPROVED)
        # Create 5 jobs so we can test limit=2 returning total=5
        for i in range(5):
            repo.create_job("batch_vlm", {"episode_ids": []}, progress_total=0, job_id=f"j_tot_{i}", status="completed")
    finally:
        repo.close()

    secret_file = tmp_path / "outside_secret.txt"
    secret_file.write_text("TOP_SECRET", encoding="utf-8")

    with make_client(ws, cfg_dir) as client:
        # GET /api/jobs returns true SQL total independent of limit
        r_jobs = client.get("/api/jobs?limit=2&offset=0", headers=LEAD_HEADERS)
        assert r_jobs.status_code == 200
        jdata = r_jobs.json()
        assert len(jdata["items"]) == 2
        assert jdata["total"] == 5

        # Export symlink inside out_dir escaping out_dir is excluded from ZIP
        r_exp = client.post(
            "/api/exports",
            headers=LEAD_HEADERS,
            json={"format": "zero_copy_index", "recipe": "bc_clean"},
        )
        assert r_exp.status_code == 200
        exp_id = r_exp.json()["export_id"]
        exp_dir = Path(r_exp.json()["output_dir"])
        (exp_dir / "leak.txt").symlink_to(secret_file)

        r_dl = client.get(f"/api/exports/{exp_id}/download", headers=LEAD_HEADERS)
        assert r_dl.status_code == 200
        with zipfile.ZipFile(io.BytesIO(r_dl.content), "r") as zf:
            assert "leak.txt" not in zf.namelist()

        # Tampered manifest output_dir escaping workspace -> 403
        mpath = ws / "artifacts" / "exports" / f"{exp_id}.json"
        mdata = json.loads(mpath.read_text(encoding="utf-8"))
        mdata["output_dir"] = str(tmp_path)
        mpath.write_text(json.dumps(mdata), encoding="utf-8")
        r_esc = client.get(f"/api/exports/{exp_id}/download", headers=LEAD_HEADERS)
        assert r_esc.status_code == 403
