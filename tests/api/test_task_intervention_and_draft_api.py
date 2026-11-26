"""API integration tests for manual task intervention, batch task assign, AI TaskSpec draft, and ingest task override."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from robo_vlm_qc.config import ConfigManager
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
from tests.e2e.helpers import make_synthetic_lerobot


def _poll_job_until_done(client: Any, job_id: str, timeout_s: float = 20.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        r = client.get(f"/api/jobs/{job_id}", headers=LEAD_HEADERS)
        assert r.status_code == 200
        job = r.json()
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(0.04)
    raise AssertionError(f"Job {job_id} did not finish within {timeout_s}s")


def _count_image_blocks(messages: list[dict[str, Any]]) -> int:
    count = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    count += 1
    return count


class _DraftMockVlmServer:
    """Dedicated OpenAI-compatible mock server for task_assist drafting tests."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._lock = threading.Lock()
        self.payload = payload
        self.fail_next_status: int | None = None
        self.requests: list[dict[str, Any]] = []

        parent = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                messages = body.get("messages", [])
                with parent._lock:
                    parent.requests.append(
                        {
                            "messages": messages,
                            "image_count": _count_image_blocks(messages),
                            "body": body,
                        }
                    )
                    fail_status = parent.fail_next_status
                    parent.fail_next_status = None
                    resp_payload = dict(parent.payload)

                if fail_status is not None:
                    raw = json.dumps({"error": {"message": "injected VLM error"}}).encode("utf-8")
                    self.send_response(fail_status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return

                resp_obj = {
                    "id": "chatcmpl-draft-mock",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(resp_payload, ensure_ascii=False),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 80, "completion_tokens": 40, "total_tokens": 120},
                }
                raw = json.dumps(resp_obj, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.port = int(self._httpd.server_address[1])
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=3.0)


def _seed_review_for_episode(
    repo: Repository,
    ep_id: str,
    run_id: str,
    *,
    action: str = "submit_review",
    duration_s: float = 2.0,
) -> str:
    repo.claim_episode(ep_id, "lead_1")
    rev = repo.save_review(
        ep_id,
        author="lead_1",
        role="lead",
        payload=ReviewWritePayload(
            base_run_id=run_id,
            action=action,  # type: ignore[arg-type]
            qc_segments=[],
            success=SuccessVerdict(
                verdict="success", passed=True, confidence=0.98, summary="reviewed"
            ),
            watch_coverage=WatchCoverage(
                watched_intervals=[(0.0, duration_s)],
                watched_s=duration_s,
                duration_s=duration_s,
                coverage_ratio=1.0,
                confirmed=True,
            ),
        ),
        min_watch_coverage=0.95,
    )
    return rev.rev_id


def test_get_episodes_tasks_for_lead_and_reviewer(api_env: dict[str, Any]) -> None:
    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]

    with make_client(ws, cfg_dir) as client:
        assert client.get("/api/episodes/tasks").status_code == 401

        for headers in (LEAD_HEADERS, REV_HEADERS):
            r = client.get("/api/episodes/tasks", headers=headers)
            assert r.status_code == 200, r.text
            body = r.json()
            tasks = body["tasks"]
            assert body["total"] == len(tasks)
            assert body["total"] >= 3

            task_ids = [t["task_id"] for t in tasks]
            # Non-_default sorted alphabetically, _default strictly last
            assert task_ids[-1] == "_default"
            assert task_ids[:-1] == sorted(task_ids[:-1])
            assert "bimanual_handover" in task_ids
            assert "pick_and_place" in task_ids

            by_id = {t["task_id"]: t for t in tasks}
            for tid, item in by_id.items():
                for key in (
                    "task_id",
                    "title",
                    "description",
                    "goal",
                    "match",
                    "aliases",
                    "grounding_objects_count",
                ):
                    assert key in item, (tid, key)
                assert isinstance(item["match"], list)
                assert isinstance(item["aliases"], list)
                assert isinstance(item["grounding_objects_count"], int)

            assert by_id["pick_and_place"]["grounding_objects_count"] == 2
            assert by_id["bimanual_handover"]["grounding_objects_count"] == 3
            assert by_id["_default"]["grounding_objects_count"] == 0


def test_patch_episode_task_lifecycle_guards_and_rbac(api_env: dict[str, Any]) -> None:
    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]
    ds_dir = api_env["data_root"] / "episode_001389"

    for name in (
        "ep_ingested.mp4",
        "ep_vlm_ready.mp4",
        "ep_reviewed.mp4",
        "ep_queued.mp4",
        "ep_running.mp4",
        "ep_frozen_lead.mp4",
        "ep_rev_default.mp4",
        "ep_rev_frozen.mp4",
        "ep_reserved.mp4",
    ):
        create_video_episode_file(ds_dir, name, duration_s=2.0, fps=15.0)

    repo = Repository(ws)
    cfg = ConfigManager(config_dir=cfg_dir, workspace_dir=ws)
    try:
        ep_ids = ingest_dir(repo, cfg_dir, ws, ds_dir, dataset="episode_001389")
        by_stem = {eid.split("/", 1)[1]: eid for eid in ep_ids}

        # Confirm unrecognized dataset fell back to _default
        for eid in ep_ids:
            assert repo.get_episode_row(eid)["task_id"] == "_default"

        # Seed active VLM run for vlm_ready
        ep_vlm_ready = by_stem["ep_vlm_ready"]
        seed_vlm_run(repo, cfg, ep_vlm_ready, run_id="run_ready_1")
        assert repo.get_episode_row(ep_vlm_ready)["status"] == EpisodeStatus.VLM_READY.value

        # Seed active VLM run + human review for reviewed
        ep_reviewed = by_stem["ep_reviewed"]
        seed_vlm_run(repo, cfg, ep_reviewed, run_id="run_rev_1")
        rev_id_reviewed = _seed_review_for_episode(repo, ep_reviewed, "run_rev_1", action="submit_review")
        assert repo.get_episode_row(ep_reviewed)["status"] == EpisodeStatus.REVIEWED.value

        # Transition queued / running
        ep_queued = by_stem["ep_queued"]
        repo.transition_status(ep_queued, EpisodeStatus.VLM_QUEUED)
        ep_running = by_stem["ep_running"]
        repo.transition_status(ep_running, EpisodeStatus.VLM_RUNNING)

        # Seed gt_frozen episodes (both still _default)
        ep_frozen_lead = by_stem["ep_frozen_lead"]
        seed_vlm_run(repo, cfg, ep_frozen_lead, run_id="run_frozen_1")
        rev_id_frozen = _seed_review_for_episode(repo, ep_frozen_lead, "run_frozen_1", action="freeze_gt")
        assert repo.get_episode_row(ep_frozen_lead)["status"] == EpisodeStatus.GT_FROZEN.value

        ep_rev_frozen = by_stem["ep_rev_frozen"]
        seed_vlm_run(repo, cfg, ep_rev_frozen, run_id="run_rev_frozen_1")
        _seed_review_for_episode(repo, ep_rev_frozen, "run_rev_frozen_1", action="freeze_gt")
        assert repo.get_episode_row(ep_rev_frozen)["status"] == EpisodeStatus.GT_FROZEN.value

        ep_ingested = by_stem["ep_ingested"]
        ep_rev_default = by_stem["ep_rev_default"]
        ep_reserved = by_stem["ep_reserved"]
    finally:
        repo.close()

    with make_client(ws, cfg_dir) as client:
        # Unauth / missing / unknown task_id / extra field
        assert (
            client.patch(
                f"/api/episodes/{ep_ingested}/task", json={"task_id": "pick_and_place"}
            ).status_code
            == 401
        )
        assert (
            client.patch(
                f"/api/episodes/{ep_ingested}/task",
                headers=LEAD_HEADERS,
                json={"task_id": "nonexistent_task"},
            ).status_code
            == 422
        )
        assert (
            client.patch(
                f"/api/episodes/{ep_ingested}/task",
                headers=LEAD_HEADERS,
                json={"task_id": "pick_and_place", "unexpected_field": True},
            ).status_code
            == 422
        )
        assert (
            client.patch(
                "/api/episodes/missing_ds/missing_ep/task",
                headers=LEAD_HEADERS,
                json={"task_id": "pick_and_place"},
            ).status_code
            == 404
        )

        # 1. Lead reassigns _default -> pick_and_place on ingested (re_run_recommended=False + SSE)
        seq0 = client.app.state.ctx.sse.latest_seq()  # type: ignore[attr-defined]
        r_ing = client.patch(
            f"/api/episodes/{ep_ingested}/task",
            headers=LEAD_HEADERS,
            json={"task_id": "pick_and_place"},
        )
        assert r_ing.status_code == 200, r_ing.text
        assert r_ing.json() == {
            "episode_id": ep_ingested,
            "previous_task_id": "_default",
            "task_id": "pick_and_place",
            "status": "ingested",
            "has_active_run": False,
            "re_run_recommended": False,
        }
        _, events = client.app.state.ctx.sse.since(seq0)  # type: ignore[attr-defined]
        assert any(
            ev.get("type") == "episode_updated"
            and ev.get("episode_id") == ep_ingested
            and ev.get("task_id") == "pick_and_place"
            and ev.get("previous_task_id") == "_default"
            for ev in events
        ), events

        # 2. Lead reassigns _default -> pick_and_place on vlm_ready and reviewed (preserves run/rev/status!)
        r_ready = client.patch(
            f"/api/episodes/{ep_vlm_ready}/task",
            headers=LEAD_HEADERS,
            json={"task_id": "pick_and_place"},
        )
        assert r_ready.status_code == 200, r_ready.text
        assert r_ready.json() == {
            "episode_id": ep_vlm_ready,
            "previous_task_id": "_default",
            "task_id": "pick_and_place",
            "status": "vlm_ready",
            "has_active_run": True,
            "re_run_recommended": True,
        }

        r_rev = client.patch(
            f"/api/episodes/{ep_reviewed}/task",
            headers=LEAD_HEADERS,
            json={"task_id": "pick_and_place"},
        )
        assert r_rev.status_code == 200, r_rev.text
        assert r_rev.json()["has_active_run"] is True
        assert r_rev.json()["re_run_recommended"] is True
        assert r_rev.json()["status"] == "reviewed"

        # Resolved episode detail endpoint reflects updated task_id while preserving run/review
        r_detail = client.get(f"/api/episodes/{ep_reviewed}", headers=LEAD_HEADERS)
        assert r_detail.status_code == 200, r_detail.text
        detail_json = r_detail.json()
        assert detail_json["meta"]["task_id"] == "pick_and_place"
        assert detail_json["status"] == "reviewed"
        assert detail_json["active_run_id"] == "run_rev_1"
        assert detail_json["active_rev_id"] == rev_id_reviewed

        # No-op reassignment on an episode with active run has re_run_recommended=False
        r_noop = client.patch(
            f"/api/episodes/{ep_vlm_ready}/task",
            headers=LEAD_HEADERS,
            json={"task_id": "pick_and_place"},
        )
        assert r_noop.status_code == 200
        assert r_noop.json()["re_run_recommended"] is False

        # 3. Active VLM guard: 409 on vlm_queued, vlm_running, and in-memory reservation
        assert (
            client.patch(
                f"/api/episodes/{ep_queued}/task",
                headers=LEAD_HEADERS,
                json={"task_id": "pick_and_place"},
            ).status_code
            == 409
        )
        assert (
            client.patch(
                f"/api/episodes/{ep_running}/task",
                headers=LEAD_HEADERS,
                json={"task_id": "pick_and_place"},
            ).status_code
            == 409
        )
        busy = client.app.state.ctx.try_reserve_vlm_episodes([ep_reserved], "job_active_test")  # type: ignore[attr-defined]
        assert busy == []
        try:
            assert (
                client.patch(
                    f"/api/episodes/{ep_reserved}/task",
                    headers=LEAD_HEADERS,
                    json={"task_id": "pick_and_place"},
                ).status_code
                == 409
            )
        finally:
            client.app.state.ctx.release_vlm_episodes("job_active_test")  # type: ignore[attr-defined]

        # 4. gt_frozen episode: 409 when allow_frozen=False, 200 when allow_frozen=True (Lead only)
        r_frozen_block = client.patch(
            f"/api/episodes/{ep_frozen_lead}/task",
            headers=LEAD_HEADERS,
            json={"task_id": "pick_and_place"},
        )
        assert r_frozen_block.status_code == 409
        detail_low = str(r_frozen_block.json().get("detail", "")).lower()
        assert "gt_frozen" in detail_low
        assert "allow_frozen" in detail_low

        r_frozen_ok = client.patch(
            f"/api/episodes/{ep_frozen_lead}/task",
            headers=LEAD_HEADERS,
            json={"task_id": "pick_and_place", "allow_frozen": True},
        )
        assert r_frozen_ok.status_code == 200, r_frozen_ok.text
        assert r_frozen_ok.json() == {
            "episode_id": ep_frozen_lead,
            "previous_task_id": "_default",
            "task_id": "pick_and_place",
            "status": "gt_frozen",
            "has_active_run": True,
            "re_run_recommended": True,
        }

        # 5. Reviewer RBAC
        # Reviewer cannot set _default -> _default
        assert (
            client.patch(
                f"/api/episodes/{ep_rev_default}/task",
                headers=REV_HEADERS,
                json={"task_id": "_default"},
            ).status_code
            == 403
        )
        # Reviewer cannot modify gt_frozen even if task_id == _default and allow_frozen=True
        assert (
            client.patch(
                f"/api/episodes/{ep_rev_frozen}/task",
                headers=REV_HEADERS,
                json={"task_id": "pick_and_place", "allow_frozen": True},
            ).status_code
            == 403
        )
        # Reviewer CAN reassign non-frozen _default -> pick_and_place (200)
        r_rev_ok = client.patch(
            f"/api/episodes/{ep_rev_default}/task",
            headers=REV_HEADERS,
            json={"task_id": "pick_and_place"},
        )
        assert r_rev_ok.status_code == 200, r_rev_ok.text
        assert r_rev_ok.json()["task_id"] == "pick_and_place"
        assert r_rev_ok.json()["previous_task_id"] == "_default"

        # Now that ep_rev_default is pick_and_place (non-_default), Reviewer gets 403 trying to change it again
        assert (
            client.patch(
                f"/api/episodes/{ep_rev_default}/task",
                headers=REV_HEADERS,
                json={"task_id": "bimanual_handover"},
            ).status_code
            == 403
        )
        # Reviewer also gets 403 trying to revert a non-_default episode back to _default
        assert (
            client.patch(
                f"/api/episodes/{ep_rev_default}/task",
                headers=REV_HEADERS,
                json={"task_id": "_default"},
            ).status_code
            == 403
        )

    # Confirm DB preserved active_run_id, active_rev_id, and status everywhere
    repo_check = Repository(ws)
    try:
        row_ready = repo_check.get_episode_row(ep_vlm_ready)
        assert row_ready["task_id"] == "pick_and_place"
        assert row_ready["active_run_id"] == "run_ready_1"
        assert row_ready["active_rev_id"] is None
        assert row_ready["status"] == EpisodeStatus.VLM_READY.value

        row_rev = repo_check.get_episode_row(ep_reviewed)
        assert row_rev["task_id"] == "pick_and_place"
        assert row_rev["active_run_id"] == "run_rev_1"
        assert row_rev["active_rev_id"] == rev_id_reviewed
        assert row_rev["status"] == EpisodeStatus.REVIEWED.value

        row_frozen = repo_check.get_episode_row(ep_frozen_lead)
        assert row_frozen["task_id"] == "pick_and_place"
        assert row_frozen["active_run_id"] == "run_frozen_1"
        assert row_frozen["active_rev_id"] == rev_id_frozen
        assert row_frozen["status"] == EpisodeStatus.GT_FROZEN.value

        # Blocked episodes stayed untouched
        assert repo_check.get_episode_row(ep_queued)["task_id"] == "_default"
        assert repo_check.get_episode_row(ep_running)["task_id"] == "_default"
        assert repo_check.get_episode_row(ep_rev_frozen)["task_id"] == "_default"
        assert repo_check.get_episode_row(ep_reserved)["task_id"] == "_default"
    finally:
        repo_check.close()


def test_post_episodes_batch_task_atomicity_rbac_and_rerun_ids(
    api_env: dict[str, Any]
) -> None:
    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]
    ds_dir = api_env["data_root"] / "batch_capture_001"

    for name in ("ep_a.mp4", "ep_b.mp4", "ep_running.mp4", "ep_frozen.mp4"):
        create_video_episode_file(ds_dir, name, duration_s=2.0, fps=15.0)

    repo = Repository(ws)
    cfg = ConfigManager(config_dir=cfg_dir, workspace_dir=ws)
    try:
        ep_ids = ingest_dir(repo, cfg_dir, ws, ds_dir, dataset="batch_capture_001")
        by_stem = {eid.split("/", 1)[1]: eid for eid in ep_ids}
        ep_a = by_stem["ep_a"]
        ep_b = by_stem["ep_b"]
        ep_running = by_stem["ep_running"]
        ep_frozen = by_stem["ep_frozen"]

        seed_vlm_run(repo, cfg, ep_b, run_id="run_b_1")
        repo.transition_status(ep_running, EpisodeStatus.VLM_RUNNING)
        seed_vlm_run(repo, cfg, ep_frozen, run_id="run_f_1")
        rev_f = _seed_review_for_episode(repo, ep_frozen, "run_f_1", action="freeze_gt")
    finally:
        repo.close()

    with make_client(ws, cfg_dir) as client:
        # Auth & RBAC (Lead-only)
        assert (
            client.post(
                "/api/episodes/batch-task",
                json={"episode_ids": [ep_a], "task_id": "pick_and_place"},
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/api/episodes/batch-task",
                headers=REV_HEADERS,
                json={"episode_ids": [ep_a], "task_id": "pick_and_place"},
            ).status_code
            == 403
        )

        # Validation: unknown task_id, empty list, extra field
        assert (
            client.post(
                "/api/episodes/batch-task",
                headers=LEAD_HEADERS,
                json={"episode_ids": [ep_a], "task_id": "unknown_xyz"},
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/episodes/batch-task",
                headers=LEAD_HEADERS,
                json={"episode_ids": [], "task_id": "pick_and_place"},
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/episodes/batch-task",
                headers=LEAD_HEADERS,
                json={"episode_ids": [ep_a], "task_id": "pick_and_place", "extra": 1},
            ).status_code
            == 422
        )

        # Atomic 404 when any episode is missing -> no partial write to ep_a
        assert (
            client.post(
                "/api/episodes/batch-task",
                headers=LEAD_HEADERS,
                json={"episode_ids": [ep_a, "batch_capture_001/no_such_ep"], "task_id": "pick_and_place"},
            ).status_code
            == 404
        )
        assert client.app.state.ctx.repo.get_episode_row(ep_a)["task_id"] == "_default"  # type: ignore[attr-defined]

        # Atomic 409 when any episode is vlm_running -> no partial write to ep_a / ep_b
        assert (
            client.post(
                "/api/episodes/batch-task",
                headers=LEAD_HEADERS,
                json={"episode_ids": [ep_a, ep_b, ep_running], "task_id": "pick_and_place"},
            ).status_code
            == 409
        )
        assert client.app.state.ctx.repo.get_episode_row(ep_a)["task_id"] == "_default"  # type: ignore[attr-defined]
        assert client.app.state.ctx.repo.get_episode_row(ep_b)["task_id"] == "_default"  # type: ignore[attr-defined]

        # Atomic 409 when any episode is gt_frozen without allow_frozen=True -> no partial write!
        assert (
            client.post(
                "/api/episodes/batch-task",
                headers=LEAD_HEADERS,
                json={"episode_ids": [ep_a, ep_b, ep_frozen], "task_id": "pick_and_place"},
            ).status_code
            == 409
        )
        assert client.app.state.ctx.repo.get_episode_row(ep_a)["task_id"] == "_default"  # type: ignore[attr-defined]
        assert client.app.state.ctx.repo.get_episode_row(ep_b)["task_id"] == "_default"  # type: ignore[attr-defined]
        assert client.app.state.ctx.repo.get_episode_row(ep_frozen)["task_id"] == "_default"  # type: ignore[attr-defined]

        # Atomic success with allow_frozen=True and duplicate ID deduplication
        seq0 = client.app.state.ctx.sse.latest_seq()  # type: ignore[attr-defined]
        r_ok = client.post(
            "/api/episodes/batch-task",
            headers=LEAD_HEADERS,
            json={
                "episode_ids": [ep_a, ep_b, ep_a, ep_frozen],
                "task_id": "pick_and_place",
                "allow_frozen": True,
            },
        )
        assert r_ok.status_code == 200, r_ok.text
        data = r_ok.json()
        assert data["task_id"] == "pick_and_place"
        assert data["updated"] == 3
        assert data["episode_ids"] == [ep_a, ep_b, ep_frozen]
        assert data["has_active_runs"] == 2
        assert set(data["re_run_recommended_ids"]) == {ep_b, ep_frozen}

        _, events = client.app.state.ctx.sse.since(seq0)  # type: ignore[attr-defined]
        updated_ev_ids = {
            ev["episode_id"]
            for ev in events
            if ev.get("type") == "episode_updated" and ev.get("task_id") == "pick_and_place"
        }
        assert updated_ev_ids == {ep_a, ep_b, ep_frozen}

    # Verify DB integrity
    repo_check = Repository(ws)
    try:
        assert repo_check.get_episode_row(ep_a)["task_id"] == "pick_and_place"
        assert repo_check.get_episode_row(ep_a)["active_run_id"] is None

        row_b = repo_check.get_episode_row(ep_b)
        assert row_b["task_id"] == "pick_and_place"
        assert row_b["active_run_id"] == "run_b_1"
        assert row_b["status"] == EpisodeStatus.VLM_READY.value

        row_f = repo_check.get_episode_row(ep_frozen)
        assert row_f["task_id"] == "pick_and_place"
        assert row_f["active_run_id"] == "run_f_1"
        assert row_f["active_rev_id"] == rev_f
        assert row_f["status"] == EpisodeStatus.GT_FROZEN.value

        assert repo_check.get_episode_row(ep_running)["task_id"] == "_default"
    finally:
        repo_check.close()


def test_lab_task_draft_save_and_immediate_assign_end_to_end(
    api_env: dict[str, Any]
) -> None:
    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]

    draft_vlm_payload = {
        "title": "Fold Tabletop Towel",
        "description": "Fold a towel twice into a neat square on the table.",
        "goal": "Fold the towel cleanly twice and place it centered on the tabletop.",
        "steps": [
            "Grasp two corners of the towel.",
            "Fold the towel in half and smooth it flat.",
            "Fold a second time and release cleanly.",
        ],
        "criteria": [
            "Towel is neatly folded twice with aligned edges.",
            "Manipulators retract without dragging the towel.",
        ],
        "task_prompt": "Check fabric grasp quality, fold symmetry, and clean release.",
        "grounding_objects": [
            {
                "id": "towel",
                "name": "Towel",
                "desc": "Fabric towel manipulated on the tabletop",
                "tag_pos": "top",
                "is_static": False,
            }
        ],
        "match": ["*fold_towel*", "*episode_001389*", "*PICO tracker capture*"],
        "aliases": ["fold towel", "PICO tracker capture"],
        "rationale": "Synthesized from visual sample frames and user description.",
    }
    draft_mock = _DraftMockVlmServer(draft_vlm_payload)
    try:
        write_workspace_config(
            ws,
            base_url=draft_mock.base_url,
            api_key_env="MOCK_VLM_API_KEY",
            extra_overrides={"vlm": {"retries": 0}},
        )

        # Seed a synthetic LeRobot _default episode with noisy dataset/task metadata
        lr_dir = api_env["data_root"] / "episode_001389"
        make_synthetic_lerobot(
            lr_dir,
            version="v3",
            n_episodes=1,
            duration_s=2.0,
            fps=15,
            task_name="PICO tracker capture",
        )
        repo = Repository(ws)
        try:
            ep_default = ingest_dir(repo, cfg_dir, ws, lr_dir)[0]
            assert repo.get_episode_row(ep_default)["task_id"] == "_default"
        finally:
            repo.close()

        with make_client(ws, cfg_dir) as client:
            # Auth & RBAC (Lead-only)
            assert (
                client.post(
                    "/api/lab/tasks/draft",
                    json={"task_id": "fold_towel", "description": "Fold a towel cleanly."},
                ).status_code
                == 401
            )
            assert (
                client.post(
                    "/api/lab/tasks/draft",
                    headers=REV_HEADERS,
                    json={"task_id": "fold_towel", "description": "Fold a towel cleanly."},
                ).status_code
                == 403
            )

            # Validation errors -> 422 / 404
            assert (
                client.post(
                    "/api/lab/tasks/draft",
                    headers=LEAD_HEADERS,
                    json={"task_id": "_default", "description": "Fold a towel cleanly."},
                ).status_code
                == 422
            )
            assert (
                client.post(
                    "/api/lab/tasks/draft",
                    headers=LEAD_HEADERS,
                    json={"task_id": "fold_towel", "description": "  "},
                ).status_code
                == 422
            )
            assert (
                client.post(
                    "/api/lab/tasks/draft",
                    headers=LEAD_HEADERS,
                    json={
                        "task_id": "fold_towel",
                        "description": "Fold a towel cleanly.",
                        "num_frames": 2,
                    },
                ).status_code
                == 422
            )
            assert (
                client.post(
                    "/api/lab/tasks/draft",
                    headers=LEAD_HEADERS,
                    json={
                        "task_id": "fold_towel",
                        "description": "Fold a towel cleanly.",
                        "num_frames": 9,
                    },
                ).status_code
                == 422
            )
            assert (
                client.post(
                    "/api/lab/tasks/draft",
                    headers=LEAD_HEADERS,
                    json={
                        "task_id": "fold_towel",
                        "description": "Fold a towel cleanly.",
                        "sample_episode_id": "episode_001389/does_not_exist",
                    },
                ).status_code
                == 404
            )

            # Upstream VLM error maps cleanly to 502
            draft_mock.fail_next_status = 500
            r_502 = client.post(
                "/api/lab/tasks/draft",
                headers=LEAD_HEADERS,
                json={"task_id": "fold_towel", "description": "Fold a towel cleanly."},
            )
            assert r_502.status_code == 502

            # 1. Text-only draft (Chinese description via unicode escapes)
            draft_mock.requests.clear()
            zh_desc = "\u6298\u53e0\u6bdb\u5dfe\u4e24\u6b21\u5e76\u6574\u9f50\u653e\u5728\u684c\u9762\u4e2d\u592e"
            r_text = client.post(
                "/api/lab/tasks/draft",
                headers=LEAD_HEADERS,
                json={"task_id": "fold_towel", "description": zh_desc},
            )
            assert r_text.status_code == 200, r_text.text
            d_text = r_text.json()
            assert d_text["advisory_only"] is True
            assert d_text["used_sample_episode_id"] is None
            assert d_text["sampled_frames_count"] == 0
            assert d_text["task_spec"]["task_id"] == "fold_towel"
            assert len(draft_mock.requests) == 1
            assert draft_mock.requests[0]["image_count"] == 0
            assert not (ws / "configs_override" / "tasks" / "fold_towel.yaml").exists()

            # 2. Multimodal draft with sample_episode_id
            draft_mock.requests.clear()
            r_mm = client.post(
                "/api/lab/tasks/draft",
                headers=LEAD_HEADERS,
                json={
                    "task_id": "fold_towel",
                    "description": "Fold a tabletop towel neatly twice.",
                    "title": "Fold Tabletop Towel",
                    "sample_episode_id": ep_default,
                    "num_frames": 5,
                },
            )
            assert r_mm.status_code == 200, r_mm.text
            d_mm = r_mm.json()
            assert d_mm["advisory_only"] is True
            assert d_mm["used_sample_episode_id"] == ep_default
            assert d_mm["sampled_frames_count"] == 5
            assert len(draft_mock.requests) == 1
            assert draft_mock.requests[0]["image_count"] == 5
            assert d_mm["warnings"], "Expected anti-hijack warnings in multimodal draft"

            spec_draft = d_mm["task_spec"]
            assert spec_draft["task_id"] == "fold_towel"
            assert "*fold_towel*" in spec_draft["match"]
            assert not any("episode_001389" in m.lower() for m in spec_draft["match"])
            assert not any("pico" in m.lower() or "tracker" in m.lower() for m in spec_draft["match"])
            # Still advisory-only before PUT /api/lab/tasks
            assert not (ws / "configs_override" / "tasks" / "fold_towel.yaml").exists()

            # 3. Save drafted TaskSpec via PUT /api/lab/tasks
            r_save = client.put("/api/lab/tasks", headers=LEAD_HEADERS, json=spec_draft)
            assert r_save.status_code == 200, r_save.text
            assert "fold_towel" in r_save.json()["tasks"]
            assert (ws / "configs_override" / "tasks" / "fold_towel.yaml").is_file()
            # Must NOT mutate global prompt overrides
            assert not (ws / "configs_override" / "prompts").exists()

            # 4. Newly saved task is immediately visible in GET /api/episodes/tasks
            r_list = client.get("/api/episodes/tasks", headers=REV_HEADERS)
            assert r_list.status_code == 200
            listed_ids = [t["task_id"] for t in r_list.json()["tasks"]]
            assert "fold_towel" in listed_ids
            assert listed_ids[-1] == "_default"

            # 5. Immediately assign the newly saved task via PATCH /api/episodes/{ep_id}/task
            r_assign = client.patch(
                f"/api/episodes/{ep_default}/task",
                headers=LEAD_HEADERS,
                json={"task_id": "fold_towel"},
            )
            assert r_assign.status_code == 200, r_assign.text
            assert r_assign.json()["previous_task_id"] == "_default"
            assert r_assign.json()["task_id"] == "fold_towel"
            assert r_assign.json()["re_run_recommended"] is False
    finally:
        draft_mock.stop()


def test_ingest_job_with_explicit_task_id_and_validation(api_env: dict[str, Any]) -> None:
    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]
    data_root: Path = api_env["data_root"]

    # Create synthetic LeRobot dataset with noisy serial dataset dir + capture device task string
    noisy_lr = data_root / "episode_001389"
    make_synthetic_lerobot(
        noisy_lr,
        version="v3",
        n_episodes=2,
        duration_s=2.0,
        fps=15,
        task_name="PICO tracker capture",
    )

    with make_client(ws, cfg_dir, allowed_data_roots=[data_root]) as client:
        # 1. Unknown task_id rejected synchronously with 422
        r_bad = client.post(
            "/api/jobs",
            headers=LEAD_HEADERS,
            json={
                "job_type": "ingest",
                "ingest_params": {
                    "data_path": str(noisy_lr),
                    "task_id": "nonexistent_task",
                },
            },
        )
        assert r_bad.status_code == 422, r_bad.text

        # 2. Default / auto ingest falls back to _default for noisy serial/device metadata
        r_auto = client.post(
            "/api/jobs",
            headers=LEAD_HEADERS,
            json={
                "job_type": "ingest",
                "ingest_params": {
                    "data_path": str(noisy_lr),
                    "dataset": "auto_ds_001389",
                    "task_id": "auto",
                },
            },
        )
        assert r_auto.status_code == 201, r_auto.text
        j_auto = _poll_job_until_done(client, r_auto.json()["job_id"])
        assert j_auto["status"] == "completed", j_auto
        assert j_auto["summary"]["inserted"] == 2
        assert all(item["task_id"] == "_default" for item in j_auto["summary"]["results"])

        # 3. Explicit task_id overrides _default fallback cleanly
        r_explicit = client.post(
            "/api/jobs",
            headers=LEAD_HEADERS,
            json={
                "job_type": "ingest",
                "ingest_params": {
                    "data_path": str(noisy_lr),
                    "dataset": "explicit_ds_001389",
                    "task_id": "pick_and_place",
                },
            },
        )
        assert r_explicit.status_code == 201, r_explicit.text
        j_exp = _poll_job_until_done(client, r_explicit.json()["job_id"])
        assert j_exp["status"] == "completed", j_exp
        assert j_exp["summary"]["inserted"] == 2
        assert all(item["task_id"] == "pick_and_place" for item in j_exp["summary"]["results"])

    repo = Repository(ws)
    try:
        rows_auto = repo.list_episodes(dataset="auto_ds_001389")
        assert len(rows_auto) == 2
        assert all(r["task_id"] == "_default" for r in rows_auto)

        rows_exp = repo.list_episodes(dataset="explicit_ds_001389")
        assert len(rows_exp) == 2
        assert all(r["task_id"] == "pick_and_place" for r in rows_exp)
    finally:
        repo.close()
