"""Live uvicorn + httpx + Playwright full-stack API, media range, SSE, and browser E2E tests."""

from __future__ import annotations

import json
import re
import shutil
import socket
import threading
import time
import zipfile
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Iterator

import cv2
import httpx
import numpy as np
import pytest
import uvicorn

from robo_vlm_qc.api import create_app
from robo_vlm_qc.config import ConfigManager, default_config_dir
from robo_vlm_qc.io import auto_detect, get_adapter
from robo_vlm_qc.models.annotation import (
    ArmSpan,
    ArmTimelines,
    ConfigFingerprintSnapshot,
    MacroStage,
    QcSegment,
    SuccessVerdict,
    VlmRunArtifact,
)
from robo_vlm_qc.models.review import ReviewWritePayload, WatchCoverage
from robo_vlm_qc.models.taxonomy import FewShotItem
from robo_vlm_qc.store.repository import Repository

REPO_ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _write_mp4(path: Path, n_frames: int = 60, fps: float = 30.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    assert w.isOpened()
    try:
        for i in range(n_frames):
            w.write(np.full((120, 160, 3), (i * 4) % 255, dtype=np.uint8))
    finally:
        w.release()


def _seed_live_workspace(
    tmp_path: Path,
    *,
    claim_ttl_s: float | None = None,
    with_few_shot: bool = False,
) -> tuple[Path, Path, str, str]:
    ws = tmp_path / "ws"
    cfg_dir = tmp_path / "configs"
    shutil.copytree(default_config_dir(), cfg_dir)
    if claim_ttl_s is not None:
        settings_p = cfg_dir / "default.yaml"
        txt = settings_p.read_text(encoding="utf-8")
        txt = re.sub(r"claim_ttl_s:\s*[0-9.]+", f"claim_ttl_s: {claim_ttl_s}", txt)
        settings_p.write_text(txt, encoding="utf-8")

    ds_dir = tmp_path / "data" / "demo_ds"
    _write_mp4(ds_dir / "ep_001.mp4", n_frames=60, fps=30.0)
    handle = auto_detect(ds_dir)[0]
    with get_adapter(handle.adapter_type).open(handle) as view:
        meta = view.meta

    cfg = ConfigManager(config_dir=cfg_dir, workspace_dir=ws)
    repo = Repository(ws)
    try:
        repo.ingest_episode(meta)
        run = VlmRunArtifact(
            run_id="run_live_1",
            episode_id=meta.episode_id,
            created_at=time.time(),
            config_fingerprint=ConfigFingerprintSnapshot(
                model_routing={"s1_grounding": "gpt-4o"},
                prompt_bundle_version=cfg.fingerprints.prompt_bundle_sha256[:12],
                label_taxonomy_version=f"v{cfg.labels.version}",
                task_spec_version="1.0.0",
            ),
            macro_stages=[MacroStage(stage_id="S1", t_start=0.0, t_end=2.0, label="Pick object")],
            arm_timelines=ArmTimelines(
                left=[ArmSpan(t_start=0.0, t_end=2.0, state="work", desc="left")],
                right=[ArmSpan(t_start=0.0, t_end=2.0, state="hold", desc="right")],
            ),
            qc_segments=[
                QcSegment(
                    seg_id="seg_1",
                    t_start=0.2,
                    t_end=0.8,
                    label_id="mid_idle_freeze",
                    training_role="badcase",
                    splice_policy="cut_clean",
                    vlm_raw_policy="cut_clean",
                )
            ],
            success=SuccessVerdict(passed=True, verdict="success", confidence=0.95),
        )
        repo.save_vlm_run_artifact(run)
        if with_few_shot:
            tile_rel = Path("artifacts") / "few_shots" / "fs_live_1.jpg"
            tile_abs = ws / tile_rel
            tile_abs.parent.mkdir(parents=True, exist_ok=True)
            ok, jpg_buf = cv2.imencode(".jpg", np.full((120, 160, 3), 120, dtype=np.uint8))
            assert ok
            tile_abs.write_bytes(jpg_buf.tobytes())
            repo.save_few_shot(
                FewShotItem(
                    example_id="fs_live_1",
                    scope="global",
                    task_id=meta.task_id,
                    label_id="mid_idle_freeze",
                    error_type="fp_overflag",
                    episode_id=meta.episode_id,
                    t_start=0.2,
                    t_end=0.8,
                    tile_image_path=tile_rel.as_posix(),
                    vlm_wrong_json={"label_id": "mid_idle_freeze"},
                    human_right_json={"action": "delete"},
                    rationale="live few-shot smoke",
                    enabled=True,
                    created_at=time.time(),
                )
            )
    finally:
        repo.close()
    return ws, cfg_dir, meta.episode_id, "run_live_1"


@contextmanager
def _live_uvicorn(ws: Path, cfg_dir: Path) -> Iterator[str]:
    port = _free_port()
    app = create_app(
        workspace=ws,
        config_dir=cfg_dir,
        reviewer_token="rev-secret",
        lead_token="lead-secret",
    )
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 10.0
    try:
        with httpx.Client(trust_env=False, timeout=2.0) as client:
            while time.time() < deadline:
                try:
                    r = client.get(f"{base_url}/api/health")
                    if r.status_code == 200:
                        break
                except httpx.HTTPError:
                    time.sleep(0.05)
            else:
                raise RuntimeError("uvicorn failed to become ready in time")
        yield base_url
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


def _require_browser_prereqs() -> tuple[Path, object]:
    web_dist_index = REPO_ROOT / "web" / "dist" / "index.html"
    if not web_dist_index.is_file():
        pytest.skip("web/dist/index.html not built")
    chrome_bin = Path("/usr/bin/google-chrome")
    if not chrome_bin.is_file():
        pytest.skip("google-chrome not installed")
    pw_mod = pytest.importorskip("playwright.sync_api")
    return chrome_bin, pw_mod


def test_preview_video_http_range_206_and_query_token(tmp_path: Path) -> None:
    ws, cfg_dir, ep_id, _ = _seed_live_workspace(tmp_path)
    with _live_uvicorn(ws, cfg_dir) as base_url:
        with httpx.Client(trust_env=False, timeout=15.0) as client:
            r_unauth = client.get(f"{base_url}/api/episodes/{ep_id}/preview")
            assert r_unauth.status_code == 401
            r_bad = client.get(f"{base_url}/api/episodes/{ep_id}/preview?token=wrong")
            assert r_bad.status_code == 401

            r_range = client.get(
                f"{base_url}/api/episodes/{ep_id}/preview?camera=main&token=rev-secret",
                headers={"Range": "bytes=0-1023"},
            )
            assert r_range.status_code == 206
            assert r_range.headers.get("accept-ranges") == "bytes"
            assert r_range.headers.get("content-range", "").startswith("bytes 0-1023/")
            assert len(r_range.content) == 1024


def test_frame_endpoint_nan_negative_and_unknown_camera(tmp_path: Path) -> None:
    ws, cfg_dir, ep_id, _ = _seed_live_workspace(tmp_path)
    with _live_uvicorn(ws, cfg_dir) as base_url:
        with httpx.Client(trust_env=False, timeout=10.0) as client:
            hdr = {"X-Auth-Token": "rev-secret"}
            r_nan = client.get(
                f"{base_url}/api/episodes/{ep_id}/frame?camera=main&t_s=NaN", headers=hdr
            )
            assert r_nan.status_code == 400

            r_neg = client.get(
                f"{base_url}/api/episodes/{ep_id}/frame?camera=main&t_s=-1.0", headers=hdr
            )
            assert r_neg.status_code == 200
            assert r_neg.content[:2] == b"\xff\xd8"

            r_cam = client.get(
                f"{base_url}/api/episodes/{ep_id}/frame?camera=unknown_cam&t_s=0.1", headers=hdr
            )
            assert r_cam.status_code == 400

            r_miss = client.get(
                f"{base_url}/api/episodes/no_ds/no_ep/frame?t_s=0.1", headers=hdr
            )
            assert r_miss.status_code == 404


def test_live_sse_events_and_job_terminal_failure_stream(tmp_path: Path) -> None:
    ws, cfg_dir, ep_id, run_id = _seed_live_workspace(tmp_path)
    with _live_uvicorn(ws, cfg_dir) as base_url:
        received: list[dict[str, object]] = []
        ready = threading.Event()
        stop = threading.Event()

        def _sse_reader() -> None:
            with httpx.Client(trust_env=False, timeout=15.0) as sc:
                with sc.stream("GET", f"{base_url}/api/events?token=lead-secret") as resp:
                    assert resp.status_code == 200
                    for line in resp.iter_lines():
                        if stop.is_set():
                            break
                        if line.startswith("data: "):
                            payload = json.loads(line[6:])
                            received.append(payload)
                            if payload.get("type") == "connected":
                                ready.set()
                            if any(ev.get("type") == "review_saved" for ev in received):
                                break

        t = threading.Thread(target=_sse_reader, daemon=True)
        t.start()
        assert ready.wait(timeout=5.0), "SSE connected event not received"

        with httpx.Client(trust_env=False, timeout=10.0) as client:
            rev_h = {"X-Auth-Token": "rev-secret", "X-User": "alice"}
            assert (
                client.post(
                    f"{base_url}/api/episodes/{ep_id}/claim", json={"user": "alice"}, headers=rev_h
                ).status_code
                == 200
            )
            assert (
                client.post(
                    f"{base_url}/api/episodes/{ep_id}/reviews",
                    headers=rev_h,
                    json={
                        "base_run_id": run_id,
                        "action": "save_draft",
                        "qc_segments": [],
                        "success": {
                            "passed": True,
                            "verdict": "success",
                            "confidence": 0.9,
                            "criteria": [],
                        },
                        "macro_stages": [],
                        "watch_coverage": {
                            "watched_intervals": [[0.0, 1.0]],
                            "watched_s": 1.0,
                            "duration_s": 2.0,
                            "coverage_ratio": 0.5,
                        },
                    },
                ).status_code
                == 200
            )
            t.join(timeout=5.0)
            stop.set()

            ev_types = [ev.get("type") for ev in received]
            assert "connected" in ev_types
            assert "episode_claimed" in ev_types
            assert "review_saved" in ev_types

            # Trigger job with nonexistent episode_id and stream /api/jobs/{id}/events
            job_resp = client.post(
                f"{base_url}/api/jobs",
                headers={"X-Auth-Token": "lead-secret"},
                json={"job_type": "render_mp4", "episode_ids": ["missing_ds/missing_ep"]},
            )
            assert job_resp.status_code in (200, 201)
            job_id = job_resp.json()["job_id"]

            job_events: list[dict[str, object]] = []
            with client.stream(
                "GET", f"{base_url}/api/jobs/{job_id}/events?token=lead-secret"
            ) as jstream:
                assert jstream.status_code == 200
                for line in jstream.iter_lines():
                    if line.startswith("data: "):
                        ev = json.loads(line[6:])
                        job_events.append(ev)
                        if ev.get("status") in {"completed", "failed", "cancelled"}:
                            break
            assert job_events
            assert job_events[-1]["status"] == "failed"
            assert job_events[-1]["summary"]["errors"]


def test_claim_lease_expiry_renew_409_and_reclaim(tmp_path: Path) -> None:
    ws, cfg_dir, ep_id, run_id = _seed_live_workspace(tmp_path, claim_ttl_s=0.5)
    with _live_uvicorn(ws, cfg_dir) as base_url:
        with httpx.Client(trust_env=False, timeout=10.0) as client:
            h_a = {"X-Auth-Token": "rev-secret", "X-User": "alice"}
            h_b = {"X-Auth-Token": "rev-secret", "X-User": "bob"}
            assert (
                client.post(
                    f"{base_url}/api/episodes/{ep_id}/claim", json={"user": "alice"}, headers=h_a
                ).status_code
                == 200
            )
            time.sleep(0.75)

            # Expired lease -> renew returns 409, B can claim, A's late save_draft returns 409
            assert (
                client.post(
                    f"{base_url}/api/episodes/{ep_id}/renew", json={"user": "alice"}, headers=h_a
                ).status_code
                == 409
            )
            assert (
                client.post(
                    f"{base_url}/api/episodes/{ep_id}/claim", json={"user": "bob"}, headers=h_b
                ).status_code
                == 200
            )
            r_late = client.post(
                f"{base_url}/api/episodes/{ep_id}/reviews",
                headers=h_a,
                json={
                    "base_run_id": run_id,
                    "action": "save_draft",
                    "qc_segments": [],
                    "success": {
                        "passed": True,
                        "verdict": "success",
                        "confidence": 0.9,
                        "criteria": [],
                    },
                    "macro_stages": [],
                    "watch_coverage": {
                        "watched_intervals": [[0.0, 1.0]],
                        "watched_s": 1.0,
                        "duration_s": 2.0,
                        "coverage_ratio": 0.5,
                    },
                },
            )
            assert r_late.status_code == 409


def test_browser_chrome_smoke(tmp_path: Path) -> None:
    """Headless Chrome smoke against live uvicorn + built web/dist/ covering AuthModal, video, 2x watch, Few-Shot img, Export ZIP."""
    chrome_bin, pw_mod = _require_browser_prereqs()

    ws, cfg_dir, ep_id, run_id = _seed_live_workspace(tmp_path, with_few_shot=True)
    # Seed a prior review on ep_001 so review_diff is non-null when Lead opens ReviewStudioPage,
    # and seed ep_002 as vlm_ready so Reviewer's default QueuePage tab shows ep_002 immediately on login.
    ep2_path = tmp_path / "data" / "demo_ds" / "ep_002.mp4"
    _write_mp4(ep2_path, n_frames=60, fps=30.0)
    h2 = [h for h in auto_detect(tmp_path / "data" / "demo_ds") if h.stem == "ep_002"][0]
    with get_adapter(h2.adapter_type).open(h2) as v2:
        meta2 = v2.meta
    repo = Repository(ws)
    try:
        repo.ingest_episode(meta2)
        repo.save_vlm_run_artifact(
            VlmRunArtifact(
                run_id="run_live_2",
                episode_id=meta2.episode_id,
                created_at=time.time(),
                config_fingerprint=ConfigFingerprintSnapshot(
                    model_routing={"s1_grounding": "gpt-4o"},
                    prompt_bundle_version="v1",
                    label_taxonomy_version="v1",
                    task_spec_version="1.0.0",
                ),
                success=SuccessVerdict(passed=True, verdict="success", confidence=0.95),
            )
        )
        repo.save_review(
            ep_id,
            "alice",
            "reviewer",
            ReviewWritePayload(
                base_run_id=run_id,
                action="submit_review",
                qc_segments=[
                    QcSegment(
                        seg_id="seg_1",
                        t_start=0.2,
                        t_end=0.8,
                        label_id="mid_idle_freeze",
                        training_role="badcase",
                        splice_policy="cut_clean",
                        vlm_raw_policy="cut_clean",
                    )
                ],
                success=SuccessVerdict(passed=True, verdict="success", confidence=0.95),
                macro_stages=[
                    MacroStage(stage_id="S1", t_start=0.0, t_end=2.0, label="Pick object")
                ],
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

    with _live_uvicorn(ws, cfg_dir) as base_url:
        with pw_mod.sync_playwright() as p:  # type: ignore[attr-defined]
            browser = p.chromium.launch(
                executable_path=str(chrome_bin),
                headless=True,
                args=["--no-sandbox", "--disable-gpu"],
            )
            try:
                page = browser.new_page()
                # 1. Unauthenticated / -> 401 -> AuthModal -> Reviewer login -> Queue reloads automatically
                page.goto(f"{base_url}/", wait_until="networkidle")
                page.get_by_test_id("auth-modal").wait_for(state="visible", timeout=5000)
                modal = page.get_by_test_id("auth-modal")
                modal.locator('input[type="password"]').fill("rev-secret")
                modal.locator('input[type="text"]').fill("rev_chrome")
                modal.locator('button[type="submit"]').click()
                page.get_by_test_id(f"episode-row-{meta2.episode_id}").wait_for(
                    state="visible", timeout=5000
                )
                assert page.locator(".error-banner").count() == 0

                # 2. Lead opens ReviewStudioPage -> <video> playable + 2x watch_coverage + live DiffOverlay
                page.goto(
                    f"{base_url}/review/{ep_id}?token=lead-secret&user=lead_chrome",
                    wait_until="networkidle",
                )
                page.get_by_test_id("review-studio-page").wait_for(state="visible", timeout=5000)
                page.wait_for_function(
                    "() => { const v = document.querySelector('video'); return v && v.readyState >= 1 && v.error === null; }",
                    timeout=10000,
                )
                # Click 2x playback speed and dispatch timeupdate steps of 0.7s
                page.get_by_test_id("studio-player").get_by_role("button", name="2x").click()
                page.evaluate(
                    """() => {
                        const v = document.querySelector('video');
                        for (const t of [0.7, 1.4, 2.0]) {
                            Object.defineProperty(v, 'currentTime', { value: t, configurable: true });
                            v.dispatchEvent(new Event('timeupdate'));
                        }
                    }"""
                )
                page.wait_for_function(
                    "() => document.querySelector('[data-testid=\"watch-coverage-badge\"]')?.textContent?.includes('100%')",
                    timeout=5000,
                )

                # Delete seg_1 -> DiffOverlay must show fp_deleted = 1 immediately
                page.get_by_test_id("delete-segment-btn").click()
                page.get_by_test_id("diff-tab-btn").click()
                diff_text = page.get_by_test_id("diff-metrics").inner_text()
                assert "1" in diff_text

                # 3. VLM Lab Few-Shot tile image naturalWidth > 0 & Export ZIP HTTP 200
                page.goto(
                    f"{base_url}/lab?token=lead-secret&user=lead_chrome",
                    wait_until="networkidle",
                )
                page.get_by_test_id("tab-few-shots").click()
                page.get_by_test_id("fewshot-card-fs_live_1").wait_for(
                    state="visible", timeout=5000
                )
                page.wait_for_function(
                    "() => { const img = document.querySelector('[data-testid=\"fewshot-card-fs_live_1\"] img'); return img && img.complete && img.naturalWidth > 0; }",
                    timeout=5000,
                )

                page.goto(
                    f"{base_url}/export?token=lead-secret&user=lead_chrome",
                    wait_until="networkidle",
                )
                page.get_by_test_id("format-card-zero_copy_index").click()
                page.locator("input[type='checkbox']").nth(2).check()  # reviewed
                page.get_by_test_id("trigger-export-btn").click()
                page.get_by_test_id("export-manifest-preview").wait_for(
                    state="visible", timeout=10000
                )
                dl_href = page.locator("a[download]").first.get_attribute("href")
                assert dl_href
                with httpx.Client(trust_env=False, timeout=10.0) as client:
                    r_zip = client.get(f"{base_url}{dl_href}")
                    assert r_zip.status_code == 200
                    with zipfile.ZipFile(BytesIO(r_zip.content)) as zf:
                        assert "qc_index.parquet" in zf.namelist()
            finally:
                browser.close()


def _seed_many_episodes_for_queue_tests(tmp_path: Path, count: int) -> tuple[Path, Path]:
    ws = tmp_path / "ws_q"
    cfg_dir = tmp_path / "configs_q"
    shutil.copytree(default_config_dir(), cfg_dir)
    mp4_path = tmp_path / "data" / "demo_ds" / "ep_001.mp4"
    _write_mp4(mp4_path, n_frames=10, fps=30.0)
    handle = auto_detect(tmp_path / "data" / "demo_ds")[0]
    with get_adapter(handle.adapter_type).open(handle) as view:
        base_meta = view.meta

    repo = Repository(ws)
    try:
        for i in range(1, count + 1):
            ep_id = f"demo_ds/ep_{i:03d}"
            meta = base_meta.model_copy(
                update={
                    "episode_id": ep_id,
                    "stem": f"ep_{i:03d}",
                    "episode_fingerprint": f"fp_{i:03d}",
                }
            )
            repo.ingest_episode(meta)
            repo.save_vlm_run_artifact(
                VlmRunArtifact(
                    run_id=f"run_{i:03d}",
                    episode_id=ep_id,
                    created_at=1.0,
                    config_fingerprint=ConfigFingerprintSnapshot(
                        model_routing={"s1_grounding": "gpt-4o"},
                        prompt_bundle_version="v1",
                        label_taxonomy_version="v1",
                        task_spec_version="1.0.0",
                    ),
                    success=SuccessVerdict(passed=True, verdict="success", confidence=0.95),
                )
            )
    finally:
        repo.close()
    return ws, cfg_dir


def test_browser_queue_pagination_and_server_filters(tmp_path: Path) -> None:
    """Headless Chrome regression for Lead filter page-reset and Reviewer submitted/my_claimed server filters."""
    chrome_bin, pw_mod = _require_browser_prereqs()

    # 1. Lead pagination reset: 205 episodes (ep_001..ep_002 reviewed, ep_003..ep_205 vlm_ready)
    ws1, cfg1 = _seed_many_episodes_for_queue_tests(tmp_path / "g1", 205)
    repo1 = Repository(ws1)
    try:
        for idx in (1, 2):
            repo1.save_review(
                f"demo_ds/ep_{idx:03d}",
                "rev_alice",
                "reviewer",
                ReviewWritePayload(
                    base_run_id=f"run_{idx:03d}",
                    action="submit_review",
                    qc_segments=[],
                    success=SuccessVerdict(passed=True, verdict="success", confidence=0.95),
                    macro_stages=[],
                    watch_coverage=WatchCoverage(
                        watched_intervals=[(0.0, 0.3)],
                        watched_s=0.3,
                        duration_s=0.3,
                        coverage_ratio=1.0,
                    ),
                ),
            )
    finally:
        repo1.close()

    with _live_uvicorn(ws1, cfg1) as base_url:
        with pw_mod.sync_playwright() as p:  # type: ignore[attr-defined]
            browser = p.chromium.launch(
                executable_path=str(chrome_bin),
                headless=True,
                args=["--no-sandbox", "--disable-gpu"],
            )
            try:
                page = browser.new_page()
                page.goto(
                    f"{base_url}/?token=lead-secret&user=lead_chrome", wait_until="networkidle"
                )
                page.get_by_test_id("queue-pagination").wait_for(state="visible", timeout=5000)
                pag_text = page.get_by_test_id("queue-pagination").inner_text()
                assert "1 / 3" in pag_text and "205" in pag_text

                # Navigate to Page 3
                page.get_by_test_id("queue-next-page").click()
                page.get_by_test_id("episode-row-demo_ds/ep_101").wait_for(
                    state="visible", timeout=5000
                )
                page.get_by_test_id("queue-next-page").click()
                page.get_by_test_id("episode-row-demo_ds/ep_205").wait_for(
                    state="visible", timeout=5000
                )

                # Click "Reviewed" status pill -> must reset to Page 1 and display both reviewed rows
                page.get_by_test_id("lead-status-strip").get_by_role(
                    "button", name=re.compile(r"^Reviewed\b", re.IGNORECASE)
                ).click()
                page.get_by_test_id("episode-row-demo_ds/ep_001").wait_for(
                    state="visible", timeout=5000
                )
                page.get_by_test_id("episode-row-demo_ds/ep_002").wait_for(
                    state="visible", timeout=5000
                )
                assert page.locator("tbody td.empty-cell").count() == 0
            finally:
                browser.close()

    # 2. Reviewer server filters: 120 episodes (ep_001..ep_110 vlm_ready/claimed_by_other,
    # ep_111..ep_120 reviewed/flagged, plus ep_109..ep_110 claimed by rev_chrome)
    ws2, cfg2 = _seed_many_episodes_for_queue_tests(tmp_path / "g2", 120)
    repo2 = Repository(ws2)
    try:
        # Claim ep_001..ep_105 by another reviewer so rev_chrome's claims sit beyond offset 100 in global order
        for idx in range(1, 106):
            repo2.claim_episode(f"demo_ds/ep_{idx:03d}", "other_reviewer")
        for idx in (109, 110):
            repo2.claim_episode(f"demo_ds/ep_{idx:03d}", "rev_chrome")
        for idx in range(111, 121):
            is_flag = idx == 120
            repo2.save_review(
                f"demo_ds/ep_{idx:03d}",
                "rev_chrome",
                "reviewer",
                ReviewWritePayload(
                    base_run_id=f"run_{idx:03d}",
                    action="flag" if is_flag else "submit_review",
                    qc_segments=[],
                    success=SuccessVerdict(
                        passed=not is_flag,
                        verdict="failure" if is_flag else "success",
                        confidence=0.9,
                    ),
                    macro_stages=[],
                    watch_coverage=WatchCoverage(
                        watched_intervals=[(0.0, 0.3)],
                        watched_s=0.3,
                        duration_s=0.3,
                        coverage_ratio=1.0,
                    ),
                    flagged_reason="uncertain" if is_flag else None,
                ),
            )
    finally:
        repo2.close()

    with _live_uvicorn(ws2, cfg2) as base_url:
        with pw_mod.sync_playwright() as p:  # type: ignore[attr-defined]
            browser = p.chromium.launch(
                executable_path=str(chrome_bin),
                headless=True,
                args=["--no-sandbox", "--disable-gpu"],
            )
            try:
                page = browser.new_page()
                page.goto(
                    f"{base_url}/?token=rev-secret&user=rev_chrome", wait_until="networkidle"
                )
                page.get_by_test_id("reviewer-queue-controls").wait_for(
                    state="visible", timeout=5000
                )
                tabs = page.get_by_test_id("reviewer-queue-controls").locator("button")

                # Submitted / Flagged tab (3rd tab) -> all 10 reviewed/flagged rows (ep_111..ep_120) on Page 1
                tabs.nth(2).click()
                for idx in range(111, 121):
                    page.get_by_test_id(f"episode-row-demo_ds/ep_{idx:03d}").wait_for(
                        state="visible", timeout=5000
                    )

                # My Claimed tab (2nd tab, count = 2) -> both ep_109 & ep_110 claimed by rev_chrome on Page 1
                assert "(2)" in tabs.nth(1).inner_text()
                tabs.nth(1).click()
                page.get_by_test_id("episode-row-demo_ds/ep_109").wait_for(
                    state="visible", timeout=5000
                )
                page.get_by_test_id("episode-row-demo_ds/ep_110").wait_for(
                    state="visible", timeout=5000
                )
            finally:
                browser.close()
