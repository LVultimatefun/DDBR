"""Tests for GET /api/episodes SQL counts, facets, search, sort, and /runs historical views."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from robo_vlm_qc.config import ConfigManager
from robo_vlm_qc.models.annotation import QcSegment, SuccessVerdict, VisualEvidence
from robo_vlm_qc.models.review import EpisodeStatus, ReviewWritePayload, WatchCoverage
from robo_vlm_qc.store import Repository
from tests.api.conftest import (
    LEAD_HEADERS,
    REV_HEADERS,
    create_video_episode_file,
    ingest_dir,
    make_client,
    seed_vlm_run,
)
from tests.e2e.helpers import make_synthetic_h5_raw_v3


def test_episodes_sql_status_counts_facets_search_and_sort(api_env: dict[str, Any]) -> None:
    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]
    data_root: Path = api_env["data_root"]

    ds_alpha = data_root / "alpha"
    ds_beta = data_root / "beta"
    make_synthetic_h5_raw_v3(ds_alpha, name="ep_100%.h5", duration_s=4.0, stationary=True)
    make_synthetic_h5_raw_v3(ds_alpha, name="ep_200_sub.h5", duration_s=6.0, stationary=True)
    create_video_episode_file(ds_beta, "ep_vid_1.mp4", duration_s=2.0, fps=15.0)
    create_video_episode_file(ds_beta, "ep_vid_2.mp4", duration_s=3.0, fps=15.0)

    repo = Repository(ws)
    try:
        alpha_ids = sorted(ingest_dir(repo, cfg_dir, ws, ds_alpha, dataset="alpha"))
        beta_ids = sorted(ingest_dir(repo, cfg_dir, ws, ds_beta, dataset="beta"))
        cfg = ConfigManager(config_dir=cfg_dir, workspace_dir=ws)

        # ep 0 (alpha/ep_100%): leave as ingested (NULL cut_ratio / mask_ratio)
        # ep 1 (alpha/ep_200_sub): vlm_ready with high cut_ratio
        seg_cut = QcSegment(
            seg_id="s_cut",
            t_start=0.0,
            t_end=3.0,
            label_id="boundary_idle",
            training_role="badcase",
            splice_policy="cut_clean",
            mask_side=None,
            affected_arm="both",
            visual_evidence=VisualEvidence(),
            vlm_raw_policy="cut_clean",
        )
        seed_vlm_run(repo, cfg, alpha_ids[1], run_id="run_a2", segments=[seg_cut])

        # beta 0: claimed by reviewer_1 after vlm_ready
        seg_mask = QcSegment(
            seg_id="s_mask",
            t_start=0.0,
            t_end=1.0,
            label_id="missed_grasp_retry",
            training_role="badcase",
            splice_policy="mask_loss",
            mask_side="left",
            affected_arm="left",
            visual_evidence=VisualEvidence(),
            vlm_raw_policy="mask_loss",
        )
        seed_vlm_run(repo, cfg, beta_ids[0], run_id="run_b1", segments=[seg_mask])
        repo.claim_episode(beta_ids[0], "reviewer_1")

        # beta 1: approved
        seed_vlm_run(repo, cfg, beta_ids[1], run_id="run_b2", segments=[])
        repo.claim_episode(beta_ids[1], "lead_1")
        repo.save_review(
            beta_ids[1],
            author="lead_1",
            role="lead",
            payload=ReviewWritePayload(
                base_run_id="run_b2",
                action="approve",
                qc_segments=[],
                success=SuccessVerdict(
                    verdict="success", passed=True, confidence=0.95, summary="ok"
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
        repo.close()

    with make_client(ws, cfg_dir) as client:
        # Filter by status=vlm_ready: status_counts MUST still include ingested, claimed, approved!
        r_st = client.get("/api/episodes?status=vlm_ready", headers=REV_HEADERS)
        assert r_st.status_code == 200
        b_st = r_st.json()
        assert b_st["total"] == 1
        assert [it["episode_id"] for it in b_st["items"]] == [alpha_ids[1]]
        sc = b_st["status_counts"]
        assert set(sc.keys()) == {st.value for st in EpisodeStatus}
        assert sc["ingested"] == 1
        assert sc["vlm_ready"] == 1
        assert sc["claimed"] == 1
        assert sc["approved"] == 1

        # Comma-separated status filter returns union while status_counts stays global
        r_multi = client.get("/api/episodes?status=vlm_ready,approved", headers=REV_HEADERS)
        assert r_multi.status_code == 200
        b_multi = r_multi.json()
        assert b_multi["total"] == 2
        assert {it["episode_id"] for it in b_multi["items"]} == {alpha_ids[1], beta_ids[1]}
        assert b_multi["status_counts"]["ingested"] == 1
        assert b_multi["status_counts"]["claimed"] == 1

        # Filter by adapter_type scopes items, total, and status_counts while facets stay workspace-global
        r_ad = client.get("/api/episodes?adapter_type=video_dir", headers=REV_HEADERS)
        assert r_ad.status_code == 200
        b_ad = r_ad.json()
        assert b_ad["total"] == 2
        assert {it["episode_id"] for it in b_ad["items"]} == set(beta_ids)
        assert b_ad["status_counts"]["claimed"] == 1
        assert b_ad["status_counts"]["approved"] == 1
        assert b_ad["status_counts"]["ingested"] == 0
        assert b_ad["facets"]["datasets"] == ["alpha", "beta"]
        assert "h5_raw_v3" in b_ad["facets"]["adapter_types"]

        # Filter by claimed_by does NOT zero out status_counts
        r_cl = client.get("/api/episodes?claimed_by=reviewer_1", headers=REV_HEADERS)
        assert r_cl.status_code == 200
        assert r_cl.json()["total"] == 1
        assert r_cl.json()["status_counts"]["approved"] == 1

        # Filter by dataset scopes status_counts to that dataset, while facets remain workspace-global
        r_ds = client.get("/api/episodes?dataset=alpha", headers=REV_HEADERS)
        assert r_ds.status_code == 200
        b_ds = r_ds.json()
        assert b_ds["status_counts"]["ingested"] == 1
        assert b_ds["status_counts"]["vlm_ready"] == 1
        assert b_ds["status_counts"]["approved"] == 0
        assert b_ds["facets"]["datasets"] == ["alpha", "beta"]
        assert "h5_raw_v3" in b_ds["facets"]["adapter_types"]
        assert "video_dir" in b_ds["facets"]["adapter_types"]
        assert len(b_ds["facets"]["task_ids"]) >= 1

        # Search LIKE escaping (% and _) and SQL injection safety
        r_pct = client.get("/api/episodes?search=100%25", headers=REV_HEADERS)
        assert r_pct.status_code == 200
        assert [it["episode_id"] for it in r_pct.json()["items"]] == [alpha_ids[0]]

        r_und = client.get("/api/episodes?search=200_sub", headers=REV_HEADERS)
        assert r_und.status_code == 200
        assert [it["episode_id"] for it in r_und.json()["items"]] == [alpha_ids[1]]

        r_sqli = client.get("/api/episodes?search=' OR 1=1--", headers=REV_HEADERS)
        assert r_sqli.status_code == 200
        assert r_sqli.json()["total"] == 0

        # Sort across all 6 columns in asc/desc, NULLs last
        for col in (
            "episode_id",
            "updated_at",
            "duration_s",
            "cut_ratio",
            "mask_ratio",
            "badcase_count",
        ):
            for sdir in ("asc", "desc"):
                rs = client.get(
                    f"/api/episodes?sort_by={col}&sort_dir={sdir}",
                    headers=REV_HEADERS,
                )
                assert rs.status_code == 200, (col, sdir, rs.text)
                items = rs.json()["items"]
                assert len(items) == 4
                if col == "duration_s":
                    durs = [it["duration_s"] for it in items]
                    assert durs == sorted(durs, reverse=(sdir == "desc"))
                if col in ("cut_ratio", "mask_ratio"):
                    # alpha/ep_100% has NULL ratio -> must sort last in both asc & desc
                    assert items[-1]["episode_id"] == alpha_ids[0]

        # SQL injection / invalid sort_by & sort_dir -> 422
        assert (
            client.get(
                "/api/episodes?sort_by=episode_id; DROP TABLE episodes;--",
                headers=REV_HEADERS,
            ).status_code
            == 422
        )
        assert (
            client.get("/api/episodes?sort_dir=asc; --", headers=REV_HEADERS).status_code == 422
        )


def test_episode_runs_and_historical_readonly_view(api_env: dict[str, Any]) -> None:
    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]
    data_dir = api_env["data_root"] / "ds_runs"
    make_synthetic_h5_raw_v3(data_dir, name="ep_r1.h5", duration_s=4.0, stationary=True)
    make_synthetic_h5_raw_v3(data_dir, name="ep_r2.h5", duration_s=4.0, stationary=True)
    make_synthetic_h5_raw_v3(data_dir, name="ep_r0.h5", duration_s=4.0, stationary=True)

    repo = Repository(ws)
    try:
        ep_r0 = ingest_dir(repo, cfg_dir, ws, data_dir / "ep_r0.h5", dataset="ds_runs")[0]
        ep_r1 = ingest_dir(repo, cfg_dir, ws, data_dir / "ep_r1.h5", dataset="ds_runs")[0]
        ep_r2 = ingest_dir(repo, cfg_dir, ws, data_dir / "ep_r2.h5", dataset="ds_runs")[0]
        cfg = ConfigManager(config_dir=cfg_dir, workspace_dir=ws)

        t_base = time.time() - 100.0
        seed_vlm_run(
            repo,
            cfg,
            ep_r1,
            run_id="run_old",
            created_at=t_base,
            prompt_version="p1",
            passed=False,
            verdict="failure",
        )
        seed_vlm_run(
            repo,
            cfg,
            ep_r1,
            run_id="run_new",
            created_at=t_base + 50.0,
            prompt_version="p2",
            passed=True,
            verdict="success",
        )
        seed_vlm_run(
            repo,
            cfg,
            ep_r2,
            run_id="run_other_ep",
            created_at=t_base + 60.0,
        )

        # Add a human review on ep_r1 so active_rev_id is populated
        repo.claim_episode(ep_r1, "reviewer_1")
        rev_art = repo.save_review(
            ep_r1,
            author="reviewer_1",
            role="reviewer",
            payload=ReviewWritePayload(
                base_run_id="run_new",
                action="submit_review",
                qc_segments=[],
                success=SuccessVerdict(
                    verdict="success", passed=True, confidence=0.95, summary="ok"
                ),
                watch_coverage=WatchCoverage(
                    watched_intervals=[[0.0, 4.0]],
                    watched_s=4.0,
                    duration_s=4.0,
                    coverage_ratio=1.0,
                ),
            ),
            min_watch_coverage=0.95,
        )
    finally:
        repo.close()

    with make_client(ws, cfg_dir) as client:
        # Unauth -> 401, Reviewer -> 403
        assert client.get(f"/api/episodes/{ep_r1}/runs").status_code == 401
        assert client.get(f"/api/episodes/{ep_r1}/runs", headers=REV_HEADERS).status_code == 403
        assert (
            client.get(f"/api/episodes/{ep_r1}?run_id=run_old", headers=REV_HEADERS).status_code
            == 403
        )
        # Reviewer without run_id still gets 200
        r_rev_cur = client.get(f"/api/episodes/{ep_r1}", headers=REV_HEADERS)
        assert r_rev_cur.status_code == 200
        assert r_rev_cur.json()["is_historical_run_view"] is False
        assert r_rev_cur.json()["selected_run_id"] == "run_new"

        # Lead gets empty list on episode with 0 runs
        r_empty = client.get(f"/api/episodes/{ep_r0}/runs", headers=LEAD_HEADERS)
        assert r_empty.status_code == 200
        assert r_empty.json() == {"episode_id": ep_r0, "active_run_id": None, "runs": []}

        # Missing episode -> 404
        assert (
            client.get("/api/episodes/ds_runs/missing/runs", headers=LEAD_HEADERS).status_code
            == 404
        )

        # Lead gets newest-first runs with ISO-8601 UTC string created_at
        r_runs = client.get(f"/api/episodes/{ep_r1}/runs", headers=LEAD_HEADERS)
        assert r_runs.status_code == 200
        b_runs = r_runs.json()
        assert b_runs["active_run_id"] == "run_new"
        assert [r["run_id"] for r in b_runs["runs"]] == ["run_new", "run_old"]
        assert isinstance(b_runs["runs"][0]["created_at"], str)
        assert "T" in b_runs["runs"][0]["created_at"]

        # Lead queries historical ?run_id=run_old (read-only view, active_rev_id=None)
        r_hist = client.get(f"/api/episodes/{ep_r1}?run_id=run_old", headers=LEAD_HEADERS)
        assert r_hist.status_code == 200
        b_hist = r_hist.json()
        assert b_hist["selected_run_id"] == "run_old"
        assert b_hist["active_run_id"] == "run_old"
        assert b_hist["is_historical_run_view"] is True
        assert b_hist["active_rev_id"] is None
        assert b_hist["is_human_reviewed"] is False
        assert b_hist["success"]["verdict"] == "failure"

        # Cross-episode run_id -> 404! Missing run_id -> 404! Blank run_id -> 422!
        assert (
            client.get(
                f"/api/episodes/{ep_r1}?run_id=run_other_ep", headers=LEAD_HEADERS
            ).status_code
            == 404
        )
        assert (
            client.get(
                f"/api/episodes/{ep_r1}?run_id=no_such_run", headers=LEAD_HEADERS
            ).status_code
            == 404
        )
        assert client.get(f"/api/episodes/{ep_r1}?run_id=", headers=LEAD_HEADERS).status_code == 422

    # Verify zero DB mutation after historical read
    repo_check = Repository(ws)
    try:
        row_after = repo_check.get_episode_row(ep_r1)
        assert row_after["active_run_id"] == "run_new"
        assert row_after["active_rev_id"] == rev_art.rev_id
        assert row_after["status"] == EpisodeStatus.REVIEWED.value
    finally:
        repo_check.close()
