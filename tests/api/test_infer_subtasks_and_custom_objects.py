"""API tests for POST /api/episodes/{ep_id}/infer-subtasks and POST /api/jobs custom_objects."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from robo_vlm_qc.config import ConfigManager
from robo_vlm_qc.models.annotation import (
    GroundingKeyframe,
    GroundingResult,
    GroundingTrack,
    MacroStage,
    QcSegment,
    SuccessVerdict,
    VisualEvidence,
)
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
from tests.e2e.conftest import MockVlmServer
from tests.e2e.helpers import make_synthetic_h5_raw_v3


def _poll_job(client: Any, job_id: str, timeout_s: float = 20.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        r = client.get(f"/api/jobs/{job_id}", headers=LEAD_HEADERS)
        assert r.status_code == 200
        j = r.json()
        if j["status"] in ("completed", "failed", "cancelled"):
            return j
        time.sleep(0.04)
    raise AssertionError(f"Job {job_id} timed out")


def test_infer_subtasks_endpoint_auth_gt_frozen_validation_and_grounding_injection(
    api_env: dict[str, Any],
) -> None:
    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]
    mock_vlm: MockVlmServer = api_env["mock_vlm"]
    ds_dir = api_env["data_root"] / "ds_subtasks"

    create_video_episode_file(ds_dir, "ep_open.mp4", duration_s=4.0, fps=15.0)
    create_video_episode_file(ds_dir, "ep_noground.mp4", duration_s=3.0, fps=15.0)
    create_video_episode_file(ds_dir, "ep_frozen.mp4", duration_s=3.0, fps=15.0)

    repo = Repository(ws)
    try:
        ep_open = ingest_dir(repo, cfg_dir, ws, ds_dir / "ep_open.mp4", dataset="ds_subtasks")[0]
        ep_noground = ingest_dir(repo, cfg_dir, ws, ds_dir / "ep_noground.mp4", dataset="ds_subtasks")[0]
        ep_frozen = ingest_dir(repo, cfg_dir, ws, ds_dir / "ep_frozen.mp4", dataset="ds_subtasks")[0]
        cfg = ConfigManager(config_dir=cfg_dir, workspace_dir=ws)

        # Seed active VLM run WITH grounding tracks on ep_open
        art_open = seed_vlm_run(repo, cfg, ep_open, run_id="run_open_base", segments=[])
        art_with_g = art_open.model_copy(
            update={
                "run_id": "run_open_g",
                "grounding": GroundingResult(
                    schema_version=1,
                    keyframes_s=[0.0, 2.0, 4.0],
                    object_tracks_2d={
                        "green_mug": GroundingTrack(
                            object_id="green_mug",
                            name="Green Mug",
                            keyframes=[
                                GroundingKeyframe(row=0, time_s=0.0, visible=True, bbox_norm_1000=[100, 300, 220, 480], state="on counter"),
                                GroundingKeyframe(row=30, time_s=2.0, visible=True, bbox_norm_1000=[350, 200, 470, 380], state="lifted"),
                                GroundingKeyframe(row=60, time_s=4.0, visible=True, bbox_norm_1000=[620, 310, 740, 490], state="in rack"),
                            ],
                        )
                    },
                )
            }
        )
        repo.save_vlm_run_artifact(art_with_g)

        # Freeze ep_frozen
        seed_vlm_run(repo, cfg, ep_frozen, run_id="run_frz_1", segments=[])
        repo.claim_episode(ep_frozen, "lead_1")
        repo.save_review(
            ep_frozen,
            author="lead_1",
            role="lead",
            payload=ReviewWritePayload(
                base_run_id="run_frz_1",
                action="freeze_gt",
                qc_segments=[],
                success=SuccessVerdict(verdict="success", passed=True, confidence=0.95, summary="gt"),
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

    captured_prompts: list[str] = []

    def _mock_s2(req: dict[str, Any]) -> dict[str, Any]:
        text_blocks = []
        for msg in req["messages"]:
            c = msg.get("content", "")
            if isinstance(c, str):
                text_blocks.append(c)
            elif isinstance(c, list):
                text_blocks.extend(str(b.get("text", "")) for b in c if b.get("type") == "text")
        captured_prompts.append("\n".join(text_blocks))
        return {
            "macro_stages": [
                {
                    "stage_id": "S1",
                    "t_start": 0.0,
                    "t_end": 0.8,
                    "label": "Leading idle",
                    "label_en": "leading_idle",
                    "instruction": "Hold initial pose",
                    "summary": "Aligned to cut_clean",
                },
                {
                    "stage_id": "S2",
                    "t_start": 0.8,
                    "t_end": 4.0,
                    "label": "Move green mug to rack",
                    "label_en": "move_green_mug",
                    "instruction": "Pick up the green mug and place it in the rack",
                    "summary": "Primary manipulation span",
                },
            ]
        }

    mock_vlm.set_stage_response("s2", _mock_s2)

    valid_body = {
        "qc_segments": [
            QcSegment(
                seg_id="seg_cut",
                t_start=0.0,
                t_end=0.8,
                label_id="boundary_idle",
                training_role="badcase",
                splice_policy="cut_clean",
                vlm_raw_policy="cut_clean",
            ).model_dump(mode="json"),
            QcSegment(
                seg_id="seg_rec",
                t_start=1.5,
                t_end=2.2,
                label_id="missed_grasp_retry",
                training_role="recovery",
                splice_policy="mask_loss",
                mask_side="right",
                affected_arm="right",
                visual_evidence=VisualEvidence(process_change="Retry grasp on mug handle"),
                vlm_raw_policy="mask_loss",
            ).model_dump(mode="json"),
        ],
        "macro_stages": [
            MacroStage(stage_id="S1", t_start=0.0, t_end=4.0, label="Draft stage").model_dump(mode="json")
        ],
        "custom_hint": "Align S2 right after the 0.8s idle cut",
    }

    with make_client(ws, cfg_dir) as client:
        # 1. Unauth -> 401
        assert client.post(f"/api/episodes/{ep_open}/infer-subtasks", json=valid_body).status_code == 401

        # 2. Strict extra="forbid" -> 422, missing qc_segments -> 422
        assert (
            client.post(
                f"/api/episodes/{ep_open}/infer-subtasks",
                headers=REV_HEADERS,
                json={**valid_body, "unexpected_field": True},
            ).status_code
            == 422
        )
        assert (
            client.post(
                f"/api/episodes/{ep_open}/infer-subtasks",
                headers=REV_HEADERS,
                json={"macro_stages": []},
            ).status_code
            == 422
        )

        # 3. Missing episode -> 404
        assert (
            client.post(
                "/api/episodes/ds_subtasks/does_not_exist/infer-subtasks",
                headers=REV_HEADERS,
                json=valid_body,
            ).status_code
            == 404
        )

        # 4. gt_frozen -> 409 Conflict with ZERO VLM calls
        mock_vlm.requests.clear()
        r_frz = client.post(
            f"/api/episodes/{ep_frozen}/infer-subtasks",
            headers=LEAD_HEADERS,
            json=valid_body,
        )
        assert r_frz.status_code == 409
        assert "gt_frozen" in r_frz.json()["detail"]
        assert mock_vlm.requests == []

        # 5. Reviewer succeeds on ep_open WITH grounding tracks & verified QC segments
        r_ok = client.post(
            f"/api/episodes/{ep_open}/infer-subtasks",
            headers=REV_HEADERS,
            json=valid_body,
        )
        assert r_ok.status_code == 200, r_ok.text
        data = r_ok.json()
        assert data["episode_id"] == ep_open
        assert data["used_grounding"] is True
        assert data["used_segments_count"] == 2
        assert len(data["macro_stages"]) == 2
        assert data["macro_stages"][1]["t_start"] == 0.8
        assert (
            data["macro_stages"][1]["instruction"]
            == "Pick up the green mug and place it in the rack"
        )
        assert captured_prompts
        assert "CUT_CLEAN EXCLUDED" in captured_prompts[-1]
        assert "RECOVERY" in captured_prompts[-1]
        assert "green_mug (Green Mug)" in captured_prompts[-1]
        assert "Align S2 right after the 0.8s idle cut" in captured_prompts[-1]

        # 6. Episode without active grounding tracks reports used_grounding=False
        r_ng = client.post(
            f"/api/episodes/{ep_noground}/infer-subtasks",
            headers=LEAD_HEADERS,
            json={"qc_segments": []},
        )
        assert r_ng.status_code == 200, r_ng.text
        data_ng = r_ng.json()
        assert data_ng["used_grounding"] is False
        assert data_ng["used_segments_count"] == 0
        assert len(data_ng["macro_stages"]) >= 1

    # Verify endpoint did NOT mutate episode DB status or active_run_id
    repo_chk = Repository(ws)
    try:
        row_open = repo_chk.get_episode_row(ep_open)
        assert row_open["active_run_id"] == "run_open_g"
        assert row_open["status"] == EpisodeStatus.VLM_READY.value
    finally:
        repo_chk.close()


def test_jobs_batch_vlm_s1_custom_objects_override_and_cache_isolation(
    api_env: dict[str, Any],
) -> None:
    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]
    mock_vlm: MockVlmServer = api_env["mock_vlm"]
    ds_dir = api_env["data_root"] / "ds_custom_obj"

    make_synthetic_h5_raw_v3(ds_dir, name="ep_co_1.h5", duration_s=2.5, stationary=True)
    repo = Repository(ws)
    try:
        ep_id = ingest_dir(repo, cfg_dir, ws, ds_dir / "ep_co_1.h5", dataset="ds_custom_obj")[0]
    finally:
        repo.close()

    with make_client(ws, cfg_dir) as client:
        # 1. Warm full pipeline without custom_objects
        r1 = client.post(
            "/api/jobs",
            headers=LEAD_HEADERS,
            json={"job_type": "batch_vlm", "episode_ids": [ep_id], "stage": "all"},
        )
        assert r1.status_code == 201
        j1 = _poll_job(client, r1.json()["job_id"])
        assert j1["status"] == "completed", j1
        ep_v1 = client.get(f"/api/episodes/{ep_id}", headers=LEAD_HEADERS).json()
        run_id_v1 = ep_v1["active_run_id"]

        # 2. Rerun S1 only with custom_objects=["Green Mug", "Dish Rack"]
        mock_vlm.reset()
        r2 = client.post(
            "/api/jobs",
            headers=LEAD_HEADERS,
            json={
                "job_type": "batch_vlm",
                "episode_ids": [ep_id],
                "stage": "s1_grounding",
                "custom_objects": ["Green Mug", "Dish Rack"],
            },
        )
        assert r2.status_code == 201
        j2 = _poll_job(client, r2.json()["job_id"])
        assert j2["status"] == "completed", j2

        # Only S1 was invoked; S2-S5 reused cache
        called_stages = {req["stage"] for req in mock_vlm.requests}
        assert called_stages == {"S1"}, f"Expected only S1 calls, got {called_stages}"

        ep_v2 = client.get(f"/api/episodes/{ep_id}", headers=LEAD_HEADERS).json()
        run_id_v2 = ep_v2["active_run_id"]
        assert run_id_v2 != run_id_v1
        tracks = ep_v2["grounding"]["object_tracks_2d"]
        assert "green_mug" in tracks
        assert "dish_rack" in tracks
        assert tracks["green_mug"]["name"] == "Green Mug"
        assert tracks["dish_rack"]["name"] == "Dish Rack"

    repo_chk = Repository(ws)
    try:
        art1 = repo_chk.get_vlm_run(run_id_v1)
        art2 = repo_chk.get_vlm_run(run_id_v2)
        assert art2.stage_refs.s1_grounding != art1.stage_refs.s1_grounding
        assert art2.stage_refs.s2_macro == art1.stage_refs.s2_macro
        assert art2.stage_refs.s3_discover == art1.stage_refs.s3_discover
        assert art2.stage_refs.s5_success == art1.stage_refs.s5_success
    finally:
        repo_chk.close()
