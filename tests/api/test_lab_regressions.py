"""Regression tests for diagnosis tolerance, few-shot task isolation, and Lab API validation."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from robo_vlm_qc.api import create_app
from robo_vlm_qc.config import ConfigManager
from robo_vlm_qc.io import auto_detect, get_adapter
from robo_vlm_qc.lab.diagnosis import diagnose_workspace
from robo_vlm_qc.lab.eval_runner import _apply_in_memory_overrides
from robo_vlm_qc.models.annotation import (
    ArmTimelines,
    ConfigFingerprintSnapshot,
    CriterionResult,
    GroundingResult,
    MacroStage,
    QcSegment,
    SuccessVerdict,
    VlmRunArtifact,
)
from robo_vlm_qc.models.review import ReviewWritePayload, WatchCoverage
from robo_vlm_qc.models.taxonomy import FewShotItem
from robo_vlm_qc.store.repository import Repository, compute_review_diff
from robo_vlm_qc.vlm.dag import VlmDag
from tests.api.test_lab_eval_prompt_assist_api import (
    _LocalMockVlm,
    _sanitize_vlm_env,
    _setup_workspace,
)


def _make_video_dataset(root: Path, ds_name: str = "diag_ds", duration_s: float = 4.0) -> Path:
    ds_dir = root / ds_name
    ds_dir.mkdir(parents=True, exist_ok=True)
    mp4 = ds_dir / "ep_000.mp4"
    fps = 10.0
    n_frames = int(round(duration_s * fps))
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    assert writer.isOpened()
    try:
        for i in range(n_frames):
            frame = np.full((120, 160, 3), (i * 5) % 255, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return ds_dir


def _ingest(ws: Path, ds_dir: Path, task_id: str = "pick_and_place") -> str:
    handles = auto_detect(ds_dir)
    assert handles
    with get_adapter(handles[0].adapter_type).open(handles[0]) as view:
        meta = view.meta.model_copy(update={"task_id": task_id})
    repo = Repository(ws)
    try:
        ep_id, _ = repo.ingest_episode(meta)
        return ep_id
    finally:
        repo.close()


def _make_run(
    ep_id: str,
    run_id: str,
    duration_s: float,
    segments: list[QcSegment],
    *,
    passed: bool = True,
    macro_stages: list[MacroStage] | None = None,
) -> VlmRunArtifact:
    stages = macro_stages or [
        MacroStage(stage_id="S1", t_start=0.0, t_end=duration_s * 0.5, label="Grasp", label_en="grasp"),
        MacroStage(stage_id="S2", t_start=duration_s * 0.5, t_end=duration_s, label="Place", label_en="place"),
    ]
    return VlmRunArtifact(
        run_id=run_id,
        episode_id=ep_id,
        created_at=time.time(),
        config_fingerprint=ConfigFingerprintSnapshot(
            model_routing={"s1_grounding": "gpt-4o"},
            prompt_bundle_version="1.0.0",
            label_taxonomy_version="v1",
            task_spec_version="1.0.0",
        ),
        grounding=GroundingResult(),
        macro_stages=stages,
        arm_timelines=ArmTimelines(),
        qc_segments=segments,
        success=SuccessVerdict(
            passed=passed,
            verdict="success" if passed else "failure",
            confidence=0.9,
            criteria=[CriterionResult(criterion="completed", met=passed)],
        ),
    )


def _seg(
    seg_id: str,
    t_start: float,
    t_end: float,
    label_id: str = "missed_grasp_retry",
    role: str = "recovery",
    policy: str = "mask_loss",
    mask_side: str | None = "left",
) -> QcSegment:
    return QcSegment(
        seg_id=seg_id,
        t_start=t_start,
        t_end=t_end,
        label_id=label_id,
        training_role=role,  # type: ignore[arg-type]
        splice_policy=policy,  # type: ignore[arg-type]
        mask_side=mask_side,  # type: ignore[arg-type]
        affected_arm="left" if mask_side is None else mask_side,  # type: ignore[arg-type]
        vlm_raw_policy=policy,  # type: ignore[arg-type]
    )


def test_exact_005s_ieee754_tolerance_in_diagnosis_and_review_diff(tmp_path: Path) -> None:
    """abs(1.0 - 1.05) == 0.050000000000000044 must NOT trigger boundary/stage errors."""
    ws = tmp_path / "ws"
    ds_dir = _make_video_dataset(tmp_path / "data", "ieee_ds", duration_s=4.0)
    ep_id = _ingest(ws, ds_dir)

    repo = Repository(ws)
    try:
        v_segs = [_seg("v_one", 1.0, 2.0, "missed_grasp_retry", "recovery", "mask_loss", "left")]
        g_segs_exact = [
            _seg("g_one", 1.05, 2.0, "missed_grasp_retry", "recovery", "mask_loss", "left")
        ]
        v_stages = [
            MacroStage(stage_id="S1", t_start=0.0, t_end=1.0, label="Stage 1", label_en="s1"),
            MacroStage(stage_id="S2", t_start=1.0, t_end=4.0, label="Stage 2", label_en="s2"),
        ]
        g_stages_exact = [
            MacroStage(stage_id="S1", t_start=0.0, t_end=1.05, label="Stage 1", label_en="s1"),
            MacroStage(stage_id="S2", t_start=1.05, t_end=4.0, label="Stage 2", label_en="s2"),
        ]

        run = _make_run(ep_id, "r_ieee", 4.0, v_segs, passed=True, macro_stages=v_stages)
        repo.save_vlm_run_artifact(run)
        repo.save_review(
            ep_id,
            "lead_u",
            "lead",
            ReviewWritePayload(
                base_run_id="r_ieee",
                action="approve",
                qc_segments=g_segs_exact,
                success=run.success,
                macro_stages=g_stages_exact,
                watch_coverage=WatchCoverage(
                    watched_intervals=[(0.0, 4.0)],
                    watched_s=4.0,
                    duration_s=4.0,
                    coverage_ratio=1.0,
                    confirmed=True,
                ),
            ),
        )

        # Direct compute_review_diff check
        diff_exact = compute_review_diff(run, g_segs_exact, run.success, g_stages_exact)
        assert diff_exact.boundary_nudges == 0
        assert diff_exact.macro_stage_edits == 0
        assert diff_exact.total_edits == 0

        # diagnose_workspace check
        rep_exact = diagnose_workspace(repo)
        assert rep_exact.episodes_evaluated == 1
        assert rep_exact.total_errors == 0
        assert rep_exact.errors == []

        # Contrast: 0.051s shift MUST still trigger boundary & stage errors
        g_segs_over = [
            _seg("g_over", 1.051, 2.0, "missed_grasp_retry", "recovery", "mask_loss", "left")
        ]
        g_stages_over = [
            MacroStage(stage_id="S1", t_start=0.0, t_end=1.051, label="Stage 1", label_en="s1"),
            MacroStage(stage_id="S2", t_start=1.051, t_end=4.0, label="Stage 2", label_en="s2"),
        ]
        diff_over = compute_review_diff(run, g_segs_over, run.success, g_stages_over)
        assert diff_over.boundary_nudges == 1
        assert diff_over.macro_stage_edits == 2  # both S1.t_end and S2.t_start shifted by 0.051

        repo.save_review(
            ep_id,
            "lead_u",
            "lead",
            ReviewWritePayload(
                base_run_id="r_ieee",
                action="approve",
                qc_segments=g_segs_over,
                success=run.success,
                macro_stages=g_stages_over,
                watch_coverage=WatchCoverage(
                    watched_intervals=[(0.0, 4.0)],
                    watched_s=4.0,
                    duration_s=4.0,
                    coverage_ratio=1.0,
                    confirmed=True,
                ),
            ),
        )
        rep_over = diagnose_workspace(repo)
        assert rep_over.counts_by_type.get("boundary") == 1
        assert rep_over.counts_by_type.get("stage") == 1
    finally:
        repo.close()


def test_cross_task_label_few_shot_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task-bound label: Few-Shots must NOT leak across tasks in Repository, VlmDag, or Lab API."""
    _sanitize_vlm_env(monkeypatch)
    mock = _LocalMockVlm()
    try:
        ws, cfg_dir, ep_id, _ = _setup_workspace(tmp_path, mock.base_url)
        repo = Repository(ws)
        try:
            # Task-specific label-scoped few shot (like unknown triage merge creates)
            repo.save_few_shot(
                FewShotItem(
                    example_id="fs_handover_lbl",
                    scope="label:collision_or_knock",
                    task_id="bimanual_handover",
                    label_id="collision_or_knock",
                    error_type="policy_flip",
                    episode_id=ep_id,
                    t_start=0.2,
                    t_end=0.8,
                    rationale="HANDOVER_ONLY_SECRET_CUE",
                    enabled=True,
                    created_at=time.time(),
                )
            )
            # Cross-task label-scoped few shot (task_id=None)
            repo.save_few_shot(
                FewShotItem(
                    example_id="fs_shared_lbl",
                    scope="label:collision_or_knock",
                    task_id=None,
                    label_id="collision_or_knock",
                    error_type="boundary",
                    episode_id=ep_id,
                    t_start=0.2,
                    t_end=0.8,
                    rationale="SHARED_LABEL_CUE",
                    enabled=True,
                    created_at=time.time(),
                )
            )

            pick_items = repo.list_few_shots(task_id="pick_and_place")
            pick_ids = {it.example_id for it in pick_items}
            assert "fs_handover_lbl" not in pick_ids
            assert "fs_shared_lbl" in pick_ids

            handover_items = repo.list_few_shots(task_id="bimanual_handover")
            handover_ids = {it.example_id for it in handover_items}
            assert "fs_handover_lbl" in handover_ids
            assert "fs_shared_lbl" in handover_ids

            # VlmDag.run_episode(few_shots=None) must NOT inject bimanual_handover's label Few-Shot into pick_and_place
            ds_dir = tmp_path / "data" / "lab_demo_ds"
            handle = auto_detect(ds_dir)[0]
            cfg = ConfigManager(config_dir=cfg_dir, workspace_dir=ws)
            dag = VlmDag(workspace=ws, config=cfg, repo=repo)
            try:
                mock.requests.clear()
                with get_adapter(handle.adapter_type).open(handle) as view:
                    meta = view.meta.model_copy(update={"task_id": "pick_and_place"})
                    view.meta = meta
                    dag.run_episode(view, few_shots=None, is_eval_sandbox=True, persist=False)
                s3_texts = [r["text"] for r in mock.requests if r["stage"] == "S3"]
                assert s3_texts
                assert all("HANDOVER_ONLY_SECRET_CUE" not in t for t in s3_texts)
                assert any("SHARED_LABEL_CUE" in t for t in s3_texts)
            finally:
                dag.client.close()
        finally:
            repo.close()

        # GET /api/lab/few-shots?task_id=pick_and_place must also isolate
        app = create_app(ws, config_dir=cfg_dir, reviewer_token="rev-tok", lead_token="lead-tok")
        client = TestClient(app)
        resp = client.get(
            "/api/lab/few-shots",
            params={"task_id": "pick_and_place"},
            headers={"X-Auth-Token": "lead-tok"},
        )
        assert resp.status_code == 200
        returned_ids = {it["example_id"] for it in resp.json()["items"]}
        assert "fs_handover_lbl" not in returned_ids
        assert "fs_shared_lbl" in returned_ids
    finally:
        mock.close()


def test_diagnose_workspace_skips_missing_base_run_artifact_file(tmp_path: Path) -> None:
    """Missing vlm_run JSON file on disk skips cleanly instead of raising FileNotFoundError."""
    ws = tmp_path / "ws"
    ds_dir = _make_video_dataset(tmp_path / "data", "missing_run_ds", duration_s=4.0)
    ep_id = _ingest(ws, ds_dir)

    repo = Repository(ws)
    try:
        run = _make_run(ep_id, "r_missing_file", 4.0, [])
        abs_run_path = repo.save_vlm_run_artifact(run)
        repo.save_review(
            ep_id,
            "lead_u",
            "lead",
            ReviewWritePayload(
                base_run_id="r_missing_file",
                action="freeze_gt",
                qc_segments=[],
                success=run.success,
                macro_stages=run.macro_stages,
                watch_coverage=WatchCoverage(
                    watched_intervals=[(0.0, 4.0)],
                    watched_s=4.0,
                    duration_s=4.0,
                    coverage_ratio=1.0,
                    confirmed=True,
                ),
            ),
        )
        abs_run_path.unlink()
        assert not abs_run_path.exists()

        rep = diagnose_workspace(repo)
        assert rep.episodes_evaluated == 0
        assert rep.total_errors == 0
        assert rep.errors == []
    finally:
        repo.close()


def test_few_shot_scope_rejects_empty_suffix_in_model_and_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty or whitespace-only task:/label: suffix is rejected with ValueError / HTTP 422."""
    for bad_scope in ("task:", "task:   ", "label:", "label:\t"):
        with pytest.raises(ValidationError):
            FewShotItem(
                example_id="fs_bad",
                scope=bad_scope,
                error_type="fp_overflag",
                episode_id="ds/ep_01",
                t_start=0.1,
                t_end=0.5,
                rationale="bad scope",
                created_at=1.0,
            )

    _sanitize_vlm_env(monkeypatch)
    mock = _LocalMockVlm()
    try:
        ws, cfg_dir, ep_id, _ = _setup_workspace(tmp_path, mock.base_url)
        app = create_app(ws, config_dir=cfg_dir, reviewer_token="rev-tok", lead_token="lead-tok")
        client = TestClient(app)
        for bad_scope in ("task:", "task:  ", "label:", "label: "):
            r = client.post(
                "/api/lab/few-shots",
                headers={"X-Auth-Token": "lead-tok"},
                json={
                    "scope": bad_scope,
                    "error_type": "fp_overflag",
                    "episode_id": ep_id,
                    "t_start": 0.2,
                    "t_end": 1.0,
                    "rationale": "bad scope",
                    "capture_tile": False,
                },
            )
            assert r.status_code == 422, (bad_scope, r.text)
    finally:
        mock.close()


def test_lab_api_eval_vlm_error_502_non_dict_candidate_prompts_422_and_partial_prompt_put(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VlmError->502 on /api/lab/eval, non-dict candidate_prompts->422, partial PUT /api/lab/prompts->200."""
    _sanitize_vlm_env(monkeypatch)
    mock = _LocalMockVlm()
    try:
        ws, cfg_dir, _, _ = _setup_workspace(tmp_path, mock.base_url)
        cfg = ConfigManager(config_dir=cfg_dir, workspace_dir=ws)

        # _apply_in_memory_overrides rejects non-dict candidate_prompts values
        with pytest.raises(ValueError, match="expected dict"):
            _apply_in_memory_overrides(cfg, candidate_prompts={"s3_discover": "not_a_dict"})

        app = create_app(ws, config_dir=cfg_dir, reviewer_token="rev-tok", lead_token="lead-tok")
        client = TestClient(app)
        lead_h = {"X-Auth-Token": "lead-tok"}

        # PUT /api/lab/prompts single-stage partial update without templates
        orig_prompts = client.get("/api/lab/prompts", headers=lead_h).json()
        orig_sys = orig_prompts["s3_discover"]["system_template"]
        r_partial = client.put(
            "/api/lab/prompts",
            headers=lead_h,
            json={"stage": "s3_discover", "version": "9.9.9"},
        )
        assert r_partial.status_code == 200, r_partial.text
        assert r_partial.json()["s3_discover"]["version"] == "9.9.9"
        assert r_partial.json()["s3_discover"]["system_template"] == orig_sys

        # Invalid single-stage name still returns 422
        r_bad_stage = client.put(
            "/api/lab/prompts",
            headers=lead_h,
            json={"stage": "no_such_stage", "version": "9.9.9"},
        )
        assert r_bad_stage.status_code == 422

        # POST /api/lab/eval with non-dict candidate_prompts -> 422
        r_bad_prompts = client.post(
            "/api/lab/eval",
            headers=lead_h,
            json={"candidate_prompts": {"s3_discover": "not_a_dict"}},
        )
        assert r_bad_prompts.status_code == 422

        # Invalid candidate_labels -> 422
        r_bad_labels = client.post(
            "/api/lab/eval",
            headers=lead_h,
            json={"candidate_labels": {"labels": [{"id": "incomplete"}]}},
        )
        assert r_bad_labels.status_code == 422

        # VLM HTTP 500 during POST /api/lab/eval -> clean HTTP 502 (not unhandled 500)
        mock.fail_count = 10
        r_vlm_fail = client.post(
            "/api/lab/eval",
            headers=lead_h,
            json={"few_shot": False},
        )
        assert r_vlm_fail.status_code == 502, r_vlm_fail.text
    finally:
        mock.close()
