"""Core E2E feature coverage: VLM Lab pure in-context loop and repository documentation.

Covers:
- lab/diagnosis.py (6 error buckets: fp_overflag, fn_missed, policy_flip, boundary,
  success_flip, stage; Top-N aggregation; clean match).
- lab/few_shot.py (2x2 keyframe crop artifacts/few_shots/<id>.jpg + few_shot_examples DB record,
  scoped global/task/label retrieval within top-K and token budgets).
- lab/eval_runner.py & `robo-vlm-qc eval` CLI (is_eval_sandbox=1 reusing S1/S2/S5 stage_cache
  with 0 upstream S1/S2/S5 requests, quantitative delta metrics, one-click promote config).
- lab/prompt_assist.py via MockVlmServer, api/routers/lab.py config CRUD with .history snapshots,
  and unknown_anomaly triage (merge & promote incrementing label version).
- Open-source documentation checks (README.md, docs/architecture.md, docs/operations.md,
  AGENTS.md, LICENSE, VlmLabPage.tsx) and verification of zero VLM SFT/DPO fine-tuning claims.
"""

from __future__ import annotations

import importlib
import inspect
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import cv2
import yaml

from tests.e2e import helpers as h


# ---------------------------------------------------------------------------
# Lazy Setup Helpers
# ---------------------------------------------------------------------------


def _ingest_dir(ws: Path, cfg_dir: Path, data_dir: Path) -> list[str]:
    res = h.run_cli(["-w", str(ws), "-c", str(cfg_dir), "ingest", "--data", str(data_dir), "--json"])
    assert res.returncode == 0, res.stderr + res.stdout
    return [ep["episode_id"] for ep in json.loads(res.stdout)["episodes"]]


def _make_vlm_run_obj(
    episode_id: str,
    run_id: str,
    duration_s: float,
    qc_segments: list[dict[str, Any]],
    *,
    passed: bool = True,
    macro_stages: list[dict[str, Any]] | None = None,
) -> Any:
    ann_mod = importlib.import_module("robo_vlm_qc.models.annotation")
    mid = round(duration_s * 0.5, 3)
    stages = (
        [ann_mod.MacroStage.model_validate(s) for s in macro_stages]
        if macro_stages is not None
        else [
            ann_mod.MacroStage(
                stage_id="S1", t_start=0.0, t_end=mid, label="Approach and grasp", label_en="grasp"
            ),
            ann_mod.MacroStage(
                stage_id="S2", t_start=mid, t_end=duration_s, label="Transfer and place", label_en="place"
            ),
        ]
    )
    segs = [ann_mod.QcSegment.model_validate(s) for s in qc_segments]
    return ann_mod.VlmRunArtifact(
        run_id=run_id,
        episode_id=episode_id,
        created_at=time.time(),
        config_fingerprint=ann_mod.ConfigFingerprintSnapshot(
            model_routing={"s1_grounding": "gpt-4o", "s3_discover": "gpt-4o"},
            prompt_bundle_version="1.0.0",
            label_taxonomy_version="1",
            task_spec_version="1.0.0",
        ),
        macro_stages=stages,
        qc_segments=segs,
        success=ann_mod.SuccessVerdict(
            passed=passed,
            verdict="success" if passed else "failure",
            confidence=0.9,
            criteria=[ann_mod.CriterionResult(criterion="Target object placed in bin", met=passed)],
        ),
    )


def _freeze_gt(
    repo: Any,
    episode_id: str,
    vlm_run: Any,
    duration_s: float,
    gt_segments: list[dict[str, Any]],
    *,
    gt_passed: bool = True,
    gt_macro_stages: list[dict[str, Any]] | None = None,
) -> Any:
    ann_mod = importlib.import_module("robo_vlm_qc.models.annotation")
    rev_mod = importlib.import_module("robo_vlm_qc.models.review")

    segs = [ann_mod.QcSegment.model_validate(s) for s in gt_segments]
    stages = (
        [ann_mod.MacroStage.model_validate(s) for s in gt_macro_stages]
        if gt_macro_stages is not None
        else vlm_run.macro_stages
    )
    sv = ann_mod.SuccessVerdict(
        passed=gt_passed,
        verdict="success" if gt_passed else "failure",
        confidence=1.0,
        criteria=[ann_mod.CriterionResult(criterion="Target object placed in bin", met=gt_passed)],
    )
    wc = rev_mod.WatchCoverage(
        watched_intervals=[(0.0, duration_s)],
        watched_s=duration_s,
        duration_s=duration_s,
        coverage_ratio=1.0,
        confirmed=True,
    )
    return repo.save_review(
        episode_id,
        author="lead_gt",
        role="lead",
        payload=rev_mod.ReviewWritePayload(
            base_run_id=vlm_run.run_id,
            action="freeze_gt",
            qc_segments=segs,
            success=sv,
            macro_stages=stages,
            watch_coverage=wc,
        ),
    )


def _run_diagnosis(repo: Any, episode_ids: list[str] | None = None, top_n: int = 20) -> Any:
    diag_mod = importlib.import_module("robo_vlm_qc.lab.diagnosis")
    fn = getattr(diag_mod, "diagnose_workspace", None) or getattr(diag_mod, "run_diagnosis", None)
    if fn is not None:
        sig = inspect.signature(fn)
        kwargs: dict[str, Any] = {}
        for pname in sig.parameters:
            if pname in ("repo", "repository"):
                kwargs[pname] = repo
            elif pname == "episode_ids":
                kwargs[pname] = episode_ids
            elif pname == "top_n":
                kwargs[pname] = top_n
        return fn(**kwargs)
    cls = getattr(diag_mod, "DiagnosisEngine")
    eng = cls(repo)
    return eng.diagnose(episode_ids=episode_ids, top_n=top_n)


# ===========================================================================
# Diagnosis (6 Error Buckets + Top-N) & Few-Shot Mistake Book
# ===========================================================================


def test_diagnosis_classifies_all_six_error_buckets(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.store import Repository

    vdir = h.make_synthetic_video_dir(tmp_path / "ds_diag6", multi_cam=False, duration_s=10.0)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, vdir)
    repo = Repository(e2e_workspace)
    try:
        vlm_segs = [
            # 1. fp_overflag: VLM flags false positive at [0.5, 1.3], absent in GT
            {
                "seg_id": "v_fp",
                "t_start": 0.5,
                "t_end": 1.3,
                "label_id": "hesitation_oscillation",
                "training_role": "badcase",
                "splice_policy": "mask_loss",
                "mask_side": "left",
                "vlm_raw_policy": "mask_loss",
            },
            # 2. policy_flip: overlapping [2.0, 3.0], VLM cut_clean vs GT keep_positive
            {
                "seg_id": "v_pflip",
                "t_start": 2.0,
                "t_end": 3.0,
                "label_id": "mid_idle_freeze",
                "training_role": "badcase",
                "splice_policy": "cut_clean",
                "mask_side": None,
                "vlm_raw_policy": "cut_clean",
            },
            # 3. boundary: overlapping [4.0, 5.0] vs GT [4.0, 5.8] (significant boundary drift)
            {
                "seg_id": "v_bnd",
                "t_start": 4.0,
                "t_end": 5.0,
                "label_id": "missed_grasp_retry",
                "training_role": "recovery",
                "splice_policy": "mask_loss",
                "mask_side": "left",
                "vlm_raw_policy": "mask_loss",
            },
        ]
        gt_segs = [
            # policy_flip counterpart
            {
                "seg_id": "g_pflip",
                "t_start": 2.0,
                "t_end": 3.0,
                "label_id": "bimanual_assist_hold",
                "training_role": "necessary",
                "splice_policy": "keep_positive",
                "mask_side": None,
                "vlm_raw_policy": "cut_clean",
            },
            # boundary counterpart
            {
                "seg_id": "g_bnd",
                "t_start": 4.0,
                "t_end": 5.8,
                "label_id": "missed_grasp_retry",
                "training_role": "recovery",
                "splice_policy": "mask_loss",
                "mask_side": "left",
                "vlm_raw_policy": "mask_loss",
            },
            # 4. fn_missed: GT has [7.0, 8.2] missed completely by VLM
            {
                "seg_id": "g_fn",
                "t_start": 7.0,
                "t_end": 8.2,
                "label_id": "collision_or_knock",
                "training_role": "badcase",
                "splice_policy": "mask_loss",
                "mask_side": "right",
                "vlm_raw_policy": "mask_loss",
            },
        ]
        gt_stages = [
            # 5. stage mismatch: altered macro_stage boundaries & label
            {"stage_id": "S1", "t_start": 0.0, "t_end": 3.0, "label": "Human Stage 1", "label_en": "custom_1"},
            {"stage_id": "S2", "t_start": 3.0, "t_end": 10.0, "label": "Human Stage 2", "label_en": "custom_2"},
        ]
        vlm = _make_vlm_run_obj(ep_ids[0], "run_diag6", 10.0, vlm_segs, passed=True)
        repo.save_vlm_run_artifact(vlm)
        # 6. success_flip: VLM passed=True vs GT passed=False
        _freeze_gt(repo, ep_ids[0], vlm, 10.0, gt_segs, gt_passed=False, gt_macro_stages=gt_stages)

        report = _run_diagnosis(repo, episode_ids=ep_ids)
        report_json = report.model_dump(mode="json") if hasattr(report, "model_dump") else dict(report)
        serialized = json.dumps(report_json, ensure_ascii=False)
        for bucket in ("fp_overflag", "fn_missed", "policy_flip", "boundary", "success_flip", "stage"):
            assert bucket in serialized, f"Missing error bucket {bucket!r} in diagnosis report: {serialized}"
    finally:
        repo.close()


def test_diagnosis_top_n_aggregation_across_episodes(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.store import Repository

    vdir = h.make_synthetic_video_dir(tmp_path / "ds_diag_topn", multi_cam=False, duration_s=4.0, n_episodes=3)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, vdir)
    repo = Repository(e2e_workspace)
    try:
        fp_seg = [
            {
                "seg_id": "v_fp",
                "t_start": 0.5,
                "t_end": 1.2,
                "label_id": "mid_idle_freeze",
                "training_role": "badcase",
                "splice_policy": "cut_clean",
                "mask_side": None,
                "vlm_raw_policy": "cut_clean",
            }
        ]
        for idx, ep_id in enumerate(ep_ids):
            vlm = _make_vlm_run_obj(ep_id, f"r_topn_{idx}", 4.0, fp_seg, passed=True)
            repo.save_vlm_run_artifact(vlm)
            _freeze_gt(repo, ep_id, vlm, 4.0, [], gt_passed=True)

        report = _run_diagnosis(repo, top_n=2)
        report_json = report.model_dump(mode="json") if hasattr(report, "model_dump") else dict(report)
        counts = report_json.get("counts_by_type") or report_json.get("summary") or report_json
        assert "fp_overflag" in json.dumps(counts)
        top_items = report_json.get("top_errors") or report_json.get("items") or []
        assert 1 <= len(top_items) <= 2
    finally:
        repo.close()


def test_diagnosis_clean_match_reports_zero_errors(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.store import Repository

    vdir = h.make_synthetic_video_dir(tmp_path / "ds_diag_clean", multi_cam=False, duration_s=4.0)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, vdir)
    repo = Repository(e2e_workspace)
    try:
        segs = [
            {
                "seg_id": "s1",
                "t_start": 1.0,
                "t_end": 1.8,
                "label_id": "missed_grasp_retry",
                "training_role": "recovery",
                "splice_policy": "mask_loss",
                "mask_side": "left",
                "vlm_raw_policy": "mask_loss",
            }
        ]
        vlm = _make_vlm_run_obj(ep_ids[0], "r_clean", 4.0, segs, passed=True)
        repo.save_vlm_run_artifact(vlm)
        _freeze_gt(repo, ep_ids[0], vlm, 4.0, segs, gt_passed=True)

        report = _run_diagnosis(repo, episode_ids=ep_ids)
        report_json = report.model_dump(mode="json") if hasattr(report, "model_dump") else dict(report)
        total_err = report_json.get("total_errors")
        if total_err is not None:
            assert total_err == 0
        else:
            items = report_json.get("items") or report_json.get("errors") or []
            assert items == []
    finally:
        repo.close()


def test_few_shot_creates_2x2_crop_and_db_record(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    fs_mod = importlib.import_module("robo_vlm_qc.lab.few_shot")
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "ds_fs_crop"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=2.0)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, h5_dir)
    repo = Repository(e2e_workspace)
    try:
        create_fn = getattr(fs_mod, "create_few_shot_example", None) or getattr(fs_mod, "add_few_shot_example")
        sig = inspect.signature(create_fn)
        kwargs: dict[str, Any] = {}
        mapping = {
            "repo": repo,
            "repository": repo,
            "workspace": e2e_workspace,
            "episode_id": ep_ids[0],
            "t_start": 0.4,
            "t_end": 1.2,
            "scope": "task:pick_and_place",
            "task_id": "pick_and_place",
            "label_id": "precision_fine_adjust",
            "error_type": "fp_overflag",
            "vlm_wrong": {"label_id": "hesitation_oscillation", "splice_policy": "mask_loss"},
            "vlm_wrong_json": {"label_id": "hesitation_oscillation", "splice_policy": "mask_loss"},
            "human_right": {"label_id": "precision_fine_adjust", "splice_policy": "keep_positive"},
            "human_right_json": {"label_id": "precision_fine_adjust", "splice_policy": "keep_positive"},
            "rationale": "Slow monotonic alignment is a positive skill and must not be flagged as hesitation.",
            "with_image": True,
            "capture_tile": True,
        }
        for pname in sig.parameters:
            if pname in mapping:
                kwargs[pname] = mapping[pname]
        item = create_fn(**kwargs)

        assert item.example_id
        assert item.tile_image_path is not None
        img_path = (
            e2e_workspace / item.tile_image_path
            if not Path(item.tile_image_path).is_absolute()
            else Path(item.tile_image_path)
        )
        assert img_path.is_file() and img_path.parent.name == "few_shots"
        img = cv2.imread(str(img_path))
        assert img is not None and img.shape[0] >= 120 and img.shape[1] >= 160

        conn = sqlite3.connect(str(e2e_workspace / "robo_vlm_qc.db"))
        try:
            row = conn.execute(
                "SELECT scope, error_type, tile_image_path FROM few_shot_examples WHERE example_id = ?",
                (item.example_id,),
            ).fetchone()
            assert row is not None
            assert row[0] == "task:pick_and_place"
            assert row[1] == "fp_overflag"
        finally:
            conn.close()
    finally:
        repo.close()


def test_few_shot_retrieval_scoped_by_global_task_label(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    fs_mod = importlib.import_module("robo_vlm_qc.lab.few_shot")
    tax_mod = importlib.import_module("robo_vlm_qc.models.taxonomy")
    from robo_vlm_qc.store import Repository

    repo = Repository(e2e_workspace)
    try:
        items = [
            tax_mod.FewShotItem(
                example_id="fs_global",
                scope="global",
                error_type="fp_overflag",
                episode_id="ds/ep0",
                t_start=0.0,
                t_end=1.0,
                rationale="Global rule",
                created_at=100.0,
            ),
            tax_mod.FewShotItem(
                example_id="fs_pick",
                scope="task:pick_and_place",
                task_id="pick_and_place",
                error_type="policy_flip",
                episode_id="ds/ep1",
                t_start=0.0,
                t_end=1.0,
                rationale="Pick-and-place task rule",
                created_at=101.0,
            ),
            tax_mod.FewShotItem(
                example_id="fs_handover",
                scope="task:bimanual_handover",
                task_id="bimanual_handover",
                error_type="fn_missed",
                episode_id="ds/ep2",
                t_start=0.0,
                t_end=1.0,
                rationale="Bimanual handover task rule",
                created_at=102.0,
            ),
            tax_mod.FewShotItem(
                example_id="fs_label_retry",
                scope="label:missed_grasp_retry",
                label_id="missed_grasp_retry",
                error_type="boundary",
                episode_id="ds/ep3",
                t_start=0.0,
                t_end=1.0,
                rationale="Missed grasp retry boundary rule",
                created_at=103.0,
            ),
        ]
        with repo.db.transaction() as conn:
            for it in items:
                conn.execute(
                    """
                    INSERT INTO few_shot_examples (
                        example_id, scope, task_id, label_id, error_type, episode_id,
                        t_start, t_end, tile_image_path, vlm_wrong_json, human_right_json,
                        rationale, enabled, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        it.example_id,
                        it.scope,
                        it.task_id,
                        it.label_id,
                        it.error_type,
                        it.episode_id,
                        it.t_start,
                        it.t_end,
                        None,
                        "{}",
                        "{}",
                        it.rationale,
                        1,
                        it.created_at,
                    ),
                )

        retrieve_fn = getattr(fs_mod, "retrieve_few_shots", None) or getattr(fs_mod, "get_relevant_few_shots")
        got = retrieve_fn(repo, task_id="pick_and_place", label_ids=["missed_grasp_retry"], top_k=10)
        got_ids = {g.example_id for g in got}
        assert {"fs_global", "fs_pick", "fs_label_retry"}.issubset(got_ids)
        assert "fs_handover" not in got_ids
    finally:
        repo.close()


def test_few_shot_retrieval_respects_top_k_and_token_budget(
    e2e_workspace: Path
) -> None:
    fs_mod = importlib.import_module("robo_vlm_qc.lab.few_shot")
    from robo_vlm_qc.store import Repository

    repo = Repository(e2e_workspace)
    try:
        with repo.db.transaction() as conn:
            for i in range(8):
                conn.execute(
                    """
                    INSERT INTO few_shot_examples (
                        example_id, scope, task_id, label_id, error_type, episode_id,
                        t_start, t_end, tile_image_path, vlm_wrong_json, human_right_json,
                        rationale, enabled, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"fs_en_{i}",
                        "global",
                        None,
                        None,
                        "fp_overflag",
                        f"ds/ep{i}",
                        0.0,
                        1.0,
                        None,
                        "{}",
                        "{}",
                        "Detailed rationale text " * 25,
                        1,
                        100.0 + i,
                    ),
                )
            conn.execute(
                """
                INSERT INTO few_shot_examples (
                    example_id, scope, task_id, label_id, error_type, episode_id,
                    t_start, t_end, tile_image_path, vlm_wrong_json, human_right_json,
                    rationale, enabled, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "fs_disabled",
                    "global",
                    None,
                    None,
                    "fp_overflag",
                    "ds/epx",
                    0.0,
                    1.0,
                    None,
                    "{}",
                    "{}",
                    "Disabled",
                    0,
                    999.0,
                ),
            )

        retrieve_fn = getattr(fs_mod, "retrieve_few_shots", None) or getattr(fs_mod, "get_relevant_few_shots")
        got_k = retrieve_fn(repo, task_id="pick_and_place", top_k=3, max_tokens=5000)
        assert len(got_k) <= 3
        assert all(g.example_id != "fs_disabled" for g in got_k)

        got_budget = retrieve_fn(repo, task_id="pick_and_place", top_k=10, max_tokens=150)
        assert len(got_budget) < 8
    finally:
        repo.close()


# ===========================================================================
# Eval Runner, `eval` CLI, Prompt Assist, Lab API & Open-Source Docs
# ===========================================================================


def test_eval_runner_sandbox_reuses_s1_s2_s5_cache_zero_upstream_calls(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.models.review import EpisodeStatus
    from robo_vlm_qc.store import Repository

    eval_mod = importlib.import_module("robo_vlm_qc.lab.eval_runner")
    vdir = h.make_synthetic_video_dir(tmp_path / "ds_eval_cache", multi_cam=False, duration_s=3.0)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, vdir)

    # Baseline run via CLI `run-vlm` -> populates S1, S2, S3, S4, S5 StageCache
    r_vlm = h.run_cli(["-w", str(e2e_workspace), "-c", str(e2e_config_dir), "run-vlm", "--episode", ep_ids[0]])
    assert r_vlm.returncode == 0, r_vlm.stderr + r_vlm.stdout

    repo = Repository(e2e_workspace)
    try:
        base_run_id = repo.get_episode_row(ep_ids[0])["active_run_id"]
        assert base_run_id
        base_run = repo.get_vlm_run(base_run_id)
        _freeze_gt(
            repo,
            ep_ids[0],
            base_run,
            3.0,
            [s.model_dump(mode="json") for s in base_run.qc_segments],
            gt_passed=True,
        )

        # Modify S3 prompt in configs_override -> invalidates S3/S4 only, S1/S2/S5 MUST hit cache
        s3_raw = yaml.safe_load((e2e_config_dir / "prompts" / "s3_discover.yaml").read_text("utf-8"))
        s3_raw["version"] = "2.0.0"
        s3_raw["user_template"] = s3_raw["user_template"] + "\n# Sandbox candidate v2"
        ov_p = e2e_workspace / "configs_override" / "prompts"
        ov_p.mkdir(parents=True, exist_ok=True)
        (ov_p / "s3_discover.yaml").write_text(yaml.safe_dump(s3_raw, allow_unicode=True), encoding="utf-8")

        mock_vlm_server.reset()
        cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
        run_eval = getattr(eval_mod, "run_gt_eval", None) or getattr(eval_mod, "run_eval_sandbox")
        sig = inspect.signature(run_eval)
        kwargs: dict[str, Any] = {}
        for pname in sig.parameters:
            if pname in ("workspace", "workspace_dir"):
                kwargs[pname] = e2e_workspace
            elif pname in ("repo", "repository"):
                kwargs[pname] = repo
            elif pname in ("config", "cfg", "config_manager", "cfg_mgr"):
                kwargs[pname] = cfg
            elif pname in ("config_dir", "configs_dir"):
                kwargs[pname] = e2e_config_dir
            elif pname == "episode_ids":
                kwargs[pname] = ep_ids
        eval_res = run_eval(**kwargs)

        # Verify 0 upstream VLM HTTP calls for S1, S2, S5, S5_CONTROL
        stages_called = [req["stage"] for req in mock_vlm_server.requests]
        assert not any(st in ("S1", "S2", "S5", "S5_CONTROL") for st in stages_called), (
            f"Expected 100% cache hit on S1/S2/S5, but saw upstream calls: {stages_called}"
        )
        assert any(st in ("S3", "S4A", "S4B") for st in stages_called)

        # Verify sandbox run saved with is_eval_sandbox=1 and gt_frozen untouched
        row = repo.get_episode_row(ep_ids[0])
        assert row["status"] == EpisodeStatus.GT_FROZEN.value
        assert row["active_run_id"] == base_run_id
        sandbox_rows = repo.db.connect().execute(
            "SELECT run_id FROM vlm_runs WHERE episode_id = ? AND is_eval_sandbox = 1",
            (ep_ids[0],),
        ).fetchall()
        assert len(sandbox_rows) >= 1

        # Quantitative delta table metrics
        eval_dict = eval_res.model_dump(mode="json") if hasattr(eval_res, "model_dump") else dict(eval_res)
        eval_json = json.dumps(eval_dict, ensure_ascii=False).lower()
        for metric_kw in ("f1", "iou", "overcut", "downgrade", "success", "token"):
            assert metric_kw in eval_json, f"Missing metric {metric_kw!r} in eval output: {eval_json}"
    finally:
        repo.close()


def test_cli_eval_runs_sandbox_and_outputs_metrics(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    from robo_vlm_qc.models.review import EpisodeStatus
    from robo_vlm_qc.store import Repository

    vdir = h.make_synthetic_video_dir(tmp_path / "ds_cli_eval", multi_cam=False, duration_s=2.0)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, vdir)
    assert h.run_cli(["-w", str(e2e_workspace), "-c", str(e2e_config_dir), "run-vlm", "--episode", ep_ids[0]]).returncode == 0

    repo = Repository(e2e_workspace)
    try:
        base_run = repo.get_vlm_run(repo.get_episode_row(ep_ids[0])["active_run_id"])
        _freeze_gt(repo, ep_ids[0], base_run, 2.0, [])
    finally:
        repo.close()

    res = h.run_cli(["-w", str(e2e_workspace), "-c", str(e2e_config_dir), "eval"])
    assert res.returncode == 0, res.stderr + res.stdout
    assert any(k in res.stdout.lower() for k in ("f1", "iou", "precision", "recall", "overcut"))

    repo2 = Repository(e2e_workspace)
    try:
        assert repo2.get_episode_row(ep_ids[0])["status"] == EpisodeStatus.GT_FROZEN.value
    finally:
        repo2.close()


def test_eval_runner_one_click_promote_config(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    eval_mod = importlib.import_module("robo_vlm_qc.lab.eval_runner")
    promote_fn = getattr(eval_mod, "promote_candidate_config", None) or getattr(eval_mod, "promote_config")

    # Seed baseline override first so promoting creates a .history snapshot
    ov_prompts = e2e_workspace / "configs_override" / "prompts"
    ov_prompts.mkdir(parents=True, exist_ok=True)
    orig_s3 = (e2e_config_dir / "prompts" / "s3_discover.yaml").read_text("utf-8")
    (ov_prompts / "s3_discover.yaml").write_text(orig_s3, encoding="utf-8")

    cand_s3 = yaml.safe_load(orig_s3)
    cand_s3["version"] = "1.5.0"
    cand_s3["user_template"] = cand_s3["user_template"] + "\n# Promoted candidate"

    sig = inspect.signature(promote_fn)
    kwargs: dict[str, Any] = {}
    for pname in sig.parameters:
        if pname in ("workspace", "workspace_dir"):
            kwargs[pname] = e2e_workspace
        elif pname in ("candidate_prompts", "prompts_override", "prompts"):
            kwargs[pname] = {"s3_discover": cand_s3}
        elif pname in ("candidate", "candidate_config"):
            kwargs[pname] = {"prompts": {"s3_discover": cand_s3}}
        elif pname == "reason":
            kwargs[pname] = "eval_promote_v1.5"
    promote_fn(**kwargs)

    updated = yaml.safe_load((ov_prompts / "s3_discover.yaml").read_text("utf-8"))
    assert updated["version"] == "1.5.0"
    assert list((e2e_workspace / ".history").rglob("*.json"))


def test_prompt_assist_generates_draft_via_mock_vlm(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any
) -> None:
    from robo_vlm_qc.config import ConfigManager

    pa_mod = importlib.import_module("robo_vlm_qc.lab.prompt_assist")
    mock_vlm_server.set_stage_response(
        "UNKNOWN",
        {
            "suggested_prompt_diff": "Distinguish slow monotonic alignment from aimless oscillation.",
            "rationale": "Add explicit exemption clause for slow precision alignment to reduce fp_overflag.",
        },
    )
    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    gen_fn = (
        getattr(pa_mod, "suggest_prompt_updates", None)
        or getattr(pa_mod, "generate_prompt_suggestion", None)
        or getattr(pa_mod, "suggest_draft")
    )
    sig = inspect.signature(gen_fn)
    kwargs: dict[str, Any] = {}
    sample_errors = [
        {
            "error_type": "fp_overflag",
            "label_id": "hesitation_oscillation",
            "count": 5,
            "note": "Slow fine adjustment misflagged as oscillation",
        }
    ]
    for pname in sig.parameters:
        if pname in ("workspace", "workspace_dir"):
            kwargs[pname] = e2e_workspace
        elif pname in ("config", "cfg", "config_manager", "cfg_mgr"):
            kwargs[pname] = cfg
        elif pname in ("diagnosis", "diagnosis_report", "errors", "top_errors"):
            kwargs[pname] = sample_errors
        elif pname == "stage":
            kwargs[pname] = "s3_discover"
    suggestion = gen_fn(**kwargs)
    assert suggestion
    assert len(mock_vlm_server.requests) >= 1
    assert any(r["stage"] == "UNKNOWN" for r in mock_vlm_server.requests)


def test_api_lab_config_crud_with_history_snapshot(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    lead_hdrs = {"X-Auth-Token": "lead-token"}
    rev_hdrs = {"X-Auth-Token": "rev-token"}

    assert client.get("/api/lab/labels", headers=rev_hdrs).status_code == 403

    r_get = client.get("/api/lab/labels", headers=lead_hdrs)
    assert r_get.status_code == 200
    labels_doc = r_get.json()
    assert len(labels_doc["labels"]) == 11

    # First update writes configs_override/labels.yaml; second update triggers .history snapshot
    labels_doc_v2 = dict(labels_doc, version=int(labels_doc.get("version", 1)) + 1)
    r_put1 = client.put("/api/lab/labels", json=labels_doc_v2, headers=lead_hdrs)
    assert r_put1.status_code == 200
    assert (e2e_workspace / "configs_override" / "labels.yaml").is_file()

    labels_doc_v3 = dict(labels_doc_v2, version=int(labels_doc_v2["version"]) + 1)
    r_put2 = client.put("/api/lab/labels", json=labels_doc_v3, headers=lead_hdrs)
    assert r_put2.status_code == 200
    assert len(list((e2e_workspace / ".history").rglob("*.json"))) >= 1

    # Prompt & TaskSpec endpoints readable/updatable
    assert client.get("/api/lab/prompts", headers=lead_hdrs).status_code == 200
    assert client.get("/api/lab/tasks", headers=lead_hdrs).status_code == 200


def test_api_lab_unknown_anomaly_triage_merge_and_promote(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.store import Repository

    vdir = h.make_synthetic_video_dir(tmp_path / "ds_unk", multi_cam=False, duration_s=3.0)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, vdir)
    repo = Repository(e2e_workspace)
    try:
        unk_segs = [
            {
                "seg_id": "s_unk_1",
                "t_start": 0.5,
                "t_end": 1.1,
                "label_id": "unknown_anomaly",
                "training_role": "badcase",
                "splice_policy": "mask_loss",
                "mask_side": "left",
                "vlm_raw_policy": "mask_loss",
                "is_unknown": True,
                "unknown_note": "Minor tool binding and jitter",
            },
            {
                "seg_id": "s_unk_2",
                "t_start": 1.8,
                "t_end": 2.5,
                "label_id": "unknown_anomaly",
                "training_role": "badcase",
                "splice_policy": "mask_loss",
                "mask_side": "right",
                "vlm_raw_policy": "mask_loss",
                "is_unknown": True,
                "unknown_note": "Gripper cable snagged on table fixture",
            },
        ]
        vlm = _make_vlm_run_obj(ep_ids[0], "r_unk", 3.0, unk_segs)
        repo.save_vlm_run_artifact(vlm)
    finally:
        repo.close()

    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    lead_hdrs = {"X-Auth-Token": "lead-token"}

    r_list = client.get("/api/lab/unknowns", headers=lead_hdrs)
    assert r_list.status_code == 200
    unk_items = (
        r_list.json()["items"] if isinstance(r_list.json(), dict) and "items" in r_list.json() else r_list.json()
    )
    assert len(unk_items) >= 2

    # 1. Merge into existing label -> creates Few-Shot example & updates label
    r_merge = client.post(
        "/api/lab/unknowns/triage",
        json={
            "action": "merge",
            "episode_id": ep_ids[0],
            "seg_id": "s_unk_1",
            "target_label_id": "hesitation_oscillation",
            "rationale": "Merge into hesitation_oscillation",
        },
        headers=lead_hdrs,
    )
    assert r_merge.status_code == 200, r_merge.text

    # 2. Promote to new label -> increments label_version & updates configs_override/labels.yaml
    r_promote = client.post(
        "/api/lab/unknowns/triage",
        json={
            "action": "promote",
            "episode_id": ep_ids[0],
            "seg_id": "s_unk_2",
            "new_label": {
                "id": "cable_snag_obstruction",
                "name": "Cable Snag Obstruction",
                "category": "failure",
                "default_training_role": "badcase",
                "default_splice_policy": "mask_loss",
                "collector_ui": {
                    "title": "Cable Snag",
                    "hint": "Arm or wrist cable catches on a tabletop fixture",
                },
                "vlm_semantics": {
                    "definition": "Cable visibly catches on a container or fixture and obstructs arm motion",
                },
            },
        },
        headers=lead_hdrs,
    )
    assert r_promote.status_code == 200, r_promote.text

    labels_after = client.get("/api/lab/labels", headers=lead_hdrs).json()
    assert int(labels_after["version"]) > 1
    assert any(lbl["id"] == "cable_snag_obstruction" for lbl in labels_after["labels"])


def test_repository_docs_and_vlm_lab_page_present() -> None:
    readme = h.REPO_ROOT / "README.md"
    agents = h.REPO_ROOT / "AGENTS.md"
    arch_doc = h.REPO_ROOT / "docs" / "architecture.md"
    ops_doc = h.REPO_ROOT / "docs" / "operations.md"
    license_file = h.REPO_ROOT / "LICENSE"
    lab_page = h.REPO_ROOT / "web" / "src" / "pages" / "VlmLabPage.tsx"

    for p in (readme, agents, arch_doc, ops_doc, license_file, lab_page):
        assert p.is_file(), f"Missing deliverable file: {p}"
        assert len(p.read_text("utf-8").strip()) > 100, f"File too short/empty: {p}"

    license_text = license_file.read_text("utf-8")
    assert "Apache License" in license_text and "Version 2.0" in license_text

    # Verify zero affirmative VLM SFT/DPO fine-tuning claims across OSS docs
    negation_markers = (
        "no ",
        "zero ",
        "never ",
        "not ",
        "without ",
        "non-goal",
        "forbidden",
        "prohibit",
        "avoid",
        "pure in-context",
        "strictly in-context",
        "no-",
    )
    ft_re = re.compile(r"\b(?:sft|dpo|rlhf)\b", re.IGNORECASE)
    for doc_path in (readme, agents, arch_doc, ops_doc):
        for lineno, line in enumerate(doc_path.read_text("utf-8").splitlines(), start=1):
            if ft_re.search(line):
                low = line.lower()
                assert any(neg in low for neg in negation_markers), (
                    f"Unexpected affirmative VLM fine-tuning claim in {doc_path.name}:{lineno}: {line}"
                )
