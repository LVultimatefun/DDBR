"""Core E2E feature coverage: foundation, config, schemas, store, and 3-in I/O adapters.

Covers:
- Repo scaffold, YAML config loading, ConfigManager layered overrides & SHA256 fingerprints,
  task priority resolution, and dynamic duration token budgets.
- Core domain models (EpisodeMeta/CameraSpec/KinematicsSpec, VlmRunArtifact, QcSegment,
  MacroStage/ArmTimelines/Grounding/Success, zero abolished legacy fields).
- Review, taxonomy, and export domain models (HumanReviewArtifact, ReviewDiff, ResolvedEpisode,
  EpisodeStatus, VisualLabelDef, LabelRegistry, TaskSpec, PromptBundle, FewShotItem,
  ExportRecipe, KeepPiece, ExportManifest).
- InputAdapter registry (3-in), MediaSource frame/batch reading, adaptive 2x2/1x2 tile composer,
  and PreviewCache.
- 3-in input adapters: h5_raw_v3 (synthetic alignment/Slerp/tactile/SBS stereo/read-only),
  lerobot (v2 & v3), and video_dir (single-cam & multi-cam, kinematics=None).
- SQLite WAL & core tables, claim leases, atomic JSON + .history snapshots, Repository CRUD,
  state transitions, resolve_episode, gt_frozen immutability, RBAC & watch coverage guards.
- `robo-vlm-qc ingest` CLI across all 3 formats + idempotent re-ingest preserving reviews.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from tests.e2e import helpers as h

# ===========================================================================
# Repo Scaffold, Configs, ConfigManager & SHA256 Fingerprints
# ===========================================================================


def test_repo_scaffold_and_base_configs_load(e2e_config_dir: Path) -> None:
    from robo_vlm_qc.config import ConfigManager

    for rel in (
        "pyproject.toml",
        "configs/default.yaml",
        "configs/labels.yaml",
        "configs/prompts/s1_grounding.yaml",
        "configs/prompts/s2_macro.yaml",
        "configs/prompts/s3_discover.yaml",
        "configs/prompts/s4_verify.yaml",
        "configs/prompts/s5_success.yaml",
        "configs/tasks/_default.yaml",
        "configs/tasks/pick_and_place.yaml",
        "configs/tasks/bimanual_handover.yaml",
        "configs/presets/openai.yaml",
        "configs/presets/dashscope.yaml",
        "configs/presets/gemini.yaml",
        "configs/presets/vllm.yaml",
    ):
        assert (h.REPO_ROOT / rel).is_file(), f"Missing scaffold file: {rel}"

    cfg = ConfigManager(config_dir=e2e_config_dir)
    vlm_cfg = cfg.settings["vlm"]
    assert vlm_cfg["provider"] in ("openai_compat", "gemini")
    assert vlm_cfg["base_url"] == "https://api.openai.com/v1"
    assert vlm_cfg["api_key_env"] == "OPENAI_API_KEY"
    assert vlm_cfg["key_file"] == ".env"
    assert vlm_cfg["thinking_mode"] == "auto"
    assert vlm_cfg["stage_models"]["s3_discover"] == "gpt-4o"
    assert vlm_cfg["s1_grounding"]["thinking"] is False
    assert cfg.settings["export"]["robot_type"] == "bimanual_robot"

    assert len(cfg.labels.labels) == 11
    assert set(cfg.prompts.model_dump().keys()) == {
        "s1_grounding",
        "s2_macro",
        "s3_discover",
        "s4_verify",
        "s5_success",
    }
    assert {"_default", "pick_and_place", "bimanual_handover"}.issubset(
        set(cfg.tasks.keys())
    )


def test_config_manager_layered_overrides(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    from robo_vlm_qc.config import ConfigManager

    h.write_workspace_config(
        e2e_workspace,
        stage_models={"s1_grounding": "gpt-4o-mini", "s2_macro": "gpt-4o-mini"},
        budget={"tokens_per_second": 7500},
    )
    ov_task_dir = e2e_workspace / "configs_override" / "tasks"
    ov_task_dir.mkdir(parents=True, exist_ok=True)
    (ov_task_dir / "pick_and_place.yaml").write_text(
        yaml.safe_dump({"goal": "Custom tabletop pick and place goal"}, allow_unicode=True),
        encoding="utf-8",
    )

    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    assert cfg.stage_model("s1_grounding") == "gpt-4o-mini"
    assert cfg.stage_model("s2_macro") == "gpt-4o-mini"
    assert cfg.stage_model("s3_discover") == "gpt-4o"
    assert cfg.settings["vlm"]["budget"]["tokens_per_second"] == 7500
    task = cfg.get_task("pick_and_place")
    assert task.goal == "Custom tabletop pick and place goal"
    assert len(task.criteria) >= 1  # inherited from base via deep_merge


def test_stage_isolated_sha256_fingerprints(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    from robo_vlm_qc.config import ConfigManager

    base_cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    fp0 = base_cfg.fingerprints
    assert len(fp0.config_sha256) == 64
    assert len(fp0.model_routing_sha256) == 64
    assert len(fp0.prompt_bundle_sha256) == 64
    assert len(fp0.label_taxonomy_sha256) == 64
    assert len(fp0.task_spec_sha256) == 64

    s1_b0 = base_cfg.s1_cache_basis("pick_and_place")
    s2_b0 = base_cfg.s2_cache_basis("pick_and_place")
    s5_b0 = base_cfg.s5_cache_basis("pick_and_place")
    s3_b0 = base_cfg.s3_cache_basis("pick_and_place")
    s4_b0 = base_cfg.s4_cache_basis("pick_and_place")

    # Modify S3 prompt and labels override -> S1/S2/S5 bases MUST stay identical
    ov_prompts = e2e_workspace / "configs_override" / "prompts"
    ov_prompts.mkdir(parents=True, exist_ok=True)
    s3_raw = yaml.safe_load((e2e_config_dir / "prompts" / "s3_discover.yaml").read_text("utf-8"))
    s3_raw["version"] = "1.1.0"
    s3_raw["user_template"] = s3_raw["user_template"] + "\n# extra check"
    (ov_prompts / "s3_discover.yaml").write_text(
        yaml.safe_dump(s3_raw, allow_unicode=True), encoding="utf-8"
    )
    labels_raw = yaml.safe_load((e2e_config_dir / "labels.yaml").read_text("utf-8"))
    labels_raw["version"] = 2
    (e2e_workspace / "configs_override" / "labels.yaml").write_text(
        yaml.safe_dump(labels_raw, allow_unicode=True), encoding="utf-8"
    )

    mod_cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    assert mod_cfg.s1_cache_basis("pick_and_place") == s1_b0
    assert mod_cfg.s2_cache_basis("pick_and_place") == s2_b0
    assert mod_cfg.s5_cache_basis("pick_and_place") == s5_b0
    assert mod_cfg.s3_cache_basis("pick_and_place") != s3_b0
    assert mod_cfg.s4_cache_basis("pick_and_place") != s4_b0
    assert mod_cfg.fingerprints.label_taxonomy_sha256 != fp0.label_taxonomy_sha256


def test_resolve_task_dataset_priority_over_h5_hint(e2e_config_dir: Path) -> None:
    from robo_vlm_qc.config import ConfigManager

    cfg = ConfigManager(config_dir=e2e_config_dir)
    # Dataset glob takes precedence over misleading h5 task_name hint
    t_pick = cfg.resolve_task("dataset_pick_and_place_batch", h5_task_name="handover")
    assert t_pick.task_id == "pick_and_place"

    # Alias / hint resolution when dataset name is generic
    t_handover = cfg.resolve_task("unmatched_batch_01", h5_task_name="two_arm_handover")
    assert t_handover.task_id == "bimanual_handover"

    # Fallback to _default when neither dataset nor hint matches
    t_def = cfg.resolve_task("unmatched_batch_02", h5_task_name="unknown_xyz")
    assert t_def.task_id == "_default"


def test_dynamic_episode_token_budget(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    from robo_vlm_qc.config import ConfigManager

    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    # Short episode clamped up to min_episode_tokens (200000)
    assert cfg.resolve_episode_budget_tokens(4.0) == 200000
    # Long 138.29s episode scales dynamically at 5000 tokens/s
    assert cfg.resolve_episode_budget_tokens(138.29) == int(math.ceil(138.29 * 5000))

    # Hard cap override
    h.write_workspace_config(
        e2e_workspace,
        budget={"tokens_per_second": 5000, "min_episode_tokens": 200000, "max_tokens_per_episode": 350000},
    )
    cfg_capped = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    assert cfg_capped.resolve_episode_budget_tokens(138.29) == 350000

    # Unlimited override
    h.write_workspace_config(e2e_workspace, budget={"unlimited": True})
    cfg_unlim = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    assert cfg_unlim.resolve_episode_budget_tokens(138.29) is None


# ===========================================================================
# Core Episode & Annotation Pydantic v2 Models
# ===========================================================================


def _make_sample_meta(*, has_kinematics: bool = True) -> Any:
    from robo_vlm_qc.models.episode import CameraSpec, EpisodeMeta, KinematicsSpec

    cams = [
        CameraSpec(name="head_rgbd", role="main", width=1280, height=720, fps=30.0),
        CameraSpec(name="torso_rgbd", role="front", width=1280, height=720, fps=30.0),
    ]
    kin = KinematicsSpec(
        has_kinematics=has_kinematics,
        n_arms=2 if has_kinematics else 0,
        arm_dof=7 if has_kinematics else 0,
        hand_dof=44 if has_kinematics else 0,
        torso_dof=4 if has_kinematics else 0,
        has_chassis=has_kinematics,
        has_tactile=has_kinematics,
        tactile_dim=30 if has_kinematics else 0,
        state_dim=81 if has_kinematics else 0,
    )
    return EpisodeMeta(
        episode_id="pick_ds/ep_0001",
        dataset="pick_ds",
        stem="ep_0001",
        task_id="pick_and_place",
        adapter_type="h5_raw_v3" if has_kinematics else "video_dir",
        source_uri="/tmp/pick_ds/ep_0001.h5",
        source_mtime=1700000000.0,
        episode_fingerprint="abc123",
        duration_s=10.0,
        fps=30.0,
        frame_count=300,
        cameras=cams,
        camera_roles={"main": "head_rgbd", "front": "torso_rgbd"},
        kinematics=kin,
        recorded_success=True,
    )


def _make_sample_vlm_run(episode_id: str = "pick_ds/ep_0001", run_id: str = "run_001") -> Any:
    from robo_vlm_qc.models.annotation import (
        ArmSpan,
        ArmTimelines,
        ConfigFingerprintSnapshot,
        ControlCheckResult,
        CriterionResult,
        GroundingKeyframe,
        GroundingResult,
        GroundingTrack,
        GuardrailMetrics,
        GuardrailReport,
        MacroStage,
        QcSegment,
        StageRefs,
        StageUsage,
        SuccessVerdict,
        UsageSummary,
        VisualEvidence,
        VlmRunArtifact,
    )

    return VlmRunArtifact(
        run_id=run_id,
        episode_id=episode_id,
        created_at=1700000100.0,
        config_fingerprint=ConfigFingerprintSnapshot(
            model_routing={"s1_grounding": "gpt-4o", "s3_discover": "gpt-4o"},
            prompt_bundle_version="1.0.0",
            label_taxonomy_version="1",
            task_spec_version="1.0.0",
        ),
        stage_refs=StageRefs(s1_grounding="aa" * 32, s2_macro="bb" * 32),
        grounding=GroundingResult(
            keyframes_s=[0.0, 5.0],
            object_tracks_2d={
                "target_object": GroundingTrack(
                    object_id="target_object",
                    name="Target Object",
                    is_static_container=False,
                    keyframes=[
                        GroundingKeyframe(
                            row=0, time_s=0.0, visible=True, bbox_norm_1000=[100, 200, 300, 400]
                        )
                    ],
                )
            },
        ),
        macro_stages=[
            MacroStage(
                stage_id="S1",
                t_start=0.0,
                t_end=5.0,
                label="Pick target object",
                label_en="pick_target_object",
            ),
            MacroStage(
                stage_id="S2",
                t_start=5.0,
                t_end=10.0,
                label="Place in target bin",
                label_en="place_in_target_bin",
            ),
        ],
        arm_timelines=ArmTimelines(
            left=[ArmSpan(t_start=0.0, t_end=10.0, state="work", desc="left work")],
            right=[ArmSpan(t_start=0.0, t_end=10.0, state="hold", desc="right hold")],
        ),
        qc_segments=[
            QcSegment(
                seg_id="seg_1",
                t_start=0.0,
                t_end=0.8,
                label_id="boundary_idle",
                name="Boundary Idle",
                training_role="badcase",
                splice_policy="cut_clean",
                mask_side=None,
                affected_arm="both",
                visual_evidence=VisualEvidence(
                    initial_state="Initially stationary.",
                    process_change="Zero displacement.",
                    end_state="Remains stationary.",
                ),
                vlm_raw_policy="cut_clean",
                guardrail=GuardrailReport(gate_type="splice_gate", passed=True),
                confidence=0.95,
            ),
            QcSegment(
                seg_id="seg_2",
                t_start=2.0,
                t_end=3.2,
                label_id="missed_grasp_retry",
                name="Missed Grasp Retry",
                training_role="recovery",
                splice_policy="mask_loss",
                mask_side="left",
                affected_arm="left",
                visual_evidence=VisualEvidence(
                    initial_state="Reaching for the target object.",
                    process_change="Closes on empty space and retries.",
                    end_state="Successfully lifts the object.",
                ),
                vlm_raw_policy="mask_loss",
                guardrail=GuardrailReport(
                    gate_type="splice_gate",
                    passed=True,
                    metrics=GuardrailMetrics(eef_pos_m=0.05),
                ),
                confidence=0.9,
            ),
        ],
        success=SuccessVerdict(
            passed=True,
            verdict="success",
            confidence=0.94,
            criteria=[
                CriterionResult(
                    criterion="Target object placed inside bin",
                    met=True,
                    satisfied_at_s=9.2,
                )
            ],
            control_check=ControlCheckResult(
                cut_s=3.0, cut_ratio=0.30, control_verdict="failure", passed=True, met=[False]
            ),
            summary="Task succeeded.",
        ),
        usage=UsageSummary(
            stages=[StageUsage(stage="s1_grounding", model="gpt-4o", total_tokens=150)],
            total_tokens=150,
            total_elapsed_s=0.4,
        ),
    )


def test_episode_meta_camera_and_kinematics_spec() -> None:
    from robo_vlm_qc.models.episode import EpisodeMeta

    meta_kin = _make_sample_meta(has_kinematics=True)
    raw = meta_kin.model_dump_json()
    restored = EpisodeMeta.model_validate_json(raw)
    assert restored == meta_kin
    assert restored.kinematics.has_kinematics is True

    meta_vid = _make_sample_meta(has_kinematics=False)
    assert meta_vid.kinematics.has_kinematics is False
    assert meta_vid.kinematics.n_arms == 0

    with pytest.raises(ValidationError):
        EpisodeMeta.model_validate({**meta_kin.model_dump(), "episode_id": "wrong/id"})


def test_vlm_run_artifact_roundtrip() -> None:
    from robo_vlm_qc.models.annotation import VlmRunArtifact

    art = _make_sample_vlm_run()
    dumped = art.model_dump_json()
    restored = VlmRunArtifact.model_validate_json(dumped)
    assert restored == art
    assert restored.grounding.object_tracks_2d["target_object"].keyframes[0].bbox_norm_1000 == [
        100,
        200,
        300,
        400,
    ]
    assert len(restored.qc_segments) == 2


def test_qc_segment_policy_role_mask_side_contract() -> None:
    from robo_vlm_qc.models.annotation import QcSegment

    # Valid keep_positive
    seg_kp = QcSegment(
        seg_id="s_kp",
        t_start=1.0,
        t_end=2.0,
        label_id="precision_fine_adjust",
        training_role="necessary",
        splice_policy="keep_positive",
        mask_side=None,
        vlm_raw_policy="keep_positive",
    )
    assert seg_kp.mask_side is None

    # Invalid: keep_positive with non-None mask_side
    with pytest.raises(ValidationError):
        QcSegment(
            seg_id="bad_kp",
            t_start=1.0,
            t_end=2.0,
            label_id="precision_fine_adjust",
            training_role="necessary",
            splice_policy="keep_positive",
            mask_side="left",
            vlm_raw_policy="keep_positive",
        )

    # Invalid: mask_loss with mask_side=None
    with pytest.raises(ValidationError):
        QcSegment(
            seg_id="bad_ml",
            t_start=1.0,
            t_end=2.0,
            label_id="missed_grasp_retry",
            training_role="recovery",
            splice_policy="mask_loss",
            mask_side=None,
            vlm_raw_policy="mask_loss",
        )

    # Invalid: cut_clean with training_role="necessary"
    with pytest.raises(ValidationError):
        QcSegment(
            seg_id="bad_cc",
            t_start=1.0,
            t_end=2.0,
            label_id="boundary_idle",
            training_role="necessary",
            splice_policy="cut_clean",
            mask_side=None,
            vlm_raw_policy="cut_clean",
        )


def test_macro_stage_arm_timelines_grounding_success_models() -> None:
    from robo_vlm_qc.models.annotation import (
        ArmSpan,
        GroundingKeyframe,
        MacroStage,
        SuccessVerdict,
    )

    ms = MacroStage(stage_id="S1", t_start=0.0, t_end=3.0, label="Grasp object", color_rgb=[10, 20, 30])
    assert ms.color_rgb == [10, 20, 30]
    with pytest.raises(ValidationError):
        MacroStage(stage_id="S1", t_start=3.0, t_end=2.0, label="Inverted span")
    with pytest.raises(ValidationError):
        ArmSpan(t_start=1.0, t_end=2.0, state="invalid_state")  # type: ignore[arg-type]

    # Invisible grounding keyframe must have [0,0,0,0]
    gk_inv = GroundingKeyframe(row=0, time_s=0.0, visible=False, bbox_norm_1000=[0, 0, 0, 0])
    assert not gk_inv.visible
    with pytest.raises(ValidationError):
        GroundingKeyframe(row=0, time_s=0.0, visible=False, bbox_norm_1000=[10, 10, 50, 50])

    sv = SuccessVerdict(passed=True, verdict="success", confidence=0.88)
    assert sv.verdict == "success"


def test_no_legacy_long_pause_or_back_and_forth_fields() -> None:
    from robo_vlm_qc.models.annotation import QcSegment, VlmRunArtifact

    art = _make_sample_vlm_run()
    h.assert_no_legacy_fields(art)
    assert "long_pause" not in VlmRunArtifact.model_fields
    assert "back_and_forth" not in VlmRunArtifact.model_fields

    for bad_label in ("long_pause", "back_and_forth"):
        with pytest.raises(ValidationError):
            QcSegment(
                seg_id="bad_leg",
                t_start=0.5,
                t_end=1.5,
                label_id=bad_label,
                training_role="badcase",
                splice_policy="mask_loss",
                mask_side="both",
                vlm_raw_policy="mask_loss",
            )
        with pytest.raises(ValidationError):
            QcSegment.model_validate(
                {
                    "seg_id": "bad_key",
                    "t_start": 0.5,
                    "t_end": 1.5,
                    "label_id": "boundary_idle",
                    "training_role": "badcase",
                    "splice_policy": "cut_clean",
                    "vlm_raw_policy": "cut_clean",
                    bad_label: True,
                }
            )


def test_visual_evidence_and_guardrail_report() -> None:
    from robo_vlm_qc.models.annotation import GuardrailMetrics, GuardrailReport, VisualEvidence

    ev = VisualEvidence(
        initial_state="Left arm hovers above object.",
        process_change="Left arm descends and misses grasp.",
        end_state="Left arm retracts upward.",
    )
    assert (
        ev.as_text()
        == "Left arm hovers above object. Left arm descends and misses grasp. Left arm retracts upward."
    )
    rep = GuardrailReport(
        gate_type="splice_gate",
        passed=False,
        downgraded=True,
        downgrade_reason="Left arm pose jump 0.05m > 0.02m",
        metrics=GuardrailMetrics(eef_pos_m=0.05, other_arm_disp_m=0.04, bimanual_coupled=True),
        opt_t_start=1.1,
        opt_t_end=2.4,
    )
    assert rep.downgraded and rep.metrics.bimanual_coupled


# ===========================================================================
# Review, Taxonomy & Export Domain Models
# ===========================================================================


def test_episode_status_and_human_review_artifact_roundtrip() -> None:
    from robo_vlm_qc.models.annotation import MacroStage, SuccessVerdict
    from robo_vlm_qc.models.review import (
        EpisodeStatus,
        HumanReviewArtifact,
        ReviewDiff,
        WatchCoverage,
    )

    assert {s.value for s in EpisodeStatus} == {
        "ingested",
        "vlm_queued",
        "vlm_running",
        "vlm_ready",
        "claimed",
        "reviewed",
        "flagged",
        "approved",
        "gt_frozen",
    }

    rev = HumanReviewArtifact(
        rev_id="rev_001",
        episode_id="pick_ds/ep_0001",
        base_run_id="run_001",
        author="alice",
        role="reviewer",
        action="submit_review",
        created_at=1700000200.0,
        watch_coverage=WatchCoverage(
            watched_intervals=[(0.0, 9.8)],
            watched_s=9.8,
            duration_s=10.0,
            coverage_ratio=0.98,
            confirmed=True,
        ),
        qc_segments=[],
        success=SuccessVerdict(passed=True, verdict="success", confidence=1.0),
        macro_stages=[MacroStage(stage_id="S1", t_start=0.0, t_end=10.0, label="Full episode")],
        notes="verified",
        diff_summary=ReviewDiff(fp_deleted=2, total_edits=2),
    )
    restored = HumanReviewArtifact.model_validate_json(rev.model_dump_json())
    assert restored == rev


def test_review_write_payload_rejects_readonly_vlm_layers() -> None:
    from robo_vlm_qc.models.annotation import SuccessVerdict
    from robo_vlm_qc.models.review import ReviewWritePayload, WatchCoverage

    base_payload = {
        "base_run_id": "run_001",
        "action": "submit_review",
        "qc_segments": [],
        "success": SuccessVerdict(passed=True, verdict="success").model_dump(),
        "macro_stages": [],
        "watch_coverage": WatchCoverage(duration_s=10.0, watched_s=9.6, coverage_ratio=0.96).model_dump(),
    }
    ok_obj = ReviewWritePayload.model_validate(base_payload)
    assert ok_obj.action == "submit_review"

    for ro_key in ("grounding", "arm_timelines", "raw_vlm_segments"):
        with pytest.raises(ValidationError):
            ReviewWritePayload.model_validate({**base_payload, ro_key: {}})


def test_resolved_episode_immutable_view() -> None:
    from robo_vlm_qc.models.review import EpisodeStatus, ResolvedEpisode

    meta = _make_sample_meta()
    vlm = _make_sample_vlm_run()
    res = ResolvedEpisode(
        episode_id=meta.episode_id,
        meta=meta,
        status=EpisodeStatus.VLM_READY,
        active_run_id=vlm.run_id,
        is_human_reviewed=False,
        qc_segments=vlm.qc_segments,
        success=vlm.success,
        macro_stages=vlm.macro_stages,
        grounding=vlm.grounding,
        arm_timelines=vlm.arm_timelines,
        raw_vlm_segments=vlm.qc_segments,
    )
    assert res.status == EpisodeStatus.VLM_READY
    with pytest.raises(ValidationError):
        res.status = EpisodeStatus.APPROVED  # type: ignore[misc]


def test_visual_label_def_and_registry_validation() -> None:
    from robo_vlm_qc.models.taxonomy import (
        CollectorUiHint,
        LabelRegistry,
        VisualLabelDef,
        VlmSemantics,
    )

    lbl = VisualLabelDef(
        id="custom_grasp_miss",
        name="Custom Missed Grasp",
        category="failure_recovery",
        default_training_role="recovery",
        default_splice_policy="mask_loss",
        collector_ui=CollectorUiHint(title="Missed grasp", hint="Gripper retries grasp"),
        vlm_semantics=VlmSemantics(
            definition="Gripper misses target and retries",
            visual_cues=["Fingers close on empty air"],
        ),
    )
    reg = LabelRegistry(categories=["failure_recovery"], labels=[lbl])
    assert reg.get("custom_grasp_miss") is not None
    assert reg.get("missing") is None

    with pytest.raises(ValidationError):
        LabelRegistry(categories=["failure_recovery"], labels=[lbl, lbl])
    with pytest.raises(ValidationError):
        VisualLabelDef(
            id="long_pause",
            name="Legacy Label",
            category="idle",
            default_training_role="badcase",
            default_splice_policy="cut_clean",
            collector_ui=CollectorUiHint(title="x"),
            vlm_semantics=VlmSemantics(definition="x"),
        )


def test_task_spec_prompt_bundle_and_few_shot_item() -> None:
    from robo_vlm_qc.models.taxonomy import (
        FewShotItem,
        GroundingObjectSpec,
        PromptBundle,
        PromptTemplate,
        TaskSpec,
    )

    ts = TaskSpec(
        task_id="test_task",
        title="Test Manipulation Task",
        match=["*test*"],
        criteria=["Target object reaches target zone"],
        steps=["Grasp", "Place"],
        grounding_objects=[GroundingObjectSpec(id="obj", name="Target Object")],
    )
    assert ts.grounding_objects[0].id == "obj"
    assert ts.grounding_objects[0].name == "Target Object"

    pb = PromptBundle(
        s1_grounding=PromptTemplate(stage="s1_grounding", system_template="s1", user_template="u1"),
        s2_macro=PromptTemplate(stage="s2_macro", system_template="s2", user_template="u2"),
        s3_discover=PromptTemplate(stage="s3_discover", system_template="s3", user_template="u3"),
        s4_verify=PromptTemplate(stage="s4_verify", system_template="s4", user_template="u4"),
        s5_success=PromptTemplate(stage="s5_success", system_template="s5", user_template="u5"),
    )
    assert pb.s3_discover.stage == "s3_discover"

    fs = FewShotItem(
        example_id="fs_01",
        scope="task:pick_and_place",
        task_id="pick_and_place",
        label_id="missed_grasp_retry",
        error_type="fp_overflag",
        episode_id="pick_ds/ep_0001",
        t_start=1.0,
        t_end=2.0,
        vlm_wrong_json={"label_id": "missed_grasp_retry"},
        human_right_json={"splice_policy": "keep_positive"},
        rationale="Normal slow alignment, not a missed grasp",
        created_at=1700000300.0,
    )
    assert fs.scope == "task:pick_and_place"
    with pytest.raises(ValidationError):
        FewShotItem(
            example_id="fs_bad",
            scope="invalid_scope",
            error_type="fp_overflag",
            episode_id="pick_ds/ep_0001",
            t_start=2.0,
            t_end=1.0,
            rationale="bad",
            created_at=1.0,
        )


def test_export_recipe_keep_piece_and_manifest() -> None:
    from robo_vlm_qc.models.export import (
        ExportEpisodeEntry,
        ExportManifest,
        ExportRecipe,
        KeepPiece,
        SubtaskSpan,
    )
    from robo_vlm_qc.models.review import EpisodeStatus

    recipe = ExportRecipe(name="bc_clean", format="lerobot_v3", chunk_size=30, smooth_tail="on_jump")
    piece = KeepPiece(
        piece_id="pick_ds__ep_0001__p000",
        episode_id="pick_ds/ep_0001",
        piece_index=0,
        start_row=0,
        end_row=30,
        own_rows=30,
        tail_rows=0,
        total_rows=30,
        fps=30.0,
        t_start_s=0.0,
        t_end_s=1.0,
        action_loss_mask=[[1.0, 1.0]] * 30,
        is_tail=[False] * 30,
        subtask_indices=[0] * 30,
        subtasks=[
            SubtaskSpan(
                subtask_index=0,
                stage_id="S1",
                label="Pick target object",
                label_en="pick_target_object",
                start_row=0,
                end_row=30,
            )
        ],
    )
    manifest = ExportManifest(
        export_id="exp_001",
        created_at=1700000400.0,
        format="lerobot_v3",
        recipe=recipe,
        output_dir="/tmp/out",
        episodes_included=1,
        pieces_exported=1,
        total_frames=30,
        episodes=[
            ExportEpisodeEntry(
                episode_id="pick_ds/ep_0001",
                status=EpisodeStatus.APPROVED,
                passed=True,
                pieces_exported=1,
                own_frames=30,
                total_frames=30,
            )
        ],
    )
    assert manifest.total_frames == piece.total_rows

    with pytest.raises(ValidationError):
        KeepPiece(
            piece_id="bad_piece",
            episode_id="pick_ds/ep_0001",
            piece_index=0,
            start_row=0,
            end_row=10,
            own_rows=9,  # mismatch
            tail_rows=0,
            total_rows=9,
            fps=30.0,
            t_start_s=0.0,
            t_end_s=0.3,
        )


# ===========================================================================
# InputAdapter Registry, MediaSource, 2x2 Tiler & PreviewCache
# ===========================================================================


def test_input_adapter_registry_and_auto_detect(tmp_path: Path) -> None:
    from robo_vlm_qc.io import (
        auto_detect,
        detect_adapter,
        get_adapter,
        list_adapters,
    )

    names = list_adapters()
    assert set(names) == {"h5_raw_v3", "lerobot", "video_dir"}
    for nm in ("h5_raw_v3", "lerobot", "video_dir"):
        assert get_adapter(nm).name == nm

    vdir = h.make_synthetic_video_dir(tmp_path / "vid_batch", multi_cam=False, duration_s=0.6)
    h5p = h.make_synthetic_h5_raw_v3(tmp_path / "h5_batch", duration_s=0.6)
    lr_dir = h.make_synthetic_lerobot(tmp_path / "lr_batch", version="v3", n_episodes=1, duration_s=0.6)
    assert detect_adapter(h5p).name == "h5_raw_v3"  # type: ignore[union-attr]
    assert detect_adapter(lr_dir).name == "lerobot"  # type: ignore[union-attr]
    handles = auto_detect(tmp_path)
    ad_types = {hd.adapter_type for hd in handles}
    assert ad_types == {"video_dir", "h5_raw_v3", "lerobot"}
    assert vdir.exists()


def test_media_source_read_frame_and_batch(tmp_path: Path) -> None:
    from robo_vlm_qc.io import MediaSource, auto_detect, get_adapter

    vdir = h.make_synthetic_video_dir(tmp_path / "batch_v", multi_cam=True, duration_s=1.0)
    handles = auto_detect(vdir, adapter_name="video_dir")
    assert len(handles) == 1
    with get_adapter("video_dir").open(handles[0]) as view:
        assert isinstance(view.media, MediaSource)
        fr0 = view.media.read_frame_rgb("main", 0.0)
        fr1 = view.media.read_frame_rgb("main", 0.5)
        assert fr0.shape == (480, 640, 3)
        assert fr0.dtype == np.uint8
        assert not np.array_equal(fr0, fr1)
        batch = view.media.read_frames_batch("main", [0.0, 0.5, 0.9])
        assert len(batch) == 3
        assert np.array_equal(batch[0], fr0)
        assert np.array_equal(batch[1], fr1)


def test_adaptive_tile_composer_wrist_2x2(tmp_path: Path) -> None:
    from robo_vlm_qc.io import auto_detect, compose_2x2_tile, get_adapter, select_tile_mode

    lr_dir = h.make_synthetic_lerobot(
        tmp_path / "lr_wrist_root", version="v3", n_episodes=1, duration_s=1.0
    )
    handles = auto_detect(lr_dir, adapter_name="lerobot")
    with get_adapter("lerobot").open(handles[0]) as view:
        assert select_tile_mode(view.camera_roles) == "wrist_2x2"
        tile = compose_2x2_tile(
            view.media,
            view.camera_roles,
            t_s=0.3,
            cell_w=320,
            cell_h=180,
            include_header=True,
            header_h=26,
        )
        assert tile.shape == (26 + 360, 640, 3)
        assert tile.dtype == np.uint8
        # Verify 4 quadrants contain distinct image content (L-WRIST, R-WRIST, MAIN, MAIN-ZOOM)
        body = tile[26:]
        q_tl = body[:180, :320]
        q_tr = body[:180, 320:]
        q_bl = body[180:, :320]
        q_br = body[180:, 320:]
        assert not np.array_equal(q_tl, q_tr)
        assert not np.array_equal(q_bl, q_br)
        assert not np.array_equal(q_tl, q_bl)


def test_adaptive_tile_composer_torso_head_2x2(tmp_path: Path) -> None:
    from robo_vlm_qc.io import auto_detect, compose_2x2_tile, get_adapter, select_tile_mode

    h5p = h.make_synthetic_h5_raw_v3(tmp_path / "pick_ds", duration_s=1.0)
    handles = auto_detect(h5p, adapter_name="h5_raw_v3")
    with get_adapter("h5_raw_v3").open(handles[0]) as view:
        assert select_tile_mode(view.camera_roles) == "torso_head_2x2"
        tile = compose_2x2_tile(
            view.media,
            view.camera_roles,
            t_s=0.4,
            cell_w=320,
            cell_h=180,
            include_header=True,
            header_h=26,
        )
        assert tile.shape == (26 + 360, 640, 3)
        body = tile[26:]
        assert not np.array_equal(body[:180, :320], body[180:, :320])  # TORSO vs HEAD
        assert not np.array_equal(body[:180, :320], body[:180, 320:])  # TORSO vs TORSO-ZOOM


def test_adaptive_tile_composer_single_1x2(tmp_path: Path) -> None:
    from robo_vlm_qc.io import auto_detect, compose_2x2_tile, get_adapter, select_tile_mode

    vdir = h.make_synthetic_video_dir(tmp_path / "single_ds", multi_cam=False, duration_s=1.0)
    handles = auto_detect(vdir, adapter_name="video_dir")
    with get_adapter("video_dir").open(handles[0]) as view:
        assert select_tile_mode(view.camera_roles) == "single_1x2"
        tile = compose_2x2_tile(
            view.media,
            view.camera_roles,
            t_s=0.4,
            cell_w=320,
            cell_h=180,
            include_header=True,
            header_h=26,
        )
        assert tile.shape == (26 + 180, 640, 3)
        body = tile[26:]
        assert not np.array_equal(body[:, :320], body[:, 320:])


def test_preview_cache_build_and_mtime_reuse(
    e2e_workspace: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.io import PreviewCache, auto_detect, get_adapter

    h5p = h.make_synthetic_h5_raw_v3(tmp_path / "pick_prev", duration_s=1.0)
    handles = auto_detect(h5p, adapter_name="h5_raw_v3")
    cache = PreviewCache(e2e_workspace)
    with get_adapter("h5_raw_v3").open(handles[0]) as view:
        p1 = cache.get_or_build(
            view.media,
            dataset=view.meta.dataset,
            stem=view.meta.stem,
            camera="head_rgbd",
            source_mtime=view.meta.source_mtime,
        )
        assert p1.is_file()
        vinfo = h.ffprobe_video_info(p1)
        assert vinfo["width"] > 0 and vinfo["height"] > 0
        assert (vinfo["nb_frames"] or 0) >= 25
        mtime1 = p1.stat().st_mtime_ns

        # Second call with same source_mtime hits cache
        p2 = cache.get_or_build(
            view.media,
            dataset=view.meta.dataset,
            stem=view.meta.stem,
            camera="head_rgbd",
            source_mtime=view.meta.source_mtime,
        )
        assert p2 == p1
        assert p2.stat().st_mtime_ns == mtime1

        # Newer source_mtime triggers rebuild
        time.sleep(0.02)
        p3 = cache.get_or_build(
            view.media,
            dataset=view.meta.dataset,
            stem=view.meta.stem,
            camera="head_rgbd",
            source_mtime=time.time() + 100.0,
        )
        assert p3.stat().st_mtime_ns > mtime1


# ===========================================================================
# 3-In Input Adapters (Synthetic, Strictly Read-Only)
# ===========================================================================


def test_h5_raw_v3_synthetic_alignment_slerp_tactile_and_stereo(
    tmp_path: Path,
) -> None:
    from robo_vlm_qc.io import KinematicsTrack, auto_detect, get_adapter

    h5p = h.make_synthetic_h5_raw_v3(
        tmp_path / "pick_ds",
        duration_s=3.0,
        with_tactile_dropout=True,
        with_sbs_stereo=True,
    )
    before = h.snapshot_tree_mtimes(h5p)
    handles = auto_detect(h5p, adapter_name="h5_raw_v3")
    assert len(handles) == 1
    with get_adapter("h5_raw_v3").open(handles[0]) as view:
        assert view.kinematics is not None
        assert isinstance(view.kinematics, KinematicsTrack)
        kin = view.kinematics
        T = view.meta.frame_count
        assert view.meta.fps == pytest.approx(30.0)
        assert kin.timestamps_s.shape == (T,)
        assert np.allclose(np.diff(kin.timestamps_s), 1.0 / 30.0, atol=1e-4)
        assert kin.eef_pos is not None and kin.eef_pos.shape == (T, 2, 3)
        assert kin.eef_rotvec is not None and kin.eef_rotvec.shape == (T, 2, 3)
        # Slerp across quaternion sign flip produces smooth rotvec steps (no pi jump)
        max_rot_step = float(np.max(np.linalg.norm(np.diff(kin.eef_rotvec, axis=0), axis=-1)))
        assert max_rot_step < 0.2

        # Short tactile dropout repaired smoothly on native grid -> all valid & finite & >= 0.35
        assert kin.tactile is not None and kin.tactile.shape == (T, 30)
        assert np.isfinite(kin.tactile).all()
        assert float(np.min(kin.tactile)) >= 0.35
        assert bool(kin.sub_valid["force"].all())

        # Virtual SBS stereo L/R crops
        cam_names = {c.name for c in view.meta.cameras}
        assert {"head_rgbd", "torso_rgbd", "head_stereo_left", "head_stereo_right"}.issubset(cam_names)
        left_rgb = view.media.read_frame_rgb("head_stereo_left", 1.0)
        right_rgb = view.media.read_frame_rgb("head_stereo_right", 1.0)
        assert left_rgb.shape == (1296, 1600, 3)
        assert right_rgb.shape == (1296, 1600, 3)
        assert not np.array_equal(left_rgb, right_rgb)

    assert h.snapshot_tree_mtimes(h5p) == before


def test_h5_raw_v3_long_tactile_dropout_marks_force_invalid(tmp_path: Path) -> None:
    from robo_vlm_qc.io import auto_detect, get_adapter

    h5p = h.make_synthetic_h5_raw_v3(
        tmp_path / "pick_long_gap",
        duration_s=3.0,
        with_tactile_dropout=False,
        with_sbs_stereo=False,
        long_tactile_dropout=True,
    )
    handles = auto_detect(h5p, adapter_name="h5_raw_v3")
    with get_adapter("h5_raw_v3").open(handles[0]) as view:
        assert view.kinematics is not None
        force_valid = view.kinematics.sub_valid["force"]
        assert not bool(force_valid.all()), "Long tactile dropout >=0.5s must mark force valid_mask False"
        assert bool(force_valid.any()), "Non-dropout region must stay True"


def test_lerobot_v3_synthetic_adapter(tmp_path: Path) -> None:
    from robo_vlm_qc.io import auto_detect, get_adapter

    ds = h.make_synthetic_lerobot(tmp_path / "lr_v3_pick", version="v3", n_episodes=2, duration_s=1.5)
    before = h.snapshot_tree_mtimes(ds)
    handles = auto_detect(ds, adapter_name="lerobot")
    assert len(handles) == 2
    ad = get_adapter("lerobot")
    for hd in handles:
        with ad.open(hd) as view:
            assert view.meta.adapter_type == "lerobot"
            assert view.meta.extra_meta["codebase_version"].startswith("v3")
            assert view.kinematics is not None
            assert view.kinematics.raw_state is not None
            assert view.kinematics.raw_state.shape[1] == 81
            assert view.kinematics.raw_action is not None
            assert view.kinematics.raw_action.shape[1] == 81
            assert len(view.meta.cameras) == 3
            fr0 = view.media.read_frame_rgb("main", 0.0)
            fr1 = view.media.read_frame_rgb("main", 1.0)
            assert fr0.shape == (480, 640, 3)
            assert not np.array_equal(fr0, fr1)
    assert h.snapshot_tree_mtimes(ds) == before


def test_lerobot_v2_synthetic_adapter(tmp_path: Path) -> None:
    from robo_vlm_qc.io import auto_detect, get_adapter

    ds = h.make_synthetic_lerobot(
        tmp_path / "lr_v2_handover",
        version="v2",
        n_episodes=2,
        duration_s=1.5,
        task_name="bimanual_handover",
    )
    handles = auto_detect(ds, adapter_name="lerobot")
    assert len(handles) == 2
    ad = get_adapter("lerobot")
    with ad.open(handles[0]) as view:
        assert view.meta.extra_meta["codebase_version"].startswith("v2")
        assert view.kinematics is not None and view.kinematics.raw_state is not None
        assert view.kinematics.raw_state.shape == (view.meta.frame_count, 81)
        assert view.media.read_frame_rgb("left_wrist", 0.5).shape == (480, 640, 3)


def test_video_dir_single_and_multicam_zero_kinematics(tmp_path: Path) -> None:
    from robo_vlm_qc.io import auto_detect, get_adapter

    single_root = h.make_synthetic_video_dir(
        tmp_path / "vid_single", multi_cam=False, duration_s=1.0, n_episodes=1
    )
    multi_root = h.make_synthetic_video_dir(
        tmp_path / "vid_multi", multi_cam=True, duration_s=1.0, n_episodes=1
    )

    ad = get_adapter("video_dir")
    h_single = auto_detect(single_root, adapter_name="video_dir")
    h_multi = auto_detect(multi_root, adapter_name="video_dir")
    assert len(h_single) == 1
    assert len(h_multi) == 1

    with ad.open(h_single[0]) as vs, ad.open(h_multi[0]) as vm:
        assert vs.kinematics is None and not vs.has_kinematics
        assert vs.meta.kinematics.has_kinematics is False
        assert len(vs.meta.cameras) == 1
        assert vs.camera_roles["main"] == vs.meta.cameras[0].name

        assert vm.kinematics is None and not vm.has_kinematics
        assert len(vm.meta.cameras) == 3
        assert {"main", "left_wrist", "right_wrist"}.issubset(set(vm.camera_roles.keys()))
        assert vm.media.read_frame_rgb("left_wrist", 0.2).shape == (480, 640, 3)


# ===========================================================================
# SQLite WAL, Claim Leases, Artifacts & Repository
# ===========================================================================


def test_sqlite_wal_and_core_tables(e2e_workspace: Path) -> None:
    from robo_vlm_qc.store import Repository

    repo = Repository(e2e_workspace)
    try:
        conn = repo.db.connect()
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert str(mode).lower() == "wal"
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {
            "episodes",
            "vlm_runs",
            "reviews",
            "few_shot_examples",
            "jobs",
            "schema_migrations",
        }.issubset(tables)
    finally:
        repo.close()


def test_claim_lease_acquire_conflict_renew_release_reap(e2e_workspace: Path) -> None:
    from robo_vlm_qc.models.review import EpisodeStatus
    from robo_vlm_qc.store import ClaimConflictError, Repository

    repo = Repository(e2e_workspace)
    try:
        meta = _make_sample_meta()
        repo.ingest_episode(meta)
        repo.save_vlm_run_artifact(_make_sample_vlm_run(meta.episode_id, "run_001"))

        c1 = repo.claim_episode(meta.episode_id, "alice", ttl_s=60.0, now=1000.0)
        assert c1["status"] == EpisodeStatus.CLAIMED.value
        assert c1["claim_expires_at"] == pytest.approx(1060.0)

        with pytest.raises(ClaimConflictError):
            repo.claim_episode(meta.episode_id, "bob", ttl_s=60.0, now=1010.0)

        renewed = repo.renew_claim(meta.episode_id, "alice", ttl_s=120.0, now=1020.0)
        assert renewed["claim_expires_at"] == pytest.approx(1140.0)

        rel_status = repo.release_claim(meta.episode_id, "alice", now=1030.0)
        assert rel_status == EpisodeStatus.VLM_READY.value

        # Re-claim and let expire -> reap_expired_claims unlocks
        repo.claim_episode(meta.episode_id, "bob", ttl_s=30.0, now=2000.0)
        reaped = repo.reap_expired_claims(now=2050.0)
        assert reaped == 1
        assert repo.get_episode_row(meta.episode_id)["status"] == EpisodeStatus.VLM_READY.value
    finally:
        repo.close()


def test_atomic_json_and_history_snapshot_backup(e2e_workspace: Path) -> None:
    from robo_vlm_qc.store import ArtifactStore

    store = ArtifactStore(e2e_workspace)
    art_path = store.write_immutable_json("artifacts/vlm_runs/test/ep/r1.json", {"v": 1})
    assert art_path.is_file()
    with pytest.raises(FileExistsError):
        store.write_immutable_json("artifacts/vlm_runs/test/ep/r1.json", {"v": 2})

    cfg_path = store.write_config_override("labels.yaml", "version: 1\n", reason="init")
    assert cfg_path.is_file()
    store.write_config_override("labels.yaml", "version: 2\n", reason="bump_v2")
    snaps = list((e2e_workspace / ".history").rglob("*.json"))
    assert len(snaps) == 1
    snap_data = json.loads(snaps[0].read_text(encoding="utf-8"))
    assert snap_data["reason"] == "bump_v2"
    assert snap_data["content"] == "version: 1\n"


def test_repository_episode_crud_and_state_transitions(e2e_workspace: Path) -> None:
    from robo_vlm_qc.models.review import EpisodeStatus
    from robo_vlm_qc.store import InvalidTransitionError, Repository

    repo = Repository(e2e_workspace)
    try:
        meta = _make_sample_meta()
        _, out1 = repo.ingest_episode(meta)
        assert out1 == "inserted"
        _, out2 = repo.ingest_episode(meta)
        assert out2 == "skipped"
        _, out3 = repo.ingest_episode(meta.model_copy(update={"source_mtime": meta.source_mtime + 5.0}))
        assert out3 == "updated"

        with pytest.raises(InvalidTransitionError):
            repo.transition_status(meta.episode_id, EpisodeStatus.APPROVED)

        repo.transition_status(meta.episode_id, EpisodeStatus.VLM_QUEUED)
        repo.transition_status(meta.episode_id, EpisodeStatus.VLM_RUNNING)
        repo.save_vlm_run_artifact(_make_sample_vlm_run(meta.episode_id, "run_crud"))

        assert len(repo.list_episodes(status=EpisodeStatus.VLM_READY, dataset="pick_ds", passed=True)) == 1
        assert len(repo.list_episodes(has_unknown=True)) == 0
    finally:
        repo.close()


def test_resolve_episode_overlays_three_layers_and_preserves_readonly(
    e2e_workspace: Path,
) -> None:
    from robo_vlm_qc.models.annotation import MacroStage, QcSegment, SuccessVerdict
    from robo_vlm_qc.models.review import (
        EpisodeStatus,
        ReviewWritePayload,
        WatchCoverage,
    )
    from robo_vlm_qc.store import Repository

    repo = Repository(e2e_workspace)
    try:
        meta = _make_sample_meta()
        repo.ingest_episode(meta)
        vlm = _make_sample_vlm_run(meta.episode_id, "run_res")
        repo.save_vlm_run_artifact(vlm)

        # Before review -> ResolvedEpisode mirrors VlmRunArtifact
        r0 = repo.resolve_episode(meta.episode_id)
        assert not r0.is_human_reviewed
        assert len(r0.qc_segments) == 2

        # Submit human review changing all 3 editable layers
        new_seg = QcSegment(
            seg_id="seg_human",
            t_start=4.0,
            t_end=4.8,
            label_id="collision_or_knock",
            training_role="badcase",
            splice_policy="mask_loss",
            mask_side="right",
            vlm_raw_policy="mask_loss",
        )
        new_stages = [MacroStage(stage_id="S1", t_start=0.0, t_end=10.0, label="Human Stage")]
        new_success = SuccessVerdict(passed=False, verdict="failure", confidence=1.0)

        repo.claim_episode(meta.episode_id, "rev_user")
        repo.save_review(
            meta.episode_id,
            author="rev_user",
            role="reviewer",
            payload=ReviewWritePayload(
                base_run_id=vlm.run_id,
                action="submit_review",
                qc_segments=[new_seg],
                success=new_success,
                macro_stages=new_stages,
                watch_coverage=WatchCoverage(duration_s=10.0, watched_s=9.9, coverage_ratio=0.99, confirmed=True),
            ),
        )

        r1 = repo.resolve_episode(meta.episode_id)
        assert r1.status == EpisodeStatus.REVIEWED
        assert r1.is_human_reviewed is True
        assert [s.seg_id for s in r1.qc_segments] == ["seg_human"]
        assert r1.success.passed is False
        assert r1.macro_stages[0].label == "Human Stage"
        # Read-only VLM layers remain untouched
        assert r1.grounding == vlm.grounding
        assert r1.arm_timelines == vlm.arm_timelines
        assert r1.raw_vlm_segments == vlm.qc_segments
        assert r1.review_diff is not None
        assert r1.review_diff.fp_deleted == 2
        assert r1.review_diff.fn_added == 1
        assert r1.review_diff.success_flipped is True
    finally:
        repo.close()


def test_gt_frozen_enforcement(e2e_workspace: Path) -> None:
    from robo_vlm_qc.models.annotation import SuccessVerdict
    from robo_vlm_qc.models.review import (
        EpisodeStatus,
        ReviewWritePayload,
        WatchCoverage,
    )
    from robo_vlm_qc.store import GtFrozenError, Repository

    repo = Repository(e2e_workspace)
    try:
        meta = _make_sample_meta()
        repo.ingest_episode(meta)
        vlm = _make_sample_vlm_run(meta.episode_id, "run_gt")
        repo.save_vlm_run_artifact(vlm)

        wc = WatchCoverage(duration_s=10.0, watched_s=10.0, coverage_ratio=1.0, confirmed=True)
        sv = SuccessVerdict(passed=True, verdict="success", confidence=1.0)
        repo.save_review(
            meta.episode_id,
            author="lead_u",
            role="lead",
            payload=ReviewWritePayload(
                base_run_id=vlm.run_id,
                action="freeze_gt",
                qc_segments=vlm.qc_segments,
                success=sv,
                macro_stages=vlm.macro_stages,
                watch_coverage=wc,
            ),
        )
        assert repo.get_episode_row(meta.episode_id)["status"] == EpisodeStatus.GT_FROZEN.value

        with pytest.raises(GtFrozenError):
            repo.claim_episode(meta.episode_id, "alice")
        with pytest.raises(GtFrozenError):
            repo.save_review(
                meta.episode_id,
                author="lead_u",
                role="lead",
                payload=ReviewWritePayload(
                    base_run_id=vlm.run_id,
                    action="approve",
                    qc_segments=[],
                    success=sv,
                    macro_stages=[],
                    watch_coverage=wc,
                ),
            )
        with pytest.raises(GtFrozenError):
            repo.ingest_episode(meta.model_copy(update={"source_mtime": meta.source_mtime + 10.0}))
        with pytest.raises(GtFrozenError):
            repo.save_vlm_run_artifact(_make_sample_vlm_run(meta.episode_id, "run_overwrite"))

        # Eval sandbox runs ARE allowed against gt_frozen episodes
        sandbox_run = _make_sample_vlm_run(meta.episode_id, "run_sandbox_1").model_copy(
            update={"is_eval_sandbox": True}
        )
        repo.save_vlm_run_artifact(sandbox_run)
        assert repo.get_episode_row(meta.episode_id)["active_run_id"] == vlm.run_id
    finally:
        repo.close()


def test_reviewer_rbac_and_watch_coverage_guards(e2e_workspace: Path) -> None:
    from robo_vlm_qc.models.annotation import SuccessVerdict
    from robo_vlm_qc.models.review import ReviewWritePayload, WatchCoverage
    from robo_vlm_qc.store import AuthorizationError, Repository, WatchCoverageError

    repo = Repository(e2e_workspace)
    try:
        meta = _make_sample_meta()
        repo.ingest_episode(meta)
        vlm = _make_sample_vlm_run(meta.episode_id, "run_rbac")
        repo.save_vlm_run_artifact(vlm)
        sv = SuccessVerdict(passed=True, verdict="success")

        with pytest.raises(AuthorizationError):
            repo.save_review(
                meta.episode_id,
                author="rev_u",
                role="reviewer",
                payload=ReviewWritePayload(
                    base_run_id=vlm.run_id,
                    action="approve",
                    qc_segments=[],
                    success=sv,
                    macro_stages=[],
                    watch_coverage=WatchCoverage(duration_s=10.0, watched_s=10.0, coverage_ratio=1.0),
                ),
            )

        with pytest.raises(WatchCoverageError):
            repo.save_review(
                meta.episode_id,
                author="rev_u",
                role="reviewer",
                payload=ReviewWritePayload(
                    base_run_id=vlm.run_id,
                    action="submit_review",
                    qc_segments=[],
                    success=sv,
                    macro_stages=[],
                    watch_coverage=WatchCoverage(duration_s=10.0, watched_s=5.0, coverage_ratio=0.50),
                ),
            )
    finally:
        repo.close()


# ===========================================================================
# `robo-vlm-qc ingest` CLI Across All 3 Formats & Idempotency
# ===========================================================================


def test_cli_ingest_h5_raw_v3(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    h5_dir = tmp_path / "sample_pick_and_place_batch"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=1.0, task_name="unmatched_hint")
    res = h.run_cli(
        [
            "-w",
            str(e2e_workspace),
            "-c",
            str(e2e_config_dir),
            "ingest",
            "--data",
            str(h5_dir),
            "--json",
        ]
    )
    assert res.returncode == 0, res.stderr + res.stdout
    payload = json.loads(res.stdout)
    assert payload["inserted"] == 1
    assert payload["episodes"][0]["adapter_type"] == "h5_raw_v3"
    assert payload["episodes"][0]["task_id"] == "pick_and_place"
    assert payload["episodes"][0]["has_kinematics"] is True


def test_cli_ingest_lerobot_v3_and_v2(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    lr3 = h.make_synthetic_lerobot(
        tmp_path / "lr3_pick", version="v3", n_episodes=2, duration_s=1.0, task_name="pick_and_place"
    )
    lr2 = h.make_synthetic_lerobot(
        tmp_path / "lr2_handover",
        version="v2",
        n_episodes=1,
        duration_s=1.0,
        task_name="bimanual_handover",
    )
    res = h.run_cli(
        [
            "-w",
            str(e2e_workspace),
            "-c",
            str(e2e_config_dir),
            "ingest",
            "--data",
            str(lr3),
            "--data",
            str(lr2),
            "--json",
        ]
    )
    assert res.returncode == 0, res.stderr + res.stdout
    payload = json.loads(res.stdout)
    assert payload["inserted"] == 3
    assert payload["per_adapter"]["lerobot"] == 3


def test_cli_ingest_video_dir(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    vdir = h.make_synthetic_video_dir(
        tmp_path / "pure_videos", multi_cam=True, duration_s=1.0, n_episodes=2
    )
    res = h.run_cli(
        [
            "-w",
            str(e2e_workspace),
            "-c",
            str(e2e_config_dir),
            "ingest",
            "--data",
            str(vdir),
            "--json",
        ]
    )
    assert res.returncode == 0, res.stderr + res.stdout
    payload = json.loads(res.stdout)
    assert payload["inserted"] == 2
    assert payload["per_adapter"]["video_dir"] == 2
    assert all(ep["has_kinematics"] is False for ep in payload["episodes"])


def test_cli_ingest_idempotent_preserves_status_and_reviews(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.models.annotation import SuccessVerdict
    from robo_vlm_qc.models.review import (
        EpisodeStatus,
        ReviewWritePayload,
        WatchCoverage,
    )
    from robo_vlm_qc.store import Repository

    vdir = h.make_synthetic_video_dir(tmp_path / "idem_vid", multi_cam=False, duration_s=1.0)
    res1 = h.run_cli(
        ["-w", str(e2e_workspace), "-c", str(e2e_config_dir), "ingest", "--data", str(vdir), "--json"]
    )
    assert res1.returncode == 0, res1.stderr + res1.stdout
    ep_id = json.loads(res1.stdout)["episodes"][0]["episode_id"]

    repo = Repository(e2e_workspace)
    try:
        vlm = _make_sample_vlm_run(ep_id, "run_idem")
        repo.save_vlm_run_artifact(vlm)
        repo.save_review(
            ep_id,
            author="rev_u",
            role="reviewer",
            payload=ReviewWritePayload(
                base_run_id="run_idem",
                action="submit_review",
                qc_segments=[],
                success=SuccessVerdict(passed=True, verdict="success"),
                macro_stages=[],
                watch_coverage=WatchCoverage(duration_s=1.0, watched_s=1.0, coverage_ratio=1.0, confirmed=True),
            ),
        )
    finally:
        repo.close()

    # Re-ingest same source
    res2 = h.run_cli(
        ["-w", str(e2e_workspace), "-c", str(e2e_config_dir), "ingest", "--data", str(vdir), "--json"]
    )
    assert res2.returncode == 0, res2.stderr + res2.stdout
    p2 = json.loads(res2.stdout)
    assert p2["skipped"] == 1 and p2["inserted"] == 0

    conn = sqlite3.connect(str(e2e_workspace / "robo_vlm_qc.db"))
    try:
        row = conn.execute(
            "SELECT status, active_run_id, active_rev_id FROM episodes WHERE episode_id = ?",
            (ep_id,),
        ).fetchone()
        assert row is not None
        assert row[0] == EpisodeStatus.REVIEWED.value
        assert row[1] == "run_idem"
        assert row[2] is not None
    finally:
        conn.close()
