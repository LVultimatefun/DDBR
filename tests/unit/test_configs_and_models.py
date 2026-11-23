"""Unit tests for configs, SHA256 stage isolation, Pydantic v2 models, and backward-compatible input aliases."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from robo_vlm_qc.config import ConfigManager, default_config_dir
from robo_vlm_qc.models import (
    CameraSpec,
    EpisodeMeta,
    GroundingObjectSpec,
    GroundingTrack,
    KeepPiece,
    KinematicsSpec,
    MacroStage,
    QcSegment,
    ReviewWritePayload,
    SubtaskSpan,
    TaskSpec,
    VisualEvidence,
    VisualLabelDef,
)
from robo_vlm_qc.store import ArtifactStore


def test_default_configs_load_11_pure_visual_labels_and_tasks() -> None:
    mgr = ConfigManager()
    assert len(mgr.labels.labels) == 11
    label_ids = {lbl.id for lbl in mgr.labels.labels}
    assert "object_slip_recovery" in label_ids
    assert "unknown_anomaly" in label_ids
    assert "long_pause" not in label_ids
    assert "back_and_forth" not in label_ids

    for tid in (
        "pick_and_place",
        "bimanual_handover",
        "_default",
    ):
        assert tid in mgr.tasks

    assert mgr.prompts.s1_grounding.stage == "s1_grounding"
    assert mgr.prompts.s5_success.stage == "s5_success"


def test_prompt_alias_resolution_from_legacy_prompt_filenames(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "configs"
    shutil.copytree(default_config_dir(), cfg_dir)
    prompts_dir = cfg_dir / "prompts"
    (prompts_dir / "s2_macro.yaml").rename(prompts_dir / "s2_macro_backbone.yaml")
    (prompts_dir / "s3_discover.yaml").rename(prompts_dir / "s3_qc_discovery.yaml")
    (prompts_dir / "s4_verify.yaml").rename(prompts_dir / "s4_local_verify.yaml")
    (prompts_dir / "s5_success.yaml").rename(prompts_dir / "s5_end_state.yaml")

    mgr = ConfigManager(config_dir=cfg_dir)
    assert mgr.prompts.s2_macro.stage == "s2_macro"
    assert mgr.prompts.s3_discover.stage == "s3_discover"
    assert mgr.prompts.s4_verify.stage == "s4_verify"
    assert mgr.prompts.s5_success.stage == "s5_success"


def test_stage_cache_fingerprint_isolation_and_history_override(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    mgr_base = ConfigManager(workspace_dir=ws)
    s1_0 = mgr_base.s1_cache_basis("pick_and_place")
    s2_0 = mgr_base.s2_cache_basis("pick_and_place")
    s5_0 = mgr_base.s5_cache_basis("pick_and_place")
    s3_0 = mgr_base.s3_cache_basis("pick_and_place")
    s4_0 = mgr_base.s4_cache_basis("pick_and_place")

    # Override labels.yaml and s3_discover.yaml in workspace
    art = ArtifactStore(ws)
    labels_yaml = (default_config_dir() / "labels.yaml").read_text(encoding="utf-8")
    art.write_config_override("labels.yaml", labels_yaml + "\n", reason="first_write")
    # Modify version in labels.yaml override -> triggers .history snapshot
    art.write_config_override(
        "labels.yaml",
        labels_yaml.replace("version: 1", "version: 2", 1),
        reason="bump_label_version",
    )
    s3_yaml = (default_config_dir() / "prompts" / "s3_discover.yaml").read_text(
        encoding="utf-8"
    )
    art.write_config_override(
        "prompts/s3_discover.yaml",
        s3_yaml.replace('version: "1.0.0"', 'version: "1.1.0"', 1),
        reason="tweak_s3_prompt",
    )

    mgr_ov = ConfigManager(workspace_dir=ws)
    assert mgr_ov.labels.version == 2
    assert mgr_ov.s1_cache_basis("pick_and_place") == s1_0
    assert mgr_ov.s2_cache_basis("pick_and_place") == s2_0
    assert mgr_ov.s5_cache_basis("pick_and_place") == s5_0
    assert mgr_ov.s3_cache_basis("pick_and_place") != s3_0
    assert mgr_ov.s4_cache_basis("pick_and_place") != s4_0

    hist_files = list((ws / ".history").rglob("*.json"))
    assert len(hist_files) == 1


def test_task_routing_priority_and_budget() -> None:
    mgr = ConfigManager()
    assert mgr.resolve_task("pick_and_place", "bimanual_handover").task_id == "pick_and_place"
    assert mgr.resolve_task("bimanual_handover", None).task_id == "bimanual_handover"
    assert mgr.resolve_task("unknown_unmatched_dataset", None).task_id == "_default"
    assert mgr.resolve_task("unknown_batch", "pick_and_place").task_id == "pick_and_place"

    assert mgr.resolve_episode_budget_tokens(10.0) == 200000
    assert mgr.resolve_episode_budget_tokens(138.3) == 691500


def test_models_reject_legacy_fields_and_enforce_triplets() -> None:
    # Reject legacy type / keys
    with pytest.raises(ValidationError):
        QcSegment(
            seg_id="s1",
            t_start=1.0,
            t_end=2.0,
            label_id="long_pause",
            training_role="badcase",
            splice_policy="cut_clean",
            vlm_raw_policy="cut_clean",
        )
    with pytest.raises(ValidationError):
        QcSegment.model_validate(
            {
                "seg_id": "s1",
                "t_start": 1.0,
                "t_end": 2.0,
                "label_id": "mid_idle_freeze",
                "type": "back_and_forth",
                "training_role": "badcase",
                "splice_policy": "cut_clean",
                "vlm_raw_policy": "cut_clean",
            }
        )

    # Reject mask_loss without mask_side
    with pytest.raises(ValidationError):
        QcSegment(
            seg_id="s1",
            t_start=1.0,
            t_end=2.0,
            label_id="missed_grasp_retry",
            training_role="recovery",
            splice_policy="mask_loss",
            mask_side=None,
            vlm_raw_policy="mask_loss",
        )

    # Reject keep_positive with mask_side != None
    with pytest.raises(ValidationError):
        QcSegment(
            seg_id="s1",
            t_start=1.0,
            t_end=2.0,
            label_id="precision_fine_adjust",
            training_role="necessary",
            splice_policy="keep_positive",
            mask_side="left",
            vlm_raw_policy="keep_positive",
        )

    # Explicitly ALLOW recovery + mask_loss (object_slip_recovery & recovery_plus)
    seg_rec = QcSegment(
        seg_id="s_rec",
        t_start=2.0,
        t_end=4.0,
        label_id="object_slip_recovery",
        training_role="recovery",
        splice_policy="mask_loss",
        mask_side="right",
        affected_arm="right",
        visual_evidence=VisualEvidence(
            initial_state="object slips",
            process_change="right hand catches object",
            end_state="object stabilized",
        ),
        vlm_raw_policy="mask_loss",
    )
    assert seg_rec.training_role == "recovery"
    assert seg_rec.splice_policy == "mask_loss"

    # Reject legacy action_operator in VisualLabelDef
    with pytest.raises(ValidationError):
        VisualLabelDef.model_validate(
            {
                "id": "custom_lbl",
                "name": "Test Label",
                "category": "failure",
                "default_training_role": "badcase",
                "default_splice_policy": "mask_loss",
                "collector_ui": {"title": "t"},
                "vlm_semantics": {"definition": "d"},
                "action_operator": {"op": "pause"},
            }
        )

    # Reject read-only layers in ReviewWritePayload
    with pytest.raises(ValidationError):
        ReviewWritePayload.model_validate(
            {
                "base_run_id": "run_1",
                "action": "submit_review",
                "qc_segments": [],
                "success": {"passed": True, "verdict": "success"},
                "macro_stages": [],
                "watch_coverage": {"duration_s": 10.0, "watched_s": 10.0, "coverage_ratio": 1.0},
                "arm_timelines": {"left": [], "right": []},
            }
        )

    # EpisodeMeta wide_right + extra_meta
    meta = EpisodeMeta(
        episode_id="ds/ep0",
        dataset="ds",
        stem="ep0",
        task_id="pick_and_place",
        adapter_type="h5_raw_v3",
        source_uri="/tmp/ep0.h5",
        source_mtime=1.0,
        episode_fingerprint="abc",
        duration_s=10.0,
        fps=30.0,
        frame_count=300,
        cameras=[
            CameraSpec(name="head_rgbd", role="main", width=1280, height=720, fps=30.0),
            CameraSpec(name="head_stereo_right", role="wide_right", width=1600, height=1296, fps=30.0),
        ],
        camera_roles={"main": "head_rgbd", "wide_right": "head_stereo_right"},
        kinematics=KinematicsSpec(has_kinematics=True),
        extra_meta={"upstream_quality": {"has_complete": True}},
    )
    assert meta.camera_roles["wide_right"] == "head_stereo_right"

    # KeepPiece validation
    kp = KeepPiece(
        piece_id="p0",
        episode_id="ds/ep0",
        piece_index=0,
        start_row=0,
        end_row=10,
        own_rows=10,
        tail_rows=2,
        total_rows=12,
        fps=30.0,
        t_start_s=0.0,
        t_end_s=0.33,
    )
    assert kp.total_rows == 12

    with pytest.raises(ValidationError):
        KeepPiece(
            piece_id="p_bad_time",
            episode_id="ds/ep0",
            piece_index=0,
            start_row=0,
            end_row=10,
            own_rows=10,
            tail_rows=0,
            total_rows=10,
            fps=30.0,
            t_start_s=5.0,
            t_end_s=2.0,
        )
    with pytest.raises(ValidationError):
        KeepPiece(
            piece_id="p_equal_time",
            episode_id="ds/ep0",
            piece_index=0,
            start_row=0,
            end_row=10,
            own_rows=10,
            tail_rows=0,
            total_rows=10,
            fps=30.0,
            t_start_s=1.0,
            t_end_s=1.0,
        )


def test_backward_compatible_input_aliases() -> None:
    """Legacy *_zh input keys map cleanly to name/title/label and serialize under new keys."""
    lbl = VisualLabelDef.model_validate(
        {
            "id": "custom_alias_label",
            "name_zh": "Alias Visual Label",
            "category": "failure",
            "default_training_role": "badcase",
            "default_splice_policy": "mask_loss",
            "collector_ui": {"title": "Alias Title"},
            "vlm_semantics": {"definition": "Alias definition"},
        }
    )
    assert lbl.name == "Alias Visual Label"
    assert "name" in lbl.model_dump()
    assert "name_zh" not in lbl.model_dump()

    gobj = GroundingObjectSpec.model_validate(
        {"id": "target_bin", "name_zh": "Target Bin", "desc": "Target container", "is_static": True}
    )
    assert gobj.name == "Target Bin"
    assert "name_zh" not in gobj.model_dump()

    tspec = TaskSpec.model_validate(
        {
            "task_id": "custom_task",
            "title_zh": "Custom Manipulation Task",
            "goal": "Place object into bin",
        }
    )
    assert tspec.title == "Custom Manipulation Task"
    assert "title_zh" not in tspec.model_dump()

    gtrack = GroundingTrack.model_validate(
        {"object_id": "block", "name_zh": "Wooden Block"}
    )
    assert gtrack.name == "Wooden Block"
    assert "name_zh" not in gtrack.model_dump()

    mstage = MacroStage.model_validate(
        {
            "stage_id": "S1",
            "t_start": 0.0,
            "t_end": 2.0,
            "label_zh": "Grasp object",
            "label_en": "grasp_object",
        }
    )
    assert mstage.label == "Grasp object"
    assert "label_zh" not in mstage.model_dump()

    qseg = QcSegment.model_validate(
        {
            "seg_id": "seg_alias",
            "t_start": 0.5,
            "t_end": 1.2,
            "label_id": "boundary_idle",
            "name_zh": "Boundary Idle",
            "training_role": "badcase",
            "splice_policy": "cut_clean",
            "vlm_raw_policy": "cut_clean",
        }
    )
    assert qseg.name == "Boundary Idle"
    assert "name_zh" not in qseg.model_dump()

    span = SubtaskSpan.model_validate(
        {
            "subtask_index": 0,
            "stage_id": "S1",
            "label_zh": "Approach and grasp",
            "label_en": "approach_and_grasp",
            "start_row": 0,
            "end_row": 30,
        }
    )
    assert span.label == "Approach and grasp"
    assert "label_zh" not in span.model_dump()

    # When both new key and legacy key are provided, new key takes precedence
    lbl_both = VisualLabelDef.model_validate(
        {
            "id": "custom_both",
            "name": "Primary Name",
            "name_zh": "Fallback Alias",
            "category": "failure",
            "default_training_role": "badcase",
            "default_splice_policy": "mask_loss",
            "collector_ui": {"title": "T"},
            "vlm_semantics": {"definition": "D"},
        }
    )
    assert lbl_both.name == "Primary Name"
