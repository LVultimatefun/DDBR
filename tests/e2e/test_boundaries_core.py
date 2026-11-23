"""Core Boundary, Corner & Negative E2E Tests (Config, Models, I/O, Store, VLM DAG, Guardrails, HUD MP4).

IMPORTANT:
- Top-level imports MUST NOT import ``robo_vlm_qc.*`` submodules so
  ``pytest --collect-only tests/e2e`` succeeds cleanly in isolated environments.
- All ``robo_vlm_qc.*`` imports are performed lazily inside test functions.
- Never read/print real API keys or ``.env`` credential files.
"""

from __future__ import annotations

import importlib
import inspect
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from tests.e2e import helpers as h


# ---------------------------------------------------------------------------
# Lazy Module Invocation Helpers (Opaque-Box Contract Adapters)
# ---------------------------------------------------------------------------


def _call_with_matching_kwargs(fn: Any, **candidates: Any) -> Any:
    sig = inspect.signature(fn)
    kwargs: dict[str, Any] = {}
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if accepts_var_kw:
        return fn(**candidates)
    for pname in sig.parameters:
        if pname in candidates:
            kwargs[pname] = candidates[pname]
    return fn(**kwargs)


def _ingest_one_video_ep(
    ws: Path,
    cfg_dir: Path,
    data_root: Path,
    *,
    duration_s: float = 2.0,
    fps: int = 30,
    multi_cam: bool = True,
) -> str:
    h.make_synthetic_video_dir(data_root, multi_cam=multi_cam, duration_s=duration_s, fps=fps)
    proc = h.run_cli(
        [
            "-w",
            str(ws),
            "-c",
            str(cfg_dir),
            "ingest",
            "--data",
            str(data_root),
            "--json",
        ]
    )
    assert proc.returncode == 0, f"ingest failed: {proc.stderr}\n{proc.stdout}"
    payload = json.loads(proc.stdout)
    assert payload["episodes"], payload
    return str(payload["episodes"][0]["episode_id"])


def _ingest_one_h5_ep(
    ws: Path,
    cfg_dir: Path,
    data_root: Path,
    *,
    duration_s: float = 4.0,
    stationary: bool = False,
) -> str:
    ds_dir = data_root / "pick_and_place_ds"
    h.make_synthetic_h5_raw_v3(
        ds_dir,
        name="episode_000000.h5",
        duration_s=duration_s,
        stationary=stationary,
        task_name="pick_and_place",
    )
    proc = h.run_cli(
        [
            "-w",
            str(ws),
            "-c",
            str(cfg_dir),
            "ingest",
            "--data",
            str(ds_dir),
            "--json",
        ]
    )
    assert proc.returncode == 0, f"ingest failed: {proc.stderr}\n{proc.stdout}"
    payload = json.loads(proc.stdout)
    return str(payload["episodes"][0]["episode_id"])


def _make_sample_vlm_run(episode_id: str, run_id: str = "run_001") -> Any:
    from robo_vlm_qc.models.annotation import (
        ArmSpan,
        ArmTimelines,
        ConfigFingerprintSnapshot,
        CriterionResult,
        GroundingKeyframe,
        GroundingResult,
        GroundingTrack,
        MacroStage,
        QcSegment,
        StageRefs,
        SuccessVerdict,
        VisualEvidence,
        VlmRunArtifact,
    )

    return VlmRunArtifact(
        run_id=run_id,
        episode_id=episode_id,
        created_at=1700000000.0,
        config_fingerprint=ConfigFingerprintSnapshot(
            model_routing={"s1_grounding": "gpt-4o"},
            prompt_bundle_version="1.0.0",
            label_taxonomy_version="1",
            task_spec_version="1.0.0",
        ),
        stage_refs=StageRefs(s1_grounding="aa" * 16),
        grounding=GroundingResult(
            keyframes_s=[0.0, 1.0],
            object_tracks_2d={
                "target_object": GroundingTrack(
                    object_id="target_object",
                    name="Target Object",
                    keyframes=[
                        GroundingKeyframe(
                            row=0,
                            time_s=0.0,
                            visible=True,
                            bbox_norm_1000=[100, 100, 250, 250],
                        )
                    ],
                )
            },
        ),
        macro_stages=[
            MacroStage(stage_id="S1", t_start=0.0, t_end=1.0, label="Grasp object"),
            MacroStage(stage_id="S2", t_start=1.0, t_end=2.0, label="Place object"),
        ],
        arm_timelines=ArmTimelines(
            left=[ArmSpan(t_start=0.0, t_end=2.0, state="work", desc="Left arm")],
            right=[ArmSpan(t_start=0.0, t_end=2.0, state="hold", desc="Right arm")],
        ),
        qc_segments=[
            QcSegment(
                seg_id="seg_0",
                t_start=0.0,
                t_end=0.4,
                label_id="boundary_idle",
                training_role="badcase",
                splice_policy="cut_clean",
                mask_side=None,
                affected_arm="both",
                visual_evidence=VisualEvidence(
                    initial_state="Initially stationary.",
                    process_change="No motion.",
                    end_state="Preparing to start.",
                ),
                vlm_raw_policy="cut_clean",
            )
        ],
        success=SuccessVerdict(
            passed=True,
            verdict="success",
            confidence=0.95,
            criteria=[CriterionResult(criterion="Goal satisfied", met=True, satisfied_at_s=1.8)],
        ),
    )


def _invoke_guardrail(
    mod_name: str,
    qc_segments: list[Any],
    kinematics: Any,
    media: Any,
    cfg: dict[str, Any],
    duration_s: float,
) -> list[Any]:
    mod = importlib.import_module(mod_name)
    for attr in (
        "apply_splice_gate",
        "run_splice_gate",
        "evaluate_splice_gate",
        "apply_hallucination_guardrail",
        "run_hallucination_check",
        "check_hallucination",
        "apply_visual_gate",
        "run_visual_gate",
        "evaluate_visual_gate",
        "apply",
        "run",
        "guard",
    ):
        fn = getattr(mod, attr, None)
        if callable(fn):
            return _call_with_matching_kwargs(
                fn,
                qc_segments=qc_segments,
                segments=qc_segments,
                kinematics=kinematics,
                media=media,
                cfg=cfg,
                config=cfg,
                duration_s=duration_s,
            )
    for cls_name in ("SpliceGate", "HallucinationGuardrail", "VisualGate"):
        cls = getattr(mod, cls_name, None)
        if cls is not None:
            inst = _call_with_matching_kwargs(cls, cfg=cfg, config=cfg)
            for mname in ("apply", "run", "evaluate", "check", "__call__"):
                meth = getattr(inst, mname, None)
                if callable(meth):
                    return _call_with_matching_kwargs(
                        meth,
                        qc_segments=qc_segments,
                        segments=qc_segments,
                        kinematics=kinematics,
                        media=media,
                        cfg=cfg,
                        config=cfg,
                        duration_s=duration_s,
                    )
    raise AssertionError(f"No recognized guardrail entrypoint found in {mod_name}")


def _invoke_dag_run(ws: Path, cfg_dir: Path, episode_ids: list[str], workers: int = 2) -> Any:
    """Run VLM DAG via CLI so full config, cache, client, guardrail & store execute."""
    args = ["-w", str(ws), "-c", str(cfg_dir), "run-vlm", "--workers", str(workers)]
    for ep_id in episode_ids:
        args.extend(["--episode", ep_id])
    return h.run_cli(args)


# ===========================================================================
# ConfigManager, Layered Overrides, Fingerprints & Token Budgets
# ===========================================================================


def test_missing_override_falls_back_to_default(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    from robo_vlm_qc.config import ConfigManager

    # Ensure no workspace overrides exist
    for p in (e2e_workspace / "configs_override").glob("*"):
        if p.is_file():
            p.unlink()

    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    assert cfg.settings["vlm"]["stage_models"]["s1_grounding"] == "gpt-4o"
    assert cfg.settings["vlm"]["budget"]["tokens_per_second"] == 5000
    assert cfg.resolve_task("unseen_random_dataset").task_id == "_default"


def test_invalid_yaml_and_unknown_task_fallback(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    from robo_vlm_qc.config import ConfigManager

    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    assert cfg.get_task("completely_unknown_task_xyz").task_id == "_default"
    assert cfg.resolve_task("unknown_ds", h5_task_name="unknown_hint").task_id == "_default"

    ov_default = e2e_workspace / "configs_override" / "default.yaml"
    ov_default.write_text("vlm:\n  broken: [unclosed\n", encoding="utf-8")
    with pytest.raises((yaml.YAMLError, ValueError)):
        cfg.reload()

    # Non-mapping YAML root must also be rejected
    ov_default.write_text("- just_a_list_item\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected YAML mapping"):
        cfg.reload()


def test_deep_merge_nested_dict_and_list_replacement(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    from robo_vlm_qc.config import ConfigManager

    ov_default = e2e_workspace / "configs_override" / "default.yaml"
    ov_default.write_text(
        yaml.safe_dump({"vlm": {"stage_models": {"s1_grounding": "gpt-4o-mini"}}}),
        encoding="utf-8",
    )
    ov_tasks = e2e_workspace / "configs_override" / "tasks"
    ov_tasks.mkdir(parents=True, exist_ok=True)
    (ov_tasks / "pick_and_place.yaml").write_text(
        yaml.safe_dump({"task_id": "pick_and_place", "match": ["*tabletop_pick*"]}),
        encoding="utf-8",
    )

    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    # Nested dict preserved sibling keys
    assert cfg.stage_model("s1_grounding") == "gpt-4o-mini"
    assert cfg.stage_model("s2_macro") == "gpt-4o"
    assert cfg.settings["vlm"]["budget"]["tokens_per_second"] == 5000
    # Nested list replaced rather than concatenated
    task = cfg.get_task("pick_and_place")
    assert task.match == ["*tabletop_pick*"]
    assert task.title  # base fields preserved


def test_fingerprint_changes_on_single_field_edit(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    from robo_vlm_qc.config import ConfigManager

    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    fp0 = cfg.fingerprints.config_sha256
    s1_0 = cfg.s1_cache_basis("pick_and_place")
    s2_0 = cfg.s2_cache_basis("pick_and_place")
    s5_0 = cfg.s5_cache_basis("pick_and_place")
    s3_0 = cfg.s3_cache_basis("pick_and_place")

    # Edit single field in S3 prompt only
    ov_prompts = e2e_workspace / "configs_override" / "prompts"
    ov_prompts.mkdir(parents=True, exist_ok=True)
    s3_data = yaml.safe_load((e2e_config_dir / "prompts" / "s3_discover.yaml").read_text("utf-8"))
    s3_data["version"] = "1.0.1"
    (ov_prompts / "s3_discover.yaml").write_text(
        yaml.safe_dump(s3_data, allow_unicode=True), encoding="utf-8"
    )
    cfg.reload()

    assert cfg.fingerprints.config_sha256 != fp0
    assert cfg.s3_cache_basis("pick_and_place") != s3_0
    # Stage isolation: S1/S2/S5 MUST remain invariant
    assert cfg.s1_cache_basis("pick_and_place") == s1_0
    assert cfg.s2_cache_basis("pick_and_place") == s2_0
    assert cfg.s5_cache_basis("pick_and_place") == s5_0


def test_token_budget_unlimited_capped_zero_and_min_edge(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    from robo_vlm_qc.config import ConfigManager

    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    # duration_s=0.0 clamps to min_episode_tokens (200000)
    assert cfg.resolve_episode_budget_tokens(0.0) == 200000
    assert cfg.resolve_episode_budget_tokens(-10.0) == 200000
    # Long episode scales linearly when uncapped
    assert cfg.resolve_episode_budget_tokens(100.0) == 500000

    # Capped edge
    h.write_workspace_config(
        e2e_workspace,
        budget={
            "tokens_per_second": 5000,
            "min_episode_tokens": 200000,
            "max_tokens_per_episode": 300000,
            "unlimited": False,
        },
    )
    cfg.reload()
    assert cfg.resolve_episode_budget_tokens(100.0) == 300000

    # Unlimited edge
    h.write_workspace_config(e2e_workspace, budget={"unlimited": True})
    cfg.reload()
    assert cfg.resolve_episode_budget_tokens(100.0) is None


# ===========================================================================
# Episode & Annotation Models
# ===========================================================================


def test_reject_t_end_less_or_equal_t_start() -> None:
    from robo_vlm_qc.models.annotation import ArmSpan, MacroStage, QcSegment

    for t_s, t_e in [(1.5, 1.0), (1.5, 1.5), (0.0, 0.0)]:
        with pytest.raises(ValidationError):
            MacroStage(stage_id="S1", t_start=t_s, t_end=t_e, label="Stage")
        with pytest.raises(ValidationError):
            ArmSpan(t_start=t_s, t_end=t_e, state="work")
        with pytest.raises(ValidationError):
            QcSegment(
                seg_id="seg_0",
                t_start=t_s,
                t_end=t_e,
                label_id="boundary_idle",
                training_role="badcase",
                splice_policy="cut_clean",
                vlm_raw_policy="cut_clean",
            )


def test_reject_unknown_arm_state_splice_policy_training_role() -> None:
    from robo_vlm_qc.models.annotation import ArmSpan, QcSegment

    with pytest.raises(ValidationError):
        ArmSpan(t_start=0.0, t_end=1.0, state="flying")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        QcSegment(
            seg_id="s1",
            t_start=0.0,
            t_end=1.0,
            label_id="boundary_idle",
            training_role="badcase",
            splice_policy="drop_hard",  # type: ignore[arg-type]
            vlm_raw_policy="cut_clean",
        )

    with pytest.raises(ValidationError):
        QcSegment(
            seg_id="s1",
            t_start=0.0,
            t_end=1.0,
            label_id="boundary_idle",
            training_role="nonexistent_role",  # type: ignore[arg-type]
            splice_policy="cut_clean",
            vlm_raw_policy="cut_clean",
        )

    # Triplet invariant checks
    with pytest.raises(ValidationError):
        QcSegment(
            seg_id="s1",
            t_start=0.0,
            t_end=1.0,
            label_id="boundary_idle",
            training_role="badcase",
            splice_policy="cut_clean",
            mask_side="left",
            vlm_raw_policy="cut_clean",
        )
    with pytest.raises(ValidationError):
        QcSegment(
            seg_id="s1",
            t_start=0.0,
            t_end=1.0,
            label_id="missed_grasp_retry",
            training_role="recovery",
            splice_policy="mask_loss",
            mask_side=None,
            vlm_raw_policy="mask_loss",
        )


def test_reject_legacy_long_pause_and_back_and_forth() -> None:
    from robo_vlm_qc.models.annotation import QcSegment, VlmRunArtifact

    for legacy in ("long_pause", "back_and_forth"):
        with pytest.raises(ValidationError):
            QcSegment(
                seg_id="s1",
                t_start=0.0,
                t_end=1.0,
                label_id=legacy,
                training_role="badcase",
                splice_policy="mask_loss",
                mask_side="both",
                vlm_raw_policy="mask_loss",
            )
        with pytest.raises(ValidationError):
            QcSegment.model_validate(
                {
                    "seg_id": "s1",
                    "t_start": 0.0,
                    "t_end": 1.0,
                    "label_id": "mid_idle_freeze",
                    "training_role": "badcase",
                    "splice_policy": "cut_clean",
                    "vlm_raw_policy": "cut_clean",
                    legacy: True,
                }
            )
        with pytest.raises(ValidationError):
            VlmRunArtifact.model_validate(
                {
                    "run_id": "r1",
                    "episode_id": "ds/ep",
                    "created_at": 1.0,
                    "config_fingerprint": {
                        "model_routing": {},
                        "prompt_bundle_version": "1",
                        "label_taxonomy_version": "1",
                        "task_spec_version": "1",
                    },
                    legacy: [],
                }
            )

    valid_run = _make_sample_vlm_run("ds/ep")
    h.assert_no_legacy_fields(valid_run)


def test_reject_out_of_range_grounding_coords() -> None:
    from robo_vlm_qc.models.annotation import GroundingKeyframe

    bad_boxes = [
        [-1, 10, 100, 100],
        [10, 10, 1001, 100],
        [10, -5, 100, 100],
        [10, 10, 100, 1005],
        [500, 100, 200, 300],  # x2 < x1
        [100, 400, 200, 300],  # y2 < y1
    ]
    for box in bad_boxes:
        with pytest.raises(ValidationError):
            GroundingKeyframe(row=0, time_s=0.0, visible=True, bbox_norm_1000=box)

    # Invisible must be [0, 0, 0, 0]
    with pytest.raises(ValidationError):
        GroundingKeyframe(
            row=0, time_s=0.0, visible=False, bbox_norm_1000=[10, 10, 50, 50]
        )


def test_empty_qc_segments_and_macro_stages_allowed() -> None:
    from robo_vlm_qc.models.annotation import ConfigFingerprintSnapshot, VlmRunArtifact
    from robo_vlm_qc.models.episode import CameraSpec, EpisodeMeta, KinematicsSpec

    run = VlmRunArtifact(
        run_id="r_empty",
        episode_id="ds/ep0",
        created_at=1.0,
        config_fingerprint=ConfigFingerprintSnapshot(
            model_routing={},
            prompt_bundle_version="1.0",
            label_taxonomy_version="1",
            task_spec_version="1.0",
        ),
        macro_stages=[],
        qc_segments=[],
    )
    assert run.macro_stages == []
    assert run.qc_segments == []

    # EpisodeMeta validation rejects malformed episode_id / non-positive duration
    cam = CameraSpec(name="main", role="main", width=640, height=480, fps=30.0)
    kin = KinematicsSpec(has_kinematics=False)
    for bad_ep_id in ("no_slash", "ds/../ep", "a/b/c", "other_ds/ep0"):
        with pytest.raises(ValidationError):
            EpisodeMeta(
                episode_id=bad_ep_id,
                dataset="ds",
                stem="ep0",
                task_id="_default",
                adapter_type="video_dir",
                source_uri="/tmp/x",
                source_mtime=1.0,
                episode_fingerprint="abc",
                duration_s=1.0,
                fps=30.0,
                frame_count=30,
                cameras=[cam],
                camera_roles={"main": "main"},
                kinematics=kin,
            )


def test_reject_invalid_color_rgb_confidence_and_episode_id() -> None:
    from robo_vlm_qc.models.annotation import MacroStage, QcSegment, SuccessVerdict

    with pytest.raises(ValidationError):
        MacroStage(
            stage_id="S1",
            t_start=0.0,
            t_end=1.0,
            label="Stage",
            color_rgb=[-1, 100, 100],
        )
    with pytest.raises(ValidationError):
        MacroStage(
            stage_id="S1",
            t_start=0.0,
            t_end=1.0,
            label="Stage",
            color_rgb=[0, 256, 0],
        )
    with pytest.raises(ValidationError):
        QcSegment(
            seg_id="s1",
            t_start=0.0,
            t_end=1.0,
            label_id="boundary_idle",
            training_role="badcase",
            splice_policy="cut_clean",
            vlm_raw_policy="cut_clean",
            confidence=1.2,
        )
    with pytest.raises(ValidationError):
        SuccessVerdict(passed=True, verdict="success", confidence=-0.1)


# ===========================================================================
# Review, Taxonomy & Export Models
# ===========================================================================


def test_reject_invalid_episode_status_transitions(e2e_workspace: Path) -> None:
    from robo_vlm_qc.models.episode import CameraSpec, EpisodeMeta, KinematicsSpec
    from robo_vlm_qc.models.review import EpisodeStatus
    from robo_vlm_qc.store.repository import InvalidTransitionError, Repository

    with pytest.raises(ValueError):
        EpisodeStatus("nonexistent_state")

    repo = Repository(e2e_workspace)
    try:
        meta = EpisodeMeta(
            episode_id="ds1/ep1",
            dataset="ds1",
            stem="ep1",
            task_id="_default",
            adapter_type="video_dir",
            source_uri=str(e2e_workspace / "ep1.mp4"),
            source_mtime=1.0,
            episode_fingerprint="fp1",
            duration_s=2.0,
            fps=30.0,
            frame_count=60,
            cameras=[CameraSpec(name="main", role="main", width=640, height=480, fps=30.0)],
            camera_roles={"main": "main"},
            kinematics=KinematicsSpec(has_kinematics=False),
        )
        repo.ingest_episode(meta)
        with pytest.raises(InvalidTransitionError):
            repo.transition_status("ds1/ep1", EpisodeStatus.APPROVED)
        with pytest.raises(InvalidTransitionError):
            repo.transition_status("ds1/ep1", EpisodeStatus.GT_FROZEN)
    finally:
        repo.close()


def test_review_diff_zero_diff_edge() -> None:
    from robo_vlm_qc.models.review import ReviewWritePayload, WatchCoverage
    from robo_vlm_qc.store.repository import compute_review_diff

    run = _make_sample_vlm_run("ds/ep")
    diff = compute_review_diff(run, run.qc_segments, run.success, run.macro_stages)
    assert diff.fp_deleted == 0
    assert diff.fn_added == 0
    assert diff.policy_flips == 0
    assert diff.label_changes == 0
    assert diff.boundary_nudges == 0
    assert diff.macro_stage_edits == 0
    assert diff.success_flipped is False
    assert diff.total_edits == 0

    # ReviewWritePayload forbids read-only layers
    for bad_key in ("arm_timelines", "grounding", "raw_vlm_segments"):
        with pytest.raises(ValidationError):
            ReviewWritePayload.model_validate(
                {
                    "base_run_id": run.run_id,
                    "action": "save_draft",
                    "qc_segments": [],
                    "success": run.success.model_dump(),
                    "macro_stages": [],
                    "watch_coverage": WatchCoverage(duration_s=2.0).model_dump(),
                    bad_key: {},
                }
            )


def test_few_shot_item_invalid_scope_and_error_type() -> None:
    from robo_vlm_qc.models.taxonomy import FewShotItem

    with pytest.raises(ValidationError):
        FewShotItem(
            example_id="fs_1",
            scope="invalid_scope_prefix",
            error_type="fp_overflag",
            episode_id="ds/ep",
            t_start=0.5,
            t_end=1.0,
            rationale="reason",
            created_at=1.0,
        )

    with pytest.raises(ValidationError):
        FewShotItem(
            example_id="fs_2",
            scope="global",
            error_type="bogus_error_type",  # type: ignore[arg-type]
            episode_id="ds/ep",
            t_start=0.5,
            t_end=1.0,
            rationale="reason",
            created_at=1.0,
        )

    with pytest.raises(ValidationError):
        FewShotItem(
            example_id="fs_3",
            scope="task:pick_and_place",
            error_type="fp_overflag",
            episode_id="ds/ep",
            t_start=1.5,
            t_end=1.0,
            rationale="reason",
            created_at=1.0,
        )


def test_export_recipe_min_piece_s_zero_and_negative_chunk_size() -> None:
    from robo_vlm_qc.models.export import ExportRecipe

    with pytest.raises(ValidationError):
        ExportRecipe(name="bc_clean", format="lerobot_v3", min_piece_s=0.0)
    with pytest.raises(ValidationError):
        ExportRecipe(name="bc_clean", format="lerobot_v3", min_piece_s=-0.5)
    with pytest.raises(ValidationError):
        ExportRecipe(name="bc_clean", format="lerobot_v3", chunk_size=-10)
    with pytest.raises(ValidationError):
        ExportRecipe(name="bc_clean", format="lerobot_v3", chunk_size=1)
    with pytest.raises(ValidationError):
        ExportRecipe(name="bc_clean", format="lerobot_v3", tail_frames=-1)


def test_keep_piece_and_subtask_span_row_invariants() -> None:
    from robo_vlm_qc.models.export import KeepPiece, SubtaskSpan

    with pytest.raises(ValidationError):
        SubtaskSpan(
            subtask_index=0, stage_id="S1", label="Grasp", start_row=10, end_row=10
        )

    with pytest.raises(ValidationError):
        KeepPiece(
            piece_id="p0",
            episode_id="ds/ep",
            piece_index=0,
            start_row=0,
            end_row=10,
            own_rows=9,  # mismatch
            tail_rows=0,
            total_rows=9,
            fps=30.0,
            t_start_s=0.0,
            t_end_s=0.33,
        )

    with pytest.raises(ValidationError):
        KeepPiece(
            piece_id="p0",
            episode_id="ds/ep",
            piece_index=0,
            start_row=0,
            end_row=10,
            own_rows=10,
            tail_rows=2,
            total_rows=10,  # mismatch own+tail
            fps=30.0,
            t_start_s=0.0,
            t_end_s=0.33,
        )


def test_resolved_episode_frozen_immutability() -> None:
    from robo_vlm_qc.models.episode import CameraSpec, EpisodeMeta, KinematicsSpec
    from robo_vlm_qc.models.review import EpisodeStatus, ResolvedEpisode

    run = _make_sample_vlm_run("ds/ep0")
    meta = EpisodeMeta(
        episode_id="ds/ep0",
        dataset="ds",
        stem="ep0",
        task_id="_default",
        adapter_type="video_dir",
        source_uri="/tmp/ep0.mp4",
        source_mtime=1.0,
        episode_fingerprint="fp",
        duration_s=2.0,
        fps=30.0,
        frame_count=60,
        cameras=[CameraSpec(name="main", role="main", width=640, height=480, fps=30.0)],
        camera_roles={"main": "main"},
        kinematics=KinematicsSpec(has_kinematics=False),
    )
    resolved = ResolvedEpisode(
        episode_id="ds/ep0",
        meta=meta,
        status=EpisodeStatus.VLM_READY,
        active_run_id=run.run_id,
        qc_segments=run.qc_segments,
        success=run.success,
        macro_stages=run.macro_stages,
        grounding=run.grounding,
        arm_timelines=run.arm_timelines,
        raw_vlm_segments=run.qc_segments,
    )
    with pytest.raises(ValidationError):
        resolved.status = EpisodeStatus.APPROVED  # type: ignore[misc]


# ===========================================================================
# InputAdapter Registry, MediaSource & 2x2 Tile Composer
# ===========================================================================


def test_unknown_input_adapter_and_duplicate_registration() -> None:
    from robo_vlm_qc.io import get_adapter, list_adapters, register_adapter
    from robo_vlm_qc.io.video_dir import VideoDirAdapter

    assert set(list_adapters()) == {"h5_raw_v3", "lerobot", "video_dir"}
    with pytest.raises(KeyError):
        get_adapter("nonexistent_adapter_xyz")

    with pytest.raises(ValueError, match="already registered"):
        register_adapter(VideoDirAdapter(), override=False)


def test_media_source_clamp_negative_and_beyond_duration(tmp_path: Path) -> None:
    from robo_vlm_qc.io import get_adapter

    vdir = h.make_synthetic_video_dir(tmp_path / "vids", multi_cam=False, duration_s=1.0)
    ad = get_adapter("video_dir")
    handles = ad.discover(vdir)
    assert len(handles) == 1
    with ad.open(handles[0]) as view:
        f0 = view.media.read_frame_rgb("main", 0.0)
        f_neg = view.media.read_frame_rgb("main", -10.0)
        f_end = view.media.read_frame_rgb("main", view.duration_s)
        f_far = view.media.read_frame_rgb("main", view.duration_s + 500.0)
        assert f_neg.shape == f0.shape and f_neg.dtype == np.uint8
        assert np.array_equal(f_neg, f0)
        assert np.array_equal(f_far, f_end)
        with pytest.raises(KeyError):
            view.media.read_frame_rgb("missing_cam_xyz", 0.0)


def test_read_frames_batch_empty_list(tmp_path: Path) -> None:
    from robo_vlm_qc.io import get_adapter
    from robo_vlm_qc.io.media import SampledMediaReader

    vdir = h.make_synthetic_video_dir(tmp_path / "vids", multi_cam=False, duration_s=1.0)
    ad = get_adapter("video_dir")
    handle = ad.discover(vdir)[0]
    with ad.open(handle) as view:
        assert view.media.read_frames_batch("main", []) == []
        sr = SampledMediaReader()
        assert sr.read_at_timestamps(view.media, ["main"], []) == {"main": []}
        assert sr.read_at_timestamps(view.media, [], [0.0]) == {}


def test_tile_composer_non_16_9_tiny_single_cam_frames(tmp_path: Path) -> None:
    from robo_vlm_qc.io import get_adapter
    from robo_vlm_qc.io.media import compose_2x2_tile

    # Tiny portrait non-16:9 video (100x150)
    ep_dir = tmp_path / "portrait_ds" / "ep_0000"
    ep_dir.mkdir(parents=True, exist_ok=True)
    frames = [h._make_frame_bgr(100, 150, i, 12, "TINY") for i in range(12)]
    h._write_mp4_frames(ep_dir / "main.mp4", frames, fps=30.0)

    ad = get_adapter("video_dir")
    handle = ad.discover(tmp_path / "portrait_ds")[0]
    with ad.open(handle) as view:
        tile = compose_2x2_tile(
            view.media,
            view.camera_roles,
            t_s=0.2,
            cell_w=64,
            cell_h=48,
            include_header=True,
            header_h=20,
        )
        assert tile.shape == (48 + 20, 128, 3)
        assert tile.dtype == np.uint8

        tile_no_hdr = compose_2x2_tile(
            view.media,
            view.camera_roles,
            t_s=0.2,
            cell_w=64,
            cell_h=48,
            include_header=False,
        )
        assert tile_no_hdr.shape == (48, 128, 3)


def test_tile_composer_missing_camera_and_corrupt_mp4(tmp_path: Path) -> None:
    from robo_vlm_qc.io import get_adapter
    from robo_vlm_qc.io.media import compose_2x2_tile, probe_video_cv2

    # Missing right_wrist role in wrist_2x2 mode still produces valid black quadrant
    vdir = h.make_synthetic_video_dir(tmp_path / "vids", multi_cam=False, duration_s=1.0)
    ad = get_adapter("video_dir")
    handle = ad.discover(vdir)[0]
    with ad.open(handle) as view:
        tile = compose_2x2_tile(
            view.media,
            {"main": "main", "left_wrist": "main"},
            t_s=0.1,
            cell_w=160,
            cell_h=90,
            include_header=False,
        )
        assert tile.shape == (180, 320, 3)
        # Top-right quadrant (R-WRIST) should be black except for panel tag in top-right corner
        tr_body = tile[30:90, 160:320]
        assert int(tr_body.max()) == 0

        with pytest.raises(KeyError):
            compose_2x2_tile(
                view.media,
                {"main": "main", "left_wrist": "missing_lw"},
                t_s=0.1,
            )

    # Corrupt MP4 raises ValueError cleanly
    corrupt_mp4 = tmp_path / "corrupt.mp4"
    corrupt_mp4.write_bytes(b"not a real mp4 file header 12345")
    with pytest.raises(ValueError):
        probe_video_cv2(corrupt_mp4)


def test_auto_detect_nonexistent_path_and_invalid_id_part(tmp_path: Path) -> None:
    from robo_vlm_qc.io import auto_detect
    from robo_vlm_qc.io.base import validate_id_part

    assert auto_detect(tmp_path / "does_not_exist_at_all") == []
    for bad in ("", ".", "..", "a/b", "a\\b", "foo..bar"):
        with pytest.raises(ValueError):
            validate_id_part(bad, "dataset")


# ===========================================================================
# 3 Input Adapters Edge & Negative Cases
# ===========================================================================


def test_h5_raw_v3_long_tactile_dropout_marked_invalid(tmp_path: Path) -> None:
    from robo_vlm_qc.io import get_adapter

    ad = get_adapter("h5_raw_v3")
    # Short dropout (<=70ms) repaired cleanly
    short_h5 = h.make_synthetic_h5_raw_v3(
        tmp_path / "short",
        duration_s=4.0,
        with_tactile_dropout=True,
        long_tactile_dropout=False,
    )
    with ad.open(ad.discover(short_h5)[0]) as view_short:
        assert view_short.kinematics is not None
        assert bool(view_short.kinematics.sub_valid["force"].all()) is True

    # Long dropout (>=0.7s) marked invalid rather than silently bridged
    long_h5 = h.make_synthetic_h5_raw_v3(
        tmp_path / "long",
        duration_s=4.0,
        with_tactile_dropout=False,
        long_tactile_dropout=True,
    )
    with ad.open(ad.discover(long_h5)[0]) as view_long:
        assert view_long.kinematics is not None
        fmask = view_long.kinematics.sub_valid["force"]
        assert not bool(fmask.all())
        assert int(np.count_nonzero(~fmask)) >= 15


def test_h5_raw_v3_missing_head_stereo_fallback(tmp_path: Path) -> None:
    from robo_vlm_qc.io import get_adapter
    from robo_vlm_qc.io.media import compose_2x2_tile, select_tile_mode

    h5_path = h.make_synthetic_h5_raw_v3(
        tmp_path / "no_stereo", duration_s=2.0, with_sbs_stereo=False
    )
    ad = get_adapter("h5_raw_v3")
    with ad.open(ad.discover(h5_path)[0]) as view:
        cam_names = {c.name for c in view.meta.cameras}
        assert "head_rgbd" in cam_names
        assert "torso_rgbd" in cam_names
        assert "head_stereo_left" not in cam_names
        assert select_tile_mode(view.camera_roles) == "torso_head_2x2"
        tile = compose_2x2_tile(view.media, view.camera_roles, 0.5)
        assert tile.ndim == 3 and tile.shape[2] == 3


def test_h5_raw_v3_non_overlapping_async_streams_error(tmp_path: Path) -> None:
    from robo_vlm_qc.io import get_adapter

    h5_path = h.make_synthetic_h5_raw_v3(
        tmp_path / "disjoint", duration_s=2.0, with_sbs_stereo=False
    )
    with h5py.File(h5_path, "r+") as f:
        f["/robot/timestamps"][:] += 5000.0

    ad = get_adapter("h5_raw_v3")
    handle = ad.discover(h5_path)[0]
    with pytest.raises(ValueError, match="do not overlap"):
        ad.open(handle)


def test_lerobot_missing_info_json_and_corrupt_parquet_error(tmp_path: Path) -> None:
    from robo_vlm_qc.io import get_adapter

    ds_dir = h.make_synthetic_lerobot(tmp_path / "lr_ds", version="v3", n_episodes=1)
    ad = get_adapter("lerobot")

    # Missing info.json -> cannot read / discovers nothing
    info_path = ds_dir / "meta" / "info.json"
    info_bak = info_path.read_text(encoding="utf-8")
    info_path.unlink()
    assert ad.can_read(ds_dir) is False
    assert ad.discover(ds_dir) == []

    # Restore info.json & corrupt parquet -> open raises cleanly
    info_path.write_text(info_bak, encoding="utf-8")
    pq_path = ds_dir / "data" / "chunk-000" / "file-000.parquet"
    pq_path.write_bytes(b"corrupt parquet bytes!")
    handles = ad.discover(ds_dir)
    assert len(handles) == 1
    with pytest.raises(Exception):
        ad.open(handles[0])


def test_lerobot_missing_video_file_error(tmp_path: Path) -> None:
    from robo_vlm_qc.io import get_adapter

    ds_dir = h.make_synthetic_lerobot(tmp_path / "lr_v2", version="v2", n_episodes=1)
    vid = ds_dir / "videos" / "chunk-000" / "observation.images.main" / "episode_000000.mp4"
    vid.unlink()

    ad = get_adapter("lerobot")
    handles = ad.discover(ds_dir)
    assert len(handles) == 1
    with pytest.raises(FileNotFoundError):
        ad.open(handles[0])


def test_video_dir_empty_dir_and_zero_byte_mp4_error(tmp_path: Path) -> None:
    from robo_vlm_qc.io import get_adapter

    ad = get_adapter("video_dir")
    empty = tmp_path / "empty_videos"
    empty.mkdir()
    assert ad.discover(empty) == []

    bad_dir = tmp_path / "zero_ds" / "ep_0000"
    bad_dir.mkdir(parents=True)
    (bad_dir / "main.mp4").write_bytes(b"")
    handles = ad.discover(tmp_path / "zero_ds")
    assert len(handles) == 1
    with pytest.raises(ValueError):
        ad.open(handles[0])


def test_video_dir_mismatched_fps_uses_min_duration(tmp_path: Path) -> None:
    from robo_vlm_qc.io import get_adapter

    ep_dir = tmp_path / "mismatch_ds" / "ep_0000"
    ep_dir.mkdir(parents=True)
    h._write_mp4_frames(
        ep_dir / "main.mp4",
        [h._make_frame_bgr(320, 240, i, 60, "MAIN") for i in range(60)],
        fps=30.0,  # 2.0s
    )
    h._write_mp4_frames(
        ep_dir / "left_wrist.mp4",
        [h._make_frame_bgr(320, 240, i, 15, "LW") for i in range(15)],
        fps=15.0,  # 1.0s
    )

    ad = get_adapter("video_dir")
    handles = ad.discover(tmp_path / "mismatch_ds")
    assert len(handles) == 1
    with ad.open(handles[0]) as view:
        assert view.meta.duration_s == pytest.approx(1.0, abs=0.05)
        f_main = view.media.read_frame_rgb("main", 0.5)
        f_lw = view.media.read_frame_rgb("left_wrist", 0.5)
        assert f_main.shape == (240, 320, 3)
        assert f_lw.shape == (240, 320, 3)


# ===========================================================================
# SQLite WAL, Claim Leases, .history Backups & gt_frozen
# ===========================================================================


def _setup_vlm_ready_repo(ws: Path) -> tuple[Any, str, Any]:
    from robo_vlm_qc.models.episode import CameraSpec, EpisodeMeta, KinematicsSpec
    from robo_vlm_qc.store.repository import Repository

    repo = Repository(ws)
    ep_id = "ds_store/ep_0000"
    meta = EpisodeMeta(
        episode_id=ep_id,
        dataset="ds_store",
        stem="ep_0000",
        task_id="_default",
        adapter_type="video_dir",
        source_uri=str(ws / "ep_0000.mp4"),
        source_mtime=10.0,
        episode_fingerprint="fp_store",
        duration_s=4.0,
        fps=30.0,
        frame_count=120,
        cameras=[CameraSpec(name="main", role="main", width=640, height=480, fps=30.0)],
        camera_roles={"main": "main"},
        kinematics=KinematicsSpec(has_kinematics=False),
    )
    repo.ingest_episode(meta)
    run = _make_sample_vlm_run(ep_id, run_id="run_store")
    repo.save_vlm_run_artifact(run)
    return repo, ep_id, run


def test_claim_lease_conflict_second_reviewer_denied(e2e_workspace: Path) -> None:
    from robo_vlm_qc.models.review import ReviewWritePayload, WatchCoverage
    from robo_vlm_qc.store.repository import ClaimConflictError

    repo, ep_id, run = _setup_vlm_ready_repo(e2e_workspace)
    try:
        repo.claim_episode(ep_id, "alice", ttl_s=120.0, now=1000.0)
        with pytest.raises(ClaimConflictError):
            repo.claim_episode(ep_id, "bob", ttl_s=120.0, now=1010.0)
        with pytest.raises(ClaimConflictError):
            repo.renew_claim(ep_id, "bob", ttl_s=120.0, now=1010.0)
        with pytest.raises(ClaimConflictError):
            repo.save_review(
                ep_id,
                "bob",
                "reviewer",
                ReviewWritePayload(
                    base_run_id=run.run_id,
                    action="save_draft",
                    qc_segments=run.qc_segments,
                    success=run.success,
                    macro_stages=run.macro_stages,
                    watch_coverage=WatchCoverage(
                        duration_s=4.0, watched_s=4.0, coverage_ratio=1.0, confirmed=True
                    ),
                ),
                now=1010.0,
            )
    finally:
        repo.close()


def test_expired_lease_takeover_allowed(e2e_workspace: Path) -> None:
    from robo_vlm_qc.models.review import EpisodeStatus

    repo, ep_id, _ = _setup_vlm_ready_repo(e2e_workspace)
    try:
        repo.claim_episode(ep_id, "alice", ttl_s=60.0, now=1000.0)
        # After expiry at 1060.0, bob can take over
        claimed = repo.claim_episode(ep_id, "bob", ttl_s=60.0, now=1065.0)
        assert claimed["claimed_by"] == "bob"

        # After bob's lease expires at 1125.0, reaper resets to vlm_ready
        reaped = repo.reap_expired_claims(now=1130.0)
        assert reaped == 1
        row = repo.get_episode_row(ep_id)
        assert row["status"] == EpisodeStatus.VLM_READY.value
        assert row["claimed_by"] is None
    finally:
        repo.close()


def test_history_preserves_multiple_timestamped_backups(e2e_workspace: Path) -> None:
    from robo_vlm_qc.store.artifacts import ArtifactStore

    store = ArtifactStore(e2e_workspace)
    store.write_config_override("labels.yaml", "version: 1\n", reason="init")
    time.sleep(0.005)
    store.write_config_override("labels.yaml", "version: 2\n", reason="edit_1")
    time.sleep(0.005)
    store.write_config_override("labels.yaml", "version: 3\n", reason="edit_2")

    snaps = sorted((e2e_workspace / ".history" / "configs_override" / "labels").glob("*.json"))
    assert len(snaps) == 2
    contents = [json.loads(p.read_text(encoding="utf-8"))["content"] for p in snaps]
    assert "version: 1\n" in contents
    assert "version: 2\n" in contents


def test_gt_frozen_blocks_overwrite_and_review_mutation(e2e_workspace: Path) -> None:
    from robo_vlm_qc.models.review import EpisodeStatus, ReviewWritePayload, WatchCoverage
    from robo_vlm_qc.store.repository import GtFrozenError

    repo, ep_id, run = _setup_vlm_ready_repo(e2e_workspace)
    try:
        payload = ReviewWritePayload(
            base_run_id=run.run_id,
            action="freeze_gt",
            qc_segments=run.qc_segments,
            success=run.success,
            macro_stages=run.macro_stages,
            watch_coverage=WatchCoverage(
                duration_s=4.0, watched_s=4.0, coverage_ratio=1.0, confirmed=True
            ),
        )
        repo.save_review(ep_id, "lead_user", "lead", payload, now=1000.0)

        with pytest.raises(GtFrozenError):
            repo.transition_status(ep_id, EpisodeStatus.APPROVED)
        with pytest.raises(GtFrozenError):
            repo.claim_episode(ep_id, "alice")
        with pytest.raises(GtFrozenError):
            repo.save_review(
                ep_id,
                "lead_user",
                "lead",
                payload.model_copy(update={"action": "save_draft"}),
            )
        with pytest.raises(GtFrozenError):
            repo.save_vlm_run_artifact(_make_sample_vlm_run(ep_id, run_id="run_overwrite"))

        meta = repo.get_episode_meta(ep_id)
        with pytest.raises(GtFrozenError):
            repo.ingest_episode(meta.model_copy(update={"source_mtime": 999.0}))

        # Eval sandbox runs ARE allowed on gt_frozen without altering active_run_id
        sandbox_run = _make_sample_vlm_run(ep_id, run_id="run_sandbox").model_copy(
            update={"is_eval_sandbox": True}
        )
        repo.save_vlm_run_artifact(sandbox_run)
        assert repo.get_episode_row(ep_id)["active_run_id"] == run.run_id
    finally:
        repo.close()


def test_missing_episode_id_raises_clean_error(e2e_workspace: Path) -> None:
    from robo_vlm_qc.store.repository import NotFoundError, Repository

    repo = Repository(e2e_workspace)
    try:
        with pytest.raises(NotFoundError):
            repo.get_episode_row("no_ds/no_ep")
        with pytest.raises(NotFoundError):
            repo.get_episode_meta("no_ds/no_ep")
        with pytest.raises(NotFoundError):
            repo.resolve_episode("no_ds/no_ep")
        with pytest.raises(NotFoundError):
            repo.get_vlm_run("missing_run")
        with pytest.raises(NotFoundError):
            repo.get_review("missing_rev")
    finally:
        repo.close()


def test_concurrent_wal_read_write_safety(e2e_workspace: Path) -> None:
    from robo_vlm_qc.models.episode import CameraSpec, EpisodeMeta, KinematicsSpec
    from robo_vlm_qc.store.repository import Repository

    repo = Repository(e2e_workspace)
    errors: list[BaseException] = []

    def _writer(tid: int) -> None:
        try:
            for j in range(10):
                ep_id = f"ds_wal/ep_{tid}_{j}"
                meta = EpisodeMeta(
                    episode_id=ep_id,
                    dataset="ds_wal",
                    stem=f"ep_{tid}_{j}",
                    task_id="_default",
                    adapter_type="video_dir",
                    source_uri=str(e2e_workspace / f"{tid}_{j}.mp4"),
                    source_mtime=1.0,
                    episode_fingerprint=f"fp_{tid}_{j}",
                    duration_s=2.0,
                    fps=30.0,
                    frame_count=60,
                    cameras=[CameraSpec(name="main", role="main", width=640, height=480, fps=30.0)],
                    camera_roles={"main": "main"},
                    kinematics=KinematicsSpec(has_kinematics=False),
                )
                repo.ingest_episode(meta)
        except BaseException as exc:
            errors.append(exc)

    def _reader() -> None:
        try:
            for _ in range(25):
                repo.list_episodes(limit=50)
        except BaseException as exc:
            errors.append(exc)

    try:
        threads = [
            threading.Thread(target=_writer, args=(0,)),
            threading.Thread(target=_writer, args=(1,)),
            threading.Thread(target=_reader),
            threading.Thread(target=_reader),
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=20.0)
        assert not errors, f"Concurrent WAL errors: {errors}"
        assert len(repo.list_episodes(dataset="ds_wal", limit=50)) == 20
    finally:
        repo.close()


def test_immutable_artifact_refuses_overwrite(e2e_workspace: Path) -> None:
    from robo_vlm_qc.store.artifacts import ArtifactStore

    store = ArtifactStore(e2e_workspace)
    rel = "artifacts/vlm_runs/ds/ep/r1.json"
    store.write_immutable_json(rel, {"v": 1})
    with pytest.raises(FileExistsError):
        store.write_immutable_json(rel, {"v": 2})


# ===========================================================================
# ingest CLI Boundary & Negative Cases
# ===========================================================================


def test_ingest_cli_empty_directory_nonzero_exit(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    empty_dir = tmp_path / "empty_src"
    empty_dir.mkdir()
    proc = h.run_cli(
        [
            "-w",
            str(e2e_workspace),
            "-c",
            str(e2e_config_dir),
            "ingest",
            "--data",
            str(empty_dir),
            "--json",
        ]
    )
    assert proc.returncode != 0
    report = json.loads(proc.stdout)
    assert report["ok"] is False
    assert report["discovered"] == 0


def test_ingest_cli_nonexistent_source_path_nonzero_exit(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    proc = h.run_cli(
        [
            "-w",
            str(e2e_workspace),
            "-c",
            str(e2e_config_dir),
            "ingest",
            "--data",
            str(tmp_path / "no_such_dir"),
        ]
    )
    assert proc.returncode != 0


def test_ingest_cli_unknown_adapter_rejected(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    vdir = h.make_synthetic_video_dir(tmp_path / "vids", multi_cam=False)
    proc = h.run_cli(
        [
            "-w",
            str(e2e_workspace),
            "-c",
            str(e2e_config_dir),
            "ingest",
            "--data",
            str(vdir),
            "--adapter",
            "bogus_adapter",
        ]
    )
    assert proc.returncode != 0


def test_ingest_cli_corrupt_file_reported_without_crashing_batch(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    batch_root = tmp_path / "mixed_batch"
    h.make_synthetic_video_dir(batch_root / "good_ds", multi_cam=False, n_episodes=1)
    bad_ep = batch_root / "bad_ds" / "ep_corrupt"
    bad_ep.mkdir(parents=True)
    (bad_ep / "main.mp4").write_bytes(b"corrupt")

    proc = h.run_cli(
        [
            "-w",
            str(e2e_workspace),
            "-c",
            str(e2e_config_dir),
            "ingest",
            "--data",
            str(batch_root),
            "--json",
        ]
    )
    assert proc.returncode != 0
    report = json.loads(proc.stdout)
    assert report["discovered"] == 2
    assert report["inserted"] == 1
    assert report["errors_count"] == 1


def test_ingest_cli_duplicate_dataset_stem_idempotency(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    vdir = h.make_synthetic_video_dir(tmp_path / "idem_ds", multi_cam=False, n_episodes=1)
    args = [
        "-w",
        str(e2e_workspace),
        "-c",
        str(e2e_config_dir),
        "ingest",
        "--data",
        str(vdir),
        "--json",
    ]
    p1 = h.run_cli(args)
    assert p1.returncode == 0
    r1 = json.loads(p1.stdout)
    assert r1["inserted"] == 1 and r1["skipped"] == 0

    p2 = h.run_cli(args)
    assert p2.returncode == 0
    r2 = json.loads(p2.stdout)
    assert r2["inserted"] == 0 and r2["skipped"] == 1

    with sqlite3.connect(str(e2e_workspace / "robo_vlm_qc.db")) as conn:
        cnt = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        assert cnt == 1


# ===========================================================================
# Pure-Visual Taxonomy & Label Registry Boundaries
# ===========================================================================


def test_duplicate_label_ids_rejected() -> None:
    from robo_vlm_qc.models.taxonomy import CollectorUiHint, LabelRegistry, VisualLabelDef, VlmSemantics

    lbl1 = VisualLabelDef(
        id="dup_lbl",
        name="Label A",
        category="idle",
        default_training_role="badcase",
        default_splice_policy="cut_clean",
        collector_ui=CollectorUiHint(title="A"),
        vlm_semantics=VlmSemantics(definition="def A"),
    )
    lbl2 = lbl1.model_copy(update={"name": "Label B"})
    with pytest.raises(ValidationError, match="Duplicate label ids"):
        LabelRegistry(labels=[lbl1, lbl2])


def test_removing_active_vs_archived_label_semantics(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.models.taxonomy import LabelRegistry

    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    raw = cfg.labels.model_dump(mode="json")
    # Archiving/retiring a label preserves registry integrity and lineage fields
    raw["labels"][1]["status"] = "retired"
    raw["labels"][1]["merged_into"] = "boundary_idle"
    reg = LabelRegistry.model_validate(raw)
    retired = reg.get(raw["labels"][1]["id"])
    assert retired is not None
    assert retired.status == "retired"
    assert retired.merged_into == "boundary_idle"
    active_ids = [lbl.id for lbl in reg.labels if lbl.status == "active"]
    assert retired.id not in active_ids


def test_unknown_category_rejected() -> None:
    from robo_vlm_qc.models.taxonomy import CollectorUiHint, VisualLabelDef, VlmSemantics

    with pytest.raises(ValidationError):
        VisualLabelDef(
            id="bad_cat_lbl",
            name="Invalid Category",
            category="sensor_torque_rule",  # type: ignore[arg-type]
            default_training_role="badcase",
            default_splice_policy="mask_loss",
            collector_ui=CollectorUiHint(title="x"),
            vlm_semantics=VlmSemantics(definition="x"),
        )


def test_boundary_idle_vs_mid_idle_freeze_category_invariants(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    from robo_vlm_qc.config import ConfigManager

    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    assert len(cfg.labels.labels) == 11
    h.assert_no_legacy_fields(cfg.labels)

    b_idle = cfg.labels.get("boundary_idle")
    m_idle = cfg.labels.get("mid_idle_freeze")
    assert b_idle is not None and m_idle is not None
    assert b_idle.category == "idle"
    assert m_idle.category == "idle"
    assert b_idle.default_training_role == "badcase"
    assert m_idle.default_training_role == "badcase"
    assert b_idle.default_splice_policy == "cut_clean"
    assert m_idle.default_splice_policy == "cut_clean"


def test_disabled_label_excluded_from_active_prompt_taxonomy(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    from robo_vlm_qc.config import ConfigManager

    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    raw_labels = cfg.labels.model_dump(mode="json")
    for lbl in raw_labels["labels"]:
        if lbl["id"] == "hesitation_oscillation":
            lbl["status"] = "retired"
    (e2e_workspace / "configs_override" / "labels.yaml").write_text(
        yaml.safe_dump(raw_labels, allow_unicode=True), encoding="utf-8"
    )

    ep_id = _ingest_one_video_ep(e2e_workspace, e2e_config_dir, tmp_path / "data")
    proc = _invoke_dag_run(e2e_workspace, e2e_config_dir, [ep_id], workers=1)
    assert proc.returncode == 0, f"run-vlm failed: {proc.stderr}\n{proc.stdout}"

    s3_reqs = [r for r in mock_vlm_server.requests if r["stage"] == "S3"]
    assert s3_reqs, "Expected S3 request to MockVlmServer"
    s3_text = json.dumps(s3_reqs[0]["messages"], ensure_ascii=False)
    assert "boundary_idle" in s3_text
    assert "hesitation_oscillation" not in s3_text


# ===========================================================================
# VlmClient Retries, Budgets & Content-Addressed StageCache
# ===========================================================================


def test_vlm_client_exhausts_retries_on_persistent_500(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    ep_id = _ingest_one_video_ep(e2e_workspace, e2e_config_dir, tmp_path / "data")
    # Default retries=3 => 4 attempts fail with persistent 500
    mock_vlm_server.fail_next(status_code=500, count=20)

    proc = _invoke_dag_run(e2e_workspace, e2e_config_dir, [ep_id], workers=1)
    assert proc.returncode != 0
    assert len(mock_vlm_server.requests) >= 3

    from robo_vlm_qc.store.repository import Repository

    repo = Repository(e2e_workspace)
    try:
        row = repo.get_episode_row(ep_id)
        assert row["active_run_id"] is None
    finally:
        repo.close()


def test_vlm_client_429_retry_succeeds_on_attempt_3(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    ep_id = _ingest_one_video_ep(e2e_workspace, e2e_config_dir, tmp_path / "data")
    mock_vlm_server.fail_next(status_code=429, count=2, stage="S1")

    proc = _invoke_dag_run(e2e_workspace, e2e_config_dir, [ep_id], workers=1)
    assert proc.returncode == 0, f"run-vlm failed after 429 retries: {proc.stderr}"
    s1_reqs = [r for r in mock_vlm_server.requests if r["stage"] == "S1"]
    assert len(s1_reqs) >= 3


def test_vlm_client_malformed_json_response_handling(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    ep_id = _ingest_one_video_ep(e2e_workspace, e2e_config_dir, tmp_path / "data")
    mock_vlm_server.fail_with_malformed_json(count=20, stage="S2")

    proc = _invoke_dag_run(e2e_workspace, e2e_config_dir, [ep_id], workers=1)
    assert proc.returncode != 0

    from robo_vlm_qc.store.repository import Repository

    repo = Repository(e2e_workspace)
    try:
        row = repo.get_episode_row(ep_id)
        assert row["active_run_id"] is None
    finally:
        repo.close()


def test_vlm_budget_exceeded_stops_cleanly(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    h.write_workspace_config(
        e2e_workspace,
        budget={
            "tokens_per_second": 1,
            "min_episode_tokens": 100,
            "max_tokens_per_episode": 100,
            "unlimited": False,
        },
    )
    ep_id = _ingest_one_video_ep(e2e_workspace, e2e_config_dir, tmp_path / "data")
    proc = _invoke_dag_run(e2e_workspace, e2e_config_dir, [ep_id], workers=1)
    # MockVlmServer bills 180 tokens per request, so 100-token budget stops cleanly
    assert len(mock_vlm_server.requests) <= 2
    assert "Traceback" not in proc.stderr

    from robo_vlm_qc.store.repository import Repository

    repo = Repository(e2e_workspace)
    try:
        row = repo.get_episode_row(ep_id)
        assert row["status"] != "vlm_running"
    finally:
        repo.close()


def test_stage_cache_invalidates_s3_on_few_shot_keeps_s1_s2_s5(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    from robo_vlm_qc.models.taxonomy import FewShotItem
    from robo_vlm_qc.store.repository import Repository

    ep_id = _ingest_one_video_ep(e2e_workspace, e2e_config_dir, tmp_path / "data")
    p1 = _invoke_dag_run(e2e_workspace, e2e_config_dir, [ep_id], workers=1)
    assert p1.returncode == 0, p1.stderr
    stages_run1 = {r["stage"] for r in mock_vlm_server.requests}
    assert {"S1", "S2", "S3", "S5"}.issubset(stages_run1)

    # Add an enabled FewShotItem directly in SQLite
    repo = Repository(e2e_workspace)
    try:
        item = FewShotItem(
            example_id="fs_cache_1",
            scope="global",
            error_type="fp_overflag",
            episode_id=ep_id,
            t_start=0.2,
            t_end=0.8,
            vlm_wrong_json={"label_id": "mid_idle_freeze"},
            human_right_json={"splice_policy": "keep_positive"},
            rationale="Normal slow progress",
            enabled=True,
            created_at=time.time(),
        )
        with repo.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO few_shot_examples (
                    example_id, scope, task_id, label_id, error_type, episode_id,
                    t_start, t_end, tile_image_path, vlm_wrong_json, human_right_json,
                    rationale, enabled, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.example_id,
                    item.scope,
                    item.task_id,
                    item.label_id,
                    item.error_type,
                    item.episode_id,
                    item.t_start,
                    item.t_end,
                    item.tile_image_path,
                    json.dumps(item.vlm_wrong_json),
                    json.dumps(item.human_right_json),
                    item.rationale,
                    1,
                    item.created_at,
                ),
            )
    finally:
        repo.close()

    mock_vlm_server.requests.clear()
    p2 = _invoke_dag_run(e2e_workspace, e2e_config_dir, [ep_id], workers=1)
    assert p2.returncode == 0, p2.stderr
    stages_run2 = {r["stage"] for r in mock_vlm_server.requests}
    # S1, S2, S5 MUST hit cache (0 requests), while S3 re-runs
    assert "S1" not in stages_run2
    assert "S2" not in stages_run2
    assert "S5" not in stages_run2
    assert "S5_CONTROL" not in stages_run2
    assert "S3" in stages_run2


def test_stage_cache_corrupted_json_ignored_and_recomputed(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    ep_id = _ingest_one_video_ep(e2e_workspace, e2e_config_dir, tmp_path / "data")
    p1 = _invoke_dag_run(e2e_workspace, e2e_config_dir, [ep_id], workers=1)
    assert p1.returncode == 0, p1.stderr

    cache_files = sorted((e2e_workspace / "stage_cache").rglob("*.json"))
    assert cache_files, "Expected populated stage_cache/*.json files"
    # Corrupt one cache file
    corrupted = cache_files[0]
    corrupted.write_text("{corrupt json !!!", encoding="utf-8")

    mock_vlm_server.requests.clear()
    p2 = _invoke_dag_run(e2e_workspace, e2e_config_dir, [ep_id], workers=1)
    assert p2.returncode == 0, p2.stderr
    assert len(mock_vlm_server.requests) >= 1
    # Recomputed cache file is now valid JSON
    parsed = json.loads(corrupted.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)


# ===========================================================================
# PromptBuilder, S1 Grounding & S2 Macro Stages Boundaries
# ===========================================================================


def test_s1_empty_detections_handled(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    mock_vlm_server.set_stage_response("S1", {"objects": {}})
    ep_id = _ingest_one_video_ep(e2e_workspace, e2e_config_dir, tmp_path / "data")
    proc = _invoke_dag_run(e2e_workspace, e2e_config_dir, [ep_id], workers=1)
    assert proc.returncode == 0, proc.stderr

    from robo_vlm_qc.store.repository import Repository

    repo = Repository(e2e_workspace)
    try:
        resolved = repo.resolve_episode(ep_id)
        assert resolved.grounding.object_tracks_2d == {}
    finally:
        repo.close()


def test_s1_static_container_jitter_stabilized_across_outlier_frame(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    # pick_and_place defines target_bin with is_static: true
    base_boxes = [
        ([400, 200, 600, 500], "ok"),
        ([404, 202, 602, 498], "ok"),
        ([420, 220, 620, 520], "jitter"),  # jitter outlier
        ([401, 201, 601, 499], "ok"),
        ([399, 199, 599, 501], "ok"),
    ]
    call_counter = {"n": 0}

    def _s1_jitter(_req: dict[str, Any]) -> dict[str, Any]:
        idx = call_counter["n"]
        call_counter["n"] += 1
        bbox, state = (
            base_boxes[idx]
            if idx < len(base_boxes)
            else ([400, 200, 600, 500], "ok")
        )
        return {
            "objects": {
                "target_bin": {
                    "visible": True,
                    "bbox": bbox,
                    "state": state,
                    "status_kind": "neutral",
                }
            }
        }

    mock_vlm_server.set_stage_response("S1", _s1_jitter)
    ep_id = _ingest_one_h5_ep(e2e_workspace, e2e_config_dir, tmp_path / "h5", duration_s=11.0)
    proc = _invoke_dag_run(e2e_workspace, e2e_config_dir, [ep_id], workers=1)
    assert proc.returncode == 0, proc.stderr

    from robo_vlm_qc.store.repository import Repository

    repo = Repository(e2e_workspace)
    try:
        resolved = repo.resolve_episode(ep_id)
        track = resolved.grounding.object_tracks_2d.get("target_bin")
        assert track is not None
        assert track.is_static_container is True
        boxes = [kf.bbox_norm_1000 for kf in track.keyframes if kf.visible]
        assert len(boxes) >= 5
        # After median stabilization, all visible boxes converge within tight jitter tolerance
        xs = [b[0] for b in boxes]
        assert max(xs) - min(xs) <= 5
    finally:
        repo.close()


def test_s2_single_frame_or_very_short_episode_macro_stages(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    # Very short 2-frame (~0.07s) video episode
    ep_dir = tmp_path / "short_ds" / "ep_short"
    ep_dir.mkdir(parents=True)
    h._write_mp4_frames(
        ep_dir / "main.mp4",
        [h._make_frame_bgr(320, 240, i, 2, "SHORT") for i in range(2)],
        fps=30.0,
    )
    ingest_proc = h.run_cli(
        [
            "-w",
            str(e2e_workspace),
            "-c",
            str(e2e_config_dir),
            "ingest",
            "--data",
            str(tmp_path / "short_ds"),
            "--json",
        ]
    )
    assert ingest_proc.returncode == 0
    ep_id = json.loads(ingest_proc.stdout)["episodes"][0]["episode_id"]

    mock_vlm_server.set_stage_response(
        "S2",
        {
            "macro_stages": [
                {
                    "stage_id": "S1",
                    "t_start": 0.0,
                    "t_end": 0.067,
                    "label": "Short operation",
                    "label_en": "short_op",
                }
            ],
            "arm_timelines": {
                "left": [{"t_start": 0.0, "t_end": 0.067, "state": "work"}],
                "right": [{"t_start": 0.0, "t_end": 0.067, "state": "idle"}],
            },
        },
    )
    mock_vlm_server.set_stage_response("S3", {"segments": []})
    proc = _invoke_dag_run(e2e_workspace, e2e_config_dir, [ep_id], workers=1)
    assert proc.returncode == 0, proc.stderr


def test_s2_mismatched_lr_timeline_clamped(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    ep_id = _ingest_one_video_ep(
        e2e_workspace, e2e_config_dir, tmp_path / "data", duration_s=2.0
    )
    mock_vlm_server.set_stage_response(
        "S2",
        {
            "macro_stages": [
                {
                    "stage_id": "S1",
                    "t_start": -1.0,
                    "t_end": 99.0,
                    "label": "Out-of-bounds stage",
                }
            ],
            "arm_timelines": {
                "left": [{"t_start": -0.5, "t_end": 50.0, "state": "work"}],
                "right": [{"t_start": 0.0, "t_end": 12.0, "state": "hold"}],
            },
        },
    )
    proc = _invoke_dag_run(e2e_workspace, e2e_config_dir, [ep_id], workers=1)
    assert proc.returncode == 0, proc.stderr

    from robo_vlm_qc.store.repository import Repository

    repo = Repository(e2e_workspace)
    try:
        resolved = repo.resolve_episode(ep_id)
        for st in resolved.macro_stages:
            assert st.t_start >= 0.0
            assert st.t_end <= resolved.meta.duration_s + 1e-6
        for span in resolved.arm_timelines.left + resolved.arm_timelines.right:
            assert span.t_start >= 0.0
            assert span.t_end <= resolved.meta.duration_s + 1e-6
    finally:
        repo.close()


def test_prompt_builder_truncates_few_shot_within_token_limit(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    from robo_vlm_qc.store.repository import Repository

    h.write_workspace_config(
        e2e_workspace,
        extra_overrides={"vlm": {"s3_discover": {"few_shot_top_k": 5, "few_shot_max_tokens": 120}}},
    )
    ep_id = _ingest_one_video_ep(e2e_workspace, e2e_config_dir, tmp_path / "data")
    repo = Repository(e2e_workspace)
    try:
        with repo.db.transaction() as conn:
            for i in range(5):
                conn.execute(
                    """
                    INSERT INTO few_shot_examples (
                        example_id, scope, task_id, label_id, error_type, episode_id,
                        t_start, t_end, tile_image_path, vlm_wrong_json, human_right_json,
                        rationale, enabled, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"fs_tok_{i}",
                        "global",
                        None,
                        None,
                        "fp_overflag",
                        ep_id,
                        0.1,
                        0.5,
                        None,
                        json.dumps({"label": "mid_idle_freeze"}),
                        json.dumps({"policy": "keep_positive"}),
                        f"LONG_RATIONALE_MARKER_{i}_" + ("x" * 600),
                        1,
                        float(1000 + i),
                    ),
                )
    finally:
        repo.close()

    proc = _invoke_dag_run(e2e_workspace, e2e_config_dir, [ep_id], workers=1)
    assert proc.returncode == 0, proc.stderr
    s3_reqs = [r for r in mock_vlm_server.requests if r["stage"] == "S3"]
    assert s3_reqs
    s3_text = json.dumps(s3_reqs[0]["messages"], ensure_ascii=False)
    present = sum(1 for i in range(5) if f"LONG_RATIONALE_MARKER_{i}_" in s3_text)
    assert present < 5


def test_prompt_builder_missing_camera_role_graceful_fallback(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    # Single-cam auxiliary video without wrist/front roles
    ep_dir = tmp_path / "single_ds" / "ep_only"
    ep_dir.mkdir(parents=True)
    h._write_mp4_frames(
        ep_dir / "custom_cam.mp4",
        [h._make_frame_bgr(320, 240, i, 30, "CUSTOM") for i in range(30)],
        fps=30.0,
    )
    ingest_proc = h.run_cli(
        [
            "-w",
            str(e2e_workspace),
            "-c",
            str(e2e_config_dir),
            "ingest",
            "--data",
            str(tmp_path / "single_ds"),
            "--json",
        ]
    )
    assert ingest_proc.returncode == 0
    ep_id = json.loads(ingest_proc.stdout)["episodes"][0]["episode_id"]

    proc = _invoke_dag_run(e2e_workspace, e2e_config_dir, [ep_id], workers=1)
    assert proc.returncode == 0, proc.stderr
    assert any(r["stage"] == "S2" for r in mock_vlm_server.requests)


# ===========================================================================
# S5 Control, S3/S4A/S4B & Concurrent DAG Boundaries
# ===========================================================================


def test_s5_30pct_control_mismatch_flags_low_confidence_warning(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    # Control check falsely claims success at 30% progress -> anti-FP check flags mismatch
    mock_vlm_server.set_stage_response(
        "S5_CONTROL",
        {
            "verdict": "success",
            "confidence": 0.95,
            "criteria": [
                {
                    "criterion": "Target object reaches expected location",
                    "met": True,
                    "satisfied_at_s": 0.3,
                    "evidence": "false early positive",
                }
            ],
            "summary": "false positive at 30%",
        },
    )
    ep_id = _ingest_one_video_ep(e2e_workspace, e2e_config_dir, tmp_path / "data")
    proc = _invoke_dag_run(e2e_workspace, e2e_config_dir, [ep_id], workers=1)
    assert proc.returncode == 0, proc.stderr

    from robo_vlm_qc.store.repository import Repository

    repo = Repository(e2e_workspace)
    try:
        resolved = repo.resolve_episode(ep_id)
        cc = resolved.success.control_check
        assert cc is not None
        assert cc.passed is False
        assert (
            resolved.success.confidence < 0.93
            or "control" in resolved.success.summary.lower()
            or resolved.success.verdict == "unsure"
        )
    finally:
        repo.close()


def test_s3_empty_window_no_anomalies_returns_empty(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    mock_vlm_server.set_stage_response("S3", {"segments": []})
    ep_id = _ingest_one_video_ep(e2e_workspace, e2e_config_dir, tmp_path / "data")
    proc = _invoke_dag_run(e2e_workspace, e2e_config_dir, [ep_id], workers=1)
    assert proc.returncode == 0, proc.stderr

    from robo_vlm_qc.store.repository import Repository

    repo = Repository(e2e_workspace)
    try:
        resolved = repo.resolve_episode(ep_id)
        assert resolved.qc_segments == []
        # No S4A/S4B calls needed when S3 yields zero candidates
        assert not any(r["stage"] in ("S4A", "S4B") for r in mock_vlm_server.requests)
    finally:
        repo.close()


def test_s4a_preserves_distinct_retries_separated_by_progress(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    ep_id = _ingest_one_video_ep(
        e2e_workspace, e2e_config_dir, tmp_path / "data", duration_s=4.0
    )
    seg_a = {
        "t_start": 0.5,
        "t_end": 1.1,
        "label_id": "missed_grasp_retry",
        "affected_arm": "left",
        "splice_policy": "mask_loss",
        "visual_evidence": {"initial_state": "a", "process_change": "b", "end_state": "c"},
        "confidence": 0.9,
        "unknown_note": "",
    }
    seg_b = {
        "t_start": 2.4,
        "t_end": 3.1,
        "label_id": "missed_grasp_retry",
        "affected_arm": "left",
        "splice_policy": "mask_loss",
        "visual_evidence": {"initial_state": "d", "process_change": "e", "end_state": "f"},
        "confidence": 0.9,
        "unknown_note": "",
    }
    mock_vlm_server.set_stage_response("S3", {"segments": [seg_a, seg_b]})
    mock_vlm_server.set_stage_response("S4A", {"merged_segments": [seg_a, seg_b]})

    proc = _invoke_dag_run(e2e_workspace, e2e_config_dir, [ep_id], workers=1)
    assert proc.returncode == 0, proc.stderr

    from robo_vlm_qc.store.repository import Repository

    repo = Repository(e2e_workspace)
    try:
        resolved = repo.resolve_episode(ep_id)
        retries = [s for s in resolved.qc_segments if s.label_id == "missed_grasp_retry"]
        assert len(retries) == 2
        assert retries[0].t_end < retries[1].t_start
    finally:
        repo.close()


def test_s4b_downgrades_false_positive_to_keep_positive(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    ep_id = _ingest_one_video_ep(
        e2e_workspace, e2e_config_dir, tmp_path / "data", duration_s=4.0
    )
    mock_vlm_server.set_stage_response(
        "S3",
        {
            "segments": [
                {
                    "t_start": 1.2,
                    "t_end": 2.0,
                    "label_id": "hesitation_oscillation",
                    "affected_arm": "right",
                    "splice_policy": "mask_loss",
                    "visual_evidence": {"initial_state": "a", "process_change": "b", "end_state": "c"},
                    "confidence": 0.72,
                    "unknown_note": "",
                }
            ]
        },
    )
    mock_vlm_server.set_stage_response(
        "S4B",
        {
            "keep_segment": True,
            "t_start": 1.2,
            "t_end": 2.0,
            "label_id": "precision_fine_adjust",
            "affected_arm": "right",
            "splice_policy": "keep_positive",
            "visual_evidence": {
                "initial_state": "Approaching target slot.",
                "process_change": "Monotonic slow precision alignment.",
                "end_state": "Stable alignment completed.",
            },
            "confidence": 0.92,
            "reject_reason": "",
        },
    )
    proc = _invoke_dag_run(e2e_workspace, e2e_config_dir, [ep_id], workers=1)
    assert proc.returncode == 0, proc.stderr

    from robo_vlm_qc.store.repository import Repository

    repo = Repository(e2e_workspace)
    try:
        resolved = repo.resolve_episode(ep_id)
        assert len(resolved.qc_segments) == 1
        assert resolved.qc_segments[0].splice_policy == "keep_positive"
        assert resolved.qc_segments[0].mask_side is None
    finally:
        repo.close()


def test_s4b_keep_segment_false_rejects_candidate(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    ep_id = _ingest_one_video_ep(
        e2e_workspace, e2e_config_dir, tmp_path / "data", duration_s=4.0
    )
    mock_vlm_server.set_stage_response(
        "S3",
        {
            "segments": [
                {
                    "t_start": 1.0,
                    "t_end": 1.8,
                    "label_id": "hesitation_oscillation",
                    "affected_arm": "left",
                    "splice_policy": "mask_loss",
                    "visual_evidence": {"initial_state": "a", "process_change": "b", "end_state": "c"},
                    "confidence": 0.65,
                    "unknown_note": "",
                }
            ]
        },
    )
    mock_vlm_server.set_stage_response(
        "S4B",
        {
            "keep_segment": False,
            "t_start": 1.0,
            "t_end": 1.8,
            "label_id": "hesitation_oscillation",
            "affected_arm": "left",
            "splice_policy": "keep_positive",
            "visual_evidence": {"initial_state": "a", "process_change": "normal", "end_state": "c"},
            "confidence": 0.90,
            "reject_reason": "False positive normal motion",
        },
    )
    proc = _invoke_dag_run(e2e_workspace, e2e_config_dir, [ep_id], workers=1)
    assert proc.returncode == 0, proc.stderr

    from robo_vlm_qc.store.repository import Repository

    repo = Repository(e2e_workspace)
    try:
        resolved = repo.resolve_episode(ep_id)
        assert resolved.qc_segments == []
    finally:
        repo.close()


def test_dag_handles_stage_failure_without_corrupting_db(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    ep_id = _ingest_one_video_ep(e2e_workspace, e2e_config_dir, tmp_path / "data")
    # S1 succeeds, S3 fails persistently
    mock_vlm_server.fail_next(status_code=500, count=20, stage="S3")
    proc = _invoke_dag_run(e2e_workspace, e2e_config_dir, [ep_id], workers=1)
    assert proc.returncode != 0

    from robo_vlm_qc.models.review import EpisodeStatus
    from robo_vlm_qc.store.repository import Repository

    repo = Repository(e2e_workspace)
    try:
        row = repo.get_episode_row(ep_id)
        assert row["active_run_id"] is None
        assert row["status"] in (EpisodeStatus.INGESTED.value, EpisodeStatus.VLM_QUEUED.value)
        # DB remains queryable and not stuck in vlm_running
        assert row["status"] != EpisodeStatus.VLM_RUNNING.value
    finally:
        repo.close()


def test_dag_concurrency_two_or_more_episodes(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    vdir = h.make_synthetic_video_dir(tmp_path / "multi_eps", multi_cam=False, n_episodes=3)
    ingest_proc = h.run_cli(
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
    assert ingest_proc.returncode == 0
    ep_ids = [ep["episode_id"] for ep in json.loads(ingest_proc.stdout)["episodes"]]
    assert len(ep_ids) == 3

    proc = _invoke_dag_run(e2e_workspace, e2e_config_dir, ep_ids, workers=3)
    assert proc.returncode == 0, proc.stderr

    from robo_vlm_qc.models.review import EpisodeStatus
    from robo_vlm_qc.store.repository import Repository

    repo = Repository(e2e_workspace)
    try:
        for ep_id in ep_ids:
            row = repo.get_episode_row(ep_id)
            assert row["status"] == EpisodeStatus.VLM_READY.value
            assert row["active_run_id"] is not None
    finally:
        repo.close()


# ===========================================================================
# Guardrail Exact Threshold Boundaries
# ===========================================================================


def _make_kinematics_for_gate(
    *,
    T: int = 120,
    fps: float = 30.0,
    left_dx: float = 0.0,
    left_rot_rad: float = 0.0,
    hand_d_rad: float = 0.0,
    right_dx: float = 0.0,
    torso_d_rad: float = 0.0,
    chassis_dx: float = 0.0,
    chassis_dyaw: float = 0.0,
    t_a_idx: int = 30,
    t_b_idx: int = 60,
) -> Any:
    from robo_vlm_qc.io.base import KinematicsData

    ts = np.arange(T, dtype=np.float64) / fps
    eef_pos = np.zeros((T, 2, 3), dtype=np.float64)
    eef_rot = np.zeros((T, 2, 3), dtype=np.float64)
    hand_q = np.zeros((T, 44), dtype=np.float64)
    torso_q = np.zeros((T, 4), dtype=np.float64)
    chassis_pose = np.zeros((T, 3), dtype=np.float64)

    # Hold constant at and outside [t_a_idx, t_b_idx] so 2D grid search sees exact jump
    eef_pos[t_b_idx:, 0, 0] = left_dx
    eef_pos[t_b_idx:, 1, 0] = right_dx
    eef_rot[t_b_idx:, 0, 2] = left_rot_rad
    hand_q[t_b_idx:, 0] = hand_d_rad
    torso_q[t_b_idx:, 0] = torso_d_rad
    chassis_pose[t_b_idx:, 0] = chassis_dx
    chassis_pose[t_b_idx:, 2] = chassis_dyaw

    return KinematicsData(
        timestamps_s=ts,
        eef_pos=eef_pos,
        eef_rotvec=eef_rot,
        hand_q=hand_q,
        torso_q=torso_q,
        chassis_pose=chassis_pose,
        valid_mask=np.ones(T, dtype=bool),
    )


def _make_cut_candidate(t_start: float = 1.0, t_end: float = 2.0, label_id: str = "mid_idle_freeze") -> Any:
    from robo_vlm_qc.models.annotation import QcSegment

    return QcSegment(
        seg_id="seg_gate",
        t_start=t_start,
        t_end=t_end,
        label_id=label_id,
        training_role="badcase",
        splice_policy="cut_clean",
        mask_side=None,
        affected_arm="both",
        vlm_raw_policy="cut_clean",
    )


def _load_guardrail_cfg(ws: Path, cfg_dir: Path) -> dict[str, Any]:
    from robo_vlm_qc.config import ConfigManager

    return ConfigManager(config_dir=cfg_dir, workspace_dir=ws).settings.get("guardrail", {})


def test_splice_gate_eef_pos_2cm_pass_vs_2_01cm_downgrade(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    cfg = _load_guardrail_cfg(e2e_workspace, e2e_config_dir)

    kin_pass = _make_kinematics_for_gate(left_dx=0.0199)  # <= 2.0cm
    res_pass = _invoke_guardrail(
        "robo_vlm_qc.guardrail.splice_gate",
        [_make_cut_candidate()],
        kin_pass,
        None,
        cfg,
        4.0,
    )
    assert res_pass[0].splice_policy == "cut_clean"
    assert res_pass[0].vlm_raw_policy == "cut_clean"

    kin_fail = _make_kinematics_for_gate(left_dx=0.0205)  # > 2.0cm
    res_fail = _invoke_guardrail(
        "robo_vlm_qc.guardrail.splice_gate",
        [_make_cut_candidate()],
        kin_fail,
        None,
        cfg,
        4.0,
    )
    assert res_fail[0].splice_policy == "mask_loss"
    assert res_fail[0].vlm_raw_policy == "cut_clean"
    assert res_fail[0].guardrail.downgraded is True
    assert res_fail[0].guardrail.downgrade_reason


def test_splice_gate_eef_rot_010rad_pass_vs_011rad_downgrade(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    cfg = _load_guardrail_cfg(e2e_workspace, e2e_config_dir)

    kin_pass = _make_kinematics_for_gate(left_rot_rad=0.099)
    res_pass = _invoke_guardrail(
        "robo_vlm_qc.guardrail.splice_gate",
        [_make_cut_candidate()],
        kin_pass,
        None,
        cfg,
        4.0,
    )
    assert res_pass[0].splice_policy == "cut_clean"

    kin_fail = _make_kinematics_for_gate(left_rot_rad=0.11)
    res_fail = _invoke_guardrail(
        "robo_vlm_qc.guardrail.splice_gate",
        [_make_cut_candidate()],
        kin_fail,
        None,
        cfg,
        4.0,
    )
    assert res_fail[0].splice_policy == "mask_loss"
    assert res_fail[0].guardrail.downgraded is True

    # Torso joint jump threshold (0.05 rad)
    kin_torso_pass = _make_kinematics_for_gate(torso_d_rad=0.049)
    res_torso_pass = _invoke_guardrail(
        "robo_vlm_qc.guardrail.splice_gate",
        [_make_cut_candidate()],
        kin_torso_pass,
        None,
        cfg,
        4.0,
    )
    assert res_torso_pass[0].splice_policy == "cut_clean"

    kin_torso_fail = _make_kinematics_for_gate(torso_d_rad=0.051)
    res_torso_fail = _invoke_guardrail(
        "robo_vlm_qc.guardrail.splice_gate",
        [_make_cut_candidate()],
        kin_torso_fail,
        None,
        cfg,
        4.0,
    )
    assert res_torso_fail[0].splice_policy == "mask_loss"
    assert res_torso_fail[0].guardrail.downgraded is True


def test_splice_gate_hand_q_030rad_pass_vs_031rad_downgrade(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    cfg = _load_guardrail_cfg(e2e_workspace, e2e_config_dir)

    kin_pass = _make_kinematics_for_gate(hand_d_rad=0.299)
    res_pass = _invoke_guardrail(
        "robo_vlm_qc.guardrail.splice_gate",
        [_make_cut_candidate()],
        kin_pass,
        None,
        cfg,
        4.0,
    )
    assert res_pass[0].splice_policy == "cut_clean"

    kin_fail = _make_kinematics_for_gate(hand_d_rad=0.31)
    res_fail = _invoke_guardrail(
        "robo_vlm_qc.guardrail.splice_gate",
        [_make_cut_candidate()],
        kin_fail,
        None,
        cfg,
        4.0,
    )
    assert res_fail[0].splice_policy == "mask_loss"
    assert res_fail[0].guardrail.downgraded is True

    # Chassis pose thresholds (0.02 m translation, 0.05 rad yaw)
    res_ch_pos_pass = _invoke_guardrail(
        "robo_vlm_qc.guardrail.splice_gate",
        [_make_cut_candidate()],
        _make_kinematics_for_gate(chassis_dx=0.019),
        None,
        cfg,
        4.0,
    )
    assert res_ch_pos_pass[0].splice_policy == "cut_clean"

    res_ch_pos_fail = _invoke_guardrail(
        "robo_vlm_qc.guardrail.splice_gate",
        [_make_cut_candidate()],
        _make_kinematics_for_gate(chassis_dx=0.021),
        None,
        cfg,
        4.0,
    )
    assert res_ch_pos_fail[0].splice_policy == "mask_loss"
    assert res_ch_pos_fail[0].guardrail.downgraded is True

    res_ch_yaw_pass = _invoke_guardrail(
        "robo_vlm_qc.guardrail.splice_gate",
        [_make_cut_candidate()],
        _make_kinematics_for_gate(chassis_dyaw=0.049),
        None,
        cfg,
        4.0,
    )
    assert res_ch_yaw_pass[0].splice_policy == "cut_clean"

    res_ch_yaw_fail = _invoke_guardrail(
        "robo_vlm_qc.guardrail.splice_gate",
        [_make_cut_candidate()],
        _make_kinematics_for_gate(chassis_dyaw=0.051),
        None,
        cfg,
        4.0,
    )
    assert res_ch_yaw_fail[0].splice_policy == "mask_loss"
    assert res_ch_yaw_fail[0].guardrail.downgraded is True


def test_splice_gate_other_arm_coupling_3cm_pass_vs_3_1cm_downgrade(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    from robo_vlm_qc.io.base import KinematicsData

    cfg = _load_guardrail_cfg(e2e_workspace, e2e_config_dir)

    # Left arm segment with right arm excursion inside [t_start, t_end]
    def _kin_other_arm(peak_disp: float) -> Any:
        T = 120
        ts = np.arange(T, dtype=np.float64) / 30.0
        eef_pos = np.zeros((T, 2, 3), dtype=np.float64)
        # Right arm moves peak_disp during [1.0s, 2.0s] then returns to 0
        eef_pos[45, 1, 0] = peak_disp
        return KinematicsData(
            timestamps_s=ts,
            eef_pos=eef_pos,
            eef_rotvec=np.zeros((T, 2, 3), dtype=np.float64),
            hand_q=np.zeros((T, 44), dtype=np.float64),
            valid_mask=np.ones(T, dtype=bool),
        )

    seg_left = _make_cut_candidate().model_copy(update={"affected_arm": "left"})
    res_pass = _invoke_guardrail(
        "robo_vlm_qc.guardrail.splice_gate",
        [seg_left],
        _kin_other_arm(0.029),  # < 3.0cm
        None,
        cfg,
        4.0,
    )
    assert res_pass[0].splice_policy == "cut_clean"

    res_fail = _invoke_guardrail(
        "robo_vlm_qc.guardrail.splice_gate",
        [seg_left.model_copy(deep=True)],
        _kin_other_arm(0.031),  # > 3.0cm coupling
        None,
        cfg,
        4.0,
    )
    assert res_fail[0].splice_policy == "mask_loss"
    assert res_fail[0].guardrail.downgraded is True


def test_splice_gate_boundary_idle_near_t0_and_duration_allowed(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    from robo_vlm_qc.io.base import KinematicsData

    cfg = _load_guardrail_cfg(e2e_workspace, e2e_config_dir)
    # Construct >2cm (0.25m) step jumps INSIDE [0.0, 0.5] (at t=0.3s, idx=9),
    # INSIDE [1.2, 1.8] (at t=1.5s, idx=45), and INSIDE [3.4, 4.0] (at t=3.6s, idx=108).
    T = 120
    fps = 30.0
    ts = np.arange(T, dtype=np.float64) / fps
    eef_pos = np.zeros((T, 2, 3), dtype=np.float64)
    eef_pos[9:45, 0, 0] = 0.25
    eef_pos[45:108, 0, 0] = 0.50
    eef_pos[108:, 0, 0] = 0.75
    kin = KinematicsData(
        timestamps_s=ts,
        eef_pos=eef_pos,
        eef_rotvec=np.zeros((T, 2, 3), dtype=np.float64),
        hand_q=np.zeros((T, 44), dtype=np.float64),
        torso_q=np.zeros((T, 4), dtype=np.float64),
        chassis_pose=np.zeros((T, 3), dtype=np.float64),
        valid_mask=np.ones(T, dtype=bool),
    )

    seg_start = _make_cut_candidate(t_start=0.0, t_end=0.5, label_id="boundary_idle")
    seg_mid = _make_cut_candidate(t_start=1.2, t_end=1.8, label_id="boundary_idle")
    seg_end = _make_cut_candidate(t_start=3.4, t_end=4.0, label_id="boundary_idle")
    res = _invoke_guardrail(
        "robo_vlm_qc.guardrail.splice_gate",
        [seg_start, seg_mid, seg_end],
        kin,
        None,
        cfg,
        4.0,
    )
    assert res[0].splice_policy == "cut_clean"
    assert res[1].splice_policy == "mask_loss"
    assert res[1].guardrail.downgraded is True
    assert res[2].splice_policy == "cut_clean"


def test_hallucination_14cm_no_override_vs_16cm_keep_positive(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    from robo_vlm_qc.io.base import KinematicsData

    cfg = _load_guardrail_cfg(e2e_workspace, e2e_config_dir)

    def _kin_monotonic(disp_m: float) -> Any:
        T = 120
        ts = np.arange(T, dtype=np.float64) / 30.0
        eef_pos = np.zeros((T, 2, 3), dtype=np.float64)
        ramp = np.linspace(0.0, disp_m, 31)
        eef_pos[30:61, 0, 0] = ramp
        eef_pos[30:61, 1, 0] = ramp
        eef_pos[61:, 0, 0] = disp_m
        eef_pos[61:, 1, 0] = disp_m
        return KinematicsData(
            timestamps_s=ts,
            eef_pos=eef_pos,
            eef_rotvec=np.zeros((T, 2, 3), dtype=np.float64),
            valid_mask=np.ones(T, dtype=bool),
        )

    seg_14 = _make_cut_candidate(t_start=1.0, t_end=2.0, label_id="mid_idle_freeze")
    res_14 = _invoke_guardrail(
        "robo_vlm_qc.guardrail.hallucination",
        [seg_14],
        _kin_monotonic(0.14),
        None,
        cfg,
        4.0,
    )
    assert res_14[0].guardrail.guardrail_override is False

    seg_16 = _make_cut_candidate(t_start=1.0, t_end=2.0, label_id="mid_idle_freeze")
    res_16 = _invoke_guardrail(
        "robo_vlm_qc.guardrail.hallucination",
        [seg_16],
        _kin_monotonic(0.16),
        None,
        cfg,
        4.0,
    )
    assert res_16[0].guardrail.guardrail_override is True
    assert res_16[0].splice_policy == "keep_positive"


def test_visual_gate_identical_frames_pass_vs_altered_scene_downgrade(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.io import get_adapter

    cfg = _load_guardrail_cfg(e2e_workspace, e2e_config_dir)

    # Create pure-video episode where frames at t=1.0s and t=2.0s are identical,
    # vs altered at t=3.0s
    ep_dir = tmp_path / "vis_gate_ds" / "ep_0000"
    ep_dir.mkdir(parents=True)
    frames: list[np.ndarray] = []
    for i in range(120):
        img = np.full((240, 320, 3), 40, dtype=np.uint8)
        if i < 75:
            cv2.rectangle(img, (40, 40), (140, 140), (200, 120, 40), -1)
        else:
            cv2.rectangle(img, (180, 100), (300, 220), (40, 220, 220), -1)
        frames.append(img)
    h._write_mp4_frames(ep_dir / "main.mp4", frames, fps=30.0)

    ad = get_adapter("video_dir")
    with ad.open(ad.discover(tmp_path / "vis_gate_ds")[0]) as view:
        seg_same = _make_cut_candidate(t_start=1.0, t_end=2.0, label_id="mid_idle_freeze")
        res_same = _invoke_guardrail(
            "robo_vlm_qc.guardrail.visual_gate",
            [seg_same],
            None,
            view.media,
            cfg,
            view.duration_s,
        )
        assert res_same[0].splice_policy == "cut_clean"

        seg_diff = _make_cut_candidate(t_start=1.0, t_end=3.0, label_id="mid_idle_freeze")
        res_diff = _invoke_guardrail(
            "robo_vlm_qc.guardrail.visual_gate",
            [seg_diff],
            None,
            view.media,
            cfg,
            view.duration_s,
        )
        assert res_diff[0].splice_policy == "mask_loss"
        assert res_diff[0].guardrail.downgraded is True


# ===========================================================================
# 1080p viz_mp4 & run-vlm / render-mp4 CLI Boundaries
# ===========================================================================


def test_viz_mp4_zero_qc_segments_and_zero_grounding_tracks(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    mock_vlm_server.set_stage_response("S1", {"objects": {}})
    mock_vlm_server.set_stage_response("S3", {"segments": []})
    ep_id = _ingest_one_video_ep(e2e_workspace, e2e_config_dir, tmp_path / "data")
    assert _invoke_dag_run(e2e_workspace, e2e_config_dir, [ep_id]).returncode == 0

    out_mp4 = tmp_path / "out_zero.mp4"
    proc = h.run_cli(
        [
            "-w",
            str(e2e_workspace),
            "-c",
            str(e2e_config_dir),
            "render-mp4",
            "--episode",
            ep_id,
            "--out",
            str(out_mp4),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    info = h.ffprobe_video_info(out_mp4)
    assert info["width"] == 1920 and info["height"] == 1080


def test_viz_mp4_long_macro_stage_label_no_crash(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    long_label = (
        "Bimanual precision alignment and steady placement into the designated tabletop target bin zone; "
        * 4
    ).strip()
    mock_vlm_server.set_stage_response(
        "S2",
        {
            "macro_stages": [
                {
                    "stage_id": "S1",
                    "t_start": 0.0,
                    "t_end": 2.0,
                    "label": long_label,
                    "label_en": "very_long_macro_stage_name",
                }
            ],
            "arm_timelines": {
                "left": [{"t_start": 0.0, "t_end": 2.0, "state": "work"}],
                "right": [{"t_start": 0.0, "t_end": 2.0, "state": "hold"}],
            },
        },
    )
    ep_id = _ingest_one_video_ep(e2e_workspace, e2e_config_dir, tmp_path / "data")
    assert _invoke_dag_run(e2e_workspace, e2e_config_dir, [ep_id]).returncode == 0

    out_mp4 = tmp_path / "out_long_label.mp4"
    proc = h.run_cli(
        [
            "-w",
            str(e2e_workspace),
            "-c",
            str(e2e_config_dir),
            "render-mp4",
            "--episode",
            ep_id,
            "--out",
            str(out_mp4),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    assert out_mp4.is_file() and out_mp4.stat().st_size > 1024


def test_run_vlm_cli_nonexistent_episode_error(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    proc = h.run_cli(
        [
            "-w",
            str(e2e_workspace),
            "-c",
            str(e2e_config_dir),
            "run-vlm",
            "--episode",
            "missing_ds/missing_ep",
        ]
    )
    assert proc.returncode != 0


def test_render_mp4_cli_before_run_vlm_error(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    ep_id = _ingest_one_video_ep(e2e_workspace, e2e_config_dir, tmp_path / "data")
    out_mp4 = tmp_path / "no_run.mp4"
    proc = h.run_cli(
        [
            "-w",
            str(e2e_workspace),
            "-c",
            str(e2e_config_dir),
            "render-mp4",
            "--episode",
            ep_id,
            "--out",
            str(out_mp4),
        ]
    )
    assert proc.returncode != 0
    assert not out_mp4.exists()


def test_render_mp4_cli_output_path_creation(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    ep_id = _ingest_one_video_ep(e2e_workspace, e2e_config_dir, tmp_path / "data")
    assert _invoke_dag_run(e2e_workspace, e2e_config_dir, [ep_id]).returncode == 0

    nested_out = tmp_path / "deep" / "nested" / "dir" / "hud.mp4"
    proc = h.run_cli(
        [
            "-w",
            str(e2e_workspace),
            "-c",
            str(e2e_config_dir),
            "render-mp4",
            "--episode",
            ep_id,
            "--out",
            str(nested_out),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    assert nested_out.is_file()
    info = h.ffprobe_video_info(nested_out)
    assert info["width"] == 1920 and info["height"] == 1080
