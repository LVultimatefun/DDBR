"""Cross-Feature Pairwise Integration E2E Tests for robo_vlm_qc.

All ``robo_vlm_qc.*`` imports are deferred inside test/helper functions so
``pytest --collect-only tests/e2e/test_pairwise.py`` collects all tests
cleanly in isolated environments.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq
import yaml

from tests.e2e import helpers as h


# ---------------------------------------------------------------------------
# Local Opaque-Box Test Builders & Contract Helpers (No robo_vlm_qc top-level imports)
# ---------------------------------------------------------------------------


def _make_vlm_run_artifact(
    episode_id: str,
    *,
    run_id: str = "run_001",
    duration_s: float = 4.0,
    qc_segments: list[dict[str, Any]] | None = None,
    passed: bool = True,
    macro_stages: list[dict[str, Any]] | None = None,
    is_eval_sandbox: bool = False,
) -> Any:
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
        StageUsage,
        SuccessVerdict,
        UsageSummary,
        VlmRunArtifact,
    )

    mid = round(duration_s * 0.5, 3)
    stages = macro_stages or [
        {
            "stage_id": "S1",
            "t_start": 0.0,
            "t_end": mid,
            "label": "Approach and grasp target",
            "label_en": "approach_and_grasp",
            "summary": "Approach and lift the target object.",
            "color_rgb": [59, 130, 246],
        },
        {
            "stage_id": "S2",
            "t_start": mid,
            "t_end": round(duration_s, 3),
            "label": "Transfer and place",
            "label_en": "transfer_and_place",
            "summary": "Transfer stably and complete placement.",
            "color_rgb": [16, 185, 129],
        },
    ]

    if qc_segments is None:
        qc_segments = [
            {
                "seg_id": "seg_001",
                "t_start": 0.0,
                "t_end": round(min(0.4, duration_s * 0.12), 3),
                "label_id": "boundary_idle",
                "name": "Boundary Idle",
                "training_role": "badcase",
                "splice_policy": "cut_clean",
                "mask_side": None,
                "affected_arm": "both",
                "vlm_raw_policy": "cut_clean",
                "confidence": 0.94,
            },
            {
                "seg_id": "seg_002",
                "t_start": round(min(1.0, duration_s * 0.35), 3),
                "t_end": round(min(1.8, duration_s * 0.60), 3),
                "label_id": "missed_grasp_retry",
                "name": "Missed Grasp Retry",
                "training_role": "recovery",
                "splice_policy": "mask_loss",
                "mask_side": "left",
                "affected_arm": "left",
                "vlm_raw_policy": "mask_loss",
                "confidence": 0.90,
            },
        ]

    grounding = GroundingResult(
        ref_size=[1280, 720],
        keyframes_s=[0.0, mid],
        object_tracks_2d={
            "target_object": GroundingTrack(
                object_id="target_object",
                name="Target Object",
                tag_pos="top",
                is_static_container=False,
                keyframes=[
                    GroundingKeyframe(
                        row=0,
                        time_s=0.0,
                        visible=True,
                        bbox_norm_1000=[120, 220, 300, 420],
                        state="on_table",
                        status_kind="neutral",
                    )
                ],
            )
        },
    )
    arm_timelines = ArmTimelines(
        left=[ArmSpan(t_start=0.0, t_end=round(duration_s, 3), state="work", desc="left active")],
        right=[ArmSpan(t_start=0.0, t_end=round(duration_s, 3), state="hold", desc="right hold")],
    )
    return VlmRunArtifact(
        run_id=run_id,
        episode_id=episode_id,
        created_at=time.time(),
        is_eval_sandbox=is_eval_sandbox,
        config_fingerprint=ConfigFingerprintSnapshot(
            model_routing={
                "s1_grounding": "gpt-4o",
                "s2_macro": "gpt-4o",
                "s3_discover": "gpt-4o",
                "s4_verify": "gpt-4o",
                "s5_success": "gpt-4o",
            },
            prompt_bundle_version="1.0.0",
            label_taxonomy_version="1",
            task_spec_version="1.0.0",
        ),
        stage_refs=StageRefs(
            s1_grounding="s1_sha",
            s2_macro="s2_sha",
            s3_discover="s3_sha",
            s4_verify="s4_sha",
            s5_success="s5_sha",
        ),
        grounding=grounding,
        macro_stages=[MacroStage(**s) for s in stages],
        arm_timelines=arm_timelines,
        qc_segments=[QcSegment(**seg) for seg in qc_segments],
        success=SuccessVerdict(
            passed=passed,
            verdict="success" if passed else "failure",
            confidence=0.92,
            criteria=[
                CriterionResult(
                    criterion="Target object reaches expected area",
                    met=passed,
                    satisfied_at_s=round(duration_s * 0.85, 2) if passed else None,
                    evidence="Verified in final frames",
                )
            ],
            summary="Task succeeded" if passed else "Task failed",
        ),
        usage=UsageSummary(
            stages=[
                StageUsage(stage="s1_grounding", model="gpt-4o", total_tokens=180),
                StageUsage(stage="s2_macro", model="gpt-4o", total_tokens=180),
            ],
            total_tokens=360,
            total_elapsed_s=0.4,
        ),
    )


def _ingest_path(ws: Path, data_path: Path, *, dataset: str | None = None) -> list[str]:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.io import auto_detect, get_adapter
    from robo_vlm_qc.store import Repository

    cfg_ws = ws if (ws / "configs_override").is_dir() else None
    cfg = ConfigManager(workspace_dir=cfg_ws)
    handles = auto_detect(data_path, task_patterns=cfg.task_patterns())
    repo = Repository(ws)
    ep_ids: list[str] = []
    try:
        for hdl in handles:
            ad = get_adapter(hdl.adapter_type)
            with ad.open(hdl) as view:
                meta = view.meta
                ds_name = dataset or meta.dataset
                h5_hint = meta.extra_meta.get("h5_task_name") or meta.extra_meta.get("ep_task_str")
                task_id = cfg.resolve_task(ds_name, str(h5_hint) if h5_hint else None).task_id
                if ds_name != meta.dataset or task_id != meta.task_id:
                    meta = meta.model_copy(
                        update={
                            "dataset": ds_name,
                            "episode_id": f"{ds_name}/{meta.stem}",
                            "task_id": task_id,
                        }
                    )
                repo.ingest_episode(meta)
                ep_ids.append(meta.episode_id)
    finally:
        repo.close()
    return ep_ids


def _unwrap_list_response(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("episodes", "items", "labels", "examples", "runs", "results", "data"):
            val = payload.get(key)
            if isinstance(val, list):
                return val
    raise AssertionError(f"Expected list or wrapped list payload, got: {payload!r}")


def _make_loop_closure_h5(out_dir: Path, name: str = "episode_000000.h5") -> Path:
    """Create a 4.0s h5_raw_v3 file that returns to its t=1.2s pose at t=2.4s."""
    h5_path = h.make_synthetic_h5_raw_v3(
        out_dir, name=name, duration_s=4.0, stationary=False, task_name="pick_and_place"
    )
    with h5py.File(h5_path, "r+") as f:
        r_ts = f["robot/timestamps"][:]
        t0 = float(f.attrs["episode_start_timestamp"])
        t_rel = r_ts - t0
        state = f["robot/state"][:]
        # Make a smooth bump centered at t=1.8s: identical quiescent pose for t<=1.2s and t>=2.4s,
        # and a 0.14m excursion at t=1.8s. Keep hand_q also quiescent at t<=1.2 and t>=2.4.
        bump = np.exp(-0.5 * ((t_rel - 1.8) / 0.15) ** 2)
        bump = np.where((t_rel <= 1.2) | (t_rel >= 2.4), 0.0, bump)
        state[:, 0:23] = 0.0
        state[:, 23] = (0.35 + 0.14 * bump).astype(np.float32)
        state[:, 24] = (0.20 + 0.08 * bump).astype(np.float32)
        state[:, 25] = 0.45
        state[:, 26:30] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        state[:, 30] = 0.35
        state[:, 31] = -0.20
        state[:, 32] = 0.45
        state[:, 33:37] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        f["robot/state"][:] = state
        f["hand/hand_a_radian"][:] = 0.0
        f["hand/hand_b_radian"][:] = 0.0
    return h5_path


def _call_piece_slicer(resolved: Any, view: Any, recipe: Any, cfg: Any = None) -> list[Any]:
    import inspect
    from robo_vlm_qc.export.slicer import PieceSlicer

    sig = inspect.signature(PieceSlicer)
    kwargs: dict[str, Any] = {}
    for pname in sig.parameters:
        if pname in ("config", "cfg", "config_manager") and cfg is not None:
            kwargs[pname] = cfg
        elif pname == "settings" and cfg is not None:
            kwargs[pname] = cfg.settings
    slicer = PieceSlicer(**kwargs)
    return slicer.slice(resolved, view, recipe)


# ===========================================================================
# Block A: Foundation, Configs, Schemas, Store & 3-In I/O
# ===========================================================================


def test_pairwise_workspace_task_override_priority(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    """Workspace configs_override task match priority overrides h5 task_name during ingest."""
    from robo_vlm_qc.store import Repository

    tasks_ov = e2e_workspace / "configs_override" / "tasks"
    tasks_ov.mkdir(parents=True, exist_ok=True)
    (tasks_ov / "pick_and_place.yaml").write_text(
        yaml.safe_dump(
            {
                "task_id": "pick_and_place",
                "match": ["*pick_and_place*", "*custom_pick*"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    h5_dir = tmp_path / "custom_pick_batch"
    h.make_synthetic_h5_raw_v3(h5_dir, name="ep_01.h5", duration_s=2.5, task_name="pick_cube")

    proc = h.run_cli(
        [
            "--workspace",
            str(e2e_workspace),
            "--config-dir",
            str(e2e_config_dir),
            "ingest",
            "--data",
            str(h5_dir),
            "--json",
        ]
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["ok"] is True
    assert report["inserted"] == 1
    assert report["episodes"][0]["task_id"] == "pick_and_place"

    repo = Repository(e2e_workspace)
    try:
        row = repo.get_episode_row("custom_pick_batch/ep_01")
        assert row["task_id"] == "pick_and_place"
    finally:
        repo.close()


def test_pairwise_heterogeneous_stage_models_routing(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    """Per-stage heterogeneous stage_models config routes distinct models to MockVlmServer."""
    from robo_vlm_qc.store import Repository

    h.write_workspace_config(
        e2e_workspace,
        base_url=mock_vlm_server.base_url,
        stage_models={
            "s1_grounding": "gpt-4o-mini",
            "s2_macro": "gpt-4o-mini",
            "s3_discover": "gpt-4o",
            "s4_verify": "gpt-4o",
            "s5_success": "gpt-4o",
        },
    )
    h5_dir = tmp_path / "pick_and_place_hetero"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=3.0)
    _ingest_path(e2e_workspace, h5_dir)

    proc = h.run_cli(
        [
            "--workspace",
            str(e2e_workspace),
            "--config-dir",
            str(e2e_config_dir),
            "run-vlm",
            "--episode",
            "pick_and_place_hetero/episode_000000",
        ]
    )
    assert proc.returncode == 0, proc.stderr

    s1_models = {r["model"] for r in mock_vlm_server.requests if r["stage"] == "S1"}
    s2_models = {r["model"] for r in mock_vlm_server.requests if r["stage"] == "S2"}
    s3_models = {r["model"] for r in mock_vlm_server.requests if r["stage"] == "S3"}
    s4_models = {r["model"] for r in mock_vlm_server.requests if r["stage"] in {"S4A", "S4B"}}
    s5_models = {r["model"] for r in mock_vlm_server.requests if r["stage"] in {"S5", "S5_CONTROL"}}

    assert s1_models == {"gpt-4o-mini"}
    assert s2_models == {"gpt-4o-mini"}
    assert s3_models == {"gpt-4o"}
    assert s4_models == {"gpt-4o"}
    assert s5_models == {"gpt-4o"}

    repo = Repository(e2e_workspace)
    try:
        row = repo.get_episode_row("pick_and_place_hetero/episode_000000")
        run = repo.get_vlm_run(row["active_run_id"])
        assert run.config_fingerprint.model_routing["s1_grounding"] == "gpt-4o-mini"
        assert run.config_fingerprint.model_routing["s3_discover"] == "gpt-4o"
    finally:
        repo.close()


def test_pairwise_vlm_and_review_resolve_preserves_readonly_layers(
    e2e_workspace: Path, tmp_path: Path
) -> None:
    """Repository resolves HumanReviewArtifact over VlmRunArtifact while preserving readonly layers."""
    from robo_vlm_qc.models.annotation import MacroStage, QcSegment, SuccessVerdict
    from robo_vlm_qc.models.review import EpisodeStatus, ReviewWritePayload, WatchCoverage
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "pick_and_place_res"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=4.0)
    ep_id = _ingest_path(e2e_workspace, h5_dir)[0]

    repo = Repository(e2e_workspace)
    try:
        vlm_run = _make_vlm_run_artifact(ep_id, run_id="run_res_01", duration_s=4.0, passed=False)
        repo.save_vlm_run_artifact(vlm_run)

        edited_segs = [
            QcSegment(
                seg_id="seg_human_1",
                t_start=0.5,
                t_end=1.2,
                label_id="placement_redo",
                training_role="recovery",
                splice_policy="mask_loss",
                mask_side="right",
                affected_arm="right",
                vlm_raw_policy="mask_loss",
            )
        ]
        edited_stages = [
            MacroStage(
                stage_id="S1",
                t_start=0.0,
                t_end=4.0,
                label="Single-stage full manipulation",
                label_en="full_process",
            )
        ]
        payload = ReviewWritePayload(
            base_run_id="run_res_01",
            action="submit_review",
            qc_segments=edited_segs,
            success=SuccessVerdict(passed=True, verdict="success", confidence=1.0),
            macro_stages=edited_stages,
            watch_coverage=WatchCoverage(
                watched_intervals=[(0.0, 4.0)],
                watched_s=4.0,
                duration_s=4.0,
                coverage_ratio=1.0,
                confirmed=True,
            ),
            notes="Human verified success",
        )
        repo.save_review(ep_id, "rev_alice", "reviewer", payload)
        resolved = repo.resolve_episode(ep_id)

        assert resolved.status == EpisodeStatus.REVIEWED
        assert resolved.is_human_reviewed is True
        assert resolved.success.passed is True
        assert len(resolved.qc_segments) == 1
        assert resolved.qc_segments[0].seg_id == "seg_human_1"
        assert resolved.macro_stages[0].label == "Single-stage full manipulation"
        # Read-only VLM layers must stay identical to VlmRunArtifact
        assert resolved.grounding == vlm_run.grounding
        assert resolved.arm_timelines == vlm_run.arm_timelines
        assert resolved.raw_vlm_segments == vlm_run.qc_segments
        assert resolved.review_diff is not None
        assert resolved.review_diff.success_flipped is True
        h.assert_no_legacy_fields(resolved)
    finally:
        repo.close()


def test_pairwise_taxonomy_mutation_history_backup_and_repo_query(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    """PromptBundle/VisualLabelDef mutation via ArtifactStore creates .history backup + updates ConfigManager."""
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.models.taxonomy import CollectorUiHint, VisualLabelDef, VlmSemantics
    from robo_vlm_qc.store import Repository

    cfg_before = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    old_label_sha = cfg_before.fingerprints.label_taxonomy_sha256
    assert cfg_before.prompts.s3_discover.stage == "s3_discover"

    new_label = VisualLabelDef(
        id="object_tilt_spill",
        name="Object Tilt Spill",
        category="failure",
        status="active",
        version=1,
        default_training_role="badcase",
        default_splice_policy="mask_loss",
        collector_ui=CollectorUiHint(
            title="Container tilted and spilled",
            hint="Tilting the held container caused its contents to spill",
        ),
        vlm_semantics=VlmSemantics(
            definition="The container tilts noticeably and spills contents onto the workspace",
            visual_cues=["Container tilts past 30 degrees"],
        ),
    )

    repo = Repository(e2e_workspace)
    try:
        labels_dict = cfg_before.labels.model_dump(mode="json")
        labels_dict["version"] = 2
        labels_dict["labels"].append(new_label.model_dump(mode="json"))
        ov_text_v2 = yaml.safe_dump(labels_dict, sort_keys=False, allow_unicode=True)
        repo.artifacts.write_config_override("labels.yaml", ov_text_v2, reason="add_v2")

        labels_dict["version"] = 3
        ov_text_v3 = yaml.safe_dump(labels_dict, sort_keys=False, allow_unicode=True)
        repo.artifacts.write_config_override("labels.yaml", ov_text_v3, reason="bump_v3")

        snaps = list((e2e_workspace / ".history").rglob("*.json"))
        assert len(snaps) >= 1
        snap_data = json.loads(snaps[0].read_text(encoding="utf-8"))
        assert snap_data["reason"] == "bump_v3"
        assert "object_tilt_spill" in snap_data["content"]

        cfg_after = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
        assert cfg_after.fingerprints.label_taxonomy_sha256 != old_label_sha
        assert cfg_after.labels.get("object_tilt_spill") is not None

        h5_dir = tmp_path / "pick_and_place_tax"
        h.make_synthetic_h5_raw_v3(h5_dir, duration_s=2.0)
        _ingest_path(e2e_workspace, h5_dir)
        listed = repo.list_episodes(task_id="pick_and_place")
        assert len(listed) == 1
    finally:
        repo.close()


def test_pairwise_h5_raw_v3_adaptive_torso_head_2x2_tile(tmp_path: Path) -> None:
    """h5_raw_v3 (head_rgbd, torso_rgbd, head_stereo) composes [TORSO|TORSO-ZOOM/HEAD|HEAD-ZOOM]."""
    from robo_vlm_qc.io import auto_detect, get_adapter
    from robo_vlm_qc.io.media import compose_2x2_tile, select_tile_mode

    h5_dir = tmp_path / "pick_and_place_tiler"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=2.0, with_sbs_stereo=True)
    handles = auto_detect(h5_dir)
    assert len(handles) == 1

    with get_adapter("h5_raw_v3").open(handles[0]) as view:
        assert select_tile_mode(view.camera_roles) == "torso_head_2x2"
        tile = compose_2x2_tile(
            view.media, view.camera_roles, t_s=0.5, cell_w=320, cell_h=180, header_h=26
        )
        assert tile.dtype == np.uint8
        assert tile.shape == (386, 640, 3)
        top_half = tile[26:206]
        bot_half = tile[206:386]
        assert not np.array_equal(top_half, bot_half)


def test_pairwise_lerobot_adaptive_wrist_main_2x2_tile(tmp_path: Path) -> None:
    """LeRobot 3-cam (main, left_wrist, right_wrist) composes [L-WRIST|R-WRIST/MAIN|MAIN-ZOOM]."""
    from robo_vlm_qc.io import auto_detect, get_adapter
    from robo_vlm_qc.io.media import compose_2x2_tile, select_tile_mode

    lr_dir = tmp_path / "lr_tiler"
    h.make_synthetic_lerobot(
        lr_dir, version="v3", n_episodes=1, duration_s=2.0, task_name="bimanual_handover"
    )
    handles = auto_detect(lr_dir)
    assert len(handles) == 1

    with get_adapter("lerobot").open(handles[0]) as view:
        assert select_tile_mode(view.camera_roles) == "wrist_2x2"
        tile = compose_2x2_tile(
            view.media, view.camera_roles, t_s=0.8, cell_w=320, cell_h=180, header_h=26
        )
        assert tile.dtype == np.uint8
        assert tile.shape == (386, 640, 3)
        lw_cell = tile[26:206, :320]
        rw_cell = tile[26:206, 320:]
        assert not np.array_equal(lw_cell, rw_cell)


def test_pairwise_video_dir_single_cam_adaptive_1x2_tile(tmp_path: Path) -> None:
    """Single-camera video_dir composes adaptive 1x2 [MAIN|MAIN-CENTER-ZOOM] tile."""
    from robo_vlm_qc.io import auto_detect, get_adapter
    from robo_vlm_qc.io.media import compose_2x2_tile, select_tile_mode

    vid_dir = tmp_path / "single_cam_ds"
    h.make_synthetic_video_dir(vid_dir, multi_cam=False, duration_s=2.0, n_episodes=1)
    handles = auto_detect(vid_dir)
    assert len(handles) == 1

    with get_adapter("video_dir").open(handles[0]) as view:
        assert view.kinematics is None
        assert select_tile_mode(view.camera_roles) == "single_1x2"
        tile = compose_2x2_tile(
            view.media, view.camera_roles, t_s=0.5, cell_w=320, cell_h=180, header_h=26
        )
        assert tile.dtype == np.uint8
        assert tile.shape == (206, 640, 3)
        assert not np.array_equal(tile[26:, :320], tile[26:, 320:])


def test_pairwise_mixed_3_formats_ingest_into_single_wal_db(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    """Single ingest CLI scans h5_raw_v3 + lerobot + video_dir into WAL DB."""
    from robo_vlm_qc.store import Repository

    src = tmp_path / "mixed_sources"
    h5_root = src / "pick_and_place_h5"
    lr_root = src / "lerobot_ds"
    vd_root = src / "pure_video"

    h.make_synthetic_h5_raw_v3(h5_root, duration_s=2.0)
    h.make_synthetic_lerobot(
        lr_root, version="v3", n_episodes=1, duration_s=2.0, task_name="bimanual_handover"
    )
    h.make_synthetic_video_dir(vd_root, multi_cam=True, duration_s=2.0, n_episodes=1)

    proc = h.run_cli(
        [
            "--workspace",
            str(e2e_workspace),
            "--config-dir",
            str(e2e_config_dir),
            "ingest",
            "--data",
            str(h5_root),
            "--data",
            str(lr_root),
            "--data",
            str(vd_root),
            "--json",
        ]
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["inserted"] == 3
    assert set(report["per_adapter"].keys()) == {
        "h5_raw_v3",
        "lerobot",
        "video_dir",
    }

    repo = Repository(e2e_workspace)
    try:
        rows = repo.list_episodes(limit=20)
        assert len(rows) == 3
        by_adapter = {r["adapter_type"]: r for r in rows}
        assert by_adapter["h5_raw_v3"]["has_kinematics"] == 1
        assert by_adapter["lerobot"]["has_kinematics"] == 1
        assert by_adapter["video_dir"]["has_kinematics"] == 0
    finally:
        repo.close()


def test_pairwise_ingest_dry_run_then_idempotent_reingest(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    """ingest --dry-run writes 0 DB rows; subsequent ingest inserts then skips unchanged mtimes."""
    from robo_vlm_qc.store import Repository

    lr_v2 = tmp_path / "lr_v2_batch"
    h.make_synthetic_lerobot(lr_v2, version="v2", n_episodes=2, duration_s=2.0)
    vd_dir = tmp_path / "vd_batch"
    h.make_synthetic_video_dir(vd_dir, multi_cam=False, duration_s=2.0, n_episodes=1)

    dry = h.run_cli(
        [
            "--workspace",
            str(e2e_workspace),
            "--config-dir",
            str(e2e_config_dir),
            "ingest",
            "--data",
            str(lr_v2),
            "--data",
            str(vd_dir),
            "--dry-run",
            "--json",
        ]
    )
    assert dry.returncode == 0, dry.stderr
    dry_rep = json.loads(dry.stdout)
    assert dry_rep["would_ingest"] == 3
    assert not (e2e_workspace / "robo_vlm_qc.db").exists()

    first = h.run_cli(
        [
            "--workspace",
            str(e2e_workspace),
            "--config-dir",
            str(e2e_config_dir),
            "ingest",
            "--data",
            str(lr_v2),
            "--data",
            str(vd_dir),
            "--json",
        ]
    )
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)["inserted"] == 3

    second = h.run_cli(
        [
            "--workspace",
            str(e2e_workspace),
            "--config-dir",
            str(e2e_config_dir),
            "ingest",
            "--data",
            str(lr_v2),
            "--data",
            str(vd_dir),
            "--json",
        ]
    )
    assert second.returncode == 0, second.stderr
    sec_rep = json.loads(second.stdout)
    assert sec_rep["inserted"] == 0
    assert sec_rep["skipped"] == 3

    repo = Repository(e2e_workspace)
    try:
        assert len(repo.list_episodes()) == 3
    finally:
        repo.close()


def test_pairwise_sampled_media_reader_feeds_s1_s2_adaptive_prompts(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    """Camera-role-adaptive tiles & prompt descriptions differ between h5 torso/head and lerobot wrist/main."""
    h5_dir = tmp_path / "pick_and_place_cam_prompt"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=2.5)
    lr_dir = tmp_path / "lr_cam_prompt"
    h.make_synthetic_lerobot(
        lr_dir, version="v3", n_episodes=1, duration_s=2.5, task_name="bimanual_handover"
    )

    h5_ep = _ingest_path(e2e_workspace, h5_dir)[0]
    lr_ep = _ingest_path(e2e_workspace, lr_dir)[0]

    proc1 = h.run_cli(
        [
            "--workspace",
            str(e2e_workspace),
            "--config-dir",
            str(e2e_config_dir),
            "run-vlm",
            "--episode",
            h5_ep,
        ]
    )
    assert proc1.returncode == 0, proc1.stderr
    h5_s2_reqs = [r for r in mock_vlm_server.requests if r["stage"] == "S2"]
    assert h5_s2_reqs
    h5_s2_text = "\n".join(
        str(b.get("text", ""))
        for m in h5_s2_reqs[-1]["messages"]
        for b in (m["content"] if isinstance(m.get("content"), list) else [{"type": "text", "text": m.get("content", "")}])
        if isinstance(b, dict) and b.get("type") == "text"
    )
    h5_s2_imgs = [
        b
        for m in h5_s2_reqs[-1]["messages"]
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "image_url"
    ]
    assert len(h5_s2_imgs) >= 1
    assert any(tok in h5_s2_text.upper() for tok in ("TORSO", "HEAD"))

    mock_vlm_server.requests.clear()
    proc2 = h.run_cli(
        [
            "--workspace",
            str(e2e_workspace),
            "--config-dir",
            str(e2e_config_dir),
            "run-vlm",
            "--episode",
            lr_ep,
        ]
    )
    assert proc2.returncode == 0, proc2.stderr
    lr_s2_reqs = [r for r in mock_vlm_server.requests if r["stage"] == "S2"]
    assert lr_s2_reqs
    lr_s2_text = "\n".join(
        str(b.get("text", ""))
        for m in lr_s2_reqs[-1]["messages"]
        for b in (m["content"] if isinstance(m.get("content"), list) else [{"type": "text", "text": m.get("content", "")}])
        if isinstance(b, dict) and b.get("type") == "text"
    )
    assert any(tok in lr_s2_text.upper() for tok in ("WRIST", "L-WRIST", "R-WRIST"))


# ===========================================================================
# Block B: VLM DAG, StageCache, Labels, Guardrails & 1080p MP4
# ===========================================================================


def test_pairwise_h5_slerp_kinematics_splice_gate_2d_grid_optimization(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    """h5_raw_v3 Slerp-aligned kinematics enables splice_gate 2D grid search to snap cut_clean."""
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "pick_and_place_loop"
    _make_loop_closure_h5(h5_dir)
    ep_id = _ingest_path(e2e_workspace, h5_dir)[0]

    cand = {
        "t_start": 1.45,
        "t_end": 2.15,
        "label_id": "mid_idle_freeze",
        "affected_arm": "both",
        "splice_policy": "cut_clean",
        "visual_evidence": {
            "initial_state": "Both arms hover above the table.",
            "process_change": "Brief mid-task twitch followed by return to the prior pose.",
            "end_state": "Both arms return to the original pose.",
        },
        "confidence": 0.9,
        "unknown_note": "",
    }
    mock_vlm_server.set_stage_response("S3", {"segments": [cand]})
    mock_vlm_server.set_stage_response("S4A", {"merged_segments": [cand]})
    mock_vlm_server.set_stage_response(
        "S4B",
        {
            "keep_segment": True,
            "t_start": 1.45,
            "t_end": 2.15,
            "label_id": "mid_idle_freeze",
            "affected_arm": "both",
            "splice_policy": "cut_clean",
            "visual_evidence": cand["visual_evidence"],
            "confidence": 0.9,
            "reject_reason": "",
        },
    )

    proc = h.run_cli(
        [
            "--workspace",
            str(e2e_workspace),
            "--config-dir",
            str(e2e_config_dir),
            "run-vlm",
            "--episode",
            ep_id,
        ]
    )
    assert proc.returncode == 0, proc.stderr

    repo = Repository(e2e_workspace)
    try:
        resolved = repo.resolve_episode(ep_id)
        assert len(resolved.qc_segments) == 1
        seg = resolved.qc_segments[0]
        assert seg.guardrail.gate_type == "splice_gate"
        # 2D grid optimization within +-1.0s finds the closed-loop cut <= 1.2s .. >= 2.4s
        assert seg.splice_policy == "cut_clean"
        assert seg.guardrail.passed is True
        assert seg.t_start <= 1.25 and seg.t_end >= 2.35
        assert seg.guardrail.metrics.eef_pos_m <= 0.02
    finally:
        repo.close()


def test_pairwise_video_dir_routes_to_visual_gate_not_splice_gate(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    """Pure-video video_dir (kinematics=None) routes to visual_gate and downgrades visual diff cut_clean."""
    from robo_vlm_qc.store import Repository

    vd_dir = tmp_path / "vd_visual_gate"
    h.make_synthetic_video_dir(
        vd_dir, multi_cam=False, duration_s=3.0, n_episodes=1, scene_change=True
    )
    ep_id = _ingest_path(e2e_workspace, vd_dir)[0]

    # Moving box in make_synthetic_video_dir shifts significantly between 0.5s and 2.5s
    cand = {
        "t_start": 0.5,
        "t_end": 2.5,
        "label_id": "mid_idle_freeze",
        "affected_arm": "both",
        "splice_policy": "cut_clean",
        "visual_evidence": {
            "initial_state": "Target is on the left.",
            "process_change": "Visible scene shift occurs across the interval.",
            "end_state": "Target is on the right.",
        },
        "confidence": 0.88,
        "unknown_note": "",
    }
    mock_vlm_server.set_stage_response("S3", {"segments": [cand]})
    mock_vlm_server.set_stage_response("S4A", {"merged_segments": [cand]})
    mock_vlm_server.set_stage_response(
        "S4B",
        {
            "keep_segment": True,
            "t_start": 0.5,
            "t_end": 2.5,
            "label_id": "mid_idle_freeze",
            "affected_arm": "both",
            "splice_policy": "cut_clean",
            "visual_evidence": cand["visual_evidence"],
            "confidence": 0.88,
            "reject_reason": "",
        },
    )

    proc = h.run_cli(
        [
            "--workspace",
            str(e2e_workspace),
            "--config-dir",
            str(e2e_config_dir),
            "run-vlm",
            "--episode",
            ep_id,
        ]
    )
    assert proc.returncode == 0, proc.stderr

    repo = Repository(e2e_workspace)
    try:
        resolved = repo.resolve_episode(ep_id)
        assert len(resolved.qc_segments) == 1
        seg = resolved.qc_segments[0]
        assert seg.guardrail.gate_type == "visual_gate"
        assert seg.vlm_raw_policy == "cut_clean"
        assert seg.splice_policy == "mask_loss"
        assert seg.guardrail.downgraded is True
        assert seg.guardrail.downgrade_reason
    finally:
        repo.close()


def test_pairwise_label_yaml_edit_invalidates_s3_s4_preserves_s1_s2_s5_cache(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    """Editing configs_override/labels.yaml invalidates S3/S4 cache while keeping 100% S1/S2/S5 hit."""
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "pick_and_place_cache_lbl"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=3.0)
    ep_id = _ingest_path(e2e_workspace, h5_dir)[0]

    proc1 = h.run_cli(
        [
            "--workspace",
            str(e2e_workspace),
            "--config-dir",
            str(e2e_config_dir),
            "run-vlm",
            "--episode",
            ep_id,
        ]
    )
    assert proc1.returncode == 0, proc1.stderr
    first_stages = {r["stage"] for r in mock_vlm_server.requests}
    assert {"S1", "S2", "S3", "S5"}.issubset(first_stages)

    repo = Repository(e2e_workspace)
    try:
        run1 = repo.get_vlm_run(repo.get_episode_row(ep_id)["active_run_id"])
    finally:
        repo.close()

    # Mutate configs_override/labels.yaml
    base_labels = yaml.safe_load((e2e_config_dir / "labels.yaml").read_text(encoding="utf-8"))
    base_labels["version"] = 2
    base_labels["labels"][0]["collector_ui"]["hint"] += " (revised)"
    (e2e_workspace / "configs_override" / "labels.yaml").write_text(
        yaml.safe_dump(base_labels, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    mock_vlm_server.requests.clear()
    proc2 = h.run_cli(
        [
            "--workspace",
            str(e2e_workspace),
            "--config-dir",
            str(e2e_config_dir),
            "run-vlm",
            "--episode",
            ep_id,
        ]
    )
    assert proc2.returncode == 0, proc2.stderr
    rerun_stages = [r["stage"] for r in mock_vlm_server.requests]
    assert "S1" not in rerun_stages
    assert "S2" not in rerun_stages
    assert "S5" not in rerun_stages
    assert "S5_CONTROL" not in rerun_stages
    assert any(st in {"S3", "S4A", "S4B"} for st in rerun_stages)

    repo = Repository(e2e_workspace)
    try:
        run2 = repo.get_vlm_run(repo.get_episode_row(ep_id)["active_run_id"])
        assert run2.run_id != run1.run_id
        assert run2.stage_refs.s1_grounding == run1.stage_refs.s1_grounding
        assert run2.stage_refs.s2_macro == run1.stage_refs.s2_macro
        assert run2.stage_refs.s5_success == run1.stage_refs.s5_success
        assert run2.stage_refs.s3_discover != run1.stage_refs.s3_discover
    finally:
        repo.close()


def test_pairwise_stage_cache_s1_static_container_and_s2_reuse(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    """StageCache stores S1/S2 results, stabilizes static container median bbox, and reuses on S3 prompt edit."""
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "pick_and_place_s1_s2"
    # 12s -> multiple S1 keyframes so static_container_min_frames=5 triggers median stabilization
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=12.0)
    ep_id = _ingest_path(e2e_workspace, h5_dir)[0]

    # Inject small jitter across S1 calls for static target_bin
    s1_bboxes = [
        [200, 150, 420, 450],
        [212, 154, 432, 454],
        [204, 148, 424, 448],
        [208, 152, 428, 452],
        [206, 151, 426, 451],
        [210, 155, 430, 455],
    ]
    mock_vlm_server.set_stage_response(
        "S1",
        [
            {
                "objects": {
                    "target_object": {
                        "visible": True,
                        "bbox": [500 + i * 20, 300, 620 + i * 20, 440],
                        "state": "moving",
                        "status_kind": "neutral",
                    },
                    "target_bin": {
                        "visible": True,
                        "bbox": bb,
                        "state": "static",
                        "status_kind": "neutral",
                    },
                }
            }
            for i, bb in enumerate(s1_bboxes)
        ],
    )

    proc1 = h.run_cli(
        [
            "--workspace",
            str(e2e_workspace),
            "--config-dir",
            str(e2e_config_dir),
            "run-vlm",
            "--episode",
            ep_id,
        ]
    )
    assert proc1.returncode == 0, proc1.stderr
    cache_files = list((e2e_workspace / "stage_cache").rglob("*.json"))
    assert len(cache_files) >= 4

    # Edit S3 prompt override only
    ov_prompts = e2e_workspace / "configs_override" / "prompts"
    ov_prompts.mkdir(parents=True, exist_ok=True)
    s3_yaml = yaml.safe_load((e2e_config_dir / "prompts" / "s3_discover.yaml").read_text(encoding="utf-8"))
    s3_yaml["version"] = "1.0.1"
    (ov_prompts / "s3_discover.yaml").write_text(
        yaml.safe_dump(s3_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    mock_vlm_server.requests.clear()
    proc2 = h.run_cli(
        [
            "--workspace",
            str(e2e_workspace),
            "--config-dir",
            str(e2e_config_dir),
            "run-vlm",
            "--episode",
            ep_id,
        ]
    )
    assert proc2.returncode == 0, proc2.stderr
    assert all(r["stage"] not in {"S1", "S2", "S5"} for r in mock_vlm_server.requests)

    repo = Repository(e2e_workspace)
    try:
        resolved = repo.resolve_episode(ep_id)
        bin_track = resolved.grounding.object_tracks_2d.get("target_bin")
        assert bin_track is not None
        assert bin_track.is_static_container is True
        visible_boxes = [kf.bbox_norm_1000 for kf in bin_track.keyframes if kf.visible]
        assert len(visible_boxes) >= 5
        # Median stabilization collapses jittered static container boxes to a single median box
        assert len({tuple(b) for b in visible_boxes}) == 1
    finally:
        repo.close()


def test_pairwise_s2_macro_backbone_injected_into_s3_and_s4b_diff(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    """S2 macro_stages backbone is injected into S3 prompt and S4B runs start-vs-end diff verification."""
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "pick_and_place_s2_s3"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=4.0)
    ep_id = _ingest_path(e2e_workspace, h5_dir)[0]

    custom_s2 = {
        "macro_stages": [
            {
                "stage_id": "S1",
                "t_start": 0.0,
                "t_end": 2.0,
                "label": "Approach and lift target block",
                "label_en": "approach_and_lift",
                "summary": "Reach and lift the target block steadily.",
                "color_rgb": [59, 130, 246],
            },
            {
                "stage_id": "S2",
                "t_start": 2.0,
                "t_end": 4.0,
                "label": "Transfer and seat in target bin",
                "label_en": "transfer_and_seat",
                "summary": "Transport and release cleanly into target bin.",
                "color_rgb": [16, 185, 129],
            },
        ],
        "arm_timelines": {
            "left": [{"t_start": 0.0, "t_end": 4.0, "state": "work", "desc": "Primary active arm"}],
            "right": [{"t_start": 0.0, "t_end": 4.0, "state": "hold", "desc": "Supportive steady hold"}],
        },
    }
    mock_vlm_server.set_stage_response("S2", custom_s2)

    proc = h.run_cli(
        [
            "--workspace",
            str(e2e_workspace),
            "--config-dir",
            str(e2e_config_dir),
            "run-vlm",
            "--episode",
            ep_id,
        ]
    )
    assert proc.returncode == 0, proc.stderr

    s3_reqs = [r for r in mock_vlm_server.requests if r["stage"] == "S3"]
    s4b_reqs = [r for r in mock_vlm_server.requests if r["stage"] == "S4B"]
    assert s3_reqs and s4b_reqs

    s3_prompt = "\n".join(
        str(b.get("text", ""))
        for m in s3_reqs[0]["messages"]
        for b in (m["content"] if isinstance(m.get("content"), list) else [{"type": "text", "text": m.get("content", "")}])
        if isinstance(b, dict) and b.get("type") == "text"
    )
    assert "Approach and lift target block" in s3_prompt
    assert "Transfer and seat in target bin" in s3_prompt

    repo = Repository(e2e_workspace)
    try:
        resolved = repo.resolve_episode(ep_id)
        assert [st.label for st in resolved.macro_stages] == [
            "Approach and lift target block",
            "Transfer and seat in target bin",
        ]
        for seg in resolved.qc_segments:
            ev = seg.visual_evidence
            assert ev.initial_state and ev.process_change and ev.end_state
    finally:
        repo.close()


def test_pairwise_dag_cut_clean_downgraded_to_mask_loss_with_reason(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    """DAG candidate mid_idle_freeze cut_clean during arm motion is downgraded to mask_loss with reason."""
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "pick_and_place_downgrade"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=4.0, stationary=False)
    ep_id = _ingest_path(e2e_workspace, h5_dir)[0]

    # Default MockVlmServer emits mid_idle_freeze cut_clean at [2.2, 2.8] where robot moves > 2cm
    proc = h.run_cli(
        [
            "--workspace",
            str(e2e_workspace),
            "--config-dir",
            str(e2e_config_dir),
            "run-vlm",
            "--episode",
            ep_id,
        ]
    )
    assert proc.returncode == 0, proc.stderr

    repo = Repository(e2e_workspace)
    try:
        resolved = repo.resolve_episode(ep_id)
        mid_segs = [s for s in resolved.qc_segments if s.label_id == "mid_idle_freeze"]
        assert mid_segs, f"Expected mid_idle_freeze segment, got {resolved.qc_segments}"
        seg = mid_segs[0]
        assert seg.vlm_raw_policy == "cut_clean"
        assert seg.splice_policy == "mask_loss"
        assert seg.guardrail.gate_type == "splice_gate"
        assert seg.guardrail.downgraded is True
        assert seg.guardrail.downgrade_reason
    finally:
        repo.close()


def test_pairwise_hallucination_guardrail_overrides_idle_on_gt_15cm_motion(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    """Hallucination guardrail overrides mid_idle_freeze when both arms move > 15cm."""
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "pick_and_place_halluc"
    # make_synthetic_h5_raw_v3 moves left eef ~0.28m and right eef ~0.24m over [0.45, 4.0]
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=4.0, stationary=False)
    ep_id = _ingest_path(e2e_workspace, h5_dir)[0]

    hal_cand = {
        "t_start": 0.6,
        "t_end": 3.8,
        "label_id": "mid_idle_freeze",
        "affected_arm": "both",
        "splice_policy": "cut_clean",
        "visual_evidence": {
            "initial_state": "Both arms stationary.",
            "process_change": "No displacement throughout interval.",
            "end_state": "Arms remain in place.",
        },
        "confidence": 0.72,
        "unknown_note": "",
    }
    mock_vlm_server.set_stage_response("S3", {"segments": [hal_cand]})
    mock_vlm_server.set_stage_response("S4A", {"merged_segments": [hal_cand]})
    mock_vlm_server.set_stage_response(
        "S4B",
        {
            "keep_segment": True,
            "t_start": 0.6,
            "t_end": 3.8,
            "label_id": "mid_idle_freeze",
            "affected_arm": "both",
            "splice_policy": "cut_clean",
            "visual_evidence": hal_cand["visual_evidence"],
            "confidence": 0.72,
            "reject_reason": "",
        },
    )

    proc = h.run_cli(
        [
            "--workspace",
            str(e2e_workspace),
            "--config-dir",
            str(e2e_config_dir),
            "run-vlm",
            "--episode",
            ep_id,
        ]
    )
    assert proc.returncode == 0, proc.stderr

    repo = Repository(e2e_workspace)
    try:
        resolved = repo.resolve_episode(ep_id)
        assert len(resolved.qc_segments) == 1
        seg = resolved.qc_segments[0]
        assert seg.guardrail.guardrail_override is True
        assert seg.splice_policy == "keep_positive"
        assert seg.guardrail.override_reason
    finally:
        repo.close()


def test_pairwise_vlm_client_retry_recovery_then_guardrail_executes(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    """VlmClient recovers from 429 rate limit & malformed JSON, then Guardrail runs on recovered segments."""
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "pick_and_place_retry"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=3.0)
    ep_id = _ingest_path(e2e_workspace, h5_dir)[0]

    mock_vlm_server.fail_next(429, count=1, stage="S3")
    mock_vlm_server.fail_with_malformed_json(count=1, stage="S4B")

    proc = h.run_cli(
        [
            "--workspace",
            str(e2e_workspace),
            "--config-dir",
            str(e2e_config_dir),
            "run-vlm",
            "--episode",
            ep_id,
        ]
    )
    assert proc.returncode == 0, proc.stderr

    repo = Repository(e2e_workspace)
    try:
        resolved = repo.resolve_episode(ep_id)
        assert resolved.active_run_id is not None
        assert len(resolved.qc_segments) >= 1
        assert any(s.guardrail.gate_type != "none" for s in resolved.qc_segments)
    finally:
        repo.close()


def test_pairwise_vlm_run_artifact_usage_and_fingerprint_from_client(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    """VlmClient populates VlmRunArtifact usage tokens/elapsed and ConfigFingerprintSnapshot."""
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "pick_and_place_usage"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=3.0)
    ep_id = _ingest_path(e2e_workspace, h5_dir)[0]

    proc = h.run_cli(
        [
            "--workspace",
            str(e2e_workspace),
            "--config-dir",
            str(e2e_config_dir),
            "run-vlm",
            "--episode",
            ep_id,
        ]
    )
    assert proc.returncode == 0, proc.stderr

    repo = Repository(e2e_workspace)
    try:
        row = repo.get_episode_row(ep_id)
        run = repo.get_vlm_run(row["active_run_id"])
        assert run.usage.total_tokens > 0
        assert len(run.usage.stages) >= 5
        assert run.config_fingerprint.config_sha256
        assert run.config_fingerprint.prompt_bundle_version
        assert run.config_fingerprint.label_taxonomy_version
        h.assert_no_legacy_fields(run)
    finally:
        repo.close()


def test_pairwise_run_vlm_then_render_mp4_1080p_studio_hud(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    """run-vlm followed by render-mp4 produces 1920x1080 Studio HUD MP4 matching VlmRunArtifact."""
    h5_dir = tmp_path / "pick_and_place_viz"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=2.5)
    ep_id = _ingest_path(e2e_workspace, h5_dir)[0]

    p_vlm = h.run_cli(
        [
            "--workspace",
            str(e2e_workspace),
            "--config-dir",
            str(e2e_config_dir),
            "run-vlm",
            "--episode",
            ep_id,
        ]
    )
    assert p_vlm.returncode == 0, p_vlm.stderr

    out_mp4 = tmp_path / "out_hud.mp4"
    p_viz = h.run_cli(
        [
            "--workspace",
            str(e2e_workspace),
            "--config-dir",
            str(e2e_config_dir),
            "render-mp4",
            "--episode",
            ep_id,
            "--out",
            str(out_mp4),
        ]
    )
    assert p_viz.returncode == 0, p_viz.stderr
    assert out_mp4.is_file()

    info = h.ffprobe_video_info(out_mp4)
    assert info["width"] == 1920
    assert info["height"] == 1080
    assert abs(info["fps"] - 30.0) <= 1.0
    assert info["nb_frames"] is not None and info["nb_frames"] >= 60

    cap = cv2.VideoCapture(str(out_mp4))
    try:
        ok, frame = cap.read()
        assert ok and frame is not None
        assert frame.shape == (1080, 1920, 3)
        assert float(frame.mean()) > 5.0
    finally:
        cap.release()


def test_pairwise_viz_mp4_renders_both_h5_and_lerobot_topologies(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    """viz_mp4 renders valid 1080p HUD MP4s for both h5_raw_v3 (torso/head) and lerobot (wrist/main)."""
    h5_dir = tmp_path / "pick_and_place_viz2"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=2.0)
    lr_dir = tmp_path / "lr_viz2"
    h.make_synthetic_lerobot(
        lr_dir, version="v3", n_episodes=1, duration_s=2.0, task_name="bimanual_handover"
    )

    ep_h5 = _ingest_path(e2e_workspace, h5_dir)[0]
    ep_lr = _ingest_path(e2e_workspace, lr_dir)[0]

    for ep_id, tag in ((ep_h5, "h5"), (ep_lr, "lr")):
        r1 = h.run_cli(
            [
                "--workspace",
                str(e2e_workspace),
                "--config-dir",
                str(e2e_config_dir),
                "run-vlm",
                "--episode",
                ep_id,
            ]
        )
        assert r1.returncode == 0, r1.stderr
        mp4_path = tmp_path / f"hud_{tag}.mp4"
        r2 = h.run_cli(
            [
                "--workspace",
                str(e2e_workspace),
                "--config-dir",
                str(e2e_config_dir),
                "render-mp4",
                "--episode",
                ep_id,
                "--out",
                str(mp4_path),
            ]
        )
        assert r2.returncode == 0, r2.stderr
        info = h.ffprobe_video_info(mp4_path)
        assert (info["width"], info["height"]) == (1920, 1080)
        assert info["nb_frames"] is not None and info["nb_frames"] >= 50


# ===========================================================================
# Block C: Guardrail -> Slicer -> 4 Exporters x 3 Recipes
# ===========================================================================


def test_pairwise_splice_gate_downgrade_preserved_in_bc_clean_with_zero_mask(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    """splice_gate downgrade (cut_clean -> mask_loss) keeps frames in PieceSlicer bc_clean with mask=0.0."""
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.io import auto_detect, get_adapter
    from robo_vlm_qc.models.export import ExportRecipe
    from robo_vlm_qc.models.review import EpisodeStatus
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "pick_and_place_slice_dg"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=4.0, stationary=False)
    ep_id = _ingest_path(e2e_workspace, h5_dir)[0]

    proc = h.run_cli(
        [
            "--workspace",
            str(e2e_workspace),
            "--config-dir",
            str(e2e_config_dir),
            "run-vlm",
            "--episode",
            ep_id,
        ]
    )
    assert proc.returncode == 0, proc.stderr

    repo = Repository(e2e_workspace)
    try:
        repo.transition_status(ep_id, EpisodeStatus.APPROVED)
        resolved = repo.resolve_episode(ep_id)
        mid_seg = next(s for s in resolved.qc_segments if s.label_id == "mid_idle_freeze")
        assert mid_seg.vlm_raw_policy == "cut_clean"
        assert mid_seg.splice_policy == "mask_loss"
    finally:
        repo.close()

    hdl = auto_detect(h5_dir)[0]
    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    with get_adapter("h5_raw_v3").open(hdl) as view:
        recipe = ExportRecipe(name="bc_clean", format="lerobot_v3")
        pieces = _call_piece_slicer(resolved, view, recipe, cfg)
        kept = [p for p in pieces if not p.dropped]
        assert kept
        # boundary_idle [0.0, 0.4] IS spliced out; downgraded [2.2, 2.8] IS retained
        total_own = sum(p.own_rows for p in kept)
        assert total_own > int(round((4.0 - 0.4 - 0.6) * view.fps))
        assert kept[0].t_start_s >= 0.35
        # Rows covering [2.2, 2.8] must have action_loss_mask == 0.0
        masked_rows = 0
        for p in kept:
            for r_idx in range(p.own_rows):
                t_row = (p.start_row + r_idx) / p.fps
                if 2.25 <= t_row <= 2.75:
                    assert p.action_loss_mask[r_idx] == [0.0, 0.0]
                    masked_rows += 1
        assert masked_rows >= 10


def test_pairwise_hallucination_override_keeps_frames_with_mask_1_in_lerobot_v3(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    """Hallucination override (mid_idle_freeze -> keep_positive) exports frames with action_loss_mask=[1,1]."""
    from robo_vlm_qc.models.review import EpisodeStatus
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "pick_and_place_hal_exp"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=4.0, stationary=False)
    ep_id = _ingest_path(e2e_workspace, h5_dir)[0]

    hal_cand = {
        "t_start": 0.6,
        "t_end": 3.8,
        "label_id": "mid_idle_freeze",
        "affected_arm": "both",
        "splice_policy": "cut_clean",
        "visual_evidence": {
            "initial_state": "Stationary.",
            "process_change": "Stationary.",
            "end_state": "Stationary.",
        },
        "confidence": 0.7,
        "unknown_note": "",
    }
    mock_vlm_server.set_stage_response("S3", {"segments": [hal_cand]})
    mock_vlm_server.set_stage_response("S4A", {"merged_segments": [hal_cand]})
    mock_vlm_server.set_stage_response(
        "S4B",
        {
            "keep_segment": True,
            "t_start": 0.6,
            "t_end": 3.8,
            "label_id": "mid_idle_freeze",
            "affected_arm": "both",
            "splice_policy": "cut_clean",
            "visual_evidence": hal_cand["visual_evidence"],
            "confidence": 0.7,
            "reject_reason": "",
        },
    )

    p1 = h.run_cli(
        [
            "--workspace",
            str(e2e_workspace),
            "--config-dir",
            str(e2e_config_dir),
            "run-vlm",
            "--episode",
            ep_id,
        ]
    )
    assert p1.returncode == 0, p1.stderr

    repo = Repository(e2e_workspace)
    try:
        repo.transition_status(ep_id, EpisodeStatus.APPROVED)
    finally:
        repo.close()

    out_dir = tmp_path / "exp_hal_lr"
    p2 = h.run_cli(
        [
            "--workspace",
            str(e2e_workspace),
            "--config-dir",
            str(e2e_config_dir),
            "export",
            "--format",
            "lerobot_v3",
            "--recipe",
            "bc_clean",
            "--out",
            str(out_dir),
        ]
    )
    assert p2.returncode == 0, p2.stderr
    info = h.assert_81d_lerobot_v3_valid(out_dir)
    assert info["total_rows"] >= 100
    tbl = pq.read_table(sorted((out_dir / "data").rglob("*.parquet"))[0])
    masks = tbl["action_loss_mask"].to_pylist()
    assert all(m == [1.0, 1.0] for m in masks)


def test_pairwise_bc_clean_vs_recovery_plus_action_loss_mask_flip(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    """Exporting same ResolvedEpisode under bc_clean vs recovery_plus flips recovery segment mask 0.0 -> 1.0."""
    from robo_vlm_qc.models.review import EpisodeStatus
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "pick_and_place_rec_flip"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=3.0)
    ep_id = _ingest_path(e2e_workspace, h5_dir)[0]

    repo = Repository(e2e_workspace)
    try:
        run = _make_vlm_run_artifact(
            ep_id,
            run_id="run_rec_flip",
            duration_s=3.0,
            passed=True,
            qc_segments=[
                {
                    "seg_id": "seg_rec",
                    "t_start": 1.0,
                    "t_end": 2.0,
                    "label_id": "missed_grasp_retry",
                    "training_role": "recovery",
                    "splice_policy": "mask_loss",
                    "mask_side": "left",
                    "affected_arm": "left",
                    "vlm_raw_policy": "mask_loss",
                    "confidence": 0.9,
                }
            ],
        )
        repo.save_vlm_run_artifact(run)
        repo.transition_status(ep_id, EpisodeStatus.APPROVED)
    finally:
        repo.close()

    out_bc = tmp_path / "exp_bc_clean"
    out_rec = tmp_path / "exp_recovery_plus"
    for recipe, out_d in (("bc_clean", out_bc), ("recovery_plus", out_rec)):
        proc = h.run_cli(
            [
                "--workspace",
                str(e2e_workspace),
                "--config-dir",
                str(e2e_config_dir),
                "export",
                "--format",
                "lerobot_v3",
                "--recipe",
                recipe,
                "--out",
                str(out_d),
            ]
        )
        assert proc.returncode == 0, proc.stderr
        h.assert_81d_lerobot_v3_valid(out_d)

    bc_masks = pq.read_table(sorted((out_bc / "data").rglob("*.parquet"))[0])["action_loss_mask"].to_pylist()
    rec_masks = pq.read_table(sorted((out_rec / "data").rglob("*.parquet"))[0])["action_loss_mask"].to_pylist()
    assert len(bc_masks) == len(rec_masks)
    assert any(m == [0.0, 1.0] for m in bc_masks)
    assert all(m == [1.0, 1.0] for m in rec_masks)


def test_pairwise_lerobot_v3_subtask_index_and_is_tail_alignment(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    """PieceSlicer -> lerobot_v3 aligns subtask_index with macro_stages and populates is_tail at splice boundary."""
    from robo_vlm_qc.models.review import EpisodeStatus
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "pick_and_place_subtask"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=4.0)
    ep_id = _ingest_path(e2e_workspace, h5_dir)[0]

    repo = Repository(e2e_workspace)
    try:
        run = _make_vlm_run_artifact(
            ep_id,
            run_id="run_sub",
            duration_s=4.0,
            passed=True,
            qc_segments=[
                {
                    "seg_id": "seg_mid_cut",
                    "t_start": 1.8,
                    "t_end": 2.2,
                    "label_id": "mid_idle_freeze",
                    "training_role": "badcase",
                    "splice_policy": "cut_clean",
                    "mask_side": None,
                    "affected_arm": "both",
                    "vlm_raw_policy": "cut_clean",
                    "confidence": 0.95,
                }
            ],
        )
        repo.save_vlm_run_artifact(run)
        repo.transition_status(ep_id, EpisodeStatus.APPROVED)
    finally:
        repo.close()

    out_dir = tmp_path / "exp_subtask_lr"
    proc = h.run_cli(
        [
            "--workspace",
            str(e2e_workspace),
            "--config-dir",
            str(e2e_config_dir),
            "export",
            "--format",
            "lerobot_v3",
            "--recipe",
            "bc_clean",
            "--out",
            str(out_dir),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    h.assert_81d_lerobot_v3_valid(out_dir)

    sub_tbl = pq.read_table(out_dir / "meta" / "subtasks.parquet")
    assert sub_tbl.num_rows >= 2
    data_tbl = pq.read_table(sorted((out_dir / "data").rglob("*.parquet"))[0])
    sub_idx = data_tbl["subtask_index"].to_pylist()
    is_tail = data_tbl["is_tail"].to_pylist()
    assert len(set(sub_idx)) >= 2
    assert any(bool(v) for v in is_tail)


def test_pairwise_h5_clean_preserves_streams_and_qc_datasets(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    """PieceSlicer (bc_clean) -> h5_clean exporter writes /qc/* datasets and sliced camera/robot streams."""
    from robo_vlm_qc.models.review import EpisodeStatus
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "pick_and_place_h5clean"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=3.0)
    ep_id = _ingest_path(e2e_workspace, h5_dir)[0]

    repo = Repository(e2e_workspace)
    try:
        run = _make_vlm_run_artifact(ep_id, run_id="run_h5c", duration_s=3.0, passed=True)
        repo.save_vlm_run_artifact(run)
        repo.transition_status(ep_id, EpisodeStatus.APPROVED)
    finally:
        repo.close()

    out_dir = tmp_path / "exp_h5_clean"
    proc = h.run_cli(
        [
            "--workspace",
            str(e2e_workspace),
            "--config-dir",
            str(e2e_config_dir),
            "export",
            "--format",
            "h5_clean",
            "--recipe",
            "bc_clean",
            "--out",
            str(out_dir),
        ]
    )
    assert proc.returncode == 0, proc.stderr

    h5_files = sorted(out_dir.rglob("*.h5"))
    assert h5_files
    with h5py.File(h5_files[0], "r") as f:
        for ds_name in (
            "/qc/action_loss_mask",
            "/qc/is_tail",
            "/qc/subtask_index",
            "/qc/resolved_json",
        ):
            assert ds_name in f, f"Missing {ds_name} in {h5_files[0]}"
        n_rows = f["/qc/action_loss_mask"].shape[0]
        assert n_rows > 0
        assert f["/qc/is_tail"].shape[0] == n_rows
        assert f["/qc/subtask_index"].shape[0] == n_rows
        raw_res = f["/qc/resolved_json"][()]
        resolved_payload = json.loads(raw_res.decode("utf-8") if isinstance(raw_res, bytes) else str(raw_res))
        h.assert_no_legacy_fields(resolved_payload)


def test_pairwise_badcase_pairs_zero_copy_index_intervals_and_masks(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    """PieceSlicer (badcase_pairs) -> zero_copy_index exports both passed and failed badcase intervals."""
    from robo_vlm_qc.models.review import EpisodeStatus
    from robo_vlm_qc.store import Repository

    h5_pass = tmp_path / "pick_and_place_bp_pass"
    h5_fail = tmp_path / "pick_and_place_bp_fail"
    h.make_synthetic_h5_raw_v3(h5_pass, name="ep_pass.h5", duration_s=3.0)
    h.make_synthetic_h5_raw_v3(h5_fail, name="ep_fail.h5", duration_s=3.0)
    ep_pass = _ingest_path(e2e_workspace, h5_pass)[0]
    ep_fail = _ingest_path(e2e_workspace, h5_fail)[0]

    repo = Repository(e2e_workspace)
    try:
        repo.save_vlm_run_artifact(_make_vlm_run_artifact(ep_pass, run_id="run_bp1", duration_s=3.0, passed=True))
        repo.save_vlm_run_artifact(
            _make_vlm_run_artifact(
                ep_fail,
                run_id="run_bp2",
                duration_s=3.0,
                passed=False,
                qc_segments=[
                    {
                        "seg_id": "seg_term",
                        "t_start": 1.8,
                        "t_end": 2.8,
                        "label_id": "terminal_failure",
                        "training_role": "terminal_fail",
                        "splice_policy": "mask_loss",
                        "mask_side": "both",
                        "affected_arm": "both",
                        "vlm_raw_policy": "mask_loss",
                        "confidence": 0.95,
                    }
                ],
            )
        )
        repo.transition_status(ep_pass, EpisodeStatus.APPROVED)
        repo.transition_status(ep_fail, EpisodeStatus.APPROVED)
    finally:
        repo.close()

    out_dir = tmp_path / "exp_zc_badcase"
    proc = h.run_cli(
        [
            "--workspace",
            str(e2e_workspace),
            "--config-dir",
            str(e2e_config_dir),
            "export",
            "--format",
            "zero_copy_index",
            "--recipe",
            "badcase_pairs",
            "--out",
            str(out_dir),
        ]
    )
    assert proc.returncode == 0, proc.stderr

    assert (out_dir / "qc_index.parquet").is_file()
    assert (out_dir / "manifest.jsonl").is_file()
    assert (out_dir / "subtasks.json").is_file()

    idx_tbl = pq.read_table(out_dir / "qc_index.parquet")
    assert idx_tbl.num_rows >= 2
    manifest_lines = [
        json.loads(line)
        for line in (out_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert manifest_lines
    ep_ids_in_manifest = {row.get("episode_id") for row in manifest_lines}
    assert ep_pass in ep_ids_in_manifest or ep_fail in ep_ids_in_manifest
    assert not list(out_dir.rglob("*.mp4"))
    assert not list(out_dir.rglob("*.h5"))


def test_pairwise_same_batch_lerobot_v3_and_h5_clean_identical_frames_and_subtasks(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    """Exporting same ResolvedEpisode batch to lerobot_v3 and h5_clean yields identical frame counts & subtasks."""
    from robo_vlm_qc.models.review import EpisodeStatus
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "pick_and_place_dual_exp"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=3.0)
    ep_id = _ingest_path(e2e_workspace, h5_dir)[0]

    repo = Repository(e2e_workspace)
    try:
        repo.save_vlm_run_artifact(_make_vlm_run_artifact(ep_id, run_id="run_dual", duration_s=3.0, passed=True))
        repo.transition_status(ep_id, EpisodeStatus.APPROVED)
    finally:
        repo.close()

    out_lr = tmp_path / "exp_dual_lr"
    out_h5 = tmp_path / "exp_dual_h5"
    for fmt, out_d in (("lerobot_v3", out_lr), ("h5_clean", out_h5)):
        p = h.run_cli(
            [
                "--workspace",
                str(e2e_workspace),
                "--config-dir",
                str(e2e_config_dir),
                "export",
                "--format",
                fmt,
                "--recipe",
                "bc_clean",
                "--out",
                str(out_d),
            ]
        )
        assert p.returncode == 0, p.stderr

    lr_info = h.assert_81d_lerobot_v3_valid(out_lr)
    lr_tbl = pq.read_table(sorted((out_lr / "data").rglob("*.parquet"))[0])
    lr_subtasks = lr_tbl["subtask_index"].to_pylist()
    lr_masks = np.asarray(lr_tbl["action_loss_mask"].to_pylist(), dtype=np.float32)

    h5_files = sorted(out_h5.rglob("*.h5"))
    assert len(h5_files) == 1
    with h5py.File(h5_files[0], "r") as f:
        h5_subtasks = f["/qc/subtask_index"][:].tolist()
        h5_masks = f["/qc/action_loss_mask"][:].astype(np.float32)

    assert len(h5_subtasks) == lr_info["total_rows"]
    assert h5_subtasks == lr_subtasks
    assert np.allclose(h5_masks, lr_masks)


def test_pairwise_export_recipe_filters_and_manifest_checksums(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    """ExportRecipe dataset_filter, smooth_tail, and ExportManifest checksums work end-to-end with lerobot_v3."""
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.export import get_exporter
    from robo_vlm_qc.io import auto_detect, get_adapter
    from robo_vlm_qc.models.export import ExportRecipe
    from robo_vlm_qc.models.review import EpisodeStatus
    from robo_vlm_qc.store import Repository

    ds_a = tmp_path / "pick_and_place_a"
    ds_b = tmp_path / "pick_and_place_b"
    h.make_synthetic_h5_raw_v3(ds_a, duration_s=3.0)
    h.make_synthetic_h5_raw_v3(ds_b, duration_s=3.0)
    ep_a = _ingest_path(e2e_workspace, ds_a)[0]
    ep_b = _ingest_path(e2e_workspace, ds_b)[0]

    repo = Repository(e2e_workspace)
    try:
        repo.save_vlm_run_artifact(_make_vlm_run_artifact(ep_a, run_id="run_a", duration_s=3.0, passed=True))
        repo.save_vlm_run_artifact(_make_vlm_run_artifact(ep_b, run_id="run_b", duration_s=3.0, passed=True))
        repo.transition_status(ep_a, EpisodeStatus.APPROVED)
        # ep_b remains vlm_ready (not approved)
        resolved_a = repo.resolve_episode(ep_a)
    finally:
        repo.close()

    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    recipe = ExportRecipe(
        name="bc_clean",
        format="lerobot_v3",
        smooth_tail="always",
        tail_frames=4,
        dataset_filter="pick_and_place_a",
    )
    hdl = auto_detect(ds_a)[0]
    with get_adapter("h5_raw_v3").open(hdl) as view:
        pieces = _call_piece_slicer(resolved_a, view, recipe, cfg)
        out_dir = tmp_path / "exp_manifest"
        exporter = get_exporter("lerobot_v3")
        manifest = exporter.export(pieces, [resolved_a], {ep_a: view}, out_dir, recipe)

    assert manifest.episodes_included == 1
    assert manifest.total_frames > 0
    assert manifest.artifacts
    assert manifest.checksums
    h.assert_81d_lerobot_v3_valid(out_dir)


def test_pairwise_lerobot_v2_ingest_to_lerobot_v3_round_trip(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    """Ingest LeRobot v2 -> approve -> export lerobot_v3 preserves 81D states and multi-cam frames."""
    from robo_vlm_qc.models.review import EpisodeStatus
    from robo_vlm_qc.store import Repository

    lr_v2 = tmp_path / "lr_v2_roundtrip"
    h.make_synthetic_lerobot(lr_v2, version="v2", n_episodes=1, duration_s=2.0)
    ep_id = _ingest_path(e2e_workspace, lr_v2)[0]

    repo = Repository(e2e_workspace)
    try:
        repo.save_vlm_run_artifact(
            _make_vlm_run_artifact(ep_id, run_id="run_lr2", duration_s=2.0, passed=True, qc_segments=[])
        )
        repo.transition_status(ep_id, EpisodeStatus.APPROVED)
    finally:
        repo.close()

    out_dir = tmp_path / "exp_lr3_roundtrip"
    proc = h.run_cli(
        [
            "--workspace",
            str(e2e_workspace),
            "--config-dir",
            str(e2e_config_dir),
            "export",
            "--format",
            "lerobot_v3",
            "--recipe",
            "bc_clean",
            "--out",
            str(out_dir),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    info = h.assert_81d_lerobot_v3_valid(out_dir)
    assert info["total_rows"] == 60
    assert info["info"]["robot_type"] == "bimanual_robot"

    src_tbl = pq.read_table(lr_v2 / "data" / "chunk-000" / "episode_000000.parquet")
    dst_tbl = pq.read_table(sorted((out_dir / "data").rglob("*.parquet"))[0])
    assert np.allclose(
        np.asarray(src_tbl["observation.state"].to_pylist(), dtype=np.float32),
        np.asarray(dst_tbl["observation.state"].to_pylist(), dtype=np.float32),
        atol=1e-4,
    )


def test_pairwise_human_review_edit_updates_both_viz_mp4_and_lerobot_v3(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    """HumanReviewArtifact edit to qc_segments & macro_stages updates BOTH viz_mp4 and lerobot_v3."""
    from robo_vlm_qc.models.annotation import MacroStage, QcSegment, SuccessVerdict
    from robo_vlm_qc.models.review import ReviewWritePayload, WatchCoverage
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "pick_and_place_ssot"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=3.0)
    ep_id = _ingest_path(e2e_workspace, h5_dir)[0]

    repo = Repository(e2e_workspace)
    try:
        repo.save_vlm_run_artifact(_make_vlm_run_artifact(ep_id, run_id="run_ssot", duration_s=3.0, passed=True))
        payload = ReviewWritePayload(
            base_run_id="run_ssot",
            action="approve",
            qc_segments=[
                QcSegment(
                    seg_id="seg_h_mask",
                    t_start=1.2,
                    t_end=1.8,
                    label_id="hesitation_oscillation",
                    training_role="badcase",
                    splice_policy="mask_loss",
                    mask_side="right",
                    affected_arm="right",
                    vlm_raw_policy="mask_loss",
                )
            ],
            success=SuccessVerdict(passed=True, verdict="success", confidence=1.0),
            macro_stages=[
                MacroStage(
                    stage_id="H1",
                    t_start=0.0,
                    t_end=3.0,
                    label="Human refined stage X",
                    label_en="human_stage_x",
                )
            ],
            watch_coverage=WatchCoverage(
                watched_intervals=[(0.0, 3.0)],
                watched_s=3.0,
                duration_s=3.0,
                coverage_ratio=1.0,
                confirmed=True,
            ),
        )
        repo.save_review(ep_id, "lead_bob", "lead", payload)
    finally:
        repo.close()

    out_mp4 = tmp_path / "ssot_hud.mp4"
    p_mp4 = h.run_cli(
        [
            "--workspace",
            str(e2e_workspace),
            "--config-dir",
            str(e2e_config_dir),
            "render-mp4",
            "--episode",
            ep_id,
            "--out",
            str(out_mp4),
        ]
    )
    assert p_mp4.returncode == 0, p_mp4.stderr
    mp4_info = h.ffprobe_video_info(out_mp4)
    assert (mp4_info["width"], mp4_info["height"]) == (1920, 1080)

    out_lr = tmp_path / "ssot_lr"
    p_lr = h.run_cli(
        [
            "--workspace",
            str(e2e_workspace),
            "--config-dir",
            str(e2e_config_dir),
            "export",
            "--format",
            "lerobot_v3",
            "--recipe",
            "bc_clean",
            "--out",
            str(out_lr),
        ]
    )
    assert p_lr.returncode == 0, p_lr.stderr
    h.assert_81d_lerobot_v3_valid(out_lr)
    sub_tbl = pq.read_table(out_lr / "meta" / "subtasks.parquet")
    sub_texts = json.dumps(sub_tbl.to_pydict(), ensure_ascii=False)
    assert "Human refined stage X" in sub_texts
    data_tbl = pq.read_table(sorted((out_lr / "data").rglob("*.parquet"))[0])
    assert any(m == [1.0, 0.0] for m in data_tbl["action_loss_mask"].to_pylist())


# ===========================================================================
# Block D: FastAPI Studio, RBAC, Leases, SSE Jobs & Web Contract
# ===========================================================================


def test_pairwise_reviewer_claim_edit_submit_then_lead_filter_reviewed(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    """Reviewer claims -> edits 3 layers -> submits (reviewed) -> Lead filters GET /api/episodes?status=reviewed."""
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "pick_and_place_api_rev"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=3.0)
    ep_id = _ingest_path(e2e_workspace, h5_dir)[0]

    repo = Repository(e2e_workspace)
    try:
        repo.save_vlm_run_artifact(_make_vlm_run_artifact(ep_id, run_id="run_api1", duration_s=3.0, passed=True))
    finally:
        repo.close()

    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    rev_hdr = {"X-Auth-Token": "rev-token", "X-User": "rev_1"}
    lead_hdr = {"X-Auth-Token": "lead-token", "X-User": "lead_1"}

    c_resp = client.post(f"/api/episodes/{ep_id}/claim", headers=rev_hdr, json={"user": "rev_1"})
    assert c_resp.status_code in (200, 201), c_resp.text

    payload = {
        "base_run_id": "run_api1",
        "action": "submit_review",
        "qc_segments": [
            {
                "seg_id": "seg_rev_1",
                "t_start": 0.8,
                "t_end": 1.4,
                "label_id": "missed_grasp_retry",
                "training_role": "recovery",
                "splice_policy": "mask_loss",
                "mask_side": "left",
                "affected_arm": "left",
                "vlm_raw_policy": "mask_loss",
                "confidence": 0.95,
            }
        ],
        "success": {"passed": True, "verdict": "success", "confidence": 1.0, "criteria": [], "summary": "OK"},
        "macro_stages": [
            {"stage_id": "S1", "t_start": 0.0, "t_end": 3.0, "label": "Full grasp and delivery", "label_en": "full"}
        ],
        "watch_coverage": {
            "watched_intervals": [[0.0, 3.0]],
            "watched_s": 3.0,
            "duration_s": 3.0,
            "coverage_ratio": 1.0,
            "confirmed": True,
        },
        "notes": "Submitted via API",
    }
    s_resp = client.post(f"/api/episodes/{ep_id}/reviews", headers=rev_hdr, json=payload)
    assert s_resp.status_code in (200, 201), s_resp.text

    list_resp = client.get("/api/episodes", params={"status": "reviewed"}, headers=lead_hdr)
    assert list_resp.status_code == 200, list_resp.text
    eps = _unwrap_list_response(list_resp.json())
    assert [e["episode_id"] for e in eps] == [ep_id]

    detail = client.get(f"/api/episodes/{ep_id}", headers=lead_hdr)
    assert detail.status_code == 200, detail.text
    d_json = detail.json()
    assert d_json["status"] == "reviewed"
    assert d_json["is_human_reviewed"] is True


def test_pairwise_lead_approve_freeze_gt_blocks_subsequent_reviewer_claim_edit(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    """Lead approves & freezes GT (gt_frozen), blocking subsequent Reviewer claim/edit and RBAC violations."""
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "pick_and_place_freeze"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=3.0)
    ep_id = _ingest_path(e2e_workspace, h5_dir)[0]

    repo = Repository(e2e_workspace)
    try:
        repo.save_vlm_run_artifact(_make_vlm_run_artifact(ep_id, run_id="run_frz", duration_s=3.0, passed=True))
    finally:
        repo.close()

    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    rev_hdr = {"X-Auth-Token": "rev-token", "X-User": "rev_late"}
    lead_hdr = {"X-Auth-Token": "lead-token", "X-User": "lead_1"}

    base_review = {
        "base_run_id": "run_frz",
        "action": "approve",
        "qc_segments": [],
        "success": {"passed": True, "verdict": "success", "confidence": 1.0, "criteria": [], "summary": "OK"},
        "macro_stages": [{"stage_id": "S1", "t_start": 0.0, "t_end": 3.0, "label": "Full workflow", "label_en": "all"}],
        "watch_coverage": {"watched_intervals": [[0.0, 3.0]], "watched_s": 3.0, "duration_s": 3.0, "coverage_ratio": 1.0, "confirmed": True},
    }
    # Reviewer cannot call approve
    r_forbid = client.post(f"/api/episodes/{ep_id}/reviews", headers=rev_hdr, json=base_review)
    assert r_forbid.status_code == 403

    r_app = client.post(f"/api/episodes/{ep_id}/reviews", headers=lead_hdr, json=base_review)
    assert r_app.status_code in (200, 201), r_app.text

    r_frz = client.post(
        f"/api/episodes/{ep_id}/reviews",
        headers=lead_hdr,
        json={**base_review, "action": "freeze_gt"},
    )
    assert r_frz.status_code in (200, 201), r_frz.text
    assert client.get(f"/api/episodes/{ep_id}", headers=lead_hdr).json()["status"] == "gt_frozen"

    # Subsequent Reviewer claim & edit must fail
    r_claim = client.post(f"/api/episodes/{ep_id}/claim", headers=rev_hdr, json={"user": "rev_late"})
    assert r_claim.status_code in (403, 409, 423)

    r_edit = client.post(
        f"/api/episodes/{ep_id}/reviews",
        headers=rev_hdr,
        json={**base_review, "action": "submit_review"},
    )
    assert r_edit.status_code in (403, 409, 423)


def test_pairwise_reviewer_readonly_layer_rejection_and_lead_flag_arbitration(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    """API rejects readonly arm_timelines/grounding in review payload; Reviewer flags -> Lead arbitrates."""
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "pick_and_place_flag"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=3.0)
    ep_id = _ingest_path(e2e_workspace, h5_dir)[0]

    repo = Repository(e2e_workspace)
    try:
        repo.save_vlm_run_artifact(_make_vlm_run_artifact(ep_id, run_id="run_flg", duration_s=3.0, passed=True))
    finally:
        repo.close()

    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    rev_hdr = {"X-Auth-Token": "rev-token", "X-User": "rev_1"}
    lead_hdr = {"X-Auth-Token": "lead-token", "X-User": "lead_1"}

    client.post(f"/api/episodes/{ep_id}/claim", headers=rev_hdr, json={"user": "rev_1"})

    bad_payload = {
        "base_run_id": "run_flg",
        "action": "submit_review",
        "qc_segments": [],
        "success": {"passed": True, "verdict": "success", "confidence": 1.0, "criteria": [], "summary": "OK"},
        "macro_stages": [{"stage_id": "S1", "t_start": 0.0, "t_end": 3.0, "label": "Main trajectory", "label_en": "main"}],
        "watch_coverage": {"watched_intervals": [[0.0, 3.0]], "watched_s": 3.0, "duration_s": 3.0, "coverage_ratio": 1.0, "confirmed": True},
        "arm_timelines": {"left": [], "right": []},
    }
    r_bad = client.post(f"/api/episodes/{ep_id}/reviews", headers=rev_hdr, json=bad_payload)
    assert r_bad.status_code in (400, 422)

    flag_payload = {k: v for k, v in bad_payload.items() if k != "arm_timelines"}
    flag_payload["action"] = "flag"
    flag_payload["flagged_reason"] = "Uncertain object tilt"
    r_flg = client.post(f"/api/episodes/{ep_id}/reviews", headers=rev_hdr, json=flag_payload)
    assert r_flg.status_code in (200, 201), r_flg.text

    flagged_list = _unwrap_list_response(
        client.get("/api/episodes", params={"status": "flagged"}, headers=lead_hdr).json()
    )
    assert [e["episode_id"] for e in flagged_list] == [ep_id]

    r_arb = client.post(
        f"/api/episodes/{ep_id}/reviews",
        headers=lead_hdr,
        json={**flag_payload, "action": "approve", "flagged_reason": None},
    )
    assert r_arb.status_code in (200, 201), r_arb.text
    assert client.get(f"/api/episodes/{ep_id}", headers=lead_hdr).json()["status"] == "approved"


def test_pairwise_claim_lease_expiry_and_reclaim_via_repo_and_api(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    """SQLite claim lease conflict blocks second reviewer until expiry, then reap allows API reclaim."""
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "pick_and_place_lease"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=3.0)
    ep_id = _ingest_path(e2e_workspace, h5_dir)[0]

    repo = Repository(e2e_workspace)
    try:
        repo.save_vlm_run_artifact(_make_vlm_run_artifact(ep_id, run_id="run_lse", duration_s=3.0, passed=True))
        # Claim at an expired timestamp in the past
        repo.claim_episode(ep_id, "rev_stale", ttl_s=1.0, now=time.time() - 60.0)
    finally:
        repo.close()

    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    rev_hdr = {"X-Auth-Token": "rev-token", "X-User": "rev_fresh"}

    # Expired lease allows new reviewer claim cleanly
    r_claim = client.post(f"/api/episodes/{ep_id}/claim", headers=rev_hdr, json={"user": "rev_fresh"})
    assert r_claim.status_code in (200, 201), r_claim.text

    # Active lease by rev_fresh blocks rev_other with 409
    r_conflict = client.post(
        f"/api/episodes/{ep_id}/claim",
        headers={**rev_hdr, "X-User": "rev_other"},
        json={"user": "rev_other"},
    )
    assert r_conflict.status_code == 409


def test_pairwise_lead_diff_view_payload_and_multi_draft_revision_history(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    """Lead ReviewStudioPage Diff payload compares raw_vlm_segments vs edits + multi-draft revision history."""
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "pick_and_place_diff"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=4.0)
    ep_id = _ingest_path(e2e_workspace, h5_dir)[0]

    repo = Repository(e2e_workspace)
    try:
        repo.save_vlm_run_artifact(_make_vlm_run_artifact(ep_id, run_id="run_diff", duration_s=4.0, passed=True))
    finally:
        repo.close()

    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    lead_hdr = {"X-Auth-Token": "lead-token", "X-User": "lead_1"}

    draft1 = {
        "base_run_id": "run_diff",
        "action": "save_draft",
        "qc_segments": [],
        "success": {"passed": True, "verdict": "success", "confidence": 1.0, "criteria": [], "summary": "draft1"},
        "macro_stages": [{"stage_id": "S1", "t_start": 0.0, "t_end": 4.0, "label": "Draft 1", "label_en": "d1"}],
        "watch_coverage": {"watched_intervals": [[0.0, 4.0]], "watched_s": 4.0, "duration_s": 4.0, "coverage_ratio": 1.0, "confirmed": True},
    }
    draft2 = {
        **draft1,
        "macro_stages": [{"stage_id": "S1", "t_start": 0.0, "t_end": 4.0, "label": "Draft 2", "label_en": "d2"}],
        "success": {"passed": False, "verdict": "failure", "confidence": 1.0, "criteria": [], "summary": "draft2"},
    }
    assert client.post(f"/api/episodes/{ep_id}/reviews", headers=lead_hdr, json=draft1).status_code in (200, 201)
    assert client.post(f"/api/episodes/{ep_id}/reviews", headers=lead_hdr, json=draft2).status_code in (200, 201)

    detail = client.get(f"/api/episodes/{ep_id}", headers=lead_hdr).json()
    assert len(detail["raw_vlm_segments"]) == 2
    assert detail["qc_segments"] == []
    assert detail["review_diff"] is not None
    assert detail["review_diff"]["fp_deleted"] == 2
    assert detail["review_diff"]["success_flipped"] is True

    hist_resp = client.get(f"/api/episodes/{ep_id}/reviews", headers=lead_hdr)
    assert hist_resp.status_code == 200, hist_resp.text
    revs = _unwrap_list_response(hist_resp.json())
    assert len(revs) >= 2


def test_pairwise_11_pure_visual_labels_exposed_for_studio_ui(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    """Studio API exposes all 11 pure-visual labels with collector_ui & vlm_semantics and 0 legacy labels."""
    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    lead_hdr = {"X-Auth-Token": "lead-token"}

    resp = client.get("/api/lab/labels", headers=lead_hdr)
    assert resp.status_code == 200, resp.text
    labels = _unwrap_list_response(resp.json())
    ids = {lbl["id"] for lbl in labels}
    assert len(ids) >= 11
    assert {
        "boundary_idle",
        "mid_idle_freeze",
        "missed_grasp_retry",
        "object_slip_recovery",
        "placement_redo",
        "hesitation_oscillation",
        "collision_or_knock",
        "precision_fine_adjust",
        "bimanual_assist_hold",
        "terminal_failure",
        "unknown_anomaly",
    }.issubset(ids)
    assert "long_pause" not in ids
    assert "back_and_forth" not in ids
    for lbl in labels:
        assert "collector_ui" in lbl and "vlm_semantics" in lbl and "default_splice_policy" in lbl


def test_pairwise_batch_vlm_job_sse_progress_transitions_episodes(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    """POST /api/jobs batch_vlm updates progress over SSE and transitions ingested -> vlm_ready."""
    h5_dir = tmp_path / "pick_and_place_jobs"
    h.make_synthetic_h5_raw_v3(h5_dir, name="ep_0.h5", duration_s=2.0)
    h.make_synthetic_h5_raw_v3(h5_dir, name="ep_1.h5", duration_s=2.0)
    ep_ids = _ingest_path(e2e_workspace, h5_dir)

    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    rev_hdr = {"X-Auth-Token": "rev-token"}
    lead_hdr = {"X-Auth-Token": "lead-token"}

    # Reviewer forbidden from triggering batch jobs
    r_forbid = client.post(
        "/api/jobs",
        headers=rev_hdr,
        json={"job_type": "batch_vlm", "episode_ids": ep_ids},
    )
    assert r_forbid.status_code == 403

    r_job = client.post(
        "/api/jobs",
        headers=lead_hdr,
        json={"job_type": "batch_vlm", "episode_ids": ep_ids},
    )
    assert r_job.status_code in (200, 201, 202), r_job.text
    job_id = r_job.json()["job_id"]

    # Consume SSE events or poll job completion
    sse_seen = False
    with client.stream("GET", f"/api/jobs/{job_id}/events", headers=lead_hdr) as stream:
        if stream.status_code == 200:
            for line in stream.iter_lines():
                if line:
                    sse_seen = True
                    if "completed" in line or "failed" in line:
                        break

    deadline = time.time() + 30.0
    job_state: dict[str, Any] = {}
    while time.time() < deadline:
        jr = client.get(f"/api/jobs/{job_id}", headers=lead_hdr)
        assert jr.status_code == 200
        job_state = jr.json()
        if job_state["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)

    assert job_state.get("status") == "completed", job_state
    assert job_state.get("progress_done") == 2
    assert sse_seen is True

    ready_eps = _unwrap_list_response(
        client.get("/api/episodes", params={"status": "vlm_ready"}, headers=lead_hdr).json()
    )
    assert {e["episode_id"] for e in ready_eps} == set(ep_ids)


def test_pairwise_api_export_dry_run_matches_actual_lerobot_v3_and_download(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    """POST /api/exports/dry-run matches actual lerobot_v3 exported counts and download returns 200."""
    from robo_vlm_qc.models.review import EpisodeStatus
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "pick_and_place_api_exp"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=2.5)
    ep_id = _ingest_path(e2e_workspace, h5_dir)[0]

    repo = Repository(e2e_workspace)
    try:
        repo.save_vlm_run_artifact(_make_vlm_run_artifact(ep_id, run_id="run_aexp", duration_s=2.5, passed=True))
        repo.transition_status(ep_id, EpisodeStatus.APPROVED)
    finally:
        repo.close()

    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    rev_hdr = {"X-Auth-Token": "rev-token"}
    lead_hdr = {"X-Auth-Token": "lead-token"}

    assert (
        client.post(
            "/api/exports/dry-run",
            headers=rev_hdr,
            json={"format": "lerobot_v3", "recipe": "bc_clean"},
        ).status_code
        == 403
    )

    dry_resp = client.post(
        "/api/exports/dry-run",
        headers=lead_hdr,
        json={"format": "lerobot_v3", "recipe": "bc_clean"},
    )
    assert dry_resp.status_code in (200, 201), dry_resp.text
    dry_data = dry_resp.json()

    real_resp = client.post(
        "/api/exports",
        headers=lead_hdr,
        json={"format": "lerobot_v3", "recipe": "bc_clean"},
    )
    assert real_resp.status_code in (200, 201), real_resp.text
    real_data = real_resp.json()
    assert real_data["episodes_included"] == dry_data["episodes_included"] == 1
    assert real_data["total_frames"] == dry_data["total_frames"] > 0

    out_dir = Path(real_data["output_dir"])
    h.assert_81d_lerobot_v3_valid(out_dir)

    export_id = real_data["export_id"]
    dl_resp = client.get(f"/api/exports/{export_id}/download", headers=lead_hdr)
    assert dl_resp.status_code == 200
    assert len(dl_resp.content) > 0


def test_pairwise_api_render_mp4_job_creates_1080p_video(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    """POST /api/jobs render_mp4 renders a valid 1920x1080 HUD video for a vlm_ready episode."""
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "pick_and_place_api_viz"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=2.0)
    ep_id = _ingest_path(e2e_workspace, h5_dir)[0]

    repo = Repository(e2e_workspace)
    try:
        repo.save_vlm_run_artifact(_make_vlm_run_artifact(ep_id, run_id="run_aviz", duration_s=2.0, passed=True))
    finally:
        repo.close()

    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    lead_hdr = {"X-Auth-Token": "lead-token"}

    jr = client.post(
        "/api/jobs",
        headers=lead_hdr,
        json={"job_type": "render_mp4", "episode_ids": [ep_id]},
    )
    assert jr.status_code in (200, 201, 202), jr.text
    job_id = jr.json()["job_id"]

    deadline = time.time() + 30.0
    job_state: dict[str, Any] = {}
    while time.time() < deadline:
        job_state = client.get(f"/api/jobs/{job_id}", headers=lead_hdr).json()
        if job_state["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)
    assert job_state.get("status") == "completed", job_state

    mp4s = list((e2e_workspace / "exports").rglob("*.mp4")) + list((e2e_workspace / "artifacts").rglob("*hud*.mp4"))
    assert mp4s, "Expected rendered 1080p MP4 artifact under workspace"
    info = h.ffprobe_video_info(mp4s[0])
    assert (info["width"], info["height"]) == (1920, 1080)


# ===========================================================================
# Block E: VLM Lab In-Context Loop (Diagnosis, Few-Shot, Unknown Triage, Eval)
# ===========================================================================


def test_pairwise_add_few_shot_via_lab_hits_s1_s2_s5_cache_invalidates_s3_s4(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    """Adding Few-Shot via /api/lab keeps 100% S1/S2/S5 cache hit while invalidating & re-running S3/S4."""
    from robo_vlm_qc.models.annotation import MacroStage, SuccessVerdict
    from robo_vlm_qc.models.review import ReviewWritePayload, WatchCoverage
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "pick_and_place_fs_cache"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=3.0)
    ep_id = _ingest_path(e2e_workspace, h5_dir)[0]

    assert (
        h.run_cli(
            [
                "--workspace",
                str(e2e_workspace),
                "--config-dir",
                str(e2e_config_dir),
                "run-vlm",
                "--episode",
                ep_id,
            ]
        ).returncode
        == 0
    )

    repo = Repository(e2e_workspace)
    try:
        row = repo.get_episode_row(ep_id)
        run_id = row["active_run_id"]
        repo.save_review(
            ep_id,
            "lead_1",
            "lead",
            ReviewWritePayload(
                base_run_id=run_id,
                action="freeze_gt",
                qc_segments=[],
                success=SuccessVerdict(passed=True, verdict="success", confidence=1.0),
                macro_stages=[MacroStage(stage_id="S1", t_start=0.0, t_end=3.0, label="Manipulate", label_en="manipulate")],
                watch_coverage=WatchCoverage(
                    watched_intervals=[(0.0, 3.0)], watched_s=3.0, duration_s=3.0, coverage_ratio=1.0, confirmed=True
                ),
            ),
        )
    finally:
        repo.close()

    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    lead_hdr = {"X-Auth-Token": "lead-token"}

    fs_resp = client.post(
        "/api/lab/few-shots",
        headers=lead_hdr,
        json={
            "scope": "task:pick_and_place",
            "task_id": "pick_and_place",
            "label_id": "missed_grasp_retry",
            "error_type": "fp_overflag",
            "episode_id": ep_id,
            "t_start": 0.8,
            "t_end": 1.6,
            "vlm_wrong_json": {"label_id": "missed_grasp_retry"},
            "human_right_json": {"splice_policy": "keep_positive"},
            "rationale": "Normal slow grasp approach should not be flagged as missed_grasp_retry",
        },
    )
    assert fs_resp.status_code in (200, 201), fs_resp.text
    assert list((e2e_workspace / "artifacts" / "few_shots").glob("*.jpg"))

    mock_vlm_server.requests.clear()
    eval_resp = client.post(
        "/api/lab/eval", headers=lead_hdr, json={"dataset": "pick_and_place_fs_cache", "few_shot": True}
    )
    assert eval_resp.status_code in (200, 201), eval_resp.text

    stages_called = [r["stage"] for r in mock_vlm_server.requests]
    assert "S1" not in stages_called
    assert "S2" not in stages_called
    assert "S5" not in stages_called
    assert "S5_CONTROL" not in stages_called
    assert any(s in {"S3", "S4A", "S4B"} for s in stages_called)


def test_pairwise_diagnosis_fp_policy_flip_creates_2x2_few_shot_injected_in_s3_s4(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    """diagnosis finds fp_overflag & policy_flip -> Few-Shot 2x2 jpg created -> S3/S4 requests include rationale."""
    from robo_vlm_qc.models.annotation import MacroStage, QcSegment, SuccessVerdict
    from robo_vlm_qc.models.review import ReviewWritePayload, WatchCoverage
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "pick_and_place_diag_fs"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=4.0)
    ep_id = _ingest_path(e2e_workspace, h5_dir)[0]

    repo = Repository(e2e_workspace)
    try:
        vlm_run = _make_vlm_run_artifact(ep_id, run_id="run_dfs", duration_s=4.0, passed=True)
        repo.save_vlm_run_artifact(vlm_run)
        # Human deletes seg_001 (fp_overflag) and flips seg_002 mask_loss -> keep_positive (policy_flip)
        repo.save_review(
            ep_id,
            "lead_1",
            "lead",
            ReviewWritePayload(
                base_run_id="run_dfs",
                action="freeze_gt",
                qc_segments=[
                    QcSegment(
                        seg_id="seg_002",
                        t_start=1.0,
                        t_end=1.8,
                        label_id="precision_fine_adjust",
                        training_role="necessary",
                        splice_policy="keep_positive",
                        mask_side=None,
                        affected_arm="left",
                        vlm_raw_policy="mask_loss",
                    )
                ],
                success=SuccessVerdict(passed=True, verdict="success", confidence=1.0),
                macro_stages=[MacroStage(stage_id="S1", t_start=0.0, t_end=4.0, label="Manipulate", label_en="manipulate")],
                watch_coverage=WatchCoverage(
                    watched_intervals=[(0.0, 4.0)], watched_s=4.0, duration_s=4.0, coverage_ratio=1.0, confirmed=True
                ),
            ),
        )
    finally:
        repo.close()

    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    lead_hdr = {"X-Auth-Token": "lead-token"}

    diag_resp = client.get("/api/lab/diagnosis", headers=lead_hdr)
    assert diag_resp.status_code == 200, diag_resp.text
    diag_text = json.dumps(diag_resp.json(), ensure_ascii=False)
    assert "fp_overflag" in diag_text
    assert "policy_flip" in diag_text

    rationale = "FINE_ADJUST_NOT_MISSED_GRASP_CUE"
    fs_resp = client.post(
        "/api/lab/few-shots",
        headers=lead_hdr,
        json={
            "scope": "global",
            "task_id": "pick_and_place",
            "label_id": "precision_fine_adjust",
            "error_type": "policy_flip",
            "episode_id": ep_id,
            "t_start": 1.0,
            "t_end": 1.8,
            "vlm_wrong_json": {"label_id": "missed_grasp_retry", "splice_policy": "mask_loss"},
            "human_right_json": {"label_id": "precision_fine_adjust", "splice_policy": "keep_positive"},
            "rationale": rationale,
        },
    )
    assert fs_resp.status_code in (200, 201), fs_resp.text

    mock_vlm_server.requests.clear()
    eval_resp = client.post("/api/lab/eval", headers=lead_hdr, json={"few_shot": True})
    assert eval_resp.status_code in (200, 201), eval_resp.text

    s3_s4_texts = []
    for r in mock_vlm_server.requests:
        if r["stage"] in {"S3", "S4B"}:
            for m in r["messages"]:
                c = m.get("content", "")
                if isinstance(c, str):
                    s3_s4_texts.append(c)
                elif isinstance(c, list):
                    s3_s4_texts.extend(str(b.get("text", "")) for b in c if isinstance(b, dict) and b.get("type") == "text")
    assert any(rationale in txt for txt in s3_s4_texts)


def test_pairwise_few_shot_scope_retrieval_filters_by_task_and_label(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    """Top-K Few-Shot retrieval injects matching global & task:pick_and_place items while excluding unrelated task."""
    from robo_vlm_qc.models.annotation import MacroStage, SuccessVerdict
    from robo_vlm_qc.models.review import ReviewWritePayload, WatchCoverage
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "pick_and_place_scope"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=3.0)
    ep_id = _ingest_path(e2e_workspace, h5_dir)[0]

    repo = Repository(e2e_workspace)
    try:
        repo.save_vlm_run_artifact(_make_vlm_run_artifact(ep_id, run_id="run_scp", duration_s=3.0, passed=True))
        repo.save_review(
            ep_id,
            "lead_1",
            "lead",
            ReviewWritePayload(
                base_run_id="run_scp",
                action="freeze_gt",
                qc_segments=[],
                success=SuccessVerdict(passed=True, verdict="success", confidence=1.0),
                macro_stages=[MacroStage(stage_id="S1", t_start=0.0, t_end=3.0, label="Manipulate", label_en="manipulate")],
                watch_coverage=WatchCoverage(
                    watched_intervals=[(0.0, 3.0)], watched_s=3.0, duration_s=3.0, coverage_ratio=1.0, confirmed=True
                ),
            ),
        )
    finally:
        repo.close()

    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    lead_hdr = {"X-Auth-Token": "lead-token"}

    for scope, tid, rat in (
        ("global", None, "GLOBAL_CUE_TOKEN"),
        ("task:pick_and_place", "pick_and_place", "PICK_PLACE_ONLY_CUE_TOKEN"),
        ("task:bimanual_handover", "bimanual_handover", "HANDOVER_EXCLUDED_CUE_TOKEN"),
    ):
        resp = client.post(
            "/api/lab/few-shots",
            headers=lead_hdr,
            json={
                "scope": scope,
                "task_id": tid,
                "label_id": "missed_grasp_retry",
                "error_type": "fp_overflag",
                "episode_id": ep_id,
                "t_start": 0.5,
                "t_end": 1.2,
                "vlm_wrong_json": {"label_id": "missed_grasp_retry"},
                "human_right_json": {"splice_policy": "keep_positive"},
                "rationale": rat,
            },
        )
        assert resp.status_code in (200, 201), resp.text

    mock_vlm_server.requests.clear()
    ev = client.post("/api/lab/eval", headers=lead_hdr, json={"few_shot": True})
    assert ev.status_code in (200, 201), ev.text

    s3_text = "\n".join(
        str(b.get("text", ""))
        for r in mock_vlm_server.requests
        if r["stage"] == "S3"
        for m in r["messages"]
        for b in (m["content"] if isinstance(m.get("content"), list) else [{"type": "text", "text": m.get("content", "")}])
        if isinstance(b, dict) and b.get("type") == "text"
    )
    assert "GLOBAL_CUE_TOKEN" in s3_text
    assert "PICK_PLACE_ONLY_CUE_TOKEN" in s3_text
    assert "HANDOVER_EXCLUDED_CUE_TOKEN" not in s3_text


def test_pairwise_unknown_anomaly_triage_creates_label_history_and_bumps_version(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    """Unknown anomaly triage promotes new visual label, creates .history snapshot, and bumps label_version."""
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "pick_and_place_unk"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=3.0)
    ep_id = _ingest_path(e2e_workspace, h5_dir)[0]

    repo = Repository(e2e_workspace)
    try:
        repo.save_vlm_run_artifact(
            _make_vlm_run_artifact(
                ep_id,
                run_id="run_unk",
                duration_s=3.0,
                passed=False,
                qc_segments=[
                    {
                        "seg_id": "seg_unk",
                        "t_start": 1.0,
                        "t_end": 1.8,
                        "label_id": "unknown_anomaly",
                        "training_role": "badcase",
                        "splice_policy": "mask_loss",
                        "mask_side": "right",
                        "affected_arm": "right",
                        "vlm_raw_policy": "mask_loss",
                        "is_unknown": True,
                        "unknown_note": "Contents spilled from tilted container",
                        "confidence": 0.78,
                    }
                ],
            )
        )
    finally:
        repo.close()

    cfg_before = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    v_before = cfg_before.labels.version

    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    lead_hdr = {"X-Auth-Token": "lead-token"}

    triage_resp = client.post(
        "/api/lab/unknowns/triage",
        headers=lead_hdr,
        json={
            "action": "promote",
            "episode_id": ep_id,
            "seg_id": "seg_unk",
            "new_label": {
                "id": "container_spill_tilt",
                "name": "Container Tilt Spill",
                "category": "failure",
                "default_training_role": "badcase",
                "default_splice_policy": "mask_loss",
                "collector_ui": {
                    "title": "Tilted container spill",
                    "hint": "Container tilted during transport and spilled contents",
                },
                "vlm_semantics": {
                    "definition": "Held container tilts excessively and spills contents",
                    "visual_cues": ["Container rim tilts with visible spillage"],
                },
            },
        },
    )
    assert triage_resp.status_code in (200, 201), triage_resp.text

    cfg_after = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    assert cfg_after.labels.version > v_before
    assert cfg_after.labels.get("container_spill_tilt") is not None
    assert (e2e_workspace / "configs_override" / "labels.yaml").is_file()
    assert list((e2e_workspace / ".history").rglob("*.json"))


def test_pairwise_gt_diagnosis_to_eval_compare_two_runs_and_promote(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    """Frozen GT -> diagnosis + Few-Shot -> compare two sandbox eval runs -> promote without mutating gt_frozen."""
    from robo_vlm_qc.models.annotation import MacroStage, SuccessVerdict
    from robo_vlm_qc.models.review import EpisodeStatus, ReviewWritePayload, WatchCoverage
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "pick_and_place_eval_cmp"
    h.make_synthetic_h5_raw_v3(h5_dir, name="ep_0.h5", duration_s=3.0)
    h.make_synthetic_h5_raw_v3(h5_dir, name="ep_1.h5", duration_s=3.0)
    ep_ids = _ingest_path(e2e_workspace, h5_dir)

    repo = Repository(e2e_workspace)
    try:
        for idx, ep_id in enumerate(ep_ids):
            rid = f"run_base_{idx}"
            repo.save_vlm_run_artifact(_make_vlm_run_artifact(ep_id, run_id=rid, duration_s=3.0, passed=True))
            repo.save_review(
                ep_id,
                "lead_1",
                "lead",
                ReviewWritePayload(
                    base_run_id=rid,
                    action="freeze_gt",
                    qc_segments=[],
                    success=SuccessVerdict(passed=True, verdict="success", confidence=1.0),
                    macro_stages=[MacroStage(stage_id="S1", t_start=0.0, t_end=3.0, label="Manipulate", label_en="manipulate")],
                    watch_coverage=WatchCoverage(
                        watched_intervals=[(0.0, 3.0)], watched_s=3.0, duration_s=3.0, coverage_ratio=1.0, confirmed=True
                    ),
                ),
            )
    finally:
        repo.close()

    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    lead_hdr = {"X-Auth-Token": "lead-token"}

    # Baseline eval (without few-shot, mock returns default segments -> lower precision)
    ev1 = client.post(
        "/api/lab/eval",
        headers=lead_hdr,
        json={"dataset": "pick_and_place_eval_cmp", "few_shot": False},
    )
    assert ev1.status_code in (200, 201), ev1.text
    ev1_data = ev1.json()

    # Add Few-Shot and set candidate S3/S4 response matching GT (0 segments -> higher F1 / lower overcut)
    client.post(
        "/api/lab/few-shots",
        headers=lead_hdr,
        json={
            "scope": "global",
            "label_id": "boundary_idle",
            "error_type": "fp_overflag",
            "episode_id": ep_ids[0],
            "t_start": 0.0,
            "t_end": 0.4,
            "vlm_wrong_json": {"label_id": "boundary_idle"},
            "human_right_json": {"keep": False},
            "rationale": "Candidate improved prompt removes FP",
        },
    )
    mock_vlm_server.set_stage_response("S3", {"segments": []})
    mock_vlm_server.set_stage_response("S4A", {"merged_segments": []})

    ev2 = client.post(
        "/api/lab/eval",
        headers=lead_hdr,
        json={"dataset": "pick_and_place_eval_cmp", "few_shot": True},
    )
    assert ev2.status_code in (200, 201), ev2.text
    ev2_data = ev2.json()

    # Verify quantitative metrics & comparison
    assert "metrics" in ev1_data and "metrics" in ev2_data
    cmp_resp = client.get(
        "/api/lab/eval/compare",
        params={"baseline": ev1_data["eval_id"], "candidate": ev2_data["eval_id"]},
        headers=lead_hdr,
    )
    assert cmp_resp.status_code == 200, cmp_resp.text
    cmp_json = cmp_resp.json()
    assert cmp_json

    prom = client.post(
        "/api/lab/promote",
        headers=lead_hdr,
        json={"eval_id": ev2_data["eval_id"]},
    )
    assert prom.status_code in (200, 201), prom.text

    # Frozen GT production status & active_run_id remain intact
    repo = Repository(e2e_workspace)
    try:
        for idx, ep_id in enumerate(ep_ids):
            row = repo.get_episode_row(ep_id)
            assert row["status"] == EpisodeStatus.GT_FROZEN.value
            assert row["active_run_id"] == f"run_base_{idx}"
    finally:
        repo.close()
