"""Core E2E feature coverage: 4-out transactional exporters and unified Dual-Role Studio.

Covers:
- OutputExporter registry & _staging->rename atomic publish, PieceSlicer across
  bc_clean / recovery_plus / badcase_pairs, tail smoothing & is_tail, macro_stages->subtask_index.
- lerobot_v3 exporter (81D Parquet, CFR MP4s, action_loss_mask, is_tail, subtask_index,
  meta/{info,stats,tasks,subtasks,episodes}, default robot_type="bimanual_robot").
- h5_clean exporter (/qc/* + raw streams) & zero_copy_index exporter (qc_index.parquet,
  manifest.jsonl, subtasks.json, minimal disk footprint).
- `robo-vlm-qc export` CLI for all 4 formats & dry-run.
- FastAPI app & RBAC auth (X-Auth-Token & ?token=), GET /api/episodes list/filter,
  GET /api/episodes/{id} ResolvedEpisode + raw VLM comparison, preview video, direct frame stream, SSE.
- Reviews router: claim/renew/release, Reviewer 3-layer save_draft/submit_review/flag,
  Lead approve/freeze_gt, and read-only guard on arm_timelines & grounding.
- Jobs router (batch/stage-rerun/render_mp4 + SSE progress), Exports router (dry-run/trigger/download),
  Reviewer 403 guards, and `robo-vlm-qc serve` CLI.
- React 19 Studio contracts (web/dist/index.html & static assets served, source pages/components,
  QueuePage role filtering, ReviewStudioPage 3-layer + Diff + Undo/Redo, ExportPage 4x3).
"""

from __future__ import annotations

import importlib
import inspect
import json
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import h5py
import httpx
import numpy as np
import pyarrow.parquet as pq
import pytest

from tests.e2e import helpers as h


# ---------------------------------------------------------------------------
# Lazy Workspace & Domain Setup Helpers
# ---------------------------------------------------------------------------


def _ingest_dir(ws: Path, cfg_dir: Path, data_dir: Path, *, dataset: str | None = None) -> list[str]:
    args = ["-w", str(ws), "-c", str(cfg_dir), "ingest", "--data", str(data_dir)]
    if dataset:
        args.extend(["--dataset", dataset])
    args.append("--json")
    res = h.run_cli(args)
    assert res.returncode == 0, f"ingest failed: {res.stderr}\n{res.stdout}"
    payload = json.loads(res.stdout)
    assert payload["inserted"] >= 1 or payload["skipped"] >= 1
    return [ep["episode_id"] for ep in payload["episodes"]]


def _open_view(repo: Any, episode_id: str) -> Any:
    io_mod = importlib.import_module("robo_vlm_qc.io")
    meta = repo.get_episode_meta(episode_id)
    adapter = io_mod.get_adapter(meta.adapter_type)
    handle = io_mod.EpisodeHandle(
        episode_id=meta.episode_id,
        dataset=meta.dataset,
        stem=meta.stem,
        task_id=meta.task_id,
        adapter_type=meta.adapter_type,
        source_uri=meta.source_uri,
        source_mtime=meta.source_mtime,
        metadata=dict(meta.extra_meta),
    )
    return adapter.open(handle)


def _seed_vlm_and_review(
    repo: Any,
    episode_id: str,
    *,
    run_id: str = "run_001",
    duration_s: float = 4.0,
    qc_segments: list[dict[str, Any]] | None = None,
    macro_stages: list[dict[str, Any]] | None = None,
    passed: bool = True,
    approve_action: str | None = "approve",
) -> tuple[Any, Any]:
    ann_mod = importlib.import_module("robo_vlm_qc.models.annotation")
    rev_mod = importlib.import_module("robo_vlm_qc.models.review")

    mid = round(duration_s * 0.5, 3)
    if macro_stages is None:
        stage_objs = [
            ann_mod.MacroStage(
                stage_id="S1",
                t_start=0.0,
                t_end=mid,
                label="Grasp target object",
                label_en="grasp_object",
                summary="Arm approaches and grasps the target object.",
                color_rgb=[59, 130, 246],
            ),
            ann_mod.MacroStage(
                stage_id="S2",
                t_start=mid,
                t_end=round(duration_s, 3),
                label="Place in target bin",
                label_en="place_object",
                summary="Transfers and places the object cleanly in the target bin.",
                color_rgb=[16, 185, 129],
            ),
        ]
    else:
        stage_objs = [ann_mod.MacroStage.model_validate(s) for s in macro_stages]

    if qc_segments is None:
        seg_objs = [
            ann_mod.QcSegment(
                seg_id="seg_idle_head",
                t_start=0.0,
                t_end=min(0.4, round(duration_s * 0.12, 3)),
                label_id="boundary_idle",
                name="Boundary Idle",
                training_role="badcase",
                splice_policy="cut_clean",
                mask_side=None,
                affected_arm="both",
                vlm_raw_policy="cut_clean",
                guardrail=ann_mod.GuardrailReport(gate_type="splice_gate", passed=True),
            ),
            ann_mod.QcSegment(
                seg_id="seg_retry_mid",
                t_start=round(duration_s * 0.25, 3),
                t_end=round(duration_s * 0.45, 3),
                label_id="missed_grasp_retry",
                name="Missed Grasp Retry",
                training_role="recovery",
                splice_policy="mask_loss",
                mask_side="left",
                affected_arm="left",
                vlm_raw_policy="mask_loss",
                guardrail=ann_mod.GuardrailReport(gate_type="none", passed=True),
            ),
        ]
    else:
        seg_objs = [ann_mod.QcSegment.model_validate(s) for s in qc_segments]

    grounding = ann_mod.GroundingResult(
        ref_size=[1280, 720],
        keyframes_s=[0.0, mid],
        object_tracks_2d={
            "target_object": ann_mod.GroundingTrack(
                object_id="target_object",
                name="Target Object",
                is_static_container=False,
                keyframes=[
                    ann_mod.GroundingKeyframe(
                        row=0, time_s=0.0, visible=True, bbox_norm_1000=[200, 220, 380, 420]
                    ),
                    ann_mod.GroundingKeyframe(
                        row=int(mid * 30),
                        time_s=mid,
                        visible=True,
                        bbox_norm_1000=[420, 240, 600, 440],
                    ),
                ],
            )
        },
    )
    arm_timelines = ann_mod.ArmTimelines(
        left=[
            ann_mod.ArmSpan(
                t_start=0.0, t_end=round(duration_s, 3), state="work", desc="Left arm working"
            )
        ],
        right=[
            ann_mod.ArmSpan(
                t_start=0.0, t_end=round(duration_s, 3), state="hold", desc="Right arm supporting"
            )
        ],
    )
    success = ann_mod.SuccessVerdict(
        passed=passed,
        verdict="success" if passed else "failure",
        confidence=0.95,
        criteria=[
            ann_mod.CriterionResult(
                criterion="Target object reaches target zone",
                met=passed,
                satisfied_at_s=round(duration_s * 0.85, 2) if passed else None,
                evidence="Final frames confirm completion state.",
            )
        ],
        summary="Passed" if passed else "Task failed",
    )

    vlm_run = ann_mod.VlmRunArtifact(
        run_id=run_id,
        episode_id=episode_id,
        created_at=time.time(),
        config_fingerprint=ann_mod.ConfigFingerprintSnapshot(
            model_routing={"s1_grounding": "gpt-4o", "s3_discover": "gpt-4o"},
            prompt_bundle_version="1.0.0",
            label_taxonomy_version="1",
            task_spec_version="1.0.0",
        ),
        grounding=grounding,
        macro_stages=stage_objs,
        arm_timelines=arm_timelines,
        qc_segments=seg_objs,
        success=success,
    )
    repo.save_vlm_run_artifact(vlm_run)

    if approve_action is not None:
        wc = rev_mod.WatchCoverage(
            watched_intervals=[(0.0, duration_s)],
            watched_s=duration_s,
            duration_s=duration_s,
            coverage_ratio=1.0,
            confirmed=True,
        )
        repo.save_review(
            episode_id,
            author="lead_e2e",
            role="lead",
            payload=rev_mod.ReviewWritePayload(
                base_run_id=run_id,
                action=approve_action,
                qc_segments=seg_objs,
                success=success,
                macro_stages=stage_objs,
                watch_coverage=wc,
                notes="E2E seeded approval",
            ),
        )
    return vlm_run, repo.resolve_episode(episode_id)


def _call_slicer(cfg_mgr: Any, resolved: Any, view: Any, recipe: Any) -> list[Any]:
    slicer_mod = importlib.import_module("robo_vlm_qc.export.slicer")
    cls = getattr(slicer_mod, "PieceSlicer")
    sig = inspect.signature(cls.__init__)
    kwargs: dict[str, Any] = {}
    for p in sig.parameters:
        if p in ("config", "cfg", "config_manager", "cfg_mgr"):
            kwargs[p] = cfg_mgr
    slicer = cls(**kwargs)
    return slicer.slice(resolved, view, recipe)


def _call_exporter(
    fmt: str,
    pieces_by_ep: Any,
    resolved_list: list[Any],
    views_by_ep: dict[str, Any],
    out_dir: Path,
    recipe: Any,
) -> Any:
    export_mod = importlib.import_module("robo_vlm_qc.export")
    exporter = export_mod.get_exporter(fmt)
    if isinstance(pieces_by_ep, dict):
        flat_pieces = [p for plist in pieces_by_ep.values() for p in plist]
    else:
        flat_pieces = list(pieces_by_ep)

    sig = inspect.signature(exporter.export)
    params = list(sig.parameters.keys())
    if len(params) >= 5 and params[2] == "views":
        return exporter.export(flat_pieces, resolved_list, views_by_ep, out_dir, recipe)
    return exporter.export(flat_pieces, resolved_list, views_by_ep, out_dir=out_dir, recipe=recipe)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# ===========================================================================
# OutputExporter Registry, Atomic _staging->rename, PieceSlicer
# ===========================================================================


def test_output_exporter_registry_lists_all_four(e2e_workspace: Path) -> None:
    export_mod = importlib.import_module("robo_vlm_qc.export")
    base_mod = importlib.import_module("robo_vlm_qc.export.base")

    registered = set(export_mod.list_exporters())
    assert {"lerobot_v3", "h5_clean", "zero_copy_index", "viz_mp4"}.issubset(registered)

    for fmt in ("lerobot_v3", "h5_clean", "zero_copy_index", "viz_mp4"):
        inst = export_mod.get_exporter(fmt)
        assert isinstance(inst, base_mod.OutputExporter)
        assert getattr(inst, "name", fmt) == fmt

    class _DummyExporter(base_mod.OutputExporter):
        name = "dummy_custom_fmt"

        def export(self, pieces: Any, resolved_list: Any, views: Any, out_dir: Path, recipe: Any) -> Any:
            raise NotImplementedError

    export_mod.register_exporter(_DummyExporter, override=True)
    assert "dummy_custom_fmt" in export_mod.list_exporters()


def test_atomic_staging_rename_and_no_leftover_staging(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.models.export import ExportManifest, ExportRecipe
    from robo_vlm_qc.store import Repository

    vdir = h.make_synthetic_video_dir(tmp_path / "ds_atomic", multi_cam=False, duration_s=2.0)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, vdir)
    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    repo = Repository(e2e_workspace)
    try:
        _, resolved = _seed_vlm_and_review(repo, ep_ids[0], duration_s=2.0, qc_segments=[])
        recipe = ExportRecipe(name="bc_clean", format="zero_copy_index")
        out_dir = e2e_workspace / "exports" / "atomic_test"

        # Pre-seed a stale staging dir to ensure atomic publish cleans/handles _staging cleanly
        stale_staging = out_dir.parent / f"{out_dir.name}._staging"
        stale_staging.mkdir(parents=True, exist_ok=True)
        (stale_staging / "stale.tmp").write_text("stale", encoding="utf-8")
        with _open_view(repo, ep_ids[0]) as view:
            pieces = _call_slicer(cfg, resolved, view, recipe)
            manifest = _call_exporter("zero_copy_index", pieces, [resolved], {ep_ids[0]: view}, out_dir, recipe)

        assert isinstance(manifest, ExportManifest)
        assert out_dir.is_dir()
        assert (out_dir / "qc_index.parquet").is_file()
        assert not (out_dir / "_staging").exists()
        leftover_staging = [p for p in out_dir.parent.iterdir() if "_staging" in p.name]
        assert leftover_staging == [], f"Leftover staging dirs: {leftover_staging}"
    finally:
        repo.close()


def test_piece_slicer_bc_clean_cuts_and_masks(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.models.export import ExportRecipe
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "ds_bc"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=4.0)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, h5_dir)
    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    repo = Repository(e2e_workspace)
    try:
        segs = [
            {
                "seg_id": "s_cut_head",
                "t_start": 0.0,
                "t_end": 0.5,
                "label_id": "boundary_idle",
                "training_role": "badcase",
                "splice_policy": "cut_clean",
                "mask_side": None,
                "vlm_raw_policy": "cut_clean",
            },
            {
                "seg_id": "s_rec_left",
                "t_start": 1.0,
                "t_end": 1.8,
                "label_id": "missed_grasp_retry",
                "training_role": "recovery",
                "splice_policy": "mask_loss",
                "mask_side": "left",
                "vlm_raw_policy": "mask_loss",
            },
            {
                "seg_id": "s_bad_right",
                "t_start": 2.4,
                "t_end": 3.0,
                "label_id": "collision_or_knock",
                "training_role": "badcase",
                "splice_policy": "mask_loss",
                "mask_side": "right",
                "vlm_raw_policy": "mask_loss",
            },
        ]
        _, resolved_pass = _seed_vlm_and_review(
            repo, ep_ids[0], duration_s=4.0, qc_segments=segs, passed=True
        )
        recipe = ExportRecipe(name="bc_clean", format="lerobot_v3")
        with _open_view(repo, ep_ids[0]) as view:
            pieces = _call_slicer(cfg, resolved_pass, view, recipe)
            assert len(pieces) >= 1
            kept = [p for p in pieces if not p.dropped]
            assert kept
            # Head cut [0.0, 0.5s] (~rows 0..15) must be removed
            assert kept[0].start_row >= 14
            all_rows = [r for p in kept for r in range(p.start_row, p.end_row)]
            all_masks = [m for p in kept for m in p.action_loss_mask[: p.own_rows]]
            assert len(all_rows) == len(all_masks)
            # Recovery (left mask_loss at 1.0-1.8s -> ~30..54) must be masked [0.0, 1.0] in bc_clean
            rec_masks = [m for r, m in zip(all_rows, all_masks) if 32 <= r < 52]
            assert rec_masks and all(m[0] == 0.0 and m[1] == 1.0 for m in rec_masks)
            # Badcase (right mask_loss at 2.4-3.0s -> ~72..90) must be masked [1.0, 0.0]
            bad_masks = [m for r, m in zip(all_rows, all_masks) if 74 <= r < 88]
            assert bad_masks and all(m[0] == 1.0 and m[1] == 0.0 for m in bad_masks)

            # Failed episode must produce 0 kept pieces under bc_clean
            _, resolved_fail = _seed_vlm_and_review(
                repo, ep_ids[0], run_id="run_fail", duration_s=4.0, qc_segments=segs, passed=False
            )
            fail_pieces = [p for p in _call_slicer(cfg, resolved_fail, view, recipe) if not p.dropped]
            assert fail_pieces == []
    finally:
        repo.close()


def test_piece_slicer_recovery_plus_keeps_recovery_unmasked(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.models.export import ExportRecipe
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "ds_rec"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=4.0)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, h5_dir)
    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    repo = Repository(e2e_workspace)
    try:
        segs = [
            {
                "seg_id": "s_rec",
                "t_start": 1.0,
                "t_end": 1.8,
                "label_id": "missed_grasp_retry",
                "training_role": "recovery",
                "splice_policy": "mask_loss",
                "mask_side": "left",
                "vlm_raw_policy": "mask_loss",
            },
            {
                "seg_id": "s_bad",
                "t_start": 2.4,
                "t_end": 3.0,
                "label_id": "hesitation_oscillation",
                "training_role": "badcase",
                "splice_policy": "mask_loss",
                "mask_side": "both",
                "vlm_raw_policy": "mask_loss",
            },
        ]
        _, resolved = _seed_vlm_and_review(
            repo, ep_ids[0], duration_s=4.0, qc_segments=segs, passed=True
        )
        recipe = ExportRecipe(name="recovery_plus", format="lerobot_v3")
        with _open_view(repo, ep_ids[0]) as view:
            kept = [p for p in _call_slicer(cfg, resolved, view, recipe) if not p.dropped]
            assert kept
            all_rows = [r for p in kept for r in range(p.start_row, p.end_row)]
            all_masks = [m for p in kept for m in p.action_loss_mask[: p.own_rows]]
            rec_masks = [m for r, m in zip(all_rows, all_masks) if 32 <= r < 52]
            bad_masks = [m for r, m in zip(all_rows, all_masks) if 74 <= r < 88]
            # Recovery becomes unmasked [1.0, 1.0]; pure badcase remains masked [0.0, 0.0]
            assert rec_masks and all(m == [1.0, 1.0] for m in rec_masks)
            assert bad_masks and all(m == [0.0, 0.0] for m in bad_masks)
    finally:
        repo.close()


def test_piece_slicer_badcase_pairs_includes_failures_and_context(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.models.export import ExportRecipe
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "ds_bad"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=4.0)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, h5_dir)
    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    repo = Repository(e2e_workspace)
    try:
        segs = [
            {
                "seg_id": "s_term_fail",
                "t_start": 2.2,
                "t_end": 2.8,
                "label_id": "terminal_failure",
                "training_role": "terminal_fail",
                "splice_policy": "mask_loss",
                "mask_side": "both",
                "vlm_raw_policy": "mask_loss",
            }
        ]
        _, resolved_fail = _seed_vlm_and_review(
            repo, ep_ids[0], duration_s=4.0, qc_segments=segs, passed=False
        )
        recipe = ExportRecipe(name="badcase_pairs", format="zero_copy_index", context_window_s=0.5)
        with _open_view(repo, ep_ids[0]) as view:
            kept = [p for p in _call_slicer(cfg, resolved_fail, view, recipe) if not p.dropped]
            assert len(kept) >= 1
            # Context window [2.2 - 0.5, 2.8 + 0.5] = [1.7s, 3.3s]
            assert any(p.t_start_s <= 1.8 and p.t_end_s >= 3.2 for p in kept)
    finally:
        repo.close()


def test_piece_slicer_tail_smoothing_is_tail_and_subtask_alignment(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.models.export import ExportRecipe
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "ds_tail"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=4.0)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, h5_dir)
    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    repo = Repository(e2e_workspace)
    try:
        # Mid cut splits trajectory into 2 pieces across moving kinematics -> pose jump triggers tail smoothing
        segs = [
            {
                "seg_id": "s_mid_cut",
                "t_start": 1.2,
                "t_end": 2.2,
                "label_id": "mid_idle_freeze",
                "training_role": "badcase",
                "splice_policy": "cut_clean",
                "mask_side": None,
                "vlm_raw_policy": "cut_clean",
            }
        ]
        _, resolved = _seed_vlm_and_review(
            repo, ep_ids[0], duration_s=4.0, qc_segments=segs, passed=True
        )
        recipe = ExportRecipe(
            name="bc_clean",
            format="lerobot_v3",
            tail_frames=6,
            smooth_tail="always",
        )
        with _open_view(repo, ep_ids[0]) as view:
            kept = [p for p in _call_slicer(cfg, resolved, view, recipe) if not p.dropped]
            assert len(kept) == 2
            p0, p1 = kept[0], kept[1]
            assert p0.tail_rows > 0
            assert p0.total_rows == p0.own_rows + p0.tail_rows
            assert len(p0.is_tail) == p0.total_rows
            assert not any(p0.is_tail[: p0.own_rows])
            assert all(p0.is_tail[p0.own_rows :])
            assert p0.smooth_tail_applied is True
            assert p1.tail_rows == 0
            # Macro stages aligned to subtask_indices
            for p in kept:
                assert len(p.subtask_indices) == p.total_rows
                assert len(p.subtasks) >= 1
                assert all(idx >= 0 for idx in p.subtask_indices)
            assert any(idx == 1 for idx in p1.subtask_indices)
    finally:
        repo.close()


# ===========================================================================
# lerobot_v3 Exporter
# ===========================================================================


def test_lerobot_v3_export_from_h5_valid_81d_and_cfr_mp4(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.models.export import ExportRecipe
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "ds_lr_h5"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=2.0, with_sbs_stereo=True)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, h5_dir)
    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    repo = Repository(e2e_workspace)
    try:
        _, resolved = _seed_vlm_and_review(repo, ep_ids[0], duration_s=2.0)
        recipe = ExportRecipe(name="bc_clean", format="lerobot_v3")
        out_dir = e2e_workspace / "exports" / "lr3_h5"
        with _open_view(repo, ep_ids[0]) as view:
            pieces = _call_slicer(cfg, resolved, view, recipe)
            manifest = _call_exporter("lerobot_v3", pieces, [resolved], {ep_ids[0]: view}, out_dir, recipe)

        assert manifest.episodes_included == 1
        summary = h.assert_81d_lerobot_v3_valid(out_dir)
        assert summary["total_rows"] > 0
        assert summary["info"]["robot_type"] == "bimanual_robot"
        # 4 camera streams from H5 (head_rgbd, torso_rgbd, head_stereo_left, head_stereo_right)
        assert len(summary["videos"]) >= 4
    finally:
        repo.close()


def test_lerobot_v3_export_from_lerobot_source_valid_81d(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.models.export import ExportRecipe
    from robo_vlm_qc.store import Repository

    lr_src = h.make_synthetic_lerobot(
        tmp_path / "ds_lr_src",
        version="v3",
        n_episodes=1,
        duration_s=2.0,
        task_name="bimanual_handover",
    )
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, lr_src)
    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    repo = Repository(e2e_workspace)
    try:
        _, resolved = _seed_vlm_and_review(repo, ep_ids[0], duration_s=2.0, qc_segments=[])
        recipe = ExportRecipe(name="bc_clean", format="lerobot_v3")
        out_dir = e2e_workspace / "exports" / "lr3_roundtrip"
        with _open_view(repo, ep_ids[0]) as view:
            pieces = _call_slicer(cfg, resolved, view, recipe)
            _call_exporter("lerobot_v3", pieces, [resolved], {ep_ids[0]: view}, out_dir, recipe)

        summary = h.assert_81d_lerobot_v3_valid(out_dir)
        assert summary["total_rows"] >= 50
        assert summary["info"]["robot_type"] == "bimanual_robot"
        assert len(summary["videos"]) == 3  # main, left_wrist, right_wrist
    finally:
        repo.close()


def test_lerobot_v3_subtasks_parquet_and_row_indices(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.models.export import ExportRecipe
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "ds_subtask"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=2.0)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, h5_dir)
    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    repo = Repository(e2e_workspace)
    try:
        _, resolved = _seed_vlm_and_review(repo, ep_ids[0], duration_s=2.0, qc_segments=[])
        recipe = ExportRecipe(name="bc_clean", format="lerobot_v3")
        out_dir = e2e_workspace / "exports" / "lr3_subtasks"
        with _open_view(repo, ep_ids[0]) as view:
            pieces = _call_slicer(cfg, resolved, view, recipe)
            _call_exporter("lerobot_v3", pieces, [resolved], {ep_ids[0]: view}, out_dir, recipe)

        subtasks_tbl = pq.read_table(out_dir / "meta" / "subtasks.parquet")
        assert subtasks_tbl.num_rows >= 2
        assert "label" in subtasks_tbl.column_names
        data_tbl = pq.read_table(sorted((out_dir / "data").rglob("*.parquet"))[0])
        sub_indices = set(data_tbl["subtask_index"].to_pylist())
        assert sub_indices.issubset(set(range(subtasks_tbl.num_rows)))
        assert len(sub_indices) >= 2
        assert data_tbl["index"].to_pylist() == list(range(data_tbl.num_rows))
        assert data_tbl["frame_index"].to_pylist() == list(range(data_tbl.num_rows))
    finally:
        repo.close()


def test_lerobot_v3_action_loss_mask_and_is_tail_values(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.models.export import ExportRecipe
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "ds_mask_tail"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=3.0)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, h5_dir)
    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    repo = Repository(e2e_workspace)
    try:
        segs = [
            {
                "seg_id": "s_left",
                "t_start": 0.3,
                "t_end": 0.9,
                "label_id": "missed_grasp_retry",
                "training_role": "recovery",
                "splice_policy": "mask_loss",
                "mask_side": "left",
                "vlm_raw_policy": "mask_loss",
            },
            {
                "seg_id": "s_cut",
                "t_start": 1.3,
                "t_end": 1.8,
                "label_id": "mid_idle_freeze",
                "training_role": "badcase",
                "splice_policy": "cut_clean",
                "mask_side": None,
                "vlm_raw_policy": "cut_clean",
            },
            {
                "seg_id": "s_right",
                "t_start": 2.2,
                "t_end": 2.7,
                "label_id": "collision_or_knock",
                "training_role": "badcase",
                "splice_policy": "mask_loss",
                "mask_side": "right",
                "vlm_raw_policy": "mask_loss",
            },
        ]
        _, resolved = _seed_vlm_and_review(repo, ep_ids[0], duration_s=3.0, qc_segments=segs)
        recipe = ExportRecipe(name="bc_clean", format="lerobot_v3", tail_frames=4, smooth_tail="always")
        out_dir = e2e_workspace / "exports" / "lr3_mask_tail"
        with _open_view(repo, ep_ids[0]) as view:
            pieces = _call_slicer(cfg, resolved, view, recipe)
            _call_exporter("lerobot_v3", pieces, [resolved], {ep_ids[0]: view}, out_dir, recipe)

        tbl = pq.read_table(sorted((out_dir / "data").rglob("*.parquet"))[0])
        masks = tbl["action_loss_mask"].to_pylist()
        tails = tbl["is_tail"].to_pylist()
        assert any(m == [0.0, 1.0] for m in masks)
        assert any(m == [1.0, 0.0] for m in masks)
        assert any(m == [1.0, 1.0] for m in masks)
        assert any(t is True for t in tails)
        assert any(t is False for t in tails)
    finally:
        repo.close()


def test_lerobot_v3_stats_json_and_episodes_metadata(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.models.export import ExportRecipe
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "ds_stats"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=2.0)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, h5_dir)
    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    repo = Repository(e2e_workspace)
    try:
        _, resolved = _seed_vlm_and_review(repo, ep_ids[0], duration_s=2.0, qc_segments=[])
        recipe = ExportRecipe(name="bc_clean", format="lerobot_v3")
        out_dir = e2e_workspace / "exports" / "lr3_stats"
        with _open_view(repo, ep_ids[0]) as view:
            pieces = _call_slicer(cfg, resolved, view, recipe)
            _call_exporter("lerobot_v3", pieces, [resolved], {ep_ids[0]: view}, out_dir, recipe)

        stats = json.loads((out_dir / "meta" / "stats.json").read_text("utf-8"))
        assert "observation.state" in stats
        assert "action" in stats
        for stat_key in ("min", "max", "mean", "std"):
            assert stat_key in stats["observation.state"]
            assert len(stats["observation.state"][stat_key]) == 81

        ep_tbl = pq.read_table(sorted((out_dir / "meta" / "episodes").rglob("*.parquet"))[0])
        assert ep_tbl.num_rows >= 1
        assert "length" in ep_tbl.column_names
        info = json.loads((out_dir / "meta" / "info.json").read_text("utf-8"))
        assert info["total_episodes"] == ep_tbl.num_rows
        assert info["fps"] == pytest.approx(30.0)
        assert info["robot_type"] == "bimanual_robot"
    finally:
        repo.close()


def test_lerobot_v3_multi_episode_aggregation_and_recovery_plus(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.models.export import ExportRecipe
    from robo_vlm_qc.store import Repository

    lr_dir = h.make_synthetic_lerobot(tmp_path / "ds_multi_lr", version="v3", n_episodes=2, duration_s=2.0)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, lr_dir)
    assert len(ep_ids) == 2
    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    repo = Repository(e2e_workspace)
    try:
        segs_pass = [
            {
                "seg_id": "s_rec",
                "t_start": 0.4,
                "t_end": 1.0,
                "label_id": "missed_grasp_retry",
                "training_role": "recovery",
                "splice_policy": "mask_loss",
                "mask_side": "both",
                "vlm_raw_policy": "mask_loss",
            }
        ]
        _, res0 = _seed_vlm_and_review(
            repo, ep_ids[0], run_id="r0", duration_s=2.0, qc_segments=segs_pass, passed=True
        )
        _, res1 = _seed_vlm_and_review(
            repo, ep_ids[1], run_id="r1", duration_s=2.0, qc_segments=[], passed=False
        )
        recipe = ExportRecipe(name="recovery_plus", format="lerobot_v3")
        out_dir = e2e_workspace / "exports" / "lr3_multi_rec"

        with _open_view(repo, ep_ids[0]) as v0, _open_view(repo, ep_ids[1]) as v1:
            p0 = _call_slicer(cfg, res0, v0, recipe)
            p1 = _call_slicer(cfg, res1, v1, recipe)
            manifest = _call_exporter(
                "lerobot_v3",
                [*p0, *p1],
                [res0, res1],
                {ep_ids[0]: v0, ep_ids[1]: v1},
                out_dir,
                recipe,
            )

        assert manifest.episodes_included == 1
        assert manifest.episodes_skipped >= 1
        summary = h.assert_81d_lerobot_v3_valid(out_dir)
        tbl = pq.read_table(sorted((out_dir / "data").rglob("*.parquet"))[0])
        masks = tbl["action_loss_mask"].to_pylist()
        assert all(m == [1.0, 1.0] for m in masks)
        assert summary["total_rows"] == manifest.total_frames
    finally:
        repo.close()


# ===========================================================================
# h5_clean & zero_copy_index Exporters
# ===========================================================================


def test_h5_clean_preserves_raw_streams_and_qc_datasets(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.models.export import ExportRecipe
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "ds_h5_clean"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=2.5)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, h5_dir)
    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    repo = Repository(e2e_workspace)
    try:
        _, resolved = _seed_vlm_and_review(repo, ep_ids[0], duration_s=2.5)
        recipe = ExportRecipe(name="bc_clean", format="h5_clean")
        out_dir = e2e_workspace / "exports" / "h5_clean_1"
        with _open_view(repo, ep_ids[0]) as view:
            pieces = _call_slicer(cfg, resolved, view, recipe)
            manifest = _call_exporter("h5_clean", pieces, [resolved], {ep_ids[0]: view}, out_dir, recipe)

        assert manifest.pieces_exported >= 1
        h5_files = sorted(out_dir.rglob("*.h5"))
        assert len(h5_files) >= 1
        with h5py.File(h5_files[0], "r") as f:
            for req_ds in ("/qc/action_loss_mask", "/qc/is_tail", "/qc/subtask_index", "/qc/resolved_json"):
                assert req_ds in f, f"Missing {req_ds} in {h5_files[0]}"
            n_rows = f["/qc/action_loss_mask"].shape[0]
            assert f["/qc/action_loss_mask"].shape == (n_rows, 2)
            assert f["/qc/is_tail"].shape == (n_rows,)
            assert f["/qc/subtask_index"].shape == (n_rows,)
            resolved_raw = f["/qc/resolved_json"][()]
            resolved_str = (
                resolved_raw.decode("utf-8") if isinstance(resolved_raw, bytes) else str(resolved_raw)
            )
            h.assert_no_legacy_fields(json.loads(resolved_str))
            # Raw streams preserved
            assert "robot" in f or "camera" in f
    finally:
        repo.close()


def test_h5_clean_recomputes_masks_for_recovery_plus_vs_bc_clean(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.models.export import ExportRecipe
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "ds_h5_cmp"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=2.0)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, h5_dir)
    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    repo = Repository(e2e_workspace)
    try:
        segs = [
            {
                "seg_id": "s_rec",
                "t_start": 0.4,
                "t_end": 1.2,
                "label_id": "object_slip_recovery",
                "training_role": "recovery",
                "splice_policy": "mask_loss",
                "mask_side": "both",
                "vlm_raw_policy": "mask_loss",
            }
        ]
        _, resolved = _seed_vlm_and_review(repo, ep_ids[0], duration_s=2.0, qc_segments=segs)
        out_bc = e2e_workspace / "exports" / "h5_bc"
        out_rp = e2e_workspace / "exports" / "h5_rp"
        with _open_view(repo, ep_ids[0]) as view:
            r_bc = ExportRecipe(name="bc_clean", format="h5_clean")
            r_rp = ExportRecipe(name="recovery_plus", format="h5_clean")
            _call_exporter(
                "h5_clean", _call_slicer(cfg, resolved, view, r_bc), [resolved], {ep_ids[0]: view}, out_bc, r_bc
            )
            _call_exporter(
                "h5_clean", _call_slicer(cfg, resolved, view, r_rp), [resolved], {ep_ids[0]: view}, out_rp, r_rp
            )

        with h5py.File(sorted(out_bc.rglob("*.h5"))[0], "r") as f_bc, h5py.File(
            sorted(out_rp.rglob("*.h5"))[0], "r"
        ) as f_rp:
            m_bc = np.asarray(f_bc["/qc/action_loss_mask"])
            m_rp = np.asarray(f_rp["/qc/action_loss_mask"])
            assert float((m_bc == 0.0).sum()) > 0
            assert bool((m_rp == 1.0).all())
    finally:
        repo.close()


def test_h5_clean_from_lerobot_source(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.models.export import ExportRecipe
    from robo_vlm_qc.store import Repository

    lr_dir = h.make_synthetic_lerobot(
        tmp_path / "ds_lr_h5",
        version="v3",
        n_episodes=1,
        duration_s=2.0,
        task_name="bimanual_handover",
    )
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, lr_dir)
    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    repo = Repository(e2e_workspace)
    try:
        _, resolved = _seed_vlm_and_review(repo, ep_ids[0], duration_s=2.0, qc_segments=[])
        recipe = ExportRecipe(name="bc_clean", format="h5_clean")
        out_dir = e2e_workspace / "exports" / "h5_from_lr"
        with _open_view(repo, ep_ids[0]) as view:
            pieces = _call_slicer(cfg, resolved, view, recipe)
            _call_exporter("h5_clean", pieces, [resolved], {ep_ids[0]: view}, out_dir, recipe)

        h5_files = sorted(out_dir.rglob("*.h5"))
        assert len(h5_files) >= 1
        with h5py.File(h5_files[0], "r") as f:
            assert "/qc/action_loss_mask" in f
            assert "/qc/is_tail" in f
            assert "/qc/subtask_index" in f
            assert "/qc/resolved_json" in f
    finally:
        repo.close()


def test_zero_copy_index_structure_and_source_uris(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.models.export import ExportRecipe
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "ds_zci_h5"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=2.0)
    vid_dir = h.make_synthetic_video_dir(tmp_path / "ds_zci_vid", multi_cam=False, duration_s=2.0)
    ep_h5 = _ingest_dir(e2e_workspace, e2e_config_dir, h5_dir)
    ep_vid = _ingest_dir(e2e_workspace, e2e_config_dir, vid_dir)

    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    repo = Repository(e2e_workspace)
    try:
        _, r_h5 = _seed_vlm_and_review(repo, ep_h5[0], run_id="r_h5", duration_s=2.0, qc_segments=[])
        _, r_vid = _seed_vlm_and_review(repo, ep_vid[0], run_id="r_vid", duration_s=2.0, qc_segments=[])
        recipe = ExportRecipe(name="bc_clean", format="zero_copy_index")
        out_dir = e2e_workspace / "exports" / "zci_multi"
        with _open_view(repo, ep_h5[0]) as vh, _open_view(repo, ep_vid[0]) as vv:
            pieces = _call_slicer(cfg, r_h5, vh, recipe) + _call_slicer(cfg, r_vid, vv, recipe)
            _call_exporter(
                "zero_copy_index",
                pieces,
                [r_h5, r_vid],
                {ep_h5[0]: vh, ep_vid[0]: vv},
                out_dir,
                recipe,
            )

        assert (out_dir / "qc_index.parquet").is_file()
        assert (out_dir / "manifest.jsonl").is_file()
        assert (out_dir / "subtasks.json").is_file()

        idx_tbl = pq.read_table(out_dir / "qc_index.parquet")
        assert idx_tbl.num_rows >= 2
        cols = set(idx_tbl.column_names)
        assert {"episode_id", "source_uri", "start_row", "end_row"}.issubset(cols)
        uris = set(idx_tbl["source_uri"].to_pylist())
        assert r_h5.meta.source_uri in uris
        assert r_vid.meta.source_uri in uris

        lines = [ln for ln in (out_dir / "manifest.jsonl").read_text("utf-8").splitlines() if ln.strip()]
        assert len(lines) >= 2
        subtasks = json.loads((out_dir / "subtasks.json").read_text("utf-8"))
        assert subtasks
    finally:
        repo.close()


def test_zero_copy_index_minimal_disk_footprint(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.models.export import ExportRecipe
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "ds_zci_size"
    h5_file = h.make_synthetic_h5_raw_v3(h5_dir, duration_s=3.0, with_sbs_stereo=True)
    src_bytes = h5_file.stat().st_size
    assert src_bytes > 1_000_000

    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, h5_dir)
    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    repo = Repository(e2e_workspace)
    try:
        _, resolved = _seed_vlm_and_review(repo, ep_ids[0], duration_s=3.0, qc_segments=[])
        recipe = ExportRecipe(name="bc_clean", format="zero_copy_index")
        out_dir = e2e_workspace / "exports" / "zci_footprint"
        with _open_view(repo, ep_ids[0]) as view:
            pieces = _call_slicer(cfg, resolved, view, recipe)
            _call_exporter("zero_copy_index", pieces, [resolved], {ep_ids[0]: view}, out_dir, recipe)

        total_export_bytes = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
        assert total_export_bytes < max(200_000, int(src_bytes * 0.01))
        assert list(out_dir.rglob("*.h5")) == []
        assert list(out_dir.rglob("*.mp4")) == []
    finally:
        repo.close()


def test_zero_copy_index_badcase_pairs_includes_failed_episodes(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.models.export import ExportRecipe
    from robo_vlm_qc.store import Repository

    vdir = h.make_synthetic_video_dir(tmp_path / "ds_zci_bad", multi_cam=False, duration_s=3.0)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, vdir)
    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    repo = Repository(e2e_workspace)
    try:
        segs = [
            {
                "seg_id": "s_fail",
                "t_start": 1.2,
                "t_end": 2.0,
                "label_id": "terminal_failure",
                "training_role": "terminal_fail",
                "splice_policy": "mask_loss",
                "mask_side": "both",
                "vlm_raw_policy": "mask_loss",
            }
        ]
        _, resolved = _seed_vlm_and_review(
            repo, ep_ids[0], duration_s=3.0, qc_segments=segs, passed=False
        )
        recipe = ExportRecipe(name="badcase_pairs", format="zero_copy_index", context_window_s=0.4)
        out_dir = e2e_workspace / "exports" / "zci_bad"
        with _open_view(repo, ep_ids[0]) as view:
            pieces = _call_slicer(cfg, resolved, view, recipe)
            manifest = _call_exporter("zero_copy_index", pieces, [resolved], {ep_ids[0]: view}, out_dir, recipe)

        assert manifest.episodes_included == 1
        idx_tbl = pq.read_table(out_dir / "qc_index.parquet")
        assert idx_tbl.num_rows >= 1
        assert ep_ids[0] in idx_tbl["episode_id"].to_pylist()
    finally:
        repo.close()


# ===========================================================================
# `robo-vlm-qc export` CLI (All 4 Formats & Dry-Run)
# ===========================================================================


def test_cli_export_lerobot_v3_and_h5_clean(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "ds_cli_exp1"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=2.0)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, h5_dir)
    repo = Repository(e2e_workspace)
    try:
        _seed_vlm_and_review(repo, ep_ids[0], duration_s=2.0, qc_segments=[])
    finally:
        repo.close()

    out_lr = tmp_path / "cli_out_lr3"
    r1 = h.run_cli(
        [
            "-w",
            str(e2e_workspace),
            "-c",
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
    assert r1.returncode == 0, r1.stderr + r1.stdout
    summary = h.assert_81d_lerobot_v3_valid(out_lr)
    assert summary["info"]["robot_type"] == "bimanual_robot"

    out_h5 = tmp_path / "cli_out_h5"
    r2 = h.run_cli(
        [
            "-w",
            str(e2e_workspace),
            "-c",
            str(e2e_config_dir),
            "export",
            "--format",
            "h5_clean",
            "--recipe",
            "bc_clean",
            "--out",
            str(out_h5),
        ]
    )
    assert r2.returncode == 0, r2.stderr + r2.stdout
    assert list(out_h5.rglob("*.h5"))


def test_cli_export_zero_copy_index_and_viz_mp4(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "ds_cli_exp2"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=1.5)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, h5_dir)
    repo = Repository(e2e_workspace)
    try:
        _seed_vlm_and_review(repo, ep_ids[0], duration_s=1.5, qc_segments=[])
    finally:
        repo.close()

    out_zci = tmp_path / "cli_out_zci"
    r1 = h.run_cli(
        [
            "-w",
            str(e2e_workspace),
            "-c",
            str(e2e_config_dir),
            "export",
            "--format",
            "zero_copy_index",
            "--recipe",
            "bc_clean",
            "--out",
            str(out_zci),
        ]
    )
    assert r1.returncode == 0, r1.stderr + r1.stdout
    assert (out_zci / "qc_index.parquet").is_file()
    assert (out_zci / "manifest.jsonl").is_file()

    out_mp4 = tmp_path / "cli_out_viz"
    r2 = h.run_cli(
        [
            "-w",
            str(e2e_workspace),
            "-c",
            str(e2e_config_dir),
            "export",
            "--format",
            "viz_mp4",
            "--recipe",
            "bc_clean",
            "--out",
            str(out_mp4),
        ]
    )
    assert r2.returncode == 0, r2.stderr + r2.stdout
    mp4s = list(out_mp4.rglob("*.mp4"))
    assert len(mp4s) >= 1
    vinfo = h.ffprobe_video_info(mp4s[0])
    assert vinfo["width"] == 1920 and vinfo["height"] == 1080


def test_cli_export_dry_run_writes_nothing(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.store import Repository

    vdir = h.make_synthetic_video_dir(tmp_path / "ds_cli_dry", multi_cam=False, duration_s=1.5)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, vdir)
    repo = Repository(e2e_workspace)
    try:
        _seed_vlm_and_review(repo, ep_ids[0], duration_s=1.5, qc_segments=[])
    finally:
        repo.close()

    out_dry = tmp_path / "cli_dry_out"
    res = h.run_cli(
        [
            "-w",
            str(e2e_workspace),
            "-c",
            str(e2e_config_dir),
            "export",
            "--format",
            "lerobot_v3",
            "--recipe",
            "bc_clean",
            "--out",
            str(out_dry),
            "--dry-run",
        ]
    )
    assert res.returncode == 0, res.stderr + res.stdout
    assert not (out_dry / "data").exists()
    assert not (out_dry / "meta").exists()


# ===========================================================================
# FastAPI Auth, Episode List/Detail, Video/Frame Stream, SSE
# ===========================================================================


def test_auth_tokens_header_and_query_rbac(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    client = h.create_test_client(
        e2e_workspace,
        config_dir=e2e_config_dir,
        reviewer_token="rev-secret",
        lead_token="lead-secret",
    )
    # Unauthenticated or bad token -> 401/403
    r_noauth = client.get("/api/episodes")
    assert r_noauth.status_code in (401, 403)
    r_bad = client.get("/api/episodes", headers={"X-Auth-Token": "wrong-token"})
    assert r_bad.status_code in (401, 403)

    # Header & ?token= query auth work for both roles
    assert client.get("/api/episodes", headers={"X-Auth-Token": "rev-secret"}).status_code == 200
    assert client.get("/api/episodes?token=lead-secret").status_code == 200

    # Reviewer MUST get 403 on Lead-only routes (/api/lab/* and /api/exports/*)
    assert client.get("/api/lab/labels", headers={"X-Auth-Token": "rev-secret"}).status_code == 403
    assert client.post(
        "/api/exports/dry-run",
        json={"format": "zero_copy_index", "recipe": "bc_clean"},
        headers={"X-Auth-Token": "rev-secret"},
    ).status_code == 403


def test_episodes_list_and_filtering(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "sample_pick_and_place_api"
    h.make_synthetic_h5_raw_v3(h5_dir, name="ep_pass.h5", duration_s=2.0, episode_index=0)
    h.make_synthetic_h5_raw_v3(h5_dir, name="ep_fail.h5", duration_s=2.0, episode_index=1)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, h5_dir)
    assert len(ep_ids) == 2

    repo = Repository(e2e_workspace)
    try:
        segs_dg = [
            {
                "seg_id": "s_dg",
                "t_start": 0.6,
                "t_end": 1.2,
                "label_id": "mid_idle_freeze",
                "training_role": "badcase",
                "splice_policy": "mask_loss",
                "mask_side": "both",
                "vlm_raw_policy": "cut_clean",
                "guardrail": {
                    "gate_type": "splice_gate",
                    "passed": False,
                    "downgraded": True,
                    "downgrade_reason": "pose jump",
                },
            }
        ]
        _seed_vlm_and_review(
            repo, ep_ids[0], run_id="r0", duration_s=2.0, qc_segments=segs_dg, passed=True, approve_action=None
        )
        _seed_vlm_and_review(
            repo, ep_ids[1], run_id="r1", duration_s=2.0, qc_segments=[], passed=False, approve_action="approve"
        )
    finally:
        repo.close()

    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    hdrs = {"X-Auth-Token": "lead-token"}

    r_all = client.get("/api/episodes", headers=hdrs)
    assert r_all.status_code == 200
    body_all = r_all.json()
    items_all = body_all["items"] if isinstance(body_all, dict) and "items" in body_all else body_all
    assert len(items_all) == 2

    r_pass = client.get("/api/episodes?passed=true", headers=hdrs)
    items_pass = (
        r_pass.json()["items"] if isinstance(r_pass.json(), dict) and "items" in r_pass.json() else r_pass.json()
    )
    assert [it["episode_id"] for it in items_pass] == [ep_ids[0]]

    r_status = client.get("/api/episodes?status=approved", headers=hdrs)
    items_st = (
        r_status.json()["items"] if isinstance(r_status.json(), dict) and "items" in r_status.json() else r_status.json()
    )
    assert [it["episode_id"] for it in items_st] == [ep_ids[1]]

    r_task = client.get("/api/episodes?task_id=pick_and_place", headers=hdrs)
    items_tk = (
        r_task.json()["items"] if isinstance(r_task.json(), dict) and "items" in r_task.json() else r_task.json()
    )
    assert len(items_tk) == 2

    r_dg = client.get("/api/episodes?guardrail_downgraded=true", headers=hdrs)
    assert r_dg.status_code == 200
    items_dg = (
        r_dg.json()["items"] if isinstance(r_dg.json(), dict) and "items" in r_dg.json() else r_dg.json()
    )
    assert [it["episode_id"] for it in items_dg] == [ep_ids[0]]


def test_episode_detail_resolved_and_raw_vlm_comparison(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.store import Repository

    vdir = h.make_synthetic_video_dir(tmp_path / "ds_detail", multi_cam=True, duration_s=2.0)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, vdir)
    repo = Repository(e2e_workspace)
    try:
        _seed_vlm_and_review(repo, ep_ids[0], duration_s=2.0, approve_action="approve")
    finally:
        repo.close()

    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    resp = client.get(f"/api/episodes/{ep_ids[0]}", headers={"X-Auth-Token": "lead-token"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["episode_id"] == ep_ids[0]
    for key in (
        "qc_segments",
        "success",
        "macro_stages",
        "grounding",
        "arm_timelines",
        "raw_vlm_segments",
    ):
        assert key in data, f"Missing field {key} in episode detail"
    h.assert_no_legacy_fields(data)


def test_episode_preview_video_endpoint(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.store import Repository

    h5_dir = tmp_path / "ds_prev"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=1.5)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, h5_dir)
    repo = Repository(e2e_workspace)
    try:
        _seed_vlm_and_review(repo, ep_ids[0], duration_s=1.5, approve_action=None)
    finally:
        repo.close()

    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    resp = client.get(f"/api/episodes/{ep_ids[0]}/preview?token=rev-token")
    if resp.status_code == 404:
        resp = client.get(f"/api/episodes/{ep_ids[0]}/video?token=rev-token")
    assert resp.status_code in (200, 206)
    assert "video" in resp.headers.get("content-type", "") or len(resp.content) > 1000


def test_episode_direct_frame_stream_endpoint(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    h5_dir = tmp_path / "ds_frame"
    h.make_synthetic_h5_raw_v3(h5_dir, duration_s=1.5)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, h5_dir)

    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    resp = client.get(
        f"/api/episodes/{ep_ids[0]}/frame",
        params={"camera": "head_rgbd", "t_s": 0.5},
        headers={"X-Auth-Token": "rev-token"},
    )
    assert resp.status_code == 200
    assert "image/" in resp.headers.get("content-type", "")
    arr = np.frombuffer(resp.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert img is not None and img.shape[0] > 0 and img.shape[1] > 0


def test_sse_events_endpoint_connectivity(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    with client.stream("GET", "/api/events?token=lead-token") as resp:
        assert resp.status_code == 200
        ctype = resp.headers.get("content-type", "")
        assert "text/event-stream" in ctype
        first_chunk = next(resp.iter_lines(), None)
        assert first_chunk is not None


# ===========================================================================
# Reviews Router — Claim Leases, 3-Layer Edits, Freeze GT, RBAC
# ===========================================================================


def test_claim_lease_lock_renew_and_release(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.store import Repository

    vdir = h.make_synthetic_video_dir(tmp_path / "ds_claim", multi_cam=False, duration_s=2.0)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, vdir)
    repo = Repository(e2e_workspace)
    try:
        _seed_vlm_and_review(repo, ep_ids[0], duration_s=2.0, approve_action=None)
    finally:
        repo.close()

    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    hdrs = {"X-Auth-Token": "rev-token"}

    r_claim = client.post(f"/api/episodes/{ep_ids[0]}/claim", json={"user": "alice"}, headers=hdrs)
    assert r_claim.status_code == 200
    assert r_claim.json()["status"] == "claimed"

    r_conflict = client.post(f"/api/episodes/{ep_ids[0]}/claim", json={"user": "bob"}, headers=hdrs)
    assert r_conflict.status_code == 409

    r_renew = client.post(f"/api/episodes/{ep_ids[0]}/renew", json={"user": "alice"}, headers=hdrs)
    assert r_renew.status_code == 200
    assert r_renew.json()["claimed_by"] == "alice"

    r_rel = client.post(f"/api/episodes/{ep_ids[0]}/release", json={"user": "alice"}, headers=hdrs)
    assert r_rel.status_code == 200
    assert client.get(f"/api/episodes/{ep_ids[0]}", headers=hdrs).json()["status"] == "vlm_ready"


def test_reviewer_save_draft_and_submit_review(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.store import Repository

    vdir = h.make_synthetic_video_dir(tmp_path / "ds_sub", multi_cam=False, duration_s=2.0)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, vdir)
    repo = Repository(e2e_workspace)
    try:
        vlm, _ = _seed_vlm_and_review(repo, ep_ids[0], duration_s=2.0, approve_action=None)
    finally:
        repo.close()

    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    hdrs = {"X-Auth-Token": "rev-token", "X-User": "rev1"}
    assert client.post(f"/api/episodes/{ep_ids[0]}/claim", json={"user": "rev1"}, headers=hdrs).status_code == 200

    draft_body = {
        "base_run_id": vlm.run_id,
        "action": "save_draft",
        "qc_segments": [],
        "success": {"passed": True, "verdict": "success", "confidence": 0.9, "criteria": [], "summary": "draft"},
        "macro_stages": [{"stage_id": "S1", "t_start": 0.0, "t_end": 2.0, "label": "Full episode operation"}],
        "watch_coverage": {
            "watched_intervals": [[0.0, 1.0]],
            "watched_s": 1.0,
            "duration_s": 2.0,
            "coverage_ratio": 0.5,
            "confirmed": False,
        },
        "notes": "saving draft",
    }
    r_draft = client.post(f"/api/episodes/{ep_ids[0]}/reviews", json=draft_body, headers=hdrs)
    assert r_draft.status_code == 200
    assert client.get(f"/api/episodes/{ep_ids[0]}", headers=hdrs).json()["status"] == "claimed"

    submit_body = dict(draft_body)
    submit_body["action"] = "submit_review"
    submit_body["watch_coverage"] = {
        "watched_intervals": [[0.0, 2.0]],
        "watched_s": 2.0,
        "duration_s": 2.0,
        "coverage_ratio": 1.0,
        "confirmed": True,
    }
    r_sub = client.post(f"/api/episodes/{ep_ids[0]}/reviews", json=submit_body, headers=hdrs)
    assert r_sub.status_code == 200
    detail = client.get(f"/api/episodes/{ep_ids[0]}", headers=hdrs).json()
    assert detail["status"] == "reviewed"
    assert detail["macro_stages"][0]["label"] == "Full episode operation"


def test_reviewer_flag_unknown_or_difficult_episode(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.store import Repository

    vdir = h.make_synthetic_video_dir(tmp_path / "ds_flag", multi_cam=False, duration_s=2.0)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, vdir)
    repo = Repository(e2e_workspace)
    try:
        vlm, _ = _seed_vlm_and_review(repo, ep_ids[0], duration_s=2.0, approve_action=None)
    finally:
        repo.close()

    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    hdrs = {"X-Auth-Token": "rev-token", "X-User": "rev1"}
    flag_body = {
        "base_run_id": vlm.run_id,
        "action": "flag",
        "qc_segments": [s.model_dump(mode="json") for s in vlm.qc_segments],
        "success": vlm.success.model_dump(mode="json"),
        "macro_stages": [m.model_dump(mode="json") for m in vlm.macro_stages],
        "watch_coverage": {
            "watched_intervals": [[0.0, 2.0]],
            "watched_s": 2.0,
            "duration_s": 2.0,
            "coverage_ratio": 1.0,
            "confirmed": True,
        },
        "flagged_reason": "Unknown occlusion requires Lead arbitration",
    }
    r_flag = client.post(f"/api/episodes/{ep_ids[0]}/reviews", json=flag_body, headers=hdrs)
    assert r_flag.status_code == 200
    detail = client.get(f"/api/episodes/{ep_ids[0]}", headers=hdrs).json()
    assert detail["status"] == "flagged"
    assert detail["flagged_reason"] == "Unknown occlusion requires Lead arbitration"


def test_lead_approve_and_freeze_gt_immutable(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.store import Repository

    vdir = h.make_synthetic_video_dir(tmp_path / "ds_gt", multi_cam=False, duration_s=2.0)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, vdir)
    repo = Repository(e2e_workspace)
    try:
        vlm, _ = _seed_vlm_and_review(repo, ep_ids[0], duration_s=2.0, approve_action=None)
    finally:
        repo.close()

    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    lead_hdrs = {"X-Auth-Token": "lead-token", "X-User": "lead1"}
    base_body = {
        "base_run_id": vlm.run_id,
        "action": "approve",
        "qc_segments": [s.model_dump(mode="json") for s in vlm.qc_segments],
        "success": vlm.success.model_dump(mode="json"),
        "macro_stages": [m.model_dump(mode="json") for m in vlm.macro_stages],
        "watch_coverage": {
            "watched_intervals": [[0.0, 2.0]],
            "watched_s": 2.0,
            "duration_s": 2.0,
            "coverage_ratio": 1.0,
            "confirmed": True,
        },
    }
    assert client.post(f"/api/episodes/{ep_ids[0]}/reviews", json=base_body, headers=lead_hdrs).status_code == 200
    assert client.get(f"/api/episodes/{ep_ids[0]}", headers=lead_hdrs).json()["status"] == "approved"

    freeze_body = dict(base_body, action="freeze_gt")
    assert client.post(f"/api/episodes/{ep_ids[0]}/reviews", json=freeze_body, headers=lead_hdrs).status_code == 200
    assert client.get(f"/api/episodes/{ep_ids[0]}", headers=lead_hdrs).json()["status"] == "gt_frozen"

    # Subsequent mutations on gt_frozen MUST be rejected
    r_mut = client.post(f"/api/episodes/{ep_ids[0]}/reviews", json=base_body, headers=lead_hdrs)
    assert r_mut.status_code in (403, 409, 423)


def test_server_rejects_readonly_arm_timelines_and_grounding_mutation(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.store import Repository

    vdir = h.make_synthetic_video_dir(tmp_path / "ds_ro", multi_cam=False, duration_s=2.0)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, vdir)
    repo = Repository(e2e_workspace)
    try:
        vlm, _ = _seed_vlm_and_review(repo, ep_ids[0], duration_s=2.0, approve_action=None)
    finally:
        repo.close()

    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    hdrs = {"X-Auth-Token": "lead-token", "X-User": "lead1"}

    bad_payload = {
        "base_run_id": vlm.run_id,
        "action": "approve",
        "qc_segments": [],
        "success": vlm.success.model_dump(mode="json"),
        "macro_stages": [],
        "watch_coverage": {
            "watched_intervals": [[0.0, 2.0]],
            "watched_s": 2.0,
            "duration_s": 2.0,
            "coverage_ratio": 1.0,
            "confirmed": True,
        },
        "arm_timelines": {"left": [], "right": []},
        "grounding": {"object_tracks_2d": {}},
    }
    r_bad = client.post(f"/api/episodes/{ep_ids[0]}/reviews", json=bad_payload, headers=hdrs)
    assert r_bad.status_code in (400, 422)

    # Valid 3-layer edit leaves grounding and arm_timelines untouched
    good_payload = {k: v for k, v in bad_payload.items() if k not in ("arm_timelines", "grounding")}
    assert client.post(f"/api/episodes/{ep_ids[0]}/reviews", json=good_payload, headers=hdrs).status_code == 200
    detail = client.get(f"/api/episodes/{ep_ids[0]}", headers=hdrs).json()
    assert detail["grounding"] == vlm.grounding.model_dump(mode="json")
    assert detail["arm_timelines"] == vlm.arm_timelines.model_dump(mode="json")


def test_reviewer_cannot_approve_or_freeze_gt(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.store import Repository

    vdir = h.make_synthetic_video_dir(tmp_path / "ds_rev_rbac", multi_cam=False, duration_s=2.0)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, vdir)
    repo = Repository(e2e_workspace)
    try:
        vlm, _ = _seed_vlm_and_review(repo, ep_ids[0], duration_s=2.0, approve_action=None)
    finally:
        repo.close()

    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    rev_hdrs = {"X-Auth-Token": "rev-token", "X-User": "rev1"}
    for forbidden_act in ("approve", "freeze_gt"):
        r = client.post(
            f"/api/episodes/{ep_ids[0]}/reviews",
            json={
                "base_run_id": vlm.run_id,
                "action": forbidden_act,
                "qc_segments": [],
                "success": vlm.success.model_dump(mode="json"),
                "macro_stages": [],
                "watch_coverage": {
                    "watched_intervals": [[0.0, 2.0]],
                    "watched_s": 2.0,
                    "duration_s": 2.0,
                    "coverage_ratio": 1.0,
                    "confirmed": True,
                },
            },
            headers=rev_hdrs,
        )
        assert r.status_code == 403


# ===========================================================================
# Jobs Router (SSE), Exports Router, `robo-vlm-qc serve` CLI
# ===========================================================================


def test_jobs_batch_vlm_create_status_and_sse(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    vdir = h.make_synthetic_video_dir(tmp_path / "ds_job_vlm", multi_cam=False, duration_s=1.0)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, vdir)

    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    lead_hdrs = {"X-Auth-Token": "lead-token"}

    r_create = client.post(
        "/api/jobs",
        json={"job_type": "batch_vlm", "episode_ids": ep_ids},
        headers=lead_hdrs,
    )
    assert r_create.status_code in (200, 201, 202)
    job_id = r_create.json()["job_id"]

    # Poll until completed
    deadline = time.time() + 30.0
    job_data: dict[str, Any] = {}
    while time.time() < deadline:
        rj = client.get(f"/api/jobs/{job_id}", headers=lead_hdrs)
        assert rj.status_code == 200
        job_data = rj.json()
        if job_data["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)
    assert job_data["status"] == "completed", job_data
    assert job_data["progress_done"] == job_data["progress_total"] >= 1

    # Verify job SSE stream emits completion/progress event
    with client.stream("GET", f"/api/jobs/{job_id}/events?token=lead-token") as sse:
        assert sse.status_code == 200
        assert "text/event-stream" in sse.headers.get("content-type", "")
        lines = [ln for _, ln in zip(range(10), sse.iter_lines()) if ln]
        assert len(lines) >= 1


def test_jobs_stage_rerun_and_render_mp4(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    from robo_vlm_qc.store import Repository

    vdir = h.make_synthetic_video_dir(tmp_path / "ds_job_rerender", multi_cam=False, duration_s=1.0)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, vdir)
    repo = Repository(e2e_workspace)
    try:
        _seed_vlm_and_review(repo, ep_ids[0], duration_s=1.0, approve_action=None)
    finally:
        repo.close()

    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    lead_hdrs = {"X-Auth-Token": "lead-token"}

    for jpayload in (
        {"job_type": "batch_vlm", "episode_ids": ep_ids, "stage": "s3_discover"},
        {"job_type": "render_mp4", "episode_ids": ep_ids},
    ):
        rc = client.post("/api/jobs", json=jpayload, headers=lead_hdrs)
        assert rc.status_code in (200, 201, 202), rc.text
        jid = rc.json()["job_id"]
        deadline = time.time() + 30.0
        st = ""
        while time.time() < deadline:
            jd = client.get(f"/api/jobs/{jid}", headers=lead_hdrs).json()
            st = jd["status"]
            if st in ("completed", "failed"):
                break
            time.sleep(0.1)
        assert st == "completed", f"Job {jpayload['job_type']} ended with {st}"


def test_exports_dry_run_preview(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.store import Repository

    vdir = h.make_synthetic_video_dir(tmp_path / "ds_exp_dry", multi_cam=False, duration_s=2.0)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, vdir)
    repo = Repository(e2e_workspace)
    try:
        _seed_vlm_and_review(repo, ep_ids[0], duration_s=2.0, approve_action="approve")
    finally:
        repo.close()

    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    resp = client.post(
        "/api/exports/dry-run",
        json={"format": "lerobot_v3", "recipe": "bc_clean"},
        headers={"X-Auth-Token": "lead-token"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["episodes_included"] == 1
    assert body["pieces_exported"] >= 1
    assert body["total_frames"] > 0
    assert list((e2e_workspace / "exports").iterdir()) == []


def test_exports_trigger_and_download_manifest(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.store import Repository

    vdir = h.make_synthetic_video_dir(tmp_path / "ds_exp_dl", multi_cam=False, duration_s=1.5)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, vdir)
    repo = Repository(e2e_workspace)
    try:
        _seed_vlm_and_review(repo, ep_ids[0], duration_s=1.5, approve_action="approve")
    finally:
        repo.close()

    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    lead_hdrs = {"X-Auth-Token": "lead-token"}

    r_exp = client.post(
        "/api/exports",
        json={"format": "zero_copy_index", "recipe": "bc_clean"},
        headers=lead_hdrs,
    )
    assert r_exp.status_code in (200, 201, 202), r_exp.text
    exp_body = r_exp.json()
    exp_id = exp_body["export_id"]

    r_get = client.get(f"/api/exports/{exp_id}", headers=lead_hdrs)
    assert r_get.status_code == 200
    manifest_data = r_get.json()
    assert manifest_data["export_id"] == exp_id
    assert manifest_data["episodes_included"] == 1

    r_dl = client.get(f"/api/exports/{exp_id}/download", headers=lead_hdrs)
    assert r_dl.status_code == 200
    assert len(r_dl.content) > 0


def test_reviewer_forbidden_from_exports_and_admin_jobs(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    rev_hdrs = {"X-Auth-Token": "rev-token"}

    assert client.get("/api/exports", headers=rev_hdrs).status_code == 403
    assert client.post(
        "/api/exports",
        json={"format": "zero_copy_index", "recipe": "bc_clean"},
        headers=rev_hdrs,
    ).status_code == 403
    assert client.post(
        "/api/jobs",
        json={"job_type": "batch_vlm", "episode_ids": []},
        headers=rev_hdrs,
    ).status_code == 403


def test_cli_serve_starts_and_responds_on_ephemeral_port(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    port = _free_port()
    cmd = h._resolve_cli_cmd() + [
        "-w",
        str(e2e_workspace),
        "-c",
        str(e2e_config_dir),
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--reviewer-token",
        "rev-cli-tok",
        "--lead-token",
        "lead-cli-tok",
    ]
    env = h.os.environ.copy()
    for key_var in ("OPENAI_API_KEY", "DASHSCOPE_API_KEY", "GEMINI_API_KEY"):
        env.pop(key_var, None)
    for pv in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        env.pop(pv, None)
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["PYTHONPATH"] = str(h.REPO_ROOT)

    proc = subprocess.Popen(
        cmd,
        cwd=str(h.REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.time() + 15.0
        ok = False
        with httpx.Client(timeout=2.0, trust_env=False) as http:
            while time.time() < deadline:
                if proc.poll() is not None:
                    break
                try:
                    r = http.get(
                        f"http://127.0.0.1:{port}/api/episodes",
                        headers={"X-Auth-Token": "lead-cli-tok"},
                    )
                    if r.status_code == 200:
                        ok = True
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.2)
        assert ok, f"serve CLI did not become ready on port {port} (rc={proc.poll()})"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3.0)


# ===========================================================================
# React 19 Studio Integration Contracts
# ===========================================================================


def test_web_dist_built_and_served_at_root(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    dist_index = h.REPO_ROOT / "web" / "dist" / "index.html"
    if not dist_index.is_file():
        pytest.skip(f"Built SPA index not present: {dist_index}")
    index_html = dist_index.read_text("utf-8")
    assert "<div" in index_html and "script" in index_html.lower()

    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    r_root = client.get("/")
    assert r_root.status_code == 200
    assert "text/html" in r_root.headers.get("content-type", "")
    assert "root" in r_root.text

    assets_dir = h.REPO_ROOT / "web" / "dist" / "assets"
    if assets_dir.is_dir():
        asset_files = [p for p in assets_dir.iterdir() if p.is_file()]
        assert len(asset_files) >= 1
        r_asset = client.get(f"/assets/{asset_files[0].name}")
        assert r_asset.status_code == 200


def test_web_source_pages_and_shared_components_present() -> None:
    web_dir = h.REPO_ROOT / "web"
    web_src = web_dir / "src"
    required_files = [
        "App.tsx",
        "main.tsx",
        "pages/QueuePage.tsx",
        "pages/ReviewStudioPage.tsx",
        "pages/VlmLabPage.tsx",
        "pages/ExportPage.tsx",
        "components/StudioPlayer.tsx",
        "components/FourTrackTimeline.tsx",
        "components/SegmentInspector.tsx",
        "components/SuccessPanel.tsx",
        "components/DiffOverlay.tsx",
    ]
    for rel in required_files:
        fp = web_src / rel
        assert fp.is_file(), f"Missing React Studio source file: {fp}"
        assert fp.stat().st_size > 50, f"Empty React Studio source file: {fp}"

    pkg_json = web_dir / "package.json"
    assert pkg_json.is_file(), f"Missing web/package.json: {pkg_json}"
    pkg = json.loads(pkg_json.read_text("utf-8"))
    scripts = pkg.get("scripts", {})
    assert "test" in scripts and "build" in scripts, (
        f"web/package.json must define 'test' and 'build' scripts: {scripts}"
    )
    fe_tests = [
        p
        for p in web_src.rglob("*")
        if p.is_file() and (".test." in p.name or ".spec." in p.name)
    ]
    assert len(fe_tests) >= 1, f"Expected frontend test files under {web_src}"

    npm_bin = shutil.which("npm")
    if npm_bin is None or not (web_dir / "node_modules").is_dir():
        pytest.skip("npm or web/node_modules not available in this environment")
    npm_env = h.os.environ.copy()
    npm_env.setdefault("CI", "true")
    npm_res = subprocess.run(
        [npm_bin, "test"],
        cwd=str(web_dir),
        env=npm_env,
        capture_output=True,
        text=True,
        timeout=120.0,
    )
    assert npm_res.returncode == 0, f"npm test failed:\n{npm_res.stdout}\n{npm_res.stderr}"


def test_queue_page_role_filtering_contract(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.store import Repository

    vdir = h.make_synthetic_video_dir(tmp_path / "ds_queue_ui", multi_cam=False, duration_s=1.5, n_episodes=2)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, vdir)
    repo = Repository(e2e_workspace)
    try:
        _seed_vlm_and_review(repo, ep_ids[0], run_id="r0", duration_s=1.5, approve_action=None)
        _seed_vlm_and_review(repo, ep_ids[1], run_id="r1", duration_s=1.5, approve_action="approve")
    finally:
        repo.close()

    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    # Reviewer sees vlm_ready queue items
    r_rev = client.get("/api/episodes?status=vlm_ready", headers={"X-Auth-Token": "rev-token"})
    assert r_rev.status_code == 200
    items_rev = r_rev.json()["items"] if isinstance(r_rev.json(), dict) and "items" in r_rev.json() else r_rev.json()
    assert [it["episode_id"] for it in items_rev] == [ep_ids[0]]

    # Lead sees global queue + stats across statuses
    r_lead = client.get("/api/episodes", headers={"X-Auth-Token": "lead-token"})
    assert r_lead.status_code == 200
    items_lead = (
        r_lead.json()["items"] if isinstance(r_lead.json(), dict) and "items" in r_lead.json() else r_lead.json()
    )
    assert len(items_lead) == 2


def test_review_studio_3layer_diff_and_undo_redo_snapshot_history(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.store import Repository

    vdir = h.make_synthetic_video_dir(tmp_path / "ds_studio_ui", multi_cam=False, duration_s=2.0)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, vdir)
    repo = Repository(e2e_workspace)
    try:
        vlm, _ = _seed_vlm_and_review(repo, ep_ids[0], duration_s=2.0, approve_action=None)
    finally:
        repo.close()

    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    rev_hdrs = {"X-Auth-Token": "rev-token", "X-User": "rev_a"}
    lead_hdrs = {"X-Auth-Token": "lead-token", "X-User": "lead_a"}

    # Revision 1: Reviewer edits qc_segments + macro_stages + success
    rev1_payload = {
        "base_run_id": vlm.run_id,
        "action": "submit_review",
        "qc_segments": [],
        "success": {"passed": False, "verdict": "failure", "confidence": 0.9, "criteria": [], "summary": "rev1"},
        "macro_stages": [{"stage_id": "S1", "t_start": 0.0, "t_end": 2.0, "label": "Revision 1"}],
        "watch_coverage": {
            "watched_intervals": [[0.0, 2.0]],
            "watched_s": 2.0,
            "duration_s": 2.0,
            "coverage_ratio": 1.0,
            "confirmed": True,
        },
    }
    assert client.post(f"/api/episodes/{ep_ids[0]}/reviews", json=rev1_payload, headers=rev_hdrs).status_code == 200

    # Revision 2: Lead edits again
    rev2_payload = dict(rev1_payload, action="approve", notes="rev2")
    rev2_payload["macro_stages"] = [{"stage_id": "S1", "t_start": 0.0, "t_end": 2.0, "label": "Revision 2"}]
    assert client.post(f"/api/episodes/{ep_ids[0]}/reviews", json=rev2_payload, headers=lead_hdrs).status_code == 200

    # Detail contains DiffOverlay fields comparing latest vs raw VLM
    detail = client.get(f"/api/episodes/{ep_ids[0]}", headers=lead_hdrs).json()
    assert detail["review_diff"]["success_flipped"] is True
    assert len(detail["raw_vlm_segments"]) == len(vlm.qc_segments)
    assert detail["macro_stages"][0]["label"] == "Revision 2"

    # History endpoint returns ordered immutable snapshots enabling Undo/Redo restore
    r_hist = client.get(f"/api/episodes/{ep_ids[0]}/reviews", headers=lead_hdrs)
    assert r_hist.status_code == 200
    revs = r_hist.json()["items"] if isinstance(r_hist.json(), dict) and "items" in r_hist.json() else r_hist.json()
    assert len(revs) >= 2


def test_export_page_4x3_matrix_dry_run_and_trigger(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.store import Repository

    vdir = h.make_synthetic_video_dir(tmp_path / "ds_exp_ui", multi_cam=False, duration_s=1.5)
    ep_ids = _ingest_dir(e2e_workspace, e2e_config_dir, vdir)
    repo = Repository(e2e_workspace)
    try:
        _seed_vlm_and_review(repo, ep_ids[0], duration_s=1.5, approve_action="approve")
    finally:
        repo.close()

    client = h.create_test_client(e2e_workspace, config_dir=e2e_config_dir)
    lead_hdrs = {"X-Auth-Token": "lead-token"}

    for fmt in ("lerobot_v3", "h5_clean", "zero_copy_index", "viz_mp4"):
        for recipe_name in ("bc_clean", "recovery_plus", "badcase_pairs"):
            r_dry = client.post(
                "/api/exports/dry-run",
                json={"format": fmt, "recipe": recipe_name},
                headers=lead_hdrs,
            )
            assert r_dry.status_code == 200, f"dry-run failed for {fmt} x {recipe_name}: {r_dry.text}"

    r_run = client.post(
        "/api/exports",
        json={"format": "zero_copy_index", "recipe": "recovery_plus"},
        headers=lead_hdrs,
    )
    assert r_run.status_code in (200, 201, 202)
    r_list = client.get("/api/exports", headers=lead_hdrs)
    assert r_list.status_code == 200
    listed = r_list.json()["items"] if isinstance(r_list.json(), dict) and "items" in r_list.json() else r_list.json()
    assert len(listed) >= 1
