"""Export, FastAPI Studio, RBAC & VLM Lab Boundary, Corner & Negative E2E Tests.

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
import time
from pathlib import Path
from typing import Any

import h5py
import httpx
import numpy as np
import pyarrow.parquet as pq
import pytest

from tests.e2e import helpers as h


# ---------------------------------------------------------------------------
# Lazy Helpers & Fixture Builders
# ---------------------------------------------------------------------------


def _call_with_matching_kwargs(fn: Any, **candidates: Any) -> Any:
    sig = inspect.signature(fn)
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if accepts_var_kw:
        return fn(**candidates)
    kwargs = {k: v for k, v in candidates.items() if k in sig.parameters}
    return fn(**kwargs)


def _setup_episode_in_repo(
    ws: Path,
    cfg_dir: Path,
    data_dir: Path,
    *,
    adapter_type: str = "h5_raw_v3",
    passed: bool | None = True,
    qc_segments: list[dict[str, Any]] | None = None,
    macro_stages: list[dict[str, Any]] | None = None,
    status_action: str = "approve",
    duration_s: float = 4.0,
) -> tuple[str, str]:
    """Ingest a synthetic episode, attach a VlmRunArtifact, and optionally review/approve/freeze."""
    from robo_vlm_qc.io import auto_detect, get_adapter
    from robo_vlm_qc.models.annotation import (
        ArmSpan,
        ArmTimelines,
        ConfigFingerprintSnapshot,
        CriterionResult,
        MacroStage,
        QcSegment,
        SuccessVerdict,
        VlmRunArtifact,
    )
    from robo_vlm_qc.models.review import ReviewWritePayload, WatchCoverage
    from robo_vlm_qc.store.repository import Repository

    if adapter_type == "h5_raw_v3":
        ds_dir = data_dir / "pick_and_place_ds"
        h.make_synthetic_h5_raw_v3(ds_dir, duration_s=duration_s, task_name="pick_and_place")
        handles = auto_detect(ds_dir, adapter_name="h5_raw_v3")
    elif adapter_type == "lerobot":
        ds_dir = data_dir / "lr_ds"
        h.make_synthetic_lerobot(
            ds_dir, version="v3", n_episodes=1, duration_s=duration_s, task_name="pick_and_place"
        )
        handles = auto_detect(ds_dir, adapter_name="lerobot")
    else:
        ds_dir = data_dir / "vid_ds"
        h.make_synthetic_video_dir(ds_dir, multi_cam=False, duration_s=duration_s)
        handles = auto_detect(ds_dir, adapter_name="video_dir")

    assert handles, f"Failed to discover synthetic {adapter_type}"
    ad = get_adapter(handles[0].adapter_type)
    with ad.open(handles[0]) as view:
        meta = view.meta

    segs = [QcSegment.model_validate(s) for s in (qc_segments or [])]
    if macro_stages is None:
        stages = [
            MacroStage(
                stage_id="S1",
                t_start=0.0,
                t_end=round(meta.duration_s * 0.5, 3),
                label="Stage One",
                label_en="stage_one",
            ),
            MacroStage(
                stage_id="S2",
                t_start=round(meta.duration_s * 0.5, 3),
                t_end=round(meta.duration_s, 3),
                label="Stage Two",
                label_en="stage_two",
            ),
        ]
    else:
        stages = [MacroStage.model_validate(m) for m in macro_stages]

    verdict = "success" if passed is True else ("failure" if passed is False else "unsure")
    run_id = f"run_{int(time.time() * 1000)}"
    run = VlmRunArtifact(
        run_id=run_id,
        episode_id=meta.episode_id,
        created_at=time.time(),
        config_fingerprint=ConfigFingerprintSnapshot(
            model_routing={"s1_grounding": "gpt-4o"},
            prompt_bundle_version="1.0.0",
            label_taxonomy_version="1",
            task_spec_version="1.0.0",
        ),
        macro_stages=stages,
        arm_timelines=ArmTimelines(
            left=[ArmSpan(t_start=0.0, t_end=meta.duration_s, state="work")],
            right=[ArmSpan(t_start=0.0, t_end=meta.duration_s, state="hold")],
        ),
        qc_segments=segs,
        success=SuccessVerdict(
            passed=passed,
            verdict=verdict,  # type: ignore[arg-type]
            confidence=0.92,
            criteria=[CriterionResult(criterion="Goal satisfied", met=passed)],
        ),
    )

    repo = Repository(ws)
    try:
        repo.ingest_episode(meta)
        repo.save_vlm_run_artifact(run)
        if status_action in ("submit_review", "flag", "approve", "freeze_gt"):
            repo.save_review(
                meta.episode_id,
                "lead_setup",
                "lead",
                ReviewWritePayload(
                    base_run_id=run_id,
                    action=status_action,  # type: ignore[arg-type]
                    qc_segments=segs,
                    success=run.success,
                    macro_stages=stages,
                    watch_coverage=WatchCoverage(
                        duration_s=meta.duration_s,
                        watched_s=meta.duration_s,
                        coverage_ratio=1.0,
                        confirmed=True,
                    ),
                ),
            )
    finally:
        repo.close()

    return meta.episode_id, run_id


def _slice_episode(resolved: Any, view: Any, recipe: Any) -> list[Any]:
    slicer_mod = importlib.import_module("robo_vlm_qc.export.slicer")
    slicer_cls = getattr(slicer_mod, "PieceSlicer", None)
    if slicer_cls is not None:
        inst = slicer_cls()
        return inst.slice(resolved, view, recipe)
    fn = getattr(slicer_mod, "slice_episode")
    return fn(resolved, view, recipe)


def _export_with(
    mod_name: str,
    cls_candidates: tuple[str, ...],
    pieces: list[Any],
    resolved_list: list[Any],
    views: dict[str, Any],
    out_dir: Path,
    recipe: Any,
) -> Any:
    mod = importlib.import_module(mod_name)
    for cname in cls_candidates:
        cls = getattr(mod, cname, None)
        if cls is not None:
            inst = cls()
            return _call_with_matching_kwargs(
                inst.export,
                pieces=pieces,
                resolved_list=resolved_list,
                views=views,
                out_dir=out_dir,
                recipe=recipe,
            )
    base_mod = importlib.import_module("robo_vlm_qc.export.base")
    get_exp = getattr(base_mod, "get_exporter", None)
    if callable(get_exp):
        inst = get_exp(recipe.format)
        return inst.export(pieces, resolved_list, views, out_dir, recipe)
    raise AssertionError(f"No exporter found in {mod_name}")


def _open_view_for_resolved(resolved: Any) -> Any:
    from robo_vlm_qc.io import auto_detect, get_adapter

    ad = get_adapter(resolved.meta.adapter_type)
    handles = auto_detect(Path(resolved.meta.source_uri), adapter_name=resolved.meta.adapter_type)
    target = next((hd for hd in handles if hd.episode_id == resolved.episode_id), handles[0])
    return ad.open(target)


# ===========================================================================
# PieceSlicer, 3 Recipes & _staging -> rename Atomic Lock
# ===========================================================================


def test_bc_clean_drops_failed_episodes(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.models.export import ExportRecipe
    from robo_vlm_qc.store.repository import Repository

    ep_id, _ = _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        tmp_path / "data",
        adapter_type="video_dir",
        passed=False,
        status_action="approve",
    )
    repo = Repository(e2e_workspace)
    try:
        resolved = repo.resolve_episode(ep_id)
    finally:
        repo.close()

    with _open_view_for_resolved(resolved) as view:
        recipe = ExportRecipe(name="bc_clean", format="lerobot_v3")
        pieces = _slice_episode(resolved, view, recipe)
        kept = [p for p in pieces if not p.dropped]
        assert kept == []


def test_piece_slicer_drops_sub_min_piece_s_fragments(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.models.export import ExportRecipe
    from robo_vlm_qc.store.repository import Repository

    # 2.0s episode with [0.0, 1.94] cut_clean leaves only 0.06s (< min_piece_s=0.2s)
    ep_id, _ = _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        tmp_path / "data",
        adapter_type="video_dir",
        passed=True,
        duration_s=2.0,
        qc_segments=[
            {
                "seg_id": "seg_cut",
                "t_start": 0.0,
                "t_end": 1.94,
                "label_id": "boundary_idle",
                "training_role": "badcase",
                "splice_policy": "cut_clean",
                "mask_side": None,
                "affected_arm": "both",
                "vlm_raw_policy": "cut_clean",
            }
        ],
    )
    repo = Repository(e2e_workspace)
    try:
        resolved = repo.resolve_episode(ep_id)
    finally:
        repo.close()

    with _open_view_for_resolved(resolved) as view:
        recipe = ExportRecipe(name="bc_clean", format="lerobot_v3", min_piece_s=0.2)
        pieces = _slice_episode(resolved, view, recipe)
        kept = [p for p in pieces if not p.dropped]
        assert kept == []


def test_cut_clean_spanning_whole_episode_yields_zero_pieces(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.models.export import ExportRecipe
    from robo_vlm_qc.store.repository import Repository

    ep_id, _ = _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        tmp_path / "data",
        adapter_type="video_dir",
        passed=True,
        duration_s=2.0,
        qc_segments=[
            {
                "seg_id": "seg_all",
                "t_start": 0.0,
                "t_end": 2.0,
                "label_id": "boundary_idle",
                "training_role": "badcase",
                "splice_policy": "cut_clean",
                "mask_side": None,
                "affected_arm": "both",
                "vlm_raw_policy": "cut_clean",
            }
        ],
    )
    repo = Repository(e2e_workspace)
    try:
        resolved = repo.resolve_episode(ep_id)
    finally:
        repo.close()

    with _open_view_for_resolved(resolved) as view:
        pieces = _slice_episode(
            resolved, view, ExportRecipe(name="bc_clean", format="lerobot_v3")
        )
        kept = [p for p in pieces if not p.dropped]
        assert kept == []


def test_recovery_plus_masks_badcase_0_and_unmasks_recovery_1(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.models.export import ExportRecipe
    from robo_vlm_qc.store.repository import Repository

    ep_id, _ = _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        tmp_path / "data",
        adapter_type="video_dir",
        passed=True,
        duration_s=3.0,
        qc_segments=[
            {
                "seg_id": "seg_rec",
                "t_start": 0.4,
                "t_end": 1.0,
                "label_id": "missed_grasp_retry",
                "training_role": "recovery",
                "splice_policy": "mask_loss",
                "mask_side": "left",
                "affected_arm": "left",
                "vlm_raw_policy": "mask_loss",
            },
            {
                "seg_id": "seg_bad",
                "t_start": 1.5,
                "t_end": 2.2,
                "label_id": "hesitation_oscillation",
                "training_role": "badcase",
                "splice_policy": "mask_loss",
                "mask_side": "both",
                "affected_arm": "both",
                "vlm_raw_policy": "mask_loss",
            },
        ],
    )
    repo = Repository(e2e_workspace)
    try:
        resolved = repo.resolve_episode(ep_id)
    finally:
        repo.close()

    with _open_view_for_resolved(resolved) as view:
        rec_pieces = _slice_episode(
            resolved, view, ExportRecipe(name="recovery_plus", format="lerobot_v3")
        )
        kept = [p for p in rec_pieces if not p.dropped]
        assert len(kept) == 1
        mask = np.asarray(kept[0].action_loss_mask, dtype=np.float32)
        # During recovery [0.4s..1.0s] -> frames ~15..25: unmasked [1.0, 1.0]
        assert np.allclose(mask[18], [1.0, 1.0])
        # During badcase [1.5s..2.2s] -> frames ~50..60: masked [0.0, 0.0]
        assert np.allclose(mask[55], [0.0, 0.0])

        # Compare with bc_clean: recovery left arm is masked [0.0, 1.0]
        bc_pieces = _slice_episode(
            resolved, view, ExportRecipe(name="bc_clean", format="lerobot_v3")
        )
        bc_mask = np.asarray([p for p in bc_pieces if not p.dropped][0].action_loss_mask)
        assert np.allclose(bc_mask[18], [0.0, 1.0])


def test_badcase_pairs_handles_episode_with_zero_badcases(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.models.export import ExportRecipe
    from robo_vlm_qc.store.repository import Repository

    ep_id, _ = _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        tmp_path / "data",
        adapter_type="video_dir",
        passed=True,
        qc_segments=[],
    )
    repo = Repository(e2e_workspace)
    try:
        resolved = repo.resolve_episode(ep_id)
    finally:
        repo.close()

    with _open_view_for_resolved(resolved) as view:
        pieces = _slice_episode(
            resolved, view, ExportRecipe(name="badcase_pairs", format="lerobot_v3")
        )
        kept = [p for p in pieces if not p.dropped]
        assert kept == []


def test_staging_failure_leaves_no_partial_final_export_dir(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.models.export import ExportRecipe
    from robo_vlm_qc.store.repository import Repository

    ep_id, _ = _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        tmp_path / "data",
        adapter_type="video_dir",
        passed=True,
    )
    repo = Repository(e2e_workspace)
    try:
        resolved = repo.resolve_episode(ep_id)
    finally:
        repo.close()

    class _BrokenView:
        meta = resolved.meta
        kinematics = None

        @property
        def media(self) -> Any:
            raise RuntimeError("Simulated mid-export I/O failure")

    out_dir = tmp_path / "atomic_fail_export"
    recipe = ExportRecipe(name="bc_clean", format="lerobot_v3")
    with _open_view_for_resolved(resolved) as real_view:
        pieces = [
            p for p in _slice_episode(resolved, real_view, recipe) if not p.dropped
        ]
    with pytest.raises(Exception):
        _export_with(
            "robo_vlm_qc.export.lerobot_v3",
            ("LeRobotV3Exporter",),
            pieces,
            [resolved],
            {ep_id: _BrokenView()},
            out_dir,
            recipe,
        )
    # Atomic staging contract: final out_dir MUST NOT contain partial incomplete export
    assert not (out_dir / "meta" / "info.json").exists()


# ===========================================================================
# LeRobot v3.0 Exporter Boundaries
# ===========================================================================


def test_lerobot_v3_zero_kept_episodes_cleanly_handled(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "empty_lr_export"
    proc = h.run_cli(
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
            str(out_dir),
        ]
    )
    # Either non-zero clean exit or 0-episode manifest without broken partial output
    if proc.returncode == 0:
        assert not (out_dir / "data").exists() or list((out_dir / "data").rglob("*.parquet")) == []
    else:
        assert not (out_dir / "meta" / "info.json").exists()


def test_lerobot_v3_chunk_splitting_max_file_mb_boundary(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    h.write_workspace_config(
        e2e_workspace,
        extra_overrides={"export": {"lerobot_v3_max_file_mb": 0.001}},
    )
    ep_id, _ = _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        tmp_path / "data",
        adapter_type="h5_raw_v3",
        passed=True,
        duration_s=2.0,
    )
    assert ep_id
    out_dir = tmp_path / "chunk_split_lr"
    proc = h.run_cli(
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
            str(out_dir),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    res = h.assert_81d_lerobot_v3_valid(out_dir)
    assert res["total_rows"] > 0
    assert res["info"]["robot_type"] == "bimanual_robot"


def test_lerobot_v3_single_frame_is_tail_edge(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.models.export import ExportRecipe
    from robo_vlm_qc.store.repository import Repository

    ep_id, _ = _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        tmp_path / "data",
        adapter_type="h5_raw_v3",
        passed=True,
        duration_s=2.0,
        qc_segments=[
            {
                "seg_id": "mid_cut",
                "t_start": 0.8,
                "t_end": 1.2,
                "label_id": "mid_idle_freeze",
                "training_role": "badcase",
                "splice_policy": "cut_clean",
                "mask_side": None,
                "affected_arm": "both",
                "vlm_raw_policy": "cut_clean",
            }
        ],
    )
    repo = Repository(e2e_workspace)
    try:
        resolved = repo.resolve_episode(ep_id)
    finally:
        repo.close()

    recipe = ExportRecipe(
        name="bc_clean",
        format="lerobot_v3",
        tail_frames=1,
        smooth_tail="always",
    )
    out_dir = tmp_path / "tail1_lr"
    with _open_view_for_resolved(resolved) as view:
        pieces = [p for p in _slice_episode(resolved, view, recipe) if not p.dropped]
        assert len(pieces) == 2
        assert pieces[0].tail_rows <= 1
        _export_with(
            "robo_vlm_qc.export.lerobot_v3",
            ("LeRobotV3Exporter",),
            pieces,
            [resolved],
            {ep_id: view},
            out_dir,
            recipe,
        )
    res = h.assert_81d_lerobot_v3_valid(out_dir)
    assert res["info"]["robot_type"] == "bimanual_robot"


def test_lerobot_v3_subtasks_parquet_when_macro_stages_empty(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        tmp_path / "data",
        adapter_type="h5_raw_v3",
        passed=True,
        macro_stages=[],
        duration_s=2.0,
    )
    out_dir = tmp_path / "empty_subtasks_lr"
    proc = h.run_cli(
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
            str(out_dir),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    h.assert_81d_lerobot_v3_valid(out_dir, require_nonempty_subtasks=False)
    sub_tbl = pq.read_table(out_dir / "meta" / "subtasks.parquet")
    assert {"subtask_index", "subtask"}.issubset(set(sub_tbl.column_names))


def test_lerobot_v3_action_at_last_frame_repeats_state_with_smoothing(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.models.export import ExportRecipe
    from robo_vlm_qc.store.repository import Repository

    ep_id, _ = _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        tmp_path / "data",
        adapter_type="h5_raw_v3",
        passed=True,
        duration_s=2.0,
    )
    repo = Repository(e2e_workspace)
    try:
        resolved = repo.resolve_episode(ep_id)
    finally:
        repo.close()

    out_dir = tmp_path / "last_action_lr"
    recipe = ExportRecipe(
        name="bc_clean", format="lerobot_v3", tail_frames=2, smooth_tail="always"
    )
    with _open_view_for_resolved(resolved) as view:
        pieces = [p for p in _slice_episode(resolved, view, recipe) if not p.dropped]
        _export_with(
            "robo_vlm_qc.export.lerobot_v3",
            ("LeRobotV3Exporter",),
            pieces,
            [resolved],
            {ep_id: view},
            out_dir,
            recipe,
        )

    pqs = sorted((out_dir / "data").rglob("*.parquet"))
    tbl = pq.read_table(pqs[0])
    s_last = np.asarray(tbl["observation.state"][-1].as_py(), dtype=np.float32)
    a_last = np.asarray(tbl["action"][-1].as_py(), dtype=np.float32)
    assert s_last.shape == (81,) and a_last.shape == (81,)
    assert np.allclose(a_last, s_last, atol=1e-4)


def test_lerobot_v3_action_loss_mask_shape_n_by_2_float32(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        tmp_path / "data",
        adapter_type="h5_raw_v3",
        passed=True,
        duration_s=2.0,
        qc_segments=[
            {
                "seg_id": "left_mask",
                "t_start": 0.5,
                "t_end": 1.0,
                "label_id": "missed_grasp_retry",
                "training_role": "recovery",
                "splice_policy": "mask_loss",
                "mask_side": "left",
                "affected_arm": "left",
                "vlm_raw_policy": "mask_loss",
            }
        ],
    )
    out_dir = tmp_path / "mask_shape_lr"
    proc = h.run_cli(
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
            str(out_dir),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    h.assert_81d_lerobot_v3_valid(out_dir)
    pqs = sorted((out_dir / "data").rglob("*.parquet"))
    tbl = pq.read_table(pqs[0])
    arr = np.asarray(tbl["action_loss_mask"].to_pylist(), dtype=np.float32)
    assert arr.ndim == 2 and arr.shape[1] == 2
    assert np.isin(arr, [0.0, 1.0]).all()
    assert (arr[:, 0] == 0.0).any()
    assert (arr[:, 1] == 1.0).all()


# ===========================================================================
# h5_clean & zero_copy_index Exporters Boundaries
# ===========================================================================


def test_h5_clean_preserves_shapes_when_cut_clean_trims_ends(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        tmp_path / "data",
        adapter_type="h5_raw_v3",
        passed=True,
        duration_s=3.0,
        qc_segments=[
            {
                "seg_id": "trim_start",
                "t_start": 0.0,
                "t_end": 0.6,
                "label_id": "boundary_idle",
                "training_role": "badcase",
                "splice_policy": "cut_clean",
                "mask_side": None,
                "affected_arm": "both",
                "vlm_raw_policy": "cut_clean",
            }
        ],
    )
    out_dir = tmp_path / "h5_clean_trim"
    proc = h.run_cli(
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
            str(out_dir),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    h5_files = list(out_dir.rglob("*.h5"))
    assert len(h5_files) >= 1
    with h5py.File(h5_files[0], "r") as f:
        mask = f["/qc/action_loss_mask"][:]
        is_tail = f["/qc/is_tail"][:]
        subtask_idx = f["/qc/subtask_index"][:]
        resolved_raw = f["/qc/resolved_json"][()]
        if isinstance(resolved_raw, bytes):
            resolved_raw = resolved_raw.decode("utf-8")
        resolved_obj = json.loads(resolved_raw)
        h.assert_no_legacy_fields(resolved_obj)
        assert mask.ndim == 2 and mask.shape[1] == 2
        assert len(is_tail) == mask.shape[0]
        assert len(subtask_idx) == mask.shape[0]
        assert mask.shape[0] < 90  # trimmed from 3.0s (~90 frames)


def test_h5_clean_on_video_dir_minimal_streams(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        tmp_path / "data",
        adapter_type="video_dir",
        passed=True,
        duration_s=1.5,
    )
    out_dir = tmp_path / "h5_clean_vid"
    proc = h.run_cli(
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
            str(out_dir),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    h5_files = list(out_dir.rglob("*.h5"))
    assert h5_files
    with h5py.File(h5_files[0], "r") as f:
        assert "/qc/action_loss_mask" in f
        assert "/qc/is_tail" in f
        assert "/qc/subtask_index" in f
        assert "/qc/resolved_json" in f


def test_zero_copy_index_never_writes_frame_data_tiny_size(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        tmp_path / "data",
        adapter_type="h5_raw_v3",
        passed=True,
        duration_s=2.0,
    )
    out_dir = tmp_path / "zero_copy_out"
    proc = h.run_cli(
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
            str(out_dir),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    assert (out_dir / "qc_index.parquet").is_file()
    assert (out_dir / "manifest.jsonl").is_file()
    assert (out_dir / "subtasks.json").is_file()

    total_bytes = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    assert 0 < total_bytes < 512 * 1024  # < 512 KB, zero frame/mp4/h5 data copied
    assert list(out_dir.rglob("*.mp4")) == []
    assert list(out_dir.rglob("*.h5")) == []


def test_zero_copy_index_preserves_exact_readonly_source_uri(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    data_root = tmp_path / "data"
    ep_id, _ = _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        data_root,
        adapter_type="h5_raw_v3",
        passed=True,
        duration_s=2.0,
    )
    snap_before = h.snapshot_tree_mtimes(data_root)

    from robo_vlm_qc.store.repository import Repository

    repo = Repository(e2e_workspace)
    try:
        expected_uri = repo.get_episode_meta(ep_id).source_uri
    finally:
        repo.close()

    out_dir = tmp_path / "zc_uri"
    proc = h.run_cli(
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
            str(out_dir),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    tbl = pq.read_table(out_dir / "qc_index.parquet")
    df = tbl.to_pandas()
    assert expected_uri in set(df["source_uri"].astype(str))
    assert h.snapshot_tree_mtimes(data_root) == snap_before


def test_zero_copy_index_missing_or_corrupt_source_handled_cleanly(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    data_root = tmp_path / "data"
    ep_id, _ = _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        data_root,
        adapter_type="video_dir",
        passed=True,
    )
    assert ep_id
    # Remove source MP4 after ingest/approve
    for mp4 in data_root.rglob("*.mp4"):
        mp4.unlink()

    out_dir = tmp_path / "zc_missing_src"
    proc = h.run_cli(
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
            str(out_dir),
        ]
    )
    # Cleanly fails or records skipped/error without unhandled crash traceback
    assert "Traceback" not in proc.stderr
    assert (
        proc.returncode != 0
        or not (out_dir / "qc_index.parquet").exists()
        or pq.read_table(out_dir / "qc_index.parquet").num_rows == 0
    )


def test_h5_clean_zero_kept_episodes_handled_cleanly(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "h5_empty_ws"
    proc = h.run_cli(
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
            str(out_dir),
        ]
    )
    if proc.returncode == 0:
        assert list(out_dir.rglob("*.h5")) == []
    else:
        assert list(out_dir.rglob("*.h5")) == []


# ===========================================================================
# export CLI Boundaries
# ===========================================================================


def test_export_cli_rejects_invalid_recipe_or_format(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    p_fmt = h.run_cli(
        [
            "-w",
            str(e2e_workspace),
            "-c",
            str(e2e_config_dir),
            "export",
            "--format",
            "bogus_format",
            "--out",
            str(tmp_path / "o1"),
        ]
    )
    assert p_fmt.returncode != 0

    p_rec = h.run_cli(
        [
            "-w",
            str(e2e_workspace),
            "-c",
            str(e2e_config_dir),
            "export",
            "--recipe",
            "bogus_recipe",
            "--out",
            str(tmp_path / "o2"),
        ]
    )
    assert p_rec.returncode != 0


def test_export_cli_on_empty_workspace(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "empty_ws_exp"
    proc = h.run_cli(
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
            str(out_dir),
        ]
    )
    # Clean handling: either non-zero message or 0-episode export without crash
    assert "Traceback" not in proc.stderr


# ===========================================================================
# FastAPI Auth, Episodes Detail & Frame Streaming Boundaries
# ===========================================================================


def test_api_missing_or_invalid_token_returns_401(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    client = h.create_test_client(e2e_workspace, e2e_config_dir)
    assert client.get("/api/episodes").status_code == 401
    assert (
        client.get("/api/episodes", headers={"X-Auth-Token": "wrong-token"}).status_code
        == 401
    )


def test_api_reviewer_hitting_lab_returns_403(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    client = h.create_test_client(e2e_workspace, e2e_config_dir)
    rev_h = {"X-Auth-Token": "rev-token"}
    assert client.get("/api/lab/labels", headers=rev_h).status_code == 403
    assert client.get("/api/lab/diagnosis", headers=rev_h).status_code == 403


def test_api_reviewer_hitting_exports_returns_403(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    client = h.create_test_client(e2e_workspace, e2e_config_dir)
    rev_h = {"X-Auth-Token": "rev-token"}
    assert client.get("/api/exports", headers=rev_h).status_code == 403
    assert (
        client.post(
            "/api/exports/dry-run",
            headers=rev_h,
            json={"format": "lerobot_v3", "recipe": "bc_clean"},
        ).status_code
        == 403
    )


def test_api_nonexistent_episode_id_returns_404(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    client = h.create_test_client(e2e_workspace, e2e_config_dir)
    lead_h = {"X-Auth-Token": "lead-token"}
    assert client.get("/api/episodes/no_ds/no_ep", headers=lead_h).status_code == 404


def test_api_filter_combination_returning_empty_list(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        tmp_path / "data",
        adapter_type="video_dir",
        passed=True,
    )
    client = h.create_test_client(e2e_workspace, e2e_config_dir)
    resp = client.get(
        "/api/episodes?status=gt_frozen&dataset=no_such_dataset",
        headers={"X-Auth-Token": "lead-token"},
    )
    assert resp.status_code == 200
    body = resp.json()
    items = body if isinstance(body, list) else body.get("episodes", body.get("items", []))
    assert items == []


def test_api_frame_read_beyond_duration_clamped_or_400(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    ep_id, _ = _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        tmp_path / "data",
        adapter_type="video_dir",
        passed=True,
        duration_s=1.0,
    )
    client = h.create_test_client(e2e_workspace, e2e_config_dir)
    resp = client.get(
        f"/api/episodes/{ep_id}/frame?camera=main&t_s=9999.0",
        headers={"X-Auth-Token": "lead-token"},
    )
    assert resp.status_code in (200, 400)
    if resp.status_code == 200:
        assert resp.content[:2] == b"\xff\xd8"  # valid JPEG SOI marker


# ===========================================================================
# FastAPI Review Locks, 3-Layer Guard, gt_frozen & Watch Coverage
# ===========================================================================


def _review_body(base_run_id: str, action: str = "save_draft", coverage: float = 1.0) -> dict[str, Any]:
    return {
        "base_run_id": base_run_id,
        "action": action,
        "qc_segments": [],
        "success": {"passed": True, "verdict": "success", "confidence": 0.95, "criteria": []},
        "macro_stages": [
            {"stage_id": "S1", "t_start": 0.0, "t_end": 1.0, "label": "Stage One"}
        ],
        "watch_coverage": {
            "watched_intervals": [[0.0, 2.0]],
            "watched_s": 2.0 * coverage,
            "duration_s": 2.0,
            "coverage_ratio": coverage,
            "confirmed": True,
        },
        "notes": "",
    }


def test_api_reviewer_editing_episode_claimed_by_other_returns_409_or_403(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.store.repository import Repository

    ep_id, run_id = _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        tmp_path / "data",
        adapter_type="video_dir",
        status_action="vlm_ready",
    )
    repo = Repository(e2e_workspace)
    try:
        repo.claim_episode(ep_id, "alice", ttl_s=300.0)
    finally:
        repo.close()

    client = h.create_test_client(e2e_workspace, e2e_config_dir)
    resp = client.post(
        f"/api/episodes/{ep_id}/reviews",
        headers={"X-Auth-Token": "rev-token", "X-User": "bob"},
        json=_review_body(run_id, action="save_draft"),
    )
    assert resp.status_code in (403, 409)


def test_api_expired_lease_allows_reclaim(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.store.repository import Repository

    ep_id, _ = _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        tmp_path / "data",
        adapter_type="video_dir",
        status_action="vlm_ready",
    )
    repo = Repository(e2e_workspace)
    try:
        repo.claim_episode(ep_id, "alice", ttl_s=1.0, now=1.0)
    finally:
        repo.close()

    client = h.create_test_client(e2e_workspace, e2e_config_dir)
    resp = client.post(
        f"/api/episodes/{ep_id}/claim",
        headers={"X-Auth-Token": "rev-token", "X-User": "bob"},
        json={"user": "bob"},
    )
    assert resp.status_code == 200
    assert resp.json()["claimed_by"] != "alice"


def test_api_reviewer_calling_approve_or_freeze_gt_returns_403(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    ep_id, run_id = _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        tmp_path / "data",
        adapter_type="video_dir",
        status_action="vlm_ready",
    )
    client = h.create_test_client(e2e_workspace, e2e_config_dir)
    rev_h = {"X-Auth-Token": "rev-token", "X-User": "rev1"}

    for action in ("approve", "freeze_gt"):
        resp = client.post(
            f"/api/episodes/{ep_id}/reviews",
            headers=rev_h,
            json=_review_body(run_id, action=action),
        )
        assert resp.status_code == 403


def test_api_human_payload_modifying_arm_timelines_or_grounding_rejected(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    ep_id, run_id = _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        tmp_path / "data",
        adapter_type="video_dir",
        status_action="vlm_ready",
    )
    client = h.create_test_client(e2e_workspace, e2e_config_dir)
    lead_h = {"X-Auth-Token": "lead-token", "X-User": "lead1"}

    for bad_field in ("arm_timelines", "grounding", "raw_vlm_segments"):
        body = _review_body(run_id, action="save_draft")
        body[bad_field] = {}
        resp = client.post(
            f"/api/episodes/{ep_id}/reviews",
            headers=lead_h,
            json=body,
        )
        assert resp.status_code in (400, 422)


def test_api_gt_frozen_rejects_submit_review_and_save_draft(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    ep_id, run_id = _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        tmp_path / "data",
        adapter_type="video_dir",
        status_action="freeze_gt",
    )
    client = h.create_test_client(e2e_workspace, e2e_config_dir)
    lead_h = {"X-Auth-Token": "lead-token", "X-User": "lead1"}

    for action in ("save_draft", "submit_review"):
        resp = client.post(
            f"/api/episodes/{ep_id}/reviews",
            headers=lead_h,
            json=_review_body(run_id, action=action),
        )
        assert resp.status_code in (403, 409)


def test_api_watch_coverage_below_threshold_rejected_on_submit_review(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    ep_id, run_id = _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        tmp_path / "data",
        adapter_type="video_dir",
        status_action="vlm_ready",
    )
    client = h.create_test_client(e2e_workspace, e2e_config_dir)
    resp = client.post(
        f"/api/episodes/{ep_id}/reviews",
        headers={"X-Auth-Token": "rev-token", "X-User": "rev1"},
        json=_review_body(run_id, action="submit_review", coverage=0.10),
    )
    assert resp.status_code in (400, 422)


# ===========================================================================
# Jobs, Exports API & serve CLI Boundaries
# ===========================================================================


def test_api_batch_job_empty_episode_list_rejected_or_noop(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    client = h.create_test_client(e2e_workspace, e2e_config_dir)
    resp = client.post(
        "/api/jobs",
        headers={"X-Auth-Token": "lead-token"},
        json={"job_type": "batch_vlm", "episode_ids": []},
    )
    assert resp.status_code in (200, 201, 400, 422)
    if resp.status_code in (200, 201):
        assert resp.json().get("progress_total", 0) == 0


def test_api_invalid_job_type_returns_422_or_400(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    client = h.create_test_client(e2e_workspace, e2e_config_dir)
    resp = client.post(
        "/api/jobs",
        headers={"X-Auth-Token": "lead-token"},
        json={"job_type": "invalid_job_xyz", "episode_ids": ["ds/ep"]},
    )
    assert resp.status_code in (400, 422)


def test_api_cancel_queued_or_running_job(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    from robo_vlm_qc.store.repository import Repository

    repo = Repository(e2e_workspace)
    try:
        with repo.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, job_type, status, params_json,
                    progress_done, progress_total, summary_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("job_cancel_1", "batch_vlm", "queued", "{}", 0, 5, None, 1000.0, 1000.0),
            )
    finally:
        repo.close()

    client = h.create_test_client(e2e_workspace, e2e_config_dir)
    resp = client.post(
        "/api/jobs/job_cancel_1/cancel",
        headers={"X-Auth-Token": "lead-token"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"


def test_api_export_dry_run_zero_qualifying_episodes(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    client = h.create_test_client(e2e_workspace, e2e_config_dir)
    resp = client.post(
        "/api/exports/dry-run",
        headers={"X-Auth-Token": "lead-token"},
        json={"format": "lerobot_v3", "recipe": "bc_clean"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("episodes_included", body.get("episode_count", 0)) == 0


def test_api_download_nonexistent_export_id_returns_404(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    client = h.create_test_client(e2e_workspace, e2e_config_dir)
    resp = client.get(
        "/api/exports/nonexistent_exp_999/download",
        headers={"X-Auth-Token": "lead-token"},
    )
    assert resp.status_code == 404


def test_serve_cli_port_and_workspace_validation(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    p_help = h.run_cli(["serve", "--help"])
    assert p_help.returncode == 0
    assert "--port" in p_help.stdout

    p_bad = h.run_cli(
        [
            "-w",
            str(e2e_workspace),
            "-c",
            str(e2e_config_dir),
            "serve",
            "--port",
            "-1",
        ]
    )
    assert p_bad.returncode != 0


# ===========================================================================
# Dual-Role RBAC, SPA Routing, Revisions & DiffOverlay
# ===========================================================================


def test_api_reviewer_denied_lead_endpoints_while_lead_succeeds(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    client = h.create_test_client(e2e_workspace, e2e_config_dir)
    rev_h = {"X-Auth-Token": "rev-token"}
    lead_h = {"X-Auth-Token": "lead-token"}

    assert client.get("/api/lab/labels", headers=rev_h).status_code == 403
    assert client.get("/api/lab/labels", headers=lead_h).status_code == 200

    assert (
        client.post(
            "/api/exports/dry-run",
            headers=rev_h,
            json={"format": "zero_copy_index", "recipe": "bc_clean"},
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/exports/dry-run",
            headers=lead_h,
            json={"format": "zero_copy_index", "recipe": "bc_clean"},
        ).status_code
        == 200
    )


def test_api_unknown_api_route_404_vs_spa_client_route(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    client = h.create_test_client(e2e_workspace, e2e_config_dir)
    api_404 = client.get(
        "/api/nonexistent_route_xyz",
        headers={"X-Auth-Token": "lead-token"},
    )
    assert api_404.status_code == 404
    assert "application/json" in api_404.headers.get("content-type", "")

    spa_resp = client.get("/review/some_ds/some_ep")
    assert spa_resp.status_code == 200
    assert "text/html" in spa_resp.headers.get("content-type", "")


def test_review_revisions_preserve_base_run_id_across_edits(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.models.annotation import MacroStage
    from robo_vlm_qc.models.review import ReviewWritePayload, WatchCoverage
    from robo_vlm_qc.store.repository import Repository

    ep_id, run_id = _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        tmp_path / "data",
        adapter_type="video_dir",
        status_action="vlm_ready",
    )
    repo = Repository(e2e_workspace)
    try:
        run = repo.get_vlm_run(run_id)
        for i in range(3):
            payload = ReviewWritePayload(
                base_run_id=run_id,
                action="save_draft",
                qc_segments=[],
                success=run.success,
                macro_stages=[
                    MacroStage(
                        stage_id="S1",
                        t_start=0.0,
                        t_end=2.0,
                        label=f"Revision {i}",
                    )
                ],
                watch_coverage=WatchCoverage(
                    duration_s=2.0, watched_s=2.0, coverage_ratio=1.0, confirmed=True
                ),
                notes=f"rev_{i}",
            )
            repo.save_review(ep_id, "alice", "reviewer", payload)

        with sqlite3.connect(str(e2e_workspace / "robo_vlm_qc.db")) as conn:
            rows = conn.execute(
                "SELECT rev_id, base_run_id FROM reviews WHERE episode_id = ?",
                (ep_id,),
            ).fetchall()
        assert len(rows) == 3
        assert len({r[0] for r in rows}) == 3
        assert all(r[1] == run_id for r in rows)
        resolved = repo.resolve_episode(ep_id)
        assert resolved.notes == "rev_2"
    finally:
        repo.close()


def test_diff_overlay_payload_contains_raw_vlm_and_human_segments(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    ep_id, run_id = _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        tmp_path / "data",
        adapter_type="video_dir",
        status_action="vlm_ready",
        qc_segments=[
            {
                "seg_id": "vlm_0",
                "t_start": 0.0,
                "t_end": 0.4,
                "label_id": "boundary_idle",
                "training_role": "badcase",
                "splice_policy": "cut_clean",
                "mask_side": None,
                "affected_arm": "both",
                "vlm_raw_policy": "cut_clean",
            }
        ],
    )
    client = h.create_test_client(e2e_workspace, e2e_config_dir)
    lead_h = {"X-Auth-Token": "lead-token", "X-User": "lead1"}
    body = _review_body(run_id, action="submit_review", coverage=1.0)
    body["qc_segments"] = []  # human deletes FP segment
    assert client.post(f"/api/episodes/{ep_id}/reviews", headers=lead_h, json=body).status_code == 200

    detail = client.get(f"/api/episodes/{ep_id}", headers=lead_h)
    assert detail.status_code == 200
    data = detail.json()
    resolved = data.get("resolved", data)
    assert len(resolved["raw_vlm_segments"]) == 1
    assert resolved["qc_segments"] == []
    assert resolved["review_diff"]["fp_deleted"] == 1


def test_api_invalid_3layer_payload_rejected_with_422(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    ep_id, run_id = _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        tmp_path / "data",
        adapter_type="video_dir",
        status_action="vlm_ready",
    )
    client = h.create_test_client(e2e_workspace, e2e_config_dir)
    body = _review_body(run_id, action="save_draft")
    body["macro_stages"] = [
        {"stage_id": "S1", "t_start": 1.5, "t_end": 0.5, "label": "Inverted span"}
    ]
    resp = client.post(
        f"/api/episodes/{ep_id}/reviews",
        headers={"X-Auth-Token": "lead-token", "X-User": "lead1"},
        json=body,
    )
    assert resp.status_code == 422


# ===========================================================================
# VLM Lab Diagnosis & Few-Shot Boundaries
# ===========================================================================


def _invoke_diagnosis(ws: Path, episode_ids: list[str] | None = None) -> Any:
    mod = importlib.import_module("robo_vlm_qc.lab.diagnosis")
    for fn_name in ("diagnose_workspace", "run_diagnosis", "compute_diagnosis", "diagnose"):
        fn = getattr(mod, fn_name, None)
        if callable(fn):
            return _call_with_matching_kwargs(
                fn, workspace=ws, workspace_dir=ws, episode_ids=episode_ids
            )
    cls = getattr(mod, "ErrorDiagnoser", None) or getattr(mod, "DiagnosisRunner", None)
    if cls is not None:
        inst = _call_with_matching_kwargs(cls, workspace=ws, workspace_dir=ws)
        for mname in ("run", "diagnose", "compute"):
            m = getattr(inst, mname, None)
            if callable(m):
                return _call_with_matching_kwargs(m, episode_ids=episode_ids)
    raise AssertionError("No diagnosis entrypoint found in robo_vlm_qc.lab.diagnosis")


def test_diagnosis_zero_diffs_100pct_agreement(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    ep_id, _ = _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        tmp_path / "data",
        adapter_type="video_dir",
        passed=True,
        status_action="freeze_gt",
        qc_segments=[
            {
                "seg_id": "s0",
                "t_start": 0.2,
                "t_end": 0.8,
                "label_id": "missed_grasp_retry",
                "training_role": "recovery",
                "splice_policy": "mask_loss",
                "mask_side": "left",
                "affected_arm": "left",
                "vlm_raw_policy": "mask_loss",
            }
        ],
    )
    report = _invoke_diagnosis(e2e_workspace, [ep_id])
    dumped = report.model_dump() if hasattr(report, "model_dump") else dict(report)
    errors = dumped.get("errors", dumped.get("items", []))
    assert errors == []


def test_diagnosis_overlapping_multiple_gt_segments_picks_best_iou(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.models.annotation import QcSegment
    from robo_vlm_qc.models.review import ReviewWritePayload, WatchCoverage
    from robo_vlm_qc.store.repository import Repository

    vlm_seg = {
        "seg_id": "v0",
        "t_start": 1.0,
        "t_end": 2.0,
        "label_id": "missed_grasp_retry",
        "training_role": "recovery",
        "splice_policy": "mask_loss",
        "mask_side": "left",
        "affected_arm": "left",
        "vlm_raw_policy": "mask_loss",
    }
    ep_id, run_id = _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        tmp_path / "data",
        adapter_type="video_dir",
        passed=True,
        duration_s=3.0,
        status_action="vlm_ready",
        qc_segments=[vlm_seg],
    )
    repo = Repository(e2e_workspace)
    try:
        run = repo.get_vlm_run(run_id)
        # GT has two segments overlapping [1.0, 2.0]: weak overlap [0.6, 1.1] vs strong [1.02, 1.98]
        gt_segs = [
            QcSegment(
                seg_id="gt_weak",
                t_start=0.6,
                t_end=1.1,
                label_id="boundary_idle",
                training_role="badcase",
                splice_policy="cut_clean",
                mask_side=None,
                affected_arm="both",
                vlm_raw_policy="cut_clean",
            ),
            QcSegment(
                seg_id="gt_strong",
                t_start=1.02,
                t_end=1.98,
                label_id="missed_grasp_retry",
                training_role="recovery",
                splice_policy="mask_loss",
                mask_side="left",
                affected_arm="left",
                vlm_raw_policy="mask_loss",
            ),
        ]
        repo.save_review(
            ep_id,
            "lead_u",
            "lead",
            ReviewWritePayload(
                base_run_id=run_id,
                action="freeze_gt",
                qc_segments=gt_segs,
                success=run.success,
                macro_stages=run.macro_stages,
                watch_coverage=WatchCoverage(
                    duration_s=3.0, watched_s=3.0, coverage_ratio=1.0, confirmed=True
                ),
            ),
        )
    finally:
        repo.close()

    report = _invoke_diagnosis(e2e_workspace, [ep_id])
    dumped = report.model_dump() if hasattr(report, "model_dump") else dict(report)
    errors = dumped.get("errors", dumped.get("items", []))
    # Should pair v0 with gt_strong (same label & policy) and report gt_weak as fn_missed (not policy_flip)
    error_types = [e.get("error_type") for e in errors]
    assert "policy_flip" not in error_types
    assert "fn_missed" in error_types


def test_few_shot_invalid_t_end_le_t_start_rejected_or_clamped(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    ep_id, _ = _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        tmp_path / "data",
        adapter_type="video_dir",
    )
    client = h.create_test_client(e2e_workspace, e2e_config_dir)
    resp = client.post(
        "/api/lab/few-shots",
        headers={"X-Auth-Token": "lead-token"},
        json={
            "scope": "global",
            "error_type": "fp_overflag",
            "episode_id": ep_id,
            "t_start": 1.5,
            "t_end": 1.0,
            "vlm_wrong_json": {},
            "human_right_json": {},
            "rationale": "invalid span",
        },
    )
    assert resp.status_code in (400, 422)


def test_few_shot_top_k_zero_or_tight_token_cap_returns_empty_or_truncated(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.store.repository import Repository

    ep_id, _ = _setup_episode_in_repo(
        e2e_workspace, e2e_config_dir, tmp_path / "data", adapter_type="video_dir"
    )
    repo = Repository(e2e_workspace)
    try:
        with repo.db.transaction() as conn:
            for i in range(3):
                conn.execute(
                    """
                    INSERT INTO few_shot_examples (
                        example_id, scope, task_id, label_id, error_type, episode_id,
                        t_start, t_end, tile_image_path, vlm_wrong_json, human_right_json,
                        rationale, enabled, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"fs_cap_{i}",
                        "global",
                        None,
                        None,
                        "fp_overflag",
                        ep_id,
                        0.1,
                        0.5,
                        None,
                        "{}",
                        "{}",
                        "R" * 400,
                        1,
                        float(100 + i),
                    ),
                )
    finally:
        repo.close()

    fs_mod = importlib.import_module("robo_vlm_qc.lab.few_shot")
    retriever = (
        getattr(fs_mod, "retrieve_few_shots", None)
        or getattr(fs_mod, "select_few_shots", None)
        or getattr(fs_mod, "FewShotBook", None)
    )
    assert retriever is not None
    if inspect.isclass(retriever):
        inst = _call_with_matching_kwargs(retriever, workspace=e2e_workspace)
        fn = getattr(inst, "retrieve", None) or getattr(inst, "select")
    else:
        fn = retriever

    res_k0 = _call_with_matching_kwargs(
        fn,
        workspace=e2e_workspace,
        task_id="_default",
        top_k=0,
        k=0,
        max_tokens=4000,
    )
    assert res_k0 == []

    res_tight = _call_with_matching_kwargs(
        fn,
        workspace=e2e_workspace,
        task_id="_default",
        top_k=5,
        k=5,
        max_tokens=1,
    )
    assert len(res_tight) < 3


def test_few_shot_disabled_excluded_from_retrieval(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.store.repository import Repository

    ep_id, _ = _setup_episode_in_repo(
        e2e_workspace, e2e_config_dir, tmp_path / "data", adapter_type="video_dir"
    )
    repo = Repository(e2e_workspace)
    try:
        with repo.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO few_shot_examples (
                    example_id, scope, task_id, label_id, error_type, episode_id,
                    t_start, t_end, tile_image_path, vlm_wrong_json, human_right_json,
                    rationale, enabled, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("fs_en", "global", None, None, "fp_overflag", ep_id, 0.1, 0.4, None, "{}", "{}", "enabled", 1, 1.0),
            )
            conn.execute(
                """
                INSERT INTO few_shot_examples (
                    example_id, scope, task_id, label_id, error_type, episode_id,
                    t_start, t_end, tile_image_path, vlm_wrong_json, human_right_json,
                    rationale, enabled, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("fs_dis", "global", None, None, "fp_overflag", ep_id, 0.5, 0.8, None, "{}", "{}", "disabled", 0, 2.0),
            )
    finally:
        repo.close()

    fs_mod = importlib.import_module("robo_vlm_qc.lab.few_shot")
    retriever = (
        getattr(fs_mod, "retrieve_few_shots", None)
        or getattr(fs_mod, "select_few_shots", None)
        or getattr(fs_mod, "FewShotBook", None)
    )
    if inspect.isclass(retriever):
        inst = _call_with_matching_kwargs(retriever, workspace=e2e_workspace)
        fn = getattr(inst, "retrieve", None) or getattr(inst, "select")
    else:
        fn = retriever

    items = _call_with_matching_kwargs(
        fn, workspace=e2e_workspace, task_id="_default", top_k=10, k=10, max_tokens=10000
    )
    ids = {it.example_id if hasattr(it, "example_id") else it["example_id"] for it in items}
    assert "fs_en" in ids
    assert "fs_dis" not in ids


def test_few_shot_on_single_cam_video_dir_builds_tile(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    ep_id, _ = _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        tmp_path / "data",
        adapter_type="video_dir",
        duration_s=2.0,
    )
    client = h.create_test_client(e2e_workspace, e2e_config_dir)
    resp = client.post(
        "/api/lab/few-shots",
        headers={"X-Auth-Token": "lead-token"},
        json={
            "scope": "global",
            "error_type": "fp_overflag",
            "episode_id": ep_id,
            "t_start": 0.2,
            "t_end": 0.8,
            "vlm_wrong_json": {"label_id": "mid_idle_freeze"},
            "human_right_json": {"splice_policy": "keep_positive"},
            "rationale": "Single cam tile creation test",
            "capture_tile": True,
        },
    )
    assert resp.status_code in (200, 201)
    tile_rel = resp.json().get("tile_image_path")
    assert tile_rel, f"Expected tile_image_path when capture_tile=True: {resp.json()}"
    tile_path = e2e_workspace / tile_rel
    assert tile_path.is_file() and tile_path.stat().st_size > 100


# ===========================================================================
# Eval Sandbox, Unknown Triage, Prompt Assist & eval CLI
# ===========================================================================


def test_eval_runner_zero_gt_frozen_episodes_handled_cleanly(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any
) -> None:
    proc = h.run_cli(
        ["-w", str(e2e_workspace), "-c", str(e2e_config_dir), "eval"]
    )
    # Clean error or 0-episode report without unhandled traceback
    assert "Traceback" not in proc.stderr


def test_eval_runner_never_alters_active_run_id_uses_eval_sandbox(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any, tmp_path: Path
) -> None:
    from robo_vlm_qc.store.repository import Repository

    ep_id, orig_run_id = _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        tmp_path / "data",
        adapter_type="video_dir",
        passed=True,
        status_action="freeze_gt",
    )
    proc = h.run_cli(
        ["-w", str(e2e_workspace), "-c", str(e2e_config_dir), "eval"]
    )
    assert proc.returncode == 0, proc.stderr

    repo = Repository(e2e_workspace)
    try:
        row = repo.get_episode_row(ep_id)
        assert row["active_run_id"] == orig_run_id
        assert row["status"] == "gt_frozen"
        with sqlite3.connect(str(e2e_workspace / "robo_vlm_qc.db")) as conn:
            sandbox_rows = conn.execute(
                "SELECT run_id FROM vlm_runs WHERE episode_id = ? AND is_eval_sandbox = 1",
                (ep_id,),
            ).fetchall()
        assert len(sandbox_rows) >= 1
    finally:
        repo.close()


def test_promote_invalid_or_nonexistent_version_rejected(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    client = h.create_test_client(e2e_workspace, e2e_config_dir)
    resp = client.post(
        "/api/lab/promote",
        headers={"X-Auth-Token": "lead-token"},
        json={"eval_id": "nonexistent_eval_999"},
    )
    assert resp.status_code in (400, 404)


def test_unknown_triage_empty_pool(
    e2e_workspace: Path, e2e_config_dir: Path
) -> None:
    client = h.create_test_client(e2e_workspace, e2e_config_dir)
    resp = client.get("/api/lab/unknowns", headers={"X-Auth-Token": "lead-token"})
    assert resp.status_code == 200
    body = resp.json()
    items = body if isinstance(body, list) else body.get("unknowns", body.get("items", []))
    assert items == []


def test_unknown_triage_duplicate_label_id_rejected(
    e2e_workspace: Path, e2e_config_dir: Path, tmp_path: Path
) -> None:
    ep_id, _ = _setup_episode_in_repo(
        e2e_workspace,
        e2e_config_dir,
        tmp_path / "data",
        adapter_type="video_dir",
        status_action="vlm_ready",
        qc_segments=[
            {
                "seg_id": "s_unk_dup",
                "t_start": 0.5,
                "t_end": 1.2,
                "label_id": "unknown_anomaly",
                "training_role": "badcase",
                "splice_policy": "mask_loss",
                "mask_side": "left",
                "affected_arm": "left",
                "vlm_raw_policy": "mask_loss",
                "is_unknown": True,
                "unknown_note": "Unclassified anomaly",
            }
        ],
    )
    client = h.create_test_client(e2e_workspace, e2e_config_dir)
    lead_h = {"X-Auth-Token": "lead-token"}

    r_triage = client.post(
        "/api/lab/unknowns/triage",
        headers=lead_h,
        json={
            "action": "promote",
            "episode_id": ep_id,
            "seg_id": "s_unk_dup",
            "new_label": {
                "id": "boundary_idle",
                "name": "Duplicate Label",
                "category": "idle",
                "default_training_role": "badcase",
                "default_splice_policy": "cut_clean",
                "collector_ui": {"title": "dup", "hint": ""},
                "vlm_semantics": {"definition": "dup"},
            },
        },
    )
    assert r_triage.status_code in (400, 409)

    labels_doc = client.get("/api/lab/labels", headers=lead_h).json()
    dup_doc = dict(labels_doc, labels=list(labels_doc["labels"]) + [labels_doc["labels"][0]])
    r_put_dup = client.put("/api/lab/labels", headers=lead_h, json=dup_doc)
    assert r_put_dup.status_code == 422


def test_prompt_assist_gracefully_handles_vlm_error(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any
) -> None:
    mock_vlm_server.fail_next(status_code=500, count=20)
    mod = importlib.import_module("robo_vlm_qc.lab.prompt_assist")
    fn = (
        getattr(mod, "suggest_prompt_updates", None)
        or getattr(mod, "suggest_updates", None)
        or getattr(mod, "generate_suggestions", None)
        or getattr(mod, "assist_prompt", None)
    )
    assert callable(fn)
    try:
        res = _call_with_matching_kwargs(
            fn,
            workspace=e2e_workspace,
            config_dir=e2e_config_dir,
            task_id="_default",
            errors=[],
        )
        # If it returns a structured fallback instead of raising, verify it is non-crashing
        assert res is not None
    except (httpx.HTTPError, RuntimeError, ValueError, OSError) as exc:
        # Must raise a clean error (never corrupting workspace configs)
        assert exc is not None
    assert len(mock_vlm_server.requests) >= 1
    assert not (e2e_workspace / "configs_override" / "prompts" / "s3_discover.yaml").exists()


def test_eval_cli_missing_workspace_or_gt_error(
    tmp_path: Path, e2e_config_dir: Path
) -> None:
    missing_ws = tmp_path / "nonexistent_ws"
    proc = h.run_cli(
        ["-w", str(missing_ws), "-c", str(e2e_config_dir), "eval"]
    )
    assert proc.returncode != 0
