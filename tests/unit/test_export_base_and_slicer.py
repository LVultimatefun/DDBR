"""Unit and integration tests for export/base.py and export/slicer.py."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robo_vlm_qc.config import ConfigManager
from robo_vlm_qc.export import (
    CANONICAL_81D_NAMES,
    NOT_SMOOTHED,
    OutputExporter,
    PieceSlicer,
    build_global_subtasks,
    extract_canonical_81d_state,
    extract_piece_kinematics_81d,
    get_exporter,
    list_exporters,
    register_exporter,
    smooth_tail_81d,
    smooth_tail_vector,
    smoothable,
    staged_export_dir,
)
from robo_vlm_qc.io.base import EpisodeView, KinematicsData
from robo_vlm_qc.models.annotation import (
    ConfigFingerprintSnapshot,
    MacroStage,
    QcSegment,
    SuccessVerdict,
    VlmRunArtifact,
)
from robo_vlm_qc.models.episode import CameraSpec, EpisodeMeta, KinematicsSpec
from robo_vlm_qc.models.export import ExportManifest, ExportRecipe
from robo_vlm_qc.models.review import EpisodeStatus, ResolvedEpisode


class _DummyMedia:
    def __init__(self, fps: float = 30.0, duration_s: float = 4.0) -> None:
        self.cameras = [
            CameraSpec(name="main", role="main", width=64, height=48, fps=fps)
        ]
        self.fps = fps
        self.duration_s = duration_s

    def read_frame_rgb(self, camera: str, t_s: float) -> np.ndarray:
        return np.zeros((48, 64, 3), dtype=np.uint8)

    def read_frames_batch(self, camera: str, times_s: list[float]) -> list[np.ndarray]:
        return [np.zeros((48, 64, 3), dtype=np.uint8) for _ in times_s]


def _make_resolved_and_view(
    *,
    T: int = 120,
    fps: float = 30.0,
    passed: bool | None = True,
    qc_segments: list[QcSegment] | None = None,
    macro_stages: list[MacroStage] | None = None,
    with_kinematics: bool = True,
    with_raw_action: bool = False,
) -> tuple[ResolvedEpisode, EpisodeView]:
    dur = T / fps
    meta = EpisodeMeta(
        episode_id="ds/ep_001",
        dataset="ds",
        stem="ep_001",
        task_id="pick_and_place",
        adapter_type="h5_raw_v3",
        source_uri="/tmp/fake/ep_001.h5",
        source_mtime=1.0,
        episode_fingerprint="fp_001",
        duration_s=dur,
        fps=fps,
        frame_count=T,
        cameras=[CameraSpec(name="main", role="main", width=64, height=48, fps=fps)],
        camera_roles={"main": "main"},
        kinematics=KinematicsSpec(
            has_kinematics=with_kinematics,
            n_arms=2 if with_kinematics else 0,
            state_dim=81 if with_kinematics else 0,
            has_tactile=with_kinematics,
            tactile_dim=30 if with_kinematics else 0,
            has_action_cmd=with_raw_action,
        ),
    )
    if macro_stages is None:
        macro_stages = [
            MacroStage(stage_id="S1", t_start=0.0, t_end=dur * 0.5, label="Pick object", label_en="pick_object"),
            MacroStage(stage_id="S2", t_start=dur * 0.5, t_end=dur, label="Place object", label_en="place_object"),
        ]
    resolved = ResolvedEpisode(
        episode_id="ds/ep_001",
        meta=meta,
        status=EpisodeStatus.APPROVED,
        active_run_id="run_1",
        qc_segments=qc_segments or [],
        success=SuccessVerdict(
            passed=passed,
            verdict="success" if passed is True else ("failure" if passed is False else "unsure"),
            confidence=0.95,
        ),
        macro_stages=macro_stages,
        grounding=VlmRunArtifact(
            run_id="run_1",
            episode_id="ds/ep_001",
            created_at=1.0,
            config_fingerprint=ConfigFingerprintSnapshot(
                model_routing={},
                prompt_bundle_version="1",
                label_taxonomy_version="1",
                task_spec_version="1",
            ),
        ).grounding,
        arm_timelines=VlmRunArtifact(
            run_id="run_1",
            episode_id="ds/ep_001",
            created_at=1.0,
            config_fingerprint=ConfigFingerprintSnapshot(
                model_routing={},
                prompt_bundle_version="1",
                label_taxonomy_version="1",
                task_spec_version="1",
            ),
        ).arm_timelines,
        raw_vlm_segments=[],
    )

    kin: KinematicsData | None = None
    if with_kinematics:
        ts = np.arange(T, dtype=np.float64) / fps
        raw_state = np.zeros((T, 81), dtype=np.float32)
        t_norm = np.linspace(0.0, 1.0, T, dtype=np.float32)
        raw_state[:, 23] = t_norm * 0.5
        raw_state[:, 30] = -t_norm * 0.4
        raw_state[:, 2] = np.linspace(-3.0, 3.0, T, dtype=np.float32)
        angles = np.linspace(0.0, 1.2, T)
        q_l = Rotation.from_euler("z", angles).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)
        q_r = Rotation.from_euler("x", angles * 0.8).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)
        raw_state[:, 26:30] = q_l
        raw_state[:, 33:37] = q_r
        raw_state[:, 37:81] = t_norm[:, None] * 0.25

        eef_pos = np.stack([raw_state[:, 23:26], raw_state[:, 30:33]], axis=1).astype(np.float64)
        eef_rv = np.stack(
            [
                Rotation.from_quat(q_l[:, [1, 2, 3, 0]]).as_rotvec(),
                Rotation.from_quat(q_r[:, [1, 2, 3, 0]]).as_rotvec(),
            ],
            axis=1,
        )
        tactile = np.arange(T * 30, dtype=np.float32).reshape(T, 30) * 0.01
        raw_action = (raw_state + 0.01) if with_raw_action else None
        kin = KinematicsData(
            timestamps_s=ts,
            eef_pos=eef_pos,
            eef_rotvec=eef_rv,
            hand_q=raw_state[:, 37:81].astype(np.float64),
            torso_q=raw_state[:, 3:7].astype(np.float64),
            chassis_pose=raw_state[:, 0:3].astype(np.float64),
            raw_state=raw_state,
            raw_action=raw_action,
            tactile=tactile,
            valid_mask=np.ones(T, dtype=bool),
        )

    view = EpisodeView(meta=meta, media=_DummyMedia(fps=fps, duration_s=dur), kinematics=kin)
    return resolved, view


def test_exporter_registry_and_staged_export_atomicity(tmp_path: Path) -> None:
    exporters = list_exporters()
    assert {"lerobot_v3", "h5_clean", "zero_copy_index", "viz_mp4"}.issubset(set(exporters))
    for name in ("lerobot_v3", "h5_clean", "zero_copy_index", "viz_mp4"):
        inst = get_exporter(name)
        assert isinstance(inst, OutputExporter)
        assert inst.name == name

    class _CustomExporter(OutputExporter):
        name = "custom_test_exporter"

        def export(self, pieces, resolved_list, views, out_dir, recipe) -> ExportManifest:
            raise NotImplementedError

    register_exporter(_CustomExporter, override=True)
    assert "custom_test_exporter" in list_exporters()
    with pytest.raises(ValueError):
        register_exporter(_CustomExporter, override=False)

    out_dir = tmp_path / "exp_out"
    out_dir.mkdir()
    (out_dir / "old_file.txt").write_text("stale", encoding="utf-8")
    stale_staging = tmp_path / "exp_out._staging"
    stale_staging.mkdir()
    (stale_staging / "leftover.txt").write_text("old_staging", encoding="utf-8")

    with pytest.raises(RuntimeError, match="boom"):
        with staged_export_dir(out_dir) as st:
            assert not (st / "leftover.txt").exists()
            (st / "partial.txt").write_text("partial", encoding="utf-8")
            raise RuntimeError("boom")

    assert (out_dir / "old_file.txt").is_file()
    assert not (out_dir / "partial.txt").exists()
    assert [p for p in tmp_path.iterdir() if "_staging" in p.name or p.name.endswith(".lock")] == []

    with staged_export_dir(out_dir) as st:
        (st / "new_file.txt").write_text("fresh", encoding="utf-8")
    assert not (out_dir / "old_file.txt").exists()
    assert (out_dir / "new_file.txt").read_text(encoding="utf-8") == "fresh"
    assert [p for p in tmp_path.iterdir() if "_staging" in p.name or p.name.endswith(".lock")] == []


def test_piece_slicer_constructors_and_recipes() -> None:
    cfg_mgr = ConfigManager()
    for slicer in (
        PieceSlicer(),
        PieceSlicer(cfg_mgr),
        PieceSlicer(config=cfg_mgr),
        PieceSlicer(config_manager=cfg_mgr),
        PieceSlicer(settings={"export": {"min_piece_s": 0.1}}),
    ):
        assert isinstance(slicer, PieceSlicer)

    segs = [
        QcSegment(
            seg_id="s_cut",
            t_start=1.2,
            t_end=2.2,
            label_id="mid_idle_freeze",
            training_role="badcase",
            splice_policy="cut_clean",
            mask_side=None,
            vlm_raw_policy="cut_clean",
        ),
        QcSegment(
            seg_id="s_rec",
            t_start=0.2,
            t_end=0.8,
            label_id="missed_grasp_retry",
            training_role="recovery",
            splice_policy="mask_loss",
            mask_side="left",
            vlm_raw_policy="mask_loss",
        ),
        QcSegment(
            seg_id="s_bad",
            t_start=2.6,
            t_end=3.2,
            label_id="collision_or_knock",
            training_role="badcase",
            splice_policy="mask_loss",
            mask_side="right",
            vlm_raw_policy="mask_loss",
        ),
    ]
    res_pass, view = _make_resolved_and_view(T=120, qc_segments=segs, passed=True)
    res_fail, _ = _make_resolved_and_view(T=120, qc_segments=segs, passed=False)
    slicer = PieceSlicer(cfg_mgr)

    bc_recipe = ExportRecipe(name="bc_clean", format="lerobot_v3", tail_frames=5, smooth_tail="on_jump")
    bc_pieces = slicer.slice(res_pass, view, bc_recipe)
    assert len(bc_pieces) == 2
    p0, p1 = bc_pieces
    assert p0.tail_rows == 5
    assert p0.smooth_tail_applied is True
    assert not any(p0.is_tail[: p0.own_rows])
    assert all(p0.is_tail[p0.own_rows :])
    assert p1.tail_rows == 0
    assert p0.action_loss_mask[10] == [0.0, 1.0]
    assert p1.action_loss_mask[18] == [1.0, 0.0]
    assert slicer.slice(res_fail, view, bc_recipe) == []

    rp_recipe = ExportRecipe(name="recovery_plus", format="lerobot_v3", tail_frames=5)
    rp_pieces = slicer.slice(res_pass, view, rp_recipe)
    assert rp_pieces[0].action_loss_mask[10] == [1.0, 1.0]
    assert rp_pieces[1].action_loss_mask[18] == [1.0, 0.0]

    bp_recipe = ExportRecipe(name="badcase_pairs", format="zero_copy_index", context_window_s=0.1)
    bp_pieces = slicer.slice(res_fail, view, bp_recipe)
    assert len(bp_pieces) == 2
    assert all(p.tail_rows == 0 for p in bp_pieces)


def test_tail_smoothing_so3_geodesic_wrapped_yaw_and_not_smoothed() -> None:
    assert len(CANONICAL_81D_NAMES) == 81
    assert NOT_SMOOTHED == ("tau", "torque", "force", "wrench", "tactile", "age_ms")
    for k in ("observation.tactile_force", "hand_tau", "wrench_ext", "age_ms"):
        assert smoothable(k) is False
    assert smoothable("observation.state") is True

    arr = np.zeros((8, 81), dtype=np.float32)
    arr[:4, 23] = 0.0
    arr[4:, 23] = 1.0
    arr[:4, 2] = 3.1
    arr[4:, 2] = -3.1
    arr[:4, 26:30] = [1.0, 0.0, 0.0, 0.0]
    q90 = Rotation.from_euler("z", math.pi / 2).as_quat()[[3, 0, 1, 2]]
    arr[4:, 26:30] = q90.astype(np.float32)
    arr[5, 37] = np.nan

    smoothed = smooth_tail_81d(arr, own_rows=4)
    assert np.isclose(smoothed[-1, 23], 1.0, atol=1e-5)
    assert 0.0 < smoothed[4, 23] < 1.0
    assert abs(float(smoothed[4, 2]) - 3.1) < 0.2 or abs(float(smoothed[4, 2]) + 3.1) < 0.2
    q_norms = np.linalg.norm(smoothed[:, 26:30], axis=1)
    assert np.allclose(q_norms, 1.0, atol=1e-5)

    tac = np.ones((8, 30), dtype=np.float32)
    tac[4:] = 9.0
    assert np.array_equal(smooth_tail_vector("tactile_force", tac, 4), tac)

    res, view = _make_resolved_and_view(T=90, qc_segments=[], with_raw_action=False)
    piece = PieceSlicer().slice(
        res, view, ExportRecipe(name="bc_clean", format="lerobot_v3", smooth_tail="always")
    )[0]
    st81, act81, tac30 = extract_piece_kinematics_81d(piece, view)
    assert st81.shape == (90, 81)
    assert act81.shape == (90, 81)
    assert tac30 is not None and tac30.shape == (90, 30)
    assert np.allclose(act81[-1], st81[-1], atol=1e-6)
    assert np.allclose(act81[:-1], st81[1:], atol=1e-6)

    g_subs, remap = build_global_subtasks([piece])
    assert len(g_subs) == 2
    assert len(remap[piece.piece_id]) == piece.total_rows
    assert set(remap[piece.piece_id]) == {0, 1}
    assert extract_canonical_81d_state(None, 5).shape == (5, 81)
