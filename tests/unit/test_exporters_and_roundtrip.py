"""Self-contained integration tests for all 4 exporters x 3 recipes, robot_type, and LeRobot round-trip."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from robo_vlm_qc.config import ConfigManager
from robo_vlm_qc.export import CANONICAL_81D_NAMES, PieceSlicer, get_exporter
from robo_vlm_qc.io import LeRobotAdapter, auto_detect, get_adapter
from robo_vlm_qc.models.annotation import (
    ConfigFingerprintSnapshot,
    MacroStage,
    QcSegment,
    SuccessVerdict,
    VlmRunArtifact,
)
from robo_vlm_qc.models.export import ExportRecipe
from robo_vlm_qc.models.review import ReviewWritePayload, WatchCoverage
from robo_vlm_qc.store import Repository


def _encode_jpeg(rgb: np.ndarray) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 60])
    assert ok
    return np.frombuffer(buf.tobytes(), dtype=np.uint8)


def _make_synthetic_h5_raw_v3(ds_dir: Path, duration_s: float = 2.0) -> Path:
    ds_dir.mkdir(parents=True, exist_ok=True)
    h5_path = ds_dir / "ep_000000.h5"
    T = max(10, int(round(duration_s * 30.0)))
    t0 = 1700000000.0
    ts = t0 + np.arange(T, dtype=np.float64) / 30.0
    vlen_u8 = h5py.vlen_dtype(np.dtype("uint8"))

    robot_state = np.zeros((T, 37), dtype=np.float32)
    t_norm = np.linspace(0.0, 1.0, T, dtype=np.float32)
    robot_state[:, 23] = 0.2 + 0.3 * t_norm
    robot_state[:, 30] = -0.2 - 0.3 * t_norm
    q = Rotation.from_euler("z", t_norm * 0.8).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)
    robot_state[:, 26:30] = q
    robot_state[:, 33:37] = q

    hand_q = (t_norm[:, None] * 0.2).repeat(22, axis=1).astype(np.float32)
    tac = np.ones((T, 5, 3), dtype=np.float32) * 0.5
    tac_valid = np.ones((T,), dtype=bool)

    with h5py.File(str(h5_path), "w") as f:
        f.attrs["schema"] = "raw_only_v3"
        f.attrs["task_name"] = "pick_and_place"
        cam_grp = f.create_group("camera")
        for cname, (h, w) in [("head_rgbd", (96, 128)), ("torso_rgbd", (96, 128)), ("head_stereo", (96, 256))]:
            cg = cam_grp.create_group(cname)
            cg.attrs["format"] = "jpeg_sbs_or_native" if cname == "head_stereo" else "jpeg"
            ds_j = cg.create_dataset("jpeg", (T,), dtype=vlen_u8)
            for i in range(T):
                img = np.full((h, w, 3), (30 + i * 2) % 240, dtype=np.uint8)
                ds_j[i] = _encode_jpeg(img)
            cg.create_dataset("timestamps", data=ts, dtype=np.float64)

        hg = f.create_group("hand")
        hg.create_dataset("hand_a_radian", data=hand_q, dtype=np.float32)
        hg.create_dataset("hand_b_radian", data=hand_q, dtype=np.float32)
        hg.create_dataset("hand_a_tactile_force", data=tac, dtype=np.float32)
        hg.create_dataset("hand_b_tactile_force", data=tac, dtype=np.float32)
        hg.create_dataset("hand_a_tactile_valid", data=tac_valid, dtype=bool)
        hg.create_dataset("hand_b_tactile_valid", data=tac_valid, dtype=bool)
        hg.create_dataset("timestamps", data=ts, dtype=np.float64)

        rg = f.create_group("robot")
        rg.create_dataset("state", data=robot_state, dtype=np.float32)
        rg.create_dataset("timestamps", data=ts, dtype=np.float64)

    return h5_path


def _make_mp4(path: Path, n_frames: int = 60, w: int = 128, h: int = 96) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    assert writer.isOpened()
    try:
        for i in range(n_frames):
            frame = np.full((h, w, 3), (20 + i * 3) % 240, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _make_synthetic_video_dir(ds_dir: Path, duration_s: float = 2.0) -> Path:
    n_frames = max(10, int(round(duration_s * 30.0)))
    _make_mp4(ds_dir / "ep_000.mp4", n_frames=n_frames)
    return ds_dir


def _make_synthetic_lerobot_v3(ds_dir: Path, duration_s: float = 2.0) -> Path:
    n_frames = max(10, int(round(duration_s * 30.0)))
    (ds_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (ds_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    cam_keys = (
        "observation.images.main",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    )
    for ck in cam_keys:
        _make_mp4(ds_dir / "videos" / ck / "chunk-000" / "file-000.mp4", n_frames=n_frames)
    states = np.zeros((n_frames, 81), dtype=np.float32)
    states[:, 26] = 1.0
    states[:, 33] = 1.0
    states[:, 23] = np.linspace(0.0, 0.3, n_frames, dtype=np.float32)
    actions = np.vstack([states[1:], states[-1:]])
    pq.write_table(
        pa.table(
            {
                "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32(), 81)),
                "action": pa.array(actions.tolist(), type=pa.list_(pa.float32(), 81)),
                "timestamp": pa.array([i / 30.0 for i in range(n_frames)], type=pa.float32()),
                "frame_index": pa.array(list(range(n_frames)), type=pa.int64()),
                "episode_index": pa.array([0] * n_frames, type=pa.int64()),
                "index": pa.array(list(range(n_frames)), type=pa.int64()),
                "task_index": pa.array([0] * n_frames, type=pa.int64()),
            }
        ),
        ds_dir / "data" / "chunk-000" / "file-000.parquet",
    )
    pq.write_table(
        pa.table({"task_index": pa.array([0], type=pa.int64()), "task": pa.array(["bimanual_handover"])}),
        ds_dir / "meta" / "tasks.parquet",
    )
    ep_row: dict[str, Any] = {
        "episode_index": 0,
        "length": n_frames,
        "tasks": ["bimanual_handover"],
        "data/chunk_index": 0,
        "data/file_index": 0,
        "dataset_from_index": 0,
        "dataset_to_index": n_frames,
    }
    for ck in cam_keys:
        ep_row[f"videos/{ck}/chunk_index"] = 0
        ep_row[f"videos/{ck}/file_index"] = 0
        ep_row[f"videos/{ck}/from_timestamp"] = 0.0
        ep_row[f"videos/{ck}/to_timestamp"] = float((n_frames - 1) / 30.0)
    pq.write_table(
        pa.Table.from_pylist([ep_row]),
        ds_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    features: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": [81], "names": list(CANONICAL_81D_NAMES)},
        "action": {"dtype": "float32", "shape": [81], "names": list(CANONICAL_81D_NAMES)},
    }
    for ck in cam_keys:
        features[ck] = {"dtype": "video", "shape": [96, 128, 3]}
    info = {
        "codebase_version": "v3.0",
        "robot_type": "bimanual_robot",
        "total_episodes": 1,
        "total_frames": n_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": 30,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    (ds_dir / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    return ds_dir


def _ffprobe_info(mp4: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    assert ffprobe
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,codec_name,pix_fmt",
            "-of",
            "json",
            str(mp4),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    st = json.loads(proc.stdout)["streams"][0]
    return {
        "width": int(st["width"]),
        "height": int(st["height"]),
        "nb_frames": int(st["nb_frames"]),
        "codec_name": st["codec_name"],
        "pix_fmt": st["pix_fmt"],
    }


def _ingest_and_approve(
    ws: Path,
    cfg_mgr: ConfigManager,
    data_path: Path,
    *,
    passed: bool = True,
    qc_segments: list[QcSegment] | None = None,
) -> list[str]:
    repo = Repository(ws)
    ep_ids: list[str] = []
    try:
        for handle in auto_detect(data_path, task_patterns=cfg_mgr.task_patterns()):
            ad = get_adapter(handle.adapter_type)
            with ad.open(handle) as view:
                meta = view.meta
                repo.ingest_episode(meta)
                ep_ids.append(meta.episode_id)
                dur = meta.duration_s
                art = VlmRunArtifact(
                    run_id=f"run_{meta.stem}",
                    episode_id=meta.episode_id,
                    created_at=1.0,
                    config_fingerprint=ConfigFingerprintSnapshot(
                        model_routing={},
                        prompt_bundle_version="1",
                        label_taxonomy_version="1",
                        task_spec_version="1",
                    ),
                    macro_stages=[
                        MacroStage(stage_id="S1", t_start=0.0, t_end=dur * 0.5, label="Grasp object", label_en="grasp"),
                        MacroStage(stage_id="S2", t_start=dur * 0.5, t_end=dur, label="Place object", label_en="place"),
                    ],
                    qc_segments=qc_segments or [],
                    success=SuccessVerdict(
                        passed=passed,
                        verdict="success" if passed else "failure",
                        confidence=0.95,
                    ),
                )
                repo.save_vlm_run_artifact(art)
                repo.save_review(
                    meta.episode_id,
                    "lead_test",
                    "lead",
                    ReviewWritePayload(
                        base_run_id=art.run_id,
                        action="approve",
                        qc_segments=art.qc_segments,
                        success=art.success,
                        macro_stages=art.macro_stages,
                        watch_coverage=WatchCoverage(
                            watched_intervals=[(0.0, dur)],
                            watched_s=dur,
                            duration_s=dur,
                            coverage_ratio=1.0,
                            confirmed=True,
                        ),
                    ),
                )
    finally:
        repo.close()
    return ep_ids


def _open_view_via_discover(meta: Any, cfg_mgr: ConfigManager) -> Any:
    from robo_vlm_qc.cli import _open_view_from_meta

    return _open_view_from_meta(meta, cfg_mgr)


def test_lerobot_v3_export_roundtrip_robot_type_and_chunk_splitting(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    data_dir = tmp_path / "h5_src"
    _make_synthetic_h5_raw_v3(data_dir, duration_s=3.0)
    cfg_mgr = ConfigManager(workspace_dir=ws)
    cfg_mgr.settings["export"]["lerobot_v3_max_file_mb"] = 0.001

    segs = [
        QcSegment(
            seg_id="s_mid",
            t_start=1.0,
            t_end=1.6,
            label_id="mid_idle_freeze",
            training_role="badcase",
            splice_policy="cut_clean",
            mask_side=None,
            vlm_raw_policy="cut_clean",
        ),
        QcSegment(
            seg_id="s_rec",
            t_start=0.2,
            t_end=0.7,
            label_id="missed_grasp_retry",
            training_role="recovery",
            splice_policy="mask_loss",
            mask_side="left",
            vlm_raw_policy="mask_loss",
        ),
    ]
    ep_ids = _ingest_and_approve(ws, cfg_mgr, data_dir, passed=True, qc_segments=segs)
    repo = Repository(ws)
    try:
        res = repo.resolve_episode(ep_ids[0])
    finally:
        repo.close()

    recipe = ExportRecipe(name="bc_clean", format="lerobot_v3", tail_frames=4, smooth_tail="always")
    out_dir = tmp_path / "lr_out"
    with _open_view_via_discover(res.meta, cfg_mgr) as view:
        pieces = PieceSlicer(cfg_mgr).slice(res, view, recipe)
        manifest = get_exporter("lerobot_v3").__class__(cfg=cfg_mgr).export(
            pieces, [res], {ep_ids[0]: view}, out_dir, recipe
        )

    assert manifest.episodes_included == 1
    assert manifest.pieces_exported == 2
    pqs = sorted((out_dir / "data").rglob("*.parquet"))
    assert len(pqs) == 2

    info = json.loads((out_dir / "meta" / "info.json").read_text("utf-8"))
    assert info["codebase_version"] == "v3.0"
    assert info["robot_type"] == "bimanual_robot"
    assert info["total_frames"] == manifest.total_frames
    assert (out_dir / "meta" / "stats.json").is_file()
    assert (out_dir / "meta" / "tasks.parquet").is_file()
    sub_tbl = pq.read_table(out_dir / "meta" / "subtasks.parquet")
    assert sub_tbl.num_rows == 2
    assert "label" in sub_tbl.column_names
    assert "label_zh" not in sub_tbl.column_names

    cam_dirs = sorted({p.parents[1] for p in (out_dir / "videos").rglob("*.mp4")})
    assert len(cam_dirs) == 4
    for cd in cam_dirs:
        tot_f = sum(_ffprobe_info(mp4)["nb_frames"] for mp4 in sorted(cd.rglob("*.mp4")))
        assert tot_f == manifest.total_frames

    # Configurable export.robot_type
    cfg_custom = ConfigManager(workspace_dir=ws)
    cfg_custom.settings["export"]["robot_type"] = "aloha_v2"
    out_custom = tmp_path / "lr_out_custom_robot"
    with _open_view_via_discover(res.meta, cfg_custom) as view:
        pieces_c = PieceSlicer(cfg_custom).slice(res, view, recipe)
        get_exporter("lerobot_v3").__class__(cfg=cfg_custom).export(
            pieces_c, [res], {ep_ids[0]: view}, out_custom, recipe
        )
    info_custom = json.loads((out_custom / "meta" / "info.json").read_text("utf-8"))
    assert info_custom["robot_type"] == "aloha_v2"

    # Round-trip through LeRobotAdapter
    ad = LeRobotAdapter()
    handles = ad.discover(out_dir)
    assert len(handles) == 2
    with ad.open(handles[0]) as rt_view:
        assert rt_view.kinematics is not None
        assert rt_view.kinematics.raw_state is not None
        assert rt_view.kinematics.raw_state.shape[1] == 81
        assert rt_view.kinematics.eef_pos is not None
        assert rt_view.kinematics.eef_rotvec is not None
        assert rt_view.kinematics.tactile is not None
        frame = rt_view.media.read_frame_rgb("head_rgbd", 0.0)
        assert frame.ndim == 3 and frame.shape[2] == 3


def test_h5_clean_and_zero_copy_and_viz_mp4_mixed_adapters(tmp_path: Path) -> None:
    ws = tmp_path / "ws_mix"
    cfg_mgr = ConfigManager(workspace_dir=ws)
    h5_dir = tmp_path / "ds_h5"
    lr_dir = tmp_path / "ds_lr"
    vid_dir = tmp_path / "ds_vid"
    _make_synthetic_h5_raw_v3(h5_dir, duration_s=2.0)
    _make_synthetic_lerobot_v3(lr_dir, duration_s=2.0)
    _make_synthetic_video_dir(vid_dir, duration_s=2.0)

    ep_ids = (
        _ingest_and_approve(ws, cfg_mgr, h5_dir, passed=True)
        + _ingest_and_approve(ws, cfg_mgr, lr_dir, passed=True)
        + _ingest_and_approve(ws, cfg_mgr, vid_dir, passed=True)
    )
    assert len(ep_ids) == 3

    repo = Repository(ws)
    try:
        resolved_list = [repo.resolve_episode(eid) for eid in ep_ids]
    finally:
        repo.close()

    views = {}
    try:
        for r in resolved_list:
            views[r.episode_id] = _open_view_via_discover(r.meta, cfg_mgr)

        # 1. h5_clean on all 3 adapter types
        r_h5 = ExportRecipe(name="bc_clean", format="h5_clean")
        pieces_h5 = [p for r in resolved_list for p in PieceSlicer(cfg_mgr).slice(r, views[r.episode_id], r_h5)]
        out_h5 = tmp_path / "out_h5"
        m_h5 = get_exporter("h5_clean").export(pieces_h5, resolved_list, views, out_h5, r_h5)
        assert m_h5.episodes_included == 3
        h5_files = sorted(out_h5.glob("*.h5"))
        assert len(h5_files) == 3
        for hp in h5_files:
            with h5py.File(hp, "r") as hf:
                assert "/qc/action_loss_mask" in hf
                assert "/qc/is_tail" in hf
                assert "/qc/subtask_index" in hf
                assert "/qc/resolved_json" in hf
                assert "camera" in hf
                if "ds_vid" in hp.name:
                    assert "robot" not in hf and "hand" not in hf
                else:
                    assert "robot" in hf and "hand" in hf

        # 2. zero_copy_index
        r_zci = ExportRecipe(name="bc_clean", format="zero_copy_index")
        pieces_zci = [p for r in resolved_list for p in PieceSlicer(cfg_mgr).slice(r, views[r.episode_id], r_zci)]
        out_zci = tmp_path / "out_zci"
        m_zci = get_exporter("zero_copy_index").export(pieces_zci, resolved_list, views, out_zci, r_zci)
        assert m_zci.episodes_included == 3
        assert list(out_zci.rglob("*.mp4")) == []
        assert list(out_zci.rglob("*.h5")) == []
        idx_tbl = pq.read_table(out_zci / "qc_index.parquet")
        assert idx_tbl.num_rows == 3
        assert set(idx_tbl["source_uri"].to_pylist()) == {r.meta.source_uri for r in resolved_list}
        lines = [ln for ln in (out_zci / "manifest.jsonl").read_text("utf-8").splitlines() if ln.strip()]
        assert len(lines) == 3
        assert json.loads((out_zci / "subtasks.json").read_text("utf-8"))["subtasks"]

        # 3. heterogeneous lerobot_v3 export across all 3 adapters
        r_lr = ExportRecipe(name="bc_clean", format="lerobot_v3")
        pieces_lr = [p for r in resolved_list for p in PieceSlicer(cfg_mgr).slice(r, views[r.episode_id], r_lr)]
        out_lr = tmp_path / "out_lr_mix"
        m_lr = get_exporter("lerobot_v3").export(pieces_lr, resolved_list, views, out_lr, r_lr)
        assert m_lr.episodes_included == 3
        cam_dirs = sorted({p.parents[1] for p in (out_lr / "videos").rglob("*.mp4")})
        assert len(cam_dirs) >= 4
        for cd in cam_dirs:
            tot_f = sum(_ffprobe_info(mp4)["nb_frames"] for mp4 in sorted(cd.rglob("*.mp4")))
            assert tot_f == m_lr.total_frames

        # 4. viz_mp4 OutputExporter on single episode
        r_viz = ExportRecipe(name="bc_clean", format="viz_mp4")
        one_res = resolved_list[2]
        pieces_viz = PieceSlicer(cfg_mgr).slice(one_res, views[one_res.episode_id], r_viz)
        out_viz = tmp_path / "out_viz"
        m_viz = get_exporter("viz_mp4").export(
            pieces_viz, [one_res], {one_res.episode_id: views[one_res.episode_id]}, out_viz, r_viz
        )
        assert m_viz.episodes_included == 1
        mp4s = list(out_viz.glob("*.mp4"))
        assert len(mp4s) == 1
        assert _ffprobe_info(mp4s[0])["width"] == 1920
    finally:
        for v in views.values():
            v.close()


def test_badcase_pairs_export_emits_metadata_and_zero_kept_is_clean(tmp_path: Path) -> None:
    ws = tmp_path / "ws_bad"
    cfg_mgr = ConfigManager(workspace_dir=ws)
    vid_dir = tmp_path / "ds_fail"
    _make_synthetic_video_dir(vid_dir, duration_s=2.0)
    segs = [
        QcSegment(
            seg_id="s_fail",
            t_start=0.8,
            t_end=1.4,
            label_id="terminal_failure",
            training_role="terminal_fail",
            splice_policy="mask_loss",
            mask_side="both",
            vlm_raw_policy="mask_loss",
        )
    ]
    ep_ids = _ingest_and_approve(ws, cfg_mgr, vid_dir, passed=False, qc_segments=segs)
    repo = Repository(ws)
    try:
        res = repo.resolve_episode(ep_ids[0])
    finally:
        repo.close()

    with _open_view_via_discover(res.meta, cfg_mgr) as view:
        r_bc = ExportRecipe(name="bc_clean", format="lerobot_v3")
        out_empty = tmp_path / "out_empty"
        m_empty = get_exporter("lerobot_v3").export([], [res], {ep_ids[0]: view}, out_empty, r_bc)
        assert m_empty.episodes_included == 0
        assert not (out_empty / "data").exists()

        r_bp = ExportRecipe(name="badcase_pairs", format="lerobot_v3", context_window_s=0.4)
        pieces_bp = PieceSlicer(cfg_mgr).slice(res, view, r_bp)
        out_bp = tmp_path / "out_bp"
        m_bp = get_exporter("lerobot_v3").export(pieces_bp, [res], {ep_ids[0]: view}, out_bp, r_bp)
        assert m_bp.episodes_included == 1
        assert (out_bp / "meta" / "badcases_and_pairs.json").is_file()
        bp_doc = json.loads((out_bp / "meta" / "badcases_and_pairs.json").read_text("utf-8"))
        assert len(bp_doc["badcases"]) == 1
