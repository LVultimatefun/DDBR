"""Unit tests for all 3 InputAdapters (h5_raw_v3, lerobot, video_dir) and media utilities."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.spatial.transform import Rotation

import robo_vlm_qc.io as io_pkg
from robo_vlm_qc.config import ConfigManager
from robo_vlm_qc.io import (
    PreviewCache,
    auto_detect,
    compose_2x2_tile,
    get_adapter,
    list_adapters,
)


def _write_synthetic_mp4(
    path: Path, n_frames: int, w: int, h: int, fps: float = 30.0
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    assert writer.isOpened()
    try:
        for i in range(n_frames):
            bgr = np.zeros((h, w, 3), dtype=np.uint8)
            bgr[:, :, 0] = (i * 17) % 256  # B
            bgr[:, :, 1] = (i * 31) % 256  # G
            bgr[:, :, 2] = (i * 53 + 20) % 256  # R
            writer.write(bgr)
    finally:
        writer.release()


def test_registry_and_priority_ordering() -> None:
    names = list_adapters()
    assert names == ["h5_raw_v3", "lerobot", "video_dir"]
    assert io_pkg.get_adapter("h5_raw_v3").priority < get_adapter("video_dir").priority


def test_tiler_all_three_modes_and_preview_cache_fallback(tmp_path: Path) -> None:
    # 1. Single-cam 1x2 + 64x64 libx264 fallback + mtime invalidation
    vid_dir = tmp_path / "demo_ds"
    vid_path = vid_dir / "ep_small.mp4"
    _write_synthetic_mp4(vid_path, n_frames=12, w=64, h=64, fps=30.0)

    ad = get_adapter("video_dir")
    handles = ad.discover(tmp_path)
    assert len(handles) == 1
    with ad.open(handles[0]) as view:
        assert view.kinematics is None
        assert view.has_kinematics is False
        tile_1x2 = compose_2x2_tile(
            view.media, view.camera_roles, t_s=0.1, cell_w=160, cell_h=120
        )
        assert tile_1x2.dtype == np.uint8
        assert tile_1x2.shape == (120 + 26, 320, 3)

        cache = PreviewCache(tmp_path / "ws")
        p1 = cache.get_or_build(
            view.media,
            dataset=view.meta.dataset,
            stem=view.meta.stem,
            camera="main",
            source_mtime=view.meta.source_mtime,
            target_height=64,
        )
        assert p1.is_file()
        mtime1 = p1.stat().st_mtime
        p2 = cache.get_or_build(
            view.media,
            dataset=view.meta.dataset,
            stem=view.meta.stem,
            camera="main",
            source_mtime=mtime1 + 100.0,
            target_height=64,
        )
        assert p2 == p1
        assert p2.stat().st_mtime >= mtime1

    # 2. Multi-cam wrist_2x2 and torso_head_2x2
    wrist_ep = tmp_path / "pick_and_place" / "success" / "ep_wrist"
    for cname, cw, ch in (("ego", 160, 120), ("left_wrist", 80, 60), ("right_wrist", 80, 60)):
        _write_synthetic_mp4(wrist_ep / f"{cname}.mp4", 6, cw, ch)
    torso_ep = tmp_path / "bimanual_handover" / "ep_torso"
    for cname in ("head_rgbd", "torso_rgbd"):
        _write_synthetic_mp4(torso_ep / f"{cname}.mp4", 6, 160, 120)

    w_handles = ad.discover(tmp_path / "pick_and_place")
    assert len(w_handles) == 1
    with ad.open(w_handles[0]) as view:
        assert view.meta.recorded_success is True
        t_wrist = compose_2x2_tile(
            view.media, view.camera_roles, t_s=0.05, cell_w=120, cell_h=90
        )
        assert t_wrist.shape == (2 * 90 + 26, 240, 3)

    t_handles = ad.discover(tmp_path / "bimanual_handover")
    assert len(t_handles) == 1
    with ad.open(t_handles[0]) as view:
        t_torso = compose_2x2_tile(
            view.media, view.camera_roles, t_s=0.05, cell_w=120, cell_h=90
        )
        assert t_torso.shape == (2 * 90 + 26, 240, 3)


def test_synthetic_h5_raw_v3_slerp_tactile_sbs_and_threads(tmp_path: Path) -> None:
    h5_path = tmp_path / "pick_and_place" / "episode_000000.h5"
    h5_path.parent.mkdir(parents=True)

    t0 = 1700000000.0
    # 5 async streams with different start/end bounds -> intersection [t0+0.1, t0+1.1]
    robot_ts = np.linspace(t0 + 0.10, t0 + 1.30, 121)
    hand_ts = np.linspace(t0 + 0.00, t0 + 1.25, 100)  # ~80Hz
    cam_head_ts = np.linspace(t0 + 0.05, t0 + 1.10, 33)
    cam_torso_ts = np.linspace(t0 + 0.02, t0 + 1.20, 36)
    cam_stereo_ts = np.linspace(t0 + 0.04, t0 + 1.15, 34)

    # Robot state (N, 37) with deliberate quaternion sign flip on ee_left
    r_state = np.zeros((len(robot_ts), 37), dtype=np.float32)
    angles = np.linspace(0.0, 0.8, len(robot_ts))
    quats_xyzw = Rotation.from_rotvec(
        np.column_stack([angles, np.zeros_like(angles), np.zeros_like(angles)])
    ).as_quat()
    quats_wxyz = quats_xyzw[:, [3, 0, 1, 2]].copy()
    quats_wxyz[40:70] *= -1.0  # deliberate sign flip
    r_state[:, 26:30] = quats_wxyz.astype(np.float32)
    r_state[:, 33] = 1.0  # ee_right qw = 1

    # Hand tactile: short 40ms dropout (<0.5s) on hand_a, long 0.65s dropout on hand_b
    ha_rad = np.ones((len(hand_ts), 22), dtype=np.float32) * 0.25
    hb_rad = np.ones((len(hand_ts), 22), dtype=np.float32) * -0.15
    ta_force = np.ones((len(hand_ts), 5, 3), dtype=np.float32) * 3.0
    ta_valid = np.ones(len(hand_ts), dtype=bool)
    ta_valid[20:23] = False  # ~37ms < 0.5s -> repaired, force valid stays True
    ta_force[20:23] = 0.0

    tb_force = np.ones((len(hand_ts), 5, 3), dtype=np.float32) * 2.0
    tb_valid = np.ones(len(hand_ts), dtype=bool)
    tb_valid[30:85] = False  # ~0.69s >= 0.5s -> flagged in sub_valid["force"]
    tb_force[30:85] = 0.0

    def _encode_jpg_list(n: int, w: int, h: int, r_val: int, sbs: bool = False) -> np.ndarray:
        arr = np.empty(n, dtype=object)
        for i in range(n):
            img = np.zeros((h, w, 3), dtype=np.uint8)
            if sbs:
                img[:, : w // 2, 2] = 200  # red left in BGR -> RGB[0]=200
                img[:, w // 2 :, 0] = 200  # blue right in BGR -> RGB[2]=200
            else:
                img[:, :, 1] = (r_val + i) % 256
            ok, buf = cv2.imencode(".jpg", img)
            assert ok
            arr[i] = buf.tobytes()
        return arr

    vlen_dt = h5py.vlen_dtype(np.dtype("uint8"))
    with h5py.File(str(h5_path), "w") as f:
        f.attrs["schema"] = "raw_only_v3"
        f.attrs["task_name"] = "pick_and_place"
        f.attrs["robot_state_dim_names"] = (
            "chassis_dx,chassis_dy,chassis_dyaw,torso_0,torso_1,torso_2,torso_3,"
            + ",".join(f"arm_left_{i}" for i in range(7))
            + ","
            + ",".join(f"arm_right_{i}" for i in range(7))
            + ",head_0,head_1,"
            + "ee_left_x,ee_left_y,ee_left_z,ee_left_qw,ee_left_qx,ee_left_qy,ee_left_qz,"
            + "ee_right_x,ee_right_y,ee_right_z,ee_right_qw,ee_right_qx,ee_right_qy,ee_right_qz"
        )
        rg = f.create_group("robot")
        rg.create_dataset("timestamps", data=robot_ts)
        rg.create_dataset("state", data=r_state)

        hg = f.create_group("hand")
        hg.create_dataset("timestamps", data=hand_ts)
        hg.create_dataset("hand_a_radian", data=ha_rad)
        hg.create_dataset("hand_b_radian", data=hb_rad)
        hg.create_dataset("hand_a_tactile_force", data=ta_force)
        hg.create_dataset("hand_a_tactile_valid", data=ta_valid)
        hg.create_dataset("hand_b_tactile_force", data=tb_force)
        hg.create_dataset("hand_b_tactile_valid", data=tb_valid)

        cg = f.create_group("camera")
        for nm, cts, w, h, sbs in (
            ("head_rgbd", cam_head_ts, 160, 120, False),
            ("torso_rgbd", cam_torso_ts, 160, 120, False),
            ("head_stereo", cam_stereo_ts, 320, 120, True),
        ):
            g = cg.create_group(nm)
            g.attrs["format"] = "jpeg_sbs_or_native" if sbs else "jpeg"
            g.create_dataset("timestamps", data=cts)
            ds = g.create_dataset("jpeg", shape=(len(cts),), dtype=vlen_dt)
            imgs = _encode_jpg_list(len(cts), w, h, 50, sbs=sbs)
            for idx, b in enumerate(imgs):
                ds[idx] = np.frombuffer(b, dtype=np.uint8)

    mgr = ConfigManager()
    handles = auto_detect(h5_path.parent, task_patterns=mgr.task_patterns())
    assert len(handles) == 1
    assert handles[0].task_id == "pick_and_place"

    ad = get_adapter("h5_raw_v3")
    with ad.open(handles[0]) as view:
        T = int(np.floor(((t0 + 1.10) - (t0 + 0.10)) * 30.0)) + 1
        assert T == 31
        assert view.kinematics is not None
        assert view.kinematics.raw_state.shape == (T, 81)
        assert view.kinematics.tactile.shape == (T, 30)
        assert view.kinematics.valid_mask.all()

        # Sign-flipped quaternion must NOT jump
        rotvec_x = view.kinematics.eef_rotvec[:, 0, 0]
        assert np.all(np.diff(rotvec_x) >= -1e-4)
        assert np.max(np.abs(np.diff(rotvec_x))) < 0.1

        # Hand A short dropout repaired cleanly (all ~3.0)
        assert np.allclose(view.kinematics.tactile[:, :15], 3.0, atol=1e-4)
        # Hand B long dropout (>=0.5s) flagged in sub_valid["force"]
        assert not view.kinematics.sub_valid["force"].all()

        # SBS stereo left/right crop roles and RGB colors
        assert view.camera_roles["wide"] == "head_stereo_left"
        assert view.camera_roles["wide_right"] == "head_stereo_right"
        left_rgb = view.media.read_frame_rgb("head_stereo_left", 0.2)
        right_rgb = view.media.read_frame_rgb("head_stereo_right", 0.2)
        assert left_rgb.shape == (120, 160, 3)
        assert right_rgb.shape == (120, 160, 3)
        assert left_rgb[60, 80, 0] > 150 and left_rgb[60, 80, 2] < 50
        assert right_rgb[60, 80, 2] > 150 and right_rgb[60, 80, 0] < 50

        # Thread-safe concurrent reads
        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = [
                pool.submit(view.media.read_frames_batch, "head_rgbd", [0.0, 0.2, 0.5])
                for _ in range(4)
            ]
            for fut in futs:
                res = fut.result()
                assert len(res) == 3 and res[0].shape == (120, 160, 3)


def test_synthetic_lerobot_v2_and_v3(tmp_path: Path) -> None:
    # LeRobot v2
    v2_root = tmp_path / "pick_and_place_v2"
    (v2_root / "meta").mkdir(parents=True)
    _write_synthetic_mp4(
        v2_root / "videos" / "chunk-000" / "observation.images.ego" / "episode_000000.mp4",
        n_frames=10,
        w=160,
        h=120,
        fps=30.0,
    )
    info_v2 = {
        "codebase_version": "v2.0",
        "fps": 30,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.ego": {"dtype": "video", "shape": [120, 160, 3]},
            "observation.state": {"dtype": "float32", "shape": [4], "names": ["j0", "j1", "j2", "j3"]},
            "action": {"dtype": "float32", "shape": [4]},
        },
    }
    (v2_root / "meta" / "info.json").write_text(json.dumps(info_v2), encoding="utf-8")
    (v2_root / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "tasks": ["pick_and_place"], "length": 10}) + "\n",
        encoding="utf-8",
    )
    pq_v2_path = v2_root / "data" / "chunk-000" / "episode_000000.parquet"
    pq_v2_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [0] * 10,
                "timestamp": [i / 30.0 for i in range(10)],
                "observation.state": [[float(i), 0.0, 0.0, 0.0] for i in range(10)],
                "action": [[float(i + 1), 0.0, 0.0, 0.0] for i in range(10)],
            }
        ),
        pq_v2_path,
    )

    handles_v2 = auto_detect(v2_root)
    assert len(handles_v2) == 1
    assert handles_v2[0].adapter_type == "lerobot"
    with get_adapter("lerobot").open(handles_v2[0]) as view:
        assert view.kinematics is not None
        assert view.kinematics.raw_state.shape == (10, 4)
        assert view.kinematics.raw_action.shape == (10, 4)
        fr = view.media.read_frame_rgb("ego", 0.1)
        assert fr.shape == (120, 160, 3)

    # LeRobot v3 (aggregated 2 episodes in 1 MP4 and 1 Parquet, ep1 offset from_timestamp > 0)
    v3_root = tmp_path / "handover_v3"
    (v3_root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (v3_root / "data" / "chunk-000").mkdir(parents=True)
    _write_synthetic_mp4(
        v3_root / "videos" / "observation.images.head_rgbd" / "chunk-000" / "file-000.mp4",
        n_frames=20,
        w=160,
        h=120,
        fps=30.0,
    )
    state_names_81 = (
        ["chassis_dx", "chassis_dy", "chassis_dyaw"]
        + [f"torso_{i}" for i in range(4)]
        + [f"arm_left_{i}" for i in range(7)]
        + [f"arm_right_{i}" for i in range(7)]
        + ["head_0", "head_1"]
        + ["ee_left_x", "ee_left_y", "ee_left_z", "ee_left_qw", "ee_left_qx", "ee_left_qy", "ee_left_qz"]
        + ["ee_right_x", "ee_right_y", "ee_right_z", "ee_right_qw", "ee_right_qx", "ee_right_qy", "ee_right_qz"]
        + [f"hand_left_{i}" for i in range(22)]
        + [f"hand_right_{i}" for i in range(22)]
    )
    info_v3 = {
        "codebase_version": "v3.0",
        "fps": 30,
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.images.head_rgbd": {"dtype": "video", "shape": [120, 160, 3]},
            "observation.state": {"dtype": "float32", "shape": [81], "names": state_names_81},
        },
    }
    (v3_root / "meta" / "info.json").write_text(json.dumps(info_v3), encoding="utf-8")
    pq.write_table(
        pa.table({"task_index": [0], "task": ["bimanual_handover"]}),
        v3_root / "meta" / "tasks.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 1],
                "length": [10, 10],
                "task_index": [0, 0],
                "data/chunk_index": [0, 0],
                "data/file_index": [0, 0],
                "dataset_from_index": [0, 10],
                "dataset_to_index": [10, 20],
                "videos/observation.images.head_rgbd/chunk_index": [0, 0],
                "videos/observation.images.head_rgbd/file_index": [0, 0],
                "videos/observation.images.head_rgbd/from_timestamp": [0.0, 10.0 / 30.0],
                "videos/observation.images.head_rgbd/to_timestamp": [10.0 / 30.0, 20.0 / 30.0],
            }
        ),
        v3_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    rows_81 = []
    for i in range(20):
        vec = [0.0] * 81
        vec[26] = 1.0  # ee_left_qw
        vec[33] = 1.0  # ee_right_qw
        vec[0] = float(i)
        rows_81.append(vec)
    pq.write_table(
        pa.table(
            {
                "episode_index": [0] * 10 + [1] * 10,
                "timestamp": [i / 30.0 for i in range(10)] + [i / 30.0 for i in range(10)],
                "observation.state": rows_81,
            }
        ),
        v3_root / "data" / "chunk-000" / "file-000.parquet",
    )

    handles_v3 = auto_detect(v3_root)
    assert len(handles_v3) == 2
    with get_adapter("lerobot").open(handles_v3[1]) as view1:
        assert view1.kinematics is not None
        assert view1.kinematics.raw_state.shape == (10, 81)
        assert view1.kinematics.eef_pos.shape == (10, 2, 3)
        assert view1.kinematics.eef_rotvec.shape == (10, 2, 3)
        assert view1.kinematics.chassis_pose[0, 0] == pytest.approx(10.0)
        with get_adapter("lerobot").open(handles_v3[0]) as view0:
            f0 = view0.media.read_frame_rgb("head_rgbd", 0.0)
            f1 = view1.media.read_frame_rgb("head_rgbd", 0.0)
            assert not np.array_equal(f0, f1)


def test_preview_cache_nvenc_320x240_zero_fallback_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    vid_dir = tmp_path / "nvenc_ds"
    _write_synthetic_mp4(vid_dir / "ep_hd.mp4", n_frames=12, w=320, h=240, fps=30.0)
    ad = get_adapter("video_dir")
    handles = ad.discover(vid_dir)
    assert len(handles) == 1
    caplog.set_level("WARNING", logger="robo_vlm_qc.io.media")
    with ad.open(handles[0]) as view:
        cache = PreviewCache(tmp_path / "ws")
        out = cache.get_or_build(
            view.media,
            dataset=view.meta.dataset,
            stem=view.meta.stem,
            camera="main",
            source_mtime=view.meta.source_mtime,
            target_height=240,
        )
        assert out.is_file() and out.stat().st_size > 0
        assert not [r for r in caplog.records if "failed" in r.getMessage()]


def test_mp4_reader_cur_pos_minus_one_recovery(tmp_path: Path) -> None:
    from robo_vlm_qc.io.media import Mp4Reader

    mp4_path = tmp_path / "sample.mp4"
    _write_synthetic_mp4(mp4_path, n_frames=10, w=64, h=64, fps=30.0)
    reader = Mp4Reader()
    try:
        fr0 = reader.read_frame_rgb(mp4_path, 0)
        st = reader._get_state(mp4_path)
        st.cur_pos = -1
        fr0_again = reader.read_frame_rgb(mp4_path, 0)
        assert np.array_equal(fr0, fr0_again)
    finally:
        reader.close()


def test_lerobot_v3_chunked_boundary_and_v2_missing_ep_idx_and_image_dtype(
    tmp_path: Path,
) -> None:
    # 1. LeRobot v3 chunked MP4 boundary clamping (3 episodes x 10 frames in 1 MP4)
    v3_root = tmp_path / "chunk_bound_v3"
    (v3_root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (v3_root / "data" / "chunk-000").mkdir(parents=True)
    mp4_path = (
        v3_root / "videos" / "observation.images.head_rgbd" / "chunk-000" / "file-000.mp4"
    )
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 64))
    assert writer.isOpened()
    try:
        for i in range(30):
            bgr = np.zeros((64, 64, 3), dtype=np.uint8)
            if i < 10:
                bgr[:, :, 2] = 220  # ep0: Red
            elif i < 20:
                bgr[:, :, 1] = 220  # ep1: Green
            else:
                bgr[:, :, 0] = 220  # ep2: Blue
            writer.write(bgr)
    finally:
        writer.release()

    info_v3 = {
        "codebase_version": "v3.0",
        "fps": 30,
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.images.head_rgbd": {"dtype": "video", "shape": [64, 64, 3]},
            "observation.state": {"dtype": "float32", "shape": [2]},
        },
    }
    (v3_root / "meta" / "info.json").write_text(json.dumps(info_v3), encoding="utf-8")
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 1, 2],
                "length": [10, 10, 10],
                "data/chunk_index": [0, 0, 0],
                "data/file_index": [0, 0, 0],
                "dataset_from_index": [0, 10, 20],
                "dataset_to_index": [10, 20, 30],
                "videos/observation.images.head_rgbd/chunk_index": [0, 0, 0],
                "videos/observation.images.head_rgbd/file_index": [0, 0, 0],
                "videos/observation.images.head_rgbd/from_timestamp": [
                    0.0,
                    10.0 / 30.0,
                    20.0 / 30.0,
                ],
                "videos/observation.images.head_rgbd/to_timestamp": [
                    10.0 / 30.0,
                    20.0 / 30.0,
                    30.0 / 30.0,
                ],
            }
        ),
        v3_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "index": list(range(30)),
                "timestamp": [(i % 10) / 30.0 for i in range(30)],
                "observation.state": [[float(i), 0.0] for i in range(30)],
            }
        ),
        v3_root / "data" / "chunk-000" / "file-000.parquet",
    )

    ad = get_adapter("lerobot")
    hs = ad.discover(v3_root)
    assert len(hs) == 3
    with ad.open(hs[0]) as v0, ad.open(hs[1]) as v1:
        end0 = v0.media.read_frame_rgb("head_rgbd", v0.duration_s)
        assert end0[32, 32, 0] > 180 and end0[32, 32, 1] < 50
        batch1 = v1.media.read_frames_batch("head_rgbd", [0.0, v1.duration_s])
        assert all(fr[32, 32, 1] > 180 and fr[32, 32, 2] < 50 for fr in batch1)
        assert v1.kinematics is not None
        assert v1.kinematics.raw_state[0, 0] == pytest.approx(10.0)

    # 2. LeRobot v2 per-episode Parquet WITHOUT episode_index column + dtype=="image"
    v2_root = tmp_path / "img_v2"
    (v2_root / "meta").mkdir(parents=True)
    info_v2 = {
        "codebase_version": "v2.0",
        "fps": 30,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "observation.images.ego": {"dtype": "image", "shape": [32, 48, 3]},
            "observation.state": {"dtype": "float32", "shape": [2]},
        },
    }
    (v2_root / "meta" / "info.json").write_text(json.dumps(info_v2), encoding="utf-8")
    (v2_root / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "tasks": ["pick_and_place"], "length": 4}) + "\n"
        + json.dumps({"episode_index": 1, "tasks": ["pick_and_place"], "length": 4}) + "\n",
        encoding="utf-8",
    )
    img_bytes_list = []
    for k in range(4):
        bgr = np.zeros((32, 48, 3), dtype=np.uint8)
        bgr[:, :, 2] = 200  # red
        ok, buf = cv2.imencode(".png", bgr)
        assert ok
        img_bytes_list.append({"bytes": buf.tobytes(), "path": f"f{k}.png"})

    pq_v2 = v2_root / "data" / "chunk-000" / "episode_000000.parquet"
    pq_v2.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "timestamp": [i / 30.0 for i in range(4)],
                "observation.images.ego": img_bytes_list,
                "observation.state": [[float(i), 1.0] for i in range(4)],
            }
        ),
        pq_v2,
    )
    pq_v2_empty = v2_root / "data" / "chunk-000" / "episode_000001.parquet"
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([], type=pa.int64()),
                "timestamp": pa.array([], type=pa.float64()),
                "observation.images.ego": pa.array([], type=pa.binary()),
                "observation.state": pa.array([], type=pa.list_(pa.float32())),
            }
        ),
        pq_v2_empty,
    )
    hs_v2 = ad.discover(v2_root)
    assert len(hs_v2) == 2
    with ad.open(hs_v2[0]) as view_img:
        fr = view_img.media.read_frame_rgb("ego", view_img.duration_s)
        assert fr.shape == (32, 48, 3)
        assert fr[16, 24, 0] > 180 and fr[16, 24, 2] < 50
    with pytest.raises(ValueError, match="rows missing"):
        ad.open(hs_v2[1])


def test_h5_raw_v3_action_resampling_and_slerp_clip(tmp_path: Path) -> None:
    from robo_vlm_qc.io.h5_raw_v3 import _resample_quat_slerp

    # 1. Slerp eval_t clip when grid_t[-1] exceeds robot_ts[-1] by tiny ULP
    r_ts = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    q = np.array([[1.0, 0.0, 0.0, 0.0]] * 3, dtype=np.float64)
    grid = np.array([0.0, 0.5, 1.0 + 1e-12], dtype=np.float64)
    rv, qw, val = _resample_quat_slerp(r_ts, q, grid)
    assert rv.shape == (3, 3) and qw.shape == (3, 4) and val.all()

    # 2. 100Hz /robot/action resampling onto 30Hz grid_t
    h5_path = tmp_path / "pick_and_place_act" / "episode_000000.h5"
    h5_path.parent.mkdir(parents=True)
    t0 = 1700000000.0
    robot_ts = np.linspace(t0, t0 + 1.0, 101)
    hand_ts = np.linspace(t0, t0 + 1.0, 81)
    cam_ts = np.linspace(t0, t0 + 1.0, 31)

    r_state = np.zeros((101, 37), dtype=np.float32)
    r_state[:, 26] = 1.0
    r_state[:, 33] = 1.0
    r_action = np.column_stack(
        [np.linspace(0.0, 10.0, 101, dtype=np.float32)]
        + [np.zeros(101, dtype=np.float32) for _ in range(80)]
    )
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    vlen_dt = h5py.vlen_dtype(np.dtype("uint8"))
    with h5py.File(str(h5_path), "w") as f:
        f.attrs["schema"] = "raw_only_v3"
        rg = f.create_group("robot")
        rg.create_dataset("timestamps", data=robot_ts)
        rg.create_dataset("state", data=r_state)
        rg.create_dataset("action", data=r_action)
        hg = f.create_group("hand")
        hg.create_dataset("timestamps", data=hand_ts)
        hg.create_dataset("hand_a_radian", data=np.zeros((81, 22), dtype=np.float32))
        hg.create_dataset("hand_b_radian", data=np.zeros((81, 22), dtype=np.float32))
        cg = f.create_group("camera")
        g = cg.create_group("head_rgbd")
        g.create_dataset("timestamps", data=cam_ts)
        ds = g.create_dataset("jpeg", shape=(len(cam_ts),), dtype=vlen_dt)
        for i in range(len(cam_ts)):
            ds[i] = np.frombuffer(buf.tobytes(), dtype=np.uint8)

    ad = get_adapter("h5_raw_v3")
    hs = ad.discover(h5_path)
    with ad.open(hs[0]) as view:
        assert view.kinematics is not None
        T = len(view.kinematics.timestamps_s)
        assert T == 31
        assert view.kinematics.raw_action is not None
        assert view.kinematics.raw_action.shape == (T, 81)
        assert view.kinematics.raw_action[0, 0] == pytest.approx(0.0, abs=1e-3)
        assert view.kinematics.raw_action[-1, 0] == pytest.approx(10.0, abs=1e-2)


def test_synthetic_h5_concurrent_decode_cache_and_close_cleanup(tmp_path: Path) -> None:
    h5_path = tmp_path / "pick_and_place" / "episode_000000.h5"
    h5_path.parent.mkdir(parents=True)
    t0 = 1700000000.0
    n_cam = 31
    ts = np.linspace(t0, t0 + 1.0, n_cam)
    vlen_dt = h5py.vlen_dtype(np.dtype("uint8"))
    with h5py.File(str(h5_path), "w") as f:
        f.attrs["schema"] = "raw_only_v3"
        f.attrs["task_name"] = "pick_and_place"
        rg = f.create_group("robot")
        rg.create_dataset("timestamps", data=ts)
        r_state = np.zeros((n_cam, 37), dtype=np.float32)
        r_state[:, 26] = 1.0
        r_state[:, 33] = 1.0
        rg.create_dataset("state", data=r_state)
        hg = f.create_group("hand")
        hg.create_dataset("timestamps", data=ts)
        hg.create_dataset("hand_a_radian", data=np.zeros((n_cam, 22), dtype=np.float32))
        hg.create_dataset("hand_b_radian", data=np.zeros((n_cam, 22), dtype=np.float32))
        cg = f.create_group("camera")
        for nm, w, h, sbs in (
            ("head_rgbd", 160, 120, False),
            ("torso_rgbd", 160, 120, False),
            ("head_stereo", 320, 120, True),
        ):
            g = cg.create_group(nm)
            g.attrs["format"] = "jpeg_sbs_or_native" if sbs else "jpeg"
            g.create_dataset("timestamps", data=ts)
            ds = g.create_dataset("jpeg", shape=(n_cam,), dtype=vlen_dt)
            for idx in range(n_cam):
                img = np.full((h, w, 3), (idx * 7) % 256, dtype=np.uint8)
                ok, buf = cv2.imencode(".jpg", img)
                assert ok
                ds[idx] = np.frombuffer(buf.tobytes(), dtype=np.uint8)

    h = auto_detect(h5_path.parent, task_patterns=ConfigManager().task_patterns())[0]
    view = get_adapter("h5_raw_v3").open(h)
    try:
        times = np.linspace(0.0, view.duration_s - 0.1, 10).tolist()
        cams = [c.name for c in view.media.cameras]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = {c: pool.submit(view.media.read_frames_batch, c, times) for c in cams}
            res = {c: f.result() for c, f in futs.items()}
            del futs
        assert len(res) == 4 and all(len(v) == 10 for v in res.values())
        assert len(view.media._decode_cache) <= 2
        del res

        _ = view.media.read_frame_rgb("head_stereo_left", 0.5)
        assert 1 <= len(view.media._decode_cache) <= 2
    finally:
        view.close()
    assert len(view.media._decode_cache) == 0
