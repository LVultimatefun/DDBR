"""Shared E2E test helpers, synthetic dataset builders, CLI/API wrappers, and validators."""

from __future__ import annotations

import importlib
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PYTHON_BIN = Path(sys.executable)

FORBIDDEN_LEGACY_FIELDS = ("long_pause", "back_and_forth")

_ROBOT_STATE_DIM_NAMES = (
    "chassis_dx,chassis_dy,chassis_dyaw,"
    "torso_0,torso_1,torso_2,torso_3,"
    "arm_left_0,arm_left_1,arm_left_2,arm_left_3,arm_left_4,arm_left_5,arm_left_6,"
    "arm_right_0,arm_right_1,arm_right_2,arm_right_3,arm_right_4,arm_right_5,arm_right_6,"
    "head_0,head_1,"
    "ee_left_x,ee_left_y,ee_left_z,ee_left_qw,ee_left_qx,ee_left_qy,ee_left_qz,"
    "ee_right_x,ee_right_y,ee_right_z,ee_right_qw,ee_right_qx,ee_right_qy,ee_right_qz"
)


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def write_workspace_config(
    ws: Path,
    *,
    base_url: str | None = None,
    api_key_env: str = "MOCK_VLM_API_KEY",
    stage_models: dict[str, str] | None = None,
    budget: dict[str, Any] | None = None,
    extra_overrides: dict[str, Any] | None = None,
) -> Path:
    """Write or merge ``<ws>/configs_override/default.yaml`` safely."""
    ws = Path(ws)
    override_dir = ws / "configs_override"
    override_dir.mkdir(parents=True, exist_ok=True)
    target = override_dir / "default.yaml"

    existing: dict[str, Any] = {}
    if target.is_file():
        loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            existing = loaded

    vlm_patch: dict[str, Any] = {}
    if base_url is not None:
        vlm_patch.update(
            {
                "provider": "openai_compat",
                "base_url": base_url,
                "api_key_env": api_key_env,
                "key_file": "__mock_unused__.env",
                "retry_min_wait_s": 0.01,
                "retry_max_wait_s": 0.05,
            }
        )
    if stage_models is not None:
        vlm_patch["stage_models"] = dict(stage_models)
    if budget is not None:
        vlm_patch["budget"] = dict(budget)

    patch: dict[str, Any] = {}
    if vlm_patch:
        patch["vlm"] = vlm_patch
    if extra_overrides:
        patch = _deep_merge_dicts(patch, extra_overrides)

    merged = _deep_merge_dicts(existing, patch)
    target.write_text(
        yaml.safe_dump(merged, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target


# ---------------------------------------------------------------------------
# Frame & Video Synthetic Helpers
# ---------------------------------------------------------------------------


def _make_frame_bgr(
    width: int,
    height: int,
    frame_idx: int,
    total_frames: int,
    label: str,
    accent_bgr: tuple[int, int, int] = (40, 180, 240),
) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    bg_val = 28 + (frame_idx * 3) % 40
    img[:, :] = (bg_val, 36, 44)

    frac = frame_idx / max(1, total_frames - 1)
    box_w = max(24, width // 10)
    box_h = max(24, height // 10)
    cx = int(round(box_w + frac * max(1, width - 2 * box_w)))
    cy = int(round(height * 0.55 + np.sin(frac * np.pi) * (-height * 0.2)))
    x1 = max(0, cx - box_w // 2)
    y1 = max(0, cy - box_h // 2)
    x2 = min(width - 1, x1 + box_w)
    y2 = min(height - 1, y1 + box_h)
    cv2.rectangle(img, (x1, y1), (x2, y2), accent_bgr, thickness=-1)
    cv2.putText(
        img,
        f"{label} #{frame_idx:04d}",
        (16, max(28, height // 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.5, min(1.2, width / 900.0)),
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    return img


def _encode_jpeg_bytes(img_bgr: np.ndarray, quality: int = 65) -> np.ndarray:
    ok, buf = cv2.imencode(
        ".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise RuntimeError("cv2.imencode('.jpg') failed")
    return np.frombuffer(buf.tobytes(), dtype=np.uint8)


def _write_mp4_frames(path: Path, frames_bgr: list[np.ndarray], fps: float = 30.0) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames_bgr:
        raise ValueError("Cannot write empty MP4")
    h, w = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {path}")
    try:
        for frame in frames_bgr:
            writer.write(frame)
    finally:
        writer.release()


# ---------------------------------------------------------------------------
# Synthetic Dataset Builders
# ---------------------------------------------------------------------------


def make_synthetic_h5_raw_v3(
    out_dir: Path,
    name: str = "episode_000000.h5",
    duration_s: float = 4.0,
    with_tactile_dropout: bool = True,
    with_sbs_stereo: bool = True,
    task_name: str = "pick_and_place",
    *,
    stationary: bool = False,
    long_tactile_dropout: bool = False,
    fps: float = 30.0,
    episode_index: int = 0,
) -> Path:
    """Create a synthetic ``raw_only_v3`` HDF5 matching the async multi-stream schema."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / name

    t0 = 1794895000.0
    dt_cam = 1.0 / float(fps)

    # Async stream timestamps whose common intersection is [t0, t0 + duration_s]
    head_ts = np.arange(t0 - 0.04, t0 + duration_s + 0.02, dt_cam, dtype=np.float64)
    torso_ts = np.arange(t0 - 0.02, t0 + duration_s + 0.04, dt_cam, dtype=np.float64)
    stereo_ts = np.arange(t0 - 0.01, t0 + duration_s + 0.03, dt_cam, dtype=np.float64)

    n_hand = max(10, int(round((duration_s + 0.08) * 78.8)))
    hand_ts = np.linspace(t0 - 0.03, t0 + duration_s + 0.05, n_hand, dtype=np.float64)

    n_robot = max(10, int(round((duration_s + 0.04) * 90.0)))
    robot_ts = np.linspace(t0, t0 + duration_s + 0.04, n_robot, dtype=np.float64)

    # Robot state (N_robot, 37)
    robot_state = np.zeros((n_robot, 37), dtype=np.float32)
    t_rel = robot_ts - t0
    idle_end = min(0.45, duration_s * 0.12)

    if stationary:
        move = np.zeros_like(t_rel)
    else:
        move = np.clip((t_rel - idle_end) / max(0.5, duration_s - idle_end), 0.0, 1.0)

    robot_state[:, 0] = (0.01 * move).astype(np.float32)
    robot_state[:, 3] = (0.04 * move).astype(np.float32)
    robot_state[:, 7:14] = (0.35 * move[:, None]).astype(np.float32)
    robot_state[:, 14:21] = (-0.30 * move[:, None]).astype(np.float32)
    robot_state[:, 21] = (0.05 * move).astype(np.float32)

    # ee_left xyz + quat (w, x, y, z)
    robot_state[:, 23] = (0.30 + 0.28 * move).astype(np.float32)
    robot_state[:, 24] = (0.20 + 0.18 * np.sin(move * np.pi)).astype(np.float32)
    robot_state[:, 25] = (0.40 + 0.12 * move).astype(np.float32)

    # ee_right xyz + quat (w, x, y, z)
    robot_state[:, 30] = (0.30 + 0.24 * move).astype(np.float32)
    robot_state[:, 31] = (-0.20 - 0.15 * np.sin(move * np.pi)).astype(np.float32)
    robot_state[:, 32] = (0.40 + 0.10 * move).astype(np.float32)

    # Quaternions with sign flips in second half to test Slerp continuity
    theta_l = 0.6 * move
    theta_r = -0.5 * move
    ql = np.stack(
        [np.cos(theta_l / 2.0), np.sin(theta_l / 2.0), np.zeros_like(theta_l), np.zeros_like(theta_l)],
        axis=1,
    )
    qr = np.stack(
        [np.cos(theta_r / 2.0), np.zeros_like(theta_r), np.sin(theta_r / 2.0), np.zeros_like(theta_r)],
        axis=1,
    )
    half = n_robot // 2
    ql[half:] *= -1.0
    qr[half:] *= -1.0
    robot_state[:, 26:30] = ql.astype(np.float32)
    robot_state[:, 33:37] = qr.astype(np.float32)

    # Hand data (N_hand, 22) + (N_hand, 5, 3) tactile
    hand_move = np.interp(hand_ts, robot_ts, move)
    hand_a_q = (0.4 * hand_move[:, None] * np.ones((1, 22))).astype(np.float32)
    hand_b_q = (-0.35 * hand_move[:, None] * np.ones((1, 22))).astype(np.float32)

    ramp = np.linspace(0.5, 2.5, n_hand, dtype=np.float32)
    hand_a_tac = np.broadcast_to(ramp[:, None, None], (n_hand, 5, 3)).copy()
    hand_b_tac = (hand_a_tac * 0.8).astype(np.float32)
    hand_a_valid = np.ones(n_hand, dtype=bool)
    hand_b_valid = np.ones(n_hand, dtype=bool)

    if with_tactile_dropout and n_hand > 20:
        mid = n_hand // 3
        # Short dropout <= 70ms (3 samples @ 78.8Hz ~= 38ms)
        hand_a_valid[mid : mid + 3] = False
        hand_a_tac[mid : mid + 3] = 0.0
        hand_b_valid[mid + 4 : mid + 7] = False
        hand_b_tac[mid + 4 : mid + 7] = 0.0

    if long_tactile_dropout and n_hand > 80:
        # Long dropout >= 0.7s (~56 samples @ 78.8Hz)
        start_gap = int(n_hand * 0.6)
        end_gap = min(n_hand - 2, start_gap + 56)
        hand_a_valid[start_gap:end_gap] = False
        hand_a_tac[start_gap:end_gap] = 0.0

    head_jpegs = [
        _encode_jpeg_bytes(
            _make_frame_bgr(1280, 720, i, len(head_ts), "HEAD", (40, 180, 240)),
            quality=60,
        )
        for i in range(len(head_ts))
    ]
    torso_jpegs = [
        _encode_jpeg_bytes(
            _make_frame_bgr(1280, 720, i, len(torso_ts), "TORSO", (60, 220, 120)),
            quality=60,
        )
        for i in range(len(torso_ts))
    ]
    stereo_jpegs: list[np.ndarray] = []
    if with_sbs_stereo:
        base_sbs = np.zeros((1296, 3200, 3), dtype=np.uint8)
        base_sbs[:, :1600] = (32, 42, 60)
        base_sbs[:, 1600:] = (32, 60, 42)
        for i in range(len(stereo_ts)):
            frame = base_sbs.copy()
            dx = int((i / max(1, len(stereo_ts) - 1)) * 1200)
            cv2.rectangle(frame, (100 + dx, 400), (260 + dx, 560), (230, 160, 40), -1)
            cv2.rectangle(
                frame, (1700 + dx, 400), (1860 + dx, 560), (60, 230, 140), -1
            )
            cv2.putText(
                frame,
                f"STEREO_L #{i:04d}",
                (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"STEREO_R #{i:04d}",
                (1640, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                2,
            )
            stereo_jpegs.append(_encode_jpeg_bytes(frame, quality=50))

    vlen_u8 = h5py.vlen_dtype(np.dtype("uint8"))
    with h5py.File(h5_path, "w") as f:
        f.attrs["schema"] = "raw_only_v3"
        f.attrs["align_mode"] = "none"
        f.attrs["camera_encoding"] = "jpeg"
        f.attrs["task_name"] = task_name
        f.attrs["episode_index"] = int(episode_index)
        f.attrs["created_at"] = "2026_11_20_12_00_00"
        f.attrs["episode_start_timestamp"] = float(t0)
        f.attrs["camera_head_rgbd_num_frames"] = len(head_ts)
        f.attrs["camera_torso_rgbd_num_frames"] = len(torso_ts)
        f.attrs["camera_head_stereo_num_frames"] = len(stereo_ts) if with_sbs_stereo else 0
        f.attrs["hand_num_frames"] = n_hand
        f.attrs["hand_angle_unit"] = "radians"
        f.attrs["hand_timestamp_meaning"] = "robot_recv_time"
        f.attrs["robot_num_frames"] = n_robot
        f.attrs["robot_state_dim"] = 37
        f.attrs["robot_state_layout"] = "chassis3|torso4|armL7|armR7|head2|eeL7|eeR7"
        f.attrs["robot_state_dim_names"] = _ROBOT_STATE_DIM_NAMES

        cam_grp = f.create_group("camera")
        g_head = cam_grp.create_group("head_rgbd")
        g_head.attrs["format"] = "jpeg"
        ds_h = g_head.create_dataset("jpeg", (len(head_jpegs),), dtype=vlen_u8)
        for i, arr in enumerate(head_jpegs):
            ds_h[i] = arr
        g_head.create_dataset("timestamps", data=head_ts, dtype=np.float64)

        g_torso = cam_grp.create_group("torso_rgbd")
        g_torso.attrs["format"] = "jpeg"
        ds_t = g_torso.create_dataset("jpeg", (len(torso_jpegs),), dtype=vlen_u8)
        for i, arr in enumerate(torso_jpegs):
            ds_t[i] = arr
        g_torso.create_dataset("timestamps", data=torso_ts, dtype=np.float64)

        if with_sbs_stereo:
            g_stereo = cam_grp.create_group("head_stereo")
            g_stereo.attrs["format"] = "jpeg_sbs_or_native"
            ds_s = g_stereo.create_dataset("jpeg", (len(stereo_jpegs),), dtype=vlen_u8)
            for i, arr in enumerate(stereo_jpegs):
                ds_s[i] = arr
            g_stereo.create_dataset("timestamps", data=stereo_ts, dtype=np.float64)

        hand_grp = f.create_group("hand")
        hand_grp.attrs["hand_a_joint_dim"] = 22
        hand_grp.attrs["hand_b_joint_dim"] = 22
        ds_ha = hand_grp.create_dataset("hand_a_radian", data=hand_a_q, dtype=np.float32)
        ds_ha.attrs["unit"] = "rad"
        ds_hb = hand_grp.create_dataset("hand_b_radian", data=hand_b_q, dtype=np.float32)
        ds_hb.attrs["unit"] = "rad"
        ds_ta = hand_grp.create_dataset(
            "hand_a_tactile_force", data=hand_a_tac, dtype=np.float32
        )
        ds_ta.attrs["axis"] = "xyz"
        ds_ta.attrs["finger_order"] = "THUMB,INDEX,MIDDLE,RING,PINKY"
        ds_ta.attrs["unit"] = "N"
        hand_grp.create_dataset("hand_a_tactile_valid", data=hand_a_valid, dtype=bool)
        ds_tb = hand_grp.create_dataset(
            "hand_b_tactile_force", data=hand_b_tac, dtype=np.float32
        )
        ds_tb.attrs["axis"] = "xyz"
        ds_tb.attrs["finger_order"] = "THUMB,INDEX,MIDDLE,RING,PINKY"
        ds_tb.attrs["unit"] = "N"
        hand_grp.create_dataset("hand_b_tactile_valid", data=hand_b_valid, dtype=bool)
        hand_grp.create_dataset("timestamps", data=hand_ts, dtype=np.float64)

        robot_grp = f.create_group("robot")
        ds_r = robot_grp.create_dataset("state", data=robot_state, dtype=np.float32)
        ds_r.attrs["chassis"] = "relative_to_episode_start"
        ds_r.attrs["dim"] = 37
        ds_r.attrs["layout"] = "chassis3|torso4|armL7|armR7|head2|eeL7|eeR7"
        robot_grp.create_dataset("timestamps", data=robot_ts, dtype=np.float64)

    return h5_path


def make_synthetic_lerobot(
    out_dir: Path,
    version: str = "v3",
    n_episodes: int = 2,
    duration_s: float = 2.0,
    *,
    fps: int = 30,
    task_name: str = "pick_and_place",
    stationary: bool = False,
) -> Path:
    """Create a valid LeRobot v2 or v3 dataset with 81D Parquet + MP4 camera files."""
    if version not in ("v2", "v3"):
        raise ValueError(f"Unsupported LeRobot version {version!r}; expected 'v2' or 'v3'")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    n_frames = max(6, int(round(duration_s * fps)))
    cam_keys = [
        ("observation.images.main", 640, 480, (50, 180, 240)),
        ("observation.images.left_wrist", 640, 480, (60, 220, 120)),
        ("observation.images.right_wrist", 640, 480, (220, 140, 60)),
    ]

    def _ep_arrays(ep_idx: int) -> tuple[np.ndarray, np.ndarray]:
        t_norm = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)
        move = np.zeros_like(t_norm) if stationary else np.clip((t_norm - 0.15) / 0.85, 0.0, 1.0)
        state = np.zeros((n_frames, 81), dtype=np.float32)
        state[:, 7:14] = (0.3 * move[:, None]).astype(np.float32)
        state[:, 14:21] = (-0.25 * move[:, None]).astype(np.float32)
        state[:, 23] = (0.30 + 0.25 * move + 0.01 * ep_idx).astype(np.float32)
        state[:, 24] = (0.20 + 0.15 * np.sin(move * np.pi)).astype(np.float32)
        state[:, 25] = (0.40 + 0.10 * move).astype(np.float32)
        state[:, 26] = 1.0  # qw
        state[:, 30] = (0.30 + 0.20 * move).astype(np.float32)
        state[:, 31] = (-0.20 - 0.12 * np.sin(move * np.pi)).astype(np.float32)
        state[:, 32] = (0.40 + 0.08 * move).astype(np.float32)
        state[:, 33] = 1.0  # qw
        state[:, 37:59] = (0.2 * move[:, None]).astype(np.float32)
        state[:, 59:81] = (0.2 * move[:, None]).astype(np.float32)
        action = np.concatenate([state[1:], state[-1:]], axis=0)
        return state, action

    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [81],
            "names": [f"state_{i}" for i in range(81)],
        },
        "action": {
            "dtype": "float32",
            "shape": [81],
            "names": [f"action_{i}" for i in range(81)],
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for ck, w, h, _ in cam_keys:
        features[ck] = {
            "dtype": "video",
            "shape": [h, w, 3],
            "names": ["height", "width", "channels"],
            "info": {"video.fps": float(fps), "video.codec": "mpeg4"},
        }

    tasks_table = pa.table(
        {"task_index": pa.array([0], type=pa.int64()), "task": pa.array([task_name], type=pa.string())}
    )
    pq.write_table(tasks_table, meta_dir / "tasks.parquet")
    (meta_dir / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": task_name}) + "\n", encoding="utf-8"
    )

    if version == "v3":
        all_states: list[list[float]] = []
        all_actions: list[list[float]] = []
        all_ts: list[float] = []
        all_fidx: list[int] = []
        all_epidx: list[int] = []
        all_gidx: list[int] = []
        all_tidx: list[int] = []
        ep_rows: list[dict[str, Any]] = []
        cam_frames_all: dict[str, list[np.ndarray]] = {ck: [] for ck, _, _, _ in cam_keys}

        g_offset = 0
        for ep_idx in range(n_episodes):
            st, act = _ep_arrays(ep_idx)
            for f_idx in range(n_frames):
                all_states.append(st[f_idx].tolist())
                all_actions.append(act[f_idx].tolist())
                all_ts.append(float(f_idx / fps))
                all_fidx.append(f_idx)
                all_epidx.append(ep_idx)
                all_gidx.append(g_offset + f_idx)
                all_tidx.append(0)
            for ck, w, h, color in cam_keys:
                short_name = ck.split(".")[-1]
                for f_idx in range(n_frames):
                    cam_frames_all[ck].append(
                        _make_frame_bgr(w, h, f_idx, n_frames, f"{short_name}_e{ep_idx}", color)
                    )
            ep_from_s = float(ep_idx * n_frames / fps)
            ep_to_s = float(((ep_idx + 1) * n_frames - 1) / fps)
            ep_meta_row: dict[str, Any] = {
                "episode_index": ep_idx,
                "tasks": [task_name],
                "length": n_frames,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": g_offset,
                "dataset_to_index": g_offset + n_frames,
            }
            for ck, _, _, _ in cam_keys:
                ep_meta_row[f"videos/{ck}/chunk_index"] = 0
                ep_meta_row[f"videos/{ck}/file_index"] = 0
                ep_meta_row[f"videos/{ck}/from_timestamp"] = ep_from_s
                ep_meta_row[f"videos/{ck}/to_timestamp"] = ep_to_s
            ep_rows.append(ep_meta_row)
            g_offset += n_frames

        data_dir = out_dir / "data" / "chunk-000"
        data_dir.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "observation.state": pa.array(all_states, type=pa.list_(pa.float32(), 81)),
                "action": pa.array(all_actions, type=pa.list_(pa.float32(), 81)),
                "timestamp": pa.array(all_ts, type=pa.float32()),
                "frame_index": pa.array(all_fidx, type=pa.int64()),
                "episode_index": pa.array(all_epidx, type=pa.int64()),
                "index": pa.array(all_gidx, type=pa.int64()),
                "task_index": pa.array(all_tidx, type=pa.int64()),
            }
        )
        pq.write_table(table, data_dir / "file-000.parquet")

        for ck, _, _, _ in cam_keys:
            vid_path = out_dir / "videos" / ck / "chunk-000" / "file-000.mp4"
            _write_mp4_frames(vid_path, cam_frames_all[ck], fps=float(fps))

        ep_meta_dir = meta_dir / "episodes" / "chunk-000"
        ep_meta_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(ep_rows), ep_meta_dir / "file-000.parquet")

        info = {
            "codebase_version": "v3.0",
            "robot_type": "bimanual_robot",
            "total_episodes": n_episodes,
            "total_frames": n_episodes * n_frames,
            "total_tasks": 1,
            "chunks_size": 1000,
            "fps": fps,
            "splits": {"train": f"0:{n_episodes}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": features,
        }
        (meta_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    else:
        ep_jsonl_lines: list[str] = []
        g_offset = 0
        for ep_idx in range(n_episodes):
            st, act = _ep_arrays(ep_idx)
            data_dir = out_dir / "data" / "chunk-000"
            data_dir.mkdir(parents=True, exist_ok=True)
            table = pa.table(
                {
                    "observation.state": pa.array(st.tolist(), type=pa.list_(pa.float32(), 81)),
                    "action": pa.array(act.tolist(), type=pa.list_(pa.float32(), 81)),
                    "timestamp": pa.array(
                        [float(i / fps) for i in range(n_frames)], type=pa.float32()
                    ),
                    "frame_index": pa.array(list(range(n_frames)), type=pa.int64()),
                    "episode_index": pa.array([ep_idx] * n_frames, type=pa.int64()),
                    "index": pa.array(
                        list(range(g_offset, g_offset + n_frames)), type=pa.int64()
                    ),
                    "task_index": pa.array([0] * n_frames, type=pa.int64()),
                }
            )
            pq.write_table(table, data_dir / f"episode_{ep_idx:06d}.parquet")
            for ck, w, h, color in cam_keys:
                short_name = ck.split(".")[-1]
                frames = [
                    _make_frame_bgr(w, h, f_idx, n_frames, f"{short_name}_e{ep_idx}", color)
                    for f_idx in range(n_frames)
                ]
                vid_path = (
                    out_dir
                    / "videos"
                    / "chunk-000"
                    / ck
                    / f"episode_{ep_idx:06d}.mp4"
                )
                _write_mp4_frames(vid_path, frames, fps=float(fps))
            ep_jsonl_lines.append(
                json.dumps(
                    {
                        "episode_index": ep_idx,
                        "tasks": [task_name],
                        "length": n_frames,
                    }
                )
            )
            g_offset += n_frames

        (meta_dir / "episodes.jsonl").write_text(
            "\n".join(ep_jsonl_lines) + "\n", encoding="utf-8"
        )
        info = {
            "codebase_version": "v2.1",
            "robot_type": "bimanual_robot",
            "total_episodes": n_episodes,
            "total_frames": n_episodes * n_frames,
            "total_tasks": 1,
            "chunks_size": 1000,
            "fps": fps,
            "splits": {"train": f"0:{n_episodes}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features,
        }
        (meta_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    return out_dir


def make_synthetic_video_dir(
    out_dir: Path,
    multi_cam: bool = True,
    duration_s: float = 2.0,
    *,
    fps: int = 30,
    n_episodes: int = 1,
    scene_change: bool = False,
) -> Path:
    """Create a pure-MP4 directory (``kinematics=None``) with 1 or 3 cameras."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_frames = max(6, int(round(duration_s * fps)))
    cams = (
        [
            ("main", (40, 180, 240)),
            ("left_wrist", (60, 220, 120)),
            ("right_wrist", (220, 140, 60)),
        ]
        if multi_cam
        else [("main", (40, 180, 240))]
    )
    for ep_idx in range(n_episodes):
        ep_dir = out_dir / f"ep_{ep_idx:04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        for cam_name, color in cams:
            frames = []
            for f_idx in range(n_frames):
                img = _make_frame_bgr(640, 480, f_idx, n_frames, f"{cam_name}_{ep_idx}", color)
                if scene_change and f_idx >= n_frames // 2:
                    # Large high-contrast scene change in second half so SSIM < 0.85 and L1 > 0.12
                    h_img, w_img = img.shape[:2]
                    img[: h_img // 2, :] = (20, 210, 230)
                    img[h_img // 2 :, : w_img // 2] = (220, 60, 190)
                frames.append(img)
            _write_mp4_frames(ep_dir / f"{cam_name}.mp4", frames, fps=float(fps))
    return out_dir


# ---------------------------------------------------------------------------
# CLI & FastAPI Test Helpers (Lazy Imports Only)
# ---------------------------------------------------------------------------


def _resolve_cli_cmd() -> list[str]:
    return [sys.executable, "-m", "robo_vlm_qc"]


def run_cli(
    args: list[str],
    cwd: Path | None = None,
    env_overrides: dict[str, str | None] | None = None,
    *,
    timeout_s: float = 120.0,
) -> subprocess.CompletedProcess[str]:
    """Run ``robo_vlm_qc`` CLI in a sanitized subprocess without leaking API keys."""
    env = os.environ.copy()
    env.pop("DASHSCOPE_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    env.pop("GEMINI_API_KEY", None)
    for proxy_var in (
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
    ):
        env.pop(proxy_var, None)
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["MOCK_VLM_API_KEY"] = "mock-e2e-key"

    existing_pp = env.get("PYTHONPATH", "")
    pp_parts = [str(REPO_ROOT)]
    if existing_pp:
        pp_parts.append(existing_pp)
    env["PYTHONPATH"] = os.pathsep.join(pp_parts)

    if env_overrides:
        for k, v in env_overrides.items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = str(v)

    cmd = _resolve_cli_cmd() + list(args)
    return subprocess.run(
        cmd,
        cwd=str(cwd or REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def create_test_client(
    workspace: Path,
    config_dir: Path | None = None,
    reviewer_token: str = "rev-token",
    lead_token: str = "lead-token",
    *,
    no_auth: bool = False,
    cors_origins: list[str] | None = None,
) -> Any:
    """Lazily import FastAPI app factory and return a ``TestClient``."""
    from fastapi.testclient import TestClient

    app_mod = importlib.import_module("robo_vlm_qc.api.app")
    factory = getattr(app_mod, "create_app", None) or getattr(app_mod, "build_app", None)
    if factory is None:
        raise AttributeError(
            "robo_vlm_qc.api.app must expose create_app(...) or build_app(...)"
        )

    cfg_dir = Path(config_dir) if config_dir is not None else (REPO_ROOT / "configs")
    sig = inspect.signature(factory)
    kwargs: dict[str, Any] = {}
    for pname in sig.parameters:
        if pname in ("workspace", "workspace_dir"):
            kwargs[pname] = Path(workspace)
        elif pname in ("config_dir", "configs_dir"):
            kwargs[pname] = cfg_dir
        elif pname == "reviewer_token":
            kwargs[pname] = reviewer_token
        elif pname == "lead_token":
            kwargs[pname] = lead_token
        elif pname == "no_auth":
            kwargs[pname] = no_auth
        elif pname == "cors_origins" and cors_origins is not None:
            kwargs[pname] = cors_origins
        elif pname == "tokens":
            kwargs[pname] = {"reviewer": reviewer_token, "lead": lead_token}

    app = factory(**kwargs)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Validators & Read-Only Guards
# ---------------------------------------------------------------------------


def ffprobe_video_info(mp4_path: Path) -> dict[str, Any]:
    """Run ``ffprobe`` on ``mp4_path`` and return stream/format summary."""
    mp4_path = Path(mp4_path)
    if not mp4_path.is_file():
        raise FileNotFoundError(f"MP4 file not found: {mp4_path}")
    raw = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(mp4_path),
        ],
        text=True,
    )
    parsed = json.loads(raw)
    v_streams = [s for s in parsed.get("streams", []) if s.get("codec_type") == "video"]
    if not v_streams:
        raise AssertionError(f"No video stream found in {mp4_path}")
    vs = v_streams[0]
    r_fps = vs.get("r_frame_rate", "0/1")
    num, _, den = str(r_fps).partition("/")
    fps = float(num) / float(den) if den and float(den) != 0 else float(num or 0.0)
    nb_frames_raw = vs.get("nb_frames")
    nb_frames = int(nb_frames_raw) if nb_frames_raw is not None else None
    duration_s = float(vs.get("duration") or parsed.get("format", {}).get("duration") or 0.0)
    return {
        "width": int(vs.get("width", 0)),
        "height": int(vs.get("height", 0)),
        "fps": fps,
        "nb_frames": nb_frames,
        "duration_s": duration_s,
        "codec_name": vs.get("codec_name", ""),
        "pix_fmt": vs.get("pix_fmt", ""),
        "raw": parsed,
    }


def assert_no_legacy_fields(obj: Any) -> None:
    """Recursively verify zero abolished legacy fields/labels (long_pause, back_and_forth)."""
    checked_label_keys = (
        "type",
        "label_id",
        "segment_type",
        "id",
        "label",
        "merged_into",
        "target_label_id",
        "suggested_label",
    )

    def _walk(node: Any, path: str) -> None:
        if hasattr(node, "model_dump"):
            node = node.model_dump(mode="json")
        if isinstance(node, dict):
            for k, v in node.items():
                if k in FORBIDDEN_LEGACY_FIELDS:
                    raise AssertionError(
                        f"Forbidden legacy key {k!r} found at {path}.{k}"
                    )
                if k in checked_label_keys and v in FORBIDDEN_LEGACY_FIELDS:
                    raise AssertionError(
                        f"Forbidden legacy label value {v!r} found at {path}.{k}"
                    )
                _walk(v, f"{path}.{k}")
        elif isinstance(node, (list, tuple)):
            for idx, item in enumerate(node):
                _walk(item, f"{path}[{idx}]")
        elif isinstance(node, str) and node in FORBIDDEN_LEGACY_FIELDS:
            raise AssertionError(
                f"Forbidden legacy string leaf {node!r} found at {path}"
            )

    _walk(obj, "$")


def assert_81d_lerobot_v3_valid(
    export_dir: Path,
    *,
    min_cameras: int = 2,
    require_nonempty_subtasks: bool = True,
) -> dict[str, Any]:
    """Validate LeRobot v3.0 export bundle structure, 81D Parquet columns, and CFR MP4s."""
    export_dir = Path(export_dir)
    info_path = export_dir / "meta" / "info.json"
    assert info_path.is_file(), f"Missing meta/info.json in {export_dir}"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    assert str(info.get("codebase_version", "")).startswith("v3"), info

    for req_meta in ("stats.json", "tasks.parquet", "subtasks.parquet"):
        assert (export_dir / "meta" / req_meta).is_file(), f"Missing meta/{req_meta}"

    sub_tbl = pq.read_table(export_dir / "meta" / "subtasks.parquet")
    if require_nonempty_subtasks:
        assert sub_tbl.num_rows >= 1, (
            f"Expected nonempty meta/subtasks.parquet, got {sub_tbl.num_rows} rows"
        )

    ep_parquets = sorted((export_dir / "meta" / "episodes").rglob("*.parquet"))
    assert ep_parquets, f"Missing meta/episodes/**/*.parquet in {export_dir}"

    data_parquets = sorted((export_dir / "data").rglob("*.parquet"))
    assert data_parquets, f"Missing data/**/*.parquet in {export_dir}"

    total_rows = 0
    required_cols = {
        "observation.state",
        "action",
        "action_loss_mask",
        "is_tail",
        "subtask_index",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }
    for pq_path in data_parquets:
        tbl = pq.read_table(pq_path)
        missing = required_cols - set(tbl.column_names)
        assert not missing, f"Missing columns in {pq_path}: {missing}"
        total_rows += tbl.num_rows
        if tbl.num_rows > 0:
            for row_idx, s_row in enumerate(tbl["observation.state"].to_pylist()):
                assert len(s_row) == 81, (
                    f"Row {row_idx} in {pq_path}: expected 81D observation.state, got {len(s_row)}"
                )
            for row_idx, a_row in enumerate(tbl["action"].to_pylist()):
                assert len(a_row) == 81, (
                    f"Row {row_idx} in {pq_path}: expected 81D action, got {len(a_row)}"
                )
            for row_idx, m_row in enumerate(tbl["action_loss_mask"].to_pylist()):
                assert len(m_row) == 2, (
                    f"Row {row_idx} in {pq_path}: expected 2D action_loss_mask, got {len(m_row)}"
                )

    assert int(info.get("total_frames", -1)) == total_rows, (
        f"info.json total_frames ({info.get('total_frames')}) != Parquet rows ({total_rows})"
    )

    mp4_files = sorted((export_dir / "videos").rglob("*.mp4"))
    assert mp4_files, f"Missing videos/**/*.mp4 in {export_dir}"
    video_infos: dict[str, dict[str, Any]] = {}
    cam_dirs = sorted({p.parents[1] for p in mp4_files})
    assert len(cam_dirs) >= min_cameras, (
        f"Expected at least {min_cameras} camera directories, got {len(cam_dirs)}: "
        f"{[c.name for c in cam_dirs]}"
    )
    for cdir in cam_dirs:
        cam_mp4s = sorted(cdir.rglob("*.mp4"))
        cam_frames = 0
        for mp4 in cam_mp4s:
            vinfo = ffprobe_video_info(mp4)
            assert vinfo["width"] > 0 and vinfo["height"] > 0
            assert vinfo["nb_frames"] is not None and vinfo["nb_frames"] > 0
            cam_frames += int(vinfo["nb_frames"])
            video_infos[str(mp4.relative_to(export_dir))] = vinfo
        assert cam_frames == total_rows, (
            f"Camera {cdir.name} frames ({cam_frames}) != Parquet rows ({total_rows})"
        )

    return {
        "total_rows": total_rows,
        "data_files": [str(p.relative_to(export_dir)) for p in data_parquets],
        "videos": video_infos,
        "info": info,
    }


def snapshot_tree_mtimes(
    path: Path, *, exclude_dirs: Iterable[str] = ()
) -> dict[str, tuple[int, int]]:
    """Snapshot ``{relative_path: (mtime_ns, size)}`` for read-only verification."""
    root = Path(path)
    if not root.exists():
        return {}
    if root.is_file():
        st = root.stat()
        return {root.name: (st.st_mtime_ns, st.st_size)}
    excluded = frozenset(exclude_dirs)
    snap: dict[str, tuple[int, int]] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root)
            if excluded and any(part in excluded for part in rel.parts[:-1]):
                continue
            st = p.stat()
            snap[str(rel)] = (st.st_mtime_ns, st.st_size)
    return snap
