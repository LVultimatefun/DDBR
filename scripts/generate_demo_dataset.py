#!/usr/bin/env python3
"""Generate a synthetic multi-format demo dataset for robo_vlm_qc quickstart.

Creates two ready-to-ingest datasets under `--out`:
1. `demo_pick_and_place/` (`video_dir` adapter):
   - 2 multi-camera episodes (`ep_0001`, `ep_0002`), 3.0s @ 30 FPS each
   - Synchronized `main.mp4`, `left_wrist.mp4`, `right_wrist.mp4`, and `episode.json`
2. `demo_lerobot_handover/` (`lerobot` v3 adapter):
   - 1 bimanual handover episode (`episode_000000`), 3.0s @ 30 FPS
   - Canonical 81-D kinematics in `data/chunk-000/file-000.parquet`
   - 3 synchronized camera MP4 streams (`main`, `left_wrist`, `right_wrist`)
   - Complete LeRobot v3 metadata (`meta/info.json`, `meta/tasks.parquet`, `meta/episodes/`)

Example:
    python scripts/generate_demo_dataset.py --out ./demo_data
    robo-vlm-qc -w ./workspace ingest --data ./demo_data
    robo-vlm-qc -w ./workspace serve --no-auth
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CANONICAL_81D_NAMES: list[str] = [
    "chassis_dx",
    "chassis_dy",
    "chassis_dyaw",
    "torso_0",
    "torso_1",
    "torso_2",
    "torso_3",
    *(f"arm_left_{i}" for i in range(7)),
    *(f"arm_right_{i}" for i in range(7)),
    "head_0",
    "head_1",
    "ee_left_x",
    "ee_left_y",
    "ee_left_z",
    "ee_left_qw",
    "ee_left_qx",
    "ee_left_qy",
    "ee_left_qz",
    "ee_right_x",
    "ee_right_y",
    "ee_right_z",
    "ee_right_qw",
    "ee_right_qx",
    "ee_right_qy",
    "ee_right_qz",
    *(f"hand_left_{i}" for i in range(22)),
    *(f"hand_right_{i}" for i in range(22)),
]

assert len(CANONICAL_81D_NAMES) == 81


def _write_mp4(path: Path, frames_bgr: list[np.ndarray], fps: float) -> None:
    if not frames_bgr:
        raise ValueError("Cannot write empty video")
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open cv2.VideoWriter for {path}")
    try:
        for frame in frames_bgr:
            writer.write(frame)
    finally:
        writer.release()


def _draw_tabletop_frame(
    *,
    width: int,
    height: int,
    t_s: float,
    duration_s: float,
    camera: str,
    title: str,
    left_xy: tuple[float, float],
    right_xy: tuple[float, float],
    obj_xy: tuple[float, float],
    obj_color: tuple[int, int, int],
    phase_label: str,
) -> np.ndarray:
    """Render a synthetic BGR frame representing a bimanual tabletop workspace."""
    img = np.full((height, width, 3), (38, 42, 48), dtype=np.uint8)

    # Tabletop surface
    table_top = int(height * 0.22)
    cv2.rectangle(img, (0, table_top), (width, height), (56, 62, 70), -1)

    # Perspective reference grid
    for gx in range(1, 6):
        x = int(width * gx / 6.0)
        cv2.line(img, (x, table_top), (x, height), (70, 76, 86), 1)
    for gy in range(1, 4):
        y = table_top + int((height - table_top) * gy / 4.0)
        cv2.line(img, (0, y), (width, y), (70, 76, 86), 1)

    # Start tray (left) & goal bin (right)
    tray_pt1 = (int(width * 0.12), int(height * 0.56))
    tray_pt2 = (int(width * 0.34), int(height * 0.84))
    bin_pt1 = (int(width * 0.66), int(height * 0.54))
    bin_pt2 = (int(width * 0.88), int(height * 0.84))
    cv2.rectangle(img, tray_pt1, tray_pt2, (120, 88, 52), -1)
    cv2.rectangle(img, tray_pt1, tray_pt2, (190, 150, 95), 2)
    cv2.rectangle(img, bin_pt1, bin_pt2, (58, 108, 72), -1)
    cv2.rectangle(img, bin_pt1, bin_pt2, (110, 200, 130), 2)

    if camera == "left_wrist":
        # Close-up left wrist viewpoint centered near left end-effector
        cx = int(width * (0.5 + (obj_xy[0] - left_xy[0]) * 1.4))
        cy = int(height * (0.55 + (obj_xy[1] - left_xy[1]) * 1.4))
        cv2.circle(img, (int(width * 0.5), int(height * 0.78)), int(min(width, height) * 0.16), (210, 150, 70), 3)
        cv2.rectangle(img, (cx - 26, cy - 26), (cx + 26, cy + 26), obj_color, -1)
        cv2.rectangle(img, (cx - 26, cy - 26), (cx + 26, cy + 26), (245, 245, 245), 2)
    elif camera == "right_wrist":
        # Close-up right wrist viewpoint centered near right end-effector
        cx = int(width * (0.5 + (obj_xy[0] - right_xy[0]) * 1.4))
        cy = int(height * (0.55 + (obj_xy[1] - right_xy[1]) * 1.4))
        cv2.circle(img, (int(width * 0.5), int(height * 0.78)), int(min(width, height) * 0.16), (95, 195, 235), 3)
        cv2.rectangle(img, (cx - 26, cy - 26), (cx + 26, cy + 26), obj_color, -1)
        cv2.rectangle(img, (cx - 26, cy - 26), (cx + 26, cy + 26), (245, 245, 245), 2)
    else:
        # Main overhead/ego view showing both arms and object
        ox, oy = int(width * obj_xy[0]), int(height * obj_xy[1])
        lx, ly = int(width * left_xy[0]), int(height * left_xy[1])
        rx, ry = int(width * right_xy[0]), int(height * right_xy[1])

        # Target block
        cv2.rectangle(img, (ox - 14, oy - 14), (ox + 14, oy + 14), obj_color, -1)
        cv2.rectangle(img, (ox - 14, oy - 14), (ox + 14, oy + 14), (245, 245, 245), 1)

        # Left arm linkage + gripper
        cv2.line(img, (int(width * 0.08), int(height * 0.24)), (lx, ly), (220, 160, 75), 6)
        cv2.circle(img, (lx, ly), 10, (240, 195, 100), -1)
        cv2.circle(img, (lx, ly), 10, (255, 255, 255), 1)

        # Right arm linkage + gripper
        cv2.line(img, (int(width * 0.92), int(height * 0.24)), (rx, ry), (85, 185, 235), 6)
        cv2.circle(img, (rx, ry), 10, (130, 215, 250), -1)
        cv2.circle(img, (rx, ry), 10, (255, 255, 255), 1)

    # Top HUD bar
    cv2.rectangle(img, (0, 0), (width, 28), (22, 26, 32), -1)
    hud_text = f"{title} | {camera.upper()} | t={t_s:04.2f}s"
    cv2.putText(
        img,
        hud_text,
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (230, 236, 244),
        1,
        cv2.LINE_AA,
    )

    # Bottom phase & progress bar
    prog_w = int(width * max(0.0, min(1.0, t_s / max(duration_s, 1e-6))))
    cv2.rectangle(img, (0, height - 18), (width, height), (24, 28, 34), -1)
    cv2.rectangle(img, (0, height - 4), (prog_w, height), (80, 195, 130), -1)
    cv2.putText(
        img,
        phase_label,
        (8, height - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (190, 200, 215),
        1,
        cv2.LINE_AA,
    )
    return img


def _generate_video_dir_dataset(
    out_root: Path,
    *,
    fps: int,
    duration_s: float,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    """Create 2 synthetic multi-camera pick_and_place episodes for `video_dir`."""
    ds_root = out_root / "demo_pick_and_place"
    n_frames = max(1, int(round(duration_s * fps)))
    episodes_spec = [
        ("ep_0001", "smooth_pick_place", True, (60, 145, 245)),
        ("ep_0002", "retry_pick_place", True, (75, 110, 235)),
    ]
    created: list[dict[str, Any]] = []

    for stem, variant, recorded_success, obj_color in episodes_spec:
        ep_dir = ds_root / stem
        ep_dir.mkdir(parents=True, exist_ok=True)
        cam_frames: dict[str, list[np.ndarray]] = {
            "main": [],
            "left_wrist": [],
            "right_wrist": [],
        }

        for i in range(n_frames):
            t_s = i / float(fps)
            u = t_s / max(duration_s, 1e-6)

            if variant == "smooth_pick_place":
                # Left hand steadies tray; right hand picks at 0.25..0.55 and places at 0.55..0.85
                left_xy = (0.20, 0.62 + 0.02 * math.sin(u * math.pi))
                if u < 0.25:
                    a = u / 0.25
                    right_xy = (0.76 - 0.52 * a, 0.36 + 0.32 * a)
                    obj_xy = (0.24, 0.70)
                    phase = "stage: approach_grasp"
                elif u < 0.75:
                    a = (u - 0.25) / 0.50
                    x = 0.24 + 0.52 * a
                    y = 0.70 - 0.16 * math.sin(a * math.pi)
                    right_xy = (x, y)
                    obj_xy = (x, y + 0.02)
                    phase = "stage: transport_to_bin"
                else:
                    a = (u - 0.75) / 0.25
                    right_xy = (0.76 + 0.06 * a, 0.70 - 0.26 * a)
                    obj_xy = (0.76, 0.70)
                    phase = "stage: release_retract"
            else:
                # Initial boundary idle (0..0.20), brief missed-grasp retry (0.35..0.55), then place
                left_xy = (0.20, 0.64)
                if u < 0.20:
                    right_xy = (0.78, 0.34)
                    obj_xy = (0.24, 0.70)
                    phase = "qc: boundary_idle"
                elif u < 0.35:
                    a = (u - 0.20) / 0.15
                    right_xy = (0.78 - 0.52 * a, 0.34 + 0.32 * a)
                    obj_xy = (0.24, 0.70)
                    phase = "stage: approach_object"
                elif u < 0.55:
                    a = (u - 0.35) / 0.20
                    wobble = 0.05 * math.sin(a * 3.0 * math.pi)
                    right_xy = (0.26 + wobble, 0.64 - abs(wobble))
                    obj_xy = (0.24 + 0.015 * math.sin(a * 2.0 * math.pi), 0.70)
                    phase = "qc: missed_grasp_retry"
                elif u < 0.85:
                    a = (u - 0.55) / 0.30
                    x = 0.24 + 0.52 * a
                    y = 0.70 - 0.14 * math.sin(a * math.pi)
                    right_xy = (x, y)
                    obj_xy = (x, y + 0.02)
                    phase = "stage: recovery_transport"
                else:
                    a = (u - 0.85) / 0.15
                    right_xy = (0.76, 0.70 - 0.24 * a)
                    obj_xy = (0.76, 0.70)
                    phase = "stage: place_complete"

            for cam in cam_frames:
                cam_frames[cam].append(
                    _draw_tabletop_frame(
                        width=width,
                        height=height,
                        t_s=t_s,
                        duration_s=duration_s,
                        camera=cam,
                        title=f"pick_and_place/{stem}",
                        left_xy=left_xy,
                        right_xy=right_xy,
                        obj_xy=obj_xy,
                        obj_color=obj_color,
                        phase_label=phase,
                    )
                )

        for cam, frames in cam_frames.items():
            _write_mp4(ep_dir / f"{cam}.mp4", frames, fps=fps)

        (ep_dir / "episode.json").write_text(
            json.dumps(
                {
                    "task_id": "pick_and_place",
                    "recorded_success": recorded_success,
                    "description": f"Synthetic demo episode ({variant})",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        created.append(
            {
                "episode_id": f"demo_pick_and_place/{stem}",
                "adapter_type": "video_dir",
                "path": str(ep_dir),
                "frames": n_frames,
                "fps": fps,
                "cameras": sorted(cam_frames.keys()),
            }
        )
    return created


def _generate_lerobot_v3_dataset(
    out_root: Path,
    *,
    fps: int,
    duration_s: float,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Create 1 synthetic LeRobot v3 bimanual handover episode with 81-D kinematics + MP4s."""
    ds_root = out_root / "demo_lerobot_handover"
    n_frames = max(1, int(round(duration_s * fps)))
    video_keys = (
        "observation.images.main",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    )

    states = np.zeros((n_frames, 81), dtype=np.float32)
    col = {nm: idx for idx, nm in enumerate(CANONICAL_81D_NAMES)}

    cam_frames: dict[str, list[np.ndarray]] = {
        "main": [],
        "left_wrist": [],
        "right_wrist": [],
    }

    for i in range(n_frames):
        t_s = i / float(fps)
        u = t_s / max(duration_s, 1e-6)

        if u < 0.45:
            a = u / 0.45
            left_xy = (0.24 + 0.24 * a, 0.70 - 0.16 * a)
            right_xy = (0.76 - 0.22 * a, 0.62 - 0.08 * a)
            obj_xy = (left_xy[0], left_xy[1] + 0.02)
            phase = "stage: left_lift_to_center"
        elif u < 0.65:
            a = (u - 0.45) / 0.20
            left_xy = (0.48 - 0.02 * a, 0.54)
            right_xy = (0.54 - 0.02 * a, 0.54)
            obj_xy = (0.50 + 0.02 * a, 0.55)
            phase = "stage: bimanual_handover"
        else:
            a = (u - 0.65) / 0.35
            left_xy = (0.46 - 0.22 * a, 0.54 + 0.04 * a)
            right_xy = (0.52 + 0.24 * a, 0.54 + 0.16 * a)
            obj_xy = (right_xy[0], right_xy[1] + 0.02)
            phase = "stage: right_place_to_bin"

        # Populate smooth 81-D kinematics state
        states[i, col["torso_0"]] = 0.05 * math.sin(u * math.pi)
        for j in range(7):
            states[i, col[f"arm_left_{j}"]] = 0.2 * math.sin(u * math.pi + j * 0.25)
            states[i, col[f"arm_right_{j}"]] = -0.2 * math.sin(u * math.pi + j * 0.25)

        # End-effector positions (meters) & unit quaternions (w, x, y, z)
        states[i, col["ee_left_x"]] = 0.25 + 0.20 * left_xy[0]
        states[i, col["ee_left_y"]] = 0.30 - 0.35 * left_xy[0]
        states[i, col["ee_left_z"]] = 0.20 + 0.25 * (1.0 - left_xy[1])
        yaw_l = 0.15 * math.sin(u * math.pi)
        states[i, col["ee_left_qw"]] = math.cos(yaw_l / 2.0)
        states[i, col["ee_left_qz"]] = math.sin(yaw_l / 2.0)

        states[i, col["ee_right_x"]] = 0.25 + 0.20 * right_xy[0]
        states[i, col["ee_right_y"]] = -0.30 + 0.35 * (1.0 - right_xy[0])
        states[i, col["ee_right_z"]] = 0.20 + 0.25 * (1.0 - right_xy[1])
        yaw_r = -0.15 * math.sin(u * math.pi)
        states[i, col["ee_right_qw"]] = math.cos(yaw_r / 2.0)
        states[i, col["ee_right_qz"]] = math.sin(yaw_r / 2.0)

        # Gripper / dexterous finger closure profile
        left_grip = 0.8 if u < 0.60 else 0.15
        right_grip = 0.15 if u < 0.50 else 0.8
        for f in range(22):
            states[i, col[f"hand_left_{f}"]] = left_grip * (0.6 + 0.02 * f)
            states[i, col[f"hand_right_{f}"]] = right_grip * (0.6 + 0.02 * f)

        for cam in cam_frames:
            cam_frames[cam].append(
                _draw_tabletop_frame(
                    width=width,
                    height=height,
                    t_s=t_s,
                    duration_s=duration_s,
                    camera=cam,
                    title="bimanual_handover/ep0",
                    left_xy=left_xy,
                    right_xy=right_xy,
                    obj_xy=obj_xy,
                    obj_color=(95, 200, 110),
                    phase_label=phase,
                )
            )

    # Next-step action target
    actions = np.vstack([states[1:], states[-1:]])

    # Write chunked MP4 streams
    for vk in video_keys:
        cam_short = vk.split(".")[-1]
        vid_path = ds_root / "videos" / vk / "chunk-000" / "file-000.mp4"
        _write_mp4(vid_path, cam_frames[cam_short], fps=fps)

    # Write data Parquet
    data_pq = ds_root / "data" / "chunk-000" / "file-000.parquet"
    data_pq.parent.mkdir(parents=True, exist_ok=True)
    timestamps = (np.arange(n_frames, dtype=np.float32) / float(fps)).tolist()
    frame_indices = list(range(n_frames))
    data_table = pa.table(
        {
            "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32())),
            "action": pa.array(actions.tolist(), type=pa.list_(pa.float32())),
            "timestamp": pa.array(timestamps, type=pa.float32()),
            "frame_index": pa.array(frame_indices, type=pa.int64()),
            "episode_index": pa.array([0] * n_frames, type=pa.int64()),
            "index": pa.array(frame_indices, type=pa.int64()),
            "task_index": pa.array([0] * n_frames, type=pa.int64()),
        }
    )
    pq.write_table(data_table, data_pq)

    # Write meta/tasks.parquet
    meta_dir = ds_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "task_index": pa.array([0], type=pa.int64()),
                "task": pa.array(["bimanual_handover"], type=pa.string()),
            }
        ),
        meta_dir / "tasks.parquet",
    )

    # Write meta/episodes/chunk-000/file-000.parquet
    ep_meta_pq = meta_dir / "episodes" / "chunk-000" / "file-000.parquet"
    ep_meta_pq.parent.mkdir(parents=True, exist_ok=True)
    ep_cols: dict[str, Any] = {
        "episode_index": pa.array([0], type=pa.int64()),
        "length": pa.array([n_frames], type=pa.int64()),
        "tasks": pa.array([["bimanual_handover"]], type=pa.list_(pa.string())),
        "task_index": pa.array([0], type=pa.int64()),
        "data/chunk_index": pa.array([0], type=pa.int64()),
        "data/file_index": pa.array([0], type=pa.int64()),
        "dataset_from_index": pa.array([0], type=pa.int64()),
        "dataset_to_index": pa.array([n_frames], type=pa.int64()),
    }
    for vk in video_keys:
        ep_cols[f"videos/{vk}/chunk_index"] = pa.array([0], type=pa.int64())
        ep_cols[f"videos/{vk}/file_index"] = pa.array([0], type=pa.int64())
        ep_cols[f"videos/{vk}/from_timestamp"] = pa.array([0.0], type=pa.float64())
        ep_cols[f"videos/{vk}/to_timestamp"] = pa.array([float(n_frames / fps)], type=pa.float64())
    pq.write_table(pa.table(ep_cols), ep_meta_pq)

    # Write meta/info.json
    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [81],
            "names": CANONICAL_81D_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": [81],
            "names": CANONICAL_81D_NAMES,
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for vk in video_keys:
        features[vk] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": float(fps),
                "video.height": height,
                "video.width": width,
                "video.channels": 3,
                "video.codec": "mp4v",
            },
        }

    info = {
        "codebase_version": "v3.0",
        "robot_type": "bimanual_robot",
        "total_episodes": 1,
        "total_frames": n_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": "0:1"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    (meta_dir / "info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )

    return {
        "episode_id": "demo_lerobot_handover/episode_000000",
        "adapter_type": "lerobot",
        "path": str(ds_root),
        "frames": n_frames,
        "fps": fps,
        "cameras": ["main", "left_wrist", "right_wrist"],
        "has_kinematics": True,
        "state_dim": 81,
    }


def _clamp_box_norm(
    cx_norm: float,
    cy_norm: float,
    half_w: float,
    half_h: float,
) -> list[int]:
    x1 = max(0, min(990, int(round((cx_norm - half_w) * 1000.0))))
    y1 = max(0, min(990, int(round((cy_norm - half_h) * 1000.0))))
    x2 = max(x1 + 10, min(1000, int(round((cx_norm + half_w) * 1000.0))))
    y2 = max(y1 + 10, min(1000, int(round((cy_norm + half_h) * 1000.0))))
    return [x1, y1, x2, y2]


def _box_norm_to_px(box_norm: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box_norm
    return [
        int(round(x1 * width / 1000.0)),
        int(round(y1 * height / 1000.0)),
        int(round(x2 * width / 1000.0)),
        int(round(y2 * height / 1000.0)),
    ]


def _sample_episode_positions(
    episode_id: str, u: float
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Deterministic synthetic trajectories matching the rendered video frames."""
    u = max(0.0, min(1.0, float(u)))
    if episode_id.endswith("/ep_0001"):
        left_xy = (0.20, 0.62 + 0.02 * math.sin(u * math.pi))
        if u < 0.25:
            a = u / 0.25
            right_xy = (0.76 - 0.52 * a, 0.36 + 0.32 * a)
            obj_xy = (0.24, 0.70)
        elif u < 0.75:
            a = (u - 0.25) / 0.50
            x = 0.24 + 0.52 * a
            y = 0.70 - 0.16 * math.sin(a * math.pi)
            right_xy = (x, y)
            obj_xy = (x, y + 0.02)
        else:
            a = (u - 0.75) / 0.25
            right_xy = (0.76 + 0.06 * a, 0.70 - 0.26 * a)
            obj_xy = (0.76, 0.70)
        return left_xy, right_xy, obj_xy

    if episode_id.endswith("/ep_0002"):
        left_xy = (0.20, 0.64)
        if u < 0.20:
            right_xy = (0.78, 0.34)
            obj_xy = (0.24, 0.70)
        elif u < 0.35:
            a = (u - 0.20) / 0.15
            right_xy = (0.78 - 0.52 * a, 0.34 + 0.32 * a)
            obj_xy = (0.24, 0.70)
        elif u < 0.55:
            a = (u - 0.35) / 0.20
            wobble = 0.05 * math.sin(a * 3.0 * math.pi)
            right_xy = (0.26 + wobble, 0.64 - abs(wobble))
            obj_xy = (0.24 + 0.015 * math.sin(a * 2.0 * math.pi), 0.70)
        elif u < 0.85:
            a = (u - 0.55) / 0.30
            x = 0.24 + 0.52 * a
            y = 0.70 - 0.14 * math.sin(a * math.pi)
            right_xy = (x, y)
            obj_xy = (x, y + 0.02)
        else:
            a = (u - 0.85) / 0.15
            right_xy = (0.76, 0.70 - 0.24 * a)
            obj_xy = (0.76, 0.70)
        return left_xy, right_xy, obj_xy

    # demo_lerobot_handover/episode_000000
    if u < 0.45:
        a = u / 0.45
        left_xy = (0.24 + 0.24 * a, 0.70 - 0.16 * a)
        right_xy = (0.76 - 0.22 * a, 0.62 - 0.08 * a)
        obj_xy = (left_xy[0], left_xy[1] + 0.02)
    elif u < 0.65:
        a = (u - 0.45) / 0.20
        left_xy = (0.48 - 0.02 * a, 0.54)
        right_xy = (0.54 - 0.02 * a, 0.54)
        obj_xy = (0.50 + 0.02 * a, 0.55)
    else:
        a = (u - 0.65) / 0.35
        left_xy = (0.46 - 0.22 * a, 0.54 + 0.04 * a)
        right_xy = (0.52 + 0.24 * a, 0.54 + 0.16 * a)
        obj_xy = (right_xy[0], right_xy[1] + 0.02)
    return left_xy, right_xy, obj_xy


def _seed_sample_runs(out_root: Path, workspace: Path) -> dict[str, Any]:
    """Ingest generated demo datasets into `workspace` and seed offline VLM runs."""
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.io import auto_detect, get_adapter
    from robo_vlm_qc.models.annotation import (
        ArmSpan,
        ArmTimelines,
        ConfigFingerprintSnapshot,
        CriterionResult,
        GroundingKeyframe,
        GroundingResult,
        GroundingTrack,
        GuardrailMetrics,
        GuardrailReport,
        MacroStage,
        QcSegment,
        StageUsage,
        SuccessVerdict,
        UsageSummary,
        VisualEvidence,
        VlmRunArtifact,
    )
    from robo_vlm_qc.store import Repository

    ws_root = Path(workspace).resolve()
    ws_root.mkdir(parents=True, exist_ok=True)
    cfg_ws = ws_root if (ws_root / "configs_override").is_dir() else None
    cfg = ConfigManager(workspace_dir=cfg_ws)
    repo = Repository(ws_root)

    has_stage_cached = "cached_tokens" in StageUsage.model_fields
    has_summary_cached = "total_cached_tokens" in UsageSummary.model_fields

    def _make_stage_usage(
        stage: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int,
        elapsed_s: float,
        estimated_cost_usd: float,
    ) -> StageUsage:
        kwargs: dict[str, Any] = {
            "stage": stage,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "elapsed_s": elapsed_s,
            "estimated_cost_usd": estimated_cost_usd,
            "cache_hit": False,
        }
        if has_stage_cached:
            kwargs["cached_tokens"] = cached_tokens
        return StageUsage(**kwargs)

    def _make_usage(stages: list[StageUsage], cached_total: int) -> UsageSummary:
        kwargs: dict[str, Any] = {
            "stages": stages,
            "total_tokens": sum(s.total_tokens for s in stages),
            "total_elapsed_s": round(sum(s.elapsed_s for s in stages), 3),
            "total_cost_usd": round(sum(s.estimated_cost_usd for s in stages), 6),
        }
        if has_summary_cached:
            kwargs["total_cached_tokens"] = cached_total
        return UsageSummary(**kwargs)

    ingested: list[dict[str, Any]] = []
    seeded_runs: list[dict[str, Any]] = []

    try:
        handles = auto_detect(out_root, task_patterns=cfg.task_patterns())
        for handle in handles:
            ad = get_adapter(handle.adapter_type)
            with ad.open(handle) as view:
                meta = view.meta
                hint = meta.extra_meta.get("h5_task_name") or meta.extra_meta.get(
                    "ep_task_str"
                )
                resolved_task = cfg.resolve_task(
                    meta.dataset, str(hint) if hint else None
                ).task_id
                if resolved_task != meta.task_id:
                    meta = meta.model_copy(update={"task_id": resolved_task})
                ep_id, outcome = repo.ingest_episode(meta)
                ingested.append(
                    {
                        "episode_id": ep_id,
                        "task_id": meta.task_id,
                        "outcome": outcome,
                    }
                )

            run_id = f"demo_sample_{meta.dataset}_{meta.stem}"
            existing_runs = {r["run_id"] for r in repo.list_vlm_runs(meta.episode_id)}
            if run_id in existing_runs:
                seeded_runs.append(
                    {
                        "episode_id": meta.episode_id,
                        "run_id": run_id,
                        "outcome": "already_exists",
                    }
                )
                continue

            dur = max(0.5, float(meta.duration_s))
            fps = max(1.0, float(meta.fps))
            main_cam = str(meta.camera_roles.get("main") or meta.cameras[0].name)
            main_spec = next(
                (c for c in meta.cameras if c.name == main_cam), meta.cameras[0]
            )
            ref_w, ref_h = int(main_spec.width), int(main_spec.height)

            track_fps = min(15.0, fps)
            n_dense = max(5, int(round(dur * track_fps)))
            dense_times = [
                round(min(dur, i * dur / float(max(1, n_dense - 1))), 4)
                for i in range(n_dense)
            ]
            kf_indices = sorted(
                {
                    0,
                    n_dense // 4,
                    n_dense // 2,
                    (3 * n_dense) // 4,
                    n_dense - 1,
                }
            )
            keyframes_s = [dense_times[i] for i in kf_indices]

            left_boxes: list[list[int]] = []
            right_boxes: list[list[int]] = []
            obj_boxes: list[list[int]] = []
            for t in dense_times:
                l_xy, r_xy, o_xy = _sample_episode_positions(
                    meta.episode_id, t / dur
                )
                left_boxes.append(_clamp_box_norm(l_xy[0], l_xy[1], 0.045, 0.06))
                right_boxes.append(_clamp_box_norm(r_xy[0], r_xy[1], 0.045, 0.06))
                obj_boxes.append(_clamp_box_norm(o_xy[0], o_xy[1], 0.055, 0.075))

            def _build_dynamic_track(
                oid: str,
                name: str,
                boxes: list[list[int]],
                tag_pos: str = "top",
                state_txt: str = "active",
            ) -> GroundingTrack:
                kfs = [
                    GroundingKeyframe(
                        row=int(round(dense_times[idx] * fps)),
                        time_s=dense_times[idx],
                        visible=True,
                        bbox_norm_1000=boxes[idx],
                        bbox_px=_box_norm_to_px(boxes[idx], ref_w, ref_h),
                        state=state_txt,
                        status_kind="good",
                    )
                    for idx in kf_indices
                ]
                return GroundingTrack(
                    object_id=oid,
                    name=name,
                    tag_pos=tag_pos,  # type: ignore[arg-type]
                    is_static_container=False,
                    keyframes=kfs,
                    tracking_mode="bidir_flow_ncc",
                    dense_boxes_norm_1000=boxes,
                    dense_flags=[1] * n_dense,
                    dense_conf_pct=[94] * n_dense,
                )

            def _build_static_track(
                oid: str,
                name: str,
                box: list[int],
                tag_pos: str = "bottom",
            ) -> GroundingTrack:
                kfs = [
                    GroundingKeyframe(
                        row=int(round(dense_times[idx] * fps)),
                        time_s=dense_times[idx],
                        visible=True,
                        bbox_norm_1000=list(box),
                        bbox_px=_box_norm_to_px(box, ref_w, ref_h),
                        state="stationary",
                        status_kind="neutral",
                    )
                    for idx in kf_indices
                ]
                return GroundingTrack(
                    object_id=oid,
                    name=name,
                    tag_pos=tag_pos,  # type: ignore[arg-type]
                    is_static_container=True,
                    keyframes=kfs,
                    tracking_mode="static_locked",
                    dense_boxes_norm_1000=[list(box) for _ in range(n_dense)],
                    dense_flags=[1] * n_dense,
                    dense_conf_pct=[99] * n_dense,
                )

            t1 = round(dur * 0.25, 3)
            t2 = round(dur * 0.75, 3)

            if meta.episode_id.endswith("/ep_0001"):
                tracks = {
                    "target_object": _build_dynamic_track(
                        "target_object", "Target Object", obj_boxes, "top", "transported"
                    ),
                    "target_bin": _build_static_track(
                        "target_bin", "Target Bin", [660, 540, 880, 840], "bottom"
                    ),
                    "right_gripper": _build_dynamic_track(
                        "right_gripper", "Right Gripper", right_boxes, "top", "grasping"
                    ),
                    "left_gripper": _build_dynamic_track(
                        "left_gripper", "Left Gripper", left_boxes, "bottom", "bracing"
                    ),
                }
                macro_stages = [
                    MacroStage(
                        stage_id="S1",
                        t_start=0.0,
                        t_end=t1,
                        label="Approach & Grasp",
                        label_en="approach_grasp",
                        instruction="Reach toward and grasp the target block on the left tray",
                        summary="Right arm approaches smoothly while left arm braces the tray.",
                        color_rgb=[59, 130, 246],
                    ),
                    MacroStage(
                        stage_id="S2",
                        t_start=t1,
                        t_end=t2,
                        label="Transport to Bin",
                        label_en="transport_to_bin",
                        instruction="Lift and transport the target block to the right goal bin",
                        summary="Smooth arc transport across the tabletop without slip.",
                        color_rgb=[16, 185, 129],
                    ),
                    MacroStage(
                        stage_id="S3",
                        t_start=t2,
                        t_end=round(dur, 3),
                        label="Release & Retract",
                        label_en="release_retract",
                        instruction="Seat the target block in the bin and retract cleanly",
                        summary="Clean placement inside bin boundaries followed by retraction.",
                        color_rgb=[139, 92, 246],
                    ),
                ]
                arm_timelines = ArmTimelines(
                    left=[
                        ArmSpan(t_start=0.0, t_end=round(dur, 3), state="hold", desc="Stabilize source tray")
                    ],
                    right=[
                        ArmSpan(t_start=0.0, t_end=t1, state="work", desc="Approach and grasp target"),
                        ArmSpan(t_start=t1, t_end=t2, state="good", desc="Transport block to bin"),
                        ArmSpan(t_start=t2, t_end=round(dur, 3), state="work", desc="Release and retract"),
                    ],
                )
                qc_segments = [
                    QcSegment(
                        seg_id="seg_fine_adjust_1",
                        t_start=round(dur * 0.20, 3),
                        t_end=round(dur * 0.32, 3),
                        label_id="precision_fine_adjust",
                        name="Precision Fine Adjust",
                        training_role="necessary",
                        splice_policy="keep_positive",
                        mask_side=None,
                        affected_arm="right",
                        visual_evidence=VisualEvidence(
                            initial_state="Right gripper descends toward the target block edge.",
                            process_change="Slow monotonic alignment centers jaws cleanly on block.",
                            end_state="Gripper establishes secure grasp and lifts cleanly.",
                            confidence_note="Positive precision skill; keep for policy training.",
                        ),
                        vlm_raw_policy="keep_positive",
                        guardrail=GuardrailReport(gate_type="none", passed=True),
                        confidence=0.94,
                    )
                ]
                success = SuccessVerdict(
                    passed=True,
                    verdict="success",
                    confidence=0.96,
                    criteria=[
                        CriterionResult(
                            criterion="Target object lifted cleanly from starting tray",
                            met=True,
                            satisfied_at_s=t1,
                            evidence="Clean lift-off with zero collision.",
                        ),
                        CriterionResult(
                            criterion="Target object rests stably inside target bin",
                            met=True,
                            satisfied_at_s=t2,
                            evidence="Block seated inside green goal bin.",
                        ),
                    ],
                    summary="Clean single-pass pick and place completed successfully.",
                )
                stage_usages = [
                    _make_stage_usage("s1_grounding", "gpt-4o", 1850, 420, 640, 1.12, 0.008025),
                    _make_stage_usage("s2_macro", "gpt-4o", 1320, 310, 520, 0.86, 0.00575),
                    _make_stage_usage("s3_discover", "gpt-4o", 3100, 540, 1480, 1.48, 0.0113),
                    _make_stage_usage("s4_verify", "gpt-4o", 1920, 360, 960, 0.94, 0.0072),
                    _make_stage_usage("s5_success", "gpt-4o", 1480, 260, 620, 0.72, 0.005525),
                ]
                cached_total = 640 + 520 + 1480 + 960 + 620

            elif meta.episode_id.endswith("/ep_0002"):
                e1 = round(dur * 0.20, 3)
                e2 = round(dur * 0.35, 3)
                e3 = round(dur * 0.55, 3)
                e4 = round(dur * 0.85, 3)
                tracks = {
                    "target_object": _build_dynamic_track(
                        "target_object", "Target Object", obj_boxes, "top", "nudged_then_grasped"
                    ),
                    "target_bin": _build_static_track(
                        "target_bin", "Target Bin", [660, 540, 880, 840], "bottom"
                    ),
                    "right_gripper": _build_dynamic_track(
                        "right_gripper", "Right Gripper", right_boxes, "top", "retry_then_place"
                    ),
                    "left_gripper": _build_dynamic_track(
                        "left_gripper", "Left Gripper", left_boxes, "bottom", "bracing"
                    ),
                }
                macro_stages = [
                    MacroStage(
                        stage_id="S1",
                        t_start=0.0,
                        t_end=e1,
                        label="Leading Idle",
                        label_en="leading_idle",
                        instruction="Stationary waiting period prior to manipulation",
                        summary="Both arms remain motionless at start of episode.",
                        color_rgb=[100, 116, 139],
                    ),
                    MacroStage(
                        stage_id="S2",
                        t_start=e1,
                        t_end=e3,
                        label="Approach & Grasp Retry",
                        label_en="approach_grasp_retry",
                        instruction="Approach block, recover from missed grasp, and secure object",
                        summary="Initial grasp slips slightly; arm re-aligns and succeeds on second attempt.",
                        color_rgb=[245, 158, 11],
                    ),
                    MacroStage(
                        stage_id="S3",
                        t_start=e3,
                        t_end=e4,
                        label="Recovery Transport",
                        label_en="recovery_transport",
                        instruction="Transport recovered block over to goal bin",
                        summary="Right arm smoothly transports block after recovery.",
                        color_rgb=[16, 185, 129],
                    ),
                    MacroStage(
                        stage_id="S4",
                        t_start=e4,
                        t_end=round(dur, 3),
                        label="Place & Complete",
                        label_en="place_complete",
                        instruction="Release block in bin and retract right arm",
                        summary="Block seated cleanly in goal bin.",
                        color_rgb=[139, 92, 246],
                    ),
                ]
                arm_timelines = ArmTimelines(
                    left=[
                        ArmSpan(t_start=0.0, t_end=e1, state="idle", desc="Stationary start"),
                        ArmSpan(t_start=e1, t_end=round(dur, 3), state="hold", desc="Stabilize tray"),
                    ],
                    right=[
                        ArmSpan(t_start=0.0, t_end=e1, state="idle", desc="Leading boundary idle"),
                        ArmSpan(t_start=e1, t_end=e2, state="work", desc="Initial approach"),
                        ArmSpan(t_start=e2, t_end=e3, state="mask", desc="Missed grasp retry (masked)"),
                        ArmSpan(t_start=e3, t_end=round(dur, 3), state="good", desc="Recovery transport and place"),
                    ],
                )
                qc_segments = [
                    QcSegment(
                        seg_id="seg_boundary_idle_1",
                        t_start=0.0,
                        t_end=e1,
                        label_id="boundary_idle",
                        name="Boundary Idle",
                        training_role="badcase",
                        splice_policy="cut_clean",
                        mask_side=None,
                        affected_arm="both",
                        visual_evidence=VisualEvidence(
                            initial_state="Both arms stationary above workspace at t=0.0s.",
                            process_change="Zero displacement across all cameras during leading wait.",
                            end_state="Right arm initiates motion immediately after boundary.",
                            confidence_note="Leading boundary idle; safe to trim with cut_clean.",
                        ),
                        vlm_raw_policy="cut_clean",
                        guardrail=GuardrailReport(
                            gate_type="visual_gate",
                            passed=True,
                            downgraded=False,
                            metrics=GuardrailMetrics(visual_ssim=0.992),
                        ),
                        confidence=0.97,
                    ),
                    QcSegment(
                        seg_id="seg_missed_grasp_retry_1",
                        t_start=e2,
                        t_end=e3,
                        label_id="missed_grasp_retry",
                        name="Missed Grasp Retry",
                        training_role="recovery",
                        splice_policy="mask_loss",
                        mask_side="right",
                        affected_arm="right",
                        visual_evidence=VisualEvidence(
                            initial_state="Right gripper closes near block edge without securing it.",
                            process_change="Arm wobbles, nudges block slightly, re-opens, and re-grasps.",
                            end_state="Block is secured in right gripper at a shifted pose.",
                            confidence_note="VLM proposed cut_clean; visual guardrail downgraded to mask_loss due to pose discontinuity.",
                        ),
                        vlm_raw_policy="cut_clean",
                        guardrail=GuardrailReport(
                            gate_type="visual_gate",
                            passed=False,
                            downgraded=True,
                            downgrade_reason=(
                                "Visual discontinuity across splice boundary (SSIM 0.71 < 0.85); "
                                "downgraded cut_clean -> mask_loss"
                            ),
                            metrics=GuardrailMetrics(visual_ssim=0.71),
                        ),
                        confidence=0.91,
                    ),
                ]
                success = SuccessVerdict(
                    passed=True,
                    verdict="success",
                    confidence=0.93,
                    criteria=[
                        CriterionResult(
                            criterion="Target object lifted after grasp recovery",
                            met=True,
                            satisfied_at_s=e3,
                            evidence="Secured on second grasp attempt.",
                        ),
                        CriterionResult(
                            criterion="Target object rests stably inside target bin",
                            met=True,
                            satisfied_at_s=e4,
                            evidence="Placed cleanly into right bin before retracting.",
                        ),
                    ],
                    summary="Task succeeded after trimming leading idle and masking a missed-grasp retry.",
                )
                stage_usages = [
                    _make_stage_usage("s1_grounding", "gpt-4o", 2040, 460, 780, 1.24, 0.008725),
                    _make_stage_usage("s2_macro", "gpt-4o", 1460, 340, 620, 0.91, 0.006275),
                    _make_stage_usage("s3_discover", "gpt-4o", 3620, 690, 1820, 1.72, 0.013675),
                    _make_stage_usage("s4_verify", "gpt-4o", 2480, 510, 1240, 1.18, 0.00975),
                    _make_stage_usage("s5_success", "gpt-4o", 1540, 290, 680, 0.78, 0.0059),
                ]
                cached_total = 780 + 620 + 1820 + 1240 + 680

            else:
                h1 = round(dur * 0.45, 3)
                h2 = round(dur * 0.65, 3)
                tracks = {
                    "handover_object": _build_dynamic_track(
                        "handover_object", "Handover Object", obj_boxes, "top", "handed_over"
                    ),
                    "source_tray": _build_static_track(
                        "source_tray", "Source Tray", [120, 560, 340, 840], "bottom"
                    ),
                    "target_tray": _build_static_track(
                        "target_tray", "Target Tray", [660, 540, 880, 840], "bottom"
                    ),
                    "left_gripper": _build_dynamic_track(
                        "left_gripper", "Left Gripper", left_boxes, "top", "donor_arm"
                    ),
                    "right_gripper": _build_dynamic_track(
                        "right_gripper", "Right Gripper", right_boxes, "top", "receiver_arm"
                    ),
                }
                macro_stages = [
                    MacroStage(
                        stage_id="S1",
                        t_start=0.0,
                        t_end=h1,
                        label="Left Lift to Center",
                        label_en="left_lift_to_center",
                        instruction="Left arm lifts object from source tray to handover zone",
                        summary="Donor arm smoothly presents block at workspace center.",
                        color_rgb=[59, 130, 246],
                    ),
                    MacroStage(
                        stage_id="S2",
                        t_start=h1,
                        t_end=h2,
                        label="Bimanual Handover",
                        label_en="bimanual_handover",
                        instruction="Right arm secures block while left arm stabilizes then releases",
                        summary="Coordinated bimanual transfer with zero slip.",
                        color_rgb=[20, 184, 166],
                    ),
                    MacroStage(
                        stage_id="S3",
                        t_start=h2,
                        t_end=round(dur, 3),
                        label="Right Place to Bin",
                        label_en="right_place_to_bin",
                        instruction="Right arm places object into target tray while left retracts",
                        summary="Clean bimanual completion with 81-D kinematics verified.",
                        color_rgb=[139, 92, 246],
                    ),
                ]
                arm_timelines = ArmTimelines(
                    left=[
                        ArmSpan(t_start=0.0, t_end=h1, state="work", desc="Lift object to center"),
                        ArmSpan(t_start=h1, t_end=h2, state="hold", desc="Stabilize during handover"),
                        ArmSpan(t_start=h2, t_end=round(dur, 3), state="work", desc="Retract to rest"),
                    ],
                    right=[
                        ArmSpan(t_start=0.0, t_end=h1, state="work", desc="Converge toward center"),
                        ArmSpan(t_start=h1, t_end=h2, state="good", desc="Receive and grasp object"),
                        ArmSpan(t_start=h2, t_end=round(dur, 3), state="good", desc="Place in target tray"),
                    ],
                )
                qc_segments = [
                    QcSegment(
                        seg_id="seg_bimanual_hold_1",
                        t_start=h1,
                        t_end=h2,
                        label_id="bimanual_assist_hold",
                        name="Bimanual Assist Hold",
                        training_role="necessary",
                        splice_policy="keep_positive",
                        mask_side=None,
                        affected_arm="left",
                        visual_evidence=VisualEvidence(
                            initial_state="Both arms meet over workspace center holding block.",
                            process_change="Left hand holds object steadily while right hand establishes grasp.",
                            end_state="Right hand takes full load and left hand releases cleanly.",
                            confidence_note="High-value bimanual coordination skill; keep positive.",
                        ),
                        vlm_raw_policy="keep_positive",
                        guardrail=GuardrailReport(gate_type="none", passed=True),
                        confidence=0.95,
                    )
                ]
                success = SuccessVerdict(
                    passed=True,
                    verdict="success",
                    confidence=0.97,
                    criteria=[
                        CriterionResult(
                            criterion="First arm presents object stably to receiving arm",
                            met=True,
                            satisfied_at_s=h1,
                            evidence="Stable center presentation.",
                        ),
                        CriterionResult(
                            criterion="Receiving arm secures grasp before donor releases",
                            met=True,
                            satisfied_at_s=h2,
                            evidence="Zero drop during bimanual transfer.",
                        ),
                        CriterionResult(
                            criterion="Object placed stably onto target tray",
                            met=True,
                            satisfied_at_s=round(dur, 3),
                            evidence="Block seated inside target tray.",
                        ),
                    ],
                    summary="Smooth bimanual handover completed cleanly with synchronized 81-D kinematics.",
                )
                stage_usages = [
                    _make_stage_usage("s1_grounding", "gpt-4o", 2150, 490, 840, 1.31, 0.009225),
                    _make_stage_usage("s2_macro", "gpt-4o", 1390, 320, 580, 0.88, 0.00595),
                    _make_stage_usage("s3_discover", "gpt-4o", 3240, 560, 1560, 1.54, 0.01175),
                    _make_stage_usage("s4_verify", "gpt-4o", 1840, 330, 920, 0.89, 0.00675),
                    _make_stage_usage("s5_success", "gpt-4o", 1510, 270, 640, 0.74, 0.005675),
                ]
                cached_total = 840 + 580 + 1560 + 920 + 640

            artifact = VlmRunArtifact(
                run_id=run_id,
                episode_id=meta.episode_id,
                created_at=time.time(),
                is_eval_sandbox=False,
                config_fingerprint=ConfigFingerprintSnapshot(
                    config_sha256=cfg.fingerprints.config_sha256,
                    model_routing=dict(cfg.settings.get("vlm", {}).get("stage_models", {})),
                    prompt_bundle_version="demo_sample_v1",
                    label_taxonomy_version=str(cfg.labels.version),
                    task_spec_version=str(cfg.get_task(meta.task_id).version),
                ),
                grounding=GroundingResult(
                    schema_version=1,
                    camera_name=main_cam,
                    ref_size=[ref_w, ref_h],
                    keyframes_s=keyframes_s,
                    track_fps=track_fps,
                    dense_times_s=dense_times,
                    camera_motion_detected=False,
                    object_tracks_2d=tracks,
                ),
                macro_stages=macro_stages,
                arm_timelines=arm_timelines,
                qc_segments=qc_segments,
                success=success,
                usage=_make_usage(stage_usages, cached_total),
            )
            repo.save_vlm_run_artifact(artifact)
            seeded_runs.append(
                {
                    "episode_id": meta.episode_id,
                    "run_id": run_id,
                    "status": "vlm_ready",
                    "qc_segments": len(qc_segments),
                    "tokens_total": artifact.usage.total_tokens,
                    "cached_tokens": cached_total,
                    "total_cost_usd": artifact.usage.total_cost_usd,
                    "outcome": "seeded",
                }
            )
    finally:
        repo.close()

    return {
        "workspace": str(ws_root),
        "ingested": ingested,
        "seeded_runs": seeded_runs,
    }


def generate_demo_dataset(
    out_dir: Path,
    *,
    fps: int = 30,
    duration_s: float = 3.0,
    width: int = 320,
    height: int = 240,
    with_sample_run: bool = False,
    workspace: Path | None = None,
) -> dict[str, Any]:
    """Generate synthetic `video_dir` and `lerobot` v3 demo datasets under `out_dir`."""
    out_root = Path(out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    video_eps = _generate_video_dir_dataset(
        out_root, fps=fps, duration_s=duration_s, width=width, height=height
    )
    lerobot_ep = _generate_lerobot_v3_dataset(
        out_root, fps=fps, duration_s=duration_s, width=width, height=height
    )
    result: dict[str, Any] = {
        "out_dir": str(out_root),
        "datasets": [
            str(out_root / "demo_pick_and_place"),
            str(out_root / "demo_lerobot_handover"),
        ],
        "episodes": [*video_eps, lerobot_ep],
    }
    if with_sample_run:
        ws_dir = Path(workspace) if workspace is not None else Path("./workspace")
        result["sample_run"] = _seed_sample_runs(out_root, ws_dir)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic multi-camera video_dir and LeRobot v3 demo datasets for robo_vlm_qc."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("./demo_data"),
        help="Output directory for generated demo datasets (default: ./demo_data).",
    )
    parser.add_argument("--fps", type=int, default=30, help="Video and kinematics FPS (default: 30).")
    parser.add_argument(
        "--duration-s",
        type=float,
        default=3.0,
        help="Duration per episode in seconds (default: 3.0).",
    )
    parser.add_argument("--width", type=int, default=320, help="Frame width in pixels (default: 320).")
    parser.add_argument("--height", type=int, default=240, help="Frame height in pixels (default: 240).")
    parser.add_argument(
        "--with-sample-run",
        action="store_true",
        help="Ingest generated episodes into --workspace and seed realistic offline VLM runs (zero API key required).",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("./workspace"),
        help="Target workspace directory when --with-sample-run is set (default: ./workspace).",
    )
    parser.add_argument("--json", action="store_true", dest="as_json", help="Print JSON summary.")
    args = parser.parse_args(argv)

    summary = generate_demo_dataset(
        args.out,
        fps=args.fps,
        duration_s=args.duration_s,
        width=args.width,
        height=args.height,
        with_sample_run=bool(args.with_sample_run),
        workspace=args.workspace,
    )
    if args.as_json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Generated {len(summary['episodes'])} synthetic demo episodes in {summary['out_dir']}:")
        for ep in summary["episodes"]:
            print(f"  - {ep['episode_id']} ({ep['adapter_type']}, {ep['frames']} frames @ {ep['fps']} FPS)")
        if args.with_sample_run and "sample_run" in summary:
            sr = summary["sample_run"]
            print(f"\nIngested and seeded {len(sr['seeded_runs'])} offline VLM runs in {sr['workspace']} (status=vlm_ready).")
            print("\nNext step (zero API key required):")
            print(f"  robo-vlm-qc -w {args.workspace} serve --no-auth")
        else:
            print("\nNext steps:")
            print(f"  robo-vlm-qc -w ./workspace ingest --data {args.out}")
            print("  robo-vlm-qc -w ./workspace serve --no-auth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
