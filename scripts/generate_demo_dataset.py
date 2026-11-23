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


def generate_demo_dataset(
    out_dir: Path,
    *,
    fps: int = 30,
    duration_s: float = 3.0,
    width: int = 320,
    height: int = 240,
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
    return {
        "out_dir": str(out_root),
        "datasets": [
            str(out_root / "demo_pick_and_place"),
            str(out_root / "demo_lerobot_handover"),
        ],
        "episodes": [*video_eps, lerobot_ep],
    }


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
    parser.add_argument("--json", action="store_true", dest="as_json", help="Print JSON summary.")
    args = parser.parse_args(argv)

    summary = generate_demo_dataset(
        args.out,
        fps=args.fps,
        duration_s=args.duration_s,
        width=args.width,
        height=args.height,
    )
    if args.as_json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Generated {len(summary['episodes'])} synthetic demo episodes in {summary['out_dir']}:")
        for ep in summary["episodes"]:
            print(f"  - {ep['episode_id']} ({ep['adapter_type']}, {ep['frames']} frames @ {ep['fps']} FPS)")
        print("\nNext steps:")
        print(f"  robo-vlm-qc -w ./workspace ingest --data {args.out}")
        print("  robo-vlm-qc -w ./workspace serve --no-auth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
