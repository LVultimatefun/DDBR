"""Unit tests for LeRobotAdapter.open discover() metadata fallback."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from robo_vlm_qc.io import EpisodeHandle, get_adapter


def _write_mp4(path: Path, n_frames: int = 8, w: int = 64, h: int = 48, fps: float = 30.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened()
    try:
        for i in range(n_frames):
            frame = np.full((h, w, 3), (i * 25) % 255, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _make_lerobot_v3_dataset(root: Path, n_eps: int = 2, n_frames: int = 8) -> None:
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    info = {
        "codebase_version": "v3.0",
        "fps": 30,
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.images.main": {"dtype": "video", "shape": [48, 64, 3]},
            "observation.state": {"dtype": "float32", "shape": [2], "names": ["j0", "j1"]},
            "action": {"dtype": "float32", "shape": [2]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")

    data_rows = []
    ep_rows = []
    _write_mp4(
        root / "videos" / "observation.images.main" / "chunk-000" / "file-000.mp4",
        n_frames=n_eps * n_frames,
    )
    global_idx = 0
    for ep_idx in range(n_eps):
        for f in range(n_frames):
            data_rows.append(
                {
                    "episode_index": ep_idx,
                    "index": global_idx,
                    "timestamp": f / 30.0,
                    "observation.state": [float(ep_idx), float(f)],
                    "action": [float(ep_idx), float(f + 1)],
                }
            )
            global_idx += 1
        ep_rows.append(
            {
                "episode_index": ep_idx,
                "length": n_frames,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": ep_idx * n_frames,
                "dataset_to_index": (ep_idx + 1) * n_frames,
                "tasks": ["pick_and_place"],
                "videos/observation.images.main/chunk_index": 0,
                "videos/observation.images.main/file_index": 0,
                "videos/observation.images.main/from_timestamp": (ep_idx * n_frames) / 30.0,
                "videos/observation.images.main/to_timestamp": ((ep_idx + 1) * n_frames) / 30.0,
            }
        )

    pq.write_table(
        pa.Table.from_pylist(data_rows),
        root / "data" / "chunk-000" / "file-000.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(ep_rows),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )


def test_lerobot_open_fallback_from_extra_meta_and_empty_metadata(tmp_path: Path) -> None:
    ds_root = tmp_path / "lr_v3_ds"
    _make_lerobot_v3_dataset(ds_root, n_eps=2, n_frames=8)

    ad = get_adapter("lerobot")
    discovered = ad.discover(ds_root)
    assert len(discovered) == 2

    with ad.open(discovered[1]) as full_view:
        meta = full_view.meta

    handle_extra = EpisodeHandle(
        episode_id=meta.episode_id,
        dataset=meta.dataset,
        stem=meta.stem,
        task_id=meta.task_id,
        adapter_type="lerobot",
        source_uri=meta.source_uri,
        source_mtime=meta.source_mtime,
        metadata=dict(meta.extra_meta),
    )
    with ad.open(handle_extra) as view:
        assert view.meta.episode_id == meta.episode_id
        assert view.meta.frame_count == 8
        rgb = view.media.read_frame_rgb("main", 0.0)
        assert rgb.shape == (48, 64, 3)

    handle_empty = EpisodeHandle(
        episode_id="custom_ds/episode_000000",
        dataset="custom_ds",
        stem="episode_000000",
        task_id="pick_and_place",
        adapter_type="lerobot",
        source_uri=str(ds_root),
        source_mtime=meta.source_mtime,
        metadata={},
    )
    with ad.open(handle_empty) as view0:
        assert view0.meta.frame_count == 8
        assert view0.meta.episode_id == "custom_ds/episode_000000"


def test_lerobot_open_missing_episode_raises_file_not_found(tmp_path: Path) -> None:
    ds_root = tmp_path / "lr_v3_ds"
    _make_lerobot_v3_dataset(ds_root, n_eps=1, n_frames=6)

    ad = get_adapter("lerobot")
    bad_handle = EpisodeHandle(
        episode_id="lr_v3_ds/episode_999999",
        dataset="lr_v3_ds",
        stem="episode_999999",
        task_id="_default",
        adapter_type="lerobot",
        source_uri=str(ds_root),
        source_mtime=0.0,
        metadata={"ep_idx": 999999},
    )
    with pytest.raises(FileNotFoundError):
        ad.open(bad_handle)

    missing_root_handle = EpisodeHandle(
        episode_id="lr_v3_ds/episode_000000",
        dataset="lr_v3_ds",
        stem="episode_000000",
        task_id="_default",
        adapter_type="lerobot",
        source_uri=str(tmp_path / "does_not_exist"),
        source_mtime=0.0,
        metadata={},
    )
    with pytest.raises(FileNotFoundError):
        ad.open(missing_root_handle)
