"""Transactional LeRobot v3.0 exporter (81D Parquet, CFR H.264 MP4s, meta/*)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..io.base import EpisodeView
from ..models.export import ExportManifest, ExportRecipe, KeepPiece
from ..models.review import ResolvedEpisode
from .base import (
    OutputExporter,
    build_global_subtasks,
    finalize_manifest,
    group_kept_pieces,
    staged_export_dir,
)
from .slicer import (
    CANONICAL_81D_NAMES,
    build_badcases_and_pairs,
    extract_piece_kinematics_81d,
    piece_source_rows,
)


def _resolve_export_cfg(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    settings = getattr(cfg, "settings", None)
    if isinstance(settings, dict):
        return dict(settings.get("export", {}))
    if isinstance(cfg, dict):
        if isinstance(cfg.get("export"), dict):
            return dict(cfg["export"])
        return dict(cfg)
    return {}


class _OnlineStats:
    """Streaming float64 min/max/mean/std accumulator."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.count = 0
        self.min_v = np.full(dim, np.inf, dtype=np.float64)
        self.max_v = np.full(dim, -np.inf, dtype=np.float64)
        self.sum_v = np.zeros(dim, dtype=np.float64)
        self.sum_sq = np.zeros(dim, dtype=np.float64)

    def update(self, arr: np.ndarray) -> None:
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim == 1:
            a = a[:, None]
        if len(a) == 0:
            return
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        self.count += len(a)
        self.min_v = np.minimum(self.min_v, a.min(axis=0))
        self.max_v = np.maximum(self.max_v, a.max(axis=0))
        self.sum_v += a.sum(axis=0)
        self.sum_sq += np.square(a).sum(axis=0)

    def to_dict(self) -> dict[str, Any]:
        if self.count == 0:
            zeros = [0.0] * self.dim
            return {
                "min": zeros,
                "max": zeros,
                "mean": zeros,
                "std": zeros,
                "count": [0],
            }
        mean = self.sum_v / float(self.count)
        var = np.maximum(0.0, (self.sum_sq / float(self.count)) - np.square(mean))
        std = np.sqrt(var)
        return {
            "min": self.min_v.tolist(),
            "max": self.max_v.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "count": [self.count],
        }


class _FfmpegPipeWriter:
    """Streams raw RGB24 frames into an ffmpeg libx264 CFR MP4 pipe."""

    def __init__(self, out_path: Path, width: int, height: int, fps: float) -> None:
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise RuntimeError("ffmpeg binary not found in PATH")
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.frames_written = 0

        cmd = [
            ffmpeg_bin,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:.6f}",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-vsync",
            "cfr",
            "-g",
            "4",
            "-keyint_min",
            "4",
            "-bf",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(self.out_path),
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert self.proc.stdin is not None

    def write_frame(self, rgb: np.ndarray) -> None:
        assert self.proc.stdin is not None
        if rgb.shape[1] != self.width or rgb.shape[0] != self.height:
            rgb = cv2.resize(
                rgb, (self.width, self.height), interpolation=cv2.INTER_AREA
            )
        if not rgb.flags["C_CONTIGUOUS"] or rgb.dtype != np.uint8:
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        self.proc.stdin.write(rgb.tobytes())
        self.frames_written += 1

    def close(self) -> None:
        if self.proc.poll() is None:
            if self.proc.stdin:
                try:
                    self.proc.stdin.close()
                except Exception:
                    pass
            stderr_b = self.proc.stderr.read() if self.proc.stderr else b""
            rc = self.proc.wait()
            if rc != 0:
                raise RuntimeError(
                    f"ffmpeg exited {rc} writing {self.out_path}: "
                    f"{stderr_b.decode('utf-8', 'replace')}"
                )

    def kill(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.kill()
                self.proc.wait()
            except Exception:
                pass


def _fixed_list_col(arr: np.ndarray, width: int, pa_type: pa.DataType) -> pa.Array:
    flat = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
    return pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa_type), width)


class LeRobotV3Exporter(OutputExporter):
    """Transactional LeRobot v3.0 dataset exporter."""

    name: ClassVar[str] = "lerobot_v3"

    def __init__(self, cfg: Any = None, *, config: Any = None) -> None:
        self.cfg = cfg or config

    def export(
        self,
        pieces: list[KeepPiece],
        resolved_list: list[ResolvedEpisode],
        views: dict[str, EpisodeView],
        out_dir: Path,
        recipe: ExportRecipe,
    ) -> ExportManifest:
        exp_cfg = _resolve_export_cfg(self.cfg)
        max_file_mb = float(exp_cfg.get("lerobot_v3_max_file_mb", 100.0))
        max_file_bytes = max(1, int(max_file_mb * 1024 * 1024))
        chunks_size = max(1, int(exp_cfg.get("chunks_size", 1000)))
        robot_type = str(exp_cfg.get("robot_type") or "bimanual_robot")

        (
            included_pairs,
            entries,
            episodes_included,
            episodes_skipped,
            pieces_exported,
            pieces_dropped,
            total_frames,
        ) = group_kept_pieces(pieces, resolved_list, recipe)

        kept_pieces_all = [p for _, ep_ps in included_pairs for p in ep_ps]
        global_subtasks, remapped_subtasks = build_global_subtasks(kept_pieces_all)

        with staged_export_dir(out_dir) as staging:
            meta_dir = staging / "meta"
            meta_dir.mkdir(parents=True, exist_ok=True)

            if not kept_pieces_all:
                self._write_empty_bundle(
                    meta_dir, chunks_size, global_subtasks, robot_type=robot_type
                )
            else:
                total_frames = self._write_populated_bundle(
                    staging=staging,
                    meta_dir=meta_dir,
                    included_pairs=included_pairs,
                    kept_pieces_all=kept_pieces_all,
                    views=views,
                    recipe=recipe,
                    entries=entries,
                    global_subtasks=global_subtasks,
                    remapped_subtasks=remapped_subtasks,
                    max_file_bytes=max_file_bytes,
                    chunks_size=chunks_size,
                    robot_type=robot_type,
                )

            manifest_preview = finalize_manifest(
                staging,
                recipe,
                entries,
                episodes_included,
                episodes_skipped,
                pieces_exported,
                pieces_dropped,
                total_frames,
            )
            (meta_dir / "qc_export_manifest.json").write_text(
                manifest_preview.model_dump_json(indent=2),
                encoding="utf-8",
            )

        return finalize_manifest(
            out_dir,
            recipe,
            entries,
            episodes_included,
            episodes_skipped,
            pieces_exported,
            pieces_dropped,
            total_frames,
            export_id=manifest_preview.export_id,
        )

    def _write_empty_bundle(
        self,
        meta_dir: Path,
        chunks_size: int,
        global_subtasks: list[dict[str, Any]],
        *,
        robot_type: str = "bimanual_robot",
    ) -> None:
        self._write_tasks_parquet(meta_dir / "tasks.parquet", {})
        self._write_subtasks_parquet(meta_dir / "subtasks.parquet", global_subtasks)
        ep_dir = meta_dir / "episodes" / "chunk-000"
        ep_dir.mkdir(parents=True, exist_ok=True)
        empty_ep_tbl = pa.table(
            {
                "episode_index": pa.array([], type=pa.int64()),
                "length": pa.array([], type=pa.int64()),
                "task_index": pa.array([], type=pa.int64()),
                "data/chunk_index": pa.array([], type=pa.int64()),
                "data/file_index": pa.array([], type=pa.int64()),
                "dataset_from_index": pa.array([], type=pa.int64()),
                "dataset_to_index": pa.array([], type=pa.int64()),
            }
        )
        pq.write_table(empty_ep_tbl, ep_dir / "file-000.parquet")
        (meta_dir / "stats.json").write_text("{}", encoding="utf-8")
        info = {
            "codebase_version": "v3.0",
            "robot_type": robot_type,
            "total_episodes": 0,
            "total_frames": 0,
            "total_tasks": 0,
            "chunks_size": chunks_size,
            "fps": 30.0,
            "splits": {"train": "0:0"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": {},
        }
        (meta_dir / "info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _write_tasks_parquet(self, path: Path, task_to_idx: dict[str, int]) -> None:
        ordered = sorted(task_to_idx.items(), key=lambda kv: kv[1])
        tbl = pa.table(
            {
                "task_index": pa.array([idx for _, idx in ordered], type=pa.int64()),
                "task": pa.array([name for name, _ in ordered], type=pa.string()),
            }
        )
        pq.write_table(tbl, path)

    def _write_subtasks_parquet(
        self, path: Path, global_subtasks: list[dict[str, Any]]
    ) -> None:
        tbl = pa.table(
            {
                "subtask_index": pa.array(
                    [int(s["subtask_index"]) for s in global_subtasks], type=pa.int64()
                ),
                "subtask": pa.array(
                    [str(s["subtask"]) for s in global_subtasks], type=pa.string()
                ),
                "stage_id": pa.array(
                    [str(s["stage_id"]) for s in global_subtasks], type=pa.string()
                ),
                "label": pa.array(
                    [str(s["label"]) for s in global_subtasks], type=pa.string()
                ),
                "label_en": pa.array(
                    [str(s["label_en"]) for s in global_subtasks], type=pa.string()
                ),
            }
        )
        pq.write_table(tbl, path)

    def _write_populated_bundle(
        self,
        *,
        staging: Path,
        meta_dir: Path,
        included_pairs: list[tuple[ResolvedEpisode, list[KeepPiece]]],
        kept_pieces_all: list[KeepPiece],
        views: dict[str, EpisodeView],
        recipe: ExportRecipe,
        entries: list[Any],
        global_subtasks: list[dict[str, Any]],
        remapped_subtasks: dict[str, list[int]],
        max_file_bytes: int,
        chunks_size: int,
        robot_type: str = "bimanual_robot",
    ) -> int:
        export_fps = float(kept_pieces_all[0].fps or 30.0)

        # Collect unified camera specs (union across included views for mixed-adapter support)
        cam_specs: list[tuple[str, str, int, int]] = []
        seen_vk: set[str] = set()
        has_tactile = False
        task_to_idx: dict[str, int] = {}

        for res, _ in included_pairs:
            task_str = res.meta.task_id or "_default"
            if task_str not in task_to_idx:
                task_to_idx[task_str] = len(task_to_idx)
            view = views.get(res.episode_id)
            if view is None:
                raise KeyError(f"Missing EpisodeView for {res.episode_id!r}")
            # Touch view.media early so broken media fails fast inside staging
            cams = list(view.media.cameras)
            for c in cams:
                vk = (
                    c.name
                    if c.name.startswith("observation.images.")
                    else f"observation.images.{c.name}"
                )
                if vk not in seen_vk:
                    seen_vk.add(vk)
                    w_even = max(2, int(c.width) - (int(c.width) % 2))
                    h_even = max(2, int(c.height) - (int(c.height) % 2))
                    cam_specs.append((c.name, vk, w_even, h_even))
            if view.kinematics is not None and getattr(view.kinematics, "tactile", None) is not None:
                has_tactile = True

        # Build PyArrow schema
        fields: list[pa.Field] = [
            pa.field("observation.state", pa.list_(pa.float32(), 81)),
            pa.field("action", pa.list_(pa.float32(), 81)),
        ]
        if has_tactile:
            fields.append(pa.field("observation.tactile_force", pa.list_(pa.float32(), 30)))
        fields.extend(
            [
                pa.field("action_loss_mask", pa.list_(pa.float32(), 2)),
                pa.field("is_tail", pa.bool_()),
                pa.field("subtask_index", pa.int64()),
                pa.field("timestamp", pa.float32()),
                pa.field("frame_index", pa.int64()),
                pa.field("episode_index", pa.int64()),
                pa.field("index", pa.int64()),
                pa.field("task_index", pa.int64()),
            ]
        )
        data_schema = pa.schema(fields)

        state_stats = _OnlineStats(81)
        action_stats = _OnlineStats(81)
        tac_stats = _OnlineStats(30) if has_tactile else None
        mask_stats = _OnlineStats(2)
        ts_stats = _OnlineStats(1)

        ep_rows: list[dict[str, Any]] = []
        chunk_idx = 0
        file_idx = 0
        global_row_idx = 0
        ep_idx = 0

        pq_writer: pq.ParquetWriter | None = None
        current_data_path: Path | None = None
        video_writers: dict[str, _FfmpegPipeWriter] = {}
        file_frame_offset = 0
        rows_in_file = 0

        def _open_chunk_file(c_i: int, f_i: int) -> None:
            nonlocal pq_writer, current_data_path, video_writers, file_frame_offset, rows_in_file
            d_path = staging / "data" / f"chunk-{c_i:03d}" / f"file-{f_i:03d}.parquet"
            d_path.parent.mkdir(parents=True, exist_ok=True)
            pq_writer = pq.ParquetWriter(d_path, data_schema, compression="snappy")
            current_data_path = d_path
            video_writers = {}
            for _, vk, w_e, h_e in cam_specs:
                v_path = (
                    staging
                    / "videos"
                    / vk
                    / f"chunk-{c_i:03d}"
                    / f"file-{f_i:03d}.mp4"
                )
                video_writers[vk] = _FfmpegPipeWriter(v_path, w_e, h_e, export_fps)
            file_frame_offset = 0
            rows_in_file = 0

        def _close_chunk_file() -> None:
            nonlocal pq_writer, video_writers
            if pq_writer is not None:
                pq_writer.close()
                pq_writer = None
            for vw in video_writers.values():
                vw.close()
            video_writers = {}

        try:
            _open_chunk_file(chunk_idx, file_idx)

            for res, ep_pieces in included_pairs:
                view = views[res.episode_id]
                task_str = res.meta.task_id or "_default"
                t_idx = task_to_idx[task_str]
                ep_cam_names = {c.name for c in view.media.cameras}
                fallback_cam = view.camera_roles.get("main") or view.media.cameras[0].name
                T_view = (
                    len(view.kinematics.timestamps_s)
                    if (view.kinematics is not None and len(view.kinematics.timestamps_s) > 0)
                    else int(view.meta.frame_count)
                )
                ep_resampled_frames = 0

                for piece in ep_pieces:
                    assert pq_writer is not None
                    n_rows = piece.total_rows
                    piece_fps = float(piece.fps or export_fps)
                    state_81, action_81, tac_30 = extract_piece_kinematics_81d(
                        piece, view
                    )
                    if has_tactile and tac_30 is None:
                        tac_30 = np.zeros((n_rows, 30), dtype=np.float32)

                    mask_arr = np.asarray(piece.action_loss_mask, dtype=np.float32)
                    if mask_arr.shape != (n_rows, 2):
                        fixed_mask = np.ones((n_rows, 2), dtype=np.float32)
                        k_copy = min(n_rows, len(mask_arr))
                        if k_copy > 0 and mask_arr.ndim == 2 and mask_arr.shape[1] == 2:
                            fixed_mask[:k_copy] = mask_arr[:k_copy]
                        mask_arr = fixed_mask

                    is_tail_arr = np.asarray(piece.is_tail, dtype=bool)
                    if len(is_tail_arr) != n_rows:
                        is_tail_arr = np.asarray(
                            [False] * piece.own_rows + [True] * piece.tail_rows,
                            dtype=bool,
                        )

                    sub_arr = np.asarray(
                        remapped_subtasks.get(piece.piece_id, piece.subtask_indices),
                        dtype=np.int64,
                    )
                    src_rows = piece_source_rows(piece, T_view)
                    view_fps = float(view.fps or piece_fps)
                    times_s = [r / view_fps for r in src_rows]

                    if abs(piece_fps - export_fps) > 1e-3:
                        target_rows = max(1, int(round((n_rows / piece_fps) * export_fps)))
                        idx_map = np.clip(
                            np.floor(
                                np.arange(target_rows, dtype=np.float64)
                                * (piece_fps / export_fps)
                            ).astype(np.int64),
                            0,
                            n_rows - 1,
                        )
                        state_81 = state_81[idx_map]
                        action_81 = action_81[idx_map]
                        if tac_30 is not None:
                            tac_30 = tac_30[idx_map]
                        mask_arr = mask_arr[idx_map]
                        is_tail_arr = is_tail_arr[idx_map]
                        sub_arr = sub_arr[idx_map]
                        times_s = [times_s[int(i)] for i in idx_map]
                        n_rows = target_rows

                    ep_resampled_frames += n_rows
                    frame_indices = np.arange(n_rows, dtype=np.int64)
                    timestamps = (frame_indices / export_fps).astype(np.float32)
                    ep_indices = np.full(n_rows, ep_idx, dtype=np.int64)
                    global_indices = np.arange(
                        global_row_idx, global_row_idx + n_rows, dtype=np.int64
                    )
                    task_indices = np.full(n_rows, t_idx, dtype=np.int64)

                    cols: dict[str, pa.Array] = {
                        "observation.state": _fixed_list_col(state_81, 81, pa.float32()),
                        "action": _fixed_list_col(action_81, 81, pa.float32()),
                    }
                    if has_tactile:
                        assert tac_30 is not None
                        cols["observation.tactile_force"] = _fixed_list_col(
                            tac_30, 30, pa.float32()
                        )
                    cols.update(
                        {
                            "action_loss_mask": _fixed_list_col(
                                mask_arr, 2, pa.float32()
                            ),
                            "is_tail": pa.array(is_tail_arr, type=pa.bool_()),
                            "subtask_index": pa.array(sub_arr, type=pa.int64()),
                            "timestamp": pa.array(timestamps, type=pa.float32()),
                            "frame_index": pa.array(frame_indices, type=pa.int64()),
                            "episode_index": pa.array(ep_indices, type=pa.int64()),
                            "index": pa.array(global_indices, type=pa.int64()),
                            "task_index": pa.array(task_indices, type=pa.int64()),
                        }
                    )
                    ep_tbl = pa.table(cols, schema=data_schema)
                    pq_writer.write_table(ep_tbl)

                    # Update stats
                    state_stats.update(state_81)
                    action_stats.update(action_81)
                    if tac_stats is not None and tac_30 is not None:
                        tac_stats.update(tac_30)
                    mask_stats.update(mask_arr)
                    ts_stats.update(timestamps)

                    # Stream video frames in bounded batches
                    ep_video_meta: dict[str, Any] = {}
                    from_ts = float(file_frame_offset / export_fps)
                    to_ts = float((file_frame_offset + n_rows) / export_fps)

                    batch_sz = 64
                    for cam_name, vk, _, _ in cam_specs:
                        read_cam = cam_name if cam_name in ep_cam_names else fallback_cam
                        vw = video_writers[vk]
                        for b_start in range(0, n_rows, batch_sz):
                            batch_t = times_s[b_start : b_start + batch_sz]
                            frames = view.media.read_frames_batch(read_cam, batch_t)
                            for rgb in frames:
                                vw.write_frame(rgb)
                        ep_video_meta[f"videos/{vk}/chunk_index"] = chunk_idx
                        ep_video_meta[f"videos/{vk}/file_index"] = file_idx
                        ep_video_meta[f"videos/{vk}/from_timestamp"] = from_ts
                        ep_video_meta[f"videos/{vk}/to_timestamp"] = to_ts

                    ep_rec: dict[str, Any] = {
                        "episode_index": ep_idx,
                        "tasks": [task_str],
                        "task_index": t_idx,
                        "length": n_rows,
                        "data/chunk_index": chunk_idx,
                        "data/file_index": file_idx,
                        "dataset_from_index": global_row_idx,
                        "dataset_to_index": global_row_idx + n_rows,
                        "source_episode_id": res.episode_id,
                        "piece_id": piece.piece_id,
                    }
                    ep_rec.update(ep_video_meta)
                    ep_rows.append(ep_rec)

                    ep_idx += 1
                    global_row_idx += n_rows
                    file_frame_offset += n_rows
                    rows_in_file += n_rows

                    # Roll over file_index / chunk_index if max_file_bytes exceeded
                    assert current_data_path is not None
                    cur_bytes = (
                        current_data_path.stat().st_size
                        if current_data_path.is_file()
                        else 0
                    )
                    est_bytes = max(cur_bytes, rows_in_file * 81 * 4 * 2)
                    if est_bytes >= max_file_bytes:
                        _close_chunk_file()
                        file_idx += 1
                        if file_idx >= chunks_size:
                            chunk_idx += 1
                            file_idx = 0
                        # Only open next file if more pieces remain
                        if ep_idx < len(kept_pieces_all):
                            _open_chunk_file(chunk_idx, file_idx)

                for ent in entries:
                    if ent.episode_id == res.episode_id and ent.total_frames != ep_resampled_frames:
                        ent.total_frames = ep_resampled_frames
                        break

            _close_chunk_file()
        except BaseException:
            if pq_writer is not None:
                try:
                    pq_writer.close()
                except Exception:
                    pass
            for vw in video_writers.values():
                vw.kill()
            raise

        # Write meta/tasks.parquet, meta/subtasks.parquet, meta/episodes/..., meta/stats.json, meta/info.json
        self._write_tasks_parquet(meta_dir / "tasks.parquet", task_to_idx)
        self._write_subtasks_parquet(meta_dir / "subtasks.parquet", global_subtasks)

        ep_meta_dir = meta_dir / "episodes" / "chunk-000"
        ep_meta_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(ep_rows),
            ep_meta_dir / "file-000.parquet",
        )

        stats: dict[str, Any] = {
            "observation.state": state_stats.to_dict(),
            "action": action_stats.to_dict(),
            "action_loss_mask": mask_stats.to_dict(),
            "timestamp": ts_stats.to_dict(),
        }
        if tac_stats is not None:
            stats["observation.tactile_force"] = tac_stats.to_dict()
        (meta_dir / "stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        features_meta: dict[str, Any] = {
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
            "action_loss_mask": {
                "dtype": "float32",
                "shape": [2],
                "names": ["left_mask", "right_mask"],
            },
            "is_tail": {"dtype": "bool", "shape": [1]},
            "subtask_index": {"dtype": "int64", "shape": [1]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        }
        if has_tactile:
            features_meta["observation.tactile_force"] = {
                "dtype": "float32",
                "shape": [30],
                "names": [f"tactile_{i}" for i in range(30)],
            }
        for _, vk, w_e, h_e in cam_specs:
            features_meta[vk] = {
                "dtype": "video",
                "shape": [h_e, w_e, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.fps": export_fps,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                },
            }

        info = {
            "codebase_version": "v3.0",
            "robot_type": robot_type,
            "total_episodes": ep_idx,
            "total_frames": global_row_idx,
            "total_tasks": len(task_to_idx),
            "total_subtasks": len(global_subtasks),
            "chunks_size": chunks_size,
            "fps": export_fps,
            "splits": {"train": f"0:{ep_idx}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": features_meta,
        }
        (meta_dir / "info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if recipe.name == "badcase_pairs":
            bp = build_badcases_and_pairs(
                [res for res, _ in included_pairs], kept_pieces_all, recipe
            )
            (meta_dir / "badcases_and_pairs.json").write_text(
                json.dumps(bp, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return global_row_idx
