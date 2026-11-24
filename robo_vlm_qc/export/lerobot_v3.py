"""Transactional LeRobot v3.0 exporter (native / 81D / video-only Parquet, CFR H.264 MP4s, meta/*)."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..io.base import EpisodeView
from ..models.export import ExportManifest, ExportRecipe, KeepPiece, StateSchemaMode
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
    extract_piece_kinematics_native,
    piece_source_rows,
)


@dataclass(frozen=True)
class ResolvedStateSchema:
    """Resolved LeRobot v3 state/action/tactile schema for a batch of views."""

    mode_used: str  # "canonical_81d" | "native" | "video_only"
    include_state: bool
    state_dim: int
    state_names: list[str]
    include_action: bool
    action_dim: int
    action_names: list[str]
    has_tactile: bool
    tactile_dim: int


def _view_native_dims(
    view: EpisodeView,
) -> tuple[int | None, list[str] | None, int | None, list[str] | None, int | None]:
    kin = view.kinematics
    if kin is None:
        return None, None, None, None, None

    raw_st = getattr(kin, "raw_state", None)
    d_s: int | None = None
    if raw_st is not None:
        st_arr = np.asarray(raw_st)
        if st_arr.ndim == 2 and st_arr.shape[1] > 0:
            d_s = int(st_arr.shape[1])
        elif st_arr.ndim == 1 and len(st_arr) > 0:
            d_s = 1

    raw_act = getattr(kin, "raw_action", None)
    d_a: int | None = None
    if raw_act is not None:
        act_arr = np.asarray(raw_act)
        if act_arr.ndim == 2 and act_arr.shape[1] > 0:
            d_a = int(act_arr.shape[1])
        elif act_arr.ndim == 1 and len(act_arr) > 0:
            d_a = 1

    raw_tac = getattr(kin, "tactile", None)
    d_tac: int | None = None
    if raw_tac is not None:
        tac_arr = np.asarray(raw_tac)
        if tac_arr.ndim == 2 and tac_arr.shape[1] > 0:
            d_tac = int(tac_arr.shape[1])

    s_names = getattr(kin, "state_names", None)
    a_names = getattr(kin, "action_names", None)
    return (
        d_s,
        list(s_names) if isinstance(s_names, list) and s_names else None,
        d_a,
        list(a_names) if isinstance(a_names, list) and a_names else None,
        d_tac,
    )


def resolve_lerobot_state_schema(
    included_pairs: list[tuple[ResolvedEpisode, list[KeepPiece]]],
    views: dict[str, EpisodeView],
    state_schema_mode: StateSchemaMode = "auto",
) -> ResolvedStateSchema:
    """Resolve LeRobot v3 Parquet/stats/info schema for the included episode views."""
    included_views: list[EpisodeView] = []
    for res, _ in included_pairs:
        view = views.get(res.episode_id)
        if view is None:
            raise KeyError(f"Missing EpisodeView for {res.episode_id!r}")
        included_views.append(view)

    has_tactile = any(
        v.kinematics is not None and getattr(v.kinematics, "tactile", None) is not None
        for v in included_views
    )

    def _canonical() -> ResolvedStateSchema:
        return ResolvedStateSchema(
            mode_used="canonical_81d",
            include_state=True,
            state_dim=81,
            state_names=list(CANONICAL_81D_NAMES),
            include_action=True,
            action_dim=81,
            action_names=list(CANONICAL_81D_NAMES),
            has_tactile=has_tactile,
            tactile_dim=30 if has_tactile else 0,
        )

    if state_schema_mode == "canonical_81d":
        return _canonical()

    if not included_views:
        return _canonical()

    all_no_kin = all(v.kinematics is None for v in included_views)
    any_no_kin = any(v.kinematics is None for v in included_views)
    all_lerobot = all(v.meta.adapter_type == "lerobot" for v in included_views)
    all_h5 = all(v.meta.adapter_type == "h5_raw_v3" for v in included_views)
    any_h5 = any(v.meta.adapter_type == "h5_raw_v3" for v in included_views)

    if all_no_kin:
        return ResolvedStateSchema(
            mode_used="video_only",
            include_state=False,
            state_dim=0,
            state_names=[],
            include_action=False,
            action_dim=0,
            action_names=[],
            has_tactile=False,
            tactile_dim=0,
        )

    if state_schema_mode == "auto":
        if all_lerobot and not any_no_kin:
            dims_info = [_view_native_dims(v) for v in included_views]
            s_dims = {d[0] for d in dims_info}
            a_dims = {d[2] for d in dims_info}
            tac_dims = {d[4] for d in dims_info if d[4] is not None}
            if len(s_dims) == 1 and len(a_dims) == 1:
                d_s = next(iter(s_dims))
                d_a = next(iter(a_dims))
                if d_s is not None or d_a is not None:
                    s_names: list[str] = []
                    if d_s is not None:
                        s_cand = next(
                            (d[1] for d in dims_info if d[1] is not None and len(d[1]) == d_s),
                            None,
                        )
                        s_names = list(s_cand) if s_cand else [f"state_{i}" for i in range(d_s)]
                    a_names: list[str] = []
                    if d_a is not None:
                        a_cand = next(
                            (d[3] for d in dims_info if d[3] is not None and len(d[3]) == d_a),
                            None,
                        )
                        a_names = list(a_cand) if a_cand else [f"action_{i}" for i in range(d_a)]
                    if has_tactile and len(tac_dims) != 1:
                        return _canonical()
                    eff_tac_dim = next(iter(tac_dims)) if (has_tactile and tac_dims) else 0
                    return ResolvedStateSchema(
                        mode_used="native",
                        include_state=d_s is not None,
                        state_dim=d_s or 0,
                        state_names=s_names,
                        include_action=d_a is not None,
                        action_dim=d_a or 0,
                        action_names=a_names,
                        has_tactile=has_tactile,
                        tactile_dim=eff_tac_dim,
                    )
        return _canonical()

    # state_schema_mode == "preserve_native"
    if any_no_kin:
        raise ValueError(
            "Cannot preserve native state/action schema when mixing kinematics "
            "episodes with pure-video episodes; use state_schema_mode='auto' or 'canonical_81d'."
        )
    if all_h5:
        return _canonical()
    if any_h5:
        # h5_raw_v3 canonical representation is 81D; only compatible with 81D lerobot views
        non_h5 = [v for v in included_views if v.meta.adapter_type != "h5_raw_v3"]
        for v in non_h5:
            d_s, _, d_a, _, _ = _view_native_dims(v)
            if d_s != 81 or (d_a is not None and d_a != 81):
                raise ValueError(
                    f"Heterogeneous native dimensions in preserve_native mode: "
                    f"h5_raw_v3 (81D) cannot be combined with {v.meta.episode_id!r} "
                    f"(state_dim={d_s}, action_dim={d_a}). Use state_schema_mode='auto' or 'canonical_81d'."
                )
        return _canonical()

    dims_info = [_view_native_dims(v) for v in included_views]
    s_dims = {d[0] for d in dims_info}
    a_dims = {d[2] for d in dims_info}
    tac_dims = {d[4] for d in dims_info if d[4] is not None}
    if len(s_dims) != 1 or len(a_dims) != 1:
        raise ValueError(
            f"Heterogeneous native state/action dimensions cannot be preserved in "
            f"preserve_native mode (state_dims={sorted(str(x) for x in s_dims)}, "
            f"action_dims={sorted(str(x) for x in a_dims)}). Use state_schema_mode='auto' or 'canonical_81d'."
        )
    if has_tactile and len(tac_dims) != 1:
        raise ValueError(
            f"Heterogeneous tactile dimensions cannot be preserved in preserve_native mode "
            f"(tactile_dims={sorted(tac_dims)})."
        )
    d_s = next(iter(s_dims))
    d_a = next(iter(a_dims))
    if d_s is None and d_a is None:
        raise ValueError(
            "Selected episodes have kinematics metadata but neither raw_state nor raw_action arrays."
        )
    s_names = []
    if d_s is not None:
        s_cand = next(
            (d[1] for d in dims_info if d[1] is not None and len(d[1]) == d_s),
            None,
        )
        s_names = list(s_cand) if s_cand else [f"state_{i}" for i in range(d_s)]
    a_names = []
    if d_a is not None:
        a_cand = next(
            (d[3] for d in dims_info if d[3] is not None and len(d[3]) == d_a),
            None,
        )
        a_names = list(a_cand) if a_cand else [f"action_{i}" for i in range(d_a)]
    eff_tac_dim = next(iter(tac_dims)) if (has_tactile and tac_dims) else 0
    return ResolvedStateSchema(
        mode_used="native",
        include_state=d_s is not None,
        state_dim=d_s or 0,
        state_names=s_names,
        include_action=d_a is not None,
        action_dim=d_a or 0,
        action_names=a_names,
        has_tactile=has_tactile,
        tactile_dim=eff_tac_dim,
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
                "instruction": pa.array(
                    [
                        str(
                            s.get("instruction")
                            or s.get("label_en")
                            or s.get("label")
                            or s.get("subtask")
                            or ""
                        )
                        for s in global_subtasks
                    ],
                    type=pa.string(),
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
        schema_spec = resolve_lerobot_state_schema(
            included_pairs, views, recipe.state_schema_mode
        )

        # Collect unified camera specs (union across included views for mixed-adapter support)
        cam_specs: list[tuple[str, str, int, int]] = []
        seen_vk: set[str] = set()
        task_to_idx: dict[str, int] = {}

        for res, ep_pieces in included_pairs:
            view = views.get(res.episode_id)
            if view is None:
                raise KeyError(f"Missing EpisodeView for {res.episode_id!r}")
            for piece in ep_pieces:
                ep_task = (
                    (piece.task_override or "").strip()
                    or res.meta.task_id
                    or "_default"
                )
                if ep_task not in task_to_idx:
                    task_to_idx[ep_task] = len(task_to_idx)
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

        # Build PyArrow schema
        fields: list[pa.Field] = []
        if schema_spec.include_state:
            fields.append(
                pa.field(
                    "observation.state", pa.list_(pa.float32(), schema_spec.state_dim)
                )
            )
        if schema_spec.include_action:
            fields.append(
                pa.field("action", pa.list_(pa.float32(), schema_spec.action_dim))
            )
        if schema_spec.has_tactile:
            fields.append(
                pa.field(
                    "observation.tactile_force",
                    pa.list_(pa.float32(), schema_spec.tactile_dim),
                )
            )
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

        state_stats = (
            _OnlineStats(schema_spec.state_dim) if schema_spec.include_state else None
        )
        action_stats = (
            _OnlineStats(schema_spec.action_dim) if schema_spec.include_action else None
        )
        tac_stats = (
            _OnlineStats(schema_spec.tactile_dim) if schema_spec.has_tactile else None
        )
        mask_stats = _OnlineStats(2)
        ts_stats = _OnlineStats(1)
        per_row_bytes = max(64, (schema_spec.state_dim + schema_spec.action_dim) * 4)

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
                    ep_task = (
                        (piece.task_override or "").strip()
                        or res.meta.task_id
                        or "_default"
                    )
                    t_idx = task_to_idx[ep_task]

                    state_arr: np.ndarray | None = None
                    action_arr: np.ndarray | None = None
                    tac_arr: np.ndarray | None = None
                    if schema_spec.mode_used == "canonical_81d":
                        state_arr, action_arr, tac_arr = extract_piece_kinematics_81d(
                            piece, view
                        )
                    elif schema_spec.mode_used == "native":
                        state_arr, action_arr, tac_arr = extract_piece_kinematics_native(
                            piece,
                            view,
                            state_dim=(
                                schema_spec.state_dim
                                if schema_spec.include_state
                                else None
                            ),
                            action_dim=(
                                schema_spec.action_dim
                                if schema_spec.include_action
                                else None
                            ),
                        )

                    if schema_spec.has_tactile and tac_arr is None:
                        tac_arr = np.zeros(
                            (n_rows, schema_spec.tactile_dim), dtype=np.float32
                        )

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
                        if state_arr is not None:
                            state_arr = state_arr[idx_map]
                        if action_arr is not None:
                            action_arr = action_arr[idx_map]
                        if tac_arr is not None:
                            tac_arr = tac_arr[idx_map]
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

                    cols: dict[str, pa.Array] = {}
                    if schema_spec.include_state:
                        assert state_arr is not None
                        cols["observation.state"] = _fixed_list_col(
                            state_arr, schema_spec.state_dim, pa.float32()
                        )
                    if schema_spec.include_action:
                        assert action_arr is not None
                        cols["action"] = _fixed_list_col(
                            action_arr, schema_spec.action_dim, pa.float32()
                        )
                    if schema_spec.has_tactile:
                        assert tac_arr is not None
                        cols["observation.tactile_force"] = _fixed_list_col(
                            tac_arr, schema_spec.tactile_dim, pa.float32()
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
                    if state_stats is not None and state_arr is not None:
                        state_stats.update(state_arr)
                    if action_stats is not None and action_arr is not None:
                        action_stats.update(action_arr)
                    if tac_stats is not None and tac_arr is not None:
                        tac_stats.update(tac_arr)
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
                        "tasks": [ep_task],
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
                    est_bytes = max(cur_bytes, rows_in_file * per_row_bytes)
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

        stats: dict[str, Any] = {}
        if state_stats is not None:
            stats["observation.state"] = state_stats.to_dict()
        if action_stats is not None:
            stats["action"] = action_stats.to_dict()
        if tac_stats is not None:
            stats["observation.tactile_force"] = tac_stats.to_dict()
        stats["action_loss_mask"] = mask_stats.to_dict()
        stats["timestamp"] = ts_stats.to_dict()
        (meta_dir / "stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        features_meta: dict[str, Any] = {}
        if schema_spec.include_state:
            features_meta["observation.state"] = {
                "dtype": "float32",
                "shape": [schema_spec.state_dim],
                "names": list(schema_spec.state_names),
            }
        if schema_spec.include_action:
            features_meta["action"] = {
                "dtype": "float32",
                "shape": [schema_spec.action_dim],
                "names": list(schema_spec.action_names),
            }
        features_meta.update(
            {
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
        )
        if schema_spec.has_tactile:
            features_meta["observation.tactile_force"] = {
                "dtype": "float32",
                "shape": [schema_spec.tactile_dim],
                "names": [f"tactile_{i}" for i in range(schema_spec.tactile_dim)],
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
