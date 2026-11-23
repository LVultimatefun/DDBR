"""Transactional zero-copy index exporter (qc_index.parquet, manifest.jsonl, subtasks.json)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

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

_QC_INDEX_SCHEMA = pa.schema(
    [
        pa.field("piece_id", pa.string()),
        pa.field("episode_id", pa.string()),
        pa.field("dataset", pa.string()),
        pa.field("task_id", pa.string()),
        pa.field("adapter_type", pa.string()),
        pa.field("source_uri", pa.string()),
        pa.field("source_mtime", pa.float64()),
        pa.field("episode_fingerprint", pa.string()),
        pa.field("piece_index", pa.int64()),
        pa.field("start_row", pa.int64()),
        pa.field("end_row", pa.int64()),
        pa.field("tail_start_row", pa.int64()),
        pa.field("tail_end_row", pa.int64()),
        pa.field("own_rows", pa.int64()),
        pa.field("tail_rows", pa.int64()),
        pa.field("total_rows", pa.int64()),
        pa.field("fps", pa.float64()),
        pa.field("t_start_s", pa.float64()),
        pa.field("t_end_s", pa.float64()),
        pa.field("t0_unix", pa.float64()),
        pa.field("passed", pa.bool_()),
        pa.field("recipe", pa.string()),
        pa.field("camera_roles_json", pa.string()),
        pa.field("action_loss_mask", pa.list_(pa.list_(pa.float32(), 2))),
        pa.field("is_tail", pa.list_(pa.bool_())),
        pa.field("subtask_index", pa.list_(pa.int64())),
        pa.field("qc_segments_json", pa.string()),
    ]
)


class ZeroCopyIndexExporter(OutputExporter):
    """Exports lightweight zero-copy reference bundle pointing at read-only sources."""

    name: ClassVar[str] = "zero_copy_index"

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
        del views  # zero-copy index does not decode or copy media bytes
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

        index_rows: list[dict[str, Any]] = []
        manifest_lines: list[str] = []
        by_episode_subtasks: dict[str, list[dict[str, Any]]] = {}

        for res, ep_pieces in included_pairs:
            segs_dump = [s.model_dump(mode="json") for s in res.qc_segments]
            segs_json = json.dumps(segs_dump, ensure_ascii=False)
            cam_roles_json = json.dumps(res.meta.camera_roles, ensure_ascii=False)
            raw_t0 = res.meta.extra_meta.get("t0_unix")
            t0_unix = float(raw_t0) if raw_t0 is not None else None

            ep_sub_seen: dict[int, dict[str, Any]] = {}
            sub_lookup = {int(s["subtask_index"]): s for s in global_subtasks}

            for p in ep_pieces:
                sub_idx = remapped_subtasks.get(p.piece_id, p.subtask_indices)
                for si in sub_idx:
                    if si >= 0 and si in sub_lookup and si not in ep_sub_seen:
                        ep_sub_seen[si] = sub_lookup[si]

                mask_rows = [
                    [float(row[0]), float(row[1])]
                    for row in p.action_loss_mask
                ]
                index_rows.append(
                    {
                        "piece_id": p.piece_id,
                        "episode_id": res.episode_id,
                        "dataset": res.meta.dataset,
                        "task_id": res.meta.task_id,
                        "adapter_type": res.meta.adapter_type,
                        "source_uri": res.meta.source_uri,
                        "source_mtime": float(res.meta.source_mtime),
                        "episode_fingerprint": res.meta.episode_fingerprint,
                        "piece_index": int(p.piece_index),
                        "start_row": int(p.start_row),
                        "end_row": int(p.end_row),
                        "tail_start_row": p.tail_start_row,
                        "tail_end_row": p.tail_end_row,
                        "own_rows": int(p.own_rows),
                        "tail_rows": int(p.tail_rows),
                        "total_rows": int(p.total_rows),
                        "fps": float(p.fps),
                        "t_start_s": float(p.t_start_s),
                        "t_end_s": float(p.t_end_s),
                        "t0_unix": t0_unix,
                        "passed": res.success.passed,
                        "recipe": recipe.name,
                        "camera_roles_json": cam_roles_json,
                        "action_loss_mask": mask_rows,
                        "is_tail": [bool(x) for x in p.is_tail],
                        "subtask_index": [int(x) for x in sub_idx],
                        "qc_segments_json": segs_json,
                    }
                )

            by_episode_subtasks[res.episode_id] = [
                ep_sub_seen[k] for k in sorted(ep_sub_seen)
            ]

            manifest_rec = {
                "episode_id": res.episode_id,
                "dataset": res.meta.dataset,
                "task_id": res.meta.task_id,
                "adapter_type": res.meta.adapter_type,
                "source_uri": res.meta.source_uri,
                "source_mtime": float(res.meta.source_mtime),
                "episode_fingerprint": res.meta.episode_fingerprint,
                "status": res.status.value,
                "passed": res.success.passed,
                "fps": float(res.meta.fps),
                "duration_s": float(res.meta.duration_s),
                "frame_count": int(res.meta.frame_count),
                "recipe": recipe.name,
                "cameras": [c.model_dump(mode="json") for c in res.meta.cameras],
                "camera_roles": res.meta.camera_roles,
                "pieces": [
                    {
                        "piece_id": p.piece_id,
                        "piece_index": p.piece_index,
                        "start_row": p.start_row,
                        "end_row": p.end_row,
                        "tail_start_row": p.tail_start_row,
                        "tail_end_row": p.tail_end_row,
                        "own_rows": p.own_rows,
                        "tail_rows": p.tail_rows,
                        "total_rows": p.total_rows,
                        "t_start_s": p.t_start_s,
                        "t_end_s": p.t_end_s,
                    }
                    for p in ep_pieces
                ],
                "qc_segments": segs_dump,
                "macro_stages": [
                    m.model_dump(mode="json") for m in res.macro_stages
                ],
            }
            manifest_lines.append(json.dumps(manifest_rec, ensure_ascii=False))

        subtasks_doc = {
            "recipe": recipe.name,
            "subtasks": global_subtasks,
            "by_episode": by_episode_subtasks,
        }

        with staged_export_dir(out_dir) as staging:
            if index_rows:
                tbl = pa.Table.from_pylist(index_rows, schema=_QC_INDEX_SCHEMA)
            else:
                tbl = _QC_INDEX_SCHEMA.empty_table()
            pq.write_table(tbl, staging / "qc_index.parquet")

            (staging / "manifest.jsonl").write_text(
                ("\n".join(manifest_lines) + "\n") if manifest_lines else "",
                encoding="utf-8",
            )
            (staging / "subtasks.json").write_text(
                json.dumps(subtasks_doc, ensure_ascii=False, indent=2),
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
        )
