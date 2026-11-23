"""Transactional sliced HDF5 exporter preserving raw sensor & JPEG camera streams."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, ClassVar

import cv2
import h5py
import numpy as np

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
    piece_source_rows,
    smooth_tail_81d,
    smooth_tail_vector,
)


def _concat_piece_arrays(
    pieces: list[KeepPiece],
    T: int,
    arr: np.ndarray,
    *,
    key: str,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for p in pieces:
        rows = piece_source_rows(p, T)
        seg = arr[rows]
        if p.smooth_tail_applied and p.tail_rows > 0:
            seg = smooth_tail_vector(key, seg, p.own_rows)
        chunks.append(seg)
    return np.concatenate(chunks, axis=0)


class H5CleanExporter(OutputExporter):
    """Exports sliced episodes to `<dataset>__<stem>.h5` with `/qc/*`, `/robot/*`, `/hand/*`, `/camera/*`."""

    name: ClassVar[str] = "h5_clean"

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
            for res, ep_pieces in included_pairs:
                view = views.get(res.episode_id)
                if view is None:
                    raise KeyError(f"Missing EpisodeView for {res.episode_id!r}")
                out_h5 = staging / f"{res.meta.dataset}__{res.meta.stem}.h5"
                self._write_episode_h5(
                    out_h5=out_h5,
                    res=res,
                    ep_pieces=ep_pieces,
                    view=view,
                    recipe=recipe,
                    global_subtasks=global_subtasks,
                    remapped_subtasks=remapped_subtasks,
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

    def _write_episode_h5(
        self,
        *,
        out_h5: Path,
        res: ResolvedEpisode,
        ep_pieces: list[KeepPiece],
        view: EpisodeView,
        recipe: ExportRecipe,
        global_subtasks: list[dict[str, Any]],
        remapped_subtasks: dict[str, list[int]],
    ) -> None:
        kin = view.kinematics
        T = (
            len(kin.timestamps_s)
            if (kin is not None and len(kin.timestamps_s) > 0)
            else int(view.meta.frame_count)
        )
        fps = float(ep_pieces[0].fps or view.fps or res.meta.fps or 30.0)

        all_src_rows: list[int] = []
        mask_chunks: list[np.ndarray] = []
        tail_chunks: list[np.ndarray] = []
        sub_chunks: list[np.ndarray] = []

        for p in ep_pieces:
            all_src_rows.extend(piece_source_rows(p, T))
            n_r = p.total_rows
            m_arr = np.asarray(p.action_loss_mask, dtype=np.float32)
            if m_arr.shape != (n_r, 2):
                fixed = np.ones((n_r, 2), dtype=np.float32)
                k = min(n_r, len(m_arr))
                if k > 0 and m_arr.ndim == 2 and m_arr.shape[1] == 2:
                    fixed[:k] = m_arr[:k]
                m_arr = fixed
            mask_chunks.append(m_arr)

            t_arr = np.asarray(p.is_tail, dtype=bool)
            if len(t_arr) != n_r:
                t_arr = np.asarray([False] * p.own_rows + [True] * p.tail_rows, dtype=bool)
            tail_chunks.append(t_arr)

            sub_list = remapped_subtasks.get(p.piece_id, p.subtask_indices)
            sub_chunks.append(np.asarray(sub_list, dtype=np.int64))

        N = len(all_src_rows)
        cat_mask = np.concatenate(mask_chunks, axis=0)
        cat_tail = np.concatenate(tail_chunks, axis=0)
        cat_sub = np.concatenate(sub_chunks, axis=0)

        utf8_dtype = h5py.string_dtype(encoding="utf-8")
        vlen_u8 = h5py.vlen_dtype(np.dtype("uint8"))

        with h5py.File(str(out_h5), "w") as hf:
            hf.attrs["schema"] = "robo_vlm_qc.h5_clean.v1"
            hf.attrs["episode_id"] = res.episode_id
            hf.attrs["dataset"] = res.meta.dataset
            hf.attrs["task_id"] = res.meta.task_id
            hf.attrs["adapter_type"] = res.meta.adapter_type
            hf.attrs["recipe"] = recipe.name
            hf.attrs["fps"] = fps
            hf.attrs["total_frames"] = int(N)
            hf.attrs["has_kinematics"] = bool(kin is not None)
            hf.attrs["source_uri"] = res.meta.source_uri

            # /qc group
            qc_grp = hf.create_group("qc")
            qc_grp.create_dataset("action_loss_mask", data=cat_mask, dtype=np.float32)
            qc_grp.create_dataset("is_tail", data=cat_tail, dtype=bool)
            qc_grp.create_dataset("subtask_index", data=cat_sub, dtype=np.int64)
            qc_grp.create_dataset(
                "resolved_json",
                data=res.model_dump_json(),
                dtype=utf8_dtype,
            )
            qc_grp.create_dataset(
                "subtasks_json",
                data=json.dumps(global_subtasks, ensure_ascii=False),
                dtype=utf8_dtype,
            )
            pieces_meta = [
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
                    "smooth_tail_applied": p.smooth_tail_applied,
                }
                for p in ep_pieces
            ]
            qc_grp.create_dataset(
                "pieces_json",
                data=json.dumps(pieces_meta, ensure_ascii=False),
                dtype=utf8_dtype,
            )

            # /robot and /hand groups when kinematics present
            if kin is not None:
                rob_grp = hf.create_group("robot")
                new_ts = np.arange(N, dtype=np.float64) / fps
                rob_grp.create_dataset("timestamps", data=new_ts, dtype=np.float64)
                if len(kin.timestamps_s) == T:
                    src_ts = np.asarray(kin.timestamps_s, dtype=np.float64)[all_src_rows]
                else:
                    src_ts = np.asarray(all_src_rows, dtype=np.float64) / fps
                rob_grp.create_dataset("source_timestamps", data=src_ts, dtype=np.float64)

                vm = getattr(kin, "valid_mask", None)
                if vm is not None and len(vm) == T:
                    rob_grp.create_dataset(
                        "valid_mask",
                        data=np.asarray(vm, dtype=bool)[all_src_rows],
                        dtype=bool,
                    )

                raw_state = getattr(kin, "raw_state", None)
                state_chunks: list[np.ndarray] = []
                if raw_state is not None:
                    rs_np = np.asarray(raw_state, dtype=np.float32)
                    for p in ep_pieces:
                        rows = piece_source_rows(p, T)
                        seg_s = rs_np[rows]
                        if p.smooth_tail_applied and p.tail_rows > 0:
                            if seg_s.ndim == 2 and seg_s.shape[1] == 81:
                                seg_s = smooth_tail_81d(seg_s, p.own_rows)
                            else:
                                seg_s = smooth_tail_vector("state", seg_s, p.own_rows)
                        state_chunks.append(seg_s)
                    rob_grp.create_dataset(
                        "state",
                        data=np.concatenate(state_chunks, axis=0),
                        dtype=np.float32,
                    )

                raw_action = getattr(kin, "raw_action", None)
                if raw_action is not None:
                    ra_np = np.asarray(raw_action, dtype=np.float32)
                    cat_act = _concat_piece_arrays(ep_pieces, T, ra_np, key="action")
                    rob_grp.create_dataset("action", data=cat_act, dtype=np.float32)
                elif state_chunks:
                    act_chunks: list[np.ndarray] = []
                    for seg_s in state_chunks:
                        a_seg = np.empty_like(seg_s)
                        if len(seg_s) > 1:
                            a_seg[:-1] = seg_s[1:]
                        if len(seg_s) > 0:
                            a_seg[-1] = seg_s[-1]
                        act_chunks.append(a_seg)
                    rob_grp.create_dataset(
                        "action",
                        data=np.concatenate(act_chunks, axis=0),
                        dtype=np.float32,
                    )

                for ch_name in (
                    "eef_pos",
                    "eef_rotvec",
                    "arm_q",
                    "torso_q",
                    "chassis_pose",
                    "head_q",
                ):
                    arr_val = getattr(kin, ch_name, None)
                    if arr_val is not None and len(arr_val) == T:
                        cat_ch = _concat_piece_arrays(
                            ep_pieces,
                            T,
                            np.asarray(arr_val, dtype=np.float32),
                            key=ch_name,
                        )
                        rob_grp.create_dataset(ch_name, data=cat_ch, dtype=np.float32)

                hand_fields = {
                    "hand_q": ("hand_q", True),
                    "tactile_force": ("tactile", False),
                    "hand_q_cmd": ("hand_q_cmd", True),
                    "hand_tau": ("hand_tau", False),
                }
                present_hand = [
                    (ds_name, attr_name, can_smooth)
                    for ds_name, (attr_name, can_smooth) in hand_fields.items()
                    if getattr(kin, attr_name, None) is not None
                    and len(getattr(kin, attr_name)) == T
                ]
                if present_hand:
                    hand_grp = hf.create_group("hand")
                    for ds_name, attr_name, can_smooth in present_hand:
                        arr_np = np.asarray(getattr(kin, attr_name), dtype=np.float32)
                        if can_smooth:
                            cat_h = _concat_piece_arrays(
                                ep_pieces, T, arr_np, key=ds_name
                            )
                        else:
                            cat_h = arr_np[all_src_rows]
                        hand_grp.create_dataset(ds_name, data=cat_h, dtype=np.float32)

            # /camera streams (bounded batches for low RSS across all input adapters)
            cam_root = hf.create_group("camera")
            cam_timestamps = np.arange(N, dtype=np.float64) / fps
            media = view.media
            is_h5_media = (
                hasattr(media, "_cam_map")
                and hasattr(media, "_row_to_idx")
                and hasattr(media, "_get_h5")
            )
            batch_sz = 64

            for cam in media.cameras:
                cg = cam_root.create_group(cam.name)
                cg.attrs["role"] = cam.role
                cg.attrs["width"] = int(cam.width)
                cg.attrs["height"] = int(cam.height)
                cg.attrs["fps"] = float(cam.fps)
                cg.attrs["source_type"] = cam.source_type
                cg.create_dataset("timestamps", data=cam_timestamps, dtype=np.float64)
                ds_jpeg = cg.create_dataset("jpeg", (N,), dtype=vlen_u8)

                # Zero-reencode fast path for non-virtual JPEG streams in H5MediaSource
                if is_h5_media and cam.source_type == "h5_jpeg" and not cam.is_virtual_crop:
                    raw_cam, crop_mode = media._cam_map[cam.name]  # type: ignore[attr-defined]
                    if crop_mode is None:
                        src_h5 = media._get_h5()  # type: ignore[attr-defined]
                        r2i = media._row_to_idx[raw_cam]  # type: ignore[attr-defined]
                        src_ds = src_h5[f"/camera/{raw_cam}/jpeg"]
                        h5_lock = getattr(media, "_h5_lock", None)
                        lock_ctx = h5_lock if h5_lock is not None else nullcontext()
                        for off in range(0, N, batch_sz):
                            chunk_rows = all_src_rows[off : off + batch_sz]
                            for k, sr in enumerate(chunk_rows):
                                raw_idx = int(r2i[sr])
                                with lock_ctx:
                                    entry = src_ds[raw_idx]
                                arr_u8 = (
                                    entry
                                    if isinstance(entry, np.ndarray)
                                    else np.frombuffer(bytes(entry), dtype=np.uint8)
                                )
                                ds_jpeg[off + k] = arr_u8
                        continue

                # General path: decode batch RGB and encode JPEG
                view_fps = float(view.fps or fps)
                for off in range(0, N, batch_sz):
                    chunk_rows = all_src_rows[off : off + batch_sz]
                    times_s = [r / view_fps for r in chunk_rows]
                    frames = media.read_frames_batch(cam.name, times_s)
                    for k, rgb in enumerate(frames):
                        bgr = np.ascontiguousarray(rgb[:, :, ::-1], dtype=np.uint8)
                        ok, buf = cv2.imencode(
                            ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90]
                        )
                        if not ok:
                            raise RuntimeError(
                                f"Failed JPEG encoding frame {off + k} for camera {cam.name}"
                            )
                        ds_jpeg[off + k] = np.frombuffer(buf.tobytes(), dtype=np.uint8)
