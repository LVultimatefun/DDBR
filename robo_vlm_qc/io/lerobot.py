"""Read-only InputAdapter for LeRobot v2 and v3 datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from ..models.episode import CameraSpec, EpisodeMeta, KinematicsSpec
from .base import (
    SKIP_DIRS,
    EpisodeHandle,
    EpisodeView,
    InputAdapter,
    KinematicsData,
    build_camera_roles,
    compute_episode_fingerprint,
    infer_camera_role,
    resolve_task_id,
    validate_id_part,
)
from .media import Mp4Reader, probe_video_cv2, time_to_frame_idx


def _strip_image_prefix(feat_key: str) -> str:
    if feat_key.startswith("observation.images."):
        return feat_key[len("observation.images.") :]
    if feat_key.startswith("observation.image."):
        return feat_key[len("observation.image.") :]
    return feat_key


def _safe_load_info(ds_root: Path) -> dict[str, Any] | None:
    info_path = ds_root / "meta" / "info.json"
    if not info_path.is_file():
        return None
    try:
        data = json.loads(info_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        if not isinstance(data.get("features"), dict):
            return None
        fps = float(data.get("fps", 0.0))
        if fps <= 0.0:
            return None
        ver = str(data.get("codebase_version", ""))
        if not (ver.startswith("v2") or ver.startswith("v3") or "data_path" in data):
            return None
        return data
    except Exception:
        return None


class LeRobotMediaSource:
    """MediaSource supporting both chunked MP4 videos and Parquet-embedded images."""

    def __init__(
        self,
        *,
        cameras: list[CameraSpec],
        video_entries: dict[str, dict[str, Any]],
        image_frames: dict[str, list[bytes]],
        fps: float,
        duration_s: float,
    ) -> None:
        self.cameras = cameras
        self._video_entries = video_entries
        self._image_frames = image_frames
        self.fps = float(fps)
        self.duration_s = float(duration_s)
        self._mp4 = Mp4Reader()
        self._role_to_cam: dict[str, str] = {}
        for c in cameras:
            if c.role not in self._role_to_cam:
                self._role_to_cam[c.role] = c.name

    def _resolve_camera(self, camera: str) -> str:
        if camera in self._video_entries or camera in self._image_frames:
            return camera
        if camera in self._role_to_cam:
            return self._role_to_cam[camera]
        raise KeyError(
            f"Unknown camera {camera!r}; available: {[c.name for c in self.cameras]}"
        )

    def _decode_image_row(self, cam: str, row_idx: int) -> np.ndarray:
        frames = self._image_frames[cam]
        idx = max(0, min(len(frames) - 1, row_idx))
        raw = frames[idx]
        arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed decoding embedded image frame {idx} for {cam}")
        return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)

    @staticmethod
    def _frame_idx_for_time(
        ent: dict[str, Any], duration_s: float, t_s: float
    ) -> int:
        t_clamped = max(0.0, min(duration_s, float(t_s)))
        raw_idx = time_to_frame_idx(
            t_clamped,
            ent["fps"],
            ent["file_frames"],
            offset_s=ent["from_timestamp"],
        )
        start_f = int(ent.get("start_frame", 0))
        end_f = int(ent.get("end_frame", max(0, int(ent["file_frames"]) - 1)))
        if end_f < start_f:
            end_f = start_f
        return max(start_f, min(end_f, raw_idx))

    def read_frame_rgb(self, camera: str, t_s: float) -> np.ndarray:
        cam = self._resolve_camera(camera)
        if cam in self._video_entries:
            ent = self._video_entries[cam]
            f_idx = self._frame_idx_for_time(ent, self.duration_s, t_s)
            return self._mp4.read_frame_rgb(Path(ent["path"]), f_idx)
        t_clamped = max(0.0, min(self.duration_s, float(t_s)))
        row_idx = int(np.clip(round(t_clamped * self.fps), 0, len(self._image_frames[cam]) - 1))
        return self._decode_image_row(cam, row_idx)

    def read_frames_batch(
        self, camera: str, times_s: list[float]
    ) -> list[np.ndarray]:
        if not times_s:
            return []
        cam = self._resolve_camera(camera)
        if cam in self._video_entries:
            ent = self._video_entries[cam]
            indices = [
                self._frame_idx_for_time(ent, self.duration_s, t)
                for t in times_s
            ]
            return self._mp4.read_frames_batch(Path(ent["path"]), indices)
        n = len(self._image_frames[cam])
        return [
            self._decode_image_row(
                cam,
                int(np.clip(round(max(0.0, min(self.duration_s, float(t))) * self.fps), 0, n - 1)),
            )
            for t in times_s
        ]

    def close(self) -> None:
        self._mp4.close()


class LeRobotAdapter(InputAdapter):
    """InputAdapter for LeRobot v2 and v3 datasets."""

    name: ClassVar[str] = "lerobot"
    priority: ClassVar[int] = 20

    def can_read(self, path: Path) -> bool:
        p = Path(path)
        return p.is_dir() and _safe_load_info(p) is not None

    def discover(
        self,
        root: Path,
        *,
        task_patterns: dict[str, list[str]] | None = None,
    ) -> list[EpisodeHandle]:
        root_p = Path(root).resolve()
        if self.can_read(root_p):
            ds_roots = [root_p]
        elif root_p.is_dir():
            ds_roots = []
            for info_file in sorted(root_p.rglob("meta/info.json")):
                rel_parts = info_file.relative_to(root_p).parts
                if any(part in SKIP_DIRS for part in rel_parts):
                    continue
                ds_dir = info_file.parent.parent
                if self.can_read(ds_dir):
                    ds_roots.append(ds_dir)
        else:
            return []

        handles: list[EpisodeHandle] = []
        for ds_root in ds_roots:
            info = _safe_load_info(ds_root)
            if info is None:
                continue
            dataset = validate_id_part(ds_root.name, "dataset")
            ver = str(info.get("codebase_version", "v2.0"))
            fps = float(info["fps"])
            chunks_size = int(info.get("chunks_size", 1000))
            features: dict[str, Any] = info["features"]
            video_keys = [
                k for k, v in features.items() if isinstance(v, dict) and v.get("dtype") == "video"
            ]

            is_v3 = ver.startswith("v3") or (ds_root / "meta" / "episodes").is_dir()
            if is_v3:
                handles.extend(
                    self._discover_v3(
                        ds_root, dataset, info, fps, video_keys, task_patterns
                    )
                )
            else:
                handles.extend(
                    self._discover_v2(
                        ds_root, dataset, info, fps, chunks_size, video_keys, task_patterns
                    )
                )
        return handles

    def _discover_v3(
        self,
        ds_root: Path,
        dataset: str,
        info: dict[str, Any],
        fps: float,
        video_keys: list[str],
        task_patterns: dict[str, list[str]] | None,
    ) -> list[EpisodeHandle]:
        import pyarrow.parquet as pq

        tasks_by_idx: dict[int, str] = {}
        tasks_pq = ds_root / "meta" / "tasks.parquet"
        if tasks_pq.is_file():
            tbl = pq.read_table(tasks_pq)
            df_t = tbl.to_pandas()
            if "task_index" in df_t.columns:
                for idx, row in df_t.iterrows():
                    task_str = str(row["task"]) if "task" in df_t.columns else str(idx)
                    tasks_by_idx[int(row["task_index"])] = task_str

        ep_parquets = sorted((ds_root / "meta" / "episodes").rglob("*.parquet"))
        data_tpl = str(info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"))
        vid_tpl = str(info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"))
        info_mtime = float((ds_root / "meta" / "info.json").stat().st_mtime)

        handles: list[EpisodeHandle] = []
        for ep_pq in ep_parquets:
            df = pq.read_table(ep_pq).to_pandas()
            for _, row in df.iterrows():
                ep_idx = int(row["episode_index"])
                length = int(row["length"])
                d_chunk = int(row.get("data/chunk_index", 0))
                d_file = int(row.get("data/file_index", 0))
                data_rel = data_tpl.format(chunk_index=d_chunk, file_index=d_file)

                ep_task_str = ""
                if "tasks" in row and row["tasks"] is not None:
                    raw_t = row["tasks"]
                    if isinstance(raw_t, (list, tuple, np.ndarray)):
                        if len(raw_t) > 0:
                            ep_task_str = str(raw_t[0]).strip()
                    else:
                        s_t = str(raw_t).strip()
                        if s_t and s_t != "[]":
                            ep_task_str = s_t
                if not ep_task_str and "task_index" in row and row["task_index"] is not None:
                    ep_task_str = tasks_by_idx.get(int(row["task_index"]), "")

                video_specs: dict[str, dict[str, Any]] = {}
                for vk in video_keys:
                    vc_col = f"videos/{vk}/chunk_index"
                    vf_col = f"videos/{vk}/file_index"
                    ft_col = f"videos/{vk}/from_timestamp"
                    tt_col = f"videos/{vk}/to_timestamp"
                    v_chunk = int(row[vc_col]) if vc_col in row else d_chunk
                    v_file = int(row[vf_col]) if vf_col in row else d_file
                    v_rel = vid_tpl.format(
                        video_key=vk, chunk_index=v_chunk, file_index=v_file
                    )
                    from_ts = float(row[ft_col]) if ft_col in row else 0.0
                    to_ts = (
                        float(row[tt_col])
                        if tt_col in row
                        else (from_ts + (length / fps))
                    )
                    video_specs[vk] = {
                        "rel_path": v_rel,
                        "from_timestamp": from_ts,
                        "to_timestamp": to_ts,
                    }

                data_abs = ds_root / data_rel
                mtime = max(
                    info_mtime,
                    float(data_abs.stat().st_mtime) if data_abs.is_file() else 0.0,
                )
                stem = f"episode_{ep_idx:06d}"
                episode_id = f"{dataset}/{stem}"
                task_id = resolve_task_id(dataset, ep_task_str, task_patterns)
                handles.append(
                    EpisodeHandle(
                        episode_id=episode_id,
                        dataset=dataset,
                        stem=stem,
                        task_id=task_id,
                        adapter_type=self.name,
                        source_uri=str(ds_root),
                        source_mtime=mtime,
                        metadata={
                            "version": "v3",
                            "ep_idx": ep_idx,
                            "fps": fps,
                            "length": length,
                            "data_rel": data_rel,
                            "dataset_from_index": (
                                int(row["dataset_from_index"])
                                if "dataset_from_index" in row
                                else None
                            ),
                            "dataset_to_index": (
                                int(row["dataset_to_index"])
                                if "dataset_to_index" in row
                                else None
                            ),
                            "video_specs": video_specs,
                            "ep_task_str": ep_task_str,
                        },
                    )
                )
        return handles

    def _discover_v2(
        self,
        ds_root: Path,
        dataset: str,
        info: dict[str, Any],
        fps: float,
        chunks_size: int,
        video_keys: list[str],
        task_patterns: dict[str, list[str]] | None,
    ) -> list[EpisodeHandle]:
        ep_jsonl = ds_root / "meta" / "episodes.jsonl"
        if not ep_jsonl.is_file():
            return []
        data_tpl = str(
            info.get(
                "data_path",
                "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            )
        )
        vid_tpl = str(
            info.get(
                "video_path",
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            )
        )
        info_mtime = float((ds_root / "meta" / "info.json").stat().st_mtime)

        handles: list[EpisodeHandle] = []
        for line in ep_jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ep_idx = int(rec["episode_index"])
            length = int(rec["length"])
            ep_chunk = ep_idx // max(1, chunks_size)
            data_rel = data_tpl.format(
                episode_chunk=ep_chunk,
                episode_index=ep_idx,
                chunk_index=ep_chunk,
                file_index=ep_idx,
            )
            tasks_val = rec.get("tasks") or []
            ep_task_str = (
                str(tasks_val[0])
                if isinstance(tasks_val, list) and tasks_val
                else str(tasks_val)
            )
            video_specs: dict[str, dict[str, Any]] = {}
            for vk in video_keys:
                v_rel = vid_tpl.format(
                    episode_chunk=ep_chunk,
                    video_key=vk,
                    episode_index=ep_idx,
                    chunk_index=ep_chunk,
                    file_index=ep_idx,
                )
                video_specs[vk] = {
                    "rel_path": v_rel,
                    "from_timestamp": 0.0,
                    "to_timestamp": float(length / fps),
                }

            data_abs = ds_root / data_rel
            mtime = max(
                info_mtime,
                float(data_abs.stat().st_mtime) if data_abs.is_file() else 0.0,
            )
            stem = f"episode_{ep_idx:06d}"
            episode_id = f"{dataset}/{stem}"
            task_id = resolve_task_id(dataset, ep_task_str, task_patterns)
            handles.append(
                EpisodeHandle(
                    episode_id=episode_id,
                    dataset=dataset,
                    stem=stem,
                    task_id=task_id,
                    adapter_type=self.name,
                    source_uri=str(ds_root),
                    source_mtime=mtime,
                    metadata={
                        "version": "v2",
                        "ep_idx": ep_idx,
                        "fps": fps,
                        "length": length,
                        "data_rel": data_rel,
                        "video_specs": video_specs,
                        "ep_task_str": ep_task_str,
                    },
                )
            )
        return handles

    def open(self, handle: EpisodeHandle) -> EpisodeView:
        import pyarrow.parquet as pq

        ds_root = handle.source_path
        if not ds_root.exists():
            raise FileNotFoundError(f"LeRobot dataset directory not found: {ds_root}")
        info = _safe_load_info(ds_root)
        if info is None:
            raise ValueError(f"Invalid LeRobot dataset at {ds_root}")
        meta_md = dict(handle.metadata)
        features: dict[str, Any] = info["features"]
        has_video_feat = any(
            isinstance(v, dict) and v.get("dtype") == "video"
            for v in features.values()
        )
        needs_discover = (
            not meta_md.get("data_rel")
            or "ep_idx" not in meta_md
            or meta_md.get("ep_idx") is None
            or (has_video_feat and not meta_md.get("video_specs"))
        )
        if needs_discover:
            discovered = self.discover(ds_root)
            matched = next(
                (h for h in discovered if h.episode_id == handle.episode_id),
                None,
            )
            if matched is None:
                matched = next(
                    (h for h in discovered if h.stem == handle.stem),
                    None,
                )
            if matched is None and meta_md.get("ep_idx") is not None:
                target_idx = int(meta_md["ep_idx"])
                matched = next(
                    (
                        h
                        for h in discovered
                        if h.metadata.get("ep_idx") is not None
                        and int(h.metadata["ep_idx"]) == target_idx
                    ),
                    None,
                )
            if matched is None:
                raise FileNotFoundError(
                    f"LeRobot episode {handle.episode_id!r} not found under {ds_root}"
                )
            merged = dict(matched.metadata)
            for k, v in meta_md.items():
                if v is not None and v != {} and v != "":
                    merged[k] = v
            meta_md = merged

        ep_idx = int(meta_md["ep_idx"])
        fps = float(meta_md.get("fps", info["fps"]))
        data_path = ds_root / str(meta_md["data_rel"])
        if not data_path.is_file():
            raise FileNotFoundError(f"LeRobot Parquet missing: {data_path}")

        pq_schema = pq.read_schema(data_path)
        has_ep_col = "episode_index" in pq_schema.names
        from_idx = meta_md.get("dataset_from_index")
        to_idx = meta_md.get("dataset_to_index")

        if has_ep_col:
            tbl = pq.read_table(
                data_path,
                filters=[("episode_index", "==", ep_idx)],
            )
            df = tbl.to_pandas().reset_index(drop=True)
        else:
            df = pq.read_table(data_path).to_pandas().reset_index(drop=True)

        if (df.empty or not has_ep_col) and from_idx is not None and to_idx is not None:
            full_df = (
                pq.read_table(data_path).to_pandas().reset_index(drop=True)
                if has_ep_col
                else df
            )
            if has_ep_col and "episode_index" in full_df.columns:
                sub = full_df[full_df["episode_index"] == ep_idx].reset_index(drop=True)
                if not sub.empty:
                    df = sub
            if df.empty or not has_ep_col:
                f_i, t_i = int(from_idx), int(to_idx)
                if "index" in full_df.columns:
                    df = full_df[
                        (full_df["index"] >= f_i) & (full_df["index"] < t_i)
                    ].reset_index(drop=True)
                elif 0 <= f_i < t_i <= len(full_df):
                    df = full_df.iloc[f_i:t_i].reset_index(drop=True)

        if df.empty and len(pq_schema.names) > 0:
            raise ValueError(f"Episode {ep_idx} rows missing in {data_path}")

        T = len(df) if len(df) > 0 else int(meta_md["length"])
        if T <= 0:
            raise ValueError(f"Empty episode {handle.episode_id}")
        duration_s = float(T / fps)

        features: dict[str, Any] = info["features"]
        cameras: list[CameraSpec] = []
        video_entries: dict[str, dict[str, Any]] = {}
        image_frames: dict[str, list[bytes]] = {}
        video_specs: dict[str, dict[str, Any]] = meta_md.get("video_specs") or {}

        for feat_key, feat_spec in features.items():
            if not isinstance(feat_spec, dict):
                continue
            dtype = feat_spec.get("dtype")
            if dtype not in ("video", "image"):
                continue
            cam_name = _strip_image_prefix(feat_key)
            role = infer_camera_role(cam_name)
            if dtype == "video":
                vspec = video_specs.get(feat_key)
                if not vspec:
                    continue
                vpath = ds_root / vspec["rel_path"]
                if not vpath.is_file():
                    raise FileNotFoundError(f"LeRobot video missing: {vpath}")
                w, h, file_frames, file_fps = probe_video_cv2(vpath)
                cameras.append(
                    CameraSpec(
                        name=cam_name,
                        role=role,
                        width=w,
                        height=h,
                        fps=fps,
                        source_type="lerobot_mp4",
                        is_virtual_crop=False,
                        source_stream=feat_key,
                    )
                )
                eff_fps = file_fps if file_fps > 0 else fps
                from_ts = float(vspec.get("from_timestamp", 0.0))
                start_frame = max(0, min(file_frames - 1, int(round(from_ts * eff_fps))))
                raw_to_ts = vspec.get("to_timestamp")
                if raw_to_ts is not None and float(raw_to_ts) > from_ts:
                    to_end = int(round(float(raw_to_ts) * eff_fps)) - 1
                else:
                    ep_len = int(meta_md.get("length") or T)
                    to_end = start_frame + max(1, int(round(ep_len * eff_fps / fps))) - 1
                ep_end = start_frame + max(1, int(round(T * eff_fps / fps))) - 1
                end_frame = max(start_frame, min(file_frames - 1, to_end, ep_end))
                video_entries[cam_name] = {
                    "path": str(vpath),
                    "file_frames": file_frames,
                    "fps": eff_fps,
                    "from_timestamp": from_ts,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                }
            elif dtype == "image" and feat_key in df.columns:
                raw_col: list[bytes] = []
                for item in df[feat_key]:
                    if isinstance(item, dict) and item.get("bytes") is not None:
                        raw_col.append(bytes(item["bytes"]))
                    elif isinstance(item, (bytes, bytearray, memoryview)):
                        raw_col.append(bytes(item))
                    else:
                        raise ValueError(f"Unsupported image cell type in {feat_key}: {type(item)}")
                img0 = cv2.imdecode(
                    np.frombuffer(raw_col[0], dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if img0 is None:
                    raise ValueError(f"Failed decoding image[0] in {feat_key}")
                h0, w0 = int(img0.shape[0]), int(img0.shape[1])
                cameras.append(
                    CameraSpec(
                        name=cam_name,
                        role=role,
                        width=w0,
                        height=h0,
                        fps=fps,
                        source_type="lerobot_image",
                        is_virtual_crop=False,
                        source_stream=feat_key,
                    )
                )
                image_frames[cam_name] = raw_col

        if not cameras:
            raise ValueError(f"No video or image features found for {handle.episode_id}")
        camera_roles = build_camera_roles(cameras)

        # Kinematics extraction
        kin_data: KinematicsData | None = None
        has_state = "observation.state" in df.columns
        has_action = "action" in df.columns
        if has_state or has_action:
            if "timestamp" in df.columns and len(df["timestamp"]) == T:
                ts_raw = df["timestamp"].to_numpy(dtype=np.float64)
                timestamps_s = ts_raw - ts_raw[0]
            else:
                timestamps_s = np.arange(T, dtype=np.float64) / fps

            raw_state = (
                np.stack([np.asarray(x, dtype=np.float32) for x in df["observation.state"]])
                if has_state
                else None
            )
            raw_action = (
                np.stack([np.asarray(x, dtype=np.float32) for x in df["action"]])
                if has_action
                else None
            )
            tactile = (
                np.stack(
                    [np.asarray(x, dtype=np.float32) for x in df["observation.tactile_force"]]
                )
                if "observation.tactile_force" in df.columns
                else None
            )

            raw_state_dim = (
                int(raw_state.shape[1])
                if (raw_state is not None and raw_state.ndim == 2)
                else None
            )
            raw_action_dim = (
                int(raw_action.shape[1])
                if (raw_action is not None and raw_action.ndim == 2)
                else None
            )
            state_names = _extract_feature_names(
                features.get("observation.state"), raw_state_dim
            )
            action_names = _extract_feature_names(
                features.get("action"), raw_action_dim
            )
            parsed = _parse_named_state(raw_state, list(state_names or []))
            valid_mask = (
                np.isfinite(raw_state).all(axis=1)
                if raw_state is not None
                else np.ones(T, dtype=bool)
            )
            kin_data = KinematicsData(
                timestamps_s=timestamps_s,
                eef_pos=parsed.get("eef_pos"),
                eef_rotvec=parsed.get("eef_rotvec"),
                hand_q=parsed.get("hand_q"),
                torso_q=parsed.get("torso_q"),
                chassis_pose=parsed.get("chassis_pose"),
                raw_state=raw_state,
                raw_action=raw_action,
                tactile=tactile,
                valid_mask=valid_mask,
                arm_q=parsed.get("arm_q"),
                head_q=parsed.get("head_q"),
                eef_quat_wxyz=parsed.get("eef_quat_wxyz"),
                state_names=state_names,
                action_names=action_names,
            )

        has_kin = kin_data is not None
        kin_spec = KinematicsSpec(
            has_kinematics=has_kin,
            n_arms=2 if (kin_data and kin_data.eef_pos is not None) else 0,
            arm_dof=(
                int(kin_data.arm_q.shape[1] // 2)
                if (kin_data and kin_data.arm_q is not None)
                else 0
            ),
            hand_dof=(
                int(kin_data.hand_q.shape[1])
                if (kin_data and kin_data.hand_q is not None)
                else 0
            ),
            torso_dof=(
                int(kin_data.torso_q.shape[1])
                if (kin_data and kin_data.torso_q is not None)
                else 0
            ),
            has_chassis=bool(kin_data and kin_data.chassis_pose is not None),
            has_tactile=bool(kin_data and kin_data.tactile is not None),
            tactile_dim=(
                int(kin_data.tactile.shape[1])
                if (kin_data and kin_data.tactile is not None)
                else 0
            ),
            state_dim=(
                int(kin_data.raw_state.shape[1])
                if (kin_data and kin_data.raw_state is not None)
                else 0
            ),
            has_action_cmd=bool(kin_data and kin_data.raw_action is not None),
            source_hz={"lerobot": fps},
        )

        media = LeRobotMediaSource(
            cameras=cameras,
            video_entries=video_entries,
            image_frames=image_frames,
            fps=fps,
            duration_s=duration_s,
        )
        fingerprint = compute_episode_fingerprint(
            handle.source_uri,
            handle.source_mtime,
            extra=f"ep={ep_idx}|T={T}|ver={meta_md.get('version')}",
        )
        meta = EpisodeMeta(
            episode_id=handle.episode_id,
            dataset=handle.dataset,
            stem=handle.stem,
            task_id=handle.task_id,
            adapter_type="lerobot",
            source_uri=handle.source_uri,
            source_mtime=handle.source_mtime,
            episode_fingerprint=fingerprint,
            duration_s=duration_s,
            fps=fps,
            frame_count=T,
            cameras=cameras,
            camera_roles=camera_roles,
            kinematics=kin_spec,
            recorded_success=None,
            extra_meta={
                "codebase_version": str(info.get("codebase_version", "")),
                "ep_idx": ep_idx,
                "ep_task_str": meta_md.get("ep_task_str", ""),
            },
        )
        return EpisodeView(meta=meta, media=media, kinematics=kin_data)


def _extract_feature_names(
    feat_spec: Any, expected_dim: int | None = None
) -> list[str] | None:
    """Extract flat list of feature dimension names from LeRobot info.json feature metadata."""
    if not isinstance(feat_spec, dict):
        return None
    raw = feat_spec.get("names")
    if raw is None:
        return None

    names: list[str] = []
    if isinstance(raw, list):
        if len(raw) == 1 and isinstance(raw[0], list):
            raw = raw[0]
        if all(not isinstance(item, (list, dict)) for item in raw):
            names = [str(item) for item in raw if str(item)]
    elif isinstance(raw, dict):
        for val in raw.values():
            if isinstance(val, list):
                names.extend(str(item) for item in val if str(item))

    if not names:
        return None
    if expected_dim is not None and len(names) != expected_dim:
        return None
    return names


def _parse_named_state(
    raw_state: np.ndarray | None, names: list[str]
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if raw_state is None or not names or len(names) != raw_state.shape[1]:
        return out
    col = {nm: i for i, nm in enumerate(names)}

    if all(k in col for k in ("chassis_dx", "chassis_dy", "chassis_dyaw")):
        out["chassis_pose"] = raw_state[
            :, [col["chassis_dx"], col["chassis_dy"], col["chassis_dyaw"]]
        ].astype(np.float64)

    torso_cols = [col[f"torso_{i}"] for i in range(4) if f"torso_{i}" in col]
    if torso_cols:
        out["torso_q"] = raw_state[:, torso_cols].astype(np.float64)

    arm_l = [col[f"arm_left_{i}"] for i in range(7) if f"arm_left_{i}" in col]
    arm_r = [col[f"arm_right_{i}"] for i in range(7) if f"arm_right_{i}" in col]
    if arm_l and arm_r:
        out["arm_q"] = raw_state[:, arm_l + arm_r].astype(np.float64)

    if "head_0" in col and "head_1" in col:
        out["head_q"] = raw_state[:, [col["head_0"], col["head_1"]]].astype(np.float64)

    if all(
        f"ee_{side}_{ax}" in col
        for side in ("left", "right")
        for ax in ("x", "y", "z")
    ):
        el = raw_state[:, [col["ee_left_x"], col["ee_left_y"], col["ee_left_z"]]].astype(np.float64)
        er = raw_state[:, [col["ee_right_x"], col["ee_right_y"], col["ee_right_z"]]].astype(np.float64)
        out["eef_pos"] = np.stack([el, er], axis=1)

    if all(
        f"ee_{side}_{q}" in col
        for side in ("left", "right")
        for q in ("qw", "qx", "qy", "qz")
    ):
        rotvecs = []
        quats = []
        for side in ("left", "right"):
            qwxyz = raw_state[
                :,
                [
                    col[f"ee_{side}_qw"],
                    col[f"ee_{side}_qx"],
                    col[f"ee_{side}_qy"],
                    col[f"ee_{side}_qz"],
                ],
            ].astype(np.float64)
            norms = np.linalg.norm(qwxyz, axis=1)
            safe = np.where(norms[:, None] > 1e-6, qwxyz / np.maximum(norms[:, None], 1e-6), [1.0, 0.0, 0.0, 0.0])
            for i in range(1, len(safe)):
                if float(np.dot(safe[i], safe[i - 1])) < 0.0:
                    safe[i] = -safe[i]
            rv = Rotation.from_quat(safe[:, [1, 2, 3, 0]]).as_rotvec()
            rotvecs.append(rv)
            quats.append(safe)
        out["eef_rotvec"] = np.stack(rotvecs, axis=1)
        out["eef_quat_wxyz"] = np.stack(quats, axis=1)

    hand_l = [col[k] for k in names if k.startswith("hand_left_")]
    hand_r = [col[k] for k in names if k.startswith("hand_right_")]
    if hand_l or hand_r:
        out["hand_q"] = raw_state[:, hand_l + hand_r].astype(np.float64)

    return out
