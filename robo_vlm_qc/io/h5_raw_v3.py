"""Read-only InputAdapter for raw_only_v3 HDF5 multi-stream trajectories."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, ClassVar

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ..models.episode import CameraSpec, EpisodeMeta, KinematicsSpec
from .base import (
    LABEL_DIRS,
    SKIP_DIRS,
    EpisodeHandle,
    EpisodeView,
    InputAdapter,
    KinematicsData,
    build_camera_roles,
    compute_episode_fingerprint,
    dataset_of,
    infer_camera_role,
    resolve_task_id,
    validate_id_part,
)

logger = logging.getLogger(__name__)

try:
    import ctypes as _ctypes

    _LIBC: Any = _ctypes.CDLL("libc.so.6")
except Exception:
    _LIBC = None


def _malloc_trim() -> None:
    if _LIBC is not None:
        try:
            _LIBC.malloc_trim(0)
        except Exception:
            pass


def _configure_large_mmap_threshold() -> None:
    # Prevent glibc's dynamic M_MMAP_THRESHOLD from trapping freed 2.7-12.4 MB
    # decoded video frames inside per-thread arenas across multi-threaded reads.
    if _LIBC is not None:
        try:
            _LIBC.mallopt(-3, 512 * 1024)
        except Exception:
            pass

_DEFAULT_DIM_NAMES = [
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
]


def _strictly_increasing_indices(ts: np.ndarray) -> np.ndarray:
    if len(ts) <= 1:
        return np.arange(len(ts), dtype=np.int64)
    keep = np.empty(len(ts), dtype=bool)
    keep[0] = True
    keep[1:] = ts[1:] > ts[:-1]
    if not keep.all():
        # Filter iteratively in case a duplicate drop leaves non-monotonic step
        idxs = [0]
        last = ts[0]
        for i in range(1, len(ts)):
            if ts[i] > last:
                idxs.append(i)
                last = ts[i]
        return np.asarray(idxs, dtype=np.int64)
    return np.arange(len(ts), dtype=np.int64)


def _resample_quat_slerp(
    robot_ts: np.ndarray,
    q_wxyz_raw: np.ndarray,
    grid_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sign-continuous Slerp resampling returning (rotvec (T,3), quat_wxyz (T,4), valid (N,))."""
    norms = np.linalg.norm(q_wxyz_raw, axis=1)
    valid = np.isfinite(q_wxyz_raw).all(axis=1) & (norms > 1e-6)
    q_wxyz = q_wxyz_raw.copy()
    if not valid.any():
        q_wxyz[:] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        if not valid.all():
            first_ok = int(np.argmax(valid))
            last_ok = q_wxyz[first_ok]
            for i in range(len(q_wxyz)):
                if valid[i]:
                    last_ok = q_wxyz[i]
                else:
                    q_wxyz[i] = last_ok
            norms = np.linalg.norm(q_wxyz, axis=1)
        q_wxyz /= norms[:, None]

    # Single-pass sign continuity
    for i in range(1, len(q_wxyz)):
        if float(np.dot(q_wxyz[i], q_wxyz[i - 1])) < 0.0:
            q_wxyz[i] = -q_wxyz[i]

    # SciPy uses [qx, qy, qz, qw]
    q_xyzw = q_wxyz[:, [1, 2, 3, 0]]
    rot = Rotation.from_quat(q_xyzw)
    slerp = Slerp(robot_ts, rot)
    eval_t = np.clip(grid_t, robot_ts[0], robot_ts[-1])
    rot_30 = slerp(eval_t)
    rotvec_30 = rot_30.as_rotvec()
    q_xyzw_30 = rot_30.as_quat()
    q_wxyz_30 = q_xyzw_30[:, [3, 0, 1, 2]]
    return rotvec_30, q_wxyz_30, valid


def _repair_and_resample_tactile(
    hand_ts: np.ndarray,
    tf_raw: np.ndarray,
    tv_raw: np.ndarray,
    grid_t: np.ndarray,
    min_invalid_gap_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Repair short tactile dropouts on native grid before 30Hz resampling."""
    n = len(hand_ts)
    tf = tf_raw.astype(np.float64).reshape(n, -1).copy()
    tv = tv_raw.astype(bool).reshape(n) & np.isfinite(tf).all(axis=1)

    unrecoverable_native = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if tv[i]:
            i += 1
            continue
        i0 = i
        while i < n and not tv[i]:
            i += 1
        i1 = i - 1
        t_left = float(hand_ts[i0 - 1] if i0 > 0 else hand_ts[0])
        t_right = float(hand_ts[i1 + 1] if i1 + 1 < n else hand_ts[-1])
        if (t_right - t_left) >= min_invalid_gap_s:
            unrecoverable_native[i0 : i1 + 1] = True

    if tv.any():
        t_valid = hand_ts[tv]
        for c in range(tf.shape[1]):
            tf[:, c] = np.interp(hand_ts, t_valid, tf[tv, c])
    else:
        tf[:] = 0.0
        unrecoverable_native[:] = True

    out = np.empty((len(grid_t), tf.shape[1]), dtype=np.float32)
    for c in range(tf.shape[1]):
        out[:, c] = np.interp(grid_t, hand_ts, tf[:, c]).astype(np.float32)

    # Project native unrecoverable intervals onto 30Hz grid
    if not unrecoverable_native.any():
        force_valid_30 = np.ones(len(grid_t), dtype=bool)
    else:
        nearest_idx = np.clip(
            np.searchsorted(hand_ts, grid_t, side="left"), 0, n - 1
        )
        prev_idx = np.clip(nearest_idx - 1, 0, n - 1)
        choose_prev = np.abs(grid_t - hand_ts[prev_idx]) <= np.abs(
            hand_ts[nearest_idx] - grid_t
        )
        mapped = np.where(choose_prev, prev_idx, nearest_idx)
        force_valid_30 = ~unrecoverable_native[mapped]

    return out, force_valid_30


class H5MediaSource:
    """Thread-local O(K) JPEG MediaSource with virtual SBS stereo splitting."""

    def __init__(
        self,
        h5_path: Path,
        cameras: list[CameraSpec],
        cam_specs_internal: dict[str, tuple[str, str | None]],
        row_to_idx: dict[str, np.ndarray],
        fps: float,
        duration_s: float,
        frame_count: int,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.cameras = cameras
        # cam_name -> (raw_h5_cam_name, crop_mode: None | "left_half" | "right_half")
        self._cam_map = cam_specs_internal
        self._role_to_cam: dict[str, str] = {}
        for c in cameras:
            if c.role not in self._role_to_cam:
                self._role_to_cam[c.role] = c.name
        self._row_to_idx = row_to_idx
        self.fps = float(fps)
        self.duration_s = float(duration_s)
        self._T = int(frame_count)
        self._tls = threading.local()
        self._files_lock = threading.Lock()
        self._h5_lock = threading.Lock()
        self._h5_file: h5py.File | None = None
        self._opened_files: list[h5py.File] = []
        self._decode_cache_lock = threading.Lock()
        self._decode_cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self._decode_cache_max = 2
        _configure_large_mmap_threshold()

    def _get_h5(self) -> h5py.File:
        f = self._h5_file
        if f is not None and f.id.valid:
            return f
        with self._files_lock:
            f = self._h5_file
            if f is not None and f.id.valid:
                return f
            f = h5py.File(
                str(self.h5_path),
                "r",
                swmr=False,
                rdcc_nbytes=256 * 1024,
            )
            self._h5_file = f
            self._opened_files.append(f)
            return f

    def _resolve_camera(self, camera: str) -> str:
        if camera in self._cam_map:
            return camera
        if camera in self._role_to_cam:
            return self._role_to_cam[camera]
        raise KeyError(
            f"Unknown camera {camera!r}; available: {list(self._cam_map)}"
        )

    def _decode_raw(
        self, raw_cam: str, raw_idx: int, *, use_cache: bool = True
    ) -> np.ndarray:
        key = (raw_cam, raw_idx)
        if use_cache and self._decode_cache_max > 0:
            with self._decode_cache_lock:
                cached = self._decode_cache.get(key)
                if cached is not None:
                    self._decode_cache.move_to_end(key)
                    return cached

        h5 = self._get_h5()
        with self._h5_lock:
            raw_entry = h5[f"/camera/{raw_cam}/jpeg"][raw_idx]
        buf = (
            raw_entry
            if isinstance(raw_entry, np.ndarray)
            else np.frombuffer(bytes(raw_entry), dtype=np.uint8)
        )
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        del buf, raw_entry
        if bgr is None:
            raise RuntimeError(
                f"Failed to decode JPEG for {raw_cam}[{raw_idx}] in {self.h5_path}"
            )
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB, dst=bgr)
        rgb = bgr if bgr.flags["C_CONTIGUOUS"] else np.ascontiguousarray(bgr, dtype=np.uint8)

        if use_cache and self._decode_cache_max > 0:
            with self._decode_cache_lock:
                self._decode_cache[key] = rgb
                self._decode_cache.move_to_end(key)
                while len(self._decode_cache) > self._decode_cache_max:
                    self._decode_cache.popitem(last=False)
        return rgb

    def read_frame_rgb(self, camera: str, t_s: float) -> np.ndarray:
        t_val = float(t_s)
        if not np.isfinite(t_val):
            raise ValueError(f"Non-finite timestamp t_s={t_s!r}")
        cam_name = self._resolve_camera(camera)
        raw_cam, crop_mode = self._cam_map[cam_name]
        row = int(np.clip(round(t_val * self.fps), 0, self._T - 1))
        raw_idx = int(self._row_to_idx[raw_cam][row])
        rgb = self._decode_raw(raw_cam, raw_idx, use_cache=True)
        if crop_mode == "left_half":
            w = rgb.shape[1]
            return np.ascontiguousarray(rgb[:, : w // 2], dtype=np.uint8)
        if crop_mode == "right_half":
            w = rgb.shape[1]
            return np.ascontiguousarray(rgb[:, w // 2 :], dtype=np.uint8)
        return rgb

    def read_frames_batch(
        self, camera: str, times_s: list[float]
    ) -> list[np.ndarray]:
        if not times_s:
            return []
        t_floats = [float(t) for t in times_s]
        if not all(np.isfinite(t) for t in t_floats):
            raise ValueError("Non-finite timestamp in times_s")
        cam_name = self._resolve_camera(camera)
        raw_cam, crop_mode = self._cam_map[cam_name]
        raw_indices = [
            int(
                self._row_to_idx[raw_cam][
                    int(np.clip(round(t * self.fps), 0, self._T - 1))
                ]
            )
            for t in t_floats
        ]
        decoded_raw: dict[int, np.ndarray] = {}
        for r_idx in sorted(set(raw_indices)):
            rgb = self._decode_raw(raw_cam, r_idx, use_cache=False)
            if crop_mode == "left_half":
                w = rgb.shape[1]
                cropped = np.ascontiguousarray(rgb[:, : w // 2], dtype=np.uint8)
                del rgb
                decoded_raw[r_idx] = cropped
            elif crop_mode == "right_half":
                w = rgb.shape[1]
                cropped = np.ascontiguousarray(rgb[:, w // 2 :], dtype=np.uint8)
                del rgb
                decoded_raw[r_idx] = cropped
            else:
                decoded_raw[r_idx] = rgb
        out = [decoded_raw[r_idx] for r_idx in raw_indices]
        decoded_raw.clear()
        _malloc_trim()
        return out

    def close(self) -> None:
        with self._decode_cache_lock:
            self._decode_cache.clear()
        with self._files_lock:
            for f in self._opened_files:
                try:
                    f.close()
                except Exception:
                    pass
            self._opened_files.clear()
            self._h5_file = None
        if hasattr(self._tls, "h5"):
            delattr(self._tls, "h5")
        _malloc_trim()


class H5RawV3Adapter(InputAdapter):
    """InputAdapter for raw_only_v3 multi-stream HDF5 recordings."""

    name: ClassVar[str] = "h5_raw_v3"
    priority: ClassVar[int] = 10

    def __init__(
        self,
        *,
        hand_mapping: dict[str, str] | None = None,
        min_invalid_gap_s: float = 0.5,
    ) -> None:
        self.default_hand_mapping = dict(
            hand_mapping or {"left": "hand_a", "right": "hand_b"}
        )
        self.min_invalid_gap_s = float(min_invalid_gap_s)

    def can_read(self, path: Path) -> bool:
        p = Path(path)
        if not p.is_file():
            return False
        if p.suffix.lower() not in (".h5", ".hdf5"):
            return False
        if ".partial" in p.name:
            return False
        try:
            with h5py.File(str(p), "r") as f:
                return (
                    "/robot/state" in f
                    and "/robot/timestamps" in f
                    and "/hand" in f
                    and "/camera" in f
                )
        except Exception:
            return False

    def discover(
        self,
        root: Path,
        *,
        task_patterns: dict[str, list[str]] | None = None,
    ) -> list[EpisodeHandle]:
        root_p = Path(root).resolve()
        if root_p.is_file():
            candidates = [root_p] if self.can_read(root_p) else []
        elif root_p.is_dir():
            raw_files = sorted(
                set(root_p.rglob("*.h5")) | set(root_p.rglob("*.hdf5"))
            )
            candidates = []
            for p in raw_files:
                rel_parts = p.relative_to(root_p).parts
                if any(part in SKIP_DIRS or ".partial" in part for part in rel_parts):
                    continue
                if self.can_read(p):
                    candidates.append(p)
        else:
            return []

        handles: list[EpisodeHandle] = []
        for p in candidates:
            dataset = dataset_of(p)
            stem = validate_id_part(p.stem, "stem")
            episode_id = f"{dataset}/{stem}"
            mtime = float(p.stat().st_mtime)
            recorded_success = LABEL_DIRS.get(p.parent.name.lower())
            h5_task_name = ""
            schema_attr = ""
            try:
                with h5py.File(str(p), "r") as f:
                    h5_task_name = str(f.attrs.get("task_name", ""))
                    schema_attr = str(f.attrs.get("schema", ""))
            except Exception:
                pass
            task_id = resolve_task_id(dataset, h5_task_name, task_patterns)
            handles.append(
                EpisodeHandle(
                    episode_id=episode_id,
                    dataset=dataset,
                    stem=stem,
                    task_id=task_id,
                    adapter_type=self.name,
                    source_uri=str(p.resolve()),
                    source_mtime=mtime,
                    metadata={
                        "recorded_success": recorded_success,
                        "h5_task_name": h5_task_name,
                        "schema": schema_attr,
                    },
                )
            )
        return handles

    def open(self, handle: EpisodeHandle) -> EpisodeView:
        h5_path = handle.source_path
        hand_map = dict(
            handle.metadata.get("hand_mapping") or self.default_hand_mapping
        )
        left_hand = hand_map.get("left", "hand_a")
        right_hand = hand_map.get("right", "hand_b")
        min_gap_s = float(
            handle.metadata.get("min_invalid_gap_s", self.min_invalid_gap_s)
        )

        with h5py.File(str(h5_path), "r", rdcc_nbytes=256 * 1024) as f:
            robot_ts_raw = f["/robot/timestamps"][:].astype(np.float64)
            robot_state_raw = f["/robot/state"][:]
            hand_ts_raw = f["/hand/timestamps"][:].astype(np.float64)

            if len(robot_ts_raw) < 2 or len(hand_ts_raw) < 2:
                raise ValueError(f"Insufficient timestamps in {h5_path}")
            if not (np.isfinite(robot_ts_raw).all() and np.isfinite(hand_ts_raw).all()):
                raise ValueError(f"Non-finite timestamps in {h5_path}")

            r_keep = _strictly_increasing_indices(robot_ts_raw)
            robot_ts = robot_ts_raw[r_keep]
            robot_state = robot_state_raw[r_keep]

            h_keep = _strictly_increasing_indices(hand_ts_raw)
            hand_ts = hand_ts_raw[h_keep]

            cam_group = f["/camera"]
            raw_cam_names = sorted(cam_group.keys())
            if not raw_cam_names:
                raise ValueError(f"No cameras found under /camera in {h5_path}")

            cam_ts_map: dict[str, np.ndarray] = {}
            all_starts = [float(robot_ts[0]), float(hand_ts[0])]
            all_ends = [float(robot_ts[-1]), float(hand_ts[-1])]

            for cname in raw_cam_names:
                cts = cam_group[cname]["timestamps"][:].astype(np.float64)
                if len(cts) < 1 or not np.isfinite(cts).all():
                    raise ValueError(f"Invalid timestamps for camera {cname!r}")
                cam_ts_map[cname] = cts
                all_starts.append(float(cts[0]))
                all_ends.append(float(cts[-1]))

            t_start = max(all_starts)
            t_end = min(all_ends)
            if t_end <= t_start:
                raise ValueError(
                    f"Streams do not overlap in {h5_path}: t_start={t_start}, t_end={t_end}"
                )

            fps = 30.0
            T = int(np.floor((t_end - t_start) * fps)) + 1
            grid_t = t_start + np.arange(T, dtype=np.float64) / fps
            timestamps_s = grid_t - t_start
            duration_s = float(T / fps)

            # Dynamic column name resolution
            dim_names_attr = f.attrs.get("robot_state_dim_names", "")
            if isinstance(dim_names_attr, bytes):
                dim_names_attr = dim_names_attr.decode("utf-8")
            dim_names = (
                [s.strip() for s in str(dim_names_attr).split(",") if s.strip()]
                if dim_names_attr
                else list(_DEFAULT_DIM_NAMES)
            )
            col_map = {name: i for i, name in enumerate(dim_names)}

            def _interp_cols(names: list[str]) -> np.ndarray:
                arr = np.empty((T, len(names)), dtype=np.float64)
                for j, nm in enumerate(names):
                    idx = col_map[nm]
                    col = robot_state[:, idx].astype(np.float64)
                    arr[:, j] = np.interp(grid_t, robot_ts, col)
                return arr

            chassis_pose = _interp_cols(["chassis_dx", "chassis_dy", "chassis_dyaw"])
            torso_q = _interp_cols([f"torso_{i}" for i in range(4)])
            arm_left = _interp_cols([f"arm_left_{i}" for i in range(7)])
            arm_right = _interp_cols([f"arm_right_{i}" for i in range(7)])
            arm_q = np.concatenate([arm_left, arm_right], axis=1)
            head_q = _interp_cols(["head_0", "head_1"])
            ee_l_pos = _interp_cols(["ee_left_x", "ee_left_y", "ee_left_z"])
            ee_r_pos = _interp_cols(["ee_right_x", "ee_right_y", "ee_right_z"])
            eef_pos = np.stack([ee_l_pos, ee_r_pos], axis=1)

            ql_raw = robot_state[
                :,
                [
                    col_map["ee_left_qw"],
                    col_map["ee_left_qx"],
                    col_map["ee_left_qy"],
                    col_map["ee_left_qz"],
                ],
            ].astype(np.float64)
            qr_raw = robot_state[
                :,
                [
                    col_map["ee_right_qw"],
                    col_map["ee_right_qx"],
                    col_map["ee_right_qy"],
                    col_map["ee_right_qz"],
                ],
            ].astype(np.float64)

            rotvec_l, qwxyz_l, qvalid_l = _resample_quat_slerp(robot_ts, ql_raw, grid_t)
            rotvec_r, qwxyz_r, qvalid_r = _resample_quat_slerp(robot_ts, qr_raw, grid_t)
            eef_rotvec = np.stack([rotvec_l, rotvec_r], axis=1)
            eef_quat_wxyz = np.stack([qwxyz_l, qwxyz_r], axis=1)

            robot_finite_native = (
                np.isfinite(robot_state).all(axis=1) & qvalid_l & qvalid_r
            )
            robot_valid_30 = _nearest_bool_mask(robot_ts, robot_finite_native, grid_t)

            # Hands & Tactile
            hand_grp = f["/hand"]
            hl_rad_raw = hand_grp[f"{left_hand}_radian"][:][h_keep].astype(np.float64)
            hr_rad_raw = hand_grp[f"{right_hand}_radian"][:][h_keep].astype(np.float64)
            hand_native_valid = (
                np.isfinite(hl_rad_raw).all(axis=1)
                & np.isfinite(hr_rad_raw).all(axis=1)
            )
            hand_l_30 = np.empty((T, hl_rad_raw.shape[1]), dtype=np.float64)
            for c in range(hl_rad_raw.shape[1]):
                hand_l_30[:, c] = np.interp(grid_t, hand_ts, hl_rad_raw[:, c])
            hand_r_30 = np.empty((T, hr_rad_raw.shape[1]), dtype=np.float64)
            for c in range(hr_rad_raw.shape[1]):
                hand_r_30[:, c] = np.interp(grid_t, hand_ts, hr_rad_raw[:, c])
            hand_q = np.concatenate([hand_l_30, hand_r_30], axis=1)
            hand_valid_30 = _nearest_bool_mask(hand_ts, hand_native_valid, grid_t)

            tactile: np.ndarray | None = None
            force_valid_30 = np.ones(T, dtype=bool)
            if f"{left_hand}_tactile_force" in hand_grp and f"{right_hand}_tactile_force" in hand_grp:
                tl_raw = hand_grp[f"{left_hand}_tactile_force"][:][h_keep]
                tr_raw = hand_grp[f"{right_hand}_tactile_force"][:][h_keep]
                vl_raw = (
                    hand_grp[f"{left_hand}_tactile_valid"][:][h_keep]
                    if f"{left_hand}_tactile_valid" in hand_grp
                    else np.ones(len(hand_ts), dtype=bool)
                )
                vr_raw = (
                    hand_grp[f"{right_hand}_tactile_valid"][:][h_keep]
                    if f"{right_hand}_tactile_valid" in hand_grp
                    else np.ones(len(hand_ts), dtype=bool)
                )
                tl_30, fl_ok = _repair_and_resample_tactile(
                    hand_ts, tl_raw, vl_raw, grid_t, min_gap_s
                )
                tr_30, fr_ok = _repair_and_resample_tactile(
                    hand_ts, tr_raw, vr_raw, grid_t, min_gap_s
                )
                tactile = np.concatenate([tl_30, tr_30], axis=1).astype(np.float32)
                force_valid_30 = fl_ok & fr_ok

            raw_state = np.concatenate(
                [
                    chassis_pose,
                    torso_q,
                    arm_left,
                    arm_right,
                    head_q,
                    ee_l_pos,
                    qwxyz_l,
                    ee_r_pos,
                    qwxyz_r,
                    hand_l_30,
                    hand_r_30,
                ],
                axis=1,
            ).astype(np.float32)

            raw_action: np.ndarray | None = None
            if "/robot/action" in f:
                act_raw = f["/robot/action"][:]
                if len(act_raw) == len(robot_ts_raw):
                    act_f = act_raw[r_keep].astype(np.float64)
                    if act_f.ndim == 1:
                        act_f = act_f[:, None]
                    act_30 = np.empty((T, act_f.shape[1]), dtype=np.float32)
                    for c in range(act_f.shape[1]):
                        act_30[:, c] = np.interp(grid_t, robot_ts, act_f[:, c]).astype(
                            np.float32
                        )
                    raw_action = act_30
                elif len(act_raw) == T:
                    act_f = act_raw.astype(np.float32)
                    raw_action = act_f[:, None] if act_f.ndim == 1 else act_f
                elif len(act_raw) >= 2:
                    act_f = act_raw.astype(np.float64)
                    if act_f.ndim == 1:
                        act_f = act_f[:, None]
                    act_ts = (
                        f["/robot/action_timestamps"][:].astype(np.float64)
                        if "/robot/action_timestamps" in f
                        else np.linspace(robot_ts[0], robot_ts[-1], len(act_f))
                    )
                    act_30 = np.empty((T, act_f.shape[1]), dtype=np.float32)
                    for c in range(act_f.shape[1]):
                        act_30[:, c] = np.interp(grid_t, act_ts, act_f[:, c]).astype(
                            np.float32
                        )
                    raw_action = act_30
                else:
                    raw_action = act_raw.astype(np.float32)

            valid_mask = robot_valid_30 & hand_valid_30

            # Probe frame 0 of each raw camera and expand virtual SBS stereo
            cameras: list[CameraSpec] = []
            cam_map_internal: dict[str, tuple[str, str | None]] = {}
            row_to_idx: dict[str, np.ndarray] = {}
            cam_hz: dict[str, float] = {}

            for raw_cam in raw_cam_names:
                cts = cam_ts_map[raw_cam]
                n_jpegs = len(cam_group[raw_cam]["jpeg"])
                if n_jpegs < 1:
                    raise ValueError(f"Camera {raw_cam!r} has no JPEG frames in {h5_path}")
                row_to_idx[raw_cam] = np.clip(
                    np.searchsorted(cts, grid_t, side="right") - 1,
                    0,
                    min(len(cts), n_jpegs) - 1,
                ).astype(np.int64)
                span = float(cts[-1] - cts[0])
                cam_hz[raw_cam] = (
                    round((len(cts) - 1) / span, 2) if span > 0 and len(cts) > 1 else fps
                )

                raw0 = bytes(cam_group[raw_cam]["jpeg"][0])
                img0 = cv2.imdecode(np.frombuffer(raw0, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img0 is None:
                    raise ValueError(f"Cannot decode frame 0 of {raw_cam} in {h5_path}")
                h0, w0 = int(img0.shape[0]), int(img0.shape[1])
                fmt_attr = str(cam_group[raw_cam].attrs.get("format", ""))
                is_sbs = raw_cam == "head_stereo" or (
                    "sbs" in fmt_attr.lower() and w0 >= 2 * h0
                )
                if is_sbs:
                    half_w = w0 // 2
                    l_name = f"{raw_cam}_left"
                    r_name = f"{raw_cam}_right"
                    cameras.append(
                        CameraSpec(
                            name=l_name,
                            role="wide",
                            width=half_w,
                            height=h0,
                            fps=fps,
                            source_type="h5_jpeg_sbs",
                            is_virtual_crop=True,
                            source_stream=raw_cam,
                        )
                    )
                    cameras.append(
                        CameraSpec(
                            name=r_name,
                            role="wide_right",
                            width=half_w,
                            height=h0,
                            fps=fps,
                            source_type="h5_jpeg_sbs",
                            is_virtual_crop=True,
                            source_stream=raw_cam,
                        )
                    )
                    cam_map_internal[l_name] = (raw_cam, "left_half")
                    cam_map_internal[r_name] = (raw_cam, "right_half")
                else:
                    cameras.append(
                        CameraSpec(
                            name=raw_cam,
                            role=infer_camera_role(raw_cam),
                            width=w0,
                            height=h0,
                            fps=fps,
                            source_type="h5_jpeg",
                            is_virtual_crop=False,
                            source_stream=raw_cam,
                        )
                    )
                    cam_map_internal[raw_cam] = (raw_cam, None)

            camera_roles = build_camera_roles(cameras)
            r_span = float(robot_ts[-1] - robot_ts[0])
            h_span = float(hand_ts[-1] - hand_ts[0])
            source_hz = {
                "robot": round((len(robot_ts) - 1) / r_span, 2) if r_span > 0 else 0.0,
                "hand": round((len(hand_ts) - 1) / h_span, 2) if h_span > 0 else 0.0,
                **cam_hz,
            }

            kin_spec = KinematicsSpec(
                has_kinematics=True,
                n_arms=2,
                arm_dof=7,
                hand_dof=int(hand_q.shape[1]),
                torso_dof=int(torso_q.shape[1]),
                has_chassis=True,
                has_tactile=tactile is not None,
                tactile_dim=int(tactile.shape[1]) if tactile is not None else 0,
                state_dim=int(raw_state.shape[1]),
                has_action_cmd=raw_action is not None,
                source_hz=source_hz,
            )

            h5_task_name = str(f.attrs.get("task_name", ""))
            created_at_attr = str(f.attrs.get("created_at", ""))
            schema_attr = str(f.attrs.get("schema", "raw_only_v3"))

        _malloc_trim()

        kin_data = KinematicsData(
            timestamps_s=timestamps_s,
            eef_pos=eef_pos,
            eef_rotvec=eef_rotvec,
            hand_q=hand_q,
            torso_q=torso_q,
            chassis_pose=chassis_pose,
            raw_state=raw_state,
            raw_action=raw_action,
            tactile=tactile,
            valid_mask=valid_mask,
            arm_q=arm_q,
            head_q=head_q,
            eef_quat_wxyz=eef_quat_wxyz,
            t0_unix=float(t_start),
            sub_valid={
                "robot": robot_valid_30,
                "hand": hand_valid_30,
                "force": force_valid_30,
            },
        )

        media = H5MediaSource(
            h5_path=h5_path,
            cameras=cameras,
            cam_specs_internal=cam_map_internal,
            row_to_idx=row_to_idx,
            fps=fps,
            duration_s=duration_s,
            frame_count=T,
        )

        fingerprint = compute_episode_fingerprint(
            handle.source_uri,
            handle.source_mtime,
            extra=f"T={T}|schema={schema_attr}",
        )
        meta = EpisodeMeta(
            episode_id=handle.episode_id,
            dataset=handle.dataset,
            stem=handle.stem,
            task_id=handle.task_id,
            adapter_type="h5_raw_v3",
            source_uri=handle.source_uri,
            source_mtime=handle.source_mtime,
            episode_fingerprint=fingerprint,
            duration_s=duration_s,
            fps=fps,
            frame_count=T,
            cameras=cameras,
            camera_roles=camera_roles,
            kinematics=kin_spec,
            recorded_success=handle.metadata.get("recorded_success"),
            extra_meta={
                "h5_task_name": h5_task_name,
                "schema": schema_attr,
                "created_at": created_at_attr,
                "t0_unix": float(t_start),
                "hand_mapping": hand_map,
            },
        )
        return EpisodeView(meta=meta, media=media, kinematics=kin_data)


def _nearest_bool_mask(
    src_ts: np.ndarray, src_valid: np.ndarray, grid_t: np.ndarray
) -> np.ndarray:
    n = len(src_ts)
    idx_r = np.clip(np.searchsorted(src_ts, grid_t, side="left"), 0, n - 1)
    idx_l = np.clip(idx_r - 1, 0, n - 1)
    choose_l = np.abs(grid_t - src_ts[idx_l]) <= np.abs(src_ts[idx_r] - grid_t)
    mapped = np.where(choose_l, idx_l, idx_r)
    return src_valid[mapped]
