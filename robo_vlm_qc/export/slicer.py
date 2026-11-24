"""Unified PieceSlicer across 3 recipes, SO(3) geodesic tail smoothing, 81D kinematics, and subtask mapping."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from ..config import ConfigManager
from ..io.base import EpisodeView, KinematicsTrack
from ..models.annotation import MacroStage, QcSegment
from ..models.export import ExportRecipe, KeepPiece, SubtaskSpan
from ..models.review import ResolvedEpisode

NOT_SMOOTHED: tuple[str, ...] = (
    "tau",
    "torque",
    "force",
    "wrench",
    "tactile",
    "age_ms",
)

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


def _extract_cfg_dicts(
    cfg: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(cfg, ConfigManager):
        return (
            dict(cfg.settings.get("export", {})),
            dict(cfg.settings.get("guardrail", {}).get("splice_gate", {})),
        )
    if isinstance(cfg, dict):
        export_cfg = (
            dict(cfg["export"])
            if isinstance(cfg.get("export"), dict)
            else dict(cfg)
        )
        splice_cfg: dict[str, Any] = {}
        if isinstance(cfg.get("guardrail"), dict) and isinstance(
            cfg["guardrail"].get("splice_gate"), dict
        ):
            splice_cfg = dict(cfg["guardrail"]["splice_gate"])
        elif isinstance(cfg.get("splice_gate"), dict):
            splice_cfg = dict(cfg["splice_gate"])
        return export_cfg, splice_cfg
    settings_attr = getattr(cfg, "settings", None)
    if isinstance(settings_attr, dict):
        return (
            dict(settings_attr.get("export", {})),
            dict(settings_attr.get("guardrail", {}).get("splice_gate", {})),
        )
    return {}, {}


def smoothable(key: str) -> bool:
    """Return False for torque, force, tactile, timestamp, and age streams."""
    low = key.lower().strip()
    if low.endswith("_ns") or low in ("sensor_time", "timestamps", "source_timestamps"):
        return False
    return not any(tok in low for tok in NOT_SMOOTHED)


def tail_weights(n: int) -> np.ndarray:
    """Smoothstep weights `w = t^2 * (3 - 2t)` over `(1..n)/n` (for n=1, returns [0.5])."""
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.array([0.5], dtype=np.float64)
    t = (np.arange(n, dtype=np.float64) + 1.0) / float(n)
    return t * t * (3.0 - 2.0 * t)


def _blend_linear_cols(
    out: np.ndarray, own_rows: int, cols: list[int], w: np.ndarray
) -> None:
    if not cols:
        return
    ref = out[own_rows - 1, cols].astype(np.float64)
    tail = out[own_rows:, cols].astype(np.float64)
    first = tail[0]
    delta = ref - first
    cand = tail + (1.0 - w)[:, None] * delta[None, :]
    valid = (
        np.isfinite(cand)
        & np.isfinite(ref)[None, :]
        & np.isfinite(first)[None, :]
    )
    out[own_rows:, cols] = np.where(valid, cand, tail).astype(np.float32)


def _blend_wrapped_angle_col(
    out: np.ndarray, own_rows: int, col: int, w: np.ndarray
) -> None:
    ref = float(out[own_rows - 1, col])
    tail = out[own_rows:, col].astype(np.float64)
    first = float(tail[0])
    if not (math.isfinite(ref) and math.isfinite(first)):
        return
    delta = math.atan2(math.sin(ref - first), math.cos(ref - first))
    cand = tail + (1.0 - w) * delta
    valid = np.isfinite(cand)
    out[own_rows:, col] = np.where(valid, cand, tail).astype(np.float32)


def _blend_quat_wxyz_slice(
    out: np.ndarray, own_rows: int, q_slice: slice, w: np.ndarray
) -> None:
    ref_q = out[own_rows - 1, q_slice].astype(np.float64)
    tail_q = out[own_rows:, q_slice].astype(np.float64)
    if not np.isfinite(ref_q).all() or np.linalg.norm(ref_q) <= 1e-6:
        return
    if not np.isfinite(tail_q[0]).all() or np.linalg.norm(tail_q[0]) <= 1e-6:
        return

    ref_n = ref_q / np.linalg.norm(ref_q)
    first_n = tail_q[0] / np.linalg.norm(tail_q[0])
    if float(np.dot(ref_n, first_n)) < 0.0:
        first_n = -first_n

    r_ref = Rotation.from_quat([ref_n[1], ref_n[2], ref_n[3], ref_n[0]])
    r_first = Rotation.from_quat([first_n[1], first_n[2], first_n[3], first_n[0]])
    offset_rv = (r_ref * r_first.inv()).as_rotvec()

    tail_norms = np.linalg.norm(tail_q, axis=1)
    valid_rows = np.isfinite(tail_q).all(axis=1) & (tail_norms > 1e-6)
    if not valid_rows.any():
        return

    tail_clean = np.empty_like(tail_q)
    tail_clean[valid_rows] = tail_q[valid_rows] / tail_norms[valid_rows, None]
    prev = ref_n
    for i in range(len(tail_q)):
        if valid_rows[i]:
            if float(np.dot(prev, tail_clean[i])) < 0.0:
                tail_clean[i] = -tail_clean[i]
            prev = tail_clean[i]

    idx_valid = np.flatnonzero(valid_rows)
    r_tail = Rotation.from_quat(tail_clean[idx_valid][:, [1, 2, 3, 0]])
    r_fade = Rotation.from_rotvec((1.0 - w[idx_valid])[:, None] * offset_rv[None, :])
    q_xyzw = (r_fade * r_tail).as_quat()
    q_wxyz = q_xyzw[:, [3, 0, 1, 2]]

    prev = ref_n
    for k, row_i in enumerate(idx_valid):
        q = q_wxyz[k]
        if float(np.dot(prev, q)) < 0.0:
            q = -q
        q_wxyz[k] = q
        prev = q

    res = tail_q.copy()
    res[idx_valid] = q_wxyz
    out[own_rows:, q_slice] = res.astype(np.float32)


def smooth_tail_81d(arr: np.ndarray, own_rows: int) -> np.ndarray:
    """Apply smoothstep + SO(3) geodesic tail fade on `(total_rows, 81)` state/action."""
    out = np.array(arr, dtype=np.float32, copy=True)
    if out.ndim != 2 or out.shape[1] != 81:
        raise ValueError(f"Expected shape (N, 81), got {out.shape}")
    if own_rows <= 0 or own_rows >= len(out):
        return out

    n_tail = len(out) - own_rows
    w = tail_weights(n_tail)

    linear_cols = [0, 1, *range(3, 26), *range(30, 33), *range(37, 81)]
    _blend_linear_cols(out, own_rows, linear_cols, w)
    _blend_wrapped_angle_col(out, own_rows, 2, w)
    _blend_quat_wxyz_slice(out, own_rows, slice(26, 30), w)
    _blend_quat_wxyz_slice(out, own_rows, slice(33, 37), w)
    return out


def smooth_tail_rotvec(arr: np.ndarray, own_rows: int) -> np.ndarray:
    """Apply SO(3) geodesic tail fade on `(total_rows, ..., 3)` rotvec arrays."""
    out = np.array(arr, dtype=np.float64, copy=True)
    if own_rows <= 0 or own_rows >= len(out):
        return out
    n_tail = len(out) - own_rows
    w = tail_weights(n_tail)

    orig_shape = out.shape
    flat = out.reshape(len(out), -1, 3)
    for arm in range(flat.shape[1]):
        ref_rv = flat[own_rows - 1, arm]
        tail_rv = flat[own_rows:, arm]
        if not (np.isfinite(ref_rv).all() and np.isfinite(tail_rv[0]).all()):
            continue
        r_ref = Rotation.from_rotvec(ref_rv)
        r_first = Rotation.from_rotvec(tail_rv[0])
        offset = (r_ref * r_first.inv()).as_rotvec()

        valid = np.isfinite(tail_rv).all(axis=1)
        if not valid.any():
            continue
        idx = np.flatnonzero(valid)
        r_tail = Rotation.from_rotvec(tail_rv[idx])
        r_fade = Rotation.from_rotvec((1.0 - w[idx])[:, None] * offset[None, :])
        rv = (r_fade * r_tail).as_rotvec()
        ang = np.linalg.norm(rv, axis=1)
        nz = ang > 1e-9
        other = rv.copy()
        other[nz] = rv[nz] * (1.0 - (2.0 * math.pi) / ang[nz, None])
        use_other = np.linalg.norm(other - tail_rv[idx], axis=1) < np.linalg.norm(
            rv - tail_rv[idx], axis=1
        )
        rv[use_other] = other[use_other]
        flat[own_rows + idx, arm] = rv

    return flat.reshape(orig_shape).astype(arr.dtype)


def smooth_tail_vector(key: str, arr: np.ndarray, own_rows: int) -> np.ndarray:
    """Smooth a tail vector unless `key` matches `NOT_SMOOTHED` or is non-float."""
    arr_np = np.asarray(arr)
    if not smoothable(key) or not np.issubdtype(arr_np.dtype, np.floating):
        return arr_np.copy()
    if own_rows <= 0 or own_rows >= len(arr_np):
        return arr_np.copy()
    if arr_np.ndim == 2 and arr_np.shape[1] == 81:
        return smooth_tail_81d(arr_np, own_rows)
    if "rotvec" in key.lower() and arr_np.shape[-1] == 3:
        return smooth_tail_rotvec(arr_np, own_rows)

    out = arr_np.astype(np.float64, copy=True)
    n_tail = len(out) - own_rows
    w = tail_weights(n_tail)
    orig_shape = out.shape
    flat = out.reshape(len(out), -1)
    ref = flat[own_rows - 1]
    tail = flat[own_rows:]
    first = tail[0]
    cand = tail + (1.0 - w)[:, None] * (ref - first)[None, :]
    ok = np.isfinite(cand) & np.isfinite(ref)[None, :] & np.isfinite(first)[None, :]
    flat[own_rows:] = np.where(ok, cand, tail)
    return flat.reshape(orig_shape).astype(arr_np.dtype)


def _rotvec_to_quat_wxyz(rv: np.ndarray) -> np.ndarray:
    """Convert `(T, 3)` rotvec to `(T, 4)` wxyz quaternion (NaN-safe, sign-continuous)."""
    T = len(rv)
    out = np.zeros((T, 4), dtype=np.float32)
    out[:, 0] = 1.0
    fin = np.isfinite(rv).all(axis=1)
    if not fin.any():
        return out
    q_xyzw = Rotation.from_rotvec(rv[fin].astype(np.float64)).as_quat()
    q_wxyz = q_xyzw[:, [3, 0, 1, 2]]
    for i in range(1, len(q_wxyz)):
        if float(np.dot(q_wxyz[i - 1], q_wxyz[i])) < 0.0:
            q_wxyz[i] = -q_wxyz[i]
    out[fin] = q_wxyz.astype(np.float32)
    return out


def extract_canonical_81d_state(
    kin: KinematicsTrack | None, T: int
) -> np.ndarray:
    """Normalize any adapter's KinematicsTrack (or None) into canonical `(T, 81)` float32."""
    if T <= 0:
        return np.zeros((0, 81), dtype=np.float32)
    if kin is None:
        return np.zeros((T, 81), dtype=np.float32)

    raw_state = getattr(kin, "raw_state", None)
    if raw_state is not None:
        rs = np.asarray(raw_state, dtype=np.float32)
        if rs.ndim == 2 and rs.shape == (T, 81):
            out = rs.copy()
            # Guard against NaNs propagating to stats/Parquet
            if not np.isfinite(out).all():
                out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            return out

    out = np.zeros((T, 81), dtype=np.float32)
    # Default identity quaternions at [26:30] and [33:37]
    out[:, 26] = 1.0
    out[:, 33] = 1.0

    used_semantic = False

    chassis = getattr(kin, "chassis_pose", None)
    if chassis is not None:
        ch = np.asarray(chassis, dtype=np.float32)
        if ch.ndim == 2 and len(ch) == T:
            d = min(3, ch.shape[1])
            out[:, :d] = ch[:, :d]
            used_semantic = True

    torso = getattr(kin, "torso_q", None)
    if torso is not None:
        tq = np.asarray(torso, dtype=np.float32)
        if tq.ndim == 2 and len(tq) == T:
            d = min(4, tq.shape[1])
            out[:, 3 : 3 + d] = tq[:, :d]
            used_semantic = True

    arm_q = getattr(kin, "arm_q", None)
    if arm_q is not None:
        aq = np.asarray(arm_q, dtype=np.float32)
        if aq.ndim == 2 and len(aq) == T:
            if aq.shape[1] == 14:
                out[:, 7:21] = aq
            elif aq.shape[1] == 12:
                out[:, 7:13] = aq[:, :6]
                out[:, 14:20] = aq[:, 6:12]
            else:
                d = min(14, aq.shape[1])
                out[:, 7 : 7 + d] = aq[:, :d]
            used_semantic = True

    head_q = getattr(kin, "head_q", None)
    if head_q is not None:
        hq = np.asarray(head_q, dtype=np.float32)
        if hq.ndim == 2 and len(hq) == T:
            d = min(2, hq.shape[1])
            out[:, 21 : 21 + d] = hq[:, :d]
            used_semantic = True

    eef_pos = getattr(kin, "eef_pos", None)
    if eef_pos is not None:
        ep = np.asarray(eef_pos, dtype=np.float32)
        if ep.ndim == 3 and len(ep) == T:
            if ep.shape[1] >= 1:
                out[:, 23:26] = ep[:, 0, :3]
            if ep.shape[1] >= 2:
                out[:, 30:33] = ep[:, 1, :3]
            used_semantic = True

    eef_quat = getattr(kin, "eef_quat_wxyz", None)
    eef_rotvec = getattr(kin, "eef_rotvec", None)
    if eef_quat is not None:
        eq = np.asarray(eef_quat, dtype=np.float32)
        if eq.ndim == 3 and len(eq) == T and eq.shape[2] == 4:
            if eq.shape[1] >= 1:
                out[:, 26:30] = eq[:, 0, :4]
            if eq.shape[1] >= 2:
                out[:, 33:37] = eq[:, 1, :4]
            used_semantic = True
    elif eef_rotvec is not None:
        er = np.asarray(eef_rotvec, dtype=np.float64)
        if er.ndim == 3 and len(er) == T and er.shape[2] == 3:
            if er.shape[1] >= 1:
                out[:, 26:30] = _rotvec_to_quat_wxyz(er[:, 0, :])
            if er.shape[1] >= 2:
                out[:, 33:37] = _rotvec_to_quat_wxyz(er[:, 1, :])
            used_semantic = True

    hand_q = getattr(kin, "hand_q", None)
    if hand_q is not None:
        hq_arr = np.asarray(hand_q, dtype=np.float32)
        if hq_arr.ndim == 2 and len(hq_arr) == T:
            d = min(44, hq_arr.shape[1])
            out[:, 37 : 37 + d] = hq_arr[:, :d]
            used_semantic = True

    if not used_semantic and raw_state is not None:
        rs = np.asarray(raw_state, dtype=np.float32)
        if rs.ndim == 2 and len(rs) == T:
            d = min(81, rs.shape[1])
            out[:, :d] = rs[:, :d]

    if not np.isfinite(out).all():
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _pad_or_trim_81d(arr: np.ndarray, T: int) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        a = a[:, None]
    if len(a) != T:
        raise ValueError(f"Expected length {T}, got {len(a)}")
    if a.shape[1] == 81:
        out = a.copy()
    elif a.shape[1] > 81:
        out = a[:, :81].copy()
    else:
        out = np.zeros((T, 81), dtype=np.float32)
        out[:, : a.shape[1]] = a
    if not np.isfinite(out).all():
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def piece_source_rows(piece: KeepPiece, T: int) -> list[int]:
    """Return source episode row indices `[own..., tail...]` for a KeepPiece."""
    own = [max(0, min(T - 1, r)) for r in range(piece.start_row, piece.end_row)]
    if piece.tail_rows <= 0:
        return own
    if piece.tail_start_row is not None and piece.tail_end_row is not None:
        tail = [
            max(0, min(T - 1, r))
            for r in range(piece.tail_start_row, piece.tail_end_row)
        ]
    else:
        tail = [
            max(0, min(T - 1, piece.end_row + k)) for k in range(piece.tail_rows)
        ]
    if len(tail) > piece.tail_rows:
        tail = tail[: piece.tail_rows]
    elif len(tail) < piece.tail_rows:
        last_r = tail[-1] if tail else own[-1]
        tail.extend([last_r] * (piece.tail_rows - len(tail)))
    return own + tail


def extract_piece_kinematics_81d(
    piece: KeepPiece, view: EpisodeView
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return `(state_81, action_81, tactile_30_or_None)` for a single KeepPiece.

    Applies smoothstep + SO(3) geodesic tail smoothing to state (and raw_action
    when present) if ``piece.smooth_tail_applied`` is True, and synthesizes
    ``action[t] = state[t+1]`` (repeating post-smoothing ``state[-1]`` at the
    last row) when ``raw_action`` is absent. Tactile is NEVER smoothed.
    """
    kin = view.kinematics
    T = (
        len(kin.timestamps_s)
        if (kin is not None and len(kin.timestamps_s) > 0)
        else int(view.meta.frame_count)
    )
    rows = piece_source_rows(piece, T)

    full_state = extract_canonical_81d_state(kin, T)
    state_piece = full_state[rows]
    if piece.smooth_tail_applied and piece.tail_rows > 0:
        state_piece = smooth_tail_81d(state_piece, piece.own_rows)

    raw_act = getattr(kin, "raw_action", None) if kin is not None else None
    if raw_act is not None:
        full_act = _pad_or_trim_81d(np.asarray(raw_act, dtype=np.float32), T)
        action_piece = full_act[rows]
        if piece.smooth_tail_applied and piece.tail_rows > 0:
            action_piece = smooth_tail_81d(action_piece, piece.own_rows)
    else:
        action_piece = np.empty_like(state_piece)
        if len(state_piece) > 1:
            action_piece[:-1] = state_piece[1:]
        if len(state_piece) > 0:
            action_piece[-1] = state_piece[-1]

    tactile_piece: np.ndarray | None = None
    raw_tac = getattr(kin, "tactile", None) if kin is not None else None
    if raw_tac is not None:
        tac_arr = np.asarray(raw_tac, dtype=np.float32)
        if tac_arr.ndim == 2 and len(tac_arr) == T:
            if tac_arr.shape[1] == 30:
                tac_30 = tac_arr[rows].copy()
            elif tac_arr.shape[1] > 30:
                tac_30 = tac_arr[rows, :30].copy()
            else:
                tac_30 = np.zeros((len(rows), 30), dtype=np.float32)
                tac_30[:, : tac_arr.shape[1]] = tac_arr[rows]
            if not np.isfinite(tac_30).all():
                tac_30 = np.nan_to_num(tac_30, nan=0.0, posinf=0.0, neginf=0.0)
            tactile_piece = tac_30

    return state_piece, action_piece, tactile_piece


def extract_piece_kinematics_native(
    piece: KeepPiece,
    view: EpisodeView,
    *,
    state_dim: int | None,
    action_dim: int | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Return native-dim `(state, action, tactile)` arrays for a KeepPiece.

    Preserves native LeRobot `[D_s]` / `[D_a]` dimensions without padding to 81D,
    applying `smooth_tail_vector` when `piece.smooth_tail_applied` is True.
    Tactile is NEVER smoothed.
    """
    kin = view.kinematics
    T = (
        len(kin.timestamps_s)
        if (kin is not None and len(kin.timestamps_s) > 0)
        else int(view.meta.frame_count)
    )
    rows = piece_source_rows(piece, T)

    state_piece: np.ndarray | None = None
    if state_dim is not None and state_dim > 0:
        raw_st = getattr(kin, "raw_state", None) if kin is not None else None
        if raw_st is None:
            state_piece = np.zeros((len(rows), state_dim), dtype=np.float32)
        else:
            st_arr = np.asarray(raw_st, dtype=np.float32)
            if st_arr.ndim == 1:
                st_arr = st_arr[:, None]
            if st_arr.shape[1] != state_dim:
                raise ValueError(
                    f"Expected native state dim {state_dim}, got {st_arr.shape[1]} "
                    f"for episode {view.meta.episode_id!r}"
                )
            state_piece = st_arr[rows].copy()
            if not np.isfinite(state_piece).all():
                state_piece = np.nan_to_num(state_piece, nan=0.0, posinf=0.0, neginf=0.0)
            if piece.smooth_tail_applied and piece.tail_rows > 0:
                state_piece = smooth_tail_vector(
                    "observation.state", state_piece, piece.own_rows
                ).astype(np.float32, copy=False)

    action_piece: np.ndarray | None = None
    if action_dim is not None and action_dim > 0:
        raw_act = getattr(kin, "raw_action", None) if kin is not None else None
        if raw_act is not None:
            act_arr = np.asarray(raw_act, dtype=np.float32)
            if act_arr.ndim == 1:
                act_arr = act_arr[:, None]
            if act_arr.shape[1] != action_dim:
                raise ValueError(
                    f"Expected native action dim {action_dim}, got {act_arr.shape[1]} "
                    f"for episode {view.meta.episode_id!r}"
                )
            action_piece = act_arr[rows].copy()
            if not np.isfinite(action_piece).all():
                action_piece = np.nan_to_num(
                    action_piece, nan=0.0, posinf=0.0, neginf=0.0
                )
            if piece.smooth_tail_applied and piece.tail_rows > 0:
                action_piece = smooth_tail_vector(
                    "action", action_piece, piece.own_rows
                ).astype(np.float32, copy=False)
        elif state_piece is not None and state_piece.shape[1] == action_dim:
            action_piece = np.empty_like(state_piece)
            if len(state_piece) > 1:
                action_piece[:-1] = state_piece[1:]
            if len(state_piece) > 0:
                action_piece[-1] = state_piece[-1]
        else:
            action_piece = np.zeros((len(rows), action_dim), dtype=np.float32)

    tactile_piece: np.ndarray | None = None
    raw_tac = getattr(kin, "tactile", None) if kin is not None else None
    if raw_tac is not None:
        tac_arr = np.asarray(raw_tac, dtype=np.float32)
        if tac_arr.ndim == 2 and len(tac_arr) == T:
            tactile_piece = tac_arr[rows].copy()
            if not np.isfinite(tactile_piece).all():
                tactile_piece = np.nan_to_num(
                    tactile_piece, nan=0.0, posinf=0.0, neginf=0.0
                )

    return state_piece, action_piece, tactile_piece


def _merge_row_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    ordered = sorted((a, b) for a, b in intervals if b > a)
    if not ordered:
        return []
    merged: list[tuple[int, int]] = [ordered[0]]
    for a, b in ordered[1:]:
        pa, pb = merged[-1]
        if a <= pb:
            merged[-1] = (pa, max(pb, b))
        else:
            merged.append((a, b))
    return merged


def _subtract_intervals(T: int, cuts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if T <= 0:
        return []
    merged = _merge_row_intervals([(max(0, a), min(T, b)) for a, b in cuts])
    kept: list[tuple[int, int]] = []
    cur = 0
    for a, b in merged:
        if a > cur:
            kept.append((cur, a))
        cur = max(cur, b)
    if cur < T:
        kept.append((cur, T))
    return kept


def _pose_jump_exceeds(
    kin: KinematicsTrack | None,
    row_a: int,
    row_b: int,
    splice_cfg: dict[str, Any],
) -> bool:
    if kin is None:
        return False
    max_pos = float(splice_cfg.get("max_pos_jump_m", 0.02))
    max_rot = float(splice_cfg.get("max_rot_jump_rad", 0.10))
    max_hand = float(splice_cfg.get("max_hand_jump_rad", 0.30))
    max_torso = float(splice_cfg.get("max_torso_jump_rad", 0.05))
    max_ch_pos = float(splice_cfg.get("max_chassis_pos_jump_m", 0.02))
    max_ch_yaw = float(splice_cfg.get("max_chassis_yaw_jump_rad", 0.05))

    eef_pos = getattr(kin, "eef_pos", None)
    if eef_pos is not None and len(eef_pos) > max(row_a, row_b):
        pa = np.asarray(eef_pos[row_a], dtype=np.float64)
        pb = np.asarray(eef_pos[row_b], dtype=np.float64)
        if np.isfinite(pa).all() and np.isfinite(pb).all():
            if float(np.linalg.norm(pb - pa, axis=-1).max()) > max_pos:
                return True

    eef_rot = getattr(kin, "eef_rotvec", None)
    if eef_rot is not None and len(eef_rot) > max(row_a, row_b):
        ra = np.asarray(eef_rot[row_a], dtype=np.float64)
        rb = np.asarray(eef_rot[row_b], dtype=np.float64)
        if np.isfinite(ra).all() and np.isfinite(rb).all():
            for arm in range(ra.shape[0]):
                qa = Rotation.from_rotvec(ra[arm]).as_quat()
                qb = Rotation.from_rotvec(rb[arm]).as_quat()
                dot = min(1.0, abs(float(np.dot(qa, qb))))
                if 2.0 * math.acos(dot) > max_rot:
                    return True

    hand_q = getattr(kin, "hand_q", None)
    if hand_q is not None and len(hand_q) > max(row_a, row_b):
        ha = np.asarray(hand_q[row_a], dtype=np.float64)
        hb = np.asarray(hand_q[row_b], dtype=np.float64)
        fin = np.isfinite(ha) & np.isfinite(hb)
        if fin.any() and float(np.abs(hb[fin] - ha[fin]).max()) > max_hand:
            return True

    torso_q = getattr(kin, "torso_q", None)
    if torso_q is not None and len(torso_q) > max(row_a, row_b):
        ta = np.asarray(torso_q[row_a], dtype=np.float64)
        tb = np.asarray(torso_q[row_b], dtype=np.float64)
        fin = np.isfinite(ta) & np.isfinite(tb)
        if fin.any() and float(np.abs(tb[fin] - ta[fin]).max()) > max_torso:
            return True

    chassis = getattr(kin, "chassis_pose", None)
    if chassis is not None and len(chassis) > max(row_a, row_b):
        ca = np.asarray(chassis[row_a], dtype=np.float64)
        cb = np.asarray(chassis[row_b], dtype=np.float64)
        if len(ca) >= 2 and np.isfinite(ca[:2]).all() and np.isfinite(cb[:2]).all():
            if float(np.linalg.norm(cb[:2] - ca[:2])) > max_ch_pos:
                return True
        if len(ca) >= 3 and math.isfinite(ca[2]) and math.isfinite(cb[2]):
            dyaw = abs(math.atan2(math.sin(cb[2] - ca[2]), math.cos(cb[2] - ca[2])))
            if dyaw > max_ch_yaw:
                return True

    has_semantic = any(
        getattr(kin, attr, None) is not None
        for attr in ("eef_pos", "eef_rotvec", "hand_q", "torso_q", "chassis_pose")
    )
    if not has_semantic:
        raw_st = getattr(kin, "raw_state", None)
        if raw_st is not None and len(raw_st) > max(row_a, row_b):
            sa = np.asarray(raw_st[row_a], dtype=np.float64)
            sb = np.asarray(raw_st[row_b], dtype=np.float64)
            fin = np.isfinite(sa) & np.isfinite(sb)
            if fin.any() and float(np.abs(sb[fin] - sa[fin]).max()) > max_rot:
                return True

    return False


def _subtask_instruction_text(span: SubtaskSpan) -> str:
    return (
        (span.instruction or "").strip()
        or (span.label_en or "").strip()
        or (span.label or "").strip()
        or span.stage_id
    )


def _align_rows_to_subtasks(
    piece_rows: list[int],
    stages: list[MacroStage],
    fps: float,
) -> tuple[list[int], list[SubtaskSpan]]:
    if not stages:
        return [-1] * len(piece_rows), []

    sorted_stages = sorted(stages, key=lambda s: (s.t_start, s.t_end))
    row_indices: list[int] = []
    for r in piece_rows:
        t_r = (float(r) + 0.5) / fps
        matched_idx: int | None = None
        for idx, st in enumerate(sorted_stages):
            if st.t_start <= t_r <= st.t_end:
                matched_idx = idx
                break
        if matched_idx is None:
            # Fallback to nearest stage midpoint/boundary
            best_dist = math.inf
            best_idx = 0
            for idx, st in enumerate(sorted_stages):
                if t_r < st.t_start:
                    d = st.t_start - t_r
                elif t_r > st.t_end:
                    d = t_r - st.t_end
                else:
                    d = 0.0
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
            matched_idx = best_idx
        row_indices.append(matched_idx)

    spans: list[SubtaskSpan] = []
    if row_indices:
        cur_idx = row_indices[0]
        cur_start = 0
        for pos in range(1, len(row_indices)):
            if row_indices[pos] != cur_idx:
                st = sorted_stages[cur_idx]
                spans.append(
                    SubtaskSpan(
                        subtask_index=cur_idx,
                        stage_id=st.stage_id,
                        label=st.label,
                        label_en=st.label_en,
                        instruction=str(getattr(st, "instruction", "") or "").strip(),
                        start_row=cur_start,
                        end_row=pos,
                    )
                )
                cur_idx = row_indices[pos]
                cur_start = pos
        st = sorted_stages[cur_idx]
        spans.append(
            SubtaskSpan(
                subtask_index=cur_idx,
                stage_id=st.stage_id,
                label=st.label,
                label_en=st.label_en,
                instruction=str(getattr(st, "instruction", "") or "").strip(),
                start_row=cur_start,
                end_row=len(row_indices),
            )
        )
    return row_indices, spans


class PieceSlicer:
    """Unified recipe-aware episode slicer with tail smoothing & subtask alignment."""

    def __init__(
        self,
        cfg: Any = None,
        *,
        config: Any = None,
        config_manager: Any = None,
        settings: Any = None,
    ) -> None:
        raw_cfg = cfg or config or config_manager or settings
        self.cfg = raw_cfg
        self.export_cfg, self.splice_cfg = _extract_cfg_dicts(raw_cfg)

    def slice(
        self,
        resolved: ResolvedEpisode,
        view: EpisodeView | None,
        recipe: ExportRecipe,
    ) -> list[KeepPiece]:
        if recipe.dataset_filter and resolved.meta.dataset != recipe.dataset_filter:
            return []
        if recipe.task_filter and resolved.meta.task_id != recipe.task_filter:
            return []

        kin = view.kinematics if view is not None else None
        T = (
            len(kin.timestamps_s)
            if (kin is not None and len(kin.timestamps_s) > 0)
            else int(resolved.meta.frame_count)
        )
        if T <= 0:
            return []

        fps = float(
            resolved.meta.fps
            or (view.fps if view is not None else 0.0)
            or 30.0
        )
        duration_s = float(resolved.meta.duration_s or (T / fps))

        chunk_size = (
            recipe.chunk_size
            if recipe.chunk_size is not None
            else self.export_cfg.get("chunk_size")
        )
        min_piece_s = float(
            recipe.min_piece_s
            if recipe.min_piece_s is not None
            else self.export_cfg.get("min_piece_s", 0.1)
        )
        min_frames = (
            int(chunk_size)
            if chunk_size
            else max(2, int(round(min_piece_s * fps)))
        )

        if recipe.tail_frames is not None:
            tail_n = int(recipe.tail_frames)
        elif self.export_cfg.get("tail_frames") is not None:
            tail_n = int(self.export_cfg["tail_frames"])
        elif chunk_size is not None:
            tail_n = int(chunk_size)
        else:
            tail_n = min_frames

        smooth_mode = recipe.smooth_tail or str(
            self.export_cfg.get("smooth_tail", "on_jump")
        )

        # Build full-episode loss mask (T, 2)
        ep_mask = np.ones((T, 2), dtype=np.float32)
        cuts: list[tuple[int, int]] = []

        def _span_to_rows(t0: float, t1: float) -> tuple[int, int]:
            s_r = max(0, min(T, int(round(float(t0) * fps))))
            e_r = max(s_r, min(T, int(round(float(t1) * fps))))
            return s_r, e_r

        if recipe.name in ("bc_clean", "recovery_plus"):
            if resolved.success.passed is not True:
                return []

            for seg in resolved.qc_segments:
                s_r, e_r = _span_to_rows(seg.t_start, seg.t_end)
                if e_r <= s_r:
                    continue
                is_recovery = seg.training_role == "recovery"
                if recipe.name == "recovery_plus" and is_recovery:
                    # Keep recovery segments unmasked and uncut in recovery_plus
                    continue
                if seg.splice_policy == "cut_clean":
                    cuts.append((s_r, e_r))
                elif seg.splice_policy == "mask_loss":
                    side = seg.mask_side or seg.affected_arm or "both"
                    if side in ("left", "both"):
                        ep_mask[s_r:e_r, 0] = 0.0
                    if side in ("right", "both"):
                        ep_mask[s_r:e_r, 1] = 0.0

            raw_intervals = _subtract_intervals(T, cuts)
            kept_intervals: list[tuple[int, int]] = []
            dropped_pieces: list[KeepPiece] = []
            for idx, (a, b) in enumerate(raw_intervals):
                if (b - a) < min_frames:
                    rows = list(range(a, b))
                    sub_idx, sub_spans = _align_rows_to_subtasks(
                        rows, resolved.macro_stages, fps
                    )
                    dropped_pieces.append(
                        KeepPiece(
                            piece_id=f"{resolved.episode_id}__drop_{idx:03d}",
                            episode_id=resolved.episode_id,
                            piece_index=idx,
                            start_row=a,
                            end_row=b,
                            own_rows=b - a,
                            tail_rows=0,
                            total_rows=b - a,
                            fps=fps,
                            t_start_s=round(a / fps, 6),
                            t_end_s=round(b / fps, 6),
                            action_loss_mask=ep_mask[rows].tolist(),
                            is_tail=[False] * (b - a),
                            subtask_indices=sub_idx,
                            subtasks=sub_spans,
                            smooth_tail_applied=False,
                            dropped=True,
                            drop_reason=f"below_min_frames({b - a}<{min_frames})",
                        )
                    )
                else:
                    kept_intervals.append((a, b))

            hand_cmd_valid = (
                getattr(kin, "hand_cmd_valid", None) if kin is not None else None
            )
            kept_pieces: list[KeepPiece] = []

            if recipe.split_by_subtask and resolved.macro_stages:
                for k, (a, b) in enumerate(kept_intervals):
                    own_rows_list = list(range(a, b))
                    _, own_spans = _align_rows_to_subtasks(
                        own_rows_list, resolved.macro_stages, fps
                    )
                    for span_pos, span in enumerate(own_spans):
                        sub_a = a + span.start_row
                        sub_b = a + span.end_row
                        sub_own = sub_b - sub_a
                        task_text = _subtask_instruction_text(span)
                        if sub_own < min_frames:
                            drop_rows = list(range(sub_a, sub_b))
                            sub_idx, sub_spans = _align_rows_to_subtasks(
                                drop_rows, resolved.macro_stages, fps
                            )
                            drop_idx = len(dropped_pieces)
                            dropped_pieces.append(
                                KeepPiece(
                                    piece_id=f"{resolved.episode_id}__drop_{drop_idx:03d}",
                                    episode_id=resolved.episode_id,
                                    piece_index=drop_idx,
                                    start_row=sub_a,
                                    end_row=sub_b,
                                    own_rows=sub_own,
                                    tail_rows=0,
                                    total_rows=sub_own,
                                    fps=fps,
                                    t_start_s=round(sub_a / fps, 6),
                                    t_end_s=round(sub_b / fps, 6),
                                    action_loss_mask=ep_mask[drop_rows].tolist(),
                                    is_tail=[False] * sub_own,
                                    subtask_indices=sub_idx,
                                    subtasks=sub_spans,
                                    task_override=task_text,
                                    smooth_tail_applied=False,
                                    dropped=True,
                                    drop_reason=f"below_min_frames({sub_own}<{min_frames})",
                                )
                            )
                            continue

                        tail_start: int | None = None
                        tail_end: int | None = None
                        tail_rows = 0
                        smooth_applied = False

                        # Only bridge tail across a cut when the next kept interval continues the SAME subtask
                        if (
                            span_pos == len(own_spans) - 1
                            and k < len(kept_intervals) - 1
                            and tail_n > 0
                        ):
                            next_a, next_b = kept_intervals[k + 1]
                            cand_tail = min(tail_n, next_b - next_a)
                            if (
                                hand_cmd_valid is not None
                                and len(hand_cmd_valid) >= next_a + cand_tail
                            ):
                                invalid_rel = np.flatnonzero(
                                    ~np.asarray(
                                        hand_cmd_valid[next_a : next_a + cand_tail],
                                        dtype=bool,
                                    )
                                )
                                if len(invalid_rel) > 0:
                                    cand_tail = int(invalid_rel[0])
                            if cand_tail > 0:
                                next_sub_idx, _ = _align_rows_to_subtasks(
                                    list(range(next_a, next_a + cand_tail)),
                                    resolved.macro_stages,
                                    fps,
                                )
                                same_sub_n = 0
                                for s_i in next_sub_idx:
                                    if s_i == span.subtask_index:
                                        same_sub_n += 1
                                    else:
                                        break
                                if same_sub_n > 0:
                                    tail_start = next_a
                                    tail_end = next_a + same_sub_n
                                    tail_rows = same_sub_n
                                    if kin is not None:
                                        if smooth_mode == "always":
                                            smooth_applied = True
                                        elif smooth_mode == "on_jump":
                                            smooth_applied = _pose_jump_exceeds(
                                                kin, sub_b - 1, next_a, self.splice_cfg
                                            )

                        piece_rows = list(range(sub_a, sub_b)) + (
                            list(range(tail_start, tail_end))
                            if (tail_start is not None and tail_end is not None)
                            else []
                        )
                        sub_idx, sub_spans = _align_rows_to_subtasks(
                            piece_rows, resolved.macro_stages, fps
                        )
                        p_idx = len(kept_pieces)
                        kept_pieces.append(
                            KeepPiece(
                                piece_id=f"{resolved.episode_id}__p{p_idx:03d}",
                                episode_id=resolved.episode_id,
                                piece_index=p_idx,
                                start_row=sub_a,
                                end_row=sub_b,
                                tail_start_row=tail_start,
                                tail_end_row=tail_end,
                                own_rows=sub_own,
                                tail_rows=tail_rows,
                                total_rows=sub_own + tail_rows,
                                fps=fps,
                                t_start_s=round(sub_a / fps, 6),
                                t_end_s=round(sub_b / fps, 6),
                                action_loss_mask=ep_mask[piece_rows].tolist(),
                                is_tail=[False] * sub_own + [True] * tail_rows,
                                subtask_indices=sub_idx,
                                subtasks=sub_spans,
                                task_override=task_text,
                                smooth_tail_applied=smooth_applied,
                                dropped=False,
                                drop_reason=None,
                            )
                        )
                return kept_pieces + dropped_pieces

            for k, (a, b) in enumerate(kept_intervals):
                own_rows = b - a
                tail_start: int | None = None
                tail_end: int | None = None
                tail_rows = 0
                smooth_applied = False

                if k < len(kept_intervals) - 1 and tail_n > 0:
                    next_a, next_b = kept_intervals[k + 1]
                    n_tail = min(tail_n, next_b - next_a)
                    if (
                        hand_cmd_valid is not None
                        and len(hand_cmd_valid) >= next_a + n_tail
                    ):
                        invalid_rel = np.flatnonzero(
                            ~np.asarray(
                                hand_cmd_valid[next_a : next_a + n_tail], dtype=bool
                            )
                        )
                        if len(invalid_rel) > 0:
                            n_tail = int(invalid_rel[0])
                    if n_tail > 0:
                        tail_start = next_a
                        tail_end = next_a + n_tail
                        tail_rows = n_tail
                        if kin is not None:
                            if smooth_mode == "always":
                                smooth_applied = True
                            elif smooth_mode == "on_jump":
                                smooth_applied = _pose_jump_exceeds(
                                    kin, b - 1, next_a, self.splice_cfg
                                )

                piece_rows = list(range(a, b)) + (
                    list(range(tail_start, tail_end))
                    if (tail_start is not None and tail_end is not None)
                    else []
                )
                sub_idx, sub_spans = _align_rows_to_subtasks(
                    piece_rows, resolved.macro_stages, fps
                )
                kept_pieces.append(
                    KeepPiece(
                        piece_id=f"{resolved.episode_id}__p{k:03d}",
                        episode_id=resolved.episode_id,
                        piece_index=k,
                        start_row=a,
                        end_row=b,
                        tail_start_row=tail_start,
                        tail_end_row=tail_end,
                        own_rows=own_rows,
                        tail_rows=tail_rows,
                        total_rows=own_rows + tail_rows,
                        fps=fps,
                        t_start_s=round(a / fps, 6),
                        t_end_s=round(b / fps, 6),
                        action_loss_mask=ep_mask[piece_rows].tolist(),
                        is_tail=[False] * own_rows + [True] * tail_rows,
                        subtask_indices=sub_idx,
                        subtasks=sub_spans,
                        smooth_tail_applied=smooth_applied,
                        dropped=False,
                        drop_reason=None,
                    )
                )

            return kept_pieces + dropped_pieces

        # badcase_pairs recipe
        bad_segs = [
            seg
            for seg in resolved.qc_segments
            if seg.training_role in ("badcase", "terminal_fail")
        ]
        if not bad_segs:
            return []

        for seg in resolved.qc_segments:
            if seg.splice_policy == "mask_loss":
                s_r, e_r = _span_to_rows(seg.t_start, seg.t_end)
                if e_r > s_r:
                    side = seg.mask_side or seg.affected_arm or "both"
                    if side in ("left", "both"):
                        ep_mask[s_r:e_r, 0] = 0.0
                    if side in ("right", "both"):
                        ep_mask[s_r:e_r, 1] = 0.0

        ctx = float(recipe.context_window_s)
        win_intervals: list[tuple[int, int]] = []
        for seg in bad_segs:
            t0 = max(0.0, float(seg.t_start) - ctx)
            t1 = min(duration_s, float(seg.t_end) + ctx)
            a, b = _span_to_rows(t0, t1)
            if b <= a:
                if a < T:
                    b = a + 1
                elif a > 0:
                    a = a - 1
            if b > a:
                win_intervals.append((a, b))

        merged_wins = _merge_row_intervals(win_intervals)
        out_pieces: list[KeepPiece] = []
        for k, (a, b) in enumerate(merged_wins):
            rows = list(range(a, b))
            sub_idx, sub_spans = _align_rows_to_subtasks(
                rows, resolved.macro_stages, fps
            )
            out_pieces.append(
                KeepPiece(
                    piece_id=f"{resolved.episode_id}__bad{k:03d}",
                    episode_id=resolved.episode_id,
                    piece_index=k,
                    start_row=a,
                    end_row=b,
                    tail_start_row=None,
                    tail_end_row=None,
                    own_rows=b - a,
                    tail_rows=0,
                    total_rows=b - a,
                    fps=fps,
                    t_start_s=round(a / fps, 6),
                    t_end_s=round(b / fps, 6),
                    action_loss_mask=ep_mask[rows].tolist(),
                    is_tail=[False] * (b - a),
                    subtask_indices=sub_idx,
                    subtasks=sub_spans,
                    smooth_tail_applied=False,
                    dropped=False,
                    drop_reason=None,
                )
            )
        return out_pieces


def slice_episode(
    resolved: ResolvedEpisode,
    view: EpisodeView | None,
    recipe: ExportRecipe,
    cfg: Any = None,
) -> list[KeepPiece]:
    return PieceSlicer(cfg).slice(resolved, view, recipe)


def build_badcases_and_pairs(
    resolved_list: list[ResolvedEpisode],
    kept_pieces: list[KeepPiece],
    recipe: ExportRecipe,
) -> dict[str, Any]:
    """Build `badcases_and_pairs.json` summary for `badcase_pairs` exports."""
    badcases: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []

    kept_by_ep: dict[str, list[KeepPiece]] = {}
    for p in kept_pieces:
        kept_by_ep.setdefault(p.episode_id, []).append(p)

    for res in resolved_list:
        if recipe.dataset_filter and res.meta.dataset != recipe.dataset_filter:
            continue
        if recipe.task_filter and res.meta.task_id != recipe.task_filter:
            continue
        ep_pieces = kept_by_ep.get(res.episode_id, [])
        if not ep_pieces:
            continue
        fps = float(res.meta.fps or 30.0)
        segs = sorted(res.qc_segments, key=lambda s: (s.t_start, s.t_end))
        bad_segs = [s for s in segs if s.training_role in ("badcase", "terminal_fail")]
        pos_segs = [s for s in segs if s.training_role in ("recovery", "necessary")]

        for bseg in bad_segs:
            badcases.append(
                {
                    "episode_id": res.episode_id,
                    "seg_id": bseg.seg_id,
                    "label_id": bseg.label_id,
                    "training_role": bseg.training_role,
                    "splice_policy": bseg.splice_policy,
                    "mask_side": bseg.mask_side,
                    "t_start": bseg.t_start,
                    "t_end": bseg.t_end,
                    "start_row": int(round(bseg.t_start * fps)),
                    "end_row": int(round(bseg.t_end * fps)),
                    "passed": res.success.passed,
                    "visual_evidence": bseg.visual_evidence.model_dump(mode="json"),
                }
            )
            if bseg.training_role != "terminal_fail":
                for pseg in pos_segs:
                    if 0.0 <= pseg.t_start - bseg.t_end <= 6.0:
                        pairs.append(
                            {
                                "episode_id": res.episode_id,
                                "pair_type": "badcase_to_recovery",
                                "rejected_seg_id": bseg.seg_id,
                                "chosen_seg_id": pseg.seg_id,
                                "rejected_span": [bseg.t_start, bseg.t_end],
                                "chosen_span": [pseg.t_start, pseg.t_end],
                            }
                        )
                        break

    return {
        "schema": "robo_vlm_qc.badcases_and_pairs.v1",
        "recipe": recipe.name,
        "badcases": badcases,
        "pairs": pairs,
        "preference_pairs": pairs,
        "pieces": [
            {
                "piece_id": p.piece_id,
                "episode_id": p.episode_id,
                "start_row": p.start_row,
                "end_row": p.end_row,
                "t_start_s": p.t_start_s,
                "t_end_s": p.t_end_s,
            }
            for p in kept_pieces
        ],
    }
