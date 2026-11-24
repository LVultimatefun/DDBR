"""Pluggable InputAdapter registry, MediaSource/KinematicsTrack protocols, and shared I/O helpers."""

from __future__ import annotations

import fnmatch
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np

from ..models.episode import CameraRole, CameraSpec, EpisodeMeta

LABEL_DIRS: dict[str, bool | None] = {
    "success": True,
    "fail": False,
    "failure": False,
    "unlabeled": None,
}

SKIP_DIRS: frozenset[str] = frozenset(
    {".history", ".qc_cache", "__pycache__", "exports", "artifacts", "_staging", ".git"}
)


@runtime_checkable
class MediaSource(Protocol):
    cameras: list[CameraSpec]
    duration_s: float
    fps: float

    def read_frame_rgb(self, camera: str, t_s: float) -> np.ndarray:
        """Return (H, W, 3) uint8 RGB array for `camera` at `t_s` seconds."""
        ...

    def read_frames_batch(self, camera: str, times_s: list[float]) -> list[np.ndarray]:
        """Return list of (H, W, 3) uint8 RGB arrays for `camera` at `times_s`."""
        ...


@runtime_checkable
class KinematicsTrack(Protocol):
    timestamps_s: np.ndarray
    eef_pos: np.ndarray | None
    eef_rotvec: np.ndarray | None
    hand_q: np.ndarray | None
    torso_q: np.ndarray | None
    chassis_pose: np.ndarray | None
    raw_state: np.ndarray | None
    raw_action: np.ndarray | None
    tactile: np.ndarray | None
    valid_mask: np.ndarray


@dataclass
class KinematicsData:
    """Concrete KinematicsTrack implementation with optional exporter/guardrail fields."""

    timestamps_s: np.ndarray
    eef_pos: np.ndarray | None = None
    eef_rotvec: np.ndarray | None = None
    hand_q: np.ndarray | None = None
    torso_q: np.ndarray | None = None
    chassis_pose: np.ndarray | None = None
    raw_state: np.ndarray | None = None
    raw_action: np.ndarray | None = None
    tactile: np.ndarray | None = None
    valid_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    # Auxiliary lossless fields
    arm_q: np.ndarray | None = None
    head_q: np.ndarray | None = None
    eef_quat_wxyz: np.ndarray | None = None
    hand_q_cmd: np.ndarray | None = None
    hand_cmd_valid: np.ndarray | None = None
    hand_tau: np.ndarray | None = None
    is_hil: np.ndarray | None = None
    t0_unix: float | None = None
    sub_valid: dict[str, np.ndarray] = field(default_factory=dict)
    state_names: list[str] | None = None
    action_names: list[str] | None = None


@dataclass(frozen=True)
class EpisodeHandle:
    episode_id: str
    dataset: str
    stem: str
    task_id: str
    adapter_type: str
    source_uri: str
    source_mtime: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source_path(self) -> Path:
        return Path(self.source_uri)


@dataclass
class EpisodeView:
    meta: EpisodeMeta
    media: MediaSource
    kinematics: KinematicsTrack | None = None

    @property
    def camera_roles(self) -> dict[str, str]:
        return dict(self.meta.camera_roles)

    @property
    def duration_s(self) -> float:
        return float(self.media.duration_s)

    @property
    def fps(self) -> float:
        return float(self.media.fps)

    @property
    def has_kinematics(self) -> bool:
        return self.kinematics is not None

    def close(self) -> None:
        closer = getattr(self.media, "close", None)
        if callable(closer):
            closer()

    def __enter__(self) -> "EpisodeView":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class InputAdapter(ABC):
    """Abstract base class for read-only embodied dataset adapters."""

    name: ClassVar[str]
    priority: ClassVar[int] = 100

    @abstractmethod
    def can_read(self, path: Path) -> bool:
        """Return True if `path` can be opened by this adapter."""
        ...

    @abstractmethod
    def discover(
        self,
        root: Path,
        *,
        task_patterns: dict[str, list[str]] | None = None,
    ) -> list[EpisodeHandle]:
        """Discover all readable episodes under `root` without modifying `root`."""
        ...

    @abstractmethod
    def open(self, handle: EpisodeHandle) -> EpisodeView:
        """Open `handle` read-only and return an EpisodeView."""
        ...


_ADAPTER_REGISTRY: dict[str, InputAdapter] = {}


def register_adapter(
    adapter: InputAdapter | type[InputAdapter], *, override: bool = False
) -> InputAdapter:
    inst = adapter() if isinstance(adapter, type) else adapter
    if not override and inst.name in _ADAPTER_REGISTRY:
        raise ValueError(f"Adapter {inst.name!r} is already registered")
    _ADAPTER_REGISTRY[inst.name] = inst
    return inst


def get_adapter(name: str) -> InputAdapter:
    if name not in _ADAPTER_REGISTRY:
        raise KeyError(
            f"Unknown adapter {name!r}; registered: {sorted(_ADAPTER_REGISTRY)}"
        )
    return _ADAPTER_REGISTRY[name]


def list_adapters() -> list[str]:
    return [
        ad.name
        for ad in sorted(
            _ADAPTER_REGISTRY.values(), key=lambda a: (a.priority, a.name)
        )
    ]


def detect_adapter(path: Path) -> InputAdapter | None:
    p = Path(path)
    for name in list_adapters():
        ad = _ADAPTER_REGISTRY[name]
        try:
            if ad.can_read(p):
                return ad
        except Exception:
            continue
    return None


def _is_under_any(path: Path, roots: list[Path]) -> bool:
    rp = path.resolve()
    for r in roots:
        if rp == r or r in rp.parents:
            return True
    return False


def auto_detect(
    root: Path,
    *,
    adapter_name: str | None = None,
    task_patterns: dict[str, list[str]] | None = None,
) -> list[EpisodeHandle]:
    root_path = Path(root).resolve()
    if adapter_name and adapter_name != "auto":
        return get_adapter(adapter_name).discover(
            root_path, task_patterns=task_patterns
        )

    results: list[EpisodeHandle] = []
    seen_keys: set[tuple[str, str]] = set()
    claimed_dirs: list[Path] = []

    for name in list_adapters():
        ad = _ADAPTER_REGISTRY[name]
        handles = ad.discover(root_path, task_patterns=task_patterns)
        for h in handles:
            src = Path(h.source_uri).resolve()
            if name == "video_dir" and _is_under_any(src, claimed_dirs):
                continue
            key = (str(src), h.episode_id)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            results.append(h)
            if name == "lerobot":
                claimed_dirs.append(src if src.is_dir() else src.parent)
    return results


def dataset_of(path: Path) -> str:
    """Resolve dataset name consistently across file and directory episodes."""
    p = Path(path).resolve()
    parent = p.parent
    if parent.name.lower() in LABEL_DIRS:
        ds = parent.parent.name
    else:
        ds = parent.name
    if not ds or ds in (".", "..") or "/" in ds or "\\" in ds:
        raise ValueError(f"Invalid dataset name derived from {path}: {ds!r}")
    return ds


def validate_id_part(part: str, label: str = "id") -> str:
    if not part or part in (".", "..") or "/" in part or "\\" in part or ".." in part:
        raise ValueError(f"Invalid {label}: {part!r}")
    return part


def compute_episode_fingerprint(
    source_uri: str, source_mtime: float, extra: str = ""
) -> str:
    payload = f"{source_uri}|{source_mtime:.6f}|{extra}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_task_id(
    dataset: str,
    fallback_task: str | None = None,
    task_patterns: dict[str, list[str]] | None = None,
) -> str:
    """Resolve canonical task_id: dataset globs first, then fallback hint, then _default."""
    patterns = task_patterns or {}
    ds_low = dataset.lower()
    ordered_tasks = sorted(tid for tid in patterns if tid != "_default")

    for tid in ordered_tasks:
        for pat in patterns[tid]:
            if fnmatch.fnmatchcase(ds_low, pat.lower()):
                return tid

    if fallback_task:
        hint = fallback_task.strip()
        hint_low = hint.lower()
        for tid in ordered_tasks:
            if tid.lower() == hint_low:
                return tid
            for pat in patterns[tid]:
                if fnmatch.fnmatchcase(hint_low, pat.lower()):
                    return tid

    return "_default"


_LEFT_WRIST_NAMES = frozenset(
    {
        "left_wrist",
        "l_wrist",
        "wrist_left",
        "wrist_l",
        "lwrist",
        "hand_left",
        "left_hand",
        "cam_left_wrist",
    }
)
_RIGHT_WRIST_NAMES = frozenset(
    {
        "right_wrist",
        "r_wrist",
        "wrist_right",
        "wrist_r",
        "rwrist",
        "hand_right",
        "right_hand",
        "cam_right_wrist",
    }
)
_FRONT_NAMES = frozenset(
    {"torso", "torso_rgbd", "torso_rgb", "chest", "front", "cam_front"}
)
_WIDE_NAMES = frozenset(
    {
        "wide",
        "stereo",
        "head_stereo",
        "head_stereo_left",
        "stereo_left",
        "overhead",
        "top",
        "third_person",
        "ext",
        "side",
    }
)
_WIDE_RIGHT_NAMES = frozenset({"head_stereo_right", "stereo_right", "wide_right"})
_MAIN_NAMES = frozenset(
    {
        "main",
        "ego",
        "head",
        "head_rgbd",
        "head_rgb",
        "rgb",
        "video",
        "cam_main",
        "cam_head",
        "cam0",
        "center",
        "cam_high",
    }
)


def is_camera_filename(stem: str) -> bool:
    s = stem.lower().strip()
    if s.startswith("observation.images."):
        s = s[len("observation.images.") :]
    return s in (
        _LEFT_WRIST_NAMES
        | _RIGHT_WRIST_NAMES
        | _FRONT_NAMES
        | _WIDE_NAMES
        | _WIDE_RIGHT_NAMES
        | _MAIN_NAMES
    )


def infer_camera_role(cam_name: str) -> CameraRole:
    s = cam_name.lower().strip()
    if s.startswith("observation.images."):
        s = s[len("observation.images.") :]
    if s in _LEFT_WRIST_NAMES:
        return "left_wrist"
    if s in _RIGHT_WRIST_NAMES:
        return "right_wrist"
    if s in _FRONT_NAMES:
        return "front"
    if s in _WIDE_NAMES:
        return "wide"
    if s in _WIDE_RIGHT_NAMES:
        return "wide_right"
    if s in _MAIN_NAMES:
        return "main"
    return "aux"


def build_camera_roles(cameras: list[CameraSpec]) -> dict[str, str]:
    roles: dict[str, str] = {}
    # Prefer explicit "ego" or "head_rgbd" for main if multiple main-like exist
    main_cams = [c for c in cameras if c.role == "main"]
    if main_cams:
        preferred = next(
            (c for c in main_cams if c.name.lower() in ("ego", "head_rgbd", "main")),
            main_cams[0],
        )
        roles["main"] = preferred.name
    for c in cameras:
        if c.role not in roles:
            roles[c.role] = c.name
    if "main" not in roles and cameras:
        non_wrist = next(
            (c for c in cameras if c.role not in ("left_wrist", "right_wrist")),
            cameras[0],
        )
        roles["main"] = non_wrist.name
    return roles


def interp_nan(arr: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN/Inf values along axis 0, clamping endpoints."""
    out = np.asarray(arr, dtype=np.float64).copy()
    if out.size == 0:
        return out
    orig_shape = out.shape
    flat = out.reshape(orig_shape[0], -1)
    idx = np.arange(flat.shape[0], dtype=np.float64)
    for col in range(flat.shape[1]):
        v = flat[:, col]
        ok = np.isfinite(v)
        if ok.all():
            continue
        if not ok.any():
            flat[:, col] = 0.0
            continue
        flat[:, col] = np.interp(idx, idx[ok], v[ok])
    return flat.reshape(orig_shape)
