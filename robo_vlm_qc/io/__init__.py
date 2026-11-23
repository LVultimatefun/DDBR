"""Pluggable 3-in InputAdapter layer (auto-registers built-in adapters on import)."""

from .base import (
    LABEL_DIRS,
    SKIP_DIRS,
    EpisodeHandle,
    EpisodeView,
    InputAdapter,
    KinematicsData,
    KinematicsTrack,
    MediaSource,
    auto_detect,
    build_camera_roles,
    compute_episode_fingerprint,
    dataset_of,
    detect_adapter,
    get_adapter,
    infer_camera_role,
    interp_nan,
    list_adapters,
    register_adapter,
    resolve_task_id,
)
from .h5_raw_v3 import H5MediaSource, H5RawV3Adapter
from .lerobot import LeRobotAdapter, LeRobotMediaSource
from .media import (
    Mp4Reader,
    PreviewCache,
    SampledMediaReader,
    ZoomBox,
    compose_2x2_tile,
    encode_jpeg_bytes,
    encode_jpeg_data_url,
    probe_video_cv2,
    probe_video_ffprobe,
    select_tile_mode,
    time_to_frame_idx,
)
from .video_dir import VideoDirAdapter, VideoDirMediaSource

# Auto-register all 3 built-in adapters in priority order
for _adapter_cls in (
    H5RawV3Adapter,
    LeRobotAdapter,
    VideoDirAdapter,
):
    register_adapter(_adapter_cls, override=True)

__all__ = [
    "EpisodeHandle",
    "EpisodeView",
    "H5MediaSource",
    "H5RawV3Adapter",
    "InputAdapter",
    "KinematicsData",
    "KinematicsTrack",
    "LABEL_DIRS",
    "LeRobotAdapter",
    "LeRobotMediaSource",
    "MediaSource",
    "Mp4Reader",
    "PreviewCache",
    "SKIP_DIRS",
    "SampledMediaReader",
    "VideoDirAdapter",
    "VideoDirMediaSource",
    "ZoomBox",
    "auto_detect",
    "build_camera_roles",
    "compose_2x2_tile",
    "compute_episode_fingerprint",
    "dataset_of",
    "detect_adapter",
    "encode_jpeg_bytes",
    "encode_jpeg_data_url",
    "get_adapter",
    "infer_camera_role",
    "interp_nan",
    "list_adapters",
    "probe_video_cv2",
    "probe_video_ffprobe",
    "register_adapter",
    "resolve_task_id",
    "select_tile_mode",
    "time_to_frame_idx",
]
