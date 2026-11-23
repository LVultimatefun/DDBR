"""Output exporters (PieceSlicer, OutputExporter registry, and 4 built-in exporters)."""

from .base import (
    OutputExporter,
    atomic_export_dir,
    build_global_subtasks,
    finalize_manifest,
    get_exporter,
    group_kept_pieces,
    list_exporters,
    register_exporter,
    staged_export_dir,
)
from .h5_clean import H5CleanExporter
from .lerobot_v3 import LeRobotV3Exporter
from .slicer import (
    CANONICAL_81D_NAMES,
    NOT_SMOOTHED,
    PieceSlicer,
    build_badcases_and_pairs,
    extract_canonical_81d_state,
    extract_piece_kinematics_81d,
    slice_episode,
    smooth_tail_81d,
    smooth_tail_vector,
    smoothable,
)
from .viz_mp4 import (
    VizMp4Exporter,
    render_episode_mp4,
    render_studio_hud_mp4,
    render_viz_mp4,
)
from .zero_copy_index import ZeroCopyIndexExporter

register_exporter(LeRobotV3Exporter, override=True)
register_exporter(H5CleanExporter, override=True)
register_exporter(ZeroCopyIndexExporter, override=True)
register_exporter(VizMp4Exporter, override=True)

__all__ = [
    "CANONICAL_81D_NAMES",
    "H5CleanExporter",
    "LeRobotV3Exporter",
    "NOT_SMOOTHED",
    "OutputExporter",
    "PieceSlicer",
    "VizMp4Exporter",
    "ZeroCopyIndexExporter",
    "atomic_export_dir",
    "build_badcases_and_pairs",
    "build_global_subtasks",
    "extract_canonical_81d_state",
    "extract_piece_kinematics_81d",
    "finalize_manifest",
    "get_exporter",
    "group_kept_pieces",
    "list_exporters",
    "register_exporter",
    "render_episode_mp4",
    "render_studio_hud_mp4",
    "render_viz_mp4",
    "slice_episode",
    "smooth_tail_81d",
    "smooth_tail_vector",
    "smoothable",
    "staged_export_dir",
]
