"""Causal 5-stage VLM DAG, multi-provider client, content-addressed cache, and prompts."""

from .cache import StageCache
from .client import (
    BudgetExceeded,
    BudgetTracker,
    VlmCancelledError,
    VlmClient,
    VlmError,
    parse_json,
)
from .dag import VlmDag
from .prompt_builder import PromptBuilder, describe_camera_layout
from .stage1_grounding import run_stage1_grounding
from .stage2_macro import run_stage2_macro
from .stage3_discover import run_stage3_discover
from .stage4_verify import run_stage4_verify, run_stage4a_merge, run_stage4b_verify
from .stage5_success import run_stage5_success
from .tracker import (
    BaftTrackingOutput,
    TrackingConfig,
    run_baft_tracking,
    track_interval_bidirectional,
)

__all__ = [
    "BaftTrackingOutput",
    "BudgetExceeded",
    "BudgetTracker",
    "PromptBuilder",
    "StageCache",
    "TrackingConfig",
    "VlmCancelledError",
    "VlmClient",
    "VlmDag",
    "VlmError",
    "describe_camera_layout",
    "parse_json",
    "run_baft_tracking",
    "run_stage1_grounding",
    "run_stage2_macro",
    "run_stage3_discover",
    "run_stage4_verify",
    "run_stage4a_merge",
    "run_stage4b_verify",
    "run_stage5_success",
    "track_interval_bidirectional",
]
