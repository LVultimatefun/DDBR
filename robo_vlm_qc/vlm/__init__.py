"""Causal 5-stage VLM DAG, multi-provider client, content-addressed cache, and prompts."""

from .cache import StageCache
from .client import (
    BudgetExceeded,
    BudgetTracker,
    VlmBlindVisionError,
    VlmCancelledError,
    VlmCircuitBreaker,
    VlmCircuitOpenError,
    VlmClient,
    VlmError,
    estimate_stage_cost_usd,
    extract_json,
    parse_json,
    redact_secrets,
    resolve_client_stage_model,
    resolve_stage_model,
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
    "VlmBlindVisionError",
    "VlmCancelledError",
    "VlmCircuitBreaker",
    "VlmCircuitOpenError",
    "VlmClient",
    "VlmDag",
    "VlmError",
    "describe_camera_layout",
    "estimate_stage_cost_usd",
    "extract_json",
    "parse_json",
    "redact_secrets",
    "resolve_client_stage_model",
    "resolve_stage_model",
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
