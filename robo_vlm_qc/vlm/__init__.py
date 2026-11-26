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
    effective_vision_min_tokens_per_image,
    estimate_stage_cost_usd,
    extract_json,
    normalize_image_detail,
    parse_json,
    redact_secrets,
    resolve_client_stage_model,
    resolve_stage_model,
)
from .dag import VlmDag
from .estimator import estimate_batch_vlm_cost
from .frame_dedup import (
    FrameDedupResult,
    deduplicate_frames,
    format_skipped_ranges_note,
)
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
    "FrameDedupResult",
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
    "deduplicate_frames",
    "describe_camera_layout",
    "effective_vision_min_tokens_per_image",
    "estimate_batch_vlm_cost",
    "estimate_stage_cost_usd",
    "extract_json",
    "format_skipped_ranges_note",
    "normalize_image_detail",
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
