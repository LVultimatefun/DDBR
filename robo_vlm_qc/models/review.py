"""Human review, diff tracking, status machine, and resolved episode models."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .annotation import (
    ArmTimelines,
    GroundingResult,
    MacroStage,
    QcSegment,
    SuccessVerdict,
)
from .episode import EpisodeMeta


class EpisodeStatus(str, Enum):
    INGESTED = "ingested"
    VLM_QUEUED = "vlm_queued"
    VLM_RUNNING = "vlm_running"
    VLM_READY = "vlm_ready"
    CLAIMED = "claimed"
    REVIEWED = "reviewed"
    FLAGGED = "flagged"
    APPROVED = "approved"
    GT_FROZEN = "gt_frozen"


ReviewRole = Literal["reviewer", "lead"]
ReviewAction = Literal["save_draft", "submit_review", "flag", "approve", "freeze_gt"]


class WatchCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    watched_intervals: list[tuple[float, float]] = Field(default_factory=list)
    watched_s: float = Field(ge=0.0, default=0.0)
    duration_s: float = Field(gt=0.0)
    coverage_ratio: float = Field(ge=0.0, le=1.0, default=0.0)
    confirmed: bool = False

    @model_validator(mode="after")
    def _validate_ratio_consistency(self) -> "WatchCoverage":
        expected = min(1.0, self.watched_s / self.duration_s)
        if self.coverage_ratio > expected + 0.05:
            raise ValueError(
                f"WatchCoverage coverage_ratio ({self.coverage_ratio:.3f}) exceeds "
                f"watched_s/duration_s ({expected:.3f})"
            )
        return self


class ReviewDiff(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fp_deleted: int = 0
    fn_added: int = 0
    policy_flips: int = 0
    label_changes: int = 0
    boundary_nudges: int = 0
    macro_stage_edits: int = 0
    success_flipped: bool = False
    total_edits: int = 0


class HumanReviewArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    rev_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=3)
    base_run_id: str = Field(min_length=1)
    author: str = Field(min_length=1)
    role: ReviewRole
    action: ReviewAction
    created_at: float
    watch_coverage: WatchCoverage
    qc_segments: list[QcSegment] = Field(default_factory=list)
    success: SuccessVerdict
    macro_stages: list[MacroStage] = Field(default_factory=list)
    notes: str = ""
    flagged_reason: str | None = None
    diff_summary: ReviewDiff = Field(default_factory=ReviewDiff)


class ReviewWritePayload(BaseModel):
    """Strict input DTO allowing ONLY the 3 editable layers + review metadata."""

    model_config = ConfigDict(extra="forbid")

    base_run_id: str = Field(min_length=1)
    action: ReviewAction
    qc_segments: list[QcSegment] = Field(default_factory=list)
    success: SuccessVerdict
    macro_stages: list[MacroStage] = Field(default_factory=list)
    watch_coverage: WatchCoverage
    notes: str = ""
    flagged_reason: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_readonly_layers(cls, data: Any) -> Any:
        if isinstance(data, dict):
            forbidden = {"grounding", "arm_timelines", "raw_vlm_segments"} & set(data.keys())
            if forbidden:
                raise ValueError(
                    f"Cannot modify read-only VLM layers in human review: {sorted(forbidden)}"
                )
        return data


class ResolvedEpisode(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    episode_id: str
    meta: EpisodeMeta
    status: EpisodeStatus
    active_run_id: str | None = None
    active_rev_id: str | None = None
    is_human_reviewed: bool = False
    qc_segments: list[QcSegment]
    success: SuccessVerdict
    macro_stages: list[MacroStage]
    grounding: GroundingResult
    arm_timelines: ArmTimelines
    raw_vlm_segments: list[QcSegment]
    raw_vlm_success: SuccessVerdict | None = None
    raw_vlm_macro_stages: list[MacroStage] = Field(default_factory=list)
    review_diff: ReviewDiff | None = None
    notes: str = ""
    flagged_reason: str | None = None
