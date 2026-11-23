"""VLM annotation domain models (strictly zero legacy rule fields)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_FORBIDDEN_LEGACY_NAMES = frozenset({"long_pause", "back_and_forth"})


def _alias_legacy_key(data: Any, legacy_key: str, target_key: str) -> Any:
    if isinstance(data, dict) and legacy_key in data:
        d = dict(data)
        legacy_val = d.pop(legacy_key)
        if target_key not in d or d[target_key] is None:
            d[target_key] = legacy_val
        return d
    return data


def _check_no_legacy(data: Any) -> Any:
    if isinstance(data, dict):
        lowered_keys = {
            k.strip().lower(): k for k in data.keys() if isinstance(k, str)
        }
        for bad_key in _FORBIDDEN_LEGACY_NAMES:
            if bad_key in lowered_keys:
                raise ValueError(
                    f"Legacy field {lowered_keys[bad_key]!r} is abolished in robo_vlm_qc"
                )
        for field_name in ("type", "label_id", "segment_type"):
            val = data.get(field_name)
            if isinstance(val, str) and val.strip().lower() in _FORBIDDEN_LEGACY_NAMES:
                raise ValueError(
                    f"Legacy segment type {val!r} is abolished in robo_vlm_qc"
                )
    return data


class GroundingKeyframe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row: int = Field(ge=0)
    time_s: float = Field(ge=0.0)
    visible: bool
    bbox_norm_1000: list[int] = Field(min_length=4, max_length=4)
    bbox_px: list[int] = Field(default_factory=lambda: [0, 0, 0, 0], min_length=4, max_length=4)
    state: str = ""
    status_kind: Literal["neutral", "warn", "error", "good"] = "neutral"

    @model_validator(mode="after")
    def _validate_bbox(self) -> "GroundingKeyframe":
        x1, y1, x2, y2 = self.bbox_norm_1000
        if not (0 <= x1 <= x2 <= 1000 and 0 <= y1 <= y2 <= 1000):
            raise ValueError(f"Invalid bbox_norm_1000: {self.bbox_norm_1000}")
        if not self.visible and self.bbox_norm_1000 != [0, 0, 0, 0]:
            raise ValueError("Invisible grounding keyframe must have [0, 0, 0, 0] bbox")
        return self


class GroundingTrack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_id: str = Field(min_length=1)
    name: str = ""
    tag_pos: Literal["top", "bottom"] = "top"
    is_static_container: bool = False
    keyframes: list[GroundingKeyframe] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _alias_legacy_fields(cls, data: Any) -> Any:
        return _alias_legacy_key(data, "name_zh", "name")


class GroundingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    ref_size: list[int] = Field(default_factory=lambda: [1280, 720], min_length=2, max_length=2)
    keyframes_s: list[float] = Field(default_factory=list)
    object_tracks_2d: dict[str, GroundingTrack] = Field(default_factory=dict)


class MacroStage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage_id: str = Field(min_length=1)
    t_start: float = Field(ge=0.0)
    t_end: float = Field(gt=0.0)
    label: str = Field(min_length=1)
    label_en: str = ""
    summary: str = ""
    color_rgb: list[int] = Field(
        default_factory=lambda: [59, 130, 246], min_length=3, max_length=3
    )

    @model_validator(mode="before")
    @classmethod
    def _alias_legacy_fields(cls, data: Any) -> Any:
        return _alias_legacy_key(data, "label_zh", "label")

    @field_validator("color_rgb")
    @classmethod
    def _validate_rgb(cls, v: list[int]) -> list[int]:
        if any(c < 0 or c > 255 for c in v):
            raise ValueError(f"color_rgb channels must be in [0, 255], got {v}")
        return v

    @model_validator(mode="after")
    def _validate_span(self) -> "MacroStage":
        if self.t_end <= self.t_start:
            raise ValueError(
                f"MacroStage t_end ({self.t_end}) must be > t_start ({self.t_start})"
            )
        return self


ArmState = Literal["idle", "work", "hold", "anom", "mask", "good"]


class ArmSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t_start: float = Field(ge=0.0)
    t_end: float = Field(gt=0.0)
    state: ArmState
    desc: str = ""

    @model_validator(mode="after")
    def _validate_span(self) -> "ArmSpan":
        if self.t_end <= self.t_start:
            raise ValueError(
                f"ArmSpan t_end ({self.t_end}) must be > t_start ({self.t_start})"
            )
        return self


class ArmTimelines(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left: list[ArmSpan] = Field(default_factory=list)
    right: list[ArmSpan] = Field(default_factory=list)


TrainingRole = Literal["necessary", "recovery", "badcase", "terminal_fail"]
SplicePolicy = Literal["keep_positive", "mask_loss", "cut_clean"]
MaskSide = Literal["left", "right", "both"]


class VisualEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_state: str = ""
    process_change: str = ""
    end_state: str = ""
    confidence_note: str = ""

    def as_text(self) -> str:
        parts = [
            p.strip()
            for p in (self.initial_state, self.process_change, self.end_state)
            if p and p.strip()
        ]
        return " ".join(parts)


class GuardrailMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eef_pos_m: float = 0.0
    eef_rot_rad: float = 0.0
    hand_rad: float = 0.0
    torso_rad: float = 0.0
    chassis_pos_m: float = 0.0
    chassis_yaw_rad: float = 0.0
    other_arm_disp_m: float = 0.0
    bimanual_coupled: bool = False
    visual_ssim: float | None = None


class GuardrailReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gate_type: Literal["splice_gate", "hallucination", "visual_gate", "none"] = "none"
    passed: bool = True
    downgraded: bool = False
    downgrade_reason: str | None = None
    guardrail_override: bool = False
    override_reason: str | None = None
    metrics: GuardrailMetrics = Field(default_factory=GuardrailMetrics)
    opt_t_start: float | None = None
    opt_t_end: float | None = None


class QcSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seg_id: str = Field(min_length=1)
    t_start: float = Field(ge=0.0)
    t_end: float = Field(gt=0.0)
    label_id: str = Field(min_length=1)
    name: str = ""
    training_role: TrainingRole
    splice_policy: SplicePolicy
    mask_side: MaskSide | None = None
    affected_arm: MaskSide = "both"
    visual_evidence: VisualEvidence = Field(default_factory=VisualEvidence)
    vlm_raw_policy: SplicePolicy
    guardrail: GuardrailReport = Field(default_factory=GuardrailReport)
    is_unknown: bool = False
    unknown_note: str = ""
    suggested_label: str | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.9)

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_inputs(cls, data: Any) -> Any:
        return _alias_legacy_key(_check_no_legacy(data), "name_zh", "name")

    @model_validator(mode="after")
    def _validate_triplet_and_span(self) -> "QcSegment":
        if self.t_end <= self.t_start:
            raise ValueError(
                f"QcSegment t_end ({self.t_end}) must be > t_start ({self.t_start})"
            )
        if self.label_id.strip().lower() in _FORBIDDEN_LEGACY_NAMES:
            raise ValueError(f"Legacy label_id {self.label_id!r} is abolished")

        if self.splice_policy == "keep_positive":
            if self.mask_side is not None:
                raise ValueError("keep_positive segment must have mask_side=None")
            if self.training_role not in {"necessary", "recovery"}:
                raise ValueError(
                    f"keep_positive invalid with training_role={self.training_role!r}"
                )
        elif self.splice_policy == "mask_loss":
            if self.mask_side not in {"left", "right", "both"}:
                raise ValueError(
                    "mask_loss segment must specify mask_side in {'left', 'right', 'both'}"
                )
            if self.training_role not in {"badcase", "terminal_fail", "recovery"}:
                raise ValueError(
                    f"mask_loss invalid with training_role={self.training_role!r}"
                )
        elif self.splice_policy == "cut_clean":
            if self.mask_side is not None:
                raise ValueError("cut_clean segment must have mask_side=None")
            if self.training_role not in {"badcase", "terminal_fail"}:
                raise ValueError(
                    f"cut_clean invalid with training_role={self.training_role!r}"
                )
        return self


class CriterionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criterion: str
    met: bool | None = None
    satisfied_at_s: float | None = None
    evidence: str = ""


class ControlCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cut_s: float
    cut_ratio: float
    control_verdict: Literal["success", "failure", "unsure"]
    passed: bool
    met: list[bool | None] = Field(default_factory=list)


class SuccessVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool | None = None
    verdict: Literal["success", "failure", "unsure"] = "unsure"
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    criteria: list[CriterionResult] = Field(default_factory=list)
    control_check: ControlCheckResult | None = None
    summary: str = ""


class StageUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    elapsed_s: float = 0.0
    estimated_cost_usd: float = 0.0
    cache_hit: bool = False


class UsageSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stages: list[StageUsage] = Field(default_factory=list)
    total_tokens: int = 0
    total_elapsed_s: float = 0.0
    total_cost_usd: float = 0.0


class ConfigFingerprintSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_routing: dict[str, str]
    prompt_bundle_version: str
    label_taxonomy_version: str
    task_spec_version: str
    few_shot_pack_version: str = "none"
    config_sha256: str = ""


class StageRefs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    s1_grounding: str | None = None
    s2_macro: str | None = None
    s3_discover: str | None = None
    s4_verify: str | None = None
    s5_success: str | None = None


class VlmRunArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    run_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=3)
    created_at: float
    is_eval_sandbox: bool = False
    config_fingerprint: ConfigFingerprintSnapshot
    stage_refs: StageRefs = Field(default_factory=StageRefs)
    grounding: GroundingResult = Field(default_factory=GroundingResult)
    macro_stages: list[MacroStage] = Field(default_factory=list)
    arm_timelines: ArmTimelines = Field(default_factory=ArmTimelines)
    qc_segments: list[QcSegment] = Field(default_factory=list)
    success: SuccessVerdict = Field(default_factory=SuccessVerdict)
    usage: UsageSummary = Field(default_factory=UsageSummary)

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_inputs(cls, data: Any) -> Any:
        return _check_no_legacy(data)
