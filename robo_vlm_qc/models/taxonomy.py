"""Visual taxonomy, task spec, prompt template, and few-shot models."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .annotation import SplicePolicy, TrainingRole

_LABEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def _alias_legacy_key(data: Any, legacy_key: str, target_key: str) -> Any:
    if isinstance(data, dict) and legacy_key in data:
        d = dict(data)
        legacy_val = d.pop(legacy_key)
        if target_key not in d or d[target_key] is None:
            d[target_key] = legacy_val
        return d
    return data

LabelCategory = Literal[
    "idle",
    "failure_recovery",
    "inefficiency",
    "failure",
    "necessary_skill",
    "terminal",
    "unknown",
]
LabelStatus = Literal["active", "proposed", "retired"]
DiagnosisErrorType = Literal[
    "fp_overflag",
    "fn_missed",
    "policy_flip",
    "boundary",
    "success_flip",
    "stage",
]


class CollectorUiHint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    hint: str = ""


class VlmSemantics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    definition: str
    counter_examples: list[str] = Field(default_factory=list)
    visual_cues: list[str] = Field(default_factory=list)


class VisualLabelDef(BaseModel):
    """Pure-visual label definition (zero rule operators)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    category: LabelCategory
    status: LabelStatus = "active"
    version: int = Field(ge=1, default=1)
    default_training_role: TrainingRole
    default_splice_policy: SplicePolicy
    collector_ui: CollectorUiHint
    vlm_semantics: VlmSemantics
    merged_from: list[str] = Field(default_factory=list)
    merged_into: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _alias_legacy_fields(cls, data: Any) -> Any:
        return _alias_legacy_key(data, "name_zh", "name")

    @field_validator("id")
    @classmethod
    def _validate_id(cls, v: str) -> str:
        if not _LABEL_ID_RE.match(v):
            raise ValueError(f"Invalid label id {v!r}; must match {_LABEL_ID_RE.pattern}")
        if v in {"long_pause", "back_and_forth"}:
            raise ValueError(f"Legacy label id {v!r} is abolished")
        return v


class LabelRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    version: int = Field(ge=1, default=1)
    categories: list[LabelCategory] = Field(default_factory=list)
    labels: list[VisualLabelDef] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> "LabelRegistry":
        ids = [lbl.id for lbl in self.labels]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate label ids in LabelRegistry: {ids}")
        return self

    def get(self, label_id: str) -> VisualLabelDef | None:
        for lbl in self.labels:
            if lbl.id == label_id:
                return lbl
        return None


class GroundingObjectSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str
    desc: str = ""
    tag_pos: Literal["top", "bottom"] = "top"
    is_static: bool = False

    @model_validator(mode="before")
    @classmethod
    def _alias_legacy_fields(cls, data: Any) -> Any:
        return _alias_legacy_key(data, "name_zh", "name")


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    version: str = "1.0.0"
    match: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    title: str = ""
    description: str = ""
    goal: str = ""
    criteria: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    task_prompt: str = ""
    grounding_objects: list[GroundingObjectSpec] = Field(default_factory=list)
    hand_mapping: dict[str, str] | None = None
    camera_roles_override: dict[str, str] | None = None

    @model_validator(mode="before")
    @classmethod
    def _alias_legacy_fields(cls, data: Any) -> Any:
        return _alias_legacy_key(data, "title_zh", "title")


class PromptTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: Literal[
        "s1_grounding",
        "s2_macro",
        "s3_discover",
        "s4_verify",
        "s5_success",
    ]
    version: str = "1.0.0"
    description: str = ""
    system_template: str = Field(min_length=1)
    user_template: str = Field(min_length=1)
    extra_templates: dict[str, str] = Field(default_factory=dict)


class PromptBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    s1_grounding: PromptTemplate
    s2_macro: PromptTemplate
    s3_discover: PromptTemplate
    s4_verify: PromptTemplate
    s5_success: PromptTemplate


class FewShotItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    example_id: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    task_id: str | None = None
    label_id: str | None = None
    error_type: DiagnosisErrorType
    episode_id: str = Field(min_length=3)
    t_start: float = Field(ge=0.0)
    t_end: float = Field(gt=0.0)
    tile_image_path: str | None = None
    vlm_wrong_json: dict[str, Any] = Field(default_factory=dict)
    human_right_json: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(min_length=1)
    enabled: bool = True
    created_at: float

    @field_validator("scope")
    @classmethod
    def _validate_scope(cls, v: str) -> str:
        if v == "global":
            return v
        if v.startswith("task:") or v.startswith("label:"):
            suffix = v.split(":", 1)[1].strip()
            if suffix:
                return v
        raise ValueError(
            f"scope must be 'global', 'task:<id>', or 'label:<id>' with non-empty id, got {v!r}"
        )

    @model_validator(mode="after")
    def _validate_span(self) -> "FewShotItem":
        if self.t_end <= self.t_start:
            raise ValueError(
                f"FewShotItem t_end ({self.t_end}) must be > t_start ({self.t_start})"
            )
        return self
