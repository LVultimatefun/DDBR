"""Export recipe, piece slicer, and manifest models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .review import EpisodeStatus

ExportFormat = Literal["lerobot_v3", "h5_clean", "zero_copy_index", "viz_mp4"]
ExportRecipeName = Literal["bc_clean", "recovery_plus", "badcase_pairs"]
SmoothTailMode = Literal["on_jump", "always", "never"]


class ExportRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: ExportRecipeName
    format: ExportFormat
    chunk_size: int | None = Field(default=None, ge=2, le=2000)
    min_piece_s: float = Field(default=0.1, gt=0.0)
    tail_frames: int | None = Field(default=None, ge=0, le=2000)
    smooth_tail: SmoothTailMode = "on_jump"
    include_statuses: list[EpisodeStatus] = Field(
        default_factory=lambda: [EpisodeStatus.APPROVED, EpisodeStatus.GT_FROZEN]
    )
    context_window_s: float = Field(default=2.0, ge=0.0)
    dataset_filter: str | None = None
    task_filter: str | None = None


class SubtaskSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subtask_index: int = Field(ge=0)
    stage_id: str
    label: str
    label_en: str = ""
    start_row: int = Field(ge=0)
    end_row: int = Field(gt=0)

    @model_validator(mode="before")
    @classmethod
    def _alias_legacy_fields(cls, data: Any) -> Any:
        if isinstance(data, dict) and "label_zh" in data:
            d = dict(data)
            legacy_val = d.pop("label_zh")
            if "label" not in d or d["label"] is None:
                d["label"] = legacy_val
            return d
        return data

    @model_validator(mode="after")
    def _validate_rows(self) -> "SubtaskSpan":
        if self.end_row <= self.start_row:
            raise ValueError("SubtaskSpan end_row must be > start_row")
        return self


class KeepPiece(BaseModel):
    model_config = ConfigDict(extra="forbid")

    piece_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=3)
    piece_index: int = Field(ge=0)
    start_row: int = Field(ge=0)
    end_row: int = Field(gt=0)
    tail_start_row: int | None = None
    tail_end_row: int | None = None
    own_rows: int = Field(gt=0)
    tail_rows: int = Field(ge=0, default=0)
    total_rows: int = Field(gt=0)
    fps: float = Field(gt=0.0)
    t_start_s: float = Field(ge=0.0)
    t_end_s: float = Field(gt=0.0)
    action_loss_mask: list[list[float]] = Field(default_factory=list)
    is_tail: list[bool] = Field(default_factory=list)
    subtask_indices: list[int] = Field(default_factory=list)
    subtasks: list[SubtaskSpan] = Field(default_factory=list)
    smooth_tail_applied: bool = False
    dropped: bool = False
    drop_reason: str | None = None

    @model_validator(mode="after")
    def _validate_piece(self) -> "KeepPiece":
        if self.end_row <= self.start_row:
            raise ValueError(
                f"KeepPiece end_row ({self.end_row}) must be > start_row ({self.start_row})"
            )
        if self.t_end_s <= self.t_start_s:
            raise ValueError(
                f"KeepPiece t_end_s ({self.t_end_s}) must be > t_start_s ({self.t_start_s})"
            )
        if self.own_rows != (self.end_row - self.start_row):
            raise ValueError(
                f"own_rows ({self.own_rows}) must equal end_row - start_row ({self.end_row - self.start_row})"
            )
        if self.total_rows != (self.own_rows + self.tail_rows):
            raise ValueError(
                f"total_rows ({self.total_rows}) must equal own_rows + tail_rows"
            )
        return self


class ExportEpisodeEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    status: EpisodeStatus
    passed: bool | None = None
    pieces_exported: int = 0
    pieces_dropped: int = 0
    own_frames: int = 0
    total_frames: int = 0
    dropped_reasons: list[str] = Field(default_factory=list)


class ExportManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    export_id: str = Field(min_length=1)
    created_at: float
    format: ExportFormat
    recipe: ExportRecipe
    output_dir: str
    episodes_included: int = 0
    episodes_skipped: int = 0
    pieces_exported: int = 0
    pieces_dropped: int = 0
    total_frames: int = 0
    episodes: list[ExportEpisodeEntry] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    checksums: dict[str, str] = Field(default_factory=dict)
