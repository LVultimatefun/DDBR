"""Episode metadata and hardware specification models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AdapterType = Literal["h5_raw_v3", "lerobot", "video_dir"]
CameraRole = Literal[
    "main",
    "front",
    "left_wrist",
    "right_wrist",
    "wide",
    "wide_right",
    "secondary",
    "aux",
]


class CameraSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    role: CameraRole
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0.0)
    source_type: str = "unknown"
    is_virtual_crop: bool = False
    source_stream: str | None = None


class KinematicsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    has_kinematics: bool
    n_arms: int = Field(ge=0, default=2)
    arm_dof: int = Field(ge=0, default=7)
    hand_dof: int = Field(ge=0, default=44)
    torso_dof: int = Field(ge=0, default=0)
    has_chassis: bool = False
    has_tactile: bool = False
    tactile_dim: int = Field(ge=0, default=0)
    state_dim: int = Field(ge=0, default=0)
    has_action_cmd: bool = False
    source_hz: dict[str, float] = Field(default_factory=dict)


class EpisodeMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(min_length=3)
    dataset: str = Field(min_length=1)
    stem: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    adapter_type: AdapterType
    source_uri: str = Field(min_length=1)
    source_mtime: float = Field(ge=0.0)
    episode_fingerprint: str = Field(min_length=1)
    duration_s: float = Field(gt=0.0)
    fps: float = Field(gt=0.0)
    frame_count: int = Field(gt=0)
    cameras: list[CameraSpec] = Field(min_length=1)
    camera_roles: dict[str, str]
    kinematics: KinematicsSpec
    recorded_success: bool | None = None
    extra_meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("dataset", "stem")
    @classmethod
    def _validate_path_component(cls, v: str) -> str:
        if not v or v in (".", "..") or "/" in v or "\\" in v or ".." in v:
            raise ValueError(f"Invalid path component: {v!r}")
        return v

    @field_validator("episode_id")
    @classmethod
    def _validate_episode_id(cls, v: str) -> str:
        if ".." in v or "\\" in v or v.count("/") != 1:
            raise ValueError(
                f"episode_id must be '<dataset>/<stem>' with a single '/', got {v!r}"
            )
        ds, st = v.split("/", 1)
        if not ds or not st or ds in (".", "..") or st in (".", ".."):
            raise ValueError(f"Invalid episode_id segments: {v!r}")
        return v

    @model_validator(mode="after")
    def _validate_consistency(self) -> "EpisodeMeta":
        expected_id = f"{self.dataset}/{self.stem}"
        if self.episode_id != expected_id:
            raise ValueError(
                f"episode_id {self.episode_id!r} must equal '{expected_id}'"
            )
        cam_names = [c.name for c in self.cameras]
        if len(cam_names) != len(set(cam_names)):
            raise ValueError(f"Duplicate camera names in EpisodeMeta: {cam_names}")
        return self
