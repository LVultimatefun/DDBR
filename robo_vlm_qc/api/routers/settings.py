"""Workspace review & batch concurrency settings endpoints."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ...config import ConfigManager, deep_merge
from ...lab.eval_runner import archive_and_write_override, workspace_config_lock
from ..auth import AuthContext, get_app_state, require_auth, require_lead
from ..state import AppState

router = APIRouter(prefix="/api/settings", tags=["settings"])


class SettingsUpdatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_watch_coverage: float | None = None
    batch_vlm_concurrency: int | None = None

    @field_validator("min_watch_coverage", mode="before")
    @classmethod
    def _validate_coverage(cls, v: Any) -> float | None:
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError("min_watch_coverage must be a float, not bool")
        try:
            f = float(v)
        except (TypeError, ValueError) as exc:
            raise ValueError("min_watch_coverage must be a finite number") from exc
        if not math.isfinite(f):
            raise ValueError("min_watch_coverage must be finite")
        if f < 0.0 or f > 1.0:
            raise ValueError("min_watch_coverage must be between 0.0 and 1.0")
        return f

    @field_validator("batch_vlm_concurrency", mode="before")
    @classmethod
    def _validate_concurrency(cls, v: Any) -> int | None:
        if v is None:
            return None
        if isinstance(v, bool) or isinstance(v, float):
            raise ValueError("batch_vlm_concurrency must be an integer between 1 and 32")
        if isinstance(v, str):
            s = v.strip()
            if not s or not s.lstrip("-+").isdigit():
                raise ValueError("batch_vlm_concurrency must be an integer between 1 and 32")
            val = int(s)
        elif isinstance(v, int):
            val = v
        else:
            raise ValueError("batch_vlm_concurrency must be an integer between 1 and 32")
        if val < 1 or val > 32:
            raise ValueError("batch_vlm_concurrency must be between 1 and 32")
        return val

    @model_validator(mode="after")
    def _require_any_field(self) -> "SettingsUpdatePayload":
        if self.min_watch_coverage is None and self.batch_vlm_concurrency is None:
            raise ValueError("At least one of min_watch_coverage or batch_vlm_concurrency is required")
        return self


def _read_settings_dict(cfg: ConfigManager) -> dict[str, Any]:
    review_cfg = cfg.settings.get("review", {})
    vlm_cfg = cfg.settings.get("vlm", {})
    min_cov = float(review_cfg.get("min_watch_coverage", 0.95))
    conc_raw = vlm_cfg.get("episode_concurrency")
    if conc_raw is None:
        conc_raw = vlm_cfg.get("batch_vlm_concurrency", 4)
    return {
        "min_watch_coverage": min_cov,
        "batch_vlm_concurrency": int(conc_raw),
    }


@router.get("")
def get_settings(
    auth: AuthContext = Depends(require_auth),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    return _read_settings_dict(state.get_config())


@router.put("")
def put_settings(
    payload: SettingsUpdatePayload,
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    with workspace_config_lock(state.workspace):
        ov_path = state.workspace / "configs_override" / "default.yaml"
        existing: dict[str, Any] = {}
        if ov_path.is_file():
            loaded = yaml.safe_load(ov_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
            elif loaded is not None:
                raise HTTPException(
                    status_code=500,
                    detail="Existing configs_override/default.yaml is not a YAML mapping",
                )

        patch: dict[str, Any] = {}
        if payload.min_watch_coverage is not None:
            patch.setdefault("review", {})["min_watch_coverage"] = float(
                payload.min_watch_coverage
            )
        if payload.batch_vlm_concurrency is not None:
            patch.setdefault("vlm", {})["episode_concurrency"] = int(
                payload.batch_vlm_concurrency
            )
            if (
                isinstance(existing.get("vlm"), dict)
                and "batch_vlm_concurrency" in existing["vlm"]
            ):
                patch["vlm"]["batch_vlm_concurrency"] = int(payload.batch_vlm_concurrency)

        merged = deep_merge(existing, patch)
        yaml_text = yaml.safe_dump(merged, sort_keys=False, allow_unicode=True)
        archive_and_write_override(
            state.workspace,
            state.config_dir,
            Path("default.yaml"),
            yaml_text,
            reason="settings_update",
        )
        state.invalidate_config()
        res = _read_settings_dict(state.get_config())

    state.sse.publish({"type": "settings_updated", **res})
    return res
