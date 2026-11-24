"""Workspace review, batch concurrency, and VLM API/model settings endpoints."""

from __future__ import annotations

import contextlib
import math
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ...config import ConfigManager, deep_merge
from ...lab.eval_runner import archive_and_write_override, workspace_config_lock
from ...vlm.client import (
    VlmClient,
    VlmError,
    _parse_env_file_for_key,
    normalize_stage_name,
    resolve_api_key,
)
from ..auth import AuthContext, get_app_state, require_auth, require_lead
from ..state import AppState

router = APIRouter(prefix="/api/settings", tags=["settings"])

VlmProvider = Literal["openai_compat", "gemini"]
VlmThinkingMode = Literal["auto", "dashscope", "disabled"]

VLM_STAGE_KEYS: tuple[str, ...] = (
    "s1_grounding",
    "s2_macro",
    "s3_discover",
    "s4_verify",
    "s5_success",
)

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_HTTP_URL_RE = re.compile(r"^https?://[^\s]+$", re.IGNORECASE)

_PRESET_META: dict[str, dict[str, Any]] = {
    "openai": {
        "label": "OpenAI",
        "description": "OpenAI GPT-4o / GPT-4.1 Vision API",
        "fallback": {
            "provider": "openai_compat",
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "thinking_mode": "disabled",
            "stage_models": {k: "gpt-4o" for k in VLM_STAGE_KEYS},
        },
    },
    "dashscope": {
        "label": "DashScope (Qwen-VL)",
        "description": "Alibaba Cloud DashScope OpenAI-compatible Qwen-VL",
        "fallback": {
            "provider": "openai_compat",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key_env": "DASHSCOPE_API_KEY",
            "thinking_mode": "dashscope",
            "stage_models": {k: "qwen-vl-max" for k in VLM_STAGE_KEYS},
        },
    },
    "gemini": {
        "label": "Google Gemini",
        "description": "Google Gemini native multimodal API",
        "fallback": {
            "provider": "gemini",
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "api_key_env": "GEMINI_API_KEY",
            "thinking_mode": "auto",
            "stage_models": {k: "gemini-2.5-pro" for k in VLM_STAGE_KEYS},
        },
    },
    "vllm": {
        "label": "Local vLLM / SGLang",
        "description": "Local OpenAI-compatible vLLM or SGLang server",
        "fallback": {
            "provider": "openai_compat",
            "base_url": "http://127.0.0.1:8001/v1",
            "api_key_env": "VLLM_API_KEY",
            "thinking_mode": "disabled",
            "stage_models": {
                k: "Qwen/Qwen2.5-VL-72B-Instruct" for k in VLM_STAGE_KEYS
            },
        },
    },
}


def mask_api_key(key: str | None) -> str:
    """Return a masked preview of an API key (e.g. 'sk-***abcd' or '***'), never plaintext."""
    if not key:
        return ""
    k = str(key).strip()
    if not k:
        return ""
    if len(k) < 10:
        return "***"
    if k.startswith("sk-"):
        return f"sk-***{k[-4:]}"
    return f"{k[:3]}***{k[-4:]}"


def _redact_secret(text: str, secret: str | None) -> str:
    out = str(text or "")
    if secret:
        s = secret.strip()
        if len(s) >= 4:
            out = out.replace(s, "[REDACTED]")
    return out[:500]


def _validate_env_key_name(raw: str) -> str:
    s = str(raw).strip()
    if not _ENV_KEY_RE.match(s):
        raise ValueError(
            f"Invalid api_key_env {raw!r}; must match [A-Za-z_][A-Za-z0-9_]{{0,127}}"
        )
    return s


def _validate_http_url(raw: str) -> str:
    s = str(raw).strip().rstrip("/")
    if not s or not _HTTP_URL_RE.match(s):
        raise ValueError(
            f"Invalid base_url {raw!r}; must be a valid http:// or https:// URL"
        )
    return s


def _validate_api_key_value(raw: str) -> str:
    if any(ch in raw for ch in ("\r", "\n", "\x00")):
        raise ValueError("api_key must not contain newline or NUL characters")
    return raw.strip()


def _validate_stage_models(raw: dict[str, str]) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError("stage_models keys and values must be strings")
        canon = normalize_stage_name(k)
        if canon not in VLM_STAGE_KEYS:
            raise ValueError(
                f"Unknown VLM stage {k!r}; expected one of {', '.join(VLM_STAGE_KEYS)}"
            )
        model_s = v.strip()
        if not model_s:
            raise ValueError(f"Model name for stage {canon!r} must be non-empty")
        cleaned[canon] = model_s
    return cleaned


def upsert_workspace_env_key(workspace: Path, env_key: str, api_key: str) -> Path:
    """Atomically upsert `ENV_KEY=VALUE` into `<workspace>/.env` with 0o600 permissions."""
    valid_key = _validate_env_key_name(env_key)
    valid_val = _validate_api_key_value(api_key)
    if not valid_val:
        raise ValueError("api_key must be non-empty when writing to .env")

    ws_root = Path(workspace).resolve()
    ws_root.mkdir(parents=True, exist_ok=True)
    env_path = ws_root / ".env"

    existing_lines: list[str] = []
    if env_path.is_file():
        existing_lines = env_path.read_text(encoding="utf-8").splitlines()

    if any(ch in valid_val for ch in (" ", "\t", "#", "'", '"')):
        escaped = valid_val.replace("\\", "\\\\").replace('"', '\\"')
        formatted = f'{valid_key}="{escaped}"'
    else:
        formatted = f"{valid_key}={valid_val}"

    out_lines: list[str] = []
    replaced = False
    for ln in existing_lines:
        stripped = ln.strip()
        check = (
            stripped[len("export ") :].strip()
            if stripped.startswith("export ")
            else stripped
        )
        if "=" in check and not stripped.startswith("#"):
            k = check.split("=", 1)[0].strip()
            if k == valid_key:
                if not replaced:
                    out_lines.append(formatted)
                    replaced = True
                continue
        out_lines.append(ln)
    if not replaced:
        out_lines.append(formatted)

    content = "\n".join(out_lines) + "\n"
    tmp_path = ws_root / f".env.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise
    os.replace(tmp_path, env_path)
    os.chmod(env_path, 0o600)
    return env_path


def resolve_workspace_api_key(
    workspace: Path,
    api_key_env: str,
    key_file: str | Path | None = ".env",
) -> str | None:
    """Resolve API key preferring `<workspace>/.env`, then `<workspace>/<key_file>`, then standard resolve_api_key."""
    ws_root = Path(workspace).resolve()
    env_var = api_key_env or "OPENAI_API_KEY"
    ws_dotenv = ws_root / ".env"
    if ws_dotenv.is_file():
        found = _parse_env_file_for_key(ws_dotenv, env_var)
        if found:
            return found
    if key_file:
        kf = Path(key_file)
        ws_env = kf if kf.is_absolute() else (ws_root / kf)
        if ws_env != ws_dotenv and ws_env.is_file():
            found = _parse_env_file_for_key(ws_env, env_var)
            if found:
                return found
    try:
        return resolve_api_key(
            api_key_env=env_var,
            key_file=key_file,
            workspace_dir=workspace,
        )
    except VlmError:
        return None


def load_vlm_presets(config_dir: Path) -> list[dict[str, Any]]:
    """Load built-in VLM presets from `<config_dir>/presets/*.yaml` plus a `custom` entry."""
    presets_dir = Path(config_dir) / "presets"
    ordered_ids = ["openai", "dashscope", "gemini", "vllm"]
    extra_ids: list[str] = []
    if presets_dir.is_dir():
        for p in sorted(presets_dir.glob("*.yaml")):
            if p.stem not in ordered_ids and p.stem != "custom":
                extra_ids.append(p.stem)

    items: list[dict[str, Any]] = []
    for pid in [*ordered_ids, *extra_ids]:
        meta = _PRESET_META.get(pid, {})
        vlm_data: dict[str, Any] = dict(meta.get("fallback") or {})
        yaml_path = presets_dir / f"{pid}.yaml"
        if yaml_path.is_file():
            try:
                loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and isinstance(loaded.get("vlm"), dict):
                    vlm_data = deep_merge(vlm_data, loaded["vlm"])
            except Exception:
                pass
        raw_sm = vlm_data.get("stage_models") or {}
        fallback_model = (
            str(vlm_data.get("default_model") or "")
            or str(raw_sm.get("s3_discover") or "")
            or next((str(v) for v in raw_sm.values() if v), "")
            or "gpt-4o"
        )
        stage_models = {
            st: str(raw_sm.get(st) or fallback_model) for st in VLM_STAGE_KEYS
        }
        provider = str(vlm_data.get("provider") or "openai_compat").strip().lower()
        if provider not in ("openai_compat", "gemini"):
            provider = "openai_compat"
        thinking_mode = str(vlm_data.get("thinking_mode") or "auto").strip().lower()
        if thinking_mode not in ("auto", "dashscope", "disabled"):
            thinking_mode = "auto"
        label = str(meta.get("label") or pid.upper())
        items.append(
            {
                "id": pid,
                "label": label,
                "name": label,
                "description": str(meta.get("description") or f"{label} preset"),
                "provider": provider,
                "base_url": str(vlm_data.get("base_url") or "").strip(),
                "api_key_env": str(
                    vlm_data.get("api_key_env") or "OPENAI_API_KEY"
                ).strip(),
                "thinking_mode": thinking_mode,
                "default_model": fallback_model,
                "stage_models": stage_models,
            }
        )

    items.append(
        {
            "id": "custom",
            "label": "Custom",
            "name": "Custom",
            "description": "Custom OpenAI-compatible relay, proxy, or self-hosted endpoint",
            "provider": "openai_compat",
            "base_url": "",
            "api_key_env": "OPENAI_API_KEY",
            "thinking_mode": "auto",
            "default_model": "",
            "stage_models": {st: "" for st in VLM_STAGE_KEYS},
        }
    )
    return items


def read_vlm_settings_dict(
    cfg: ConfigManager,
    *,
    workspace: Path,
    config_dir: Path,
) -> dict[str, Any]:
    """Read current VLM settings with API key masked (NEVER plaintext)."""
    vlm_cfg = cfg.settings.get("vlm", {})
    provider = str(vlm_cfg.get("provider", "openai_compat")).strip().lower()
    if provider not in ("openai_compat", "gemini"):
        provider = "openai_compat"
    base_url = str(vlm_cfg.get("base_url", "https://api.openai.com/v1")).strip()
    api_key_env = str(vlm_cfg.get("api_key_env", "OPENAI_API_KEY")).strip() or "OPENAI_API_KEY"
    key_file = vlm_cfg.get("key_file", ".env")
    thinking_mode = str(vlm_cfg.get("thinking_mode", "auto")).strip().lower()
    if thinking_mode not in ("auto", "dashscope", "disabled"):
        thinking_mode = "auto"

    raw_sm = vlm_cfg.get("stage_models") or {}
    fallback_m = (
        str(vlm_cfg.get("default_model") or "")
        or str(raw_sm.get("s3_discover") or "")
        or "gpt-4o"
    )
    stage_models = {
        st: str(raw_sm.get(st) or fallback_m) for st in VLM_STAGE_KEYS
    }

    resolved_key = resolve_workspace_api_key(workspace, api_key_env, key_file)
    return {
        "provider": provider,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "api_key_configured": bool(resolved_key),
        "api_key_masked": mask_api_key(resolved_key),
        "thinking_mode": thinking_mode,
        "stage_models": stage_models,
        "presets": load_vlm_presets(config_dir),
    }


def apply_vlm_settings_update(
    *,
    workspace: Path,
    config_dir: Path,
    provider: VlmProvider | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    api_key: str | None = None,
    thinking_mode: VlmThinkingMode | None = None,
    default_model: str | None = None,
    stage_models: dict[str, str] | None = None,
    reason: str = "vlm_settings_update",
) -> ConfigManager:
    """Safely persist non-secret VLM config to configs_override/default.yaml and secret key to .env (0600)."""
    ws = Path(workspace).resolve()
    cfg_dir = Path(config_dir).resolve()
    ws.mkdir(parents=True, exist_ok=True)

    clean_base_url = _validate_http_url(base_url) if base_url is not None else None
    clean_env = _validate_env_key_name(api_key_env) if api_key_env is not None else None
    clean_key = (
        _validate_api_key_value(api_key)
        if (api_key is not None and api_key.strip())
        else None
    )
    clean_default_model = (
        default_model.strip()
        if (default_model is not None and default_model.strip())
        else None
    )
    clean_stage_models = (
        _validate_stage_models(stage_models) if stage_models is not None else None
    )

    with workspace_config_lock(ws):
        ov_path = ws / "configs_override" / "default.yaml"
        existing: dict[str, Any] = {}
        if ov_path.is_file():
            loaded = yaml.safe_load(ov_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
            elif loaded is not None:
                raise RuntimeError(
                    "Existing configs_override/default.yaml is not a YAML mapping"
                )

        vlm_patch: dict[str, Any] = {}
        if provider is not None:
            vlm_patch["provider"] = provider
        if clean_base_url is not None:
            vlm_patch["base_url"] = clean_base_url
        if clean_env is not None:
            vlm_patch["api_key_env"] = clean_env
        if thinking_mode is not None:
            vlm_patch["thinking_mode"] = thinking_mode

        sm_patch: dict[str, str] = {}
        if clean_default_model is not None:
            for st in VLM_STAGE_KEYS:
                sm_patch[st] = clean_default_model
        if clean_stage_models:
            sm_patch.update(clean_stage_models)
        if sm_patch:
            vlm_patch["stage_models"] = sm_patch
        if clean_key is not None:
            # Ensure VlmClient resolves the saved secret from <workspace>/.env
            vlm_patch["key_file"] = ".env"

        if vlm_patch:
            merged = deep_merge(existing, {"vlm": vlm_patch})
            # Guarantee plaintext api_key is NEVER written into YAML override
            if isinstance(merged.get("vlm"), dict):
                merged["vlm"].pop("api_key", None)
            yaml_text = yaml.safe_dump(merged, sort_keys=False, allow_unicode=True)
            archive_and_write_override(
                ws,
                cfg_dir,
                Path("default.yaml"),
                yaml_text,
                reason=reason,
            )

        cfg_after = ConfigManager(config_dir=cfg_dir, workspace_dir=ws)
        if clean_key is not None:
            eff_env = clean_env or str(
                cfg_after.settings.get("vlm", {}).get("api_key_env") or "OPENAI_API_KEY"
            ).strip()
            upsert_workspace_env_key(ws, eff_env, clean_key)

        return cfg_after


def probe_vlm_connection(
    *,
    cfg: ConfigManager,
    workspace: Path,
    provider: VlmProvider | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    thinking_mode: VlmThinkingMode | None = None,
) -> dict[str, Any]:
    """Run a fast minimal connectivity probe against a candidate or saved VLM endpoint."""
    vlm_cfg = cfg.settings.get("vlm", {})
    eff_provider = str(
        provider or vlm_cfg.get("provider") or "openai_compat"
    ).strip().lower()
    eff_base_url = (
        _validate_http_url(base_url)
        if (base_url is not None and base_url.strip())
        else str(vlm_cfg.get("base_url") or "https://api.openai.com/v1").strip().rstrip("/")
    )
    eff_env = (
        _validate_env_key_name(api_key_env)
        if (api_key_env is not None and api_key_env.strip())
        else str(vlm_cfg.get("api_key_env") or "OPENAI_API_KEY").strip()
    )
    eff_thinking = str(
        thinking_mode or vlm_cfg.get("thinking_mode") or "auto"
    ).strip().lower()
    raw_sm = vlm_cfg.get("stage_models") or {}
    eff_model = (
        model.strip()
        if (model is not None and model.strip())
        else str(
            raw_sm.get("s3_discover")
            or raw_sm.get("s1_grounding")
            or vlm_cfg.get("default_model")
            or "gpt-4o"
        ).strip()
    )

    explicit_key = (
        _validate_api_key_value(api_key)
        if (api_key is not None and api_key.strip())
        else None
    )
    eff_key = explicit_key or resolve_workspace_api_key(
        workspace, eff_env, vlm_cfg.get("key_file", ".env")
    )

    t0 = time.perf_counter()
    if not eff_key:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "ok": False,
            "latency_ms": round(elapsed_ms, 2),
            "model": eff_model,
            "detail": f"API key not configured for env var {eff_env!r}",
        }

    try:
        with VlmClient(
            provider=eff_provider,
            base_url=eff_base_url,
            api_key=eff_key,
            api_key_env=eff_env,
            thinking_mode=eff_thinking,
            timeout_s=15.0,
            retries=0,
            workspace_dir=workspace,
        ) as client:
            resp_obj = client.chat_json(
                stage="s3_discover",
                model=eff_model,
                system_prompt='Respond ONLY with valid JSON: {"ok": true}',
                user_prompt='Connectivity probe. Reply with {"ok": true}.',
                thinking=False,
                max_tokens=16,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "ok": True,
            "latency_ms": round(elapsed_ms, 2),
            "model": eff_model,
            "detail": _redact_secret(
                f"Connected successfully ({len(resp_obj)} JSON key(s) returned)",
                eff_key,
            ),
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "ok": False,
            "latency_ms": round(elapsed_ms, 2),
            "model": eff_model,
            "detail": _redact_secret(str(exc), eff_key),
        }


class VlmSettingsUpdatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: VlmProvider | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    api_key: str | None = None
    thinking_mode: VlmThinkingMode | None = None
    default_model: str | None = None
    stage_models: dict[str, str] | None = None

    @field_validator("base_url")
    @classmethod
    def _val_base_url(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _validate_http_url(v)

    @field_validator("api_key_env")
    @classmethod
    def _val_env(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _validate_env_key_name(v)

    @field_validator("api_key")
    @classmethod
    def _val_api_key(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _validate_api_key_value(v)

    @field_validator("default_model")
    @classmethod
    def _val_default_model(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = v.strip()
        return s or None

    @field_validator("stage_models")
    @classmethod
    def _val_stage_models(cls, v: dict[str, str] | None) -> dict[str, str] | None:
        if v is None:
            return None
        if not v:
            return None
        return _validate_stage_models(v)

    @model_validator(mode="after")
    def _require_any(self) -> "VlmSettingsUpdatePayload":
        has_any = any(
            x is not None
            for x in (
                self.provider,
                self.base_url,
                self.api_key_env,
                self.thinking_mode,
                self.default_model,
                self.stage_models,
            )
        ) or bool(self.api_key)
        if not has_any:
            raise ValueError("At least one VLM setting field is required")
        return self


class VlmTestPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: VlmProvider | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    api_key: str | None = None
    model: str | None = None
    thinking_mode: VlmThinkingMode | None = None

    @field_validator("base_url")
    @classmethod
    def _val_base_url(cls, v: str | None) -> str | None:
        if v is None or not v.strip():
            return None
        return _validate_http_url(v)

    @field_validator("api_key_env")
    @classmethod
    def _val_env(cls, v: str | None) -> str | None:
        if v is None or not v.strip():
            return None
        return _validate_env_key_name(v)

    @field_validator("api_key")
    @classmethod
    def _val_api_key(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return _validate_api_key_value(v)

    @field_validator("model")
    @classmethod
    def _val_model(cls, v: str | None) -> str | None:
        if v is None or not v.strip():
            return None
        return v.strip()


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


@router.get("/vlm")
def get_vlm_settings(
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    return read_vlm_settings_dict(
        state.get_config(),
        workspace=state.workspace,
        config_dir=state.config_dir,
    )


@router.put("/vlm")
def put_vlm_settings(
    payload: VlmSettingsUpdatePayload,
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    try:
        apply_vlm_settings_update(
            workspace=state.workspace,
            config_dir=state.config_dir,
            provider=payload.provider,
            base_url=payload.base_url,
            api_key_env=payload.api_key_env,
            api_key=payload.api_key,
            thinking_mode=payload.thinking_mode,
            default_model=payload.default_model,
            stage_models=payload.stage_models,
            reason="vlm_settings_update",
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    state.invalidate_config()
    res = read_vlm_settings_dict(
        state.get_config(),
        workspace=state.workspace,
        config_dir=state.config_dir,
    )
    state.sse.publish(
        {
            "type": "vlm_settings_updated",
            "provider": res["provider"],
            "base_url": res["base_url"],
            "api_key_env": res["api_key_env"],
            "api_key_configured": res["api_key_configured"],
            "thinking_mode": res["thinking_mode"],
            "stage_models": res["stage_models"],
        }
    )
    return res


@router.post("/vlm/test")
async def test_vlm_settings(
    payload: VlmTestPayload | None = Body(default=None),
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    req = payload or VlmTestPayload()
    cfg = state.get_config()
    try:
        return await run_in_threadpool(
            probe_vlm_connection,
            cfg=cfg,
            workspace=state.workspace,
            provider=req.provider,
            base_url=req.base_url,
            api_key_env=req.api_key_env,
            api_key=req.api_key,
            model=req.model,
            thinking_mode=req.thinking_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
