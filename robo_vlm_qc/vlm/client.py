"""Multi-provider httpx VLM client (OpenAI-compatible & Gemini native /v1beta) with resilience and budget."""

from __future__ import annotations

import contextlib
import ipaddress
import json
import logging
import os
import random
import re
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote, quote_plus, urlsplit

import httpx
import numpy as np

from ..config import ConfigManager
from ..io.media import encode_jpeg_data_url
from ..models.annotation import StageUsage, UsageSummary

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OPENAI_COMPAT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_COMPAT_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_OPENAI_COMPAT_KEY_FILE = ".env"
DEFAULT_OPENAI_COMPAT_MODEL = "gpt-4o"

DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
DEFAULT_GEMINI_KEY_FILE = ".env"
DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"
DEFAULT_GEMINI_MEDIA_RESOLUTION = "MEDIA_RESOLUTION_HIGH"
DEFAULT_VISION_MIN_TOKENS_PER_IMAGE = 550
DEFAULT_VISION_BLIND_NODE_RETRIES = 5

_SUPPORTED_PROVIDERS = frozenset({"gemini", "openai_compat"})
_SK_TOKEN_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")
_AIZA_TOKEN_RE = re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b")
_GEMINI_LEVEL_MODEL_RE = re.compile(r"^gemini-(?:3|4)(?:\b|[.-])", re.IGNORECASE)
_GEMINI_MEDIA_RESOLUTION_ALIASES: dict[str, str] = {
    "low": "MEDIA_RESOLUTION_LOW",
    "media_resolution_low": "MEDIA_RESOLUTION_LOW",
    "medium": "MEDIA_RESOLUTION_MEDIUM",
    "media_resolution_medium": "MEDIA_RESOLUTION_MEDIUM",
    "high": "MEDIA_RESOLUTION_HIGH",
    "media_resolution_high": "MEDIA_RESOLUTION_HIGH",
    "unspecified": "MEDIA_RESOLUTION_UNSPECIFIED",
    "media_resolution_unspecified": "MEDIA_RESOLUTION_UNSPECIFIED",
}
_DISABLED_SENTINELS = frozenset({"", "none", "off", "false", "disabled", "0"})
_GEMINI_BLOCK_FINISH_REASONS = frozenset(
    {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"}
)
_GEMINI_LEVEL_HEADROOM: dict[str, int] = {
    "minimal": 512,
    "low": 1024,
    "medium": 2048,
    "high": 4096,
    "xhigh": 8192,
}
_KEY_QUERY_RE = re.compile(
    r"([?&](?:key|api_key|apikey)=)([^&\s'\"`<>)]+)",
    re.IGNORECASE,
)
_BEARER_RE = re.compile(
    r"(Bearer\s+)([^\s'\"`<>)]+)",
    re.IGNORECASE,
)
_GOOG_KEY_HDR_RE = re.compile(
    r"(x-goog-api-key['\"]?\s*[:=]\s*['\"]?)([^\s'\"`,}]+)",
    re.IGNORECASE,
)
_GOOG_KEY_TUPLE_RE = re.compile(
    r"((?:b?['\"]x-goog-api-key['\"]\s*,\s*b?['\"]))([^'\"]+)",
    re.IGNORECASE,
)
_AUTH_TUPLE_RE = re.compile(
    r"((?:b?['\"]authorization['\"]\s*,\s*b?['\"])(?:Bearer\s+)?)([^'\"]+)",
    re.IGNORECASE,
)
_HTTP_LOGGERS_TO_REDACT: tuple[str, ...] = (
    "httpx",
    "httpcore",
    "httpcore.connection",
    "httpcore.http11",
    "httpcore.http2",
    "httpcore.proxy",
)

_STAGE_CONFIG_ALIASES: dict[str, str] = {
    "s1": "s1_grounding",
    "s1_grounding": "s1_grounding",
    "s2": "s2_macro",
    "s2_macro": "s2_macro",
    "s3": "s3_discover",
    "s3_discover": "s3_discover",
    "s4": "s4_verify",
    "s4a": "s4_verify",
    "s4a_merge": "s4_verify",
    "s4_merge": "s4_verify",
    "s4b": "s4_verify",
    "s4b_verify": "s4_verify",
    "s4_verify": "s4_verify",
    "s5": "s5_success",
    "s5_success": "s5_success",
    "s5_control": "s5_success",
}


class VlmError(RuntimeError):
    """VLM API call, auth, or response parsing failure."""


class BudgetExceeded(VlmError):
    """Episode or global token/call budget exhausted."""


class VlmCancelledError(VlmError):
    """Cooperative cancellation signal raised when a VLM job is cancelled mid-flight."""


class VlmBlindVisionError(VlmError):
    """Raised when an upstream Gemini vision endpoint repeatedly returns degraded/blind token density."""


class VlmCircuitOpenError(VlmError):
    """Raised when the VLM circuit breaker is open due to quota exhaustion or repeated failures."""

    def __init__(
        self,
        message: str,
        *,
        reason: str | None = None,
        snapshot: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason or message
        self.snapshot = snapshot


_QUOTA_ERROR_MARKERS: tuple[str, ...] = (
    "pre_consume_token_quota_failed",
    "insufficient_quota",
    "token_quota",
    "quota_exceeded",
    "quota",
)


class VlmCircuitBreaker:
    """Thread-safe circuit breaker tripping immediately on quota/403 or after N consecutive failures."""

    def __init__(
        self,
        failure_threshold: int = 3,
        trip_on_quota_403: bool = True,
        enabled: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.failure_threshold = max(1, int(failure_threshold))
        self.trip_on_quota_403 = bool(trip_on_quota_403)
        self._lock = threading.Lock()
        self._is_open = False
        self._consecutive_failures = 0
        self._total_failures = 0
        self._total_successes = 0
        self._opened_at: float | None = None
        self._reason: str | None = None

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._is_open

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    @staticmethod
    def is_quota_error(exc: BaseException) -> bool:
        sc = getattr(exc, "_status_code", None)
        if sc is not None and int(sc) == 403:
            return True
        body = str(getattr(exc, "_body_text", "") or "")
        combined = f"{exc} {body}".lower()
        if "http 403" in combined:
            return True
        return any(marker in combined for marker in _QUOTA_ERROR_MARKERS)

    def check(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if not self._is_open:
                return
            reason = self._reason or "VLM circuit breaker is open"
            snap = self._snapshot_locked()
        raise VlmCircuitOpenError(
            f"VLM circuit breaker open: {reason}",
            reason=reason,
            snapshot=snap,
        )

    def record_success(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._consecutive_failures = 0
            self._total_successes += 1

    def record_failure(self, exc: BaseException) -> bool:
        """Record a provider failure; returns True if the circuit is now open."""
        if not self.enabled:
            return False
        if isinstance(exc, (VlmCancelledError, VlmCircuitOpenError, BudgetExceeded)):
            return self.is_open
        with self._lock:
            self._consecutive_failures += 1
            self._total_failures += 1
            if self._is_open:
                return True
            if self.trip_on_quota_403 and self.is_quota_error(exc):
                self._is_open = True
                self._opened_at = time.time()
                self._reason = f"Quota/403 failure: {str(exc)[:240]}"
                return True
            if self._consecutive_failures >= self.failure_threshold:
                self._is_open = True
                self._opened_at = time.time()
                self._reason = (
                    f"Consecutive failures reached {self._consecutive_failures}/"
                    f"{self.failure_threshold}: {str(exc)[:240]}"
                )
                return True
            return False

    def _snapshot_locked(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "is_open": self._is_open,
            "failure_threshold": self.failure_threshold,
            "trip_on_quota_403": self.trip_on_quota_403,
            "consecutive_failures": self._consecutive_failures,
            "total_failures": self._total_failures,
            "total_successes": self._total_successes,
            "opened_at": self._opened_at,
            "reason": self._reason,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()


def _is_loopback_url(url: str) -> bool:
    """Return True if *url* targets a local loopback address (127.0.0.1, localhost, ::1, etc.)."""
    try:
        host = (urlsplit(str(url or "")).hostname or "").strip().lower()
    except ValueError:
        return False
    if not host:
        return False
    if host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _strip_gemini_model_prefix(model: str) -> str:
    clean = str(model or "").strip()
    if clean.startswith("models/"):
        clean = clean[len("models/") :]
    return clean


def is_gemini_hd_vision_model(model: str) -> bool:
    """Return True for Gemini 3+ family models that encode HD images at ~1075-1350 tok/img."""
    return bool(_GEMINI_LEVEL_MODEL_RE.match(_strip_gemini_model_prefix(model)))


def normalize_gemini_media_resolution(value: Any) -> str | None:
    """Normalize a Gemini generationConfig.mediaResolution setting; return None if disabled."""
    if value is None:
        return None
    if isinstance(value, bool):
        return DEFAULT_GEMINI_MEDIA_RESOLUTION if value else None
    raw = str(value).strip()
    low = raw.lower()
    if low in _DISABLED_SENTINELS:
        return None
    if low in _GEMINI_MEDIA_RESOLUTION_ALIASES:
        return _GEMINI_MEDIA_RESOLUTION_ALIASES[low]
    up = raw.upper()
    if up.startswith("MEDIA_RESOLUTION_"):
        return up
    return up


def _estimate_text_tokens(text: str) -> int:
    """Conservative text-token estimator (~0.80/CJK char, ~0.25/other char).

    Intentionally slightly conservative so subtracting it from ``prompt_tokens`` never
    triggers false positives on genuine HD vision responses (~1075-1350 tok/img), while
    reliably detecting degraded ~258 tok/img responses even on long prompts with few images.
    """
    if not text:
        return 0
    cjk = 0
    for ch in text:
        o = ord(ch)
        if (
            0x3000 <= o <= 0x303F
            or 0x3400 <= o <= 0x4DBF
            or 0x4E00 <= o <= 0x9FFF
            or 0xF900 <= o <= 0xFAFF
            or 0xFF00 <= o <= 0xFFEF
        ):
            cjk += 1
    other = len(text) - cjk
    return int(cjk * 0.80 + other * 0.25)


def count_gemini_payload_media(
    sys_instruction: dict[str, Any] | None,
    contents: Sequence[dict[str, Any]],
) -> tuple[int, int]:
    """Return ``(n_images, est_text_tokens)`` for a prepared Gemini payload."""
    n_images = 0
    est_text = 0
    if isinstance(sys_instruction, dict):
        for p in sys_instruction.get("parts") or ():
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                est_text += _estimate_text_tokens(p["text"])
    for item in contents:
        if not isinstance(item, dict):
            continue
        for p in item.get("parts") or ():
            if not isinstance(p, dict):
                continue
            if "inlineData" in p and isinstance(p["inlineData"], dict):
                mime = str(p["inlineData"].get("mimeType") or "image/jpeg").lower()
                if mime.startswith("image/"):
                    n_images += 1
            elif isinstance(p.get("text"), str):
                est_text += _estimate_text_tokens(p["text"])
    return n_images, est_text


def evaluate_vision_token_density(
    *,
    n_images: int,
    prompt_tokens: int,
    exact_image_tokens: int | None,
    est_text_tokens: int,
    min_tokens_per_image: int,
) -> tuple[bool, float | None, int, str]:
    """Check whether a Gemini multimodal response indicates a blind/degraded vision node.

    Returns ``(is_blind, tok_per_image, effective_image_tokens, density_source)``.
    """
    if n_images <= 0 or min_tokens_per_image <= 0 or prompt_tokens <= 0:
        return False, None, 0, "skipped"
    if exact_image_tokens is not None and exact_image_tokens > 0:
        eff_img = int(exact_image_tokens)
        src = "promptTokensDetails.IMAGE"
    else:
        eff_img = max(0, int(prompt_tokens) - max(0, int(est_text_tokens)))
        src = "prompt_tokens-est_text"
    tok_per_img = eff_img / float(n_images)
    is_blind = eff_img < (n_images * int(min_tokens_per_image))
    return is_blind, tok_per_img, eff_img, src


def check_client_cancelled(client: Any) -> None:
    """Raise VlmCancelledError if a VlmClient (or test double) signals cancellation."""
    is_c = getattr(client, "is_cancelled", None)
    if callable(is_c) and is_c():
        raise VlmCancelledError("VLM request cancelled")


def normalize_stage_name(stage: str) -> str:
    s = stage.strip().lower()
    return _STAGE_CONFIG_ALIASES.get(s, s)


def estimate_stage_cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    pricing: dict[str, Any] | None = None,
    *,
    reasoning_tokens: int = 0,
    reasoning_included_in_completion: bool = False,
) -> float:
    """Estimate USD cost for a call using vlm.pricing config and actual model name.

    When ``reasoning_included_in_completion=False`` (default, matching Gemini native
    ``candidatesTokenCount`` + ``thoughtsTokenCount``), ``reasoning_tokens`` are added to
    billed output tokens. When ``True`` (matching OpenAI-compat where
    ``completion_tokens`` already includes ``completion_tokens_details.reasoning_tokens``),
    output tokens are never double-counted.
    """
    if not pricing:
        return 0.0
    clean = _strip_gemini_model_prefix(model)
    entry = pricing.get(model) or pricing.get(clean) or pricing.get("default")
    if not isinstance(entry, dict):
        return 0.0
    try:
        in_rate = float(entry.get("input_per_1m_usd", 0.0) or 0.0)
        out_rate = float(entry.get("output_per_1m_usd", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    p_tok = max(0, int(prompt_tokens))
    c_tok = max(0, int(completion_tokens))
    r_tok = max(0, int(reasoning_tokens))
    billed_out = max(c_tok, r_tok) if reasoning_included_in_completion else (c_tok + r_tok)
    cost = (p_tok * in_rate + billed_out * out_rate) / 1_000_000.0
    return round(max(0.0, cost), 8)


def resolve_client_stage_model(
    client: Any,
    stage: str,
    model_override: str | None = None,
) -> str:
    """Safely resolve a stage's effective model on VlmClient, ConfigManager, or test doubles."""
    fn = getattr(client, "resolve_stage_model", None)
    if callable(fn):
        return str(fn(stage, model_override=model_override))
    fn2 = getattr(client, "model_for_stage", None)
    if callable(fn2):
        try:
            return str(fn2(stage, override=model_override))
        except TypeError:
            return str(model_override or fn2(stage))
    if model_override:
        return str(model_override)
    fn3 = getattr(client, "stage_model", None)
    if callable(fn3):
        return str(fn3(normalize_stage_name(stage)))
    return ""


resolve_stage_model = resolve_client_stage_model


def parse_json(text: str) -> dict[str, Any]:
    """Parse JSON object from model output, stripping markdown fences or surrounding prose."""
    if not isinstance(text, str):
        raise ValueError(f"Expected string response, got {type(text).__name__}")
    raw = text.strip()
    if not raw:
        raise ValueError("Empty response text")

    # 1. Direct parse
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2. Extract markdown code fence blocks anywhere in text before generic brace scan
    for match in re.finditer(r"```(?:[a-zA-Z0-9_-]+)?\s*(.*?)```", raw, re.DOTALL):
        fenced = match.group(1).strip()
        if not fenced:
            continue
        try:
            parsed = json.loads(fenced)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        fenced = "\n".join(lines).strip()
        try:
            parsed = json.loads(fenced)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # 3. Scan '{' positions with JSONDecoder.raw_decode
    decoder = json.JSONDecoder()
    idx = raw.find("{")
    while idx != -1:
        try:
            parsed, _ = decoder.raw_decode(raw, idx)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        idx = raw.find("{", idx + 1)

    raise ValueError(f"Response does not contain a valid JSON object: {raw[:200]!r}")


extract_json = parse_json


def redact_secrets(text: str, secret: str | None = None) -> str:
    """Redact API keys, query ?key= values, and auth headers from strings/logs/errors."""
    if not text:
        return text
    out = _KEY_QUERY_RE.sub(r"\1[REDACTED]", text)
    out = _BEARER_RE.sub(r"\1[REDACTED]", out)
    out = _GOOG_KEY_HDR_RE.sub(r"\1[REDACTED]", out)
    out = _GOOG_KEY_TUPLE_RE.sub(r"\1[REDACTED]", out)
    out = _AUTH_TUPLE_RE.sub(r"\1[REDACTED]", out)
    out = _SK_TOKEN_RE.sub("[REDACTED]", out)
    out = _AIZA_TOKEN_RE.sub("[REDACTED]", out)
    if secret:
        raw = str(secret).strip()
        if raw:
            out = out.replace(raw, "[REDACTED]")
            enc = quote(raw, safe="")
            if enc != raw:
                out = out.replace(enc, "[REDACTED]")
            enc_plus = quote_plus(raw)
            if enc_plus != raw and enc_plus != enc:
                out = out.replace(enc_plus, "[REDACTED]")
    return out


class _SecretRedactingLogFilter(logging.Filter):
    """Scrub API keys (?key=, x-goog-api-key, Bearer) from httpx/httpcore log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rendered = record.getMessage()
        except Exception:
            rendered = str(record.msg)
        redacted = redact_secrets(rendered)
        if redacted != rendered or record.args:
            record.msg = redacted
            record.args = None
        if isinstance(record.exc_text, str) and record.exc_text:
            record.exc_text = redact_secrets(record.exc_text)
        return True


_REDACT_LOG_FILTER = _SecretRedactingLogFilter()


def install_http_log_redaction() -> None:
    """Idempotently attach secret-redacting filter to httpx/httpcore loggers."""
    for logger_name in _HTTP_LOGGERS_TO_REDACT:
        lgr = logging.getLogger(logger_name)
        if not any(isinstance(f, _SecretRedactingLogFilter) for f in lgr.filters):
            lgr.addFilter(_REDACT_LOG_FILTER)


install_http_log_redaction()


def normalize_gemini_base_url(base_url: str) -> str:
    """Normalize a Gemini API base URL to end with /v1beta without duplicating it."""
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        return DEFAULT_GEMINI_BASE_URL
    if base.endswith("/v1beta"):
        return base
    if base.endswith("/v1"):
        return base[:-3] + "/v1beta"
    return f"{base}/v1beta"


def gemini_generate_url(base_url: str, model: str) -> str:
    """Build the Gemini native /v1beta/models/{model}:generateContent endpoint URL."""
    clean_model = str(model or "").strip()
    if clean_model.startswith("models/"):
        clean_model = clean_model[len("models/") :]
    if not clean_model:
        raise VlmError("Gemini model name must be non-empty")
    encoded_model = quote(clean_model, safe="-._")
    return f"{normalize_gemini_base_url(base_url)}/models/{encoded_model}:generateContent"


def _parse_data_uri(img_url: str) -> tuple[str, str]:
    if not (img_url.startswith("data:") and ";base64," in img_url):
        raise VlmError(
            "Unsupported image_url for Gemini native API: expected data:<mime>;base64,<data>"
        )
    hdr, b64 = img_url.split(";base64,", 1)
    mime = hdr[5:].split(";", 1)[0].strip() or "image/jpeg"
    return mime, b64


def messages_to_gemini_payload(
    messages: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Translate internal OpenAI-style messages into Gemini systemInstruction + contents."""
    sys_parts: list[dict[str, Any]] = []
    contents: list[dict[str, Any]] = []

    for msg in messages:
        role = str(msg.get("role", "user")).strip().lower()
        raw_c = msg.get("content", "")
        parts: list[dict[str, Any]] = []
        if isinstance(raw_c, str):
            if raw_c or role not in ("system", "developer"):
                parts.append({"text": raw_c})
        elif isinstance(raw_c, list):
            for blk in raw_c:
                if not isinstance(blk, dict):
                    continue
                btype = blk.get("type")
                if btype == "text":
                    parts.append({"text": str(blk.get("text", ""))})
                elif btype == "image_url":
                    img_obj = blk.get("image_url")
                    img_url = (
                        str(img_obj.get("url", ""))
                        if isinstance(img_obj, dict)
                        else str(img_obj or "")
                    )
                    mime, b64 = _parse_data_uri(img_url)
                    parts.append({"inlineData": {"mimeType": mime, "data": b64}})
                elif "inlineData" in blk and isinstance(blk["inlineData"], dict):
                    parts.append({"inlineData": dict(blk["inlineData"])})
                elif "inline_data" in blk and isinstance(blk["inline_data"], dict):
                    id_obj = blk["inline_data"]
                    parts.append(
                        {
                            "inlineData": {
                                "mimeType": str(
                                    id_obj.get("mimeType")
                                    or id_obj.get("mime_type")
                                    or "image/jpeg"
                                ),
                                "data": str(id_obj.get("data", "")),
                            }
                        }
                    )
                elif "text" in blk and btype is None:
                    parts.append({"text": str(blk["text"])})

        if role in ("system", "developer"):
            for p in parts:
                if "text" in p and str(p["text"]):
                    sys_parts.append({"text": str(p["text"])})
            continue

        if not parts:
            continue
        gem_role = "model" if role in ("assistant", "model") else "user"
        if contents and contents[-1]["role"] == gem_role:
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": gem_role, "parts": parts})

    if not contents:
        raise VlmError("Cannot invoke Gemini generateContent with empty contents")

    sys_instruction = {"parts": sys_parts} if sys_parts else None
    return sys_instruction, contents


def build_gemini_generation_config(
    *,
    model: str,
    thinking: bool,
    thinking_budget: int | None,
    thinking_level: str | None = None,
    max_tokens: int,
    temperature: float | None = None,
    top_p: float | None = None,
    json_mode: bool = True,
    media_resolution: str | None = None,
) -> dict[str, Any]:
    """Build Gemini native generationConfig with model-family-aware thinkingConfig."""
    clean_model = _strip_gemini_model_prefix(model)

    thinking_cfg: dict[str, Any] | None = None
    eff_level: str | None = (
        str(thinking_level).strip().lower()
        if thinking_level is not None and str(thinking_level).strip()
        else None
    )

    if eff_level is not None:
        thinking_cfg = {"thinkingLevel": eff_level}
    elif thinking:
        if _GEMINI_LEVEL_MODEL_RE.match(clean_model):
            if thinking_budget is None or int(thinking_budget) <= 0:
                eff_level = "medium"
            elif int(thinking_budget) <= 1024:
                eff_level = "low"
            elif int(thinking_budget) <= 2048:
                eff_level = "medium"
            else:
                eff_level = "high"
            thinking_cfg = {"thinkingLevel": eff_level}
        elif thinking_budget is not None and int(thinking_budget) > 0:
            thinking_cfg = {"thinkingBudget": int(thinking_budget)}

    headroom = 0
    if thinking_cfg is not None and "thinkingLevel" in thinking_cfg:
        level_headroom = _GEMINI_LEVEL_HEADROOM.get(
            str(thinking_cfg["thinkingLevel"]), 2048
        )
        if thinking and thinking_budget is not None and int(thinking_budget) > 0:
            headroom = max(int(thinking_budget), level_headroom)
        else:
            headroom = level_headroom
    elif thinking and thinking_budget is not None and int(thinking_budget) > 0:
        headroom = int(thinking_budget)

    gen_cfg: dict[str, Any] = {
        "maxOutputTokens": int(max_tokens) + headroom,
    }
    if temperature is not None:
        gen_cfg["temperature"] = float(temperature)
    if top_p is not None:
        gen_cfg["topP"] = float(top_p)
    if json_mode:
        gen_cfg["responseMimeType"] = "application/json"
    if thinking_cfg is not None:
        gen_cfg["thinkingConfig"] = thinking_cfg
    norm_mr = normalize_gemini_media_resolution(media_resolution)
    if norm_mr is not None:
        gen_cfg["mediaResolution"] = norm_mr
    return gen_cfg


def parse_gemini_response(
    data: dict[str, Any],
    *,
    secret: str | None = None,
) -> tuple[str, dict[str, int], dict[str, Any]]:
    """Parse Gemini generateContent response, separating thought parts from answer text."""
    cands = data.get("candidates") or []
    if not cands or not isinstance(cands[0], dict):
        pf = data.get("promptFeedback")
        preview = redact_secrets(str(data)[:300], secret)
        raise VlmError(
            f"Empty candidates in Gemini response (promptFeedback={pf!r}): {preview}"
        )

    cand0 = cands[0]
    finish_reason = cand0.get("finishReason")
    parts_out = (cand0.get("content") or {}).get("parts") or []

    thought_parts: list[str] = []
    answer_parts: list[str] = []
    for p in parts_out:
        if not isinstance(p, dict):
            continue
        txt = p.get("text")
        if txt is None:
            continue
        if p.get("thought") is True:
            thought_parts.append(str(txt))
        else:
            answer_parts.append(str(txt))

    answer_text = "".join(answer_parts)
    thought_text = "".join(thought_parts)

    um = data.get("usageMetadata") or {}
    p_tok = int(um.get("promptTokenCount") or 0)
    c_tok = int(um.get("candidatesTokenCount") or 0)
    r_tok = int(um.get("thoughtsTokenCount") or 0)
    t_tok = int(um.get("totalTokenCount") or (p_tok + c_tok + r_tok))
    usage = {
        "prompt_tokens": p_tok,
        "completion_tokens": c_tok,
        "total_tokens": t_tok,
        "reasoning_tokens": r_tok,
    }

    exact_img_tokens: int | None = None
    exact_txt_tokens: int | None = None
    raw_details = um.get("promptTokensDetails")
    if isinstance(raw_details, list):
        img_sum = 0
        txt_sum = 0
        has_img = False
        has_txt = False
        for entry in raw_details:
            if not isinstance(entry, dict):
                continue
            mod = str(entry.get("modality") or "").strip().upper()
            cnt = int(entry.get("tokenCount") or 0)
            if mod == "IMAGE":
                img_sum += cnt
                has_img = True
            elif mod == "TEXT":
                txt_sum += cnt
                has_txt = True
        if has_img:
            exact_img_tokens = img_sum
        if has_txt:
            exact_txt_tokens = txt_sum

    if (
        not answer_text.strip()
        and str(finish_reason or "").upper() in _GEMINI_BLOCK_FINISH_REASONS
    ):
        raise VlmError(f"Gemini response blocked (finishReason={finish_reason!r})")

    meta = {
        "finish_reason": finish_reason,
        "thought_text": thought_text,
        "model_version": data.get("modelVersion"),
        "image_prompt_tokens": exact_img_tokens,
        "text_prompt_tokens": exact_txt_tokens,
        "reasoning_included_in_completion": bool(
            r_tok > 0 and t_tok == (p_tok + c_tok) and c_tok >= r_tok
        ),
    }
    return answer_text, usage, meta


def _parse_env_file_for_key(path: Path, key_name: str) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    bare_tokens: list[str] = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("export "):
            s = s[len("export ") :].strip()
        if "=" in s:
            k, v = s.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            if k == key_name and v:
                return v
        else:
            tok = s.strip("'\"")
            if tok:
                bare_tokens.append(tok)
    if len(bare_tokens) == 1:
        return bare_tokens[0]
    return None


def resolve_api_key(
    api_key_env: str = DEFAULT_OPENAI_COMPAT_API_KEY_ENV,
    key_file: str | Path | None = DEFAULT_OPENAI_COMPAT_KEY_FILE,
    workspace_dir: Path | None = None,
) -> str:
    """Resolve API key from env var or key_file without ever logging its value."""
    if api_key_env:
        env_val = os.environ.get(api_key_env, "").strip()
        if env_val:
            return env_val

    if key_file:
        kf = Path(key_file)
        candidates: list[Path] = []
        if kf.is_absolute():
            candidates.append(kf)
        else:
            if workspace_dir is not None:
                candidates.append(Path(workspace_dir) / kf)
            candidates.append(Path.cwd() / kf)
            candidates.append(REPO_ROOT / kf)

        for cand in candidates:
            if cand.is_file():
                found = _parse_env_file_for_key(
                    cand, api_key_env or DEFAULT_OPENAI_COMPAT_API_KEY_ENV
                )
                if found:
                    return found

    raise VlmError(
        f"API key not found in env var {api_key_env!r} or key_file {str(key_file)!r}"
    )


def build_interleaved_messages(
    *,
    system_prompt: str,
    user_prompt: str,
    images: Sequence[np.ndarray] | None = None,
    timestamps_s: Sequence[float | None] | None = None,
    labels: Sequence[str] | None = None,
    jpeg_quality: int = 74,
    text_first: bool = False,
) -> list[dict[str, Any]]:
    """Build OpenAI-compatible interleaved timestamped image + text messages."""
    msgs: list[dict[str, Any]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})

    imgs = list(images or [])
    if not imgs:
        msgs.append({"role": "user", "content": user_prompt})
        return msgs

    blocks: list[dict[str, Any]] = []
    if text_first and user_prompt:
        blocks.append({"type": "text", "text": user_prompt})

    for idx, img in enumerate(imgs):
        raw_t = (
            timestamps_s[idx]
            if timestamps_s is not None and idx < len(timestamps_s)
            else float(idx)
        )
        lbl = (
            str(labels[idx]).strip()
            if labels is not None and idx < len(labels) and labels[idx]
            else ""
        )
        if raw_t is None:
            tag = f"[{lbl}]" if lbl else f"[Frame {idx + 1}]"
        else:
            t_val = float(raw_t)
            tag = f"[t={t_val:05.2f}s | {lbl}]" if lbl else f"[t={t_val:05.2f}s]"
        blocks.append({"type": "text", "text": tag})
        url = encode_jpeg_data_url(img, quality=jpeg_quality)
        blocks.append({"type": "image_url", "image_url": {"url": url}})

    if not text_first and user_prompt:
        blocks.append({"type": "text", "text": user_prompt})

    msgs.append({"role": "user", "content": blocks})
    return msgs


@dataclass
class _StageAccum:
    stage: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    elapsed_s: float = 0.0
    estimated_cost_usd: float = 0.0
    calls: int = 0
    cache_hits: int = 0


class BudgetTracker:
    """Thread-safe hierarchical token and call budget tracker."""

    def __init__(
        self,
        *,
        max_tokens: int | None = None,
        max_calls: int | None = None,
        parent: "BudgetTracker | None" = None,
        pricing: dict[str, Any] | None = None,
    ) -> None:
        self.max_tokens = int(max_tokens) if max_tokens is not None else None
        self.max_calls = int(max_calls) if max_calls is not None else None
        self.parent = parent
        self.pricing: dict[str, Any] | None = (
            dict(pricing)
            if pricing is not None
            else (dict(parent.pricing) if (parent is not None and parent.pricing is not None) else None)
        )
        self._lock = threading.Lock()
        self.tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.calls = 0
        self.cached_stages = 0
        self.elapsed_s = 0.0
        self.total_cost_usd = 0.0
        self._reserved = 0
        self._stages: dict[tuple[str, str], _StageAccum] = {}

    def check(self) -> None:
        with self._lock:
            if self.max_tokens is not None and self.tokens >= self.max_tokens:
                raise BudgetExceeded(
                    f"Token budget exceeded: {self.tokens} >= {self.max_tokens}"
                )
            if (
                self.max_calls is not None
                and (self.calls + self._reserved) >= self.max_calls
            ):
                raise BudgetExceeded(
                    f"Call budget exceeded: {self.calls + self._reserved} >= {self.max_calls}"
                )
        if self.parent is not None:
            self.parent.check()

    def reserve(self) -> None:
        with self._lock:
            if self.max_tokens is not None and self.tokens >= self.max_tokens:
                raise BudgetExceeded(
                    f"Token budget exceeded: {self.tokens} >= {self.max_tokens}"
                )
            if (
                self.max_calls is not None
                and (self.calls + self._reserved) >= self.max_calls
            ):
                raise BudgetExceeded(
                    f"Call budget exceeded: {self.calls + self._reserved} >= {self.max_calls}"
                )
            self._reserved += 1
        if self.parent is not None:
            try:
                self.parent.reserve()
            except BaseException:
                with self._lock:
                    self._reserved = max(0, self._reserved - 1)
                raise

    def release(self) -> None:
        with self._lock:
            self._reserved = max(0, self._reserved - 1)
        if self.parent is not None:
            self.parent.release()

    def record(
        self,
        *,
        stage: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int | None = None,
        reasoning_tokens: int = 0,
        reasoning_included_in_completion: bool | None = None,
        elapsed_s: float = 0.0,
        cache_hit: bool = False,
    ) -> None:
        p_tok = max(0, int(prompt_tokens))
        c_tok = max(0, int(completion_tokens))
        r_tok = max(0, int(reasoning_tokens))

        if reasoning_included_in_completion is None:
            if r_tok <= 0:
                eff_included = True
            elif total_tokens is not None:
                eff_included = (int(total_tokens) == (p_tok + c_tok)) and (c_tok >= r_tok)
            else:
                eff_included = False
        else:
            eff_included = bool(reasoning_included_in_completion)

        if total_tokens is not None:
            tot = max(0, int(total_tokens))
        else:
            tot = (p_tok + c_tok) if eff_included else (p_tok + c_tok + r_tok)

        call_cost = (
            0.0
            if cache_hit
            else estimate_stage_cost_usd(
                model,
                p_tok,
                c_tok,
                self.pricing,
                reasoning_tokens=r_tok,
                reasoning_included_in_completion=eff_included,
            )
        )
        stage_key = (stage, model)
        with self._lock:
            if not cache_hit:
                self.calls += 1
                self.prompt_tokens += p_tok
                self.completion_tokens += c_tok
                self.reasoning_tokens += r_tok
                self.tokens += tot
                self.elapsed_s += float(elapsed_s)
                self.total_cost_usd += call_cost
            else:
                self.cached_stages += 1

            acc = self._stages.get(stage_key)
            if acc is None:
                acc = _StageAccum(stage=stage, model=model)
                self._stages[stage_key] = acc
            acc.prompt_tokens += p_tok
            acc.completion_tokens += c_tok
            acc.total_tokens += tot
            acc.reasoning_tokens += r_tok
            acc.elapsed_s += float(elapsed_s)
            acc.estimated_cost_usd += call_cost
            if cache_hit:
                acc.cache_hits += 1
            else:
                acc.calls += 1

        if self.parent is not None:
            self.parent.record(
                stage=stage,
                model=model,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                total_tokens=tot,
                reasoning_tokens=r_tok,
                reasoning_included_in_completion=eff_included,
                elapsed_s=elapsed_s,
                cache_hit=cache_hit,
            )

    def to_usage_summary(self) -> UsageSummary:
        with self._lock:
            stages = [
                StageUsage(
                    stage=acc.stage,
                    model=acc.model,
                    prompt_tokens=acc.prompt_tokens,
                    completion_tokens=acc.completion_tokens,
                    total_tokens=acc.total_tokens,
                    elapsed_s=round(acc.elapsed_s, 4),
                    estimated_cost_usd=round(acc.estimated_cost_usd, 6),
                    cache_hit=(acc.calls == 0 and acc.cache_hits > 0),
                )
                for acc in self._stages.values()
            ]
            return UsageSummary(
                stages=stages,
                total_tokens=self.tokens,
                total_elapsed_s=round(self.elapsed_s, 4),
                total_cost_usd=round(self.total_cost_usd, 6),
            )


def _should_retry_http(status_code: int, body_text: str) -> bool:
    low = body_text.lower()
    if status_code == 503 and (
        '"model_not_found"' in low or "no available channel for model" in low
    ):
        return False
    if status_code in (408, 429) or status_code >= 500:
        return True
    if status_code == 400 and "timed out" in low:
        return True
    return False


class VlmClient:
    """Multi-provider httpx VLM client (openai_compat & gemini native /v1beta) with resilience and budget."""

    def __init__(
        self,
        config: ConfigManager | dict[str, Any] | None = None,
        *,
        cfg: ConfigManager | dict[str, Any] | None = None,
        config_manager: ConfigManager | None = None,
        settings: dict[str, Any] | None = None,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        key_file: str | Path | None = None,
        thinking_mode: str | None = None,
        timeout_s: float | None = None,
        retries: int | None = None,
        retry_min_wait_s: float | None = None,
        retry_max_wait_s: float | None = None,
        concurrency: int | None = None,
        stage_models: dict[str, str] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        thinking_level: str | None = None,
        gemini_media_resolution: str | bool | None = None,
        vision_min_tokens_per_image: int | None = None,
        vision_blind_node_retries: int | None = None,
        vision_blind_node_guard: bool | str | None = None,
        vision_blind_node_on_exhaust: str | None = None,
        fallback_model: str | None = None,
        fallback_provider: str | None = None,
        fallback_base_url: str | None = None,
        fallback_api_key: str | None = None,
        fallback_api_key_env: str | None = None,
        fallback_key_file: str | Path | None = None,
        circuit_breaker: VlmCircuitBreaker | None = None,
        workspace_dir: Path | None = None,
        budget: BudgetTracker | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        src = config_manager or config or cfg
        ws_dir = workspace_dir
        vlm_cfg: dict[str, Any] = {}
        if isinstance(src, ConfigManager):
            vlm_cfg = dict(src.settings.get("vlm", {}))
            if ws_dir is None:
                ws_dir = src.workspace_dir
        elif isinstance(src, dict):
            vlm_cfg = dict(src["vlm"]) if "vlm" in src and isinstance(src["vlm"], dict) else dict(src)
        if settings:
            s_vlm = settings["vlm"] if "vlm" in settings and isinstance(settings["vlm"], dict) else settings
            vlm_cfg = {**vlm_cfg, **s_vlm}

        self.provider = str(provider or vlm_cfg.get("provider") or "openai_compat").strip().lower()
        if self.provider not in _SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported VLM provider {self.provider!r}; expected one of {sorted(_SUPPORTED_PROVIDERS)}"
            )

        if self.provider == "gemini":
            def_base_url = DEFAULT_GEMINI_BASE_URL
            def_api_key_env = DEFAULT_GEMINI_API_KEY_ENV
            def_key_file: str | None = DEFAULT_GEMINI_KEY_FILE
            def_model = DEFAULT_GEMINI_MODEL
        else:
            def_base_url = DEFAULT_OPENAI_COMPAT_BASE_URL
            def_api_key_env = DEFAULT_OPENAI_COMPAT_API_KEY_ENV
            def_key_file = DEFAULT_OPENAI_COMPAT_KEY_FILE
            def_model = DEFAULT_OPENAI_COMPAT_MODEL

        raw_base_url = str(base_url or vlm_cfg.get("base_url") or def_base_url).strip().rstrip("/")
        self.base_url = (
            normalize_gemini_base_url(raw_base_url)
            if self.provider == "gemini"
            else raw_base_url
        )
        self.api_key_env = (
            api_key_env
            if api_key_env is not None
            else str(vlm_cfg.get("api_key_env") or def_api_key_env)
        )
        self.key_file = (
            key_file if key_file is not None else vlm_cfg.get("key_file", def_key_file)
        )
        raw_tm = str(
            thinking_mode
            if thinking_mode is not None
            else vlm_cfg.get("thinking_mode", "auto")
        ).strip().lower()
        if raw_tm not in {"auto", "dashscope", "disabled"}:
            raise ValueError(
                f"Invalid thinking_mode {raw_tm!r}; expected 'auto', 'dashscope', or 'disabled'"
            )
        self.thinking_mode = raw_tm
        self.workspace_dir = Path(ws_dir) if ws_dir is not None else None
        self._explicit_api_key = api_key

        self.timeout_s = float(
            timeout_s if timeout_s is not None else vlm_cfg.get("timeout_s", 180.0)
        )
        self.retries = int(retries if retries is not None else vlm_cfg.get("retries", 3))
        self.retry_min_wait_s = float(
            retry_min_wait_s
            if retry_min_wait_s is not None
            else vlm_cfg.get("retry_min_wait_s", 1.0)
        )
        self.retry_max_wait_s = float(
            retry_max_wait_s
            if retry_max_wait_s is not None
            else vlm_cfg.get("retry_max_wait_s", 16.0)
        )
        self.concurrency = max(
            1, int(concurrency if concurrency is not None else vlm_cfg.get("concurrency", 32))
        )
        self.stage_models: dict[str, str] = {
            normalize_stage_name(str(k)): str(v)
            for k, v in (stage_models or vlm_cfg.get("stage_models") or {}).items()
        }
        self._stage_configs: dict[str, dict[str, Any]] = {
            normalize_stage_name(str(k)): dict(v)
            for k, v in vlm_cfg.items()
            if isinstance(v, dict) and normalize_stage_name(str(k)) in _STAGE_CONFIG_ALIASES.values()
        }
        raw_temp = temperature if temperature is not None else vlm_cfg.get("temperature")
        self.temperature: float | None = float(raw_temp) if raw_temp is not None else None
        raw_top_p = top_p if top_p is not None else vlm_cfg.get("top_p")
        self.top_p: float | None = float(raw_top_p) if raw_top_p is not None else None
        raw_tl = thinking_level if thinking_level is not None else vlm_cfg.get("thinking_level")
        self.thinking_level: str | None = (
            str(raw_tl).strip().lower() if raw_tl is not None and str(raw_tl).strip() else None
        )
        self._is_loopback: bool = _is_loopback_url(self.base_url)

        self._gemini_media_resolution_explicit: bool = False
        if gemini_media_resolution is not None:
            self._gemini_media_resolution_explicit = True
            self.gemini_media_resolution: str | None = normalize_gemini_media_resolution(
                gemini_media_resolution
            )
        elif "gemini_media_resolution" in vlm_cfg and vlm_cfg["gemini_media_resolution"] is not None:
            self._gemini_media_resolution_explicit = True
            self.gemini_media_resolution = normalize_gemini_media_resolution(
                vlm_cfg["gemini_media_resolution"]
            )
        elif self.provider == "gemini" and not self._is_loopback:
            self.gemini_media_resolution = DEFAULT_GEMINI_MEDIA_RESOLUTION
        else:
            self.gemini_media_resolution = None

        raw_min_tpi = (
            vision_min_tokens_per_image
            if vision_min_tokens_per_image is not None
            else vlm_cfg.get("vision_min_tokens_per_image", DEFAULT_VISION_MIN_TOKENS_PER_IMAGE)
        )
        self.vision_min_tokens_per_image: int = max(0, int(raw_min_tpi or 0))

        raw_blind_retries = (
            vision_blind_node_retries
            if vision_blind_node_retries is not None
            else vlm_cfg.get("vision_blind_node_retries", DEFAULT_VISION_BLIND_NODE_RETRIES)
        )
        self.vision_blind_node_retries: int = max(0, int(raw_blind_retries or 0))

        raw_guard = (
            vision_blind_node_guard
            if vision_blind_node_guard is not None
            else vlm_cfg.get("vision_blind_node_guard", "auto")
        )
        if isinstance(raw_guard, bool):
            self.vision_blind_node_guard: bool | str = raw_guard
        else:
            g_low = str(raw_guard if raw_guard is not None else "auto").strip().lower()
            if g_low in {"", "auto"}:
                self.vision_blind_node_guard = "auto"
            elif g_low in {"true", "1", "yes", "on", "always", "force"}:
                self.vision_blind_node_guard = True
            elif g_low in _DISABLED_SENTINELS or g_low in {"no", "never"}:
                self.vision_blind_node_guard = False
            else:
                raise ValueError(
                    f"Unsupported vision_blind_node_guard {raw_guard!r}; expected 'auto', True, or False"
                )

        raw_exhaust = (
            vision_blind_node_on_exhaust
            if vision_blind_node_on_exhaust is not None
            else vlm_cfg.get("vision_blind_node_on_exhaust", "raise")
        )
        self.vision_blind_node_on_exhaust: str = str(raw_exhaust or "raise").strip().lower()
        if self.vision_blind_node_on_exhaust not in {"raise", "warn"}:
            raise ValueError(
                f"Unsupported vision_blind_node_on_exhaust {raw_exhaust!r}; expected 'raise' or 'warn'"
            )

        self.default_model = str(
            vlm_cfg.get("default_model")
            or self.stage_models.get("s3_discover")
            or def_model
        )

        # Opt-in fallback model/provider (disabled when fallback_model is None)
        raw_fb_model = (
            fallback_model
            if fallback_model is not None
            else vlm_cfg.get("fallback_model")
        )
        self.fallback_model: str | None = (
            str(raw_fb_model).strip()
            if raw_fb_model is not None and str(raw_fb_model).strip()
            else None
        )
        raw_fb_prov = (
            fallback_provider
            if fallback_provider is not None
            else vlm_cfg.get("fallback_provider")
        )
        self.fallback_provider: str | None = (
            str(raw_fb_prov).strip().lower()
            if raw_fb_prov is not None and str(raw_fb_prov).strip()
            else None
        )
        if self.fallback_provider is not None and self.fallback_provider not in _SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported fallback VLM provider {self.fallback_provider!r}; "
                f"expected one of {sorted(_SUPPORTED_PROVIDERS)}"
            )
        raw_fb_base = (
            fallback_base_url
            if fallback_base_url is not None
            else vlm_cfg.get("fallback_base_url")
        )
        self.fallback_base_url: str | None = (
            str(raw_fb_base).strip().rstrip("/")
            if raw_fb_base is not None and str(raw_fb_base).strip()
            else None
        )
        self._fallback_explicit_api_key: str | None = fallback_api_key
        raw_fb_env = (
            fallback_api_key_env
            if fallback_api_key_env is not None
            else vlm_cfg.get("fallback_api_key_env")
        )
        self.fallback_api_key_env: str | None = (
            str(raw_fb_env).strip()
            if raw_fb_env is not None and str(raw_fb_env).strip()
            else None
        )
        raw_fb_kf = (
            fallback_key_file
            if fallback_key_file is not None
            else vlm_cfg.get("fallback_key_file")
        )
        self.fallback_key_file: str | Path | None = raw_fb_kf

        # Optional circuit breaker (explicitly passed or configured in vlm.circuit_breaker)
        if circuit_breaker is not None:
            self.circuit_breaker: VlmCircuitBreaker | None = circuit_breaker
        elif isinstance(vlm_cfg.get("circuit_breaker"), dict):
            cb_cfg = vlm_cfg["circuit_breaker"]
            self.circuit_breaker = VlmCircuitBreaker(
                failure_threshold=int(cb_cfg.get("failure_threshold", 3)),
                trip_on_quota_403=bool(cb_cfg.get("trip_on_quota_403", True)),
                enabled=bool(cb_cfg.get("enabled", True)),
            )
        else:
            self.circuit_breaker = None

        self.pricing: dict[str, Any] = (
            dict(vlm_cfg["pricing"]) if isinstance(vlm_cfg.get("pricing"), dict) else {}
        )
        self.budget = budget or BudgetTracker(pricing=self.pricing or None)
        self._cancel_event = cancel_event or threading.Event()
        self._ephemeral_lock = threading.Lock()
        self._ephemeral_clients: set[httpx.Client] = set()

        install_http_log_redaction()
        self._sem = threading.Semaphore(self.concurrency)
        self._http = httpx.Client(
            trust_env=False,
            timeout=httpx.Timeout(self.timeout_s),
            limits=httpx.Limits(
                max_connections=max(64, self.concurrency * 2),
                max_keepalive_connections=max(32, self.concurrency),
            ),
        )

    @classmethod
    def from_config(
        cls,
        config: ConfigManager | dict[str, Any],
        *,
        workspace_dir: Path | None = None,
        budget: BudgetTracker | None = None,
        cancel_event: threading.Event | None = None,
        circuit_breaker: VlmCircuitBreaker | None = None,
    ) -> "VlmClient":
        return cls(
            config=config,
            workspace_dir=workspace_dir,
            budget=budget,
            cancel_event=cancel_event,
            circuit_breaker=circuit_breaker,
        )

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def _check_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise VlmCancelledError("VLM request cancelled")

    @staticmethod
    def _abort_http_client_sockets(http_client: httpx.Client) -> None:
        """Shutdown active TCP sockets on a given httpx.Client."""
        transport = getattr(http_client, "_transport", None)
        pool = getattr(transport, "_pool", None)
        if pool is None:
            return
        conns = getattr(pool, "connections", None)
        if not conns:
            return
        for conn in list(conns):
            with contextlib.suppress(Exception):
                inner = getattr(conn, "_connection", None)
                ns = getattr(conn, "_network_stream", None) or getattr(
                    inner, "_network_stream", None
                )
                sock = getattr(ns, "_sock", None) if ns is not None else None
                if sock is not None:
                    with contextlib.suppress(OSError):
                        sock.shutdown(socket.SHUT_RDWR)

    def _abort_active_sockets(self) -> None:
        """Shutdown active TCP sockets so threads blocked in recv() wake up immediately."""
        self._abort_http_client_sockets(self._http)
        with self._ephemeral_lock:
            ephemerals = list(self._ephemeral_clients)
        for hc in ephemerals:
            self._abort_http_client_sockets(hc)

    def cancel(self) -> None:
        """Signal cooperative cancellation and abort in-flight HTTP requests immediately."""
        self._cancel_event.set()
        self._abort_active_sockets()
        with contextlib.suppress(Exception):
            self._http.close()
        with self._ephemeral_lock:
            ephemerals = list(self._ephemeral_clients)
        for hc in ephemerals:
            with contextlib.suppress(Exception):
                hc.close()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._http.close()
        with self._ephemeral_lock:
            ephemerals = list(self._ephemeral_clients)
            self._ephemeral_clients.clear()
        for hc in ephemerals:
            with contextlib.suppress(Exception):
                hc.close()

    def _media_resolution_for_request(
        self,
        model: str,
        n_images: int,
        *,
        provider: str | None = None,
        base_url: str | None = None,
    ) -> str | None:
        eff_provider = (provider or self.provider).strip().lower()
        if (
            eff_provider != "gemini"
            or n_images <= 0
            or not is_gemini_hd_vision_model(model)
        ):
            return None
        if self.gemini_media_resolution:
            return self.gemini_media_resolution
        if self._gemini_media_resolution_explicit:
            return None
        eff_is_loopback = _is_loopback_url(base_url) if base_url else self._is_loopback
        if not eff_is_loopback:
            return DEFAULT_GEMINI_MEDIA_RESOLUTION
        return None

    def _should_enforce_blind_node_guard(
        self,
        model: str,
        n_images: int,
        *,
        provider: str | None = None,
        base_url: str | None = None,
    ) -> bool:
        eff_provider = (provider or self.provider).strip().lower()
        if (
            eff_provider != "gemini"
            or n_images <= 0
            or self.vision_min_tokens_per_image <= 0
            or not is_gemini_hd_vision_model(model)
        ):
            return False
        if self.vision_blind_node_guard is False:
            return False
        if self.vision_blind_node_guard is True:
            return True
        eff_is_loopback = _is_loopback_url(base_url) if base_url else self._is_loopback
        return not eff_is_loopback

    def __enter__(self) -> "VlmClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _get_api_key(self) -> str:
        if self._explicit_api_key:
            return self._explicit_api_key
        return resolve_api_key(
            api_key_env=self.api_key_env,
            key_file=self.key_file,
            workspace_dir=self.workspace_dir,
        )

    def model_for_stage(self, stage: str, override: str | None = None) -> str:
        if override:
            return override
        norm = normalize_stage_name(stage)
        if norm in self.stage_models:
            return self.stage_models[norm]
        if stage in self.stage_models:
            return self.stage_models[stage]
        return self.default_model

    def resolve_stage_model(
        self, stage: str, model_override: str | None = None
    ) -> str:
        """Resolve effective VLM model for a stage, honoring an optional model_override."""
        return self.model_for_stage(stage, override=model_override)

    def _post_openai_compat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        thinking: bool,
        thinking_budget: int | None,
        max_tokens: int,
        temperature: float | None = None,
        top_p: float | None = None,
        json_mode: bool = True,
        base_url_override: str | None = None,
        api_key_override: str | None = None,
    ) -> tuple[str, dict[str, int], dict[str, Any]]:
        api_key = api_key_override if api_key_override is not None else self._get_api_key()
        eff_base_url = (base_url_override or self.base_url).rstrip("/")
        url = f"{eff_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Connection": "keep-alive",
        }
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if temperature is not None:
            body["temperature"] = float(temperature)
        if top_p is not None:
            body["top_p"] = float(top_p)
        use_dashscope_thinking = bool(thinking) and (
            self.thinking_mode == "dashscope"
            or (self.thinking_mode == "auto" and "dashscope" in eff_base_url.lower())
        )
        eff_budget = int(thinking_budget or 0)
        if use_dashscope_thinking:
            body["enable_thinking"] = True
            if eff_budget > 0:
                body["thinking_budget"] = eff_budget
                body["max_tokens"] = eff_budget + int(max_tokens)
            else:
                body["max_tokens"] = int(max_tokens)
            # Omit response_format whenever DashScope thinking is enabled
        else:
            body["max_tokens"] = int(max_tokens)
            if json_mode:
                body["response_format"] = {"type": "json_object"}

        try:
            resp = self._http.post(url, headers=headers, json=body)
        except (
            httpx.TimeoutException,
            httpx.TransportError,
            httpx.HTTPError,
            OSError,
        ) as exc:
            self._check_cancelled()
            safe_err = redact_secrets(f"{type(exc).__name__}: {exc}", api_key)
            err = VlmError(f"OpenAI-compat API transport error: {safe_err}")
            setattr(err, "_transient", True)
            raise err from None

        if resp.status_code >= 400:
            body_preview = redact_secrets(resp.text[:500], api_key)
            err = VlmError(f"HTTP {resp.status_code} from VLM API: {body_preview}")
            setattr(err, "_status_code", resp.status_code)
            setattr(err, "_body_text", body_preview)
            raise err

        try:
            data = resp.json()
        except (ValueError, json.JSONDecodeError) as exc:
            safe_err = redact_secrets(str(exc), api_key)
            preview = redact_secrets(resp.text[:300], api_key)
            err = VlmError(f"Invalid JSON from VLM API: {safe_err} (body={preview!r})")
            setattr(err, "_transient", True)
            raise err from None

        choices = data.get("choices") or []
        if not choices:
            preview = redact_secrets(str(data)[:300], api_key)
            raise VlmError(f"Empty choices in VLM response: {preview}")
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, list):
            content = "".join(
                str(b.get("text", ""))
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        text = str(content or "")
        usage = data.get("usage") or {}
        p_tok = int(usage.get("prompt_tokens") or 0)
        c_tok = int(usage.get("completion_tokens") or 0)
        t_tok = int(usage.get("total_tokens") or (p_tok + c_tok))
        details = usage.get("completion_tokens_details") or {}
        r_tok = int(details.get("reasoning_tokens") or 0)
        return (
            text,
            {
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "total_tokens": t_tok,
                "reasoning_tokens": r_tok,
            },
            {
                "finish_reason": choices[0].get("finish_reason"),
                "reasoning_included_in_completion": not (
                    r_tok > 0 and t_tok >= (p_tok + c_tok + r_tok)
                ),
            },
        )

    def _post_gemini(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        thinking: bool,
        thinking_budget: int | None,
        thinking_level: str | None = None,
        max_tokens: int,
        temperature: float | None = None,
        top_p: float | None = None,
        json_mode: bool = True,
        force_new_connection: bool = False,
        base_url_override: str | None = None,
        api_key_override: str | None = None,
    ) -> tuple[str, dict[str, int], dict[str, Any]]:
        api_key = api_key_override if api_key_override is not None else self._get_api_key()
        eff_base = base_url_override or self.base_url
        url = gemini_generate_url(eff_base, model)
        # Send both ?key= query param (native v1beta convention) and x-goog-api-key header;
        # NEVER send Authorization: Bearer to Gemini native endpoints.
        headers = {
            "Content-Type": "application/json",
            "Connection": "close" if force_new_connection else "keep-alive",
            "x-goog-api-key": api_key,
        }
        sys_instruction, contents = messages_to_gemini_payload(messages)
        n_images, est_text_tokens = count_gemini_payload_media(sys_instruction, contents)
        eff_media_res = self._media_resolution_for_request(
            model, n_images, provider="gemini", base_url=eff_base
        )
        eff_thinking = bool(thinking) and self.thinking_mode != "disabled"
        eff_thinking_level = None if self.thinking_mode == "disabled" else thinking_level
        gen_cfg = build_gemini_generation_config(
            model=model,
            thinking=eff_thinking,
            thinking_budget=thinking_budget,
            thinking_level=eff_thinking_level,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            json_mode=json_mode,
            media_resolution=eff_media_res,
        )
        body: dict[str, Any] = {"contents": contents, "generationConfig": gen_cfg}
        if sys_instruction is not None:
            body["systemInstruction"] = sys_instruction

        ephemeral_http: httpx.Client | None = None
        if force_new_connection:
            ephemeral_http = httpx.Client(
                trust_env=False,
                timeout=httpx.Timeout(self.timeout_s),
                limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
            )
            with self._ephemeral_lock:
                self._ephemeral_clients.add(ephemeral_http)
            http_client = ephemeral_http
        else:
            http_client = self._http

        try:
            self._check_cancelled()
            resp = http_client.post(
                url,
                params={"key": api_key},
                headers=headers,
                json=body,
            )
        except VlmCancelledError:
            raise
        except (
            httpx.TimeoutException,
            httpx.TransportError,
            httpx.HTTPError,
            OSError,
        ) as exc:
            self._check_cancelled()
            safe_err = redact_secrets(f"{type(exc).__name__}: {exc}", api_key)
            err = VlmError(f"Gemini API transport error: {safe_err}")
            setattr(err, "_transient", True)
            raise err from None
        finally:
            if ephemeral_http is not None:
                with self._ephemeral_lock:
                    self._ephemeral_clients.discard(ephemeral_http)
                with contextlib.suppress(Exception):
                    ephemeral_http.close()

        if resp.status_code >= 400:
            body_preview = redact_secrets(resp.text[:500], api_key)
            err = VlmError(f"HTTP {resp.status_code} from Gemini API: {body_preview}")
            setattr(err, "_status_code", resp.status_code)
            setattr(err, "_body_text", body_preview)
            raise err

        try:
            data = resp.json()
        except (ValueError, json.JSONDecodeError) as exc:
            safe_err = redact_secrets(str(exc), api_key)
            preview = redact_secrets(resp.text[:300], api_key)
            err = VlmError(f"Invalid JSON from Gemini API: {safe_err} (body={preview!r})")
            setattr(err, "_transient", True)
            raise err from None

        if not isinstance(data, dict):
            preview = redact_secrets(str(data)[:300], api_key)
            raise VlmError(f"Unexpected non-object JSON from Gemini API: {preview}")

        answer_text, usage, meta = parse_gemini_response(data, secret=api_key)
        meta["n_images"] = n_images
        meta["est_text_tokens"] = est_text_tokens
        return answer_text, usage, meta

    def _call_once_with_http_retries(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        thinking: bool,
        thinking_budget: int | None,
        thinking_level: str | None = None,
        max_tokens: int,
        temperature: float | None = None,
        top_p: float | None = None,
        json_mode: bool = True,
        force_new_connection: bool = False,
        provider_override: str | None = None,
        base_url_override: str | None = None,
        api_key_override: str | None = None,
    ) -> tuple[str, dict[str, int], dict[str, Any]]:
        eff_provider = (provider_override or self.provider).strip().lower()
        last_exc: Exception | None = None
        max_attempts = max(1, self.retries + 1)
        for attempt in range(max_attempts):
            self._check_cancelled()
            try:
                if eff_provider == "gemini":
                    return self._post_gemini(
                        model=model,
                        messages=messages,
                        thinking=thinking,
                        thinking_budget=thinking_budget,
                        thinking_level=thinking_level,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        json_mode=json_mode,
                        force_new_connection=force_new_connection or (attempt > 0),
                        base_url_override=base_url_override,
                        api_key_override=api_key_override,
                    )
                return self._post_openai_compat(
                    model=model,
                    messages=messages,
                    thinking=thinking,
                    thinking_budget=thinking_budget,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    json_mode=json_mode,
                    base_url_override=base_url_override,
                    api_key_override=api_key_override,
                )
            except VlmCancelledError:
                raise
            except VlmError as exc:
                self._check_cancelled()
                last_exc = exc
                if getattr(exc, "_transient", False):
                    transient = True
                else:
                    sc = getattr(exc, "_status_code", None)
                    bt = getattr(exc, "_body_text", "")
                    transient = sc is not None and _should_retry_http(int(sc), str(bt))
            except (
                httpx.TimeoutException,
                httpx.TransportError,
                httpx.HTTPError,
                OSError,
                json.JSONDecodeError,
            ) as exc:
                self._check_cancelled()
                secret_for_redact = api_key_override or self._explicit_api_key
                safe_err = redact_secrets(
                    f"{type(exc).__name__}: {exc}", secret_for_redact
                )
                last_exc = VlmError(f"VLM transport error: {safe_err}")
                setattr(last_exc, "_transient", True)
                transient = True
            except Exception:
                self._check_cancelled()
                raise

            self._check_cancelled()
            if not transient or attempt >= max_attempts - 1:
                break
            wait = min(
                self.retry_max_wait_s,
                max(self.retry_min_wait_s, self.retry_min_wait_s * (2.0**attempt)),
            )
            jitter = wait * random.uniform(0.0, 0.25)
            sleep_s = min(self.retry_max_wait_s, wait + jitter)
            if self._cancel_event.wait(timeout=sleep_s):
                raise VlmCancelledError("VLM request cancelled during retry wait")

        if isinstance(last_exc, VlmError):
            if getattr(last_exc, "_transient", False) and max_attempts > 1:
                raise VlmError(
                    f"VLM request failed after {max_attempts} attempts: {last_exc}"
                ) from None
            raise last_exc from None
        secret_for_redact = api_key_override or self._explicit_api_key
        safe_msg = redact_secrets(str(last_exc), secret_for_redact)
        raise VlmError(
            f"VLM request failed after {max_attempts} attempts: {safe_msg}"
        ) from None

    def _resolve_fallback_target(self) -> tuple[str, str, str, str]:
        """Resolve (provider, base_url, api_key, model) for opt-in fallback invocation."""
        assert self.fallback_model is not None
        fb_provider = self.fallback_provider or self.provider
        if fb_provider == self.provider:
            fb_base = self.fallback_base_url or self.base_url
            if self._fallback_explicit_api_key:
                fb_key = self._fallback_explicit_api_key
            elif self.fallback_api_key_env or self.fallback_key_file:
                fb_key = resolve_api_key(
                    api_key_env=self.fallback_api_key_env or self.api_key_env,
                    key_file=self.fallback_key_file or self.key_file,
                    workspace_dir=self.workspace_dir,
                )
            else:
                fb_key = self._get_api_key()
        else:
            if fb_provider == "gemini":
                def_base = DEFAULT_GEMINI_BASE_URL
                def_env = DEFAULT_GEMINI_API_KEY_ENV
                def_kf = DEFAULT_GEMINI_KEY_FILE
            else:
                def_base = DEFAULT_OPENAI_COMPAT_BASE_URL
                def_env = DEFAULT_OPENAI_COMPAT_API_KEY_ENV
                def_kf = DEFAULT_OPENAI_COMPAT_KEY_FILE
            raw_base = self.fallback_base_url or def_base
            fb_base = (
                normalize_gemini_base_url(raw_base)
                if fb_provider == "gemini"
                else raw_base.rstrip("/")
            )
            if self._fallback_explicit_api_key:
                fb_key = self._fallback_explicit_api_key
            elif self._explicit_api_key and not (self.fallback_api_key_env or self.fallback_key_file):
                fb_key = self._explicit_api_key
            else:
                fb_key = resolve_api_key(
                    api_key_env=self.fallback_api_key_env or def_env,
                    key_file=(
                        self.fallback_key_file
                        if self.fallback_key_file is not None
                        else def_kf
                    ),
                    workspace_dir=self.workspace_dir,
                )
        return fb_provider, fb_base, fb_key, self.fallback_model

    def _run_chat_json_loop(
        self,
        *,
        stage: str,
        eff_model: str,
        eff_messages: list[dict[str, Any]],
        thinking: bool,
        thinking_budget: int | None,
        eff_thinking_level: str | None,
        max_tokens: int,
        eff_temp: float | None,
        eff_top_p: float | None,
        json_mode: bool,
        tracker: BudgetTracker,
        provider_override: str | None = None,
        base_url_override: str | None = None,
        api_key_override: str | None = None,
    ) -> dict[str, Any]:
        eff_provider = (provider_override or self.provider).strip().lower()
        last_json_err: Exception | None = None
        last_finish_reason: Any = None
        blind_retries_used = 0
        json_attempts = 0
        max_json_attempts = 2
        force_new_conn = False

        while json_attempts < max_json_attempts:
            self._check_cancelled()
            tracker.reserve()
            t0 = time.perf_counter()
            try:
                with self._sem:
                    self._check_cancelled()
                    text, usage, meta = self._call_once_with_http_retries(
                        model=eff_model,
                        messages=eff_messages,
                        thinking=thinking,
                        thinking_budget=thinking_budget,
                        thinking_level=eff_thinking_level,
                        max_tokens=max_tokens,
                        temperature=eff_temp,
                        top_p=eff_top_p,
                        json_mode=json_mode,
                        force_new_connection=force_new_conn,
                        provider_override=eff_provider,
                        base_url_override=base_url_override,
                        api_key_override=api_key_override,
                    )
            finally:
                tracker.release()
            self._check_cancelled()
            elapsed = time.perf_counter() - t0
            tracker.record(
                stage=stage,
                model=eff_model,
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"],
                reasoning_tokens=usage["reasoning_tokens"],
                reasoning_included_in_completion=meta.get("reasoning_included_in_completion"),
                elapsed_s=elapsed,
                cache_hit=False,
            )

            n_images = int(meta.get("n_images") or 0)
            if eff_provider == "gemini" and self._should_enforce_blind_node_guard(
                eff_model,
                n_images,
                provider=eff_provider,
                base_url=base_url_override or self.base_url,
            ):
                is_blind, tok_per_img, eff_img_toks, density_src = (
                    evaluate_vision_token_density(
                        n_images=n_images,
                        prompt_tokens=usage["prompt_tokens"],
                        exact_image_tokens=meta.get("image_prompt_tokens"),
                        est_text_tokens=int(meta.get("est_text_tokens") or 0),
                        min_tokens_per_image=self.vision_min_tokens_per_image,
                    )
                )
                if is_blind:
                    max_blind = self.vision_blind_node_retries
                    logger.warning(
                        "VLM blind-node detected (stage=%s model=%s n_images=%d "
                        "prompt_tokens=%d eff_img_tokens=%d tok_per_img=%.1f "
                        "min_required=%d src=%s retry=%d/%d)",
                        stage,
                        eff_model,
                        n_images,
                        usage["prompt_tokens"],
                        eff_img_toks,
                        float(tok_per_img or 0.0),
                        self.vision_min_tokens_per_image,
                        density_src,
                        blind_retries_used,
                        max_blind,
                    )
                    if blind_retries_used < max_blind:
                        wait = min(
                            self.retry_max_wait_s,
                            max(
                                self.retry_min_wait_s,
                                self.retry_min_wait_s * (1.6**blind_retries_used),
                            ),
                        )
                        jitter = wait * random.uniform(0.15, 0.45)
                        sleep_s = min(self.retry_max_wait_s, wait + jitter)
                        blind_retries_used += 1
                        force_new_conn = True
                        if self._cancel_event.wait(timeout=sleep_s):
                            raise VlmCancelledError(
                                "VLM request cancelled during blind-node retry wait"
                            )
                        continue
                    err_msg = (
                        f"Gemini blind/degraded vision node persisted after "
                        f"{blind_retries_used + 1} attempts (stage={stage}, "
                        f"model={eff_model}, n_images={n_images}, "
                        f"prompt_tokens={usage['prompt_tokens']}, "
                        f"eff_img_tokens={eff_img_toks}, "
                        f"tok_per_img={float(tok_per_img or 0.0):.1f} < "
                        f"{self.vision_min_tokens_per_image}, src={density_src})"
                    )
                    if self.vision_blind_node_on_exhaust == "warn":
                        logger.error(
                            "%s; returning degraded response (vision_blind_node_on_exhaust=warn)",
                            err_msg,
                        )
                    else:
                        raise VlmBlindVisionError(err_msg)
                elif blind_retries_used > 0:
                    logger.info(
                        "VLM blind-node recovered after %d retry(ies) (stage=%s model=%s "
                        "n_images=%d prompt_tokens=%d eff_img_tokens=%d tok_per_img=%.1f src=%s)",
                        blind_retries_used,
                        stage,
                        eff_model,
                        n_images,
                        usage["prompt_tokens"],
                        eff_img_toks,
                        float(tok_per_img or 0.0),
                        density_src,
                    )
                    force_new_conn = False

            json_attempts += 1
            last_finish_reason = meta.get("finish_reason")
            try:
                return parse_json(text)
            except (ValueError, json.JSONDecodeError) as exc:
                last_json_err = exc
                continue

        reason_hint = (
            f" (finishReason={last_finish_reason!r})" if last_finish_reason else ""
        )
        raise VlmError(
            f"Model did not return valid JSON after retry{reason_hint}: {last_json_err}"
        )

    def chat_json(
        self,
        *,
        stage: str = "s3_discover",
        model: str | None = None,
        model_override: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        system_prompt: str = "",
        user_prompt: str = "",
        images: Sequence[np.ndarray] | None = None,
        timestamps_s: Sequence[float | None] | None = None,
        labels: Sequence[str] | None = None,
        thinking: bool = False,
        thinking_budget: int | None = None,
        thinking_level: str | None = None,
        max_tokens: int = 2048,
        temperature: float | None = None,
        top_p: float | None = None,
        json_mode: bool = True,
        jpeg_quality: int = 74,
        text_first: bool = False,
        budget: BudgetTracker | None = None,
    ) -> dict[str, Any]:
        """Call VLM endpoint and parse a JSON dict, with circuit breaker, retries, and opt-in fallback."""
        self._check_cancelled()
        eff_override = model_override if model_override is not None else model
        eff_model = self.resolve_stage_model(stage, model_override=eff_override)
        norm_stage = normalize_stage_name(stage)
        stage_cfg = self._stage_configs.get(norm_stage, {})
        eff_temp = temperature if temperature is not None else stage_cfg.get("temperature", self.temperature)
        eff_top_p = top_p if top_p is not None else stage_cfg.get("top_p", self.top_p)
        eff_thinking_level = (
            thinking_level
            if thinking_level is not None
            else (
                stage_cfg.get("thinking_level", self.thinking_level)
                if thinking
                else None
            )
        )

        if messages is None:
            eff_messages = build_interleaved_messages(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images=images,
                timestamps_s=timestamps_s,
                labels=labels,
                jpeg_quality=jpeg_quality,
                text_first=text_first,
            )
        elif images:
            # Attach images to last user message if both messages & images passed
            eff_messages = list(messages)
            last_user = ""
            for m in reversed(eff_messages):
                if m.get("role") == "user" and isinstance(m.get("content"), str):
                    last_user = m["content"]
                    break
            built = build_interleaved_messages(
                system_prompt="",
                user_prompt=last_user,
                images=images,
                timestamps_s=timestamps_s,
                labels=labels,
                jpeg_quality=jpeg_quality,
                text_first=text_first,
            )
            replaced = False
            for i in range(len(eff_messages) - 1, -1, -1):
                if eff_messages[i].get("role") == "user":
                    eff_messages[i] = built[-1]
                    replaced = True
                    break
            if not replaced:
                eff_messages.extend(built)
        else:
            eff_messages = messages

        tracker = budget or self.budget
        can_fallback = bool(
            self.fallback_model
            and (
                self.fallback_model != eff_model
                or (self.fallback_provider and self.fallback_provider != self.provider)
            )
        )

        primary_err: Exception | None = None
        breaker_tripped = False
        try:
            if self.circuit_breaker is not None:
                self.circuit_breaker.check()
            parsed = self._run_chat_json_loop(
                stage=stage,
                eff_model=eff_model,
                eff_messages=eff_messages,
                thinking=thinking,
                thinking_budget=thinking_budget,
                eff_thinking_level=eff_thinking_level,
                max_tokens=max_tokens,
                eff_temp=eff_temp,
                eff_top_p=eff_top_p,
                json_mode=json_mode,
                tracker=tracker,
            )
            if self.circuit_breaker is not None:
                self.circuit_breaker.record_success()
            return parsed
        except (VlmCancelledError, BudgetExceeded):
            raise
        except VlmCircuitOpenError as cb_exc:
            primary_err = cb_exc
            breaker_tripped = True
        except VlmError as exc:
            primary_err = exc
            if self.circuit_breaker is not None:
                breaker_tripped = self.circuit_breaker.record_failure(exc)

        if can_fallback and primary_err is not None:
            self._check_cancelled()
            fb_provider, fb_base, fb_key, fb_model = self._resolve_fallback_target()
            logger.warning(
                "Primary VLM call failed (stage=%s provider=%s model=%s err=%s); "
                "falling back once to provider=%s model=%s",
                stage,
                self.provider,
                eff_model,
                str(primary_err)[:200],
                fb_provider,
                fb_model,
            )
            return self._run_chat_json_loop(
                stage=stage,
                eff_model=fb_model,
                eff_messages=eff_messages,
                thinking=thinking,
                thinking_budget=thinking_budget,
                eff_thinking_level=eff_thinking_level,
                max_tokens=max_tokens,
                eff_temp=eff_temp,
                eff_top_p=eff_top_p,
                json_mode=json_mode,
                tracker=tracker,
                provider_override=fb_provider,
                base_url_override=fb_base,
                api_key_override=fb_key,
            )

        if breaker_tripped and self.circuit_breaker is not None:
            self.circuit_breaker.check()
        assert primary_err is not None
        raise primary_err

    call_json = chat_json
    complete_json = chat_json
    call_stage = chat_json
