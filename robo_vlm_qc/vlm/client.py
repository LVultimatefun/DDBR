"""Multi-provider httpx VLM client (OpenAI-compatible, DashScope, Gemini) with retries and budget."""

from __future__ import annotations

import contextlib
import json
import os
import random
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import httpx
import numpy as np

from ..config import ConfigManager
from ..io.media import encode_jpeg_data_url
from ..models.annotation import StageUsage, UsageSummary

REPO_ROOT = Path(__file__).resolve().parents[2]

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


def check_client_cancelled(client: Any) -> None:
    """Raise VlmCancelledError if a VlmClient (or test double) signals cancellation."""
    is_c = getattr(client, "is_cancelled", None)
    if callable(is_c) and is_c():
        raise VlmCancelledError("VLM request cancelled")


def normalize_stage_name(stage: str) -> str:
    s = stage.strip().lower()
    return _STAGE_CONFIG_ALIASES.get(s, s)


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
    api_key_env: str = "OPENAI_API_KEY",
    key_file: str | Path | None = ".env",
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
                found = _parse_env_file_for_key(cand, api_key_env or "OPENAI_API_KEY")
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
    timestamps_s: Sequence[float] | None = None,
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
        t_val = (
            float(timestamps_s[idx])
            if timestamps_s is not None and idx < len(timestamps_s)
            else float(idx)
        )
        lbl = (
            str(labels[idx]).strip()
            if labels is not None and idx < len(labels) and labels[idx]
            else ""
        )
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
    ) -> None:
        self.max_tokens = int(max_tokens) if max_tokens is not None else None
        self.max_calls = int(max_calls) if max_calls is not None else None
        self.parent = parent
        self._lock = threading.Lock()
        self.tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.calls = 0
        self.cached_stages = 0
        self.elapsed_s = 0.0
        self._reserved = 0
        self._stages: dict[str, _StageAccum] = {}

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
        elapsed_s: float = 0.0,
        cache_hit: bool = False,
    ) -> None:
        tot = (
            int(total_tokens)
            if total_tokens is not None
            else int(prompt_tokens) + int(completion_tokens)
        )
        with self._lock:
            if not cache_hit:
                self.calls += 1
                self.prompt_tokens += int(prompt_tokens)
                self.completion_tokens += int(completion_tokens)
                self.reasoning_tokens += int(reasoning_tokens)
                self.tokens += tot
                self.elapsed_s += float(elapsed_s)
            else:
                self.cached_stages += 1

            acc = self._stages.get(stage)
            if acc is None:
                acc = _StageAccum(stage=stage, model=model)
                self._stages[stage] = acc
            acc.model = model
            acc.prompt_tokens += int(prompt_tokens)
            acc.completion_tokens += int(completion_tokens)
            acc.total_tokens += tot
            acc.reasoning_tokens += int(reasoning_tokens)
            acc.elapsed_s += float(elapsed_s)
            if cache_hit:
                acc.cache_hits += 1
            else:
                acc.calls += 1

        if self.parent is not None:
            self.parent.record(
                stage=stage,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=tot,
                reasoning_tokens=reasoning_tokens,
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
                    estimated_cost_usd=0.0,
                    cache_hit=(acc.calls == 0 and acc.cache_hits > 0),
                )
                for acc in self._stages.values()
            ]
            return UsageSummary(
                stages=stages,
                total_tokens=self.tokens,
                total_elapsed_s=round(self.elapsed_s, 4),
                total_cost_usd=0.0,
            )


def _should_retry_http(status_code: int, body_text: str) -> bool:
    if status_code in (408, 429) or status_code >= 500:
        return True
    if status_code == 400 and "timed out" in body_text.lower():
        return True
    return False


class VlmClient:
    """Multi-provider httpx VLM client (openai_compat & gemini) with retries and budget."""

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

        self.provider = str(provider or vlm_cfg.get("provider", "openai_compat")).lower()
        self.base_url = str(
            base_url
            or vlm_cfg.get("base_url", "https://api.openai.com/v1")
        ).rstrip("/")
        self.api_key_env = (
            api_key_env
            if api_key_env is not None
            else str(vlm_cfg.get("api_key_env", "OPENAI_API_KEY"))
        )
        self.key_file = (
            key_file if key_file is not None else vlm_cfg.get("key_file", ".env")
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
        self.default_model = str(
            vlm_cfg.get("default_model")
            or self.stage_models.get("s3_discover")
            or "gpt-4o"
        )
        self.budget = budget or BudgetTracker()
        self._cancel_event = cancel_event or threading.Event()

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
    ) -> "VlmClient":
        return cls(
            config=config,
            workspace_dir=workspace_dir,
            budget=budget,
            cancel_event=cancel_event,
        )

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def _check_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise VlmCancelledError("VLM request cancelled")

    def _abort_active_sockets(self) -> None:
        """Shutdown active TCP sockets so threads blocked in recv() wake up immediately."""
        transport = getattr(self._http, "_transport", None)
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

    def cancel(self) -> None:
        """Signal cooperative cancellation and abort in-flight HTTP requests immediately."""
        self._cancel_event.set()
        self._abort_active_sockets()
        with contextlib.suppress(Exception):
            self._http.close()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._http.close()

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

    def _post_openai_compat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        thinking: bool,
        thinking_budget: int | None,
        max_tokens: int,
    ) -> tuple[str, dict[str, int]]:
        api_key = self._get_api_key()
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Connection": "keep-alive",
        }
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        use_dashscope_thinking = bool(thinking) and (
            self.thinking_mode == "dashscope"
            or (self.thinking_mode == "auto" and "dashscope" in self.base_url.lower())
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
            body["response_format"] = {"type": "json_object"}

        resp = self._http.post(url, headers=headers, json=body)
        if resp.status_code >= 400:
            body_preview = resp.text[:500]
            err = VlmError(f"HTTP {resp.status_code} from VLM API: {body_preview}")
            setattr(err, "_status_code", resp.status_code)
            setattr(err, "_body_text", body_preview)
            raise err

        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise VlmError(f"Empty choices in VLM response: {str(data)[:300]}")
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
        return text, {
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "total_tokens": t_tok,
            "reasoning_tokens": r_tok,
        }

    def _post_gemini(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        thinking: bool,
        thinking_budget: int | None,
        max_tokens: int,
    ) -> tuple[str, dict[str, int]]:
        api_key = self._get_api_key()
        url = f"{self.base_url}/models/{model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }
        sys_parts: list[dict[str, str]] = []
        contents: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user")
            raw_c = m.get("content", "")
            parts: list[dict[str, Any]] = []
            if isinstance(raw_c, str):
                parts.append({"text": raw_c})
            elif isinstance(raw_c, list):
                for blk in raw_c:
                    if not isinstance(blk, dict):
                        continue
                    if blk.get("type") == "text":
                        parts.append({"text": str(blk.get("text", ""))})
                    elif blk.get("type") == "image_url":
                        img_url = str((blk.get("image_url") or {}).get("url", ""))
                        if img_url.startswith("data:") and ";base64," in img_url:
                            hdr, b64 = img_url.split(";base64,", 1)
                            mime = hdr[5:] or "image/jpeg"
                            parts.append(
                                {"inlineData": {"mimeType": mime, "data": b64}}
                            )
            if role == "system":
                sys_parts.extend(parts)
            else:
                contents.append(
                    {"role": "model" if role == "assistant" else "user", "parts": parts}
                )

        eff_thinking = bool(thinking) and self.thinking_mode != "disabled"
        gen_cfg: dict[str, Any] = {
            "maxOutputTokens": int(max_tokens) + (int(thinking_budget or 0) if eff_thinking else 0),
            "responseMimeType": "application/json",
        }
        if eff_thinking and thinking_budget:
            gen_cfg["thinkingConfig"] = {"thinkingBudget": int(thinking_budget)}
        body: dict[str, Any] = {"contents": contents, "generationConfig": gen_cfg}
        if sys_parts:
            body["systemInstruction"] = {"parts": sys_parts}

        resp = self._http.post(url, headers=headers, json=body)
        if resp.status_code >= 400:
            body_preview = resp.text[:500]
            err = VlmError(f"HTTP {resp.status_code} from Gemini API: {body_preview}")
            setattr(err, "_status_code", resp.status_code)
            setattr(err, "_body_text", body_preview)
            raise err

        data = resp.json()
        cands = data.get("candidates") or []
        if not cands:
            raise VlmError(f"Empty candidates in Gemini response: {str(data)[:300]}")
        parts_out = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(str(p.get("text", "")) for p in parts_out if isinstance(p, dict))
        um = data.get("usageMetadata") or {}
        p_tok = int(um.get("promptTokenCount") or 0)
        c_tok = int(um.get("candidatesTokenCount") or 0)
        r_tok = int(um.get("thoughtsTokenCount") or 0)
        t_tok = int(um.get("totalTokenCount") or (p_tok + c_tok + r_tok))
        return text, {
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "total_tokens": t_tok,
            "reasoning_tokens": r_tok,
        }

    def _call_once_with_http_retries(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        thinking: bool,
        thinking_budget: int | None,
        max_tokens: int,
    ) -> tuple[str, dict[str, int]]:
        last_exc: Exception | None = None
        max_attempts = max(1, self.retries + 1)
        for attempt in range(max_attempts):
            self._check_cancelled()
            try:
                if self.provider == "gemini":
                    return self._post_gemini(
                        model=model,
                        messages=messages,
                        thinking=thinking,
                        thinking_budget=thinking_budget,
                        max_tokens=max_tokens,
                    )
                return self._post_openai_compat(
                    model=model,
                    messages=messages,
                    thinking=thinking,
                    thinking_budget=thinking_budget,
                    max_tokens=max_tokens,
                )
            except VlmCancelledError:
                raise
            except VlmError as exc:
                self._check_cancelled()
                last_exc = exc
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
                last_exc = exc
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
            raise last_exc
        raise VlmError(f"VLM request failed after {max_attempts} attempts: {last_exc}") from last_exc

    def chat_json(
        self,
        *,
        stage: str = "s3_discover",
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        system_prompt: str = "",
        user_prompt: str = "",
        images: Sequence[np.ndarray] | None = None,
        timestamps_s: Sequence[float] | None = None,
        labels: Sequence[str] | None = None,
        thinking: bool = False,
        thinking_budget: int | None = None,
        max_tokens: int = 2048,
        jpeg_quality: int = 74,
        text_first: bool = False,
        budget: BudgetTracker | None = None,
    ) -> dict[str, Any]:
        """Call VLM endpoint and parse a JSON dict, with HTTP + malformed-JSON retries."""
        self._check_cancelled()
        eff_model = self.model_for_stage(stage, override=model)
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
        last_json_err: Exception | None = None
        for _json_attempt in range(2):
            self._check_cancelled()
            tracker.reserve()
            t0 = time.perf_counter()
            try:
                with self._sem:
                    self._check_cancelled()
                    text, usage = self._call_once_with_http_retries(
                        model=eff_model,
                        messages=eff_messages,
                        thinking=thinking,
                        thinking_budget=thinking_budget,
                        max_tokens=max_tokens,
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
                elapsed_s=elapsed,
                cache_hit=False,
            )
            try:
                return parse_json(text)
            except (ValueError, json.JSONDecodeError) as exc:
                last_json_err = exc
                continue

        raise VlmError(f"Model did not return valid JSON after retry: {last_json_err}")

    complete_json = chat_json
    call_stage = chat_json
