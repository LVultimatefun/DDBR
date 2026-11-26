"""Comprehensive offline unit & wire-level tests for Gemini native /v1beta protocol, resilience, and cost."""

from __future__ import annotations

import base64
import json
import logging
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, quote_plus, urlsplit

import numpy as np
import pytest
import yaml

import robo_vlm_qc.vlm as vlm_pkg
import robo_vlm_qc.vlm.client as vlm_client_mod
from robo_vlm_qc.config import ConfigManager, default_config_dir
from robo_vlm_qc.vlm.cache import StageCache
from robo_vlm_qc.vlm.client import (
    DEFAULT_GEMINI_API_KEY_ENV,
    DEFAULT_GEMINI_BASE_URL,
    DEFAULT_GEMINI_KEY_FILE,
    DEFAULT_GEMINI_MEDIA_RESOLUTION,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENAI_COMPAT_API_KEY_ENV,
    DEFAULT_OPENAI_COMPAT_BASE_URL,
    DEFAULT_OPENAI_COMPAT_KEY_FILE,
    DEFAULT_OPENAI_COMPAT_MODEL,
    DEFAULT_VISION_BLIND_NODE_RETRIES,
    DEFAULT_VISION_MIN_TOKENS_PER_IMAGE,
    BudgetExceeded,
    BudgetTracker,
    VlmBlindVisionError,
    VlmCancelledError,
    VlmCircuitBreaker,
    VlmCircuitOpenError,
    VlmClient,
    VlmError,
    build_gemini_generation_config,
    estimate_stage_cost_usd,
    evaluate_vision_token_density,
    extract_json,
    gemini_generate_url,
    is_gemini_hd_vision_model,
    messages_to_gemini_payload,
    normalize_gemini_base_url,
    normalize_gemini_media_resolution,
    parse_gemini_response,
    redact_secrets,
    resolve_client_stage_model,
    resolve_stage_model,
)


def _gem_ok(
    obj: dict[str, Any],
    *,
    prompt_tokens: int = 40,
    completion_tokens: int = 15,
    thoughts_tokens: int = 0,
    model_version: str | None = None,
    prompt_tokens_details: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    um: dict[str, Any] = {
        "promptTokenCount": prompt_tokens,
        "candidatesTokenCount": completion_tokens,
        "thoughtsTokenCount": thoughts_tokens,
        "totalTokenCount": prompt_tokens + completion_tokens + thoughts_tokens,
    }
    if prompt_tokens_details is not None:
        um["promptTokensDetails"] = prompt_tokens_details
    payload: dict[str, Any] = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": json.dumps(obj, ensure_ascii=False)}],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": um,
    }
    if model_version is not None:
        payload["modelVersion"] = model_version
    return payload


def _dummy_images(n: int) -> list[np.ndarray]:
    return [np.full((32, 48, 3), 80 + i * 20, dtype=np.uint8) for i in range(n)]


class _DualMockServer:
    """In-process localhost mock server recording path, query, headers, TCP port, and JSON body."""

    def __init__(self, *, slow_s: float = 0.0) -> None:
        self.slow_s = slow_s
        self.started = threading.Event()
        self.release = threading.Event()
        self.requests: list[dict[str, Any]] = []
        self.queue: list[tuple[int, Any]] = []
        self._lock = threading.Lock()
        srv = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A002
                return

            def do_POST(self) -> None:  # noqa: N802
                n = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(n).decode("utf-8") if n > 0 else "{}"
                body = json.loads(raw)
                parsed = urlsplit(self.path)
                hdrs = {k.lower(): v for k, v in self.headers.items()}
                rec = {
                    "raw_path": self.path,
                    "url_path": parsed.path,
                    "query": parse_qs(parsed.query),
                    "headers": hdrs,
                    "client_port": int(self.client_address[1]),
                    "body": body,
                }
                with srv._lock:
                    srv.requests.append(rec)
                    item = srv.queue.pop(0) if srv.queue else None

                srv.started.set()
                if srv.slow_s > 0:
                    srv.release.wait(timeout=srv.slow_s)

                if item is not None:
                    code, payload = item
                elif ":generateContent" in parsed.path:
                    code = 200
                    payload = _gem_ok(
                        {"ok": True, "via": "gemini"},
                        prompt_tokens=40,
                        completion_tokens=15,
                        thoughts_tokens=10,
                    )
                else:
                    code = 200
                    payload = {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": '{"ok": true, "via": "openai"}',
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 30,
                            "completion_tokens": 12,
                            "total_tokens": 42,
                        },
                    }

                out = (
                    payload.encode("utf-8")
                    if isinstance(payload, str)
                    else json.dumps(payload, ensure_ascii=False).encode("utf-8")
                )
                try:
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(out)))
                    if hdrs.get("connection", "").lower() == "close":
                        self.send_header("Connection", "close")
                        self.close_connection = True
                    self.end_headers()
                    self.wfile.write(out)
                except OSError:
                    pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.port = int(self._httpd.server_address[1])
        self.root_url = f"http://127.0.0.1:{self.port}"
        self.v1beta_url = f"http://127.0.0.1:{self.port}/v1beta"
        self.v1_url = f"http://127.0.0.1:{self.port}/v1"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.release.set()
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2.0)


def test_frozen_symbol_exports() -> None:
    expected = (
        "VlmBlindVisionError",
        "VlmCircuitOpenError",
        "VlmCircuitBreaker",
        "resolve_client_stage_model",
        "resolve_stage_model",
        "estimate_stage_cost_usd",
        "extract_json",
        "redact_secrets",
    )
    for name in expected:
        assert hasattr(vlm_client_mod, name), f"Missing {name} in robo_vlm_qc.vlm.client"
        assert hasattr(vlm_pkg, name), f"Missing {name} in robo_vlm_qc.vlm"
        assert name in vlm_pkg.__all__, f"Missing {name} in robo_vlm_qc.vlm.__all__"
        assert getattr(vlm_pkg, name) is getattr(vlm_client_mod, name)


def test_provider_aware_defaults_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    cfg = ConfigManager()
    assert cfg.vlm_provider() == "openai_compat"

    oai = VlmClient.from_config(cfg)
    try:
        assert oai.provider == "openai_compat"
        assert oai.base_url == DEFAULT_OPENAI_COMPAT_BASE_URL == "https://api.openai.com/v1"
        assert oai.api_key_env == DEFAULT_OPENAI_COMPAT_API_KEY_ENV == "OPENAI_API_KEY"
        assert oai.key_file == DEFAULT_OPENAI_COMPAT_KEY_FILE == ".env"
        assert oai.default_model == DEFAULT_OPENAI_COMPAT_MODEL == "gpt-4o"
        assert oai.thinking_mode == "auto"
        assert oai.circuit_breaker is not None and oai.circuit_breaker.enabled is True
    finally:
        oai.close()

    gem = VlmClient(provider="gemini")
    try:
        assert gem.provider == "gemini"
        assert gem.base_url == DEFAULT_GEMINI_BASE_URL == "https://generativelanguage.googleapis.com/v1beta"
        assert gem.api_key_env == DEFAULT_GEMINI_API_KEY_ENV == "GEMINI_API_KEY"
        assert gem.key_file == DEFAULT_GEMINI_KEY_FILE == ".env"
        assert gem.default_model == DEFAULT_GEMINI_MODEL == "gemini-2.5-pro"
        assert gem.gemini_media_resolution == DEFAULT_GEMINI_MEDIA_RESOLUTION
        assert gem.vision_min_tokens_per_image == DEFAULT_VISION_MIN_TOKENS_PER_IMAGE
        assert gem.vision_blind_node_retries == DEFAULT_VISION_BLIND_NODE_RETRIES
    finally:
        gem.close()

    with pytest.raises(ValueError, match="Unsupported VLM provider"):
        VlmClient({"vlm": {"provider": "unsupported_provider"}})


def test_gemini_url_normalization_and_wire_auth() -> None:
    base_google = "https://generativelanguage.googleapis.com"
    assert normalize_gemini_base_url(base_google) == f"{base_google}/v1beta"
    assert normalize_gemini_base_url(f"{base_google}/") == f"{base_google}/v1beta"
    assert normalize_gemini_base_url(f"{base_google}/v1beta") == f"{base_google}/v1beta"
    assert normalize_gemini_base_url(f"{base_google}/v1beta/") == f"{base_google}/v1beta"
    assert normalize_gemini_base_url(f"{base_google}/v1") == f"{base_google}/v1beta"
    assert normalize_gemini_base_url("http://127.0.0.1:8123/v1/") == "http://127.0.0.1:8123/v1beta"

    assert (
        gemini_generate_url(f"{base_google}/", "gemini-2.5-pro")
        == f"{base_google}/v1beta/models/gemini-2.5-pro:generateContent"
    )
    assert (
        gemini_generate_url(f"{base_google}/v1beta/", "models/gemini-2.5-flash")
        == f"{base_google}/v1beta/models/gemini-2.5-flash:generateContent"
    )
    with pytest.raises(VlmError):
        gemini_generate_url(f"{base_google}/v1beta", "")

    srv = _DualMockServer()
    try:
        for base in (srv.root_url, f"{srv.root_url}/", srv.v1beta_url, f"{srv.v1beta_url}/", srv.v1_url):
            client = VlmClient(
                provider="gemini",
                base_url=base,
                api_key="fake-gemini-key-99",
                stage_models={"s1_grounding": "gemini-2.5-pro"},
            )
            try:
                res = client.call_json(stage="s1_grounding", user_prompt="ping")
                assert res == {"ok": True, "via": "gemini"}
                req = srv.requests[-1]
                assert req["url_path"] == "/v1beta/models/gemini-2.5-pro:generateContent"
                assert req["query"].get("key") == ["fake-gemini-key-99"]
                assert req["headers"].get("x-goog-api-key") == "fake-gemini-key-99"
                assert "authorization" not in req["headers"]
            finally:
                client.close()
    finally:
        srv.stop()


def test_gemini_multimodal_payload_and_generation_config_matrix() -> None:
    sys_inst, contents = messages_to_gemini_payload(
        [
            {"role": "system", "content": "sys line 1"},
            {"role": "developer", "content": "sys line 2"},
            {"role": "user", "content": "turn 1a"},
            {"role": "user", "content": [{"type": "text", "text": "turn 1b"}]},
            {"role": "assistant", "content": "model reply"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "check frame"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ"}},
                    {"inlineData": {"mimeType": "image/png", "data": "iVBORw=="}},
                    {"inline_data": {"mime_type": "image/webp", "data": "UklGRg=="}},
                ],
            },
        ]
    )
    assert sys_inst == {"parts": [{"text": "sys line 1"}, {"text": "sys line 2"}]}
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert contents[0]["parts"] == [{"text": "turn 1a"}, {"text": "turn 1b"}]
    assert contents[1]["parts"] == [{"text": "model reply"}]
    assert contents[2]["parts"][1] == {"inlineData": {"mimeType": "image/jpeg", "data": "/9j/4AAQ"}}
    assert contents[2]["parts"][2] == {"inlineData": {"mimeType": "image/png", "data": "iVBORw=="}}
    assert contents[2]["parts"][3] == {"inlineData": {"mimeType": "image/webp", "data": "UklGRg=="}}

    with pytest.raises(VlmError, match="Unsupported image_url"):
        messages_to_gemini_payload(
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}}]}]
        )

    # Gemini 3 family maps thinking_budget -> thinkingLevel + max(budget, level_headroom)
    c_low = build_gemini_generation_config(
        model="gemini-3-pro-preview",
        thinking=True,
        thinking_budget=768,
        max_tokens=1000,
        temperature=0.1,
        top_p=0.9,
        json_mode=True,
    )
    assert c_low == {
        "maxOutputTokens": 2024,
        "temperature": 0.1,
        "topP": 0.9,
        "responseMimeType": "application/json",
        "thinkingConfig": {"thinkingLevel": "low"},
    }

    c_med = build_gemini_generation_config(
        model="gemini-3-pro",
        thinking=True,
        thinking_budget=1280,
        max_tokens=1536,
    )
    assert c_med["thinkingConfig"] == {"thinkingLevel": "medium"}
    assert c_med["maxOutputTokens"] == 1536 + 2048

    c_high = build_gemini_generation_config(
        model="gemini-3-pro",
        thinking=True,
        thinking_budget=6000,
        max_tokens=2000,
    )
    assert c_high["thinkingConfig"] == {"thinkingLevel": "high"}
    assert c_high["maxOutputTokens"] == 2000 + 6000

    # Legacy Gemini 2.5 uses integer thinkingBudget
    c_25 = build_gemini_generation_config(
        model="gemini-2.5-pro",
        thinking=True,
        thinking_budget=512,
        max_tokens=1024,
    )
    assert c_25["thinkingConfig"] == {"thinkingBudget": 512}
    assert c_25["maxOutputTokens"] == 1536

    # thinking=False omits thinkingConfig and adds no headroom
    c_off = build_gemini_generation_config(
        model="gemini-3-pro",
        thinking=False,
        thinking_budget=1536,
        max_tokens=600,
        json_mode=False,
    )
    assert "thinkingConfig" not in c_off
    assert "responseMimeType" not in c_off
    assert c_off["maxOutputTokens"] == 600

    # Wire-level multimodal call + thinking_mode="disabled" + thinking_level inheritance guard
    srv = _DualMockServer()
    try:
        client = VlmClient(
            {
                "vlm": {
                    "provider": "gemini",
                    "base_url": srv.v1beta_url,
                    "thinking_level": "medium",
                    "stage_models": {"s1_grounding": "gemini-3-pro-preview", "s3_discover": "gemini-3-pro-preview"},
                    "s3_discover": {"thinking_level": "high"},
                }
            },
            api_key="fake-key",
            retries=0,
        )
        try:
            img = np.full((24, 32, 3), 120, dtype=np.uint8)
            res = client.chat_json(
                stage="s1_grounding",
                system_prompt="Vision inspector.",
                user_prompt="Return JSON.",
                images=[img],
                timestamps_s=[1.25],
                labels=["main"],
                thinking=True,
                thinking_budget=768,
                max_tokens=512,
                temperature=0.2,
                top_p=0.85,
            )
            assert res["ok"] is True
            body = srv.requests[-1]["body"]
            assert body["systemInstruction"] == {"parts": [{"text": "Vision inspector."}]}
            parts = body["contents"][0]["parts"]
            assert any("[t=01.25s | main]" in p.get("text", "") for p in parts)
            inline_parts = [p["inlineData"] for p in parts if "inlineData" in p]
            assert len(inline_parts) == 1
            assert base64.b64decode(inline_parts[0]["data"])[:2] == b"\xff\xd8"

            # thinking=False must NOT inherit client/stage thinking_level
            client.call_json(stage="s3_discover", user_prompt="no thinking", thinking=False, max_tokens=500)
            gc_off = srv.requests[-1]["body"]["generationConfig"]
            assert "thinkingConfig" not in gc_off
            assert gc_off["maxOutputTokens"] == 500

            # thinking=True inherits stage thinking_level="high"
            client.call_json(stage="s3_discover", user_prompt="stage thinking", thinking=True, max_tokens=500)
            gc_on = srv.requests[-1]["body"]["generationConfig"]
            assert gc_on["thinkingConfig"] == {"thinkingLevel": "high"}
            assert gc_on["maxOutputTokens"] == 500 + 4096

            # Explicit caller thinking_level="low" works even with thinking=False
            client.call_json(
                stage="s3_discover",
                user_prompt="explicit low",
                thinking=False,
                thinking_level="low",
                max_tokens=500,
            )
            gc_exp = srv.requests[-1]["body"]["generationConfig"]
            assert gc_exp["thinkingConfig"] == {"thinkingLevel": "low"}
            assert gc_exp["maxOutputTokens"] == 500 + 1024
        finally:
            client.close()

        # thinking_mode="disabled" suppresses Gemini thinking even when thinking=True
        disabled_client = VlmClient(
            provider="gemini",
            base_url=srv.v1beta_url,
            api_key="fake-key",
            thinking_mode="disabled",
            stage_models={"s2_macro": "gemini-2.5-pro"},
        )
        try:
            disabled_client.chat_json(
                stage="s2_macro",
                user_prompt="no thinking allowed",
                thinking=True,
                thinking_budget=1024,
                max_tokens=400,
            )
            gc_dis = srv.requests[-1]["body"]["generationConfig"]
            assert "thinkingConfig" not in gc_dis
            assert gc_dis["maxOutputTokens"] == 400
        finally:
            disabled_client.close()
    finally:
        srv.stop()


def test_gemini_response_parsing_retries_and_503_fast_fail() -> None:
    answer, usage, meta = parse_gemini_response(
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"thought": True, "text": '{"wrong": "decoy", broken'},
                            {"text": '```json\n{"verdict": "success", "score": 0.95}\n```'},
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 120,
                "candidatesTokenCount": 45,
                "thoughtsTokenCount": 80,
                "totalTokenCount": 245,
                "promptTokensDetails": [
                    {"modality": "IMAGE", "tokenCount": 100},
                    {"modality": "TEXT", "tokenCount": 20},
                ],
            },
        }
    )
    assert extract_json(answer) == {"verdict": "success", "score": 0.95}
    assert meta["finish_reason"] == "STOP"
    assert "decoy" in meta["thought_text"]
    assert meta["image_prompt_tokens"] == 100
    assert meta["text_prompt_tokens"] == 20
    assert meta["reasoning_included_in_completion"] is False
    assert usage == {
        "prompt_tokens": 120,
        "completion_tokens": 45,
        "total_tokens": 245,
        "reasoning_tokens": 80,
    }

    with pytest.raises(VlmError, match="blocked"):
        parse_gemini_response({"candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}]})

    srv = _DualMockServer()
    try:
        budget = BudgetTracker(max_tokens=10000)
        client = VlmClient(
            provider="gemini",
            base_url=srv.v1beta_url,
            api_key="fake-key",
            retries=3,
            retry_min_wait_s=0.01,
            retry_max_wait_s=0.02,
            budget=budget,
        )
        try:
            srv.queue.extend(
                [
                    (429, {"error": {"message": "rate limited"}}),
                    (500, {"error": {"message": "upstream transient"}}),
                    (
                        200,
                        {
                            "candidates": [{"content": {"parts": [{"text": "{malformed"}]}, "finishReason": "STOP"}],
                            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 5, "totalTokenCount": 10},
                        },
                    ),
                    (
                        200,
                        {
                            "candidates": [
                                {
                                    "content": {
                                        "parts": [
                                            {"thought": True, "text": '{"decoy": true}'},
                                            {"text": '{"final": 42}'},
                                        ]
                                    },
                                    "finishReason": "STOP",
                                }
                            ],
                            "usageMetadata": {
                                "promptTokenCount": 100,
                                "candidatesTokenCount": 25,
                                "thoughtsTokenCount": 50,
                                "totalTokenCount": 175,
                            },
                        },
                    ),
                ]
            )
            res = client.call_json(stage="s3_discover", user_prompt="test retries")
            assert res == {"final": 42}
            assert len(srv.requests) == 4
            assert budget.prompt_tokens == 105
            assert budget.completion_tokens == 30
            assert budget.reasoning_tokens == 50
            assert budget.tokens == 185

            # Non-retryable 503 model_not_found must fast-fail without retrying
            srv.queue.append(
                (
                    503,
                    {
                        "error": {
                            "code": "model_not_found",
                            "message": "No available channel for model gemini-2.5-pro",
                        }
                    },
                )
            )
            before = len(srv.requests)
            with pytest.raises(VlmError, match="model_not_found"):
                client.call_json(stage="s3_discover", user_prompt="fast fail 503")
            assert len(srv.requests) - before == 1
        finally:
            client.close()
    finally:
        srv.stop()


def test_media_resolution_and_blind_node_guard_recovery_and_exhaustion(
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert normalize_gemini_media_resolution("high") == "MEDIA_RESOLUTION_HIGH"
    assert normalize_gemini_media_resolution("MEDIA_RESOLUTION_MEDIUM") == "MEDIA_RESOLUTION_MEDIUM"
    assert normalize_gemini_media_resolution("low") == "MEDIA_RESOLUTION_LOW"
    assert normalize_gemini_media_resolution(True) == DEFAULT_GEMINI_MEDIA_RESOLUTION
    assert normalize_gemini_media_resolution(False) is None
    assert normalize_gemini_media_resolution("off") is None

    assert is_gemini_hd_vision_model("gemini-3-pro-preview") is True
    assert is_gemini_hd_vision_model("models/gemini-3-pro") is True
    assert is_gemini_hd_vision_model("gemini-2.5-pro") is False
    assert is_gemini_hd_vision_model("gpt-4o") is False

    is_blind, tpi, eff_img, src = evaluate_vision_token_density(
        n_images=4,
        prompt_tokens=11032,
        exact_image_tokens=4 * 258,
        est_text_tokens=500,
        min_tokens_per_image=550,
    )
    assert is_blind is True
    assert eff_img == 1032
    assert tpi == pytest.approx(258.0)
    assert src == "promptTokensDetails.IMAGE"

    srv = _DualMockServer()
    try:
        # 1. Media resolution injected ONLY for multimodal Gemini 3+ (not text-only or Gemini 2.5)
        mr_client = VlmClient(
            {
                "vlm": {
                    "provider": "gemini",
                    "base_url": srv.v1beta_url,
                    "gemini_media_resolution": "MEDIA_RESOLUTION_HIGH",
                    "stage_models": {
                        "s1_grounding": "gemini-3-pro-preview",
                        "s2_macro": "gemini-2.5-flash",
                    },
                }
            },
            api_key="fake-mr-key",
            retries=0,
        )
        try:
            mr_client.chat_json(stage="s1_grounding", user_prompt="locate", images=_dummy_images(2))
            assert srv.requests[-1]["body"]["generationConfig"].get("mediaResolution") == "MEDIA_RESOLUTION_HIGH"

            mr_client.chat_json(stage="s1_grounding", user_prompt="text only")
            assert "mediaResolution" not in srv.requests[-1]["body"]["generationConfig"]

            mr_client.chat_json(stage="s2_macro", user_prompt="legacy", images=_dummy_images(2))
            assert "mediaResolution" not in srv.requests[-1]["body"]["generationConfig"]
        finally:
            mr_client.close()

        # 2. Blind-node retries ~258 tok/img with fresh TCP connections until HD ~1120 recovers
        srv.requests.clear()
        srv.queue.extend(
            [
                (200, _gem_ok({"source": "blind_1"}, prompt_tokens=4 * 258 + 50, completion_tokens=110)),
                (200, _gem_ok({"source": "blind_2"}, prompt_tokens=4 * 258 + 50, completion_tokens=120)),
                (
                    200,
                    _gem_ok(
                        {"source": "real_hd"},
                        prompt_tokens=4 * 1120 + 50,
                        completion_tokens=54,
                        model_version="gemini-3-pro-preview",
                        prompt_tokens_details=[
                            {"modality": "IMAGE", "tokenCount": 4400},
                            {"modality": "TEXT", "tokenCount": 130},
                        ],
                    ),
                ),
            ]
        )
        budget = BudgetTracker(max_tokens=100_000)
        guard_client = VlmClient(
            provider="gemini",
            base_url=srv.v1beta_url,
            api_key="fake-blind-secret-key",
            retries=0,
            retry_min_wait_s=0.005,
            retry_max_wait_s=0.015,
            stage_models={"s1_grounding": "gemini-3-pro-preview"},
            gemini_media_resolution="MEDIA_RESOLUTION_HIGH",
            vision_min_tokens_per_image=550,
            vision_blind_node_retries=4,
            vision_blind_node_guard=True,
            budget=budget,
        )
        try:
            with caplog.at_level(logging.INFO, logger="robo_vlm_qc.vlm.client"):
                res = guard_client.chat_json(
                    stage="s1_grounding",
                    system_prompt="Locate objects.",
                    user_prompt="Return bboxes.",
                    images=_dummy_images(4),
                    timestamps_s=[0.0, 2.5, 5.0, 7.5],
                )
            assert res == {"source": "real_hd"}
            assert len(srv.requests) == 3
            assert srv.requests[0]["headers"].get("connection") == "keep-alive"
            assert srv.requests[1]["headers"].get("connection") == "close"
            assert srv.requests[2]["headers"].get("connection") == "close"
            assert srv.requests[1]["client_port"] != srv.requests[0]["client_port"]
            assert srv.requests[2]["client_port"] != srv.requests[1]["client_port"]
            assert budget.calls == 3
            assert budget._reserved == 0
            for rec in caplog.records:
                assert "fake-blind-secret-key" not in rec.getMessage()
        finally:
            guard_client.close()

        # 3. Exhaustion raises VlmBlindVisionError (default) or returns payload when on_exhaust="warn"
        for _ in range(3):
            srv.queue.append((200, _gem_ok({"verdict": "wrong"}, prompt_tokens=4 * 258 + 20)))
        raise_client = VlmClient(
            provider="gemini",
            base_url=srv.v1beta_url,
            api_key="sk-SUPER-SECRET-BLIND-EXHAUST-999",
            retries=0,
            retry_min_wait_s=0.005,
            retry_max_wait_s=0.01,
            stage_models={"s5_success": "gemini-3-pro-preview"},
            vision_blind_node_retries=2,
            vision_blind_node_guard=True,
        )
        try:
            with pytest.raises(VlmBlindVisionError) as exc_info:
                raise_client.chat_json(stage="s5_success", user_prompt="judge", images=_dummy_images(4))
            assert isinstance(exc_info.value, VlmError)
            assert "sk-SUPER-SECRET-BLIND-EXHAUST-999" not in str(exc_info.value)
        finally:
            raise_client.close()

        srv.queue.extend(
            [
                (200, _gem_ok({"verdict": "degraded_1"}, prompt_tokens=2 * 258 + 20)),
                (200, _gem_ok({"verdict": "degraded_2"}, prompt_tokens=2 * 258 + 20)),
            ]
        )
        warn_client = VlmClient(
            provider="gemini",
            base_url=srv.v1beta_url,
            api_key="fake-key",
            retries=0,
            retry_min_wait_s=0.005,
            retry_max_wait_s=0.01,
            stage_models={"s5_success": "gemini-3-pro-preview"},
            vision_blind_node_retries=1,
            vision_blind_node_guard=True,
            vision_blind_node_on_exhaust="warn",
        )
        try:
            assert warn_client.chat_json(stage="s5_success", user_prompt="judge", images=_dummy_images(2)) == {
                "verdict": "degraded_2"
            }
        finally:
            warn_client.close()

        # 4. Loopback auto mode auto-skips guard even when mock returns low promptTokenCount
        auto_client = VlmClient(
            provider="gemini",
            base_url=srv.v1beta_url,
            api_key="fake-key",
            retries=0,
            stage_models={"s1_grounding": "gemini-3-pro-preview"},
        )
        try:
            before = len(srv.requests)
            res_auto = auto_client.chat_json(stage="s1_grounding", user_prompt="mock", images=_dummy_images(4))
            assert res_auto["ok"] is True
            assert len(srv.requests) - before == 1
        finally:
            auto_client.close()
    finally:
        srv.stop()


def test_circuit_breaker_and_opt_in_fallback() -> None:
    cb = VlmCircuitBreaker(failure_threshold=3, trip_on_quota_403=True, enabled=True)
    cb.check()
    assert cb.record_failure(VlmError("HTTP 500 temp 1")) is False
    assert cb.record_failure(VlmError("HTTP 500 temp 2")) is False
    assert cb.record_failure(VlmCancelledError("cancel")) is False
    assert cb.record_failure(BudgetExceeded("budget")) is False
    cb.record_success()
    assert cb.snapshot()["consecutive_failures"] == 0
    assert cb.record_failure(VlmError("fail 1")) is False
    assert cb.record_failure(VlmError("fail 2")) is False
    assert cb.record_failure(VlmError("fail 3")) is True
    assert cb.is_open is True
    with pytest.raises(VlmCircuitOpenError):
        cb.check()

    cb_q = VlmCircuitBreaker(failure_threshold=5, trip_on_quota_403=True, enabled=True)
    err_q = VlmError("HTTP 403 from Gemini API: insufficient_quota")
    setattr(err_q, "_status_code", 403)
    assert cb_q.record_failure(err_q) is True
    assert cb_q.is_open is True

    srv = _DualMockServer()
    try:
        # Wire-level circuit trip on 403 blocks subsequent call before hitting server
        breaker = VlmCircuitBreaker(failure_threshold=3, trip_on_quota_403=True)
        c1 = VlmClient(
            provider="gemini",
            base_url=srv.v1beta_url,
            api_key="k1",
            retries=0,
            circuit_breaker=breaker,
        )
        try:
            srv.queue.append((403, {"error": {"code": "insufficient_quota"}}))
            with pytest.raises(VlmCircuitOpenError):
                c1.chat_json(stage="s3_discover", user_prompt="call 1")
            reqs_after = len(srv.requests)
            assert reqs_after == 1
            with pytest.raises(VlmCircuitOpenError):
                c1.chat_json(stage="s3_discover", user_prompt="call 2 blocked")
            assert len(srv.requests) == reqs_after
        finally:
            c1.close()

        # Opt-in cross-provider fallback (gemini -> openai_compat) succeeds once
        c_fb = VlmClient(
            provider="gemini",
            base_url=srv.v1beta_url,
            api_key="k-gem",
            retries=0,
            fallback_model="gpt-4o",
            fallback_provider="openai_compat",
            fallback_base_url=srv.v1_url,
            fallback_api_key="k-fallback",
            circuit_breaker=VlmCircuitBreaker(failure_threshold=2, trip_on_quota_403=True),
        )
        try:
            srv.queue.append((403, {"error": {"message": "quota_exceeded"}}))
            before = len(srv.requests)
            out = c_fb.chat_json(stage="s5_success", user_prompt="fallback test")
            assert out == {"ok": True, "via": "openai"}
            assert len(srv.requests) - before == 2
            assert srv.requests[-1]["url_path"] == "/v1/chat/completions"
            assert srv.requests[-1]["headers"].get("authorization") == "Bearer k-fallback"
            assert srv.requests[-1]["body"]["model"] == "gpt-4o"
        finally:
            c_fb.close()
    finally:
        srv.stop()


def test_secret_redaction_in_errors_and_httpx_logs() -> None:
    secret = "sk-UNIT-SECRET+val/999=xyz"
    aiza_secret = "AIzaSyD1234567890abcdefghijklmnop"
    enc_secret = quote(secret, safe="")
    enc_plus = quote_plus(secret)

    sample = (
        f"POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={enc_secret} "
        f"x-goog-api-key: {secret} Authorization: Bearer {secret} raw={secret} plus={enc_plus} {aiza_secret}"
    )
    redacted = redact_secrets(sample, secret)
    assert secret not in redacted
    assert enc_secret not in redacted
    assert enc_plus not in redacted
    assert aiza_secret not in redacted
    assert "[REDACTED]" in redacted

    # Capture httpx/httpcore INFO & DEBUG logs and verify zero secret leakage
    gem_secret = "sk-LOG-LEAK-GEM+test/777=end"
    oai_secret = "sk-LOG-LEAK-OAI+test/888=end"
    enc_gem = quote(gem_secret, safe="")
    enc_gem_plus = quote_plus(gem_secret)
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture()
    target_names = ("httpx", "httpcore", "httpcore.connection", "httpcore.http11", "httpcore.http2")
    saved: list[tuple[logging.Logger, int, bool]] = []
    for name in target_names:
        lg = logging.getLogger(name)
        saved.append((lg, lg.level, lg.propagate))
        lg.setLevel(logging.DEBUG)
        lg.addHandler(handler)

    srv = _DualMockServer()
    try:
        gem = VlmClient(provider="gemini", base_url=srv.v1beta_url, api_key=gem_secret, retries=0)
        oai = VlmClient(provider="openai_compat", base_url=srv.v1_url, api_key=oai_secret, retries=0)
        try:
            assert gem.call_json(stage="s1_grounding", user_prompt="log check")["via"] == "gemini"
            assert oai.call_json(stage="s1_grounding", user_prompt="log check")["via"] == "openai"

            # HTTP 403 echoing secret must redact it from raised VlmError
            srv.queue.append((403, f"Blocked ?key={enc_gem} header x-goog-api-key={gem_secret}"))
            with pytest.raises(VlmError) as exc_403:
                gem.call_json(stage="s1_grounding", user_prompt="err403")
            assert gem_secret not in str(exc_403.value)
            assert enc_gem not in str(exc_403.value)
            assert "[REDACTED]" in str(exc_403.value)
        finally:
            gem.close()
            oai.close()
    finally:
        srv.stop()
        for lg, prev_level, prev_prop in saved:
            lg.removeHandler(handler)
            lg.setLevel(prev_level)
            lg.propagate = prev_prop

    assert len(records) >= 2
    for rec in records:
        combined = f"{rec.getMessage()} {rec.args!r}"
        assert gem_secret not in combined
        assert enc_gem not in combined
        assert enc_gem_plus not in combined
        assert oai_secret not in combined

    # Transport error against closed port must redact ?key= and drop __cause__
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        unused_port = sock.getsockname()[1]

    dead = VlmClient(
        provider="gemini",
        base_url=f"http://127.0.0.1:{unused_port}/v1beta",
        api_key=secret,
        retries=0,
        timeout_s=2.0,
    )
    try:
        with pytest.raises(VlmError) as exc_conn:
            dead.call_json(stage="s1_grounding", user_prompt="conn error")
        assert secret not in str(exc_conn.value)
        assert enc_secret not in str(exc_conn.value)
        assert exc_conn.value.__cause__ is None
    finally:
        dead.close()


def test_gemini_cooperative_cancellation() -> None:
    srv = _DualMockServer(slow_s=5.0)
    try:
        client = VlmClient(
            provider="gemini",
            base_url=srv.v1beta_url,
            api_key="fake-cancel-key",
            retries=3,
            retry_min_wait_s=0.5,
            timeout_s=10.0,
        )
        raised: list[BaseException] = []

        def _worker() -> None:
            try:
                client.call_json(stage="s2_macro", user_prompt="slow gemini call")
            except BaseException as exc:  # noqa: BLE001
                raised.append(exc)

        th = threading.Thread(target=_worker)
        th.start()
        assert srv.started.wait(timeout=3.0)

        t0 = time.monotonic()
        client.cancel()
        th.join(timeout=2.0)
        elapsed = time.monotonic() - t0

        assert not th.is_alive()
        assert len(raised) == 1 and isinstance(raised[0], VlmCancelledError)
        assert elapsed < 1.0
        assert len(srv.requests) == 1
    finally:
        srv.stop()


def test_cost_accounting_no_double_count_and_mixed_model_budget() -> None:
    cfg = ConfigManager()
    pricing = cfg.settings["vlm"]["pricing"]

    # Gemini: candidatesTokenCount + thoughtsTokenCount both billed as output
    # gemini-2.5-pro: in=1.25/1M, out=10.0/1M -> 0.1*1.25 + (0.01+0.05)*10.0 = 0.125 + 0.600 = 0.725
    assert estimate_stage_cost_usd(
        "gemini-2.5-pro",
        100_000,
        10_000,
        pricing,
        reasoning_tokens=50_000,
        reasoning_included_in_completion=False,
    ) == pytest.approx(0.725, abs=1e-6)

    # OpenAI-compat: completion_tokens already includes reasoning_tokens -> never double-count
    # gpt-4o: in=2.50/1M, out=10.0/1M -> 0.1*2.50 + 0.06*10.0 = 0.25 + 0.60 = 0.85
    assert estimate_stage_cost_usd(
        "gpt-4o",
        100_000,
        60_000,
        pricing,
        reasoning_tokens=50_000,
        reasoning_included_in_completion=True,
    ) == pytest.approx(0.85, abs=1e-6)

    srv = _DualMockServer()
    try:
        srv.queue.extend(
            [
                (
                    200,
                    _gem_ok(
                        {"ok": True},
                        prompt_tokens=100_000,
                        completion_tokens=10_000,
                        thoughts_tokens=50_000,
                    ),
                ),
                (
                    200,
                    _gem_ok(
                        {"ok": True},
                        prompt_tokens=100_000,
                        completion_tokens=10_000,
                        thoughts_tokens=50_000,
                    ),
                ),
                (
                    200,
                    {
                        "choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}],
                        "usage": {
                            "prompt_tokens": 100_000,
                            "completion_tokens": 60_000,
                            "total_tokens": 160_000,
                            "completion_tokens_details": {"reasoning_tokens": 50_000},
                        },
                    },
                ),
            ]
        )

        tracker = BudgetTracker(pricing=pricing)
        gem = VlmClient(
            provider="gemini",
            base_url=srv.v1beta_url,
            api_key="test-key",
            stage_models={"s5_success": "gemini-2.5-pro"},
            settings={"pricing": pricing},
        )
        try:
            gem.chat_json(stage="s5_success", model_override="gemini-2.5-flash", user_prompt="fast", budget=tracker)
            gem.chat_json(stage="s5_success", user_prompt="strong", budget=tracker)
        finally:
            gem.close()

        summary = tracker.to_usage_summary()
        assert len(summary.stages) == 2
        by_model = {s.model: s for s in summary.stages}
        # gemini-2.5-flash: 0.1*0.30 + 0.06*2.50 = 0.03 + 0.15 = 0.18
        assert by_model["gemini-2.5-flash"].estimated_cost_usd == pytest.approx(0.18, abs=1e-5)
        # gemini-2.5-pro: 0.1*1.25 + 0.06*10.0 = 0.125 + 0.60 = 0.725
        assert by_model["gemini-2.5-pro"].estimated_cost_usd == pytest.approx(0.725, abs=1e-5)
        assert summary.total_cost_usd == pytest.approx(0.905, abs=1e-5)

        oai_tracker = BudgetTracker(pricing=pricing)
        oai = VlmClient(
            provider="openai_compat",
            base_url=srv.v1_url,
            api_key="test-key",
            stage_models={"s5_success": "gpt-4o"},
            settings={"pricing": pricing},
        )
        try:
            oai.chat_json(stage="s5_success", user_prompt="oai", budget=oai_tracker)
        finally:
            oai.close()
        assert oai_tracker.to_usage_summary().total_cost_usd == pytest.approx(0.85, abs=1e-5)
    finally:
        srv.stop()


def test_stage_cache_isolates_openai_compat_vs_gemini(tmp_path: Path) -> None:
    ws_oai = tmp_path / "ws_oai"
    ws_gem = tmp_path / "ws_gem"
    for ws in (ws_oai, ws_gem):
        (ws / "configs_override").mkdir(parents=True, exist_ok=True)

    (ws_gem / "configs_override" / "default.yaml").write_text(
        yaml.safe_dump({"vlm": {"provider": "gemini"}}),
        encoding="utf-8",
    )
    cfg_oai = ConfigManager(config_dir=default_config_dir(), workspace_dir=ws_oai)
    cfg_gem = ConfigManager(config_dir=default_config_dir(), workspace_dir=ws_gem)
    assert cfg_oai.vlm_provider() == "openai_compat"
    assert cfg_gem.vlm_provider() == "gemini"
    assert cfg_oai.fingerprints.model_routing_sha256 != cfg_gem.fingerprints.model_routing_sha256

    cache = StageCache(workspace=tmp_path / "shared_cache")
    for fn_name in (
        "s1_cache_basis",
        "s2_cache_basis",
        "s3_cache_basis",
        "s4_cache_basis",
        "s5_cache_basis",
    ):
        k_oai = StageCache.compute_key(
            episode_fingerprint="ep1", stage_basis=getattr(cfg_oai, fn_name)("pick_and_place")
        )
        k_gem = StageCache.compute_key(
            episode_fingerprint="ep1", stage_basis=getattr(cfg_gem, fn_name)("pick_and_place")
        )
        assert k_oai != k_gem
        cache.put(k_oai, {"provider": "openai_compat"})
        cache.put(k_gem, {"provider": "gemini"})
        assert cache.get(k_oai) == {"provider": "openai_compat"}
        assert cache.get(k_gem) == {"provider": "gemini"}
