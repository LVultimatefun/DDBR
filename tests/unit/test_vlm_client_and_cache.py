"""Unit and integration tests for VlmClient, BudgetTracker, key resolution, and StageCache."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robo_vlm_qc.config import ConfigManager
from robo_vlm_qc.vlm.cache import StageCache
from robo_vlm_qc.vlm.client import (
    BudgetExceeded,
    BudgetTracker,
    VlmClient,
    VlmError,
    parse_json,
    resolve_api_key,
)


class _LocalMockServer:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.queue: list[tuple[int, Any]] = []
        self._lock = threading.Lock()
        srv = self

        class _H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A002
                return

            def do_POST(self) -> None:  # noqa: N802
                n = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(n).decode("utf-8")
                body = json.loads(raw)
                with srv._lock:
                    srv.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                    item = srv.queue.pop(0) if srv.queue else None
                if item is not None:
                    code, payload = item
                else:
                    code = 200
                    if "generateContent" in self.path:
                        payload = {
                            "candidates": [{"content": {"parts": [{"text": '{"ok": true, "provider": "gemini"}'}]}}],
                            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "thoughtsTokenCount": 3, "totalTokenCount": 18},
                        }
                    else:
                        payload = {
                            "choices": [{"message": {"content": '{"ok": true, "provider": "openai"}'}}],
                            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30, "completion_tokens_details": {"reasoning_tokens": 4}},
                        }
                out = json.dumps(payload).encode("utf-8") if not isinstance(payload, str) else payload.encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _H)
        self.base_url = f"http://127.0.0.1:{self._httpd.server_address[1]}/v1"
        self._t = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._t.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._t.join(timeout=2.0)


def test_parse_json_variants_and_errors() -> None:
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('```json\n{"b": "hello"}\n```') == {"b": "hello"}
    assert parse_json('Thinking...\n{"c": [1, 2]}\nDone.') == {"c": [1, 2]}
    with pytest.raises(ValueError):
        parse_json("")
    with pytest.raises(ValueError):
        parse_json("[1, 2, 3]")
    with pytest.raises(ValueError):
        parse_json("no braces here")


def test_vlm_client_defaults_and_config() -> None:
    client = VlmClient(api_key="dummy")
    try:
        assert client.base_url == "https://api.openai.com/v1"
        assert client.api_key_env == "OPENAI_API_KEY"
        assert client.key_file == ".env"
        assert client.default_model == "gpt-4o"
        assert client.thinking_mode == "auto"
    finally:
        client.close()

    cfg = ConfigManager()
    vlm_cfg = cfg.settings["vlm"]
    assert vlm_cfg["api_key_env"] == "OPENAI_API_KEY"
    assert vlm_cfg["key_file"] == ".env"
    assert vlm_cfg["stage_models"]["s1_grounding"] == "gpt-4o"
    assert vlm_cfg.get("thinking_mode", "auto") == "auto"


def test_resolve_api_key_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_VLM_API_KEY", raising=False)
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "k.env").write_text("TEST_VLM_API_KEY=ws-secret-val\n", encoding="utf-8")
    assert resolve_api_key("TEST_VLM_API_KEY", "k.env", ws) == "ws-secret-val"

    monkeypatch.setenv("TEST_VLM_API_KEY", "env-secret-val")
    assert resolve_api_key("TEST_VLM_API_KEY", "k.env", ws) == "env-secret-val"

    monkeypatch.delenv("TEST_VLM_API_KEY", raising=False)
    with pytest.raises(VlmError) as exc_info:
        resolve_api_key("TEST_VLM_API_KEY", "missing.env", ws)
    assert "TEST_VLM_API_KEY" in str(exc_info.value)


def test_vlm_client_thinking_modes_and_retries(tmp_path: Path) -> None:
    srv = _LocalMockServer()
    try:
        # 1. DashScope thinking_mode="dashscope": sends enable_thinking/thinking_budget and omits response_format
        ds_client = VlmClient(
            base_url=srv.base_url,
            api_key="test-key",
            retries=3,
            retry_min_wait_s=0.01,
            retry_max_wait_s=0.02,
            thinking_mode="dashscope",
            stage_models={"s1_grounding": "qwen-vl-max", "s3_discover": "qwen-vl-max"},
        )
        try:
            res1 = ds_client.chat_json(
                stage="s1_grounding",
                system_prompt="sys",
                user_prompt="usr",
                images=[np.zeros((32, 32, 3), dtype=np.uint8)],
                timestamps_s=[1.5],
                thinking=True,
                thinking_budget=768,
                max_tokens=800,
            )
            assert res1["ok"] is True
            body1 = srv.requests[-1]["body"]
            assert body1["model"] == "qwen-vl-max"
            assert body1["max_tokens"] == 1568
            assert body1["enable_thinking"] is True
            assert body1["thinking_budget"] == 768
            assert "response_format" not in body1
            # Timestamp tag uses ASCII [t=01.50s] format
            user_blocks = body1["messages"][1]["content"]
            text_blocks = [b["text"] for b in user_blocks if b.get("type") == "text"]
            assert any("[t=01.50s" in t for t in text_blocks)

            # Thinking=False => response_format json_object included
            ds_client.chat_json(
                stage="s3_discover",
                system_prompt="sys",
                user_prompt="usr",
                thinking=False,
                max_tokens=500,
            )
            body2 = srv.requests[-1]["body"]
            assert body2["model"] == "qwen-vl-max"
            assert body2["max_tokens"] == 500
            assert body2["response_format"] == {"type": "json_object"}
            assert "enable_thinking" not in body2
            assert "thinking_budget" not in body2

            # Transient 408 + 429 + 400 timed out + malformed JSON retry
            srv.queue.extend(
                [
                    (408, {"error": "timeout"}),
                    (429, {"error": "rate limit"}),
                    (400, {"error": {"message": "Upstream request timed out"}}),
                    (200, {"choices": [{"message": {"content": "{broken json"}}], "usage": {"total_tokens": 5}}),
                ]
            )
            before = len(srv.requests)
            res3 = ds_client.chat_json(stage="s3_discover", user_prompt="retry test")
            assert res3["ok"] is True
            assert len(srv.requests) - before == 5

            # Non-transient 401 fails immediately without retrying
            srv.queue.append((401, {"error": "unauthorized"}))
            before401 = len(srv.requests)
            with pytest.raises(VlmError):
                ds_client.chat_json(stage="s3_discover", user_prompt="auth fail")
            assert len(srv.requests) - before401 == 1
        finally:
            ds_client.close()

        # 2. Standard OpenAI-compatible endpoint (thinking_mode="auto", non-DashScope URL):
        #    even when thinking=True, omits DashScope-only fields and keeps response_format
        openai_client = VlmClient(
            base_url=srv.base_url,
            api_key="test-key",
            thinking_mode="auto",
            stage_models={"s1_grounding": "gpt-4o"},
        )
        try:
            res_oai = openai_client.chat_json(
                stage="s1_grounding",
                system_prompt="sys",
                user_prompt="usr",
                thinking=True,
                thinking_budget=768,
                max_tokens=800,
            )
            assert res_oai["ok"] is True
            body_oai = srv.requests[-1]["body"]
            assert body_oai["model"] == "gpt-4o"
            assert body_oai["max_tokens"] == 800
            assert "enable_thinking" not in body_oai
            assert "thinking_budget" not in body_oai
            assert body_oai["response_format"] == {"type": "json_object"}
        finally:
            openai_client.close()
    finally:
        srv.stop()


def test_vlm_client_gemini_provider() -> None:
    srv = _LocalMockServer()
    try:
        client = VlmClient(
            provider="gemini",
            base_url=srv.base_url,
            api_key="gem-key",
            stage_models={"s2_macro": "gemini-2.5-pro"},
        )
        try:
            res = client.chat_json(
                stage="s2_macro",
                system_prompt="sys",
                user_prompt="usr",
                images=[np.zeros((16, 16, 3), dtype=np.uint8)],
                thinking=True,
                thinking_budget=512,
                max_tokens=1024,
            )
            assert res["provider"] == "gemini"
            req = srv.requests[-1]
            assert "/models/gemini-2.5-pro:generateContent" in req["path"]
            assert req["headers"]["x-goog-api-key"] == "gem-key"
        finally:
            client.close()
    finally:
        srv.stop()


def test_budget_tracker_hierarchy_and_stage_cache(tmp_path: Path) -> None:
    parent = BudgetTracker(max_tokens=1000)
    child = BudgetTracker(max_tokens=200, parent=parent)
    child.reserve()
    child.record(stage="s1_grounding", model="m1", prompt_tokens=150, completion_tokens=60)
    child.release()
    assert parent.tokens == 210
    with pytest.raises(BudgetExceeded):
        child.reserve()

    cache = StageCache(workspace=tmp_path)
    key = StageCache.compute_key(
        episode_fingerprint="ep1", stage_basis="b1", upstream_hash="u1"
    )
    path = cache.put(key, {"hello": "world"})
    assert path == tmp_path / "stage_cache" / key[:2] / f"{key}.json"
    assert cache.get(key) == {"hello": "world"}

    path.write_text("{corrupt", encoding="utf-8")
    assert cache.get(key) is None

    cfg = ConfigManager()
    assert cfg.resolve_episode_budget_tokens(10.0) == 200000
    assert cfg.resolve_episode_budget_tokens(100.0) == 500000
