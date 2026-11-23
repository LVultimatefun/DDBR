"""Pytest fixtures and MockVlmServer for robo_vlm_qc E2E tests."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _import_helpers() -> Any:
    try:
        from tests.e2e import helpers as _helpers

        return _helpers
    except ModuleNotFoundError:
        e2e_dir = Path(__file__).resolve().parent
        tests_pkg = sys.modules.get("tests")
        if tests_pkg is None or str(e2e_dir.parent) not in [
            str(Path(p).resolve()) for p in (getattr(tests_pkg, "__path__", []) or [])
        ]:
            tests_pkg = types.ModuleType("tests")
            tests_pkg.__path__ = [str(e2e_dir.parent)]  # type: ignore[attr-defined]
            sys.modules["tests"] = tests_pkg

        e2e_spec = importlib.util.spec_from_file_location(
            "tests.e2e",
            e2e_dir / "__init__.py",
            submodule_search_locations=[str(e2e_dir)],
        )
        assert e2e_spec is not None and e2e_spec.loader is not None
        e2e_mod = importlib.util.module_from_spec(e2e_spec)
        sys.modules["tests.e2e"] = e2e_mod
        tests_pkg.e2e = e2e_mod  # type: ignore[attr-defined]
        e2e_spec.loader.exec_module(e2e_mod)

        helpers_spec = importlib.util.spec_from_file_location(
            "tests.e2e.helpers", e2e_dir / "helpers.py"
        )
        assert helpers_spec is not None and helpers_spec.loader is not None
        helpers_mod = importlib.util.module_from_spec(helpers_spec)
        sys.modules["tests.e2e.helpers"] = helpers_mod
        e2e_mod.helpers = helpers_mod  # type: ignore[attr-defined]
        helpers_spec.loader.exec_module(helpers_mod)
        return helpers_mod


_helpers = _import_helpers()
write_workspace_config = _helpers.write_workspace_config

__all__ = [
    "MockVlmServer",
    "REPO_ROOT",
    "e2e_config_dir",
    "e2e_workspace",
    "mock_vlm_server",
]

_STAGE_ALIASES: dict[str, str] = {
    "s1": "S1",
    "s1_grounding": "S1",
    "s2": "S2",
    "s2_macro": "S2",
    "s3": "S3",
    "s3_discover": "S3",
    "s4a": "S4A",
    "s4_merge": "S4A",
    "s4b": "S4B",
    "s4_verify": "S4B",
    "s5": "S5",
    "s5_success": "S5",
    "s5_control": "S5_CONTROL",
}


def _normalize_stage(stage: str) -> str:
    key = stage.strip().lower()
    return _STAGE_ALIASES.get(key, stage.strip().upper())


def _extract_messages_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
    return "\n".join(parts)


def _classify_stage(messages: list[dict[str, Any]]) -> str:
    text = _extract_messages_text(messages)
    low = text.lower()
    if (
        "suggested_prompt_diff" in text
        or "prompt_assist" in text
        or "prompt revision" in low
        or "prompt refinement" in low
    ):
        return "UNKNOWN"
    if (
        "anti-false-positive" in low
        or "anti-fp" in low
        or "early control" in low
        or "truncated to the first" in low
        or "truncated at" in low
        or "control_system" in low
        or "control check" in low
    ):
        return "S5_CONTROL"
    if "Stage 4A" in text or "merged_segments" in text:
        return "S4A"
    if "Stage 4B" in text or "keep_segment" in text:
        return "S4B"
    if "Precision-First" in text or "unknown_note" in text or "sliding window" in low:
        return "S3"
    if "macro_stages" in text and "arm_timelines" in text:
        return "S2"
    if "2D Grounding" in text or "bbox_norm_1000" in text or '"objects":' in text:
        return "S1"
    if "satisfied_at_s" in text or "completion criteria" in low:
        return "S5"
    return "UNKNOWN"


def _parse_duration_s(text: str, default: float = 4.0) -> float:
    for pattern in (
        r"(?:total\s+duration|duration|total)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*s",
        r"/\s*([0-9]+(?:\.[0-9]+)?)\s*s",
        r"0\.0s\s+to\s+([0-9]+(?:\.[0-9]+)?)\s*s",
    ):
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return max(0.5, float(m.group(1)))
    return default


def _default_s1_payload(text: str) -> dict[str, Any]:
    obj_ids: list[str] = []
    for m in re.finditer(r"(?:^|\n)\s*[-*]?\s*(?:id:\s*)?([a-z][a-z0-9_]{1,31})\s*[:(]", text):
        cand = m.group(1)
        if cand not in {
            "http",
            "https",
            "task",
            "note",
            "bbox",
            "camera",
            "layout",
            "goal",
            "context",
            "objects",
            "target",
        } and cand not in obj_ids:
            obj_ids.append(cand)
    if not obj_ids:
        obj_ids = ["target_object"]
    objects: dict[str, Any] = {}
    for idx, oid in enumerate(obj_ids[:6]):
        x1 = min(700, 120 + idx * 140)
        y1 = min(650, 220 + idx * 60)
        objects[oid] = {
            "visible": True,
            "bbox": [x1, y1, x1 + 180, y1 + 200],
            "state": "visible",
            "status_kind": "neutral",
        }
    return {"objects": objects}


def _default_s2_payload(text: str) -> dict[str, Any]:
    dur = _parse_duration_s(text, 4.0)
    mid = round(dur * 0.5, 3)
    prep = round(min(0.4, dur * 0.15), 3)
    return {
        "macro_stages": [
            {
                "stage_id": "S1",
                "t_start": 0.0,
                "t_end": mid,
                "label": "Approach and grasp object",
                "label_en": "approach_and_grasp",
                "summary": "Arm approaches the target object and establishes a stable grasp.",
                "color_rgb": [59, 130, 246],
            },
            {
                "stage_id": "S2",
                "t_start": mid,
                "t_end": round(dur, 3),
                "label": "Transfer and place object",
                "label_en": "transfer_and_place",
                "summary": "Transfers the object smoothly and places it in the target zone.",
                "color_rgb": [16, 185, 129],
            },
        ],
        "arm_timelines": {
            "left": [
                {"t_start": 0.0, "t_end": prep, "state": "idle", "desc": "Initial ready pose"},
                {"t_start": prep, "t_end": round(dur, 3), "state": "work", "desc": "Left arm primary manipulation"},
            ],
            "right": [
                {"t_start": 0.0, "t_end": round(dur, 3), "state": "hold", "desc": "Right arm bimanual support"},
            ],
        },
    }


def _default_s3_candidates(dur: float) -> list[dict[str, Any]]:
    scale = min(1.0, dur / 4.0)
    return [
        {
            "t_start": 0.0,
            "t_end": round(0.4 * scale, 3),
            "label_id": "boundary_idle",
            "affected_arm": "both",
            "splice_policy": "cut_clean",
            "visual_evidence": {
                "initial_state": "Both arms remain at the initial standby pose without touching tabletop objects.",
                "process_change": "Both arms and all tabletop objects remain completely stationary with no task progress.",
                "end_state": "Both arms remain in the initial idle pose before starting motion.",
            },
            "confidence": 0.94,
            "unknown_note": "",
        },
        {
            "t_start": round(1.0 * scale, 3),
            "t_end": round(1.8 * scale, 3),
            "label_id": "missed_grasp_retry",
            "affected_arm": "left",
            "splice_policy": "mask_loss",
            "visual_evidence": {
                "initial_state": "Left arm reaches toward the target object for the first grasp attempt.",
                "process_change": "Fingers close on empty air, reopen, retract slightly, and realign.",
                "end_state": "Left arm closes fingers again and successfully lifts the target object.",
            },
            "confidence": 0.91,
            "unknown_note": "",
        },
        {
            "t_start": round(2.2 * scale, 3),
            "t_end": round(2.8 * scale, 3),
            "label_id": "mid_idle_freeze",
            "affected_arm": "both",
            "splice_policy": "cut_clean",
            "visual_evidence": {
                "initial_state": "Both arms hover above the workspace.",
                "process_change": "Both arms freeze mid-trajectory without progressing the subtask.",
                "end_state": "Both arms resume motion from the same hovered pose.",
            },
            "confidence": 0.87,
            "unknown_note": "",
        },
        {
            "t_start": round(3.0 * scale, 3),
            "t_end": round(3.5 * scale, 3),
            "label_id": "unknown_anomaly",
            "affected_arm": "right",
            "splice_policy": "mask_loss",
            "visual_evidence": {
                "initial_state": "Right arm is positioned in an assistive posture.",
                "process_change": "An undefined local visual disturbance appears in the camera view.",
                "end_state": "The disturbance ends and normal manipulation continues.",
            },
            "confidence": 0.78,
            "unknown_note": "Unexpected visual occlusion outside known taxonomy",
        },
    ]


def _default_s3_payload(text: str) -> dict[str, Any]:
    dur = _parse_duration_s(text, 4.0)
    win_m = re.search(
        r"(?:window|sliding\s+window)\s*[:=]?\s*\[\s*([0-9]+(?:\.[0-9]+)?)\s*s?\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*s?\s*\]",
        text,
        re.IGNORECASE,
    )
    all_cands = _default_s3_candidates(dur)
    if win_m:
        w_start = float(win_m.group(1))
        w_end = float(win_m.group(2))
        filtered = [
            c
            for c in all_cands
            if w_start <= (0.5 * (c["t_start"] + c["t_end"])) <= w_end
        ]
        return {"segments": filtered}
    return {"segments": all_cands}


def _default_s4a_payload(text: str) -> dict[str, Any]:
    dur = _parse_duration_s(text, 4.0)
    decoder = json.JSONDecoder()
    anchor = -1
    for marker in ("raw candidates", "Stage 3", "candidates"):
        idx = text.lower().find(marker.lower())
        if idx != -1:
            anchor = idx
            break
    search_from = anchor if anchor != -1 else 0
    for m in re.finditer(r"\[\s*\{", text[search_from:]):
        start = search_from + m.start()
        try:
            raw_list, _ = decoder.raw_decode(text, start)
            if isinstance(raw_list, list) and all(
                isinstance(item, dict) and "label_id" in item for item in raw_list
            ):
                return {"merged_segments": raw_list}
        except json.JSONDecodeError:
            continue
    return {"merged_segments": _default_s3_candidates(dur)}


def _default_s4b_payload(text: str) -> dict[str, Any]:
    span_m = re.search(
        r"\[\s*([0-9]+(?:\.[0-9]+)?)\s*s?\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*s?\s*\][,;\s|]*(?:initial\s+label|label_id|label)\s*[:=]\s*([a-z0-9_-]+)",
        text,
        re.IGNORECASE,
    )
    if span_m:
        t_start = float(span_m.group(1))
        t_end = float(span_m.group(2))
        label_id = span_m.group(3)
    else:
        t_start, t_end, label_id = 0.0, 0.4, "boundary_idle"

    if label_id in ("boundary_idle", "mid_idle_freeze"):
        policy = "cut_clean"
        arm = "both"
    elif label_id in ("precision_fine_adjust", "bimanual_assist_hold"):
        policy = "keep_positive"
        arm = "both"
    elif label_id == "missed_grasp_retry":
        policy = "mask_loss"
        arm = "left"
    else:
        policy = "mask_loss"
        arm = "right"

    return {
        "keep_segment": True,
        "t_start": t_start,
        "t_end": t_end,
        "label_id": label_id,
        "affected_arm": arm,
        "splice_policy": policy,
        "visual_evidence": {
            "initial_state": f"At start frame ({t_start:.2f}s), candidate onset cues are visible.",
            "process_change": f"Across the interval, visual dynamics match {label_id}.",
            "end_state": f"At end frame ({t_end:.2f}s), the candidate interval concludes.",
        },
        "confidence": 0.91,
        "reject_reason": "",
    }


def _extract_criteria_lines(text: str) -> list[str]:
    m = re.search(
        r"(?:completion\s+criteria|success\s+criteria|criteria)\s*:\s*\n(.*?)(?:\n(?:camera|view|task|please|evaluate)|$)",
        text,
        re.S | re.IGNORECASE,
    )
    if not m:
        return ["Target object reaches the expected goal state"]
    lines = [
        re.sub(r"^\s*[-*0-9.)]+\s*", "", ln).strip()
        for ln in m.group(1).splitlines()
        if ln.strip()
    ]
    return lines or ["Target object reaches the expected goal state"]


def _default_s5_payload(text: str) -> dict[str, Any]:
    dur = _parse_duration_s(text, 4.0)
    sat_s = round(dur * 0.85, 2)
    criteria = _extract_criteria_lines(text)
    return {
        "verdict": "success",
        "confidence": 0.93,
        "criteria": [
            {
                "criterion": c,
                "met": True,
                "satisfied_at_s": sat_s,
                "evidence": f"Tail frames around {sat_s:.2f}s confirm the criterion is satisfied.",
            }
            for c in criteria
        ],
        "summary": "All task completion criteria are stably satisfied in the final frames.",
    }


def _default_s5_control_payload(text: str) -> dict[str, Any]:
    criteria = _extract_criteria_lines(text)
    return {
        "verdict": "failure",
        "confidence": 0.91,
        "criteria": [
            {
                "criterion": c,
                "met": False,
                "satisfied_at_s": None,
                "evidence": "Task is not yet completed in the early truncated control frames.",
            }
            for c in criteria
        ],
        "summary": "Early control truncation does not yet satisfy the task completion state.",
    }


class MockVlmServer:
    """In-process threaded HTTP server emulating OpenAI-compatible VLM endpoints."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests: list[dict[str, Any]] = []
        self._failures: list[dict[str, Any]] = []
        self._stage_overrides: dict[str, Any] = {}
        self._counter = 0

        server_ref = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def _send_json(self, code: int, payload: Any) -> None:
                raw = (
                    payload.encode("utf-8")
                    if isinstance(payload, str)
                    else json.dumps(payload, ensure_ascii=False).encode("utf-8")
                )
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:  # noqa: N802
                if self.path.rstrip("/") in ("/health", "/v1/models"):
                    self._send_json(200, {"status": "ok", "data": []})
                    return
                self._send_json(404, {"error": {"message": "not found"}})

            def do_POST(self) -> None:  # noqa: N802
                if not self.path.endswith("/chat/completions"):
                    self._send_json(404, {"error": {"message": f"unknown path {self.path}"}})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                raw_bytes = self.rfile.read(length)
                body = json.loads(raw_bytes.decode("utf-8"))
                messages = body.get("messages", [])
                stage = _classify_stage(messages)
                headers_lower = {k.lower(): v for k, v in self.headers.items()}

                failure_spec: dict[str, Any] | None = None
                override_spec: Any = None
                with server_ref._lock:
                    server_ref._counter += 1
                    req_idx = server_ref._counter
                    req_record = {
                        "index": req_idx,
                        "path": self.path,
                        "headers": headers_lower,
                        "model": body.get("model"),
                        "messages": messages,
                        "max_tokens": body.get("max_tokens"),
                        "extra_body": body.get("extra_body"),
                        "stage": stage,
                        "body": body,
                    }
                    server_ref.requests.append(req_record)
                    for fi, fspec in enumerate(server_ref._failures):
                        if fspec["stage"] is None or fspec["stage"] == stage:
                            failure_spec = server_ref._failures.pop(fi)
                            break
                    if failure_spec is None and stage in server_ref._stage_overrides:
                        ov = server_ref._stage_overrides[stage]
                        if isinstance(ov, list):
                            override_spec = ov.pop(0) if len(ov) > 1 else ov[0]
                        else:
                            override_spec = ov

                if failure_spec is not None:
                    status_code = int(failure_spec["status_code"])
                    err_body = failure_spec["body"]
                    if err_body is None:
                        err_body = {
                            "error": {
                                "message": f"Mock programmed error {status_code}",
                                "type": "mock_error",
                                "code": status_code,
                            }
                        }
                    self._send_json(status_code, err_body)
                    return

                if callable(override_spec):
                    content_obj = override_spec(req_record)
                elif override_spec is not None:
                    content_obj = override_spec
                else:
                    content_obj = server_ref._build_default_payload(stage, messages)

                content_str = (
                    content_obj
                    if isinstance(content_obj, str)
                    else json.dumps(content_obj, ensure_ascii=False)
                )
                resp = {
                    "id": f"chatcmpl-mock-{req_idx}",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": body.get("model") or "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": content_str,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 120,
                        "completion_tokens": 60,
                        "total_tokens": 180,
                    },
                }
                self._send_json(200, resp)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.port = int(self._httpd.server_address[1])
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self.chat_url = f"{self.base_url}/chat/completions"
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True, name="mock-vlm-server"
        )
        self._thread.start()

    @staticmethod
    def _build_default_payload(stage: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        text = _extract_messages_text(messages)
        if stage == "S1":
            return _default_s1_payload(text)
        if stage == "S2":
            return _default_s2_payload(text)
        if stage == "S3":
            return _default_s3_payload(text)
        if stage == "S4A":
            return _default_s4a_payload(text)
        if stage == "S4B":
            return _default_s4b_payload(text)
        if stage == "S5":
            return _default_s5_payload(text)
        if stage == "S5_CONTROL":
            return _default_s5_control_payload(text)
        return {"ok": True}

    def fail_next(
        self,
        status_code: int = 429,
        count: int = 1,
        *,
        body: dict[str, Any] | str | None = None,
        stage: str | None = None,
    ) -> None:
        norm_stage = _normalize_stage(stage) if stage is not None else None
        with self._lock:
            for _ in range(max(0, int(count))):
                self._failures.append(
                    {"status_code": int(status_code), "body": body, "stage": norm_stage}
                )

    def fail_with_malformed_json(
        self, count: int = 1, *, stage: str | None = None
    ) -> None:
        norm_stage = _normalize_stage(stage) if stage is not None else None
        with self._lock:
            for _ in range(max(0, int(count))):
                self._failures.append(
                    {
                        "status_code": 200,
                        "body": {
                            "id": "chatcmpl-malformed",
                            "object": "chat.completion",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "{not_valid_json",
                                    },
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 50,
                                "completion_tokens": 10,
                                "total_tokens": 60,
                            },
                        },
                        "stage": norm_stage,
                    }
                )

    def set_stage_response(
        self,
        stage: str,
        payload: dict[str, Any] | list[dict[str, Any]] | Callable[[dict[str, Any]], Any],
    ) -> None:
        norm_stage = _normalize_stage(stage)
        with self._lock:
            if isinstance(payload, list):
                self._stage_overrides[norm_stage] = list(payload)
            else:
                self._stage_overrides[norm_stage] = payload

    def clear_stage_responses(self) -> None:
        with self._lock:
            self._stage_overrides.clear()

    def reset(self) -> None:
        with self._lock:
            self.requests.clear()
            self._failures.clear()
            self._stage_overrides.clear()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=3.0)


def _sanitize_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    for proxy_var in (
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
    ):
        monkeypatch.delenv(proxy_var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("MOCK_VLM_API_KEY", "mock-e2e-key")


@pytest.fixture
def mock_vlm_server(monkeypatch: pytest.MonkeyPatch) -> MockVlmServer:
    """Yield a running localhost OpenAI-compatible MockVlmServer."""
    _sanitize_test_env(monkeypatch)
    srv = MockVlmServer()
    try:
        yield srv
    finally:
        srv.stop()


@pytest.fixture
def e2e_workspace(
    tmp_path: Path, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Create an isolated workspace under ``tmp_path / 'ws'`` with copied configs."""
    _sanitize_test_env(monkeypatch)

    cfg_dst = tmp_path / "configs"
    if not cfg_dst.exists():
        shutil.copytree(REPO_ROOT / "configs", cfg_dst)

    ws = tmp_path / "ws"
    for sub in ("artifacts", "stage_cache", "exports", "configs_override", ".history"):
        (ws / sub).mkdir(parents=True, exist_ok=True)

    if "mock_vlm_server" in request.fixturenames:
        srv: MockVlmServer = request.getfixturevalue("mock_vlm_server")
        write_workspace_config(ws, base_url=srv.base_url, api_key_env="MOCK_VLM_API_KEY")

    return ws


@pytest.fixture
def e2e_config_dir(e2e_workspace: Path) -> Path:
    """Return the isolated ``tmp_path / 'configs'`` directory paired with ``e2e_workspace``."""
    return e2e_workspace.parent / "configs"
