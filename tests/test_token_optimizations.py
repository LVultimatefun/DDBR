"""Comprehensive unit & integration tests for token optimization Items #1-#5, #7, #8."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pytest

from robo_vlm_qc.config import ConfigManager
from robo_vlm_qc.io import auto_detect, get_adapter
from robo_vlm_qc.io.base import EpisodeView
from robo_vlm_qc.models.annotation import (
    MacroStage,
    QcSegment,
    StageUsage,
    UsageSummary,
    VisualEvidence,
    VlmRunArtifact,
)
from robo_vlm_qc.models.episode import CameraSpec, EpisodeMeta, KinematicsSpec
from robo_vlm_qc.models.taxonomy import TaskSpec
from robo_vlm_qc.store import NotFoundError, Repository
from robo_vlm_qc.vlm.client import (
    BudgetTracker,
    VlmBlindVisionError,
    VlmClient,
    build_interleaved_messages,
    effective_vision_min_tokens_per_image,
    estimate_stage_cost_usd,
    evaluate_vision_token_density,
    normalize_image_detail,
)
from robo_vlm_qc.vlm.dag import VlmDag
from robo_vlm_qc.vlm.estimator import estimate_batch_vlm_cost
from robo_vlm_qc.vlm.frame_dedup import (
    FrameDedupResult,
    deduplicate_frames,
    format_skipped_ranges_note,
)
from robo_vlm_qc.vlm.prompt_builder import PromptBuilder
from robo_vlm_qc.vlm.stage1_grounding import run_stage1_grounding
from robo_vlm_qc.vlm.stage2_macro import run_stage2_macro
from robo_vlm_qc.vlm.stage3_discover import (
    _plan_adaptive_window_times,
    _plan_window_times,
    run_stage3_discover,
)
from robo_vlm_qc.vlm.stage4_verify import _plan_s4b_frame_count, run_stage4b_verify
from robo_vlm_qc.vlm.stage5_success import run_stage5_success


class _MockHttpServer:
    """In-process HTTP server supporting both OpenAI-compat and Gemini native responses."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.queue: list[tuple[int, dict[str, Any]]] = []
        self._lock = threading.Lock()
        srv = self

        class _H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A002
                return

            def do_POST(self) -> None:  # noqa: N802
                n = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(n).decode("utf-8") if n > 0 else "{}"
                body = json.loads(raw)
                with srv._lock:
                    srv.requests.append(
                        {"path": self.path, "headers": dict(self.headers), "body": body}
                    )
                    item = srv.queue.pop(0) if srv.queue else None
                if item is not None:
                    code, payload = item
                elif ":generateContent" in self.path:
                    code = 200
                    payload = {
                        "candidates": [
                            {
                                "content": {"parts": [{"text": '{"ok": true}'}]},
                                "finishReason": "STOP",
                            }
                        ],
                        "usageMetadata": {
                            "promptTokenCount": 400,
                            "candidatesTokenCount": 50,
                            "totalTokenCount": 450,
                        },
                    }
                else:
                    code = 200
                    payload = {
                        "choices": [
                            {
                                "message": {"content": '{"ok": true}'},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 200,
                            "completion_tokens": 40,
                            "total_tokens": 240,
                        },
                    }
                out = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _H)
        self.port = int(self._httpd.server_address[1])
        self.v1_url = f"http://127.0.0.1:{self.port}/v1"
        self.v1beta_url = f"http://127.0.0.1:{self.port}/v1beta"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2.0)


class _RecordingClient:
    """In-memory deterministic VlmClient test double capturing all chat_json kwargs."""

    def __init__(
        self,
        handler: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        cached_tokens_per_call: int = 0,
    ) -> None:
        self.handler = handler
        self.calls: list[dict[str, Any]] = []
        self.budget = BudgetTracker()
        self.cached_tokens_per_call = cached_tokens_per_call
        self._lock = threading.Lock()

    def is_cancelled(self) -> bool:
        return False

    def model_for_stage(self, stage: str, override: str | None = None) -> str:
        return override or "gpt-4o"

    def resolve_stage_model(self, stage: str, model_override: str | None = None) -> str:
        return model_override or "gpt-4o"

    def chat_json(self, **kwargs: Any) -> dict[str, Any]:
        with self._lock:
            self.calls.append(kwargs)
        if kwargs.get("budget") is not None:
            kwargs["budget"].record(
                stage=kwargs.get("stage", "unknown"),
                model=kwargs.get("model_override") or "gpt-4o",
                prompt_tokens=1000,
                completion_tokens=200,
                total_tokens=1200,
                cached_tokens=self.cached_tokens_per_call,
            )
        return self.handler(kwargs)


class _SynthMedia:
    """In-memory MediaSource with configurable per-timestamp image generator."""

    def __init__(
        self,
        duration_s: float = 20.0,
        fps: float = 30.0,
        *,
        w: int = 320,
        h: int = 180,
        frame_fn: Callable[[float], np.ndarray] | None = None,
    ) -> None:
        self.duration_s = float(duration_s)
        self.fps = float(fps)
        self._w = w
        self._h = h
        self._frame_fn = frame_fn
        self.cameras = [CameraSpec(name="main", role="main", width=w, height=h, fps=fps)]

    def read_frame_rgb(self, camera: str, t_s: float) -> np.ndarray:
        if self._frame_fn is not None:
            return self._frame_fn(float(t_s))
        return np.full((self._h, self._w, 3), 90, dtype=np.uint8)

    def read_frames_batch(self, camera: str, times_s: list[float]) -> list[np.ndarray]:
        return [self.read_frame_rgb(camera, t) for t in times_s]


def _make_synth_view(
    *,
    duration_s: float = 20.0,
    task_id: str = "_default",
    frame_fn: Callable[[float], np.ndarray] | None = None,
    episode_id: str = "synth_ds/ep_000",
) -> EpisodeView:
    meta = EpisodeMeta(
        episode_id=episode_id,
        dataset=episode_id.split("/")[0],
        stem=episode_id.split("/")[-1],
        task_id=task_id,
        adapter_type="video_dir",
        source_uri="/tmp/synth_ep",
        source_mtime=1.0,
        episode_fingerprint=f"fp_{episode_id}_{duration_s}_{task_id}",
        duration_s=float(duration_s),
        fps=30.0,
        frame_count=max(1, int(round(duration_s * 30.0))),
        cameras=[CameraSpec(name="main", role="main", width=320, height=180, fps=30.0)],
        camera_roles={"main": "main"},
        kinematics=KinematicsSpec(has_kinematics=False),
    )
    return EpisodeView(
        meta=meta,
        media=_SynthMedia(duration_s=duration_s, frame_fn=frame_fn),  # type: ignore[arg-type]
    )


def _dummy_imgs(n: int, val: int = 80) -> list[np.ndarray]:
    return [np.full((48, 64, 3), val, dtype=np.uint8) for _ in range(n)]


# ---------------------------------------------------------------------------
# Item #0 & #1: Config defaults, OpenAI image_url.detail, Gemini resolution,
# and blind-node guard calibration
# ---------------------------------------------------------------------------


def test_default_yaml_token_optimization_and_openai_invariants() -> None:
    cfg = ConfigManager()
    vlm = cfg.settings["vlm"]

    # OpenAI OSS defaults strictly untouched
    assert vlm["provider"] == "openai_compat"
    assert vlm["base_url"] == "https://api.openai.com/v1"
    assert vlm["api_key_env"] == "OPENAI_API_KEY"
    assert vlm["key_file"] == ".env"
    assert vlm["thinking_mode"] == "auto"
    for stage in ("s1_grounding", "s2_macro", "s3_discover", "s4_verify", "s5_success"):
        assert vlm["stage_models"][stage] == "gpt-4o"
        assert vlm[stage]["thinking"] is False

    # Preview cache config for @sys-api-dx
    assert cfg.settings["io"]["preview_cache"] == {"max_cache_gb": 10.0}

    # Per-stage detail & optimization knobs
    assert vlm["s1_grounding"]["pass0_image_detail"] == "low"
    assert vlm["s1_grounding"]["image_detail"] == "high"
    assert vlm["s1_grounding"]["merge_pass0_short_episode"] is True

    assert vlm["s2_macro"]["image_detail"] == "low"
    assert vlm["s2_macro"]["static_dedup_enabled"] is True
    assert vlm["s2_macro"]["static_dedup_mad_thresh"] == 4.5
    assert vlm["s2_macro"]["static_dedup_max_gap_s"] == 6.0

    assert vlm["s3_discover"]["image_detail"] == "auto"
    assert vlm["s3_discover"]["window_s"] == 16.0
    assert vlm["s3_discover"]["window_stride_s"] == 12.0
    assert vlm["s3_discover"]["window_step_s"] == 0.8
    assert vlm["s3_discover"]["adaptive_step_enabled"] is True
    assert vlm["s3_discover"]["idle_step_s"] == 1.4
    assert vlm["s3_discover"]["static_dedup_enabled"] is True
    assert vlm["s3_discover"]["static_dedup_mad_thresh"] == 3.5
    assert vlm["s3_discover"]["static_dedup_max_gap_s"] == 3.2

    assert vlm["s4_verify"]["image_detail"] == "auto"
    assert vlm["s4_verify"]["dynamic_refine_frames"] is True
    assert vlm["s4_verify"]["refine_min_frames"] == 4

    assert vlm["s5_success"]["image_detail"] == "auto"


def test_image_detail_openai_wire_gemini_mapping_and_blind_node_calibration() -> None:
    assert normalize_image_detail(None) is None
    assert normalize_image_detail("LOW") == "low"
    assert normalize_image_detail("high") == "high"
    assert normalize_image_detail("auto") == "auto"
    with pytest.raises(ValueError, match="Unsupported image_detail"):
        normalize_image_detail("ultra")

    # build_interleaved_messages emits detail only when set
    msgs_none = build_interleaved_messages(
        system_prompt="sys", user_prompt="usr", images=_dummy_imgs(1)
    )
    img_blk_none = [b for b in msgs_none[1]["content"] if b["type"] == "image_url"][0]
    assert "detail" not in img_blk_none["image_url"]

    msgs_low = build_interleaved_messages(
        system_prompt="sys", user_prompt="usr", images=_dummy_imgs(1), image_detail="low"
    )
    img_blk_low = [b for b in msgs_low[1]["content"] if b["type"] == "image_url"][0]
    assert img_blk_low["image_url"]["detail"] == "low"

    # Blind-node threshold calibration for low detail
    assert effective_vision_min_tokens_per_image(550, image_detail="low") == 60
    assert effective_vision_min_tokens_per_image(40, image_detail="low") == 40
    assert effective_vision_min_tokens_per_image(550, image_detail="high") == 550
    assert effective_vision_min_tokens_per_image(550, image_detail="auto") == 550
    assert effective_vision_min_tokens_per_image(
        550, media_resolution="MEDIA_RESOLUTION_LOW"
    ) == 60

    # OpenAI low-detail (~85 tok/img) NEVER triggers blind-node alert
    is_blind_low, tpi_low, _, _ = evaluate_vision_token_density(
        n_images=4,
        prompt_tokens=4 * 85 + 60,
        exact_image_tokens=4 * 85,
        est_text_tokens=60,
        min_tokens_per_image=550,
        image_detail="low",
    )
    assert is_blind_low is False
    assert tpi_low == pytest.approx(85.0)

    # Truly blind response (20 tok/img) still triggers even on low detail
    is_blind_zero, _, _, _ = evaluate_vision_token_density(
        n_images=4,
        prompt_tokens=4 * 20 + 60,
        exact_image_tokens=4 * 20,
        est_text_tokens=60,
        min_tokens_per_image=550,
        image_detail="low",
    )
    assert is_blind_zero is True

    srv = _MockHttpServer()
    try:
        # 1. OpenAI-compat sends image_url.detail on wire
        oai = VlmClient(
            provider="openai_compat",
            base_url=srv.v1_url,
            api_key="test-key",
            retries=0,
        )
        try:
            oai.chat_json(
                stage="s2_macro",
                user_prompt="inspect",
                images=_dummy_imgs(2),
                image_detail="low",
            )
            blocks = srv.requests[-1]["body"]["messages"][0]["content"]
            img_blocks = [b["image_url"] for b in blocks if b["type"] == "image_url"]
            assert len(img_blocks) == 2
            assert all(ib.get("detail") == "low" for ib in img_blocks)
        finally:
            oai.close()

        # 2. Gemini native maps low->MEDIA_RESOLUTION_LOW, high->MEDIA_RESOLUTION_HIGH,
        #    auto->configured gemini_media_resolution, AND low-detail skips false blind-node retries!
        gem = VlmClient(
            provider="gemini",
            base_url=srv.v1beta_url,
            api_key="gem-test-key",
            retries=0,
            retry_min_wait_s=0.005,
            retry_max_wait_s=0.01,
            stage_models={"s2_macro": "gemini-3-pro-preview"},
            gemini_media_resolution="MEDIA_RESOLUTION_HIGH",
            vision_min_tokens_per_image=550,
            vision_blind_node_retries=2,
            vision_blind_node_guard=True,
        )
        try:
            # Low detail returning ~85 tok/img MUST succeed in 1 request (zero false blind retries)
            srv.requests.clear()
            srv.queue.append(
                (
                    200,
                    {
                        "candidates": [
                            {
                                "content": {"parts": [{"text": '{"detail": "low_ok"}'}]},
                                "finishReason": "STOP",
                            }
                        ],
                        "usageMetadata": {
                            "promptTokenCount": 4 * 85 + 30,
                            "candidatesTokenCount": 20,
                            "totalTokenCount": 390,
                            "promptTokensDetails": [
                                {"modality": "IMAGE", "tokenCount": 4 * 85},
                                {"modality": "TEXT", "tokenCount": 30},
                            ],
                        },
                    },
                )
            )
            res_low = gem.chat_json(
                stage="s2_macro",
                user_prompt="low detail check",
                images=_dummy_imgs(4),
                image_detail="low",
            )
            assert res_low == {"detail": "low_ok"}
            assert len(srv.requests) == 1
            assert (
                srv.requests[-1]["body"]["generationConfig"]["mediaResolution"]
                == "MEDIA_RESOLUTION_LOW"
            )

            # High detail maps to MEDIA_RESOLUTION_HIGH and enforces full 550 tok/img guard
            srv.requests.clear()
            for _ in range(3):
                srv.queue.append(
                    (
                        200,
                        {
                            "candidates": [
                                {
                                    "content": {"parts": [{"text": '{"bad": true}'}]},
                                    "finishReason": "STOP",
                                }
                            ],
                            "usageMetadata": {
                                "promptTokenCount": 4 * 258 + 20,
                                "candidatesTokenCount": 15,
                                "totalTokenCount": 1067,
                                "promptTokensDetails": [
                                    {"modality": "IMAGE", "tokenCount": 4 * 258},
                                    {"modality": "TEXT", "tokenCount": 20},
                                ],
                            },
                        },
                    )
                )
            with pytest.raises(VlmBlindVisionError):
                gem.chat_json(
                    stage="s2_macro",
                    user_prompt="high detail blind check",
                    images=_dummy_imgs(4),
                    image_detail="high",
                )
            assert len(srv.requests) == 3
            assert (
                srv.requests[-1]["body"]["generationConfig"]["mediaResolution"]
                == "MEDIA_RESOLUTION_HIGH"
            )

            # Auto detail falls back to configured gemini_media_resolution
            srv.requests.clear()
            srv.queue.append(
                (
                    200,
                    {
                        "candidates": [
                            {
                                "content": {"parts": [{"text": '{"auto": true}'}]},
                                "finishReason": "STOP",
                            }
                        ],
                        "usageMetadata": {
                            "promptTokenCount": 2 * 1120 + 20,
                            "candidatesTokenCount": 20,
                            "totalTokenCount": 2280,
                            "promptTokensDetails": [
                                {"modality": "IMAGE", "tokenCount": 2 * 1120},
                                {"modality": "TEXT", "tokenCount": 20},
                            ],
                        },
                    },
                )
            )
            res_auto = gem.chat_json(
                stage="s2_macro",
                user_prompt="auto detail",
                images=_dummy_imgs(2),
                image_detail="auto",
            )
            assert res_auto == {"auto": True}
            assert (
                srv.requests[-1]["body"]["generationConfig"]["mediaResolution"]
                == "MEDIA_RESOLUTION_HIGH"
            )
        finally:
            gem.close()
    finally:
        srv.stop()


def test_stages_s1_to_s5_wire_configured_image_detail() -> None:
    cfg = ConfigManager()
    cfg.settings["vlm"]["s1_grounding"]["tracking"]["enabled"] = False
    view_long = _make_synth_view(duration_s=20.0, task_id="_default")

    # S1 Pass-0 gets "low", S1 batches get "high"
    cl_s1 = _RecordingClient(
        lambda kw: (
            {"canonical_objects": [{"id": "mug", "name": "Mug", "desc": "mug", "is_static": False}]}
            if "canonical_objects" in kw["system_prompt"]
            else {"objects": {"mug": {"visible": True, "bbox": [100, 100, 250, 250]}}}
        )
    )
    run_stage1_grounding(view=view_long, task=cfg.get_task("_default"), client=cl_s1, cfg=cfg)  # type: ignore[arg-type]
    assert cl_s1.calls[0]["image_detail"] == "low"
    assert all(c["image_detail"] == "high" for c in cl_s1.calls[1:])

    # S2 gets "low"
    cl_s2 = _RecordingClient(lambda _kw: {"macro_stages": [], "arm_timelines": {}})
    run_stage2_macro(view=view_long, client=cl_s2, cfg=cfg)  # type: ignore[arg-type]
    assert cl_s2.calls[0]["image_detail"] == "low"

    # S3 gets "auto"
    cl_s3 = _RecordingClient(lambda _kw: {"segments": []})
    run_stage3_discover(view=view_long, client=cl_s3, cfg=cfg)  # type: ignore[arg-type]
    assert all(c["image_detail"] == "auto" for c in cl_s3.calls)

    # S4B gets "auto"
    cand = QcSegment(
        seg_id="c1",
        t_start=4.0,
        t_end=6.0,
        label_id="missed_grasp_retry",
        training_role="recovery",
        splice_policy="mask_loss",
        mask_side="left",
        affected_arm="left",
        visual_evidence=VisualEvidence(process_change="retry"),
        vlm_raw_policy="mask_loss",
    )
    cl_s4b = _RecordingClient(lambda _kw: {"keep_segment": True})
    run_stage4b_verify(view=view_long, candidates=[cand], client=cl_s4b, cfg=cfg)  # type: ignore[arg-type]
    assert cl_s4b.calls[0]["image_detail"] == "auto"

    # S5 primary and control both get "auto"
    cl_s5 = _RecordingClient(
        lambda kw: {
            "verdict": "failure" if kw["stage"] == "s5_control" else "success",
            "confidence": 0.9,
            "criteria": [
                {"criterion": "done", "met": kw["stage"] != "s5_control", "evidence": "ok"}
            ],
            "summary": "ok",
        }
    )
    run_stage5_success(view=view_long, task=cfg.get_task("pick_and_place"), client=cl_s5, cfg=cfg)  # type: ignore[arg-type]
    assert [c["stage"] for c in cl_s5.calls] == ["s5_success", "s5_control"]
    assert all(c["image_detail"] == "auto" for c in cl_s5.calls)


# ---------------------------------------------------------------------------
# Item #2: Static-Frame Deduplication (unit + S2 & S3 integration)
# ---------------------------------------------------------------------------


def test_frame_deduplication_mad_max_gap_min_keep_and_prompt_note() -> None:
    # 1. Fully static 11-frame sequence [0..20s step 2s], max_gap_s=5.0
    static_imgs = _dummy_imgs(11, val=110)
    ts = [float(i * 2) for i in range(11)]
    res = deduplicate_frames(static_imgs, ts, mad_threshold=4.0, max_gap_s=5.0, min_keep=2)
    assert isinstance(res, FrameDedupResult)
    assert res.kept_indices[0] == 0
    assert res.kept_indices[-1] == 10
    assert res.kept_count == len(res.kept_indices)
    assert res.dropped_count == 11 - res.kept_count
    assert res.dropped_count >= 5
    # Strict max_gap_s enforcement across every adjacent pair of kept frames
    kept_ts = [ts[i] for i in res.kept_indices]
    for a, b in zip(kept_ts, kept_ts[1:]):
        assert (b - a) <= 5.0 + 1e-6

    note = format_skipped_ranges_note(res.skipped_ranges)
    assert "[Static-Frame Deduplication Note]" in note
    assert "0.00s" in note
    assert format_skipped_ranges_note([]) == ""

    # 2. Highly dynamic sequence keeps every frame (0 dropped)
    dyn_imgs = [np.full((48, 64, 3), (i * 25) % 256, dtype=np.uint8) for i in range(8)]
    dyn_ts = [float(i) for i in range(8)]
    res_dyn = deduplicate_frames(dyn_imgs, dyn_ts, mad_threshold=4.0, max_gap_s=6.0, min_keep=2)
    assert res_dyn.kept_indices == list(range(8))
    assert res_dyn.dropped_count == 0
    assert res_dyn.skipped_ranges == []

    # 3. min_keep enforcement on static short sequence with huge max_gap_s
    res_min = deduplicate_frames(
        _dummy_imgs(6, val=50),
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
        mad_threshold=4.0,
        max_gap_s=99.0,
        min_keep=4,
    )
    assert res_min.kept_count == 4
    assert res_min.kept_indices[0] == 0 and res_min.kept_indices[-1] == 5

    # 4. S2 & S3 integration: static video drops redundant frames (ignoring 26px TIME header) & adds note
    cfg = ConfigManager()
    static_view = _make_synth_view(duration_s=16.0)

    cl_s2_on = _RecordingClient(lambda _kw: {"macro_stages": [], "arm_timelines": {}})
    run_stage2_macro(view=static_view, client=cl_s2_on, cfg=cfg)  # type: ignore[arg-type]
    assert len(cl_s2_on.calls[0]["images"]) < 9  # raw 16s @ 2s = 9 frames -> deduped to ~4 frames
    assert "[Static-Frame Deduplication Note]" in cl_s2_on.calls[0]["user_prompt"]

    cfg_off = ConfigManager()
    cfg_off.settings["vlm"]["s2_macro"]["static_dedup_enabled"] = False
    cl_s2_off = _RecordingClient(lambda _kw: {"macro_stages": [], "arm_timelines": {}})
    run_stage2_macro(view=static_view, client=cl_s2_off, cfg=cfg_off)  # type: ignore[arg-type]
    assert len(cl_s2_off.calls[0]["images"]) == 9
    assert "[Static-Frame Deduplication Note]" not in cl_s2_off.calls[0]["user_prompt"]

    cl_s3_on = _RecordingClient(lambda _kw: {"segments": []})
    run_stage3_discover(view=static_view, client=cl_s3_on, cfg=cfg)  # type: ignore[arg-type]
    assert len(cl_s3_on.calls[0]["images"]) <= 8  # raw 16s @ 0.8s = 21 frames -> compressed!
    assert "[Static-Frame Deduplication Note]" in cl_s3_on.calls[0]["user_prompt"]


# ---------------------------------------------------------------------------
# Item #3: S3 Adaptive Step Sampling
# ---------------------------------------------------------------------------


def test_s3_adaptive_step_sampling_dense_near_boundaries_and_coarse_in_calm_interiors() -> None:
    # Single calm 30s macro stage -> window [10s, 26s] is entirely calm interior (>2s from 0 and 30)
    calm_stages = [MacroStage(stage_id="S1", t_start=0.0, t_end=30.0, label="Steady work")]
    uniform = _plan_window_times(10.0, 26.0, 0.8)
    adaptive_calm = _plan_adaptive_window_times(
        10.0,
        26.0,
        fine_step_s=0.8,
        idle_step_s=1.4,
        duration_s=30.0,
        macro_stages=calm_stages,
    )
    assert len(adaptive_calm) < len(uniform)
    assert adaptive_calm[0] == 10.0 and adaptive_calm[-1] == 26.0
    # Interior steps are ~1.4s
    mid_gaps = [round(b - a, 2) for a, b in zip(adaptive_calm[:-1], adaptive_calm[1:])]
    assert any(g >= 1.35 for g in mid_gaps)

    # Window crossing a macro transition at t=15.0s -> dense 0.8s within +/-2.0s of 15.0s
    split_stages = [
        MacroStage(stage_id="S1", t_start=0.0, t_end=15.0, label="Grasp"),
        MacroStage(stage_id="S2", t_start=15.0, t_end=30.0, label="Place"),
    ]
    adaptive_trans = _plan_adaptive_window_times(
        6.0,
        22.0,
        fine_step_s=0.8,
        idle_step_s=1.4,
        duration_s=30.0,
        macro_stages=split_stages,
    )
    near_15 = [t for t in adaptive_trans if 13.0 <= t <= 17.0]
    assert len(near_15) >= 5
    for a, b in zip(near_15, near_15[1:]):
        assert (b - a) <= 0.85

    # Short macro stages (<=5.0s) stay finely sampled throughout
    busy_stages = [
        MacroStage(stage_id="S1", t_start=0.0, t_end=4.0, label="A"),
        MacroStage(stage_id="S2", t_start=4.0, t_end=8.0, label="B"),
        MacroStage(stage_id="S3", t_start=8.0, t_end=12.0, label="C"),
    ]
    adaptive_busy = _plan_adaptive_window_times(
        0.0,
        12.0,
        fine_step_s=0.8,
        idle_step_s=1.4,
        duration_s=12.0,
        macro_stages=busy_stages,
    )
    for a, b in zip(adaptive_busy, adaptive_busy[1:]):
        assert (b - a) <= 0.85

    # Short window ALWAYS guarantees >=4 frames
    short_win = _plan_adaptive_window_times(
        0.0,
        1.2,
        fine_step_s=0.8,
        idle_step_s=1.4,
        duration_s=1.2,
        macro_stages=[MacroStage(stage_id="S1", t_start=0.0, t_end=1.2, label="Short")],
    )
    assert len(short_win) >= 4

    # Integration with run_stage3_discover (dedup disabled to isolate adaptive stepping)
    dyn_view = _make_synth_view(
        duration_s=30.0,
        frame_fn=lambda t: np.full((180, 320, 3), int(t * 17) % 255, dtype=np.uint8),
    )
    cfg_on = ConfigManager()
    cfg_on.settings["vlm"]["s3_discover"]["static_dedup_enabled"] = False
    cfg_on.settings["vlm"]["s3_discover"]["adaptive_step_enabled"] = True

    cfg_off = ConfigManager()
    cfg_off.settings["vlm"]["s3_discover"]["static_dedup_enabled"] = False
    cfg_off.settings["vlm"]["s3_discover"]["adaptive_step_enabled"] = False

    cl_on = _RecordingClient(lambda _kw: {"segments": []})
    cl_off = _RecordingClient(lambda _kw: {"segments": []})
    run_stage3_discover(view=dyn_view, macro_stages=calm_stages, client=cl_on, cfg=cfg_on)  # type: ignore[arg-type]
    run_stage3_discover(view=dyn_view, macro_stages=calm_stages, client=cl_off, cfg=cfg_off)  # type: ignore[arg-type]
    imgs_on = sum(len(c["images"]) for c in cl_on.calls)
    imgs_off = sum(len(c["images"]) for c in cl_off.calls)
    assert imgs_on < imgs_off


# ---------------------------------------------------------------------------
# Item #4: Short-Episode S1 Pass-0 + Batch-0 Merge & Fallbacks
# ---------------------------------------------------------------------------


def test_s1_short_episode_pass0_batch0_merge_and_graceful_fallbacks() -> None:
    cfg = ConfigManager()
    cfg.settings["vlm"]["s1_grounding"]["tracking"]["enabled"] = False
    cfg.settings["vlm"]["s1_grounding"]["batch_size"] = 4
    cfg.settings["vlm"]["s1_grounding"]["interval_s"] = 2.5
    # 7.0s @ 2.5s => [0.0, 2.5, 5.0, 7.0] => 4 keyframes <= batch_size=4
    short_view = _make_synth_view(duration_s=7.0, task_id="_default")

    # 1. Happy path: combined discovery + grounding finishes in 1 call with image_detail="high"
    def _combined_handler(kw: dict[str, Any]) -> dict[str, Any]:
        ts = list(kw["timestamps_s"])
        return {
            "canonical_objects": [
                {"id": "blue_cup", "name": "Blue Cup", "desc": "Ceramic cup", "is_static": False},
                {"id": "tray", "name": "Tray", "desc": "Serving tray", "is_static": True},
            ],
            "frames": [
                {
                    "frame_index": i,
                    "time_s": float(t),
                    "objects": {
                        "blue_cup": {"visible": True, "bbox": [120 + i * 20, 200, 240 + i * 20, 360], "state": "held"},
                        "tray": {"visible": True, "bbox": [550, 300, 880, 620], "state": "stationary"},
                    },
                }
                for i, t in enumerate(ts)
            ],
        }

    cl_merge = _RecordingClient(_combined_handler)
    res_merge = run_stage1_grounding(
        view=short_view, task=cfg.get_task("_default"), client=cl_merge, cfg=cfg  # type: ignore[arg-type]
    )
    assert len(cl_merge.calls) == 1
    assert cl_merge.calls[0]["image_detail"] == "high"
    assert set(res_merge.object_tracks_2d.keys()) == {"blue_cup", "tray"}
    assert res_merge.object_tracks_2d["blue_cup"].name == "Blue Cup"
    assert len(res_merge.object_tracks_2d["blue_cup"].keyframes) == 4
    assert all(k.visible for k in res_merge.object_tracks_2d["blue_cup"].keyframes)

    # 2. Graceful fallback when merged call returns canonical_objects only (missing boxes)
    call_idx = {"n": 0}

    def _missing_boxes_handler(kw: dict[str, Any]) -> dict[str, Any]:
        call_idx["n"] += 1
        if call_idx["n"] == 1:
            return {
                "canonical_objects": [
                    {"id": "red_block", "name": "Red Block", "desc": "Cube", "is_static": False}
                ]
            }
        assert "red_block" in kw["user_prompt"]
        ts = list(kw["timestamps_s"])
        return {
            "frames": [
                {
                    "frame_index": i,
                    "time_s": float(t),
                    "objects": {
                        "red_block": {"visible": True, "bbox": [200, 200, 340, 360], "state": "ok"}
                    },
                }
                for i, t in enumerate(ts)
            ]
        }

    cl_fb1 = _RecordingClient(_missing_boxes_handler)
    res_fb1 = run_stage1_grounding(
        view=short_view, task=cfg.get_task("_default"), client=cl_fb1, cfg=cfg  # type: ignore[arg-type]
    )
    assert len(cl_fb1.calls) == 2
    assert "red_block" in res_fb1.object_tracks_2d
    assert all(k.visible for k in res_fb1.object_tracks_2d["red_block"].keyframes)

    # 3. Full fallback when merged call returns neither objects nor boxes -> separate Pass-0 (low) + Batch-0 (high)
    call_idx2 = {"n": 0}

    def _empty_first_handler(kw: dict[str, Any]) -> dict[str, Any]:
        call_idx2["n"] += 1
        if call_idx2["n"] == 1:
            return {"frames": []}
        if call_idx2["n"] == 2:
            assert kw["image_detail"] == "low"
            return {
                "canonical_objects": [
                    {"id": "spoon", "name": "Spoon", "desc": "Metal spoon", "is_static": False}
                ]
            }
        assert kw["image_detail"] == "high"
        ts = list(kw["timestamps_s"])
        return {
            "frames": [
                {
                    "frame_index": i,
                    "time_s": float(t),
                    "objects": {
                        "spoon": {"visible": True, "bbox": [300, 220, 420, 350], "state": "ok"}
                    },
                }
                for i, t in enumerate(ts)
            ]
        }

    cl_fb2 = _RecordingClient(_empty_first_handler)
    res_fb2 = run_stage1_grounding(
        view=short_view, task=cfg.get_task("_default"), client=cl_fb2, cfg=cfg  # type: ignore[arg-type]
    )
    assert len(cl_fb2.calls) == 3
    assert "spoon" in res_fb2.object_tracks_2d

    # 4. Disabling merge_pass0_short_episode forces separate Pass-0 (low) + Batch-0 (high)
    cfg_no_merge = ConfigManager()
    cfg_no_merge.settings["vlm"]["s1_grounding"]["tracking"]["enabled"] = False
    cfg_no_merge.settings["vlm"]["s1_grounding"]["merge_pass0_short_episode"] = False
    cl_sep = _RecordingClient(
        lambda kw: (
            {"canonical_objects": [{"id": "cup", "name": "Cup", "desc": "cup", "is_static": False}]}
            if "canonical_objects" in kw["system_prompt"]
            else {"objects": {"cup": {"visible": True, "bbox": [100, 100, 250, 250]}}}
        )
    )
    run_stage1_grounding(
        view=short_view, task=cfg_no_merge.get_task("_default"), client=cl_sep, cfg=cfg_no_merge  # type: ignore[arg-type]
    )
    assert len(cl_sep.calls) == 2
    assert cl_sep.calls[0]["image_detail"] == "low"
    assert cl_sep.calls[1]["image_detail"] == "high"


# ---------------------------------------------------------------------------
# Item #5: Prefix Caching, cached_tokens Telemetry & Cost Discounting
# ---------------------------------------------------------------------------


def test_prefix_caching_cached_tokens_telemetry_and_cost_discount(tmp_path: Path) -> None:
    cfg = ConfigManager()
    pricing = cfg.settings["vlm"]["pricing"]

    # 1. Cost estimation discounts cached prompt tokens at 50% by default
    # gpt-4o: in=2.50/1M, out=10.0/1M -> 100k prompt (60k cached) + 10k comp
    # uncached: 40k * 2.50 / 1M = 0.100; cached: 60k * 1.25 / 1M = 0.075; out: 10k * 10.0 / 1M = 0.100 => 0.275
    cost_default = estimate_stage_cost_usd(
        "gpt-4o", 100_000, 10_000, pricing, cached_tokens=60_000
    )
    assert cost_default == pytest.approx(0.275, abs=1e-6)

    # Custom cached_input_per_1m_usd overrides 50% default
    custom_pricing = {
        "gpt-4o": {
            "input_per_1m_usd": 2.50,
            "cached_input_per_1m_usd": 0.50,
            "output_per_1m_usd": 10.0,
        }
    }
    cost_custom = estimate_stage_cost_usd(
        "gpt-4o", 100_000, 10_000, custom_pricing, cached_tokens=60_000
    )
    # 40k*2.50/1M (0.10) + 60k*0.50/1M (0.03) + 10k*10.0/1M (0.10) = 0.23
    assert cost_custom == pytest.approx(0.23, abs=1e-6)

    # 2. Wire-level parsing for OpenAI (prompt_tokens_details.cached_tokens + prompt_cache_hit_tokens)
    #    and Gemini (usageMetadata.cachedContentTokenCount)
    srv = _MockHttpServer()
    try:
        parent_bt = BudgetTracker(pricing=pricing)
        child_bt = BudgetTracker(parent=parent_bt, pricing=pricing)
        oai = VlmClient(
            provider="openai_compat",
            base_url=srv.v1_url,
            api_key="test-key",
            retries=0,
            stage_models={"s3_discover": "gpt-4o"},
            settings={"pricing": pricing},
        )
        try:
            srv.queue.extend(
                [
                    (
                        200,
                        {
                            "choices": [{"message": {"content": '{"ok": 1}'}, "finish_reason": "stop"}],
                            "usage": {
                                "prompt_tokens": 100_000,
                                "completion_tokens": 10_000,
                                "total_tokens": 110_000,
                                "prompt_tokens_details": {"cached_tokens": 60_000},
                            },
                        },
                    ),
                    (
                        200,
                        {
                            "choices": [{"message": {"content": '{"ok": 2}'}, "finish_reason": "stop"}],
                            "usage": {
                                "prompt_tokens": 50_000,
                                "completion_tokens": 5_000,
                                "total_tokens": 55_000,
                                "prompt_cache_hit_tokens": 30_000,
                            },
                        },
                    ),
                ]
            )
            oai.chat_json(stage="s3_discover", user_prompt="w1", budget=child_bt)
            oai.chat_json(stage="s3_discover", user_prompt="w2", budget=child_bt)
        finally:
            oai.close()

        summary_oai = child_bt.to_usage_summary()
        assert summary_oai.total_cached_tokens == 90_000
        assert summary_oai.stages[0].cached_tokens == 90_000
        assert parent_bt.cached_tokens == 90_000
        # 0.275 + (20k*2.5 + 30k*1.25 + 5k*10)/1M = 0.275 + 0.1375 = 0.4125
        assert summary_oai.total_cost_usd == pytest.approx(0.4125, abs=1e-5)

        gem_bt = BudgetTracker(pricing=pricing)
        gem = VlmClient(
            provider="gemini",
            base_url=srv.v1beta_url,
            api_key="test-key",
            retries=0,
            stage_models={"s3_discover": "gemini-2.5-pro"},
            settings={"pricing": pricing},
        )
        try:
            srv.queue.append(
                (
                    200,
                    {
                        "candidates": [
                            {"content": {"parts": [{"text": '{"ok": 3}'}]}, "finishReason": "STOP"}
                        ],
                        "usageMetadata": {
                            "promptTokenCount": 100_000,
                            "candidatesTokenCount": 10_000,
                            "cachedContentTokenCount": 80_000,
                            "totalTokenCount": 110_000,
                        },
                    },
                )
            )
            gem.chat_json(stage="s3_discover", user_prompt="g1", budget=gem_bt)
        finally:
            gem.close()

        summary_gem = gem_bt.to_usage_summary()
        assert summary_gem.total_cached_tokens == 80_000
        assert summary_gem.stages[0].cached_tokens == 80_000
        # gemini-2.5-pro: in=1.25, cached=0.625, out=10.0 -> 20k*1.25 + 80k*0.625 + 10k*10.0 = 0.025 + 0.050 + 0.100 = 0.175
        assert summary_gem.total_cost_usd == pytest.approx(0.175, abs=1e-5)
    finally:
        srv.stop()

    # 3. Stable system prefix across S3 windows & S4B candidates for prefix cache hits
    pb = PromptBuilder(cfg)
    task = cfg.get_task("pick_and_place")
    roles = {"main": "main", "left_wrist": "left_wrist", "right_wrist": "right_wrist"}
    stages = [MacroStage(stage_id="S1", t_start=0.0, t_end=30.0, label="Work")]

    sys_w0, usr_w0 = pb.build_s3(
        task=task, camera_roles=roles, macro_stages=stages, window_start_s=0.0, window_end_s=16.0, duration_s=30.0
    )
    sys_w1, usr_w1 = pb.build_s3(
        task=task, camera_roles=roles, macro_stages=stages, window_start_s=12.0, window_end_s=28.0, duration_s=30.0
    )
    assert sys_w0 == sys_w1  # Byte-for-byte identical system prefix across windows!
    assert "[Active Pure-Visual QC Label Registry]" in sys_w0
    assert "0.0s" not in sys_w0 and "12.0s" not in sys_w0
    assert "Window: [0.0s, 16.0s]" in usr_w0 and "Window: [12.0s, 28.0s]" in usr_w1

    c1 = QcSegment(
        seg_id="c1", t_start=2.0, t_end=4.0, label_id="missed_grasp_retry",
        training_role="recovery", splice_policy="mask_loss", mask_side="left",
        affected_arm="left", vlm_raw_policy="mask_loss",
    )
    c2 = QcSegment(
        seg_id="c2", t_start=10.0, t_end=13.0, label_id="placement_redo",
        training_role="recovery", splice_policy="mask_loss", mask_side="right",
        affected_arm="right", vlm_raw_policy="mask_loss",
    )
    sys_b1, usr_b1 = pb.build_s4b(
        task=task, camera_roles=roles, macro_stages=stages, candidate=c1, window_start_s=0.5, window_end_s=5.5
    )
    sys_b2, usr_b2 = pb.build_s4b(
        task=task, camera_roles=roles, macro_stages=stages, candidate=c2, window_start_s=8.5, window_end_s=14.5
    )
    assert sys_b1 == sys_b2  # Byte-for-byte identical system prefix across candidates!
    assert "[Active Pure-Visual QC Label Registry]" in sys_b1
    assert "c1" in usr_b1 and "c2" in usr_b2

    # 4. Backward-compatible defaults on StageUsage & UsageSummary + DAG propagation
    legacy_su = StageUsage.model_validate({"stage": "s2_macro", "model": "gpt-4o", "total_tokens": 100})
    assert legacy_su.cached_tokens == 0
    legacy_us = UsageSummary.model_validate({"stages": [legacy_su.model_dump()], "total_tokens": 100})
    assert legacy_us.total_cached_tokens == 0

    ws = tmp_path / "ws_dag_cache_tok"
    ep_dir = tmp_path / "data" / "ds" / "ep_000"
    ep_dir.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(ep_dir / "main.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (160, 120))
    for i in range(60):
        w.write(np.full((120, 160, 3), (40 + i % 40, 80, 120), dtype=np.uint8))
    w.release()
    handle = auto_detect(tmp_path / "data" / "ds", adapter_name="video_dir")[0]

    cfg_ws = ConfigManager(workspace_dir=ws)
    cfg_ws.settings["vlm"]["s1_grounding"]["tracking"]["enabled"] = False
    repo = Repository(ws)
    try:
        with get_adapter("video_dir").open(handle) as view:
            repo.ingest_episode(view.meta)
            cl_dag = _RecordingClient(
                lambda kw: (
                    {"objects": {"target_object": {"visible": True, "bbox": [100, 100, 300, 300]}}}
                    if kw["stage"] == "s1_grounding"
                    else {"macro_stages": [], "arm_timelines": {}}
                    if kw["stage"] == "s2_macro"
                    else {"verdict": "failure", "confidence": 0.9, "criteria": []}
                    if kw["stage"].startswith("s5")
                    else {"segments": []}
                ),
                cached_tokens_per_call=400,
            )
            dag = VlmDag(workspace=ws, config=cfg_ws, client=cl_dag, repo=repo)  # type: ignore[arg-type]
            art = dag.run_episode(view)
            assert art.usage.total_cached_tokens >= 1600
            assert all(s.cached_tokens == 400 for s in art.usage.stages if not s.cache_hit)
            roundtrip = VlmRunArtifact.model_validate_json(art.model_dump_json())
            assert roundtrip.usage.total_cached_tokens == art.usage.total_cached_tokens
    finally:
        repo.close()


# ---------------------------------------------------------------------------
# Item #7: S4B Dynamic Frame Cap
# ---------------------------------------------------------------------------


def test_s4b_dynamic_frame_cap() -> None:
    # Formula: min(refine_max_frames, max(refine_min_frames, ceil((w1 - w0) / refine_step_s) + 1))
    assert _plan_s4b_frame_count(0.0, 0.6, refine_step_s=0.5, refine_min_frames=4, refine_max_frames=18) == 4
    assert _plan_s4b_frame_count(1.0, 4.5, refine_step_s=0.5, refine_min_frames=4, refine_max_frames=18) == 8
    assert _plan_s4b_frame_count(0.0, 15.0, refine_step_s=0.5, refine_min_frames=4, refine_max_frames=18) == 18
    assert _plan_s4b_frame_count(0.0, 0.6, refine_step_s=0.5, refine_min_frames=4, refine_max_frames=18, dynamic_refine_frames=False) == 18

    # End-to-end check via run_stage4b_verify
    view = _make_synth_view(duration_s=20.0)
    short_cand = QcSegment(
        seg_id="c_short",
        t_start=5.0,
        t_end=5.8,  # +1.5 margin each side => [3.5, 7.3], span=3.8s => ceil(3.8/0.5)+1 = 9 frames
        label_id="missed_grasp_retry",
        training_role="recovery",
        splice_policy="mask_loss",
        mask_side="left",
        affected_arm="left",
        vlm_raw_policy="mask_loss",
    )
    cfg_dyn = ConfigManager()
    cl_dyn = _RecordingClient(lambda _kw: {"keep_segment": True})
    run_stage4b_verify(view=view, candidates=[short_cand], client=cl_dyn, cfg=cfg_dyn)  # type: ignore[arg-type]
    assert len(cl_dyn.calls[0]["images"]) == 9

    cfg_fixed = ConfigManager()
    cfg_fixed.settings["vlm"]["s4_verify"]["dynamic_refine_frames"] = False
    cl_fixed = _RecordingClient(lambda _kw: {"keep_segment": True})
    run_stage4b_verify(view=view, candidates=[short_cand], client=cl_fixed, cfg=cfg_fixed)  # type: ignore[arg-type]
    assert len(cl_fixed.calls[0]["images"]) == 18


# ---------------------------------------------------------------------------
# Item #8: Pre-Dispatch Cost Estimator (<50ms, exact schema, cache hits)
# ---------------------------------------------------------------------------


def test_pre_dispatch_cost_estimator_schema_cache_hits_and_speed(tmp_path: Path) -> None:
    ws = tmp_path / "ws_est"
    ep_dir = tmp_path / "data" / "pick_place_ds" / "ep_0000"
    ep_dir.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(ep_dir / "main.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (160, 120))
    for i in range(90):
        w.write(np.full((120, 160, 3), (30 + i % 40, 80, 110), dtype=np.uint8))
    w.release()
    handle = auto_detect(tmp_path / "data" / "pick_place_ds", adapter_name="video_dir")[0]

    cfg = ConfigManager(workspace_dir=ws)
    cfg.settings["vlm"]["s1_grounding"]["tracking"]["enabled"] = False
    repo = Repository(ws)
    try:
        with get_adapter("video_dir").open(handle) as view:
            repo.ingest_episode(view.meta)
            ep_id = view.meta.episode_id

            # 1. Cold estimate: exact schema, zero VLM calls, <50ms
            t0 = time.perf_counter()
            est = estimate_batch_vlm_cost(cfg, repo, ws, [ep_id], stage="all", fresh=False)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            assert elapsed_ms < 50.0, f"Estimator took {elapsed_ms:.1f}ms (expected <50ms)"

            expected_top_keys = {
                "episode_count",
                "total_duration_s",
                "stage",
                "fresh",
                "estimated_prompt_tokens",
                "estimated_completion_tokens",
                "estimated_total_tokens",
                "estimated_cost_usd",
                "estimated_cache_hits",
                "per_stage",
                "token_saving_active",
            }
            assert set(est.keys()) == expected_top_keys
            assert est["episode_count"] == 1
            assert est["total_duration_s"] == pytest.approx(view.duration_s, abs=0.05)
            assert est["stage"] == "all"
            assert est["fresh"] is False
            assert est["estimated_cache_hits"] == 0
            assert est["estimated_prompt_tokens"] > 0
            assert est["estimated_completion_tokens"] > 0
            assert (
                est["estimated_total_tokens"]
                == est["estimated_prompt_tokens"] + est["estimated_completion_tokens"]
            )
            assert est["estimated_cost_usd"] > 0.0

            assert len(est["per_stage"]) == 5
            expected_stage_keys = {
                "stage",
                "model",
                "estimated_calls",
                "estimated_images",
                "image_detail",
                "estimated_tokens",
                "estimated_cost_usd",
                "cached_episodes",
            }
            by_stage = {row["stage"]: row for row in est["per_stage"]}
            for st_name in ("s1_grounding", "s2_macro", "s3_discover", "s4_verify", "s5_success"):
                assert st_name in by_stage
                assert set(by_stage[st_name].keys()) == expected_stage_keys
                assert by_stage[st_name]["model"] == "gpt-4o"
            assert by_stage["s1_grounding"]["image_detail"] == "high"
            assert by_stage["s2_macro"]["image_detail"] == "low"
            assert by_stage["s3_discover"]["image_detail"] == "auto"

            tsa = est["token_saving_active"]
            assert set(tsa.keys()) == {
                "low_detail_stages",
                "static_frame_dedup",
                "adaptive_s3_step",
                "short_episode_s1_merge",
                "s4b_dynamic_frames",
            }
            assert "s1_pass0" in tsa["low_detail_stages"]
            assert "s2_macro" in tsa["low_detail_stages"]
            assert tsa["static_frame_dedup"] is True
            assert tsa["adaptive_s3_step"] is True
            assert tsa["short_episode_s1_merge"] is True
            assert tsa["s4b_dynamic_frames"] is True

            # 2. Turning off token-saving optimizations raises estimated tokens & cost
            cfg_heavy = ConfigManager(workspace_dir=ws)
            cfg_heavy.settings["vlm"]["s2_macro"]["image_detail"] = "high"
            cfg_heavy.settings["vlm"]["s2_macro"]["static_dedup_enabled"] = False
            cfg_heavy.settings["vlm"]["s3_discover"]["static_dedup_enabled"] = False
            cfg_heavy.settings["vlm"]["s3_discover"]["adaptive_step_enabled"] = False
            cfg_heavy.settings["vlm"]["s1_grounding"]["merge_pass0_short_episode"] = False
            cfg_heavy.settings["vlm"]["s4_verify"]["dynamic_refine_frames"] = False
            est_heavy = estimate_batch_vlm_cost(cfg_heavy, repo, ws, [ep_id], stage="all", fresh=False)
            assert est_heavy["estimated_total_tokens"] > est["estimated_total_tokens"]
            assert est_heavy["estimated_cost_usd"] > est["estimated_cost_usd"]
            assert est_heavy["token_saving_active"]["static_frame_dedup"] is False
            assert est_heavy["token_saving_active"]["adaptive_s3_step"] is False
            assert est_heavy["token_saving_active"]["short_episode_s1_merge"] is False
            assert est_heavy["token_saving_active"]["s4b_dynamic_frames"] is False

            # 3. After DAG populates StageCache, fresh=False detects all 5 cache hits ($0),
            #    stage="s3_s4" hits S1/S2/S5 caches and charges only S3+S4, and fresh=True charges all 5
            cl_dag = _RecordingClient(
                lambda kw: (
                    {"objects": {"target_object": {"visible": True, "bbox": [100, 100, 300, 300]}}}
                    if kw["stage"] == "s1_grounding"
                    else {"macro_stages": [], "arm_timelines": {}}
                    if kw["stage"] == "s2_macro"
                    else {"verdict": "failure", "confidence": 0.9, "criteria": []}
                    if kw["stage"].startswith("s5")
                    else {"segments": []}
                )
            )
            dag = VlmDag(workspace=ws, config=cfg, client=cl_dag, repo=repo)  # type: ignore[arg-type]
            dag.run_episode(view)

            est_warm = estimate_batch_vlm_cost(cfg, repo, ws, [ep_id], stage="all", fresh=False)
            assert est_warm["estimated_cache_hits"] == 5
            assert est_warm["estimated_total_tokens"] == 0
            assert est_warm["estimated_cost_usd"] == 0.0

            est_s34 = estimate_batch_vlm_cost(cfg, repo, ws, [ep_id], stage="s3_s4", fresh=False)
            assert est_s34["estimated_cache_hits"] == 3  # S1, S2, S5 cached
            s34_by_st = {r["stage"]: r for r in est_s34["per_stage"]}
            assert s34_by_st["s1_grounding"]["cached_episodes"] == 1
            assert s34_by_st["s2_macro"]["cached_episodes"] == 1
            assert s34_by_st["s5_success"]["cached_episodes"] == 1
            assert s34_by_st["s3_discover"]["cached_episodes"] == 0
            assert s34_by_st["s4_verify"]["cached_episodes"] == 0
            assert est_s34["estimated_cost_usd"] > 0.0

            est_fresh = estimate_batch_vlm_cost(cfg, repo, ws, [ep_id], stage="all", fresh=True)
            assert est_fresh["estimated_cache_hits"] == 0
            assert est_fresh["estimated_cost_usd"] == pytest.approx(est["estimated_cost_usd"])

            # 4. Validation errors on unknown stage or missing episode
            with pytest.raises(ValueError, match="Invalid stage"):
                estimate_batch_vlm_cost(cfg, repo, ws, [ep_id], stage="bogus_stage")
            with pytest.raises(NotFoundError):
                estimate_batch_vlm_cost(cfg, repo, ws, ["missing/ep_999"])
    finally:
        repo.close()
