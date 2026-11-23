"""Regression tests for guardrails, VlmClient, stages, DAG, viz_mp4, and CLI."""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import yaml
from click.testing import CliRunner

from robo_vlm_qc.cli import main
from robo_vlm_qc.config import ConfigManager
from robo_vlm_qc.export.viz_mp4 import _HudCompositor
from robo_vlm_qc.guardrail import apply_splice_gate
from robo_vlm_qc.io import auto_detect, get_adapter
from robo_vlm_qc.io.base import EpisodeMeta, EpisodeView, KinematicsData
from robo_vlm_qc.models.annotation import (
    ArmTimelines,
    ConfigFingerprintSnapshot,
    GroundingResult,
    QcSegment,
    SuccessVerdict,
    VisualEvidence,
    VlmRunArtifact,
)
from robo_vlm_qc.models.episode import CameraSpec, KinematicsSpec
from robo_vlm_qc.models.review import EpisodeStatus, ResolvedEpisode
from robo_vlm_qc.store import GtFrozenError, Repository
from robo_vlm_qc.vlm.client import VlmClient, parse_json
from robo_vlm_qc.vlm.dag import VlmDag
from robo_vlm_qc.vlm.stage1_grounding import _normalize_bbox, run_stage1_grounding
from robo_vlm_qc.vlm.stage3_discover import run_stage3_discover
from robo_vlm_qc.vlm.stage4_verify import (
    _deterministic_merge_raw_segments,
    _resolve_overlaps,
    run_stage4a_merge,
    run_stage4b_verify,
)


def _seg(
    seg_id: str = "s1",
    t_start: float = 1.0,
    t_end: float = 2.0,
    arm: str = "left",
    policy: str = "cut_clean",
    label_id: str = "mid_idle_freeze",
) -> QcSegment:
    return QcSegment(
        seg_id=seg_id,
        t_start=t_start,
        t_end=t_end,
        label_id=label_id,
        training_role="badcase" if policy != "keep_positive" else "necessary",
        splice_policy=policy,  # type: ignore[arg-type]
        mask_side=arm if policy == "mask_loss" else None,  # type: ignore[arg-type]
        affected_arm=arm,  # type: ignore[arg-type]
        visual_evidence=VisualEvidence(process_change="test"),
        vlm_raw_policy=policy,  # type: ignore[arg-type]
    )


def _kin(T: int = 120, fps: float = 30.0) -> KinematicsData:
    ts = np.arange(T, dtype=np.float64) / fps
    return KinematicsData(
        timestamps_s=ts,
        eef_pos=np.zeros((T, 2, 3), dtype=np.float64),
        eef_rotvec=np.zeros((T, 2, 3), dtype=np.float64),
        hand_q=np.zeros((T, 12), dtype=np.float64),
        torso_q=np.zeros((T, 4), dtype=np.float64),
        chassis_pose=np.zeros((T, 3), dtype=np.float64),
        valid_mask=np.ones(T, dtype=bool),
    )


def test_splice_gate_nan_valid_mask_and_tail_boundary() -> None:
    cfg = ConfigManager()

    # 1. NaN + valid_mask=False at boundary frame picks finite grid candidate & roundtrips JSON
    kin1 = _kin(120)
    kin1.eef_pos[29, 0, 0] = np.nan  # type: ignore[index]
    kin1.eef_rotvec[29, 0, 0] = np.nan  # type: ignore[index]
    kin1.valid_mask[29] = False
    out1 = apply_splice_gate([_seg()], kinematics=kin1, cfg=cfg, duration_s=4.0)
    assert out1[0].splice_policy == "cut_clean"
    art1 = VlmRunArtifact(
        run_id="r1",
        episode_id="ds/ep",
        created_at=1.0,
        config_fingerprint=ConfigFingerprintSnapshot(
            model_routing={},
            prompt_bundle_version="1",
            label_taxonomy_version="1",
            task_spec_version="1",
        ),
        qc_segments=out1,
    )
    roundtrip1 = VlmRunArtifact.model_validate_json(art1.model_dump_json())
    assert roundtrip1.qc_segments[0].guardrail.metrics.eef_pos_m == 0.0

    # 2. All candidates invalid/non-finite -> finite 0.0 metrics, downgraded=True, pose_closure_failed
    kin_all_bad = _kin(120)
    kin_all_bad.valid_mask[:] = False
    kin_all_bad.eef_pos[:] = np.nan  # type: ignore[index]
    out2 = apply_splice_gate([_seg()], kinematics=kin_all_bad, cfg=cfg, duration_s=4.0)
    assert out2[0].splice_policy == "mask_loss"
    assert out2[0].guardrail.downgraded is True
    assert out2[0].guardrail.downgrade_reason == "pose_closure_failed"
    art2 = art1.model_copy(update={"qc_segments": out2})
    VlmRunArtifact.model_validate_json(art2.model_dump_json())

    # 3. NaN / valid_mask=False inside other_seg during bimanual coupling check
    kin_bim = _kin(120)
    kin_bim.eef_pos[45:50, 0, 0] = np.nan  # type: ignore[index]
    kin_bim.valid_mask[45:50] = False
    kin_bim.eef_pos[52, 0, 0] = 0.04  # type: ignore[index]
    out3 = apply_splice_gate([_seg(arm="right")], kinematics=kin_bim, cfg=cfg, duration_s=4.0)
    assert out3[0].splice_policy == "mask_loss"
    assert out3[0].guardrail.metrics.bimanual_coupled is True
    assert out3[0].guardrail.metrics.other_arm_disp_m >= 0.04

    # 4. Tail-edge boundary + tiny search_margin_s never emits t_start == t_end
    kin_tail = _kin(120)
    tail_seg = _seg(t_start=float(kin_tail.timestamps_s[-1]), t_end=4.0)
    out4 = apply_splice_gate(
        [tail_seg],
        kinematics=kin_tail,
        cfg={"search_margin_s": 0.01},
        duration_s=float(kin_tail.timestamps_s[-1]),
    )
    assert out4[0].t_end > out4[0].t_start
    QcSegment.model_validate(out4[0].model_dump())


def test_vlm_client_stage_routing_thinking_and_parse_json() -> None:
    txt = 'Here:\n```json\n{"a": 1}\n```\nNote: placeholder {foo}'
    assert parse_json(txt) == {"a": 1}

    reqs: list[dict[str, Any]] = []

    class _H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length", "0"))
            reqs.append(json.loads(self.rfile.read(n).decode("utf-8")))
            out = json.dumps(
                {
                    "choices": [{"message": {"content": '{"ok": true}'}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        client = VlmClient(
            base_url=f"http://127.0.0.1:{httpd.server_address[1]}/v1",
            api_key="test-key",
            thinking_mode="dashscope",
            stage_models={"s1": "m-s1", "s3_discover": "m-s3", "s4": "m-s4"},
        )
        try:
            assert client.model_for_stage("s1_grounding") == "m-s1"
            assert client.model_for_stage("s4a_merge") == "m-s4"
            assert client.model_for_stage("s4b_verify") == "m-s4"

            # thinking=True with DashScope mode omits response_format
            client.chat_json(
                stage="s4a_merge",
                user_prompt="merge",
                thinking=True,
                thinking_budget=None,
                max_tokens=400,
            )
            assert reqs[-1]["model"] == "m-s4"
            assert reqs[-1]["enable_thinking"] is True
            assert "response_format" not in reqs[-1]

            # thinking=False includes response_format
            client.chat_json(
                stage="s4b_verify",
                user_prompt="verify",
                thinking=False,
                max_tokens=400,
            )
            assert reqs[-1]["model"] == "m-s4"
            assert reqs[-1]["response_format"] == {"type": "json_object"}
        finally:
            client.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=2.0)


def _make_video_ds(root: Path, duration_s: float = 4.0) -> Path:
    ep_dir = root / "ds" / "ep_0000"
    ep_dir.mkdir(parents=True, exist_ok=True)
    n = int(round(duration_s * 30.0))
    w = cv2.VideoWriter(str(ep_dir / "main.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (160, 120))
    for i in range(n):
        w.write(np.full((120, 160, 3), (30 + i % 40, 80, 120), dtype=np.uint8))
    w.release()
    return root / "ds"


class _FakeClient:
    def __init__(self, handler: Any) -> None:
        self.handler = handler
        self.calls: list[dict[str, Any]] = []
        from robo_vlm_qc.vlm.client import BudgetTracker

        self.budget = BudgetTracker()

    def model_for_stage(self, stage: str, override: str | None = None) -> str:
        return override or "fake-model"

    def chat_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if kwargs.get("budget") is not None:
            kwargs["budget"].record(
                stage=kwargs.get("stage", "unknown"),
                model="fake-model",
                prompt_tokens=10,
                completion_tokens=10,
                total_tokens=20,
            )
        return self.handler(kwargs)

    def close(self) -> None:
        return


def test_stage4_overlaps_empty_s4a_and_case_insensitive_legacy(tmp_path: Path) -> None:
    outer = _seg("o1", t_start=1.0, t_end=5.0)
    inner = _seg("i1", t_start=1.2, t_end=2.0)
    res = _resolve_overlaps([outer, inner], min_len_s=0.3)
    assert len(res) == 1
    assert res[0].t_start == 1.0 and res[0].t_end == 5.0

    ds = _make_video_ds(tmp_path, duration_s=8.0)
    handle = auto_detect(ds, adapter_name="video_dir")[0]
    cfg = ConfigManager()
    a3 = _seg("a3", t_start=4.8, t_end=6.0)
    a4 = _seg("a4", t_start=5.85, t_end=6.2)

    def _s4b(kw: dict[str, Any]) -> dict[str, Any]:
        return {"keep_segment": True}

    with get_adapter("video_dir").open(handle) as view:
        verified = run_stage4b_verify(
            view=view, candidates=[a3, a4], client=_FakeClient(_s4b), cfg=cfg  # type: ignore[arg-type]
        )
        assert len(verified) == 1
        assert verified[0].t_start == 4.8 and verified[0].t_end == 6.0

        merged_empty = run_stage4a_merge(
            view=view,
            candidates=[a3],
            client=_FakeClient(lambda _kw: {"merged_segments": []}),  # type: ignore[arg-type]
            cfg=cfg,
        )
        assert merged_empty == []

        s3_out = run_stage3_discover(
            view=view,
            client=_FakeClient(
                lambda _kw: {
                    "segments": [
                        {
                            "t_start": 1.0,
                            "t_end": 2.0,
                            "label_id": "LONG_PAUSE",
                            "affected_arm": "left",
                        },
                        {
                            "t_start": 2.5,
                            "t_end": 3.5,
                            "label_id": "Back_And_Forth",
                            "affected_arm": "right",
                        },
                    ]
                }
            ),  # type: ignore[arg-type]
            cfg=cfg,
        )
        assert s3_out == []


def test_dag_reload_gt_frozen_fast_fail_and_s1_bbox_and_jitter(tmp_path: Path) -> None:
    ok_f, b_f, _ = _normalize_bbox([0.0, 0.0, 1.0, 1.0], img_w=960, img_h=540, ref_w=1280, ref_h=720, min_box_px=12)
    assert ok_f is True and b_f == [0, 0, 1000, 1000]
    ok_i, b_i, _ = _normalize_bbox([0, 0, 1, 1], img_w=960, img_h=540, ref_w=1280, ref_h=720, min_box_px=12)
    assert ok_i is False and b_i == [0, 0, 0, 0]

    ds = _make_video_ds(tmp_path / "data", duration_s=12.0)
    handle = auto_detect(ds, adapter_name="video_dir")[0]
    cfg_s1 = ConfigManager()
    boxes_s1 = [
        [400, 200, 600, 500],
        [400, 200, 600, 500],
        [400, 200, 600, 500],
        [800, 600, 950, 700],  # extreme outlier > 85px
        [810, 610, 960, 710],  # extreme outlier > 85px
    ]
    idx_s1 = {"n": 0}

    def _s1_handler(_kw: dict[str, Any]) -> dict[str, Any]:
        b = boxes_s1[idx_s1["n"] % len(boxes_s1)]
        idx_s1["n"] += 1
        return {"objects": {"bin": {"visible": True, "bbox": b, "state": "ok"}}}

    with get_adapter("video_dir").open(handle) as view:
        s1_res = run_stage1_grounding(
            view=view,
            task=cfg_s1.get_task("pick_and_place"),
            client=_FakeClient(_s1_handler),  # type: ignore[arg-type]
            cfg=cfg_s1,
        )
        container = s1_res.object_tracks_2d["bin"]
        assert container.keyframes[0].bbox_norm_1000 == [400, 200, 600, 500]

    ws = tmp_path / "ws"
    ov_prompts = ws / "configs_override" / "prompts"
    ov_prompts.mkdir(parents=True, exist_ok=True)
    base_s3 = (Path(__file__).resolve().parents[2] / "configs" / "prompts" / "s3_discover.yaml").read_text(encoding="utf-8")
    d_s3 = yaml.safe_load(base_s3)
    d_s3["user_template"] = d_s3["user_template"] + "\nRELOAD_MARKER_V1"
    (ov_prompts / "s3_discover.yaml").write_text(yaml.safe_dump(d_s3, allow_unicode=True), encoding="utf-8")

    seen_s3_prompts: list[str] = []

    def _dag_handler(kw: dict[str, Any]) -> dict[str, Any]:
        st = kw["stage"]
        if st == "s1_grounding":
            return {"objects": {}}
        if st == "s2_macro":
            return {"macro_stages": [], "arm_timelines": {}}
        if st.startswith("s5"):
            return {"verdict": "failure", "confidence": 0.9, "criteria": []}
        if st == "s3_discover":
            seen_s3_prompts.append(kw["user_prompt"])
            return {"segments": []}
        return {"merged_segments": []}

    repo = Repository(ws)
    try:
        with get_adapter("video_dir").open(handle) as view:
            repo.ingest_episode(view.meta)
            fake_cl = _FakeClient(_dag_handler)
            dag = VlmDag(workspace=ws, client=fake_cl, repo=repo)  # type: ignore[arg-type]
            dag.run_episode(view)
            assert any("RELOAD_MARKER_V1" in p for p in seen_s3_prompts)

            d_s3["user_template"] = d_s3["user_template"].replace("RELOAD_MARKER_V1", "RELOAD_MARKER_V2")
            (ov_prompts / "s3_discover.yaml").write_text(yaml.safe_dump(d_s3, allow_unicode=True), encoding="utf-8")
            dag.cfg.reload()
            seen_s3_prompts.clear()
            dag.run_episode(view)
            assert any("RELOAD_MARKER_V2" in p for p in seen_s3_prompts)

            repo.transition_status(view.meta.episode_id, EpisodeStatus.APPROVED)
            repo.transition_status(view.meta.episode_id, EpisodeStatus.GT_FROZEN)
            calls_before = len(fake_cl.calls)
            with pytest.raises(GtFrozenError):
                dag.run_episode(view)
            assert len(fake_cl.calls) == calls_before

            art_sb = dag.run_episode(view, is_eval_sandbox=True)
            assert art_sb.is_eval_sandbox is True
    finally:
        repo.close()


def test_viz_mp4_torso_head_fallback_and_cli_stage_validation(tmp_path: Path) -> None:
    class _DummyMedia:
        cameras = [CameraSpec(name="head", role="main", width=160, height=120, fps=30.0)]
        duration_s = 1.0
        fps = 30.0

        def read_frame_rgb(self, camera: str, t_s: float) -> np.ndarray:
            return np.zeros((120, 160, 3), dtype=np.uint8)

        def read_frames_batch(self, camera: str, times_s: list[float]) -> list[np.ndarray]:
            return [np.zeros((120, 160, 3), dtype=np.uint8) for _ in times_s]

    meta = EpisodeMeta(
        episode_id="ds/ep",
        dataset="ds",
        stem="ep",
        task_id="_default",
        adapter_type="video_dir",
        source_uri="/tmp/x",
        source_mtime=1.0,
        episode_fingerprint="fp_dummy_1234",
        duration_s=1.0,
        fps=30.0,
        frame_count=30,
        cameras=[
            CameraSpec(name="head", role="main", width=160, height=120, fps=30.0),
            CameraSpec(name="aux", role="secondary", width=160, height=120, fps=30.0),
            CameraSpec(name="wr", role="wide_right", width=160, height=120, fps=30.0),
        ],
        camera_roles={"main": "head", "secondary": "aux", "wide_right": "wr"},
        kinematics=KinematicsSpec(has_kinematics=False),
    )
    resolved = ResolvedEpisode(
        episode_id="ds/ep",
        meta=meta,
        status=EpisodeStatus.VLM_READY,
        qc_segments=[],
        macro_stages=[],
        grounding=GroundingResult(),
        arm_timelines=ArmTimelines(),
        raw_vlm_segments=[],
        success=SuccessVerdict(verdict="failure", passed=False, confidence=0.9),
    )
    view = EpisodeView(meta=meta, media=_DummyMedia())  # type: ignore[arg-type]
    comp = _HudCompositor(resolved, view)
    assert comp.tile_mode == "torso_head_2x2"
    fr = comp.render_frame_bgr(0, 30, 0.0, np.zeros((120, 160, 3), dtype=np.uint8))
    assert fr.shape == (1080, 1920, 3)

    runner = CliRunner()
    res_bad = runner.invoke(main, ["-w", str(tmp_path / "ws"), "run-vlm", "--stage", "bad_stage"])
    assert res_bad.exit_code == 2

    ds = _make_video_ds(tmp_path / "cli_data", duration_s=1.5)
    ws_cli = tmp_path / "cli_ws"
    res_ing = runner.invoke(main, ["-w", str(ws_cli), "ingest", "--data", str(ds), "--json"])
    assert res_ing.exit_code == 0, res_ing.output

    stage_calls: list[str] = []

    class _MockSrv(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n).decode("utf-8"))
            sys_txt = str(body.get("messages", [{}])[0].get("content", ""))
            if "2D Grounding" in sys_txt or '"objects":' in sys_txt:
                stage_calls.append("s1")
                payload_txt = '{"objects": {}}'
            elif "macro_stages" in sys_txt and "arm_timelines" in sys_txt:
                stage_calls.append("s2")
                payload_txt = '{"macro_stages": [], "arm_timelines": {}}'
            elif "satisfied_at_s" in sys_txt:
                stage_calls.append("s5")
                payload_txt = '{"verdict": "failure", "confidence": 0.9, "criteria": []}'
            elif "Precision-First" in sys_txt or "unknown_note" in sys_txt:
                stage_calls.append("s3")
                payload_txt = '{"segments": [{"t_start": 0.2, "t_end": 1.0, "label_id": "mid_idle_freeze", "affected_arm": "left"}]}'
            elif "Stage 4A" in sys_txt or "merged_segments" in sys_txt:
                stage_calls.append("s4a")
                payload_txt = '{"merged_segments": [{"t_start": 0.2, "t_end": 1.0, "label_id": "mid_idle_freeze", "affected_arm": "left"}]}'
            else:
                stage_calls.append("s4b")
                payload_txt = '{"keep_segment": true}'
            out = json.dumps(
                {
                    "choices": [{"message": {"content": payload_txt}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _MockSrv)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        ov_dir = ws_cli / "configs_override"
        ov_dir.mkdir(parents=True, exist_ok=True)
        (ov_dir / "default.yaml").write_text(
            f"vlm:\n  base_url: http://127.0.0.1:{httpd.server_address[1]}/v1\n  api_key_env: TEST_DUMMY_KEY\n",
            encoding="utf-8",
        )
        old_key = os.environ.get("TEST_DUMMY_KEY")
        os.environ["TEST_DUMMY_KEY"] = "dummy-key"
        try:
            r1 = runner.invoke(main, ["-w", str(ws_cli), "run-vlm", "--workers", "1", "--json"])
            assert r1.exit_code == 0, r1.output
            first_calls = len(stage_calls)
            assert first_calls >= 4

            stage_calls.clear()
            r2 = runner.invoke(main, ["-w", str(ws_cli), "run-vlm", "--stage", "s3", "--workers", "1", "--json"])
            assert r2.exit_code == 0, r2.output
            assert "s3" in stage_calls
            assert "s1" not in stage_calls
            assert "s2" not in stage_calls
            assert "s5" not in stage_calls
        finally:
            if old_key is None:
                os.environ.pop("TEST_DUMMY_KEY", None)
            else:
                os.environ["TEST_DUMMY_KEY"] = old_key
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=2.0)


def test_short_edge_boundary_idle_preserved_while_short_interior_filtered(
    tmp_path: Path,
) -> None:
    ds = _make_video_ds(tmp_path, duration_s=4.0)
    handle = auto_detect(ds, adapter_name="video_dir")[0]
    cfg = ConfigManager()

    edge_head = _seg("b_head", t_start=0.0, t_end=0.2, arm="both", label_id="boundary_idle")
    edge_tail = _seg("b_tail", t_start=3.8, t_end=4.0, arm="both", label_id="boundary_idle")
    short_int_mid = _seg("m_int", t_start=0.8, t_end=1.0, arm="both", label_id="mid_idle_freeze")
    short_int_bnd = _seg("b_int", t_start=1.8, t_end=2.0, arm="both", label_id="boundary_idle")

    merged_fb = _deterministic_merge_raw_segments(
        [edge_head, short_int_mid, short_int_bnd, edge_tail],
        min_len_s=0.35,
        duration_s=4.0,
    )
    assert [(s.t_start, s.t_end, s.label_id) for s in merged_fb] == [
        (0.0, 0.2, "boundary_idle"),
        (3.8, 4.0, "boundary_idle"),
    ]

    resolved_ov = _resolve_overlaps(
        [edge_head, short_int_mid, short_int_bnd, edge_tail],
        min_len_s=0.3,
        duration_s=4.0,
    )
    assert [(s.t_start, s.t_end, s.label_id) for s in resolved_ov] == [
        (0.0, 0.2, "boundary_idle"),
        (3.8, 4.0, "boundary_idle"),
    ]

    with get_adapter("video_dir").open(handle) as view:
        s3_out = run_stage3_discover(
            view=view,
            client=_FakeClient(
                lambda _kw: {
                    "segments": [
                        {
                            "t_start": 0.0,
                            "t_end": 0.2,
                            "label_id": "boundary_idle",
                            "affected_arm": "both",
                        },
                        {
                            "t_start": 0.8,
                            "t_end": 1.0,
                            "label_id": "mid_idle_freeze",
                            "affected_arm": "both",
                        },
                        {
                            "t_start": 1.8,
                            "t_end": 2.0,
                            "label_id": "boundary_idle",
                            "affected_arm": "both",
                        },
                    ]
                }
            ),  # type: ignore[arg-type]
            cfg=cfg,
        )
        assert [(s.t_start, s.t_end, s.label_id) for s in s3_out] == [
            (0.0, 0.2, "boundary_idle"),
        ]

        s4b_out = run_stage4b_verify(
            view=view,
            candidates=[edge_head, short_int_mid, short_int_bnd],
            client=_FakeClient(lambda _kw: {"keep_segment": True}),  # type: ignore[arg-type]
            cfg=cfg,
        )
        assert [(s.t_start, s.t_end, s.label_id) for s in s4b_out] == [
            (0.0, 0.2, "boundary_idle"),
        ]

    # Exact 0.1s IEEE-754 trailing edge [4.0, 4.1] at dur=4.1 is preserved
    ds_41 = _make_video_ds(tmp_path / "ds41", duration_s=4.1)
    handle_41 = auto_detect(ds_41, adapter_name="video_dir")[0]
    tail_41 = _seg("b_tail_41", t_start=4.0, t_end=4.1, arm="both", label_id="boundary_idle")
    assert [
        (s.t_start, s.t_end, s.label_id)
        for s in _deterministic_merge_raw_segments([tail_41], min_len_s=0.35, duration_s=4.1)
    ] == [(4.0, 4.1, "boundary_idle")]
    assert [
        (s.t_start, s.t_end, s.label_id)
        for s in _resolve_overlaps([tail_41], min_len_s=0.3, duration_s=4.1)
    ] == [(4.0, 4.1, "boundary_idle")]
    with get_adapter("video_dir").open(handle_41) as view_41:
        s3_41 = run_stage3_discover(
            view=view_41,
            client=_FakeClient(
                lambda _kw: {
                    "segments": [
                        {
                            "t_start": 4.0,
                            "t_end": 4.1,
                            "label_id": "boundary_idle",
                            "affected_arm": "both",
                        }
                    ]
                }
            ),  # type: ignore[arg-type]
            cfg=cfg,
        )
        assert [(s.t_start, s.t_end, s.label_id) for s in s3_41] == [
            (4.0, 4.1, "boundary_idle"),
        ]
        s4b_41 = run_stage4b_verify(
            view=view_41,
            candidates=[tail_41],
            client=_FakeClient(lambda _kw: {"keep_segment": True}),  # type: ignore[arg-type]
            cfg=cfg,
        )
        assert [(s.t_start, s.t_end, s.label_id) for s in s4b_41] == [
            (4.0, 4.1, "boundary_idle"),
        ]
