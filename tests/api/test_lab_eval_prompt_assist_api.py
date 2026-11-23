"""Integration tests for eval_runner, prompt_assist, Lab API router, and `cli.py eval`."""

from __future__ import annotations

import json
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import yaml
from click.testing import CliRunner
from fastapi.testclient import TestClient

from robo_vlm_qc.api import create_app
from robo_vlm_qc.cli import main
from robo_vlm_qc.config import default_config_dir
from robo_vlm_qc.io import auto_detect, get_adapter
from robo_vlm_qc.lab.eval_runner import compare_eval_runs, promote_candidate_config, run_gt_eval
from robo_vlm_qc.lab.prompt_assist import suggest_prompt_updates
from robo_vlm_qc.models.annotation import (
    ArmTimelines,
    ConfigFingerprintSnapshot,
    GroundingResult,
    MacroStage,
    QcSegment,
    SuccessVerdict,
    VlmRunArtifact,
)
from robo_vlm_qc.models.review import EpisodeStatus, ReviewWritePayload, WatchCoverage
from robo_vlm_qc.store import NotFoundError, Repository
from robo_vlm_qc.vlm.client import VlmError


def _sanitize_vlm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "mock-test-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    for var in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        monkeypatch.delenv(var, raising=False)


class _LocalMockVlm:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests: list[dict[str, Any]] = []
        self.overrides: dict[str, dict[str, Any]] = {}
        self.fail_count = 0

        parent = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                msgs = body.get("messages", [])
                text = "\n".join(
                    b.get("text", "") if isinstance(b, dict) else ""
                    for m in msgs
                    for b in (
                        m["content"]
                        if isinstance(m.get("content"), list)
                        else [{"type": "text", "text": str(m.get("content", ""))}]
                    )
                )
                low = text.lower()
                if (
                    "suggested_prompt_diff" in text
                    or "prompt_assist" in text
                    or "prompt revision" in low
                    or "prompt refinement" in low
                ):
                    stage = "PROMPT_ASSIST"
                elif (
                    "anti-false-positive" in low
                    or "anti-fp" in low
                    or "early control" in low
                    or "truncated at" in low
                    or "truncated to the first" in low
                    or "control verifier" in low
                ):
                    stage = "S5_CONTROL"
                elif "Stage 4A" in text or "merged_segments" in text:
                    stage = "S4A"
                elif "Stage 4B" in text or "keep_segment" in text:
                    stage = "S4B"
                elif (
                    "Precision-First" in text
                    or "unknown_note" in text
                    or "sliding window" in low
                ):
                    stage = "S3"
                elif "macro_stages" in text and "arm_timelines" in text:
                    stage = "S2"
                elif "2D Grounding" in text or "bbox_norm_1000" in text or '"objects":' in text:
                    stage = "S1"
                elif "satisfied_at_s" in text or "completion criteria" in low:
                    stage = "S5"
                else:
                    stage = "UNKNOWN"

                with parent._lock:
                    parent.requests.append({"stage": stage, "text": text})
                    should_fail = parent.fail_count > 0
                    if should_fail:
                        parent.fail_count -= 1
                    ov = parent.overrides.get(stage)

                if should_fail:
                    raw = b'{"error":{"message":"injected 500"}}'
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return

                if ov is not None:
                    payload = ov
                elif stage == "S1":
                    payload = {
                        "objects": {
                            "target_object": {
                                "visible": True,
                                "bbox": [100, 100, 300, 300],
                                "state": "ok",
                                "status_kind": "neutral",
                            }
                        }
                    }
                elif stage == "S2":
                    payload = {
                        "macro_stages": [
                            {
                                "stage_id": "S1",
                                "t_start": 0.0,
                                "t_end": 2.0,
                                "label": "Operation",
                                "label_en": "act",
                                "color_rgb": [59, 130, 246],
                            }
                        ],
                        "arm_timelines": {
                            "left": [
                                {"t_start": 0.0, "t_end": 2.0, "state": "work", "desc": "work"}
                            ],
                            "right": [],
                        },
                    }
                elif stage == "S5":
                    payload = {
                        "verdict": "success",
                        "confidence": 0.9,
                        "criteria": [
                            {
                                "criterion": "done",
                                "met": True,
                                "satisfied_at_s": 1.6,
                                "evidence": "ok",
                            }
                        ],
                        "summary": "ok",
                    }
                elif stage == "S5_CONTROL":
                    payload = {
                        "verdict": "failure",
                        "confidence": 0.9,
                        "criteria": [
                            {
                                "criterion": "done",
                                "met": False,
                                "satisfied_at_s": None,
                                "evidence": "early",
                            }
                        ],
                        "summary": "early",
                    }
                elif stage == "S3":
                    payload = {
                        "segments": [
                            {
                                "t_start": 0.2,
                                "t_end": 1.0,
                                "label_id": "missed_grasp_retry",
                                "affected_arm": "left",
                                "splice_policy": "mask_loss",
                                "visual_evidence": {
                                    "initial_state": "a",
                                    "process_change": "b",
                                    "end_state": "c",
                                },
                                "confidence": 0.85,
                                "unknown_note": "",
                            }
                        ]
                    }
                elif stage == "S4A":
                    payload = {"merged_segments": []}
                elif stage == "S4B":
                    payload = {
                        "keep_segment": True,
                        "t_start": 0.2,
                        "t_end": 1.0,
                        "label_id": "missed_grasp_retry",
                        "affected_arm": "left",
                        "splice_policy": "mask_loss",
                    }
                else:
                    payload = {
                        "suggested_prompt_diff": "Add slow motion exemption",
                        "rationale": "Reduce FP",
                    }

                resp_obj = {
                    "id": "mock_1",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(payload, ensure_ascii=False),
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
                }
                raw = json.dumps(resp_obj, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _setup_workspace(
    tmp_path: Path, base_url: str, *, with_unknown: bool = False
) -> tuple[Path, Path, str, str]:
    ws = tmp_path / "ws"
    cfg_dir = tmp_path / "configs"
    # Copy base configs into tmp_path so tests never touch repo configs
    shutil.copytree(default_config_dir(), cfg_dir)
    default_yaml = yaml.safe_load((cfg_dir / "default.yaml").read_text("utf-8"))
    default_yaml["vlm"]["base_url"] = base_url
    default_yaml["vlm"]["api_key_env"] = "OPENAI_API_KEY"
    default_yaml["vlm"]["key_file"] = "__mock_unused__.env"
    default_yaml["vlm"]["retries"] = 0
    default_yaml["vlm"]["retry_min_wait_s"] = 0.01
    default_yaml["vlm"]["retry_max_wait_s"] = 0.02
    (cfg_dir / "default.yaml").write_text(
        yaml.safe_dump(default_yaml, sort_keys=False), encoding="utf-8"
    )

    ds_dir = tmp_path / "data" / "lab_demo_ds"
    ds_dir.mkdir(parents=True, exist_ok=True)
    mp4 = ds_dir / "ep_01.mp4"
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    assert writer.isOpened()
    try:
        for i in range(20):
            writer.write(np.full((120, 160, 3), (i * 10) % 255, dtype=np.uint8))
    finally:
        writer.release()

    handles = auto_detect(ds_dir)
    with get_adapter(handles[0].adapter_type).open(handles[0]) as view:
        meta = view.meta.model_copy(update={"task_id": "pick_and_place"})

    repo = Repository(ws)
    try:
        ep_id, _ = repo.ingest_episode(meta)
        segs = [
            QcSegment(
                seg_id="seg_fp",
                t_start=0.2,
                t_end=1.0,
                label_id="missed_grasp_retry",
                training_role="recovery",
                splice_policy="mask_loss",
                mask_side="left",
                affected_arm="left",
                vlm_raw_policy="mask_loss",
            )
        ]
        if with_unknown:
            segs.append(
                QcSegment(
                    seg_id="seg_unk_1",
                    t_start=1.1,
                    t_end=1.7,
                    label_id="unknown_anomaly",
                    training_role="badcase",
                    splice_policy="mask_loss",
                    mask_side="right",
                    affected_arm="right",
                    vlm_raw_policy="mask_loss",
                    is_unknown=True,
                    unknown_note="Cable snag",
                )
            )
        run = VlmRunArtifact(
            run_id="run_orig_1",
            episode_id=ep_id,
            created_at=time.time(),
            config_fingerprint=ConfigFingerprintSnapshot(
                model_routing={"s1_grounding": "gpt-4o"},
                prompt_bundle_version="1.0.0",
                label_taxonomy_version="v1",
                task_spec_version="1.0.0",
            ),
            grounding=GroundingResult(),
            macro_stages=[
                MacroStage(stage_id="S1", t_start=0.0, t_end=2.0, label="Operation", label_en="act")
            ],
            arm_timelines=ArmTimelines(),
            qc_segments=segs,
            success=SuccessVerdict(passed=True, verdict="success", confidence=0.9),
        )
        repo.save_vlm_run_artifact(run)
        repo.save_review(
            ep_id,
            "lead_u",
            "lead",
            ReviewWritePayload(
                base_run_id="run_orig_1",
                action="freeze_gt",
                qc_segments=[s for s in segs if s.seg_id == "seg_unk_1"] if with_unknown else [],
                success=SuccessVerdict(passed=True, verdict="success", confidence=1.0),
                macro_stages=run.macro_stages,
                watch_coverage=WatchCoverage(
                    watched_intervals=[(0.0, 2.0)],
                    watched_s=2.0,
                    duration_s=2.0,
                    coverage_ratio=1.0,
                    confirmed=True,
                ),
            ),
        )
    finally:
        repo.close()
    return ws, cfg_dir, ep_id, "run_orig_1"


def test_eval_runner_cache_hit_compare_promote_and_prompt_assist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sanitize_vlm_env(monkeypatch)
    mock = _LocalMockVlm()
    try:
        ws, cfg_dir, ep_id, orig_run_id = _setup_workspace(tmp_path, mock.base_url)

        # Run 1 (populates StageCache S1..S5)
        mock.overrides["S4A"] = {
            "merged_segments": [
                {
                    "t_start": 0.2,
                    "t_end": 1.0,
                    "label_id": "missed_grasp_retry",
                    "affected_arm": "left",
                    "splice_policy": "mask_loss",
                }
            ]
        }
        ev1 = run_gt_eval(ws, config_dir=cfg_dir, few_shot=False)
        assert ev1.metrics.episodes_evaluated == 1
        assert ev1.metrics.f1 == 0.0  # FP on empty GT

        # Verify gt_frozen & active_run_id untouched
        repo = Repository(ws)
        try:
            row = repo.get_episode_row(ep_id)
            assert row["status"] == EpisodeStatus.GT_FROZEN.value
            assert row["active_run_id"] == orig_run_id
        finally:
            repo.close()

        # Run 2 with candidate_prompts (in-memory overlay) + empty S3 -> S1/S2/S5 100% cache hit!
        mock.requests.clear()
        mock.overrides["S3"] = {"segments": []}
        mock.overrides["S4A"] = {"merged_segments": []}
        ev2 = run_gt_eval(
            ws,
            config_dir=cfg_dir,
            few_shot=True,
            candidate_prompts={
                "s3_discover": {"version": "2.0.0", "user_template": "Updated\n{{ task_id }}"}
            },
        )
        called_stages = [r["stage"] for r in mock.requests]
        assert not any(s in {"S1", "S2", "S5", "S5_CONTROL"} for s in called_stages), called_stages
        assert "S3" in called_stages
        assert ev2.metrics.f1 == 1.0
        # Candidate overlay did NOT mutate configs_override/ yet
        assert not (ws / "configs_override" / "prompts" / "s3_discover.yaml").exists()

        cmp_res = compare_eval_runs(ws, ev1.eval_id, ev2.eval_id)
        assert cmp_res["deltas"]["delta_f1"] == 1.0

        # Nonexistent eval_id raises NotFoundError
        with pytest.raises(NotFoundError):
            promote_candidate_config(ws, config_dir=cfg_dir, eval_id="eval_missing_xyz")

        # Promote ev2 -> writes configs_override/prompts/s3_discover.yaml AND creates .history/ on first override
        prom = promote_candidate_config(
            ws, config_dir=cfg_dir, eval_id=ev2.eval_id, reason="unit_promote"
        )
        assert prom["ok"] is True
        assert "configs_override/prompts/s3_discover.yaml" in prom["promoted_files"]
        assert list((ws / ".history").rglob("*.json"))

        # Prompt assist calls VLM with cue, never writes configs_override
        mock.requests.clear()
        sugg = suggest_prompt_updates(ws, config_dir=cfg_dir, stage="s3_discover")
        assert "suggested_prompt_diff" in sugg
        assert len(mock.requests) == 1

        # Prompt assist VLM failure raises clean VlmError
        mock.fail_count = 3
        with pytest.raises(VlmError):
            suggest_prompt_updates(ws, config_dir=cfg_dir, stage="s3_discover")
    finally:
        mock.close()


def test_lab_api_full_surface_rbac_triage_few_shot_eval_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sanitize_vlm_env(monkeypatch)
    mock = _LocalMockVlm()
    try:
        ws, cfg_dir, ep_id, _ = _setup_workspace(tmp_path, mock.base_url, with_unknown=True)
        app = create_app(ws, config_dir=cfg_dir, reviewer_token="rev-tok", lead_token="lead-tok")
        client = TestClient(app)
        rev_h = {"X-Auth-Token": "rev-tok"}
        lead_h = {"X-Auth-Token": "lead-tok"}

        # RBAC: Reviewer gets 403 on all lab endpoints, Lead gets 200
        for method, path in (
            ("GET", "/api/lab/labels"),
            ("GET", "/api/lab/prompts"),
            ("GET", "/api/lab/tasks"),
            ("GET", "/api/lab/diagnosis"),
            ("GET", "/api/lab/unknowns"),
            ("GET", "/api/lab/few-shots"),
            ("GET", "/api/lab/eval/compare"),
        ):
            assert client.request(method, path, headers=rev_h).status_code == 403
        assert client.get("/api/lab/labels").status_code == 401

        # Diagnosis supports single, comma-separated, and repeated episode_ids
        r_diag = client.get(
            "/api/lab/diagnosis", params={"episode_ids": ep_id, "top_n": 5}, headers=lead_h
        )
        assert r_diag.status_code == 200
        assert "fp_overflag" in json.dumps(r_diag.json())

        # Unknowns pool & duplicate promote -> 409 (never 422)
        r_unk = client.get("/api/lab/unknowns", headers=lead_h)
        assert r_unk.status_code == 200
        assert r_unk.json()["total"] == 1
        assert r_unk.json()["items"][0]["seg_id"] == "seg_unk_1"

        r_dup = client.post(
            "/api/lab/unknowns/triage",
            headers=lead_h,
            json={
                "episode_id": ep_id,
                "seg_id": "seg_unk_1",
                "action": "promote",
                "new_label": {
                    "id": "boundary_idle",
                    "name": "Duplicate",
                    "category": "idle",
                    "default_training_role": "badcase",
                    "default_splice_policy": "cut_clean",
                    "collector_ui": {"title": "dup"},
                    "vlm_semantics": {"definition": "dup"},
                },
            },
        )
        assert r_dup.status_code == 409

        # Promote new label on first override creates .history/ AND increments version
        r_promo_lbl = client.post(
            "/api/lab/unknowns/triage",
            headers=lead_h,
            json={
                "episode_id": ep_id,
                "seg_id": "seg_unk_1",
                "action": "promote",
                "new_label": {
                    "id": "cable_snag_custom",
                    "name": "Cable Snag",
                    "category": "failure",
                    "default_training_role": "badcase",
                    "default_splice_policy": "mask_loss",
                    "collector_ui": {"title": "Cable Snag"},
                    "vlm_semantics": {"definition": "Cable catches on a tabletop object"},
                },
            },
        )
        assert r_promo_lbl.status_code == 200
        assert r_promo_lbl.json()["label_version"] == 2
        assert list((ws / ".history").rglob("*.json"))
        # Unknown is now triaged and removed from active pool
        assert client.get("/api/lab/unknowns", headers=lead_h).json()["total"] == 0

        # PUT /api/lab/labels rejects duplicate IDs with 422
        lbls = client.get("/api/lab/labels", headers=lead_h).json()
        bad_lbls = dict(lbls, labels=list(lbls["labels"]) + [lbls["labels"][0]])
        assert client.put("/api/lab/labels", headers=lead_h, json=bad_lbls).status_code == 422

        # GET/PUT prompts & tasks
        prompts = client.get("/api/lab/prompts", headers=lead_h).json()
        prompts["s3_discover"]["version"] = "1.1.0"
        r_put_p = client.put(
            "/api/lab/prompts", headers=lead_h, json={"s3_discover": prompts["s3_discover"]}
        )
        assert r_put_p.status_code == 200
        assert r_put_p.json()["s3_discover"]["version"] == "1.1.0"

        tasks = client.get("/api/lab/tasks", headers=lead_h).json()["tasks"]
        tasks["pick_and_place"]["version"] = "1.1.0"
        r_put_t = client.put("/api/lab/tasks", headers=lead_h, json=tasks["pick_and_place"])
        assert r_put_t.status_code == 200
        assert r_put_t.json()["tasks"]["pick_and_place"]["version"] == "1.1.0"

        # Few-Shots POST, GET, PATCH, /tile
        r_fs_bad = client.post(
            "/api/lab/few-shots",
            headers=lead_h,
            json={
                "scope": "global",
                "error_type": "fp_overflag",
                "episode_id": ep_id,
                "t_start": 1.5,
                "t_end": 1.0,
                "rationale": "bad",
            },
        )
        assert r_fs_bad.status_code == 422

        r_fs = client.post(
            "/api/lab/few-shots",
            headers=lead_h,
            json={
                "scope": "task:pick_and_place",
                "error_type": "fp_overflag",
                "episode_id": ep_id,
                "t_start": 0.2,
                "t_end": 1.0,
                "rationale": "Good motion",
            },
        )
        assert r_fs.status_code == 200
        ex_id = r_fs.json()["example_id"]
        r_tile = client.get(f"/api/lab/few-shots/{ex_id}/tile", headers=lead_h)
        assert r_tile.status_code == 200
        assert r_tile.headers["content-type"] == "image/jpeg"

        r_patch = client.patch(
            f"/api/lab/few-shots/{ex_id}", headers=lead_h, json={"enabled": False}
        )
        assert r_patch.status_code == 200
        assert r_patch.json()["enabled"] is False

        # Eval + Compare + Promote (404 on unknown eval_id)
        r_ev1 = client.post("/api/lab/eval", headers=lead_h, json={"few_shot": False})
        r_ev2 = client.post("/api/lab/eval", headers=lead_h, json={"few_shot": True})
        assert r_ev1.status_code == 200 and r_ev2.status_code == 200

        r_cmp = client.get(
            "/api/lab/eval/compare",
            params={"baseline": r_ev1.json()["eval_id"], "candidate": r_ev2.json()["eval_id"]},
            headers=lead_h,
        )
        assert r_cmp.status_code == 200
        assert "deltas" in r_cmp.json()

        assert (
            client.post(
                "/api/lab/promote", headers=lead_h, json={"eval_id": "missing_eval_000"}
            ).status_code
            == 404
        )
        assert (
            client.post(
                "/api/lab/promote", headers=lead_h, json={"eval_id": r_ev2.json()["eval_id"]}
            ).status_code
            == 200
        )

        # Prompt assist endpoint
        r_pa = client.post("/api/lab/prompt-assist", headers=lead_h, json={"stage": "s3_discover"})
        assert r_pa.status_code == 200
        assert "suggested_prompt_diff" in r_pa.json()

        # CLI `eval` exit codes: missing workspace -> 2; gt_frozen -> 0 with --json
        runner = CliRunner()
        r_missing = runner.invoke(main, ["-w", str(tmp_path / "no_ws"), "-c", str(cfg_dir), "eval"])
        assert r_missing.exit_code == 2

        r_cli_eval = runner.invoke(main, ["-w", str(ws), "-c", str(cfg_dir), "eval", "--json"])
        assert r_cli_eval.exit_code == 0, r_cli_eval.output
        cli_json = json.loads(r_cli_eval.output)
        assert cli_json["ok"] is True
        assert cli_json["metrics"]["episodes_evaluated"] == 1
    finally:
        mock.close()
