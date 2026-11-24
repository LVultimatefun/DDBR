"""Backend adversarial and regression tests covering VLM, guardrails, reviews, I/O, and exports."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from robo_vlm_qc.api import create_app
from robo_vlm_qc.config import ConfigManager
from robo_vlm_qc.export import PieceSlicer, get_exporter
from robo_vlm_qc.export.slicer import smooth_tail_81d, tail_weights
from robo_vlm_qc.guardrail import apply_hallucination_check, apply_splice_gate, apply_visual_gate
from robo_vlm_qc.io import auto_detect, get_adapter
from robo_vlm_qc.io.base import EpisodeView, KinematicsData
from robo_vlm_qc.models.annotation import (
    ConfigFingerprintSnapshot,
    QcSegment,
    SuccessVerdict,
    VlmRunArtifact,
)
from robo_vlm_qc.models.episode import CameraSpec, EpisodeMeta, KinematicsSpec
from robo_vlm_qc.models.export import ExportRecipe
from robo_vlm_qc.models.review import (
    EpisodeStatus,
    ResolvedEpisode,
    ReviewWritePayload,
    WatchCoverage,
)
from robo_vlm_qc.store.repository import Repository
from robo_vlm_qc.vlm.client import BudgetTracker, VlmError, parse_json
from robo_vlm_qc.vlm.dag import VlmDag
from robo_vlm_qc.vlm.stage1_grounding import run_stage1_grounding
from robo_vlm_qc.vlm.stage3_discover import run_stage3_discover
from robo_vlm_qc.vlm.stage4_verify import _is_short_edge_boundary_idle


def _seg(
    seg_id: str = "s1",
    t_start: float = 0.0,
    t_end: float = 0.3,
    label_id: str = "boundary_idle",
    policy: str = "cut_clean",
    arm: str = "both",
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
        vlm_raw_policy=policy,  # type: ignore[arg-type]
    )


def _write_mp4(path: Path, n_frames: int = 30, fps: float = 30.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64))
    assert w.isOpened()
    try:
        for i in range(n_frames):
            w.write(np.full((64, 64, 3), (i * 11) % 255, dtype=np.uint8))
    finally:
        w.release()


def test_ieee754_edge_tol_boundary_in_vlm_and_guardrails() -> None:
    """Exact trailing-edge segments must not fail due to unrounded duration_s - edge_tol_s."""
    # 2.45 - 0.15 == 2.3000000000000003 in IEEE-754 float64
    seg_245 = _seg(t_start=2.15, t_end=2.3, label_id="boundary_idle")
    assert _is_short_edge_boundary_idle(seg_245, duration_s=2.45, edge_tol_s=0.15) is True

    class _DummyMedia:
        cameras = [CameraSpec(name="main", role="main", width=64, height=64, fps=30.0)]
        duration_s = 2.45
        fps = 30.0

        def read_frame_rgb(self, camera: str, t_s: float) -> np.ndarray:
            return np.zeros((64, 64, 3), dtype=np.uint8)

        def read_frames_batch(self, camera: str, times_s: list[float]) -> list[np.ndarray]:
            return [np.zeros((64, 64, 3), dtype=np.uint8) for _ in times_s]

    meta = EpisodeMeta(
        episode_id="ds/ep_245",
        dataset="ds",
        stem="ep_245",
        task_id="_default",
        adapter_type="video_dir",
        source_uri="/tmp/ep_245.mp4",
        source_mtime=1.0,
        episode_fingerprint="fp_245",
        duration_s=2.45,
        fps=30.0,
        frame_count=74,
        cameras=[CameraSpec(name="main", role="main", width=64, height=64, fps=30.0)],
        camera_roles={"main": "main"},
        kinematics=KinematicsSpec(has_kinematics=False),
    )

    class _S3Fake:
        budget = BudgetTracker()

        def model_for_stage(self, stage: str, override: str | None = None) -> str:
            return "m"

        def chat_json(self, **kw: Any) -> dict[str, Any]:
            return {
                "segments": [
                    {
                        "t_start": 2.15,
                        "t_end": 2.3,
                        "label_id": "boundary_idle",
                        "affected_arm": "both",
                    }
                ]
            }

    s3_out = run_stage3_discover(
        view=EpisodeView(meta=meta, media=_DummyMedia()),
        client=_S3Fake(),  # type: ignore[arg-type]
        cfg=ConfigManager(),
    )
    assert len(s3_out) == 1

    # 1.1 - 0.5 == 0.6000000000000001 in IEEE-754 float64
    seg_11 = _seg(t_start=0.55, t_end=0.6, label_id="boundary_idle")

    class _ShiftMedia:
        cameras = [CameraSpec(name="main", role="main", width=64, height=64, fps=30.0)]
        duration_s = 1.1
        fps = 30.0

        def read_frame_rgb(self, camera: str, t_s: float) -> np.ndarray:
            return (
                np.zeros((64, 64, 3), dtype=np.uint8)
                if t_s < 0.58
                else np.full((64, 64, 3), 255, dtype=np.uint8)
            )

        def read_frames_batch(self, camera: str, times_s: list[float]) -> list[np.ndarray]:
            return [self.read_frame_rgb(camera, t) for t in times_s]

    vg_out = apply_visual_gate(
        [seg_11],
        media=_ShiftMedia(),
        cfg={"splice_gate": {"boundary_edge_tol_s": 0.5}},
        duration_s=1.1,
    )
    assert vg_out[0].guardrail.downgraded is False

    kin = KinematicsData(
        timestamps_s=np.arange(33, dtype=np.float64) / 30.0,
        eef_pos=np.zeros((33, 2, 3), dtype=np.float64),
        eef_rotvec=np.zeros((33, 2, 3), dtype=np.float64),
        valid_mask=np.ones(33, dtype=bool),
    )
    kin.eef_pos[18:, 0, 0] = 0.5
    sg_out = apply_splice_gate(
        [seg_11],
        kinematics=kin,
        cfg={"boundary_edge_tol_s": 0.5, "search_margin_s": 0.01},
        duration_s=1.1,
    )
    assert sg_out[0].guardrail.downgraded is False


def test_dag_failed_rerun_preserves_status_and_fresh_stages_invalidates_downstream(
    tmp_path: Path,
) -> None:
    """Failed VlmDag re-run restores stable status (INGESTED/VLM_READY, never stuck in VLM_QUEUED); fresh_stages cascades downstream."""
    ds_dir = tmp_path / "data" / "ds"
    _write_mp4(ds_dir / "ep0.mp4", n_frames=30, fps=30.0)
    handle = auto_detect(ds_dir)[0]
    ws = tmp_path / "ws"
    repo = Repository(ws)
    try:
        with get_adapter(handle.adapter_type).open(handle) as view:
            repo.ingest_episode(view.meta)
            repo.transition_status(view.meta.episode_id, EpisodeStatus.VLM_QUEUED)

            class _FailS3:
                budget = BudgetTracker()

                def model_for_stage(self, stage: str, override: str | None = None) -> str:
                    return "m"

                def chat_json(self, **kw: Any) -> dict[str, Any]:
                    st = kw["stage"]
                    if st == "s1_grounding":
                        return {"objects": {}}
                    if st == "s2_macro":
                        return {"macro_stages": [], "arm_timelines": {}}
                    if st.startswith("s5"):
                        return {"verdict": "success", "confidence": 0.9, "criteria": []}
                    raise VlmError("Simulated S3 outage")

                def close(self) -> None:
                    return

            dag_fail = VlmDag(workspace=ws, client=_FailS3(), repo=repo)  # type: ignore[arg-type]
            with pytest.raises(VlmError):
                dag_fail.run_episode(view, fresh=True)
            assert (
                repo.get_episode_row(view.meta.episode_id)["status"]
                == EpisodeStatus.INGESTED.value
            )

            repo.save_vlm_run_artifact(
                VlmRunArtifact(
                    run_id="run_seed",
                    episode_id=view.meta.episode_id,
                    created_at=1.0,
                    config_fingerprint=ConfigFingerprintSnapshot(
                        model_routing={},
                        prompt_bundle_version="1",
                        label_taxonomy_version="1",
                        task_spec_version="1",
                    ),
                    success=SuccessVerdict(passed=True, verdict="success", confidence=0.9),
                )
            )
            with pytest.raises(VlmError):
                dag_fail.run_episode(view, fresh=True)
            assert (
                repo.get_episode_row(view.meta.episode_id)["status"]
                == EpisodeStatus.VLM_READY.value
            )

            seen_stages: list[str] = []

            class _OkClient:
                budget = BudgetTracker()

                def model_for_stage(self, stage: str, override: str | None = None) -> str:
                    return "m"

                def chat_json(self, **kw: Any) -> dict[str, Any]:
                    st = kw["stage"]
                    seen_stages.append(st)
                    if st == "s1_grounding":
                        return {"objects": {}}
                    if st == "s2_macro":
                        return {"macro_stages": [], "arm_timelines": {}}
                    if st.startswith("s5"):
                        return {"verdict": "success", "confidence": 0.9, "criteria": []}
                    if st == "s3_discover":
                        return {
                            "segments": [
                                {
                                    "t_start": 0.1,
                                    "t_end": 0.8,
                                    "label_id": "mid_idle_freeze",
                                    "affected_arm": "left",
                                }
                            ]
                        }
                    if st == "s4a_merge":
                        return {
                            "merged_segments": [
                                {
                                    "t_start": 0.1,
                                    "t_end": 0.8,
                                    "label_id": "mid_idle_freeze",
                                    "affected_arm": "left",
                                }
                            ]
                        }
                    return {"keep_segment": True}

                def close(self) -> None:
                    return

            dag_ok = VlmDag(workspace=ws, client=_OkClient(), repo=repo)  # type: ignore[arg-type]
            art_warm = dag_ok.run_episode(view, fresh=True)
            seen_stages.clear()
            dag_ok.run_episode(view, fresh_stages={"s3_discover"})
            assert "s3_discover" in seen_stages
            assert any(s.startswith("s4") for s in seen_stages)
            assert "s1_grounding" not in seen_stages
            assert "s2_macro" not in seen_stages
            assert not any(s.startswith("s5") for s in seen_stages)

            # S2 still cascades downstream to S3 and S4 without re-running S1 or S5
            seen_stages.clear()
            dag_ok.run_episode(view, fresh_stages={"s2_macro"})
            assert {"s2_macro", "s3_discover"}.issubset(set(seen_stages))
            assert any(s.startswith("s4") for s in seen_stages)
            assert "s1_grounding" not in seen_stages
            assert not any(s.startswith("s5") for s in seen_stages)

            # Refresh art_warm reference after S2 cascade (S3/S4 keys updated)
            art_after_s2 = dag_ok.run_episode(view, fresh=False)

            # S1-only rerun MUST re-run ONLY s1_grounding and reuse S2-S5 caches (0 S2-S5 VLM calls)
            seen_stages.clear()
            art_s1 = dag_ok.run_episode(view, fresh_stages={"s1_grounding"})
            assert set(seen_stages) == {"s1_grounding"}
            assert art_s1.stage_refs.s2_macro == art_after_s2.stage_refs.s2_macro
            assert art_s1.stage_refs.s3_discover == art_after_s2.stage_refs.s3_discover
            assert art_s1.stage_refs.s4_verify == art_after_s2.stage_refs.s4_verify
            assert art_s1.stage_refs.s5_success == art_after_s2.stage_refs.s5_success

            # Changing S1 config/prompt invalidates ONLY S1 cache key, leaving S2-S5 cache keys intact
            ov_dir = ws / "configs_override"
            (ov_dir / "prompts").mkdir(parents=True, exist_ok=True)
            (ov_dir / "default.yaml").write_text(
                "vlm:\n  s1_grounding:\n    max_tokens: 1234\n",
                encoding="utf-8",
            )
            (ov_dir / "prompts" / "s1_grounding.yaml").write_text(
                (dag_ok.cfg.config_dir / "prompts" / "s1_grounding.yaml").read_text(encoding="utf-8")
                + "\n# s1 prompt tweak for cache isolation test\n",
                encoding="utf-8",
            )
            dag_ok.cfg.reload()
            seen_stages.clear()
            art_s1_cfg = dag_ok.run_episode(view, fresh=False)
            assert set(seen_stages) == {"s1_grounding"}
            assert art_s1_cfg.stage_refs.s1_grounding != art_s1.stage_refs.s1_grounding
            assert art_s1_cfg.stage_refs.s2_macro == art_s1.stage_refs.s2_macro
            assert art_s1_cfg.stage_refs.s3_discover == art_s1.stage_refs.s3_discover
            assert art_s1_cfg.stage_refs.s4_verify == art_s1.stage_refs.s4_verify
            assert art_s1_cfg.stage_refs.s5_success == art_s1.stage_refs.s5_success
            assert art_warm.episode_id == art_s1_cfg.episode_id
    finally:
        repo.close()


def test_stage1_and_stage3_partial_errors_must_not_cache(tmp_path: Path) -> None:
    """Partial window/keyframe VlmErrors in S1 and S3 must raise rather than silently returning truncated output."""
    ds_dir = tmp_path / "data" / "ds"
    _write_mp4(ds_dir / "ep_long.mp4", n_frames=20 * 30, fps=30.0)
    handle = auto_detect(ds_dir)[0]
    calls = {"s1": 0, "s3": 0}

    class _PartialFailClient:
        budget = BudgetTracker()

        def model_for_stage(self, stage: str, override: str | None = None) -> str:
            return "m"

        def chat_json(self, **kw: Any) -> dict[str, Any]:
            st = kw["stage"]
            if st == "s1_grounding":
                calls["s1"] += 1
                if calls["s1"] == 2:
                    raise VlmError("Keyframe 2 HTTP 500")
                return {"objects": {}}
            calls["s3"] += 1
            if calls["s3"] == 2:
                raise VlmError("Window 2 HTTP 500")
            return {"segments": []}

    with get_adapter(handle.adapter_type).open(handle) as view:
        with pytest.raises(VlmError, match="Keyframe 2"):
            run_stage1_grounding(view=view, client=_PartialFailClient(), cfg=ConfigManager())  # type: ignore[arg-type]
        with pytest.raises(VlmError, match="Window 2"):
            run_stage3_discover(view=view, client=_PartialFailClient(), cfg=ConfigManager())  # type: ignore[arg-type]


def test_parse_json_raw_decode_with_trailing_braces() -> None:
    assert parse_json('Result: {"segments": []} (see {note})') == {"segments": []}
    assert parse_json('Prefix {bad} then {"ok": true, "n": 2} trailing {x}') == {"ok": True, "n": 2}


def test_guardrail_nan_hallucination_and_zero_arm_splice_gate() -> None:
    kin_h = KinematicsData(
        timestamps_s=np.arange(30, dtype=np.float64) / 30.0,
        eef_pos=np.zeros((30, 2, 3), dtype=np.float64),
        valid_mask=np.ones(30, dtype=bool),
    )
    kin_h.eef_pos[:, :, 0] = np.linspace(0.0, 0.5, 30)[:, None]
    kin_h.eef_pos[15, 0, 0] = np.nan
    kin_h.valid_mask[15] = False
    out_h = apply_hallucination_check(
        [_seg(t_start=0.0, t_end=1.0, label_id="mid_idle_freeze", policy="cut_clean")],
        kinematics=kin_h,
    )
    assert out_h[0].guardrail.guardrail_override is True

    kin_0 = KinematicsData(
        timestamps_s=np.arange(10, dtype=np.float64) / 30.0,
        eef_pos=np.zeros((10, 0, 3), dtype=np.float64),
        eef_rotvec=np.zeros((10, 0, 3), dtype=np.float64),
        valid_mask=np.ones(10, dtype=bool),
    )
    out_0 = apply_splice_gate(
        [_seg(t_start=0.05, t_end=0.2, label_id="mid_idle_freeze", policy="cut_clean")],
        kinematics=kin_0,
        duration_s=0.33,
    )
    assert len(out_0) == 1


def test_reviewer_release_hijack_and_abandoned_draft_isolation(tmp_path: Path) -> None:
    """Draft+release/reap must not pollute active_rev_id; Reviewer B cannot hijack/release A's claim."""
    ds_dir = tmp_path / "data" / "ds"
    _write_mp4(ds_dir / "ep.mp4", n_frames=30, fps=30.0)
    handle = auto_detect(ds_dir)[0]
    ws = tmp_path / "ws"
    repo = Repository(ws)
    try:
        with get_adapter(handle.adapter_type).open(handle) as view:
            repo.ingest_episode(view.meta)
            ep_id = view.meta.episode_id
        repo.save_vlm_run_artifact(
            VlmRunArtifact(
                run_id="r1",
                episode_id=ep_id,
                created_at=1.0,
                config_fingerprint=ConfigFingerprintSnapshot(
                    model_routing={},
                    prompt_bundle_version="1",
                    label_taxonomy_version="1",
                    task_spec_version="1",
                ),
                success=SuccessVerdict(passed=True, verdict="success", confidence=0.9),
            )
        )
        repo.claim_episode(ep_id, "alice")
        repo.save_review(
            ep_id,
            "alice",
            "reviewer",
            ReviewWritePayload(
                base_run_id="r1",
                action="save_draft",
                qc_segments=[_seg(t_start=0.1, t_end=0.5, label_id="mid_idle_freeze")],
                success=SuccessVerdict(passed=False, verdict="failure", confidence=0.9),
                macro_stages=[],
                watch_coverage=WatchCoverage(duration_s=1.0, watched_s=0.1, coverage_ratio=0.1),
            ),
        )
        assert repo.release_claim(ep_id, "alice") == EpisodeStatus.VLM_READY.value
        resolved = repo.resolve_episode(ep_id)
        assert resolved.is_human_reviewed is False
        assert resolved.active_rev_id is None
        assert resolved.qc_segments == []
        assert resolved.success.passed is True
        assert repo.get_episode_row(ep_id)["resolved_passed"] == 1

        # Also verify reap_expired_claims cleans abandoned draft
        repo.claim_episode(ep_id, "alice", ttl_s=1.0, now=100.0)
        repo.save_review(
            ep_id,
            "alice",
            "reviewer",
            ReviewWritePayload(
                base_run_id="r1",
                action="save_draft",
                qc_segments=[_seg(t_start=0.1, t_end=0.5, label_id="mid_idle_freeze")],
                success=SuccessVerdict(passed=False, verdict="failure", confidence=0.9),
                macro_stages=[],
                watch_coverage=WatchCoverage(duration_s=1.0, watched_s=0.1, coverage_ratio=0.1),
            ),
            now=100.5,
        )
        assert repo.reap_expired_claims(now=102.0) == 1
        resolved_reaped = repo.resolve_episode(ep_id)
        assert resolved_reaped.is_human_reviewed is False
        assert resolved_reaped.success.passed is True

        repo.claim_episode(ep_id, "alice")
    finally:
        repo.close()

    client = TestClient(create_app(ws, reviewer_token="r-tok", lead_token="l-tok"))
    for ep_action in ("release", "renew", "claim"):
        resp = client.post(
            f"/api/episodes/{ep_id}/{ep_action}",
            headers={"X-Auth-Token": "r-tok", "X-User": "bob"},
            json={"user": "alice"},
        )
        assert resp.status_code in (403, 409)


def test_concurrent_unknown_triage_promote_and_non_unknown_rejection(tmp_path: Path) -> None:
    """Concurrent unknown promote requests must not lose either label; non-unknown seg_id rejected."""
    ws = tmp_path / "ws"
    repo = Repository(ws)
    meta = EpisodeMeta(
        episode_id="ds/ep_unk",
        dataset="ds",
        stem="ep_unk",
        task_id="_default",
        adapter_type="video_dir",
        source_uri="/tmp/ep_unk.mp4",
        source_mtime=1.0,
        episode_fingerprint="fp_unk",
        duration_s=2.0,
        fps=30.0,
        frame_count=60,
        cameras=[CameraSpec(name="main", role="main", width=64, height=64, fps=30.0)],
        camera_roles={"main": "main"},
        kinematics=KinematicsSpec(has_kinematics=False),
    )
    repo.ingest_episode(meta)
    repo.save_vlm_run_artifact(
        VlmRunArtifact(
            run_id="r_unk",
            episode_id="ds/ep_unk",
            created_at=1.0,
            config_fingerprint=ConfigFingerprintSnapshot(
                model_routing={},
                prompt_bundle_version="1",
                label_taxonomy_version="1",
                task_spec_version="1",
            ),
            qc_segments=[
                _seg("u1", 0.1, 0.5, "unknown_anomaly", "mask_loss").model_copy(
                    update={"is_unknown": True}
                ),
                _seg("u2", 0.6, 1.0, "unknown_anomaly", "mask_loss").model_copy(
                    update={"is_unknown": True}
                ),
                _seg("norm", 1.1, 1.5, "mid_idle_freeze", "cut_clean"),
            ],
            success=SuccessVerdict(passed=True, verdict="success", confidence=0.9),
        )
    )
    repo.close()

    app = create_app(ws, reviewer_token="r-tok", lead_token="l-tok")
    c_norm = TestClient(app).post(
        "/api/lab/unknowns/triage",
        headers={"X-Auth-Token": "l-tok"},
        json={
            "episode_id": "ds/ep_unk",
            "seg_id": "norm",
            "action": "merge",
            "target_label_id": "mid_idle_freeze",
        },
    )
    assert c_norm.status_code in (400, 422)

    def _promote(idx: int) -> int:
        c = TestClient(app)
        return c.post(
            "/api/lab/unknowns/triage",
            headers={"X-Auth-Token": "l-tok"},
            json={
                "episode_id": "ds/ep_unk",
                "seg_id": f"u{idx}",
                "action": "promote",
                "new_label": {
                    "id": f"race_lbl_{idx}",
                    "name": f"Race Label {idx}",
                    "category": "failure",
                    "default_training_role": "badcase",
                    "default_splice_policy": "mask_loss",
                    "collector_ui": {"title": f"Race Label {idx}"},
                    "vlm_semantics": {"definition": "d"},
                },
            },
        ).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        codes = list(pool.map(_promote, [1, 2]))
    assert codes == [200, 200]
    labels_doc = TestClient(app).get("/api/lab/labels", headers={"X-Auth-Token": "l-tok"}).json()
    ids = {lbl["id"] for lbl in labels_doc["labels"]}
    assert {"race_lbl_1", "race_lbl_2"}.issubset(ids)


def test_case_insensitive_legacy_rejection_and_missing_media_404(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        _seg(label_id="LONG_PAUSE")
    with pytest.raises(ValidationError):
        QcSegment.model_validate(
            {
                "seg_id": "s_bad",
                "t_start": 0.0,
                "t_end": 0.5,
                "label_id": "Back_And_Forth",
                "training_role": "badcase",
                "splice_policy": "mask_loss",
                "mask_side": "both",
                "vlm_raw_policy": "mask_loss",
            }
        )

    ws = tmp_path / "ws"
    repo = Repository(ws)
    meta = EpisodeMeta(
        episode_id="ds/ep_miss",
        dataset="ds",
        stem="ep_miss",
        task_id="_default",
        adapter_type="video_dir",
        source_uri=str(tmp_path / "nonexistent.mp4"),
        source_mtime=1.0,
        episode_fingerprint="fp_miss",
        duration_s=1.0,
        fps=30.0,
        frame_count=30,
        cameras=[CameraSpec(name="main", role="main", width=64, height=64, fps=30.0)],
        camera_roles={"main": "main"},
        kinematics=KinematicsSpec(has_kinematics=False),
    )
    repo.ingest_episode(meta)
    repo.close()

    client = TestClient(
        create_app(ws, reviewer_token="r-tok", lead_token="l-tok"), raise_server_exceptions=False
    )
    r_fr = client.get("/api/episodes/ds/ep_miss/frame?t_s=0.1", headers={"X-Auth-Token": "r-tok"})
    assert r_fr.status_code == 404
    r_pv = client.get("/api/episodes/ds/ep_miss/preview?token=r-tok")
    assert r_pv.status_code == 404


def test_mixed_fps_lerobot_v3_export_and_empty_lerobot_tasks_and_export_id_sync(
    tmp_path: Path,
) -> None:
    """Mixed-FPS LeRobotV3 export keeps Parquet & MP4 timestamps synced; empty tasks=[] uses tasks_by_idx; export_id matches dir."""
    # 1. Mixed-FPS LeRobotV3 export (30fps + 15fps, both 1.0s)
    ds0 = tmp_path / "ds0"
    ds1 = tmp_path / "ds1"
    _write_mp4(ds0 / "ep0.mp4", n_frames=30, fps=30.0)
    _write_mp4(ds1 / "ep1.mp4", n_frames=15, fps=15.0)
    h0 = auto_detect(ds0)[0]
    h1 = auto_detect(ds1)[0]
    views: dict[str, EpisodeView] = {}
    resolved: list[ResolvedEpisode] = []
    pieces = []
    slicer = PieceSlicer()
    rec = ExportRecipe(name="bc_clean", format="lerobot_v3")
    with get_adapter(h0.adapter_type).open(h0) as v0, get_adapter(h1.adapter_type).open(h1) as v1:
        for idx, v in enumerate((v0, v1)):
            ep_id = f"mixed/ep{idx}"
            meta = v.meta.model_copy(
                update={"episode_id": ep_id, "dataset": "mixed", "stem": f"ep{idx}"}
            )
            v.meta = meta
            res = ResolvedEpisode(
                episode_id=ep_id,
                meta=meta,
                status=EpisodeStatus.APPROVED,
                qc_segments=[],
                success=SuccessVerdict(passed=True, verdict="success", confidence=0.9),
                macro_stages=[],
                grounding={"keyframes_s": [], "object_tracks_2d": {}},
                arm_timelines={"left": [], "right": []},
                raw_vlm_segments=[],
            )
            views[ep_id] = v
            resolved.append(res)
            pieces.extend(slicer.slice(res, v, rec))

        out_lr = tmp_path / "out_mixed_lr"
        manifest = get_exporter("lerobot_v3").export(pieces, resolved, views, out_lr, rec)

    info = json.loads((out_lr / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["robot_type"] == "bimanual_robot"
    assert manifest.total_frames == info["total_frames"] == 60
    ep_tbl = pq.read_table(
        out_lr / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).to_pandas()
    assert list(ep_tbl["length"]) == [30, 30]
    assert (
        ep_tbl.iloc[1]["videos/observation.images.main/to_timestamp"]
        - ep_tbl.iloc[1]["videos/observation.images.main/from_timestamp"]
        == pytest.approx(1.0, abs=1e-4)
    )
    data_tbl = pq.read_table(out_lr / "data" / "chunk-000" / "file-000.parquet").to_pandas()
    ep1_ts = data_tbl[data_tbl["episode_index"] == 1]["timestamp"].to_numpy()
    assert len(ep1_ts) == 30
    assert float(ep1_ts[-1]) == pytest.approx(29.0 / 30.0, abs=1e-4)

    # 2. LeRobot v3 tasks=[] fallback to tasks_by_idx[task_index]
    lr_empty_tasks = tmp_path / "lr_empty_tasks"
    (lr_empty_tasks / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (lr_empty_tasks / "data" / "chunk-000").mkdir(parents=True)
    (lr_empty_tasks / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": 30,
                "chunks_size": 1000,
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [81]},
                    "observation.images.main": {"dtype": "video", "shape": [64, 64, 3]},
                },
            }
        ),
        encoding="utf-8",
    )
    pq.write_table(
        pa.table(
            {
                "task_index": pa.array([0], type=pa.int64()),
                "task": pa.array(["pick_and_place"], type=pa.string()),
            }
        ),
        lr_empty_tasks / "meta" / "tasks.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([0], type=pa.int64()),
                "length": pa.array([30], type=pa.int64()),
                "tasks": pa.array([[]], type=pa.list_(pa.string())),
                "task_index": pa.array([0], type=pa.int64()),
                "data/chunk_index": pa.array([0], type=pa.int64()),
                "data/file_index": pa.array([0], type=pa.int64()),
            }
        ),
        lr_empty_tasks / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    lr_handles = get_adapter("lerobot").discover(
        lr_empty_tasks, task_patterns={"pick_and_place": ["*pick_and_place*"]}
    )
    assert len(lr_handles) == 1
    assert lr_handles[0].metadata["ep_task_str"] == "pick_and_place"
    assert lr_handles[0].task_id == "pick_and_place"

    # 3. POST /api/exports manifest.export_id matches exports/<export_id> dir
    ws = tmp_path / "ws_exp"
    repo = Repository(ws)
    try:
        with get_adapter(h0.adapter_type).open(h0) as v0:
            repo.ingest_episode(v0.meta)
            repo.save_vlm_run_artifact(
                VlmRunArtifact(
                    run_id="r_exp",
                    episode_id=v0.meta.episode_id,
                    created_at=1.0,
                    config_fingerprint=ConfigFingerprintSnapshot(
                        model_routing={},
                        prompt_bundle_version="1",
                        label_taxonomy_version="1",
                        task_spec_version="1",
                    ),
                    success=SuccessVerdict(passed=True, verdict="success", confidence=0.9),
                )
            )
            repo.save_review(
                v0.meta.episode_id,
                "lead_u",
                "lead",
                ReviewWritePayload(
                    base_run_id="r_exp",
                    action="approve",
                    qc_segments=[],
                    success=SuccessVerdict(passed=True, verdict="success", confidence=0.9),
                    macro_stages=[],
                    watch_coverage=WatchCoverage(
                        duration_s=1.0, watched_s=1.0, coverage_ratio=1.0
                    ),
                ),
            )
    finally:
        repo.close()

    client = TestClient(create_app(ws, reviewer_token="r-tok", lead_token="l-tok"))
    exp_resp = client.post(
        "/api/exports",
        headers={"X-Auth-Token": "l-tok"},
        json={"format": "zero_copy_index", "recipe": "bc_clean"},
    )
    assert exp_resp.status_code == 200
    exp_data = exp_resp.json()
    assert Path(exp_data["output_dir"]).name == exp_data["export_id"]
    assert (ws / "exports" / exp_data["export_id"]).is_dir()
    dl = client.get(
        f"/api/exports/{exp_data['export_id']}/download", headers={"X-Auth-Token": "l-tok"}
    )
    assert dl.status_code == 200


def test_io_export_slicer_and_watch_coverage_edge_cases(tmp_path: Path) -> None:
    """H5Clean empty timestamps, truncated H5 jpeg, NaN H5 read, smooth_tail_81d, and WatchCoverage."""
    # 1. Empty kinematics timestamps_s in H5CleanExporter
    meta = EpisodeMeta(
        episode_id="ds/ep_e",
        dataset="ds",
        stem="ep_e",
        task_id="_default",
        adapter_type="video_dir",
        source_uri="/tmp/x.mp4",
        source_mtime=1.0,
        episode_fingerprint="fp_e",
        duration_s=1.0,
        fps=30.0,
        frame_count=30,
        cameras=[CameraSpec(name="main", role="main", width=64, height=64, fps=30.0)],
        camera_roles={"main": "main"},
        kinematics=KinematicsSpec(has_kinematics=False),
    )
    res = ResolvedEpisode(
        episode_id="ds/ep_e",
        meta=meta,
        status=EpisodeStatus.APPROVED,
        qc_segments=[],
        success=SuccessVerdict(passed=True, verdict="success", confidence=0.9),
        macro_stages=[],
        grounding={"keyframes_s": [], "object_tracks_2d": {}},
        arm_timelines={"left": [], "right": []},
        raw_vlm_segments=[],
    )

    class _M:
        cameras = [CameraSpec(name="main", role="main", width=64, height=64, fps=30.0)]
        fps = 30.0
        duration_s = 1.0

        def read_frame_rgb(self, c: str, t: float) -> np.ndarray:
            return np.zeros((64, 64, 3), dtype=np.uint8)

        def read_frames_batch(self, c: str, ts: list[float]) -> list[np.ndarray]:
            return [np.zeros((64, 64, 3), dtype=np.uint8) for _ in ts]

    view = EpisodeView(meta=meta, media=_M(), kinematics=KinematicsData(timestamps_s=np.zeros(0)))
    rec = ExportRecipe(name="bc_clean", format="h5_clean")
    pcs = PieceSlicer().slice(res, view, rec)
    manifest = get_exporter("h5_clean").export(
        pcs, [res], {res.episode_id: view}, tmp_path / "h5_out", rec
    )
    assert manifest.episodes_included == 1

    # 2. Truncated H5 jpeg dataset clamps safely & NaN t_s raises ValueError
    h5_trunc = tmp_path / "ds_h5" / "trunc.h5"
    h5_trunc.parent.mkdir(parents=True)
    vlen_u8 = h5py.vlen_dtype(np.dtype("uint8"))
    ok, jpg_b = cv2.imencode(".jpg", np.zeros((32, 32, 3), dtype=np.uint8))
    assert ok
    with h5py.File(str(h5_trunc), "w") as f:
        rg = f.create_group("robot")
        rg.create_dataset("timestamps", data=np.linspace(10.0, 11.0, 31))
        st = np.zeros((31, 37), dtype=np.float32)
        st[:, 26] = 1.0
        st[:, 33] = 1.0
        rg.create_dataset("state", data=st)
        hg = f.create_group("hand")
        hg.create_dataset("timestamps", data=np.linspace(10.0, 11.0, 31))
        hg.create_dataset("hand_a_radian", data=np.zeros((31, 22), dtype=np.float32))
        hg.create_dataset("hand_b_radian", data=np.zeros((31, 22), dtype=np.float32))
        cg = f.create_group("camera")
        g = cg.create_group("head_rgbd")
        g.create_dataset("timestamps", data=np.linspace(10.0, 11.0, 31))
        dsj = g.create_dataset("jpeg", (10,), dtype=vlen_u8)
        for i in range(10):
            dsj[i] = np.frombuffer(jpg_b.tobytes(), dtype=np.uint8)

    ad_h5 = get_adapter("h5_raw_v3")
    hd = ad_h5.discover(h5_trunc)[0]
    with ad_h5.open(hd) as v_t:
        fr = v_t.media.read_frame_rgb("head_rgbd", 1.0)
        assert fr.shape == (32, 32, 3)
        with pytest.raises(ValueError):
            v_t.media.read_frame_rgb("head_rgbd", float("nan"))

    # 3. tail_weights(1) == [0.5] and smooth_tail_81d(tail_rows=1) blends cleanly
    assert np.allclose(tail_weights(1), [0.5])
    arr81 = np.zeros((3, 81), dtype=np.float32)
    arr81[:, 26] = 1.0
    arr81[:, 33] = 1.0
    arr81[2, 23] = 1.0
    sm1 = smooth_tail_81d(arr81, own_rows=2)
    assert sm1[2, 23] == pytest.approx(0.5, abs=1e-5)

    # 4. WatchCoverage rejects forged coverage_ratio when watched_s=0.0
    with pytest.raises(ValidationError):
        WatchCoverage(
            watched_intervals=[], watched_s=0.0, duration_s=10.0, coverage_ratio=1.0, confirmed=True
        )


def test_episodes_multi_status_and_claimed_by_server_pagination(tmp_path: Path) -> None:
    """Backend contract: multi-status CSV, claimed_by filter, pagination, and 422 validation."""
    ws = tmp_path / "ws_multi"
    repo = Repository(ws)
    try:
        for idx in range(1, 12):
            ep_id = f"ds/ep_{idx:03d}"
            meta = EpisodeMeta(
                episode_id=ep_id,
                dataset="ds",
                stem=f"ep_{idx:03d}",
                task_id="_default",
                adapter_type="video_dir",
                source_uri=f"/tmp/ep_{idx:03d}.mp4",
                source_mtime=1.0,
                episode_fingerprint=f"fp_{idx:03d}",
                duration_s=1.0,
                fps=30.0,
                frame_count=30,
                cameras=[CameraSpec(name="main", role="main", width=64, height=64, fps=30.0)],
                camera_roles={"main": "main"},
                kinematics=KinematicsSpec(has_kinematics=False),
            )
            repo.ingest_episode(meta)
            repo.save_vlm_run_artifact(
                VlmRunArtifact(
                    run_id=f"r_{idx:03d}",
                    episode_id=ep_id,
                    created_at=1.0,
                    config_fingerprint=ConfigFingerprintSnapshot(
                        model_routing={},
                        prompt_bundle_version="1",
                        label_taxonomy_version="1",
                        task_spec_version="1",
                    ),
                    success=SuccessVerdict(passed=True, verdict="success", confidence=0.9),
                )
            )
            if idx <= 5:
                # 5 vlm_ready remain untouched
                continue
            if idx <= 7:
                repo.claim_episode(ep_id, "alice")
            elif idx == 8:
                repo.claim_episode(ep_id, "bob")
            elif idx <= 10:
                repo.save_review(
                    ep_id,
                    "alice",
                    "reviewer",
                    ReviewWritePayload(
                        base_run_id=f"r_{idx:03d}",
                        action="submit_review",
                        qc_segments=[],
                        success=SuccessVerdict(passed=True, verdict="success", confidence=0.9),
                        macro_stages=[],
                        watch_coverage=WatchCoverage(
                            watched_intervals=[(0.0, 1.0)],
                            watched_s=1.0,
                            duration_s=1.0,
                            coverage_ratio=1.0,
                        ),
                    ),
                )
            else:
                repo.save_review(
                    ep_id,
                    "alice",
                    "reviewer",
                    ReviewWritePayload(
                        base_run_id=f"r_{idx:03d}",
                        action="flag",
                        qc_segments=[],
                        success=SuccessVerdict(passed=False, verdict="failure", confidence=0.8),
                        macro_stages=[],
                        watch_coverage=WatchCoverage(
                            watched_intervals=[(0.0, 1.0)],
                            watched_s=1.0,
                            duration_s=1.0,
                            coverage_ratio=1.0,
                        ),
                        flagged_reason="ambiguous",
                    ),
                )

        # Repository direct checks
        assert repo.count_episodes(status="reviewed,flagged") == 3
        assert repo.count_episodes(status="claimed", claimed_by="alice") == 2
        assert [
            r["episode_id"]
            for r in repo.list_episodes(status="reviewed,flagged", limit=2, offset=1)
        ] == [
            "ds/ep_010",
            "ds/ep_011",
        ]
    finally:
        repo.close()

    client = TestClient(create_app(ws, reviewer_token="r-tok", lead_token="l-tok"))
    rev_alice = {"X-Auth-Token": "r-tok", "X-User": "alice"}

    # Backward compatibility: single status
    r_single = client.get("/api/episodes?status=vlm_ready", headers=rev_alice)
    assert r_single.status_code == 200
    assert r_single.json()["total"] == 5
    assert r_single.json()["my_claimed_count"] == 2

    # Multi-status CSV with pagination
    r_sub_p1 = client.get(
        "/api/episodes?status=reviewed,flagged&limit=2&offset=0", headers=rev_alice
    )
    assert r_sub_p1.status_code == 200
    d1 = r_sub_p1.json()
    assert d1["total"] == 3
    assert [it["episode_id"] for it in d1["items"]] == ["ds/ep_009", "ds/ep_010"]

    r_sub_p2 = client.get(
        "/api/episodes?status=reviewed,flagged&limit=2&offset=2", headers=rev_alice
    )
    assert r_sub_p2.status_code == 200
    assert [it["episode_id"] for it in r_sub_p2.json()["items"]] == ["ds/ep_011"]

    # claimed_by filter
    r_mine = client.get("/api/episodes?status=claimed&claimed_by=alice", headers=rev_alice)
    assert r_mine.status_code == 200
    assert r_mine.json()["total"] == 2
    assert [it["episode_id"] for it in r_mine.json()["items"]] == ["ds/ep_006", "ds/ep_007"]

    # Invalid single/multi/empty status -> HTTP 422
    assert client.get("/api/episodes?status=bogus", headers=rev_alice).status_code == 422
    assert client.get("/api/episodes?status=reviewed,bogus", headers=rev_alice).status_code == 422
    assert client.get("/api/episodes?status=", headers=rev_alice).status_code == 422
