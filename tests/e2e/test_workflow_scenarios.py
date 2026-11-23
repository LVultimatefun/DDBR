"""End-to-End Application Workflow Scenario Tests for robo_vlm_qc.

IMPORTANT:
- Top-level imports MUST NOT import ``robo_vlm_qc.*`` submodules.
  All ``robo_vlm_qc.*`` imports are performed lazily inside test functions.
- Never read, print, or log real API keys or ``.env`` credential files.
- All workspaces, DBs, artifacts, caches, and exports live under ``tmp_path``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow.parquet as pq
import pytest

from tests.e2e import helpers as h

# ---------------------------------------------------------------------------
# Local Helpers (Lazy Imports Only)
# ---------------------------------------------------------------------------


def _cli(
    workspace: Path,
    config_dir: Path,
    args: list[str],
    *,
    timeout_s: float = 180.0,
) -> Any:
    return h.run_cli(
        ["-w", str(workspace), "-c", str(config_dir), *args],
        timeout_s=timeout_s,
    )


def _parse_json_stdout(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        raise AssertionError("Expected JSON stdout from CLI, got empty output")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        idx = text.find("{")
        if idx == -1:
            raise
        data = json.loads(text[idx:])
    assert isinstance(data, dict), f"Expected JSON object, got {type(data).__name__}"
    return data


def _extract_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("episodes", "items", "results", "unknowns", "diagnostics", "examples", "exports", "jobs"):
            val = payload.get(key)
            if isinstance(val, list):
                return val
    raise AssertionError(f"Unable to extract list items from payload: {payload!r}")


def _find_lerobot_root(out_dir: Path) -> Path:
    if (out_dir / "meta" / "info.json").is_file():
        return out_dir
    infos = sorted(out_dir.rglob("meta/info.json"))
    assert infos, f"No LeRobot meta/info.json found under {out_dir}"
    return infos[0].parents[1]


def _make_watch_coverage(duration_s: float) -> dict[str, Any]:
    return {
        "watched_intervals": [[0.0, float(duration_s)]],
        "watched_s": float(duration_s),
        "duration_s": float(duration_s),
        "coverage_ratio": 1.0,
        "confirmed": True,
    }


def _make_qc_segment_dict(
    seg_id: str,
    t_start: float,
    t_end: float,
    label_id: str,
    training_role: str,
    splice_policy: str,
    *,
    mask_side: str | None = None,
    affected_arm: str = "both",
    vlm_raw_policy: str | None = None,
    is_unknown: bool = False,
    unknown_note: str = "",
) -> dict[str, Any]:
    return {
        "seg_id": seg_id,
        "t_start": float(t_start),
        "t_end": float(t_end),
        "label_id": label_id,
        "name": label_id,
        "training_role": training_role,
        "splice_policy": splice_policy,
        "mask_side": mask_side,
        "affected_arm": affected_arm,
        "visual_evidence": {
            "initial_state": f"Start state at {t_start:.2f}s.",
            "process_change": f"Observed {label_id} between {t_start:.2f}s and {t_end:.2f}s.",
            "end_state": f"End state at {t_end:.2f}s.",
            "confidence_note": "",
        },
        "vlm_raw_policy": vlm_raw_policy or splice_policy,
        "guardrail": {
            "gate_type": "none",
            "passed": True,
            "downgraded": False,
            "downgrade_reason": None,
            "guardrail_override": False,
            "override_reason": None,
            "opt_t_start": None,
            "opt_t_end": None,
        },
        "is_unknown": is_unknown,
        "unknown_note": unknown_note,
        "suggested_label": None,
        "confidence": 0.9,
    }


def _make_success_dict(passed: bool = True, duration_s: float = 4.0) -> dict[str, Any]:
    return {
        "passed": passed,
        "verdict": "success" if passed else "failure",
        "confidence": 0.95,
        "criteria": [
            {
                "criterion": "Target object placed cleanly in destination.",
                "met": passed,
                "satisfied_at_s": round(duration_s * 0.85, 2) if passed else None,
                "evidence": "Verified in end-state frames.",
            }
        ],
        "control_check": None,
        "summary": "Task succeeded." if passed else "Task failed.",
    }


def _make_macro_stages_dict(duration_s: float = 4.0) -> list[dict[str, Any]]:
    mid = round(duration_s * 0.5, 3)
    return [
        {
            "stage_id": "S1",
            "t_start": 0.0,
            "t_end": mid,
            "label": "Approach and grasp object",
            "label_en": "approach_and_grasp",
            "summary": "Approach and grasp object.",
            "color_rgb": [59, 130, 246],
        },
        {
            "stage_id": "S2",
            "t_start": mid,
            "t_end": round(duration_s, 3),
            "label": "Transport and place object",
            "label_en": "transport_and_place",
            "summary": "Transport and place object.",
            "color_rgb": [16, 185, 129],
        },
    ]


# ---------------------------------------------------------------------------
# Scenario 1: Synthetic h5_raw_v3 Read-Only Ingest & Multi-Stream Alignment
# ---------------------------------------------------------------------------


def test_scenario_01_synthetic_h5_readonly_ingest_and_align(
    e2e_workspace: Path,
    e2e_config_dir: Path,
    tmp_path: Path,
) -> None:
    from scipy.spatial.transform import Rotation

    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.io import auto_detect, compose_2x2_tile, get_adapter, select_tile_mode
    from robo_vlm_qc.store import Repository

    data_dir = tmp_path / "pick_and_place_s1"
    h5_path = h.make_synthetic_h5_raw_v3(
        data_dir,
        name="episode_000000.h5",
        duration_s=4.0,
        with_tactile_dropout=True,
        with_sbs_stereo=True,
        task_name="pick_and_place",
    )
    assert h5_path.is_file()
    before_snap = h.snapshot_tree_mtimes(h5_path)

    proc = _cli(
        e2e_workspace,
        e2e_config_dir,
        ["ingest", "--data", str(h5_path), "--json"],
        timeout_s=120.0,
    )
    assert proc.returncode == 0, f"ingest failed: {proc.stderr}\n{proc.stdout}"
    report = _parse_json_stdout(proc.stdout)
    assert report["ok"] is True
    assert report["discovered"] == 1
    assert report["inserted"] == 1

    repo = Repository(e2e_workspace)
    try:
        ep_id = "pick_and_place_s1/episode_000000"
        row = repo.get_episode_row(ep_id)
        assert row["task_id"] == "pick_and_place"
        assert row["adapter_type"] == "h5_raw_v3"
        assert row["status"] == "ingested"
        assert row["has_kinematics"] == 1
        assert 3.5 <= float(row["duration_s"]) <= 4.5
        assert float(row["fps"]) == pytest.approx(30.0)
        meta = repo.get_episode_meta(ep_id)
        assert meta.episode_id == ep_id
    finally:
        repo.close()

    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    handles = auto_detect(
        h5_path, adapter_name="h5_raw_v3", task_patterns=cfg.task_patterns()
    )
    assert len(handles) == 1

    adapter = get_adapter("h5_raw_v3")
    with adapter.open(handles[0]) as view:
        assert 3.5 <= view.duration_s <= 4.5
        assert view.fps == pytest.approx(30.0)
        assert view.kinematics is not None
        kin = view.kinematics
        T = len(kin.timestamps_s)
        assert T == view.meta.frame_count
        assert T >= 100
        assert kin.valid_mask.all()

        # Verify Slerp quaternion & SO(3) geodesic continuity (no sign-flip spikes across consecutive frames)
        assert kin.eef_rotvec is not None
        assert np.isfinite(kin.eef_rotvec).all()
        for arm_idx in range(kin.eef_rotvec.shape[1]):
            rots = Rotation.from_rotvec(kin.eef_rotvec[:, arm_idx, :])
            geo_step = (rots[:-1].inv() * rots[1:]).magnitude()
            assert float(np.max(geo_step)) < 0.2, (
                f"Arm {arm_idx} SO(3) step spike: {float(np.max(geo_step))}"
            )
        if getattr(kin, "eef_quat_wxyz", None) is not None:
            q = kin.eef_quat_wxyz
            dots = np.sum(q[:-1] * q[1:], axis=-1)
            assert np.all(dots >= -1e-6), f"Unexpected quaternion sign flip: min dot={dots.min()}"

        # Verify native tactile dropout repair yields finite 30D tactile series
        assert kin.tactile is not None
        assert kin.tactile.shape == (T, 30)
        assert np.isfinite(kin.tactile).all()

        # Verify SBS stereo L/R crop (1600x1296) + head_rgbd + torso_rgbd reads
        cam_by_name = {c.name: c for c in view.media.cameras}
        assert "head_stereo_left" in cam_by_name
        assert "head_stereo_right" in cam_by_name
        assert (cam_by_name["head_stereo_left"].width, cam_by_name["head_stereo_left"].height) == (1600, 1296)
        assert (cam_by_name["head_stereo_right"].width, cam_by_name["head_stereo_right"].height) == (1600, 1296)

        l_rgb = view.media.read_frame_rgb("head_stereo_left", 1.0)
        r_rgb = view.media.read_frame_rgb("head_stereo_right", 1.0)
        assert l_rgb.shape == (1296, 1600, 3) and l_rgb.dtype == np.uint8
        assert r_rgb.shape == (1296, 1600, 3) and r_rgb.dtype == np.uint8
        assert not np.array_equal(l_rgb, r_rgb)

        head_batch = view.media.read_frames_batch("head_rgbd", [0.0, 1.5, 3.0])
        torso_batch = view.media.read_frames_batch("torso_rgbd", [0.0, 1.5, 3.0])
        assert len(head_batch) == 3 and all(fr.dtype == np.uint8 and fr.ndim == 3 for fr in head_batch)
        assert len(torso_batch) == 3 and all(fr.dtype == np.uint8 and fr.ndim == 3 for fr in torso_batch)

        # Verify [TORSO|TORSO-ZOOM/HEAD|HEAD-ZOOM] 2x2 tile composition
        assert select_tile_mode(view.camera_roles) == "torso_head_2x2"
        tile = compose_2x2_tile(view.media, view.camera_roles, t_s=2.0)
        assert tile.dtype == np.uint8
        assert tile.shape == (386, 640, 3)

    after_snap = h.snapshot_tree_mtimes(h5_path)
    assert after_snap == before_snap, "Read-only source H5 was modified!"


# ---------------------------------------------------------------------------
# Scenario 2: Causal VLM DAG + splice_gate Downgrade + 1080p Studio HUD MP4
# ---------------------------------------------------------------------------


def test_scenario_02_causal_vlm_dag_splice_gate_and_1080p_mp4(
    e2e_workspace: Path,
    e2e_config_dir: Path,
    mock_vlm_server: Any,
    tmp_path: Path,
) -> None:
    from robo_vlm_qc.store import Repository

    data_dir = tmp_path / "pick_and_place_dag"
    h.make_synthetic_h5_raw_v3(
        data_dir,
        name="episode_000000.h5",
        duration_s=4.0,
        stationary=False,
        task_name="pick_and_place",
    )

    # Program MockVlmServer to suggest an unsafe mid-episode cut_clean while arms move
    unsafe_seg = {
        "t_start": 1.2,
        "t_end": 2.8,
        "label_id": "mid_idle_freeze",
        "affected_arm": "both",
        "splice_policy": "cut_clean",
        "visual_evidence": {
            "initial_state": "Both arms hover over the workspace.",
            "process_change": "Both arms appear stalled mid-trajectory without progressing.",
            "end_state": "Both arms hold briefly before continuing motion.",
        },
        "confidence": 0.89,
        "unknown_note": "",
    }
    mock_vlm_server.set_stage_response("S3", {"segments": [unsafe_seg]})
    mock_vlm_server.set_stage_response("S4A", {"merged_segments": [unsafe_seg]})
    mock_vlm_server.set_stage_response(
        "S4B",
        {
            "keep_segment": True,
            "t_start": 1.2,
            "t_end": 2.8,
            "label_id": "mid_idle_freeze",
            "affected_arm": "both",
            "splice_policy": "cut_clean",
            "visual_evidence": unsafe_seg["visual_evidence"],
            "confidence": 0.90,
            "reject_reason": "",
        },
    )

    ing = _cli(e2e_workspace, e2e_config_dir, ["ingest", "--data", str(data_dir), "--json"])
    assert ing.returncode == 0, f"ingest failed: {ing.stderr}\n{ing.stdout}"
    ep_id = "pick_and_place_dag/episode_000000"

    run_res = _cli(
        e2e_workspace,
        e2e_config_dir,
        ["run-vlm", "--episode", ep_id],
        timeout_s=120.0,
    )
    assert run_res.returncode == 0, f"run-vlm failed: {run_res.stderr}\n{run_res.stdout}"

    # Verify causal DAG stages executed against MockVlmServer and populated stage_cache
    called_stages = {r["stage"] for r in mock_vlm_server.requests}
    assert {"S1", "S2", "S3", "S4A", "S4B", "S5"} <= called_stages, called_stages
    cache_files = list((e2e_workspace / "stage_cache").rglob("*.json"))
    assert len(cache_files) >= 5

    repo = Repository(e2e_workspace)
    try:
        row = repo.get_episode_row(ep_id)
        assert row["status"] == "vlm_ready"
        assert row["active_run_id"]
        vlm_run = repo.get_vlm_run(row["active_run_id"])
        h.assert_no_legacy_fields(vlm_run)

        mid_segs = [s for s in vlm_run.qc_segments if s.label_id == "mid_idle_freeze"]
        assert mid_segs, f"Expected mid_idle_freeze segment, got {vlm_run.qc_segments}"
        seg = mid_segs[0]
        assert seg.vlm_raw_policy == "cut_clean"
        assert seg.splice_policy == "mask_loss", "splice_gate must downgrade unsafe cut_clean to mask_loss"
        assert seg.guardrail.gate_type == "splice_gate"
        assert seg.guardrail.downgraded is True
        assert seg.guardrail.downgrade_reason
    finally:
        repo.close()

    mp4_out = tmp_path / "render" / "studio_hud.mp4"
    mp4_res = _cli(
        e2e_workspace,
        e2e_config_dir,
        ["render-mp4", "--episode", ep_id, "--out", str(mp4_out)],
        timeout_s=120.0,
    )
    assert mp4_res.returncode == 0, f"render-mp4 failed: {mp4_res.stderr}\n{mp4_res.stdout}"
    vinfo = h.ffprobe_video_info(mp4_out)
    assert vinfo["width"] == 1920
    assert vinfo["height"] == 1080
    assert 29.0 <= vinfo["fps"] <= 31.0
    assert vinfo["nb_frames"] is not None and vinfo["nb_frames"] >= 90


# ---------------------------------------------------------------------------
# Scenario 3: Pure-Video video_dir + visual_gate + Single-Cam Tile + zero_copy_index
# ---------------------------------------------------------------------------


def test_scenario_03_pure_video_dir_visual_gate_and_zero_copy(
    e2e_workspace: Path,
    e2e_config_dir: Path,
    mock_vlm_server: Any,
    tmp_path: Path,
) -> None:
    from robo_vlm_qc.io import auto_detect, compose_2x2_tile, get_adapter, select_tile_mode
    from robo_vlm_qc.models.review import ReviewWritePayload, SuccessVerdict, WatchCoverage
    from robo_vlm_qc.store import Repository

    multi_root = h.make_synthetic_video_dir(
        tmp_path / "pure_multi", multi_cam=True, duration_s=2.0, n_episodes=1, scene_change=True
    )
    single_root = h.make_synthetic_video_dir(
        tmp_path / "pure_single", multi_cam=False, duration_s=2.0, n_episodes=1, scene_change=True
    )

    # Program VLM to propose both a boundary cut_clean (kept) and a mid cut_clean (downgraded by visual_gate)
    cands = [
        {
            "t_start": 0.0,
            "t_end": 0.2,
            "label_id": "boundary_idle",
            "affected_arm": "both",
            "splice_policy": "cut_clean",
            "visual_evidence": {
                "initial_state": "Initial standby frame.",
                "process_change": "Short leading stillness.",
                "end_state": "Ready to begin motion.",
            },
            "confidence": 0.92,
            "unknown_note": "",
        },
        {
            "t_start": 0.5,
            "t_end": 1.6,
            "label_id": "mid_idle_freeze",
            "affected_arm": "both",
            "splice_policy": "cut_clean",
            "visual_evidence": {
                "initial_state": "Mid-episode frame.",
                "process_change": "Apparent mid-trajectory stall.",
                "end_state": "Motion resumes afterwards.",
            },
            "confidence": 0.88,
            "unknown_note": "",
        },
    ]
    mock_vlm_server.set_stage_response("S3", {"segments": cands})
    mock_vlm_server.set_stage_response("S4A", {"merged_segments": cands})

    def _s4b_handler(req: dict[str, Any]) -> dict[str, Any]:
        txt = json.dumps(req["messages"], ensure_ascii=False)
        if "boundary_idle" in txt:
            return {
                "keep_segment": True,
                "t_start": 0.0,
                "t_end": 0.2,
                "label_id": "boundary_idle",
                "affected_arm": "both",
                "splice_policy": "cut_clean",
                "visual_evidence": cands[0]["visual_evidence"],
                "confidence": 0.92,
                "reject_reason": "",
            }
        return {
            "keep_segment": True,
            "t_start": 0.5,
            "t_end": 1.6,
            "label_id": "mid_idle_freeze",
            "affected_arm": "both",
            "splice_policy": "cut_clean",
            "visual_evidence": cands[1]["visual_evidence"],
            "confidence": 0.88,
            "reject_reason": "",
        }

    mock_vlm_server.set_stage_response("S4B", _s4b_handler)

    ing = _cli(
        e2e_workspace,
        e2e_config_dir,
        [
            "ingest",
            "--data",
            str(multi_root),
            "--data",
            str(single_root),
            "--json",
        ],
    )
    assert ing.returncode == 0, f"ingest failed: {ing.stderr}\n{ing.stdout}"
    rep = _parse_json_stdout(ing.stdout)
    assert rep["inserted"] == 2

    # Check single-cam 1x2 tile and kinematics=None
    s_handles = auto_detect(single_root, adapter_name="video_dir")
    assert len(s_handles) == 1
    with get_adapter("video_dir").open(s_handles[0]) as view:
        assert view.kinematics is None
        assert select_tile_mode(view.camera_roles) == "single_1x2"
        tile = compose_2x2_tile(view.media, view.camera_roles, t_s=0.5, cell_w=240, cell_h=180)
        assert tile.shape == (180 + 26, 480, 3)

    run_res = _cli(e2e_workspace, e2e_config_dir, ["run-vlm"], timeout_s=120.0)
    assert run_res.returncode == 0, f"run-vlm failed: {run_res.stderr}\n{run_res.stdout}"

    repo = Repository(e2e_workspace)
    try:
        eps = repo.list_episodes()
        assert len(eps) == 2
        for ep in eps:
            assert ep["has_kinematics"] == 0
            assert ep["adapter_type"] == "video_dir"
            resolved = repo.resolve_episode(ep["episode_id"])
            bnd = [s for s in resolved.qc_segments if s.label_id == "boundary_idle"]
            mid = [s for s in resolved.qc_segments if s.label_id == "mid_idle_freeze"]
            assert bnd and bnd[0].splice_policy == "cut_clean"
            assert mid and mid[0].vlm_raw_policy == "cut_clean"
            assert mid[0].splice_policy == "mask_loss"
            assert mid[0].guardrail.gate_type == "visual_gate"
            assert mid[0].guardrail.downgraded is True

            # Approve for zero_copy_index export
            repo.save_review(
                ep["episode_id"],
                author="lead_e2e",
                role="lead",
                payload=ReviewWritePayload(
                    base_run_id=ep["active_run_id"],
                    action="approve",
                    qc_segments=resolved.qc_segments,
                    success=SuccessVerdict(passed=True, verdict="success", confidence=0.95),
                    macro_stages=resolved.macro_stages,
                    watch_coverage=WatchCoverage(
                        watched_intervals=[(0.0, ep["duration_s"])],
                        watched_s=ep["duration_s"],
                        duration_s=ep["duration_s"],
                        coverage_ratio=1.0,
                        confirmed=True,
                    ),
                ),
            )
    finally:
        repo.close()

    zci_out = tmp_path / "out_zci"
    exp_res = _cli(
        e2e_workspace,
        e2e_config_dir,
        ["export", "--format", "zero_copy_index", "--recipe", "bc_clean", "--out", str(zci_out)],
    )
    assert exp_res.returncode == 0, f"export failed: {exp_res.stderr}\n{exp_res.stdout}"

    pq_files = list(zci_out.rglob("qc_index.parquet"))
    manifest_files = list(zci_out.rglob("manifest.jsonl"))
    subtasks_files = list(zci_out.rglob("subtasks.json"))
    assert pq_files, f"Missing qc_index.parquet under {zci_out}"
    assert manifest_files, f"Missing manifest.jsonl under {zci_out}"
    assert subtasks_files, f"Missing subtasks.json under {zci_out}"
    # Strictly zero copied video payloads
    assert not list(zci_out.rglob("*.mp4")), "zero_copy_index must not copy MP4 payloads!"

    tbl = pq.read_table(pq_files[0])
    assert tbl.num_rows >= 2


# ---------------------------------------------------------------------------
# Scenario 4: LeRobot Round-Trip Export (81D + CFR MP4s + subtasks.parquet)
# ---------------------------------------------------------------------------


def test_scenario_04_lerobot_roundtrip_export_81d_and_subtasks(
    e2e_workspace: Path,
    e2e_config_dir: Path,
    mock_vlm_server: Any,
    tmp_path: Path,
) -> None:
    from robo_vlm_qc.models.review import ReviewWritePayload, SuccessVerdict, WatchCoverage
    from robo_vlm_qc.store import Repository

    lr_src = h.make_synthetic_lerobot(
        tmp_path / "pick_and_place_lr_src",
        version="v3",
        n_episodes=2,
        duration_s=3.0,
        task_name="pick_and_place",
        stationary=True,
    )

    cands = [
        {
            "t_start": 0.0,
            "t_end": 0.4,
            "label_id": "boundary_idle",
            "affected_arm": "both",
            "splice_policy": "cut_clean",
            "visual_evidence": {
                "initial_state": "Leading idle pose.",
                "process_change": "No motion.",
                "end_state": "Ready to start.",
            },
            "confidence": 0.94,
            "unknown_note": "",
        },
        {
            "t_start": 0.8,
            "t_end": 1.4,
            "label_id": "missed_grasp_retry",
            "affected_arm": "left",
            "splice_policy": "mask_loss",
            "visual_evidence": {
                "initial_state": "Left arm reaches toward the target object.",
                "process_change": "Initial grasp misses and retries.",
                "end_state": "Grasp succeeds on retry.",
            },
            "confidence": 0.90,
            "unknown_note": "",
        },
        {
            "t_start": 1.8,
            "t_end": 2.2,
            "label_id": "mid_idle_freeze",
            "affected_arm": "both",
            "splice_policy": "cut_clean",
            "visual_evidence": {
                "initial_state": "Both arms hover.",
                "process_change": "Mid-episode stationary pause.",
                "end_state": "Manipulation resumes.",
            },
            "confidence": 0.92,
            "unknown_note": "",
        },
    ]
    mock_vlm_server.set_stage_response("S3", {"segments": cands})
    mock_vlm_server.set_stage_response("S4A", {"merged_segments": cands})

    def _s4b_handler(req: dict[str, Any]) -> dict[str, Any]:
        txt = json.dumps(req["messages"], ensure_ascii=False)
        if "boundary_idle" in txt:
            seg = cands[0]
        elif "mid_idle_freeze" in txt:
            seg = cands[2]
        else:
            seg = cands[1]
        return {
            "keep_segment": True,
            "t_start": seg["t_start"],
            "t_end": seg["t_end"],
            "label_id": seg["label_id"],
            "affected_arm": seg["affected_arm"],
            "splice_policy": seg["splice_policy"],
            "visual_evidence": seg["visual_evidence"],
            "confidence": 0.91,
            "reject_reason": "",
        }

    mock_vlm_server.set_stage_response("S4B", _s4b_handler)

    ing = _cli(e2e_workspace, e2e_config_dir, ["ingest", "--data", str(lr_src), "--json"])
    assert ing.returncode == 0, f"ingest failed: {ing.stderr}\n{ing.stdout}"
    assert _parse_json_stdout(ing.stdout)["inserted"] == 2

    run_res = _cli(e2e_workspace, e2e_config_dir, ["run-vlm"], timeout_s=120.0)
    assert run_res.returncode == 0, f"run-vlm failed: {run_res.stderr}\n{run_res.stdout}"

    repo = Repository(e2e_workspace)
    try:
        eps = repo.list_episodes()
        assert len(eps) == 2
        for ep in eps:
            assert ep["adapter_type"] == "lerobot"
            resolved = repo.resolve_episode(ep["episode_id"])
            repo.save_review(
                ep["episode_id"],
                author="lead_e2e",
                role="lead",
                payload=ReviewWritePayload(
                    base_run_id=ep["active_run_id"],
                    action="approve",
                    qc_segments=resolved.qc_segments,
                    success=SuccessVerdict(passed=True, verdict="success", confidence=0.95),
                    macro_stages=resolved.macro_stages,
                    watch_coverage=WatchCoverage(
                        watched_intervals=[(0.0, ep["duration_s"])],
                        watched_s=ep["duration_s"],
                        duration_s=ep["duration_s"],
                        coverage_ratio=1.0,
                        confirmed=True,
                    ),
                ),
            )
    finally:
        repo.close()

    out_dir = tmp_path / "lerobot_export"
    exp_res = _cli(
        e2e_workspace,
        e2e_config_dir,
        ["export", "--format", "lerobot_v3", "--recipe", "bc_clean", "--out", str(out_dir)],
        timeout_s=120.0,
    )
    assert exp_res.returncode == 0, f"export failed: {exp_res.stderr}\n{exp_res.stdout}"

    export_root = _find_lerobot_root(out_dir)
    summary = h.assert_81d_lerobot_v3_valid(export_root)
    assert summary["info"]["robot_type"] == "bimanual_robot"
    assert 0 < summary["total_rows"] < 180  # boundary_idle cut trimmed frames

    subtasks_tbl = pq.read_table(export_root / "meta" / "subtasks.parquet")
    assert subtasks_tbl.num_rows >= 2

    data_tbl = pq.read_table(sorted((export_root / "data").rglob("*.parquet"))[0])
    masks = np.asarray(data_tbl["action_loss_mask"].to_pylist(), dtype=np.float32)
    is_tail = np.asarray(data_tbl["is_tail"].to_pylist(), dtype=bool)
    subtask_idx = np.asarray(data_tbl["subtask_index"].to_pylist(), dtype=np.int64)
    assert np.any(masks[:, 0] == 0.0), "Expected masked frames during missed_grasp_retry"
    assert np.any(masks[:, 0] == 1.0), "Expected unmasked positive frames"
    assert np.any(is_tail), "Expected tail frames flagged near piece end"
    assert len(set(subtask_idx.tolist())) >= 2


# ---------------------------------------------------------------------------
# Scenario 5: Dual-Role Reviewer -> Lead Studio Workflow (FastAPI TestClient)
# ---------------------------------------------------------------------------


def test_scenario_05_dual_role_reviewer_to_lead_studio_workflow(
    e2e_workspace: Path,
    e2e_config_dir: Path,
    mock_vlm_server: Any,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "pick_and_place_studio"
    h.make_synthetic_h5_raw_v3(data_dir, duration_s=4.0, task_name="pick_and_place")
    assert _cli(e2e_workspace, e2e_config_dir, ["ingest", "--data", str(data_dir)]).returncode == 0
    assert _cli(e2e_workspace, e2e_config_dir, ["run-vlm"]).returncode == 0

    ep_id = "pick_and_place_studio/episode_000000"
    client = h.create_test_client(
        e2e_workspace, e2e_config_dir, reviewer_token="rev-token", lead_token="lead-token"
    )
    rev_headers = {"X-Auth-Token": "rev-token", "X-User": "reviewer_alice"}
    lead_headers = {"X-Auth-Token": "lead-token", "X-User": "lead_bob"}

    # 1. Reviewer fetches queue and claims episode
    q_resp = client.get("/api/episodes", headers=rev_headers)
    assert q_resp.status_code == 200, q_resp.text
    items = _extract_items(q_resp.json())
    assert any(it["episode_id"] == ep_id for it in items)

    detail_resp = client.get(f"/api/episodes/{ep_id}", headers=rev_headers)
    assert detail_resp.status_code == 200, detail_resp.text
    detail = detail_resp.json()
    base_run_id = detail["active_run_id"]
    dur = float(detail["meta"]["duration_s"])

    claim_resp = client.post(
        f"/api/episodes/{ep_id}/claim",
        headers=rev_headers,
        json={"user": "reviewer_alice"},
    )
    assert claim_resp.status_code == 200, claim_resp.text
    assert claim_resp.json()["status"] == "claimed"

    # 2. Reviewer attempts forbidden read-only layer edits (arm_timelines / grounding) -> 422 or 400
    bad_payload = {
        "base_run_id": base_run_id,
        "action": "save_draft",
        "qc_segments": [],
        "success": _make_success_dict(True, dur),
        "macro_stages": _make_macro_stages_dict(dur),
        "watch_coverage": _make_watch_coverage(dur),
        "arm_timelines": {"left": [], "right": []},
        "grounding": {"object_tracks_2d": {}},
    }
    bad_resp = client.post(
        f"/api/episodes/{ep_id}/reviews", headers=rev_headers, json=bad_payload
    )
    assert bad_resp.status_code in (400, 422), f"Expected 400/422, got {bad_resp.status_code}: {bad_resp.text}"

    # 3. Reviewer saves valid 3-layer edit and flags episode for Lead
    edited_segs = [
        _make_qc_segment_dict(
            "rev_seg_1",
            0.0,
            0.4,
            "boundary_idle",
            "badcase",
            "cut_clean",
        ),
        _make_qc_segment_dict(
            "rev_seg_2",
            1.1,
            1.9,
            "missed_grasp_retry",
            "recovery",
            "mask_loss",
            mask_side="left",
            affected_arm="left",
        ),
    ]
    flag_payload = {
        "base_run_id": base_run_id,
        "action": "flag",
        "qc_segments": edited_segs,
        "success": _make_success_dict(True, dur),
        "macro_stages": _make_macro_stages_dict(dur),
        "watch_coverage": _make_watch_coverage(dur),
        "notes": "Reviewer edited boundary and flagged ambiguous retry.",
        "flagged_reason": "Needs lead arbitration on retry window.",
    }
    flag_resp = client.post(
        f"/api/episodes/{ep_id}/reviews", headers=rev_headers, json=flag_payload
    )
    assert flag_resp.status_code == 200, flag_resp.text

    # 4. Lead views Diff (raw_vlm_segments vs edited qc_segments), arbitrates, approves, freezes GT
    lead_detail = client.get(f"/api/episodes/{ep_id}", headers=lead_headers)
    assert lead_detail.status_code == 200, lead_detail.text
    ld = lead_detail.json()
    assert ld["status"] == "flagged"
    assert ld["is_human_reviewed"] is True
    assert len(ld["raw_vlm_segments"]) >= 1
    assert len(ld["qc_segments"]) == 2
    assert ld["review_diff"] is not None

    approve_payload = {
        "base_run_id": base_run_id,
        "action": "approve",
        "qc_segments": edited_segs,
        "success": _make_success_dict(True, dur),
        "macro_stages": _make_macro_stages_dict(dur),
        "watch_coverage": _make_watch_coverage(dur),
        "notes": "Lead arbitrated and approved.",
    }
    app_resp = client.post(
        f"/api/episodes/{ep_id}/reviews", headers=lead_headers, json=approve_payload
    )
    assert app_resp.status_code == 200, app_resp.text

    freeze_payload = dict(approve_payload, action="freeze_gt", notes="Frozen as GT")
    frz_resp = client.post(
        f"/api/episodes/{ep_id}/reviews", headers=lead_headers, json=freeze_payload
    )
    assert frz_resp.status_code == 200, frz_resp.text

    frozen_detail = client.get(f"/api/episodes/{ep_id}", headers=lead_headers).json()
    assert frozen_detail["status"] == "gt_frozen"

    # 5. Further Reviewer edit and claim must be blocked
    blocked_edit = client.post(
        f"/api/episodes/{ep_id}/reviews",
        headers=rev_headers,
        json=dict(approve_payload, action="submit_review"),
    )
    assert blocked_edit.status_code in (403, 409, 423), blocked_edit.text
    blocked_claim = client.post(
        f"/api/episodes/{ep_id}/claim",
        headers=rev_headers,
        json={"user": "reviewer_alice"},
    )
    assert blocked_claim.status_code in (403, 409, 423), blocked_claim.text


# ---------------------------------------------------------------------------
# Scenario 6: VLM Lab Closed Loop (Diagnosis -> Few-Shot -> Cached Eval -> Promote)
# ---------------------------------------------------------------------------


def test_scenario_06_vlm_lab_closed_loop_few_shot_and_cached_eval(
    e2e_workspace: Path,
    e2e_config_dir: Path,
    mock_vlm_server: Any,
    tmp_path: Path,
) -> None:
    from robo_vlm_qc.store import Repository

    data_dir = tmp_path / "pick_and_place_lab"
    h.make_synthetic_h5_raw_v3(data_dir, duration_s=4.0, task_name="pick_and_place")

    # Baseline VLM run produces an intentional FP segment (missed_grasp_retry [1.0, 1.8])
    fp_cand = {
        "t_start": 1.0,
        "t_end": 1.8,
        "label_id": "missed_grasp_retry",
        "affected_arm": "left",
        "splice_policy": "mask_loss",
        "visual_evidence": {
            "initial_state": "Left arm approaches target object.",
            "process_change": "Falsely flagged as a missed grasp retry.",
            "end_state": "Object is grasped cleanly.",
        },
        "confidence": 0.86,
        "unknown_note": "",
    }
    mock_vlm_server.set_stage_response("S3", {"segments": [fp_cand]})
    mock_vlm_server.set_stage_response("S4A", {"merged_segments": [fp_cand]})
    mock_vlm_server.set_stage_response(
        "S4B",
        {
            "keep_segment": True,
            "t_start": 1.0,
            "t_end": 1.8,
            "label_id": "missed_grasp_retry",
            "affected_arm": "left",
            "splice_policy": "mask_loss",
            "visual_evidence": fp_cand["visual_evidence"],
            "confidence": 0.86,
            "reject_reason": "",
        },
    )

    assert _cli(e2e_workspace, e2e_config_dir, ["ingest", "--data", str(data_dir)]).returncode == 0
    assert _cli(e2e_workspace, e2e_config_dir, ["run-vlm"]).returncode == 0

    ep_id = "pick_and_place_lab/episode_000000"
    client = h.create_test_client(
        e2e_workspace, e2e_config_dir, reviewer_token="rev-token", lead_token="lead-token"
    )
    lead_headers = {"X-Auth-Token": "lead-token", "X-User": "lead_bob"}

    detail = client.get(f"/api/episodes/{ep_id}", headers=lead_headers).json()
    base_run_id = detail["active_run_id"]
    dur = float(detail["meta"]["duration_s"])

    # Lead removes the FP segment and freezes GT
    frz = client.post(
        f"/api/episodes/{ep_id}/reviews",
        headers=lead_headers,
        json={
            "base_run_id": base_run_id,
            "action": "freeze_gt",
            "qc_segments": [],
            "success": _make_success_dict(True, dur),
            "macro_stages": detail["macro_stages"],
            "watch_coverage": _make_watch_coverage(dur),
            "notes": "GT has zero QC anomalies.",
        },
    )
    assert frz.status_code == 200, frz.text

    # 1. Run diagnosis -> detects fp_overflag
    diag_resp = client.get(
        "/api/lab/diagnosis",
        headers=lead_headers,
        params={"episode_ids": ep_id},
    )
    assert diag_resp.status_code == 200, diag_resp.text
    diag_text = json.dumps(diag_resp.json(), ensure_ascii=False)
    assert "fp_overflag" in diag_text

    # 2. 1-click Few-Shot creation -> generates 2x2 JPEG tile + DB entry
    fs_resp = client.post(
        "/api/lab/few-shots",
        headers=lead_headers,
        json={
            "episode_id": ep_id,
            "t_start": 1.0,
            "t_end": 1.8,
            "label_id": "missed_grasp_retry",
            "error_type": "fp_overflag",
            "scope": "task:pick_and_place",
            "vlm_wrong_json": {"label_id": "missed_grasp_retry", "splice_policy": "mask_loss"},
            "human_right_json": {"label_id": "none", "splice_policy": "keep_positive"},
            "rationale": "Slow steady grasp approach is normal positive motion, not a retry.",
        },
    )
    assert fs_resp.status_code in (200, 201), fs_resp.text
    fs_tiles = list((e2e_workspace / "artifacts" / "few_shots").glob("*.jpg"))
    assert fs_tiles, "Expected 2x2 JPEG tile saved under artifacts/few_shots/"

    repo = Repository(e2e_workspace)
    try:
        fs_rows = repo.db.connect().execute("SELECT * FROM few_shot_examples").fetchall()
        assert len(fs_rows) == 1
    finally:
        repo.close()

    # 3. Reset request log & configure improved S3 response; run eval sandbox
    mock_vlm_server.reset()
    mock_vlm_server.set_stage_response("S3", {"segments": []})

    eval_resp = client.post(
        "/api/lab/eval",
        headers=lead_headers,
        json={"dataset": "pick_and_place_lab", "few_shot": True},
    )
    assert eval_resp.status_code == 200, eval_resp.text

    # S1, S2, S5 MUST hit StageCache (0 new HTTP calls), while S3 executes with Few-Shot
    eval_stages = [r["stage"] for r in mock_vlm_server.requests]
    assert "S1" not in eval_stages, f"S1 missed StageCache during eval: {eval_stages}"
    assert "S2" not in eval_stages, f"S2 missed StageCache during eval: {eval_stages}"
    assert "S5" not in eval_stages and "S5_CONTROL" not in eval_stages, (
        f"S5 missed StageCache during eval: {eval_stages}"
    )
    assert "S3" in eval_stages, f"S3 did not re-run during eval: {eval_stages}"
    s3_prompt_dump = json.dumps(
        [r["messages"] for r in mock_vlm_server.requests if r["stage"] == "S3"],
        ensure_ascii=False,
    )
    assert "Slow steady grasp approach" in s3_prompt_dump

    # Also verify CLI `robo-vlm-qc eval` runs cleanly in sandbox mode
    cli_eval = _cli(
        e2e_workspace,
        e2e_config_dir,
        ["eval", "--dataset", "pick_and_place_lab", "--few-shot"],
        timeout_s=120.0,
    )
    assert cli_eval.returncode == 0, f"cli eval failed: {cli_eval.stderr}\n{cli_eval.stdout}"

    repo = Repository(e2e_workspace)
    try:
        sandbox_runs = (
            repo.db.connect()
            .execute("SELECT * FROM vlm_runs WHERE is_eval_sandbox = 1")
            .fetchall()
        )
        assert len(sandbox_runs) >= 1
    finally:
        repo.close()

    # 4. Promote candidate config
    promote_resp = client.post(
        "/api/lab/promote",
        headers=lead_headers,
        json={"dataset": "pick_and_place_lab"},
    )
    assert promote_resp.status_code == 200, promote_resp.text


# ---------------------------------------------------------------------------
# Scenario 7: Unknown Anomaly Triage & Label Evolution (.history + version)
# ---------------------------------------------------------------------------


def test_scenario_07_unknown_anomaly_triage_and_label_evolution(
    e2e_workspace: Path,
    e2e_config_dir: Path,
    mock_vlm_server: Any,
    tmp_path: Path,
) -> None:
    from robo_vlm_qc.config import ConfigManager

    data_dir = tmp_path / "pick_and_place_unknown"
    h.make_synthetic_h5_raw_v3(
        data_dir, name="episode_000000.h5", duration_s=4.0, task_name="pick_and_place"
    )
    h.make_synthetic_h5_raw_v3(
        data_dir, name="episode_000001.h5", duration_s=4.0, task_name="pick_and_place", episode_index=1
    )

    unk_cands = [
        {
            "t_start": 1.2,
            "t_end": 2.0,
            "label_id": "unknown_anomaly",
            "affected_arm": "right",
            "splice_policy": "mask_loss",
            "visual_evidence": {
                "initial_state": "Right arm moves near the edge of the tabletop.",
                "process_change": "Wrist cable catches on a tabletop fixture.",
                "end_state": "Cable frees itself and manipulation continues.",
            },
            "confidence": 0.77,
            "unknown_note": "Wrist cable snagging on table edge",
        }
    ]
    mock_vlm_server.set_stage_response("S3", {"segments": unk_cands})
    mock_vlm_server.set_stage_response("S4A", {"merged_segments": unk_cands})
    mock_vlm_server.set_stage_response(
        "S4B",
        {
            "keep_segment": True,
            "t_start": 1.2,
            "t_end": 2.0,
            "label_id": "unknown_anomaly",
            "affected_arm": "right",
            "splice_policy": "mask_loss",
            "visual_evidence": unk_cands[0]["visual_evidence"],
            "confidence": 0.78,
            "reject_reason": "",
        },
    )

    assert _cli(e2e_workspace, e2e_config_dir, ["ingest", "--data", str(data_dir)]).returncode == 0
    assert _cli(e2e_workspace, e2e_config_dir, ["run-vlm"]).returncode == 0

    client = h.create_test_client(
        e2e_workspace, e2e_config_dir, reviewer_token="rev-token", lead_token="lead-token"
    )
    rev_headers = {"X-Auth-Token": "rev-token", "X-User": "rev_unk"}
    lead_headers = {"X-Auth-Token": "lead-token", "X-User": "lead_unk"}

    ep0 = "pick_and_place_unknown/episode_000000"
    ep1 = "pick_and_place_unknown/episode_000001"
    for ep_id in (ep0, ep1):
        det = client.get(f"/api/episodes/{ep_id}", headers=rev_headers).json()
        dur = float(det["meta"]["duration_s"])
        r = client.post(
            f"/api/episodes/{ep_id}/reviews",
            headers=rev_headers,
            json={
                "base_run_id": det["active_run_id"],
                "action": "flag",
                "qc_segments": det["qc_segments"],
                "success": det["success"],
                "macro_stages": det["macro_stages"],
                "watch_coverage": _make_watch_coverage(dur),
                "flagged_reason": "Unknown visual anomaly needs taxonomy triage.",
            },
        )
        assert r.status_code == 200, r.text

    unk_resp = client.get("/api/lab/unknowns", headers=lead_headers)
    assert unk_resp.status_code == 200, unk_resp.text
    unknowns = _extract_items(unk_resp.json())
    assert len(unknowns) >= 2

    # (a) Merge first unknown into existing label as Few-Shot
    merge_resp = client.post(
        "/api/lab/unknowns/triage",
        headers=lead_headers,
        json={
            "episode_id": ep0,
            "seg_id": unknowns[0].get("seg_id"),
            "action": "merge",
            "target_label_id": "collision_or_knock",
            "rationale": "Cable snagging against fixture belongs under collision_or_knock.",
        },
    )
    assert merge_resp.status_code == 200, merge_resp.text

    # (b) Promote second unknown into a new visual label (and a second label update so .history archives)
    promo1 = client.post(
        "/api/lab/unknowns/triage",
        headers=lead_headers,
        json={
            "episode_id": ep1,
            "seg_id": unknowns[1].get("seg_id"),
            "action": "promote",
            "new_label": {
                "id": "cable_snag_obstruction",
                "name": "Cable Snag Obstruction",
                "category": "failure",
                "default_training_role": "badcase",
                "default_splice_policy": "mask_loss",
                "collector_ui": {
                    "title": "Cable or hose snagged",
                    "hint": "Arm or wrist cable catches on a table edge or fixture and impedes motion.",
                },
                "vlm_semantics": {
                    "definition": "Visible external cable snagging against workspace fixtures and obstructing motion.",
                    "counter_examples": ["Normal unobstructed trajectory"],
                    "visual_cues": ["Cable pulled taut or caught on table corner"],
                },
            },
        },
    )
    assert promo1.status_code in (200, 201), promo1.text

    # Update labels.yaml via Lab config endpoint so .history backup of prior override is guaranteed
    cfg_before = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    labels_payload = cfg_before.labels.model_dump(mode="json")
    labels_payload["version"] = int(labels_payload["version"]) + 1
    upd_resp = client.put(
        "/api/lab/labels",
        headers=lead_headers,
        json=labels_payload,
    )
    assert upd_resp.status_code == 200, upd_resp.text

    override_labels = e2e_workspace / "configs_override" / "labels.yaml"
    assert override_labels.is_file()
    history_snaps = list((e2e_workspace / ".history").rglob("*.json"))
    assert history_snaps, "Expected .history snapshot backup when updating labels.yaml"

    cfg_after = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    assert cfg_after.labels.version >= 2
    assert cfg_after.labels.get("cable_snag_obstruction") is not None


# ---------------------------------------------------------------------------
# Scenario 8: Three-Recipe Export Matrix Comparison (bc_clean / recovery_plus / badcase_pairs)
# ---------------------------------------------------------------------------


def test_scenario_08_three_recipe_export_matrix_comparison(
    e2e_workspace: Path,
    e2e_config_dir: Path,
    tmp_path: Path,
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.models.annotation import (
        ArmSpan,
        ArmTimelines,
        ConfigFingerprintSnapshot,
        MacroStage,
        QcSegment,
        SuccessVerdict,
        VlmRunArtifact,
    )
    from robo_vlm_qc.models.review import ReviewWritePayload, WatchCoverage
    from robo_vlm_qc.store import Repository

    data_dir = tmp_path / "pick_and_place_matrix"
    h.make_synthetic_h5_raw_v3(
        data_dir, name="ep_pass.h5", duration_s=4.0, task_name="pick_and_place", episode_index=0
    )
    h.make_synthetic_h5_raw_v3(
        data_dir, name="ep_fail.h5", duration_s=4.0, task_name="pick_and_place", episode_index=1
    )
    assert _cli(e2e_workspace, e2e_config_dir, ["ingest", "--data", str(data_dir)]).returncode == 0

    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    fp = ConfigFingerprintSnapshot(
        model_routing=cfg.settings.get("vlm", {}).get("stage_models", {}),
        prompt_bundle_version=cfg.fingerprints.prompt_bundle_sha256[:12],
        label_taxonomy_version=str(cfg.labels.version),
        task_spec_version="1.0.0",
    )

    ep_pass_id = "pick_and_place_matrix/ep_pass"
    ep_fail_id = "pick_and_place_matrix/ep_fail"
    stages = [MacroStage.model_validate(s) for s in _make_macro_stages_dict(4.0)]
    arms = ArmTimelines(
        left=[ArmSpan(t_start=0.0, t_end=4.0, state="work", desc="left")],
        right=[ArmSpan(t_start=0.0, t_end=4.0, state="hold", desc="right")],
    )

    pass_segs = [
        QcSegment.model_validate(
            _make_qc_segment_dict("s_cut", 0.0, 0.4, "boundary_idle", "badcase", "cut_clean")
        ),
        QcSegment.model_validate(
            _make_qc_segment_dict(
                "s_rec",
                1.0,
                1.6,
                "missed_grasp_retry",
                "recovery",
                "mask_loss",
                mask_side="left",
                affected_arm="left",
            )
        ),
        QcSegment.model_validate(
            _make_qc_segment_dict(
                "s_bad",
                2.2,
                2.8,
                "hesitation_oscillation",
                "badcase",
                "mask_loss",
                mask_side="both",
                affected_arm="both",
            )
        ),
    ]
    fail_segs = [
        QcSegment.model_validate(
            _make_qc_segment_dict(
                "s_term",
                2.4,
                3.6,
                "terminal_failure",
                "terminal_fail",
                "mask_loss",
                mask_side="both",
                affected_arm="both",
            )
        )
    ]

    repo = Repository(e2e_workspace)
    try:
        now = time.time()
        for ep_id, segs, passed in (
            (ep_pass_id, pass_segs, True),
            (ep_fail_id, fail_segs, False),
        ):
            run_id = f"run_{ep_id.split('/')[-1]}"
            run_art = VlmRunArtifact(
                run_id=run_id,
                episode_id=ep_id,
                created_at=now,
                config_fingerprint=fp,
                macro_stages=stages,
                arm_timelines=arms,
                qc_segments=segs,
                success=SuccessVerdict(passed=passed, verdict="success" if passed else "failure", confidence=0.95),
            )
            repo.save_vlm_run_artifact(run_art)
            repo.save_review(
                ep_id,
                author="lead_e2e",
                role="lead",
                payload=ReviewWritePayload(
                    base_run_id=run_id,
                    action="approve",
                    qc_segments=segs,
                    success=run_art.success,
                    macro_stages=stages,
                    watch_coverage=WatchCoverage(
                        watched_intervals=[(0.0, 4.0)],
                        watched_s=4.0,
                        duration_s=4.0,
                        coverage_ratio=1.0,
                        confirmed=True,
                    ),
                ),
            )
    finally:
        repo.close()

    # Export all 3 recipes in both lerobot_v3 and h5_clean
    for recipe in ("bc_clean", "recovery_plus", "badcase_pairs"):
        lr_out = tmp_path / f"exp_lr_{recipe}"
        h5_out = tmp_path / f"exp_h5_{recipe}"
        r_lr = _cli(
            e2e_workspace,
            e2e_config_dir,
            ["export", "--format", "lerobot_v3", "--recipe", recipe, "--out", str(lr_out)],
            timeout_s=120.0,
        )
        assert r_lr.returncode == 0, f"lerobot_v3 {recipe} failed: {r_lr.stderr}\n{r_lr.stdout}"
        r_h5 = _cli(
            e2e_workspace,
            e2e_config_dir,
            ["export", "--format", "h5_clean", "--recipe", recipe, "--out", str(h5_out)],
            timeout_s=120.0,
        )
        assert r_h5.returncode == 0, f"h5_clean {recipe} failed: {r_h5.stderr}\n{r_h5.stdout}"

    # Verify bc_clean vs recovery_plus on lerobot_v3
    bc_root = _find_lerobot_root(tmp_path / "exp_lr_bc_clean")
    rec_root = _find_lerobot_root(tmp_path / "exp_lr_recovery_plus")
    bad_root = _find_lerobot_root(tmp_path / "exp_lr_badcase_pairs")

    bc_info = h.assert_81d_lerobot_v3_valid(bc_root)
    rec_info = h.assert_81d_lerobot_v3_valid(rec_root)
    assert bc_info["info"]["total_episodes"] == 1
    assert rec_info["info"]["total_episodes"] == 1

    bc_tbl = pq.read_table(sorted((bc_root / "data").rglob("*.parquet"))[0])
    rec_tbl = pq.read_table(sorted((rec_root / "data").rglob("*.parquet"))[0])
    bc_mask = np.asarray(bc_tbl["action_loss_mask"].to_pylist(), dtype=np.float32)
    rec_mask = np.asarray(rec_tbl["action_loss_mask"].to_pylist(), dtype=np.float32)

    # In bc_clean, both recovery (left=0) and badcase (both=0) are masked;
    # In recovery_plus, recovery is unmasked (1.0) so strictly fewer left-masked rows!
    bc_left_zero = int(np.sum(bc_mask[:, 0] == 0.0))
    rec_left_zero = int(np.sum(rec_mask[:, 0] == 0.0))
    assert bc_left_zero > rec_left_zero > 0, (
        f"Expected bc_clean masked rows ({bc_left_zero}) > recovery_plus ({rec_left_zero}) > 0"
    )

    # Verify badcase_pairs includes failed episode & exports badcase windows in both lerobot_v3 and h5_clean
    bad_h5_files = sorted((tmp_path / "exp_h5_badcase_pairs").rglob("*.h5"))
    assert len(bad_h5_files) >= 1
    found_fail_h5 = False
    for hp in bad_h5_files:
        with h5py.File(str(hp), "r") as hf:
            assert "/qc/action_loss_mask" in hf
            assert "/qc/is_tail" in hf
            assert "/qc/subtask_index" in hf
            assert "/qc/resolved_json" in hf
            if "ep_fail" in hp.name or "terminal_failure" in str(hf["/qc/resolved_json"][()]):
                found_fail_h5 = True
    assert found_fail_h5, "badcase_pairs h5_clean export must include failed episode/terminal_failure!"

    bad_parquets = sorted((bad_root / "data").rglob("*.parquet"))
    assert bad_parquets
    bad_rows = sum(pq.read_table(p).num_rows for p in bad_parquets)
    assert bad_rows > 0


# ---------------------------------------------------------------------------
# Scenario 9: Concurrent Batch Job SSE Progress & Claim Lease Timeout Recovery
# ---------------------------------------------------------------------------


def test_scenario_09_concurrent_batch_job_sse_and_lease_recovery(
    e2e_workspace: Path,
    e2e_config_dir: Path,
    mock_vlm_server: Any,
    tmp_path: Path,
) -> None:
    h.write_workspace_config(
        e2e_workspace,
        extra_overrides={"review": {"claim_ttl_s": 0.1}},
    )

    data_dir = tmp_path / "pick_and_place_batch_sse"
    h.make_synthetic_h5_raw_v3(
        data_dir, name="ep_00.h5", duration_s=2.0, task_name="pick_and_place", episode_index=0
    )
    h.make_synthetic_h5_raw_v3(
        data_dir, name="ep_01.h5", duration_s=2.0, task_name="pick_and_place", episode_index=1
    )
    assert _cli(e2e_workspace, e2e_config_dir, ["ingest", "--data", str(data_dir)]).returncode == 0

    ep0 = "pick_and_place_batch_sse/ep_00"
    ep1 = "pick_and_place_batch_sse/ep_01"
    client = h.create_test_client(
        e2e_workspace, e2e_config_dir, reviewer_token="rev-token", lead_token="lead-token"
    )
    lead_headers = {"X-Auth-Token": "lead-token", "X-User": "lead_09"}
    rev_headers = {"X-Auth-Token": "rev-token"}

    # Trigger batch VLM job
    job_resp = client.post(
        "/api/jobs",
        headers=lead_headers,
        json={"job_type": "batch_vlm", "episode_ids": [ep0, ep1]},
    )
    assert job_resp.status_code in (200, 201, 202), job_resp.text
    job_id = job_resp.json()["job_id"]

    # Stream SSE progress events until terminal completion
    sse_events: list[dict[str, Any]] = []
    with client.stream("GET", f"/api/jobs/{job_id}/events", headers=lead_headers) as stream_resp:
        assert stream_resp.status_code == 200
        for line in stream_resp.iter_lines():
            if not line:
                continue
            if line.startswith("data:"):
                raw_data = line[len("data:") :].strip()
                if raw_data:
                    evt = json.loads(raw_data)
                    sse_events.append(evt)
                    if evt.get("status") in ("completed", "failed"):
                        break

    assert sse_events, "Expected at least one SSE job event"
    assert sse_events[-1]["status"] == "completed", sse_events[-1]
    assert sse_events[-1]["progress_done"] == 2

    # Claim lease expiration & recovery
    c1 = client.post(
        f"/api/episodes/{ep0}/claim",
        headers={**rev_headers, "X-User": "reviewer_one"},
        json={"user": "reviewer_one"},
    )
    assert c1.status_code == 200, c1.text
    assert c1.json()["claimed_by"] == "reviewer_one"

    # Second reviewer blocked while lease is active
    c2_blocked = client.post(
        f"/api/episodes/{ep0}/claim",
        headers={**rev_headers, "X-User": "reviewer_two"},
        json={"user": "reviewer_two"},
    )
    assert c2_blocked.status_code in (409, 423), c2_blocked.text

    # Wait for claim_ttl_s=0.1s to expire, then reclaim cleanly
    time.sleep(0.25)
    c2_ok = client.post(
        f"/api/episodes/{ep0}/claim",
        headers={**rev_headers, "X-User": "reviewer_two"},
        json={"user": "reviewer_two"},
    )
    assert c2_ok.status_code == 200, c2_ok.text
    assert c2_ok.json()["claimed_by"] == "reviewer_two"


# ---------------------------------------------------------------------------
# Scenario 10: Multi-Format (3-In) Serve SPA + Artifact Export & Download
# ---------------------------------------------------------------------------


def test_scenario_10_full_multiformat_serve_spa_and_export_download(
    e2e_workspace: Path,
    e2e_config_dir: Path,
    mock_vlm_server: Any,
    tmp_path: Path,
) -> None:
    ds_h5 = h.make_synthetic_h5_raw_v3(
        tmp_path / "fmt_h5", duration_s=2.0, task_name="pick_and_place"
    ).parent
    ds_lr = h.make_synthetic_lerobot(
        tmp_path / "fmt_lerobot",
        version="v3",
        n_episodes=1,
        duration_s=2.0,
        task_name="bimanual_handover",
    )
    ds_vd = h.make_synthetic_video_dir(
        tmp_path / "fmt_videodir", multi_cam=True, duration_s=2.0, n_episodes=1
    )

    ing = _cli(
        e2e_workspace,
        e2e_config_dir,
        [
            "ingest",
            "--data",
            str(ds_h5),
            "--data",
            str(ds_lr),
            "--data",
            str(ds_vd),
            "--json",
        ],
        timeout_s=120.0,
    )
    assert ing.returncode == 0, f"ingest failed: {ing.stderr}\n{ing.stdout}"
    assert _parse_json_stdout(ing.stdout)["inserted"] == 3

    run_res = _cli(e2e_workspace, e2e_config_dir, ["run-vlm"], timeout_s=180.0)
    assert run_res.returncode == 0, f"run-vlm failed: {run_res.stderr}\n{run_res.stdout}"

    client = h.create_test_client(
        e2e_workspace, e2e_config_dir, reviewer_token="rev-token", lead_token="lead-token"
    )
    lead_headers = {"X-Auth-Token": "lead-token", "X-User": "lead_10"}

    # Check React SPA root (web/dist must be built)
    spa_index = h.REPO_ROOT / "web" / "dist" / "index.html"
    assert spa_index.is_file(), f"Built SPA index missing: {spa_index}"
    root_resp = client.get("/")
    assert root_resp.status_code == 200, root_resp.text
    assert "<html" in root_resp.text.lower()

    # Verify /api/episodes returns all 3 adapter types
    list_resp = client.get("/api/episodes", headers=lead_headers)
    assert list_resp.status_code == 200, list_resp.text
    ep_rows = _extract_items(list_resp.json())
    assert len(ep_rows) == 3
    assert {r["adapter_type"] for r in ep_rows} == {
        "h5_raw_v3",
        "lerobot",
        "video_dir",
    }

    # Lead approves all 3 episodes
    for ep_row in ep_rows:
        ep_id = ep_row["episode_id"]
        det = client.get(f"/api/episodes/{ep_id}", headers=lead_headers).json()
        dur = float(det["meta"]["duration_s"])
        app = client.post(
            f"/api/episodes/{ep_id}/reviews",
            headers=lead_headers,
            json={
                "base_run_id": det["active_run_id"],
                "action": "approve",
                "qc_segments": det["qc_segments"],
                "success": _make_success_dict(True, dur),
                "macro_stages": det["macro_stages"],
                "watch_coverage": _make_watch_coverage(dur),
            },
        )
        assert app.status_code == 200, app.text

    # Trigger h5_clean, lerobot_v3, and viz_mp4 exports via /api/exports & validate manifests + download
    for fmt in ("h5_clean", "lerobot_v3", "viz_mp4"):
        exp_resp = client.post(
            "/api/exports",
            headers=lead_headers,
            json={"format": fmt, "recipe": "bc_clean"},
        )
        assert exp_resp.status_code in (200, 201, 202), exp_resp.text
        exp_data = exp_resp.json()
        export_id = exp_data["export_id"]
        out_dir = Path(exp_data["output_dir"])
        if not out_dir.is_absolute():
            out_dir = e2e_workspace / out_dir
        assert out_dir.is_dir(), f"Export dir missing: {out_dir}"

        if fmt == "lerobot_v3":
            lr_root = out_dir if (out_dir / "meta" / "info.json").is_file() else next(out_dir.rglob("meta/info.json")).parents[1]
            summary = h.assert_81d_lerobot_v3_valid(lr_root)
            assert summary["info"]["robot_type"] == "bimanual_robot"
        elif fmt == "h5_clean":
            h5_files = list(out_dir.rglob("*.h5"))
            assert h5_files, f"Expected exported .h5 under {out_dir}"
            with h5py.File(str(h5_files[0]), "r") as hf:
                assert "/qc/action_loss_mask" in hf
                assert "/qc/subtask_index" in hf
        elif fmt == "viz_mp4":
            mp4s = list(out_dir.rglob("*.mp4"))
            assert mp4s, f"Expected exported viz_mp4 under {out_dir}"
            vi = h.ffprobe_video_info(mp4s[0])
            assert vi["width"] == 1920 and vi["height"] == 1080

        dl_resp = client.get(
            f"/api/exports/{export_id}/download", headers=lead_headers
        )
        assert dl_resp.status_code == 200, dl_resp.text
        assert len(dl_resp.content) > 0
