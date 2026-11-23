"""Unit and integration tests for guardrails, 1080p viz_mp4, and render-mp4 CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from click.testing import CliRunner

from robo_vlm_qc.cli import main
from robo_vlm_qc.config import ConfigManager
from robo_vlm_qc.export.viz_mp4 import render_viz_mp4
from robo_vlm_qc.guardrail import apply_hallucination_check, apply_splice_gate, apply_visual_gate
from robo_vlm_qc.io import auto_detect, get_adapter
from robo_vlm_qc.io.base import KinematicsData
from robo_vlm_qc.models.annotation import (
    ArmSpan,
    ArmTimelines,
    ConfigFingerprintSnapshot,
    CriterionResult,
    GroundingKeyframe,
    GroundingResult,
    GroundingTrack,
    MacroStage,
    QcSegment,
    StageRefs,
    SuccessVerdict,
    VisualEvidence,
    VlmRunArtifact,
)
from robo_vlm_qc.store import Repository


def _kin(T: int = 120, fps: float = 30.0) -> KinematicsData:
    ts = np.arange(T, dtype=np.float64) / fps
    return KinematicsData(
        timestamps_s=ts,
        eef_pos=np.zeros((T, 2, 3), dtype=np.float64),
        eef_rotvec=np.zeros((T, 2, 3), dtype=np.float64),
        hand_q=np.zeros((T, 44), dtype=np.float64),
        torso_q=np.zeros((T, 4), dtype=np.float64),
        chassis_pose=np.zeros((T, 3), dtype=np.float64),
        valid_mask=np.ones(T, dtype=bool),
    )


def _cut_seg(t_start: float = 1.0, t_end: float = 2.0, arm: str = "left") -> QcSegment:
    return QcSegment(
        seg_id="seg_1",
        t_start=t_start,
        t_end=t_end,
        label_id="mid_idle_freeze",
        name="Mid-trajectory idle freeze",
        training_role="badcase",
        splice_policy="cut_clean",
        mask_side=None,
        affected_arm=arm,  # type: ignore[arg-type]
        visual_evidence=VisualEvidence(
            initial_state="Arms hover in place.",
            process_change="Mid-trajectory freeze with no motion.",
            end_state="Freeze ends and motion resumes.",
        ),
        vlm_raw_policy="cut_clean",
    )


def test_splice_gate_thresholds_and_bimanual_excursion() -> None:
    cfg = ConfigManager()

    # Torso jump > 0.05 rad downgrades
    k_torso = _kin()
    k_torso.torso_q[60:, 0] = 0.06  # type: ignore[index]
    res_torso = apply_splice_gate([_cut_seg()], kinematics=k_torso, cfg=cfg, duration_s=4.0)
    assert res_torso[0].splice_policy == "mask_loss"
    assert res_torso[0].mask_side == "left"
    assert res_torso[0].vlm_raw_policy == "cut_clean"
    assert res_torso[0].guardrail.downgraded is True
    assert "torso_rad" in (res_torso[0].guardrail.downgrade_reason or "")

    # Chassis yaw wrapped across pi: 3.10 -> -3.10 is 0.083 rad > 0.05 rad -> downgrade
    k_yaw = _kin()
    k_yaw.chassis_pose[:60, 2] = 3.10  # type: ignore[index]
    k_yaw.chassis_pose[60:, 2] = -3.10  # type: ignore[index]
    res_yaw = apply_splice_gate([_cut_seg()], kinematics=k_yaw, cfg=cfg, duration_s=4.0)
    assert res_yaw[0].splice_policy == "mask_loss"
    assert "chassis_yaw_rad" in (res_yaw[0].guardrail.downgrade_reason or "")

    # Mid-span other-arm excursion returning to 0 at end >= 3.0cm -> bimanual_coupled downgrade
    k_bim = _kin()
    k_bim.eef_pos[45, 1, 0] = 0.035  # type: ignore[index]
    res_bim = apply_splice_gate([_cut_seg(arm="left")], kinematics=k_bim, cfg=cfg, duration_s=4.0)
    assert res_bim[0].splice_policy == "mask_loss"
    assert res_bim[0].guardrail.metrics.bimanual_coupled is True
    assert res_bim[0].guardrail.metrics.other_arm_disp_m >= 0.03


def test_hallucination_and_visual_gate(tmp_path: Path) -> None:
    cfg = ConfigManager()
    k_move = _kin()
    ramp = np.linspace(0.0, 0.22, 31)
    k_move.eef_pos[30:61, 0, 0] = ramp  # type: ignore[index]
    k_move.eef_pos[30:61, 1, 0] = ramp  # type: ignore[index]
    k_move.eef_pos[61:, :, 0] = 0.22  # type: ignore[index]

    res_h = apply_hallucination_check([_cut_seg()], kinematics=k_move, cfg=cfg, duration_s=4.0)
    assert res_h[0].splice_policy == "keep_positive"
    assert res_h[0].mask_side is None
    assert res_h[0].guardrail.guardrail_override is True

    # Visual gate pure video
    ep_dir = tmp_path / "vg_ds" / "ep_0000"
    ep_dir.mkdir(parents=True)
    writer = cv2.VideoWriter(str(ep_dir / "main.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (160, 120))
    for i in range(90):
        fr = np.full((120, 160, 3), 50, dtype=np.uint8)
        if i >= 60:
            fr[:, :] = (20, 220, 240)
        writer.write(fr)
    writer.release()

    ad = get_adapter("video_dir")
    with ad.open(ad.discover(tmp_path / "vg_ds")[0]) as view:
        out_same = apply_visual_gate([_cut_seg(0.4, 1.2)], media=view.media, cfg=cfg, duration_s=view.duration_s)
        assert out_same[0].splice_policy == "cut_clean"
        out_diff = apply_visual_gate([_cut_seg(0.4, 2.5)], media=view.media, cfg=cfg, duration_s=view.duration_s)
        assert out_diff[0].splice_policy == "mask_loss"
        assert out_diff[0].guardrail.gate_type == "visual_gate"


def test_viz_mp4_and_cli_render_mp4(tmp_path: Path) -> None:
    ds_dir = tmp_path / "hud_ds" / "ep_0000"
    ds_dir.mkdir(parents=True)
    writer = cv2.VideoWriter(str(ds_dir / "main.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    for i in range(45):
        writer.write(np.full((240, 320, 3), (40 + i, 90, 150), dtype=np.uint8))
    writer.release()

    ws = tmp_path / "ws"
    runner = CliRunner()
    res_ing = runner.invoke(main, ["-w", str(ws), "ingest", "--data", str(tmp_path / "hud_ds"), "--json"])
    assert res_ing.exit_code == 0, res_ing.output
    ep_id = json.loads(res_ing.output)["episodes"][0]["episode_id"]

    repo = Repository(ws)
    try:
        art = VlmRunArtifact(
            run_id="run_test_hud_01",
            episode_id=ep_id,
            created_at=1700000000.0,
            config_fingerprint=ConfigFingerprintSnapshot(
                model_routing={"s1_grounding": "gpt-4o"},
                prompt_bundle_version="1.0.0",
                label_taxonomy_version="v1",
                task_spec_version="1.0.0",
            ),
            stage_refs=StageRefs(s1_grounding="abc123"),
            grounding=GroundingResult(
                keyframes_s=[0.0, 1.5],
                object_tracks_2d={
                    "target_object": GroundingTrack(
                        object_id="target_object",
                        name="Target Object",
                        keyframes=[
                            GroundingKeyframe(row=0, time_s=0.0, visible=True, bbox_norm_1000=[200, 200, 450, 500], state="ready"),
                            GroundingKeyframe(row=44, time_s=1.5, visible=True, bbox_norm_1000=[300, 220, 550, 520], state="placed", status_kind="good"),
                        ],
                    )
                },
            ),
            macro_stages=[MacroStage(stage_id="S1", t_start=0.0, t_end=1.5, label="Pick and place object")],
            arm_timelines=ArmTimelines(
                left=[ArmSpan(t_start=0.0, t_end=1.5, state="work", desc="Left arm moves")],
                right=[ArmSpan(t_start=0.0, t_end=1.5, state="hold", desc="Right arm holds")],
            ),
            qc_segments=[
                QcSegment(
                    seg_id="seg_001",
                    t_start=0.3,
                    t_end=0.9,
                    label_id="missed_grasp_retry",
                    name="Missed grasp retry",
                    training_role="recovery",
                    splice_policy="mask_loss",
                    mask_side="left",
                    affected_arm="left",
                    visual_evidence=VisualEvidence(initial_state="a", process_change="b", end_state="c"),
                    vlm_raw_policy="mask_loss",
                )
            ],
            success=SuccessVerdict(
                passed=True,
                verdict="success",
                confidence=0.95,
                criteria=[CriterionResult(criterion="Target object placed stably", met=True, satisfied_at_s=1.2)],
            ),
        )
        repo.save_vlm_run_artifact(art)
        resolved = repo.resolve_episode(ep_id)
    finally:
        repo.close()

    handle = auto_detect(tmp_path / "hud_ds", adapter_name="video_dir")[0]
    with get_adapter("video_dir").open(handle) as view:
        out_mp4 = render_viz_mp4(resolved, view, tmp_path / "direct_hud.mp4")
    assert out_mp4.is_file()

    cli_mp4 = tmp_path / "cli_hud.mp4"
    res_mp4 = runner.invoke(
        main,
        ["-w", str(ws), "render-mp4", "--episode", ep_id, "--out", str(cli_mp4)],
    )
    assert res_mp4.exit_code == 0, res_mp4.output
    assert cli_mp4.is_file()

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,codec_name", "-of", "json", str(cli_mp4)],
        capture_output=True,
        text=True,
        check=True,
    )
    st = json.loads(probe.stdout)["streams"][0]
    assert st["width"] == 1920 and st["height"] == 1080 and st["codec_name"] == "h264"

    mod_res = subprocess.run(
        [sys.executable, "-m", "robo_vlm_qc", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert mod_res.returncode == 0
    assert "run-vlm" in mod_res.stdout and "render-mp4" in mod_res.stdout
