"""Unit & integration tests for widened MAIN-ZOOM, START-vs-END compare, 30Hz kinematics summary,
upgraded English prompts/labels, S4A short-circuit & S4B kinematics edge-idle skip,
sequential on-demand S5, DAG wall-clock/pricing, and arm-aware/edge-snapping guardrails."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from robo_vlm_qc.config import ConfigManager
from robo_vlm_qc.guardrail import apply_hallucination_check, apply_splice_gate
from robo_vlm_qc.io import auto_detect, get_adapter
from robo_vlm_qc.io.base import EpisodeView, KinematicsData
from robo_vlm_qc.io.media import (
    DEFAULT_MAIN_ZOOM,
    ZoomBox,
    compose_2x2_tile,
    compose_start_end_compare,
)
from robo_vlm_qc.models.annotation import (
    CriterionResult,
    MacroStage,
    QcSegment,
    VisualEvidence,
)
from robo_vlm_qc.store import Repository
from robo_vlm_qc.vlm.client import (
    BudgetTracker,
    VlmBlindVisionError,
    VlmCircuitOpenError,
    VlmError,
)
from robo_vlm_qc.vlm.dag import VlmDag
from robo_vlm_qc.vlm.prompt_builder import (
    PromptBuilder,
    format_window_kinematics_summary,
)
from robo_vlm_qc.vlm.stage2_macro import run_stage2_macro
from robo_vlm_qc.vlm.stage3_discover import run_stage3_discover
from robo_vlm_qc.vlm.stage4_verify import (
    _needs_s4a_vlm_merge,
    run_stage4_verify,
    run_stage4a_merge,
    run_stage4b_verify,
)
from robo_vlm_qc.vlm.stage5_success import (
    _derive_verdict,
    _plan_dense_tail_times,
    run_stage5_success,
)

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")


class _FakeClient:
    """In-memory deterministic VlmClient double capturing calls, tiles, and overrides."""

    def __init__(self, handler: Any) -> None:
        self.handler = handler
        self.calls: list[dict[str, Any]] = []
        self.budget = BudgetTracker()

    def is_cancelled(self) -> bool:
        return False

    def model_for_stage(self, stage: str, override: str | None = None) -> str:
        return override or "gpt-4o"

    def chat_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        model = kwargs.get("model_override") or "gpt-4o"
        if kwargs.get("budget") is not None:
            kwargs["budget"].record(
                stage=kwargs.get("stage", "unknown"),
                model=model,
                prompt_tokens=1000,
                completion_tokens=500,
                total_tokens=1500,
            )
        return self.handler(kwargs)


def _make_kin(
    duration_s: float = 6.0,
    fps: float = 30.0,
    *,
    left_pos_fn: Any = None,
    right_pos_fn: Any = None,
) -> KinematicsData:
    n = int(round(duration_s * fps)) + 1
    ts = np.arange(n, dtype=np.float64) / fps
    eef_pos = np.zeros((n, 2, 3), dtype=np.float64)
    if left_pos_fn is not None:
        eef_pos[:, 0, :] = left_pos_fn(ts)
    if right_pos_fn is not None:
        eef_pos[:, 1, :] = right_pos_fn(ts)
    return KinematicsData(
        timestamps_s=ts,
        eef_pos=eef_pos,
        eef_rotvec=np.zeros((n, 2, 3), dtype=np.float64),
        hand_q=np.zeros((n, 12), dtype=np.float64),
        torso_q=np.zeros((n, 4), dtype=np.float64),
        chassis_pose=np.zeros((n, 3), dtype=np.float64),
        valid_mask=np.ones(n, dtype=bool),
    )


def _make_wrist_video_dir(root: Path, duration_s: float = 4.0) -> Path:
    ep_dir = root / "bench_ds" / "ep_test"
    ep_dir.mkdir(parents=True, exist_ok=True)
    n = int(round(duration_s * 30.0))
    w_main = cv2.VideoWriter(
        str(ep_dir / "main.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 360)
    )
    for i in range(n):
        fr = np.full((360, 640, 3), (40, 40, 40), dtype=np.uint8)
        # BGR bright red patch at left hovering zone (x in [0.14..0.22], y in [0.40..0.55]) -> outside old x0=0.25
        fr[144:198, 90:140] = (0, 0, 240)
        # BGR bright green patch at foreground drop zone (x in [0.45..0.58], y in [0.72..0.84]) -> outside old y1=0.648
        fr[260:302, 288:371] = (0, 240, 0)
        if i >= n // 2:
            # Blue patch in second half so START vs END visibly differ
            fr[100:150, 400:460] = (240, 20, 20)
        w_main.write(fr)
    w_main.release()
    for cam in ("left_wrist", "right_wrist"):
        w = cv2.VideoWriter(
            str(ep_dir / f"{cam}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240)
        )
        for _ in range(n):
            w.write(np.full((240, 320, 3), (70, 70, 70), dtype=np.uint8))
        w.release()
    return root / "bench_ds"


def _seg(
    seg_id: str,
    t_start: float,
    t_end: float,
    *,
    label_id: str = "mid_idle_freeze",
    arm: str = "both",
    policy: str = "cut_clean",
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
        visual_evidence=VisualEvidence(process_change="candidate"),
        vlm_raw_policy=policy,  # type: ignore[arg-type]
    )


def test_default_main_zoom_and_start_end_compare(tmp_path: Path) -> None:
    assert DEFAULT_MAIN_ZOOM == ZoomBox(x0=0.12, y0=0.10, x1=0.88, y1=0.86)
    assert DEFAULT_MAIN_ZOOM.x0 <= 0.15
    assert DEFAULT_MAIN_ZOOM.y1 >= 0.85

    ds_dir = _make_wrist_video_dir(tmp_path, duration_s=1.0)
    handle = auto_detect(ds_dir, adapter_name="video_dir")[0]
    with get_adapter("video_dir").open(handle) as view:
        tile = compose_2x2_tile(
            view.media,
            view.camera_roles,
            t_s=0.2,
            cell_w=420,
            cell_h=236,
            include_header=True,
            header_h=26,
        )
        assert tile.shape == (26 + 236 * 2, 420 * 2, 3)
        main_zoom = tile[26 + 236 :, 420:, :]
        red_pixels = np.sum(
            (main_zoom[:, :, 0] > 180) & (main_zoom[:, :, 1] < 60) & (main_zoom[:, :, 2] < 60)
        )
        green_pixels = np.sum(
            (main_zoom[:, :, 1] > 180) & (main_zoom[:, :, 0] < 60) & (main_zoom[:, :, 2] < 60)
        )
        assert red_pixels > 200
        assert green_pixels > 200

        cmp_img = compose_start_end_compare(
            view.media, view.camera_roles, 0.0, view.duration_s
        )
        assert cmp_img.shape == (30 + 360, 640 * 2 + 4, 3)
        # Red center divider
        assert tuple(cmp_img[50, 641]) == (255, 40, 40)
        # Start panel vs end panel differ visually
        assert not np.array_equal(cmp_img[30:, :640], cmp_img[30:, 644:])

        cmp_no_hdr = compose_start_end_compare(
            view.media, view.camera_roles, 0.0, view.duration_s, header_h=0
        )
        assert cmp_no_hdr.shape == (360, 640 * 2 + 4, 3)


def test_format_window_kinematics_summary_english_and_degradation() -> None:
    # 1. High-backtrack oscillation (L) vs stationary stall (R)
    kin_osc = _make_kin(
        duration_s=4.0,
        left_pos_fn=lambda ts: np.stack(
            [0.04 * np.sin(2.0 * np.pi * 0.8 * ts), np.zeros_like(ts), np.zeros_like(ts)],
            axis=1,
        ),
        right_pos_fn=lambda ts: np.zeros((len(ts), 3), dtype=np.float64),
    )
    txt_osc = format_window_kinematics_summary(kin_osc, 0.0, 4.0, slice_s=2.0)
    assert txt_osc != ""
    assert not _CJK_RE.search(txt_osc)
    assert "NEVER apply mechanically" in txt_osc
    assert "wiping" in txt_osc.lower()
    assert "[high-backtrack / low-net-progress]" in txt_osc
    assert "[low-speed stall]" in txt_osc
    assert "reversals=" in txt_osc and "backtrack=" in txt_osc
    assert "long_pause" not in txt_osc and "back_and_forth" not in txt_osc

    # 2. Steady unidirectional progress
    kin_steady = _make_kin(
        duration_s=4.0,
        left_pos_fn=lambda ts: np.stack(
            [0.08 * ts, np.zeros_like(ts), np.zeros_like(ts)],
            axis=1,
        ),
        right_pos_fn=lambda ts: np.stack(
            [np.zeros_like(ts), 0.06 * ts, np.zeros_like(ts)],
            axis=1,
        ),
    )
    txt_steady = format_window_kinematics_summary(kin_steady, 0.0, 4.0, slice_s=2.0)
    assert "[steady unidirectional progress]" in txt_steady
    assert "[high-backtrack / low-net-progress]" not in txt_steady
    assert "reversals=0" in txt_steady

    # 3. NaN & valid_mask=False handled cleanly
    kin_masked = _make_kin(
        duration_s=4.0,
        left_pos_fn=lambda ts: np.stack(
            [0.07 * ts, np.zeros_like(ts), np.zeros_like(ts)],
            axis=1,
        ),
    )
    kin_masked.valid_mask[10:20] = False
    kin_masked.eef_pos[10:20, 0, :] = np.nan  # type: ignore[index]
    txt_masked = format_window_kinematics_summary(kin_masked, 0.0, 4.0, slice_s=2.0)
    assert "[steady unidirectional progress]" in txt_masked
    assert "nan" not in txt_masked.lower()

    # 4. Graceful degradation when kinematics absent or invalid
    assert format_window_kinematics_summary(None, 0.0, 4.0) == ""
    assert format_window_kinematics_summary(
        KinematicsData(timestamps_s=np.linspace(0.0, 4.0, 121), eef_pos=None), 0.0, 4.0
    ) == ""
    kin_invalid = _make_kin(duration_s=4.0)
    kin_invalid.valid_mask[:] = False
    assert format_window_kinematics_summary(kin_invalid, 0.0, 4.0) == ""


def test_upgraded_prompts_and_labels_are_english_and_preserve_oss_invariants() -> None:
    cfg = ConfigManager()
    pb = PromptBuilder(cfg)
    task = cfg.get_task("pick_and_place")
    roles = {"main": "main", "left_wrist": "left_wrist", "right_wrist": "right_wrist"}
    stages = [
        MacroStage(
            stage_id="S1",
            t_start=0.0,
            t_end=20.0,
            label="Pick up target",
            instruction="Pick up the target object from the table",
            summary="Left arm probes repeatedly near target",
        )
    ]
    cand = _seg("s4a_1", 12.0, 24.0, label_id="hesitation_oscillation", arm="left", policy="mask_loss")

    # labels.yaml refined hesitation_oscillation & precision_fine_adjust
    hes = cfg.labels.get("hesitation_oscillation")
    pfa = cfg.labels.get("precision_fine_adjust")
    assert hes is not None and ">=2" in hes.vlm_semantics.definition and ">5s" in hes.vlm_semantics.definition
    assert pfa is not None and "<=4-5s" in pfa.vlm_semantics.definition

    # S2: physical milestones + anom + preserves instruction & grounding summary
    sys_s2, usr_s2 = pb.build_s2(task=task, camera_roles=roles, duration_s=60.0)
    assert "physical milestones" in sys_s2.lower()
    assert 'state="anom"' in sys_s2
    assert '"instruction"' in sys_s2
    assert "NEVER force-fit" in usr_s2

    # S3: S2 does not exempt local anomalies, injects task_goal/task_prompt & optional kinematics
    sys_s3, usr_s3_nokin = pb.build_s3(
        task=task,
        camera_roles=roles,
        macro_stages=stages,
        window_start_s=0.0,
        window_end_s=16.0,
        duration_s=60.0,
    )
    assert "Precision-First" in sys_s3
    assert "hesitation_oscillation" in sys_s3 and "precision_fine_adjust" in sys_s3
    assert "NEVER exempt" in sys_s3
    assert "NOT exempted" in usr_s3_nokin
    assert "30Hz Objective Kinematics Telemetry" not in usr_s3_nokin

    sample_kin = "Note: telemetry\n- [0.00s..2.00s] L: path=18.0cm, net=1.0cm [high-backtrack / low-net-progress]"
    _, usr_s3_kin = pb.build_s3(
        task=task,
        camera_roles=roles,
        macro_stages=stages,
        window_start_s=0.0,
        window_end_s=16.0,
        duration_s=60.0,
        kinematics_summary=sample_kin,
    )
    assert "30Hz Objective Kinematics Telemetry" in usr_s3_kin
    assert "[high-backtrack / low-net-progress]" in usr_s3_kin

    # S4B: Process-Anomaly Anti-False-Rejection Redline + labels_desc + optional kinematics
    sys_s4b, usr_s4b_nokin = pb.build_s4b(
        task=task,
        camera_roles=roles,
        macro_stages=stages,
        candidate=cand,
        window_start_s=10.5,
        window_end_s=25.5,
    )
    assert "Stage 4B" in sys_s4b
    assert "Process-Anomaly Anti-False-Rejection Redline" in sys_s4b
    for lbl_id in ("hesitation_oscillation", "missed_grasp_retry", "object_slip_recovery", "placement_redo"):
        assert lbl_id in sys_s4b
    assert "Active Pure-Visual QC Label Registry" in usr_s4b_nokin
    assert "30Hz Objective Kinematics Telemetry" not in usr_s4b_nokin

    _, usr_s4b_kin = pb.build_s4b(
        task=task,
        camera_roles=roles,
        macro_stages=stages,
        candidate=cand,
        window_start_s=10.5,
        window_end_s=25.5,
        kinematics_summary=sample_kin,
    )
    assert "30Hz Objective Kinematics Telemetry" in usr_s4b_kin
    assert "[high-backtrack / low-net-progress]" in usr_s4b_kin

    # S5 & S5_CONTROL: START_vs_END / START_vs_CUTOFF + tabletop drop scan + task_goal/prompt
    sys_s5, usr_s5 = pb.build_s5(task=task, camera_roles=roles, duration_s=60.0)
    assert "START_vs_END" in sys_s5 and "MAIN-ZOOM" in sys_s5 and "met=false" in sys_s5
    assert "START_vs_END" in usr_s5 and "MAIN-ZOOM" in usr_s5
    sys_c, usr_c = pb.build_s5_control(task=task, cut_s=18.0)
    assert "START_vs_CUTOFF" in sys_c and "START_vs_CUTOFF" in usr_c
    assert task.goal in usr_c

    # Zero CJK across all rendered prompts
    for txt in (sys_s2, usr_s2, sys_s3, usr_s3_kin, sys_s4b, usr_s4b_kin, sys_s5, usr_s5, sys_c, usr_c):
        assert not _CJK_RE.search(txt), f"Unexpected CJK in prompt: {txt[:200]}"

    # Retired label IDs are excluded from BOTH system and user prompts in S3 and S4B
    from robo_vlm_qc.models.taxonomy import LabelRegistry

    raw_lbls = cfg.labels.model_dump(mode="json")
    for lbl in raw_lbls["labels"]:
        if lbl["id"] == "hesitation_oscillation":
            lbl["status"] = "retired"
    pb_ret = PromptBuilder(cfg, labels=LabelRegistry.model_validate(raw_lbls))
    s3_ret_sys, s3_ret_usr = pb_ret.build_s3(
        task=task, camera_roles=roles, macro_stages=stages, window_start_s=0.0, window_end_s=16.0, duration_s=60.0
    )
    assert "hesitation_oscillation" not in s3_ret_sys
    assert "hesitation_oscillation" not in s3_ret_usr
    cand_active = _seg("s4a_2", 12.0, 24.0, label_id="mid_idle_freeze", arm="left", policy="cut_clean")
    s4b_ret_sys, s4b_ret_usr = pb_ret.build_s4b(
        task=task, camera_roles=roles, macro_stages=stages, candidate=cand_active, window_start_s=10.5, window_end_s=25.5
    )
    assert "hesitation_oscillation" not in s4b_ret_sys
    assert "hesitation_oscillation" not in s4b_ret_usr


def test_stage3_stage4b_kinematics_and_stage_overrides(tmp_path: Path) -> None:
    ds_dir = _make_wrist_video_dir(tmp_path, duration_s=4.0)
    handle = auto_detect(ds_dir, adapter_name="video_dir")[0]
    cfg = ConfigManager()
    kin = _make_kin(
        duration_s=4.0,
        left_pos_fn=lambda ts: np.stack(
            [0.04 * np.sin(2.0 * np.pi * 0.8 * ts), np.zeros_like(ts), np.zeros_like(ts)],
            axis=1,
        ),
        right_pos_fn=lambda ts: np.zeros((len(ts), 3), dtype=np.float64),
    )

    with get_adapter("video_dir").open(handle) as base_view:
        view_kin = EpisodeView(meta=base_view.meta, media=base_view.media, kinematics=kin)

        # S2 model_override & thinking_budget_override
        cl_s2 = _FakeClient(lambda _kw: {"macro_stages": [], "arm_timelines": {}})
        run_stage2_macro(
            view=view_kin,
            client=cl_s2,  # type: ignore[arg-type]
            cfg=cfg,
            model_override="gpt-4o-mini",
            thinking_budget_override=256,
        )
        assert cl_s2.calls[0]["model_override"] == "gpt-4o-mini"
        assert cl_s2.calls[0]["thinking_budget"] == 256

        # S3 injects kinematics and uses 420x236, quality=76
        cl_s3 = _FakeClient(
            lambda _kw: {
                "segments": [
                    {
                        "t_start": 0.5,
                        "t_end": 3.5,
                        "label_id": "hesitation_oscillation",
                        "affected_arm": "left",
                        "splice_policy": "mask_loss",
                    }
                ]
            }
        )
        s3_segs = run_stage3_discover(
            view=view_kin,
            task=cfg.get_task("pick_and_place"),
            client=cl_s3,  # type: ignore[arg-type]
            cfg=cfg,
            model_override="gpt-4o-mini",
            thinking_budget_override=512,
        )
        assert len(s3_segs) == 1
        assert cl_s3.calls[0]["jpeg_quality"] == 76
        assert cl_s3.calls[0]["images"][0].shape == (26 + 236 * 2, 420 * 2, 3)
        assert "[high-backtrack / low-net-progress]" in cl_s3.calls[0]["user_prompt"]
        assert cl_s3.calls[0]["model_override"] == "gpt-4o-mini"
        assert cl_s3.calls[0]["thinking_budget"] == 512

        # S4B injects kinematics & labels_desc + forwards model_override
        cl_s4b = _FakeClient(lambda _kw: {"keep_segment": True})
        s4b_segs = run_stage4b_verify(
            view=view_kin,
            candidates=s3_segs,
            task=cfg.get_task("pick_and_place"),
            client=cl_s4b,  # type: ignore[arg-type]
            cfg=cfg,
            model_override="gpt-4o",
        )
        assert len(s4b_segs) == 1
        assert "[high-backtrack / low-net-progress]" in cl_s4b.calls[0]["user_prompt"]
        assert "Active Pure-Visual QC Label Registry" in cl_s4b.calls[0]["user_prompt"]
        assert cl_s4b.calls[0]["model_override"] == "gpt-4o"

        # Without kinematics -> graceful degradation
        cl_s3_nokin = _FakeClient(lambda _kw: {"segments": []})
        run_stage3_discover(view=base_view, client=cl_s3_nokin, cfg=cfg)  # type: ignore[arg-type]
        assert "30Hz Objective Kinematics Telemetry" not in cl_s3_nokin.calls[0]["user_prompt"]


def test_stage4a_short_circuit_and_s4_error_propagation(tmp_path: Path) -> None:
    s_a = _seg("a", 0.5, 1.2)
    s_b = _seg("b", 1.5, 2.5)  # gap 0.3s <= 2.0s
    s_far = _seg("c", 5.0, 6.0)  # gap 3.8s > 2.0s

    assert _needs_s4a_vlm_merge([]) is False
    assert _needs_s4a_vlm_merge([s_a]) is False
    assert _needs_s4a_vlm_merge([s_a, s_far]) is False
    assert _needs_s4a_vlm_merge([s_a, s_b]) is True

    ds_dir = _make_wrist_video_dir(tmp_path, duration_s=8.0)
    handle = auto_detect(ds_dir, adapter_name="video_dir")[0]
    cfg = ConfigManager()

    with get_adapter("video_dir").open(handle) as view:
        # 1. Single candidate & widely separated candidates skip S4A VLM call completely
        cl_zero = _FakeClient(lambda _kw: {"merged_segments": []})
        out_single = run_stage4a_merge(view=view, candidates=[s_a], client=cl_zero, cfg=cfg)  # type: ignore[arg-type]
        assert len(out_single) == 1 and len(cl_zero.calls) == 0

        out_far = run_stage4a_merge(view=view, candidates=[s_a, s_far], client=cl_zero, cfg=cfg)  # type: ignore[arg-type]
        assert len(out_far) == 2 and len(cl_zero.calls) == 0

        # 2. S4A re-raises VlmCircuitOpenError and VlmBlindVisionError
        cl_cb = _FakeClient(lambda _kw: (_ for _ in ()).throw(VlmCircuitOpenError("circuit open")))
        with pytest.raises(VlmCircuitOpenError):
            run_stage4a_merge(view=view, candidates=[s_a, s_b], client=cl_cb, cfg=cfg)  # type: ignore[arg-type]

        cl_blind = _FakeClient(lambda _kw: (_ for _ in ()).throw(VlmBlindVisionError("blind")))
        with pytest.raises(VlmBlindVisionError):
            run_stage4a_merge(view=view, candidates=[s_a, s_b], client=cl_blind, cfg=cfg)  # type: ignore[arg-type]

        # 3. run_stage4_verify short-circuits before S4B when S4A returns empty []
        cl_s4_empty = _FakeClient(lambda _kw: {"merged_segments": []})
        out_empty = run_stage4_verify(view=view, candidates=[s_a, s_b], client=cl_s4_empty, cfg=cfg)  # type: ignore[arg-type]
        assert out_empty == []
        assert [c["stage"] for c in cl_s4_empty.calls] == ["s4a_merge"]

        # 4. S4B propagates VlmError / VlmCircuitOpenError / VlmBlindVisionError instead of swallowing
        for exc_cls in (VlmError, VlmCircuitOpenError, VlmBlindVisionError):
            cl_err = _FakeClient(lambda _kw, e=exc_cls: (_ for _ in ()).throw(e("boom")))
            with pytest.raises(exc_cls):
                run_stage4b_verify(view=view, candidates=[s_a], client=cl_err, cfg=cfg)  # type: ignore[arg-type]


def test_stage4b_kinematics_verified_edge_boundary_idle_skip(tmp_path: Path) -> None:
    ds_dir = _make_wrist_video_dir(tmp_path, duration_s=4.0)
    handle = auto_detect(ds_dir, adapter_name="video_dir")[0]
    cfg = ConfigManager()

    kin_still = _make_kin(duration_s=4.0)
    kin_moving = _make_kin(
        duration_s=4.0,
        left_pos_fn=lambda ts: np.stack([0.15 * ts, np.zeros_like(ts), np.zeros_like(ts)], axis=1),
    )

    edge_head = _seg("b0", 0.15, 1.0, label_id="boundary_idle", arm="both", policy="cut_clean")
    edge_tail = _seg("b1", 3.1, 3.85, label_id="boundary_idle", arm="both", policy="cut_clean")

    with get_adapter("video_dir").open(handle) as base_view:
        view_still = EpisodeView(meta=base_view.meta, media=base_view.media, kinematics=kin_still)
        view_moving = EpisodeView(meta=base_view.meta, media=base_view.media, kinematics=kin_moving)

        # 1. Stationary kinematics skips S4B VLM (0 calls), snaps to exact 0.0 / 4.0, sets confidence_note & guardrail
        cl_skip = _FakeClient(lambda _kw: {"keep_segment": True})
        out_skip = run_stage4b_verify(
            view=view_still, candidates=[edge_head, edge_tail], client=cl_skip, cfg=cfg  # type: ignore[arg-type]
        )
        assert len(cl_skip.calls) == 0
        assert len(out_skip) == 2
        assert out_skip[0].t_start == 0.0 and out_skip[0].t_end == 1.0
        assert out_skip[1].t_start == 3.1 and out_skip[1].t_end == 4.0
        assert all(
            s.visual_evidence.confidence_note == "kinematics_verified_edge_idle"
            and s.guardrail.passed is True
            and s.splice_policy == "cut_clean"
            for s in out_skip
        )

        # 2. Moving arm does NOT skip S4B VLM
        cl_move = _FakeClient(lambda _kw: {"keep_segment": True})
        run_stage4b_verify(
            view=view_moving, candidates=[edge_head], client=cl_move, cfg=cfg  # type: ignore[arg-type]
        )
        assert len(cl_move.calls) == 1

        # 3. Config boundary_idle_kin_skip_s4b=False disables skip
        cfg_off = ConfigManager()
        cfg_off.settings["vlm"]["s4_verify"]["boundary_idle_kin_skip_s4b"] = False
        cl_off = _FakeClient(lambda _kw: {"keep_segment": True})
        run_stage4b_verify(
            view=view_still, candidates=[edge_head], client=cl_off, cfg=cfg_off  # type: ignore[arg-type]
        )
        assert len(cl_off.calls) == 1

        # 4. Pure video (kinematics=None) invokes S4B VLM
        cl_vid = _FakeClient(lambda _kw: {"keep_segment": True})
        run_stage4b_verify(
            view=base_view, candidates=[edge_head], client=cl_vid, cfg=cfg  # type: ignore[arg-type]
        )
        assert len(cl_vid.calls) == 1


def test_stage5_sequential_on_demand_control_adaptive_tail_and_verdict(tmp_path: Path) -> None:
    # 1. Adaptive tail on a 121.5s episode covers [96.5s..121.5s] with <=16 tiles
    times_121 = _plan_dense_tail_times(121.5, tail_s=12.0, body_frames=6, tail_frames=10)
    assert len(times_121) <= 16
    assert times_121[0] == 0.0 and times_121[-1] == 121.5
    drop_band = [t for t in times_121 if 102.0 <= t <= 114.0]
    assert len(drop_band) >= 3

    # 2. _derive_verdict preserves explicit raw_verdict == "failure" even when criteria have met=None
    assert _derive_verdict("failure", [CriterionResult(criterion="c1", met=None)]) == "failure"
    assert _derive_verdict("success", [CriterionResult(criterion="c1", met=False)]) == "failure"
    assert _derive_verdict("unsure", [CriterionResult(criterion="c1", met=True)]) == "success"

    ds_dir = _make_wrist_video_dir(tmp_path, duration_s=4.0)
    handle = auto_detect(ds_dir, adapter_name="video_dir")[0]
    cfg = ConfigManager()

    with get_adapter("video_dir").open(handle) as view:
        # 3. Failure primary verdict skips s5_control completely (only 1 VLM call)
        cl_fail = _FakeClient(
            lambda _kw: {
                "verdict": "failure",
                "confidence": 0.92,
                "criteria": [{"criterion": "done", "met": None, "evidence": "dropped"}],
                "summary": "dropped on table",
            }
        )
        v_fail = run_stage5_success(
            view=view, task=cfg.get_task("pick_and_place"), client=cl_fail, cfg=cfg  # type: ignore[arg-type]
        )
        assert v_fail.verdict == "failure" and v_fail.passed is False
        assert v_fail.control_check is None
        assert [c["stage"] for c in cl_fail.calls] == ["s5_success"]

        # 4. Success primary verdict runs s5_control on-demand with START_vs_END & START_vs_CUTOFF tiles
        cl_ok = _FakeClient(
            lambda kw: {
                "verdict": "failure" if kw["stage"] == "s5_control" else "success",
                "confidence": 0.95,
                "criteria": [
                    {
                        "criterion": "target placed",
                        "met": kw["stage"] != "s5_control",
                        "satisfied_at_s": 3.5 if kw["stage"] != "s5_control" else None,
                        "evidence": "checked",
                    }
                ],
                "summary": "ok",
            }
        )
        v_ok = run_stage5_success(
            view=view,
            task=cfg.get_task("pick_and_place"),
            client=cl_ok,  # type: ignore[arg-type]
            cfg=cfg,
            model_override="gpt-4o",
            thinking_budget_override=256,
        )
        assert v_ok.verdict == "success" and v_ok.passed is True
        assert v_ok.control_check is not None and v_ok.control_check.passed is True
        assert [c["stage"] for c in cl_ok.calls] == ["s5_success", "s5_control"]
        full_call, ctrl_call = cl_ok.calls
        assert full_call["jpeg_quality"] == 78
        assert full_call["images"][0].shape == (30 + 360, 640 * 2 + 4, 3)
        assert full_call["images"][1].shape == (26 + 236 * 2, 420 * 2, 3)
        assert full_call["labels"][0].startswith("START_vs_END")
        assert ctrl_call["labels"][0].startswith("START_vs_CUTOFF")
        assert ctrl_call["images"][0].shape == (30 + 360, 640 * 2 + 4, 3)
        assert full_call["model_override"] == "gpt-4o" and full_call["thinking_budget"] == 256

        # 5. run_control_check=False skips s5_control even on success
        cl_noctrl = _FakeClient(
            lambda _kw: {
                "verdict": "success",
                "confidence": 0.94,
                "criteria": [{"criterion": "done", "met": True, "satisfied_at_s": 3.0}],
                "summary": "ok",
            }
        )
        v_noctrl = run_stage5_success(
            view=view,
            task=cfg.get_task("pick_and_place"),
            client=cl_noctrl,  # type: ignore[arg-type]
            cfg=cfg,
            run_control_check=False,
        )
        assert v_noctrl.verdict == "success" and v_noctrl.passed is True
        assert v_noctrl.control_check is None
        assert [c["stage"] for c in cl_noctrl.calls] == ["s5_success"]


def test_guardrail_arm_aware_hallucination_and_splice_gate_edge_snap() -> None:
    cfg = ConfigManager()
    # Left arm moves monotonically by 0.24m over [1.0, 2.2]; right arm stays completely still
    kin = _make_kin(
        duration_s=4.0,
        left_pos_fn=lambda ts: np.stack(
            [np.clip((ts - 1.0) * 0.20, 0.0, 0.24), np.zeros_like(ts), np.zeros_like(ts)],
            axis=1,
        ),
        right_pos_fn=lambda ts: np.zeros((len(ts), 3), dtype=np.float64),
    )

    seg_left = _seg("h_l", 1.0, 2.2, label_id="mid_idle_freeze", arm="left", policy="cut_clean")
    seg_right = _seg("h_r", 1.0, 2.2, label_id="mid_idle_freeze", arm="right", policy="cut_clean")
    seg_both = _seg("h_b", 1.0, 2.2, label_id="mid_idle_freeze", arm="both", policy="cut_clean")

    out_h = apply_hallucination_check(
        [seg_left, seg_right, seg_both], kinematics=kin, cfg=cfg, duration_s=4.0
    )
    # Left arm moving -> overrides affected_arm="left" and affected_arm="both"
    assert out_h[0].splice_policy == "keep_positive" and out_h[0].guardrail.guardrail_override is True
    assert "left-arm" in out_h[0].guardrail.override_reason
    # Right arm stationary -> does NOT override affected_arm="right"
    assert out_h[1].splice_policy == "cut_clean" and out_h[1].guardrail.guardrail_override is False
    # affected_arm="both" triggers on any moving arm (single-arm)
    assert out_h[2].splice_policy == "keep_positive" and out_h[2].guardrail.guardrail_override is True
    assert "single-arm" in out_h[2].guardrail.override_reason

    # splice_gate snaps edge boundary_idle to exact 0.0 / round(dur, 3)
    kin_still = _make_kin(duration_s=4.0)
    b_head = _seg("bh", 0.22, 1.1, label_id="boundary_idle", arm="both", policy="cut_clean")
    b_tail = _seg("bt", 3.0, 3.78, label_id="boundary_idle", arm="both", policy="cut_clean")
    out_sg = apply_splice_gate([b_head, b_tail], kinematics=kin_still, cfg=cfg, duration_s=4.0)
    assert out_sg[0].t_start == 0.0 and out_sg[0].t_end == 1.1
    assert out_sg[0].guardrail.opt_t_start == 0.0 and out_sg[0].guardrail.opt_t_end == 1.1
    assert out_sg[1].t_start == 3.0 and out_sg[1].t_end == 4.0
    assert out_sg[1].guardrail.opt_t_start == 3.0 and out_sg[1].guardrail.opt_t_end == 4.0


def test_dag_wall_clock_elapsed_and_pricing_cost_accounting(tmp_path: Path) -> None:
    ds_dir = _make_wrist_video_dir(tmp_path / "data", duration_s=2.0)
    handle = auto_detect(ds_dir, adapter_name="video_dir")[0]
    ws = tmp_path / "ws"
    cfg = ConfigManager(workspace_dir=ws)
    cfg.settings["vlm"]["s1_grounding"]["tracking"]["enabled"] = False
    repo = Repository(ws)

    def _handler(kw: dict[str, Any]) -> dict[str, Any]:
        st = kw["stage"]
        if st == "s1_grounding":
            return {"objects": {}}
        if st == "s2_macro":
            return {"macro_stages": [], "arm_timelines": {}}
        if st == "s5_success":
            return {"verdict": "failure", "confidence": 0.9, "criteria": []}
        if st == "s3_discover":
            return {"segments": []}
        return {"merged_segments": []}

    try:
        with get_adapter("video_dir").open(handle) as view:
            repo.ingest_episode(view.meta)
            cl = _FakeClient(_handler)
            dag = VlmDag(workspace=ws, config=cfg, client=cl, repo=repo)  # type: ignore[arg-type]
            art = dag.run_episode(view)
            assert art.usage.total_elapsed_s >= 0.0
            assert art.usage.total_cost_usd > 0.0
            assert any(s.estimated_cost_usd > 0.0 for s in art.usage.stages)
            # Re-running hits stage cache with resolve_client_stage_model
            art_hit = dag.run_episode(view)
            assert art_hit.usage.total_elapsed_s >= 0.0
            assert all(s.cache_hit for s in art_hit.usage.stages)
    finally:
        repo.close()
