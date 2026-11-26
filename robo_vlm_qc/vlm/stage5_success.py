"""Stage 5: Dense-tail task completion verification with sequential on-demand 30% anti-FP control check."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from ..config import ConfigManager
from ..io.base import EpisodeView
from ..io.media import compose_2x2_tile, compose_start_end_compare
from ..models.annotation import ControlCheckResult, CriterionResult, SuccessVerdict
from ..models.taxonomy import TaskSpec
from .client import BudgetTracker, VlmClient, check_client_cancelled
from .prompt_builder import PromptBuilder


def _resolve_s5_cfg(cfg: ConfigManager | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(cfg, ConfigManager):
        return dict(cfg.settings.get("vlm", {}).get("s5_success", {}))
    if isinstance(cfg, dict):
        if "vlm" in cfg and isinstance(cfg["vlm"], dict):
            return dict(cfg["vlm"].get("s5_success", {}))
        if "s5_success" in cfg and isinstance(cfg["s5_success"], dict):
            return dict(cfg["s5_success"])
        return dict(cfg)
    return {}


def _plan_dense_tail_times(
    duration_s: float,
    tail_s: float,
    body_frames: int,
    tail_frames: int,
    *,
    tail_ratio: float = 0.22,
    max_tail_s: float = 25.0,
) -> list[float]:
    dur = max(0.05, float(duration_s))
    eff_tail_s = max(float(tail_s), min(float(max_tail_s), dur * max(0.0, float(tail_ratio))))
    if dur <= eff_tail_s + 0.5:
        n = min(max(4, body_frames + tail_frames), max(4, int(round(dur * 4.0))))
        return sorted({round(float(t), 3) for t in np.linspace(0.0, dur, n)})
    split_t = max(0.0, dur - eff_tail_s)
    body = list(np.linspace(0.0, max(0.0, split_t - 0.1), max(2, body_frames)))
    tail = list(np.linspace(split_t, dur, max(2, tail_frames)))
    return sorted({round(float(t), 3) for t in (body + tail)})


def _plan_control_times(
    duration_s: float, control_ratio: float, control_frames: int
) -> tuple[float, float, list[float]]:
    dur = max(0.05, float(duration_s))
    ratio = max(0.05, min(0.9, float(control_ratio)))
    cut_s = min(dur, max(min(dur, 1.0), dur * ratio))
    if dur < 1.0:
        cut_s = max(0.05, dur * ratio)
    eff_ratio = cut_s / dur if dur > 0 else ratio
    n = max(2, min(int(control_frames), max(2, int(round(cut_s * 6.0)) + 2)))
    times = sorted({round(float(t), 3) for t in np.linspace(0.0, cut_s, n)})
    return round(cut_s, 3), round(eff_ratio, 4), times


def _parse_criteria(
    raw_criteria: Any, task_criteria: list[str]
) -> list[CriterionResult]:
    out: list[CriterionResult] = []
    if isinstance(raw_criteria, list):
        for idx, item in enumerate(raw_criteria):
            if not isinstance(item, dict):
                continue
            crit = str(item.get("criterion") or "").strip()
            if not crit and idx < len(task_criteria):
                crit = task_criteria[idx]
            if not crit:
                continue
            met_raw = item.get("met")
            met: bool | None = None if met_raw is None else bool(met_raw)
            sat_raw = item.get("satisfied_at_s")
            try:
                sat_s = float(sat_raw) if sat_raw is not None else None
            except (TypeError, ValueError):
                sat_s = None
            out.append(
                CriterionResult(
                    criterion=crit,
                    met=met,
                    satisfied_at_s=sat_s,
                    evidence=str(item.get("evidence") or ""),
                )
            )
    if not out and task_criteria:
        out = [CriterionResult(criterion=c, met=None, evidence="") for c in task_criteria]
    return out


def _derive_verdict(
    raw_verdict: str, criteria: list[CriterionResult]
) -> str:
    v = raw_verdict.strip().lower()
    if v == "failure":
        return "failure"
    if criteria:
        mets = [c.met for c in criteria]
        if any(m is False for m in mets):
            return "failure"
        if all(m is True for m in mets):
            return "success"
    if v in {"success", "failure", "unsure"}:
        return v
    return "unsure"


def run_stage5_success(
    view: EpisodeView | None = None,
    *,
    episode_view: EpisodeView | None = None,
    task: TaskSpec | None = None,
    task_spec: TaskSpec | None = None,
    client: VlmClient | None = None,
    vlm_client: VlmClient | None = None,
    cfg: ConfigManager | dict[str, Any] | None = None,
    config: ConfigManager | dict[str, Any] | None = None,
    prompt_builder: PromptBuilder | None = None,
    budget: BudgetTracker | None = None,
    run_control_check: bool | None = None,
    model_override: str | None = None,
    thinking_budget_override: int | None = None,
) -> SuccessVerdict:
    """Execute Stage 5 dense-tail success evaluation + sequential on-demand 30% anti-FP control check."""
    ev = view or episode_view
    cl = client or vlm_client
    cm = cfg if isinstance(cfg, ConfigManager) else (config if isinstance(config, ConfigManager) else None)
    t_spec = task or task_spec or (cm.get_task(ev.meta.task_id) if cm and ev else TaskSpec(task_id="_default"))
    assert ev is not None and cl is not None

    s5_cfg = _resolve_s5_cfg(cfg or config)
    if not bool(s5_cfg.get("enabled", True)):
        return SuccessVerdict(passed=None, verdict="unsure", confidence=0.0, summary="Stage 5 disabled")

    eff_run_control = True if run_control_check is None else bool(run_control_check)
    tail_s = float(s5_cfg.get("tail_s", 12.0))
    tail_ratio = float(s5_cfg.get("tail_ratio", 0.22))
    max_tail_s = float(s5_cfg.get("max_tail_s", 25.0))
    body_frames = int(s5_cfg.get("body_frames", 6))
    tail_frames = int(s5_cfg.get("tail_frames", 10))
    control_ratio = float(s5_cfg.get("control_ratio", 0.30))
    control_frames = int(s5_cfg.get("control_frames", 8))
    cell_w = int(s5_cfg.get("cell_w", 420))
    cell_h = int(s5_cfg.get("cell_h", 236))
    quality = int(s5_cfg.get("quality", 78))
    image_detail = s5_cfg.get("image_detail", "auto")
    start_end_compare = bool(s5_cfg.get("start_end_compare", True))
    compare_panel_w = int(s5_cfg.get("compare_panel_w", 640))
    compare_panel_h = int(s5_cfg.get("compare_panel_h", 360))
    thinking = bool(s5_cfg.get("thinking", False))
    thinking_budget = (
        int(thinking_budget_override)
        if thinking_budget_override is not None
        else int(s5_cfg.get("thinking_budget", 1024))
    )
    max_tokens = int(s5_cfg.get("max_tokens", 1536))

    roles = ev.camera_roles
    full_times = _plan_dense_tail_times(
        ev.duration_s,
        tail_s,
        body_frames,
        tail_frames,
        tail_ratio=tail_ratio,
        max_tail_s=max_tail_s,
    )

    check_client_cancelled(cl)

    def _build_tile(t: float) -> np.ndarray:
        check_client_cancelled(cl)
        return compose_2x2_tile(
            ev.media, roles, t, cell_w=cell_w, cell_h=cell_h, include_header=True
        )

    with ThreadPoolExecutor(max_workers=min(6, max(1, len(full_times)))) as pool:
        tile_map = dict(zip(full_times, pool.map(_build_tile, full_times)))

    check_client_cancelled(cl)

    full_tiles = [tile_map[t] for t in full_times]
    full_send_times: list[float | None] = list(full_times)
    full_labels: list[str] | None = None

    if start_end_compare and full_times:
        end_t = full_times[-1]
        full_cmp = compose_start_end_compare(
            ev.media,
            roles,
            0.0,
            end_t,
            panel_w=compare_panel_w,
            panel_h=compare_panel_h,
        )
        full_tiles = [full_cmp, *full_tiles]
        full_send_times = [end_t, *full_send_times]
        full_labels = [
            f"START_vs_END (0.00s vs {end_t:.2f}s)",
            *["" for _ in full_times],
        ]

    pb = prompt_builder or (PromptBuilder(cm) if cm is not None else None)
    if pb is not None:
        sys_full, usr_full = pb.build_s5(
            task=t_spec, camera_roles=roles, duration_s=ev.duration_s
        )
    else:
        crit_str = "\n".join(f"- {c}" for c in t_spec.criteria) or "- Target object reaches the intended goal state"
        sys_full = (
            "You are an embodied manipulation task completion judge. "
            'Output strict JSON only: {"verdict": "success|failure|unsure", "confidence": float, "criteria": [{"criterion": str, "met": bool|null, "satisfied_at_s": float|null, "evidence": str}], "summary": str}.'
        )
        usr_full = f"Task: {t_spec.task_id} (duration {ev.duration_s:.2f}s)\nCompletion criteria:\n{crit_str}\nCamera layout: 2x2"

    # Sequential on-demand S5: always run primary s5_success FIRST
    check_client_cancelled(cl)
    full_chat_kwargs: dict[str, Any] = {
        "stage": "s5_success",
        "system_prompt": sys_full,
        "user_prompt": usr_full,
        "images": full_tiles,
        "timestamps_s": full_send_times,
        "labels": full_labels,
        "thinking": thinking,
        "thinking_budget": thinking_budget,
        "max_tokens": max_tokens,
        "jpeg_quality": quality,
        "image_detail": image_detail,
        "budget": budget,
    }
    if model_override is not None:
        full_chat_kwargs["model_override"] = model_override
    resp_full = cl.chat_json(**full_chat_kwargs)

    criteria = _parse_criteria(resp_full.get("criteria"), t_spec.criteria)
    verdict = _derive_verdict(str(resp_full.get("verdict") or "unsure"), criteria)
    try:
        conf = float(resp_full.get("confidence", 0.8 if verdict != "unsure" else 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    summary = str(resp_full.get("summary") or "")

    # ONLY run s5_control on demand when primary verdict == "success" AND eff_run_control is True
    control_result: ControlCheckResult | None = None
    if eff_run_control and verdict == "success":
        check_client_cancelled(cl)
        cut_s, eff_ratio, ctrl_times = _plan_control_times(
            ev.duration_s, control_ratio, control_frames
        )
        missing_ctrl_times = [t for t in ctrl_times if t not in tile_map]
        if missing_ctrl_times:
            with ThreadPoolExecutor(
                max_workers=min(6, max(1, len(missing_ctrl_times)))
            ) as pool:
                for t_val, tile_img in zip(
                    missing_ctrl_times, pool.map(_build_tile, missing_ctrl_times)
                ):
                    tile_map[t_val] = tile_img
        check_client_cancelled(cl)

        ctrl_tiles = [tile_map[t] for t in ctrl_times]
        ctrl_send_times: list[float | None] = list(ctrl_times)
        ctrl_labels: list[str] | None = None
        if start_end_compare and ctrl_times:
            ctrl_end_t = ctrl_times[-1]
            ctrl_cmp = compose_start_end_compare(
                ev.media,
                roles,
                0.0,
                ctrl_end_t,
                panel_w=compare_panel_w,
                panel_h=compare_panel_h,
                end_label=f"CUTOFF t={ctrl_end_t:05.2f}s (MAIN-ZOOM)",
            )
            ctrl_tiles = [ctrl_cmp, *ctrl_tiles]
            ctrl_send_times = [ctrl_end_t, *ctrl_send_times]
            ctrl_labels = [
                f"START_vs_CUTOFF (0.00s vs {ctrl_end_t:.2f}s)",
                *["" for _ in ctrl_times],
            ]

        if pb is not None:
            sys_ctrl, usr_ctrl = pb.build_s5_control(task=t_spec, cut_s=cut_s)
        else:
            crit_str = "\n".join(f"- {c}" for c in t_spec.criteria) or "- Target object reaches the intended goal state"
            sys_ctrl = "You are an anti-false-positive control verifier. This clip is truncated early (~30% of the episode)."
            usr_ctrl = f"Task: {t_spec.task_id} (early control clip truncated at {cut_s:.2f}s)\nCompletion criteria:\n{crit_str}"

        ctrl_chat_kwargs: dict[str, Any] = {
            "stage": "s5_control",
            "system_prompt": sys_ctrl,
            "user_prompt": usr_ctrl,
            "images": ctrl_tiles,
            "timestamps_s": ctrl_send_times,
            "labels": ctrl_labels,
            "thinking": thinking,
            "thinking_budget": thinking_budget,
            "max_tokens": max_tokens,
            "jpeg_quality": quality,
            "image_detail": image_detail,
            "budget": budget,
        }
        if model_override is not None:
            ctrl_chat_kwargs["model_override"] = model_override
        resp_ctrl = cl.chat_json(**ctrl_chat_kwargs)

        ctrl_criteria = _parse_criteria(resp_ctrl.get("criteria"), t_spec.criteria)
        ctrl_verdict = _derive_verdict(str(resp_ctrl.get("verdict") or "unsure"), ctrl_criteria)
        ctrl_all_met = bool(ctrl_criteria) and all(c.met is True for c in ctrl_criteria)
        ctrl_false_positive = (ctrl_verdict == "success") or ctrl_all_met

        ctrl_passed = not (verdict == "success" and ctrl_false_positive)
        if not ctrl_passed:
            verdict = "unsure"
            passed: bool | None = None
            conf = 0.0
            summary = (
                f"{summary} [Anti-FP control check failed at {eff_ratio:.0%}]".strip()
            )
        else:
            passed = True

        control_result = ControlCheckResult(
            cut_s=cut_s,
            cut_ratio=eff_ratio,
            control_verdict=ctrl_verdict,  # type: ignore[arg-type]
            passed=ctrl_passed,
            met=[c.met for c in ctrl_criteria],
        )
    else:
        if verdict == "success":
            passed = True
        elif verdict == "failure":
            passed = False
        else:
            passed = None

    return SuccessVerdict(
        passed=passed,
        verdict=verdict,  # type: ignore[arg-type]
        confidence=conf,
        criteria=criteria,
        control_check=control_result,
        summary=summary,
    )


run_s5 = run_stage5_success
