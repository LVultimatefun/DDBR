"""Episodes listing, ResolvedEpisode detail, direct JPEG frame & MP4 preview endpoints."""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ...guardrail import apply_hallucination_check, apply_splice_gate
from ...guardrail.visual_gate import evaluate_visual_segment
from ...io.media import encode_jpeg_bytes
from ...models.annotation import (
    GroundingResult,
    GuardrailMetrics,
    MacroStage,
    MaskSide,
    QcSegment,
    SplicePolicy,
)
from ...models.review import EpisodeStatus
from ...store import NotFoundError
from ...vlm.client import VlmClient, VlmError
from ...vlm.prompt_builder import PromptBuilder
from ...vlm.stage2_macro import infer_subtasks_from_segments
from ..auth import AuthContext, get_app_state, require_auth, require_lead
from ..state import AppState

router = APIRouter(prefix="/api/episodes", tags=["episodes"])


class CheckSpliceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t_start: float = Field(ge=0.0)
    t_end: float = Field(gt=0.0)
    label_id: str = Field(min_length=1)
    splice_policy: SplicePolicy
    affected_arm: MaskSide = "both"

    @field_validator("t_start", "t_end", mode="before")
    @classmethod
    def _reject_bool_time(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("Time boundary must be a finite number, not bool")
        return v

    @model_validator(mode="after")
    def _validate_span(self) -> "CheckSpliceRequest":
        if not math.isfinite(self.t_start) or not math.isfinite(self.t_end):
            raise ValueError("t_start and t_end must be finite numbers")
        if self.t_end <= self.t_start:
            raise ValueError(f"t_end ({self.t_end}) must be > t_start ({self.t_start})")
        if not self.label_id.strip():
            raise ValueError("label_id must be non-empty")
        return self


def _format_episode_item(state: AppState, row: dict[str, Any]) -> dict[str, Any]:
    cam_roles_raw = row.get("camera_roles_json") or "{}"
    try:
        cam_roles = json.loads(cam_roles_raw) if isinstance(cam_roles_raw, str) else dict(cam_roles_raw)
    except Exception:
        cam_roles = {}
    rp = row.get("resolved_passed")
    return {
        "episode_id": row["episode_id"],
        "dataset": row["dataset"],
        "task_id": row["task_id"],
        "adapter_type": row["adapter_type"],
        "source_uri": row["source_uri"],
        "source_mtime": float(row["source_mtime"]),
        "duration_s": float(row["duration_s"]),
        "fps": float(row["fps"]),
        "has_kinematics": bool(row["has_kinematics"]),
        "camera_roles": cam_roles,
        "status": row["status"],
        "active_run_id": row["active_run_id"],
        "active_rev_id": row["active_rev_id"],
        "claimed_by": row["claimed_by"],
        "claim_expires_at": row["claim_expires_at"],
        "resolved_passed": None if rp is None else bool(rp),
        "cut_ratio": row["cut_ratio"],
        "mask_ratio": row["mask_ratio"],
        "badcase_count": int(row["badcase_count"] or 0),
        "unknown_count": int(row["unknown_count"] or 0),
        "guardrail_downgraded": state.repo.episode_guardrail_downgraded(row),
        "updated_at": float(row["updated_at"]),
    }


@router.get("")
def list_episodes(
    status: str | None = Query(default=None),
    dataset: str | None = Query(default=None),
    task_id: str | None = Query(default=None),
    adapter_type: str | None = Query(default=None),
    passed: bool | None = Query(default=None),
    guardrail_downgraded: bool | None = Query(default=None),
    has_unknown: bool | None = Query(default=None),
    claimed_by: str | None = Query(default=None),
    search: str | None = Query(default=None),
    sort_by: Literal[
        "updated_at",
        "duration_s",
        "cut_ratio",
        "mask_ratio",
        "badcase_count",
        "episode_id",
    ] = Query(default="episode_id"),
    sort_dir: Literal["asc", "desc"] = Query(default="asc"),
    limit: int = Query(default=100, ge=1, le=10000),
    offset: int = Query(default=0, ge=0),
    auth: AuthContext = Depends(require_auth),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    state.repo.reap_expired_claims()
    try:
        rows = state.repo.list_episodes(
            status=status,
            dataset=dataset,
            task_id=task_id,
            adapter_type=adapter_type,
            passed=passed,
            has_unknown=has_unknown,
            guardrail_downgraded=guardrail_downgraded,
            claimed_by=claimed_by,
            search=search,
            sort_by=sort_by,
            sort_dir=sort_dir,
            limit=limit,
            offset=offset,
        )
        total = state.repo.count_episodes(
            status=status,
            dataset=dataset,
            task_id=task_id,
            adapter_type=adapter_type,
            passed=passed,
            has_unknown=has_unknown,
            guardrail_downgraded=guardrail_downgraded,
            claimed_by=claimed_by,
            search=search,
        )
        status_counts = state.repo.count_episodes_by_status(
            dataset=dataset,
            task_id=task_id,
            adapter_type=adapter_type,
            passed=passed,
            has_unknown=has_unknown,
            search=search,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    items = [_format_episode_item(state, r) for r in rows]
    my_claimed_count = state.repo.count_episodes(status="claimed", claimed_by=auth.user)
    facets = state.repo.list_episode_facets()
    return {
        "items": items,
        "total": total,
        "my_claimed_count": my_claimed_count,
        "status_counts": status_counts,
        "facets": facets,
    }


class AssignTaskPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    allow_frozen: bool = False


class BatchAssignTaskPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_ids: list[str]
    task_id: str = Field(min_length=1)
    allow_frozen: bool = False


def _is_vlm_active(state: AppState, episode_id: str, row_status: str) -> bool:
    if row_status in (
        EpisodeStatus.VLM_QUEUED.value,
        EpisodeStatus.VLM_RUNNING.value,
    ):
        return True
    lock = getattr(state, "_jobs_lock", None)
    active_map = getattr(state, "_active_vlm_episodes", None)
    if lock is not None and isinstance(active_map, dict):
        with lock:
            return episode_id in active_map
    return False


@router.get("/tasks")
def list_episode_tasks(
    auth: AuthContext = Depends(require_auth),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    cfg = state.get_config()
    ordered_ids = sorted(tid for tid in cfg.tasks if tid != "_default")
    if "_default" in cfg.tasks:
        ordered_ids.append("_default")
    tasks = [
        {
            "task_id": spec.task_id,
            "title": spec.title,
            "description": spec.description,
            "goal": spec.goal,
            "match": list(spec.match),
            "aliases": list(spec.aliases),
            "grounding_objects_count": len(spec.grounding_objects),
        }
        for tid in ordered_ids
        for spec in (cfg.tasks[tid],)
    ]
    return {"tasks": tasks, "total": len(tasks)}


@router.post("/batch-task")
def batch_assign_episode_task(
    payload: BatchAssignTaskPayload,
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    new_tid = payload.task_id.strip()
    cfg = state.get_config()
    if new_tid not in cfg.tasks:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown task_id: {new_tid!r}",
        )

    unique_ids = list(
        dict.fromkeys(str(eid).strip() for eid in payload.episode_ids if str(eid).strip())
    )
    if not unique_ids:
        raise HTTPException(
            status_code=422,
            detail="episode_ids must contain at least one episode_id",
        )

    rows_by_id: dict[str, dict[str, Any]] = {}
    for eid in unique_ids:
        try:
            rows_by_id[eid] = state.repo.get_episode_row(eid)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    active_vlm_ids = [
        eid
        for eid in unique_ids
        if _is_vlm_active(state, eid, str(rows_by_id[eid]["status"]))
    ]
    if active_vlm_ids:
        raise HTTPException(
            status_code=409,
            detail=(
                "Cannot reassign task_id while VLM is queued or running for episode(s): "
                f"{', '.join(active_vlm_ids)}"
            ),
        )

    if not payload.allow_frozen:
        frozen_ids = [
            eid
            for eid in unique_ids
            if str(rows_by_id[eid]["status"]) == EpisodeStatus.GT_FROZEN.value
        ]
        if frozen_ids:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Episode(s) are gt_frozen ({', '.join(frozen_ids)}); "
                    "pass allow_frozen=true to confirm task_id reassignment"
                ),
            )

    try:
        updated_rows = state.repo.update_episodes_task_id(unique_ids, new_tid)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    updated_by_id = {str(r["episode_id"]): r for r in updated_rows}
    has_active_runs = 0
    re_run_recommended_ids: list[str] = []
    for eid in unique_ids:
        old_tid = str(rows_by_id[eid]["task_id"])
        after_row = updated_by_id[eid]
        has_run = bool(after_row.get("active_run_id"))
        if has_run:
            has_active_runs += 1
        if old_tid != new_tid:
            state.sse.publish(
                {
                    "type": "episode_updated",
                    "episode_id": eid,
                    "task_id": new_tid,
                    "previous_task_id": old_tid,
                    "status": str(after_row["status"]),
                }
            )
            if has_run:
                re_run_recommended_ids.append(eid)

    return {
        "task_id": new_tid,
        "updated": len(updated_rows),
        "episode_ids": unique_ids,
        "has_active_runs": has_active_runs,
        "re_run_recommended_ids": re_run_recommended_ids,
    }


@router.patch("/{ep_id:path}/task")
def assign_episode_task(
    ep_id: str,
    payload: AssignTaskPayload,
    auth: AuthContext = Depends(require_auth),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    new_tid = payload.task_id.strip()
    cfg = state.get_config()
    if new_tid not in cfg.tasks:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown task_id: {new_tid!r}",
        )

    try:
        row = state.repo.get_episode_row(ep_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    ep_status = str(row["status"])
    old_tid = str(row["task_id"])

    if _is_vlm_active(state, ep_id, ep_status):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Episode {ep_id!r} has an active VLM job ({ep_status}); "
                "wait or cancel before reassigning task_id"
            ),
        )

    if auth.role == "reviewer":
        if (
            old_tid != "_default"
            or ep_status == EpisodeStatus.GT_FROZEN.value
            or new_tid == "_default"
        ):
            raise HTTPException(
                status_code=403,
                detail=(
                    "Reviewers may only reassign non-frozen unrecognized (_default) episodes "
                    "to an existing non-_default task"
                ),
            )
    else:
        if ep_status == EpisodeStatus.GT_FROZEN.value and not payload.allow_frozen:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Episode {ep_id!r} is gt_frozen; pass allow_frozen=true "
                    "to confirm task_id reassignment"
                ),
            )

    try:
        updated_row = state.repo.update_episode_task_id(ep_id, new_tid)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if old_tid != new_tid:
        state.sse.publish(
            {
                "type": "episode_updated",
                "episode_id": ep_id,
                "task_id": new_tid,
                "previous_task_id": old_tid,
                "status": str(updated_row["status"]),
            }
        )

    has_active_run = bool(updated_row.get("active_run_id"))
    re_run_recommended = (old_tid != new_tid) and has_active_run
    return {
        "episode_id": ep_id,
        "previous_task_id": old_tid,
        "task_id": new_tid,
        "status": str(updated_row["status"]),
        "has_active_run": has_active_run,
        "re_run_recommended": re_run_recommended,
    }


def _build_check_splice_metrics(
    gm: GuardrailMetrics | None = None,
    *,
    include_kinematics: bool = False,
    visual_ssim: float | None = None,
    visual_l1: float | None = None,
) -> dict[str, Any]:
    if gm is not None and include_kinematics:
        return {
            "eef_pos_m": float(gm.eef_pos_m),
            "eef_rot_rad": float(gm.eef_rot_rad),
            "hand_rad": float(gm.hand_rad),
            "torso_rad": float(gm.torso_rad),
            "other_arm_disp_m": float(gm.other_arm_disp_m),
            "bimanual_coupled": bool(gm.bimanual_coupled),
            "visual_ssim": visual_ssim,
            "visual_l1": visual_l1,
        }
    return {
        "eef_pos_m": None,
        "eef_rot_rad": None,
        "hand_rad": None,
        "torso_rad": None,
        "other_arm_disp_m": None,
        "bimanual_coupled": False,
        "visual_ssim": visual_ssim,
        "visual_l1": visual_l1,
    }


def _check_splice_sync(
    state: AppState, ep_id: str, payload: CheckSpliceRequest
) -> dict[str, Any]:
    t0 = time.perf_counter()
    cfg = state.get_config()
    meta = state.repo.get_episode_meta(ep_id)

    label_id = payload.label_id.strip()
    lbl = cfg.labels.get(label_id)
    if payload.splice_policy == "keep_positive":
        role = (
            lbl.default_training_role
            if lbl is not None and lbl.default_training_role in {"necessary", "recovery"}
            else "necessary"
        )
        mask_side: MaskSide | None = None
    elif payload.splice_policy == "mask_loss":
        role = (
            lbl.default_training_role
            if lbl is not None and lbl.default_training_role in {"badcase", "terminal_fail", "recovery"}
            else "badcase"
        )
        mask_side = payload.affected_arm
    else:
        role = (
            lbl.default_training_role
            if lbl is not None and lbl.default_training_role in {"badcase", "terminal_fail"}
            else "badcase"
        )
        mask_side = None

    seg = QcSegment(
        seg_id="check_splice",
        t_start=float(payload.t_start),
        t_end=float(payload.t_end),
        label_id=label_id,
        name=lbl.name if lbl is not None else "",
        training_role=role,  # type: ignore[arg-type]
        splice_policy=payload.splice_policy,
        mask_side=mask_side,
        affected_arm=payload.affected_arm,
        vlm_raw_policy=payload.splice_policy,
    )

    if payload.splice_policy == "keep_positive":
        return {
            "passed": True,
            "downgraded": False,
            "downgrade_reason": None,
            "effective_splice_policy": "keep_positive",
            "opt_t_start": None,
            "opt_t_end": None,
            "metrics": _build_check_splice_metrics(),
            "compute_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            "mode": "skipped",
        }

    if payload.splice_policy == "mask_loss":
        if meta.kinematics.has_kinematics and label_id in {"mid_idle_freeze", "boundary_idle"}:
            kin, dur, _fps = state.get_or_load_kinematics(meta, cfg)
            if kin is not None:
                hallu = apply_hallucination_check(
                    [seg], kinematics=kin, cfg=cfg, duration_s=dur
                )[0]
                if hallu.guardrail.guardrail_override:
                    return {
                        "passed": True,
                        "downgraded": False,
                        "downgrade_reason": hallu.guardrail.override_reason,
                        "effective_splice_policy": hallu.splice_policy,
                        "opt_t_start": None,
                        "opt_t_end": None,
                        "metrics": _build_check_splice_metrics(),
                        "compute_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                        "mode": "kinematics",
                    }
        return {
            "passed": True,
            "downgraded": False,
            "downgrade_reason": None,
            "effective_splice_policy": "mask_loss",
            "opt_t_start": None,
            "opt_t_end": None,
            "metrics": _build_check_splice_metrics(),
            "compute_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            "mode": "skipped",
        }

    # cut_clean
    if meta.kinematics.has_kinematics:
        kin, dur, _fps = state.get_or_load_kinematics(meta, cfg)
        if kin is not None:
            hallu = apply_hallucination_check(
                [seg], kinematics=kin, cfg=cfg, duration_s=dur
            )[0]
            if hallu.guardrail.guardrail_override:
                return {
                    "passed": True,
                    "downgraded": False,
                    "downgrade_reason": hallu.guardrail.override_reason,
                    "effective_splice_policy": hallu.splice_policy,
                    "opt_t_start": None,
                    "opt_t_end": None,
                    "metrics": _build_check_splice_metrics(),
                    "compute_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                    "mode": "kinematics",
                }
            sg = apply_splice_gate(
                [hallu], kinematics=kin, cfg=cfg, duration_s=dur
            )[0]
            if sg.guardrail.gate_type == "splice_gate":
                return {
                    "passed": bool(sg.guardrail.passed),
                    "downgraded": bool(sg.guardrail.downgraded),
                    "downgrade_reason": sg.guardrail.downgrade_reason,
                    "effective_splice_policy": sg.splice_policy,
                    "opt_t_start": sg.guardrail.opt_t_start,
                    "opt_t_end": sg.guardrail.opt_t_end,
                    "metrics": _build_check_splice_metrics(
                        sg.guardrail.metrics, include_kinematics=True
                    ),
                    "compute_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                    "mode": "kinematics",
                }

    # Pure-video visual gate fallback
    with state.open_view(meta, cfg) as view:
        if view.media.cameras:
            vis_seg, l1_diff = evaluate_visual_segment(
                seg,
                media=view.media,
                cfg=cfg,
                duration_s=view.duration_s,
            )
            return {
                "passed": bool(vis_seg.guardrail.passed),
                "downgraded": bool(vis_seg.guardrail.downgraded),
                "downgrade_reason": vis_seg.guardrail.downgrade_reason,
                "effective_splice_policy": vis_seg.splice_policy,
                "opt_t_start": vis_seg.guardrail.opt_t_start,
                "opt_t_end": vis_seg.guardrail.opt_t_end,
                "metrics": _build_check_splice_metrics(
                    visual_ssim=vis_seg.guardrail.metrics.visual_ssim,
                    visual_l1=round(l1_diff, 4) if l1_diff is not None else None,
                ),
                "compute_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                "mode": "visual",
            }

    return {
        "passed": True,
        "downgraded": False,
        "downgrade_reason": None,
        "effective_splice_policy": seg.splice_policy,
        "opt_t_start": None,
        "opt_t_end": None,
        "metrics": _build_check_splice_metrics(),
        "compute_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        "mode": "skipped",
    }


class InferSubtasksRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    qc_segments: list[QcSegment]
    macro_stages: list[MacroStage] = Field(default_factory=list)
    custom_hint: str | None = None


def _infer_subtasks_sync(
    state: AppState, ep_id: str, payload: InferSubtasksRequest
) -> dict[str, Any]:
    cfg = state.get_config()
    row = state.repo.get_episode_row(ep_id)
    if row["status"] == EpisodeStatus.GT_FROZEN.value:
        raise HTTPException(
            status_code=409,
            detail=f"Episode {ep_id!r} is gt_frozen and cannot re-infer subtasks",
        )
    meta = state.repo.get_episode_meta(ep_id)

    grounding: GroundingResult | None = None
    active_run_id = row.get("active_run_id")
    if active_run_id:
        try:
            run_art = state.repo.get_vlm_run(str(active_run_id))
            if run_art.grounding and run_art.grounding.object_tracks_2d:
                grounding = run_art.grounding
        except NotFoundError:
            grounding = None

    if meta.task_id and meta.task_id != "_default":
        task_spec = cfg.get_task(meta.task_id)
    else:
        hint = meta.extra_meta.get("h5_task_name") or meta.extra_meta.get("ep_task_str")
        task_spec = cfg.resolve_task(meta.dataset, str(hint) if hint else None)

    pb = PromptBuilder(cfg)
    with state.open_view(meta, cfg) as view:
        with VlmClient.from_config(cfg, workspace_dir=state.workspace) as vlm_client:
            inferred = infer_subtasks_from_segments(
                view=view,
                qc_segments=payload.qc_segments,
                macro_stages=payload.macro_stages,
                grounding=grounding,
                custom_hint=payload.custom_hint,
                task=task_spec,
                client=vlm_client,
                cfg=cfg,
                prompt_builder=pb,
            )

    return {
        "episode_id": ep_id,
        "macro_stages": [st.model_dump(mode="json") for st in inferred],
        "used_grounding": grounding is not None,
        "used_segments_count": len(payload.qc_segments),
    }


@router.post("/{ep_id:path}/infer-subtasks")
async def infer_episode_subtasks(
    ep_id: str,
    payload: InferSubtasksRequest,
    auth: AuthContext = Depends(require_auth),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    try:
        return await run_in_threadpool(_infer_subtasks_sync, state, ep_id, payload)
    except HTTPException:
        raise
    except (NotFoundError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except VlmError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/{ep_id:path}/check-splice")
async def check_episode_splice(
    ep_id: str,
    payload: CheckSpliceRequest,
    auth: AuthContext = Depends(require_auth),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    try:
        return await run_in_threadpool(_check_splice_sync, state, ep_id, payload)
    except (NotFoundError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/{ep_id:path}/runs")
def list_episode_runs(
    ep_id: str,
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    try:
        row = state.repo.get_episode_row(ep_id)
        runs_raw = state.repo.list_vlm_runs(ep_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    runs = [
        {
            "run_id": r["run_id"],
            "created_at": datetime.fromtimestamp(
                float(r["created_at"]), tz=timezone.utc
            ).isoformat(),
            "prompt_version": r["prompt_version"],
            "label_version": r["label_version"],
            "few_shot_pack_id": r["few_shot_pack_id"],
            "tokens_total": int(r["tokens_total"] or 0),
            "is_eval_sandbox": bool(r["is_eval_sandbox"]),
        }
        for r in runs_raw
    ]
    return {
        "episode_id": ep_id,
        "active_run_id": row["active_run_id"],
        "runs": runs,
    }


def _resolve_camera_name(meta: Any, camera: str | None) -> str:
    if not camera:
        return str(meta.camera_roles.get("main") or meta.cameras[0].name)
    cam_names = {c.name for c in meta.cameras}
    if camera in cam_names:
        return camera
    if camera in meta.camera_roles and meta.camera_roles[camera] in cam_names:
        return str(meta.camera_roles[camera])
    raise HTTPException(
        status_code=400,
        detail=f"Unknown camera {camera!r}; available: {sorted(cam_names)}",
    )


def _read_jpeg_frame_sync(state: AppState, ep_id: str, camera: str | None, t_s: float) -> bytes:
    cfg = state.get_config()
    meta = state.repo.get_episode_meta(ep_id)
    cam = _resolve_camera_name(meta, camera)
    t_clamped = max(0.0, min(float(meta.duration_s), float(t_s)))
    with state.open_view(meta, cfg) as view:
        rgb = view.media.read_frame_rgb(cam, t_clamped)
    return encode_jpeg_bytes(rgb, quality=85)


@router.get("/{ep_id:path}/frame")
async def get_episode_frame(
    ep_id: str,
    camera: str | None = Query(default=None),
    t_s: float = Query(...),
    auth: AuthContext = Depends(require_auth),
    state: AppState = Depends(get_app_state),
) -> Response:
    _ = auth
    if not math.isfinite(t_s):
        raise HTTPException(status_code=400, detail="t_s must be finite")
    try:
        jpeg_bytes = await run_in_threadpool(_read_jpeg_frame_sync, state, ep_id, camera, t_s)
    except (NotFoundError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(content=jpeg_bytes, media_type="image/jpeg")


def _build_preview_sync(state: AppState, ep_id: str, camera: str | None, hd: bool) -> Any:
    cfg = state.get_config()
    meta = state.repo.get_episode_meta(ep_id)
    cam = _resolve_camera_name(meta, camera)
    cached = state.preview_cache.preview_path(meta.dataset, meta.stem, cam, hd=hd)
    if cached.is_file() and cached.stat().st_mtime >= meta.source_mtime:
        return cached
    with state.open_view(meta, cfg) as view:
        return state.preview_cache.get_or_build(
            view.media,
            dataset=meta.dataset,
            stem=meta.stem,
            camera=cam,
            source_mtime=meta.source_mtime,
            hd=hd,
        )


@router.get("/{ep_id:path}/preview")
async def get_episode_preview(
    ep_id: str,
    camera: str | None = Query(default=None),
    hd: bool = Query(default=False),
    auth: AuthContext = Depends(require_auth),
    state: AppState = Depends(get_app_state),
) -> FileResponse:
    _ = auth
    try:
        mp4_path = await run_in_threadpool(_build_preview_sync, state, ep_id, camera, hd)
    except (NotFoundError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path=mp4_path, media_type="video/mp4", filename=mp4_path.name)


@router.get("/{ep_id:path}")
def get_episode_detail(
    ep_id: str,
    run_id: str | None = Query(default=None),
    auth: AuthContext = Depends(require_auth),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    eff_run_id: str | None = None
    if run_id is not None:
        if auth.role != "lead":
            raise HTTPException(
                status_code=403,
                detail="Lead role required to view historical VLM runs",
            )
        eff_run_id = run_id.strip()
        if not eff_run_id:
            raise HTTPException(
                status_code=422,
                detail="run_id must be non-empty when provided",
            )

    state.repo.reap_expired_claims()
    try:
        resolved = state.repo.resolve_episode_for_api(ep_id, run_id=eff_run_id)
        row = state.repo.get_episode_row(ep_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    payload = resolved.model_dump(mode="json")
    payload["claimed_by"] = row["claimed_by"]
    payload["claim_expires_at"] = row["claim_expires_at"]
    if eff_run_id is not None:
        payload["guardrail_downgraded"] = any(
            s.guardrail.downgraded for s in resolved.qc_segments
        )
        payload["selected_run_id"] = eff_run_id
        payload["is_historical_run_view"] = True
    else:
        payload["guardrail_downgraded"] = state.repo.episode_guardrail_downgraded(row)
        payload["selected_run_id"] = row["active_run_id"]
        payload["is_historical_run_view"] = False
    return payload
