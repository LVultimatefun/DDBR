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
from ...models.annotation import GuardrailMetrics, MaskSide, QcSegment, SplicePolicy
from ...store import NotFoundError
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
