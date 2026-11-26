"""Unified async jobs (batch_vlm concurrent, render_mp4, export, eval_gt, ingest), artifact download, and SSE streaming."""

from __future__ import annotations

import asyncio
import contextlib
import json
import tempfile
import threading
import time
import uuid
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ...export.viz_mp4 import render_viz_mp4
from ...io import auto_detect, get_adapter
from ...io.base import validate_id_part
from ...lab.eval_runner import run_gt_eval
from ...models.review import EpisodeStatus
from ...store import InvalidTransitionError, NotFoundError
from ...vlm import VlmDag
from ...vlm.client import VlmCancelledError, VlmClient, normalize_stage_name
from ..auth import AuthContext, get_app_state, require_auth, require_lead
from ..state import AppState
from .exports import ExportTriggerPayload, _run_export_sync
from .ingest import is_within_allowed_roots, resolve_allowed_path
from .lab import EvalRunPayload

router = APIRouter(tags=["jobs"])

_VALID_STAGES = frozenset(
    {
        "all",
        "s1",
        "s1_grounding",
        "s1_s5",
        "s1-s5",
        "s1/s5",
        "s2",
        "s2_macro",
        "s3",
        "s3_discover",
        "s3_s4",
        "s3/s4",
        "s3-s4",
        "s4",
        "s4a",
        "s4a_merge",
        "s4_merge",
        "s4b",
        "s4b_verify",
        "s4_verify",
        "s5",
        "s5_success",
    }
)

_DOWNSTREAM_STAGE_MAP: dict[str, set[str]] = {
    "s1_grounding": {"s1_grounding"},
    "s2_macro": {"s2_macro", "s3_discover", "s4_verify"},
    "s3_discover": {"s3_discover", "s4_verify"},
    "s4_verify": {"s4_verify"},
    "s5_success": {"s5_success"},
}


class _JobCancelled(Exception):
    """Internal signal to abort an active job worker cleanly without overwriting cancelled status."""


class IngestParamsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_path: str | None = None
    data_paths: list[str] | None = None
    path: str | None = None
    adapter: Literal["auto", "h5_raw_v3", "lerobot", "video_dir"] = "auto"
    dataset: str | None = None
    task_id: str | None = None


def _normalize_ingest_task_override(raw_task_id: str | None) -> str | None:
    if raw_task_id is None:
        return None
    cleaned = str(raw_task_id).strip()
    if not cleaned or cleaned.lower() == "auto":
        return None
    return cleaned


class JobEstimatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_ids: list[str]
    stage: str | None = None
    fresh: bool = False


class JobCreatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_type: Literal["batch_vlm", "render_mp4", "export", "eval_gt", "ingest"]
    episode_ids: list[str] = Field(default_factory=list)
    stage: str | None = None
    fresh: bool = False
    concurrency: int | None = Field(default=None, ge=1, le=32)
    is_eval_sandbox: bool | None = None
    custom_objects: list[str] | None = None
    export_params: dict[str, Any] | None = None
    eval_params: dict[str, Any] | None = None
    ingest_params: dict[str, Any] | None = None

    @field_validator("concurrency", mode="before")
    @classmethod
    def _reject_bool_concurrency(cls, v: Any) -> Any:
        if isinstance(v, bool) or isinstance(v, float):
            raise ValueError("concurrency must be an integer between 1 and 32")
        return v

    @field_validator("custom_objects")
    @classmethod
    def _normalize_custom_objects(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in v:
            s = item.strip()
            if not s:
                continue
            low = s.lower()
            if low not in seen:
                seen.add(low)
                cleaned.append(s)
        return cleaned or None


def _resolve_stage_options(
    stage: str | None, fresh: bool
) -> tuple[bool, set[str] | None]:
    if stage is None:
        return bool(fresh), None
    s_norm = stage.strip().lower()
    if not s_norm or s_norm == "all":
        return bool(fresh), None
    if s_norm in {"s1_s5", "s1-s5", "s1/s5"}:
        return True, None
    if s_norm in {"s3_s4", "s3/s4", "s3-s4"}:
        return False, {"s3_discover", "s4_verify"}
    if s_norm not in _VALID_STAGES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid stage {stage!r}; expected 'all', 's1_s5', 's3_s4', or s1..s5",
        )
    canon = normalize_stage_name(s_norm)
    return False, _DOWNSTREAM_STAGE_MAP.get(canon, {canon})


def _collect_ingest_paths(ingest_payload: IngestParamsPayload) -> list[str]:
    raw: list[str] = []
    if ingest_payload.data_paths:
        raw.extend(ingest_payload.data_paths)
    if ingest_payload.data_path:
        raw.append(ingest_payload.data_path)
    if ingest_payload.path:
        raw.append(ingest_payload.path)
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in raw:
        s = str(item).strip()
        if s and s not in seen:
            seen.add(s)
            cleaned.append(s)
    return cleaned


def _resolve_batch_concurrency(payload: JobCreatePayload, cfg: Any, n_items: int) -> int:
    if payload.concurrency is not None:
        req = int(payload.concurrency)
    else:
        vlm_cfg = cfg.settings.get("vlm", {})
        req = int(
            vlm_cfg.get("episode_concurrency")
            or vlm_cfg.get("batch_vlm_concurrency")
            or 4
        )
    return max(1, min(32, req, max(1, n_items)))


def _publish_episode_status(
    state: AppState, episode_id: str, new_status: EpisodeStatus
) -> None:
    state.sse.publish(
        {
            "type": "episode_updated",
            "episode_id": episode_id,
            "status": new_status.value,
        }
    )


def _run_job_worker(
    state: AppState,
    job_id: str,
    payload: JobCreatePayload,
    eff_fresh: bool,
    fresh_stages: set[str] | None,
) -> None:
    cancel_event = state.get_job_cancel_event(job_id)
    sandbox_flag = (
        bool(payload.is_eval_sandbox)
        if payload.is_eval_sandbox is not None
        else False
    )
    restore_needed_on_fail = False
    try:
        cur = state.repo.get_job(job_id)
        if cancel_event.is_set() or cur["status"] == "cancelled":
            cancel_event.set()
            return
        cur = state.repo.update_job(job_id, status="running", progress_done=0)
        if cancel_event.is_set() or cur["status"] == "cancelled":
            cancel_event.set()
            return
        state.sse.publish({"type": "job_update", "job": cur})

        cfg = state.get_config()
        results: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        artifacts: list[dict[str, Any]] = []
        summary_extra: dict[str, Any] = {}
        done = 0

        if payload.job_type == "batch_vlm":
            ep_ids = list(payload.episode_ids)
            eff_workers = _resolve_batch_concurrency(payload, cfg, len(ep_ids))
            prev_statuses = state.get_job_prev_statuses(job_id)
            vlm_client = VlmClient.from_config(
                cfg,
                workspace_dir=state.workspace,
                cancel_event=cancel_event,
            )
            state.add_job_cancel_callback(job_id, vlm_client.cancel)
            dag = VlmDag(
                workspace=state.workspace,
                config=cfg,
                client=vlm_client,
                repo=state.repo,
                cancel_event=cancel_event,
            )
            progress_lock = threading.Lock()
            results_map: dict[int, dict[str, Any]] = {}
            errors_map: dict[int, dict[str, str]] = {}
            running_tokens = 0
            running_cached = 0
            running_cost = 0.0

            def _on_ep_status(ep_id_cb: str, st_cb: EpisodeStatus) -> None:
                _publish_episode_status(state, ep_id_cb, st_cb)

            def _batch_one(idx: int, ep_id: str) -> None:
                nonlocal done, running_tokens, running_cached, running_cost
                try:
                    if cancel_event.is_set() or state.repo.get_job(job_id)["status"] == "cancelled":
                        cancel_event.set()
                        return
                    try:
                        meta = state.repo.get_episode_meta(ep_id)
                        if cancel_event.is_set():
                            return
                        with state.open_view(meta, cfg) as view:
                            art = dag.run_episode(
                                view,
                                fresh=eff_fresh,
                                fresh_stages=fresh_stages,
                                custom_objects=payload.custom_objects,
                                is_eval_sandbox=sandbox_flag,
                                cancel_event=cancel_event,
                                on_status_change=_on_ep_status,
                                can_rollback=lambda eid: state.can_job_restore_episode(
                                    job_id, eid
                                ),
                            )
                        art_tokens = int(
                            art.usage.total_tokens
                            or sum(s.total_tokens for s in art.usage.stages)
                        )
                        art_cached = int(
                            getattr(art.usage, "total_cached_tokens", 0)
                            or sum(
                                int(getattr(s, "cached_tokens", 0) or 0)
                                for s in art.usage.stages
                            )
                        )
                        art_cost = float(
                            getattr(art.usage, "total_cost_usd", 0.0)
                            or sum(
                                float(getattr(s, "estimated_cost_usd", 0.0) or 0.0)
                                for s in art.usage.stages
                            )
                        )
                        art_elapsed = float(
                            art.usage.total_elapsed_s
                            or sum(s.elapsed_s for s in art.usage.stages)
                        )
                        item_res: dict[str, Any] | None = {
                            "episode_id": ep_id,
                            "run_id": art.run_id,
                            "verdict": art.success.verdict,
                            "tokens_total": art_tokens,
                            "cached_tokens": art_cached,
                            "estimated_cost_usd": round(art_cost, 6),
                            "elapsed_s": round(art_elapsed, 4),
                        }
                        item_err: dict[str, str] | None = None
                    except VlmCancelledError:
                        cancel_event.set()
                        return
                    except Exception as exc:
                        if cancel_event.is_set() or state.repo.get_job(job_id)["status"] == "cancelled":
                            cancel_event.set()
                            return
                        if not sandbox_flag and state.can_job_restore_episode(job_id, ep_id):
                            with contextlib.suppress(Exception):
                                restored_one = state.repo.restore_vlm_episodes_on_cancel(
                                    [ep_id], prev_statuses=prev_statuses or None
                                )
                                for r_ep, r_st in restored_one:
                                    _on_ep_status(r_ep, r_st)
                        item_res = None
                        item_err = {"episode_id": ep_id, "error": str(exc)}

                    with progress_lock:
                        if item_res is not None:
                            results_map[idx] = item_res
                            running_tokens += int(item_res["tokens_total"])
                            running_cached += int(item_res["cached_tokens"])
                            running_cost += float(item_res["estimated_cost_usd"])
                        if item_err is not None:
                            errors_map[idx] = item_err
                        done += 1
                        if cancel_event.is_set() or state.repo.get_job(job_id)["status"] == "cancelled":
                            cancel_event.set()
                            return
                        live_summary = {
                            "tokens_total": running_tokens,
                            "cached_tokens": running_cached,
                            "estimated_cost_usd": round(running_cost, 6),
                            "results": [results_map[i] for i in sorted(results_map)],
                            "errors": [errors_map[i] for i in sorted(errors_map)],
                        }
                        upd = state.repo.update_job(
                            job_id, progress_done=done, summary=live_summary
                        )
                        if upd["status"] != "cancelled":
                            state.sse.publish({"type": "job_update", "job": upd})
                finally:
                    state.repo.db.close_thread_connection()

            try:
                with ThreadPoolExecutor(
                    max_workers=eff_workers,
                    thread_name_prefix=f"vlm-batch-{job_id[:8]}",
                ) as pool:
                    futures = [
                        pool.submit(_batch_one, idx, ep_id)
                        for idx, ep_id in enumerate(ep_ids)
                    ]
                    for fut in futures:
                        fut.result()
            finally:
                with contextlib.suppress(Exception):
                    dag.client.close()

            if cancel_event.is_set() or state.repo.get_job(job_id)["status"] == "cancelled":
                cancel_event.set()
                return
            results = [results_map[i] for i in sorted(results_map)]
            errors = [errors_map[i] for i in sorted(errors_map)]
            summary_extra = {
                "tokens_total": running_tokens,
                "cached_tokens": running_cached,
                "estimated_cost_usd": round(running_cost, 6),
            }

        elif payload.job_type == "render_mp4":
            for ep_id in payload.episode_ids:
                if cancel_event.is_set() or state.repo.get_job(job_id)["status"] == "cancelled":
                    cancel_event.set()
                    return
                try:
                    resolved = state.repo.resolve_episode(ep_id)
                    out_path = (
                        state.workspace
                        / "exports"
                        / "viz_mp4"
                        / f"{resolved.meta.dataset}__{resolved.meta.stem}.mp4"
                    )
                    with state.open_view(resolved.meta, cfg) as view:
                        saved = render_viz_mp4(resolved, view, out_path, cfg=cfg)
                    saved_res = saved.resolve()
                    results.append({"episode_id": ep_id, "out_path": str(saved_res)})
                    artifacts.append(
                        {
                            "name": saved_res.name,
                            "path": str(saved_res),
                            "kind": "mp4",
                            "export_id": None,
                            "episode_id": ep_id,
                        }
                    )
                except Exception as exc:
                    errors.append({"episode_id": ep_id, "error": str(exc)})
                done += 1
                upd = state.repo.update_job(job_id, progress_done=done)
                if upd["status"] != "cancelled":
                    state.sse.publish({"type": "job_update", "job": upd})

        elif payload.job_type == "export":
            if cancel_event.is_set() or state.repo.get_job(job_id)["status"] == "cancelled":
                cancel_event.set()
                return
            exp_dict = dict(payload.export_params or {})
            if exp_dict.get("episode_ids") is None and payload.episode_ids:
                exp_dict["episode_ids"] = list(payload.episode_ids)
            exp_payload = ExportTriggerPayload.model_validate(exp_dict)
            manifest = _run_export_sync(state, exp_payload)
            if cancel_event.is_set() or state.repo.get_job(job_id)["status"] == "cancelled":
                cancel_event.set()
                return
            done = int(cur.get("progress_total") or 1)
            out_resolved = Path(manifest.output_dir).resolve()
            artifacts.append(
                {
                    "name": f"{manifest.export_id}.zip",
                    "path": str(out_resolved),
                    "kind": "export",
                    "export_id": manifest.export_id,
                    "episode_id": None,
                }
            )
            results.append(
                {
                    "export_id": manifest.export_id,
                    "episodes_included": manifest.episodes_included,
                    "episodes_skipped": manifest.episodes_skipped,
                    "pieces_exported": manifest.pieces_exported,
                    "output_dir": str(out_resolved),
                }
            )
            summary_extra = {
                "export_id": manifest.export_id,
                "manifest": manifest.model_dump(mode="json"),
            }

        elif payload.job_type == "eval_gt":
            eval_dict = dict(payload.eval_params or {})
            if eval_dict.get("episode_ids") is None and payload.episode_ids:
                eval_dict["episode_ids"] = list(payload.episode_ids)
            eval_payload = EvalRunPayload.model_validate(eval_dict)
            eff_workers = (
                int(payload.concurrency)
                if payload.concurrency is not None
                else int(eval_payload.workers)
            )
            per_ep_results: list[dict[str, Any]] = []
            cache_hits_acc: Counter[str] = Counter()

            def _on_eval_progress(
                done_n: int, total_n: int, _ep_id: str, info: dict[str, Any]
            ) -> None:
                nonlocal done
                if cancel_event.is_set() or state.repo.get_job(job_id)["status"] == "cancelled":
                    cancel_event.set()
                    raise _JobCancelled()
                cache_hits_acc.update(info.get("cache_hit_stages", []))
                per_ep_results.append(info)
                done = done_n
                upd = state.repo.update_job(
                    job_id,
                    progress_done=done_n,
                    progress_total=total_n,
                    summary={
                        "cache_hit_stages": dict(cache_hits_acc),
                        "results": per_ep_results,
                    },
                )
                if upd["status"] != "cancelled":
                    state.sse.publish({"type": "job_update", "job": upd})

            try:
                report = run_gt_eval(
                    state.repo,
                    config=cfg,
                    config_dir=state.config_dir,
                    dataset=eval_payload.dataset,
                    task_id=eval_payload.task_id,
                    episode_ids=eval_payload.episode_ids,
                    few_shot=eval_payload.few_shot,
                    top_k=eval_payload.top_k,
                    max_tokens=eval_payload.max_tokens,
                    workers=eff_workers,
                    candidate_prompts=eval_payload.candidate_prompts,
                    candidate_labels=eval_payload.candidate_labels,
                    on_progress=_on_eval_progress,
                    cancel_event=cancel_event,
                    on_cancel_register=lambda cb: state.add_job_cancel_callback(job_id, cb),
                )
            except (_JobCancelled, VlmCancelledError):
                cancel_event.set()
                return

            done = report.metrics.episodes_evaluated
            state.repo.update_job(job_id, progress_total=done)
            results = per_ep_results
            report_dict = report.model_dump(mode="json")
            summary_extra = {
                "eval_run_id": report.eval_id,
                "eval_id": report.eval_id,
                "metrics": report.metrics.model_dump(mode="json"),
                "cache_hit_stages": dict(report.metrics.cache_hit_stages),
                "report": report_dict,
                "eval_report": report_dict,
            }

        elif payload.job_type == "ingest":
            ing_payload = IngestParamsPayload.model_validate(payload.ingest_params or {})
            raw_paths = _collect_ingest_paths(ing_payload)
            dataset_override = (
                validate_id_part(ing_payload.dataset.strip(), "dataset")
                if ing_payload.dataset and ing_payload.dataset.strip()
                else None
            )
            task_override = _normalize_ingest_task_override(ing_payload.task_id)
            if task_override is not None and task_override not in cfg.tasks:
                raise ValueError(f"Unknown task_id: {task_override!r}")
            resolved_roots = [
                resolve_allowed_path(p, state.allowed_data_roots) for p in raw_paths
            ]
            task_patterns = cfg.task_patterns()
            all_handles = []
            for data_root in resolved_roots:
                handles = auto_detect(
                    data_root,
                    adapter_name=None if ing_payload.adapter == "auto" else ing_payload.adapter,
                    task_patterns=task_patterns,
                )
                all_handles.extend(handles)

            upd_total = state.repo.update_job(
                job_id,
                progress_total=len(all_handles),
                progress_done=0,
                summary={
                    "discovered": len(all_handles),
                    "inserted": 0,
                    "updated": 0,
                    "skipped": 0,
                    "current_episode_id": None,
                },
            )
            if upd_total["status"] != "cancelled":
                state.sse.publish({"type": "job_update", "job": upd_total})

            inserted = 0
            updated = 0
            skipped = 0
            last_episode_id: str | None = None
            for handle in all_handles:
                if cancel_event.is_set() or state.repo.get_job(job_id)["status"] == "cancelled":
                    cancel_event.set()
                    return
                current_episode_id = handle.episode_id
                try:
                    src_resolved = Path(handle.source_uri).resolve()
                    if not is_within_allowed_roots(src_resolved, state.allowed_data_roots):
                        raise PermissionError(
                            f"Discovered episode source escapes allowed_data_roots: {handle.source_uri}"
                        )
                    ad = get_adapter(handle.adapter_type)
                    with ad.open(handle) as view:
                        meta = view.meta
                        ds_name = dataset_override or meta.dataset
                        if task_override is not None:
                            resolved_task = task_override
                        else:
                            h5_hint = meta.extra_meta.get("h5_task_name") or meta.extra_meta.get(
                                "ep_task_str"
                            )
                            resolved_task = cfg.resolve_task(
                                ds_name, str(h5_hint) if h5_hint else None
                            ).task_id
                        if ds_name != meta.dataset or resolved_task != meta.task_id:
                            meta = meta.model_copy(
                                update={
                                    "dataset": ds_name,
                                    "episode_id": f"{ds_name}/{meta.stem}",
                                    "task_id": resolved_task,
                                }
                            )
                        current_episode_id = meta.episode_id
                        _, outcome = state.repo.ingest_episode(meta)
                        if outcome == "inserted":
                            inserted += 1
                        elif outcome == "updated":
                            updated += 1
                            state.invalidate_kinematics(meta.episode_id)
                        else:
                            skipped += 1
                        results.append(
                            {
                                "episode_id": meta.episode_id,
                                "adapter_type": meta.adapter_type,
                                "task_id": meta.task_id,
                                "outcome": outcome,
                            }
                        )
                except Exception as exc:
                    errors.append({"episode_id": handle.episode_id, "error": str(exc)})
                last_episode_id = current_episode_id
                done += 1
                upd = state.repo.update_job(
                    job_id,
                    progress_done=done,
                    summary={
                        "discovered": len(all_handles),
                        "inserted": inserted,
                        "updated": updated,
                        "skipped": skipped,
                        "current_episode_id": current_episode_id,
                    },
                )
                if upd["status"] != "cancelled":
                    state.sse.publish({"type": "job_update", "job": upd})

            summary_extra = {
                "discovered": len(all_handles),
                "inserted": inserted,
                "updated": updated,
                "skipped": skipped,
                "current_episode_id": last_episode_id,
            }

        if cancel_event.is_set() or state.repo.get_job(job_id)["status"] == "cancelled":
            return
        final_status = "failed" if errors else "completed"
        summary_payload: dict[str, Any] = {
            **summary_extra,
            "results": results,
            "errors": errors,
            "artifacts": artifacts,
        }
        final_job = state.repo.update_job(
            job_id,
            status=final_status,
            progress_done=done,
            summary=summary_payload,
        )
        if final_job["status"] != "cancelled":
            state.sse.publish({"type": "job_update", "job": final_job})
    except Exception as exc:
        restore_needed_on_fail = True
        with contextlib.suppress(Exception):
            cur_after = state.repo.get_job(job_id)
            if not cancel_event.is_set() and cur_after["status"] != "cancelled":
                fail_job = state.repo.update_job(
                    job_id,
                    status="failed",
                    summary={
                        "results": [],
                        "errors": [{"episode_id": "*", "error": str(exc)}],
                        "artifacts": [],
                    },
                )
                if fail_job["status"] != "cancelled":
                    state.sse.publish({"type": "job_update", "job": fail_job})
    finally:
        prev_map = state.get_job_prev_statuses(job_id)
        if payload.job_type == "batch_vlm" and not sandbox_flag:
            is_cancelled = cancel_event.is_set()
            if not is_cancelled:
                with contextlib.suppress(Exception):
                    is_cancelled = state.repo.get_job(job_id)["status"] == "cancelled"
            if is_cancelled or restore_needed_on_fail:
                restorable_ids = [
                    eid
                    for eid in payload.episode_ids
                    if state.can_job_restore_episode(job_id, eid)
                ]
                if restorable_ids:
                    with contextlib.suppress(Exception):
                        restored = state.repo.restore_vlm_episodes_on_cancel(
                            restorable_ids, prev_statuses=prev_map or None
                        )
                        for r_ep, r_st in restored:
                            _publish_episode_status(state, r_ep, r_st)
        state.clear_job_control(job_id)
        state.repo.db.close_thread_connection()


@router.get("/api/jobs")
def list_jobs(
    status_filter: str | None = Query(default=None, alias="status"),
    job_type: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    items = state.repo.list_jobs(
        status=status_filter,
        job_type=job_type,
        limit=limit,
        offset=offset,
    )
    total = state.repo.count_jobs(
        status=status_filter,
        job_type=job_type,
    )
    status_counts = state.repo.count_jobs_by_status(job_type=job_type)
    return {"items": items, "total": total, "status_counts": status_counts}


@router.delete("/api/jobs")
def clear_terminal_jobs(
    scope: Literal["finished", "failed_cancelled"] = Query(default="finished"),
    job_type: str | None = Query(default=None),
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    """Clear terminal job records from SQLite (Lead-only; never touches active jobs or disk artifacts)."""
    _ = auth
    target_statuses = (
        ("completed", "failed", "cancelled")
        if scope == "finished"
        else ("failed", "cancelled")
    )
    active_control_ids = state.active_job_control_ids()
    deleted_ids = state.repo.delete_terminal_jobs(
        statuses=target_statuses,
        job_type=job_type,
        exclude_job_ids=active_control_ids,
    )
    status_counts = state.repo.count_jobs_by_status(job_type=job_type)
    if deleted_ids:
        state.sse.publish(
            {
                "type": "jobs_deleted",
                "job_ids": deleted_ids,
                "scope": scope,
                "status_counts": status_counts,
            }
        )
    return {
        "deleted": len(deleted_ids),
        "job_ids": deleted_ids,
        "scope": scope,
        "status_counts": status_counts,
    }


@router.post("/api/jobs", status_code=status.HTTP_201_CREATED)
def create_job(
    payload: JobCreatePayload,
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    eff_fresh, fresh_stages = _resolve_stage_options(payload.stage, payload.fresh)
    raw_ep_ids = list(payload.episode_ids)
    if payload.job_type == "batch_vlm" and raw_ep_ids:
        ep_ids = list(dict.fromkeys(raw_ep_ids))
        if ep_ids != raw_ep_ids:
            payload = payload.model_copy(update={"episode_ids": ep_ids})
    else:
        ep_ids = raw_ep_ids

    if payload.job_type in ("batch_vlm", "render_mp4") and not ep_ids:
        job = state.repo.create_job(
            payload.job_type,
            payload.model_dump(mode="json"),
            progress_total=0,
            status="completed",
        )
        job = state.repo.update_job(
            job["job_id"],
            status="completed",
            progress_done=0,
            summary={"results": [], "errors": [], "artifacts": []},
        )
        state.sse.publish({"type": "job_update", "job": job})
        return job

    progress_total = len(ep_ids)
    if payload.job_type == "export":
        exp_dict = dict(payload.export_params or {})
        if exp_dict.get("episode_ids") is None and ep_ids:
            exp_dict["episode_ids"] = ep_ids
        try:
            exp_payload = ExportTriggerPayload.model_validate(exp_dict)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        progress_total = (
            len(exp_payload.episode_ids) if exp_payload.episode_ids is not None else 1
        )
    elif payload.job_type == "eval_gt":
        eval_dict = dict(payload.eval_params or {})
        if eval_dict.get("episode_ids") is None and ep_ids:
            eval_dict["episode_ids"] = ep_ids
        try:
            eval_payload = EvalRunPayload.model_validate(eval_dict)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        frozen = state.repo.list_episodes(
            status=EpisodeStatus.GT_FROZEN,
            dataset=eval_payload.dataset,
            task_id=eval_payload.task_id,
            limit=100000,
        )
        if eval_payload.episode_ids is not None:
            wanted = set(eval_payload.episode_ids)
            frozen = [r for r in frozen if r["episode_id"] in wanted]
        progress_total = len(frozen)
    elif payload.job_type == "ingest":
        try:
            ing_payload = IngestParamsPayload.model_validate(payload.ingest_params or {})
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        raw_paths = _collect_ingest_paths(ing_payload)
        if not raw_paths:
            raise HTTPException(
                status_code=422,
                detail="ingest_params must provide at least one non-empty path",
            )
        if ing_payload.dataset is not None and ing_payload.dataset.strip():
            try:
                validate_id_part(ing_payload.dataset.strip(), "dataset")
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        task_override = _normalize_ingest_task_override(ing_payload.task_id)
        if task_override is not None:
            cfg = state.get_config()
            if task_override not in cfg.tasks:
                raise HTTPException(
                    status_code=422,
                    detail=f"Unknown task_id: {task_override!r}",
                )
        for rp in raw_paths:
            resolved_p = resolve_allowed_path(rp, state.allowed_data_roots)
            if not resolved_p.exists():
                raise HTTPException(status_code=404, detail=f"Data path not found: {rp}")
        progress_total = len(raw_paths)

    sandbox_flag = (
        bool(payload.is_eval_sandbox)
        if payload.is_eval_sandbox is not None
        else False
    )
    reserve_token: str | None = None
    if payload.job_type == "batch_vlm" and not sandbox_flag and ep_ids:
        reserve_token = f"pending_{uuid.uuid4().hex}"
        busy_mem = state.try_reserve_vlm_episodes(ep_ids, reserve_token)
        if busy_mem:
            raise HTTPException(
                status_code=409,
                detail=f"Episodes already queued or running VLM: {', '.join(busy_mem)}",
            )
        busy_db: list[str] = []
        for eid in ep_ids:
            try:
                row = state.repo.get_episode_row(eid)
                if row["status"] in (
                    EpisodeStatus.VLM_QUEUED.value,
                    EpisodeStatus.VLM_RUNNING.value,
                ):
                    busy_db.append(eid)
            except NotFoundError:
                continue
        if busy_db:
            state.release_vlm_episodes(reserve_token)
            raise HTTPException(
                status_code=409,
                detail=f"Episodes already queued or running VLM: {', '.join(busy_db)}",
            )

    try:
        job = state.repo.create_job(
            payload.job_type,
            payload.model_dump(mode="json"),
            progress_total=progress_total,
            status="queued",
        )
    except Exception:
        if reserve_token is not None:
            state.release_vlm_episodes(reserve_token)
        raise

    if reserve_token is not None:
        state.rebind_vlm_episodes(reserve_token, job["job_id"])

    prev_statuses: dict[str, str] = {}
    if payload.job_type == "batch_vlm" and not sandbox_flag and ep_ids:
        queued_eps = state.repo.queue_vlm_episodes(ep_ids)
        prev_statuses = {eid: prev_st.value for eid, prev_st in queued_eps}
        state.register_job_control(job["job_id"], prev_statuses=prev_statuses)
        for eid, _ in queued_eps:
            _publish_episode_status(state, eid, EpisodeStatus.VLM_QUEUED)
    else:
        state.register_job_control(job["job_id"])

    state.sse.publish({"type": "job_update", "job": job})
    try:
        state.executor.submit(
            _run_job_worker, state, job["job_id"], payload, eff_fresh, fresh_stages
        )
    except Exception:
        if payload.job_type == "batch_vlm" and not sandbox_flag and ep_ids:
            with contextlib.suppress(Exception):
                restored = state.repo.restore_vlm_episodes_on_cancel(
                    ep_ids, prev_statuses=prev_statuses or None
                )
                for r_ep, r_st in restored:
                    _publish_episode_status(state, r_ep, r_st)
        state.clear_job_control(job["job_id"])
        raise
    return job


@router.post("/api/jobs/estimate")
def estimate_job_cost(
    payload: JobEstimatePayload,
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    """Estimate token usage and USD cost for a batch VLM job without calling VLM APIs."""
    _ = auth
    _resolve_stage_options(payload.stage, payload.fresh)
    unique_ids = list(
        dict.fromkeys(str(eid).strip() for eid in payload.episode_ids if str(eid).strip())
    )
    for eid in unique_ids:
        try:
            state.repo.get_episode_row(eid)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    from ...vlm.estimator import estimate_batch_vlm_cost

    cfg = state.get_config()
    eff_stage = payload.stage.strip() if payload.stage and payload.stage.strip() else "all"
    try:
        return estimate_batch_vlm_cost(
            cfg,
            state.repo,
            state.workspace,
            unique_ids,
            stage=eff_stage,
            fresh=bool(payload.fresh),
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/api/jobs/{job_id}/artifact")
async def download_job_artifact(
    job_id: str,
    index: int = Query(default=0, ge=0),
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> Any:
    _ = auth
    try:
        job = state.repo.get_job(job_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    summary = job.get("summary") or {}
    artifacts = list(summary.get("artifacts") or [])
    if not artifacts and job.get("job_type") == "render_mp4":
        for r in summary.get("results") or []:
            if isinstance(r, dict) and r.get("out_path"):
                p_r = Path(str(r["out_path"]))
                artifacts.append(
                    {
                        "name": p_r.name,
                        "path": str(p_r),
                        "kind": "mp4",
                        "export_id": None,
                        "episode_id": r.get("episode_id"),
                    }
                )

    if index >= len(artifacts):
        raise HTTPException(
            status_code=404,
            detail=f"Artifact index {index} not found for job {job_id!r}",
        )

    art = artifacts[index]
    raw_path = art.get("path")
    if not raw_path:
        raise HTTPException(status_code=404, detail="Artifact entry has no path")

    ws_root = state.workspace.resolve()
    p = Path(str(raw_path))
    resolved = p.resolve() if p.is_absolute() else (ws_root / p).resolve()
    if not resolved.is_relative_to(ws_root):
        raise HTTPException(
            status_code=403,
            detail="Artifact path escapes workspace boundary",
        )

    is_export_dir = art.get("kind") == "export" or resolved.is_dir()
    if is_export_dir:
        if not resolved.is_dir():
            raise HTTPException(status_code=404, detail="Export artifact directory missing")
        safe_name = (
            Path(str(art.get("name") or f"{resolved.name}.zip"))
            .name.replace('"', "")
            .replace("\r", "")
            .replace("\n", "")
        )
        if not safe_name.endswith(".zip"):
            safe_name = f"{safe_name}.zip"

        def _build_zip() -> tempfile.SpooledTemporaryFile[bytes]:
            tmp = tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024)
            with zipfile.ZipFile(tmp, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for fpath in sorted(resolved.rglob("*")):
                    try:
                        f_res = fpath.resolve()
                    except OSError:
                        continue
                    if not f_res.is_relative_to(resolved):
                        continue
                    if f_res.is_file():
                        zf.write(f_res, arcname=fpath.relative_to(resolved).as_posix())
            tmp.seek(0)
            return tmp

        tmp_file = await run_in_threadpool(_build_zip)

        def _iter_chunks():
            try:
                while True:
                    chunk = tmp_file.read(1024 * 1024)
                    if not chunk:
                        break
                    yield chunk
            finally:
                tmp_file.close()

        return StreamingResponse(
            _iter_chunks(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{safe_name}"'},
        )

    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="Artifact file missing")
    safe_name = Path(str(art.get("name") or resolved.name)).name
    media_type = "video/mp4" if resolved.suffix.lower() == ".mp4" else "application/octet-stream"
    return FileResponse(path=resolved, media_type=media_type, filename=safe_name)


@router.get("/api/jobs/{job_id}")
def get_job(
    job_id: str,
    auth: AuthContext = Depends(require_auth),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    try:
        return state.repo.get_job(job_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/api/jobs/{job_id}")
def delete_job(
    job_id: str,
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    """Delete a single terminal job record from SQLite (Lead-only; never deletes disk artifacts)."""
    _ = auth
    if state.has_job_control(job_id):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete active job {job_id!r} while execution is in progress; cancel it first.",
        )
    try:
        state.repo.delete_job(job_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvalidTransitionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    status_counts = state.repo.count_jobs_by_status()
    state.sse.publish(
        {
            "type": "jobs_deleted",
            "job_ids": [job_id],
            "scope": "single",
            "status_counts": status_counts,
        }
    )
    return {
        "deleted": 1,
        "job_ids": [job_id],
        "scope": "single",
        "status_counts": status_counts,
    }


@router.post("/api/jobs/{job_id}/cancel")
def cancel_job(
    job_id: str,
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    try:
        before = state.repo.get_job(job_id)
        was_active = (
            before["status"] in ("queued", "running")
            or state.has_job_control(job_id)
        )
        job = state.repo.cancel_job(job_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if job["status"] == "cancelled":
        prev_statuses = state.trigger_job_cancel(job_id)
        if was_active and job["job_type"] == "batch_vlm":
            params = job.get("params") or {}
            if not bool(params.get("is_eval_sandbox")):
                ep_ids = [str(e) for e in (params.get("episode_ids") or []) if e]
                if ep_ids:
                    restored = state.repo.restore_vlm_episodes_on_cancel(
                        ep_ids, prev_statuses=prev_statuses or None
                    )
                    for r_ep, r_st in restored:
                        _publish_episode_status(state, r_ep, r_st)
                state.release_vlm_episodes(job_id)

    state.sse.publish({"type": "job_update", "job": job})
    return job


@router.get("/api/jobs/{job_id}/events")
async def stream_job_events(
    job_id: str,
    request: Request,
    auth: AuthContext = Depends(require_auth),
    state: AppState = Depends(get_app_state),
) -> StreamingResponse:
    _ = auth
    try:
        initial = await run_in_threadpool(state.repo.get_job, job_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    async def _gen():
        yield f"data: {json.dumps(initial, ensure_ascii=False)}\n\n"
        if initial["status"] in ("completed", "failed", "cancelled"):
            return
        last_sig = (initial["status"], initial["progress_done"], initial["updated_at"])
        while not await request.is_disconnected():
            await asyncio.sleep(0.08)
            try:
                cur = await run_in_threadpool(state.repo.get_job, job_id)
            except NotFoundError:
                break
            sig = (cur["status"], cur["progress_done"], cur["updated_at"])
            if sig != last_sig:
                yield f"data: {json.dumps(cur, ensure_ascii=False)}\n\n"
                last_sig = sig
            if cur["status"] in ("completed", "failed", "cancelled"):
                break

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/api/events")
async def stream_global_events(
    request: Request,
    auth: AuthContext = Depends(require_auth),
    state: AppState = Depends(get_app_state),
) -> StreamingResponse:
    _ = auth
    start_seq = state.sse.latest_seq()

    async def _gen():
        yield f"data: {json.dumps({'type': 'connected', 'ts': time.time()})}\n\n"
        seq = start_seq
        last_ping = time.monotonic()
        while not await request.is_disconnected():
            new_seq, events = state.sse.since(seq)
            if events:
                seq = new_seq
                for ev in events:
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                last_ping = time.monotonic()
            else:
                if time.monotonic() - last_ping >= 15.0:
                    yield ": keepalive\n\n"
                    last_ping = time.monotonic()
                await asyncio.sleep(0.15)

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
