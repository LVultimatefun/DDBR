"""Lead-only dataset export dry-run, synchronous transactional export, and ZIP download."""

from __future__ import annotations

import contextlib
import json
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from ...config import ConfigManager
from ...export import PieceSlicer, get_exporter, group_kept_pieces
from ...io.base import EpisodeView, validate_id_part
from ...models.export import (
    ExportFormat,
    ExportManifest,
    ExportRecipe,
    ExportRecipeName,
    KeepPiece,
    SmoothTailMode,
)
from ...models.review import EpisodeStatus, ResolvedEpisode
from ...store import NotFoundError
from ...store.artifacts import atomic_write
from ..auth import AuthContext, get_app_state, require_lead
from ..state import AppState

router = APIRouter(prefix="/api/exports", tags=["exports"])


class ExportTriggerPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: ExportFormat
    recipe: ExportRecipeName
    chunk_size: int | None = Field(default=None, ge=2, le=2000)
    min_piece_s: float | None = Field(default=None, gt=0.0)
    tail_frames: int | None = Field(default=None, ge=0, le=2000)
    smooth_tail: SmoothTailMode | None = None
    include_statuses: list[EpisodeStatus] | None = None
    context_window_s: float = Field(default=2.0, ge=0.0)
    dataset_filter: str | None = None
    task_filter: str | None = None
    episode_ids: list[str] | None = None


def _build_recipe(payload: ExportTriggerPayload, cfg: ConfigManager) -> ExportRecipe:
    exp_cfg = cfg.settings.get("export", {})
    eff_chunk = payload.chunk_size if payload.chunk_size is not None else exp_cfg.get("chunk_size")
    eff_min_piece = (
        float(payload.min_piece_s)
        if payload.min_piece_s is not None
        else float(exp_cfg.get("min_piece_s", 0.1))
    )
    eff_tail = payload.tail_frames if payload.tail_frames is not None else exp_cfg.get("tail_frames")
    eff_smooth = (
        payload.smooth_tail
        if payload.smooth_tail is not None
        else str(exp_cfg.get("smooth_tail", "on_jump"))
    )
    eff_statuses = (
        list(payload.include_statuses)
        if payload.include_statuses is not None
        else [EpisodeStatus.APPROVED, EpisodeStatus.GT_FROZEN]
    )
    return ExportRecipe(
        name=payload.recipe,
        format=payload.format,
        chunk_size=eff_chunk,
        min_piece_s=eff_min_piece,
        tail_frames=eff_tail,
        smooth_tail=eff_smooth,  # type: ignore[arg-type]
        include_statuses=eff_statuses,
        context_window_s=payload.context_window_s,
        dataset_filter=payload.dataset_filter,
        task_filter=payload.task_filter,
    )


def _exports_manifest_dir(state: AppState) -> Path:
    d = state.workspace / "artifacts" / "exports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_manifest(state: AppState, export_id: str) -> ExportManifest:
    try:
        eid = validate_id_part(export_id, "export_id")
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    p = _exports_manifest_dir(state) / f"{eid}.json"
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"Export not found: {export_id!r}")
    return ExportManifest.model_validate_json(p.read_text(encoding="utf-8"))


def _gather_slices(
    state: AppState,
    cfg: ConfigManager,
    recipe: ExportRecipe,
    episode_ids: list[str] | None,
    stack: contextlib.ExitStack,
) -> tuple[list[KeepPiece], list[ResolvedEpisode], dict[str, EpisodeView]]:
    slicer = PieceSlicer(cfg)
    allowed_statuses = set(recipe.include_statuses)
    if episode_ids is not None:
        candidate_rows = [state.repo.get_episode_row(ep_id) for ep_id in episode_ids]
    else:
        candidate_rows = state.repo.list_episodes(
            dataset=recipe.dataset_filter,
            task_id=recipe.task_filter,
            limit=100000,
        )

    resolved_list: list[ResolvedEpisode] = []
    views_by_ep: dict[str, EpisodeView] = {}
    all_pieces: list[KeepPiece] = []
    for row in candidate_rows:
        ep_id = str(row["episode_id"])
        if EpisodeStatus(str(row["status"])) not in allowed_statuses:
            continue
        if not row.get("active_run_id"):
            continue
        resolved = state.repo.resolve_episode(ep_id)
        view = stack.enter_context(state.open_view(resolved.meta, cfg))
        ep_pieces = slicer.slice(resolved, view, recipe)
        resolved_list.append(resolved)
        views_by_ep[ep_id] = view
        all_pieces.extend(ep_pieces)
    return all_pieces, resolved_list, views_by_ep


def _run_dry_run_sync(state: AppState, payload: ExportTriggerPayload) -> ExportManifest:
    cfg = state.get_config()
    recipe = _build_recipe(payload, cfg)
    with contextlib.ExitStack() as stack:
        all_pieces, resolved_list, _ = _gather_slices(
            state, cfg, recipe, payload.episode_ids, stack
        )
        _, entries, ep_inc, ep_skp, p_exp, p_drop, tot_f = group_kept_pieces(
            all_pieces, resolved_list, recipe
        )
    return ExportManifest(
        export_id="dry_run",
        created_at=time.time(),
        format=recipe.format,
        recipe=recipe,
        output_dir="",
        episodes_included=ep_inc,
        episodes_skipped=ep_skp,
        pieces_exported=p_exp,
        pieces_dropped=p_drop,
        total_frames=tot_f,
        episodes=entries,
    )


def _run_export_sync(state: AppState, payload: ExportTriggerPayload) -> ExportManifest:
    cfg = state.get_config()
    recipe = _build_recipe(payload, cfg)
    export_id = f"exp_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    out_dir = (state.workspace / "exports" / export_id).resolve()

    with contextlib.ExitStack() as stack:
        all_pieces, resolved_list, views_by_ep = _gather_slices(
            state, cfg, recipe, payload.episode_ids, stack
        )
        base_exporter = get_exporter(recipe.format)
        try:
            exporter = type(base_exporter)(cfg=cfg)
        except TypeError:
            exporter = base_exporter
        manifest = exporter.export(
            all_pieces, resolved_list, views_by_ep, out_dir, recipe
        )
        if manifest.export_id != export_id:
            manifest = manifest.model_copy(update={"export_id": export_id})
            qc_m_path = out_dir / "meta" / "qc_export_manifest.json"
            if qc_m_path.is_file():
                atomic_write(qc_m_path, manifest.model_dump_json(indent=2))

    manifest_path = _exports_manifest_dir(state) / f"{manifest.export_id}.json"
    atomic_write(manifest_path, manifest.model_dump_json(indent=2))
    state.sse.publish(
        {
            "type": "export_completed",
            "export_id": manifest.export_id,
            "format": manifest.format,
            "recipe": manifest.recipe.name,
        }
    )
    return manifest


@router.post("/dry-run")
async def dry_run_export(
    payload: ExportTriggerPayload,
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    try:
        manifest = await run_in_threadpool(_run_dry_run_sync, state, payload)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return manifest.model_dump(mode="json")


@router.post("")
async def trigger_export(
    payload: ExportTriggerPayload,
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    try:
        manifest = await run_in_threadpool(_run_export_sync, state, payload)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return manifest.model_dump(mode="json")


@router.get("")
def list_exports(
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    mdir = _exports_manifest_dir(state)
    items: list[ExportManifest] = []
    for p in sorted(mdir.glob("*.json")):
        try:
            items.append(ExportManifest.model_validate_json(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    items.sort(key=lambda m: m.created_at, reverse=True)
    return {
        "items": [m.model_dump(mode="json") for m in items],
        "total": len(items),
    }


@router.get("/{export_id}/download")
async def download_export(
    export_id: str,
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> StreamingResponse:
    _ = auth
    manifest = _load_manifest(state, export_id)
    ws_root = state.workspace.resolve()
    raw_out = Path(manifest.output_dir)
    out_dir = raw_out.resolve() if raw_out.is_absolute() else (ws_root / raw_out).resolve()
    if not out_dir.is_relative_to(ws_root):
        raise HTTPException(
            status_code=403,
            detail="Export output directory escapes workspace boundary",
        )
    if not out_dir.is_dir():
        raise HTTPException(
            status_code=404, detail=f"Export output directory missing for {export_id!r}"
        )

    def _build_zip() -> tempfile.SpooledTemporaryFile[bytes]:
        tmp = tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024)
        with zipfile.ZipFile(tmp, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fpath in sorted(out_dir.rglob("*")):
                try:
                    f_res = fpath.resolve()
                except OSError:
                    continue
                if not f_res.is_relative_to(out_dir):
                    continue
                if f_res.is_file():
                    zf.write(f_res, arcname=fpath.relative_to(out_dir).as_posix())
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
        headers={
            "Content-Disposition": f'attachment; filename="{manifest.export_id}.zip"'
        },
    )


@router.get("/{export_id}")
def get_export(
    export_id: str,
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    return _load_manifest(state, export_id).model_dump(mode="json")
