"""Lead-only VLM Lab router: config CRUD w/ .history, diagnosis, unknowns triage, Few-Shots, GT eval, promote, prompt assist."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ...config import PROMPT_STAGE_FILES, ConfigManager
from ...io.base import validate_id_part
from ...lab.diagnosis import diagnose_workspace
from ...lab.eval_runner import (
    SnapshotValidationError,
    archive_and_write_override,
    compare_eval_runs,
    diff_history_snapshot,
    list_eval_reports,
    list_history_snapshots,
    load_eval_report,
    promote_candidate_config,
    rollback_history_snapshot,
    run_gt_eval,
    workspace_config_lock,
)
from ...lab.few_shot import create_few_shot_example
from ...lab.prompt_assist import suggest_prompt_updates
from ...models.annotation import QcSegment
from ...models.taxonomy import (
    DiagnosisErrorType,
    FewShotItem,
    LabelRegistry,
    PromptTemplate,
    TaskSpec,
    VisualLabelDef,
)
from ...store.artifacts import atomic_write
from ...store.repository import NotFoundError
from ...vlm.client import VlmError, normalize_stage_name
from ..auth import AuthContext, get_app_state, require_lead
from ..state import AppState

router = APIRouter(
    prefix="/api/lab",
    tags=["lab"],
    dependencies=[Depends(require_lead)],
)


class UnknownTriagePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(min_length=3)
    seg_id: str = Field(min_length=1)
    action: Literal["merge", "promote"]
    target_label_id: str | None = None
    rationale: str | None = None
    new_label: dict[str, Any] | None = None


class FewShotCreatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: str = Field(default="global", min_length=1)
    task_id: str | None = None
    label_id: str | None = None
    error_type: DiagnosisErrorType = "fp_overflag"
    episode_id: str = Field(min_length=3)
    t_start: float = Field(ge=0.0)
    t_end: float = Field(gt=0.0)
    vlm_wrong_json: dict[str, Any] = Field(default_factory=dict)
    human_right_json: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(min_length=1)
    capture_tile: bool = True

    @field_validator("scope")
    @classmethod
    def _validate_scope(cls, v: str) -> str:
        if v == "global":
            return v
        if v.startswith("task:") or v.startswith("label:"):
            suffix = v.split(":", 1)[1].strip()
            if suffix:
                return v
        raise ValueError(
            f"scope must be 'global', 'task:<id>', or 'label:<id>' with non-empty id, got {v!r}"
        )

    @model_validator(mode="after")
    def _validate_span(self) -> "FewShotCreatePayload":
        if self.t_end <= self.t_start:
            raise ValueError(f"t_end ({self.t_end}) must be > t_start ({self.t_start})")
        return self


class FewShotPatchPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool


class EvalRunPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str | None = None
    task_id: str | None = None
    episode_ids: list[str] | None = None
    few_shot: bool = True
    top_k: int | None = Field(default=None, ge=0, le=50)
    max_tokens: int | None = Field(default=None, ge=1, le=50000)
    workers: int = Field(default=4, ge=1, le=32)
    candidate_prompts: dict[str, Any] | None = None
    candidate_labels: dict[str, Any] | None = None


class PromotePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eval_id: str | None = None
    dataset: str | None = None
    candidate_prompts: dict[str, Any] | None = None
    candidate_labels: dict[str, Any] | None = None
    candidate_tasks: dict[str, Any] | None = None
    reason: str = "lab_promote"


class PromptAssistPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str = "s3_discover"
    task_id: str | None = None
    episode_ids: list[str] | None = None
    top_n: int = Field(default=10, ge=1, le=200)


def _unknown_triage_path(workspace: Path) -> Path:
    return Path(workspace).resolve() / "artifacts" / "unknown_triage.json"


def _load_triaged_records(workspace: Path) -> list[dict[str, Any]]:
    path = _unknown_triage_path(workspace)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data) if isinstance(data, list) else []
    except Exception:
        return []


def _record_unknown_triage(workspace: Path, entry: dict[str, Any]) -> None:
    with workspace_config_lock(workspace):
        records = _load_triaged_records(workspace)
        records = [
            r
            for r in records
            if not (
                r.get("episode_id") == entry.get("episode_id")
                and r.get("seg_id") == entry.get("seg_id")
            )
        ]
        records.append(entry)
        atomic_write(
            _unknown_triage_path(workspace),
            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        )


def _episode_segments_for_unknowns(state: AppState, row: dict[str, Any]) -> list[QcSegment]:
    if row.get("active_rev_id"):
        try:
            rev = state.repo.get_review(str(row["active_rev_id"]))
            return list(rev.qc_segments)
        except NotFoundError:
            pass
    if row.get("active_run_id"):
        try:
            run = state.repo.get_vlm_run(str(row["active_run_id"]))
            return list(run.qc_segments)
        except NotFoundError:
            pass
    return []


@router.get("/labels")
def get_lab_labels(
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    cfg = state.get_config()
    return cfg.labels.model_dump(mode="json")


@router.put("/labels")
def put_lab_labels(
    payload: LabelRegistry,
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    yaml_text = yaml.safe_dump(
        payload.model_dump(mode="json"), sort_keys=False, allow_unicode=True
    )
    with workspace_config_lock(state.workspace):
        archive_and_write_override(
            state.workspace,
            state.config_dir,
            Path("labels.yaml"),
            yaml_text,
            reason="lab_labels_update",
        )
        state.invalidate_config()
        updated = state.get_config().labels.model_dump(mode="json")
    state.sse.publish(
        {"type": "config_updated", "target": "labels", "version": payload.version}
    )
    return updated


@router.get("/prompts")
def get_lab_prompts(
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    return state.get_config().prompts.model_dump(mode="json")


@router.put("/prompts")
def put_lab_prompts(
    body: dict[str, Any],
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    cfg = state.get_config()
    stage_updates: dict[str, PromptTemplate] = {}

    if "stage" in body:
        raw_stage = str(body["stage"])
        norm = normalize_stage_name(raw_stage)
        if norm not in PROMPT_STAGE_FILES:
            raise HTTPException(status_code=422, detail=f"Invalid prompt stage: {raw_stage!r}")
        base_dict = getattr(cfg.prompts, norm).model_dump(mode="json")
        try:
            stage_updates[norm] = PromptTemplate.model_validate({**base_dict, **body, "stage": norm})
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    else:
        for raw_key, raw_val in body.items():
            norm = normalize_stage_name(str(raw_key))
            if norm not in PROMPT_STAGE_FILES or not isinstance(raw_val, dict):
                raise HTTPException(
                    status_code=422, detail=f"Invalid prompt stage payload for {raw_key!r}"
                )
            base_dict = getattr(cfg.prompts, norm).model_dump(mode="json")
            try:
                stage_updates[norm] = PromptTemplate.model_validate(
                    {**base_dict, **raw_val, "stage": norm}
                )
            except ValidationError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not stage_updates:
        raise HTTPException(status_code=422, detail="No prompt stages provided")

    with workspace_config_lock(state.workspace):
        for norm, tpl in stage_updates.items():
            rel = Path("prompts") / f"{norm}.yaml"
            base_cands = [state.config_dir / "prompts" / fn for fn in PROMPT_STAGE_FILES[norm]]
            archive_and_write_override(
                state.workspace,
                state.config_dir,
                rel,
                yaml.safe_dump(tpl.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
                reason="lab_prompts_update",
                base_candidates=base_cands,
            )
        state.invalidate_config()
        updated_prompts = state.get_config().prompts.model_dump(mode="json")

    state.sse.publish(
        {"type": "config_updated", "target": "prompts", "stages": sorted(stage_updates)}
    )
    return updated_prompts


@router.get("/tasks")
def get_lab_tasks(
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    cfg = state.get_config()
    return {
        "tasks": {tid: spec.model_dump(mode="json") for tid, spec in cfg.tasks.items()},
        "total": len(cfg.tasks),
    }


@router.put("/tasks")
def put_lab_tasks(
    body: dict[str, Any],
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    specs: dict[str, TaskSpec] = {}
    if isinstance(body.get("tasks"), dict):
        for tid_raw, raw_spec in body["tasks"].items():
            if not isinstance(raw_spec, dict):
                raise HTTPException(status_code=422, detail=f"Invalid task spec for {tid_raw!r}")
            try:
                tid = validate_id_part(str(tid_raw), "task_id")
                specs[tid] = TaskSpec.model_validate({**raw_spec, "task_id": tid})
            except (ValueError, ValidationError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
    elif "task_id" in body:
        try:
            tid = validate_id_part(str(body["task_id"]), "task_id")
            specs[tid] = TaskSpec.model_validate(body)
        except (ValueError, ValidationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    else:
        raise HTTPException(
            status_code=422,
            detail="Expected TaskSpec object or {'tasks': {...}} map",
        )

    if not specs:
        raise HTTPException(status_code=422, detail="No task specs provided")

    with workspace_config_lock(state.workspace):
        for tid, spec in specs.items():
            rel = Path("tasks") / f"{tid}.yaml"
            archive_and_write_override(
                state.workspace,
                state.config_dir,
                rel,
                yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
                reason="lab_tasks_update",
            )
        state.invalidate_config()
        cfg_after = state.get_config()

    state.sse.publish(
        {"type": "config_updated", "target": "tasks", "task_ids": sorted(specs)}
    )
    return {
        "tasks": {tid: spec.model_dump(mode="json") for tid, spec in cfg_after.tasks.items()},
        "total": len(cfg_after.tasks),
    }


@router.get("/diagnosis")
async def get_lab_diagnosis(
    episode_ids: list[str] | None = Query(default=None),
    top_n: int = Query(default=20, ge=1, le=500),
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    norm_ids: list[str] | None = None
    if episode_ids is not None:
        norm_ids = []
        for item in episode_ids:
            for part in item.split(","):
                cleaned = part.strip()
                if cleaned:
                    norm_ids.append(cleaned)
    report = await run_in_threadpool(
        diagnose_workspace,
        state.repo,
        episode_ids=norm_ids,
        top_n=top_n,
    )
    return report.model_dump(mode="json")


@router.get("/unknowns")
def get_lab_unknowns(
    dataset: str | None = None,
    limit: int = Query(default=200, ge=1, le=2000),
    include_triaged: bool = False,
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    triaged_keys = {
        (str(r.get("episode_id")), str(r.get("seg_id")))
        for r in _load_triaged_records(state.workspace)
    }
    rows = state.repo.list_episodes(dataset=dataset, limit=10000)
    items: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("active_run_id"):
            continue
        ep_id = str(row["episode_id"])
        segs = _episode_segments_for_unknowns(state, row)
        for seg in segs:
            if not (seg.is_unknown or seg.label_id == "unknown_anomaly"):
                continue
            if not include_triaged and (ep_id, seg.seg_id) in triaged_keys:
                continue
            items.append(
                {
                    "episode_id": ep_id,
                    "dataset": row["dataset"],
                    "task_id": row["task_id"],
                    "status": row["status"],
                    "seg_id": seg.seg_id,
                    "t_start": seg.t_start,
                    "t_end": seg.t_end,
                    "label_id": seg.label_id,
                    "affected_arm": seg.affected_arm,
                    "unknown_note": seg.unknown_note,
                    "suggested_label": seg.suggested_label,
                    "visual_evidence": seg.visual_evidence.model_dump(mode="json"),
                    "confidence": seg.confidence,
                }
            )
    return {"items": items[:limit], "total": len(items)}


@router.post("/unknowns/triage")
async def post_lab_unknowns_triage(
    payload: UnknownTriagePayload,
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    try:
        ep_row = state.repo.get_episode_row(payload.episode_id)
        ep_meta = state.repo.get_episode_meta(payload.episode_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    segs = _episode_segments_for_unknowns(state, ep_row)
    seg = next((s for s in segs if s.seg_id == payload.seg_id), None)
    if seg is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown segment {payload.seg_id!r} not found in {payload.episode_id!r}",
        )
    if not (seg.is_unknown or seg.label_id == "unknown_anomaly"):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Segment {payload.seg_id!r} in {payload.episode_id!r} is not an unknown anomaly "
                f"(label_id={seg.label_id!r})"
            ),
        )

    cfg = state.get_config()

    if payload.action == "merge":
        if not payload.target_label_id:
            raise HTTPException(
                status_code=400, detail="target_label_id is required for action='merge'"
            )
        target_lbl = cfg.labels.get(payload.target_label_id)
        if target_lbl is None:
            raise HTTPException(
                status_code=404,
                detail=f"target_label_id {payload.target_label_id!r} not found in registry",
            )
        rationale = (
            (payload.rationale or "").strip()
            or seg.unknown_note.strip()
            or f"Merged unknown anomaly into {target_lbl.id}"
        )
        fs_item = await run_in_threadpool(
            create_few_shot_example,
            state.repo,
            config=cfg,
            episode_id=payload.episode_id,
            t_start=seg.t_start,
            t_end=seg.t_end,
            scope=f"label:{target_lbl.id}",
            task_id=ep_meta.task_id,
            label_id=target_lbl.id,
            error_type="policy_flip",
            vlm_wrong_json={
                "label_id": seg.label_id,
                "unknown_note": seg.unknown_note,
            },
            human_right_json={
                "label_id": target_lbl.id,
                "splice_policy": target_lbl.default_splice_policy,
                "training_role": target_lbl.default_training_role,
            },
            rationale=rationale,
            capture_tile=True,
        )
        _record_unknown_triage(
            state.workspace,
            {
                "episode_id": payload.episode_id,
                "seg_id": payload.seg_id,
                "action": "merge",
                "target_label_id": target_lbl.id,
                "few_shot_id": fs_item.example_id,
                "triaged_at": time.time(),
            },
        )
        return {
            "ok": True,
            "action": "merge",
            "episode_id": payload.episode_id,
            "seg_id": payload.seg_id,
            "target_label_id": target_lbl.id,
            "few_shot_id": fs_item.example_id,
            "label_version": cfg.labels.version,
        }

    # action == "promote"
    if not payload.new_label:
        raise HTTPException(
            status_code=400, detail="new_label is required for action='promote'"
        )
    try:
        new_lbl = VisualLabelDef.model_validate(payload.new_label)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    with workspace_config_lock(state.workspace):
        cfg = state.get_config()
        if cfg.labels.get(new_lbl.id) is not None:
            raise HTTPException(
                status_code=409,
                detail=f"Label {new_lbl.id!r} already exists in LabelRegistry",
            )

        categories = list(cfg.labels.categories)
        if new_lbl.category not in categories:
            categories.append(new_lbl.category)

        updated_reg = cfg.labels.model_copy(
            update={
                "version": int(cfg.labels.version) + 1,
                "categories": categories,
                "labels": [*cfg.labels.labels, new_lbl],
            }
        )
        yaml_text = yaml.safe_dump(
            updated_reg.model_dump(mode="json"), sort_keys=False, allow_unicode=True
        )
        archive_and_write_override(
            state.workspace,
            state.config_dir,
            Path("labels.yaml"),
            yaml_text,
            reason=f"unknown_promote:{new_lbl.id}",
        )
        state.invalidate_config()
        _record_unknown_triage(
            state.workspace,
            {
                "episode_id": payload.episode_id,
                "seg_id": payload.seg_id,
                "action": "promote",
                "new_label_id": new_lbl.id,
                "label_version": updated_reg.version,
                "triaged_at": time.time(),
            },
        )
    state.sse.publish(
        {
            "type": "config_updated",
            "target": "labels",
            "version": updated_reg.version,
            "promoted_label_id": new_lbl.id,
        }
    )
    return {
        "ok": True,
        "action": "promote",
        "episode_id": payload.episode_id,
        "seg_id": payload.seg_id,
        "label_id": new_lbl.id,
        "label_version": updated_reg.version,
    }


@router.get("/few-shots")
def get_lab_few_shots(
    task_id: str | None = None,
    label_id: str | None = None,
    scope: str | None = None,
    enabled_only: bool = False,
    limit: int = Query(default=200, ge=1, le=1000),
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    items = state.repo.list_few_shots(
        task_id=task_id,
        label_id=label_id,
        enabled_only=enabled_only,
        limit=limit,
    )
    if scope is not None:
        items = [it for it in items if it.scope == scope]
    return {
        "items": [it.model_dump(mode="json") for it in items],
        "total": len(items),
    }


@router.post("/few-shots")
async def post_lab_few_shots(
    payload: FewShotCreatePayload,
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    cfg = state.get_config()
    try:
        item = await run_in_threadpool(
            create_few_shot_example,
            state.repo,
            config=cfg,
            episode_id=payload.episode_id,
            t_start=payload.t_start,
            t_end=payload.t_end,
            scope=payload.scope,
            task_id=payload.task_id,
            label_id=payload.label_id,
            error_type=payload.error_type,
            vlm_wrong_json=payload.vlm_wrong_json,
            human_right_json=payload.human_right_json,
            rationale=payload.rationale,
            capture_tile=payload.capture_tile,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return item.model_dump(mode="json")


@router.patch("/few-shots/{example_id}")
def patch_lab_few_shot(
    example_id: str,
    payload: FewShotPatchPayload,
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    with state.repo.db.transaction() as conn:
        cur = conn.execute(
            "UPDATE few_shot_examples SET enabled = ? WHERE example_id = ?",
            (1 if payload.enabled else 0, example_id),
        )
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail=f"Few-Shot not found: {example_id!r}")
    items = state.repo.list_few_shots(enabled_only=False, limit=2000)
    found = next((it for it in items if it.example_id == example_id), None)
    if found is None:
        raise HTTPException(status_code=404, detail=f"Few-Shot not found: {example_id!r}")
    return found.model_dump(mode="json")


@router.get("/few-shots/{example_id}/tile")
def get_lab_few_shot_tile(
    example_id: str,
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> FileResponse:
    _ = auth
    row = (
        state.repo.db.connect()
        .execute(
            "SELECT tile_image_path FROM few_shot_examples WHERE example_id = ?",
            (example_id,),
        )
        .fetchone()
    )
    if row is None or not row["tile_image_path"]:
        raise HTTPException(status_code=404, detail=f"Few-Shot tile not found: {example_id!r}")
    rel = Path(str(row["tile_image_path"]))
    tiles_root = (state.workspace / "artifacts" / "few_shots").resolve()
    target = (state.workspace / rel).resolve()
    if not target.is_relative_to(tiles_root) or not target.is_file():
        raise HTTPException(status_code=404, detail=f"Few-Shot tile file missing: {example_id!r}")
    return FileResponse(target, media_type="image/jpeg")


@router.post("/eval")
async def post_lab_eval(
    payload: EvalRunPayload,
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    cfg = state.get_config()
    try:
        report = await run_in_threadpool(
            run_gt_eval,
            state.repo,
            config=cfg,
            config_dir=state.config_dir,
            dataset=payload.dataset,
            task_id=payload.task_id,
            episode_ids=payload.episode_ids,
            few_shot=payload.few_shot,
            top_k=payload.top_k,
            max_tokens=payload.max_tokens,
            workers=payload.workers,
            candidate_prompts=payload.candidate_prompts,
            candidate_labels=payload.candidate_labels,
        )
    except (ValueError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except VlmError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return report.model_dump(mode="json")


@router.get("/eval/compare")
def get_lab_eval_compare(
    baseline: str | None = None,
    candidate: str | None = None,
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    reports = list_eval_reports(state.workspace)
    eval_list = [r.model_dump(mode="json") for r in reports]
    if baseline and candidate:
        try:
            cmp_res = compare_eval_runs(state.workspace, baseline, candidate)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"evals": eval_list, **cmp_res}

    if baseline:
        try:
            b_rep = load_eval_report(state.workspace, baseline).model_dump(mode="json")
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"evals": eval_list, "baseline": b_rep, "candidate": None, "deltas": None}

    if candidate:
        try:
            c_rep = load_eval_report(state.workspace, candidate).model_dump(mode="json")
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"evals": eval_list, "baseline": None, "candidate": c_rep, "deltas": None}

    if len(reports) >= 2:
        cmp_res = compare_eval_runs(
            state.workspace, reports[1].eval_id, reports[0].eval_id
        )
        return {"evals": eval_list, **cmp_res}
    return {"evals": eval_list, "baseline": None, "candidate": None, "deltas": None}


@router.post("/promote")
def post_lab_promote(
    payload: PromotePayload,
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    try:
        res = promote_candidate_config(
            state.workspace,
            config_dir=state.config_dir,
            eval_id=payload.eval_id,
            dataset=payload.dataset,
            candidate_prompts=payload.candidate_prompts,
            candidate_labels=payload.candidate_labels,
            candidate_tasks=payload.candidate_tasks,
            reason=payload.reason,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, ValidationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    state.invalidate_config()
    state.get_config()
    state.sse.publish(
        {
            "type": "config_updated",
            "target": "promote",
            "eval_id": payload.eval_id,
            "promoted_files": res.get("promoted_files", []),
        }
    )
    return res


@router.post("/prompt-assist")
async def post_lab_prompt_assist(
    payload: PromptAssistPayload,
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    cfg = state.get_config()
    diag = await run_in_threadpool(
        diagnose_workspace,
        state.repo,
        episode_ids=payload.episode_ids,
        top_n=payload.top_n,
    )
    try:
        suggestion = await run_in_threadpool(
            suggest_prompt_updates,
            workspace=state.workspace,
            config=cfg,
            diagnosis=diag,
            stage=payload.stage,
            task_id=payload.task_id,
        )
    except VlmError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return suggestion


@router.get("/history")
def get_lab_history(
    target: Literal["labels", "prompts", "tasks", "default"] | None = Query(default=None),
    name: str | None = Query(default=None),
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    items = list_history_snapshots(state.workspace, target=target, name=name)
    return {"items": items, "total": len(items)}


@router.get("/history/{snapshot_id}/diff")
def get_lab_history_diff(
    snapshot_id: str,
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    try:
        return diff_history_snapshot(state.workspace, state.config_dir, snapshot_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/history/{snapshot_id}/rollback")
def post_lab_history_rollback(
    snapshot_id: str,
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    try:
        res = rollback_history_snapshot(state.workspace, state.config_dir, snapshot_id)
    except SnapshotValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    state.invalidate_config()
    cfg_after = state.get_config()
    config_sha256 = cfg_after.fingerprints.config_sha256
    state.sse.publish(
        {
            "type": "config_updated",
            "target": res["target"],
            "relative_path": res["relative_path"],
            "reason": "rollback",
            "snapshot_id": snapshot_id,
            "config_sha256": config_sha256,
        }
    )
    return {
        "ok": True,
        "relative_path": res["relative_path"],
        "config_sha256": config_sha256,
    }
