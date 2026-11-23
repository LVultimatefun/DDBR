"""GT regression evaluation sandbox (reusing S1/S2/S5 StageCache) and config promotion."""

from __future__ import annotations

import base64
import binascii
import copy
import difflib
import fcntl
import json
import re
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ..config import PROMPT_STAGE_FILES, ConfigManager, default_config_dir
from ..io.base import validate_id_part
from ..models.annotation import QcSegment, VlmRunArtifact
from ..models.review import EpisodeStatus
from ..models.taxonomy import LabelRegistry, PromptTemplate, TaskSpec
from ..store.artifacts import ArtifactStore, archive_before, atomic_write
from ..store.repository import NotFoundError, Repository
from ..vlm.client import normalize_stage_name
from ..vlm.dag import VlmDag
from .few_shot import open_episode_view, retrieve_few_shots


class EpisodeEvalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    task_id: str
    sandbox_run_id: str
    gt_rev_id: str | None = None
    precision: float
    recall: float
    f1: float
    mean_iou: float
    overcut_rate: float
    guardrail_downgrade_rate: float
    success_correct: bool
    pred_segments: int
    gt_segments: int
    matched_segments: int
    tokens_total: int
    cache_hit_stages: list[str] = Field(default_factory=list)


class EvalMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episodes_evaluated: int = 0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    mean_iou: float = 0.0
    overcut_rate: float = 0.0
    guardrail_downgrade_rate: float = 0.0
    success_accuracy: float = 0.0
    total_tokens: int = 0
    cache_hit_stages: dict[str, int] = Field(default_factory=dict)
    cache_hit_rate: float = 0.0


class EvalRunReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eval_id: str
    created_at: float
    dataset: str | None = None
    task_id: str | None = None
    few_shot: bool = True
    metrics: EvalMetrics
    per_episode: list[EpisodeEvalResult] = Field(default_factory=list)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    candidate_overrides: dict[str, Any] = Field(default_factory=dict)


def _segment_iou(a: QcSegment, b: QcSegment) -> float:
    inter = max(0.0, min(a.t_end, b.t_end) - max(a.t_start, b.t_start))
    if inter <= 0.0:
        return 0.0
    union = max(a.t_end, b.t_end) - min(a.t_start, b.t_start)
    return inter / union if union > 0.0 else 0.0


def _greedy_match_ious(
    pred_segs: Sequence[QcSegment],
    gt_segs: Sequence[QcSegment],
    iou_threshold: float = 0.3,
) -> list[float]:
    cands: list[tuple[float, int, int]] = []
    for pi, ps in enumerate(pred_segs):
        for gi, gs in enumerate(gt_segs):
            iou = _segment_iou(ps, gs)
            if iou >= iou_threshold:
                cands.append((iou, pi, gi))
    cands.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_p: set[int] = set()
    used_g: set[int] = set()
    matched_ious: list[float] = []
    for iou, pi, gi in cands:
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
            matched_ious.append(iou)
    return matched_ious


def _apply_in_memory_overrides(
    cfg: ConfigManager,
    *,
    candidate_prompts: dict[str, Any] | None = None,
    candidate_labels: dict[str, Any] | LabelRegistry | None = None,
    prompt_ver: str | None = None,
    label_ver: str | None = None,
) -> ConfigManager:
    if not candidate_prompts and not candidate_labels and not prompt_ver and not label_ver:
        return cfg

    cloned = copy.deepcopy(cfg)
    if candidate_labels is not None:
        if isinstance(candidate_labels, LabelRegistry):
            cloned.labels = candidate_labels
        else:
            cloned.labels = LabelRegistry.model_validate(candidate_labels)
    if label_ver is not None:
        try:
            ver_num = int(str(label_ver).lstrip("vV"))
            if ver_num >= 1:
                cloned.labels = cloned.labels.model_copy(update={"version": ver_num})
        except ValueError:
            pass

    if candidate_prompts:
        prompt_updates: dict[str, PromptTemplate] = {}
        for raw_stage, raw_val in candidate_prompts.items():
            if not isinstance(raw_val, (dict, PromptTemplate)):
                raise ValueError(
                    f"Invalid candidate_prompts entry for {raw_stage!r}: "
                    f"expected dict, got {type(raw_val).__name__}"
                )
            norm = normalize_stage_name(str(raw_stage))
            if norm not in PROMPT_STAGE_FILES:
                continue
            if isinstance(raw_val, PromptTemplate):
                prompt_updates[norm] = raw_val
            else:
                base_dict = getattr(cloned.prompts, norm).model_dump(mode="json")
                merged = {**base_dict, **raw_val, "stage": norm}
                prompt_updates[norm] = PromptTemplate.model_validate(merged)
        if prompt_updates:
            cloned.prompts = cloned.prompts.model_copy(update=prompt_updates)

    if prompt_ver is not None:
        p_updates: dict[str, PromptTemplate] = {}
        for st_name in PROMPT_STAGE_FILES:
            tpl = getattr(cloned.prompts, st_name)
            p_updates[st_name] = tpl.model_copy(update={"version": str(prompt_ver)})
        cloned.prompts = cloned.prompts.model_copy(update=p_updates)

    cloned.fingerprints = cloned._compute_fingerprints()
    return cloned


def eval_runs_dir(workspace: Path) -> Path:
    d = Path(workspace).resolve() / "artifacts" / "eval_runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_eval_report(workspace: Path, eval_id: str) -> EvalRunReport:
    try:
        eid = validate_id_part(eval_id, "eval_id")
    except ValueError as exc:
        raise NotFoundError(f"Invalid eval_id: {eval_id!r}") from exc
    path = eval_runs_dir(workspace) / f"{eid}.json"
    if not path.is_file():
        raise NotFoundError(f"Eval run not found: {eval_id!r}")
    return EvalRunReport.model_validate_json(path.read_text(encoding="utf-8"))


def list_eval_reports(workspace: Path) -> list[EvalRunReport]:
    d = eval_runs_dir(workspace)
    items: list[EvalRunReport] = []
    for p in sorted(d.glob("*.json")):
        try:
            items.append(EvalRunReport.model_validate_json(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    items.sort(key=lambda r: r.created_at, reverse=True)
    return items


def run_gt_eval(
    workspace_or_repo: Path | str | Repository | None = None,
    *,
    workspace: Path | str | None = None,
    workspace_dir: Path | str | None = None,
    repo: Repository | None = None,
    repository: Repository | None = None,
    config: ConfigManager | None = None,
    cfg: ConfigManager | None = None,
    config_manager: ConfigManager | None = None,
    cfg_mgr: ConfigManager | None = None,
    config_dir: Path | str | None = None,
    configs_dir: Path | str | None = None,
    dataset: str | None = None,
    task_id: str | None = None,
    episode_ids: Sequence[str] | None = None,
    few_shot: bool = True,
    top_k: int | None = None,
    max_tokens: int | None = None,
    workers: int = 4,
    candidate_prompts: dict[str, Any] | None = None,
    candidate_labels: dict[str, Any] | LabelRegistry | None = None,
    prompt_ver: str | None = None,
    label_ver: str | None = None,
    persist_report: bool = True,
    on_progress: Callable[[int, int, str, dict[str, Any]], None] | None = None,
) -> EvalRunReport:
    """Run VlmDag in is_eval_sandbox=True mode over gt_frozen episodes and compute metrics."""
    owns_repo = False
    if repo is not None:
        eff_repo = repo
    elif repository is not None:
        eff_repo = repository
    elif isinstance(workspace_or_repo, Repository):
        eff_repo = workspace_or_repo
    else:
        ws_val = workspace or workspace_dir or workspace_or_repo
        if ws_val is None:
            raise ValueError("workspace or Repository is required")
        eff_repo = Repository(Path(ws_val))
        owns_repo = True

    ws = eff_repo.workspace
    raw_cfg_dir = config_dir or configs_dir
    base_cfg = (
        config
        or cfg
        or config_manager
        or cfg_mgr
        or ConfigManager(
            config_dir=Path(raw_cfg_dir) if raw_cfg_dir is not None else None,
            workspace_dir=ws,
        )
    )
    eff_cfg = _apply_in_memory_overrides(
        base_cfg,
        candidate_prompts=candidate_prompts,
        candidate_labels=candidate_labels,
        prompt_ver=prompt_ver,
        label_ver=label_ver,
    )

    try:
        frozen_rows = eff_repo.list_episodes(
            status=EpisodeStatus.GT_FROZEN,
            dataset=dataset,
            task_id=task_id,
            limit=100000,
        )
        if episode_ids is not None:
            wanted = set(episode_ids)
            frozen_rows = [r for r in frozen_rows if r["episode_id"] in wanted]

        eval_id = f"eval_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        if not frozen_rows:
            report = EvalRunReport(
                eval_id=eval_id,
                created_at=time.time(),
                dataset=dataset,
                task_id=task_id,
                few_shot=bool(few_shot),
                metrics=EvalMetrics(episodes_evaluated=0),
                per_episode=[],
                config_snapshot={
                    "config_sha256": eff_cfg.fingerprints.config_sha256,
                    "label_version": eff_cfg.labels.version,
                    "prompt_bundle_sha256": eff_cfg.fingerprints.prompt_bundle_sha256,
                },
                candidate_overrides={
                    "prompts": candidate_prompts or {},
                    "labels": (
                        candidate_labels.model_dump(mode="json")
                        if isinstance(candidate_labels, LabelRegistry)
                        else (candidate_labels or {})
                    ),
                },
            )
            if persist_report:
                atomic_write(
                    eval_runs_dir(ws) / f"{eval_id}.json",
                    report.model_dump_json(indent=2) + "\n",
                )
            return report

        dag = VlmDag(workspace=ws, config=eff_cfg, repo=eff_repo)
        s3_cfg = eff_cfg.settings.get("vlm", {}).get("s3_discover", {})
        eff_top_k = int(top_k if top_k is not None else s3_cfg.get("few_shot_top_k", 4))
        eff_max_tok = int(
            max_tokens if max_tokens is not None else s3_cfg.get("few_shot_max_tokens", 4000)
        )

        def _eval_one(row: dict[str, Any]) -> tuple[EpisodeEvalResult, VlmRunArtifact]:
            ep_id = str(row["episode_id"])
            resolved_gt = eff_repo.resolve_episode(ep_id)
            with open_episode_view(resolved_gt.meta, eff_cfg) as view:
                t_spec = dag._resolve_task(view, None)
                if few_shot:
                    fs_items = retrieve_few_shots(
                        eff_repo,
                        task_id=t_spec.task_id,
                        top_k=eff_top_k,
                        max_tokens=eff_max_tok,
                    )
                else:
                    fs_items = []
                sandbox_run = dag.run_episode(
                    view,
                    task_spec=t_spec,
                    few_shots=fs_items,
                    is_eval_sandbox=True,
                    persist=True,
                )

            pred_segs = sandbox_run.qc_segments
            gt_segs = resolved_gt.qc_segments
            matched_ious = _greedy_match_ious(pred_segs, gt_segs, iou_threshold=0.3)
            tp = len(matched_ious)
            fp = len(pred_segs) - tp
            fn = len(gt_segs) - tp
            if tp + fp == 0 and tp + fn == 0:
                prec = 1.0
                rec = 1.0
                f1 = 1.0
                mean_iou = 1.0
            else:
                prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if tp + fn == 0 else 0.0)
                rec = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if tp + fp == 0 else 0.0)
                f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                mean_iou = (sum(matched_ious) / len(matched_ious)) if matched_ious else 0.0

            dur = max(float(resolved_gt.meta.duration_s), 1e-6)
            pred_cut_s = sum(
                max(0.0, s.t_end - s.t_start)
                for s in pred_segs
                if s.splice_policy == "cut_clean"
            )
            gt_cut_s = sum(
                max(0.0, s.t_end - s.t_start)
                for s in gt_segs
                if s.splice_policy == "cut_clean"
            )
            overcut_rate = min(1.0, max(0.0, pred_cut_s - gt_cut_s) / dur)
            downgrade_rate = (
                sum(1 for s in pred_segs if s.guardrail.downgraded) / len(pred_segs)
                if pred_segs
                else 0.0
            )
            succ_ok = sandbox_run.success.passed == resolved_gt.success.passed
            hit_stages = [st.stage for st in sandbox_run.usage.stages if st.cache_hit]

            ep_res = EpisodeEvalResult(
                episode_id=ep_id,
                task_id=t_spec.task_id,
                sandbox_run_id=sandbox_run.run_id,
                gt_rev_id=resolved_gt.active_rev_id,
                precision=round(prec, 4),
                recall=round(rec, 4),
                f1=round(f1, 4),
                mean_iou=round(mean_iou, 4),
                overcut_rate=round(overcut_rate, 4),
                guardrail_downgrade_rate=round(downgrade_rate, 4),
                success_correct=succ_ok,
                pred_segments=len(pred_segs),
                gt_segments=len(gt_segs),
                matched_segments=tp,
                tokens_total=int(sandbox_run.usage.total_tokens),
                cache_hit_stages=hit_stages,
            )
            return ep_res, sandbox_run

        def _progress_info(ep_res: EpisodeEvalResult) -> dict[str, Any]:
            return {
                "episode_id": ep_res.episode_id,
                "sandbox_run_id": ep_res.sandbox_run_id,
                "f1": ep_res.f1,
                "precision": ep_res.precision,
                "recall": ep_res.recall,
                "mean_iou": ep_res.mean_iou,
                "success_correct": ep_res.success_correct,
                "tokens_total": ep_res.tokens_total,
                "cache_hit_stages": list(ep_res.cache_hit_stages),
            }

        try:
            total_eps = len(frozen_rows)
            max_w = max(1, min(int(workers), total_eps))
            if max_w == 1:
                outcomes = []
                for idx, r in enumerate(frozen_rows, start=1):
                    ep_res, run_art = _eval_one(r)
                    outcomes.append((ep_res, run_art))
                    if on_progress is not None:
                        on_progress(idx, total_eps, ep_res.episode_id, _progress_info(ep_res))
            else:
                def _eval_worker(row: dict[str, Any]) -> tuple[EpisodeEvalResult, VlmRunArtifact]:
                    try:
                        return _eval_one(row)
                    finally:
                        eff_repo.db.close_thread_connection()

                ordered_outcomes: list[tuple[EpisodeEvalResult, VlmRunArtifact] | None] = [
                    None
                ] * total_eps
                done_cnt = 0
                with ThreadPoolExecutor(max_workers=max_w) as pool:
                    fut_to_idx = {
                        pool.submit(_eval_worker, r): idx for idx, r in enumerate(frozen_rows)
                    }
                    try:
                        for fut in as_completed(fut_to_idx):
                            idx = fut_to_idx[fut]
                            ep_res, run_art = fut.result()
                            ordered_outcomes[idx] = (ep_res, run_art)
                            done_cnt += 1
                            if on_progress is not None:
                                on_progress(
                                    done_cnt,
                                    total_eps,
                                    ep_res.episode_id,
                                    _progress_info(ep_res),
                                )
                    except BaseException:
                        for f in fut_to_idx:
                            f.cancel()
                        raise
                outcomes = [o for o in ordered_outcomes if o is not None]
        finally:
            try:
                dag.client.close()
            except Exception:
                pass

        per_episode = [o[0] for o in outcomes]
        runs = [o[1] for o in outcomes]

        total_tp = sum(e.matched_segments for e in per_episode)
        total_pred = sum(e.pred_segments for e in per_episode)
        total_gt = sum(e.gt_segments for e in per_episode)
        if total_pred == 0 and total_gt == 0:
            agg_prec = 1.0
            agg_rec = 1.0
            agg_f1 = 1.0
        else:
            agg_prec = total_tp / total_pred if total_pred > 0 else 0.0
            agg_rec = total_tp / total_gt if total_gt > 0 else 0.0
            agg_f1 = (
                2.0 * agg_prec * agg_rec / (agg_prec + agg_rec)
                if (agg_prec + agg_rec) > 0
                else 0.0
            )
        agg_iou = sum(e.mean_iou for e in per_episode) / len(per_episode)
        agg_overcut = sum(e.overcut_rate for e in per_episode) / len(per_episode)
        total_downgraded = sum(
            sum(1 for s in r.qc_segments if s.guardrail.downgraded) for r in runs
        )
        agg_downgrade = (total_downgraded / total_pred) if total_pred > 0 else 0.0
        agg_succ = sum(1 for e in per_episode if e.success_correct) / len(per_episode)
        total_tok = sum(e.tokens_total for e in per_episode)

        stage_hits: Counter[str] = Counter()
        total_stage_execs = 0
        total_stage_hits = 0
        for r in runs:
            for st in r.usage.stages:
                total_stage_execs += 1
                if st.cache_hit:
                    total_stage_hits += 1
                    stage_hits[st.stage] += 1

        metrics = EvalMetrics(
            episodes_evaluated=len(per_episode),
            f1=round(agg_f1, 4),
            precision=round(agg_prec, 4),
            recall=round(agg_rec, 4),
            mean_iou=round(agg_iou, 4),
            overcut_rate=round(agg_overcut, 4),
            guardrail_downgrade_rate=round(agg_downgrade, 4),
            success_accuracy=round(agg_succ, 4),
            total_tokens=total_tok,
            cache_hit_stages=dict(stage_hits),
            cache_hit_rate=round(
                total_stage_hits / total_stage_execs if total_stage_execs > 0 else 0.0, 4
            ),
        )
        report = EvalRunReport(
            eval_id=eval_id,
            created_at=time.time(),
            dataset=dataset,
            task_id=task_id,
            few_shot=bool(few_shot),
            metrics=metrics,
            per_episode=per_episode,
            config_snapshot={
                "config_sha256": eff_cfg.fingerprints.config_sha256,
                "label_version": eff_cfg.labels.version,
                "prompt_bundle_sha256": eff_cfg.fingerprints.prompt_bundle_sha256,
            },
            candidate_overrides={
                "prompts": candidate_prompts or {},
                "labels": (
                    candidate_labels.model_dump(mode="json")
                    if isinstance(candidate_labels, LabelRegistry)
                    else (candidate_labels or {})
                ),
            },
        )
        if persist_report:
            atomic_write(
                eval_runs_dir(ws) / f"{eval_id}.json",
                report.model_dump_json(indent=2) + "\n",
            )
        return report
    finally:
        if owns_repo:
            eff_repo.close()


run_eval_sandbox = run_gt_eval


def compare_eval_runs(
    workspace: Path | str,
    baseline_id: str,
    candidate_id: str,
) -> dict[str, Any]:
    """Compare two persisted eval reports and compute quantitative metric deltas."""
    ws = Path(workspace).resolve()
    base = load_eval_report(ws, baseline_id)
    cand = load_eval_report(ws, candidate_id)
    bm = base.metrics
    cm = cand.metrics
    deltas = {
        "delta_f1": round(cm.f1 - bm.f1, 4),
        "delta_precision": round(cm.precision - bm.precision, 4),
        "delta_recall": round(cm.recall - bm.recall, 4),
        "delta_mean_iou": round(cm.mean_iou - bm.mean_iou, 4),
        "delta_overcut_rate": round(cm.overcut_rate - bm.overcut_rate, 4),
        "delta_guardrail_downgrade_rate": round(
            cm.guardrail_downgrade_rate - bm.guardrail_downgrade_rate, 4
        ),
        "delta_success_accuracy": round(cm.success_accuracy - bm.success_accuracy, 4),
        "delta_total_tokens": int(cm.total_tokens - bm.total_tokens),
        "delta_cache_hit_rate": round(cm.cache_hit_rate - bm.cache_hit_rate, 4),
    }
    return {
        "baseline": base.model_dump(mode="json"),
        "candidate": cand.model_dump(mode="json"),
        "deltas": deltas,
    }


_WS_CONFIG_LOCKS_GUARD = threading.Lock()
_WS_CONFIG_LOCKS: dict[Path, threading.RLock] = {}
_WS_CONFIG_LOCK_DEPTH = threading.local()


def _get_ws_config_lock(workspace: Path) -> threading.RLock:
    key = Path(workspace).resolve()
    with _WS_CONFIG_LOCKS_GUARD:
        lock = _WS_CONFIG_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _WS_CONFIG_LOCKS[key] = lock
        return lock


@contextmanager
def workspace_config_lock(workspace: Path) -> Iterator[None]:
    """Serialize config-override & triage read-modify-write across threads and processes."""
    ws = Path(workspace).resolve()
    lock_dir = ws / ".history"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / ".config_override.lock"
    t_lock = _get_ws_config_lock(ws)
    depths: dict[Path, int] = getattr(_WS_CONFIG_LOCK_DEPTH, "depths", None)
    if depths is None:
        depths = {}
        _WS_CONFIG_LOCK_DEPTH.depths = depths
    with t_lock:
        depth = depths.get(ws, 0)
        depths[ws] = depth + 1
        try:
            if depth == 0:
                with open(lock_path, "a+", encoding="utf-8") as fd:
                    fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
                    try:
                        yield
                    finally:
                        fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
            else:
                yield
        finally:
            if depth == 0:
                depths.pop(ws, None)
            else:
                depths[ws] = depth


def archive_and_write_override(
    workspace: Path,
    config_dir: Path,
    rel_path: str | Path,
    content_str: str,
    *,
    reason: str,
    base_candidates: Sequence[Path] = (),
) -> Path:
    """Write configs_override/<rel_path>, archiving base config on very first override."""
    ws = Path(workspace).resolve()
    cfg_d = Path(config_dir).resolve()
    rel = Path(rel_path)
    with workspace_config_lock(ws):
        override_target = ws / "configs_override" / rel
        if not override_target.is_file():
            candidates = list(base_candidates) if base_candidates else [cfg_d / rel]
            for cand in candidates:
                if cand.is_file():
                    archive_before(ws, cand, reason=f"{reason}:initial_base_snapshot")
                    break
        return ArtifactStore(ws).write_config_override(rel, content_str, reason=reason)


def promote_candidate_config(
    workspace_or_repo: Path | str | Repository | None = None,
    *,
    workspace: Path | str | None = None,
    workspace_dir: Path | str | None = None,
    config_dir: Path | str | None = None,
    eval_id: str | None = None,
    dataset: str | None = None,
    candidate_prompts: dict[str, Any] | None = None,
    prompts_override: dict[str, Any] | None = None,
    prompts: dict[str, Any] | None = None,
    candidate_labels: dict[str, Any] | LabelRegistry | None = None,
    candidate_tasks: dict[str, Any] | None = None,
    candidate: dict[str, Any] | None = None,
    candidate_config: dict[str, Any] | None = None,
    reason: str = "eval_promote",
) -> dict[str, Any]:
    """Promote validated candidate prompts/labels/tasks into configs_override/ with .history/."""
    if isinstance(workspace_or_repo, Repository):
        ws = workspace_or_repo.workspace
    else:
        ws_val = workspace or workspace_dir or workspace_or_repo
        if ws_val is None:
            raise ValueError("workspace is required for promote_candidate_config")
        ws = Path(ws_val).resolve()

    cfg_dir = Path(config_dir).resolve() if config_dir is not None else default_config_dir()

    with workspace_config_lock(ws):
        eval_report: EvalRunReport | None = None
        if eval_id is not None:
            eval_report = load_eval_report(ws, eval_id)

        cand_bundle = candidate or candidate_config or {}
        eff_prompts = (
            candidate_prompts
            or prompts_override
            or prompts
            or cand_bundle.get("prompts")
            or (eval_report.candidate_overrides.get("prompts") if eval_report else None)
        )
        eff_labels = (
            candidate_labels
            or cand_bundle.get("labels")
            or (eval_report.candidate_overrides.get("labels") if eval_report else None)
        )
        eff_tasks = candidate_tasks or cand_bundle.get("tasks")

        cfg_now = ConfigManager(config_dir=cfg_dir, workspace_dir=ws)
        promoted_files: list[str] = []

        if eff_prompts:
            for raw_stage, raw_val in eff_prompts.items():
                norm = normalize_stage_name(str(raw_stage))
                if norm not in PROMPT_STAGE_FILES:
                    raise ValueError(f"Unknown prompt stage: {raw_stage!r}")
                base_tpl = getattr(cfg_now.prompts, norm).model_dump(mode="json")
                if isinstance(raw_val, PromptTemplate):
                    tpl = raw_val
                elif isinstance(raw_val, dict):
                    tpl = PromptTemplate.model_validate({**base_tpl, **raw_val, "stage": norm})
                else:
                    raise ValueError(f"Invalid prompt payload for {raw_stage!r}")
                rel = Path("prompts") / f"{norm}.yaml"
                base_cands = [cfg_dir / "prompts" / fn for fn in PROMPT_STAGE_FILES[norm]]
                written = archive_and_write_override(
                    ws,
                    cfg_dir,
                    rel,
                    yaml.safe_dump(tpl.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
                    reason=reason,
                    base_candidates=base_cands,
                )
                promoted_files.append(written.relative_to(ws).as_posix())

        if eff_labels:
            reg = (
                eff_labels
                if isinstance(eff_labels, LabelRegistry)
                else LabelRegistry.model_validate(eff_labels)
            )
            written = archive_and_write_override(
                ws,
                cfg_dir,
                Path("labels.yaml"),
                yaml.safe_dump(reg.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
                reason=reason,
            )
            promoted_files.append(written.relative_to(ws).as_posix())

        if eff_tasks:
            for tid_raw, raw_spec in eff_tasks.items():
                tid = validate_id_part(str(tid_raw), "task_id")
                if isinstance(raw_spec, TaskSpec):
                    spec = raw_spec
                elif isinstance(raw_spec, dict):
                    spec = TaskSpec.model_validate({**raw_spec, "task_id": tid})
                else:
                    raise ValueError(f"Invalid TaskSpec payload for {tid!r}")
                rel = Path("tasks") / f"{tid}.yaml"
                written = archive_and_write_override(
                    ws,
                    cfg_dir,
                    rel,
                    yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
                    reason=reason,
                )
                promoted_files.append(written.relative_to(ws).as_posix())

        # Record active promoted marker in artifacts/promoted_eval.json
        promo_record = {
            "eval_id": eval_id,
            "dataset": dataset,
            "reason": reason,
            "promoted_at": time.time(),
            "promoted_files": promoted_files,
        }
        atomic_write(
            ws / "artifacts" / "promoted_eval.json",
            json.dumps(promo_record, ensure_ascii=False, indent=2) + "\n",
        )

        cfg_after = ConfigManager(config_dir=cfg_dir, workspace_dir=ws)
        return {
            "ok": True,
            "eval_id": eval_id,
            "dataset": dataset,
            "promoted_files": promoted_files,
            "label_version": cfg_after.labels.version,
            "config_sha256": cfg_after.fingerprints.config_sha256,
        }


promote_config = promote_candidate_config


# ---------------------------------------------------------------------------
# .history/ Config Snapshot Listing, Diffing & Atomic Rollback (F3.2a)
# ---------------------------------------------------------------------------

_SNAPSHOT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,512}$")
_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


class SnapshotValidationError(ValueError):
    """Raised when a .history snapshot fails YAML/Pydantic validation before rollback."""


_PROMPT_ALIAS_TO_STAGE: dict[str, str] = {
    fn: stage
    for stage, files in PROMPT_STAGE_FILES.items()
    for fn in files
}


def encode_snapshot_id(history_rel: Path) -> str:
    """Encode a .history/-relative POSIX path into a URL-safe base64 snapshot ID."""
    raw = history_rel.as_posix().encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def resolve_snapshot_path(workspace: Path, snapshot_id: str) -> Path:
    """Decode snapshot_id and verify strict containment inside <workspace>/.history/."""
    if not _SNAPSHOT_ID_RE.match(snapshot_id):
        raise ValueError("Invalid snapshot_id format")
    pad = "=" * (-len(snapshot_id) % 4)
    try:
        rel_str = base64.urlsafe_b64decode((snapshot_id + pad).encode("ascii")).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise FileNotFoundError(f"Snapshot not found: {snapshot_id!r}") from exc
    if not rel_str or "\x00" in rel_str or "\\" in rel_str or rel_str.startswith("/"):
        raise ValueError("Invalid snapshot path")
    rel = Path(rel_str)
    if ".." in rel.parts or rel.name.startswith(".") or rel.suffix != ".json":
        raise ValueError("Invalid snapshot path")
    history_root = (Path(workspace).resolve() / ".history").resolve()
    candidate = (history_root / rel).resolve()
    if not candidate.is_relative_to(history_root) or candidate == history_root:
        raise PermissionError("Snapshot path escapes .history/")
    if not candidate.is_file():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_id!r}")
    return candidate


def _decode_snapshot_text(data: dict[str, Any]) -> str:
    content = data.get("content")
    if not isinstance(content, str):
        raise ValueError("Snapshot missing string 'content'")
    enc = str(data.get("encoding") or "utf-8").lower()
    if enc == "base64":
        return base64.b64decode(content.encode("ascii")).decode("utf-8")
    if enc == "utf-8":
        return content
    raise ValueError(f"Unsupported snapshot encoding: {enc!r}")


def _classify_snapshot(data: dict[str, Any], snap_path: Path) -> tuple[str, str, str]:
    raw_rel = str(data.get("relative_path") or "").replace("\\", "/")
    raw_path = str(data.get("path") or "").replace("\\", "/")
    reason = str(data.get("reason") or "")

    stripped = raw_rel
    for prefix in ("configs_override/", "configs/"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :]
            break

    rel_name = Path(stripped or raw_rel).name
    parent_name = Path(raw_path).parent.name if raw_path else ""

    if stripped == "labels.yaml" or rel_name == "labels.yaml":
        return "labels", "labels", "labels.yaml"
    if stripped == "default.yaml" or rel_name == "default.yaml":
        return "default", "default", "default.yaml"
    if (
        stripped.startswith("prompts/")
        or parent_name == "prompts"
        or rel_name in _PROMPT_ALIAS_TO_STAGE
    ):
        fname = rel_name
        stage = _PROMPT_ALIAS_TO_STAGE.get(fname, Path(fname).stem)
        return "prompts", stage, f"prompts/{stage}.yaml"
    if (
        stripped.startswith("tasks/")
        or parent_name == "tasks"
        or reason.startswith(("lab_tasks_update", "eval_promote", "lab_promote"))
    ):
        tid = Path(stripped or raw_rel).stem
        return "tasks", tid, f"tasks/{tid}.yaml"
    return "other", Path(stripped or snap_path.parent.name).stem, stripped or rel_name


def _read_active_config_text(
    workspace: Path,
    config_dir: Path,
    target: str,
    name: str,
    canonical_rel: str,
) -> str:
    ws = Path(workspace).resolve()
    cfg_d = Path(config_dir).resolve()
    ov_dir = ws / "configs_override"
    if target == "prompts" and name in PROMPT_STAGE_FILES:
        for base_dir in (ov_dir / "prompts", cfg_d / "prompts"):
            for fname in PROMPT_STAGE_FILES[name]:
                cand = base_dir / fname
                if cand.is_file():
                    return cand.read_text(encoding="utf-8")
        return ""
    ov_file = ov_dir / canonical_rel
    if ov_file.is_file():
        return ov_file.read_text(encoding="utf-8")
    base_file = cfg_d / canonical_rel
    if base_file.is_file():
        return base_file.read_text(encoding="utf-8")
    return ""


def _parse_snapshot_json(snap_path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(snap_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("content"), str):
            return None
        return raw
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def list_history_snapshots(
    workspace: Path,
    *,
    target: str | None = None,
    name: str | None = None,
) -> list[dict[str, Any]]:
    """Scan <workspace>/.history/ (both base & override layouts) and return newest-first snapshots."""
    ws = Path(workspace).resolve()
    history_root = (ws / ".history").resolve()
    if not history_root.is_dir():
        return []

    name_clean = name.strip() if name is not None else None
    name_stem = Path(name_clean).stem if name_clean else None
    name_stage: str | None = None
    if name_clean:
        try:
            cand_stage = normalize_stage_name(name_clean)
            if cand_stage in PROMPT_STAGE_FILES:
                name_stage = cand_stage
        except Exception:
            pass

    items: list[dict[str, Any]] = []
    for p in history_root.rglob("*.json"):
        if p.name.startswith("."):
            continue
        try:
            rp = p.resolve()
        except OSError:
            continue
        if not rp.is_relative_to(history_root) or not rp.is_file():
            continue
        data = _parse_snapshot_json(rp)
        if data is None:
            continue
        snap_target, snap_name, canonical_rel = _classify_snapshot(data, rp)
        if target is not None and snap_target != target:
            continue
        if name_clean:
            if not (
                snap_name == name_clean
                or (name_stem is not None and snap_name == name_stem)
                or (name_stage is not None and snap_name == name_stage)
                or canonical_rel == name_clean
            ):
                continue
        st = rp.stat()
        created_at = float(data.get("created_at") or st.st_mtime)
        reason = str(data.get("reason") or "")
        items.append(
            {
                "snapshot_id": encode_snapshot_id(rp.relative_to(history_root)),
                "target": snap_target,
                "name": snap_name,
                "relative_path": canonical_rel,
                "created_at": created_at,
                "reason": reason,
                "size_bytes": int(st.st_size),
            }
        )

    items.sort(key=lambda it: (it["created_at"], it["snapshot_id"]), reverse=True)
    return items


def diff_history_snapshot(
    workspace: Path,
    config_dir: Path,
    snapshot_id: str,
) -> dict[str, Any]:
    """Compute unified_diff and structured lines between a .history snapshot and active config."""
    snap_path = resolve_snapshot_path(workspace, snapshot_id)
    data = _parse_snapshot_json(snap_path)
    if data is None:
        raise ValueError(f"Corrupt or unreadable snapshot: {snapshot_id!r}")
    snap_text = _decode_snapshot_text(data)
    target, name, canonical_rel = _classify_snapshot(data, snap_path)
    active_text = _read_active_config_text(
        workspace, config_dir, target, name, canonical_rel
    )

    diff_iter = list(
        difflib.unified_diff(
            snap_text.splitlines(),
            active_text.splitlines(),
            fromfile=f"snapshot/{canonical_rel}",
            tofile=f"active/{canonical_rel}",
            lineterm="",
        )
    )
    unified_diff = "\n".join(diff_iter)
    lines: list[dict[str, Any]] = []
    old_ln = 0
    new_ln = 0
    for ln in diff_iter:
        if ln.startswith("--- ") or ln.startswith("+++ "):
            continue
        m = _HUNK_HEADER_RE.match(ln)
        if m:
            old_ln = int(m.group(1))
            new_ln = int(m.group(2))
            lines.append(
                {"type": "hunk", "old_line": None, "new_line": None, "text": ln}
            )
        elif ln.startswith(" "):
            lines.append(
                {
                    "type": "context",
                    "old_line": old_ln,
                    "new_line": new_ln,
                    "text": ln[1:],
                }
            )
            old_ln += 1
            new_ln += 1
        elif ln.startswith("-"):
            lines.append(
                {
                    "type": "remove",
                    "old_line": old_ln,
                    "new_line": None,
                    "text": ln[1:],
                }
            )
            old_ln += 1
        elif ln.startswith("+"):
            lines.append(
                {
                    "type": "add",
                    "old_line": None,
                    "new_line": new_ln,
                    "text": ln[1:],
                }
            )
            new_ln += 1

    st = snap_path.stat()
    return {
        "snapshot_id": snapshot_id,
        "relative_path": canonical_rel,
        "created_at": float(data.get("created_at") or st.st_mtime),
        "reason": str(data.get("reason") or ""),
        "unified_diff": unified_diff,
        "lines": lines,
    }


def rollback_history_snapshot(
    workspace: Path,
    config_dir: Path,
    snapshot_id: str,
) -> dict[str, Any]:
    """Validate snapshot content, archive active config, and atomically restore override."""
    ws = Path(workspace).resolve()
    cfg_d = Path(config_dir).resolve()
    with workspace_config_lock(ws):
        snap_path = resolve_snapshot_path(ws, snapshot_id)
        data = _parse_snapshot_json(snap_path)
        if data is None:
            raise ValueError(f"Corrupt or unreadable snapshot: {snapshot_id!r}")
        restored_text = _decode_snapshot_text(data)
        target, name, canonical_rel = _classify_snapshot(data, snap_path)

        try:
            parsed = yaml.safe_load(restored_text)
        except yaml.YAMLError as exc:
            raise SnapshotValidationError(f"Snapshot YAML is invalid: {exc}") from exc

        if parsed is None and target in ("default", "tasks"):
            parsed = {}
        if not isinstance(parsed, dict):
            raise SnapshotValidationError(
                f"Snapshot must parse to a YAML mapping, got {type(parsed).__name__}"
            )

        try:
            if target == "labels":
                LabelRegistry.model_validate(parsed)
            elif target == "prompts":
                PromptTemplate.model_validate({**parsed, "stage": name})
            elif target == "tasks":
                tid = validate_id_part(name, "task_id")
                TaskSpec.model_validate({**parsed, "task_id": tid})
            elif target == "default":
                pass
            else:
                raise SnapshotValidationError(f"Unsupported rollback target {target!r}")
        except ValueError as exc:
            raise SnapshotValidationError(str(exc)) from exc

        base_cands = (
            [cfg_d / "prompts" / fn for fn in PROMPT_STAGE_FILES[name]]
            if target == "prompts" and name in PROMPT_STAGE_FILES
            else ()
        )
        archive_and_write_override(
            ws,
            cfg_d,
            Path(canonical_rel),
            restored_text,
            reason=f"rollback:{snapshot_id}",
            base_candidates=base_cands,
        )
        cfg_after = ConfigManager(config_dir=cfg_d, workspace_dir=ws)
        return {
            "ok": True,
            "target": target,
            "name": name,
            "relative_path": canonical_rel,
            "config_sha256": cfg_after.fingerprints.config_sha256,
        }
