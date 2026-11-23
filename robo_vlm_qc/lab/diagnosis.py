"""Automated VLM Raw vs Human GT error diagnosis across 6 structured buckets."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field

from ..models.annotation import MacroStage, QcSegment
from ..models.taxonomy import DiagnosisErrorType
from ..store.repository import NotFoundError, Repository


class DiagnosisErrorItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error_id: str
    episode_id: str
    task_id: str
    error_type: DiagnosisErrorType
    label_id: str | None = None
    t_start: float | None = None
    t_end: float | None = None
    vlm_seg_id: str | None = None
    gt_seg_id: str | None = None
    vlm_value: Any = None
    gt_value: Any = None
    iou: float | None = None
    severity: float = 1.0
    summary: str = ""


class DiagnosisCluster(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_key: str
    error_type: DiagnosisErrorType
    label_id: str | None = None
    task_id: str | None = None
    count: int
    episode_ids: list[str] = Field(default_factory=list)
    sample_summary: str = ""


class DiagnosisReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episodes_evaluated: int = 0
    total_errors: int = 0
    counts_by_type: dict[str, int] = Field(default_factory=dict)
    counts_by_label: dict[str, int] = Field(default_factory=dict)
    counts_by_task: dict[str, int] = Field(default_factory=dict)
    errors: list[DiagnosisErrorItem] = Field(default_factory=list)
    top_errors: list[DiagnosisErrorItem] = Field(default_factory=list)
    clusters: list[DiagnosisCluster] = Field(default_factory=list)


def _segment_iou(a: QcSegment, b: QcSegment) -> float:
    inter = max(0.0, min(a.t_end, b.t_end) - max(a.t_start, b.t_start))
    if inter <= 0.0:
        return 0.0
    union = max(a.t_end, b.t_end) - min(a.t_start, b.t_start)
    return inter / union if union > 0.0 else 0.0


def _macro_stages_differ(
    v_stages: Sequence[MacroStage],
    g_stages: Sequence[MacroStage],
    boundary_tol_s: float,
) -> tuple[bool, str]:
    if len(v_stages) != len(g_stages):
        return True, f"Macro stage count {len(v_stages)} != GT {len(g_stages)}"
    tol = boundary_tol_s + 1e-9
    for idx, (vs, gs) in enumerate(zip(v_stages, g_stages)):
        if (
            vs.label != gs.label
            or abs(vs.t_start - gs.t_start) > tol
            or abs(vs.t_end - gs.t_end) > tol
        ):
            return (
                True,
                f"Stage {idx + 1} mismatch: VLM '{vs.label}' [{vs.t_start:.2f},{vs.t_end:.2f}] "
                f"vs GT '{gs.label}' [{gs.t_start:.2f},{gs.t_end:.2f}]",
            )
    return False, ""


def _resolve_repo(
    repo_or_workspace: Repository | Path | str | None,
    *,
    repo: Repository | None = None,
    repository: Repository | None = None,
    workspace: Path | str | None = None,
    workspace_dir: Path | str | None = None,
) -> tuple[Repository, bool]:
    if repo is not None:
        return repo, False
    if repository is not None:
        return repository, False
    if isinstance(repo_or_workspace, Repository):
        return repo_or_workspace, False
    ws = workspace or workspace_dir or repo_or_workspace
    if ws is None:
        raise ValueError("Repository or workspace path is required for diagnosis")
    return Repository(Path(ws)), True


def diagnose_workspace(
    repo_or_workspace: Repository | Path | str | None = None,
    *,
    repo: Repository | None = None,
    repository: Repository | None = None,
    workspace: Path | str | None = None,
    workspace_dir: Path | str | None = None,
    episode_ids: Sequence[str] | None = None,
    top_n: int = 20,
    iou_threshold: float = 0.3,
    boundary_tol_s: float = 0.05,
) -> DiagnosisReport:
    """Compare active human reviews against their base VLM runs across 6 error buckets."""
    eff_repo, owns_repo = _resolve_repo(
        repo_or_workspace,
        repo=repo,
        repository=repository,
        workspace=workspace,
        workspace_dir=workspace_dir,
    )
    try:
        target_rows: list[dict[str, Any]] = []
        if episode_ids is not None:
            for ep_id in episode_ids:
                try:
                    row = eff_repo.get_episode_row(ep_id)
                except NotFoundError:
                    continue
                if row.get("active_rev_id"):
                    target_rows.append(row)
        else:
            all_rows = eff_repo.list_episodes(limit=100000)
            target_rows = [r for r in all_rows if r.get("active_rev_id")]

        errors: list[DiagnosisErrorItem] = []
        episodes_evaluated = 0

        for row in target_rows:
            ep_id = str(row["episode_id"])
            task_id = str(row["task_id"])
            rev_id = str(row["active_rev_id"])
            try:
                review = eff_repo.get_review(rev_id)
                base_run = eff_repo.get_vlm_run(review.base_run_id)
            except (NotFoundError, FileNotFoundError):
                continue

            episodes_evaluated += 1
            vlm_segs = base_run.qc_segments
            gt_segs = review.qc_segments

            # Greedy best-IoU pairing (highest IoU first, >= iou_threshold)
            candidates: list[tuple[float, int, int]] = []
            for vi, vs in enumerate(vlm_segs):
                for gi, gs in enumerate(gt_segs):
                    iou = _segment_iou(vs, gs)
                    if iou >= iou_threshold:
                        candidates.append((iou, vi, gi))
            candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

            matched_v: set[int] = set()
            matched_g: set[int] = set()
            paired: list[tuple[float, QcSegment, QcSegment]] = []
            for iou, vi, gi in candidates:
                if vi not in matched_v and gi not in matched_g:
                    matched_v.add(vi)
                    matched_g.add(gi)
                    paired.append((iou, vlm_segs[vi], gt_segs[gi]))

            # 1. fp_overflag: VLM flagged segment absent in GT
            for vi, vs in enumerate(vlm_segs):
                if vi in matched_v:
                    continue
                errors.append(
                    DiagnosisErrorItem(
                        error_id=f"{ep_id}::fp::{vs.seg_id}",
                        episode_id=ep_id,
                        task_id=task_id,
                        error_type="fp_overflag",
                        label_id=vs.label_id,
                        t_start=vs.t_start,
                        t_end=vs.t_end,
                        vlm_seg_id=vs.seg_id,
                        vlm_value={
                            "label_id": vs.label_id,
                            "splice_policy": vs.splice_policy,
                            "t_start": vs.t_start,
                            "t_end": vs.t_end,
                        },
                        gt_value=None,
                        severity=max(0.5, min(2.0, vs.t_end - vs.t_start)),
                        summary=(
                            f"VLM over-flagged {vs.label_id} ({vs.splice_policy}) "
                            f"at [{vs.t_start:.2f}s, {vs.t_end:.2f}s] not present in GT"
                        ),
                    )
                )

            # 2. fn_missed: GT segment missed by VLM
            for gi, gs in enumerate(gt_segs):
                if gi in matched_g:
                    continue
                errors.append(
                    DiagnosisErrorItem(
                        error_id=f"{ep_id}::fn::{gs.seg_id}",
                        episode_id=ep_id,
                        task_id=task_id,
                        error_type="fn_missed",
                        label_id=gs.label_id,
                        t_start=gs.t_start,
                        t_end=gs.t_end,
                        gt_seg_id=gs.seg_id,
                        vlm_value=None,
                        gt_value={
                            "label_id": gs.label_id,
                            "splice_policy": gs.splice_policy,
                            "t_start": gs.t_start,
                            "t_end": gs.t_end,
                        },
                        severity=max(0.5, min(2.0, gs.t_end - gs.t_start)),
                        summary=(
                            f"VLM missed GT {gs.label_id} ({gs.splice_policy}) "
                            f"at [{gs.t_start:.2f}s, {gs.t_end:.2f}s]"
                        ),
                    )
                )

            # 3. policy_flip & 4. boundary on matched pairs
            for iou, vs, gs in paired:
                if vs.splice_policy != gs.splice_policy or vs.label_id != gs.label_id:
                    errors.append(
                        DiagnosisErrorItem(
                            error_id=f"{ep_id}::pflip::{vs.seg_id}::{gs.seg_id}",
                            episode_id=ep_id,
                            task_id=task_id,
                            error_type="policy_flip",
                            label_id=gs.label_id,
                            t_start=gs.t_start,
                            t_end=gs.t_end,
                            vlm_seg_id=vs.seg_id,
                            gt_seg_id=gs.seg_id,
                            vlm_value={
                                "label_id": vs.label_id,
                                "splice_policy": vs.splice_policy,
                            },
                            gt_value={
                                "label_id": gs.label_id,
                                "splice_policy": gs.splice_policy,
                            },
                            iou=round(iou, 4),
                            severity=1.5 if vs.splice_policy != gs.splice_policy else 1.0,
                            summary=(
                                f"VLM {vs.label_id}/{vs.splice_policy} -> "
                                f"GT {gs.label_id}/{gs.splice_policy} "
                                f"at [{gs.t_start:.2f}s, {gs.t_end:.2f}s]"
                            ),
                        )
                    )

                dt_start = abs(vs.t_start - gs.t_start)
                dt_end = abs(vs.t_end - gs.t_end)
                if dt_start > boundary_tol_s + 1e-9 or dt_end > boundary_tol_s + 1e-9:
                    drift = max(dt_start, dt_end)
                    errors.append(
                        DiagnosisErrorItem(
                            error_id=f"{ep_id}::bnd::{vs.seg_id}::{gs.seg_id}",
                            episode_id=ep_id,
                            task_id=task_id,
                            error_type="boundary",
                            label_id=gs.label_id,
                            t_start=gs.t_start,
                            t_end=gs.t_end,
                            vlm_seg_id=vs.seg_id,
                            gt_seg_id=gs.seg_id,
                            vlm_value={"t_start": vs.t_start, "t_end": vs.t_end},
                            gt_value={"t_start": gs.t_start, "t_end": gs.t_end},
                            iou=round(iou, 4),
                            severity=min(2.0, 0.5 + drift),
                            summary=(
                                f"Boundary drift {drift:.2f}s on {gs.label_id}: "
                                f"VLM [{vs.t_start:.2f}, {vs.t_end:.2f}] vs GT [{gs.t_start:.2f}, {gs.t_end:.2f}]"
                            ),
                        )
                    )

            # 5. success_flip
            if (
                base_run.success.passed != review.success.passed
                or base_run.success.verdict != review.success.verdict
            ):
                errors.append(
                    DiagnosisErrorItem(
                        error_id=f"{ep_id}::success",
                        episode_id=ep_id,
                        task_id=task_id,
                        error_type="success_flip",
                        vlm_value={
                            "passed": base_run.success.passed,
                            "verdict": base_run.success.verdict,
                        },
                        gt_value={
                            "passed": review.success.passed,
                            "verdict": review.success.verdict,
                        },
                        severity=2.0,
                        summary=(
                            f"Success flip: VLM {base_run.success.verdict} (passed={base_run.success.passed}) "
                            f"vs GT {review.success.verdict} (passed={review.success.passed})"
                        ),
                    )
                )

            # 6. stage (macro_stages mismatch)
            stage_diff, stage_msg = _macro_stages_differ(
                base_run.macro_stages, review.macro_stages, boundary_tol_s
            )
            if stage_diff:
                errors.append(
                    DiagnosisErrorItem(
                        error_id=f"{ep_id}::stage",
                        episode_id=ep_id,
                        task_id=task_id,
                        error_type="stage",
                        vlm_value=[s.model_dump(mode="json") for s in base_run.macro_stages],
                        gt_value=[s.model_dump(mode="json") for s in review.macro_stages],
                        severity=1.0,
                        summary=stage_msg,
                    )
                )

        counts_by_type: Counter[str] = Counter()
        counts_by_label: Counter[str] = Counter()
        counts_by_task: Counter[str] = Counter()
        cluster_items: dict[tuple[str, str, str], list[DiagnosisErrorItem]] = defaultdict(list)

        for err in errors:
            counts_by_type[err.error_type] += 1
            if err.label_id:
                counts_by_label[err.label_id] += 1
            counts_by_task[err.task_id] += 1
            cluster_items[(err.error_type, err.label_id or "", err.task_id)].append(err)

        clusters: list[DiagnosisCluster] = []
        for (etype, lid, tid), items in cluster_items.items():
            ep_list = sorted({it.episode_id for it in items})
            clusters.append(
                DiagnosisCluster(
                    cluster_key=f"{etype}:{lid or '*'}:{tid}",
                    error_type=etype,  # type: ignore[arg-type]
                    label_id=lid or None,
                    task_id=tid,
                    count=len(items),
                    episode_ids=ep_list,
                    sample_summary=items[0].summary,
                )
            )
        clusters.sort(key=lambda c: (-c.count, c.error_type, c.label_id or "", c.task_id or ""))

        ranked_errors = sorted(
            errors,
            key=lambda e: (
                -len(cluster_items[(e.error_type, e.label_id or "", e.task_id)]),
                -e.severity,
                e.episode_id,
                e.t_start if e.t_start is not None else -1.0,
                e.error_id,
            ),
        )
        eff_top_n = max(0, int(top_n))
        top_errors = ranked_errors[:eff_top_n]

        return DiagnosisReport(
            episodes_evaluated=episodes_evaluated,
            total_errors=len(errors),
            counts_by_type=dict(counts_by_type),
            counts_by_label=dict(counts_by_label),
            counts_by_task=dict(counts_by_task),
            errors=errors,
            top_errors=top_errors,
            clusters=clusters,
        )
    finally:
        if owns_repo:
            eff_repo.close()


run_diagnosis = diagnose_workspace
compute_diagnosis = diagnose_workspace
diagnose = diagnose_workspace


class DiagnosisEngine:
    """Object-oriented wrapper around diagnose_workspace."""

    def __init__(
        self,
        repo_or_workspace: Repository | Path | str | None = None,
        *,
        repo: Repository | None = None,
        repository: Repository | None = None,
        workspace: Path | str | None = None,
        workspace_dir: Path | str | None = None,
    ) -> None:
        self._target = repo or repository or workspace or workspace_dir or repo_or_workspace

    def diagnose(
        self,
        episode_ids: Sequence[str] | None = None,
        top_n: int = 20,
    ) -> DiagnosisReport:
        return diagnose_workspace(self._target, episode_ids=episode_ids, top_n=top_n)

    run = diagnose
    compute = diagnose


ErrorDiagnoser = DiagnosisEngine
DiagnosisRunner = DiagnosisEngine
