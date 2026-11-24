"""Domain repository: episode registry, state machine, lease locks, and ResolvedEpisode."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from ..models.annotation import (
    ArmTimelines,
    GroundingResult,
    MacroStage,
    QcSegment,
    SuccessVerdict,
    VlmRunArtifact,
)
from ..models.episode import EpisodeMeta
from ..models.taxonomy import FewShotItem
from ..models.review import (
    EpisodeStatus,
    HumanReviewArtifact,
    ResolvedEpisode,
    ReviewDiff,
    ReviewRole,
    ReviewWritePayload,
)
from .artifacts import ArtifactStore
from .db import DatabasePool


class RepositoryError(Exception):
    """Base domain repository error."""


class NotFoundError(RepositoryError):
    """Requested episode, VLM run, or review does not exist."""


class InvalidTransitionError(RepositoryError):
    """Disallowed episode status transition."""


class ClaimConflictError(RepositoryError):
    """Episode is currently claimed by another active reviewer."""


class GtFrozenError(RepositoryError):
    """Episode is frozen as GT and cannot be modified."""


class WatchCoverageError(RepositoryError):
    """Reviewer did not meet minimum watch coverage before submitting."""


class AuthorizationError(RepositoryError):
    """Role does not have permission for this action."""


class ReadOnlyFieldError(RepositoryError):
    """Attempted to mutate a read-only VLM field."""


ALLOWED_TRANSITIONS: dict[EpisodeStatus, set[EpisodeStatus]] = {
    EpisodeStatus.INGESTED: {
        EpisodeStatus.VLM_QUEUED,
        EpisodeStatus.VLM_RUNNING,
        EpisodeStatus.VLM_READY,
    },
    EpisodeStatus.VLM_QUEUED: {
        EpisodeStatus.VLM_RUNNING,
        EpisodeStatus.VLM_READY,
        EpisodeStatus.INGESTED,
    },
    EpisodeStatus.VLM_RUNNING: {
        EpisodeStatus.VLM_READY,
        EpisodeStatus.VLM_QUEUED,
        EpisodeStatus.INGESTED,
    },
    EpisodeStatus.VLM_READY: {
        EpisodeStatus.VLM_QUEUED,
        EpisodeStatus.VLM_RUNNING,
        EpisodeStatus.CLAIMED,
        EpisodeStatus.APPROVED,
    },
    EpisodeStatus.CLAIMED: {
        EpisodeStatus.VLM_READY,
        EpisodeStatus.REVIEWED,
        EpisodeStatus.FLAGGED,
        EpisodeStatus.APPROVED,
    },
    EpisodeStatus.REVIEWED: {
        EpisodeStatus.CLAIMED,
        EpisodeStatus.APPROVED,
        EpisodeStatus.FLAGGED,
        EpisodeStatus.VLM_QUEUED,
    },
    EpisodeStatus.FLAGGED: {
        EpisodeStatus.CLAIMED,
        EpisodeStatus.APPROVED,
        EpisodeStatus.VLM_QUEUED,
    },
    EpisodeStatus.APPROVED: {
        EpisodeStatus.GT_FROZEN,
        EpisodeStatus.CLAIMED,
        EpisodeStatus.REVIEWED,
    },
    EpisodeStatus.GT_FROZEN: set(),
}

_PROTECTED_REVIEW_STATUSES = {
    EpisodeStatus.CLAIMED.value,
    EpisodeStatus.REVIEWED.value,
    EpisodeStatus.FLAGGED.value,
    EpisodeStatus.APPROVED.value,
    EpisodeStatus.GT_FROZEN.value,
}


def _segment_iou(a: QcSegment, b: QcSegment) -> float:
    inter = max(0.0, min(a.t_end, b.t_end) - max(a.t_start, b.t_start))
    if inter <= 0.0:
        return 0.0
    union = max(a.t_end, b.t_end) - min(a.t_start, b.t_start)
    return inter / union if union > 0.0 else 0.0


def compute_review_diff(
    base_run: VlmRunArtifact,
    qc_segments: list[QcSegment],
    success: SuccessVerdict,
    macro_stages: list[MacroStage],
) -> ReviewDiff:
    """Compute structured differences between VLM raw run and human review."""
    vlm_segs = base_run.qc_segments
    hum_segs = qc_segments

    candidates: list[tuple[float, int, int]] = []
    for vi, vs in enumerate(vlm_segs):
        for hi, hs in enumerate(hum_segs):
            iou = _segment_iou(vs, hs)
            if iou >= 0.5:
                candidates.append((iou, vi, hi))
    candidates.sort(reverse=True)

    matched_v: set[int] = set()
    matched_h: set[int] = set()
    pairs: list[tuple[QcSegment, QcSegment]] = []
    for _, vi, hi in candidates:
        if vi not in matched_v and hi not in matched_h:
            matched_v.add(vi)
            matched_h.add(hi)
            pairs.append((vlm_segs[vi], hum_segs[hi]))

    fp_deleted = len(vlm_segs) - len(matched_v)
    fn_added = len(hum_segs) - len(matched_h)
    policy_flips = sum(1 for v, h in pairs if v.splice_policy != h.splice_policy)
    label_changes = sum(1 for v, h in pairs if v.label_id != h.label_id)
    boundary_nudges = sum(
        1
        for v, h in pairs
        if abs(v.t_start - h.t_start) > 0.05 + 1e-9
        or abs(v.t_end - h.t_end) > 0.05 + 1e-9
    )

    v_stages = base_run.macro_stages
    macro_stage_edits = abs(len(v_stages) - len(macro_stages))
    for vs, hs in zip(v_stages, macro_stages):
        if (
            vs.label != hs.label
            or abs(vs.t_start - hs.t_start) > 0.05 + 1e-9
            or abs(vs.t_end - hs.t_end) > 0.05 + 1e-9
        ):
            macro_stage_edits += 1

    success_flipped = (
        base_run.success.passed != success.passed
        or base_run.success.verdict != success.verdict
    )
    total_edits = (
        fp_deleted
        + fn_added
        + policy_flips
        + label_changes
        + boundary_nudges
        + macro_stage_edits
        + (1 if success_flipped else 0)
    )
    return ReviewDiff(
        fp_deleted=fp_deleted,
        fn_added=fn_added,
        policy_flips=policy_flips,
        label_changes=label_changes,
        boundary_nudges=boundary_nudges,
        macro_stage_edits=macro_stage_edits,
        success_flipped=success_flipped,
        total_edits=total_edits,
    )


def _compute_episode_metrics(
    duration_s: float,
    qc_segments: list[QcSegment],
    success: SuccessVerdict,
) -> dict[str, Any]:
    dur = max(duration_s, 1e-6)
    cut_s = sum(
        max(0.0, s.t_end - s.t_start)
        for s in qc_segments
        if s.splice_policy == "cut_clean"
    )
    mask_s = sum(
        max(0.0, s.t_end - s.t_start)
        for s in qc_segments
        if s.splice_policy == "mask_loss"
    )
    badcase_count = sum(
        1
        for s in qc_segments
        if s.training_role in {"badcase", "terminal_fail"}
    )
    unknown_count = sum(
        1 for s in qc_segments if s.is_unknown or s.label_id == "unknown_anomaly"
    )
    passed_int: int | None = None
    if success.passed is True:
        passed_int = 1
    elif success.passed is False:
        passed_int = 0
    return {
        "resolved_passed": passed_int,
        "cut_ratio": min(1.0, cut_s / dur),
        "mask_ratio": min(1.0, mask_s / dur),
        "badcase_count": badcase_count,
        "unknown_count": unknown_count,
    }


class Repository:
    """High-level persistence facade over SQLite WAL + ArtifactStore."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.db = DatabasePool(self.workspace / "robo_vlm_qc.db")
        self.artifacts = ArtifactStore(self.workspace)

    def close(self) -> None:
        self.db.close_all()

    def ingest_episode(
        self, meta: EpisodeMeta
    ) -> tuple[str, Literal["inserted", "updated", "skipped"]]:
        now = time.time()
        cam_roles_json = json.dumps(meta.camera_roles, ensure_ascii=False)
        meta_json = meta.model_dump_json()

        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT source_mtime, source_uri, status FROM episodes WHERE episode_id = ?",
                (meta.episode_id,),
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO episodes (
                        episode_id, dataset, task_id, adapter_type, source_uri,
                        source_mtime, duration_s, fps, has_kinematics,
                        camera_roles_json, meta_json, status, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        meta.episode_id,
                        meta.dataset,
                        meta.task_id,
                        meta.adapter_type,
                        meta.source_uri,
                        meta.source_mtime,
                        meta.duration_s,
                        meta.fps,
                        1 if meta.kinematics.has_kinematics else 0,
                        cam_roles_json,
                        meta_json,
                        EpisodeStatus.INGESTED.value,
                        now,
                    ),
                )
                return meta.episode_id, "inserted"

            if (
                abs(float(row["source_mtime"]) - meta.source_mtime) < 1e-6
                and row["source_uri"] == meta.source_uri
            ):
                return meta.episode_id, "skipped"

            if row["status"] == EpisodeStatus.GT_FROZEN.value:
                raise GtFrozenError(
                    f"Episode {meta.episode_id!r} is gt_frozen; refusing re-ingest overwrite"
                )

            conn.execute(
                """
                UPDATE episodes
                   SET task_id = ?,
                       adapter_type = ?,
                       source_uri = ?,
                       source_mtime = ?,
                       duration_s = ?,
                       fps = ?,
                       has_kinematics = ?,
                       camera_roles_json = ?,
                       meta_json = ?,
                       updated_at = ?
                 WHERE episode_id = ?
                """,
                (
                    meta.task_id,
                    meta.adapter_type,
                    meta.source_uri,
                    meta.source_mtime,
                    meta.duration_s,
                    meta.fps,
                    1 if meta.kinematics.has_kinematics else 0,
                    cam_roles_json,
                    meta_json,
                    now,
                    meta.episode_id,
                ),
            )
            return meta.episode_id, "updated"

    def get_episode_row(self, episode_id: str) -> dict[str, Any]:
        conn = self.db.connect()
        row = conn.execute(
            "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        if row is None:
            raise NotFoundError(f"Episode not found: {episode_id!r}")
        return dict(row)

    def get_episode_meta(self, episode_id: str) -> EpisodeMeta:
        row = self.get_episode_row(episode_id)
        return EpisodeMeta.model_validate_json(row["meta_json"])

    def episode_guardrail_downgraded(self, row: dict[str, Any]) -> bool:
        if not row.get("active_run_id"):
            return False
        try:
            if row.get("active_rev_id"):
                rev = self.get_review(str(row["active_rev_id"]))
                return any(s.guardrail.downgraded for s in rev.qc_segments)
            run = self.get_vlm_run(str(row["active_run_id"]))
            return any(s.guardrail.downgraded for s in run.qc_segments)
        except NotFoundError:
            return False

    @staticmethod
    def _normalize_status_values(
        status: EpisodeStatus | str | Sequence[EpisodeStatus | str] | None,
    ) -> list[str] | None:
        if status is None:
            return None
        raw_items: list[EpisodeStatus | str]
        if isinstance(status, EpisodeStatus):
            raw_items = [status]
        elif isinstance(status, str):
            stripped = status.strip()
            if not stripped:
                raise ValueError("Empty status filter")
            raw_items = [part.strip() for part in stripped.split(",")]
        else:
            raw_items = list(status)
            if not raw_items:
                raise ValueError("Empty status filter")
        out: list[str] = []
        seen: set[str] = set()
        for item in raw_items:
            if isinstance(item, EpisodeStatus):
                val = item.value
            else:
                s = str(item).strip()
                if not s:
                    raise ValueError("Empty status value in status filter")
                val = EpisodeStatus(s).value
            if val not in seen:
                seen.add(val)
                out.append(val)
        return out

    @staticmethod
    def _escape_like(value: str) -> str:
        return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    _SORT_COLUMN_MAP: dict[str, str] = {
        "episode_id": "episode_id",
        "updated_at": "updated_at",
        "duration_s": "duration_s",
        "cut_ratio": "cut_ratio",
        "mask_ratio": "mask_ratio",
        "badcase_count": "badcase_count",
    }
    _SORT_DIR_MAP: dict[str, str] = {
        "asc": "ASC",
        "desc": "DESC",
    }

    def _build_episode_where(
        self,
        *,
        status: EpisodeStatus | str | Sequence[EpisodeStatus | str] | None = None,
        dataset: str | None = None,
        task_id: str | None = None,
        adapter_type: str | None = None,
        passed: bool | None = None,
        has_unknown: bool | None = None,
        claimed_by: str | None = None,
        search: str | None = None,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        status_values = self._normalize_status_values(status)
        if status_values is not None:
            if len(status_values) == 1:
                clauses.append("status = ?")
                params.append(status_values[0])
            else:
                placeholders = ", ".join("?" for _ in status_values)
                clauses.append(f"status IN ({placeholders})")
                params.extend(status_values)
        if dataset is not None:
            clauses.append("dataset = ?")
            params.append(dataset)
        if task_id is not None:
            clauses.append("task_id = ?")
            params.append(task_id)
        if adapter_type is not None:
            clauses.append("adapter_type = ?")
            params.append(adapter_type)
        if passed is not None:
            clauses.append("resolved_passed = ?")
            params.append(1 if passed else 0)
        if has_unknown is True:
            clauses.append("unknown_count > 0")
        elif has_unknown is False:
            clauses.append("unknown_count = 0")
        if claimed_by is not None:
            clauses.append("claimed_by = ?")
            params.append(claimed_by)
        if search is not None:
            q = search.strip()
            if q:
                clauses.append("episode_id LIKE ? ESCAPE '\\'")
                params.append(f"%{self._escape_like(q)}%")
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return where_sql, params

    def list_episodes(
        self,
        *,
        status: EpisodeStatus | str | Sequence[EpisodeStatus | str] | None = None,
        dataset: str | None = None,
        task_id: str | None = None,
        adapter_type: str | None = None,
        passed: bool | None = None,
        has_unknown: bool | None = None,
        guardrail_downgraded: bool | None = None,
        claimed_by: str | None = None,
        search: str | None = None,
        sort_by: str | None = None,
        sort_dir: str = "asc",
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        where_sql, params = self._build_episode_where(
            status=status,
            dataset=dataset,
            task_id=task_id,
            adapter_type=adapter_type,
            passed=passed,
            has_unknown=has_unknown,
            claimed_by=claimed_by,
            search=search,
        )
        if sort_by is None:
            order_sql = "ORDER BY dataset ASC, episode_id ASC"
        else:
            col = self._SORT_COLUMN_MAP.get(sort_by)
            if col is None:
                raise ValueError(f"Invalid sort_by: {sort_by!r}")
            dir_sql = self._SORT_DIR_MAP.get(str(sort_dir).lower())
            if dir_sql is None:
                raise ValueError(f"Invalid sort_dir: {sort_dir!r}")
            if col == "episode_id":
                order_sql = f"ORDER BY episode_id {dir_sql}"
            else:
                order_sql = (
                    f"ORDER BY CASE WHEN {col} IS NULL THEN 1 ELSE 0 END ASC, "
                    f"{col} {dir_sql}, episode_id ASC"
                )
        base_sql = f"""
            SELECT episode_id, dataset, task_id, adapter_type, source_uri,
                   source_mtime, duration_s, fps, has_kinematics, camera_roles_json,
                   status, active_run_id, active_rev_id, claimed_by, claim_expires_at,
                   resolved_passed, cut_ratio, mask_ratio, badcase_count,
                   unknown_count, updated_at
              FROM episodes
              {where_sql}
             {order_sql}
        """
        if guardrail_downgraded is None:
            sql = base_sql + "\n             LIMIT ? OFFSET ?"
            params_exec = [*params, limit, offset]
            rows = self.db.connect().execute(sql, params_exec).fetchall()
            return [dict(r) for r in rows]

        rows = self.db.connect().execute(base_sql, params).fetchall()
        filtered = [
            d
            for r in rows
            for d in (dict(r),)
            if self.episode_guardrail_downgraded(d) is guardrail_downgraded
        ]
        return filtered[offset : offset + limit]

    def count_episodes(
        self,
        *,
        status: EpisodeStatus | str | Sequence[EpisodeStatus | str] | None = None,
        dataset: str | None = None,
        task_id: str | None = None,
        adapter_type: str | None = None,
        passed: bool | None = None,
        has_unknown: bool | None = None,
        guardrail_downgraded: bool | None = None,
        claimed_by: str | None = None,
        search: str | None = None,
    ) -> int:
        where_sql, params = self._build_episode_where(
            status=status,
            dataset=dataset,
            task_id=task_id,
            adapter_type=adapter_type,
            passed=passed,
            has_unknown=has_unknown,
            claimed_by=claimed_by,
            search=search,
        )
        if guardrail_downgraded is None:
            row = (
                self.db.connect()
                .execute(f"SELECT COUNT(*) AS cnt FROM episodes {where_sql}", params)
                .fetchone()
            )
            return int(row["cnt"]) if row is not None else 0
        rows = (
            self.db.connect()
            .execute(
                f"SELECT episode_id, active_run_id, active_rev_id FROM episodes {where_sql}",
                params,
            )
            .fetchall()
        )
        return sum(
            1
            for r in rows
            if self.episode_guardrail_downgraded(dict(r)) is guardrail_downgraded
        )

    def count_episodes_by_status(
        self,
        *,
        dataset: str | None = None,
        task_id: str | None = None,
        adapter_type: str | None = None,
        passed: bool | None = None,
        has_unknown: bool | None = None,
        search: str | None = None,
    ) -> dict[str, int]:
        """Return pure SQL counts grouped by status across all 9 EpisodeStatus values."""
        counts = {st.value: 0 for st in EpisodeStatus}
        where_sql, params = self._build_episode_where(
            dataset=dataset,
            task_id=task_id,
            adapter_type=adapter_type,
            passed=passed,
            has_unknown=has_unknown,
            search=search,
        )
        rows = (
            self.db.connect()
            .execute(
                f"SELECT status, COUNT(*) AS cnt FROM episodes {where_sql} GROUP BY status",
                params,
            )
            .fetchall()
        )
        for r in rows:
            st_key = str(r["status"])
            counts[st_key] = int(r["cnt"])
        return counts

    def list_episode_facets(self) -> dict[str, list[str]]:
        """Return sorted distinct workspace facets for datasets, task_ids, and adapter_types."""
        conn = self.db.connect()
        datasets = [
            str(r["dataset"])
            for r in conn.execute(
                "SELECT DISTINCT dataset FROM episodes WHERE dataset IS NOT NULL ORDER BY dataset ASC"
            ).fetchall()
        ]
        task_ids = [
            str(r["task_id"])
            for r in conn.execute(
                "SELECT DISTINCT task_id FROM episodes WHERE task_id IS NOT NULL ORDER BY task_id ASC"
            ).fetchall()
        ]
        adapter_types = [
            str(r["adapter_type"])
            for r in conn.execute(
                "SELECT DISTINCT adapter_type FROM episodes WHERE adapter_type IS NOT NULL ORDER BY adapter_type ASC"
            ).fetchall()
        ]
        return {
            "datasets": datasets,
            "task_ids": task_ids,
            "adapter_types": adapter_types,
        }

    def transition_status(
        self, episode_id: str, new_status: EpisodeStatus
    ) -> EpisodeStatus:
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT status FROM episodes WHERE episode_id = ?", (episode_id,)
            ).fetchone()
            if row is None:
                raise NotFoundError(f"Episode not found: {episode_id!r}")
            current = EpisodeStatus(row["status"])
            if current == EpisodeStatus.GT_FROZEN:
                raise GtFrozenError(f"Episode {episode_id!r} is gt_frozen")
            if new_status != current and new_status not in ALLOWED_TRANSITIONS[current]:
                raise InvalidTransitionError(
                    f"Cannot transition {episode_id!r} from {current.value} to {new_status.value}"
                )
            conn.execute(
                "UPDATE episodes SET status = ?, updated_at = ? WHERE episode_id = ?",
                (new_status.value, time.time(), episode_id),
            )
            return new_status

    def queue_vlm_episodes(
        self,
        episode_ids: Sequence[str],
        *,
        now: float | None = None,
    ) -> list[tuple[str, EpisodeStatus]]:
        """Atomically move eligible (ingested, vlm_ready) episodes to vlm_queued.

        Returns [(episode_id, prev_status), ...] for episodes actually transitioned;
        protected review states and already-active episodes are left untouched.
        """
        unique_ids = list(dict.fromkeys(str(e) for e in episode_ids if e))
        if not unique_ids:
            return []
        ts_now = time.time() if now is None else float(now)
        eligible = {EpisodeStatus.INGESTED.value, EpisodeStatus.VLM_READY.value}
        queued: list[tuple[str, EpisodeStatus]] = []
        with self.db.transaction() as conn:
            for ep_id in unique_ids:
                row = conn.execute(
                    "SELECT status FROM episodes WHERE episode_id = ?", (ep_id,)
                ).fetchone()
                if row is None or row["status"] not in eligible:
                    continue
                prev_st = EpisodeStatus(row["status"])
                conn.execute(
                    "UPDATE episodes SET status = ?, updated_at = ? WHERE episode_id = ?",
                    (EpisodeStatus.VLM_QUEUED.value, ts_now, ep_id),
                )
                queued.append((ep_id, prev_st))
        return queued

    def restore_vlm_episodes_on_cancel(
        self,
        episode_ids: Sequence[str],
        prev_statuses: dict[str, str | EpisodeStatus] | None = None,
        *,
        now: float | None = None,
    ) -> list[tuple[str, EpisodeStatus]]:
        """Atomically revert episodes currently in vlm_queued/vlm_running back to pre-job status.

        Only touches episodes whose DB status is still vlm_queued or vlm_running (never
        clobbers completed vlm_ready or protected review statuses). Restores to prev_statuses
        if in {ingested, vlm_ready}, else vlm_ready when active_run_id is set, else ingested.
        Returns [(episode_id, restored_status), ...] for episodes actually updated.
        """
        unique_ids = list(dict.fromkeys(str(e) for e in episode_ids if e))
        if not unique_ids:
            return []
        ts_now = time.time() if now is None else float(now)
        transient = {EpisodeStatus.VLM_QUEUED.value, EpisodeStatus.VLM_RUNNING.value}
        restorable = {EpisodeStatus.INGESTED.value, EpisodeStatus.VLM_READY.value}
        restored: list[tuple[str, EpisodeStatus]] = []
        with self.db.transaction() as conn:
            for ep_id in unique_ids:
                row = conn.execute(
                    "SELECT status, active_run_id FROM episodes WHERE episode_id = ?",
                    (ep_id,),
                ).fetchone()
                if row is None or row["status"] not in transient:
                    continue
                raw_prev = (prev_statuses or {}).get(ep_id)
                prev_val = (
                    raw_prev.value
                    if isinstance(raw_prev, EpisodeStatus)
                    else (str(raw_prev) if raw_prev is not None else None)
                )
                if prev_val in restorable:
                    target = EpisodeStatus(prev_val)
                else:
                    target = (
                        EpisodeStatus.VLM_READY
                        if row["active_run_id"]
                        else EpisodeStatus.INGESTED
                    )
                conn.execute(
                    "UPDATE episodes SET status = ?, updated_at = ? WHERE episode_id = ?",
                    (target.value, ts_now, ep_id),
                )
                restored.append((ep_id, target))
        return restored

    def save_vlm_run_artifact(
        self,
        artifact: VlmRunArtifact,
        *,
        before_commit_check: Callable[[], None] | None = None,
    ) -> Path:
        meta = self.get_episode_meta(artifact.episode_id)
        row = self.get_episode_row(artifact.episode_id)
        if not artifact.is_eval_sandbox and row["status"] == EpisodeStatus.GT_FROZEN.value:
            raise GtFrozenError(
                f"Episode {artifact.episode_id!r} is gt_frozen; cannot overwrite active VLM run"
            )

        rel_path = self.artifacts.vlm_run_relpath(
            meta.dataset, meta.stem, artifact.run_id
        )

        total_tokens = artifact.usage.total_tokens or sum(
            s.total_tokens for s in artifact.usage.stages
        )
        total_elapsed = artifact.usage.total_elapsed_s or sum(
            s.elapsed_s for s in artifact.usage.stages
        )
        fp = artifact.config_fingerprint

        with self.db.transaction() as conn:
            if before_commit_check is not None:
                before_commit_check()
            cur_row = conn.execute(
                "SELECT status, duration_s FROM episodes WHERE episode_id = ?",
                (artifact.episode_id,),
            ).fetchone()
            if cur_row is None:
                raise NotFoundError(f"Episode not found: {artifact.episode_id!r}")
            if not artifact.is_eval_sandbox and cur_row["status"] == EpisodeStatus.GT_FROZEN.value:
                raise GtFrozenError(f"Episode {artifact.episode_id!r} is gt_frozen")

            abs_path = self.artifacts.write_immutable_json(rel_path, artifact)

            conn.execute(
                """
                INSERT INTO vlm_runs (
                    run_id, episode_id, is_eval_sandbox, model_routing_json,
                    prompt_version, label_version, task_spec_version,
                    few_shot_pack_id, artifact_path, status,
                    tokens_total, elapsed_s, error_message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.run_id,
                    artifact.episode_id,
                    1 if artifact.is_eval_sandbox else 0,
                    json.dumps(fp.model_routing, ensure_ascii=False),
                    fp.prompt_bundle_version,
                    fp.label_taxonomy_version,
                    fp.task_spec_version,
                    fp.few_shot_pack_version,
                    rel_path.as_posix(),
                    "completed",
                    total_tokens,
                    total_elapsed,
                    None,
                    artifact.created_at,
                ),
            )

            if (
                not artifact.is_eval_sandbox
                and cur_row["status"] not in _PROTECTED_REVIEW_STATUSES
            ):
                metrics = _compute_episode_metrics(
                    float(cur_row["duration_s"]),
                    artifact.qc_segments,
                    artifact.success,
                )
                conn.execute(
                    """
                    UPDATE episodes
                       SET active_run_id = ?,
                           status = ?,
                           resolved_passed = ?,
                           cut_ratio = ?,
                           mask_ratio = ?,
                           badcase_count = ?,
                           unknown_count = ?,
                           updated_at = ?
                     WHERE episode_id = ?
                    """,
                    (
                        artifact.run_id,
                        EpisodeStatus.VLM_READY.value,
                        metrics["resolved_passed"],
                        metrics["cut_ratio"],
                        metrics["mask_ratio"],
                        metrics["badcase_count"],
                        metrics["unknown_count"],
                        time.time(),
                        artifact.episode_id,
                    ),
                )
        return abs_path

    def list_vlm_runs(self, episode_id: str) -> list[dict[str, Any]]:
        """List historical VlmRunArtifact metadata rows for an episode (newest first)."""
        self.get_episode_row(episode_id)
        rows = (
            self.db.connect()
            .execute(
                """
                SELECT run_id, episode_id, is_eval_sandbox, model_routing_json,
                       prompt_version, label_version, task_spec_version,
                       few_shot_pack_id, status, tokens_total, elapsed_s,
                       error_message, created_at
                  FROM vlm_runs
                 WHERE episode_id = ?
                 ORDER BY created_at DESC, rowid DESC
                """,
                (episode_id,),
            )
            .fetchall()
        )
        out: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            d["is_eval_sandbox"] = bool(d.get("is_eval_sandbox"))
            d["tokens_total"] = int(d.get("tokens_total") or 0)
            d["elapsed_s"] = float(d.get("elapsed_s") or 0.0)
            d["created_at"] = float(d.get("created_at") or 0.0)
            out.append(d)
        return out

    def get_vlm_run(self, run_id: str) -> VlmRunArtifact:
        row = (
            self.db.connect()
            .execute("SELECT artifact_path FROM vlm_runs WHERE run_id = ?", (run_id,))
            .fetchone()
        )
        if row is None:
            raise NotFoundError(f"VLM run not found: {run_id!r}")
        data = self.artifacts.read_json(row["artifact_path"])
        return VlmRunArtifact.model_validate(data)

    def get_review(self, rev_id: str) -> HumanReviewArtifact:
        row = (
            self.db.connect()
            .execute("SELECT artifact_path FROM reviews WHERE rev_id = ?", (rev_id,))
            .fetchone()
        )
        if row is None:
            raise NotFoundError(f"Review not found: {rev_id!r}")
        data = self.artifacts.read_json(row["artifact_path"])
        return HumanReviewArtifact.model_validate(data)

    def claim_episode(
        self,
        episode_id: str,
        user: str,
        *,
        ttl_s: float = 120.0,
        now: float | None = None,
    ) -> dict[str, Any]:
        ts_now = time.time() if now is None else float(now)
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT status, claimed_by, claim_expires_at FROM episodes WHERE episode_id = ?",
                (episode_id,),
            ).fetchone()
            if row is None:
                raise NotFoundError(f"Episode not found: {episode_id!r}")
            st = EpisodeStatus(row["status"])
            if st == EpisodeStatus.GT_FROZEN:
                raise GtFrozenError(f"Episode {episode_id!r} is gt_frozen")
            if st not in {
                EpisodeStatus.VLM_READY,
                EpisodeStatus.CLAIMED,
                EpisodeStatus.REVIEWED,
                EpisodeStatus.FLAGGED,
                EpisodeStatus.APPROVED,
            }:
                raise InvalidTransitionError(
                    f"Episode {episode_id!r} in status {st.value!r} cannot be claimed"
                )
            cur_claimer = row["claimed_by"]
            cur_exp = float(row["claim_expires_at"] or 0.0)
            if (
                st == EpisodeStatus.CLAIMED
                and cur_claimer is not None
                and cur_claimer != user
                and cur_exp > ts_now
            ):
                raise ClaimConflictError(
                    f"Episode {episode_id!r} is already claimed by {cur_claimer!r} until {cur_exp}"
                )
            expires_at = ts_now + ttl_s
            conn.execute(
                """
                UPDATE episodes
                   SET status = ?, claimed_by = ?, claim_expires_at = ?, updated_at = ?
                 WHERE episode_id = ?
                """,
                (EpisodeStatus.CLAIMED.value, user, expires_at, ts_now, episode_id),
            )
            return {
                "episode_id": episode_id,
                "claimed_by": user,
                "claim_expires_at": expires_at,
                "status": EpisodeStatus.CLAIMED.value,
            }

    def renew_claim(
        self,
        episode_id: str,
        user: str,
        *,
        ttl_s: float = 120.0,
        now: float | None = None,
    ) -> dict[str, Any]:
        ts_now = time.time() if now is None else float(now)
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT status, claimed_by, claim_expires_at FROM episodes WHERE episode_id = ?",
                (episode_id,),
            ).fetchone()
            if row is None:
                raise NotFoundError(f"Episode not found: {episode_id!r}")
            if row["status"] != EpisodeStatus.CLAIMED.value:
                raise ClaimConflictError(f"Episode {episode_id!r} is not claimed")
            cur_claimer = row["claimed_by"]
            cur_exp = float(row["claim_expires_at"] or 0.0)
            if cur_claimer != user:
                raise ClaimConflictError(
                    f"Episode {episode_id!r} is claimed by {cur_claimer!r}, not {user!r}"
                )
            if cur_exp <= ts_now:
                raise ClaimConflictError(f"Claim on {episode_id!r} already expired")
            expires_at = ts_now + ttl_s
            conn.execute(
                "UPDATE episodes SET claim_expires_at = ?, updated_at = ? WHERE episode_id = ?",
                (expires_at, ts_now, episode_id),
            )
            return {
                "episode_id": episode_id,
                "claimed_by": user,
                "claim_expires_at": expires_at,
            }

    def _restore_for_release(
        self,
        conn: Any,
        episode_id: str,
        active_run_id: str | None,
        duration_s: float,
    ) -> tuple[str, str | None, dict[str, Any]]:
        rev_row = conn.execute(
            """
            SELECT rev_id, action, artifact_path FROM reviews
             WHERE episode_id = ? AND action != 'save_draft'
             ORDER BY created_at DESC, rowid DESC LIMIT 1
            """,
            (episode_id,),
        ).fetchone()
        if rev_row is not None:
            action_to_status = {
                "submit_review": EpisodeStatus.REVIEWED.value,
                "flag": EpisodeStatus.FLAGGED.value,
                "approve": EpisodeStatus.APPROVED.value,
                "freeze_gt": EpisodeStatus.GT_FROZEN.value,
            }
            next_status = action_to_status.get(
                rev_row["action"], EpisodeStatus.VLM_READY.value
            )
            rev_art = HumanReviewArtifact.model_validate(
                self.artifacts.read_json(rev_row["artifact_path"])
            )
            metrics = _compute_episode_metrics(
                duration_s, rev_art.qc_segments, rev_art.success
            )
            return next_status, str(rev_row["rev_id"]), metrics

        if active_run_id:
            run_row = conn.execute(
                "SELECT artifact_path FROM vlm_runs WHERE run_id = ?",
                (active_run_id,),
            ).fetchone()
            if run_row is not None:
                vlm_run = VlmRunArtifact.model_validate(
                    self.artifacts.read_json(run_row["artifact_path"])
                )
                metrics = _compute_episode_metrics(
                    duration_s, vlm_run.qc_segments, vlm_run.success
                )
                return EpisodeStatus.VLM_READY.value, None, metrics

        return (
            EpisodeStatus.VLM_READY.value if active_run_id else EpisodeStatus.INGESTED.value,
            None,
            {
                "resolved_passed": None,
                "cut_ratio": 0.0,
                "mask_ratio": 0.0,
                "badcase_count": 0,
                "unknown_count": 0,
            },
        )

    def release_claim(
        self,
        episode_id: str,
        user: str | None = None,
        *,
        force: bool = False,
        now: float | None = None,
    ) -> str:
        ts_now = time.time() if now is None else float(now)
        with self.db.transaction() as conn:
            row = conn.execute(
                """
                SELECT status, claimed_by, claim_expires_at, active_rev_id,
                       active_run_id, duration_s
                  FROM episodes WHERE episode_id = ?
                """,
                (episode_id,),
            ).fetchone()
            if row is None:
                raise NotFoundError(f"Episode not found: {episode_id!r}")
            if row["status"] == EpisodeStatus.GT_FROZEN.value:
                raise GtFrozenError(f"Episode {episode_id!r} is gt_frozen")
            if row["status"] != EpisodeStatus.CLAIMED.value:
                return str(row["status"])
            cur_claimer = row["claimed_by"]
            cur_exp = float(row["claim_expires_at"] or 0.0)
            if not force and user is not None and cur_claimer != user and cur_exp > ts_now:
                raise ClaimConflictError(
                    f"Cannot release active claim held by {cur_claimer!r}"
                )
            next_status, restored_rev_id, metrics = self._restore_for_release(
                conn,
                episode_id,
                row["active_run_id"],
                float(row["duration_s"]),
            )
            conn.execute(
                """
                UPDATE episodes
                   SET status = ?,
                       active_rev_id = ?,
                       claimed_by = NULL,
                       claim_expires_at = NULL,
                       resolved_passed = ?,
                       cut_ratio = ?,
                       mask_ratio = ?,
                       badcase_count = ?,
                       unknown_count = ?,
                       updated_at = ?
                 WHERE episode_id = ?
                """,
                (
                    next_status,
                    restored_rev_id,
                    metrics["resolved_passed"],
                    metrics["cut_ratio"],
                    metrics["mask_ratio"],
                    metrics["badcase_count"],
                    metrics["unknown_count"],
                    ts_now,
                    episode_id,
                ),
            )
            return next_status

    def reap_expired_claims(self, now: float | None = None) -> int:
        ts_now = time.time() if now is None else float(now)
        with self.db.transaction() as conn:
            rows = conn.execute(
                """
                SELECT episode_id, active_rev_id, active_run_id, duration_s
                  FROM episodes
                 WHERE status = ? AND claim_expires_at IS NOT NULL AND claim_expires_at <= ?
                """,
                (EpisodeStatus.CLAIMED.value, ts_now),
            ).fetchall()
            for r in rows:
                next_status, restored_rev_id, metrics = self._restore_for_release(
                    conn,
                    str(r["episode_id"]),
                    r["active_run_id"],
                    float(r["duration_s"]),
                )
                conn.execute(
                    """
                    UPDATE episodes
                       SET status = ?,
                           active_rev_id = ?,
                           claimed_by = NULL,
                           claim_expires_at = NULL,
                           resolved_passed = ?,
                           cut_ratio = ?,
                           mask_ratio = ?,
                           badcase_count = ?,
                           unknown_count = ?,
                           updated_at = ?
                     WHERE episode_id = ?
                    """,
                    (
                        next_status,
                        restored_rev_id,
                        metrics["resolved_passed"],
                        metrics["cut_ratio"],
                        metrics["mask_ratio"],
                        metrics["badcase_count"],
                        metrics["unknown_count"],
                        ts_now,
                        r["episode_id"],
                    ),
                )
            return len(rows)

    def save_review(
        self,
        episode_id: str,
        author: str,
        role: ReviewRole,
        payload: ReviewWritePayload,
        *,
        min_watch_coverage: float = 0.95,
        now: float | None = None,
    ) -> HumanReviewArtifact:
        ts_now = time.time() if now is None else float(now)
        if payload.action in {"approve", "freeze_gt"} and role != "lead":
            raise AuthorizationError(
                f"Action {payload.action!r} requires 'lead' role, got {role!r}"
            )
        eff_watch_ratio = min(
            payload.watch_coverage.coverage_ratio,
            payload.watch_coverage.watched_s / max(payload.watch_coverage.duration_s, 1e-6),
        )
        if (
            payload.action == "submit_review"
            and role == "reviewer"
            and eff_watch_ratio < min_watch_coverage
        ):
            raise WatchCoverageError(
                f"Watch coverage {eff_watch_ratio:.3f} < {min_watch_coverage:.3f}"
            )

        with self.db.transaction() as conn:
            ep_row = conn.execute(
                "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
            ).fetchone()
            if ep_row is None:
                raise NotFoundError(f"Episode not found: {episode_id!r}")
            cur_status = EpisodeStatus(ep_row["status"])
            if cur_status == EpisodeStatus.GT_FROZEN:
                raise GtFrozenError(f"Episode {episode_id!r} is gt_frozen")
            if cur_status in {
                EpisodeStatus.INGESTED,
                EpisodeStatus.VLM_QUEUED,
                EpisodeStatus.VLM_RUNNING,
            }:
                raise InvalidTransitionError(
                    f"Episode {episode_id!r} in status {cur_status.value!r} cannot be reviewed while VLM is not ready"
                )
            if cur_status == EpisodeStatus.CLAIMED:
                claimer = ep_row["claimed_by"]
                exp = float(ep_row["claim_expires_at"] or 0.0)
                if claimer is not None and claimer != author and exp > ts_now:
                    raise ClaimConflictError(
                        f"Episode {episode_id!r} is claimed by {claimer!r}"
                    )

            run_row = conn.execute(
                "SELECT artifact_path FROM vlm_runs WHERE run_id = ?",
                (payload.base_run_id,),
            ).fetchone()
            if run_row is None:
                raise NotFoundError(f"Base VLM run not found: {payload.base_run_id!r}")

            base_run = VlmRunArtifact.model_validate(
                self.artifacts.read_json(run_row["artifact_path"])
            )
            if base_run.episode_id != episode_id:
                raise RepositoryError(
                    f"Base run {payload.base_run_id!r} belongs to {base_run.episode_id!r}, not {episode_id!r}"
                )

            diff_summary = compute_review_diff(
                base_run,
                payload.qc_segments,
                payload.success,
                payload.macro_stages,
            )
            meta = EpisodeMeta.model_validate_json(ep_row["meta_json"])
            rev_id = f"rev_{int(ts_now * 1000)}_{uuid.uuid4().hex[:8]}"
            artifact = HumanReviewArtifact(
                rev_id=rev_id,
                episode_id=episode_id,
                base_run_id=payload.base_run_id,
                author=author,
                role=role,
                action=payload.action,
                created_at=ts_now,
                watch_coverage=payload.watch_coverage,
                qc_segments=payload.qc_segments,
                success=payload.success,
                macro_stages=payload.macro_stages,
                notes=payload.notes,
                flagged_reason=payload.flagged_reason,
                diff_summary=diff_summary,
            )
            rel_path = self.artifacts.review_relpath(meta.dataset, meta.stem, rev_id)
            self.artifacts.write_immutable_json(rel_path, artifact)

            conn.execute(
                """
                INSERT INTO reviews (
                    rev_id, episode_id, base_run_id, author, role, action,
                    artifact_path, diff_stats_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rev_id,
                    episode_id,
                    payload.base_run_id,
                    author,
                    role,
                    payload.action,
                    rel_path.as_posix(),
                    diff_summary.model_dump_json(),
                    ts_now,
                ),
            )

            if payload.action == "save_draft":
                next_status = cur_status.value
                claimed_by = ep_row["claimed_by"]
                claim_exp = ep_row["claim_expires_at"]
            else:
                action_status = {
                    "submit_review": EpisodeStatus.REVIEWED.value,
                    "flag": EpisodeStatus.FLAGGED.value,
                    "approve": EpisodeStatus.APPROVED.value,
                    "freeze_gt": EpisodeStatus.GT_FROZEN.value,
                }
                next_status = action_status[payload.action]
                claimed_by = None
                claim_exp = None

            metrics = _compute_episode_metrics(
                float(ep_row["duration_s"]),
                payload.qc_segments,
                payload.success,
            )
            conn.execute(
                """
                UPDATE episodes
                   SET active_rev_id = ?,
                       status = ?,
                       claimed_by = ?,
                       claim_expires_at = ?,
                       resolved_passed = ?,
                       cut_ratio = ?,
                       mask_ratio = ?,
                       badcase_count = ?,
                       unknown_count = ?,
                       updated_at = ?
                 WHERE episode_id = ?
                """,
                (
                    rev_id,
                    next_status,
                    claimed_by,
                    claim_exp,
                    metrics["resolved_passed"],
                    metrics["cut_ratio"],
                    metrics["mask_ratio"],
                    metrics["badcase_count"],
                    metrics["unknown_count"],
                    ts_now,
                    episode_id,
                ),
            )
            return artifact

    def resolve_episode(
        self, episode_id: str, *, run_id: str | None = None
    ) -> ResolvedEpisode:
        row = self.get_episode_row(episode_id)
        meta = EpisodeMeta.model_validate_json(row["meta_json"])
        target_run_id = run_id or row["active_run_id"]
        if not target_run_id:
            raise NotFoundError(f"Episode {episode_id!r} has no active VLM run")

        vlm_run = self.get_vlm_run(target_run_id)
        if vlm_run.episode_id != episode_id:
            raise NotFoundError(
                f"VLM run {target_run_id!r} does not belong to episode {episode_id!r}"
            )
        active_rev_id = row["active_rev_id"] if run_id is None else None
        if active_rev_id:
            review = self.get_review(active_rev_id)
            return ResolvedEpisode(
                episode_id=episode_id,
                meta=meta,
                status=EpisodeStatus(row["status"]),
                active_run_id=target_run_id,
                active_rev_id=active_rev_id,
                is_human_reviewed=True,
                qc_segments=review.qc_segments,
                success=review.success,
                macro_stages=review.macro_stages,
                grounding=vlm_run.grounding,
                arm_timelines=vlm_run.arm_timelines,
                raw_vlm_segments=vlm_run.qc_segments,
                raw_vlm_success=vlm_run.success,
                raw_vlm_macro_stages=vlm_run.macro_stages,
                review_diff=review.diff_summary,
                notes=review.notes,
                flagged_reason=review.flagged_reason,
            )

        return ResolvedEpisode(
            episode_id=episode_id,
            meta=meta,
            status=EpisodeStatus(row["status"]),
            active_run_id=target_run_id,
            active_rev_id=None,
            is_human_reviewed=False,
            qc_segments=vlm_run.qc_segments,
            success=vlm_run.success,
            macro_stages=vlm_run.macro_stages,
            grounding=vlm_run.grounding,
            arm_timelines=vlm_run.arm_timelines,
            raw_vlm_segments=vlm_run.qc_segments,
            raw_vlm_success=vlm_run.success,
            raw_vlm_macro_stages=vlm_run.macro_stages,
        )

    def resolve_episode_for_api(
        self, episode_id: str, *, run_id: str | None = None
    ) -> ResolvedEpisode:
        """Resolve an episode for API consumption, tolerating episodes with no VLM run yet."""
        row = self.get_episode_row(episode_id)
        if run_id or row["active_run_id"]:
            return self.resolve_episode(episode_id, run_id=run_id)
        meta = EpisodeMeta.model_validate_json(row["meta_json"])
        return ResolvedEpisode(
            episode_id=episode_id,
            meta=meta,
            status=EpisodeStatus(row["status"]),
            active_run_id=None,
            active_rev_id=None,
            is_human_reviewed=False,
            qc_segments=[],
            success=SuccessVerdict(),
            macro_stages=[],
            grounding=GroundingResult(),
            arm_timelines=ArmTimelines(),
            raw_vlm_segments=[],
            raw_vlm_success=None,
            raw_vlm_macro_stages=[],
        )

    def list_reviews(self, episode_id: str) -> list[HumanReviewArtifact]:
        """Return chronological immutable HumanReviewArtifact snapshots for an episode."""
        self.get_episode_row(episode_id)
        rows = (
            self.db.connect()
            .execute(
                """
                SELECT artifact_path FROM reviews
                 WHERE episode_id = ?
                 ORDER BY created_at ASC, rowid ASC
                """,
                (episode_id,),
            )
            .fetchall()
        )
        return [
            HumanReviewArtifact.model_validate(self.artifacts.read_json(r["artifact_path"]))
            for r in rows
        ]

    @staticmethod
    def _job_row_to_dict(row: Any) -> dict[str, Any]:
        params_raw = row["params_json"]
        summary_raw = row["summary_json"]
        return {
            "job_id": str(row["job_id"]),
            "job_type": str(row["job_type"]),
            "status": str(row["status"]),
            "params": json.loads(params_raw) if params_raw else {},
            "progress_done": int(row["progress_done"] or 0),
            "progress_total": int(row["progress_total"] or 0),
            "summary": json.loads(summary_raw) if summary_raw else None,
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
        }

    def create_job(
        self,
        job_type: str,
        params: dict[str, Any],
        *,
        progress_total: int = 0,
        job_id: str | None = None,
        status: str = "queued",
        now: float | None = None,
    ) -> dict[str, Any]:
        ts_now = time.time() if now is None else float(now)
        jid = job_id or f"job_{int(ts_now * 1000)}_{uuid.uuid4().hex[:8]}"
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, job_type, status, params_json,
                    progress_done, progress_total, summary_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    jid,
                    job_type,
                    status,
                    json.dumps(params, ensure_ascii=False),
                    0,
                    int(progress_total),
                    None,
                    ts_now,
                    ts_now,
                ),
            )
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (jid,)).fetchone()
            assert row is not None
            return self._job_row_to_dict(row)

    def get_job(self, job_id: str) -> dict[str, Any]:
        row = (
            self.db.connect()
            .execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            .fetchone()
        )
        if row is None:
            raise NotFoundError(f"Job not found: {job_id!r}")
        return self._job_row_to_dict(row)

    def list_jobs(
        self,
        *,
        status: str | None = None,
        job_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if job_type is not None:
            clauses.append("job_type = ?")
            params.append(job_type)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT * FROM jobs
            {where_sql}
             ORDER BY created_at DESC, rowid DESC
             LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        rows = self.db.connect().execute(sql, params).fetchall()
        return [self._job_row_to_dict(r) for r in rows]

    def count_jobs(
        self,
        *,
        status: str | None = None,
        job_type: str | None = None,
    ) -> int:
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if job_type is not None:
            clauses.append("job_type = ?")
            params.append(job_type)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        row = (
            self.db.connect()
            .execute(f"SELECT COUNT(*) AS cnt FROM jobs {where_sql}", params)
            .fetchone()
        )
        return int(row["cnt"]) if row is not None else 0

    def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        progress_done: int | None = None,
        progress_total: int | None = None,
        summary: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        ts_now = time.time() if now is None else float(now)
        with self.db.transaction() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"Job not found: {job_id!r}")
            if row["status"] == "cancelled":
                return self._job_row_to_dict(row)
            sets: list[str] = ["updated_at = ?"]
            vals: list[Any] = [ts_now]
            if status is not None:
                sets.append("status = ?")
                vals.append(status)
            if progress_done is not None:
                sets.append("progress_done = ?")
                vals.append(int(progress_done))
            if progress_total is not None:
                sets.append("progress_total = ?")
                vals.append(int(progress_total))
            if summary is not None:
                sets.append("summary_json = ?")
                vals.append(json.dumps(summary, ensure_ascii=False))
            vals.append(job_id)
            conn.execute(
                f"UPDATE jobs SET {', '.join(sets)} WHERE job_id = ?",
                vals,
            )
            updated = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            assert updated is not None
            return self._job_row_to_dict(updated)

    def cancel_job(self, job_id: str, *, now: float | None = None) -> dict[str, Any]:
        ts_now = time.time() if now is None else float(now)
        with self.db.transaction() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"Job not found: {job_id!r}")
            if row["status"] in ("queued", "running"):
                conn.execute(
                    "UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ?",
                    ("cancelled", ts_now, job_id),
                )
            updated = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            assert updated is not None
            return self._job_row_to_dict(updated)

    def save_few_shot(self, item: FewShotItem) -> str:
        """Insert or update a FewShotItem in SQLite."""
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO few_shot_examples (
                    example_id, scope, task_id, label_id, error_type, episode_id,
                    t_start, t_end, tile_image_path, vlm_wrong_json, human_right_json,
                    rationale, enabled, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(example_id) DO UPDATE SET
                    scope = excluded.scope,
                    task_id = excluded.task_id,
                    label_id = excluded.label_id,
                    error_type = excluded.error_type,
                    episode_id = excluded.episode_id,
                    t_start = excluded.t_start,
                    t_end = excluded.t_end,
                    tile_image_path = excluded.tile_image_path,
                    vlm_wrong_json = excluded.vlm_wrong_json,
                    human_right_json = excluded.human_right_json,
                    rationale = excluded.rationale,
                    enabled = excluded.enabled,
                    created_at = excluded.created_at
                """,
                (
                    item.example_id,
                    item.scope,
                    item.task_id,
                    item.label_id,
                    item.error_type,
                    item.episode_id,
                    float(item.t_start),
                    float(item.t_end),
                    item.tile_image_path,
                    json.dumps(item.vlm_wrong_json, ensure_ascii=False),
                    json.dumps(item.human_right_json, ensure_ascii=False),
                    item.rationale,
                    1 if item.enabled else 0,
                    float(item.created_at),
                ),
            )
        return item.example_id

    def list_few_shots(
        self,
        *,
        task_id: str | None = None,
        label_id: str | None = None,
        enabled_only: bool = True,
        limit: int = 200,
    ) -> list[FewShotItem]:
        """List FewShotItems, optionally filtered by task/label/enabled."""
        clauses: list[str] = []
        params: list[Any] = []
        if enabled_only:
            clauses.append("enabled = 1")
        if task_id is not None:
            clauses.append(
                "(scope = 'global' OR scope = ? OR (scope LIKE 'label:%' AND (task_id IS NULL OR task_id = ?)) OR task_id = ?)"
            )
            params.extend([f"task:{task_id}", task_id, task_id])
        if label_id is not None:
            clauses.append("(label_id = ? OR scope = ?)")
            params.extend([label_id, f"label:{label_id}"])

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT example_id, scope, task_id, label_id, error_type, episode_id,
                   t_start, t_end, tile_image_path, vlm_wrong_json, human_right_json,
                   rationale, enabled, created_at
              FROM few_shot_examples
              {where_sql}
             ORDER BY created_at DESC, example_id ASC
             LIMIT ?
        """
        params.append(limit)
        rows = self.db.connect().execute(sql, params).fetchall()
        items: list[FewShotItem] = []
        for r in rows:
            wrong_raw = r["vlm_wrong_json"]
            right_raw = r["human_right_json"]
            items.append(
                FewShotItem(
                    example_id=r["example_id"],
                    scope=r["scope"],
                    task_id=r["task_id"],
                    label_id=r["label_id"],
                    error_type=r["error_type"],
                    episode_id=r["episode_id"],
                    t_start=float(r["t_start"]),
                    t_end=float(r["t_end"]),
                    tile_image_path=r["tile_image_path"],
                    vlm_wrong_json=json.loads(wrong_raw) if isinstance(wrong_raw, str) else dict(wrong_raw or {}),
                    human_right_json=json.loads(right_raw) if isinstance(right_raw, str) else dict(right_raw or {}),
                    rationale=r["rationale"],
                    enabled=bool(r["enabled"]),
                    created_at=float(r["created_at"]),
                )
            )
        return items
