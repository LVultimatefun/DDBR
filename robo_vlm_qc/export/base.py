"""OutputExporter ABC, registry, atomic _staging -> rename lock, and manifest helpers."""

from __future__ import annotations

import fcntl
import hashlib
import shutil
import threading
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, Iterator

from ..io.base import EpisodeView
from ..models.export import (
    ExportEpisodeEntry,
    ExportManifest,
    ExportRecipe,
    KeepPiece,
)
from ..models.review import ResolvedEpisode

_STAGING_LOCKS_GUARD = threading.Lock()
_STAGING_LOCKS: dict[Path, threading.RLock] = {}


def _get_dir_lock(path: Path) -> threading.RLock:
    key = path.resolve()
    with _STAGING_LOCKS_GUARD:
        lock = _STAGING_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _STAGING_LOCKS[key] = lock
        return lock


class OutputExporter(ABC):
    """Abstract base class for transactional dataset/visualization exporters."""

    name: ClassVar[str]

    @abstractmethod
    def export(
        self,
        pieces: list[KeepPiece],
        resolved_list: list[ResolvedEpisode],
        views: dict[str, EpisodeView],
        out_dir: Path,
        recipe: ExportRecipe,
    ) -> ExportManifest:
        """Export sliced pieces atomically into `out_dir` and return an ExportManifest."""
        ...


_EXPORTER_REGISTRY: dict[str, OutputExporter] = {}


def register_exporter(
    exporter: OutputExporter | type[OutputExporter], *, override: bool = False
) -> OutputExporter:
    inst = exporter() if isinstance(exporter, type) else exporter
    if not getattr(inst, "name", None):
        raise ValueError(f"Exporter {inst!r} must define non-empty class attribute 'name'")
    if not override and inst.name in _EXPORTER_REGISTRY:
        raise ValueError(f"Exporter {inst.name!r} is already registered")
    _EXPORTER_REGISTRY[inst.name] = inst
    return inst


def get_exporter(name: str) -> OutputExporter:
    if name not in _EXPORTER_REGISTRY:
        raise KeyError(
            f"Unknown exporter {name!r}; registered: {sorted(_EXPORTER_REGISTRY)}"
        )
    return _EXPORTER_REGISTRY[name]


def list_exporters() -> list[str]:
    return sorted(_EXPORTER_REGISTRY.keys())


@contextmanager
def staged_export_dir(out_dir: Path) -> Iterator[Path]:
    """Transactional `_staging -> rename` context manager with file + thread locking.

    Uses sibling ``out_dir.parent / f"{out_dir.name}._staging"`` so ``out_dir``
    never contains ``_staging`` or ``.lock`` files, cleans pre-existing stale
    staging directories before starting, and rolls back cleanly on any failure.
    """
    final_dir = Path(out_dir).resolve()
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = final_dir.parent / f"{final_dir.name}._staging"
    lock_file_path = final_dir.parent / f".{final_dir.name}.export.lock"

    thread_lock = _get_dir_lock(final_dir)
    with thread_lock:
        lock_fd = open(lock_file_path, "a+", encoding="utf-8")
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
            staging_dir.mkdir(parents=True, exist_ok=True)

            try:
                yield staging_dir
            except BaseException:
                shutil.rmtree(staging_dir, ignore_errors=True)
                raise

            bak_dir: Path | None = None
            if final_dir.exists():
                bak_dir = final_dir.parent / f".{final_dir.name}._bak_{uuid.uuid4().hex[:8]}"
                final_dir.replace(bak_dir)
            try:
                staging_dir.replace(final_dir)
            except BaseException:
                if bak_dir is not None and bak_dir.exists() and not final_dir.exists():
                    try:
                        bak_dir.replace(final_dir)
                    except Exception:
                        pass
                shutil.rmtree(staging_dir, ignore_errors=True)
                raise
            if bak_dir is not None and bak_dir.exists():
                shutil.rmtree(bak_dir, ignore_errors=True)
        finally:
            try:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                lock_fd.close()
            except Exception:
                pass
            try:
                lock_file_path.unlink(missing_ok=True)
            except Exception:
                pass


atomic_export_dir = staged_export_dir


def filter_resolved_for_recipe(
    resolved_list: list[ResolvedEpisode], recipe: ExportRecipe
) -> tuple[list[ResolvedEpisode], list[ResolvedEpisode]]:
    """Return (matching_resolved, skipped_by_filter) according to recipe filters."""
    matched: list[ResolvedEpisode] = []
    skipped: list[ResolvedEpisode] = []
    for res in resolved_list:
        if recipe.dataset_filter and res.meta.dataset != recipe.dataset_filter:
            skipped.append(res)
            continue
        if recipe.task_filter and res.meta.task_id != recipe.task_filter:
            skipped.append(res)
            continue
        matched.append(res)
    return matched, skipped


def group_kept_pieces(
    pieces: list[KeepPiece],
    resolved_list: list[ResolvedEpisode],
    recipe: ExportRecipe,
) -> tuple[
    list[tuple[ResolvedEpisode, list[KeepPiece]]],
    list[ExportEpisodeEntry],
    int,
    int,
    int,
    int,
    int,
]:
    """Group kept & dropped pieces per ResolvedEpisode.

    Returns:
        (included_pairs, entries, episodes_included, episodes_skipped,
         pieces_exported, pieces_dropped, total_frames)
    """
    by_ep: dict[str, list[KeepPiece]] = {}
    for p in pieces:
        by_ep.setdefault(p.episode_id, []).append(p)
    for ep_list in by_ep.values():
        ep_list.sort(key=lambda item: (item.piece_index, item.start_row))

    matched, filtered_out = filter_resolved_for_recipe(resolved_list, recipe)
    included_pairs: list[tuple[ResolvedEpisode, list[KeepPiece]]] = []
    entries: list[ExportEpisodeEntry] = []
    episodes_included = 0
    episodes_skipped = len(filtered_out)
    pieces_exported = 0
    pieces_dropped = 0
    total_frames = 0

    for res in filtered_out:
        entries.append(
            ExportEpisodeEntry(
                episode_id=res.episode_id,
                status=res.status,
                passed=res.success.passed,
                pieces_exported=0,
                pieces_dropped=0,
                own_frames=0,
                total_frames=0,
                dropped_reasons=["filtered_out"],
            )
        )

    for res in matched:
        ep_pieces = by_ep.get(res.episode_id, [])
        kept = [p for p in ep_pieces if not p.dropped]
        dropped = [p for p in ep_pieces if p.dropped]
        reasons = [p.drop_reason or "dropped" for p in dropped]
        pieces_dropped += len(dropped)

        if kept:
            episodes_included += 1
            pieces_exported += len(kept)
            own_f = sum(p.own_rows for p in kept)
            tot_f = sum(p.total_rows for p in kept)
            total_frames += tot_f
            included_pairs.append((res, kept))
            entries.append(
                ExportEpisodeEntry(
                    episode_id=res.episode_id,
                    status=res.status,
                    passed=res.success.passed,
                    pieces_exported=len(kept),
                    pieces_dropped=len(dropped),
                    own_frames=own_f,
                    total_frames=tot_f,
                    dropped_reasons=reasons,
                )
            )
        else:
            episodes_skipped += 1
            if not reasons:
                reasons = ["no_kept_pieces"]
            entries.append(
                ExportEpisodeEntry(
                    episode_id=res.episode_id,
                    status=res.status,
                    passed=res.success.passed,
                    pieces_exported=0,
                    pieces_dropped=len(dropped),
                    own_frames=0,
                    total_frames=0,
                    dropped_reasons=reasons,
                )
            )

    return (
        included_pairs,
        entries,
        episodes_included,
        episodes_skipped,
        pieces_exported,
        pieces_dropped,
        total_frames,
    )


def build_global_subtasks(
    kept_pieces: list[KeepPiece],
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    """Deduplicate subtasks globally by `(label, label_en, instruction)` and remap piece indices.

    Returns:
        - global_subtasks: list of dicts ``{subtask_index, subtask, instruction, stage_id, label, label_en}``
        - remapped_indices: ``{piece.piece_id: list[int]}`` of length ``piece.total_rows``
    """
    key_to_global: dict[tuple[str, str, str], int] = {}
    global_subtasks: list[dict[str, Any]] = []
    remapped: dict[str, list[int]] = {}

    for piece in kept_pieces:
        local_to_global: dict[int, int] = {}
        for span in piece.subtasks:
            instr_raw = str(getattr(span, "instruction", "") or "").strip()
            eff_instr = (
                instr_raw
                or span.label_en.strip()
                or span.label.strip()
                or span.stage_id
            )
            key = (span.label.strip(), span.label_en.strip(), eff_instr)
            if key not in key_to_global:
                g_idx = len(global_subtasks)
                key_to_global[key] = g_idx
                subtask_text = (
                    f"{span.label} ({span.label_en})"
                    if span.label_en and span.label_en != span.label
                    else (span.label or span.label_en or span.stage_id)
                )
                global_subtasks.append(
                    {
                        "subtask_index": g_idx,
                        "subtask": subtask_text,
                        "instruction": eff_instr,
                        "stage_id": span.stage_id,
                        "label": span.label,
                        "label_en": span.label_en,
                    }
                )
            local_to_global[span.subtask_index] = key_to_global[key]

        mapped_rows: list[int] = []
        for idx in piece.subtask_indices:
            if idx < 0:
                mapped_rows.append(-1)
            elif idx in local_to_global:
                mapped_rows.append(local_to_global[idx])
            else:
                mapped_rows.append(-1)
        if len(mapped_rows) < piece.total_rows:
            fill_val = mapped_rows[-1] if mapped_rows else -1
            mapped_rows.extend([fill_val] * (piece.total_rows - len(mapped_rows)))
        elif len(mapped_rows) > piece.total_rows:
            mapped_rows = mapped_rows[: piece.total_rows]
        remapped[piece.piece_id] = mapped_rows

    return global_subtasks, remapped


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def collect_artifacts_and_checksums(
    out_dir: Path,
) -> tuple[list[str], dict[str, str]]:
    root = Path(out_dir).resolve()
    artifacts: list[str] = []
    checksums: dict[str, str] = {}
    if not root.is_dir():
        return artifacts, checksums
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            artifacts.append(rel)
            checksums[rel] = sha256_file(p)
    return artifacts, checksums


def finalize_manifest(
    out_dir: Path,
    recipe: ExportRecipe,
    entries: list[ExportEpisodeEntry],
    episodes_included: int,
    episodes_skipped: int,
    pieces_exported: int,
    pieces_dropped: int,
    total_frames: int,
    *,
    export_id: str | None = None,
) -> ExportManifest:
    """Build ExportManifest scanning `out_dir` for artifacts & SHA256 checksums."""
    artifacts, checksums = collect_artifacts_and_checksums(out_dir)
    eid = export_id or f"exp_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    return ExportManifest(
        export_id=eid,
        created_at=time.time(),
        format=recipe.format,
        recipe=recipe,
        output_dir=str(Path(out_dir).resolve()),
        episodes_included=episodes_included,
        episodes_skipped=episodes_skipped,
        pieces_exported=pieces_exported,
        pieces_dropped=pieces_dropped,
        total_frames=total_frames,
        episodes=entries,
        artifacts=artifacts,
        checksums=checksums,
    )
