"""Visual 2x2 + text Few-Shot mistake book management and scoped retrieval."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..config import ConfigManager
from ..io import EpisodeHandle, EpisodeView, get_adapter
from ..io.media import compose_2x2_tile, encode_jpeg_bytes
from ..models.episode import EpisodeMeta
from ..models.taxonomy import DiagnosisErrorType, FewShotItem
from ..store.artifacts import atomic_write
from ..store.repository import Repository
from ..vlm.prompt_builder import _estimate_tokens, _format_single_few_shot


def open_episode_view(meta: EpisodeMeta, cfg_mgr: ConfigManager) -> EpisodeView:
    """Re-open a read-only EpisodeView from persisted EpisodeMeta without importing api."""
    adapter = get_adapter(meta.adapter_type)
    h5_cfg = cfg_mgr.settings.get("io", {}).get("h5", {})
    md = dict(meta.extra_meta)
    md.setdefault("hand_mapping", h5_cfg.get("hand_mapping"))
    md.setdefault("min_invalid_gap_s", h5_cfg.get("min_invalid_gap_s", 0.5))
    handle = EpisodeHandle(
        episode_id=meta.episode_id,
        dataset=meta.dataset,
        stem=meta.stem,
        task_id=meta.task_id,
        adapter_type=meta.adapter_type,
        source_uri=meta.source_uri,
        source_mtime=meta.source_mtime,
        metadata=md,
    )
    view = adapter.open(handle)
    if view.meta.episode_id != meta.episode_id or view.meta.task_id != meta.task_id:
        view.meta = meta
    return view


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
        raise ValueError("Repository or workspace path is required")
    return Repository(Path(ws)), True


def _render_start_end_comparison_jpeg(
    *,
    repo: Repository,
    episode_id: str,
    t_start: float,
    t_end: float,
    cfg_mgr: ConfigManager,
    out_path: Path,
) -> None:
    meta = repo.get_episode_meta(episode_id)
    with open_episode_view(meta, cfg_mgr) as view:
        t0 = max(0.0, min(float(view.duration_s), float(t_start)))
        t1 = max(0.0, min(float(view.duration_s), float(t_end)))
        tile_start = compose_2x2_tile(
            view.media,
            view.camera_roles,
            t0,
            cell_w=320,
            cell_h=180,
            include_header=True,
        )
        tile_end = compose_2x2_tile(
            view.media,
            view.camera_roles,
            t1,
            cell_w=320,
            cell_h=180,
            include_header=True,
        )
    if tile_start.shape[0] != tile_end.shape[0]:
        h = min(tile_start.shape[0], tile_end.shape[0])
        tile_start = tile_start[:h]
        tile_end = tile_end[:h]
    sep = np.full((tile_start.shape[0], 4, 3), 40, dtype=np.uint8)
    combined = np.ascontiguousarray(np.hstack([tile_start, sep, tile_end]), dtype=np.uint8)
    jpeg_bytes = encode_jpeg_bytes(combined, quality=85)
    atomic_write(out_path, jpeg_bytes)


def create_few_shot_example(
    repo_or_workspace: Repository | Path | str | None = None,
    *,
    repo: Repository | None = None,
    repository: Repository | None = None,
    workspace: Path | str | None = None,
    workspace_dir: Path | str | None = None,
    config: ConfigManager | None = None,
    cfg: ConfigManager | None = None,
    config_dir: Path | str | None = None,
    episode_id: str,
    t_start: float,
    t_end: float,
    scope: str = "global",
    task_id: str | None = None,
    label_id: str | None = None,
    error_type: DiagnosisErrorType = "fp_overflag",
    vlm_wrong_json: dict[str, Any] | None = None,
    vlm_wrong: dict[str, Any] | None = None,
    human_right_json: dict[str, Any] | None = None,
    human_right: dict[str, Any] | None = None,
    rationale: str,
    capture_tile: bool | None = None,
    with_image: bool | None = None,
    enabled: bool = True,
    example_id: str | None = None,
) -> FewShotItem:
    """Create a FewShotItem with optional 2x2 start-vs-end JPEG crop and persist to SQLite."""
    ts = float(t_start)
    te = float(t_end)
    if ts < 0.0:
        raise ValueError(f"t_start ({ts}) must be >= 0.0")
    if te <= ts:
        raise ValueError(f"t_end ({te}) must be > t_start ({ts})")

    eff_repo, owns_repo = _resolve_repo(
        repo_or_workspace,
        repo=repo,
        repository=repository,
        workspace=workspace,
        workspace_dir=workspace_dir,
    )
    try:
        meta = eff_repo.get_episode_meta(episode_id)
        ws = eff_repo.workspace

        eff_task_id = task_id
        if eff_task_id is None and scope.startswith("task:"):
            eff_task_id = scope.split(":", 1)[1]
        eff_label_id = label_id
        if eff_label_id is None and scope.startswith("label:"):
            eff_label_id = scope.split(":", 1)[1]

        wrong_payload = dict(
            vlm_wrong_json if vlm_wrong_json is not None else (vlm_wrong or {})
        )
        right_payload = dict(
            human_right_json if human_right_json is not None else (human_right or {})
        )

        do_capture = True
        if capture_tile is not None:
            do_capture = bool(capture_tile)
        elif with_image is not None:
            do_capture = bool(with_image)

        ex_id = example_id or f"fs_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        tile_rel_path: str | None = None
        if do_capture:
            cfg_mgr = (
                config
                or cfg
                or ConfigManager(
                    config_dir=Path(config_dir) if config_dir is not None else None,
                    workspace_dir=ws,
                )
            )
            rel = Path("artifacts") / "few_shots" / f"{ex_id}.jpg"
            _render_start_end_comparison_jpeg(
                repo=eff_repo,
                episode_id=meta.episode_id,
                t_start=ts,
                t_end=te,
                cfg_mgr=cfg_mgr,
                out_path=ws / rel,
            )
            tile_rel_path = rel.as_posix()

        item = FewShotItem(
            example_id=ex_id,
            scope=scope,
            task_id=eff_task_id,
            label_id=eff_label_id,
            error_type=error_type,
            episode_id=episode_id,
            t_start=ts,
            t_end=te,
            tile_image_path=tile_rel_path,
            vlm_wrong_json=wrong_payload,
            human_right_json=right_payload,
            rationale=rationale.strip(),
            enabled=bool(enabled),
            created_at=time.time(),
        )
        eff_repo.save_few_shot(item)
        return item
    finally:
        if owns_repo:
            eff_repo.close()


add_few_shot_example = create_few_shot_example


def _matches_scope(
    fs: FewShotItem,
    *,
    task_id: str | None,
    label_set: set[str] | None,
) -> bool:
    if not fs.enabled:
        return False
    if fs.scope == "global":
        return True
    if fs.scope.startswith("task:"):
        sc_task = fs.scope.split(":", 1)[1]
        return task_id is None or sc_task == task_id
    if fs.scope.startswith("label:"):
        sc_lbl = fs.scope.split(":", 1)[1]
        task_ok = task_id is None or fs.task_id is None or fs.task_id == task_id
        lbl_ok = (
            label_set is None
            or sc_lbl in label_set
            or (fs.label_id is not None and fs.label_id in label_set)
        )
        return task_ok and lbl_ok
    return task_id is None or fs.task_id is None or fs.task_id == task_id


def retrieve_few_shots(
    repo_or_workspace: Repository | Path | str | None = None,
    *,
    repo: Repository | None = None,
    repository: Repository | None = None,
    workspace: Path | str | None = None,
    workspace_dir: Path | str | None = None,
    task_id: str | None = None,
    label_ids: Sequence[str] | None = None,
    top_k: int = 4,
    k: int | None = None,
    max_tokens: int = 4000,
) -> list[FewShotItem]:
    """Retrieve enabled Few-Shot items matching global/task/label within top_k and max_tokens."""
    eff_k = int(k if k is not None else top_k)
    eff_max_tok = int(max_tokens)
    if eff_k <= 0 or eff_max_tok <= 0:
        return []

    eff_repo, owns_repo = _resolve_repo(
        repo_or_workspace,
        repo=repo,
        repository=repository,
        workspace=workspace,
        workspace_dir=workspace_dir,
    )
    try:
        all_enabled = eff_repo.list_few_shots(enabled_only=True, limit=2000)
    finally:
        if owns_repo:
            eff_repo.close()

    label_set = set(label_ids) if label_ids is not None else None
    filtered = [
        fs
        for fs in all_enabled
        if _matches_scope(fs, task_id=task_id, label_set=label_set)
    ]

    def _priority(fs: FewShotItem) -> tuple[int, float, str]:
        if task_id and (
            fs.scope == f"task:{task_id}"
            or (fs.task_id == task_id and fs.scope != "global")
        ):
            tier = 0
        elif fs.scope.startswith("label:"):
            tier = 1
        elif fs.scope == "global":
            tier = 2
        else:
            tier = 3
        return (tier, -float(fs.created_at), fs.example_id)

    ordered = sorted(filtered, key=_priority)
    selected: list[FewShotItem] = []
    used_tokens = 0
    for fs in ordered:
        tok = _estimate_tokens(_format_single_few_shot(fs))
        if used_tokens + tok > eff_max_tok:
            continue
        selected.append(fs)
        used_tokens += tok
        if len(selected) >= eff_k:
            break
    return selected


get_relevant_few_shots = retrieve_few_shots
select_few_shots = retrieve_few_shots


class FewShotBook:
    """Object-oriented helper around Few-Shot creation and scoped retrieval."""

    def __init__(
        self,
        repo_or_workspace: Repository | Path | str | None = None,
        *,
        repo: Repository | None = None,
        repository: Repository | None = None,
        workspace: Path | str | None = None,
        workspace_dir: Path | str | None = None,
        config: ConfigManager | None = None,
    ) -> None:
        self._target = repo or repository or workspace or workspace_dir or repo_or_workspace
        self._config = config

    def add(self, **kwargs: Any) -> FewShotItem:
        if self._config is not None and "config" not in kwargs and "cfg" not in kwargs:
            kwargs["config"] = self._config
        return create_few_shot_example(self._target, **kwargs)

    create = add

    def retrieve(
        self,
        *,
        task_id: str | None = None,
        label_ids: Sequence[str] | None = None,
        top_k: int = 4,
        k: int | None = None,
        max_tokens: int = 4000,
    ) -> list[FewShotItem]:
        return retrieve_few_shots(
            self._target,
            task_id=task_id,
            label_ids=label_ids,
            top_k=top_k,
            k=k,
            max_tokens=max_tokens,
        )

    select = retrieve
