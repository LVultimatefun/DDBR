"""Shared FastAPI application state, SSE broker, and media view opener."""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import ConfigManager, default_config_dir
from ..io import EpisodeHandle, EpisodeView, KinematicsTrack, PreviewCache, get_adapter
from ..models.episode import EpisodeMeta
from ..store import Repository


class SSEBroker:
    """Thread-safe ring-buffer broker for server-sent events."""

    def __init__(self, maxlen: int = 500) -> None:
        self._lock = threading.Lock()
        self._events: deque[tuple[int, dict[str, Any]]] = deque(maxlen=maxlen)
        self._seq = 0

    def publish(self, event: dict[str, Any]) -> int:
        payload = dict(event)
        payload.setdefault("ts", time.time())
        with self._lock:
            self._seq += 1
            seq = self._seq
            self._events.append((seq, payload))
            return seq

    def latest_seq(self) -> int:
        with self._lock:
            return self._seq

    def since(self, last_seq: int) -> tuple[int, list[dict[str, Any]]]:
        with self._lock:
            items = [ev for seq, ev in self._events if seq > last_seq]
            return self._seq, items


@dataclass
class AuthConfig:
    reviewer_token: str | None = None
    lead_token: str | None = None
    no_auth: bool = False


@dataclass
class _JobControl:
    cancel_event: threading.Event = field(default_factory=threading.Event)
    callbacks: list[Callable[[], None]] = field(default_factory=list)
    prev_statuses: dict[str, str] = field(default_factory=dict)


@dataclass
class AppState:
    workspace: Path
    config_dir: Path
    repo: Repository
    preview_cache: PreviewCache
    auth: AuthConfig
    executor: ThreadPoolExecutor
    sse: SSEBroker = field(default_factory=SSEBroker)
    web_dist: Path = field(default_factory=lambda: resolve_default_web_dist(None))
    allowed_data_roots: list[Path] | None = None
    kinematics_cache_max_items: int = 16
    _kin_cache_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _kin_cache: OrderedDict[
        tuple[str, str, float], tuple[KinematicsTrack | None, float, float]
    ] = field(default_factory=OrderedDict, init=False, repr=False)
    _cfg_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _cached_cfg: ConfigManager | None = field(
        default=None, init=False, repr=False
    )
    _cached_cfg_sig: tuple[tuple[str, int, int], ...] | None = field(
        default=None, init=False, repr=False
    )
    _jobs_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _active_jobs: dict[str, _JobControl] = field(
        default_factory=dict, init=False, repr=False
    )
    _active_vlm_episodes: dict[str, str] = field(
        default_factory=dict, init=False, repr=False
    )

    def try_reserve_vlm_episodes(
        self, episode_ids: Sequence[str], owner_id: str
    ) -> list[str]:
        unique_ids = list(dict.fromkeys(str(e) for e in episode_ids if e))
        with self._jobs_lock:
            busy = [
                eid
                for eid in unique_ids
                if eid in self._active_vlm_episodes
                and self._active_vlm_episodes[eid] != owner_id
            ]
            if busy:
                return busy
            for eid in unique_ids:
                self._active_vlm_episodes[eid] = owner_id
            return []

    def rebind_vlm_episodes(self, old_owner: str, new_owner: str) -> None:
        with self._jobs_lock:
            for eid, cur_owner in list(self._active_vlm_episodes.items()):
                if cur_owner == old_owner:
                    self._active_vlm_episodes[eid] = new_owner

    def release_vlm_episodes(self, owner_id: str) -> None:
        with self._jobs_lock:
            for eid, cur_owner in list(self._active_vlm_episodes.items()):
                if cur_owner == owner_id:
                    del self._active_vlm_episodes[eid]

    def can_job_restore_episode(self, job_id: str, episode_id: str) -> bool:
        with self._jobs_lock:
            owner = self._active_vlm_episodes.get(episode_id)
            return owner is None or owner == job_id

    def register_job_control(
        self,
        job_id: str,
        *,
        prev_statuses: dict[str, str] | None = None,
    ) -> threading.Event:
        with self._jobs_lock:
            ctrl = self._active_jobs.get(job_id)
            if ctrl is None:
                ctrl = _JobControl()
                self._active_jobs[job_id] = ctrl
            if prev_statuses:
                ctrl.prev_statuses.update(prev_statuses)
            return ctrl.cancel_event

    def get_job_cancel_event(self, job_id: str) -> threading.Event:
        return self.register_job_control(job_id)

    def get_job_prev_statuses(self, job_id: str) -> dict[str, str]:
        with self._jobs_lock:
            ctrl = self._active_jobs.get(job_id)
            return dict(ctrl.prev_statuses) if ctrl is not None else {}

    def has_job_control(self, job_id: str) -> bool:
        with self._jobs_lock:
            return job_id in self._active_jobs

    def add_job_cancel_callback(
        self, job_id: str, callback: Callable[[], None]
    ) -> None:
        with self._jobs_lock:
            ctrl = self._active_jobs.get(job_id)
            if ctrl is None:
                ctrl = _JobControl()
                self._active_jobs[job_id] = ctrl
            already_cancelled = ctrl.cancel_event.is_set()
            if not already_cancelled:
                ctrl.callbacks.append(callback)
        if already_cancelled:
            with contextlib.suppress(Exception):
                callback()

    def trigger_job_cancel(self, job_id: str) -> dict[str, str]:
        with self._jobs_lock:
            ctrl = self._active_jobs.get(job_id)
            if ctrl is None:
                return {}
            ctrl.cancel_event.set()
            callbacks = list(ctrl.callbacks)
            ctrl.callbacks.clear()
            prev = dict(ctrl.prev_statuses)
        for cb in callbacks:
            with contextlib.suppress(Exception):
                cb()
        return prev

    def clear_job_control(self, job_id: str) -> dict[str, str]:
        with self._jobs_lock:
            ctrl = self._active_jobs.pop(job_id, None)
            for eid, cur_owner in list(self._active_vlm_episodes.items()):
                if cur_owner == job_id:
                    del self._active_vlm_episodes[eid]
            return dict(ctrl.prev_statuses) if ctrl is not None else {}

    def cancel_all_jobs(self) -> None:
        with self._jobs_lock:
            job_ids = list(self._active_jobs.keys())
        for jid in job_ids:
            self.trigger_job_cancel(jid)

    def _config_files_signature(self) -> tuple[tuple[str, int, int], ...]:
        sig_items: list[tuple[str, int, int]] = []
        roots = [self.config_dir, self.workspace / "configs_override"]
        for root in roots:
            if not root.is_dir():
                continue
            for p in sorted(root.rglob("*.yaml")):
                try:
                    st = p.stat()
                    sig_items.append((str(p), int(st.st_mtime_ns), int(st.st_size)))
                except OSError:
                    continue
        return tuple(sig_items)

    def get_config(self) -> ConfigManager:
        with self._cfg_lock:
            sig = self._config_files_signature()
            if self._cached_cfg is not None and self._cached_cfg_sig == sig:
                return self._cached_cfg
            cfg = ConfigManager(config_dir=self.config_dir, workspace_dir=self.workspace)
            self._cached_cfg = cfg
            self._cached_cfg_sig = sig
            return cfg

    def invalidate_config(self) -> None:
        with self._cfg_lock:
            self._cached_cfg = None
            self._cached_cfg_sig = None

    def open_view(self, meta: EpisodeMeta, cfg_mgr: ConfigManager | None = None) -> EpisodeView:
        cfg = cfg_mgr or self.get_config()
        return open_view_from_meta(meta, cfg)

    def get_or_load_kinematics(
        self, meta: EpisodeMeta, cfg_mgr: ConfigManager | None = None
    ) -> tuple[KinematicsTrack | None, float, float]:
        eff_mtime = float(meta.source_mtime)
        try:
            src_p = Path(meta.source_uri)
            if src_p.exists():
                eff_mtime = max(eff_mtime, float(src_p.stat().st_mtime))
        except OSError:
            pass
        key = (meta.episode_id, meta.source_uri, eff_mtime)
        with self._kin_cache_lock:
            hit = self._kin_cache.get(key)
            if hit is not None:
                self._kin_cache.move_to_end(key)
                return hit

        cfg = cfg_mgr or self.get_config()
        with self.open_view(meta, cfg) as view:
            entry: tuple[KinematicsTrack | None, float, float] = (
                view.kinematics,
                float(view.duration_s),
                float(view.fps),
            )

        with self._kin_cache_lock:
            stale_keys = [
                k for k in self._kin_cache if k[0] == meta.episode_id and k != key
            ]
            for k in stale_keys:
                self._kin_cache.pop(k, None)
            self._kin_cache[key] = entry
            self._kin_cache.move_to_end(key)
            limit = max(1, int(self.kinematics_cache_max_items))
            while len(self._kin_cache) > limit:
                self._kin_cache.popitem(last=False)
        return entry

    def invalidate_kinematics(self, episode_id: str | None = None) -> None:
        with self._kin_cache_lock:
            if episode_id is None:
                self._kin_cache.clear()
            else:
                for k in [k for k in self._kin_cache if k[0] == episode_id]:
                    self._kin_cache.pop(k, None)


def open_view_from_meta(meta: EpisodeMeta, cfg_mgr: ConfigManager) -> EpisodeView:
    """Re-open a read-only EpisodeView from persisted EpisodeMeta."""
    src_path = Path(meta.source_uri)
    if not src_path.exists():
        raise FileNotFoundError(f"Episode source not found: {meta.source_uri}")
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


def resolve_default_config_dir(config_dir: Path | None) -> Path:
    return Path(config_dir).resolve() if config_dir is not None else default_config_dir()


def resolve_default_web_dist(web_dist_dir: Path | None = None) -> Path:
    if web_dist_dir is not None:
        return Path(web_dist_dir).expanduser().resolve()
    env_dist = os.environ.get("ROBO_VLM_QC_WEB_DIST", "").strip()
    if env_dist:
        return Path(env_dist).expanduser().resolve()
    pkg_dir = Path(__file__).resolve().parent.parent
    candidates = (
        pkg_dir.parent / "web" / "dist",
        pkg_dir / "_web_dist",
        Path.cwd() / "web" / "dist",
    )
    for cand in candidates:
        if (cand / "index.html").is_file():
            return cand.resolve()
    return candidates[0].resolve()
