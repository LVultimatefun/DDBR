"""SQLite WAL connection pool, schema migrations, and write serialization."""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA_VERSION = 1

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    INTEGER PRIMARY KEY,
    applied_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS episodes (
    episode_id        TEXT PRIMARY KEY,
    dataset           TEXT NOT NULL,
    task_id           TEXT NOT NULL,
    adapter_type      TEXT NOT NULL,
    source_uri        TEXT NOT NULL,
    source_mtime      REAL NOT NULL,
    duration_s        REAL NOT NULL,
    fps               REAL NOT NULL,
    has_kinematics    INTEGER NOT NULL,
    camera_roles_json TEXT NOT NULL,
    meta_json         TEXT NOT NULL DEFAULT '{}',
    status            TEXT NOT NULL,
    active_run_id     TEXT,
    active_rev_id     TEXT,
    claimed_by        TEXT,
    claim_expires_at  REAL,
    resolved_passed   INTEGER,
    cut_ratio         REAL,
    mask_ratio        REAL,
    badcase_count     INTEGER DEFAULT 0,
    unknown_count     INTEGER DEFAULT 0,
    updated_at        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodes_status ON episodes(status);
CREATE INDEX IF NOT EXISTS idx_episodes_dataset_task ON episodes(dataset, task_id);
CREATE INDEX IF NOT EXISTS idx_episodes_claim ON episodes(claimed_by, claim_expires_at);

CREATE TABLE IF NOT EXISTS vlm_runs (
    run_id             TEXT PRIMARY KEY,
    episode_id         TEXT NOT NULL REFERENCES episodes(episode_id),
    is_eval_sandbox    INTEGER NOT NULL DEFAULT 0,
    model_routing_json TEXT NOT NULL,
    prompt_version     TEXT NOT NULL,
    label_version      TEXT NOT NULL,
    task_spec_version  TEXT NOT NULL,
    few_shot_pack_id   TEXT,
    artifact_path      TEXT NOT NULL,
    status             TEXT NOT NULL,
    tokens_total       INTEGER DEFAULT 0,
    elapsed_s          REAL DEFAULT 0,
    error_message      TEXT,
    created_at         REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_vlm_runs_episode ON vlm_runs(episode_id, is_eval_sandbox, created_at);

CREATE TABLE IF NOT EXISTS reviews (
    rev_id          TEXT PRIMARY KEY,
    episode_id      TEXT NOT NULL REFERENCES episodes(episode_id),
    base_run_id     TEXT NOT NULL REFERENCES vlm_runs(run_id),
    author          TEXT NOT NULL,
    role            TEXT NOT NULL,
    action          TEXT NOT NULL,
    artifact_path   TEXT NOT NULL,
    diff_stats_json TEXT NOT NULL,
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reviews_episode ON reviews(episode_id, created_at);

CREATE TABLE IF NOT EXISTS few_shot_examples (
    example_id       TEXT PRIMARY KEY,
    scope            TEXT NOT NULL,
    task_id          TEXT,
    label_id         TEXT,
    error_type       TEXT NOT NULL,
    episode_id       TEXT NOT NULL,
    t_start          REAL NOT NULL,
    t_end            REAL NOT NULL,
    tile_image_path  TEXT,
    vlm_wrong_json   TEXT NOT NULL,
    human_right_json TEXT NOT NULL,
    rationale        TEXT NOT NULL,
    enabled          INTEGER NOT NULL DEFAULT 1,
    created_at       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_few_shot_scope ON few_shot_examples(scope, enabled);

CREATE TABLE IF NOT EXISTS jobs (
    job_id         TEXT PRIMARY KEY,
    job_type       TEXT NOT NULL,
    status         TEXT NOT NULL,
    params_json    TEXT NOT NULL,
    progress_done  INTEGER DEFAULT 0,
    progress_total INTEGER DEFAULT 0,
    summary_json   TEXT,
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at);
"""


class DatabasePool:
    """Thread-safe SQLite WAL connection pool with serialized BEGIN IMMEDIATE writes."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._tls = threading.local()
        self._write_lock = threading.RLock()
        self._conn_lock = threading.Lock()
        self._conns: set[sqlite3.Connection] = set()
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._tls, "conn", None)
        if conn is not None:
            return conn
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=10.0,
            isolation_level=None,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        self._tls.conn = conn
        with self._conn_lock:
            self._conns.add(conn)
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock:
            conn = self.connect()
            conn.execute("BEGIN IMMEDIATE;")
            try:
                yield conn
            except BaseException:
                try:
                    conn.execute("ROLLBACK;")
                except sqlite3.Error:
                    pass
                raise
            else:
                conn.execute("COMMIT;")

    def initialize(self) -> None:
        with self.transaction() as conn:
            conn.executescript(SCHEMA_DDL)
            # executescript issues implicit COMMIT in sqlite3; re-open tx for migration row
            conn.execute("BEGIN IMMEDIATE;")
            row = conn.execute(
                "SELECT version FROM schema_migrations WHERE version = ?",
                (SCHEMA_VERSION,),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                    (SCHEMA_VERSION, time.time()),
                )

    def close_thread_connection(self) -> None:
        """Close and release the current thread's SQLite connection, if open."""
        conn: sqlite3.Connection | None = getattr(self._tls, "conn", None)
        if conn is not None:
            delattr(self._tls, "conn")
            with self._conn_lock:
                self._conns.discard(conn)
            try:
                conn.close()
            except sqlite3.Error:
                pass

    def close_all(self) -> None:
        with self._conn_lock:
            for conn in list(self._conns):
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
            self._conns.clear()
        if hasattr(self._tls, "conn"):
            delattr(self._tls, "conn")
