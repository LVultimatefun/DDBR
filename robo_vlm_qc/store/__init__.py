"""Persistence layer: SQLite WAL, atomic JSON artifacts, and repository."""

from .artifacts import ArtifactStore, archive_before, atomic_write
from .db import DatabasePool
from .repository import (
    ALLOWED_TRANSITIONS,
    AuthorizationError,
    ClaimConflictError,
    GtFrozenError,
    InvalidTransitionError,
    NotFoundError,
    ReadOnlyFieldError,
    Repository,
    RepositoryError,
    WatchCoverageError,
    compute_review_diff,
)

__all__ = [
    "ALLOWED_TRANSITIONS",
    "ArtifactStore",
    "AuthorizationError",
    "ClaimConflictError",
    "DatabasePool",
    "GtFrozenError",
    "InvalidTransitionError",
    "NotFoundError",
    "ReadOnlyFieldError",
    "Repository",
    "RepositoryError",
    "WatchCoverageError",
    "archive_before",
    "atomic_write",
    "compute_review_diff",
]
