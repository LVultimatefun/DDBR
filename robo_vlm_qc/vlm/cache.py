"""Content-addressed stage cache (`stage_cache/<sha256[:2]>/<sha256>.json`)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..config import canonical_sha256
from ..store.artifacts import atomic_write

_SAFE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]{8,128}$")


class StageCache:
    """Content-addressed atomic JSON stage cache."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        *,
        workspace: Path | None = None,
        workspace_dir: Path | None = None,
    ) -> None:
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir).resolve()
        elif workspace is not None or workspace_dir is not None:
            ws = Path(workspace if workspace is not None else workspace_dir).resolve()  # type: ignore[arg-type]
            self.cache_dir = ws / "stage_cache"
        else:
            raise ValueError("StageCache requires cache_dir or workspace")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_key(
        *,
        episode_fingerprint: str,
        stage_basis: str,
        upstream_hash: str = "",
        extra: Any = None,
    ) -> str:
        """Compute deterministic SHA256 stage cache key."""
        return canonical_sha256(
            {
                "episode_fingerprint": episode_fingerprint,
                "upstream_hash": upstream_hash,
                "stage_basis": stage_basis,
                "extra": extra,
            }
        )

    def path_for(self, key: str) -> Path:
        if not _SAFE_KEY_RE.match(key):
            raise ValueError(f"Invalid StageCache key: {key!r}")
        return self.cache_dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self.path_for(key)
        if not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(raw, dict):
            return None
        return raw

    def put(self, key: str, payload: dict[str, Any]) -> Path:
        path = self.path_for(key)
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        atomic_write(path, text)
        return path
