"""Atomic JSON/text persistence, immutable artifacts, and .history snapshots."""

from __future__ import annotations

import base64
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def _sync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write(path: Path, data: str | bytes) -> None:
    """Write text or bytes atomically via mkstemp + fsync + os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_bytes = isinstance(data, (bytes, bytearray))
    mode = "wb" if is_bytes else "w"
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        kwargs: dict[str, Any] = {} if is_bytes else {"encoding": "utf-8"}
        with os.fdopen(fd, mode, **kwargs) as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        _sync_dir(path.parent)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def archive_before(workspace: Path, path: Path, reason: str) -> Path | None:
    """Save a pre-modification snapshot under <workspace>/.history/."""
    workspace = Path(workspace).resolve()
    path = Path(path).resolve()
    if not path.is_file():
        return None
    raw = path.read_bytes()
    try:
        content = raw.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        content = base64.b64encode(raw).decode("ascii")
        encoding = "base64"

    try:
        rel = path.relative_to(workspace)
    except ValueError:
        rel = Path(path.name)

    snap_dir = workspace / ".history" / rel.parent / rel.stem
    snap_name = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.json"
    snap_path = snap_dir / snap_name
    payload = {
        "path": str(path),
        "relative_path": rel.as_posix(),
        "created_at": time.time(),
        "reason": reason,
        "encoding": encoding,
        "content": content,
    }
    atomic_write(snap_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return snap_path


class ArtifactStore:
    """Manages immutable JSON artifacts and .history-protected config overrides."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def _resolve_inside(self, rel_path: str | Path) -> Path:
        p = Path(rel_path)
        if p.is_absolute():
            resolved = p.resolve()
        else:
            resolved = (self.workspace / p).resolve()
        try:
            resolved.relative_to(self.workspace)
        except ValueError as exc:
            raise ValueError(f"Path {rel_path!r} escapes workspace {self.workspace}") from exc
        return resolved

    @staticmethod
    def vlm_run_relpath(dataset: str, stem: str, run_id: str) -> Path:
        return Path("artifacts") / "vlm_runs" / dataset / stem / f"{run_id}.json"

    @staticmethod
    def review_relpath(dataset: str, stem: str, rev_id: str) -> Path:
        return Path("artifacts") / "reviews" / dataset / stem / f"{rev_id}.json"

    def write_immutable_json(
        self, rel_path: str | Path, payload: BaseModel | dict[str, Any]
    ) -> Path:
        target = self._resolve_inside(rel_path)
        if target.exists():
            raise FileExistsError(
                f"Refusing to overwrite immutable artifact: {target}"
            )
        if isinstance(payload, BaseModel):
            text = payload.model_dump_json(indent=2) + "\n"
        else:
            text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        atomic_write(target, text)
        return target

    def read_json(self, rel_path: str | Path) -> dict[str, Any]:
        target = self._resolve_inside(rel_path)
        return json.loads(target.read_text(encoding="utf-8"))

    def write_config_override(
        self,
        rel_path: str | Path,
        content: str | dict[str, Any],
        reason: str = "config_update",
    ) -> Path:
        target = self._resolve_inside(Path("configs_override") / rel_path)
        archive_before(self.workspace, target, reason=reason)
        if isinstance(content, dict):
            text = json.dumps(content, ensure_ascii=False, indent=2) + "\n"
        else:
            text = content
        atomic_write(target, text)
        return target
