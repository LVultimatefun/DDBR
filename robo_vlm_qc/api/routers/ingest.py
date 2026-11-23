"""Read-only directory browser & adapter preflight endpoints with --allowed-data-roots containment."""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from ...io.base import SKIP_DIRS, infer_camera_role, is_camera_filename
from ...io.h5_raw_v3 import H5RawV3Adapter
from ...io.lerobot import LeRobotAdapter
from ...io.video_dir import VIDEO_EXTS, _has_structured_ancestor
from ..auth import AuthContext, get_app_state, require_lead
from ..state import AppState

router = APIRouter(prefix="/api/ingest", tags=["ingest"])


def is_within_allowed_roots(
    resolved_path: Path, allowed_roots: list[Path] | None
) -> bool:
    """Return True if allowed_roots is None or resolved_path lies inside at least one root."""
    if allowed_roots is None:
        return True
    if not allowed_roots:
        return False
    rp = resolved_path.resolve()
    return any(rp.is_relative_to(r.resolve()) for r in allowed_roots)


def resolve_allowed_path(
    raw_path: str | Path, allowed_roots: list[Path] | None
) -> Path:
    """Resolve raw_path (expanding ~ and symlinks) and enforce allowed_roots containment (403 on escape)."""
    s = str(raw_path)
    if "\x00" in s:
        raise HTTPException(status_code=400, detail="Invalid null byte in path")
    try:
        resolved = Path(s).expanduser().resolve()
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid path {s!r}: {exc}") from exc

    if allowed_roots is not None:
        if not allowed_roots:
            raise HTTPException(
                status_code=403,
                detail="No allowed data roots configured on server",
            )
        if not any(resolved.is_relative_to(r.resolve()) for r in allowed_roots):
            raise HTTPException(
                status_code=403,
                detail=f"Path {s!r} is outside allowed_data_roots",
            )
    return resolved


def preflight_detect_adapters(
    target: Path,
    allowed_roots: list[Path] | None,
    *,
    max_depth: int = 4,
    max_entries: int = 5000,
) -> list[dict[str, Any]]:
    """Bounded iterative os.scandir preflight detecting episode counts per adapter without hanging."""
    counts: dict[str, int] = {
        "h5_raw_v3": 0,
        "lerobot": 0,
        "video_dir": 0,
    }
    h5_ad = H5RawV3Adapter()
    lr_ad = LeRobotAdapter()

    if target.is_file():
        if (
            target.suffix.lower() in (".h5", ".hdf5")
            and ".partial" not in target.name
            and h5_ad.can_read(target)
        ):
            counts["h5_raw_v3"] += 1
        elif (
            target.suffix.lower() in VIDEO_EXTS
            and ".partial" not in target.name
            and not _has_structured_ancestor(target)
        ):
            counts["video_dir"] += 1
        return [
            {"adapter_type": name, "episode_count": counts[name]}
            for name in ("h5_raw_v3", "lerobot", "video_dir")
            if counts[name] > 0
        ]

    visited_inodes: set[tuple[int, int]] = set()
    queue: deque[tuple[Path, int]] = deque([(target, 0)])
    entries_seen = 0

    while queue and entries_seen < max_entries:
        cur_dir, depth = queue.popleft()
        try:
            st = cur_dir.stat()
        except OSError:
            continue
        inode_key = (int(st.st_dev), int(st.st_ino))
        if inode_key in visited_inodes:
            continue
        visited_inodes.add(inode_key)

        if lr_ad.can_read(cur_dir):
            try:
                counts["lerobot"] += len(lr_ad.discover(cur_dir))
            except Exception:
                pass
            continue

        direct_videos: list[Path] = []
        subdirs: list[Path] = []
        try:
            with os.scandir(cur_dir) as it:
                for ent in it:
                    entries_seen += 1
                    if entries_seen > max_entries:
                        break
                    name = ent.name
                    if name.startswith(".") or name in SKIP_DIRS:
                        continue
                    try:
                        rp = Path(ent.path).resolve(strict=True)
                    except OSError:
                        continue
                    if not is_within_allowed_roots(rp, allowed_roots):
                        continue
                    try:
                        if rp.is_dir():
                            subdirs.append(rp)
                        elif rp.is_file():
                            suf = rp.suffix.lower()
                            if suf in (".h5", ".hdf5") and ".partial" not in rp.name:
                                if h5_ad.can_read(rp):
                                    counts["h5_raw_v3"] += 1
                            elif suf in VIDEO_EXTS and ".partial" not in rp.name:
                                direct_videos.append(rp)
                    except OSError:
                        continue
        except (PermissionError, OSError):
            continue

        if direct_videos and not _has_structured_ancestor(cur_dir):
            has_manifest = (cur_dir / "episode.json").is_file() or (
                cur_dir / "cameras.json"
            ).is_file()
            if has_manifest or len(direct_videos) == 1:
                counts["video_dir"] += 1
            else:
                all_cam = all(is_camera_filename(v.stem) for v in direct_videos)
                roles = [infer_camera_role(v.stem) for v in direct_videos]
                if all_cam and len(roles) == len(set(roles)):
                    counts["video_dir"] += 1
                else:
                    counts["video_dir"] += len(direct_videos)

        if depth < max_depth and entries_seen < max_entries:
            for sd in sorted(subdirs):
                queue.append((sd, depth + 1))

    return [
        {"adapter_type": name, "episode_count": counts[name]}
        for name in ("h5_raw_v3", "lerobot", "video_dir")
        if counts[name] > 0
    ]


@router.get("/browse")
def browse_ingest_directory(
    path: str | None = Query(default=None),
    auth: AuthContext = Depends(require_lead),
    state: AppState = Depends(get_app_state),
) -> dict[str, Any]:
    _ = auth
    allowed_roots = state.allowed_data_roots
    if path is None or not path.strip():
        if allowed_roots is not None:
            if not allowed_roots:
                raise HTTPException(
                    status_code=403,
                    detail="No allowed data roots configured on server",
                )
            target = allowed_roots[0].resolve()
        else:
            target = Path.cwd().resolve()
    else:
        target = resolve_allowed_path(path.strip(), allowed_roots)

    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {target}")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {target}")

    parent = target.parent
    if parent == target:
        parent_path: str | None = None
    elif not is_within_allowed_roots(parent, allowed_roots):
        parent_path = None
    else:
        parent_path = str(parent)

    entries: list[dict[str, Any]] = []
    try:
        with os.scandir(target) as it:
            for ent in it:
                if ent.name.startswith("."):
                    continue
                try:
                    resolved_child = Path(ent.path).resolve(strict=True)
                except OSError:
                    continue
                if not is_within_allowed_roots(resolved_child, allowed_roots):
                    continue
                try:
                    is_dir = resolved_child.is_dir()
                except OSError:
                    continue
                entries.append(
                    {
                        "name": ent.name,
                        "path": str(resolved_child),
                        "is_dir": is_dir,
                    }
                )
    except PermissionError as exc:
        raise HTTPException(
            status_code=403, detail=f"Permission denied reading {target}: {exc}"
        ) from exc

    entries.sort(key=lambda e: (not e["is_dir"], e["name"].lower()))
    detected_adapters = preflight_detect_adapters(target, allowed_roots)

    return {
        "current_path": str(target),
        "parent_path": parent_path,
        "allowed_roots": [str(r.resolve()) for r in allowed_roots]
        if allowed_roots is not None
        else [],
        "entries": entries,
        "detected_adapters": detected_adapters,
    }
