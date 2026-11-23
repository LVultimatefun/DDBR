"""Read-only InputAdapter for generic single/multi-camera pure-video directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ..models.episode import CameraSpec, EpisodeMeta, KinematicsSpec
from .base import (
    LABEL_DIRS,
    SKIP_DIRS,
    EpisodeHandle,
    EpisodeView,
    InputAdapter,
    build_camera_roles,
    compute_episode_fingerprint,
    dataset_of,
    infer_camera_role,
    is_camera_filename,
    resolve_task_id,
    validate_id_part,
)
from .media import Mp4Reader, probe_video_cv2, time_to_frame_idx

VIDEO_EXTS: frozenset[str] = frozenset({".mp4", ".mov", ".mkv", ".avi"})


def _has_structured_ancestor(path: Path) -> bool:
    cur = path.resolve()
    if cur.is_file():
        cur = cur.parent
    for anc in (cur, *cur.parents):
        if (anc / "meta" / "info.json").is_file():
            return True
    return False


def _direct_videos(d: Path) -> list[Path]:
    vids: list[Path] = []
    for p in sorted(d.iterdir()):
        if (
            p.is_file()
            and p.suffix.lower() in VIDEO_EXTS
            and ".partial" not in p.name
            and not p.name.startswith(".")
        ):
            vids.append(p)
    return vids


class VideoDirMediaSource:
    """Thread-safe MediaSource over one or more MP4 files (zero kinematics)."""

    def __init__(
        self,
        *,
        cameras: list[CameraSpec],
        cam_entries: dict[str, dict[str, Any]],
        fps: float,
        duration_s: float,
    ) -> None:
        self.cameras = cameras
        self._cam_entries = cam_entries
        self.fps = float(fps)
        self.duration_s = float(duration_s)
        self._mp4 = Mp4Reader()
        self._role_to_cam: dict[str, str] = {}
        for c in cameras:
            if c.role not in self._role_to_cam:
                self._role_to_cam[c.role] = c.name

    def _resolve_camera(self, camera: str) -> str:
        if camera in self._cam_entries:
            return camera
        if camera in self._role_to_cam:
            return self._role_to_cam[camera]
        raise KeyError(
            f"Unknown camera {camera!r}; available: {list(self._cam_entries)}"
        )

    def read_frame_rgb(self, camera: str, t_s: float) -> np.ndarray:
        cam = self._resolve_camera(camera)
        ent = self._cam_entries[cam]
        t_clamped = max(0.0, min(ent["duration_s"], float(t_s)))
        f_idx = time_to_frame_idx(t_clamped, ent["fps"], ent["frame_count"])
        return self._mp4.read_frame_rgb(ent["path"], f_idx)

    def read_frames_batch(
        self, camera: str, times_s: list[float]
    ) -> list[np.ndarray]:
        if not times_s:
            return []
        cam = self._resolve_camera(camera)
        ent = self._cam_entries[cam]
        indices = [
            time_to_frame_idx(
                max(0.0, min(ent["duration_s"], float(t))),
                ent["fps"],
                ent["frame_count"],
            )
            for t in times_s
        ]
        return self._mp4.read_frames_batch(ent["path"], indices)

    def close(self) -> None:
        self._mp4.close()


class VideoDirAdapter(InputAdapter):
    """InputAdapter for pure-video single-cam files or multi-cam directories."""

    name: ClassVar[str] = "video_dir"
    priority: ClassVar[int] = 90

    def can_read(self, path: Path) -> bool:
        p = Path(path)
        if _has_structured_ancestor(p):
            return False
        if p.is_file():
            return p.suffix.lower() in VIDEO_EXTS and ".partial" not in p.name
        if p.is_dir():
            for child in p.rglob("*"):
                if any(part in SKIP_DIRS for part in child.relative_to(p).parts):
                    continue
                if child.is_file() and child.suffix.lower() in VIDEO_EXTS and ".partial" not in child.name:
                    if not _has_structured_ancestor(child):
                        return True
        return False

    def discover(
        self,
        root: Path,
        *,
        task_patterns: dict[str, list[str]] | None = None,
    ) -> list[EpisodeHandle]:
        root_p = Path(root).resolve()
        if root_p.is_file():
            if not self.can_read(root_p):
                return []
            return [self._handle_for_file(root_p, task_patterns)]

        if not root_p.is_dir():
            return []

        dirs_with_videos: set[Path] = set()
        for p in sorted(root_p.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS or ".partial" in p.name:
                continue
            if p.name.startswith("."):
                continue
            rel_parts = p.relative_to(root_p).parts
            if any(part in SKIP_DIRS for part in rel_parts):
                continue
            if _has_structured_ancestor(p):
                continue
            dirs_with_videos.add(p.parent)

        handles: list[EpisodeHandle] = []
        for d in sorted(dirs_with_videos):
            vids = _direct_videos(d)
            if not vids:
                continue
            has_manifest = (d / "episode.json").is_file() or (d / "cameras.json").is_file()
            if has_manifest:
                handles.append(self._handle_for_multicam_dir(d, vids, task_patterns))
                continue

            if len(vids) == 1:
                if is_camera_filename(vids[0].stem):
                    handles.append(self._handle_for_multicam_dir(d, vids, task_patterns))
                else:
                    handles.append(self._handle_for_file(vids[0], task_patterns))
            else:
                all_cam_names = all(is_camera_filename(v.stem) for v in vids)
                roles = [infer_camera_role(v.stem) for v in vids]
                if all_cam_names and len(roles) == len(set(roles)):
                    handles.append(self._handle_for_multicam_dir(d, vids, task_patterns))
                else:
                    for v in vids:
                        handles.append(self._handle_for_file(v, task_patterns))
        return handles

    def _handle_for_file(
        self, vid: Path, task_patterns: dict[str, list[str]] | None
    ) -> EpisodeHandle:
        dataset = dataset_of(vid)
        stem = validate_id_part(vid.stem, "stem")
        episode_id = f"{dataset}/{stem}"
        task_id = resolve_task_id(dataset, None, task_patterns)
        mtime = float(vid.stat().st_mtime)
        recorded = LABEL_DIRS.get(vid.parent.name.lower())
        return EpisodeHandle(
            episode_id=episode_id,
            dataset=dataset,
            stem=stem,
            task_id=task_id,
            adapter_type=self.name,
            source_uri=str(vid.resolve()),
            source_mtime=mtime,
            metadata={
                "is_multicam_dir": False,
                "recorded_success": recorded,
                "videos": {"main": str(vid.resolve())},
            },
        )

    def _handle_for_multicam_dir(
        self,
        ep_dir: Path,
        vids: list[Path],
        task_patterns: dict[str, list[str]] | None,
    ) -> EpisodeHandle:
        dataset = dataset_of(ep_dir)
        stem = validate_id_part(ep_dir.name, "stem")
        episode_id = f"{dataset}/{stem}"
        hint_task = None
        recorded = LABEL_DIRS.get(ep_dir.parent.name.lower())

        ep_json = ep_dir / "episode.json"
        if ep_json.is_file():
            try:
                meta_j = json.loads(ep_json.read_text(encoding="utf-8"))
                hint_task = meta_j.get("task_id") or meta_j.get("task")
                if "recorded_success" in meta_j:
                    recorded = meta_j["recorded_success"]
            except Exception:
                pass

        task_id = resolve_task_id(dataset, hint_task, task_patterns)
        mtime = max(float(v.stat().st_mtime) for v in vids)
        return EpisodeHandle(
            episode_id=episode_id,
            dataset=dataset,
            stem=stem,
            task_id=task_id,
            adapter_type=self.name,
            source_uri=str(ep_dir.resolve()),
            source_mtime=mtime,
            metadata={
                "is_multicam_dir": True,
                "recorded_success": recorded,
                "videos": {v.stem: str(v.resolve()) for v in vids},
            },
        )

    def open(self, handle: EpisodeHandle) -> EpisodeView:
        vid_map: dict[str, str] = handle.metadata.get("videos") or {}
        if not vid_map:
            src = handle.source_path
            if src.is_file():
                vid_map = {"main": str(src)}
            else:
                vid_map = {v.stem: str(v) for v in _direct_videos(src)}
        if not vid_map:
            raise ValueError(f"No video files found for {handle.episode_id}")

        cam_overrides: dict[str, str] = {}
        if handle.source_path.is_dir():
            cams_json = handle.source_path / "cameras.json"
            if cams_json.is_file():
                try:
                    raw_c = json.loads(cams_json.read_text(encoding="utf-8"))
                    if isinstance(raw_c, dict):
                        cam_overrides = {str(k): str(v) for k, v in raw_c.items()}
                except Exception:
                    pass

        cameras: list[CameraSpec] = []
        cam_entries: dict[str, dict[str, Any]] = {}
        single_cam = len(vid_map) == 1

        for cam_name, p_str in sorted(vid_map.items()):
            vpath = Path(p_str)
            w, h, n_frames, fps = probe_video_cv2(vpath)
            if fps <= 0.0:
                fps = 30.0
            role = (
                "main"
                if single_cam
                else (cam_overrides.get(cam_name) or infer_camera_role(cam_name))
            )
            cameras.append(
                CameraSpec(
                    name=cam_name,
                    role=role,  # type: ignore[arg-type]
                    width=w,
                    height=h,
                    fps=fps,
                    source_type="video_mp4",
                    is_virtual_crop=False,
                    source_stream=vpath.name,
                )
            )
            cam_entries[cam_name] = {
                "path": vpath,
                "frame_count": n_frames,
                "fps": fps,
                "duration_s": float(n_frames / fps),
            }

        # Ensure a "main" camera exists in multi-cam dir
        if not any(c.role == "main" for c in cameras):
            idx = next(
                (
                    i
                    for i, c in enumerate(cameras)
                    if c.role not in ("left_wrist", "right_wrist")
                ),
                0,
            )
            c0 = cameras[idx]
            cameras[idx] = c0.model_copy(update={"role": "main"})

        camera_roles = build_camera_roles(cameras)
        main_cam_name = camera_roles["main"]
        ep_fps = cam_entries[main_cam_name]["fps"]
        duration_s = min(ent["duration_s"] for ent in cam_entries.values())
        frame_count = max(1, int(round(duration_s * ep_fps)))

        media = VideoDirMediaSource(
            cameras=cameras,
            cam_entries=cam_entries,
            fps=ep_fps,
            duration_s=duration_s,
        )
        kin_spec = KinematicsSpec(
            has_kinematics=False,
            n_arms=0,
            arm_dof=0,
            hand_dof=0,
            torso_dof=0,
            has_chassis=False,
            has_tactile=False,
            tactile_dim=0,
            state_dim=0,
            has_action_cmd=False,
        )
        fingerprint = compute_episode_fingerprint(
            handle.source_uri,
            handle.source_mtime,
            extra=f"T={frame_count}|cams={','.join(sorted(vid_map))}",
        )
        meta = EpisodeMeta(
            episode_id=handle.episode_id,
            dataset=handle.dataset,
            stem=handle.stem,
            task_id=handle.task_id,
            adapter_type="video_dir",
            source_uri=handle.source_uri,
            source_mtime=handle.source_mtime,
            episode_fingerprint=fingerprint,
            duration_s=duration_s,
            fps=ep_fps,
            frame_count=frame_count,
            cameras=cameras,
            camera_roles=camera_roles,
            kinematics=kin_spec,
            recorded_success=handle.metadata.get("recorded_success"),
            extra_meta={"is_multicam_dir": bool(handle.metadata.get("is_multicam_dir"))},
        )
        return EpisodeView(meta=meta, media=media, kinematics=None)
