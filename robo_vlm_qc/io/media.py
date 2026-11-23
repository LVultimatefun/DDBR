"""Unified MP4 reader (cv2 + ffprobe), 2x2 adaptive tiler, and PreviewCache."""

from __future__ import annotations

import base64
import json
import logging
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from .base import MediaSource, validate_id_part

logger = logging.getLogger(__name__)

TileMode = Literal["wrist_2x2", "torso_head_2x2", "single_1x2"]


@dataclass(frozen=True)
class ZoomBox:
    x0: float
    y0: float
    x1: float
    y1: float


DEFAULT_MAIN_ZOOM = ZoomBox(x0=0.25, y0=0.056, x1=0.875, y1=0.648)
DEFAULT_TORSO_ZOOM = ZoomBox(x0=0.20, y0=0.20, x1=0.80, y1=0.85)
DEFAULT_SINGLE_ZOOM = ZoomBox(x0=0.20, y0=0.20, x1=0.80, y1=0.85)


def time_to_frame_idx(
    t_s: float,
    fps: float,
    frame_count: int,
    *,
    timestamps_s: np.ndarray | None = None,
    offset_s: float = 0.0,
) -> int:
    if frame_count <= 1:
        return 0
    if timestamps_s is not None and len(timestamps_s) > 0:
        idx = int(np.searchsorted(timestamps_s, t_s + offset_s, side="right")) - 1
        return int(max(0, min(frame_count - 1, idx)))
    idx = int(np.floor((t_s + offset_s) * fps + 1e-6))
    return int(max(0, min(frame_count - 1, idx)))


def probe_video_ffprobe(video_path: Path) -> tuple[int, int, int, float]:
    """Probe (width, height, frame_count, fps) via ffprobe JSON output."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise ValueError(f"ffprobe failed on {video_path}: {proc.stderr.strip()}")
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise ValueError(f"No video streams found in {video_path}")
    st = streams[0]
    w = int(st.get("width") or 0)
    h = int(st.get("height") or 0)

    def _parse_frac(val: str | None) -> float:
        if not val or val == "0/0":
            return 0.0
        if "/" in val:
            num, den = val.split("/", 1)
            d = float(den)
            return float(num) / d if d != 0.0 else 0.0
        return float(val)

    fps = _parse_frac(st.get("avg_frame_rate")) or _parse_frac(st.get("r_frame_rate"))
    if fps <= 0.0 or not np.isfinite(fps):
        fps = 30.0

    nb_raw = st.get("nb_frames")
    if nb_raw is not None and str(nb_raw).isdigit():
        n_frames = int(nb_raw)
    else:
        dur = st.get("duration") or (data.get("format") or {}).get("duration")
        dur_f = float(dur) if dur is not None else 0.0
        n_frames = max(1, int(round(dur_f * fps))) if dur_f > 0 else 0
    return w, h, n_frames, fps


def probe_video_cv2(video_path: Path) -> tuple[int, int, int, float]:
    """Probe (width, height, frame_count, fps) using cv2 with ffprobe fallback."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        w = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        h = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n_frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        cap.release()

    if w <= 0 or h <= 0 or n_frames <= 0 or not np.isfinite(fps) or fps <= 0.0:
        w, h, n_frames, fps = probe_video_ffprobe(video_path)
    if w <= 0 or h <= 0 or n_frames <= 0:
        raise ValueError(f"Invalid video dimensions/frames in {video_path}")
    return w, h, n_frames, fps


class _CapState:
    def __init__(self, cap: cv2.VideoCapture, w: int, h: int, n_frames: int, fps: float) -> None:
        self.cap = cap
        self.w = w
        self.h = h
        self.n_frames = n_frames
        self.fps = fps
        self.cur_pos = 0


class Mp4Reader:
    """Thread-safe cv2.VideoCapture reader with grab-vs-seek batch optimization."""

    def __init__(self) -> None:
        self._tls = threading.local()
        self._lock = threading.Lock()
        self._all_caps: list[cv2.VideoCapture] = []

    def _get_state(self, path: Path) -> _CapState:
        cache: dict[str, _CapState] | None = getattr(self._tls, "caps", None)
        if cache is None:
            cache = {}
            self._tls.caps = cache
        key = str(path)
        st = cache.get(key)
        if st is not None and st.cap.isOpened():
            return st
        w, h, n_frames, fps = probe_video_cv2(path)
        cap = cv2.VideoCapture(key)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video capture: {path}")
        st = _CapState(cap=cap, w=w, h=h, n_frames=n_frames, fps=fps)
        cache[key] = st
        with self._lock:
            self._all_caps.append(cap)
        return st

    def _decode_at(self, st: _CapState, frame_idx: int, path: Path) -> np.ndarray:
        target = max(0, min(st.n_frames - 1, int(frame_idx)))
        delta = target - st.cur_pos
        if st.cur_pos >= 0 and delta == 0:
            ok, bgr = st.cap.read()
        elif st.cur_pos >= 0 and 0 < delta <= 16:
            ok_grab = True
            for _ in range(delta):
                if not st.cap.grab():
                    ok_grab = False
                    break
            if ok_grab:
                ok, bgr = st.cap.read()
            else:
                st.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ok, bgr = st.cap.read()
        else:
            st.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, bgr = st.cap.read()

        if not ok or bgr is None:
            logger.warning("Failed decoding frame %d from %s; returning black frame", target, path)
            st.cur_pos = -1
            return np.zeros((st.h, st.w, 3), dtype=np.uint8)

        st.cur_pos = target + 1
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    def read_frame_rgb(self, path: Path, frame_idx: int) -> np.ndarray:
        st = self._get_state(path)
        return self._decode_at(st, frame_idx, path)

    def read_frames_batch(
        self, path: Path, frame_indices: list[int]
    ) -> list[np.ndarray]:
        if not frame_indices:
            return []
        st = self._get_state(path)
        clamped = [max(0, min(st.n_frames - 1, int(i))) for i in frame_indices]
        unique_sorted = sorted(set(clamped))
        decoded: dict[int, np.ndarray] = {}
        for idx in unique_sorted:
            decoded[idx] = self._decode_at(st, idx, path)
        return [decoded[idx] for idx in clamped]

    def close(self) -> None:
        with self._lock:
            for cap in self._all_caps:
                try:
                    cap.release()
                except Exception:
                    pass
            self._all_caps.clear()
        if hasattr(self._tls, "caps"):
            delattr(self._tls, "caps")


class SampledMediaReader:
    """Concurrent multi-camera frame sampler over any MediaSource."""

    def read_at_timestamps(
        self,
        media: MediaSource,
        cameras: list[str],
        times_s: list[float],
    ) -> dict[str, list[np.ndarray]]:
        if not cameras or not times_s:
            return {cam: [] for cam in cameras}
        workers = min(4, len(cameras))
        if workers <= 1:
            return {cam: media.read_frames_batch(cam, times_s) for cam in cameras}
        out: dict[str, list[np.ndarray]] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                cam: pool.submit(media.read_frames_batch, cam, times_s)
                for cam in cameras
            }
            for cam, fut in futs.items():
                out[cam] = fut.result()
        return out


def select_tile_mode(camera_roles: dict[str, str]) -> TileMode:
    if "left_wrist" in camera_roles or "right_wrist" in camera_roles:
        return "wrist_2x2"
    if (
        "front" in camera_roles
        or "wide" in camera_roles
        or "wide_right" in camera_roles
        or "secondary" in camera_roles
    ):
        return "torso_head_2x2"
    return "single_1x2"


def _crop_zoom(
    rgb: np.ndarray, box: ZoomBox, target_w: int, target_h: int
) -> np.ndarray:
    h, w = rgb.shape[:2]
    x0 = max(0, min(w - 2, int(round(box.x0 * w))))
    y0 = max(0, min(h - 2, int(round(box.y0 * h))))
    x1 = max(x0 + 2, min(w, int(round(box.x1 * w))))
    y1 = max(y0 + 2, min(h, int(round(box.y1 * h))))
    crop = rgb[y0:y1, x0:x1]
    return cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def _resize_full(rgb: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    return cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)


def _tag_panel(rgb: np.ndarray, label: str) -> np.ndarray:
    out = rgb.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.38, min(0.65, w / 640.0))
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    pad = 4
    bx1 = w
    bx0 = max(0, w - tw - 2 * pad)
    by0 = 0
    by1 = min(h, th + baseline + 2 * pad)
    out[by0:by1, bx0:bx1] = (0, 0, 0)
    cv2.putText(
        out,
        label,
        (bx0 + pad, by1 - baseline - pad),
        font,
        scale,
        (160, 255, 255),  # cyan in RGB
        thickness,
        cv2.LINE_AA,
    )
    return out


def compose_2x2_tile(
    media: MediaSource,
    camera_roles: dict[str, str],
    t_s: float,
    *,
    cell_w: int = 320,
    cell_h: int = 180,
    include_header: bool = True,
    header_h: int = 26,
    zoom_main: ZoomBox = DEFAULT_MAIN_ZOOM,
    zoom_torso: ZoomBox = DEFAULT_TORSO_ZOOM,
    zoom_single: ZoomBox = DEFAULT_SINGLE_ZOOM,
) -> np.ndarray:
    """Compose a camera-role-adaptive 2x2 (or 1x2 single-cam) RGB uint8 tile."""
    mode = select_tile_mode(camera_roles)
    t_clamped = max(0.0, min(float(media.duration_s), float(t_s)))
    frame_row = int(round(t_clamped * float(media.fps)))

    def _safe_read(role_key: str) -> np.ndarray | None:
        cam_name = camera_roles.get(role_key)
        if not cam_name:
            return None
        return media.read_frame_rgb(cam_name, t_clamped)

    main_rgb = _safe_read("main")
    if main_rgb is None and media.cameras:
        main_rgb = media.read_frame_rgb(media.cameras[0].name, t_clamped)
    if main_rgb is None:
        main_rgb = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)

    if mode == "wrist_2x2":
        lw_rgb = _safe_read("left_wrist")
        rw_rgb = _safe_read("right_wrist")
        tl = _tag_panel(
            _resize_full(lw_rgb, cell_w, cell_h)
            if lw_rgb is not None
            else np.zeros((cell_h, cell_w, 3), dtype=np.uint8),
            "L-WRIST",
        )
        tr = _tag_panel(
            _resize_full(rw_rgb, cell_w, cell_h)
            if rw_rgb is not None
            else np.zeros((cell_h, cell_w, 3), dtype=np.uint8),
            "R-WRIST",
        )
        bl = _tag_panel(_resize_full(main_rgb, cell_w, cell_h), "MAIN")
        br = _tag_panel(_crop_zoom(main_rgb, zoom_main, cell_w, cell_h), "MAIN-ZOOM")
        grid = np.vstack([np.hstack([tl, tr]), np.hstack([bl, br])])

    elif mode == "torso_head_2x2":
        if "front" in camera_roles:
            top_rgb = _safe_read("front")
            top_lbl, top_zoom_lbl = "TORSO", "TORSO-ZOOM"
        elif "wide" in camera_roles:
            top_rgb = _safe_read("wide")
            top_lbl, top_zoom_lbl = "WIDE", "WIDE-ZOOM"
        elif "wide_right" in camera_roles:
            top_rgb = _safe_read("wide_right")
            top_lbl, top_zoom_lbl = "WIDE", "WIDE-ZOOM"
        else:
            top_rgb = _safe_read("secondary")
            top_lbl, top_zoom_lbl = "AUX", "AUX-ZOOM"

        if top_rgb is None:
            top_rgb = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        tl = _tag_panel(_resize_full(top_rgb, cell_w, cell_h), top_lbl)
        tr = _tag_panel(_crop_zoom(top_rgb, zoom_torso, cell_w, cell_h), top_zoom_lbl)
        bl = _tag_panel(_resize_full(main_rgb, cell_w, cell_h), "HEAD")
        br = _tag_panel(_crop_zoom(main_rgb, zoom_main, cell_w, cell_h), "HEAD-ZOOM")
        grid = np.vstack([np.hstack([tl, tr]), np.hstack([bl, br])])

    else:  # single_1x2
        left = _tag_panel(_resize_full(main_rgb, cell_w, cell_h), "MAIN")
        right = _tag_panel(
            _crop_zoom(main_rgb, zoom_single, cell_w, cell_h), "MAIN-CENTER-ZOOM"
        )
        grid = np.hstack([left, right])

    if include_header:
        bar = np.zeros((header_h, grid.shape[1], 3), dtype=np.uint8)
        text = f"TIME = {t_clamped:05.2f}s  (frame {frame_row:04d})"
        cv2.putText(
            bar,
            text,
            (8, header_h - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 0),  # yellow in RGB
            1,
            cv2.LINE_AA,
        )
        canvas = np.vstack([bar, grid])
    else:
        canvas = grid
    return np.ascontiguousarray(canvas, dtype=np.uint8)


def encode_jpeg_bytes(rgb: np.ndarray, quality: int = 74) -> bytes:
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    ok, buf = cv2.imencode(
        ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def encode_jpeg_data_url(rgb: np.ndarray, quality: int = 74) -> str:
    b64 = base64.b64encode(encode_jpeg_bytes(rgb, quality=quality)).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


_NVENC_LOCK = threading.Lock()
_NVENC_AVAILABLE: bool | None = None


def _is_nvenc_available() -> bool:
    global _NVENC_AVAILABLE
    with _NVENC_LOCK:
        if _NVENC_AVAILABLE is not None:
            return _NVENC_AVAILABLE
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:d=0.1",
            "-c:v",
            "h264_nvenc",
            "-preset",
            "fast",
            "-cq",
            "26",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "null",
            "-",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, check=False, timeout=10)
            _NVENC_AVAILABLE = proc.returncode == 0
        except Exception:
            _NVENC_AVAILABLE = False
        return _NVENC_AVAILABLE


class PreviewCache:
    """Transcode and cache browser-friendly CFR H.264 MP4 previews under workspace."""

    def __init__(self, workspace: Path, *, max_workers: int = 2) -> None:
        self.workspace = Path(workspace).resolve()
        self.previews_root = self.workspace / "artifacts" / "previews"
        self.previews_root.mkdir(parents=True, exist_ok=True)
        self._sem = threading.Semaphore(max(1, max_workers))
        self._locks_guard = threading.Lock()
        self._path_locks: dict[str, threading.Lock] = {}

    def _lock_for(self, path: Path) -> threading.Lock:
        key = str(path)
        with self._locks_guard:
            lk = self._path_locks.get(key)
            if lk is None:
                lk = threading.Lock()
                self._path_locks[key] = lk
            return lk

    @staticmethod
    def _even(n: int) -> int:
        v = max(2, int(round(n / 2.0)) * 2)
        return v

    def preview_path(
        self, dataset: str, stem: str, camera: str, *, hd: bool = False
    ) -> Path:
        ds = validate_id_part(dataset, "dataset")
        st = validate_id_part(stem, "stem")
        cam = validate_id_part(camera, "camera")
        suffix = "_hd_v1.mp4" if hd else "_v1.mp4"
        return self.previews_root / ds / st / f"{cam}{suffix}"

    def _encode_stream(
        self,
        media: MediaSource,
        camera: str,
        out_w: int,
        out_h: int,
        fps: float,
        times_s: list[float],
        tmp_path: Path,
        encoder: str,
    ) -> None:
        if encoder == "h264_nvenc":
            codec_args = ["-c:v", "h264_nvenc", "-preset", "fast", "-cq", "26"]
        else:
            codec_args = [
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "25",
                "-tune",
                "fastdecode",
                "-profile:v",
                "main",
                "-sc_threshold",
                "0",
            ]
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{out_w}x{out_h}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "-",
            *codec_args,
            "-g",
            "4",
            "-keyint_min",
            "4",
            "-bf",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(tmp_path),
        ]
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert proc.stdin is not None
        try:
            batch_size = 64
            for start in range(0, len(times_s), batch_size):
                chunk_t = times_s[start : start + batch_size]
                frames = media.read_frames_batch(camera, chunk_t)
                for fr in frames:
                    if fr.shape[1] != out_w or fr.shape[0] != out_h:
                        fr = cv2.resize(
                            fr, (out_w, out_h), interpolation=cv2.INTER_AREA
                        )
                    proc.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
            proc.stdin.close()
            stderr_bytes = proc.stderr.read() if proc.stderr else b""
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(
                    f"ffmpeg ({encoder}) exited {rc}: {stderr_bytes.decode('utf-8', 'replace')}"
                )
        except BaseException:
            try:
                proc.kill()
                proc.wait()
            except Exception:
                pass
            raise

    def get_or_build(
        self,
        media: MediaSource,
        *,
        dataset: str,
        stem: str,
        camera: str,
        source_mtime: float,
        hd: bool = False,
        target_height: int | None = None,
    ) -> Path:
        out_path = self.preview_path(dataset, stem, camera, hd=hd)
        cam_spec = next((c for c in media.cameras if c.name == camera), None)
        if cam_spec is None:
            raise KeyError(f"Camera {camera!r} not found in MediaSource")

        if target_height is None:
            if hd:
                target_height = min(720, cam_spec.height)
            elif cam_spec.role in ("left_wrist", "right_wrist"):
                target_height = min(270, cam_spec.height)
            else:
                target_height = min(360, cam_spec.height)

        out_h = self._even(target_height)
        out_w = self._even(round(cam_spec.width * out_h / cam_spec.height))

        with self._lock_for(out_path):
            if out_path.is_file() and out_path.stat().st_mtime >= source_mtime:
                return out_path

            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = out_path.with_suffix(".partial.mp4")

            fps = float(media.fps) if media.fps > 0 else 30.0
            n_frames = max(1, int(round(media.duration_s * fps)))
            times_s = [min(media.duration_s, i / fps) for i in range(n_frames)]

            encoders = ["libx264"]
            if min(out_w, out_h) >= 160 and _is_nvenc_available():
                encoders = ["h264_nvenc", "libx264"]

            with self._sem:
                last_err: Exception | None = None
                for enc in encoders:
                    try:
                        tmp_path.unlink(missing_ok=True)
                        self._encode_stream(
                            media, camera, out_w, out_h, fps, times_s, tmp_path, enc
                        )
                        tmp_path.replace(out_path)
                        last_err = None
                        break
                    except Exception as exc:
                        last_err = exc
                        tmp_path.unlink(missing_ok=True)
                        logger.warning(
                            "Preview encode with %s failed (%s), trying fallback if available",
                            enc,
                            exc,
                        )
                if last_err is not None:
                    raise last_err
            return out_path
