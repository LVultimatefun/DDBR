"""1920x1080@30fps Studio HUD MP4 renderer consuming ResolvedEpisode + EpisodeView only."""

from __future__ import annotations

import bisect
import functools
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, ClassVar, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..io.base import EpisodeView
from ..io.media import DEFAULT_MAIN_ZOOM, select_tile_mode
from ..models.annotation import (
    ArmSpan,
    GroundingKeyframe,
    GroundingResult,
    GroundingTrack,
    MacroStage,
    QcSegment,
)
from ..models.export import ExportManifest, ExportRecipe, KeepPiece
from ..models.review import ResolvedEpisode
from .base import (
    OutputExporter,
    finalize_manifest,
    group_kept_pieces,
    staged_export_dir,
)

CANVAS_W = 1920
CANVAS_H = 1080
BG_RGB = (11, 15, 25)
PANEL_RGB = (17, 24, 39)
CARD_RGB = (22, 30, 48)
BORDER_RGB = (51, 65, 85)
TEXT_RGB = (241, 245, 249)
MUTED_RGB = (148, 163, 184)

_ARM_COLORS: dict[str, tuple[int, int, int]] = {
    "idle": (100, 116, 139),
    "work": (59, 130, 246),
    "hold": (14, 165, 233),
    "anom": (245, 158, 11),
    "mask": (168, 85, 247),
    "good": (16, 185, 129),
}

_POLICY_COLORS: dict[str, tuple[int, int, int]] = {
    "keep_positive": (16, 185, 129),
    "mask_loss": (168, 85, 247),
    "cut_clean": (245, 158, 11),
}


@functools.lru_cache(maxsize=16)
def _load_font(size: int, *, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates: list[Path] = []
    if mono:
        candidates.extend(
            [
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
                Path("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf"),
                Path("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"),
            ]
        )
    elif bold:
        candidates.extend(
            [
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
                Path("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"),
                Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
                Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
            ]
        )
    else:
        candidates.extend(
            [
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
                Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
                Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            ]
        )

    for p in candidates:
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size=size)
            except Exception:
                continue

    font_root = Path("/usr/share/fonts")
    if font_root.is_dir():
        for pattern in ("*DejaVuSans*", "*NotoSans*"):
            for p in sorted(font_root.rglob(pattern)):
                if not p.is_file():
                    continue
                try:
                    return ImageFont.truetype(str(p), size=size)
                except Exception:
                    continue
    return ImageFont.load_default()


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: Any) -> int:
    if not text:
        return 0
    try:
        box = draw.textbbox((0, 0), text, font=font)
        return max(0, int(box[2] - box[0]))
    except Exception:
        return len(text) * 8


def _fit_single_line(
    draw: ImageDraw.ImageDraw, text: str, font: Any, max_w: int
) -> str:
    s = (text or "").replace("\n", " ").strip()
    if max_w <= 12 or not s:
        return ""
    if _text_width(draw, s, font) <= max_w:
        return s
    ell = "..."
    lo, hi = 0, len(s)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = s[:mid] + ell
        if _text_width(draw, cand, font) <= max_w:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1
    return best or ell


def _wrap_text(
    draw: ImageDraw.ImageDraw, text: str, font: Any, max_w: int, max_lines: int
) -> list[str]:
    s = (text or "").strip()
    if not s or max_lines <= 0 or max_w <= 20:
        return []
    lines: list[str] = []
    cur = ""
    for ch in s:
        if ch == "\n":
            lines.append(cur)
            cur = ""
            if len(lines) >= max_lines:
                break
            continue
        cand = cur + ch
        if _text_width(draw, cand, font) <= max_w:
            cur = cand
        else:
            if cur:
                lines.append(cur)
                cur = ch
            else:
                lines.append(ch)
                cur = ""
            if len(lines) >= max_lines:
                break
    if len(lines) < max_lines and cur:
        lines.append(cur)
    if len(lines) == max_lines and (cur and cur != lines[-1]):
        lines[-1] = _fit_single_line(draw, lines[-1], font, max_w)
    elif len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = _fit_single_line(draw, lines[-1], font, max_w)
    return lines


def _compute_center_crop_rect(
    src_w: int, src_h: int, target_w: int, target_h: int
) -> tuple[int, int, int, int]:
    """Return exact center-crop rectangle (x0, y0, crop_w, crop_h) used by `_fit_crop_resize`."""
    if src_w <= 0 or src_h <= 0 or target_w <= 0 or target_h <= 0:
        return 0, 0, max(1, src_w), max(1, src_h)
    target_ar = target_w / float(target_h)
    src_ar = src_w / float(src_h)
    if abs(src_ar - target_ar) > 0.05:
        if src_ar > target_ar:
            new_w = max(2, int(round(src_h * target_ar)))
            x0 = max(0, (src_w - new_w) // 2)
            return x0, 0, min(new_w, src_w - x0), src_h
        new_h = max(2, int(round(src_w / target_ar)))
        y0 = max(0, (src_h - new_h) // 2)
        return 0, y0, src_w, min(new_h, src_h - y0)
    return 0, 0, src_w, src_h


def _fit_crop_resize(rgb: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize RGB image to target_w x target_h preserving aspect ratio via center crop."""
    h, w = rgb.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0, y0, cw, ch = _compute_center_crop_rect(w, h, target_w, target_h)
    if x0 != 0 or y0 != 0 or cw != w or ch != h:
        rgb = rgb[y0 : y0 + ch, x0 : x0 + cw]
    interp = cv2.INTER_AREA if rgb.shape[1] >= target_w else cv2.INTER_LINEAR
    return cv2.resize(rgb, (target_w, target_h), interpolation=interp)


def _project_norm_box_for_crop(
    bbox_norm_1000: Sequence[float],
    src_w: int,
    src_h: int,
    target_w: int = 1280,
    target_h: int = 720,
) -> list[int] | None:
    """Project [0,1000] bbox from original uncropped frame into center-cropped target view."""
    if src_w <= 0 or src_h <= 0 or target_w <= 0 or target_h <= 0:
        return None
    x0, y0, cw, ch = _compute_center_crop_rect(src_w, src_h, target_w, target_h)
    sx1 = float(bbox_norm_1000[0]) * src_w / 1000.0
    sy1 = float(bbox_norm_1000[1]) * src_h / 1000.0
    sx2 = float(bbox_norm_1000[2]) * src_w / 1000.0
    sy2 = float(bbox_norm_1000[3]) * src_h / 1000.0

    scale_x = target_w / float(max(1, cw))
    scale_y = target_h / float(max(1, ch))
    tx1 = (sx1 - x0) * scale_x
    ty1 = (sy1 - y0) * scale_y
    tx2 = (sx2 - x0) * scale_x
    ty2 = (sy2 - y0) * scale_y

    if tx2 <= 0.0 or tx1 >= target_w or ty2 <= 0.0 or ty1 >= target_h:
        return None

    ix1 = int(round(max(0.0, min(float(target_w - 1), tx1))))
    iy1 = int(round(max(0.0, min(float(target_h - 1), ty1))))
    ix2 = int(round(max(0.0, min(float(target_w - 1), tx2))))
    iy2 = int(round(max(0.0, min(float(target_h - 1), ty2))))
    if ix2 <= ix1 + 1 or iy2 <= iy1 + 1:
        return None
    return [ix1, iy1, ix2, iy2]


def _pick_tick_step(duration_s: float) -> float:
    if duration_s <= 4.0:
        return 0.5
    if duration_s <= 15.0:
        return 2.0
    if duration_s <= 40.0:
        return 5.0
    if duration_s <= 100.0:
        return 10.0
    return 20.0


def _smoothstep(u: float) -> float:
    x = max(0.0, min(1.0, u))
    return x * x * (3.0 - 2.0 * x)


def _nearest_visible_anchor_meta(track: GroundingTrack, t_s: float) -> tuple[str, str]:
    vis_kfs = [k for k in track.keyframes if k.visible]
    if not vis_kfs:
        return "", "neutral"
    times = [k.time_s for k in vis_kfs]
    pos = bisect.bisect_left(times, t_s)
    if pos <= 0:
        k = vis_kfs[0]
    elif pos >= len(vis_kfs):
        k = vis_kfs[-1]
    else:
        k = vis_kfs[pos - 1] if (t_s - times[pos - 1]) <= (times[pos] - t_s) else vis_kfs[pos]
    return k.state, k.status_kind


def _interp_track_box(
    track: GroundingTrack,
    t_s: float,
    view_w: int,
    view_h: int,
    *,
    grounding: GroundingResult | None = None,
    src_w: int | None = None,
    src_h: int | None = None,
) -> tuple[list[int], str, str] | None:
    eff_src_w = int(src_w) if src_w and src_w > 0 else view_w
    eff_src_h = int(src_h) if src_h and src_h > 0 else view_h

    # Fast dense path (schema_version >= 2)
    if (
        grounding is not None
        and grounding.dense_times_s
        and len(track.dense_boxes_norm_1000) == len(grounding.dense_times_s)
    ):
        times = grounding.dense_times_s
        boxes = track.dense_boxes_norm_1000
        flags = track.dense_flags
        n = len(times)

        def _flag_at(idx: int) -> int:
            if flags:
                return flags[idx]
            return 0 if boxes[idx] == [0, 0, 0, 0] else 1

        pos = bisect.bisect_left(times, t_s)
        norm_box: list[float] | None = None
        if pos <= 0:
            if _flag_at(0) > 0:
                norm_box = [float(v) for v in boxes[0]]
        elif pos >= n:
            if _flag_at(n - 1) > 0:
                norm_box = [float(v) for v in boxes[n - 1]]
        else:
            i0, i1 = pos - 1, pos
            f0, f1 = _flag_at(i0), _flag_at(i1)
            dt = times[i1] - times[i0]
            if f0 > 0 and f1 > 0 and dt <= 0.5:
                u = (t_s - times[i0]) / max(1e-6, dt)
                b0, b1 = boxes[i0], boxes[i1]
                norm_box = [(1.0 - u) * b0[j] + u * b1[j] for j in range(4)]
            else:
                pick = i0 if (t_s - times[i0]) <= (times[i1] - t_s) else i1
                if _flag_at(pick) > 0:
                    norm_box = [float(v) for v in boxes[pick]]

        if norm_box is None:
            return None
        proj = _project_norm_box_for_crop(norm_box, eff_src_w, eff_src_h, view_w, view_h)
        if proj is None:
            return None
        state, skind = _nearest_visible_anchor_meta(track, t_s)
        return proj, state, skind

    # Legacy sparse fallback (schema_version == 1)
    kfs: list[GroundingKeyframe] = list(track.keyframes)
    if not kfs:
        return None
    kf_times = [k.time_s for k in kfs]
    pos = bisect.bisect_left(kf_times, t_s)

    if pos <= 0:
        k = kfs[0]
        if not k.visible:
            return None
        if abs(t_s - k.time_s) > 4.5 and not track.is_static_container:
            return None
        proj = _project_norm_box_for_crop(k.bbox_norm_1000, eff_src_w, eff_src_h, view_w, view_h)
        return (proj, k.state, k.status_kind) if proj is not None else None

    if pos >= len(kfs):
        k = kfs[-1]
        if not k.visible:
            return None
        if abs(t_s - k.time_s) > 4.5 and not track.is_static_container:
            return None
        proj = _project_norm_box_for_crop(k.bbox_norm_1000, eff_src_w, eff_src_h, view_w, view_h)
        return (proj, k.state, k.status_kind) if proj is not None else None

    k0, k1 = kfs[pos - 1], kfs[pos]
    span = max(1e-6, k1.time_s - k0.time_s)
    if k0.visible and k1.visible and (span <= 4.5 or track.is_static_container):
        w = _smoothstep((t_s - k0.time_s) / span)
        b0, b1 = k0.bbox_norm_1000, k1.bbox_norm_1000
        norm_b = [(1.0 - w) * b0[j] + w * b1[j] for j in range(4)]
        proj = _project_norm_box_for_crop(norm_b, eff_src_w, eff_src_h, view_w, view_h)
        if proj is None:
            return None
        ref_k = k0 if w < 0.5 else k1
        return proj, ref_k.state, ref_k.status_kind

    nearest = k0 if (t_s - k0.time_s) <= (k1.time_s - t_s) else k1
    if not nearest.visible:
        return None
    if abs(t_s - nearest.time_s) > 2.0 and not track.is_static_container:
        return None
    proj = _project_norm_box_for_crop(nearest.bbox_norm_1000, eff_src_w, eff_src_h, view_w, view_h)
    return (proj, nearest.state, nearest.status_kind) if proj is not None else None


def _find_active_idx(spans: Sequence[Any], t_s: float) -> int:
    for i, sp in enumerate(spans):
        if sp.t_start <= t_s <= sp.t_end + 1e-4:
            return i
    return -1


def _draw_hatched_rect(
    img: Image.Image,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    fill_rgb: tuple[int, int, int],
    stripe_rgb: tuple[int, int, int],
    spacing: int = 9,
) -> None:
    if x1 <= x0 or y1 <= y0:
        return
    draw = ImageDraw.Draw(img)
    draw.rectangle((x0, y0, x1, y1), fill=fill_rgb)
    h = y1 - y0
    for dx in range(-h, (x1 - x0) + h, spacing):
        sx0 = max(x0, x0 + dx)
        sy0 = y0 + max(0, sx0 - (x0 + dx))
        sx1 = min(x1, x0 + dx + h)
        sy1 = y0 + min(h, sx1 - (x0 + dx))
        if sx1 > sx0 and sy1 > sy0:
            draw.line((sx0, sy0, sx1, sy1), fill=stripe_rgb, width=2)


class _HudCompositor:
    """Renders 1920x1080 Studio HUD frames from ResolvedEpisode."""

    def __init__(self, resolved: ResolvedEpisode, view: EpisodeView) -> None:
        self.resolved = resolved
        self.view = view
        self.duration_s = max(0.05, float(resolved.meta.duration_s or view.duration_s))
        self.roles = view.camera_roles
        self.tile_mode = select_tile_mode(self.roles)

        self.f_bold_16 = _load_font(16, bold=True)
        self.f_bold_14 = _load_font(14, bold=True)
        self.f_reg_14 = _load_font(14, bold=False)
        self.f_reg_12 = _load_font(12, bold=False)
        self.f_mono_14 = _load_font(14, mono=True)

        self._base_bg = self._build_static_background()
        self._info_cache: dict[tuple[Any, ...], np.ndarray] = {}

    def _build_static_background(self) -> np.ndarray:
        img = Image.new("RGB", (CANVAS_W, CANVAS_H), BG_RGB)
        draw = ImageDraw.Draw(img)

        # Top header bar
        draw.rectangle((0, 0, CANVAS_W, 50), fill=PANEL_RGB)
        draw.rounded_rectangle((16, 10, 310, 40), radius=6, fill=(30, 58, 138))
        draw.text((28, 16), "ROBO-VLM-QC STUDIO HUD", font=self.f_bold_14, fill=(219, 234, 254))

        mode_tag = "HUMAN REVIEWED" if self.resolved.is_human_reviewed else "VLM RAW"
        meta_txt = (
            f"{self.resolved.episode_id}  |  task={self.resolved.meta.task_id}  |  "
            f"{self.resolved.status.value.upper()} ({mode_tag})"
        )
        draw.text(
            (324, 16),
            _fit_single_line(draw, meta_txt, self.f_reg_14, 905),
            font=self.f_reg_14,
            fill=TEXT_RGB,
        )

        # Aux panel headers (y=60..84)
        if self.tile_mode == "wrist_2x2":
            l_title, r_title = "L-WRIST | Left Wrist", "R-WRIST | Right Wrist"
        elif self.tile_mode == "torso_head_2x2":
            if "front" in self.roles:
                l_title = "TORSO | Front View"
            elif "secondary" in self.roles:
                l_title = "AUX | Aux View"
            else:
                l_title = "MAIN | Main View"

            if "wide" in self.roles:
                r_title = "WIDE-L | Wide Left"
            elif "wide_right" in self.roles:
                r_title = "WIDE-R | Wide Right"
            elif "front" in self.roles or "secondary" in self.roles:
                r_title = "TORSO-ZOOM | Crop"
            else:
                r_title = "MAIN-ZOOM | Crop"
        else:
            l_title, r_title = "MAIN-ZOOM | Workspace Crop", "NO AUX CAM"

        draw.rectangle((1310, 60, 1601, 84), fill=CARD_RGB)
        draw.text((1318, 64), _fit_single_line(draw, l_title, self.f_bold_14, 276), font=self.f_bold_14, fill=(186, 230, 253))
        draw.rectangle((1613, 60, 1904, 84), fill=CARD_RGB)
        draw.text((1621, 64), _fit_single_line(draw, r_title, self.f_bold_14, 276), font=self.f_bold_14, fill=(186, 230, 253))

        # Timeline outer card (16, 792) - (1904, 1068)
        draw.rounded_rectangle((16, 792, 1904, 1068), radius=8, fill=PANEL_RGB, outline=BORDER_RGB, width=1)
        tx0, tx1 = 228, 1882
        step = _pick_tick_step(self.duration_s)
        t_tick = 0.0
        while t_tick <= self.duration_s + 1e-4:
            px = int(round(tx0 + (t_tick / self.duration_s) * (tx1 - tx0)))
            draw.line((px, 814, px, 1020), fill=(35, 45, 65), width=1)
            draw.text((px + 2, 797), f"{t_tick:.1f}s", font=self.f_reg_12, fill=MUTED_RGB)
            t_tick += step

        # Track labels
        track_defs = [
            (828, 864, "1. Macro Stages"),
            (874, 910, "2. Left Arm (L-ARM)"),
            (920, 956, "3. Right Arm (R-ARM)"),
            (966, 1014, "4. QC Segments"),
        ]
        for ty0, ty1, lbl in track_defs:
            draw.rectangle((tx0, ty0, tx1, ty1), fill=(15, 23, 42), outline=(38, 49, 68), width=1)
            draw.text((28, (ty0 + ty1) // 2 - 8), lbl, font=self.f_bold_14, fill=TEXT_RGB)

        # Track 1: Macro stages
        for st in self.resolved.macro_stages:
            x0 = int(round(tx0 + (st.t_start / self.duration_s) * (tx1 - tx0)))
            x1 = int(round(tx0 + (st.t_end / self.duration_s) * (tx1 - tx0)))
            col = tuple(int(c) for c in st.color_rgb)
            if x1 > x0 + 1:
                draw.rectangle((x0, 829, x1, 863), fill=col, outline=(15, 23, 42), width=1)
                lbl_fit = _fit_single_line(draw, f"{st.stage_id} {st.label}", self.f_reg_12, max(0, x1 - x0 - 8))
                if lbl_fit:
                    draw.text((x0 + 4, 839), lbl_fit, font=self.f_reg_12, fill=(255, 255, 255))

        # Track 2 & 3: Arm timelines
        for ty0, ty1, spans in (
            (875, 909, self.resolved.arm_timelines.left),
            (921, 955, self.resolved.arm_timelines.right),
        ):
            for sp in spans:
                x0 = int(round(tx0 + (sp.t_start / self.duration_s) * (tx1 - tx0)))
                x1 = int(round(tx0 + (sp.t_end / self.duration_s) * (tx1 - tx0)))
                col = _ARM_COLORS.get(sp.state, (100, 116, 139))
                if x1 > x0 + 1:
                    draw.rectangle((x0, ty0, x1, ty1), fill=col, outline=(15, 23, 42), width=1)
                    st_fit = _fit_single_line(draw, sp.state.upper(), self.f_reg_12, max(0, x1 - x0 - 6))
                    if st_fit:
                        draw.text((x0 + 3, ty0 + 9), st_fit, font=self.f_reg_12, fill=(255, 255, 255))

        # Track 4: QC Segments with 9px diagonal hatching for mask_loss
        for seg in self.resolved.qc_segments:
            x0 = int(round(tx0 + (seg.t_start / self.duration_s) * (tx1 - tx0)))
            x1 = int(round(tx0 + (seg.t_end / self.duration_s) * (tx1 - tx0)))
            x1 = max(x0 + 3, x1)
            if seg.splice_policy == "mask_loss":
                _draw_hatched_rect(
                    img,
                    x0,
                    967,
                    x1,
                    1013,
                    fill_rgb=(91, 33, 182),
                    stripe_rgb=(192, 132, 252),
                    spacing=9,
                )
            elif seg.training_role == "terminal_fail":
                draw.rectangle((x0, 967, x1, 1013), fill=(185, 28, 28), outline=(254, 202, 202), width=1)
            else:
                col = _POLICY_COLORS.get(seg.splice_policy, (245, 158, 11))
                draw.rectangle((x0, 967, x1, 1013), fill=col, outline=(15, 23, 42), width=1)
            tag = _fit_single_line(draw, seg.label_id, self.f_reg_12, max(0, x1 - x0 - 6))
            if tag:
                draw.text((x0 + 3, 983), tag, font=self.f_reg_12, fill=(255, 255, 255))

        # Legend (y=1034..1051)
        lx = 228
        for col, lbl in (
            ((16, 185, 129), "keep_positive (Keep)"),
            ((168, 85, 247), "mask_loss (Hatched Mask)"),
            ((245, 158, 11), "cut_clean (Splice Cut)"),
            ((185, 28, 28), "terminal_fail (Terminal)"),
        ):
            draw.rectangle((lx, 1036, lx + 14, 1050), fill=col)
            draw.text((lx + 20, 1035), lbl, font=self.f_reg_12, fill=MUTED_RGB)
            lx += 220

        return np.ascontiguousarray(np.asarray(img, dtype=np.uint8)[:, :, ::-1])  # BGR

    def _render_info_panel(
        self,
        t_s: float,
        stage_idx: int,
        l_idx: int,
        r_idx: int,
        qc_idx: int,
    ) -> np.ndarray:
        criteria = self.resolved.success.criteria[:3]
        crit_key = tuple(
            (
                c.met,
                c.satisfied_at_s is not None and t_s >= c.satisfied_at_s,
            )
            for c in criteria
        )
        verdict_revealed = t_s >= 0.85 * self.duration_s
        cache_key = (stage_idx, l_idx, r_idx, qc_idx, crit_key, verdict_revealed)
        cached = self._info_cache.get(cache_key)
        if cached is not None:
            return cached

        panel = Image.new("RGB", (594, 524), PANEL_RGB)
        draw = ImageDraw.Draw(panel)

        # Card B (0..150): Macro stage + L/R arm status
        draw.rounded_rectangle((0, 0, 593, 150), radius=8, fill=CARD_RGB, outline=BORDER_RGB, width=1)
        if 0 <= stage_idx < len(self.resolved.macro_stages):
            st = self.resolved.macro_stages[stage_idx]
            st_hdr = f"Macro Stage {st.stage_id}: {st.label} ({st.t_start:.1f}s-{st.t_end:.1f}s)"
        else:
            st_hdr = "Macro Stage: Full Episode Context"
        draw.text((12, 10), _fit_single_line(draw, st_hdr, self.f_bold_14, 570), font=self.f_bold_14, fill=(147, 197, 253))

        for y_arm, prefix, spans, idx in (
            (46, "L-ARM", self.resolved.arm_timelines.left, l_idx),
            (98, "R-ARM", self.resolved.arm_timelines.right, r_idx),
        ):
            sp: ArmSpan | None = spans[idx] if 0 <= idx < len(spans) else None
            st_name = sp.state.upper() if sp else "IDLE"
            st_col = _ARM_COLORS.get(sp.state if sp else "idle", (100, 116, 139))
            draw.rounded_rectangle((12, y_arm, 150, y_arm + 34), radius=4, fill=st_col)
            draw.text((20, y_arm + 8), f"{prefix}: {st_name}", font=self.f_bold_14, fill=(255, 255, 255))
            desc = sp.desc if sp and sp.desc else "Active"
            draw.text((160, y_arm + 8), _fit_single_line(draw, desc, self.f_reg_14, 420), font=self.f_reg_14, fill=TEXT_RGB)

        # Card C (160..394): Active QC segment card
        draw.rounded_rectangle((0, 160, 593, 394), radius=8, fill=CARD_RGB, outline=BORDER_RGB, width=1)
        if 0 <= qc_idx < len(self.resolved.qc_segments):
            seg: QcSegment = self.resolved.qc_segments[qc_idx]
            pol_col = _POLICY_COLORS.get(seg.splice_policy, (168, 85, 247))
            seg_label_txt = f"{seg.label_id} | {seg.name}" if seg.name else seg.label_id
            draw.text(
                (12, 170),
                _fit_single_line(
                    draw,
                    f"{seg.seg_id}  [{seg.t_start:.2f}s - {seg.t_end:.2f}s]  {seg_label_txt}",
                    self.f_bold_14,
                    570,
                ),
                font=self.f_bold_14,
                fill=(253, 224, 71),
            )
            draw.rounded_rectangle((12, 198, 170, 224), radius=4, fill=(51, 65, 85))
            draw.text((20, 203), f"Role: {seg.training_role}", font=self.f_reg_12, fill=TEXT_RGB)

            pol_txt = f"{seg.splice_policy}" + (f" [{seg.mask_side}]" if seg.mask_side else "")
            draw.rounded_rectangle((180, 198, 370, 224), radius=4, fill=pol_col)
            draw.text((188, 203), _fit_single_line(draw, pol_txt, self.f_bold_14, 174), font=self.f_reg_12, fill=(255, 255, 255))

            g_txt = f"Gate: {seg.guardrail.gate_type}"
            if seg.guardrail.downgraded:
                g_txt += " (DOWNGRADED)"
            elif seg.guardrail.guardrail_override:
                g_txt += " (OVERRIDE)"
            draw.text((380, 203), _fit_single_line(draw, g_txt, self.f_reg_12, 200), font=self.f_reg_12, fill=(186, 230, 253))

            ev_lines = _wrap_text(
                draw,
                f"VLM Visual Evidence: {seg.visual_evidence.as_text() or seg.unknown_note or 'Confirmed by multi-view visual cues'}",
                self.f_reg_14,
                570,
                4,
            )
            for li, ln in enumerate(ev_lines):
                draw.text((12, 238 + li * 24), ln, font=self.f_reg_14, fill=TEXT_RGB)

            if seg.guardrail.downgrade_reason or seg.guardrail.override_reason:
                reason = seg.guardrail.downgrade_reason or seg.guardrail.override_reason or ""
                draw.text(
                    (12, 358),
                    _fit_single_line(draw, f"Guardrail: {reason}", self.f_reg_12, 570),
                    font=self.f_reg_12,
                    fill=(251, 191, 36),
                )
        else:
            draw.text((12, 180), "[Normal Positive Progression]", font=self.f_bold_16, fill=(52, 211, 153))
            draw.text(
                (12, 216),
                "No active anomaly; normal positive training span.",
                font=self.f_reg_14,
                fill=MUTED_RGB,
            )

        # Card D (404..523): End-state success criteria checklist
        draw.rounded_rectangle((0, 404, 593, 523), radius=8, fill=CARD_RGB, outline=BORDER_RGB, width=1)
        sv = self.resolved.success
        v_col = (16, 185, 129) if sv.verdict == "success" else ((220, 38, 38) if sv.verdict == "failure" else (245, 158, 11))
        ctrl_txt = ""
        if sv.control_check is not None:
            ctrl_txt = " (Control OK)" if sv.control_check.passed else " (Control Failed)"
        draw.text(
            (12, 412),
            _fit_single_line(draw, f"VERDICT: {sv.verdict.upper()}{ctrl_txt}", self.f_bold_14, 570),
            font=self.f_bold_14,
            fill=v_col,
        )
        for ci, c in enumerate(criteria):
            if c.met is True and (c.satisfied_at_s is None or t_s >= c.satisfied_at_s):
                mark, mc = "[+]", (52, 211, 153)
            elif c.met is False and verdict_revealed:
                mark, mc = "[x]", (248, 113, 113)
            else:
                mark, mc = "[.]", MUTED_RGB
            draw.text(
                (12, 442 + ci * 24),
                _fit_single_line(draw, f"{mark} {c.criterion}", self.f_reg_14, 570),
                font=self.f_reg_14,
                fill=mc,
            )

        bgr = np.ascontiguousarray(np.asarray(panel, dtype=np.uint8)[:, :, ::-1])
        self._info_cache[cache_key] = bgr
        return bgr

    def _aux_frames(self, main_rgb: np.ndarray, t_s: float) -> tuple[np.ndarray, np.ndarray]:
        if self.tile_mode == "wrist_2x2":
            l_name = self.roles.get("left_wrist")
            r_name = self.roles.get("right_wrist")
            l_rgb = self.view.media.read_frame_rgb(l_name, t_s) if l_name else np.zeros((164, 291, 3), dtype=np.uint8)
            r_rgb = self.view.media.read_frame_rgb(r_name, t_s) if r_name else np.zeros((164, 291, 3), dtype=np.uint8)
            return _fit_crop_resize(l_rgb, 291, 164), _fit_crop_resize(r_rgb, 291, 164)
        if self.tile_mode == "torso_head_2x2":
            front_name = self.roles.get("front") or self.roles.get("secondary")
            wide_name = self.roles.get("wide") or self.roles.get("wide_right")
            l_rgb = self.view.media.read_frame_rgb(front_name, t_s) if front_name else main_rgb
            if wide_name:
                r_rgb = self.view.media.read_frame_rgb(wide_name, t_s)
            else:
                h, w = l_rgb.shape[:2]
                z = DEFAULT_MAIN_ZOOM
                r_rgb = l_rgb[int(z.y0 * h) : int(z.y1 * h), int(z.x0 * w) : int(z.x1 * w)]
            return _fit_crop_resize(l_rgb, 291, 164), _fit_crop_resize(r_rgb, 291, 164)

        # single_1x2
        h, w = main_rgb.shape[:2]
        z = DEFAULT_MAIN_ZOOM
        crop = main_rgb[max(0, int(z.y0 * h)) : max(2, int(z.y1 * h)), max(0, int(z.x0 * w)) : max(2, int(z.x1 * w))]
        l_panel = _fit_crop_resize(crop, 291, 164)
        r_panel = np.full((164, 291, 3), 24, dtype=np.uint8)
        cv2.putText(r_panel, "NO AUX CAM", (75, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (148, 163, 184), 2, cv2.LINE_AA)
        return l_panel, r_panel

    def render_frame_bgr(
        self, frame_idx: int, total_frames: int, t_s: float, main_rgb: np.ndarray
    ) -> np.ndarray:
        canvas = self._base_bg.copy()

        # Main view (1280x720 -> [60:780, 16:1296])
        src_h, src_w = main_rgb.shape[:2]
        main_resized = _fit_crop_resize(main_rgb, 1280, 720)
        main_bgr = np.ascontiguousarray(main_resized[:, :, ::-1])

        # Draw 2D grounding boxes on main_bgr (only when grounded camera matches main cam)
        main_cam = self.roles.get("main") or (
            self.view.media.cameras[0].name if self.view.media.cameras else ""
        )
        grounded_cam = self.resolved.grounding.camera_name
        if not grounded_cam or grounded_cam == main_cam:
            for oid, track in self.resolved.grounding.object_tracks_2d.items():
                res = _interp_track_box(
                    track,
                    t_s,
                    1280,
                    720,
                    grounding=self.resolved.grounding,
                    src_w=src_w,
                    src_h=src_h,
                )
                if res is None:
                    continue
                box, state, skind = res
                x1, y1, x2, y2 = [
                    max(0, min(1279, box[0])),
                    max(0, min(719, box[1])),
                    max(0, min(1279, box[2])),
                    max(0, min(719, box[3])),
                ]
                if x2 <= x1 + 4 or y2 <= y1 + 4:
                    continue
                bgr_col = (
                    (68, 68, 239)
                    if skind == "error"
                    else (
                        (11, 158, 245)
                        if skind == "warn"
                        else (
                            (129, 185, 16) if skind == "good" else (246, 130, 59)
                        )
                    )
                )
                # Corner brackets
                blen = min(18, (x2 - x1) // 3, (y2 - y1) // 3)
                cv2.rectangle(main_bgr, (x1, y1), (x2, y2), bgr_col, 1)
                for cx, cy, dx, dy in (
                    (x1, y1, 1, 1),
                    (x2, y1, -1, 1),
                    (x1, y2, 1, -1),
                    (x2, y2, -1, -1),
                ):
                    cv2.line(main_bgr, (cx, cy), (cx + dx * blen, cy), bgr_col, 3)
                    cv2.line(main_bgr, (cx, cy), (cx, cy + dy * blen), bgr_col, 3)

                tag_txt = f"{track.name or oid}" + (f" | {state}" if state else "")
                pil_tag = Image.fromarray(main_bgr[:, :, ::-1])
                d_tag = ImageDraw.Draw(pil_tag)
                tw = min(320, _text_width(d_tag, tag_txt, self.f_reg_12) + 12)
                ty = max(0, y1 - 22) if track.tag_pos == "top" else min(698, y2 + 2)
                d_tag.rectangle((x1, ty, min(1279, x1 + tw), ty + 20), fill=bgr_col[::-1])
                d_tag.text(
                    (x1 + 6, ty + 3),
                    _fit_single_line(d_tag, tag_txt, self.f_reg_12, tw - 8),
                    font=self.f_reg_12,
                    fill=(255, 255, 255),
                )
                main_bgr = np.ascontiguousarray(np.asarray(pil_tag, dtype=np.uint8)[:, :, ::-1])

        stage_idx = _find_active_idx(self.resolved.macro_stages, t_s)
        l_idx = _find_active_idx(self.resolved.arm_timelines.left, t_s)
        r_idx = _find_active_idx(self.resolved.arm_timelines.right, t_s)
        qc_idx = _find_active_idx(self.resolved.qc_segments, t_s)

        # Active QC overlay banner + border on main_bgr
        if 0 <= qc_idx < len(self.resolved.qc_segments):
            seg = self.resolved.qc_segments[qc_idx]
            pol_rgb = _POLICY_COLORS.get(seg.splice_policy, (168, 85, 247))
            pol_bgr = pol_rgb[::-1]
            cv2.rectangle(main_bgr, (0, 0), (1279, 719), pol_bgr, 3)
            pil_m = Image.fromarray(main_bgr[:, :, ::-1])
            dm = ImageDraw.Draw(pil_m)
            banner_lbl = f"{seg.label_id} | {seg.name}" if seg.name else seg.label_id
            banner = f"QC [{banner_lbl}] -> {seg.splice_policy.upper()}"
            bw = min(900, _text_width(dm, banner, self.f_bold_16) + 32)
            bx0 = (1280 - bw) // 2
            dm.rounded_rectangle((bx0, 676, bx0 + bw, 710), radius=6, fill=pol_rgb)
            dm.text((bx0 + 16, 684), _fit_single_line(dm, banner, self.f_bold_16, bw - 24), font=self.f_bold_16, fill=(255, 255, 255))
            main_bgr = np.ascontiguousarray(np.asarray(pil_m, dtype=np.uint8)[:, :, ::-1])

        canvas[60:780, 16:1296] = main_bgr

        # Right top aux panels (291x164)
        l_aux, r_aux = self._aux_frames(main_rgb, t_s)
        canvas[84:248, 1310:1601] = np.ascontiguousarray(l_aux[:, :, ::-1])
        canvas[84:248, 1613:1904] = np.ascontiguousarray(r_aux[:, :, ::-1])

        l_st = self.resolved.arm_timelines.left[l_idx].state if 0 <= l_idx < len(self.resolved.arm_timelines.left) else "idle"
        r_st = self.resolved.arm_timelines.right[r_idx].state if 0 <= r_idx < len(self.resolved.arm_timelines.right) else "idle"
        cv2.rectangle(canvas, (1310, 60), (1600, 247), _ARM_COLORS.get(l_st, (100, 116, 139))[::-1], 2)
        cv2.rectangle(canvas, (1613, 60), (1903, 247), _ARM_COLORS.get(r_st, (100, 116, 139))[::-1], 2)

        # Right info card
        canvas[256:780, 1310:1904] = self._render_info_panel(t_s, stage_idx, l_idx, r_idx, qc_idx)

        # Dynamic supervision pill & clock in top bar
        pill_rgb = (16, 185, 129)
        pill_txt = "NORMAL | KEEP"
        if 0 <= qc_idx < len(self.resolved.qc_segments):
            seg = self.resolved.qc_segments[qc_idx]
            pill_rgb = _POLICY_COLORS.get(seg.splice_policy, (168, 85, 247))
            pill_txt = f"{seg.splice_policy.upper()}"
        top_pil = Image.fromarray(canvas[0:50, 1240:1920, ::-1])
        dt = ImageDraw.Draw(top_pil)
        dt.rounded_rectangle((6, 9, 310, 41), radius=6, fill=pill_rgb)
        dt.text((22, 16), pill_txt, font=self.f_bold_14, fill=(255, 255, 255))
        clock_str = f"t = {t_s:05.2f}s / {self.duration_s:05.2f}s | F {frame_idx + 1:04d}/{total_frames:04d}"
        dt.text((328, 17), clock_str, font=self.f_mono_14, fill=TEXT_RGB)
        canvas[0:50, 1240:1920] = np.ascontiguousarray(np.asarray(top_pil, dtype=np.uint8)[:, :, ::-1])

        # Playhead vertical line on timeline
        tx0, tx1 = 228, 1882
        px = int(round(tx0 + max(0.0, min(1.0, t_s / self.duration_s)) * (tx1 - tx0)))
        cv2.line(canvas, (px, 814), (px, 1022), (255, 255, 255), 2)
        cv2.circle(canvas, (px, 814), 5, (212, 182, 45), -1)

        return canvas


def render_viz_mp4(
    resolved: ResolvedEpisode | None = None,
    view: EpisodeView | None = None,
    out_path: Path | str | None = None,
    *,
    episode: ResolvedEpisode | None = None,
    episode_view: EpisodeView | None = None,
    output_path: Path | str | None = None,
    mp4_path: Path | str | None = None,
    cfg: Any = None,
    config: Any = None,
    fps: float = 30.0,
    crf: int = 20,
    preset: str = "fast",
) -> Path:
    """Render a 1920x1080@30fps Studio HUD MP4 from ResolvedEpisode + EpisodeView."""
    res_ep = resolved or episode
    ev = view or episode_view
    target_raw = out_path or output_path or mp4_path
    assert res_ep is not None and ev is not None and target_raw is not None

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg binary not found in PATH")

    out_p = Path(target_raw)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_p.with_name(f"{out_p.stem}.{time.time_ns()}.partial.mp4")

    target_fps = float(fps) if fps > 0 else 30.0
    duration_s = max(0.05, float(res_ep.meta.duration_s or ev.duration_s))
    total_frames = max(1, int(round(duration_s * target_fps)))
    times_s = [min(duration_s, i / target_fps) for i in range(total_frames)]

    main_cam = ev.camera_roles.get("main") or ev.media.cameras[0].name
    compositor = _HudCompositor(res_ep, ev)

    cmd = [
        ffmpeg_bin,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{CANVAS_W}x{CANVAS_H}",
        "-r",
        f"{target_fps:.6f}",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        str(preset),
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(tmp_out),
    ]

    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert proc.stdin is not None
    try:
        batch_size = 32
        for start in range(0, total_frames, batch_size):
            chunk_t = times_s[start : start + batch_size]
            main_frames = ev.media.read_frames_batch(main_cam, chunk_t)
            for offset, main_rgb in enumerate(main_frames):
                f_idx = start + offset
                bgr_frame = compositor.render_frame_bgr(
                    f_idx, total_frames, chunk_t[offset], main_rgb
                )
                proc.stdin.write(bgr_frame.tobytes())
        proc.stdin.close()
        stderr_bytes = proc.stderr.read() if proc.stderr else b""
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(
                f"ffmpeg exited {rc}: {stderr_bytes.decode('utf-8', 'replace')}"
            )
        tmp_out.replace(out_p)
        return out_p
    except BaseException:
        try:
            proc.kill()
            proc.wait()
        except Exception:
            pass
        tmp_out.unlink(missing_ok=True)
        raise


render_episode_mp4 = render_viz_mp4
render_studio_hud_mp4 = render_viz_mp4


class VizMp4Exporter(OutputExporter):
    """Transactional 1080p Studio HUD MP4 exporter & OOP wrapper around `render_viz_mp4`."""

    name: ClassVar[str] = "viz_mp4"

    def __init__(self, cfg: Any = None, *, config: Any = None) -> None:
        self.cfg = cfg or config

    def render_episode(
        self,
        resolved: ResolvedEpisode | None = None,
        view: EpisodeView | None = None,
        out_path: Path | str | None = None,
        **kwargs: Any,
    ) -> Path:
        return render_viz_mp4(
            resolved=resolved,
            view=view,
            out_path=out_path,
            cfg=self.cfg,
            **kwargs,
        )

    render = render_episode

    def export(
        self,
        pieces: list[KeepPiece],
        resolved_list: list[ResolvedEpisode],
        views: dict[str, EpisodeView],
        out_dir: Path,
        recipe: ExportRecipe,
    ) -> ExportManifest:
        (
            included_pairs,
            entries,
            episodes_included,
            episodes_skipped,
            pieces_exported,
            pieces_dropped,
            total_frames,
        ) = group_kept_pieces(pieces, resolved_list, recipe)

        with staged_export_dir(out_dir) as staging:
            for res, _ in included_pairs:
                view = views.get(res.episode_id)
                if view is None:
                    raise KeyError(f"Missing EpisodeView for {res.episode_id!r}")
                mp4_name = f"{res.meta.dataset}__{res.meta.stem}.mp4"
                render_viz_mp4(res, view, staging / mp4_name, cfg=self.cfg)

            preview = finalize_manifest(
                staging,
                recipe,
                entries,
                episodes_included,
                episodes_skipped,
                pieces_exported,
                pieces_dropped,
                total_frames,
            )
            (staging / "manifest.json").write_text(
                preview.model_dump_json(indent=2), encoding="utf-8"
            )

        return finalize_manifest(
            out_dir,
            recipe,
            entries,
            episodes_included,
            episodes_skipped,
            pieces_exported,
            pieces_dropped,
            total_frames,
            export_id=preview.export_id,
        )
