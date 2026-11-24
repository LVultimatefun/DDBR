import type {
  CameraSpec,
  GroundingKeyframe,
  GroundingResult,
  GroundingStatusKind,
  GroundingTrack,
  GroundingTrackingMode,
} from "../api/types";

export interface ContentRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export type GroundingProvenance = "vlm_anchor" | "tracked" | "static_locked";

export interface SampledTrackBox {
  bbox: [number, number, number, number];
  state: string;
  statusKind: GroundingStatusKind;
  provenance: GroundingProvenance;
  confPct: number | null;
}

export interface LabelPlacement {
  x: number;
  y: number;
}

const DENSE_MAX_INTERP_GAP_S = 0.5;
const SPARSE_MAX_INTERP_GAP_S = 4.5;
const SPARSE_MAX_JUMP_NORM = 150;

function boxCenterDistNorm(
  a: readonly [number, number, number, number],
  b: readonly [number, number, number, number],
): number {
  const cxA = (a[0] + a[2]) * 0.5;
  const cyA = (a[1] + a[3]) * 0.5;
  const cxB = (b[0] + b[2]) * 0.5;
  const cyB = (b[1] + b[3]) * 0.5;
  return Math.hypot(cxB - cxA, cyB - cyA);
}

function clamp(v: number, lo: number, hi: number): number {
  if (!Number.isFinite(v)) return lo;
  return Math.max(lo, Math.min(hi, v));
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function isBoxValid(box: readonly [number, number, number, number] | undefined | null): boolean {
  if (!box || box.length !== 4) return false;
  const [x1, y1, x2, y2] = box;
  return (
    Number.isFinite(x1) &&
    Number.isFinite(y1) &&
    Number.isFinite(x2) &&
    Number.isFinite(y2) &&
    x2 > x1 &&
    y2 > y1
  );
}

/**
 * Compute exact displayed content rectangle for `object-fit: contain` inside a container,
 * handling both letterboxing (top/bottom bars) and pillarboxing (left/right bars).
 */
export function computeContainRect(
  containerW: number,
  containerH: number,
  mediaW: number,
  mediaH: number,
): ContentRect {
  if (!Number.isFinite(containerW) || containerW <= 0 || !Number.isFinite(containerH) || containerH <= 0) {
    return {
      x: 0,
      y: 0,
      width: Number.isFinite(containerW) && containerW > 0 ? containerW : 0,
      height: Number.isFinite(containerH) && containerH > 0 ? containerH : 0,
    };
  }
  if (!Number.isFinite(mediaW) || mediaW <= 0 || !Number.isFinite(mediaH) || mediaH <= 0) {
    return { x: 0, y: 0, width: containerW, height: containerH };
  }

  const containerAR = containerW / containerH;
  const mediaAR = mediaW / mediaH;

  if (Math.abs(containerAR - mediaAR) <= 1e-6) {
    return { x: 0, y: 0, width: containerW, height: containerH };
  }

  if (mediaAR > containerAR) {
    // Wider media -> letterbox (top & bottom bars)
    const width = containerW;
    const height = containerW / mediaAR;
    return {
      x: 0,
      y: (containerH - height) / 2,
      width,
      height,
    };
  }

  // Narrower/taller media -> pillarbox (left & right bars)
  const height = containerH;
  const width = containerH * mediaAR;
  return {
    x: (containerW - width) / 2,
    y: 0,
    width,
    height,
  };
}

/**
 * Project a normalized [0, 1000] bbox [x1, y1, x2, y2] strictly into a displayed video ContentRect.
 */
export function projectNormBoxToRect(
  bboxNorm1000: readonly [number, number, number, number],
  rect: ContentRect,
): ContentRect {
  const rawX1 = clamp(bboxNorm1000[0], 0, 1000);
  const rawY1 = clamp(bboxNorm1000[1], 0, 1000);
  const rawX2 = clamp(bboxNorm1000[2], 0, 1000);
  const rawY2 = clamp(bboxNorm1000[3], 0, 1000);
  const nx1 = Math.min(rawX1, rawX2);
  const nx2 = Math.max(rawX1, rawX2);
  const ny1 = Math.min(rawY1, rawY2);
  const ny2 = Math.max(rawY1, rawY2);

  const x = rect.x + (nx1 / 1000) * rect.width;
  const y = rect.y + (ny1 / 1000) * rect.height;
  const width = Math.max(2, ((nx2 - nx1) / 1000) * rect.width);
  const height = Math.max(2, ((ny2 - ny1) / 1000) * rect.height);
  return { x, y, width, height };
}

/**
 * Clamp label pill position strictly inside the video content rectangle so tags never bleed into black bars.
 */
export function computeLabelPlacement(
  box: ContentRect,
  contentRect: ContentRect,
  labelW: number,
  labelH: number,
  tagPos: "top" | "bottom",
): LabelPlacement {
  const minX = contentRect.x;
  const maxX = Math.max(minX, contentRect.x + contentRect.width - labelW);
  const x = clamp(box.x, minX, maxX);

  const minY = contentRect.y;
  const maxY = Math.max(minY, contentRect.y + contentRect.height - labelH);

  let preferredY: number;
  if (tagPos === "bottom") {
    const below = box.y + box.height + 2;
    const above = box.y - labelH - 2;
    preferredY = below <= maxY ? below : above >= minY ? above : below;
  } else {
    const above = box.y - labelH - 2;
    const insideTop = box.y + 2;
    preferredY = above >= minY ? above : insideTop <= maxY ? insideTop : minY;
  }

  return {
    x,
    y: clamp(preferredY, minY, maxY),
  };
}

/**
 * Determine which camera the GroundingResult belongs to (explicit schema v2 camera_name or primary fallback).
 */
export function resolveGroundedCameraName(
  grounding: GroundingResult | null | undefined,
  primaryCamera: string,
): string {
  const explicit = grounding?.camera_name?.trim();
  return explicit ? explicit : primaryCamera;
}

/**
 * Resolve natural media dimensions in priority order:
 * active <video> videoWidth/Height -> HD <img> naturalWidth/Height -> CameraSpec -> grounding.ref_size -> fallback.
 */
export function resolveMediaDimensions(params: {
  videoEl?: Pick<HTMLVideoElement, "videoWidth" | "videoHeight"> | null;
  imgEl?: Pick<HTMLImageElement, "naturalWidth" | "naturalHeight"> | null;
  hdFrameMode: boolean;
  videoFailed: boolean;
  cameraSpec?: Pick<CameraSpec, "width" | "height"> | null;
  grounding?: Pick<GroundingResult, "ref_size"> | null;
  fallbackW: number;
  fallbackH: number;
}): [number, number] {
  const { videoEl, imgEl, hdFrameMode, videoFailed, cameraSpec, grounding, fallbackW, fallbackH } = params;

  if (!hdFrameMode && !videoFailed && videoEl && videoEl.videoWidth > 0 && videoEl.videoHeight > 0) {
    return [videoEl.videoWidth, videoEl.videoHeight];
  }
  if ((hdFrameMode || videoFailed) && imgEl && imgEl.naturalWidth > 0 && imgEl.naturalHeight > 0) {
    return [imgEl.naturalWidth, imgEl.naturalHeight];
  }
  if (cameraSpec && cameraSpec.width > 0 && cameraSpec.height > 0) {
    return [cameraSpec.width, cameraSpec.height];
  }
  if (
    grounding?.ref_size &&
    grounding.ref_size.length >= 2 &&
    grounding.ref_size[0] > 0 &&
    grounding.ref_size[1] > 0
  ) {
    return [grounding.ref_size[0], grounding.ref_size[1]];
  }
  return [fallbackW, fallbackH];
}

/**
 * O(log N) binary search returning enclosing bracket [lo, hi] in a sorted ascending timestamp array.
 */
export function findTimeBracket(times: readonly number[], tS: number): [number, number] {
  const n = times.length;
  if (n <= 1) return [0, 0];
  if (tS <= times[0]) return [0, 0];
  if (tS >= times[n - 1]) return [n - 1, n - 1];

  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (times[mid] <= tS) lo = mid;
    else hi = mid;
  }
  return [lo, hi];
}

function findKeyframeBracket(keyframes: readonly GroundingKeyframe[], tS: number): [number, number] {
  const n = keyframes.length;
  if (n <= 1) return [0, 0];
  if (tS <= keyframes[0].time_s) return [0, 0];
  if (tS >= keyframes[n - 1].time_s) return [n - 1, n - 1];

  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (keyframes[mid].time_s <= tS) lo = mid;
    else hi = mid;
  }
  return [lo, hi];
}

function pickNearestVisibleAnchorMeta(
  keyframes: readonly GroundingKeyframe[] | undefined,
  tS: number,
): { state: string; statusKind: GroundingStatusKind } {
  if (!keyframes || keyframes.length === 0) {
    return { state: "", statusKind: "neutral" };
  }
  const [lo, hi] = findKeyframeBracket(keyframes, tS);

  let leftIdx = lo;
  while (leftIdx >= 0 && !keyframes[leftIdx].visible) {
    leftIdx -= 1;
  }
  let rightIdx = hi;
  while (rightIdx < keyframes.length && !keyframes[rightIdx].visible) {
    rightIdx += 1;
  }

  let chosen: GroundingKeyframe | null = null;
  if (leftIdx >= 0 && rightIdx < keyframes.length) {
    const dL = Math.abs(tS - keyframes[leftIdx].time_s);
    const dR = Math.abs(keyframes[rightIdx].time_s - tS);
    chosen = dL <= dR ? keyframes[leftIdx] : keyframes[rightIdx];
  } else if (leftIdx >= 0) {
    chosen = keyframes[leftIdx];
  } else if (rightIdx < keyframes.length) {
    chosen = keyframes[rightIdx];
  }

  if (!chosen) {
    return { state: "", statusKind: "neutral" };
  }
  return {
    state: chosen.state || "",
    statusKind: chosen.status_kind || "neutral",
  };
}

function flagToProvenance(
  flag: number,
  trackingMode?: GroundingTrackingMode,
  isStaticContainer?: boolean,
): GroundingProvenance {
  if (flag === 2) return "vlm_anchor";
  if (flag === 3 || trackingMode === "static_locked") return "static_locked";
  if (flag === 1) return "tracked";
  return isStaticContainer ? "static_locked" : "tracked";
}

/**
 * Sample a 2D grounding track at timestamp `tS` with O(log N) binary search:
 * - Dense frame-level path: linear interpolation between adjacent visible dense frames (dt <= 0.5s),
 *   occlusion hiding whenever flag === 0, provenance & confidence tracking, and state/statusKind from
 *   nearest visible VLM anchor.
 * - Legacy sparse VLM fallback: smoothstep interpolation between adjacent visible keyframes (dt <= 4.5s
 *   and center jump <= 150 normalized units), snapping to nearest when jump > 150 or gap > 4.5s, and
 *   occlusion hiding when nearest anchor has visible === false.
 */
export function sampleTrackBoxAt(
  grounding: GroundingResult | null | undefined,
  track: GroundingTrack,
  tS: number,
): SampledTrackBox | null {
  const safeT = Number.isFinite(tS) ? Math.max(0, tS) : 0;

  const denseTimes = grounding?.dense_times_s;
  const denseBoxes = track.dense_boxes_norm_1000;
  const hasDense =
    Array.isArray(denseTimes) &&
    denseTimes.length > 0 &&
    Array.isArray(denseBoxes) &&
    denseBoxes.length === denseTimes.length;

  if (hasDense) {
    const flags = track.dense_flags;
    const hasFlags = Array.isArray(flags) && flags.length === denseTimes.length;
    const confs = track.dense_conf_pct;
    const hasConfs = Array.isArray(confs) && confs.length === denseTimes.length;

    const flagAt = (idx: number): number => {
      if (hasFlags) return flags[idx];
      if (!isBoxValid(denseBoxes[idx])) return 0;
      return track.tracking_mode === "static_locked" ? 3 : 1;
    };

    const confAt = (idx: number): number | null => {
      if (!hasConfs) return null;
      const val = confs[idx];
      return Number.isFinite(val) ? Math.round(clamp(val, 0, 100)) : null;
    };

    const [i0, i1] = findTimeBracket(denseTimes, safeT);
    const f0 = flagAt(i0);
    const f1 = flagAt(i1);
    const b0 = denseBoxes[i0];
    const b1 = denseBoxes[i1];
    const v0 = f0 > 0 && isBoxValid(b0);
    const v1 = f1 > 0 && isBoxValid(b1);

    if (i0 === i1) {
      if (!v0) return null;
      const meta = pickNearestVisibleAnchorMeta(track.keyframes, safeT);
      return {
        bbox: [b0[0], b0[1], b0[2], b0[3]],
        state: meta.state,
        statusKind: meta.statusKind,
        provenance: flagToProvenance(f0, track.tracking_mode, track.is_static_container),
        confPct: confAt(i0),
      };
    }

    const t0 = denseTimes[i0];
    const t1 = denseTimes[i1];
    const dt = t1 - t0;

    if (v0 && v1 && dt > 1e-6 && dt <= DENSE_MAX_INTERP_GAP_S) {
      const alpha = clamp((safeT - t0) / dt, 0, 1);
      const bbox: [number, number, number, number] = [
        lerp(b0[0], b1[0], alpha),
        lerp(b0[1], b1[1], alpha),
        lerp(b0[2], b1[2], alpha),
        lerp(b0[3], b1[3], alpha),
      ];

      const anchorTol = Math.min(0.035, dt * 0.35);
      let provenance: GroundingProvenance;
      if (f0 === 2 && Math.abs(safeT - t0) <= anchorTol) {
        provenance = "vlm_anchor";
      } else if (f1 === 2 && Math.abs(t1 - safeT) <= anchorTol) {
        provenance = "vlm_anchor";
      } else if ((f0 === 3 && f1 === 3) || track.tracking_mode === "static_locked") {
        provenance = "static_locked";
      } else {
        provenance = "tracked";
      }

      const c0 = confAt(i0);
      const c1 = confAt(i1);
      const confPct = c0 !== null && c1 !== null ? Math.round(lerp(c0, c1, alpha)) : c0 ?? c1;
      const meta = pickNearestVisibleAnchorMeta(track.keyframes, safeT);

      return {
        bbox,
        state: meta.state,
        statusKind: meta.statusKind,
        provenance,
        confPct,
      };
    }

    // Boundary crosses flag === 0 (occlusion) or large gap (> 0.5s): snap to nearest
    const nearestIdx = Math.abs(safeT - t0) <= Math.abs(t1 - safeT) ? i0 : i1;
    const nf = flagAt(nearestIdx);
    const nb = denseBoxes[nearestIdx];
    if (nf === 0 || !isBoxValid(nb)) return null;

    const meta = pickNearestVisibleAnchorMeta(track.keyframes, safeT);
    return {
      bbox: [nb[0], nb[1], nb[2], nb[3]],
      state: meta.state,
      statusKind: meta.statusKind,
      provenance: flagToProvenance(nf, track.tracking_mode, track.is_static_container),
      confPct: confAt(nearestIdx),
    };
  }

  // Legacy sparse VLM fallback (schema_version=1 without dense arrays)
  const kfs = track.keyframes;
  if (!kfs || kfs.length === 0) return null;

  const [i0, i1] = findKeyframeBracket(kfs, safeT);
  const k0 = kfs[i0];
  const k1 = kfs[i1];
  const nearest = Math.abs(safeT - k0.time_s) <= Math.abs(k1.time_s - safeT) ? k0 : k1;
  const staticMode = track.tracking_mode === "static_locked" || track.is_static_container;

  if (i0 === i1) {
    if (!k0.visible || !isBoxValid(k0.bbox_norm_1000)) return null;
    return {
      bbox: [k0.bbox_norm_1000[0], k0.bbox_norm_1000[1], k0.bbox_norm_1000[2], k0.bbox_norm_1000[3]],
      state: k0.state || "",
      statusKind: k0.status_kind || "neutral",
      provenance: staticMode ? "static_locked" : "vlm_anchor",
      confPct: null,
    };
  }

  const v0 = k0.visible && isBoxValid(k0.bbox_norm_1000);
  const v1 = k1.visible && isBoxValid(k1.bbox_norm_1000);
  const dt = k1.time_s - k0.time_s;

  if (
    v0 &&
    v1 &&
    dt > 1e-6 &&
    dt <= SPARSE_MAX_INTERP_GAP_S &&
    boxCenterDistNorm(k0.bbox_norm_1000, k1.bbox_norm_1000) <= SPARSE_MAX_JUMP_NORM
  ) {
    const u = clamp((safeT - k0.time_s) / dt, 0, 1);
    const smooth = u * u * (3 - 2 * u);
    const b0 = k0.bbox_norm_1000;
    const b1 = k1.bbox_norm_1000;
    const nearAnchor = Math.abs(safeT - nearest.time_s) <= 0.05;
    return {
      bbox: [
        lerp(b0[0], b1[0], smooth),
        lerp(b0[1], b1[1], smooth),
        lerp(b0[2], b1[2], smooth),
        lerp(b0[3], b1[3], smooth),
      ],
      state: nearest.state || "",
      statusKind: nearest.status_kind || "neutral",
      provenance: staticMode ? "static_locked" : nearAnchor ? "vlm_anchor" : "tracked",
      confPct: null,
    };
  }

  // Occlusion, sparse gap > 4.5s, or jump > 150 normalized units: snap to nearest; hide if nearest anchor is invisible
  if (!nearest.visible || !isBoxValid(nearest.bbox_norm_1000)) return null;
  return {
    bbox: [
      nearest.bbox_norm_1000[0],
      nearest.bbox_norm_1000[1],
      nearest.bbox_norm_1000[2],
      nearest.bbox_norm_1000[3],
    ],
    state: nearest.state || "",
    statusKind: nearest.status_kind || "neutral",
    provenance: staticMode ? "static_locked" : "vlm_anchor",
    confPct: null,
  };
}
