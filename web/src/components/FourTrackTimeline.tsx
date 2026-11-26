import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  Loader2,
  Lock,
  Magnet,
  Plus,
  Scissors,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import type {
  ArmState,
  ArmTimelines,
  MacroStage,
  QcSegment,
  SplicePolicy,
  WatchCoverage,
} from "../api/types";
import { computeUnwatchedIntervals } from "../stores/reviewHistoryStore";
import { useI18n } from "../stores/i18nStore";

export interface StageSpanUpdate {
  stageId: string;
  tStart: number;
  tEnd: number;
}

export interface FourTrackTimelineProps {
  durationS: number;
  currentTime: number;
  macroStages: MacroStage[];
  armTimelines: ArmTimelines;
  qcSegments: QcSegment[];
  rawVlmSegments?: QcSegment[];
  watchCoverage?: WatchCoverage;
  showDiffTrack?: boolean;
  readOnly?: boolean;
  snapEnabled?: boolean;
  compact?: boolean;
  selectedSegmentId: string | null;
  selectedStageId: string | null;
  onSeek: (tS: number) => void;
  onSelectSegment: (segId: string | null) => void;
  onSelectStage: (stageId: string | null) => void;
  onAddSegment: () => void;
  onAddMacroStage: () => void;
  onCreateSegmentSpan?: (tStart: number, tEnd: number) => void;
  onSplitMacroStage?: () => void;
  onUpdateSegmentSpan?: (segId: string, tStart: number, tEnd: number) => void;
  onUpdateStageSpan?: (stageId: string, tStart: number, tEnd: number) => void;
  onUpdateStageSpans?: (updates: StageSpanUpdate[]) => void;
  onAlignStagesToQc?: () => void;
  onInferSubtasks?: () => void;
  inferringSubtasks?: boolean;
}

type ActiveDrag =
  | {
      kind: "segment";
      segId: string;
      edge: "left" | "right";
      tStart: number;
      tEnd: number;
      origStart: number;
      origEnd: number;
      snappedTime?: number | null;
    }
  | {
      kind: "stage";
      stageId: string;
      edge: "left" | "right";
      tStart: number;
      tEnd: number;
      origStart: number;
      origEnd: number;
      snappedToQcTime?: number | null;
      neighbor?: {
        stageId: string;
        tStart: number;
        tEnd: number;
        origStart: number;
        origEnd: number;
        linked: boolean;
      };
    }
  | {
      kind: "brush";
      anchorT: number;
      tStart: number;
      tEnd: number;
    };

interface PendingScrollAnchor {
  ratio: number;
  viewportOffsetX: number;
}

const ARM_STATE_COLORS: Record<ArmState, string> = {
  idle: "#64748b",
  work: "#2563eb",
  hold: "#0d9488",
  anom: "#ef4444",
  mask: "#f59e0b",
  good: "#16a34a",
};

const POLICY_CLASS: Record<SplicePolicy, string> = {
  keep_positive: "seg-keep",
  mask_loss: "seg-mask",
  cut_clean: "seg-cut",
};

const TRACK_LABEL_WIDTH_PX = 120;
const MIN_ZOOM = 1;
const MAX_ZOOM = 8;
const ZOOM_PRESETS = [1, 2, 4] as const;
const MIN_SPAN_S = 0.1;
const MAGNETIC_GAP_S = 0.05;
const MAGNETIC_SNAP_THRESHOLD_S = 0.15;
const QC_SNAP_THRESHOLD_S = 0.08;
const MIN_BRUSH_SPAN_S = 0.2;

function snapTimeToTargets(
  rawT: number,
  targets: number[],
  minT: number,
  maxT: number,
  thresholdS: number,
): { value: number; snappedTo: number | null } {
  let bestTarget: number | null = null;
  let bestDist = thresholdS + 1e-9;
  for (const candidate of targets) {
    if (!Number.isFinite(candidate)) continue;
    if (candidate < minT - 1e-9 || candidate > maxT + 1e-9) continue;
    const dist = Math.abs(candidate - rawT);
    if (dist <= bestDist) {
      bestDist = dist;
      bestTarget = clamp(candidate, minT, maxT);
    }
  }
  return bestTarget !== null
    ? { value: bestTarget, snappedTo: bestTarget }
    : { value: rawT, snappedTo: null };
}

function snapTimeToQcBoundary(
  rawT: number,
  qcSegments: QcSegment[],
  minT: number,
  maxT: number,
): { value: number; snappedTo: number | null } {
  const targets: number[] = [];
  for (const seg of qcSegments) {
    targets.push(seg.t_start, seg.t_end);
  }
  return snapTimeToTargets(rawT, targets, minT, maxT, QC_SNAP_THRESHOLD_S);
}

function pct(t: number, durationS: number): number {
  if (durationS <= 0) return 0;
  return Math.max(0, Math.min(100, (t / durationS) * 100));
}

function roundTime(t: number): number {
  return Number(t.toFixed(2));
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function rgbCss(rgb: [number, number, number]): string {
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

function clientXToTime(clientX: number, targetEl: HTMLElement | null, durationS: number): number {
  if (!targetEl || durationS <= 0) return 0;
  const rect = targetEl.getBoundingClientRect();
  if (rect.width <= 0) return 0;
  const ratio = clamp((clientX - rect.left) / rect.width, 0, 1);
  return Number((ratio * durationS).toFixed(3));
}

export const FourTrackTimeline: React.FC<FourTrackTimelineProps> = ({
  durationS,
  currentTime,
  macroStages,
  armTimelines,
  qcSegments,
  rawVlmSegments = [],
  watchCoverage,
  showDiffTrack = false,
  readOnly = false,
  snapEnabled = true,
  compact = false,
  selectedSegmentId,
  selectedStageId,
  onSeek,
  onSelectSegment,
  onSelectStage,
  onAddSegment,
  onAddMacroStage,
  onCreateSegmentSpan,
  onSplitMacroStage,
  onUpdateSegmentSpan,
  onUpdateStageSpan,
  onUpdateStageSpans,
  onAlignStagesToQc,
  onInferSubtasks,
  inferringSubtasks = false,
}) => {
  const { t } = useI18n();
  const [zoom, setZoom] = useState<number>(1);
  const [activeDrag, setActiveDrag] = useState<ActiveDrag | null>(null);

  const scrollRef = useRef<HTMLDivElement | null>(null);
  const rulerRef = useRef<HTMLDivElement | null>(null);
  const macroLaneRef = useRef<HTMLDivElement | null>(null);
  const qcLaneRef = useRef<HTMLDivElement | null>(null);
  const rafIdRef = useRef<number | null>(null);
  const pendingSeekRef = useRef<number | null>(null);
  const suppressClickRef = useRef<boolean>(false);
  const cleanupDragListenersRef = useRef<(() => void) | null>(null);
  const pendingScrollAnchorRef = useRef<PendingScrollAnchor | null>(null);

  const zoomRef = useRef<number>(zoom);
  zoomRef.current = zoom;
  const currentTimeRef = useRef<number>(currentTime);
  currentTimeRef.current = currentTime;
  const durationRef = useRef<number>(durationS);
  durationRef.current = durationS;

  useEffect(() => {
    return () => {
      cleanupDragListenersRef.current?.();
      if (rafIdRef.current !== null) {
        window.cancelAnimationFrame(rafIdRef.current);
        rafIdRef.current = null;
      }
    };
  }, []);

  // Re-anchor horizontal scroll around cursor/playhead timestamp right after zoom commits
  useLayoutEffect(() => {
    const anchor = pendingScrollAnchorRef.current;
    const scrollEl = scrollRef.current;
    if (!anchor || !scrollEl) return;
    pendingScrollAnchorRef.current = null;

    if (zoom <= 1) {
      scrollEl.scrollLeft = 0;
      return;
    }

    const clientW = scrollEl.clientWidth;
    const canvasWidth =
      scrollEl.scrollWidth > clientW ? scrollEl.scrollWidth : Math.max(clientW, 1) * zoom;
    const newLaneWidth = Math.max(1, canvasWidth - TRACK_LABEL_WIDTH_PX);
    const targetContentX = TRACK_LABEL_WIDTH_PX + anchor.ratio * newLaneWidth;
    const maxScroll = Math.max(0, canvasWidth - clientW);
    scrollEl.scrollLeft = clamp(targetContentX - anchor.viewportOffsetX, 0, maxScroll);
  }, [zoom]);

  // Native non-passive wheel listener for smooth Ctrl/Alt/Meta + Wheel cursor-anchored zoom
  useEffect(() => {
    const scrollEl = scrollRef.current;
    if (!scrollEl) return;

    const handleWheel = (e: WheelEvent) => {
      if (!(e.ctrlKey || e.altKey || e.metaKey)) return;
      e.preventDefault();

      const curZoom = zoomRef.current;
      const factor = e.deltaY < 0 ? 1.25 : 0.8;
      const nextZoom = clamp(Number((curZoom * factor).toFixed(2)), MIN_ZOOM, MAX_ZOOM);
      if (Math.abs(nextZoom - curZoom) < 1e-4) return;

      const rect = scrollEl.getBoundingClientRect();
      const clientW = scrollEl.clientWidth || rect.width;
      const oldCanvasWidth =
        scrollEl.scrollWidth > clientW ? scrollEl.scrollWidth : Math.max(clientW, 1) * curZoom;
      const oldLaneWidth = Math.max(1, oldCanvasWidth - TRACK_LABEL_WIDTH_PX);
      const pointerViewportX = e.clientX - rect.left;
      const oldContentX = scrollEl.scrollLeft + pointerViewportX;

      let ratio: number;
      let viewportOffsetX: number;
      if (rect.width > 0 && oldContentX >= TRACK_LABEL_WIDTH_PX) {
        ratio = clamp((oldContentX - TRACK_LABEL_WIDTH_PX) / oldLaneWidth, 0, 1);
        viewportOffsetX = pointerViewportX;
      } else {
        const dur = durationRef.current;
        ratio = dur > 0 ? clamp(currentTimeRef.current / dur, 0, 1) : 0;
        viewportOffsetX =
          clientW > TRACK_LABEL_WIDTH_PX
            ? TRACK_LABEL_WIDTH_PX + (clientW - TRACK_LABEL_WIDTH_PX) * 0.5
            : TRACK_LABEL_WIDTH_PX;
      }

      pendingScrollAnchorRef.current = { ratio, viewportOffsetX };
      setZoom(nextZoom);
    };

    scrollEl.addEventListener("wheel", handleWheel, { passive: false });
    return () => {
      scrollEl.removeEventListener("wheel", handleWheel);
    };
  }, []);

  const applyPlayheadAnchoredZoom = (nextZoom: number) => {
    const clampedZoom = clamp(nextZoom, MIN_ZOOM, MAX_ZOOM);
    if (Math.abs(clampedZoom - zoom) < 1e-4) return;
    const scrollEl = scrollRef.current;
    if (scrollEl && durationS > 0) {
      const clientW = scrollEl.clientWidth || scrollEl.getBoundingClientRect().width;
      const ratio = clamp(currentTime / durationS, 0, 1);
      const viewportOffsetX =
        clientW > TRACK_LABEL_WIDTH_PX
          ? TRACK_LABEL_WIDTH_PX + (clientW - TRACK_LABEL_WIDTH_PX) * 0.5
          : TRACK_LABEL_WIDTH_PX;
      pendingScrollAnchorRef.current = { ratio, viewportOffsetX };
    }
    setZoom(clampedZoom);
  };

  const scheduleSeek = (tS: number) => {
    pendingSeekRef.current = tS;
    if (rafIdRef.current !== null) return;
    rafIdRef.current = window.requestAnimationFrame(() => {
      rafIdRef.current = null;
      if (pendingSeekRef.current !== null) {
        const nextT = pendingSeekRef.current;
        pendingSeekRef.current = null;
        onSeek(nextT);
      }
    });
  };

  const flushSeek = (tS: number) => {
    if (rafIdRef.current !== null) {
      window.cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    }
    pendingSeekRef.current = null;
    onSeek(tS);
  };

  const markSuppressNextClick = () => {
    suppressClickRef.current = true;
    window.setTimeout(() => {
      suppressClickRef.current = false;
    }, 0);
  };

  const ticks = useMemo(() => {
    const count = Math.max(4, Math.min(16, Math.ceil(durationS / 5) + 1));
    const arr: number[] = [];
    for (let i = 0; i <= count; i += 1) {
      arr.push((durationS * i) / count);
    }
    return arr;
  }, [durationS]);

  const watchedIntervals = watchCoverage?.watched_intervals ?? [];
  const unwatchedGaps = useMemo(
    () => computeUnwatchedIntervals(watchedIntervals, durationS),
    [watchedIntervals, durationS],
  );

  const canSplitStage = useMemo(
    () =>
      !readOnly &&
      macroStages.some(
        (st) => currentTime >= st.t_start + MIN_SPAN_S && currentTime <= st.t_end - MIN_SPAN_S,
      ),
    [currentTime, macroStages, readOnly],
  );

  const seekFromClientX = (clientX: number, targetEl: HTMLElement | null) => {
    if (!targetEl || durationS <= 0) return;
    onSeek(clientXToTime(clientX, targetEl, durationS));
  };

  const startSegmentEdgeDrag = (
    e: React.PointerEvent<HTMLDivElement>,
    seg: QcSegment,
    edge: "left" | "right",
  ) => {
    if (readOnly || e.button !== 0 || !onUpdateSegmentSpan) return;
    e.stopPropagation();
    e.preventDefault();
    try {
      e.currentTarget.setPointerCapture?.(e.pointerId);
    } catch {
      // jsdom does not implement pointer capture
    }
    onSelectSegment(seg.seg_id);
    onSelectStage(null);

    // Freeze snap targets at drag start so scrub preview does not pull the playhead target along
    const snapTargets: number[] = [currentTime];
    for (const st of macroStages) {
      snapTargets.push(st.t_start, st.t_end);
    }
    for (const other of qcSegments) {
      if (other.seg_id === seg.seg_id) continue;
      snapTargets.push(other.t_start, other.t_end);
    }

    let latest: Extract<ActiveDrag, { kind: "segment" }> = {
      kind: "segment",
      segId: seg.seg_id,
      edge,
      tStart: seg.t_start,
      tEnd: seg.t_end,
      origStart: seg.t_start,
      origEnd: seg.t_end,
      snappedTime: null,
    };
    setActiveDrag(latest);

    const onMove = (ev: PointerEvent) => {
      const rawT = clientXToTime(ev.clientX, qcLaneRef.current, durationS);
      const allowSnap = snapEnabled && !ev.shiftKey && !ev.altKey;
      if (edge === "left") {
        const minStart = 0;
        const maxStart = Math.max(0, latest.origEnd - MIN_SPAN_S);
        const clampedStart = clamp(rawT, minStart, maxStart);
        const snapped = allowSnap
          ? snapTimeToTargets(clampedStart, snapTargets, minStart, maxStart, MAGNETIC_SNAP_THRESHOLD_S)
          : { value: clampedStart, snappedTo: null };
        latest = {
          ...latest,
          tStart: snapped.value,
          tEnd: latest.origEnd,
          snappedTime: snapped.snappedTo,
        };
        setActiveDrag({ ...latest });
        scheduleSeek(snapped.value);
      } else {
        const minEnd = Math.min(durationS, latest.origStart + MIN_SPAN_S);
        const maxEnd = durationS;
        const clampedEnd = clamp(rawT, minEnd, maxEnd);
        const snapped = allowSnap
          ? snapTimeToTargets(clampedEnd, snapTargets, minEnd, maxEnd, MAGNETIC_SNAP_THRESHOLD_S)
          : { value: clampedEnd, snappedTo: null };
        latest = {
          ...latest,
          tStart: latest.origStart,
          tEnd: snapped.value,
          snappedTime: snapped.snappedTo,
        };
        setActiveDrag({ ...latest });
        scheduleSeek(snapped.value);
      }
    };

    const cleanup = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
      cleanupDragListenersRef.current = null;
    };

    const onUp = () => {
      cleanup();
      setActiveDrag(null);
      markSuppressNextClick();
      const finalEdge = latest.edge === "left" ? latest.tStart : latest.tEnd;
      flushSeek(finalEdge);
      const changed =
        Math.abs(latest.tStart - latest.origStart) > 1e-4 ||
        Math.abs(latest.tEnd - latest.origEnd) > 1e-4;
      if (changed) {
        onUpdateSegmentSpan(seg.seg_id, roundTime(latest.tStart), roundTime(latest.tEnd));
      }
    };

    cleanupDragListenersRef.current?.();
    cleanupDragListenersRef.current = cleanup;
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
  };

  const startStageEdgeDrag = (
    e: React.PointerEvent<HTMLDivElement>,
    stage: MacroStage,
    edge: "left" | "right",
  ) => {
    if (readOnly || e.button !== 0 || (!onUpdateStageSpan && !onUpdateStageSpans)) return;
    e.stopPropagation();
    e.preventDefault();
    try {
      e.currentTarget.setPointerCapture?.(e.pointerId);
    } catch {
      // jsdom does not implement pointer capture
    }
    onSelectStage(stage.stage_id);
    onSelectSegment(null);

    // Find magnetic neighbor touching this edge (< 0.05s gap)
    let neighborStage: MacroStage | null = null;
    let bestGap = MAGNETIC_GAP_S;
    for (const candidate of macroStages) {
      if (candidate.stage_id === stage.stage_id) continue;
      const gap =
        edge === "right"
          ? Math.abs(candidate.t_start - stage.t_end)
          : Math.abs(candidate.t_end - stage.t_start);
      if (gap < bestGap) {
        bestGap = gap;
        neighborStage = candidate;
      }
    }

    let latest: Extract<ActiveDrag, { kind: "stage" }> = {
      kind: "stage",
      stageId: stage.stage_id,
      edge,
      tStart: stage.t_start,
      tEnd: stage.t_end,
      origStart: stage.t_start,
      origEnd: stage.t_end,
      neighbor: neighborStage
        ? {
            stageId: neighborStage.stage_id,
            tStart: neighborStage.t_start,
            tEnd: neighborStage.t_end,
            origStart: neighborStage.t_start,
            origEnd: neighborStage.t_end,
            linked: !e.altKey,
          }
        : undefined,
    };
    setActiveDrag(latest);

    const onMove = (ev: PointerEvent) => {
      const rawT = clientXToTime(ev.clientX, macroLaneRef.current, durationS);
      const linked = Boolean(latest.neighbor) && !ev.altKey;
      const allowSnap = snapEnabled && !ev.altKey && !ev.shiftKey && qcSegments.length > 0;

      if (edge === "right") {
        const maxEnd =
          linked && latest.neighbor
            ? Math.min(durationS, latest.neighbor.origEnd - MIN_SPAN_S)
            : durationS;
        const minEnd = Math.min(maxEnd, latest.origStart + MIN_SPAN_S);
        const clampedEnd = clamp(rawT, minEnd, maxEnd);
        const snapped = allowSnap
          ? snapTimeToQcBoundary(clampedEnd, qcSegments, minEnd, maxEnd)
          : { value: clampedEnd, snappedTo: null };
        const nextEnd = snapped.value;
        latest = {
          ...latest,
          tStart: latest.origStart,
          tEnd: nextEnd,
          snappedToQcTime: snapped.snappedTo,
          neighbor: latest.neighbor
            ? {
                ...latest.neighbor,
                linked,
                tStart: linked ? nextEnd : latest.neighbor.origStart,
                tEnd: latest.neighbor.origEnd,
              }
            : undefined,
        };
        setActiveDrag({ ...latest });
        scheduleSeek(nextEnd);
      } else {
        const minStart =
          linked && latest.neighbor ? Math.max(0, latest.neighbor.origStart + MIN_SPAN_S) : 0;
        const maxStart = Math.max(minStart, latest.origEnd - MIN_SPAN_S);
        const clampedStart = clamp(rawT, minStart, maxStart);
        const snapped = allowSnap
          ? snapTimeToQcBoundary(clampedStart, qcSegments, minStart, maxStart)
          : { value: clampedStart, snappedTo: null };
        const nextStart = snapped.value;
        latest = {
          ...latest,
          tStart: nextStart,
          tEnd: latest.origEnd,
          snappedToQcTime: snapped.snappedTo,
          neighbor: latest.neighbor
            ? {
                ...latest.neighbor,
                linked,
                tStart: latest.neighbor.origStart,
                tEnd: linked ? nextStart : latest.neighbor.origEnd,
              }
            : undefined,
        };
        setActiveDrag({ ...latest });
        scheduleSeek(nextStart);
      }
    };

    const cleanup = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
      cleanupDragListenersRef.current = null;
    };

    const onUp = () => {
      cleanup();
      setActiveDrag(null);
      markSuppressNextClick();
      const finalEdge = latest.edge === "left" ? latest.tStart : latest.tEnd;
      flushSeek(finalEdge);

      const primaryChanged =
        Math.abs(latest.tStart - latest.origStart) > 1e-4 ||
        Math.abs(latest.tEnd - latest.origEnd) > 1e-4;
      const neighborChanged = Boolean(
        latest.neighbor &&
          latest.neighbor.linked &&
          (Math.abs(latest.neighbor.tStart - latest.neighbor.origStart) > 1e-4 ||
            Math.abs(latest.neighbor.tEnd - latest.neighbor.origEnd) > 1e-4),
      );

      if (!primaryChanged && !neighborChanged) return;

      const updates: StageSpanUpdate[] = [
        {
          stageId: stage.stage_id,
          tStart: roundTime(latest.tStart),
          tEnd: roundTime(latest.tEnd),
        },
      ];
      if (neighborChanged && latest.neighbor) {
        updates.push({
          stageId: latest.neighbor.stageId,
          tStart: roundTime(latest.neighbor.tStart),
          tEnd: roundTime(latest.neighbor.tEnd),
        });
      }

      if (updates.length > 1 && onUpdateStageSpans) {
        onUpdateStageSpans(updates);
      } else if (onUpdateStageSpan) {
        onUpdateStageSpan(updates[0].stageId, updates[0].tStart, updates[0].tEnd);
        if (updates.length > 1) {
          onUpdateStageSpan(updates[1].stageId, updates[1].tStart, updates[1].tEnd);
        }
      } else if (onUpdateStageSpans) {
        onUpdateStageSpans(updates);
      }
    };

    cleanupDragListenersRef.current?.();
    cleanupDragListenersRef.current = cleanup;
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
  };

  const startQcLaneBrush = (e: React.PointerEvent<HTMLDivElement>) => {
    if (readOnly || e.button !== 0 || e.target !== e.currentTarget) return;
    e.preventDefault();
    const laneEl = qcLaneRef.current ?? e.currentTarget;
    const anchorT = clientXToTime(e.clientX, laneEl, durationS);

    let latest: Extract<ActiveDrag, { kind: "brush" }> = {
      kind: "brush",
      anchorT,
      tStart: anchorT,
      tEnd: anchorT,
    };
    setActiveDrag(latest);

    const onMove = (ev: PointerEvent) => {
      const curT = clientXToTime(ev.clientX, laneEl, durationS);
      latest = {
        kind: "brush",
        anchorT,
        tStart: Math.min(anchorT, curT),
        tEnd: Math.max(anchorT, curT),
      };
      setActiveDrag({ ...latest });
      scheduleSeek(curT);
    };

    const cleanup = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
      cleanupDragListenersRef.current = null;
    };

    const onUp = () => {
      cleanup();
      setActiveDrag(null);
      markSuppressNextClick();
      if (latest.tEnd - latest.tStart >= MIN_BRUSH_SPAN_S && onCreateSegmentSpan) {
        const s0 = roundTime(latest.tStart);
        const s1 = roundTime(latest.tEnd);
        flushSeek(s0);
        onCreateSegmentSpan(s0, s1);
      } else {
        flushSeek(anchorT);
      }
    };

    cleanupDragListenersRef.current?.();
    cleanupDragListenersRef.current = cleanup;
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
  };

  const getStageRenderBounds = (
    stage: MacroStage,
  ): { tStart: number; tEnd: number; dragging: boolean; magneticLinked: boolean } => {
    if (activeDrag?.kind === "stage") {
      if (activeDrag.stageId === stage.stage_id) {
        return {
          tStart: activeDrag.tStart,
          tEnd: activeDrag.tEnd,
          dragging: true,
          magneticLinked: Boolean(activeDrag.neighbor?.linked),
        };
      }
      if (activeDrag.neighbor && activeDrag.neighbor.stageId === stage.stage_id) {
        return {
          tStart: activeDrag.neighbor.tStart,
          tEnd: activeDrag.neighbor.tEnd,
          dragging: false,
          magneticLinked: activeDrag.neighbor.linked,
        };
      }
    }
    return { tStart: stage.t_start, tEnd: stage.t_end, dragging: false, magneticLinked: false };
  };

  const getSegmentRenderBounds = (
    seg: QcSegment,
  ): { tStart: number; tEnd: number; dragging: boolean } => {
    if (activeDrag?.kind === "segment" && activeDrag.segId === seg.seg_id) {
      return { tStart: activeDrag.tStart, tEnd: activeDrag.tEnd, dragging: true };
    }
    return { tStart: seg.t_start, tEnd: seg.t_end, dragging: false };
  };

  const activeSnapTime: number | null =
    activeDrag?.kind === "stage" &&
    activeDrag.snappedToQcTime !== null &&
    activeDrag.snappedToQcTime !== undefined
      ? activeDrag.snappedToQcTime
      : activeDrag?.kind === "segment" &&
          activeDrag.snappedTime !== null &&
          activeDrag.snappedTime !== undefined
        ? activeDrag.snappedTime
        : null;

  const isPresetZoom = ZOOM_PRESETS.some((z) => Math.abs(zoom - z) < 0.01);
  const playheadLeft = `${pct(currentTime, durationS)}%`;

  return (
    <div
      className={`four-track-timeline${compact ? " compact" : ""}`}
      data-testid="four-track-timeline"
      data-compact={compact ? "true" : "false"}
      data-snap-enabled={snapEnabled ? "true" : "false"}
    >
      <div className="timeline-toolbar">
        <div className="timeline-title">{t("timeline.title")}</div>
        <div className="timeline-actions">
          {!readOnly && (
            <>
              {onAlignStagesToQc && (
                <button
                  type="button"
                  className="mini-btn"
                  disabled={macroStages.length === 0 || qcSegments.length === 0}
                  onClick={onAlignStagesToQc}
                  data-testid="align-stages-qc-btn"
                  title={t("timeline.alignToQcTitle")}
                >
                  <Magnet size={13} /> {t("timeline.alignToQc")}
                </button>
              )}
              {onInferSubtasks && (
                <button
                  type="button"
                  className="mini-btn accent"
                  disabled={inferringSubtasks}
                  onClick={onInferSubtasks}
                  data-testid="infer-subtasks-btn"
                  title={t("timeline.inferSubtasksTitle")}
                >
                  {inferringSubtasks ? (
                    <Loader2 size={13} className="studio-spin" />
                  ) : (
                    <Sparkles size={13} />
                  )}{" "}
                  {inferringSubtasks ? t("timeline.inferring") : t("timeline.inferSubtasks")}
                </button>
              )}
              {onSplitMacroStage && (
                <button
                  type="button"
                  className="mini-btn"
                  disabled={!canSplitStage}
                  onClick={onSplitMacroStage}
                  data-testid="split-macro-stage-btn"
                  title={t("timeline.splitStageTitle")}
                >
                  <Scissors size={13} /> {t("timeline.splitStage")}
                </button>
              )}
              <button
                type="button"
                className="mini-btn"
                onClick={onAddMacroStage}
                data-testid="add-macro-stage-btn"
              >
                <Plus size={13} /> {t("timeline.addSubtaskStage")}
              </button>
              <button
                type="button"
                className="mini-btn primary"
                onClick={onAddSegment}
                data-testid="add-qc-segment-btn"
              >
                <Plus size={13} /> {t("timeline.addQcSegment")}
              </button>
            </>
          )}
          <div className="zoom-group" title={t("studio.zoomTip")}>
            {ZOOM_PRESETS.map((z) => (
              <button
                key={z}
                type="button"
                className={`zoom-btn tabular-nums ${Math.abs(zoom - z) < 0.01 ? "active" : ""}`}
                onClick={() => applyPlayheadAnchoredZoom(z)}
              >
                {z}x
              </button>
            ))}
            {!isPresetZoom && (
              <span className="zoom-readout-pill tabular-nums" data-testid="timeline-zoom-readout">
                {zoom.toFixed(2)}x
              </span>
            )}
          </div>
        </div>
      </div>

      <div className="timeline-scroll" ref={scrollRef} data-testid="timeline-scroll">
        <div
          className="timeline-canvas"
          data-testid="timeline-canvas"
          style={{ width: `${zoom * 100}%` }}
        >
          {/* Time Ruler */}
          <div className="track-row ruler-row">
            <div className="track-label ruler-label">{t("timeline.rulerLabel")}</div>
            <div
              ref={rulerRef}
              className="timeline-ruler"
              onClick={(e) => seekFromClientX(e.clientX, e.currentTarget)}
              role="slider"
              aria-label={t("timeline.rulerAria")}
              aria-valuemin={0}
              aria-valuemax={durationS}
              aria-valuenow={currentTime}
            >
              {ticks.map((tick, idx) => (
                <span
                  key={idx}
                  className="ruler-tick tabular-nums"
                  style={{ left: `${pct(tick, durationS)}%` }}
                >
                  {tick.toFixed(1)}s
                </span>
              ))}
            </div>
          </div>

          {/* 4px Watched / Unwatched Heatmap Bar beneath Time Ruler */}
          <div className="track-row heatmap-row" data-testid="watch-heatmap-row">
            <div className="track-label heatmap-label">{t("timeline.watchMapLabel")}</div>
            <div
              className="watch-heatmap-track"
              data-testid="watch-heatmap-bar"
              onClick={(e) => seekFromClientX(e.clientX, e.currentTarget)}
              title={t("timeline.watchMapTitle")}
            >
              <div className="watch-heatmap-bg" />
              {watchedIntervals.map(([a, b], idx) => {
                const left = pct(a, durationS);
                const width = Math.max(0.4, pct(b, durationS) - left);
                return (
                  <div
                    key={`w-${idx}`}
                    className="heatmap-watched"
                    data-testid={`heatmap-watched-${idx}`}
                    style={{ left: `${left}%`, width: `${width}%` }}
                  />
                );
              })}
              {unwatchedGaps.map(([g0, g1], idx) => {
                const left = pct(g0, durationS);
                const width = Math.max(0.8, pct(g1, durationS) - left);
                return (
                  <button
                    key={`u-${idx}`}
                    type="button"
                    className="heatmap-unwatched-gap"
                    data-testid={`heatmap-unwatched-gap-${idx}`}
                    style={{ left: `${left}%`, width: `${width}%` }}
                    aria-label={t("timeline.unwatchedGapAria", {
                      start: g0.toFixed(1),
                      end: g1.toFixed(1),
                    })}
                    title={t("timeline.unwatchedGapTitle", {
                      start: g0.toFixed(1),
                      end: g1.toFixed(1),
                    })}
                    onClick={(e) => {
                      e.stopPropagation();
                      onSeek(g0);
                    }}
                  />
                );
              })}
            </div>
          </div>

          {/* Track 1: Subtask Stages (Editable + Edge Resize + Magnetic Ripple + QC Snap) */}
          <div className="track-row" data-testid="track-macro-stages">
            <div className="track-label">
              <span>{t("timeline.track1Label")}</span>
              <span className="track-badge editable">{t("timeline.badgeEditable")}</span>
            </div>
            <div
              ref={macroLaneRef}
              className="track-lane"
              data-testid="macro-track-lane"
              onClick={(e) => {
                if (suppressClickRef.current) return;
                if (e.target === e.currentTarget) {
                  seekFromClientX(e.clientX, e.currentTarget);
                }
              }}
            >
              {macroStages.map((stage) => {
                const bounds = getStageRenderBounds(stage);
                const left = pct(bounds.tStart, durationS);
                const width = Math.max(1.2, pct(bounds.tEnd, durationS) - left);
                const isSelected = selectedStageId === stage.stage_id;
                return (
                  <div
                    key={stage.stage_id}
                    role="button"
                    tabIndex={0}
                    data-testid={`macro-stage-${stage.stage_id}`}
                    className={`track-block macro-block nle-resizable ${isSelected ? "selected" : ""} ${bounds.dragging ? "dragging" : ""} ${bounds.magneticLinked ? "magnetic-linked" : ""}`}
                    style={{
                      left: `${left}%`,
                      width: `${width}%`,
                      backgroundColor: rgbCss(stage.color_rgb),
                    }}
                    title={
                      stage.instruction
                        ? `${stage.label} (${bounds.tStart.toFixed(2)}s - ${bounds.tEnd.toFixed(2)}s)${t("timeline.stageInstructionSuffix", { instruction: stage.instruction })}`
                        : `${stage.label} (${bounds.tStart.toFixed(2)}s - ${bounds.tEnd.toFixed(2)}s)`
                    }
                    onClick={(e) => {
                      if (suppressClickRef.current) return;
                      e.stopPropagation();
                      onSelectStage(stage.stage_id);
                      onSelectSegment(null);
                      onSeek(stage.t_start);
                    }}
                  >
                    {!readOnly && (onUpdateStageSpan || onUpdateStageSpans) && (
                      <div
                        className="resize-handle resize-handle-left"
                        data-testid={`resize-left-stage-${stage.stage_id}`}
                        title={t("timeline.resizeStageLeftTitle")}
                        onPointerDown={(e) => startStageEdgeDrag(e, stage, "left")}
                        onClick={(e) => e.stopPropagation()}
                      />
                    )}
                    <span className="block-text">{stage.label}</span>
                    {!readOnly && onUpdateStageSpan && isSelected && (
                      <span className="nudge-controls">
                        <button
                          type="button"
                          className="nudge-btn"
                          title={t("timeline.nudgeStartMinus", { delta: "0.2" })}
                          onClick={(e) => {
                            e.stopPropagation();
                            onUpdateStageSpan(
                              stage.stage_id,
                              Math.max(0, roundTime(stage.t_start - 0.2)),
                              stage.t_end,
                            );
                          }}
                        >
                          ◀
                        </button>
                        <button
                          type="button"
                          className="nudge-btn"
                          title={t("timeline.nudgeEndPlus", { delta: "0.2" })}
                          onClick={(e) => {
                            e.stopPropagation();
                            onUpdateStageSpan(
                              stage.stage_id,
                              stage.t_start,
                              Math.min(durationS, roundTime(stage.t_end + 0.2)),
                            );
                          }}
                        >
                          ▶
                        </button>
                      </span>
                    )}
                    {!readOnly && (onUpdateStageSpan || onUpdateStageSpans) && (
                      <div
                        className="resize-handle resize-handle-right"
                        data-testid={`resize-right-stage-${stage.stage_id}`}
                        title={t("timeline.resizeStageRightTitle")}
                        onPointerDown={(e) => startStageEdgeDrag(e, stage, "right")}
                        onClick={(e) => e.stopPropagation()}
                      />
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Track 2: Arm Timelines (Strictly READ-ONLY, zero buttons) */}
          <div className="track-row readonly-row" data-testid="track-arm-timelines">
            <div className="track-label">
              <span>{t("timeline.track2Label")}</span>
              <span className="track-badge readonly" data-testid="arm-timelines-readonly">
                <Lock size={11} /> {t("timeline.badgeReadOnly")}
              </span>
            </div>
            <div className="track-lane dual-sublane" aria-readonly="true">
              <div className="sublane" data-testid="arm-lane-left">
                <span className="sublane-tag">L</span>
                {armTimelines.left.map((span, i) => {
                  const left = pct(span.t_start, durationS);
                  const width = Math.max(0.8, pct(span.t_end, durationS) - left);
                  return (
                    <div
                      key={`L-${i}`}
                      className="arm-span-block"
                      style={{
                        left: `${left}%`,
                        width: `${width}%`,
                        backgroundColor: ARM_STATE_COLORS[span.state],
                      }}
                      title={t("timeline.armLeftTitle", {
                        state: span.state,
                        start: span.t_start.toFixed(2),
                        end: span.t_end.toFixed(2),
                        desc: span.desc,
                      })}
                    >
                      <span>{span.state}</span>
                    </div>
                  );
                })}
              </div>
              <div className="sublane" data-testid="arm-lane-right">
                <span className="sublane-tag">R</span>
                {armTimelines.right.map((span, i) => {
                  const left = pct(span.t_start, durationS);
                  const width = Math.max(0.8, pct(span.t_end, durationS) - left);
                  return (
                    <div
                      key={`R-${i}`}
                      className="arm-span-block"
                      style={{
                        left: `${left}%`,
                        width: `${width}%`,
                        backgroundColor: ARM_STATE_COLORS[span.state],
                      }}
                      title={t("timeline.armRightTitle", {
                        state: span.state,
                        start: span.t_start.toFixed(2),
                        end: span.t_end.toFixed(2),
                        desc: span.desc,
                      })}
                    >
                      <span>{span.state}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Track 3: QC Segments (Editable + Edge Resize + Magnetic Snap + Drag-to-Create Brush) */}
          <div className="track-row" data-testid="track-qc-segments">
            <div className="track-label">
              <span>{t("timeline.track3Label")}</span>
              <span className="track-badge editable">{t("timeline.badgeEditable")}</span>
            </div>
            <div
              ref={qcLaneRef}
              className={`track-lane ${!readOnly && onCreateSegmentSpan ? "qc-brushable" : ""}`}
              data-testid="qc-track-lane"
              onPointerDown={startQcLaneBrush}
              onClick={(e) => {
                if (suppressClickRef.current) return;
                if (e.target === e.currentTarget) {
                  seekFromClientX(e.clientX, e.currentTarget);
                }
              }}
            >
              {activeDrag?.kind === "brush" && activeDrag.tEnd - activeDrag.tStart >= 0.05 && (
                <div
                  className="qc-brush-preview tabular-nums"
                  data-testid="qc-brush-preview"
                  style={{
                    left: `${pct(activeDrag.tStart, durationS)}%`,
                    width: `${Math.max(0.6, pct(activeDrag.tEnd, durationS) - pct(activeDrag.tStart, durationS))}%`,
                  }}
                >
                  {activeDrag.tStart.toFixed(2)}s – {activeDrag.tEnd.toFixed(2)}s
                </div>
              )}

              {qcSegments.map((seg) => {
                const bounds = getSegmentRenderBounds(seg);
                const left = pct(bounds.tStart, durationS);
                const width = Math.max(1.4, pct(bounds.tEnd, durationS) - left);
                const isSelected = selectedSegmentId === seg.seg_id;
                const maskCode =
                  seg.splice_policy === "mask_loss"
                    ? seg.mask_side === "left"
                      ? "L"
                      : seg.mask_side === "right"
                        ? "R"
                        : "LR"
                    : null;

                return (
                  <div
                    key={seg.seg_id}
                    role="button"
                    tabIndex={0}
                    data-testid={`qc-seg-${seg.seg_id}`}
                    className={`track-block qc-block nle-resizable ${POLICY_CLASS[seg.splice_policy]} ${isSelected ? "selected" : ""} ${bounds.dragging ? "dragging" : ""}`}
                    style={{ left: `${left}%`, width: `${width}%` }}
                    title={`${seg.name || seg.label_id} (${seg.splice_policy}) ${bounds.tStart.toFixed(2)}s-${bounds.tEnd.toFixed(2)}s`}
                    onClick={(e) => {
                      if (suppressClickRef.current) return;
                      e.stopPropagation();
                      onSelectSegment(seg.seg_id);
                      onSelectStage(null);
                      onSeek(seg.t_start);
                    }}
                  >
                    {!readOnly && onUpdateSegmentSpan && (
                      <div
                        className="resize-handle resize-handle-left"
                        data-testid={`resize-left-seg-${seg.seg_id}`}
                        title={t("timeline.resizeSegLeftTitle")}
                        onPointerDown={(e) => startSegmentEdgeDrag(e, seg, "left")}
                        onClick={(e) => e.stopPropagation()}
                      />
                    )}
                    <span className="block-text">{seg.name || seg.label_id}</span>
                    {maskCode && <span className="mask-chip">{maskCode}</span>}
                    {seg.guardrail.downgraded && (
                      <span
                        className="guardrail-chip downgraded"
                        data-testid={`guardrail-downgraded-${seg.seg_id}`}
                        title={
                          seg.guardrail.downgrade_reason ?? t("timeline.downgradedDefaultTitle")
                        }
                      >
                        <ShieldAlert size={11} /> {t("timeline.downgradedBadge")}
                      </span>
                    )}
                    {!seg.guardrail.downgraded && seg.guardrail.gate_type === "splice_gate" && (
                      <span
                        className="guardrail-chip passed"
                        title={t("timeline.guardrailPassedTitle")}
                      >
                        <ShieldCheck size={11} />
                      </span>
                    )}
                    {seg.guardrail.guardrail_override && (
                      <span
                        className="guardrail-chip override"
                        title={
                          seg.guardrail.override_reason ?? t("timeline.guardrailOverrideTitle")
                        }
                      >
                        <AlertTriangle size={11} />
                      </span>
                    )}
                    {!readOnly && onUpdateSegmentSpan && isSelected && (
                      <span className="nudge-controls">
                        <button
                          type="button"
                          className="nudge-btn"
                          title={t("timeline.nudgeStartMinus", { delta: "0.1" })}
                          onClick={(e) => {
                            e.stopPropagation();
                            onUpdateSegmentSpan(
                              seg.seg_id,
                              Math.max(0, roundTime(seg.t_start - 0.1)),
                              seg.t_end,
                            );
                          }}
                        >
                          ◀
                        </button>
                        <button
                          type="button"
                          className="nudge-btn"
                          title={t("timeline.nudgeEndPlus", { delta: "0.1" })}
                          onClick={(e) => {
                            e.stopPropagation();
                            onUpdateSegmentSpan(
                              seg.seg_id,
                              seg.t_start,
                              Math.min(durationS, roundTime(seg.t_end + 0.1)),
                            );
                          }}
                        >
                          ▶
                        </button>
                      </span>
                    )}
                    {!readOnly && onUpdateSegmentSpan && (
                      <div
                        className="resize-handle resize-handle-right"
                        data-testid={`resize-right-seg-${seg.seg_id}`}
                        title={t("timeline.resizeSegRightTitle")}
                        onPointerDown={(e) => startSegmentEdgeDrag(e, seg, "right")}
                        onClick={(e) => e.stopPropagation()}
                      />
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Track 4: Lead-only Ghost VLM Diff Track */}
          {showDiffTrack && (
            <div className="track-row diff-row" data-testid="track-ghost-vlm">
              <div className="track-label">
                <span>{t("timeline.track4Label")}</span>
                <span className="track-badge readonly">
                  <Lock size={11} /> {t("timeline.badgeGhostDiff")}
                </span>
              </div>
              <div className="track-lane">
                {rawVlmSegments.map((seg) => {
                  const left = pct(seg.t_start, durationS);
                  const width = Math.max(1.2, pct(seg.t_end, durationS) - left);
                  return (
                    <div
                      key={`ghost-${seg.seg_id}`}
                      className={`track-block ghost-block ${POLICY_CLASS[seg.splice_policy]}`}
                      style={{ left: `${left}%`, width: `${width}%` }}
                      title={t("timeline.vlmGhostTitle", {
                        name: seg.name || seg.label_id,
                        policy: seg.splice_policy,
                        start: seg.t_start.toFixed(2),
                        end: seg.t_end.toFixed(2),
                      })}
                      onClick={() => onSeek(seg.t_start)}
                    >
                      <span className="block-text">
                        {t("timeline.vlmGhostText", { name: seg.name || seg.label_id })}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Cross-track Magnetic Snap Guide Indicator */}
          {activeSnapTime !== null && (
            <div
              className="timeline-snap-guide-overlay"
              data-testid={activeDrag?.kind === "stage" ? "qc-snap-guide" : undefined}
            >
              <div
                className="timeline-snap-guide"
                data-testid="timeline-snap-guide"
                data-snap-time={activeSnapTime.toFixed(2)}
                style={{ left: `${pct(activeSnapTime, durationS)}%` }}
              >
                <span className="timeline-snap-readout tabular-nums">
                  {activeSnapTime.toFixed(2)}s
                </span>
              </div>
            </div>
          )}

          {/* Synchronized Playhead aligned with .track-lane */}
          <div className="timeline-playhead-overlay">
            <div className="timeline-playhead" style={{ left: playheadLeft }} />
          </div>
        </div>
      </div>
    </div>
  );
};
