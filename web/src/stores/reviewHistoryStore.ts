import { useCallback, useState } from "react";
import type {
  HumanReviewArtifact,
  MacroStage,
  QcSegment,
  SuccessVerdict,
  WatchCoverage,
} from "../api/types";

export const MIN_WATCH_COVERAGE = 0.95;
const MAX_HISTORY_STEPS = 100;

export interface EditableReviewState {
  qc_segments: QcSegment[];
  success: SuccessVerdict;
  macro_stages: MacroStage[];
}

export interface ReviewHistoryStack {
  past: EditableReviewState[];
  present: EditableReviewState;
  future: EditableReviewState[];
  dirty: boolean;
}

export function cloneEditableState(state: EditableReviewState): EditableReviewState {
  return JSON.parse(JSON.stringify(state)) as EditableReviewState;
}

export function createInitialHistory(initial: EditableReviewState): ReviewHistoryStack {
  return {
    past: [],
    present: cloneEditableState(initial),
    future: [],
    dirty: false,
  };
}

export function pushHistoryEdit(
  stack: ReviewHistoryStack,
  updater: (prev: EditableReviewState) => EditableReviewState,
): ReviewHistoryStack {
  const nextPresent = cloneEditableState(updater(cloneEditableState(stack.present)));
  const nextPast = [...stack.past, cloneEditableState(stack.present)];
  if (nextPast.length > MAX_HISTORY_STEPS) {
    nextPast.shift();
  }
  return {
    past: nextPast,
    present: nextPresent,
    future: [],
    dirty: true,
  };
}

export function undoHistory(stack: ReviewHistoryStack): ReviewHistoryStack {
  if (stack.past.length === 0) return stack;
  const previous = stack.past[stack.past.length - 1];
  return {
    past: stack.past.slice(0, -1),
    present: cloneEditableState(previous),
    future: [cloneEditableState(stack.present), ...stack.future],
    dirty: true,
  };
}

export function redoHistory(stack: ReviewHistoryStack): ReviewHistoryStack {
  if (stack.future.length === 0) return stack;
  const next = stack.future[0];
  return {
    past: [...stack.past, cloneEditableState(stack.present)],
    present: cloneEditableState(next),
    future: stack.future.slice(1),
    dirty: true,
  };
}

export function restoreHistorySnapshot(
  stack: ReviewHistoryStack,
  rev: Pick<HumanReviewArtifact, "qc_segments" | "success" | "macro_stages">,
): ReviewHistoryStack {
  return pushHistoryEdit(stack, () => ({
    qc_segments: rev.qc_segments,
    success: rev.success,
    macro_stages: rev.macro_stages,
  }));
}

export function useReviewHistory(initial: EditableReviewState) {
  const [stack, setStack] = useState<ReviewHistoryStack>(() => createInitialHistory(initial));

  const init = useCallback((nextInitial: EditableReviewState) => {
    setStack(createInitialHistory(nextInitial));
  }, []);

  const applyEdit = useCallback((updater: (prev: EditableReviewState) => EditableReviewState) => {
    setStack((prev) => pushHistoryEdit(prev, updater));
  }, []);

  const undo = useCallback(() => {
    setStack((prev) => undoHistory(prev));
  }, []);

  const redo = useCallback(() => {
    setStack((prev) => redoHistory(prev));
  }, []);

  const restoreSnapshot = useCallback(
    (rev: Pick<HumanReviewArtifact, "qc_segments" | "success" | "macro_stages">) => {
      setStack((prev) => restoreHistorySnapshot(prev, rev));
    },
    [],
  );

  const markSaved = useCallback(() => {
    setStack((prev) => ({ ...prev, dirty: false }));
  }, []);

  return {
    state: stack.present,
    canUndo: stack.past.length > 0,
    canRedo: stack.future.length > 0,
    undoCount: stack.past.length,
    redoCount: stack.future.length,
    dirty: stack.dirty,
    init,
    applyEdit,
    undo,
    redo,
    restoreSnapshot,
    markSaved,
  };
}

export function mergeIntervals(intervals: Array<[number, number]>): Array<[number, number]> {
  const valid = intervals
    .filter(([a, b]) => Number.isFinite(a) && Number.isFinite(b) && b > a)
    .map(([a, b]) => [Math.max(0, a), b] as [number, number])
    .sort((x, y) => x[0] - y[0]);

  if (valid.length === 0) return [];
  const merged: Array<[number, number]> = [[valid[0][0], valid[0][1]]];
  for (let i = 1; i < valid.length; i += 1) {
    const [start, end] = valid[i];
    const last = merged[merged.length - 1];
    if (start <= last[1] + 0.05) {
      last[1] = Math.max(last[1], end);
    } else {
      merged.push([start, end]);
    }
  }
  return merged;
}

export function computeUnwatchedIntervals(
  watchedIntervals: Array<[number, number]>,
  durationS: number,
  minGapS = 0.05,
): Array<[number, number]> {
  const dur = Math.max(0, durationS);
  if (dur <= minGapS) return [];
  const merged = mergeIntervals(
    watchedIntervals.map(([a, b]) => [Math.max(0, a), Math.min(dur, b)]),
  );
  if (merged.length === 0) {
    return [[0, Number(dur.toFixed(3))]];
  }
  const gaps: Array<[number, number]> = [];
  let cursor = 0;
  for (const [start, end] of merged) {
    if (start - cursor >= minGapS) {
      gaps.push([Number(cursor.toFixed(3)), Number(start.toFixed(3))]);
    }
    cursor = Math.max(cursor, end);
  }
  if (dur - cursor >= minGapS) {
    gaps.push([Number(cursor.toFixed(3)), Number(dur.toFixed(3))]);
  }
  return gaps;
}

export class WatchCoverageTracker {
  private intervals: Array<[number, number]> = [];
  readonly duration_s: number;
  confirmed = false;

  constructor(durationS: number, initialIntervals: Array<[number, number]> = []) {
    this.duration_s = Math.max(0.01, durationS);
    this.intervals = mergeIntervals(
      initialIntervals.map(([a, b]) => [Math.max(0, a), Math.min(this.duration_s, b)]),
    );
  }

  recordInterval(tStart: number, tEnd: number, minCoverage: number = MIN_WATCH_COVERAGE): WatchCoverage {
    const a = Math.max(0, Math.min(tStart, tEnd));
    const b = Math.min(this.duration_s, Math.max(tStart, tEnd));
    if (b > a) {
      this.intervals = mergeIntervals([...this.intervals, [a, b]]);
    }
    return this.snapshot(minCoverage);
  }

  unwatchedIntervals(minGapS = 0.05): Array<[number, number]> {
    return computeUnwatchedIntervals(this.intervals, this.duration_s, minGapS);
  }

  snapshot(minCoverage: number = MIN_WATCH_COVERAGE): WatchCoverage {
    const watchedS = this.intervals.reduce((acc, [a, b]) => acc + (b - a), 0);
    const ratio = Math.min(1, Math.max(0, watchedS / this.duration_s));
    return {
      watched_intervals: this.intervals.map(([a, b]) => [
        Number(a.toFixed(3)),
        Number(b.toFixed(3)),
      ]),
      watched_s: Number(watchedS.toFixed(3)),
      duration_s: Number(this.duration_s.toFixed(3)),
      coverage_ratio: Number(ratio.toFixed(4)),
      confirmed: this.confirmed || ratio >= minCoverage - 1e-6,
    };
  }
}

export function canReviewerSubmit(
  coverage: WatchCoverage,
  minCoverage: number = MIN_WATCH_COVERAGE,
): boolean {
  return coverage.coverage_ratio >= minCoverage - 1e-6;
}
