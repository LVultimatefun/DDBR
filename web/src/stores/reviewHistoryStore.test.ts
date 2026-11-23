import { describe, expect, it } from "vitest";
import { normalizeSegmentTriplet, validateSegmentTriplet } from "../constants/labels";
import { makeQcSegment, makeReviewArtifact } from "../test/fixtures";
import {
  canReviewerSubmit,
  createInitialHistory,
  pushHistoryEdit,
  redoHistory,
  restoreHistorySnapshot,
  undoHistory,
  WatchCoverageTracker,
} from "./reviewHistoryStore";

describe("reviewHistoryStore", () => {
  it("supports multi-step undo/redo and snapshot restore across 3 editable layers", () => {
    const initial = {
      qc_segments: [makeQcSegment()],
      success: {
        passed: true,
        verdict: "success" as const,
        confidence: 0.9,
        criteria: [],
        control_check: null,
        summary: "init",
      },
      macro_stages: [
        {
          stage_id: "M1",
          t_start: 0,
          t_end: 5,
          label: "Stage 1",
          label_en: "stage_1",
          summary: "",
          color_rgb: [59, 130, 246] as [number, number, number],
        },
      ],
    };

    let stack = createInitialHistory(initial);
    expect(stack.dirty).toBe(false);

    // Edit 1: update success verdict
    stack = pushHistoryEdit(stack, (prev) => ({
      ...prev,
      success: { ...prev.success, passed: false, verdict: "failure", summary: "edited" },
    }));
    expect(stack.present.success.verdict).toBe("failure");
    expect(stack.past).toHaveLength(1);
    expect(stack.dirty).toBe(true);

    // Undo restores initial
    stack = undoHistory(stack);
    expect(stack.present.success.verdict).toBe("success");
    expect(stack.future).toHaveLength(1);

    // Redo reapplies Edit 1
    stack = redoHistory(stack);
    expect(stack.present.success.verdict).toBe("failure");

    // Snapshot restore pushes current to past and sets snapshot state
    const rev = makeReviewArtifact({
      qc_segments: [],
      macro_stages: [],
      success: {
        passed: false,
        verdict: "failure",
        confidence: 1,
        criteria: [],
        control_check: null,
        summary: "restored_snapshot",
      },
    });
    stack = restoreHistorySnapshot(stack, rev);
    expect(stack.present.qc_segments).toHaveLength(0);
    expect(stack.present.success.summary).toBe("restored_snapshot");
    expect(stack.past).toHaveLength(2);
  });

  it("enforces QcSegment triplet invariants and rejects abolished legacy labels", () => {
    const seg = makeQcSegment({
      splice_policy: "mask_loss",
      mask_side: "left",
      training_role: "recovery",
    });

    // Switching to keep_positive clears mask_side to null
    const keepPos = normalizeSegmentTriplet(seg, { splice_policy: "keep_positive" });
    expect(keepPos.mask_side).toBeNull();
    expect(validateSegmentTriplet(keepPos)).toBeNull();

    // Switching to cut_clean clears mask_side and coerces training_role to badcase
    const cutClean = normalizeSegmentTriplet(seg, { splice_policy: "cut_clean" });
    expect(cutClean.mask_side).toBeNull();
    expect(cutClean.training_role).toBe("badcase");
    expect(validateSegmentTriplet(cutClean)).toBeNull();

    // Switching back to mask_loss enforces left|right|both
    const maskLoss = normalizeSegmentTriplet(cutClean, { splice_policy: "mask_loss" });
    expect(["left", "right", "both"]).toContain(maskLoss.mask_side);
    expect(validateSegmentTriplet(maskLoss)).toBeNull();

    // Rejects abolished legacy labels
    expect(() => normalizeSegmentTriplet(seg, { label_id: "long_pause" })).toThrow();
    expect(() => normalizeSegmentTriplet(seg, { label_id: "back_and_forth" })).toThrow();

    // Clamps negative t_start to >= 0 before checking t_end <= t_start
    const clampedNeg = normalizeSegmentTriplet(seg, { t_start: -2.0, t_end: 2.0 });
    expect(clampedNeg.t_start).toBe(0);
    expect(clampedNeg.t_end).toBe(2.0);
    expect(validateSegmentTriplet(clampedNeg)).toBeNull();
  });

  it("merges WatchCoverageTracker intervals and enforces 95% threshold", () => {
    const tracker = new WatchCoverageTracker(10.0);
    tracker.recordInterval(0, 4.0);
    tracker.recordInterval(3.98, 7.0); // overlaps within 0.05s tolerance
    let snap = tracker.snapshot();
    expect(snap.watched_intervals).toEqual([[0, 7]]);
    expect(snap.coverage_ratio).toBeCloseTo(0.7, 3);
    expect(canReviewerSubmit(snap)).toBe(false);

    tracker.recordInterval(7.0, 9.6);
    snap = tracker.snapshot();
    expect(snap.coverage_ratio).toBeGreaterThanOrEqual(0.95);
    expect(canReviewerSubmit(snap)).toBe(true);
  });
});
