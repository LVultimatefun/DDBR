import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { makeQcSegment, makeReviewArtifact } from "../test/fixtures";
import { DiffOverlay } from "./DiffOverlay";

describe("DiffOverlay", () => {
  it("is hidden for Reviewer and renders diff metrics, comparison rows, and snapshot restore for Lead", () => {
    const rawSeg = makeQcSegment({ seg_id: "raw_1", splice_policy: "cut_clean", mask_side: null });
    const rev = makeReviewArtifact({ rev_id: "rev_42" });
    const onRestore = vi.fn();
    const onToggleDiff = vi.fn();

    const { rerender } = render(
      <DiffOverlay
        isLead={false}
        diffEnabled={false}
        onToggleDiff={onToggleDiff}
        reviewDiff={{
          fp_deleted: 2,
          fn_added: 1,
          policy_flips: 1,
          label_changes: 0,
          boundary_nudges: 1,
          macro_stage_edits: 1,
          success_flipped: true,
          total_edits: 6,
        }}
        rawVlmSegments={[rawSeg]}
        currentSegments={[]}
        rawVlmSuccess={null}
        currentSuccess={rev.success}
        rawVlmMacroStages={[]}
        currentMacroStages={[]}
        reviews={[rev]}
        onRestoreSnapshot={onRestore}
      />,
    );

    expect(screen.queryByTestId("diff-overlay")).toBeNull();

    rerender(
      <DiffOverlay
        isLead={true}
        diffEnabled={true}
        onToggleDiff={onToggleDiff}
        reviewDiff={{
          fp_deleted: 2,
          fn_added: 1,
          policy_flips: 1,
          label_changes: 0,
          boundary_nudges: 1,
          macro_stage_edits: 1,
          success_flipped: true,
          total_edits: 6,
        }}
        rawVlmSegments={[rawSeg]}
        currentSegments={[]}
        rawVlmSuccess={null}
        currentSuccess={rev.success}
        rawVlmMacroStages={[]}
        currentMacroStages={[]}
        reviews={[rev]}
        onRestoreSnapshot={onRestore}
      />,
    );

    const metrics = screen.getByTestId("diff-metrics");
    expect(metrics).toHaveTextContent("fp_deleted");
    expect(metrics).toHaveTextContent("fn_added");
    expect(metrics).toHaveTextContent("policy_flips");
    expect(metrics).toHaveTextContent("success_flipped");

    fireEvent.click(screen.getByTestId("restore-rev-rev_42"));
    expect(onRestore).toHaveBeenCalledWith(rev);
  });
});
