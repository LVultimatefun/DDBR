import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { buildReviewWritePayload, frameUrl, previewVideoUrl } from "../api/client";
import { AuthProvider } from "../stores/authStore";
import { makeQcSegment, makeResolvedEpisode } from "../test/fixtures";
import { SegmentInspector } from "./SegmentInspector";
import { StudioPlayer } from "./StudioPlayer";
import { SuccessPanel } from "./SuccessPanel";

describe("StudioPlayer, SegmentInspector, SuccessPanel & API payload contract", () => {
  it("renders StudioPlayer with read-only 2D Grounding canvas and builds token/t_s media URLs", () => {
    const ep = makeResolvedEpisode();
    render(
      <AuthProvider initialAuth={{ token: "tok123", username: "rev1", role: "reviewer" }}>
        <StudioPlayer
          episodeId={ep.episode_id}
          cameras={ep.meta.cameras}
          durationS={ep.meta.duration_s}
          fps={ep.meta.fps}
          grounding={ep.grounding}
          currentTime={0}
          onTimeUpdate={vi.fn()}
        />
      </AuthProvider>,
    );

    expect(screen.getByTestId("studio-player")).toBeInTheDocument();
    expect(screen.getByTestId("grounding-canvas")).toBeInTheDocument();
    expect(previewVideoUrl("demo_ds/ep_001", "head_rgbd", "tok123")).toBe(
      "/api/episodes/demo_ds/ep_001/preview?camera=head_rgbd&token=tok123",
    );
    expect(frameUrl("demo_ds/ep_001", "head_rgbd", 1.25, "tok123")).toBe(
      "/api/episodes/demo_ds/ep_001/frame?camera=head_rgbd&t_s=1.2500&token=tok123",
    );
  });

  it("normalizes SegmentInspector triplet edits and triggers SuccessPanel seek", () => {
    const seg = makeQcSegment({ splice_policy: "mask_loss", mask_side: "left", training_role: "recovery" });
    const onUpdateSegment = vi.fn();
    const onSeek = vi.fn();
    const onSuccessChange = vi.fn();

    render(
      <>
        <SegmentInspector
          episodeId="demo_ds/ep_001"
          taskId="pick_and_place"
          durationS={10.0}
          currentTime={2.0}
          isLead={false}
          selectedSegment={seg}
          selectedStage={null}
          onUpdateSegment={onUpdateSegment}
          onDeleteSegment={vi.fn()}
          onUpdateStage={vi.fn()}
          onDeleteStage={vi.fn()}
        />
        <SuccessPanel
          success={{
            passed: true,
            verdict: "success",
            confidence: 0.9,
            criteria: [
              { criterion: "Object placed stably", met: true, satisfied_at_s: 8.5, evidence: "ok" },
            ],
            control_check: null,
            summary: "good",
          }}
          onChange={onSuccessChange}
          onSeek={onSeek}
        />
      </>,
    );

    // Changing policy to keep_positive automatically clears mask_side to null
    fireEvent.change(screen.getByLabelText("Splice Policy"), { target: { value: "keep_positive" } });
    expect(onUpdateSegment).toHaveBeenCalledWith(
      expect.objectContaining({ splice_policy: "keep_positive", mask_side: null }),
    );

    // Clicking satisfied_at_s seek pill jumps to 8.5s
    fireEvent.click(screen.getByText(/8\.50s/));
    expect(onSeek).toHaveBeenCalledWith(8.5);
  });

  it("buildReviewWritePayload strictly omits author, grounding, arm_timelines, and raw_vlm_segments", () => {
    const payload = buildReviewWritePayload({
      baseRunId: "run_1",
      action: "submit_review",
      qcSegments: [makeQcSegment()],
      success: { passed: true, verdict: "success", confidence: 1, criteria: [], control_check: null, summary: "" },
      macroStages: [],
      watchCoverage: {
        watched_intervals: [[0, 10]],
        watched_s: 10,
        duration_s: 10,
        coverage_ratio: 1,
        confirmed: true,
      },
      notes: "ok",
    });

    expect(Object.keys(payload).sort()).toEqual([
      "action",
      "base_run_id",
      "macro_stages",
      "notes",
      "qc_segments",
      "success",
      "watch_coverage",
    ]);
  });
});
