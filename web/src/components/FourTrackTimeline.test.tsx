import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { makeQcSegment, makeResolvedEpisode } from "../test/fixtures";
import { FourTrackTimeline } from "./FourTrackTimeline";

describe("FourTrackTimeline", () => {
  it("renders Track 2 arm_timelines strictly read-only while supporting macro/qc editing and guardrail badges", () => {
    const ep = makeResolvedEpisode({
      qc_segments: [
        makeQcSegment({
          seg_id: "seg_downgraded",
          splice_policy: "mask_loss",
          mask_side: "both",
          vlm_raw_policy: "cut_clean",
          guardrail: {
            gate_type: "splice_gate",
            passed: false,
            downgraded: true,
            downgrade_reason: "Left arm pose jump 14.2cm > 2.0cm; downgraded to mask_loss",
            guardrail_override: false,
            override_reason: null,
            metrics: {
              eef_pos_m: 0.142,
              eef_rot_rad: 0.05,
              hand_rad: 0.1,
              torso_rad: 0.01,
              chassis_pos_m: 0,
              chassis_yaw_rad: 0,
              other_arm_disp_m: 0.01,
              bimanual_coupled: false,
              visual_ssim: null,
            },
            opt_t_start: null,
            opt_t_end: null,
          },
        }),
      ],
    });

    const onSelectSegment = vi.fn();
    const onSelectStage = vi.fn();
    const onSeek = vi.fn();
    const onAddSegment = vi.fn();
    const onAddMacroStage = vi.fn();

    render(
      <FourTrackTimeline
        durationS={ep.meta.duration_s}
        currentTime={1.5}
        macroStages={ep.macro_stages}
        armTimelines={ep.arm_timelines}
        qcSegments={ep.qc_segments}
        rawVlmSegments={ep.raw_vlm_segments}
        showDiffTrack={true}
        selectedSegmentId={null}
        selectedStageId={null}
        onSeek={onSeek}
        onSelectSegment={onSelectSegment}
        onSelectStage={onSelectStage}
        onAddSegment={onAddSegment}
        onAddMacroStage={onAddMacroStage}
      />,
    );

    // Track 2 must explicitly mark read-only and have aria-readonly="true" with zero edit buttons
    const armTrack = screen.getByTestId("track-arm-timelines");
    expect(screen.getByTestId("arm-timelines-readonly")).toHaveTextContent("Read-Only");
    const lane = armTrack.querySelector('[aria-readonly="true"]');
    expect(lane).not.toBeNull();
    expect(armTrack.querySelectorAll("button")).toHaveLength(0);

    // Guardrail downgrade badge is rendered on Track 3
    expect(screen.getByTestId("guardrail-downgraded-seg_downgraded")).toHaveTextContent("Downgraded");

    // Clicking Track 1 Macro Stage & Track 3 QC Segment triggers selection & seek
    fireEvent.click(screen.getByTestId("macro-stage-M1"));
    expect(onSelectStage).toHaveBeenCalledWith("M1");

    fireEvent.click(screen.getByTestId("qc-seg-seg_downgraded"));
    expect(onSelectSegment).toHaveBeenCalledWith("seg_downgraded");

    // Add buttons work
    fireEvent.click(screen.getByTestId("add-qc-segment-btn"));
    expect(onAddSegment).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByTestId("add-macro-stage-btn"));
    expect(onAddMacroStage).toHaveBeenCalledTimes(1);

    // Ghost VLM diff track rendered when showDiffTrack=true
    expect(screen.getByTestId("track-ghost-vlm")).toBeInTheDocument();
  });
});
