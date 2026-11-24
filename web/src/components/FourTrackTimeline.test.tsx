import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { I18nProvider } from "../stores/i18nStore";
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
    // Track 1 header renamed cleanly to Subtask Stages
    expect(screen.getByTestId("track-macro-stages")).toHaveTextContent("1 · Subtask Stages");
  });

  it("supports cross-track magnetic snap to QC segment boundaries within 0.08s and renders Align / Infer Subtasks actions", () => {
    const ep = makeResolvedEpisode({
      qc_segments: [
        makeQcSegment({
          seg_id: "seg_cut",
          t_start: 3.0,
          t_end: 4.5,
          splice_policy: "cut_clean",
          mask_side: null,
        }),
      ],
      macro_stages: [
        {
          stage_id: "M1",
          t_start: 0.0,
          t_end: 5.0,
          label: "Pick",
          label_en: "pick",
          summary: "",
          color_rgb: [59, 130, 246],
        },
        {
          stage_id: "M2",
          t_start: 5.0,
          t_end: 10.0,
          label: "Place",
          label_en: "place",
          summary: "",
          color_rgb: [16, 185, 129],
        },
      ],
    });

    const onUpdateStageSpans = vi.fn();
    const onUpdateStageSpan = vi.fn();
    const onAlignStagesToQc = vi.fn();
    const onInferSubtasks = vi.fn();

    render(
      <FourTrackTimeline
        durationS={10.0}
        currentTime={1.5}
        macroStages={ep.macro_stages}
        armTimelines={ep.arm_timelines}
        qcSegments={ep.qc_segments}
        selectedSegmentId={null}
        selectedStageId={null}
        onSeek={vi.fn()}
        onSelectSegment={vi.fn()}
        onSelectStage={vi.fn()}
        onAddSegment={vi.fn()}
        onAddMacroStage={vi.fn()}
        onUpdateStageSpan={onUpdateStageSpan}
        onUpdateStageSpans={onUpdateStageSpans}
        onAlignStagesToQc={onAlignStagesToQc}
        onInferSubtasks={onInferSubtasks}
      />,
    );

    // Align and Infer Subtasks toolbar buttons work
    fireEvent.click(screen.getByTestId("align-stages-qc-btn"));
    expect(onAlignStagesToQc).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByTestId("infer-subtasks-btn"));
    expect(onInferSubtasks).toHaveBeenCalledTimes(1);

    // Drag M1 right edge to 4.45s (within 0.08s of seg_cut.t_end = 4.50s) -> snaps to 4.50s and ripples M2!
    const macroLane = screen.getByTestId("macro-track-lane");
    vi.spyOn(macroLane, "getBoundingClientRect").mockReturnValue({
      left: 0,
      top: 0,
      right: 1000,
      bottom: 42,
      width: 1000,
      height: 42,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    });

    const m1RightHandle = screen.getByTestId("resize-right-stage-M1");
    fireEvent.pointerDown(m1RightHandle, { button: 0, clientX: 500, altKey: false });
    fireEvent.pointerMove(window, { clientX: 445, altKey: false }); // 4.45s -> snaps to 4.50s
    expect(screen.getByTestId("qc-snap-guide")).toBeInTheDocument();
    fireEvent.pointerUp(window, { clientX: 445, altKey: false });

    expect(onUpdateStageSpans).toHaveBeenCalledWith([
      { stageId: "M1", tStart: 0, tEnd: 4.5 },
      { stageId: "M2", tStart: 4.5, tEnd: 10 },
    ]);

    // Holding Alt disables cross-track QC snap (and neighbor ripple)
    fireEvent.pointerDown(m1RightHandle, { button: 0, clientX: 500, altKey: true });
    fireEvent.pointerMove(window, { clientX: 445, altKey: true });
    expect(screen.queryByTestId("qc-snap-guide")).toBeNull();
    fireEvent.pointerUp(window, { clientX: 445, altKey: true });
    expect(onUpdateStageSpan).toHaveBeenCalledWith("M1", 0, 4.45);
  });

  it("renders Chinese (locale='zh') track headings and toolbar actions", () => {
    const ep = makeResolvedEpisode();
    render(
      <I18nProvider initialLocale="zh">
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
          onSeek={vi.fn()}
          onSelectSegment={vi.fn()}
          onSelectStage={vi.fn()}
          onAddSegment={vi.fn()}
          onAddMacroStage={vi.fn()}
          onAlignStagesToQc={vi.fn()}
          onInferSubtasks={vi.fn()}
        />
      </I18nProvider>,
    );

    expect(screen.getByTestId("four-track-timeline")).toHaveTextContent("四轨同步时间轴");
    expect(screen.getByTestId("track-macro-stages")).toHaveTextContent("1 · 子任务阶段");
    expect(screen.getByTestId("track-arm-timelines")).toHaveTextContent("2. 双臂状态");
    expect(screen.getByTestId("arm-timelines-readonly")).toHaveTextContent("只读");
    expect(screen.getByTestId("track-qc-segments")).toHaveTextContent("3. 质检分段");
    expect(screen.getByTestId("align-stages-qc-btn")).toHaveTextContent("对齐 QC 切点");
    expect(screen.getByTestId("infer-subtasks-btn")).toHaveTextContent("反推子任务");
  });
});
