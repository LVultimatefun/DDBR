import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { buildReviewWritePayload, frameUrl, previewVideoUrl } from "../api/client";
import { AuthProvider } from "../stores/authStore";
import { I18nProvider } from "../stores/i18nStore";
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

  it("SuccessPanel supports full-width multi-line editing, tri-state pills, Now/Clear timestamps, add/edit/delete criteria, and non-destructive Verdict sync", () => {
    let state = {
      passed: true as boolean | null,
      verdict: "success" as "success" | "failure" | "unsure",
      confidence: 0.9,
      criteria: [
        {
          criterion: "Green mug grasped securely",
          met: true as boolean | null,
          satisfied_at_s: 0.89 as number | null,
          evidence: "At t=00.89s, the green mug is lifted cleanly from the counter",
        },
        {
          criterion: "Mug placed upright in drying rack",
          met: null as boolean | null,
          satisfied_at_s: null as number | null,
          evidence: "",
        },
      ],
      control_check: null,
      summary: "Initial summary text",
    };

    const onSeek = vi.fn();
    const onChange = vi.fn((next) => {
      state = next;
    });

    const { rerender } = render(
      <SuccessPanel
        success={state}
        currentTime={4.25}
        durationS={10.0}
        onChange={onChange}
        onSeek={onSeek}
      />,
    );

    const rerenderPanel = (curTime = 4.25, ro = false) =>
      rerender(
        <SuccessPanel
          success={state}
          currentTime={curTime}
          durationS={10.0}
          readOnly={ro}
          onChange={onChange}
          onSeek={onSeek}
        />,
      );

    // Full-width multi-line Verdict Summary (rows=3) and Criterion evidence (rows=2)
    const summaryTa = screen.getByTestId("verdict-summary-textarea") as HTMLTextAreaElement;
    expect(summaryTa.tagName).toBe("TEXTAREA");
    expect(summaryTa.rows).toBe(3);

    const evidence0 = screen.getByTestId("criterion-evidence-0") as HTMLTextAreaElement;
    expect(evidence0.tagName).toBe("TEXTAREA");
    expect(evidence0.rows).toBe(2);
    expect(evidence0.value).toContain("green mug is lifted cleanly");

    // Mix of Met + Unsure with verdict="success" -> non-destructive sync bar suggests Unsure
    expect(screen.getByTestId("verdict-sync-bar")).toHaveTextContent("Unsure");

    // Existing timestamp: seek pill works, Now stamps currentTime (4.25), numeric edit works, Clear resets to null
    fireEvent.click(screen.getByTestId("criterion-seek-0"));
    expect(onSeek).toHaveBeenCalledWith(0.89);

    fireEvent.click(screen.getByTestId("criterion-now-0"));
    expect(state.criteria[0].satisfied_at_s).toBe(4.25);
    rerenderPanel();

    fireEvent.change(screen.getByTestId("criterion-time-input-0"), { target: { value: "5.5" } });
    expect(state.criteria[0].satisfied_at_s).toBe(5.5);
    rerenderPanel();

    fireEvent.click(screen.getByTestId("criterion-clear-time-0"));
    expect(state.criteria[0].satisfied_at_s).toBeNull();
    rerenderPanel();

    // Null timestamp displays "+ Set @ Now (4.25s)" button and stamps playhead
    const setNowBtn = screen.getByTestId("criterion-set-now-1");
    expect(setNowBtn).toHaveTextContent("+ Set @ Now (4.25s)");
    fireEvent.click(setNowBtn);
    expect(state.criteria[1].satisfied_at_s).toBe(4.25);
    rerenderPanel();

    // Tri-state pill buttons replace <select>: click Not Met on criterion 1 -> sync bar suggests Failure
    const notMetBtn = screen.getByTestId("criterion-1-not-met");
    fireEvent.click(notMetBtn);
    expect(state.criteria[1].met).toBe(false);
    // Non-destructive: top verdict is still "success" until user clicks Sync Verdict
    expect(state.verdict).toBe("success");
    rerenderPanel();

    expect(screen.getByTestId("criterion-1-not-met")).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByTestId("verdict-sync-bar")).toHaveTextContent("Failure");
    fireEvent.click(screen.getByTestId("sync-verdict-btn"));
    expect(state.verdict).toBe("failure");
    expect(state.passed).toBe(false);
    rerenderPanel();
    // Sync bar disappears once aligned
    expect(screen.queryByTestId("verdict-sync-bar")).toBeNull();

    // Add Criterion, edit title, delete criterion
    fireEvent.click(screen.getByTestId("add-criterion-btn"));
    expect(state.criteria).toHaveLength(3);
    rerenderPanel();

    fireEvent.change(screen.getByTestId("criterion-title-2"), {
      target: { value: "No water spilled on counter" },
    });
    expect(state.criteria[2].criterion).toBe("No water spilled on counter");
    rerenderPanel();

    fireEvent.click(screen.getByTestId("delete-criterion-1"));
    expect(state.criteria).toHaveLength(2);
    rerenderPanel();

    // Read-only hides mutation controls and disables textareas
    rerenderPanel(4.25, true);
    expect(screen.queryByTestId("add-criterion-btn")).toBeNull();
    expect(screen.queryByTestId("delete-criterion-0")).toBeNull();
    expect(screen.getByTestId("verdict-summary-textarea")).toBeDisabled();
    expect(screen.getByTestId("criterion-evidence-0")).toBeDisabled();
  });

  it("SegmentInspector Stage view edits Subtask Training Instruction (instruction)", () => {
    const onUpdateStage = vi.fn();
    render(
      <SegmentInspector
        episodeId="demo_ds/ep_001"
        taskId="pick_and_place"
        durationS={10.0}
        currentTime={2.0}
        isLead={false}
        selectedSegment={null}
        selectedStage={{
          stage_id: "M1",
          t_start: 0,
          t_end: 5,
          label: "Pick Mug",
          label_en: "pick_mug",
          summary: "Arm approaches mug",
          instruction: "Pick up the green mug",
          color_rgb: [59, 130, 246],
        }}
        onUpdateSegment={vi.fn()}
        onDeleteSegment={vi.fn()}
        onUpdateStage={onUpdateStage}
        onDeleteStage={vi.fn()}
      />,
    );

    const instrInput = screen.getByTestId("stage-instruction-input") as HTMLTextAreaElement;
    expect(instrInput.value).toBe("Pick up the green mug");
    fireEvent.change(instrInput, {
      target: { value: "Pick up the green mug and place it in the rack" },
    });
    expect(onUpdateStage).toHaveBeenCalledWith(
      expect.objectContaining({
        stage_id: "M1",
        instruction: "Pick up the green mug and place it in the rack",
      }),
    );
  });

  it("renders Chinese (locale='zh') UI labels across SuccessPanel, SegmentInspector, and StudioPlayer", () => {
    const ep = makeResolvedEpisode();
    render(
      <I18nProvider initialLocale="zh">
        <AuthProvider initialAuth={{ token: "tok", username: "rev1", role: "reviewer" }}>
          <StudioPlayer
            episodeId={ep.episode_id}
            cameras={ep.meta.cameras}
            durationS={ep.meta.duration_s}
            fps={ep.meta.fps}
            grounding={ep.grounding}
            currentTime={0}
            onTimeUpdate={vi.fn()}
          />
          <SegmentInspector
            episodeId={ep.episode_id}
            taskId="pick_and_place"
            durationS={10.0}
            currentTime={2.0}
            isLead={false}
            selectedSegment={null}
            selectedStage={ep.macro_stages[0]}
            onUpdateSegment={vi.fn()}
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
                {
                  criterion: "物体稳妥放置",
                  met: false,
                  satisfied_at_s: null,
                  evidence: "未放稳",
                },
              ],
              control_check: null,
              summary: "总结",
            }}
            currentTime={3.5}
            durationS={10.0}
            onChange={vi.fn()}
            onSeek={vi.fn()}
          />
        </AuthProvider>
      </I18nProvider>,
    );

    // StudioPlayer Chinese labels
    expect(screen.getByTestId("toggle-grounding")).toHaveTextContent("2D 目标框 (1)");
    expect(screen.getByTestId("grounding-object-bar")).toHaveTextContent("物体过滤：");
    expect(screen.getByTestId("play-pause-btn")).toHaveTextContent("播放");
    expect(screen.getByTestId("toggle-hd-frame")).toHaveTextContent("高清单帧");

    // SegmentInspector Stage view Chinese labels
    expect(screen.getByTestId("macro-stage-inspector")).toHaveTextContent("子任务阶段 (M1)");
    expect(screen.getByTestId("macro-stage-inspector")).toHaveTextContent(
      "子任务训练指令 (instruction)",
    );

    // SuccessPanel Chinese labels
    expect(screen.getByTestId("success-panel")).toHaveTextContent("任务成功判定（第 2 层）");
    expect(screen.getByTestId("verdict-success-btn")).toHaveTextContent("成功");
    expect(screen.getByTestId("verdict-failure-btn")).toHaveTextContent("失败");
    expect(screen.getByTestId("verdict-sync-bar")).toHaveTextContent("同步结论");
    expect(screen.getByTestId("add-criterion-btn")).toHaveTextContent("新增判据");
    expect(screen.getByTestId("criterion-0-met")).toHaveTextContent("已满足");
    expect(screen.getByTestId("criterion-0-not-met")).toHaveTextContent("未满足");
    expect(screen.getByTestId("criterion-set-now-0")).toHaveTextContent("+ 设为当前时间 (3.50s)");
  });
});
