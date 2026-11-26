import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { UsageSummary } from "../api/types";
import { FourTrackTimeline } from "../components/FourTrackTimeline";
import { DEFAULT_VISUAL_LABELS } from "../constants/labels";
import { buildDiagnosisReviewLink, fmtCacheHits, fmtDelta } from "../pages/lab/labUtils";
import { ReviewStudioPage } from "../pages/ReviewStudioPage";
import VlmLabPageDefault, { VlmLabPage } from "../pages/VlmLabPage";
import { AuthProvider } from "../stores/authStore";
import { I18nProvider, translate } from "../stores/i18nStore";
import { makeQcSegment, makeResolvedEpisode } from "../test/fixtures";

function mockRect(el: Element, width = 1000, left = 0, height = 42) {
  vi.spyOn(el, "getBoundingClientRect").mockReturnValue({
    left,
    top: 0,
    right: left + width,
    bottom: height,
    width,
    height,
    x: left,
    y: 0,
    toJSON: () => ({}),
  });
}

function makeSampleUsage(): UsageSummary {
  return {
    stages: [
      {
        stage: "s1_grounding",
        model: "gpt-4o",
        prompt_tokens: 8000,
        completion_tokens: 600,
        cached_tokens: 4000,
        total_tokens: 8600,
        elapsed_s: 2.34,
        estimated_cost_usd: 0.018,
        cache_hit: false,
      },
      {
        stage: "s2_macro",
        model: "gpt-4o-mini",
        prompt_tokens: 4200,
        completion_tokens: 300,
        cached_tokens: 2000,
        total_tokens: 4500,
        elapsed_s: 0.85,
        estimated_cost_usd: 0.004,
        cache_hit: true,
      },
      {
        stage: "s3_discover",
        model: "gpt-4o",
        prompt_tokens: 10500,
        completion_tokens: 900,
        cached_tokens: 3800,
        total_tokens: 11400,
        elapsed_s: 3.12,
        estimated_cost_usd: 0.02,
        cache_hit: false,
      },
    ],
    total_tokens: 24500,
    total_cached_tokens: 9800,
    total_elapsed_s: 6.31,
    total_cost_usd: 0.042,
  };
}

describe("ReviewStudioPage run cost telemetry & Display/Shortcuts popover", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.localStorage.clear();
  });

  function stubStudioFetch(usage: UsageSummary | null = makeSampleUsage()) {
    const ep = makeResolvedEpisode({ usage });
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.endsWith("/reviews")) {
          return new Response(JSON.stringify({ items: [] }), { status: 200 });
        }
        if (url.endsWith("/runs")) {
          return new Response(
            JSON.stringify({ episode_id: ep.episode_id, active_run_id: "run_001", runs: [] }),
            { status: 200 },
          );
        }
        if (url.includes("/api/episodes/demo_ds/ep_001")) {
          return new Response(JSON.stringify(ep), { status: 200 });
        }
        if (url.includes("/api/settings")) {
          return new Response(
            JSON.stringify({ min_watch_coverage: 0.95, batch_vlm_concurrency: 4 }),
            { status: 200 },
          );
        }
        if (url.includes("/labels")) {
          return new Response(
            JSON.stringify({ schema_version: "1.0", version: 1, categories: [], labels: [] }),
            { status: 200 },
          );
        }
        return new Response(JSON.stringify({}), { status: 200 });
      }),
    );
  }

  it("renders StudioSkeleton while loading, then shows run cost chip + per-stage breakdown popover", async () => {
    let resolveEp!: (res: Response) => void;
    const ep = makeResolvedEpisode({ usage: makeSampleUsage() });

    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.includes("/api/episodes/demo_ds/ep_001") && !url.endsWith("/reviews") && !url.endsWith("/runs")) {
          return new Promise<Response>((resolve) => {
            resolveEp = resolve;
          });
        }
        if (url.endsWith("/reviews")) {
          return Promise.resolve(new Response(JSON.stringify({ items: [] }), { status: 200 }));
        }
        if (url.endsWith("/runs")) {
          return Promise.resolve(
            new Response(JSON.stringify({ episode_id: ep.episode_id, active_run_id: "run_001", runs: [] }), {
              status: 200,
            }),
          );
        }
        return Promise.resolve(
          new Response(JSON.stringify({ schema_version: "1.0", version: 1, categories: [], labels: [] }), {
            status: 200,
          }),
        );
      }),
    );

    render(
      <AuthProvider initialAuth={{ token: "lead-tok", username: "lead_1", role: "lead" }}>
        <MemoryRouter initialEntries={["/review/demo_ds/ep_001"]}>
          <Routes>
            <Route path="/review/:dataset/:stem" element={<ReviewStudioPage />} />
          </Routes>
        </MemoryRouter>
      </AuthProvider>,
    );

    expect(screen.getByTestId("review-studio-loading")).toBeInTheDocument();
    expect(screen.getByTestId("studio-skeleton")).toBeInTheDocument();

    resolveEp(new Response(JSON.stringify(ep), { status: 200 }));
    await waitFor(() => expect(screen.getByTestId("review-studio-page")).toBeInTheDocument());

    const chip = screen.getByTestId("studio-run-cost-chip");
    expect(chip).toHaveTextContent("24.5k tok");
    expect(chip).toHaveTextContent("$0.042");
    expect(screen.getByTestId("studio-run-cost-cached")).toHaveTextContent("40% cached");

    // Open breakdown popover
    fireEvent.click(chip);
    const popover = screen.getByTestId("studio-run-cost-popover");
    expect(popover).toHaveTextContent("VLM Run Cost & Token Usage");
    expect(popover).toHaveTextContent("Per-Stage Breakdown");
    expect(popover).toHaveTextContent("s1_grounding");
    expect(popover).toHaveTextContent("gpt-4o-mini");
    expect(popover).toHaveTextContent("cache hit");
    expect(popover).toHaveTextContent("6.31s");
    expect(popover).toHaveTextContent("$0.042");

    // Escape closes the cost popover
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByTestId("studio-run-cost-popover")).toBeNull();
  });

  it("hides run cost chip when episode.usage is null and wires Display & Shortcuts popover toggles to FourTrackTimeline", async () => {
    stubStudioFetch(null);

    render(
      <AuthProvider initialAuth={{ token: "rev-tok", username: "rev_1", role: "reviewer" }}>
        <MemoryRouter initialEntries={["/review/demo_ds/ep_001"]}>
          <Routes>
            <Route path="/review/:dataset/:stem" element={<ReviewStudioPage />} />
          </Routes>
        </MemoryRouter>
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("review-studio-page")).toBeInTheDocument());
    expect(screen.queryByTestId("studio-run-cost-chip")).toBeNull();

    const timeline = screen.getByTestId("four-track-timeline");
    expect(timeline.getAttribute("data-snap-enabled")).toBe("true");
    expect(timeline.getAttribute("data-compact")).toBe("false");

    fireEvent.click(screen.getByTestId("studio-display-popover-btn"));
    const displayPopover = screen.getByTestId("studio-display-popover");
    expect(displayPopover).toHaveTextContent("Display & Shortcuts");
    expect(displayPopover).toHaveTextContent("Magnetic Timeline Snap");
    expect(displayPopover).toHaveTextContent("Hold Shift while dragging to bypass snap");
    expect(displayPopover).toHaveTextContent("Compact Track Density");
    expect(displayPopover).toHaveTextContent("Ctrl/Alt + Wheel zooms around cursor");

    // Toggle compact tracks ON & magnetic snap OFF
    fireEvent.click(screen.getByTestId("toggle-compact-tracks"));
    expect(timeline.getAttribute("data-compact")).toBe("true");
    expect(timeline.className).toContain("compact");
    expect(window.localStorage.getItem("robo_vlm_qc.timeline_compact")).toBe("true");

    fireEvent.click(screen.getByTestId("toggle-timeline-snap"));
    expect(timeline.getAttribute("data-snap-enabled")).toBe("false");
    expect(window.localStorage.getItem("robo_vlm_qc.timeline_snap")).toBe("false");

    // Open full shortcuts modal from inside Display popover
    fireEvent.click(screen.getByTestId("display-open-full-shortcuts-btn"));
    expect(screen.queryByTestId("studio-display-popover")).toBeNull();
    expect(screen.getByTestId("shortcuts-modal")).toBeInTheDocument();
  });

  it("renders Chinese (locale='zh') cost telemetry and Display & Shortcuts popover strings", async () => {
    stubStudioFetch(makeSampleUsage());

    render(
      <I18nProvider initialLocale="zh">
        <AuthProvider initialAuth={{ token: "lead-tok", username: "lead_1", role: "lead" }}>
          <MemoryRouter initialEntries={["/review/demo_ds/ep_001"]}>
            <Routes>
              <Route path="/review/:dataset/:stem" element={<ReviewStudioPage />} />
            </Routes>
          </MemoryRouter>
        </AuthProvider>
      </I18nProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("review-studio-page")).toBeInTheDocument());
    expect(screen.getByTestId("studio-run-cost-cached")).toHaveTextContent(
      translate("zh", "studio.costChipCached", { pct: 40 }),
    );

    fireEvent.click(screen.getByTestId("studio-run-cost-chip"));
    const costPopover = screen.getByTestId("studio-run-cost-popover");
    expect(costPopover).toHaveTextContent(translate("zh", "studio.costChipTitle"));
    expect(costPopover).toHaveTextContent(translate("zh", "studio.costStageBreakdown"));
    expect(costPopover).toHaveTextContent(translate("zh", "studio.costStageCacheHit"));

    fireEvent.click(screen.getByTestId("studio-display-popover-btn"));
    const displayPopover = screen.getByTestId("studio-display-popover");
    expect(displayPopover).toHaveTextContent(translate("zh", "studio.displayControls"));
    expect(displayPopover).toHaveTextContent(translate("zh", "studio.snapToGuides"));
    expect(displayPopover).toHaveTextContent(translate("zh", "studio.compactTracks"));
    expect(displayPopover).toHaveTextContent(translate("zh", "studio.zoomTip"));
  });
});

describe("FourTrackTimeline cursor-anchored zoom, magnetic snap guide & compact density", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("snaps QC segment edge drags to macro stage, playhead, and adjacent segment edges within 0.15s, renders timeline-snap-guide, and bypasses on Shift or snapEnabled=false", () => {
    const ep = makeResolvedEpisode({
      qc_segments: [
        makeQcSegment({ seg_id: "seg_a", t_start: 1.0, t_end: 3.0 }),
        makeQcSegment({ seg_id: "seg_b", t_start: 6.2, t_end: 8.0 }),
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

    const onUpdateSegmentSpan = vi.fn();
    const { rerender } = render(
      <FourTrackTimeline
        durationS={10.0}
        currentTime={4.0}
        macroStages={ep.macro_stages}
        armTimelines={ep.arm_timelines}
        qcSegments={ep.qc_segments}
        selectedSegmentId="seg_a"
        selectedStageId={null}
        snapEnabled={true}
        compact={true}
        onSeek={vi.fn()}
        onSelectSegment={vi.fn()}
        onSelectStage={vi.fn()}
        onAddSegment={vi.fn()}
        onAddMacroStage={vi.fn()}
        onUpdateSegmentSpan={onUpdateSegmentSpan}
      />,
    );

    const timeline = screen.getByTestId("four-track-timeline");
    expect(timeline.className).toContain("compact");
    expect(timeline.getAttribute("data-compact")).toBe("true");

    const qcLane = screen.getByTestId("qc-track-lane");
    mockRect(qcLane, 1000, 0);

    const rightHandle = screen.getByTestId("resize-right-seg-seg_a");

    // 1. Snap to macro stage boundary at 5.0s (drag to 4.92s -> dist 0.08s <= 0.15s)
    fireEvent.pointerDown(rightHandle, { button: 0, clientX: 300 });
    fireEvent.pointerMove(window, { clientX: 492, shiftKey: false });
    const snapGuide1 = screen.getByTestId("timeline-snap-guide");
    expect(snapGuide1.style.left).toBe("50%");
    expect(snapGuide1).toHaveTextContent("5.00s");
    fireEvent.pointerUp(window, { clientX: 492, shiftKey: false });
    expect(onUpdateSegmentSpan).toHaveBeenLastCalledWith("seg_a", 1, 5);

    // 2. Snap to playhead at 4.0s (drag to 3.9s -> dist 0.10s <= 0.15s)
    fireEvent.pointerDown(rightHandle, { button: 0, clientX: 300 });
    fireEvent.pointerMove(window, { clientX: 390, shiftKey: false });
    const snapGuide2 = screen.getByTestId("timeline-snap-guide");
    expect(snapGuide2.style.left).toBe("40%");
    fireEvent.pointerUp(window, { clientX: 390, shiftKey: false });
    expect(onUpdateSegmentSpan).toHaveBeenLastCalledWith("seg_a", 1, 4);

    // 3. Snap to adjacent QC segment start edge at 6.2s (drag to 6.12s -> dist 0.08s <= 0.15s)
    fireEvent.pointerDown(rightHandle, { button: 0, clientX: 300 });
    fireEvent.pointerMove(window, { clientX: 612, shiftKey: false });
    expect(screen.getByTestId("timeline-snap-guide").style.left).toBe("62%");
    fireEvent.pointerUp(window, { clientX: 612, shiftKey: false });
    expect(onUpdateSegmentSpan).toHaveBeenLastCalledWith("seg_a", 1, 6.2);

    // 4. Holding Shift bypasses magnetic snapping & hides snap guide
    fireEvent.pointerDown(rightHandle, { button: 0, clientX: 300 });
    fireEvent.pointerMove(window, { clientX: 492, shiftKey: true });
    expect(screen.queryByTestId("timeline-snap-guide")).toBeNull();
    fireEvent.pointerUp(window, { clientX: 492, shiftKey: true });
    expect(onUpdateSegmentSpan).toHaveBeenLastCalledWith("seg_a", 1, 4.92);

    // 5. Disabling snapEnabled prop bypasses magnetic snapping
    rerender(
      <FourTrackTimeline
        durationS={10.0}
        currentTime={4.0}
        macroStages={ep.macro_stages}
        armTimelines={ep.arm_timelines}
        qcSegments={ep.qc_segments}
        selectedSegmentId="seg_a"
        selectedStageId={null}
        snapEnabled={false}
        compact={false}
        onSeek={vi.fn()}
        onSelectSegment={vi.fn()}
        onSelectStage={vi.fn()}
        onAddSegment={vi.fn()}
        onAddMacroStage={vi.fn()}
        onUpdateSegmentSpan={onUpdateSegmentSpan}
      />,
    );

    expect(timeline.className).not.toContain("compact");
    fireEvent.pointerDown(rightHandle, { button: 0, clientX: 300 });
    fireEvent.pointerMove(window, { clientX: 492, shiftKey: false });
    expect(screen.queryByTestId("timeline-snap-guide")).toBeNull();
    fireEvent.pointerUp(window, { clientX: 492, shiftKey: false });
    expect(onUpdateSegmentSpan).toHaveBeenLastCalledWith("seg_a", 1, 4.92);
  });

  it("re-anchors horizontal scroll around mouse cursor on Ctrl+Wheel zoom", () => {
    const ep = makeResolvedEpisode();
    render(
      <FourTrackTimeline
        durationS={10.0}
        currentTime={5.0}
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
      />,
    );

    const scrollEl = screen.getByTestId("timeline-scroll");
    Object.defineProperty(scrollEl, "clientWidth", { value: 1120, configurable: true });
    mockRect(scrollEl, 1120, 0, 220);

    // Pointer at x=620 => 500px into 1000px lane (ratio = 0.50). Zooming in to 1.25x should scroll right (>0) to keep t=5s under cursor!
    fireEvent.wheel(scrollEl, { ctrlKey: true, deltaY: -120, clientX: 620 });

    const canvas = screen.getByTestId("timeline-canvas");
    expect(canvas.style.width).toBe("125%");
    expect(screen.getByTestId("timeline-zoom-readout")).toHaveTextContent("1.25x");
    expect(scrollEl.scrollLeft).toBeGreaterThan(100);
  });
});

describe("Modularized VlmLabPage & labUtils", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("preserves named and default VlmLabPage exports and labUtils helpers", () => {
    expect(VlmLabPage).toBe(VlmLabPageDefault);
    expect(fmtDelta(0.15, { pct: true })).toEqual({ text: "+15.0%", cls: "delta-pos" });
    expect(fmtDelta(-0.1, { pct: true, lowerBetter: true })).toEqual({
      text: "-10.0%",
      cls: "delta-pos",
    });
    expect(fmtCacheHits({ s1_grounding: 2, s2_macro: 1 }, 0.6)).toBe("3 (s1:2 s2:1, 60.0%)");
    expect(
      buildDiagnosisReviewLink({
        error_id: "e1",
        episode_id: "ds/ep_1",
        task_id: "pick",
        error_type: "fp_overflag",
        t_start: 2.5,
        t_end: 4.0,
        vlm_seg_id: "s1",
        summary: "fp",
      }),
    ).toBe("/review/ds/ep_1?t=2.5&seg=s1&tab=diff");
  });

  it("switches cleanly across all 4 modularized Lab tabs and taxonomy subtabs", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.startsWith("/api/lab/labels")) {
          return new Response(
            JSON.stringify({
              schema_version: "1.0",
              version: 1,
              categories: ["idle", "failure"],
              labels: DEFAULT_VISUAL_LABELS,
            }),
            { status: 200 },
          );
        }
        if (url.startsWith("/api/lab/prompts")) {
          const tpl = (stage: string) => ({
            stage,
            version: "1.0.0",
            description: stage,
            system_template: `${stage} sys`,
            user_template: `${stage} usr`,
            extra_templates: {},
          });
          return new Response(
            JSON.stringify({
              s1_grounding: tpl("s1_grounding"),
              s2_macro: tpl("s2_macro"),
              s3_discover: tpl("s3_discover"),
              s4_verify: tpl("s4_verify"),
              s5_success: tpl("s5_success"),
            }),
            { status: 200 },
          );
        }
        if (url.startsWith("/api/lab/tasks")) {
          return new Response(JSON.stringify({ tasks: {}, total: 0 }), { status: 200 });
        }
        if (url.startsWith("/api/lab/diagnosis")) {
          return new Response(
            JSON.stringify({
              episodes_evaluated: 1,
              total_errors: 0,
              counts_by_type: {},
              counts_by_label: {},
              counts_by_task: {},
              errors: [],
              top_errors: [],
            }),
            { status: 200 },
          );
        }
        if (url.startsWith("/api/lab/unknowns")) {
          return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200 });
        }
        if (url.startsWith("/api/lab/few-shots")) {
          return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200 });
        }
        if (url.startsWith("/api/lab/eval/compare")) {
          return new Response(
            JSON.stringify({ evals: [], baseline: null, candidate: null, deltas: null }),
            { status: 200 },
          );
        }
        return new Response(JSON.stringify({}), { status: 200 });
      }),
    );

    render(
      <AuthProvider initialAuth={{ token: "lead-tok", username: "lead_1", role: "lead" }}>
        <MemoryRouter>
          <VlmLabPage />
        </MemoryRouter>
      </AuthProvider>,
    );

    expect(await screen.findByTestId("lab-panel-diagnosis")).toBeInTheDocument();

    fireEvent.click(screen.getByTestId("tab-few-shots"));
    expect(await screen.findByTestId("lab-panel-few-shots")).toBeInTheDocument();

    fireEvent.click(screen.getByTestId("tab-taxonomy"));
    expect(await screen.findByTestId("labels-editor")).toBeInTheDocument();
    fireEvent.click(screen.getByTestId("taxonomy-subtab-prompts"));
    expect(await screen.findByTestId("prompts-editor")).toBeInTheDocument();
    fireEvent.click(screen.getByTestId("taxonomy-subtab-tasks"));
    expect(await screen.findByTestId("tasks-editor")).toBeInTheDocument();

    fireEvent.click(screen.getByTestId("tab-eval"));
    expect(await screen.findByTestId("lab-panel-eval")).toBeInTheDocument();
  });
});
