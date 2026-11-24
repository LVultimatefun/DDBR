import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { FourTrackTimeline } from "../components/FourTrackTimeline";
import { SegmentInspector } from "../components/SegmentInspector";
import { StudioPlayer } from "../components/StudioPlayer";
import { AuthProvider } from "../stores/authStore";
import { JobProvider } from "../stores/jobStore";
import { computeUnwatchedIntervals } from "../stores/reviewHistoryStore";
import { ToastProvider } from "../stores/toastStore";
import {
  makeEpisodeListItem,
  makeJobStatus,
  makeQcSegment,
  makeResolvedEpisode,
  makeReviewArtifact,
} from "../test/fixtures";
import { ReviewStudioPage } from "./ReviewStudioPage";

const LocationEcho = () => {
  const loc = useLocation();
  return <div data-testid="location-echo">{`${loc.pathname}${loc.search}`}</div>;
};

function mockRect(el: Element, width = 1000, left = 0) {
  vi.spyOn(el, "getBoundingClientRect").mockReturnValue({
    left,
    top: 0,
    right: left + width,
    bottom: 42,
    width,
    height: 42,
    x: left,
    y: 0,
    toJSON: () => ({}),
  });
}

describe("Review Studio NLE, Multi-Cam, Live Splice, Shortcuts & Lead Workflows", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.localStorage.clear();
  });

  it("1. FourTrackTimeline: pointer-drag edge resize pushes single undo, brush-creates >=0.2s, magnetic ripples stages & Alt detaches, heatmap jumps to unwatched gap", () => {
    const ep = makeResolvedEpisode({
      qc_segments: [makeQcSegment({ seg_id: "seg_1", t_start: 1.0, t_end: 3.0 })],
      macro_stages: [
        {
          stage_id: "M1",
          t_start: 0.0,
          t_end: 5.0,
          label: "Stage 1",
          label_en: "stage_1",
          summary: "",
          color_rgb: [59, 130, 246],
        },
        {
          stage_id: "M2",
          t_start: 5.0,
          t_end: 10.0,
          label: "Stage 2",
          label_en: "stage_2",
          summary: "",
          color_rgb: [16, 185, 129],
        },
      ],
    });

    const onSeek = vi.fn();
    const onUpdateSegmentSpan = vi.fn();
    const onCreateSegmentSpan = vi.fn();
    const onUpdateStageSpan = vi.fn();
    const onUpdateStageSpans = vi.fn();
    const onSplitMacroStage = vi.fn();

    render(
      <FourTrackTimeline
        durationS={10.0}
        currentTime={2.5}
        macroStages={ep.macro_stages}
        armTimelines={ep.arm_timelines}
        qcSegments={ep.qc_segments}
        watchCoverage={{
          watched_intervals: [
            [0, 2],
            [6, 10],
          ],
          watched_s: 6,
          duration_s: 10,
          coverage_ratio: 0.6,
          confirmed: false,
        }}
        selectedSegmentId="seg_1"
        selectedStageId={null}
        onSeek={onSeek}
        onSelectSegment={vi.fn()}
        onSelectStage={vi.fn()}
        onAddSegment={vi.fn()}
        onAddMacroStage={vi.fn()}
        onCreateSegmentSpan={onCreateSegmentSpan}
        onSplitMacroStage={onSplitMacroStage}
        onUpdateSegmentSpan={onUpdateSegmentSpan}
        onUpdateStageSpan={onUpdateStageSpan}
        onUpdateStageSpans={onUpdateStageSpans}
      />,
    );

    // Track 2 remains strictly read-only with 0 buttons
    const armTrack = screen.getByTestId("track-arm-timelines");
    expect(armTrack.querySelectorAll("button")).toHaveLength(0);

    // Heatmap bar shows unwatched gap [2, 6] and clicking jumps to 2.0s
    expect(computeUnwatchedIntervals([[0, 2], [6, 10]], 10)).toEqual([[2, 6]]);
    const gapBtn = screen.getByTestId("heatmap-unwatched-gap-0");
    fireEvent.click(gapBtn);
    expect(onSeek).toHaveBeenCalledWith(2);

    // 1a. QC segment right edge drag across multiple moves -> SINGLE commit on pointerup
    const qcLane = screen.getByTestId("qc-track-lane");
    mockRect(qcLane, 1000, 0);
    const segRightHandle = screen.getByTestId("resize-right-seg-seg_1");
    fireEvent.pointerDown(segRightHandle, { button: 0, clientX: 300 });
    fireEvent.pointerMove(window, { clientX: 350 });
    fireEvent.pointerMove(window, { clientX: 400 });
    fireEvent.pointerMove(window, { clientX: 450 });
    expect(onUpdateSegmentSpan).not.toHaveBeenCalled();
    fireEvent.pointerUp(window, { clientX: 450 });
    expect(onUpdateSegmentSpan).toHaveBeenCalledTimes(1);
    expect(onUpdateSegmentSpan).toHaveBeenCalledWith("seg_1", 1, 4.5);

    // 1b. Brush-create on empty QC track lane (>= 0.2s)
    fireEvent.pointerDown(qcLane, { button: 0, clientX: 600 });
    fireEvent.pointerMove(window, { clientX: 750 });
    expect(screen.getByTestId("qc-brush-preview")).toBeInTheDocument();
    fireEvent.pointerUp(window, { clientX: 750 });
    expect(onCreateSegmentSpan).toHaveBeenCalledTimes(1);
    expect(onCreateSegmentSpan).toHaveBeenCalledWith(6, 7.5);

    // Sub-threshold click (<0.2s) on empty lane does not brush-create
    fireEvent.pointerDown(qcLane, { button: 0, clientX: 800 });
    fireEvent.pointerMove(window, { clientX: 810 }); // 0.1s < 0.2s
    fireEvent.pointerUp(window, { clientX: 810 });
    expect(onCreateSegmentSpan).toHaveBeenCalledTimes(1);

    // 1c. Magnetic shared-boundary ripple dragging for adjacent stages M1 & M2
    const macroLane = screen.getByTestId("macro-track-lane");
    mockRect(macroLane, 1000, 0);
    const m1RightHandle = screen.getByTestId("resize-right-stage-M1");
    fireEvent.pointerDown(m1RightHandle, { button: 0, clientX: 500, altKey: false });
    fireEvent.pointerMove(window, { clientX: 620, altKey: false });
    fireEvent.pointerUp(window, { clientX: 620, altKey: false });
    expect(onUpdateStageSpans).toHaveBeenCalledTimes(1);
    expect(onUpdateStageSpans).toHaveBeenCalledWith([
      { stageId: "M1", tStart: 0, tEnd: 6.2 },
      { stageId: "M2", tStart: 6.2, tEnd: 10 },
    ]);

    // 1d. Alt key detaches magnetic ripple -> moves ONLY M1
    fireEvent.pointerDown(m1RightHandle, { button: 0, clientX: 500, altKey: true });
    fireEvent.pointerMove(window, { clientX: 420, altKey: true });
    fireEvent.pointerUp(window, { clientX: 420, altKey: true });
    expect(onUpdateStageSpan).toHaveBeenCalledWith("M1", 0, 4.2);
    expect(onUpdateStageSpans).toHaveBeenCalledTimes(1);

    // 1e. Split button enabled at currentTime=2.5 (inside M1 [0,5])
    fireEvent.click(screen.getByTestId("split-macro-stage-btn"));
    expect(onSplitMacroStage).toHaveBeenCalledTimes(1);
  });

  it("2. StudioPlayer: Multi-cam Single/PiP/2x2 synced playback, 3x speed, Cut-Skip (K), MASK banner, Segment Loop (R), and stepping watch coverage", () => {
    const ep = makeResolvedEpisode();
    ep.meta.cameras = [
      ...ep.meta.cameras,
      {
        name: "left_wrist",
        role: "left_wrist",
        width: 640,
        height: 480,
        fps: 30,
        source_type: "h5_jpeg",
        is_virtual_crop: false,
        source_stream: "left_wrist",
      },
      {
        name: "right_wrist",
        role: "right_wrist",
        width: 640,
        height: 480,
        fps: 30,
        source_type: "h5_jpeg",
        is_virtual_crop: false,
        source_stream: "right_wrist",
      },
    ];
    const cutSeg = makeQcSegment({
      seg_id: "seg_cut",
      t_start: 2.0,
      t_end: 4.0,
      splice_policy: "cut_clean",
      mask_side: null,
      training_role: "badcase",
    });
    const maskSeg = makeQcSegment({
      seg_id: "seg_mask",
      t_start: 5.0,
      t_end: 6.5,
      splice_policy: "mask_loss",
      mask_side: "left",
      training_role: "recovery",
    });

    const onTimeUpdate = vi.fn();
    const onWatchInterval = vi.fn();

    const { rerender } = render(
      <AuthProvider initialAuth={{ token: "tok", username: "rev1", role: "reviewer" }}>
        <StudioPlayer
          episodeId={ep.episode_id}
          cameras={ep.meta.cameras}
          durationS={10.0}
          fps={30}
          grounding={ep.grounding}
          currentTime={1.0}
          onTimeUpdate={onTimeUpdate}
          onWatchInterval={onWatchInterval}
          qcSegments={[cutSeg, maskSeg]}
          selectedSegment={maskSeg}
        />
      </AuthProvider>,
    );

    // 3x speed option exists
    fireEvent.click(screen.getByText("3x"));
    const primaryVideo = screen.getByTestId("primary-video") as HTMLVideoElement;
    expect(primaryVideo.playbackRate).toBe(3);

    // Switch to PiP mode -> secondary PiP overlay mounted, Grounding canvas stays inside primary pane
    fireEvent.click(screen.getByTestId("view-mode-pip"));
    expect(screen.getByTestId("pip-cam-torso_rgbd")).toBeInTheDocument();
    expect(screen.getByTestId("primary-cam-pane")).toContainElement(
      screen.getByTestId("grounding-canvas"),
    );

    // Switch to 2x2 Grid mode -> all 3 secondary cameras mounted & playbackRate synced to 3x
    fireEvent.click(screen.getByTestId("view-mode-grid"));
    expect(screen.getByTestId("grid-cam-torso_rgbd")).toBeInTheDocument();
    expect(screen.getByTestId("grid-cam-left_wrist")).toBeInTheDocument();
    expect(screen.getByTestId("grid-cam-right_wrist")).toBeInTheDocument();
    const secVid = screen.getByTestId("secondary-video-torso_rgbd") as HTMLVideoElement;
    expect(secVid.playbackRate).toBe(3);

    // Enable Cut-Skip Preview -> timeupdate entering cut_clean [2.0, 4.0] skips to 4.02 & flashes badge
    fireEvent.click(screen.getByTestId("toggle-cut-skip"));
    Object.defineProperty(primaryVideo, "currentTime", {
      value: 2.1,
      writable: true,
      configurable: true,
    });
    fireEvent.timeUpdate(primaryVideo);
    expect(onTimeUpdate).toHaveBeenCalledWith(4.02);
    expect(screen.getByTestId("cut-skip-flash")).toHaveTextContent("Skipped cut_clean");

    // Re-render at t=5.5 (inside mask_loss [5.0, 6.5]) -> subtle MASK banner shown
    rerender(
      <AuthProvider initialAuth={{ token: "tok", username: "rev1", role: "reviewer" }}>
        <StudioPlayer
          episodeId={ep.episode_id}
          cameras={ep.meta.cameras}
          durationS={10.0}
          fps={30}
          grounding={ep.grounding}
          currentTime={5.5}
          onTimeUpdate={onTimeUpdate}
          onWatchInterval={onWatchInterval}
          qcSegments={[cutSeg, maskSeg]}
          selectedSegment={maskSeg}
          cutSkipEnabled={true}
          loopSegmentEnabled={true}
        />
      </AuthProvider>,
    );
    expect(screen.getByTestId("mask-loss-banner")).toHaveTextContent("MASK");
    expect(screen.getByTestId("loop-range-pill")).toHaveTextContent("4.5–7.0s");

    // Loop wrap-around when crossing t_end + 0.5 = 7.0s -> wraps back to 4.5s
    Object.defineProperty(primaryVideo, "currentTime", {
      value: 7.05,
      writable: true,
      configurable: true,
    });
    fireEvent.timeUpdate(primaryVideo);
    expect(onTimeUpdate).toHaveBeenCalledWith(4.5);

    // Stepping accumulates watch coverage cleanly
    fireEvent.click(screen.getByText("+0.5s"));
    expect(onWatchInterval).toHaveBeenCalledWith(5.5, 6.0);
  });

  it("3. SegmentInspector: Start vs End side-by-side thumbnails, debounced 250ms checkSplice & 1-click opt-cut snap", async () => {
    const checkCalls: Array<Record<string, unknown>> = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.includes("/check-splice") && init?.method === "POST") {
          const body = JSON.parse(String(init.body)) as Record<string, unknown>;
          checkCalls.push(body);
          const isSnapped = Number(body.t_start) === 1.2 && Number(body.t_end) === 2.8;
          return new Response(
            JSON.stringify({
              passed: isSnapped,
              downgraded: !isSnapped,
              downgrade_reason: isSnapped ? null : "EEF displacement 6.80cm > 2.00cm",
              effective_splice_policy: isSnapped ? "cut_clean" : "mask_loss",
              opt_t_start: isSnapped ? null : 1.2,
              opt_t_end: isSnapped ? null : 2.8,
              metrics: {
                eef_pos_m: isSnapped ? 0.0056 : 0.068,
                eef_rot_rad: 0.02,
                hand_rad: 0.03,
                torso_rad: 0.01,
                other_arm_disp_m: 0.004,
                bimanual_coupled: false,
                visual_ssim: null,
                visual_l1: null,
              },
              compute_ms: 8.4,
              mode: "kinematics",
            }),
            { status: 200 },
          );
        }
        return new Response(JSON.stringify({}), { status: 200 });
      }),
    );

    const seg = makeQcSegment({
      seg_id: "seg_cut_1",
      t_start: 1.0,
      t_end: 3.0,
      splice_policy: "cut_clean",
      mask_side: null,
      training_role: "badcase",
    });
    const onSeek = vi.fn();
    const onUpdateSegment = vi.fn();

    const { rerender } = render(
      <SegmentInspector
        episodeId="demo_ds/ep_001"
        taskId="pick_and_place"
        durationS={10.0}
        currentTime={1.5}
        isLead={true}
        primaryCamera="head_rgbd"
        authToken="lead-tok"
        selectedSegment={seg}
        selectedStage={null}
        onSeek={onSeek}
        onUpdateSegment={onUpdateSegment}
        onDeleteSegment={vi.fn()}
        onUpdateStage={vi.fn()}
        onDeleteStage={vi.fn()}
      />,
    );

    // Side-by-side start & end thumbnails with frameUrl + click to seek + lightbox
    const startThumb = screen.getByTestId("segment-start-thumb");
    const endThumb = screen.getByTestId("segment-end-thumb");
    expect(startThumb.querySelector("img")?.getAttribute("src")).toContain("t_s=1.0000");
    expect(endThumb.querySelector("img")?.getAttribute("src")).toContain("t_s=3.0000");
    fireEvent.click(startThumb);
    expect(onSeek).toHaveBeenCalledWith(1.0);
    fireEvent.click(endThumb);
    expect(onSeek).toHaveBeenCalledWith(3.0);

    fireEvent.click(screen.getByTestId("open-frame-lightbox"));
    expect(screen.getByTestId("frame-compare-lightbox")).toBeInTheDocument();

    // Debounced 250ms live checkSplice returns jump warning + opt-cut snap button
    await waitFor(
      () => expect(screen.getByTestId("live-splice-badge")).toHaveTextContent("Jump exceeds threshold"),
      { timeout: 1500 },
    );
    expect(screen.getByTestId("live-splice-badge")).toHaveTextContent("6.80cm");
    expect(checkCalls).toHaveLength(1);

    // Click 1-click opt-cut snap button
    fireEvent.click(screen.getByTestId("snap-opt-cut-btn"));
    expect(onUpdateSegment).toHaveBeenCalledWith(
      expect.objectContaining({ t_start: 1.2, t_end: 2.8 }),
    );

    // Re-render with snapped bounds -> live check refreshes to green pass
    rerender(
      <SegmentInspector
        episodeId="demo_ds/ep_001"
        taskId="pick_and_place"
        durationS={10.0}
        currentTime={1.5}
        isLead={true}
        primaryCamera="head_rgbd"
        authToken="lead-tok"
        selectedSegment={{ ...seg, t_start: 1.2, t_end: 2.8 }}
        selectedStage={null}
        onSeek={onSeek}
        onUpdateSegment={onUpdateSegment}
        onDeleteSegment={vi.fn()}
        onUpdateStage={vi.fn()}
        onDeleteStage={vi.fn()}
      />,
    );

    await waitFor(
      () => expect(screen.getByTestId("live-splice-badge")).toHaveTextContent("Seamless cut OK"),
      { timeout: 1500 },
    );
    expect(screen.getByTestId("live-splice-badge")).toHaveTextContent("0.56cm");
  });

  it("4. ReviewStudioPage: full shortcuts suite, ShortcutsModal (?), input shielding, unsaved guard & draggable split resizer", async () => {
    const ep = makeResolvedEpisode({
      qc_segments: [makeQcSegment({ seg_id: "seg_1", t_start: 1.0, t_end: 3.0 })],
    });
    let draftSaves = 0;

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.endsWith("/reviews") && init?.method === "POST") {
          draftSaves += 1;
          return new Response(JSON.stringify(makeReviewArtifact({ rev_id: "rev_s1", action: "save_draft" })), {
            status: 200,
          });
        }
        if (url.endsWith("/reviews")) {
          return new Response(JSON.stringify({ items: [] }), { status: 200 });
        }
        if (url.includes("/api/settings")) {
          return new Response(JSON.stringify({ min_watch_coverage: 0.95, batch_vlm_concurrency: 4 }), {
            status: 200,
          });
        }
        if (url.includes("/api/episodes/demo_ds/ep_001")) {
          return new Response(JSON.stringify(ep), { status: 200 });
        }
        return new Response(JSON.stringify({ schema_version: "1.0", version: 1, categories: [], labels: [] }), {
          status: 200,
        });
      }),
    );

    render(
      <AuthProvider initialAuth={{ token: "rev-tok", username: "rev_1", role: "reviewer" }}>
        <ToastProvider>
          <MemoryRouter initialEntries={["/review/demo_ds/ep_001"]}>
            <Routes>
              <Route path="/" element={<div data-testid="queue-home">Queue</div>} />
              <Route path="/review/:dataset/:stem" element={<ReviewStudioPage />} />
            </Routes>
          </MemoryRouter>
        </ToastProvider>
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("review-studio-page")).toBeInTheDocument());

    // ? toggles ShortcutsModal; Escape closes it
    fireEvent.keyDown(window, { key: "?" });
    expect(screen.getByTestId("shortcuts-modal")).toBeInTheDocument();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByTestId("shortcuts-modal")).toBeNull();

    // Shortcuts ignored while typing in notes <input>
    const notesInput = screen.getByPlaceholderText("Review notes (notes)...");
    fireEvent.keyDown(notesInput, { key: "?" });
    expect(screen.queryByTestId("shortcuts-modal")).toBeNull();
    fireEvent.keyDown(notesInput, { key: "n" });
    expect(screen.getByTestId("undo-btn")).toHaveTextContent("Undo (0)");

    // L (+0.5s), Shift+L (+2.0s) -> currentTime becomes 2.50s
    fireEvent.keyDown(window, { key: "l" });
    fireEvent.keyDown(window, { key: "L", shiftKey: true });
    expect(screen.getByTestId("studio-player")).toHaveTextContent("2.50s");

    // K toggles cut-skip; R toggles segment loop
    fireEvent.keyDown(window, { key: "k" });
    expect(screen.getByTestId("toggle-cut-skip").className).toContain("active");
    fireEvent.keyDown(window, { key: "r" });
    expect(screen.getByTestId("toggle-segment-loop").className).toContain("active");

    // 1/2/3 switches policy of selected segment; [ sets t_start to currentTime
    fireEvent.keyDown(window, { key: "1" });
    expect((screen.getByLabelText("Splice Policy") as HTMLSelectElement).value).toBe("keep_positive");
    fireEvent.keyDown(window, { key: "3" });
    expect((screen.getByLabelText("Splice Policy") as HTMLSelectElement).value).toBe("cut_clean");

    // S splits macro stage M1 [0,5] at t=2.5s
    fireEvent.keyDown(window, { key: "s" });
    expect(screen.getByTestId("track-macro-stages").querySelectorAll(".macro-block")).toHaveLength(3);

    // N creates new QC segment; Delete removes it; Ctrl+Z restores it
    fireEvent.keyDown(window, { key: "n" });
    expect(screen.getByTestId("track-qc-segments").querySelectorAll(".qc-block")).toHaveLength(2);
    fireEvent.keyDown(window, { key: "Delete" });
    expect(screen.getByTestId("track-qc-segments").querySelectorAll(".qc-block")).toHaveLength(1);
    fireEvent.keyDown(window, { key: "z", ctrlKey: true });
    expect(screen.getByTestId("track-qc-segments").querySelectorAll(".qc-block")).toHaveLength(2);

    // Dirty indicator + beforeunload + confirm guard on navigation
    expect(screen.getByTestId("unsaved-indicator")).toBeInTheDocument();
    const unloadEvt = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(unloadEvt);
    expect(unloadEvt.defaultPrevented).toBe(true);

    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(false);
    fireEvent.click(screen.getByTestId("studio-back-link"));
    expect(confirmSpy).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId("review-studio-page")).toBeInTheDocument();

    // Ctrl+S saves draft and clears dirty flag
    fireEvent.keyDown(window, { key: "s", ctrlKey: true });
    await waitFor(() => expect(draftSaves).toBe(1));
    expect(screen.queryByTestId("unsaved-indicator")).toBeNull();

    // Split resizer drags within 50%-75% and persists to localStorage
    const grid = screen.getByTestId("studio-split-grid");
    mockRect(grid, 1000, 0);
    const resizer = screen.getByTestId("studio-split-resizer");
    fireEvent.pointerDown(resizer, { button: 0, clientX: 610 });
    fireEvent.pointerMove(window, { clientX: 700 });
    fireEvent.pointerUp(window, { clientX: 700 });
    expect(window.localStorage.getItem("robo_vlm_qc.studio_split_pct")).toBe("70");
  });

  it("5. ReviewStudioPage: dynamic min_watch_coverage from /api/settings + auto-advance to next vlm_ready on Reviewer submit", async () => {
    const ep1 = makeResolvedEpisode({ episode_id: "demo_ds/ep_001", status: "claimed", claimed_by: "rev_1" });
    const ep2 = makeResolvedEpisode({ episode_id: "demo_ds/ep_002", status: "claimed", claimed_by: "rev_1" });
    let claimedEp2 = false;

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.includes("/api/settings")) {
          return new Response(JSON.stringify({ min_watch_coverage: 0.4, batch_vlm_concurrency: 4 }), {
            status: 200,
          });
        }
        if (url.includes("/api/episodes/demo_ds/ep_002/claim") && init?.method === "POST") {
          claimedEp2 = true;
          return new Response(
            JSON.stringify({
              episode_id: "demo_ds/ep_002",
              claimed_by: "rev_1",
              claim_expires_at: 9999999999,
              status: "claimed",
            }),
            { status: 200 },
          );
        }
        if (url.endsWith("/reviews") && init?.method === "POST") {
          return new Response(JSON.stringify(makeReviewArtifact({ action: "submit_review" })), { status: 200 });
        }
        if (url.endsWith("/reviews")) {
          return new Response(JSON.stringify({ items: [] }), { status: 200 });
        }
        if (url.includes("/api/episodes/demo_ds/ep_001")) {
          return new Response(JSON.stringify(ep1), { status: 200 });
        }
        if (url.includes("/api/episodes/demo_ds/ep_002")) {
          return new Response(JSON.stringify(ep2), { status: 200 });
        }
        if (url.includes("/api/episodes?status=vlm_ready")) {
          return new Response(
            JSON.stringify({
              items: [makeEpisodeListItem({ episode_id: "demo_ds/ep_002", status: "vlm_ready" })],
              total: 1,
            }),
            { status: 200 },
          );
        }
        return new Response(JSON.stringify({ schema_version: "1.0", version: 1, categories: [], labels: [] }), {
          status: 200,
        });
      }),
    );

    const { container } = render(
      <AuthProvider initialAuth={{ token: "rev-tok", username: "rev_1", role: "reviewer" }}>
        <MemoryRouter initialEntries={["/review/demo_ds/ep_001"]}>
          <LocationEcho />
          <Routes>
            <Route path="/review/:dataset/:stem" element={<ReviewStudioPage />} />
          </Routes>
        </MemoryRouter>
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("watch-coverage-badge")).toHaveTextContent("40%"));
    expect(screen.getByTestId("action-submit-review")).toBeDisabled();

    // Watch 4.5s out of 10s (45% >= 40% dynamic setting, even though < 95%)
    const video = container.querySelector("video") as HTMLVideoElement;
    for (const t of [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]) {
      Object.defineProperty(video, "currentTime", { value: t, configurable: true });
      fireEvent.timeUpdate(video);
    }

    await waitFor(() => expect(screen.getByTestId("action-submit-review")).toBeEnabled());
    expect((screen.getByTestId("auto-advance-toggle") as HTMLInputElement).checked).toBe(true);

    // Submit -> auto-claims ep_002 and navigates
    fireEvent.click(screen.getByTestId("action-submit-review"));
    await waitFor(() => expect(claimedEp2).toBe(true));
    await waitFor(() =>
      expect(screen.getByTestId("location-echo")).toHaveTextContent("/review/demo_ds/ep_002"),
    );
  });

  it("6. ReviewStudioPage: URL deep-link (?t=&seg=&tab=diff), Lead Multi-Run switcher (read-only compare), and Single-Episode Local Re-Run VLM", async () => {
    const seg1 = makeQcSegment({ seg_id: "seg_1", t_start: 1.0, t_end: 2.0 });
    const seg2 = makeQcSegment({ seg_id: "seg_2", t_start: 3.5, t_end: 5.0 });
    const epCurrent = makeResolvedEpisode({
      active_run_id: "run_new",
      qc_segments: [seg1, seg2],
    });
    const epOldRun = makeResolvedEpisode({
      active_run_id: "run_old",
      selected_run_id: "run_old",
      is_historical_run_view: true,
      qc_segments: [makeQcSegment({ seg_id: "seg_old", t_start: 0.5, t_end: 1.5 })],
    });

    const jobBodies: Array<Record<string, unknown>> = [];
    const episodeUrls: string[] = [];

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.endsWith("/api/jobs") && init?.method === "POST") {
          const body = JSON.parse(String(init.body)) as Record<string, unknown>;
          jobBodies.push(body);
          return new Response(
            JSON.stringify(
              makeJobStatus({
                job_id: `job_${jobBodies.length}`,
                job_type: "batch_vlm",
                status: "queued",
                progress_done: 0,
                progress_total: 1,
              }),
            ),
            { status: 201 },
          );
        }
        if (url.includes("/api/jobs")) {
          return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200 });
        }
        if (url.endsWith("/runs")) {
          return new Response(
            JSON.stringify({
              episode_id: "demo_ds/ep_001",
              active_run_id: "run_new",
              runs: [
                {
                  run_id: "run_new",
                  created_at: "2026-11-21T10:00:00Z",
                  prompt_version: "v2.1",
                  label_version: "1",
                  few_shot_pack_id: null,
                  tokens_total: 12000,
                  is_eval_sandbox: false,
                },
                {
                  run_id: "run_old",
                  created_at: "2026-11-20T10:00:00Z",
                  prompt_version: "v1.0",
                  label_version: "1",
                  few_shot_pack_id: null,
                  tokens_total: 15000,
                  is_eval_sandbox: false,
                },
              ],
            }),
            { status: 200 },
          );
        }
        if (url.endsWith("/reviews")) {
          return new Response(JSON.stringify({ items: [] }), { status: 200 });
        }
        if (url.includes("/api/episodes/demo_ds/ep_001")) {
          episodeUrls.push(url);
          if (url.includes("run_id=run_old")) {
            return new Response(JSON.stringify(epOldRun), { status: 200 });
          }
          return new Response(JSON.stringify(epCurrent), { status: 200 });
        }
        if (url.includes("/api/settings")) {
          return new Response(JSON.stringify({ min_watch_coverage: 0.95, batch_vlm_concurrency: 4 }), {
            status: 200,
          });
        }
        return new Response(JSON.stringify({ schema_version: "1.0", version: 1, categories: [], labels: [] }), {
          status: 200,
        });
      }),
    );

    render(
      <AuthProvider initialAuth={{ token: "lead-tok", username: "lead_1", role: "lead" }}>
        <ToastProvider>
          <JobProvider>
            <MemoryRouter initialEntries={["/review/demo_ds/ep_001?t=3.5&seg=seg_2&tab=diff"]}>
              <Routes>
                <Route path="/review/:dataset/:stem" element={<ReviewStudioPage />} />
              </Routes>
            </MemoryRouter>
          </JobProvider>
        </ToastProvider>
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("review-studio-page")).toBeInTheDocument());

    // Deep-link automatically seeks to 3.50s, selects seg_2, opens Diff tab, and turns on ghost track
    expect(screen.getByTestId("studio-player")).toHaveTextContent("3.50s");
    expect(screen.getByTestId("diff-overlay")).toBeInTheDocument();
    expect(screen.getByTestId("track-ghost-vlm")).toBeInTheDocument();
    expect(screen.getByTestId("qc-seg-seg_2").className).toContain("selected");

    // Multi-Run Version Switcher loads historical run_old in read-only compare mode
    await waitFor(() =>
      expect(screen.getByTestId("run-version-select").querySelectorAll("option")).toHaveLength(3),
    );
    await act(async () => {
      fireEvent.change(screen.getByTestId("run-version-select"), { target: { value: "run_old" } });
    });

    await waitFor(() => expect(screen.getByTestId("historical-run-banner")).toBeInTheDocument());
    expect(episodeUrls.some((u) => u.includes("run_id=run_old"))).toBe(true);
    expect(screen.getByTestId("action-approve")).toBeDisabled();
    expect(screen.getByTestId("action-freeze-gt")).toBeDisabled();
    expect(screen.getByTestId("action-save-draft")).toBeDisabled();

    // Exit historical compare mode returns to editable view
    await act(async () => {
      fireEvent.click(screen.getByTestId("exit-historical-run-btn"));
    });
    await waitFor(() => expect(screen.queryByTestId("historical-run-banner")).toBeNull());
    expect(screen.getByTestId("action-approve")).toBeEnabled();

    // Single-Episode Local Re-Run VLM: S3/S4 fast rerun + S1-S5 full rerun via split menu
    await act(async () => {
      fireEvent.click(screen.getByTestId("rerun-vlm-s3s4-btn"));
    });
    await waitFor(() => expect(jobBodies).toHaveLength(1));
    expect(jobBodies[0]).toEqual(
      expect.objectContaining({
        job_type: "batch_vlm",
        episode_ids: ["demo_ds/ep_001"],
        stage: "s3_s4",
        fresh: false,
      }),
    );

    // Verify floating toast replaced static info-banner (no static .info-banner element)
    await act(async () => {
      expect(document.querySelector(".info-banner")).toBeNull();
    });
  });

  it("7. ReviewStudioPage: Lead split menu triggers full S1-S5 local VLM rerun", async () => {
    const ep = makeResolvedEpisode();
    const jobBodies: Array<Record<string, unknown>> = [];

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.endsWith("/api/jobs") && init?.method === "POST") {
          const body = JSON.parse(String(init.body)) as Record<string, unknown>;
          jobBodies.push(body);
          return new Response(
            JSON.stringify(
              makeJobStatus({
                job_id: "job_full_1",
                job_type: "batch_vlm",
                status: "completed",
                progress_done: 1,
                progress_total: 1,
              }),
            ),
            { status: 201 },
          );
        }
        if (url.endsWith("/runs")) {
          return new Response(JSON.stringify({ episode_id: ep.episode_id, active_run_id: "run_001", runs: [] }), {
            status: 200,
          });
        }
        if (url.endsWith("/reviews")) {
          return new Response(JSON.stringify({ items: [] }), { status: 200 });
        }
        if (url.includes("/api/episodes/demo_ds/ep_001")) {
          return new Response(JSON.stringify(ep), { status: 200 });
        }
        return new Response(JSON.stringify({ schema_version: "1.0", version: 1, categories: [], labels: [] }), {
          status: 200,
        });
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

    await waitFor(() => expect(screen.getByTestId("review-studio-page")).toBeInTheDocument());
    fireEvent.click(screen.getByTestId("rerun-vlm-menu-toggle"));
    await act(async () => {
      fireEvent.click(screen.getByTestId("rerun-vlm-all-btn"));
    });

    await waitFor(() => expect(jobBodies).toHaveLength(1));
    expect(jobBodies[0]).toEqual(
      expect.objectContaining({
        job_type: "batch_vlm",
        episode_ids: ["demo_ds/ep_001"],
        stage: "s1_s5",
        fresh: true,
      }),
    );
    // Floating standalone toast renders without static info-banner
    expect(screen.getByTestId("studio-info-banner")).toHaveTextContent("job_full_1");
    expect(screen.getByTestId("refresh-after-rerun-btn")).toBeInTheDocument();
  });

  it("8. ReviewStudioPage: inline cancel and external SSE cancel/fail of single-episode VLM rerun reset rerunning, show toast, and refresh status badge", async () => {
    const sseInstances: Array<{ onmessage: ((evt: MessageEvent) => void) | null; close: () => void }> = [];
    class MockEventSource {
      url: string;
      onopen: (() => void) | null = null;
      onmessage: ((evt: MessageEvent) => void) | null = null;
      onerror: (() => void) | null = null;
      close = vi.fn();
      constructor(url: string) {
        this.url = url;
        sseInstances.push(this);
      }
    }
    vi.stubGlobal("EventSource", MockEventSource);

    let epStatus: "vlm_running" | "vlm_ready" = "vlm_ready";
    let jobSeq = 0;
    let cancelCalls = 0;
    let episodeFetches = 0;

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        const method = init?.method ?? "GET";
        if (url.endsWith("/api/jobs") && method === "POST") {
          jobSeq += 1;
          epStatus = "vlm_running";
          return new Response(
            JSON.stringify(
              makeJobStatus({
                job_id: `job_studio_${jobSeq}`,
                job_type: "batch_vlm",
                status: "running",
                progress_done: 0,
                progress_total: 1,
                params: { episode_ids: ["demo_ds/ep_001"] },
              }),
            ),
            { status: 201 },
          );
        }
        if (url.includes("/cancel") && method === "POST") {
          cancelCalls += 1;
          epStatus = "vlm_ready";
          return new Response(
            JSON.stringify(
              makeJobStatus({
                job_id: `job_studio_${jobSeq}`,
                job_type: "batch_vlm",
                status: "cancelled",
                progress_done: 0,
                progress_total: 1,
                params: { episode_ids: ["demo_ds/ep_001"] },
              }),
            ),
            { status: 200 },
          );
        }
        if (url.startsWith("/api/jobs")) {
          return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200 });
        }
        if (url.endsWith("/runs")) {
          return new Response(
            JSON.stringify({ episode_id: "demo_ds/ep_001", active_run_id: "run_001", runs: [] }),
            { status: 200 },
          );
        }
        if (url.endsWith("/reviews")) {
          return new Response(JSON.stringify({ items: [] }), { status: 200 });
        }
        if (url.includes("/api/episodes/demo_ds/ep_001")) {
          episodeFetches += 1;
          return new Response(
            JSON.stringify(makeResolvedEpisode({ episode_id: "demo_ds/ep_001", status: epStatus })),
            { status: 200 },
          );
        }
        return new Response(JSON.stringify({ schema_version: "1.0", version: 1, categories: [], labels: [] }), {
          status: 200,
        });
      }),
    );

    render(
      <AuthProvider initialAuth={{ token: "lead-tok", username: "lead_1", role: "lead" }}>
        <ToastProvider>
          <JobProvider>
            <MemoryRouter initialEntries={["/review/demo_ds/ep_001"]}>
              <Routes>
                <Route path="/review/:dataset/:stem" element={<ReviewStudioPage />} />
              </Routes>
            </MemoryRouter>
          </JobProvider>
        </ToastProvider>
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("review-studio-page")).toBeInTheDocument());

    // 1. Trigger rerun and cancel directly via inline cancel button
    fireEvent.click(screen.getByTestId("rerun-vlm-s3s4-btn"));
    const inlineCancelBtn = await screen.findByTestId("cancel-rerun-vlm-btn");
    expect(screen.getByTestId("rerun-vlm-s3s4-btn")).toBeDisabled();

    const fetchesBeforeInlineCancel = episodeFetches;
    fireEvent.click(inlineCancelBtn);

    await waitFor(() => expect(cancelCalls).toBe(1));
    await waitFor(() => expect(episodeFetches).toBeGreaterThan(fetchesBeforeInlineCancel));
    await waitFor(() => expect(screen.getByTestId("rerun-vlm-s3s4-btn")).toBeEnabled());
    expect(screen.queryByTestId("cancel-rerun-vlm-btn")).toBeNull();

    // 2. Trigger a second rerun and cancel externally via SSE job_update -> resets rerunning + reloads episode
    fireEvent.click(screen.getByTestId("rerun-vlm-s3s4-btn"));
    await screen.findByTestId("cancel-rerun-vlm-btn");
    epStatus = "vlm_ready";
    const fetchesBeforeExternalCancel = episodeFetches;

    act(() => {
      for (const es of sseInstances) {
        es.onmessage?.({
          data: JSON.stringify({
            type: "job_update",
            job: makeJobStatus({
              job_id: "job_studio_2",
              job_type: "batch_vlm",
              status: "cancelled",
              progress_done: 0,
              progress_total: 1,
              params: { episode_ids: ["demo_ds/ep_001"] },
            }),
          }),
        } as MessageEvent);
      }
    });

    await waitFor(() => expect(episodeFetches).toBeGreaterThan(fetchesBeforeExternalCancel));
    await waitFor(() => expect(screen.getByTestId("rerun-vlm-s3s4-btn")).toBeEnabled());
    expect(screen.getByTestId("review-studio-page")).toHaveTextContent("Ready for Review");
  });

  it("9. ReviewStudioPage: Lead split menu triggers S1-only rerun (s1_grounding, fresh=false) and updates grounding badge from v1 Sparse to v2 Dense on refresh", async () => {
    const epV1 = makeResolvedEpisode({
      episode_id: "demo_ds/ep_001",
      active_run_id: "run_v1",
    });
    const epV2 = makeResolvedEpisode({
      episode_id: "demo_ds/ep_001",
      active_run_id: "run_v2",
      grounding: {
        schema_version: 2,
        camera_name: "head_rgbd",
        ref_size: [1280, 720],
        keyframes_s: [0, 2.5, 5.0],
        dense_times_s: [0.0, 0.1, 0.2],
        object_tracks_2d: {
          block: {
            object_id: "block",
            name: "Target Block",
            tag_pos: "top",
            is_static_container: false,
            tracking_mode: "bidir_flow_ncc",
            keyframes: epV1.grounding!.object_tracks_2d.block.keyframes,
            dense_boxes_norm_1000: [
              [200, 300, 350, 500],
              [210, 300, 360, 500],
              [220, 300, 370, 500],
            ],
            dense_flags: [2, 1, 1],
            dense_conf_pct: [100, 96, 95],
          },
        },
      },
    });

    let currentEp = epV1;
    const jobBodies: Array<Record<string, unknown>> = [];

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.endsWith("/api/jobs") && init?.method === "POST") {
          const body = JSON.parse(String(init.body)) as Record<string, unknown>;
          jobBodies.push(body);
          currentEp = epV2;
          return new Response(
            JSON.stringify(
              makeJobStatus({
                job_id: "job_s1_only_1",
                job_type: "batch_vlm",
                status: "completed",
                progress_done: 1,
                progress_total: 1,
                params: body,
              }),
            ),
            { status: 201 },
          );
        }
        if (url.endsWith("/runs")) {
          return new Response(
            JSON.stringify({
              episode_id: "demo_ds/ep_001",
              active_run_id: currentEp.active_run_id,
              runs: [],
            }),
            { status: 200 },
          );
        }
        if (url.endsWith("/reviews")) {
          return new Response(JSON.stringify({ items: [] }), { status: 200 });
        }
        if (url.includes("/api/episodes/demo_ds/ep_001")) {
          return new Response(JSON.stringify(currentEp), { status: 200 });
        }
        if (url.includes("/api/settings")) {
          return new Response(JSON.stringify({ min_watch_coverage: 0.95, batch_vlm_concurrency: 4 }), {
            status: 200,
          });
        }
        return new Response(JSON.stringify({ schema_version: "1.0", version: 1, categories: [], labels: [] }), {
          status: 200,
        });
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

    await waitFor(() => expect(screen.getByTestId("review-studio-page")).toBeInTheDocument());
    expect(screen.getByTestId("grounding-version-badge")).toHaveTextContent("v1 Sparse");

    fireEvent.click(screen.getByTestId("rerun-vlm-menu-toggle"));
    const s1Btn = screen.getByTestId("rerun-vlm-s1-btn");
    expect(s1Btn).toHaveTextContent("Rerun S1 Only (Refresh 2D Boxes, Reuse S2-S5 Cache)");

    await act(async () => {
      fireEvent.click(s1Btn);
    });

    await waitFor(() => expect(jobBodies).toHaveLength(1));
    expect(jobBodies[0]).toEqual({
      job_type: "batch_vlm",
      episode_ids: ["demo_ds/ep_001"],
      stage: "s1_grounding",
      fresh: false,
    });

    const refreshBtn = await screen.findByTestId("refresh-after-rerun-btn");
    await act(async () => {
      fireEvent.click(refreshBtn);
    });

    await waitFor(() =>
      expect(screen.getByTestId("grounding-version-badge")).toHaveTextContent("v2 Dense"),
    );
  });

  it("10. ReviewStudioPage: Infer Subtasks from QC (undoable), Align Stages to QC Cuts, Rerun S1 with Custom Objects, SuccessPanel Now stamp, and Lead Open in Export Studio", async () => {
    const ep = makeResolvedEpisode({
      episode_id: "demo_ds/ep_001",
      status: "vlm_ready",
      qc_segments: [
        makeQcSegment({
          seg_id: "seg_cut",
          t_start: 4.85,
          t_end: 5.5,
          splice_policy: "cut_clean",
          mask_side: null,
        }),
      ],
      macro_stages: [
        {
          stage_id: "M1",
          t_start: 0.0,
          t_end: 5.0,
          label: "Pick Object",
          label_en: "pick_object",
          summary: "Grasp block",
          color_rgb: [59, 130, 246],
        },
        {
          stage_id: "M2",
          t_start: 5.0,
          t_end: 10.0,
          label: "Place in Bin",
          label_en: "place_in_bin",
          summary: "Place block",
          color_rgb: [16, 185, 129],
        },
      ],
    });

    const inferBodies: Array<Record<string, unknown>> = [];
    const jobBodies: Array<Record<string, unknown>> = [];

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.endsWith("/infer-subtasks") && init?.method === "POST") {
          const body = JSON.parse(String(init.body)) as Record<string, unknown>;
          inferBodies.push(body);
          return new Response(
            JSON.stringify({
              episode_id: "demo_ds/ep_001",
              used_grounding: true,
              used_segments_count: 1,
              macro_stages: [
                {
                  stage_id: "S1",
                  t_start: 0.0,
                  t_end: 4.85,
                  label: "Grasp Green Mug",
                  label_en: "grasp_green_mug",
                  summary: "Pick up green mug",
                  instruction: "Pick up the green mug from the counter",
                  color_rgb: [59, 130, 246],
                },
                {
                  stage_id: "S2",
                  t_start: 5.5,
                  t_end: 10.0,
                  label: "Place in Rack",
                  label_en: "place_in_rack",
                  summary: "Place mug into rack",
                  instruction: "Place the green mug upright in the dish rack",
                  color_rgb: [16, 185, 129],
                },
              ],
            }),
            { status: 200 },
          );
        }
        if (url.endsWith("/api/jobs") && init?.method === "POST") {
          const body = JSON.parse(String(init.body)) as Record<string, unknown>;
          jobBodies.push(body);
          return new Response(
            JSON.stringify(
              makeJobStatus({
                job_id: "job_custom_s1",
                job_type: "batch_vlm",
                status: "queued",
                progress_done: 0,
                progress_total: 1,
                params: body,
              }),
            ),
            { status: 201 },
          );
        }
        if (url.endsWith("/runs")) {
          return new Response(
            JSON.stringify({ episode_id: "demo_ds/ep_001", active_run_id: "run_001", runs: [] }),
            { status: 200 },
          );
        }
        if (url.endsWith("/reviews")) {
          return new Response(JSON.stringify({ items: [] }), { status: 200 });
        }
        if (url.includes("/api/episodes/demo_ds/ep_001")) {
          return new Response(JSON.stringify(ep), { status: 200 });
        }
        if (url.includes("/api/settings")) {
          return new Response(JSON.stringify({ min_watch_coverage: 0.95, batch_vlm_concurrency: 4 }), {
            status: 200,
          });
        }
        return new Response(JSON.stringify({ schema_version: "1.0", version: 1, categories: [], labels: [] }), {
          status: 200,
        });
      }),
    );

    render(
      <AuthProvider initialAuth={{ token: "lead-tok", username: "lead_1", role: "lead" }}>
        <MemoryRouter initialEntries={["/review/demo_ds/ep_001"]}>
          <LocationEcho />
          <Routes>
            <Route path="/review/:dataset/:stem" element={<ReviewStudioPage />} />
            <Route path="/export" element={<div data-testid="export-studio-route">Export Studio</div>} />
          </Routes>
        </MemoryRouter>
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("review-studio-page")).toBeInTheDocument());

    // 10a. Align Stages to QC Cuts snaps M1.t_end & M2.t_start (5.0 -> 4.85, within 0.25s) and is Undoable
    fireEvent.click(screen.getByTestId("align-stages-qc-btn"));
    fireEvent.click(screen.getByTestId("macro-stage-M1"));
    expect((screen.getByLabelText("Stage end") as HTMLInputElement).value).toBe("4.85");
    fireEvent.click(screen.getByTestId("undo-btn"));
    expect((screen.getByLabelText("Stage end") as HTMLInputElement).value).toBe("5");

    // 10b. Infer Subtasks from QC POSTs current qc_segments & macro_stages, applies result into undoable history
    await act(async () => {
      fireEvent.click(screen.getByTestId("infer-subtasks-btn"));
    });
    await waitFor(() => expect(inferBodies).toHaveLength(1));
    expect(Array.isArray(inferBodies[0].qc_segments)).toBe(true);
    expect(screen.getByTestId("macro-stage-S1")).toBeInTheDocument();
    expect(screen.getByTestId("macro-stage-S2")).toBeInTheDocument();
    expect((screen.getByTestId("stage-instruction-input") as HTMLTextAreaElement).value).toBe(
      "Pick up the green mug from the counter",
    );

    // Ctrl+Z undoes inferred subtasks cleanly back to M1/M2
    fireEvent.click(screen.getByTestId("undo-btn"));
    expect(screen.getByTestId("macro-stage-M1")).toBeInTheDocument();
    expect(screen.queryByTestId("macro-stage-S1")).toBeNull();

    // 10c. Rerun S1 with Custom Objects modal submits custom_objects array to /api/jobs
    fireEvent.click(screen.getByTestId("rerun-vlm-menu-toggle"));
    fireEvent.click(screen.getByTestId("rerun-vlm-custom-objects-btn"));
    expect(screen.getByTestId("custom-objects-modal")).toBeInTheDocument();
    fireEvent.change(screen.getByTestId("custom-objects-input"), {
      target: { value: "green mug, dish rack" },
    });
    expect(screen.getByTestId("custom-objects-preview")).toHaveTextContent("green mug");
    expect(screen.getByTestId("custom-objects-preview")).toHaveTextContent("dish rack");
    await act(async () => {
      fireEvent.click(screen.getByTestId("submit-custom-objects-btn"));
    });
    await waitFor(() => expect(jobBodies).toHaveLength(1));
    expect(jobBodies[0]).toEqual({
      job_type: "batch_vlm",
      episode_ids: ["demo_ds/ep_001"],
      stage: "s1_grounding",
      fresh: false,
      custom_objects: ["green mug", "dish rack"],
    });

    // 10d. SuccessPanel receives live currentTime so clicking Now stamps current playhead
    fireEvent.click(screen.getByText("+0.5s"));
    fireEvent.click(screen.getByText("Success Verdict"));
    fireEvent.click(screen.getByTestId("criterion-now-0"));
    expect(screen.getByTestId("criterion-seek-0")).toHaveTextContent("0.50s");

    // 10e. Lead Export popover -> Open in Export Studio navigates with pre-filled query params
    // Note: clear unsaved confirm for navigation (useNavigate doesn't trigger anchor click guard)
    fireEvent.click(screen.getByTestId("studio-export-btn"));
    fireEvent.click(screen.getByTestId("export-open-studio-btn"));
    await waitFor(() => expect(screen.getByTestId("export-studio-route")).toBeInTheDocument());
    expect(screen.getByTestId("location-echo")).toHaveTextContent(
      "/export?episode_ids=demo_ds%2Fep_001&dataset=demo_ds&include_status=vlm_ready",
    );
  });
});
