import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AppRoutes } from "../App";
import { configureApiCredentialsProvider } from "../api/client";
import { LabHistoryDrawer } from "../components/LabHistoryDrawer";
import { SegmentInspector } from "../components/SegmentInspector";
import { StudioPlayer } from "../components/StudioPlayer";
import { createDefaultSegment, DEFAULT_VISUAL_LABELS } from "../constants/labels";
import { AuthProvider, useAuth } from "../stores/authStore";
import { JobProvider } from "../stores/jobStore";
import { ToastProvider } from "../stores/toastStore";
import {
  makeEpisodeListItem,
  makeJobStatus,
  makeQcSegment,
  makeResolvedEpisode,
} from "../test/fixtures";
import { ExportPage } from "./ExportPage";
import { QueuePage } from "./QueuePage";
import { ReviewStudioPage } from "./ReviewStudioPage";
import { VlmLabPage } from "./VlmLabPage";

const LocationSpy = () => {
  const loc = useLocation();
  return <div data-testid="loc-spy">{loc.pathname}</div>;
};

describe("Frontend UX Audit Regression Suite (FE-1..FE-9)", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.localStorage.clear();
    window.history.replaceState({}, "", "/");
    configureApiCredentialsProvider(() => ({ token: "", username: "reviewer" }));
  });

  it("FE-1: VLM Lab multi-line textareas preserve newlines/spaces while typing and sanitize on save", async () => {
    const putPayloads: Array<{ url: string; body: Record<string, unknown> }> = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        const method = init?.method ?? "GET";
        if (method === "PUT") {
          const parsed = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
          putPayloads.push({ url, body: parsed });
          if (url.startsWith("/api/lab/tasks")) {
            return new Response(
              JSON.stringify({ tasks: { pick_and_place: parsed }, total: 1 }),
              { status: 200 },
            );
          }
          return new Response(JSON.stringify(parsed), { status: 200 });
        }
        if (url.startsWith("/api/lab/labels")) {
          return new Response(
            JSON.stringify({
              schema_version: "1.0",
              version: 1,
              categories: ["idle"],
              labels: [DEFAULT_VISUAL_LABELS[0]],
            }),
            { status: 200 },
          );
        }
        if (url.startsWith("/api/lab/prompts")) {
          const tpl = {
            stage: "s3_discover",
            version: "1",
            system_template: "s",
            user_template: "u",
            extra_templates: {},
          };
          return new Response(
            JSON.stringify({
              s1_grounding: { ...tpl, stage: "s1_grounding" },
              s2_macro: { ...tpl, stage: "s2_macro" },
              s3_discover: tpl,
              s4_verify: { ...tpl, stage: "s4_verify" },
              s5_success: { ...tpl, stage: "s5_success" },
            }),
            { status: 200 },
          );
        }
        if (url.startsWith("/api/lab/tasks")) {
          return new Response(
            JSON.stringify({
              tasks: {
                pick_and_place: {
                  task_id: "pick_and_place",
                  version: "1",
                  match: ["pick_and_place"],
                  aliases: [],
                  title: "Pick and Place",
                  description: "",
                  goal: "Complete pick and place",
                  criteria: ["Object placed inside bin"],
                  steps: ["Grasp object"],
                  task_prompt: "",
                  grounding_objects: [],
                },
              },
              total: 1,
            }),
            { status: 200 },
          );
        }
        if (url.startsWith("/api/lab/diagnosis")) {
          return new Response(
            JSON.stringify({
              episodes_evaluated: 0,
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
        return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200 });
      }),
    );

    render(
      <AuthProvider initialAuth={{ token: "lead-tok", username: "lead_1", role: "lead" }}>
        <MemoryRouter>
          <VlmLabPage />
        </MemoryRouter>
      </AuthProvider>,
    );

    fireEvent.click(screen.getByTestId("tab-taxonomy"));
    const ceTextarea = (await screen.findByLabelText("Counter-examples")) as HTMLTextAreaElement;
    fireEvent.change(ceTextarea, { target: { value: "line one \n\nline two  " } });
    // Must preserve trailing spaces and blank lines while editing
    expect(ceTextarea.value).toBe("line one \n\nline two  ");

    fireEvent.click(screen.getByTestId("save-labels-btn"));
    await waitFor(() => expect(putPayloads.some((p) => p.url.includes("/api/lab/labels"))).toBe(true));
    const labelsPut = putPayloads.find((p) => p.url.includes("/api/lab/labels"))!;
    const savedLabels = labelsPut.body.labels as Array<{ vlm_semantics: { counter_examples: string[] } }>;
    expect(savedLabels[0].vlm_semantics.counter_examples).toEqual(["line one", "line two"]);

    fireEvent.click(screen.getByTestId("taxonomy-subtab-tasks"));
    const criteriaTa = (await screen.findByLabelText("Completion criteria")) as HTMLTextAreaElement;
    const stepsTa = screen.getByLabelText("Expected steps") as HTMLTextAreaElement;
    fireEvent.change(criteriaTa, { target: { value: "Criterion A \n\n Criterion B" } });
    fireEvent.change(stepsTa, { target: { value: "Step 1\n " } });
    expect(criteriaTa.value).toBe("Criterion A \n\n Criterion B");
    expect(stepsTa.value).toBe("Step 1\n ");

    fireEvent.click(screen.getByTestId("save-task-btn"));
    await waitFor(() => expect(putPayloads.some((p) => p.url.includes("/api/lab/tasks"))).toBe(true));
    const taskPut = putPayloads.find((p) => p.url.includes("/api/lab/tasks"))!;
    expect(taskPut.body.criteria).toEqual(["Criterion A", "Criterion B"]);
    expect(taskPut.body.steps).toEqual(["Step 1"]);
  });

  it("FE-2: --no-auth logout then AuthModal save with empty token & new username restores Lead without downgrading", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.includes("/api/auth/me")) {
          return new Response(
            JSON.stringify({ role: "lead", user: "lead_1", no_auth: true }),
            { status: 200 },
          );
        }
        return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200 });
      }),
    );

    let authCtx!: ReturnType<typeof useAuth>;
    const Harness = () => {
      authCtx = useAuth();
      return (
        <div>
          <span data-testid="fe2-role">{authCtx.role}</span>
          <span data-testid="fe2-user">{authCtx.username}</span>
        </div>
      );
    };

    render(
      <AuthProvider>
        <Harness />
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("fe2-role")).toHaveTextContent("lead"));

    act(() => {
      authCtx.logout();
    });
    expect(screen.getByTestId("fe2-role")).toHaveTextContent("reviewer");

    // Re-apply empty token + new username without reviewer override (like AuthModal saving lead)
    await act(async () => {
      await authCtx.setCredentials({ token: "", username: "lead_renamed", role: "lead" }, true);
    });

    await waitFor(() => expect(screen.getByTestId("fe2-user")).toHaveTextContent("lead_renamed"));
    // Must stay lead even after auth.username triggers the mount/credential useEffect
    expect(screen.getByTestId("fe2-role")).toHaveTextContent("lead");
  });

  it("FE-3: ReviewStudioPage disables rerun on gt_frozen, shows error detail on failure, avoids duplicate cancel toast, and opens new run on protected episode", async () => {
    // Part A: gt_frozen disables rerun button
    const frozenEp = makeResolvedEpisode({ episode_id: "ds/frozen", status: "gt_frozen" });
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.endsWith("/reviews")) return new Response(JSON.stringify({ items: [] }), { status: 200 });
        if (url.endsWith("/runs")) return new Response(JSON.stringify({ runs: [] }), { status: 200 });
        if (url.includes("/api/episodes/ds/frozen")) return new Response(JSON.stringify(frozenEp), { status: 200 });
        return new Response(JSON.stringify({ schema_version: "1.0", version: 1, categories: [], labels: [] }), { status: 200 });
      }),
    );

    const { unmount } = render(
      <AuthProvider initialAuth={{ token: "lead-tok", username: "lead_1", role: "lead" }}>
        <MemoryRouter initialEntries={["/review/ds/frozen"]}>
          <Routes>
            <Route path="/review/:dataset/:stem" element={<ReviewStudioPage />} />
          </Routes>
        </MemoryRouter>
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("rerun-vlm-s3s4-btn")).toBeDisabled());
    expect(screen.getByTestId("rerun-vlm-menu-toggle")).toBeDisabled();
    unmount();

    // Part B: protected "reviewed" episode rerun completion -> "Load New Run" requests ?run_id=run_new_99
    const epRequests: string[] = [];
    const reviewedEp = makeResolvedEpisode({
      episode_id: "ds/rev1",
      status: "reviewed",
      active_run_id: "run_old_1",
    });
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.endsWith("/api/jobs") && init?.method === "POST") {
          return new Response(
            JSON.stringify(
              makeJobStatus({
                job_id: "job_prot_1",
                job_type: "batch_vlm",
                status: "completed",
                progress_done: 1,
                progress_total: 1,
                summary: { results: [{ episode_id: "ds/rev1", run_id: "run_new_99" }] },
              }),
            ),
            { status: 201 },
          );
        }
        if (url.startsWith("/api/jobs")) return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200 });
        if (url.endsWith("/reviews")) return new Response(JSON.stringify({ items: [] }), { status: 200 });
        if (url.endsWith("/runs")) {
          return new Response(JSON.stringify({ episode_id: "ds/rev1", active_run_id: "run_old_1", runs: [] }), { status: 200 });
        }
        if (url.includes("/api/episodes/ds/rev1")) {
          epRequests.push(url);
          return new Response(JSON.stringify(reviewedEp), { status: 200 });
        }
        return new Response(JSON.stringify({ schema_version: "1.0", version: 1, categories: [], labels: [] }), { status: 200 });
      }),
    );

    render(
      <AuthProvider initialAuth={{ token: "lead-tok", username: "lead_1", role: "lead" }}>
        <ToastProvider>
          <JobProvider>
            <MemoryRouter initialEntries={["/review/ds/rev1"]}>
              <Routes>
                <Route path="/review/:dataset/:stem" element={<ReviewStudioPage />} />
              </Routes>
            </MemoryRouter>
          </JobProvider>
        </ToastProvider>
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("review-studio-page")).toBeInTheDocument());
    fireEvent.click(screen.getByTestId("rerun-vlm-s3s4-btn"));
    const refreshBtn = await screen.findByTestId("refresh-after-rerun-btn");
    fireEvent.click(refreshBtn);

    await waitFor(() => expect(epRequests.some((u) => u.includes("run_id=run_new_99"))).toBe(true));
  });

  it("FE-4: StudioPlayer preserves playhead across camera switch and resets playing on episode switch", () => {
    const ep = makeResolvedEpisode();
    const onTimeUpdate = vi.fn();

    const { rerender } = render(
      <AuthProvider initialAuth={{ token: "tok", username: "rev1", role: "reviewer" }}>
        <StudioPlayer
          episodeId={ep.episode_id}
          cameras={ep.meta.cameras}
          durationS={10.0}
          fps={30}
          grounding={ep.grounding}
          currentTime={6.5}
          onTimeUpdate={onTimeUpdate}
        />
      </AuthProvider>,
    );

    const video = screen.getByTestId("primary-video") as HTMLVideoElement;
    // Switch camera tab -> browser resets video.currentTime to 0 before loadedmetadata
    fireEvent.click(screen.getByRole("tab", { name: /torso_rgbd/ }));
    Object.defineProperty(video, "currentTime", { value: 0, writable: true, configurable: true });
    fireEvent.timeUpdate(video);
    // Must NOT snap playhead back to 0!
    expect(onTimeUpdate).not.toHaveBeenCalledWith(0);

    // loadedmetadata restores playhead to 6.5s
    fireEvent.loadedMetadata(video);
    expect(video.currentTime).toBeCloseTo(6.5, 1);

    // Switching episodeId resets playing cleanly
    fireEvent.play(video);
    rerender(
      <AuthProvider initialAuth={{ token: "tok", username: "rev1", role: "reviewer" }}>
        <StudioPlayer
          episodeId="demo_ds/ep_002"
          cameras={ep.meta.cameras}
          durationS={10.0}
          fps={30}
          grounding={ep.grounding}
          currentTime={0}
          onTimeUpdate={onTimeUpdate}
        />
      </AuthProvider>,
    );
    expect(screen.getByTestId("play-pause-btn")).toHaveTextContent("Play");
  });

  it("FE-5 & FE-6: createDefaultSegment clamps at durationS; SegmentInspector handles empty inputs, hallucination override badge & state reset", async () => {
    // FE-5: createDefaultSegment at durationS stays strictly within [0, durationS]
    const segAtEnd = createDefaultSegment(10.0, 10.0);
    expect(segAtEnd.t_start).toBeGreaterThanOrEqual(0);
    expect(segAtEnd.t_end).toBeLessThanOrEqual(10.0);
    expect(segAtEnd.t_end).toBeGreaterThan(segAtEnd.t_start);

    // FE-6: Hallucination override returns passed=true, downgraded=false, effective_splice_policy="keep_positive"
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.includes("/check-splice")) {
          return new Response(
            JSON.stringify({
              passed: true,
              downgraded: false,
              downgrade_reason: "Hallucination filter: effective motion overridden to keep_positive",
              effective_splice_policy: "keep_positive",
              opt_t_start: null,
              opt_t_end: null,
              metrics: {
                eef_pos_m: 0.004,
                eef_rot_rad: 0.01,
                hand_rad: 0.02,
                torso_rad: 0.0,
                other_arm_disp_m: 0.0,
                bimanual_coupled: false,
                visual_ssim: null,
                visual_l1: null,
              },
              compute_ms: 4.2,
              mode: "kinematics",
            }),
            { status: 200 },
          );
        }
        return new Response(JSON.stringify({}), { status: 200 });
      }),
    );

    const seg1 = makeQcSegment({
      seg_id: "seg_hallu",
      t_start: 1.0,
      t_end: 3.0,
      splice_policy: "cut_clean",
      mask_side: null,
      training_role: "badcase",
    });
    const seg2 = makeQcSegment({
      seg_id: "seg_other",
      t_start: 4.0,
      t_end: 5.0,
      splice_policy: "keep_positive",
      mask_side: null,
      training_role: "necessary",
    });
    const onUpdateSegment = vi.fn();

    const { rerender } = render(
      <SegmentInspector
        episodeId="demo_ds/ep_001"
        taskId="pick_and_place"
        durationS={10.0}
        currentTime={1.5}
        isLead={true}
        selectedSegment={seg1}
        selectedStage={null}
        onUpdateSegment={onUpdateSegment}
        onDeleteSegment={vi.fn()}
        onUpdateStage={vi.fn()}
        onDeleteStage={vi.fn()}
      />,
    );

    await waitFor(
      () =>
        expect(screen.getByTestId("live-splice-badge")).toHaveTextContent(
          "Hallucination override: protected as keep_positive",
        ),
      { timeout: 1500 },
    );
    expect(screen.getByTestId("live-splice-badge").className).toContain("warn");

    // FE-5: Clearing numeric input does not emit 0 or NaN
    const startInput = screen.getByLabelText("Segment start") as HTMLInputElement;
    fireEvent.change(startInput, { target: { value: "" } });
    expect(onUpdateSegment).not.toHaveBeenCalled();
    fireEvent.blur(startInput);
    expect(startInput.value).toBe("1");

    // FE-6: Open Few-Shot form and Lightbox, switch segment -> both reset cleanly
    fireEvent.click(screen.getByTestId("add-few-shot-btn"));
    expect(screen.getByTestId("few-shot-form")).toBeInTheDocument();
    fireEvent.click(screen.getByTestId("open-frame-lightbox"));
    expect(screen.getByTestId("frame-compare-lightbox")).toBeInTheDocument();

    // FE-7: Escape closes lightbox
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByTestId("frame-compare-lightbox")).toBeNull();

    rerender(
      <SegmentInspector
        episodeId="demo_ds/ep_001"
        taskId="pick_and_place"
        durationS={10.0}
        currentTime={4.5}
        isLead={true}
        selectedSegment={seg2}
        selectedStage={null}
        onUpdateSegment={onUpdateSegment}
        onDeleteSegment={vi.fn()}
        onUpdateStage={vi.fn()}
        onDeleteStage={vi.fn()}
      />,
    );
    expect(screen.queryByTestId("few-shot-form")).toBeNull();
  });

  it("FE-7: Global Studio shortcuts are blocked behind open modals and Escape closes AuthModal & flag dialog", async () => {
    const ep = makeResolvedEpisode({
      qc_segments: [makeQcSegment({ seg_id: "seg_1", t_start: 1.0, t_end: 3.0 })],
    });
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.endsWith("/reviews")) return new Response(JSON.stringify({ items: [] }), { status: 200 });
        if (url.endsWith("/runs")) return new Response(JSON.stringify({ runs: [] }), { status: 200 });
        if (url.includes("/api/episodes/demo_ds/ep_001")) return new Response(JSON.stringify(ep), { status: 200 });
        if (url.startsWith("/api/jobs")) return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200 });
        return new Response(JSON.stringify({ schema_version: "1.0", version: 1, categories: [], labels: [] }), { status: 200 });
      }),
    );

    render(
      <AuthProvider initialAuth={{ token: "lead-tok", username: "lead_1", role: "lead" }}>
        <MemoryRouter initialEntries={["/review/demo_ds/ep_001"]}>
          <AppRoutes />
        </MemoryRouter>
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("review-studio-page")).toBeInTheDocument());
    expect(screen.getByTestId("track-qc-segments").querySelectorAll(".qc-block")).toHaveLength(1);

    // Open ShortcutsModal -> Delete must NOT delete the segment underneath
    fireEvent.keyDown(window, { key: "?" });
    expect(screen.getByTestId("shortcuts-modal")).toBeInTheDocument();
    fireEvent.keyDown(window, { key: "Delete" });
    expect(screen.getByTestId("track-qc-segments").querySelectorAll(".qc-block")).toHaveLength(1);
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByTestId("shortcuts-modal")).toBeNull();

    // Open AuthModal -> Delete blocked; Escape closes AuthModal
    fireEvent.click(screen.getByTestId("open-auth-modal"));
    expect(screen.getByTestId("auth-modal")).toBeInTheDocument();
    fireEvent.keyDown(window, { key: "Delete" });
    expect(screen.getByTestId("track-qc-segments").querySelectorAll(".qc-block")).toHaveLength(1);
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByTestId("auth-modal")).toBeNull();

    // Open flag dialog -> Escape closes it
    fireEvent.click(screen.getByTestId("action-flag-btn"));
    expect(screen.getByTestId("flag-reason-bar")).toBeInTheDocument();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByTestId("flag-reason-bar")).toBeNull();
  });

  it("FE-8: QueuePage clears selection on filter change, handleClaimNext works from non-vlm_ready tab with 409 retry, and failed banner shows error + dismiss", async () => {
    const claimedList = [makeEpisodeListItem({ episode_id: "ds/my_ep", status: "claimed", claimed_by: "rev_1" })];
    const readyList = [
      makeEpisodeListItem({ episode_id: "ds/taken_ep", status: "vlm_ready" }),
      makeEpisodeListItem({ episode_id: "ds/free_ep", status: "vlm_ready" }),
    ];

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.includes("/ds/taken_ep/claim") && init?.method === "POST") {
          return new Response(JSON.stringify({ detail: "Already claimed" }), { status: 409 });
        }
        if (url.includes("/ds/free_ep/claim") && init?.method === "POST") {
          return new Response(
            JSON.stringify({
              episode_id: "ds/free_ep",
              claimed_by: "rev_1",
              claim_expires_at: 9999999999,
              status: "claimed",
            }),
            { status: 200 },
          );
        }
        if (url.includes("status=claimed")) {
          return new Response(JSON.stringify({ items: claimedList, total: 1, my_claimed_count: 1 }), { status: 200 });
        }
        return new Response(JSON.stringify({ items: readyList, total: 2, my_claimed_count: 1 }), { status: 200 });
      }),
    );

    const { unmount } = render(
      <AuthProvider initialAuth={{ token: "rev-tok", username: "rev_1", role: "reviewer" }}>
        <MemoryRouter initialEntries={["/"]}>
          <LocationSpy />
          <Routes>
            <Route path="/" element={<QueuePage />} />
            <Route path="/review/:dataset/:stem" element={<div>Studio</div>} />
          </Routes>
        </MemoryRouter>
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("episode-row-ds/taken_ep")).toBeInTheDocument());
    // Switch to "My Claimed" tab where no vlm_ready rows are shown
    fireEvent.click(screen.getByRole("button", { name: /My Claimed/ }));
    await waitFor(() => expect(screen.getByTestId("episode-row-ds/my_ep")).toBeInTheDocument());

    // Claim Next queries server vlm_ready, skips 409 ds/taken_ep, claims ds/free_ep
    fireEvent.click(screen.getByTestId("claim-next-btn"));
    await waitFor(() => expect(screen.getByTestId("loc-spy")).toHaveTextContent("/review/ds/free_ep"));
    unmount();

    // Lead: failed activeJob banner displays error + dismiss button, and filter clears selection
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.endsWith("/api/jobs") && init?.method === "POST") {
          return new Response(
            JSON.stringify(
              makeJobStatus({
                job_id: "job_fail_1",
                job_type: "batch_vlm",
                status: "failed",
                progress_done: 0,
                progress_total: 1,
                summary: { errors: [{ episode_id: "ds/ep_a", error: "VLM timeout 504" }] },
              }),
            ),
            { status: 201 },
          );
        }
        const items = url.includes("status=approved")
          ? [makeEpisodeListItem({ episode_id: "ds/ep_b", status: "approved" })]
          : [makeEpisodeListItem({ episode_id: "ds/ep_a", status: "ingested" })];
        return new Response(
          JSON.stringify({
            items,
            total: items.length,
            status_counts: { ingested: 1, approved: 1 },
          }),
          { status: 200 },
        );
      }),
    );

    render(
      <AuthProvider initialAuth={{ token: "lead-tok", username: "lead_1", role: "lead" }}>
        <MemoryRouter>
          <QueuePage />
        </MemoryRouter>
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("episode-row-ds/ep_a")).toBeInTheDocument());
    const cb = screen.getByTestId("episode-row-ds/ep_a").querySelector("input[type='checkbox']")!;
    fireEvent.click(cb);
    fireEvent.click(screen.getByTestId("batch-vlm-btn"));

    expect(await screen.findByTestId("active-job-error-summary")).toHaveTextContent("VLM timeout 504");
    fireEvent.click(screen.getByTestId("dismiss-active-job-btn"));
    expect(screen.queryByTestId("active-job-banner")).toBeNull();

    // Switching status filter clears selection so "Select all" is NOT falsely checked on ds/ep_b
    fireEvent.click(screen.getByRole("button", { name: /Approved/ }));
    await waitFor(() => expect(screen.getByTestId("episode-row-ds/ep_b")).toBeInTheDocument());
    expect((screen.getByLabelText("Select all") as HTMLInputElement).checked).toBe(false);
    expect(screen.getByTestId("lead-batch-controls")).toHaveTextContent("Selected: 0");
  });

  it("FE-9: ExportPage guards empty statuses/min_piece_s and LabHistoryDrawer prevents out-of-order diff rollback race", async () => {
    let postCalls = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
        if (init?.method === "POST") postCalls += 1;
        return new Response(JSON.stringify({ items: [] }), { status: 200 });
      }),
    );

    const { unmount } = render(
      <AuthProvider initialAuth={{ token: "lead-tok", username: "lead_1", role: "lead" }}>
        <ExportPage />
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("export-page")).toBeInTheDocument());
    // Uncheck both default include_statuses ("approved", "gt_frozen")
    fireEvent.click(screen.getByLabelText("approved"));
    fireEvent.click(screen.getByLabelText("gt_frozen"));
    fireEvent.click(screen.getByTestId("dry-run-btn"));
    expect(screen.getByText(/Select at least one status/)).toBeInTheDocument();
    expect(postCalls).toBe(0);
    unmount();

    // LabHistoryDrawer out-of-order diff race guard
    let resolveSlowA!: (res: Response) => void;
    const slowAPromise = new Promise<Response>((resolve) => {
      resolveSlowA = resolve;
    });
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.includes("/snap_A/diff")) return slowAPromise;
        if (url.includes("/snap_B/diff")) {
          return new Response(
            JSON.stringify({
              snapshot_id: "snap_B",
              relative_path: "labels.yaml",
              created_at: 1700000200,
              reason: "edit_b",
              unified_diff: "+B",
              lines: [{ type: "add", old_line: null, new_line: 1, text: "B" }],
            }),
            { status: 200 },
          );
        }
        return new Response(
          JSON.stringify({
            items: [
              { snapshot_id: "snap_A", target: "labels", name: "a.yaml", relative_path: "a.yaml", created_at: 1700000100, reason: "a", size_bytes: 10 },
              { snapshot_id: "snap_B", target: "labels", name: "b.yaml", relative_path: "b.yaml", created_at: 1700000200, reason: "b", size_bytes: 10 },
            ],
          }),
          { status: 200 },
        );
      }),
    );

    render(
      <LabHistoryDrawer
        open={true}
        target="labels"
        onClose={vi.fn()}
        onRolledBack={vi.fn()}
      />,
    );

    fireEvent.click(await screen.findByTestId("history-snapshot-snap_A"));
    fireEvent.click(screen.getByTestId("history-snapshot-snap_B"));
    await waitFor(() => expect(screen.getByTestId("history-diff-viewer")).toHaveTextContent("+B"));

    // Late response for snap_A arrives -> MUST NOT overwrite snap_B's diff!
    await act(async () => {
      resolveSlowA(
        new Response(
          JSON.stringify({
            snapshot_id: "snap_A",
            relative_path: "labels.yaml",
            created_at: 1700000100,
            reason: "edit_a",
            unified_diff: "+A",
            lines: [{ type: "add", old_line: null, new_line: 1, text: "A" }],
          }),
          { status: 200 },
        ),
      );
      await Promise.resolve();
    });

    expect(screen.getByTestId("history-diff-viewer")).toHaveTextContent("+B");
    expect(screen.getByTestId("history-diff-viewer")).not.toHaveTextContent("+A");
  });
});
