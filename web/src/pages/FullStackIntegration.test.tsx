import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AppRoutes } from "../App";
import { configureApiCredentialsProvider } from "../api/client";
import { DiffOverlay } from "../components/DiffOverlay";
import { StudioPlayer } from "../components/StudioPlayer";
import { DEFAULT_VISUAL_LABELS, normalizeSegmentTriplet } from "../constants/labels";
import { AuthProvider } from "../stores/authStore";
import { WatchCoverageTracker } from "../stores/reviewHistoryStore";
import {
  makeEpisodeListItem,
  makeQcSegment,
  makeResolvedEpisode,
  makeReviewArtifact,
} from "../test/fixtures";
import { QueuePage } from "./QueuePage";
import { VlmLabPage } from "./VlmLabPage";

describe("Full-Stack Frontend Integration & Regression Tests", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.localStorage.clear();
    window.history.replaceState({}, "", "/");
    configureApiCredentialsProvider(() => ({ token: "", username: "reviewer" }));
  });

  it("re-fetches QueuePage automatically after 401 -> AuthModal login as Reviewer", async () => {
    const ep1 = makeEpisodeListItem({ episode_id: "ds/ep1", status: "vlm_ready" });
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        const headers = new Headers(init?.headers);
        const tok = headers.get("X-Auth-Token");
        if (url.includes("/api/auth/me")) {
          return new Response(JSON.stringify({ role: "reviewer", user: "rev_1" }), { status: 200 });
        }
        if (url.includes("/api/episodes")) {
          if (tok !== "rev-secret") {
            return new Response(JSON.stringify({ detail: "Missing token" }), { status: 401 });
          }
          return new Response(JSON.stringify({ items: [ep1], total: 1 }), { status: 200 });
        }
        return new Response(JSON.stringify({}), { status: 200 });
      }),
    );

    render(
      <AuthProvider>
        <MemoryRouter initialEntries={["/"]}>
          <AppRoutes />
        </MemoryRouter>
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("auth-modal")).toBeInTheDocument());
    const modal = screen.getByTestId("auth-modal");
    fireEvent.change(modal.querySelector('input[type="password"]')!, {
      target: { value: "rev-secret" },
    });
    fireEvent.submit(modal.querySelector("form")!);

    await waitFor(() => expect(screen.getByTestId("episode-row-ds/ep1")).toBeInTheDocument());
  });

  it("re-fetches ReviewStudioPage automatically after 401 -> AuthModal login as Reviewer", async () => {
    const ep = makeResolvedEpisode({ episode_id: "demo_ds/ep_001" });
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        const headers = new Headers(init?.headers);
        const tok = headers.get("X-Auth-Token");
        if (url.includes("/api/auth/me")) {
          return new Response(JSON.stringify({ role: "reviewer", user: "rev_1" }), { status: 200 });
        }
        if (tok !== "rev-secret") {
          return new Response(JSON.stringify({ detail: "Missing token" }), { status: 401 });
        }
        if (url.endsWith("/reviews")) {
          return new Response(JSON.stringify({ items: [] }), { status: 200 });
        }
        if (url.includes("/api/episodes/demo_ds/ep_001")) {
          return new Response(JSON.stringify(ep), { status: 200 });
        }
        if (url.includes("/api/labels")) {
          return new Response(
            JSON.stringify({ schema_version: "1.0", version: 1, categories: [], labels: DEFAULT_VISUAL_LABELS }),
            { status: 200 },
          );
        }
        return new Response(JSON.stringify({}), { status: 200 });
      }),
    );

    render(
      <AuthProvider>
        <MemoryRouter initialEntries={["/review/demo_ds/ep_001"]}>
          <AppRoutes />
        </MemoryRouter>
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("auth-modal")).toBeInTheDocument());
    const modal = screen.getByTestId("auth-modal");
    fireEvent.change(modal.querySelector('input[type="password"]')!, {
      target: { value: "rev-secret" },
    });
    fireEvent.submit(modal.querySelector("form")!);

    await waitFor(() => expect(screen.getByTestId("review-studio-page")).toBeInTheDocument());
  });

  it("records watch coverage during 2x playback when timeupdate delta is 0.7s", () => {
    const tracker = new WatchCoverageTracker(2.1);
    const onWatchInterval = vi.fn((t0: number, t1: number) => {
      tracker.recordInterval(t0, t1);
    });
    const onTimeUpdate = vi.fn();

    const { container } = render(
      <AuthProvider initialAuth={{ token: "rev-token", username: "rev_1", role: "reviewer" }}>
        <StudioPlayer
          episodeId="ds/ep1"
          cameras={[
            {
              name: "main",
              role: "main",
              width: 640,
              height: 480,
              fps: 30,
              source_type: "mp4",
              is_virtual_crop: false,
              source_stream: "main",
            },
          ]}
          durationS={2.1}
          fps={30}
          grounding={{ schema_version: 1, ref_size: [1280, 720], keyframes_s: [], object_tracks_2d: {} }}
          currentTime={0}
          onTimeUpdate={onTimeUpdate}
          onWatchInterval={onWatchInterval}
        />
      </AuthProvider>,
    );

    fireEvent.click(screen.getByText("2x"));
    const video = container.querySelector("video") as HTMLVideoElement;
    for (const t of [0.7, 1.4, 2.1]) {
      Object.defineProperty(video, "currentTime", { value: t, configurable: true });
      fireEvent.timeUpdate(video);
    }

    expect(onWatchInterval).toHaveBeenCalledTimes(3);
    expect(tracker.snapshot().coverage_ratio).toBeGreaterThanOrEqual(0.95);
  });

  it("DiffOverlay computes live diffs immediately even when reviewDiff prop is non-null", () => {
    const rawSeg = makeQcSegment({ seg_id: "raw_1", t_start: 0.2, t_end: 0.8, splice_policy: "cut_clean", mask_side: null });
    const rev = makeReviewArtifact({ rev_id: "rev_1" });

    render(
      <DiffOverlay
        isLead={true}
        diffEnabled={false}
        onToggleDiff={vi.fn()}
        reviewDiff={{
          fp_deleted: 0,
          fn_added: 0,
          policy_flips: 0,
          label_changes: 0,
          boundary_nudges: 0,
          macro_stage_edits: 0,
          success_flipped: false,
          total_edits: 0,
        }}
        rawVlmSegments={[rawSeg]}
        currentSegments={[]}
        rawVlmSuccess={{ passed: true, verdict: "success", confidence: 0.9, criteria: [], control_check: null, summary: "" }}
        currentSuccess={{ passed: true, verdict: "success", confidence: 0.9, criteria: [], control_check: null, summary: "" }}
        rawVlmMacroStages={[]}
        currentMacroStages={[]}
        reviews={[rev]}
        onRestoreSnapshot={vi.fn()}
      />,
    );

    const metrics = screen.getByTestId("diff-metrics");
    const fpStat = metrics.children[0];
    expect(fpStat).toHaveTextContent("1");
    expect(fpStat).toHaveTextContent("fp_deleted");
  });

  it("normalizeSegmentTriplet clears is_unknown when changing label_id away from unknown_anomaly", () => {
    const unk = makeQcSegment({
      seg_id: "s_unk",
      label_id: "unknown_anomaly",
      is_unknown: true,
    });
    const next = normalizeSegmentTriplet(unk, { label_id: "missed_grasp_retry" });
    expect(next.label_id).toBe("missed_grasp_retry");
    expect(next.is_unknown).toBe(false);
  });

  it("VlmLabPage filter typing does not re-fetch labels, prompts, or tasks", async () => {
    let labelsCalls = 0;
    let promptsCalls = 0;
    let tasksCalls = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.startsWith("/api/lab/labels")) {
          labelsCalls += 1;
          return new Response(
            JSON.stringify({ schema_version: "1.0", version: 1, categories: [], labels: DEFAULT_VISUAL_LABELS }),
            { status: 200 },
          );
        }
        if (url.startsWith("/api/lab/prompts")) {
          promptsCalls += 1;
          return new Response(
            JSON.stringify({
              s1_grounding: { stage: "s1_grounding", version: "1", system_template: "s", user_template: "u", extra_templates: {} },
              s2_macro: { stage: "s2_macro", version: "1", system_template: "s", user_template: "u", extra_templates: {} },
              s3_discover: { stage: "s3_discover", version: "1", system_template: "s", user_template: "u", extra_templates: {} },
              s4_verify: { stage: "s4_verify", version: "1", system_template: "s", user_template: "u", extra_templates: {} },
              s5_success: { stage: "s5_success", version: "1", system_template: "s", user_template: "u", extra_templates: {} },
            }),
            { status: 200 },
          );
        }
        if (url.startsWith("/api/lab/tasks")) {
          tasksCalls += 1;
          return new Response(JSON.stringify({ tasks: {}, total: 0 }), { status: 200 });
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
        if (url.startsWith("/api/lab/unknowns")) {
          return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200 });
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

    await waitFor(() => expect(labelsCalls).toBe(1));
    const unkFilter = screen.getByLabelText("Filter unknown anomaly dataset");
    fireEvent.change(unkFilter, { target: { value: "d" } });
    fireEvent.change(unkFilter, { target: { value: "de" } });
    fireEvent.change(unkFilter, { target: { value: "demo" } });

    expect(labelsCalls).toBe(1);
    expect(promptsCalls).toBe(1);
    expect(tasksCalls).toBe(1);
  });

  it("?token= matching localStorage preserves lead role and skips spurious reviewer QueuePage fetch", async () => {
    window.localStorage.setItem(
      "robo_vlm_qc.auth",
      JSON.stringify({ token: "lead-secret", username: "lead_1", role: "lead" }),
    );
    window.history.replaceState({}, "", "/?token=lead-secret&user=lead_1");

    const episodeUrls: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.includes("/api/auth/me")) {
          return new Response(JSON.stringify({ role: "lead", user: "lead_1" }), { status: 200 });
        }
        if (url.includes("/api/episodes")) {
          episodeUrls.push(url);
          return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200 });
        }
        return new Response(JSON.stringify({}), { status: 200 });
      }),
    );

    render(
      <AuthProvider>
        <MemoryRouter initialEntries={["/"]}>
          <QueuePage />
        </MemoryRouter>
      </AuthProvider>,
    );

    await waitFor(() => expect(episodeUrls.length).toBeGreaterThanOrEqual(1));
    expect(episodeUrls.some((u) => u.includes("status=vlm_ready"))).toBe(false);
  });

  it("ReviewStudioPage save_draft preserves Undo/Redo stack", async () => {
    const ep = makeResolvedEpisode({ episode_id: "demo_ds/ep_001" });
    const rev = makeReviewArtifact({ rev_id: "rev_draft_1", action: "save_draft" });
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.endsWith("/reviews") && init?.method === "POST") {
          return new Response(JSON.stringify(rev), { status: 200 });
        }
        if (url.endsWith("/reviews")) {
          return new Response(JSON.stringify({ items: [] }), { status: 200 });
        }
        if (url.includes("/api/episodes/demo_ds/ep_001")) {
          return new Response(JSON.stringify(ep), { status: 200 });
        }
        if (url.includes("/labels")) {
          return new Response(
            JSON.stringify({ schema_version: "1.0", version: 1, categories: [], labels: DEFAULT_VISUAL_LABELS }),
            { status: 200 },
          );
        }
        return new Response(JSON.stringify({}), { status: 200 });
      }),
    );

    render(
      <AuthProvider initialAuth={{ token: "rev-tok", username: "rev_1", role: "reviewer" }}>
        <MemoryRouter initialEntries={["/review/demo_ds/ep_001"]}>
          <AppRoutes />
        </MemoryRouter>
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("review-studio-page")).toBeInTheDocument());
    fireEvent.click(screen.getByTestId("delete-segment-btn"));
    expect(screen.getByTestId("undo-btn")).toHaveTextContent("Undo (1)");

    await act(async () => {
      fireEvent.click(screen.getByTestId("action-save-draft"));
    });
    await waitFor(() => expect(screen.getByTestId("studio-info-banner")).toHaveTextContent("rev_draft_1"));
    expect(screen.getByTestId("undo-btn")).toHaveTextContent("Undo (1)");
  });

  it("QueuePage displays my_claimed_count, pagination controls, and encodeEpPath links", async () => {
    const ep1 = makeEpisodeListItem({ episode_id: "ds/ep 1", status: "vlm_ready" });
    const ep2 = makeEpisodeListItem({ episode_id: "ds/ep_101", status: "vlm_ready" });

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        const isPage2 = url.includes("offset=100");
        return new Response(
          JSON.stringify({
            items: isPage2 ? [ep2] : [ep1],
            total: 101,
            my_claimed_count: 3,
          }),
          { status: 200 },
        );
      }),
    );

    render(
      <AuthProvider initialAuth={{ token: "rev-tok", username: "rev_1", role: "reviewer" }}>
        <MemoryRouter>
          <QueuePage />
        </MemoryRouter>
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("episode-row-ds/ep 1")).toBeInTheDocument());
    expect(screen.getByText("My Claimed (3)")).toBeInTheDocument();
    const reviewLink = screen.getByText("Open Studio").closest("a");
    expect(reviewLink?.getAttribute("href")).toBe("/review/ds/ep%201");

    expect(screen.getByTestId("queue-pagination")).toBeInTheDocument();
    fireEvent.click(screen.getByTestId("queue-next-page"));
    await waitFor(() => expect(screen.getByTestId("episode-row-ds/ep_101")).toBeInTheDocument());
  });

  it("resets Lead pagination to offset=0 on filter changes and clamps stale non-empty totals", async () => {
    const page1Lead = makeEpisodeListItem({ episode_id: "ds/ep_001", status: "vlm_ready" });
    const page2Lead = makeEpisodeListItem({ episode_id: "ds/ep_101", status: "vlm_ready" });
    const reviewedEp = makeEpisodeListItem({ episode_id: "ds/ep_rev_1", status: "reviewed" });
    const datasetEp = makeEpisodeListItem({ episode_id: "ds_other/ep_x", dataset: "ds_other", status: "vlm_ready" });

    const requestedUrls: string[] = [];
    let simulateStaleRace = false;

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        requestedUrls.push(url);
        const parsed = new URL(url, "http://localhost");
        const offset = Number(parsed.searchParams.get("offset") ?? "0");
        const status = parsed.searchParams.get("status") ?? "";
        const dataset = parsed.searchParams.get("dataset") ?? "";

        if (simulateStaleRace && offset > 0) {
          return new Response(JSON.stringify({ items: [], total: 2, my_claimed_count: 0 }), { status: 200 });
        }
        if (status === "reviewed") {
          return new Response(
            JSON.stringify({ items: offset === 0 ? [reviewedEp] : [], total: 1, my_claimed_count: 0 }),
            { status: 200 },
          );
        }
        if (dataset === "ds_other") {
          return new Response(
            JSON.stringify({ items: offset === 0 ? [datasetEp] : [], total: 1, my_claimed_count: 0 }),
            { status: 200 },
          );
        }
        return new Response(
          JSON.stringify({
            items: offset >= 100 ? [page2Lead] : [page1Lead],
            total: 150,
            my_claimed_count: 0,
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

    await waitFor(() => expect(screen.getByTestId("episode-row-ds/ep_001")).toBeInTheDocument());
    fireEvent.click(screen.getByTestId("queue-next-page"));
    await waitFor(() => expect(screen.getByTestId("episode-row-ds/ep_101")).toBeInTheDocument());

    // 1. Clicking Lead status pill from Page 2 must reset offset=0 and render Page 1 results
    fireEvent.click(screen.getByRole("button", { name: /^Reviewed \(/ }));
    await waitFor(() => expect(screen.getByTestId("episode-row-ds/ep_rev_1")).toBeInTheDocument());
    const lastRevUrl = requestedUrls.filter((u) => u.includes("status=reviewed")).pop();
    expect(lastRevUrl).toContain("offset=0");

    // 2. Navigate to Page 2 again and change dataset input -> resets offset=0
    fireEvent.click(screen.getByRole("button", { name: /^All \(/ }));
    await waitFor(() => expect(screen.getByTestId("episode-row-ds/ep_001")).toBeInTheDocument());
    fireEvent.click(screen.getByTestId("queue-next-page"));
    await waitFor(() => expect(screen.getByTestId("episode-row-ds/ep_101")).toBeInTheDocument());

    fireEvent.change(screen.getByPlaceholderText("Filter by dataset (dataset)"), {
      target: { value: "ds_other" },
    });
    await waitFor(() => expect(screen.getByTestId("episode-row-ds_other/ep_x")).toBeInTheDocument());
    const lastDsUrl = requestedUrls.filter((u) => u.includes("dataset=ds_other")).pop();
    expect(lastDsUrl).toContain("offset=0");

    // 3. Stale page race where offset=100 returns items=[] with total=2 -> clamps back to Page 1
    fireEvent.change(screen.getByPlaceholderText("Filter by dataset (dataset)"), {
      target: { value: "" },
    });
    await waitFor(() => expect(screen.getByTestId("episode-row-ds/ep_001")).toBeInTheDocument());
    fireEvent.click(screen.getByTestId("queue-next-page"));
    await waitFor(() => expect(screen.getByTestId("episode-row-ds/ep_101")).toBeInTheDocument());

    simulateStaleRace = true;
    fireEvent.click(screen.getByRole("button", { name: /Refresh/ }));
    await waitFor(() => expect(screen.getByTestId("episode-row-ds/ep_001")).toBeInTheDocument());
  });

  it("Reviewer submitted and my_claimed tabs send server-side filters and paginate", async () => {
    const readyEp = makeEpisodeListItem({ episode_id: "ds/ep_ready", status: "vlm_ready" });
    const myClaimedEp = makeEpisodeListItem({
      episode_id: "ds/ep_mine",
      status: "claimed",
      claimed_by: "rev_1",
    });
    const subP1 = makeEpisodeListItem({ episode_id: "ds/ep_sub_1", status: "reviewed" });
    const subP2 = makeEpisodeListItem({ episode_id: "ds/ep_sub_101", status: "flagged" });

    const seenQueries: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        seenQueries.push(url);
        const parsed = new URL(url, "http://localhost");
        const status = parsed.searchParams.get("status");
        const claimedBy = parsed.searchParams.get("claimed_by");
        const offset = Number(parsed.searchParams.get("offset") ?? "0");

        if (status === "reviewed,flagged") {
          return new Response(
            JSON.stringify({
              items: offset >= 100 ? [subP2] : [subP1],
              total: 105,
              my_claimed_count: 1,
            }),
            { status: 200 },
          );
        }
        if (status === "claimed" && claimedBy === "rev_1") {
          return new Response(
            JSON.stringify({ items: [myClaimedEp], total: 1, my_claimed_count: 1 }),
            { status: 200 },
          );
        }
        return new Response(
          JSON.stringify({ items: [readyEp], total: 1, my_claimed_count: 1 }),
          { status: 200 },
        );
      }),
    );

    render(
      <AuthProvider initialAuth={{ token: "rev-tok", username: "rev_1", role: "reviewer" }}>
        <MemoryRouter>
          <QueuePage />
        </MemoryRouter>
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("episode-row-ds/ep_ready")).toBeInTheDocument());

    // Switch to "My Claimed" -> sends status=claimed&claimed_by=rev_1
    fireEvent.click(screen.getByRole("button", { name: /My Claimed/ }));
    await waitFor(() => expect(screen.getByTestId("episode-row-ds/ep_mine")).toBeInTheDocument());
    expect(
      seenQueries.some((u) => u.includes("status=claimed") && u.includes("claimed_by=rev_1")),
    ).toBe(true);

    // Switch to "Submitted / Flagged" -> sends status=reviewed,flagged and paginates Page 1 -> Page 2
    fireEvent.click(screen.getByRole("button", { name: /Submitted \/ Flagged/ }));
    await waitFor(() => expect(screen.getByTestId("episode-row-ds/ep_sub_1")).toBeInTheDocument());
    expect(
      seenQueries.some((u) => {
        const p = new URL(u, "http://localhost");
        return p.searchParams.get("status") === "reviewed,flagged" && p.searchParams.get("offset") === "0";
      }),
    ).toBe(true);

    fireEvent.click(screen.getByTestId("queue-next-page"));
    await waitFor(() => expect(screen.getByTestId("episode-row-ds/ep_sub_101")).toBeInTheDocument());
  });
});
