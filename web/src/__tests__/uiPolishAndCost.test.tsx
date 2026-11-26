import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AppRoutes } from "../App";
import type { JobEstimateResponse, VlmSettingsResponse } from "../api/types";
import { DEMO_QUICKSTART_COMMAND, OnboardingEmptyState } from "../components/OnboardingEmptyState";
import { PageSkeleton, StudioSkeleton, TableSkeleton } from "../components/SkeletonLoader";
import { VlmSettingsModal } from "../components/VlmSettingsModal";
import { commonEn, commonZh } from "../i18n/common";
import { labEn, labZh } from "../i18n/lab";
import { studioEn, studioZh } from "../i18n/studio";
import { QueuePage } from "../pages/QueuePage";
import { AuthProvider } from "../stores/authStore";
import { I18nProvider } from "../stores/i18nStore";
import { makeEpisodeListItem, makeJobStatus } from "../test/fixtures";

const LocationCapture: React.FC = () => {
  const loc = useLocation();
  return (
    <div data-testid="loc-capture">
      {loc.pathname}
      {loc.search}
    </div>
  );
};

function makeEstimateResponse(overrides: Partial<JobEstimateResponse> = {}): JobEstimateResponse {
  return {
    episode_count: 2,
    total_duration_s: 24.5,
    stage: "all",
    fresh: false,
    estimated_prompt_tokens: 18400,
    estimated_completion_tokens: 2600,
    estimated_total_tokens: 21000,
    estimated_cost_usd: 0.048,
    estimated_cache_hits: 3,
    per_stage: [
      {
        stage: "s1_grounding",
        model: "gpt-4o-mini",
        estimated_calls: 2,
        estimated_images: 12,
        image_detail: "high",
        estimated_tokens: 6500,
        estimated_cost_usd: 0.012,
        cached_episodes: 1,
      },
      {
        stage: "s2_macro",
        model: "gpt-4o-mini",
        estimated_calls: 2,
        estimated_images: 16,
        image_detail: "low",
        estimated_tokens: 3200,
        estimated_cost_usd: 0.004,
        cached_episodes: 1,
      },
      {
        stage: "s3_discover",
        model: "gpt-4o",
        estimated_calls: 4,
        estimated_images: 28,
        image_detail: "auto",
        estimated_tokens: 11300,
        estimated_cost_usd: 0.032,
        cached_episodes: 1,
      },
    ],
    token_saving_active: {
      low_detail_stages: true,
      static_frame_dedup: true,
      adaptive_s3_step: true,
      short_episode_s1_merge: true,
      s4b_dynamic_frames: true,
    },
    ...overrides,
  };
}

function makeVlmSettingsBase(overrides: Partial<VlmSettingsResponse> = {}): VlmSettingsResponse {
  return {
    provider: "openai_compat",
    base_url: "https://api.openai.com/v1",
    api_key_env: "OPENAI_API_KEY",
    api_key_configured: true,
    api_key_masked: "sk-***1234",
    thinking_mode: "disabled",
    stage_models: {
      s1_grounding: "gpt-4o",
      s2_macro: "gpt-4o",
      s3_discover: "gpt-4o",
      s4_verify: "gpt-4o",
      s5_success: "gpt-4o",
    },
    presets: [],
    ...overrides,
  };
}

describe("UI Polish, Command Palette, Onboarding, Cost Telemetry & i18n Parity", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.localStorage.clear();
    delete document.documentElement.dataset.theme;
  });

  it("opens CommandPalette via Cmd+K / button, lazy-fetches episodes, fuzzy searches, respects role, and runs actions", async () => {
    const fetchedUrls: string[] = [];
    const items = [
      makeEpisodeListItem({ episode_id: "demo_ds/ep_alpha", task_id: "pick_and_place", status: "vlm_ready" }),
      makeEpisodeListItem({ episode_id: "demo_ds/ep_beta", task_id: "fold_towel", status: "approved" }),
    ];

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        fetchedUrls.push(url);
        if (url.startsWith("/api/jobs")) {
          return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200 });
        }
        if (url === "/api/settings/vlm") {
          return new Response(JSON.stringify(makeVlmSettingsBase()), { status: 200 });
        }
        return new Response(
          JSON.stringify({ items, total: items.length, status_counts: { vlm_ready: 1, approved: 1 } }),
          { status: 200 },
        );
      }),
    );

    // 1. Lead role: Cmd+K opens palette, lazy-fetches limit=50 episodes, shows Lead commands
    const { unmount } = render(
      <AuthProvider initialAuth={{ token: "lead-tok", username: "lead_1", role: "lead" }}>
        <MemoryRouter initialEntries={["/"]}>
          <AppRoutes />
        </MemoryRouter>
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("queue-page")).toBeInTheDocument());
    expect(screen.getByTestId("brand-mark-svg")).toBeInTheDocument();
    expect(fetchedUrls.some((u) => u.includes("limit=50"))).toBe(false);

    fireEvent.keyDown(window, { key: "k", ctrlKey: true });
    expect(await screen.findByTestId("command-palette")).toBeInTheDocument();
    await waitFor(() => expect(fetchedUrls.some((u) => u.includes("limit=50"))).toBe(true));

    expect(screen.getByTestId("cmd-palette-item-nav-export")).toBeInTheDocument();
    expect(screen.getByTestId("cmd-palette-item-act-vlm-settings")).toBeInTheDocument();
    expect(screen.getByTestId("cmd-palette-item-act-job-drawer")).toBeInTheDocument();
    expect(await screen.findByTestId("cmd-palette-item-ep-demo_ds/ep_alpha")).toBeInTheDocument();

    // Fuzzy search for episode beta and press Enter -> navigates or switches theme
    const input = screen.getByTestId("cmd-palette-input");
    fireEvent.change(input, { target: { value: "theme" } });
    expect(screen.getByTestId("cmd-palette-item-act-theme")).toBeInTheDocument();
    fireEvent.keyDown(window, { key: "Enter" });
    await waitFor(() => expect(screen.queryByTestId("command-palette")).toBeNull());
    expect(document.documentElement.dataset.theme).toBe("light");

    // Open via TopNav pill button and Escape closes
    fireEvent.click(screen.getByTestId("open-cmd-palette-btn"));
    expect(await screen.findByTestId("command-palette")).toBeInTheDocument();
    fireEvent.keyDown(window, { key: "Escape" });
    await waitFor(() => expect(screen.queryByTestId("command-palette")).toBeNull());
    unmount();

    // 2. Reviewer role: hides Lead-only commands
    render(
      <AuthProvider initialAuth={{ token: "rev-tok", username: "rev_1", role: "reviewer" }}>
        <MemoryRouter initialEntries={["/"]}>
          <AppRoutes />
        </MemoryRouter>
      </AuthProvider>,
    );
    await waitFor(() => expect(screen.getByTestId("queue-page")).toBeInTheDocument());
    fireEvent.click(screen.getByTestId("open-cmd-palette-btn"));
    expect(await screen.findByTestId("command-palette")).toBeInTheDocument();
    expect(screen.queryByTestId("cmd-palette-item-nav-export")).toBeNull();
    expect(screen.queryByTestId("cmd-palette-item-nav-lab")).toBeNull();
    expect(screen.queryByTestId("cmd-palette-item-act-vlm-settings")).toBeNull();
    expect(screen.queryByTestId("cmd-palette-item-act-job-drawer")).toBeNull();
    expect(screen.getByTestId("cmd-palette-item-act-locale")).toBeInTheDocument();
  });

  it("renders OnboardingEmptyState in empty QueuePage with copyable quickstart command and Lead actions", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText },
      configurable: true,
    });

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url === "/api/settings/vlm") {
          return new Response(JSON.stringify(makeVlmSettingsBase()), { status: 200 });
        }
        if (url.startsWith("/api/ingest/browse")) {
          return new Response(
            JSON.stringify({
              current_path: "/data",
              parent_path: null,
              allowed_roots: ["/data"],
              entries: [],
              detected_adapters: [],
            }),
            { status: 200 },
          );
        }
        return new Response(JSON.stringify({ items: [], total: 0, status_counts: {} }), {
          status: 200,
        });
      }),
    );

    render(
      <AuthProvider initialAuth={{ token: "lead-tok", username: "lead_1", role: "lead" }}>
        <MemoryRouter>
          <QueuePage />
        </MemoryRouter>
      </AuthProvider>,
    );

    const card = await screen.findByTestId("onboarding-empty-state");
    expect(card).toBeInTheDocument();
    expect(screen.getByTestId("onboarding-demo-cmd")).toHaveTextContent(DEMO_QUICKSTART_COMMAND);

    fireEvent.click(screen.getByTestId("onboarding-copy-cmd-btn"));
    await waitFor(() => expect(writeText).toHaveBeenCalledWith(DEMO_QUICKSTART_COMMAND));

    // Clicking Configure VLM API opens VlmSettingsModal
    fireEvent.click(screen.getByTestId("onboarding-vlm-btn"));
    expect(await screen.findByTestId("vlm-settings-modal")).toBeInTheDocument();
    fireEvent.click(screen.getByTestId("close-vlm-settings-btn"));

    // Clicking Ingest Local Data opens IngestModal
    fireEvent.click(screen.getByTestId("onboarding-ingest-btn"));
    expect(await screen.findByTestId("ingest-modal")).toBeInTheDocument();
  });

  it("hides Lead buttons on OnboardingEmptyState for Reviewer and renders SkeletonLoader variants", () => {
    const { unmount } = render(
      <I18nProvider initialLocale="en">
        <OnboardingEmptyState isLead={false} />
      </I18nProvider>,
    );
    expect(screen.getByTestId("onboarding-empty-state")).toBeInTheDocument();
    expect(screen.queryByTestId("onboarding-ingest-btn")).toBeNull();
    expect(screen.queryByTestId("onboarding-vlm-btn")).toBeNull();
    unmount();

    render(
      <I18nProvider initialLocale="en">
        <table>
          <tbody>
            <TableSkeleton rows={3} columns={4} />
          </tbody>
        </table>
        <StudioSkeleton />
        <PageSkeleton />
      </I18nProvider>,
    );
    expect(screen.getByTestId("table-skeleton")).toBeInTheDocument();
    expect(screen.getByTestId("studio-skeleton")).toBeInTheDocument();
    expect(screen.getByTestId("page-skeleton")).toBeInTheDocument();
  });

  it("applies VlmSettingsModal Cost Tiers with provider auto-detection and auto-expands heterogeneous routing", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(
          JSON.stringify(
            makeVlmSettingsBase({
              provider: "openai_compat",
              base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1",
              stage_models: {
                s1_grounding: "qwen-vl-max",
                s2_macro: "qwen-vl-max",
                s3_discover: "qwen-vl-max",
                s4_verify: "qwen-vl-max",
                s5_success: "qwen-vl-max",
              },
            }),
          ),
          { status: 200 },
        ),
      ),
    );

    render(<VlmSettingsModal open onClose={vi.fn()} />);

    expect(await screen.findByTestId("cost-tier-fast")).toBeInTheDocument();
    expect(screen.getByTestId("cost-tier-balanced")).toBeInTheDocument();
    expect(screen.getByTestId("cost-tier-high_precision")).toBeInTheDocument();

    // Active base_url is dashscope -> clicking Balanced populates heterogeneous DashScope models and auto-expands panel
    fireEvent.click(screen.getByTestId("cost-tier-balanced"));
    expect(screen.getByTestId("vlm-stage-routing-panel")).toBeInTheDocument();
    expect((screen.getByTestId("vlm-stage-model-s1_grounding") as HTMLInputElement).value).toBe(
      "qwen-vl-plus",
    );
    expect((screen.getByTestId("vlm-stage-model-s3_discover") as HTMLInputElement).value).toBe(
      "qwen-vl-max",
    );

    // Switch provider to Gemini and click Fast -> applies Gemini flash models
    fireEvent.change(screen.getByTestId("vlm-provider-select"), { target: { value: "gemini" } });
    fireEvent.click(screen.getByTestId("cost-tier-fast"));
    expect((screen.getByTestId("vlm-global-model-input") as HTMLInputElement).value).toBe(
      "gemini-2.5-flash",
    );
  });

  it("syncs QueuePage URL searchParams, supports J/K/X/Enter/slash keyboard flow, renders cost cell, and runs CostEstimatePopover", async () => {
    const requestedUrls: string[] = [];
    const estimateBodies: unknown[] = [];
    const jobBodies: unknown[] = [];

    const items = [
      makeEpisodeListItem({
        episode_id: "demo_ds/ep_001",
        dataset: "demo_ds",
        task_id: "pick_and_place",
        status: "vlm_ready",
        tokens_total: 12500,
        cached_tokens: 5000,
        estimated_cost_usd: 0.0312,
      }),
      makeEpisodeListItem({
        episode_id: "demo_ds/ep_002",
        dataset: "demo_ds",
        task_id: "pick_and_place",
        status: "vlm_ready",
        tokens_total: null,
        estimated_cost_usd: null,
      }),
    ];

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        const method = init?.method ?? "GET";
        requestedUrls.push(`${method} ${url}`);
        if (url === "/api/jobs/estimate" && method === "POST") {
          estimateBodies.push(JSON.parse(String(init?.body ?? "{}")));
          return new Response(JSON.stringify(makeEstimateResponse()), { status: 200 });
        }
        if (url === "/api/jobs" && method === "POST") {
          jobBodies.push(JSON.parse(String(init?.body ?? "{}")));
          return new Response(
            JSON.stringify(
              makeJobStatus({
                job_id: "job_est_run_1",
                job_type: "batch_vlm",
                status: "queued",
                progress_done: 0,
                progress_total: 1,
              }),
            ),
            { status: 201 },
          );
        }
        if (url === "/api/episodes/tasks") {
          return new Response(JSON.stringify({ tasks: [], total: 0 }), { status: 200 });
        }
        return new Response(
          JSON.stringify({
            items,
            total: items.length,
            status_counts: { vlm_ready: 2 },
            facets: {
              datasets: ["demo_ds"],
              task_ids: ["pick_and_place"],
              adapter_types: ["h5_raw_v3"],
            },
          }),
          { status: 200 },
        );
      }),
    );

    render(
      <AuthProvider initialAuth={{ token: "lead-tok", username: "lead_1", role: "lead" }}>
        <MemoryRouter initialEntries={["/?status=vlm_ready&dataset=demo_ds&sort_by=duration_s&sort_dir=desc"]}>
          <LocationCapture />
          <Routes>
            <Route path="/" element={<QueuePage />} />
            <Route path="/review/:dataset/:stem" element={<div data-testid="mock-studio">Studio</div>} />
          </Routes>
        </MemoryRouter>
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("episode-row-demo_ds/ep_001")).toBeInTheDocument());

    // Initial URL params hydrated into API call
    expect(
      requestedUrls.some(
        (u) =>
          u.includes("status=vlm_ready") &&
          u.includes("dataset=demo_ds") &&
          u.includes("sort_by=duration_s") &&
          u.includes("sort_dir=desc"),
      ),
    ).toBe(true);

    // Tokens / Cost cell displays formatted tokens, USD cost, and cached % badge
    const costCell1 = screen.getByTestId("episode-cost-demo_ds/ep_001");
    expect(costCell1).toHaveTextContent("12.5k");
    expect(costCell1).toHaveTextContent("$0.031");
    expect(costCell1).toHaveTextContent("40% cached");
    expect(screen.getByTestId("episode-cost-demo_ds/ep_002")).toHaveTextContent("—");

    // Keyboard navigation: j focuses row 0, j focuses row 1, k moves back to row 0, x selects row 0
    fireEvent.keyDown(window, { key: "j" });
    expect(screen.getByTestId("episode-row-demo_ds/ep_001")).toHaveAttribute("data-focused", "true");
    fireEvent.keyDown(window, { key: "j" });
    expect(screen.getByTestId("episode-row-demo_ds/ep_002")).toHaveAttribute("data-focused", "true");
    fireEvent.keyDown(window, { key: "k" });
    expect(screen.getByTestId("episode-row-demo_ds/ep_001")).toHaveAttribute("data-focused", "true");

    fireEvent.keyDown(window, { key: "x" });
    expect(screen.getByTestId("lead-batch-controls")).toHaveTextContent("Selected: 1");

    // '/' focuses search input
    fireEvent.keyDown(window, { key: "/" });
    expect(document.activeElement).toBe(screen.getByTestId("queue-search-input"));
    (document.activeElement as HTMLElement)?.blur();

    // Open CostEstimatePopover -> fetches POST /api/jobs/estimate -> renders table & savings -> Confirm dispatches job
    fireEvent.click(screen.getByTestId("estimate-vlm-cost-btn"));
    expect(await screen.findByTestId("cost-estimate-popover")).toBeInTheDocument();
    expect(await screen.findByTestId("estimate-total-tokens")).toHaveTextContent("21.0k");
    expect(screen.getByTestId("estimate-total-cost")).toHaveTextContent("$0.048");
    expect(screen.getByTestId("cost-estimate-stage-table")).toHaveTextContent("s1_grounding");
    expect(screen.getByTestId("cost-estimate-saving-pills")).toHaveTextContent("Static-frame dedup");
    expect(estimateBodies).toContainEqual({
      episode_ids: ["demo_ds/ep_001"],
      fresh: false,
    });

    fireEvent.click(screen.getByTestId("cost-estimate-confirm-btn"));
    await waitFor(() => expect(jobBodies).toHaveLength(1));
    expect(jobBodies[0]).toEqual({
      job_type: "batch_vlm",
      episode_ids: ["demo_ds/ep_001"],
    });
    await waitFor(() => expect(screen.queryByTestId("cost-estimate-popover")).toBeNull());

    // Enter on focused row navigates to Review Studio
    fireEvent.keyDown(window, { key: "Enter" });
    expect(await screen.findByTestId("mock-studio")).toBeInTheDocument();
    expect(screen.getByTestId("loc-capture")).toHaveTextContent("/review/demo_ds/ep_001");
  });

  it("displays token/cost telemetry on batch_vlm cards and live current_episode_id on running ingest cards in JobCenterDrawer", async () => {
    const vlmJob = makeJobStatus({
      job_id: "job_cost_1",
      job_type: "batch_vlm",
      status: "completed",
      progress_done: 2,
      progress_total: 2,
      summary: {
        tokens_total: 42000,
        cached_tokens: 21000,
        estimated_cost_usd: 0.085,
        results: [{ episode_id: "ds/ep1" }],
        errors: [],
      },
    });
    const ingestJob = makeJobStatus({
      job_id: "job_ing_live",
      job_type: "ingest",
      status: "running",
      progress_done: 3,
      progress_total: 10,
      summary: {
        discovered: 10,
        inserted: 2,
        updated: 1,
        skipped: 0,
        current_episode_id: "demo_ds/ep_004",
      },
    });

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.startsWith("/api/jobs")) {
          return new Response(JSON.stringify({ items: [vlmJob, ingestJob], total: 2 }), {
            status: 200,
          });
        }
        return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200 });
      }),
    );

    render(
      <AuthProvider initialAuth={{ token: "lead-tok", username: "lead_1", role: "lead" }}>
        <MemoryRouter initialEntries={["/"]}>
          <AppRoutes />
        </MemoryRouter>
      </AuthProvider>,
    );

    fireEvent.click(await screen.findByTestId("live-job-pill"));
    expect(await screen.findByTestId("job-center-drawer")).toBeInTheDocument();

    const costStrip = screen.getByTestId("drawer-job-cost-job_cost_1");
    expect(costStrip).toHaveTextContent("42.0k tok");
    expect(costStrip).toHaveTextContent("$0.085");
    expect(costStrip).toHaveTextContent("50% cached");

    const ingestCurrent = screen.getByTestId("drawer-ingest-current-job_ing_live");
    expect(ingestCurrent).toHaveTextContent("demo_ds/ep_004");
  });

  it("maintains 100% EN/ZH i18n key parity across common, studio, and lab dictionaries including all studio contract keys", () => {
    const pairs: Array<[string, Record<string, string>, Record<string, string>]> = [
      ["common", commonEn, commonZh],
      ["studio", studioEn, studioZh],
      ["lab", labEn, labZh],
    ];
    for (const [, enDict, zhDict] of pairs) {
      const enKeys = Object.keys(enDict).sort();
      const zhKeys = Object.keys(zhDict).sort();
      expect(zhKeys).toEqual(enKeys);
      for (const key of enKeys) {
        expect(enDict[key].trim().length).toBeGreaterThan(0);
        expect(zhDict[key].trim().length).toBeGreaterThan(0);
      }
    }

    const requiredStudioKeys = [
      "studio.costChipTitle",
      "studio.costChipTokens",
      "studio.costChipCached",
      "studio.costStageBreakdown",
      "studio.costStageCacheHit",
      "studio.displayControls",
      "studio.snapToGuides",
      "studio.snapHint",
      "studio.compactTracks",
      "studio.shortcutsHeading",
      "studio.zoomTip",
    ] as const;

    for (const k of requiredStudioKeys) {
      expect(studioEn[k]).toBeTruthy();
      expect(studioZh[k]).toBeTruthy();
    }
  });
});
