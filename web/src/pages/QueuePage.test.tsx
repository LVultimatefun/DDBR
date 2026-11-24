import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AppRoutes } from "../App";
import { AuthProvider } from "../stores/authStore";
import { makeEpisodeListItem, makeJobStatus } from "../test/fixtures";
import { QueuePage } from "./QueuePage";

describe("QueuePage", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders Reviewer claim-next queue without batch controls, and Lead global filters/batch triggers", async () => {
    const items = [
      makeEpisodeListItem({ episode_id: "ds/ep1", status: "vlm_ready" }),
      makeEpisodeListItem({ episode_id: "ds/ep2", status: "approved" }),
    ];

    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.includes("/api/jobs") && init?.method === "POST") {
        return new Response(
          JSON.stringify({
            job_id: "job_123",
            job_type: "batch_vlm",
            status: "queued",
            progress_done: 0,
            progress_total: 1,
          }),
          { status: 200 },
        );
      }
      const filtered = url.includes("status=vlm_ready") ? [items[0]] : items;
      return new Response(JSON.stringify({ items: filtered, total: filtered.length }), {
        status: 200,
      });
    });
    vi.stubGlobal("fetch", fetchMock);

    // Reviewer view
    const { unmount } = render(
      <AuthProvider initialAuth={{ token: "rev-token", username: "rev_1", role: "reviewer" }}>
        <MemoryRouter>
          <QueuePage />
        </MemoryRouter>
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("episode-row-ds/ep1")).toBeInTheDocument());
    expect(screen.getByTestId("claim-next-btn")).toBeInTheDocument();
    expect(screen.queryByTestId("lead-batch-controls")).toBeNull();
    unmount();

    // Lead view
    render(
      <AuthProvider initialAuth={{ token: "lead-token", username: "lead_1", role: "lead" }}>
        <MemoryRouter>
          <QueuePage />
        </MemoryRouter>
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("episode-row-ds/ep2")).toBeInTheDocument());
    expect(screen.getByTestId("lead-status-strip")).toBeInTheDocument();
    expect(screen.getByTestId("lead-filter-toolbar")).toBeInTheDocument();
    expect(screen.getByTestId("lead-batch-controls")).toBeInTheDocument();

    // Select first row and trigger batch_vlm
    const rowCheckbox = screen.getByTestId("episode-row-ds/ep1").querySelector("input[type='checkbox']")!;
    fireEvent.click(rowCheckbox);
    fireEvent.click(screen.getByTestId("batch-vlm-btn"));

    await waitFor(() => expect(screen.getByTestId("active-job-banner")).toBeInTheDocument());
  });

  it("refreshes queue when /api/events emits episode_claimed or review_saved", async () => {
    const instances: Array<{ onmessage: ((evt: MessageEvent) => void) | null; close: () => void }> = [];
    class MockEventSource {
      url: string;
      onopen: (() => void) | null = null;
      onmessage: ((evt: MessageEvent) => void) | null = null;
      onerror: (() => void) | null = null;
      close = vi.fn();
      constructor(url: string) {
        this.url = url;
        instances.push(this);
      }
    }
    vi.stubGlobal("EventSource", MockEventSource);

    let listCalls = 0;
    const items = [makeEpisodeListItem({ episode_id: "ds/ep1", status: "vlm_ready" })];
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        listCalls += 1;
        return new Response(JSON.stringify({ items, total: items.length }), { status: 200 });
      }),
    );

    render(
      <AuthProvider initialAuth={{ token: "rev-token", username: "rev_1", role: "reviewer" }}>
        <MemoryRouter>
          <QueuePage />
        </MemoryRouter>
      </AuthProvider>,
    );

    await waitFor(() => expect(listCalls).toBe(1));
    expect(instances.length).toBeGreaterThanOrEqual(1);
    const es = instances[0];

    act(() => {
      es.onmessage?.({
        data: JSON.stringify({ type: "episode_claimed", episode_id: "ds/ep1" }),
      } as MessageEvent);
    });
    await waitFor(() => expect(listCalls).toBe(2));

    act(() => {
      es.onmessage?.({
        data: JSON.stringify({ type: "review_saved", episode_id: "ds/ep1", action: "submit_review" }),
      } as MessageEvent);
    });
    await waitFor(() => expect(listCalls).toBe(3));
  });

  it("uses SQL global status_counts, facet comboboxes, search/sort params, and Lead IngestModal", async () => {
    const requestedUrls: string[] = [];
    const jobBodies: unknown[] = [];
    const items = [
      makeEpisodeListItem({
        episode_id: "demo_ds/ep1",
        dataset: "demo_ds",
        task_id: "pick_and_place",
        status: "vlm_ready",
        adapter_type: "h5_raw_v3",
      }),
      makeEpisodeListItem({
        episode_id: "handover_ds/ep2",
        dataset: "handover_ds",
        task_id: "bimanual_handover",
        status: "approved",
        adapter_type: "lerobot",
      }),
    ];

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        requestedUrls.push(url);

        if (url.startsWith("/api/ingest/browse")) {
          return new Response(
            JSON.stringify({
              current_path: "/data/demo_batch",
              parent_path: "/data",
              allowed_roots: ["/data"],
              entries: [{ name: "sub_run", path: "/data/demo_batch/sub_run", is_dir: true }],
              detected_adapters: [{ adapter_type: "h5_raw_v3", episode_count: 12 }],
            }),
            { status: 200 },
          );
        }
        if (url.endsWith("/api/jobs") && init?.method === "POST") {
          const body = init.body ? JSON.parse(String(init.body)) : {};
          jobBodies.push(body);
          return new Response(
            JSON.stringify({
              job_id: "job_ing_1",
              job_type: "ingest",
              status: "running",
              progress_done: 0,
              progress_total: 12,
            }),
            { status: 201 },
          );
        }

        const filtered = url.includes("status=approved") ? [items[1]] : items;
        return new Response(
          JSON.stringify({
            items: filtered,
            total: filtered.length,
            status_counts: { vlm_ready: 5, approved: 9, gt_frozen: 2 },
            facets: {
              datasets: ["demo_ds", "handover_ds"],
              task_ids: ["pick_and_place", "bimanual_handover"],
              adapter_types: ["h5_raw_v3", "lerobot"],
            },
          }),
          { status: 200 },
        );
      }),
    );

    render(
      <AuthProvider initialAuth={{ token: "lead-token", username: "lead_1", role: "lead" }}>
        <MemoryRouter>
          <QueuePage />
        </MemoryRouter>
      </AuthProvider>,
    );

    const statusStrip = await screen.findByTestId("lead-status-strip");
    await waitFor(() => expect(statusStrip).toHaveTextContent("All (16)"));
    expect(statusStrip).toHaveTextContent("Ready for Review (5)");
    expect(statusStrip).toHaveTextContent("Approved (9)");

    // Clicking a status pill preserves global status_counts on remaining pills
    fireEvent.click(screen.getByText("Approved (9)"));
    await waitFor(() => expect(screen.queryByTestId("episode-row-demo_ds/ep1")).toBeNull());
    expect(screen.getByTestId("lead-status-strip")).toHaveTextContent("Ready for Review (5)");

    // Facet combobox dropdown selection
    const dsInput = screen.getByPlaceholderText("Filter by dataset (dataset)");
    fireEvent.focus(dsInput);
    fireEvent.mouseDown(await screen.findByRole("option", { name: "demo_ds" }));
    await waitFor(() =>
      expect(requestedUrls.some((u) => u.includes("dataset=demo_ds"))).toBe(true),
    );

    // Search + Sort + Adapter filter
    fireEvent.change(screen.getByTestId("queue-search-input"), { target: { value: "ep1" } });
    fireEvent.change(screen.getByTestId("queue-sort-by"), { target: { value: "duration_s" } });
    fireEvent.change(screen.getByTestId("queue-sort-dir"), { target: { value: "desc" } });
    fireEvent.change(screen.getByTestId("adapter-filter-select"), { target: { value: "h5_raw_v3" } });

    await waitFor(() =>
      expect(
        requestedUrls.some(
          (u) =>
            u.includes("search=ep1") &&
            u.includes("sort_by=duration_s") &&
            u.includes("sort_dir=desc") &&
            u.includes("adapter_type=h5_raw_v3"),
        ),
      ).toBe(true),
    );

    // Lead IngestModal flow
    fireEvent.click(screen.getByTestId("open-ingest-modal-btn"));
    expect(await screen.findByTestId("ingest-modal")).toBeInTheDocument();
    expect(await screen.findByTestId("ingest-detected-adapters")).toHaveTextContent(
      "h5_raw_v3 · 12 episodes",
    );

    fireEvent.change(screen.getByTestId("ingest-dataset-input"), { target: { value: "demo_new" } });
    fireEvent.click(screen.getByTestId("submit-ingest-btn"));

    await waitFor(() =>
      expect(jobBodies).toContainEqual({
        job_type: "ingest",
        ingest_params: {
          path: "/data/demo_batch",
          adapter: "auto",
          dataset: "demo_new",
        },
      }),
    );
    expect(await screen.findByTestId("active-job-banner")).toHaveTextContent("job_ing_1");
  });

  it("supports clickable sortable table columns and Lead workspace settings popover", async () => {
    const requestedUrls: string[] = [];
    const putBodies: unknown[] = [];
    const items = [
      makeEpisodeListItem({
        episode_id: "demo_ds/ep1",
        status: "vlm_ready",
        duration_s: 12.5,
        cut_ratio: 0.2,
        badcase_count: 3,
        updated_at: 1700000500,
      }),
    ];

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        const method = init?.method ?? "GET";
        requestedUrls.push(`${method} ${url}`);

        if (url === "/api/settings" && method === "GET") {
          return new Response(
            JSON.stringify({ min_watch_coverage: 0.9, batch_vlm_concurrency: 4 }),
            { status: 200 },
          );
        }
        if (url === "/api/settings" && method === "PUT") {
          const parsed = init?.body ? JSON.parse(String(init.body)) : {};
          putBodies.push(parsed);
          return new Response(JSON.stringify(parsed), { status: 200 });
        }
        return new Response(
          JSON.stringify({
            items,
            total: 1,
            status_counts: { vlm_ready: 1 },
            facets: { datasets: ["demo_ds"], task_ids: ["pick_and_place"], adapter_types: ["h5_raw_v3"] },
          }),
          { status: 200 },
        );
      }),
    );

    render(
      <AuthProvider initialAuth={{ token: "lead-token", username: "lead_1", role: "lead" }}>
        <MemoryRouter>
          <QueuePage />
        </MemoryRouter>
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("episode-row-demo_ds/ep1")).toBeInTheDocument());

    // Click sortable column headers: updated_at, cut_ratio, badcase_count
    fireEvent.click(screen.getByTestId("sort-col-updated_at"));
    await waitFor(() =>
      expect(
        requestedUrls.some((u) => u.includes("sort_by=updated_at") && u.includes("sort_dir=desc")),
      ).toBe(true),
    );
    fireEvent.click(screen.getByTestId("sort-col-updated_at"));
    await waitFor(() =>
      expect(
        requestedUrls.some((u) => u.includes("sort_by=updated_at") && u.includes("sort_dir=asc")),
      ).toBe(true),
    );

    fireEvent.click(screen.getByTestId("sort-col-cut_ratio"));
    await waitFor(() =>
      expect(
        requestedUrls.some((u) => u.includes("sort_by=cut_ratio") && u.includes("sort_dir=desc")),
      ).toBe(true),
    );

    fireEvent.click(screen.getByTestId("sort-col-badcase_count"));
    await waitFor(() =>
      expect(
        requestedUrls.some((u) => u.includes("sort_by=badcase_count") && u.includes("sort_dir=desc")),
      ).toBe(true),
    );

    // Open Lead workspace settings popover, edit and save
    fireEvent.click(screen.getByTestId("open-settings-popover-btn"));
    expect(await screen.findByTestId("workspace-settings-popover")).toBeInTheDocument();

    const covInput = (await screen.findByTestId("setting-min-watch-coverage")) as HTMLInputElement;
    const concInput = screen.getByTestId("setting-batch-concurrency") as HTMLInputElement;
    await waitFor(() => expect(covInput.value).toBe("0.9"));
    expect(concInput.value).toBe("4");

    fireEvent.change(covInput, { target: { value: "0.8" } });
    fireEvent.change(concInput, { target: { value: "6" } });
    fireEvent.click(screen.getByTestId("save-settings-btn"));

    await waitFor(() =>
      expect(putBodies).toContainEqual({
        min_watch_coverage: 0.8,
        batch_vlm_concurrency: 6,
      }),
    );
    expect(await screen.findByTestId("queue-settings-toast")).toHaveTextContent("80%");
    expect(screen.queryByTestId("workspace-settings-popover")).toBeNull();
  });

  it("immediately refreshes episode statuses when cancelling batch_vlm from Queue banner or JobCenterDrawer, and streams SSE in --no-auth mode", async () => {
    const sseInstances: Array<{ url: string; onmessage: ((evt: MessageEvent) => void) | null; close: () => void }> = [];
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

    let epStatus: "ingested" | "vlm_queued" | "vlm_running" = "ingested";
    let listCalls = 0;
    const runningJob = makeJobStatus({
      job_id: "job_vlm_cancel_1",
      job_type: "batch_vlm",
      status: "running",
      progress_done: 0,
      progress_total: 1,
      params: { episode_ids: ["ds/ep_cancel"] },
    });

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        const method = init?.method ?? "GET";
        if (url.includes("/api/auth/me")) {
          return new Response(
            JSON.stringify({ role: "lead", user: "lead_noauth", no_auth: true }),
            { status: 200 },
          );
        }
        if (url.endsWith("/api/jobs") && method === "POST") {
          epStatus = "vlm_queued";
          return new Response(JSON.stringify(runningJob), { status: 201 });
        }
        if (url.endsWith("/api/jobs/job_vlm_cancel_1/cancel") && method === "POST") {
          epStatus = "ingested";
          return new Response(
            JSON.stringify({ ...runningJob, status: "cancelled" }),
            { status: 200 },
          );
        }
        if (url.startsWith("/api/jobs")) {
          return new Response(
            JSON.stringify({
              items: epStatus === "ingested" ? [] : [runningJob],
              total: epStatus === "ingested" ? 0 : 1,
            }),
            { status: 200 },
          );
        }
        if (url.includes("/api/episodes")) {
          listCalls += 1;
          return new Response(
            JSON.stringify({
              items: [makeEpisodeListItem({ episode_id: "ds/ep_cancel", status: epStatus })],
              total: 1,
              status_counts: { [epStatus]: 1 },
            }),
            { status: 200 },
          );
        }
        return new Response(JSON.stringify({}), { status: 200 });
      }),
    );

    render(
      <AuthProvider initialAuth={{ token: "", username: "lead_noauth", role: "lead", noAuth: true }}>
        <MemoryRouter initialEntries={["/"]}>
          <AppRoutes />
        </MemoryRouter>
      </AuthProvider>,
    );

    // Initial load shows ingested ("Ingested") and connects /api/events in --no-auth mode
    await waitFor(() =>
      expect(screen.getByTestId("episode-row-ds/ep_cancel")).toHaveTextContent("Ingested"),
    );
    expect(sseInstances.some((es) => es.url === "/api/events")).toBe(true);

    // Select episode and start Batch VLM -> status refreshes to vlm_queued ("VLM Queued")
    const cb = screen.getByTestId("episode-row-ds/ep_cancel").querySelector("input[type='checkbox']")!;
    fireEvent.click(cb);
    fireEvent.click(screen.getByTestId("batch-vlm-btn"));

    await waitFor(() =>
      expect(screen.getByTestId("episode-row-ds/ep_cancel")).toHaveTextContent("VLM Queued"),
    );
    const callsBeforeDrawerCancel = listCalls;

    // Cancel from JobCenterDrawer -> immediately refreshes QueuePage back to ingested + shows warning toast
    fireEvent.click(screen.getByTestId("live-job-pill"));
    fireEvent.click(await screen.findByTestId("drawer-cancel-job-job_vlm_cancel_1"));

    await waitFor(() => expect(listCalls).toBeGreaterThan(callsBeforeDrawerCancel));
    await waitFor(() =>
      expect(screen.getByTestId("episode-row-ds/ep_cancel")).toHaveTextContent("Ingested"),
    );
    expect(
      await screen.findByTestId("job-cancelled-toast-job_vlm_cancel_1"),
    ).toHaveTextContent("cancelled; unfinished episode statuses reverted");

    // Re-trigger and cancel from Queue banner -> also refreshes immediately back to ingested
    fireEvent.click(screen.getByTestId("close-job-drawer-btn"));
    fireEvent.click(screen.getByTestId("batch-vlm-btn"));
    await waitFor(() =>
      expect(screen.getByTestId("episode-row-ds/ep_cancel")).toHaveTextContent("VLM Queued"),
    );
    const callsBeforeBannerCancel = listCalls;
    const banner = screen.getByTestId("active-job-banner");
    fireEvent.click(banner.querySelector("button.danger-btn")!);

    await waitFor(() => expect(listCalls).toBeGreaterThan(callsBeforeBannerCancel));
    await waitFor(() =>
      expect(screen.getByTestId("episode-row-ds/ep_cancel")).toHaveTextContent("Ingested"),
    );
  });

  it("exposes s1_grounding stage option in batch VLM selector and sends stage: s1_grounding to POST /api/jobs", async () => {
    const jobBodies: Array<Record<string, unknown>> = [];
    const items = [makeEpisodeListItem({ episode_id: "ds/ep_s1", status: "vlm_ready" })];

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        const method = init?.method ?? "GET";
        if (url.endsWith("/api/jobs") && method === "POST") {
          const body = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
          jobBodies.push(body);
          return new Response(
            JSON.stringify(
              makeJobStatus({
                job_id: "job_s1_only",
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
        return new Response(
          JSON.stringify({
            items,
            total: items.length,
            status_counts: { vlm_ready: 1 },
            facets: { datasets: ["ds"], task_ids: ["pick_and_place"], adapter_types: ["h5_raw_v3"] },
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

    await waitFor(() => expect(screen.getByTestId("episode-row-ds/ep_s1")).toBeInTheDocument());

    const stageSelect = screen.getByLabelText("Rerun stage") as HTMLSelectElement;
    const s1Option = Array.from(stageSelect.options).find((opt) => opt.value === "s1_grounding");
    expect(s1Option).toBeDefined();
    expect(s1Option?.textContent).toBe("Rerun S1 only (refresh 2D boxes, reuse S2-S5 cache)");

    fireEvent.change(stageSelect, { target: { value: "s1_grounding" } });
    expect(stageSelect.value).toBe("s1_grounding");

    const rowCheckbox = screen.getByTestId("episode-row-ds/ep_s1").querySelector("input[type='checkbox']")!;
    fireEvent.click(rowCheckbox);
    fireEvent.click(screen.getByTestId("batch-vlm-btn"));

    await waitFor(() => expect(jobBodies).toHaveLength(1));
    expect(jobBodies[0]).toEqual({
      job_type: "batch_vlm",
      episode_ids: ["ds/ep_s1"],
      stage: "s1_grounding",
    });
  });
});
