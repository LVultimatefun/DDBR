import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AppRoutes } from "../App";
import { AuthProvider } from "../stores/authStore";
import { useToast } from "../stores/toastStore";
import { makeEpisodeListItem, makeJobStatus } from "../test/fixtures";

describe("Shell, Theme, Toast, Live Job Pill & JobCenterDrawer", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.localStorage.clear();
    delete document.documentElement.dataset.theme;
  });

  it("toggles dark/light theme and persists choice to localStorage", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify({ items: [], total: 0 }), { status: 200 })),
    );

    render(
      <AuthProvider initialAuth={{ token: "rev-token", username: "rev_1", role: "reviewer" }}>
        <MemoryRouter initialEntries={["/"]}>
          <AppRoutes />
        </MemoryRouter>
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("queue-page")).toBeInTheDocument());
    expect(document.documentElement.dataset.theme).toBe("dark");
    const btn = screen.getByTestId("theme-toggle-btn");
    fireEvent.click(btn);

    expect(document.documentElement.dataset.theme).toBe("light");
    expect(window.localStorage.getItem("robo_vlm_qc.theme")).toBe("light");

    fireEvent.click(btn);
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(window.localStorage.getItem("robo_vlm_qc.theme")).toBe("dark");
  });

  it("deduplicates floating toasts by testId and supports standalone fallback outside ToastProvider", () => {
    const DemoToastCaller = () => {
      const toast = useToast();
      return (
        <div>
          <button
            type="button"
            onClick={() =>
              toast.showToast({ message: "First msg", variant: "info", testId: "dedupe-toast" })
            }
          >
            Emit 1
          </button>
          <button
            type="button"
            onClick={() =>
              toast.showToast({ message: "Second msg", variant: "success", testId: "dedupe-toast" })
            }
          >
            Emit 2
          </button>
          {toast.StandaloneHost}
        </div>
      );
    };

    render(<DemoToastCaller />);

    fireEvent.click(screen.getByText("Emit 1"));
    expect(screen.getByTestId("dedupe-toast")).toHaveTextContent("First msg");

    fireEvent.click(screen.getByText("Emit 2"));
    const matches = screen.getAllByTestId("dedupe-toast");
    expect(matches).toHaveLength(1);
    expect(matches[0]).toHaveTextContent("Second msg");

    fireEvent.click(screen.getByLabelText("Dismiss notification"));
    expect(screen.queryByTestId("dedupe-toast")).toBeNull();
  });

  it("hides Live Job Pill for Reviewer and hydrates/streams/cancels/downloads jobs for Lead", async () => {
    const instances: Array<{
      url: string;
      onmessage: ((evt: MessageEvent) => void) | null;
      close: () => void;
    }> = [];
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

    const runningJob = makeJobStatus({
      job_id: "job_run_1",
      job_type: "batch_vlm",
      status: "running",
      progress_done: 1,
      progress_total: 4,
    });
    const doneRenderJob = makeJobStatus({
      job_id: "job_mp4_1",
      job_type: "render_mp4",
      status: "completed",
      progress_done: 2,
      progress_total: 2,
      summary: {
        results: [{ episode_id: "ds/ep1" }],
        errors: [],
        artifacts: [
          {
            name: "ds__ep1.mp4",
            path: "exports/viz_mp4/ds__ep1.mp4",
            kind: "mp4",
            episode_id: "ds/ep1",
          },
        ],
      },
    });
    const doneIngestJob = makeJobStatus({
      job_id: "job_ing_1",
      job_type: "ingest",
      status: "completed",
      progress_done: 5,
      progress_total: 5,
      summary: {
        discovered: 5,
        inserted: 3,
        updated: 1,
        skipped: 1,
        results: [],
        errors: [],
        artifacts: [],
      },
    });

    const requestedUrls: string[] = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      requestedUrls.push(url);
      if (url.startsWith("/api/jobs/job_run_1/cancel") && init?.method === "POST") {
        return new Response(
          JSON.stringify({ ...runningJob, status: "cancelled" }),
          { status: 200 },
        );
      }
      if (url.startsWith("/api/jobs")) {
        return new Response(
          JSON.stringify({ items: [runningJob, doneRenderJob, doneIngestJob], total: 3 }),
          { status: 200 },
        );
      }
      return new Response(
        JSON.stringify({
          items: [makeEpisodeListItem({ episode_id: "ds/ep1" })],
          total: 1,
        }),
        { status: 200 },
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    // 1. Reviewer: zero /api/jobs calls, no pill
    const { unmount } = render(
      <AuthProvider initialAuth={{ token: "rev-token", username: "rev_1", role: "reviewer" }}>
        <MemoryRouter initialEntries={["/"]}>
          <AppRoutes />
        </MemoryRouter>
      </AuthProvider>,
    );
    await waitFor(() => expect(screen.getByTestId("queue-page")).toBeInTheDocument());
    expect(screen.queryByTestId("live-job-pill")).toBeNull();
    expect(requestedUrls.some((u) => u.startsWith("/api/jobs"))).toBe(false);
    unmount();

    // 2. Lead: hydrates /api/jobs?limit=20, live pill progress, SSE update, cancel & download
    instances.length = 0;
    requestedUrls.length = 0;
    render(
      <AuthProvider initialAuth={{ token: "lead-token", username: "lead_1", role: "lead" }}>
        <MemoryRouter initialEntries={["/"]}>
          <AppRoutes />
        </MemoryRouter>
      </AuthProvider>,
    );

    const pill = await screen.findByTestId("live-job-pill");
    await waitFor(() => expect(pill).toHaveTextContent("Running 1/4 (25%)"));
    expect(requestedUrls.some((u) => u.includes("/api/jobs?limit=20"))).toBe(true);

    // Open drawer and check completed artifact download + cancel button
    fireEvent.click(pill);
    const drawer = await screen.findByTestId("job-center-drawer");
    expect(drawer).toHaveTextContent("job_run_1");
    expect(drawer).toHaveTextContent("job_mp4_1");
    expect(drawer).toHaveTextContent("Discovered 5 · Inserted 3 · Updated 1 · Skipped 1");

    const dl = screen.getByTestId("drawer-download-job-job_mp4_1-0") as HTMLAnchorElement;
    expect(dl.getAttribute("href")).toBe("/api/jobs/job_mp4_1/artifact?index=0&token=lead-token");

    // SSE job_update updates progress to 2/4 (50%)
    const eventStreams = instances.filter((es) => es.url.includes("/api/events"));
    expect(eventStreams.length).toBeGreaterThanOrEqual(1);
    act(() => {
      for (const es of eventStreams) {
        es.onmessage?.({
          data: JSON.stringify({
            type: "job_update",
            job: { ...runningJob, progress_done: 2, progress_total: 4 },
          }),
        } as MessageEvent);
      }
    });
    await waitFor(() => expect(pill).toHaveTextContent("Running 2/4 (50%)"));

    // Cancel running job inside drawer
    fireEvent.click(screen.getByTestId("drawer-cancel-job-job_run_1"));
    await waitFor(() =>
      expect(requestedUrls.some((u) => u.endsWith("/api/jobs/job_run_1/cancel"))).toBe(true),
    );
    await waitFor(() => expect(pill).toHaveTextContent("Jobs (3)"));

    // Escape closes the drawer cleanly
    fireEvent.keyDown(window, { key: "Escape" });
    await waitFor(() => expect(screen.queryByTestId("job-center-drawer")).toBeNull());
  });
});
