import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AuthProvider } from "../stores/authStore";
import { I18nProvider } from "../stores/i18nStore";
import { ExportPage } from "./ExportPage";

describe("ExportPage", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("blocks Reviewer role with 403 view and lets Lead trigger 4x3 dry-run and async export job", async () => {
    const manifest = {
      export_id: "exp_001",
      created_at: 1700000300,
      format: "zero_copy_index",
      recipe: {
        name: "recovery_plus",
        format: "zero_copy_index",
        chunk_size: null,
        min_piece_s: 0.1,
        tail_frames: null,
        smooth_tail: "on_jump",
        include_statuses: ["approved", "gt_frozen"],
        context_window_s: 2.0,
        dataset_filter: null,
        task_filter: null,
      },
      output_dir: "/tmp/exports/exp_001",
      episodes_included: 2,
      episodes_skipped: 0,
      pieces_exported: 4,
      pieces_dropped: 0,
      total_frames: 600,
      episodes: [],
      artifacts: ["qc_index.parquet"],
      checksums: {},
    };

    const calls: Array<{ url: string; method: string; body?: unknown }> = [];
    let pollCount = 0;
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      const method = init?.method ?? "GET";
      const body = init?.body ? JSON.parse(String(init.body)) : undefined;
      calls.push({ url, method, body });

      if (url.endsWith("/api/exports/dry-run") && method === "POST") {
        return new Response(JSON.stringify(manifest), { status: 200 });
      }
      if (url.endsWith("/api/jobs") && method === "POST") {
        return new Response(
          JSON.stringify({
            job_id: "job_exp_1",
            job_type: "export",
            status: "running",
            progress_done: 0,
            progress_total: 1,
          }),
          { status: 201 },
        );
      }
      if (url.endsWith("/api/jobs/job_exp_1") && method === "GET") {
        pollCount += 1;
        return new Response(
          JSON.stringify({
            job_id: "job_exp_1",
            job_type: "export",
            status: "completed",
            progress_done: 1,
            progress_total: 1,
            summary: {
              export_id: "exp_001",
              manifest,
              artifacts: [
                {
                  name: "exp_001.zip",
                  path: "exports/exp_001.zip",
                  kind: "export",
                  export_id: "exp_001",
                  episode_id: null,
                },
              ],
            },
          }),
          { status: 200 },
        );
      }
      return new Response(JSON.stringify({ items: [manifest] }), { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);

    // Reviewer sees 403 card
    const { unmount } = render(
      <AuthProvider initialAuth={{ token: "rev-token", username: "rev_1", role: "reviewer" }}>
        <MemoryRouter>
          <ExportPage />
        </MemoryRouter>
      </AuthProvider>,
    );
    expect(screen.getByTestId("export-forbidden")).toBeInTheDocument();
    unmount();

    // Lead sees 4x3 matrix, dry-run, and async export job trigger
    render(
      <AuthProvider initialAuth={{ token: "lead-token", username: "lead_1", role: "lead" }}>
        <MemoryRouter>
          <ExportPage />
        </MemoryRouter>
      </AuthProvider>,
    );

    for (const fmt of ["lerobot_v3", "h5_clean", "zero_copy_index", "viz_mp4"]) {
      expect(screen.getByTestId(`format-card-${fmt}`)).toBeInTheDocument();
    }
    for (const rec of ["bc_clean", "recovery_plus", "badcase_pairs"]) {
      expect(screen.getByTestId(`recipe-card-${rec}`)).toBeInTheDocument();
    }

    fireEvent.click(screen.getByTestId("format-card-zero_copy_index"));
    fireEvent.click(screen.getByTestId("recipe-card-recovery_plus"));
    fireEvent.click(screen.getByTestId("dry-run-btn"));

    await waitFor(() =>
      expect(screen.getByTestId("export-manifest-preview")).toBeInTheDocument(),
    );

    fireEvent.click(screen.getByTestId("trigger-export-btn"));

    await waitFor(() => {
      const jobPost = calls.find((c) => c.url.endsWith("/api/jobs") && c.method === "POST");
      expect(jobPost).toBeDefined();
      expect(jobPost?.body).toMatchObject({
        job_type: "export",
        export_params: {
          format: "zero_copy_index",
          recipe: "recovery_plus",
        },
      });
    });

    await waitFor(() => expect(pollCount).toBeGreaterThanOrEqual(1));
    await waitFor(() =>
      expect(screen.getByTestId("export-info-banner")).toHaveTextContent(/Export completed: exp_001/),
    );
    const dlLink = screen.getByTestId("export-artifact-download-0") as HTMLAnchorElement;
    expect(dlLink.getAttribute("href")).toBe("/api/jobs/job_exp_1/artifact?index=0&token=lead-token");
  });

  it("pre-populates URL params (episode_ids, dataset, include_status), forwards state_schema_mode & split_by_subtask, and clears scope", async () => {
    const manifest = {
      export_id: "dry_run",
      created_at: 1700000300,
      format: "lerobot_v3",
      recipe: {
        name: "bc_clean",
        format: "lerobot_v3",
        chunk_size: null,
        min_piece_s: 0.1,
        tail_frames: null,
        smooth_tail: "on_jump",
        include_statuses: ["vlm_ready", "reviewed"],
        context_window_s: 2.0,
        dataset_filter: "kitchen_ds",
        task_filter: null,
        state_schema_mode: "preserve_native",
        split_by_subtask: true,
      },
      output_dir: "",
      episodes_included: 2,
      episodes_skipped: 0,
      pieces_exported: 6,
      pieces_dropped: 0,
      total_frames: 450,
      episodes: [],
      artifacts: [],
      checksums: {},
    };

    const calls: Array<{ url: string; method: string; body?: Record<string, unknown> }> = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        const method = init?.method ?? "GET";
        const body = init?.body ? (JSON.parse(String(init.body)) as Record<string, unknown>) : undefined;
        calls.push({ url, method, body });

        if (url.endsWith("/api/exports/dry-run") && method === "POST") {
          return new Response(JSON.stringify(manifest), { status: 200 });
        }
        if (url.endsWith("/api/jobs") && method === "POST") {
          return new Response(
            JSON.stringify({
              job_id: "job_scoped_1",
              job_type: "export",
              status: "completed",
              progress_done: 1,
              progress_total: 1,
              summary: { export_id: "exp_scoped_1", manifest, artifacts: [] },
            }),
            { status: 201 },
          );
        }
        return new Response(JSON.stringify({ items: [] }), { status: 200 });
      }),
    );

    render(
      <AuthProvider initialAuth={{ token: "lead-token", username: "lead_1", role: "lead" }}>
        <MemoryRouter
          initialEntries={[
            "/export?episode_ids=kitchen_ds/ep1,kitchen_ds/ep2&dataset=kitchen_ds&include_status=vlm_ready,reviewed",
          ]}
        >
          <ExportPage />
        </MemoryRouter>
      </AuthProvider>,
    );

    // Scoped episode pill banner is displayed
    const scopedBanner = screen.getByTestId("scoped-episodes-banner");
    expect(scopedBanner).toHaveTextContent("Scoped to 2 selected episode(s)");
    expect(scopedBanner).toHaveTextContent("kitchen_ds/ep1");
    expect(scopedBanner).toHaveTextContent("kitchen_ds/ep2");

    // Dataset input and include_statuses checkboxes are pre-populated from URL
    expect(
      (screen.getByPlaceholderText("Leave blank to export all datasets") as HTMLInputElement).value,
    ).toBe("kitchen_ds");

    // Configure state_schema_mode and split_by_subtask
    fireEvent.change(screen.getByTestId("state-schema-mode-select"), {
      target: { value: "preserve_native" },
    });
    fireEvent.click(screen.getByTestId("split-by-subtask-checkbox"));

    // Dry run forwards episode_ids, include_statuses, state_schema_mode, split_by_subtask
    fireEvent.click(screen.getByTestId("dry-run-btn"));
    await waitFor(() =>
      expect(screen.getByTestId("export-manifest-preview")).toBeInTheDocument(),
    );

    const dryRunCall = calls.find((c) => c.url.endsWith("/api/exports/dry-run"));
    expect(dryRunCall?.body).toMatchObject({
      dataset_filter: "kitchen_ds",
      episode_ids: ["kitchen_ds/ep1", "kitchen_ds/ep2"],
      include_statuses: ["vlm_ready", "reviewed"],
      state_schema_mode: "preserve_native",
      split_by_subtask: true,
    });

    // Clear scoped episodes and trigger export -> episode_ids no longer sent
    fireEvent.click(screen.getByTestId("clear-scoped-episodes-btn"));
    expect(screen.queryByTestId("scoped-episodes-banner")).toBeNull();

    fireEvent.click(screen.getByTestId("trigger-export-btn"));
    await waitFor(() => {
      const jobCall = calls.find((c) => c.url.endsWith("/api/jobs") && c.method === "POST");
      expect(jobCall).toBeDefined();
      const epParams = jobCall?.body?.export_params as Record<string, unknown>;
      expect(epParams.state_schema_mode).toBe("preserve_native");
      expect(epParams.split_by_subtask).toBe(true);
      expect("episode_ids" in epParams).toBe(false);
    });
  });

  it("shows smart 0-match anti-empty banner and retries dry-run with reviewed & vlm_ready on one click", async () => {
    const dryRunBodies: Array<Record<string, unknown>> = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        const method = init?.method ?? "GET";
        if (url.endsWith("/api/exports/dry-run") && method === "POST") {
          const body = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
          dryRunBodies.push(body);
          const statuses = (body.include_statuses as string[]) ?? [];
          const hasDrafts = statuses.includes("reviewed") && statuses.includes("vlm_ready");
          return new Response(
            JSON.stringify({
              export_id: "dry_run",
              created_at: 1700000300,
              format: "lerobot_v3",
              recipe: {
                name: "bc_clean",
                format: "lerobot_v3",
                chunk_size: null,
                min_piece_s: 0.1,
                tail_frames: null,
                smooth_tail: "on_jump",
                include_statuses: statuses,
                context_window_s: 2.0,
                dataset_filter: null,
                task_filter: null,
              },
              output_dir: "",
              episodes_included: hasDrafts ? 3 : 0,
              episodes_skipped: 0,
              pieces_exported: hasDrafts ? 5 : 0,
              pieces_dropped: 0,
              total_frames: hasDrafts ? 900 : 0,
              episodes: [],
              artifacts: [],
              checksums: {},
            }),
            { status: 200 },
          );
        }
        return new Response(JSON.stringify({ items: [] }), { status: 200 });
      }),
    );

    render(
      <AuthProvider initialAuth={{ token: "lead-token", username: "lead_1", role: "lead" }}>
        <MemoryRouter initialEntries={["/export"]}>
          <ExportPage />
        </MemoryRouter>
      </AuthProvider>,
    );

    // First dry-run with default [approved, gt_frozen] returns 0 -> shows smart warning banner
    fireEvent.click(screen.getByTestId("dry-run-btn"));
    expect(await screen.findByTestId("empty-export-warning-banner")).toBeInTheDocument();

    // One-click retry includes reviewed & vlm_ready and re-runs dry-run immediately
    fireEvent.click(screen.getByTestId("include-reviewed-vlm-ready-retry-btn"));
    await waitFor(() => expect(dryRunBodies).toHaveLength(2));
    expect(dryRunBodies[1].include_statuses).toEqual([
      "approved",
      "gt_frozen",
      "reviewed",
      "vlm_ready",
    ]);
    await waitFor(() =>
      expect(screen.getByTestId("export-manifest-preview")).toHaveTextContent(
        "Episodes Included: 3",
      ),
    );
    expect(screen.queryByTestId("empty-export-warning-banner")).toBeNull();
  });

  it("renders Chinese copy cleanly when I18nProvider locale is zh", async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ items: [] }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    render(
      <I18nProvider initialLocale="zh">
        <AuthProvider initialAuth={{ token: "lead-token", username: "lead_1", role: "lead" }}>
          <MemoryRouter initialEntries={["/export?episode_ids=ds/ep1"]}>
            <ExportPage />
          </MemoryRouter>
        </AuthProvider>
      </I18nProvider>,
    );

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    expect(screen.getByText("事务化数据集导出中心")).toBeInTheDocument();
    expect(screen.getByTestId("scoped-episodes-banner")).toHaveTextContent(
      "当前限定导出 1 条已选片段",
    );
    expect(screen.getByTestId("dry-run-btn")).toHaveTextContent("预演预览 (Dry Run)");
    expect(screen.getByTestId("trigger-export-btn")).toHaveTextContent("执行事务导出");
    expect(screen.getByText(/按 Subtask 切分为独立 Episode/)).toBeInTheDocument();
  });
});
