import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AuthProvider } from "../stores/authStore";
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
});
