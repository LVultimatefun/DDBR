import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AuthProvider } from "../stores/authStore";
import { I18nProvider } from "../stores/i18nStore";
import { makeResolvedEpisode, makeReviewArtifact } from "../test/fixtures";
import { ReviewStudioPage } from "./ReviewStudioPage";

describe("ReviewStudioPage", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  function setupFetchMock(epOverrides = {}) {
    const ep = makeResolvedEpisode(epOverrides);
    const reviews = [makeReviewArtifact()];
    let postedPayload: Record<string, unknown> | null = null;

    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.endsWith("/reviews") && init?.method === "POST") {
        postedPayload = JSON.parse(String(init.body)) as Record<string, unknown>;
        return new Response(JSON.stringify(reviews[0]), { status: 200 });
      }
      if (url.endsWith("/reviews")) {
        return new Response(JSON.stringify({ items: reviews }), { status: 200 });
      }
      if (url.includes("/api/episodes/demo_ds/ep_001")) {
        return new Response(JSON.stringify(ep), { status: 200 });
      }
      if (url.includes("/labels")) {
        return new Response(JSON.stringify({ schema_version: "1.0", version: 1, categories: [], labels: [] }), {
          status: 200,
        });
      }
      return new Response(JSON.stringify({}), { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);
    return { getPostedPayload: () => postedPayload };
  }

  it("hides Lead-only actions for Reviewer and gates submit_review on >=95% watch coverage", async () => {
    setupFetchMock();

    render(
      <AuthProvider initialAuth={{ token: "rev-token", username: "rev_a", role: "reviewer" }}>
        <MemoryRouter initialEntries={["/review/demo_ds/ep_001"]}>
          <Routes>
            <Route path="/review/:dataset/:stem" element={<ReviewStudioPage />} />
          </Routes>
        </MemoryRouter>
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("review-studio-page")).toBeInTheDocument());

    // Reviewer cannot see approve, freeze_gt, DiffOverlay tab, or Lead diagnostics / Few-Shot bookmark
    expect(screen.queryByTestId("action-approve")).toBeNull();
    expect(screen.queryByTestId("action-freeze-gt")).toBeNull();
    expect(screen.queryByTestId("diff-tab-btn")).toBeNull();
    expect(screen.queryByTestId("lead-guardrail-diagnostics")).toBeNull();
    expect(screen.queryByTestId("add-few-shot-btn")).toBeNull();

    // Watch coverage starts at 0% -> submit_review is disabled for Reviewer
    expect(screen.getByTestId("action-submit-review")).toBeDisabled();
    // save_draft remains enabled
    expect(screen.getByTestId("action-save-draft")).toBeEnabled();
  });

  it("shows Lead controls and sends strict ReviewWritePayload without author or read-only layers", async () => {
    const { getPostedPayload } = setupFetchMock();

    render(
      <AuthProvider initialAuth={{ token: "lead-token", username: "lead_a", role: "lead" }}>
        <MemoryRouter initialEntries={["/review/demo_ds/ep_001"]}>
          <Routes>
            <Route path="/review/:dataset/:stem" element={<ReviewStudioPage />} />
          </Routes>
        </MemoryRouter>
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("review-studio-page")).toBeInTheDocument());

    // Lead sees approve, freeze_gt, render_mp4, Diff tab, and Guardrail diagnostics
    expect(screen.getByTestId("action-approve")).toBeInTheDocument();
    expect(screen.getByTestId("action-freeze-gt")).toBeInTheDocument();
    expect(screen.getByTestId("action-render-mp4")).toBeInTheDocument();
    expect(screen.getByTestId("diff-tab-btn")).toBeInTheDocument();
    expect(screen.getByTestId("lead-guardrail-diagnostics")).toBeInTheDocument();

    // Click approve -> verify ReviewWritePayload has ONLY allowed keys
    fireEvent.click(screen.getByTestId("action-approve"));
    await waitFor(() => expect(getPostedPayload()).not.toBeNull());

    const payload = getPostedPayload()!;
    expect(payload).not.toHaveProperty("author");
    expect(payload).not.toHaveProperty("grounding");
    expect(payload).not.toHaveProperty("arm_timelines");
    expect(payload).not.toHaveProperty("raw_vlm_segments");
    expect(payload.action).toBe("approve");
    expect(payload.base_run_id).toBe("run_001");
    expect(Array.isArray(payload.qc_segments)).toBe(true);
    expect(Array.isArray(payload.macro_stages)).toBe(true);
    expect(payload.success).toBeDefined();
    expect(payload.watch_coverage).toBeDefined();
  });

  it("disables approve and freeze_gt when episode is claimed by another user", async () => {
    setupFetchMock({ status: "claimed", claimed_by: "other_reviewer" });

    render(
      <AuthProvider initialAuth={{ token: "lead-token", username: "lead_a", role: "lead" }}>
        <MemoryRouter initialEntries={["/review/demo_ds/ep_001"]}>
          <Routes>
            <Route path="/review/:dataset/:stem" element={<ReviewStudioPage />} />
          </Routes>
        </MemoryRouter>
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("review-studio-page")).toBeInTheDocument());
    expect(screen.getByTestId("claim-conflict-banner")).toBeInTheDocument();
    expect(screen.getByTestId("action-approve")).toBeDisabled();
    expect(screen.getByTestId("action-freeze-gt")).toBeDisabled();
    expect(screen.getByTestId("action-save-draft")).toBeDisabled();
  });

  it("updates status badge on claim and preserves watch coverage across save_draft reload", async () => {
    const ep = makeResolvedEpisode({ status: "vlm_ready", claimed_by: null });
    const reviews = [makeReviewArtifact()];
    let draftSaved = false;

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.endsWith("/claim") && init?.method === "POST") {
          return new Response(
            JSON.stringify({
              episode_id: ep.episode_id,
              claimed_by: "rev_a",
              claim_expires_at: 9999999999,
              status: "claimed",
            }),
            { status: 200 },
          );
        }
        if (url.endsWith("/reviews") && init?.method === "POST") {
          draftSaved = true;
          return new Response(JSON.stringify(reviews[0]), { status: 200 });
        }
        if (url.endsWith("/reviews")) {
          return new Response(JSON.stringify({ items: reviews }), { status: 200 });
        }
        if (url.includes("/api/episodes/demo_ds/ep_001")) {
          return new Response(
            JSON.stringify({
              ...ep,
              status: draftSaved ? "claimed" : ep.status,
              claimed_by: draftSaved ? "rev_a" : null,
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
      <AuthProvider initialAuth={{ token: "rev-token", username: "rev_a", role: "reviewer" }}>
        <MemoryRouter initialEntries={["/review/demo_ds/ep_001"]}>
          <Routes>
            <Route path="/review/:dataset/:stem" element={<ReviewStudioPage />} />
          </Routes>
        </MemoryRouter>
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("review-studio-page")).toBeInTheDocument());

    // Claim updates local episode status badge to claimed ("Claimed")
    fireEvent.click(screen.getByTestId("claim-btn"));
    await waitFor(() => expect(screen.getByText("Claimed")).toBeInTheDocument());

    // Simulate playback progress from 0 to 10s in 0.5s steps so watch coverage reaches 100%
    const video = container.querySelector("video") as HTMLVideoElement;
    expect(video).not.toBeNull();
    for (let t = 0.5; t <= 10.0; t += 0.5) {
      Object.defineProperty(video, "currentTime", { value: t, configurable: true });
      fireEvent.timeUpdate(video);
    }
    await waitFor(() => expect(screen.getByTestId("watch-coverage-badge")).toHaveTextContent("100%"));

    // Save draft triggers reload of same episodeId; watch coverage must remain 100%
    fireEvent.click(screen.getByTestId("action-save-draft"));
    await waitFor(() => expect(draftSaved).toBe(true));
    await waitFor(() => expect(screen.getByTestId("watch-coverage-badge")).toHaveTextContent("100%"));
  });

  it("provides Export popover: downloads Episode JSON for Reviewer and prompts Lead auth for Export Studio", async () => {
    setupFetchMock();

    const createdBlobs: Blob[] = [];
    const origCreateObjectURL = URL.createObjectURL;
    const origRevokeObjectURL = URL.revokeObjectURL;
    const anchorClickSpy = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    URL.createObjectURL = vi.fn((blob: Blob | MediaSource) => {
      createdBlobs.push(blob as Blob);
      return "blob:mock-episode-json";
    });
    URL.revokeObjectURL = vi.fn();

    try {
      render(
        <AuthProvider initialAuth={{ token: "rev-token", username: "rev_a", role: "reviewer" }}>
          <MemoryRouter initialEntries={["/review/demo_ds/ep_001"]}>
            <Routes>
              <Route path="/review/:dataset/:stem" element={<ReviewStudioPage />} />
              <Route path="/export" element={<div data-testid="export-page-mock">Export</div>} />
            </Routes>
          </MemoryRouter>
        </AuthProvider>,
      );

      await waitFor(() => expect(screen.getByTestId("review-studio-page")).toBeInTheDocument());

      // Open Export popover and click Download Episode JSON
      fireEvent.click(screen.getByTestId("studio-export-btn"));
      expect(screen.getByTestId("studio-export-menu")).toBeInTheDocument();

      fireEvent.click(screen.getByTestId("export-download-json-btn"));
      expect(anchorClickSpy).toHaveBeenCalledTimes(1);
      expect(createdBlobs).toHaveLength(1);
      const text = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result));
        reader.onerror = () => reject(reader.error);
        reader.readAsText(createdBlobs[0]);
      });
      const parsed = JSON.parse(text) as Record<string, unknown>;
      expect(parsed.episode_id).toBe("demo_ds/ep_001");
      expect(Array.isArray(parsed.qc_segments)).toBe(true);
      expect(Array.isArray(parsed.macro_stages)).toBe(true);
      expect(parsed.success).toBeDefined();
      expect(parsed.meta).toBeDefined();
      expect(parsed).toHaveProperty("notes");

      // Reviewer clicking Open in Export Studio stays on studio and sees warning toast
      fireEvent.click(screen.getByTestId("studio-export-btn"));
      fireEvent.click(screen.getByTestId("export-open-studio-btn"));
      expect(screen.queryByTestId("export-page-mock")).toBeNull();
      expect(screen.getByTestId("studio-info-banner")).toHaveTextContent("Lead role");
    } finally {
      URL.createObjectURL = origCreateObjectURL;
      URL.revokeObjectURL = origRevokeObjectURL;
    }
  });

  it("renders Chinese (locale='zh') ReviewStudioPage controls, tabs, status badge, Export popover, and Shortcuts modal", async () => {
    setupFetchMock({ status: "vlm_ready" });

    render(
      <I18nProvider initialLocale="zh">
        <AuthProvider initialAuth={{ token: "lead-token", username: "lead_a", role: "lead" }}>
          <MemoryRouter initialEntries={["/review/demo_ds/ep_001"]}>
            <Routes>
              <Route path="/review/:dataset/:stem" element={<ReviewStudioPage />} />
            </Routes>
          </MemoryRouter>
        </AuthProvider>
      </I18nProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("review-studio-page")).toBeInTheDocument());

    const page = screen.getByTestId("review-studio-page");
    expect(page).toHaveTextContent("待审核");
    expect(page).toHaveTextContent("分段 / 阶段");
    expect(page).toHaveTextContent("成功判定");
    expect(page).toHaveTextContent("VLM 差异与历史");
    expect(screen.getByTestId("action-save-draft")).toHaveTextContent("保存草稿");
    expect(screen.getByTestId("action-submit-review")).toHaveTextContent("提交审核");
    expect(screen.getByTestId("action-approve")).toHaveTextContent("审核通过");
    expect(screen.getByTestId("action-freeze-gt")).toHaveTextContent("冻结为 GT");
    expect(screen.getByTestId("studio-export-btn")).toHaveTextContent("导出");

    // Open Export popover in Chinese
    fireEvent.click(screen.getByTestId("studio-export-btn"));
    expect(screen.getByTestId("export-download-json-btn")).toHaveTextContent("下载单条标注 JSON");
    expect(screen.getByTestId("export-open-studio-btn")).toHaveTextContent("前往导出中心 →");

    // Open ShortcutsModal in Chinese
    fireEvent.click(screen.getByTestId("open-shortcuts-btn"));
    expect(screen.getByTestId("shortcuts-modal")).toHaveTextContent("审核工作台快捷键速查");
    expect(screen.getByTestId("shortcuts-modal")).toHaveTextContent("播放 / 暂停");
  });

  it("enforces RBAC on interactive header task badge: Reviewer only reassigns non-frozen _default without AI button, Lead reassigns any with frozen confirm and 1-click Re-run VLM", async () => {
    const taskCatalog = [
      {
        task_id: "pick_and_place",
        title: "Pick and Place",
        description: "",
        goal: "",
        match: [],
        aliases: [],
        grounding_objects_count: 1,
      },
      {
        task_id: "fold_towel",
        title: "Fold Towel",
        description: "",
        goal: "",
        match: [],
        aliases: [],
        grounding_objects_count: 1,
      },
      {
        task_id: "_default",
        title: "Default",
        description: "",
        goal: "",
        match: [],
        aliases: [],
        grounding_objects_count: 0,
      },
    ];

    const patchCalls: Array<{ url: string; body: unknown }> = [];
    const jobCalls: unknown[] = [];

    function mockStudioFetch(epOverrides: Record<string, unknown>) {
      const base = makeResolvedEpisode();
      const ep = {
        ...base,
        ...epOverrides,
        meta: {
          ...base.meta,
          ...((epOverrides.meta as Record<string, unknown> | undefined) ?? {}),
        },
      };
      vi.stubGlobal(
        "fetch",
        vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
          const url = typeof input === "string" ? input : input.toString();
          const method = init?.method ?? "GET";
          if (url === "/api/episodes/tasks") {
            return new Response(JSON.stringify({ tasks: taskCatalog, total: taskCatalog.length }), {
              status: 200,
            });
          }
          if (url.endsWith("/task") && method === "PATCH") {
            const body = JSON.parse(String(init?.body ?? "{}"));
            patchCalls.push({ url, body });
            return new Response(
              JSON.stringify({
                episode_id: ep.episode_id,
                previous_task_id: ep.meta.task_id,
                task_id: body.task_id,
                status: ep.status,
                has_active_run: true,
                re_run_recommended: true,
              }),
              { status: 200 },
            );
          }
          if (url.endsWith("/api/jobs") && method === "POST") {
            const body = JSON.parse(String(init?.body ?? "{}"));
            jobCalls.push(body);
            return new Response(
              JSON.stringify({
                job_id: "job_studio_rerun",
                job_type: "batch_vlm",
                status: "queued",
                progress_done: 0,
                progress_total: 1,
              }),
              { status: 201 },
            );
          }
          if (url.endsWith("/reviews")) {
            return new Response(JSON.stringify({ items: [] }), { status: 200 });
          }
          if (url.endsWith("/runs")) {
            return new Response(JSON.stringify({ runs: [] }), { status: 200 });
          }
          if (url.includes("/api/episodes/demo_ds/ep_001")) {
            return new Response(JSON.stringify(ep), { status: 200 });
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

    // 1. Reviewer on already-classified episode -> read-only badge only
    mockStudioFetch({ status: "vlm_ready", meta: { task_id: "pick_and_place" } });
    const r1 = render(
      <AuthProvider initialAuth={{ token: "rev-tok", username: "rev_1", role: "reviewer" }}>
        <MemoryRouter initialEntries={["/review/demo_ds/ep_001"]}>
          <Routes>
            <Route path="/review/:dataset/:stem" element={<ReviewStudioPage />} />
          </Routes>
        </MemoryRouter>
      </AuthProvider>,
    );
    await waitFor(() => expect(screen.getByTestId("review-studio-page")).toBeInTheDocument());
    expect(screen.getByTestId("studio-task-badge-readonly")).toBeInTheDocument();
    expect(screen.queryByTestId("studio-task-badge-btn")).toBeNull();
    r1.unmount();

    // 2. Reviewer on gt_frozen _default episode -> read-only badge only
    mockStudioFetch({ status: "gt_frozen", meta: { task_id: "_default" } });
    const r2 = render(
      <AuthProvider initialAuth={{ token: "rev-tok", username: "rev_1", role: "reviewer" }}>
        <MemoryRouter initialEntries={["/review/demo_ds/ep_001"]}>
          <Routes>
            <Route path="/review/:dataset/:stem" element={<ReviewStudioPage />} />
          </Routes>
        </MemoryRouter>
      </AuthProvider>,
    );
    await waitFor(() => expect(screen.getByTestId("review-studio-page")).toBeInTheDocument());
    expect(screen.getByTestId("studio-task-badge-readonly")).toBeInTheDocument();
    expect(screen.queryByTestId("studio-task-badge-btn")).toBeNull();
    r2.unmount();

    // 3. Reviewer on non-frozen _default episode -> can reassign to non-_default, NO "+ New Task (AI)"
    mockStudioFetch({ status: "vlm_ready", meta: { task_id: "_default" } });
    const r3 = render(
      <AuthProvider initialAuth={{ token: "rev-tok", username: "rev_1", role: "reviewer" }}>
        <MemoryRouter initialEntries={["/review/demo_ds/ep_001"]}>
          <Routes>
            <Route path="/review/:dataset/:stem" element={<ReviewStudioPage />} />
          </Routes>
        </MemoryRouter>
      </AuthProvider>,
    );
    await waitFor(() => expect(screen.getByTestId("studio-task-badge-btn")).toBeInTheDocument());
    fireEvent.click(screen.getByTestId("studio-task-badge-btn"));
    expect(screen.getByTestId("studio-task-popover")).toBeInTheDocument();
    expect(screen.queryByTestId("studio-new-task-ai-btn")).toBeNull();

    const revSelect = screen.getByTestId("studio-task-select") as HTMLSelectElement;
    expect(Array.from(revSelect.options).some((o) => o.value === "_default")).toBe(false);
    fireEvent.change(revSelect, { target: { value: "fold_towel" } });
    fireEvent.click(screen.getByTestId("studio-task-save-btn"));

    await waitFor(() => expect(patchCalls).toHaveLength(1));
    expect(patchCalls[0]).toEqual({
      url: "/api/episodes/demo_ds/ep_001/task",
      body: { task_id: "fold_towel" },
    });
    // Reviewer should not see a Re-run VLM button even when re_run_recommended is true
    expect(screen.queryByTestId("studio-rerun-vlm-now-btn")).toBeNull();
    r3.unmount();

    // 4. Lead on gt_frozen episode -> requires confirmation -> sends allow_frozen: true -> offers 1-click Re-run VLM
    mockStudioFetch({ status: "gt_frozen", meta: { task_id: "_default" } });
    render(
      <AuthProvider initialAuth={{ token: "lead-tok", username: "lead_1", role: "lead" }}>
        <MemoryRouter initialEntries={["/review/demo_ds/ep_001"]}>
          <Routes>
            <Route path="/review/:dataset/:stem" element={<ReviewStudioPage />} />
          </Routes>
        </MemoryRouter>
      </AuthProvider>,
    );
    await waitFor(() => expect(screen.getByTestId("studio-task-badge-btn")).toBeInTheDocument());
    fireEvent.click(screen.getByTestId("studio-task-badge-btn"));
    expect(screen.getByTestId("studio-new-task-ai-btn")).toBeInTheDocument();

    fireEvent.change(screen.getByTestId("studio-task-select"), {
      target: { value: "pick_and_place" },
    });
    fireEvent.click(screen.getByTestId("studio-task-save-btn"));

    expect(await screen.findByTestId("studio-frozen-task-confirm-modal")).toBeInTheDocument();
    expect(patchCalls).toHaveLength(1);
    fireEvent.click(screen.getByTestId("studio-confirm-frozen-task-btn"));

    await waitFor(() => expect(patchCalls).toHaveLength(2));
    expect(patchCalls[1]).toEqual({
      url: "/api/episodes/demo_ds/ep_001/task",
      body: { task_id: "pick_and_place", allow_frozen: true },
    });

    expect(await screen.findByTestId("studio-rerun-vlm-banner")).toBeInTheDocument();
    fireEvent.click(screen.getByTestId("studio-rerun-vlm-now-btn"));
    await waitFor(() =>
      expect(jobCalls).toContainEqual({
        job_type: "batch_vlm",
        episode_ids: ["demo_ds/ep_001"],
        fresh: true,
      }),
    );
  });
});
