import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AuthProvider } from "../stores/authStore";
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
});
