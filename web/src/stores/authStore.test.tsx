import React from "react";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { configureApiCredentialsProvider, listEpisodes } from "../api/client";
import { AppRoutes } from "../App";
import { makeEpisodeListItem } from "../test/fixtures";
import { AuthProvider, useAuth } from "./authStore";

const AuthProbeHarness: React.FC = () => {
  const { role, token, username, isLead, setCredentials } = useAuth();
  return (
    <div>
      <span data-testid="probe-role">{role}</span>
      <span data-testid="probe-is-lead">{String(isLead)}</span>
      <span data-testid="probe-token">{token}</span>
      <span data-testid="probe-user">{username}</span>
      <button
        type="button"
        data-testid="try-lead-btn"
        onClick={() => {
          void setCredentials({ token: "rev-secret", username: "rev_1", role: "lead" }, true);
        }}
      >
        TryLead
      </button>
    </div>
  );
};

describe("authStore", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.localStorage.clear();
    window.history.replaceState({}, "", "/");
    configureApiCredentialsProvider(() => ({ token: "", username: "reviewer" }));
  });

  it("forces reviewer when /api/auth/me or /api/lab/labels returns 403 despite role='lead' input", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.includes("/api/auth/me")) {
          return new Response(JSON.stringify({ detail: "Forbidden" }), { status: 403 });
        }
        if (url.includes("/api/lab/labels")) {
          return new Response(JSON.stringify({ detail: "Lead role required" }), { status: 403 });
        }
        return new Response(JSON.stringify({}), { status: 200 });
      }),
    );

    render(
      <AuthProvider initialAuth={{ token: "", username: "rev_1", role: "reviewer" }}>
        <AuthProbeHarness />
      </AuthProvider>,
    );

    fireEvent.click(screen.getByTestId("try-lead-btn"));
    await waitFor(() => expect(screen.getByTestId("probe-token")).toHaveTextContent("rev-secret"));
    expect(screen.getByTestId("probe-role")).toHaveTextContent("reviewer");
    expect(screen.getByTestId("probe-is-lead")).toHaveTextContent("false");
  });

  it("probes /api/auth/me on mount when token is present in URL and derives authoritative role", async () => {
    window.history.replaceState({}, "", "/?token=opaque-secret-tok&role=lead");
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.includes("/api/auth/me")) {
          return new Response(JSON.stringify({ role: "reviewer", user: "reviewer" }), { status: 200 });
        }
        return new Response(JSON.stringify({}), { status: 200 });
      }),
    );

    render(
      <AuthProvider>
        <AuthProbeHarness />
      </AuthProvider>,
    );

    await waitFor(() => expect(screen.getByTestId("probe-role")).toHaveTextContent("reviewer"));
    expect(screen.getByTestId("probe-is-lead")).toHaveTextContent("false");
  });

  it("sends X-Auth-Token from localStorage on first QueuePage mount inside React.StrictMode without opening AuthModal", async () => {
    window.localStorage.setItem(
      "robo_vlm_qc.auth",
      JSON.stringify({ token: "saved-lead-tok", username: "lead_1", role: "lead" }),
    );
    const episodeCalls: Array<{ token: string | null; user: string | null }> = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        const headers = new Headers(init?.headers);
        if (url.includes("/api/auth/me")) {
          return new Response(JSON.stringify({ role: "lead", user: "lead_1" }), { status: 200 });
        }
        if (url.includes("/api/episodes")) {
          const token = headers.get("X-Auth-Token");
          const user = headers.get("X-User");
          episodeCalls.push({ token, user });
          if (token !== "saved-lead-tok") {
            return new Response(JSON.stringify({ detail: "Unauthorized" }), { status: 401 });
          }
          return new Response(
            JSON.stringify([makeEpisodeListItem({ episode_id: "ds/ep1", status: "vlm_ready" })]),
            { status: 200 },
          );
        }
        return new Response(JSON.stringify({}), { status: 200 });
      }),
    );

    render(
      <React.StrictMode>
        <AuthProvider>
          <MemoryRouter initialEntries={["/"]}>
            <AppRoutes />
          </MemoryRouter>
        </AuthProvider>
      </React.StrictMode>,
    );

    await waitFor(() => expect(screen.getByTestId("episode-row-ds/ep1")).toBeInTheDocument());
    expect(episodeCalls.length).toBeGreaterThanOrEqual(1);
    expect(episodeCalls[0].token).toBe("saved-lead-tok");
    expect(episodeCalls[0].user).toBe("lead_1");
    expect(screen.queryByTestId("auth-modal")).toBeNull();
  });

  it("sends X-Auth-Token from ?token= URL param on first QueuePage mount without opening AuthModal", async () => {
    window.history.replaceState({}, "", "/?token=url-rev-tok&user=rev_url");
    const episodeCalls: Array<{ token: string | null; user: string | null }> = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        const headers = new Headers(init?.headers);
        if (url.includes("/api/auth/me")) {
          return new Response(JSON.stringify({ role: "reviewer", user: "rev_url" }), { status: 200 });
        }
        if (url.includes("/api/episodes")) {
          const token = headers.get("X-Auth-Token");
          const user = headers.get("X-User");
          episodeCalls.push({ token, user });
          if (token !== "url-rev-tok") {
            return new Response(JSON.stringify({ detail: "Unauthorized" }), { status: 401 });
          }
          return new Response(
            JSON.stringify([makeEpisodeListItem({ episode_id: "ds/ep1", status: "vlm_ready" })]),
            { status: 200 },
          );
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

    await waitFor(() => expect(screen.getByTestId("episode-row-ds/ep1")).toBeInTheDocument());
    expect(episodeCalls.length).toBeGreaterThanOrEqual(1);
    expect(episodeCalls[0].token).toBe("url-rev-tok");
    expect(episodeCalls[0].user).toBe("rev_url");
    expect(screen.queryByTestId("auth-modal")).toBeNull();
  });

  it("synchronously updates credentials provider on setCredentials and logout", async () => {
    const seenTokens: Array<string | null> = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        const headers = new Headers(init?.headers);
        if (url.includes("/api/episodes")) {
          seenTokens.push(headers.get("X-Auth-Token"));
          return new Response(JSON.stringify([]), { status: 200 });
        }
        return new Response(JSON.stringify({}), { status: 200 });
      }),
    );

    let ctxRef: ReturnType<typeof useAuth> | null = null;
    const Harness: React.FC = () => {
      ctxRef = useAuth();
      return <span data-testid="sync-token">{ctxRef.token}</span>;
    };

    render(
      <AuthProvider initialAuth={{ token: "", username: "reviewer_1", role: "reviewer" }}>
        <Harness />
      </AuthProvider>,
    );

    await act(async () => {
      await ctxRef!.setCredentials({ token: "fresh-tok", username: "lead_2", role: "lead" }, false);
      await listEpisodes();
    });
    expect(seenTokens[0]).toBe("fresh-tok");

    act(() => {
      ctxRef!.logout();
    });
    await listEpisodes();
    expect(seenTokens[1]).toBeNull();
  });

  it("does not allow in-flight setCredentials(..., true) to overwrite a concurrent setCredentials(..., false)", async () => {
    let resolveSlowMe!: (res: Response) => void;
    const slowMePromise = new Promise<Response>((resolve) => {
      resolveSlowMe = resolve;
    });
    const seenTokens: Array<string | null> = [];
    let meCalls = 0;

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        const headers = new Headers(init?.headers);
        if (url.includes("/api/auth/me")) {
          meCalls += 1;
          return slowMePromise;
        }
        if (url.includes("/api/episodes")) {
          seenTokens.push(headers.get("X-Auth-Token"));
          return new Response(JSON.stringify([]), { status: 200 });
        }
        return new Response(JSON.stringify({}), { status: 200 });
      }),
    );

    let ctxRef: ReturnType<typeof useAuth> | null = null;
    const Harness: React.FC = () => {
      ctxRef = useAuth();
      return (
        <div>
          <span data-testid="race-token">{ctxRef.token}</span>
          <span data-testid="race-user">{ctxRef.username}</span>
          <span data-testid="race-role">{ctxRef.role}</span>
        </div>
      );
    };

    render(
      <AuthProvider initialAuth={{ token: "", username: "reviewer_1", role: "reviewer" }}>
        <Harness />
      </AuthProvider>,
    );

    let slowPromise!: Promise<void>;
    act(() => {
      slowPromise = ctxRef!.setCredentials(
        { token: "tok_1_slow", username: "slow_user", role: "lead" },
        true,
      );
    });
    await waitFor(() => expect(meCalls).toBe(1));

    await act(async () => {
      await ctxRef!.setCredentials(
        { token: "tok_2_fast", username: "fast_user", role: "reviewer" },
        false,
      );
    });
    expect(screen.getByTestId("race-token")).toHaveTextContent("tok_2_fast");
    expect(screen.getByTestId("race-user")).toHaveTextContent("fast_user");
    expect(screen.getByTestId("race-role")).toHaveTextContent("reviewer");

    await act(async () => {
      resolveSlowMe(new Response(JSON.stringify({ role: "lead", user: "slow_user" }), { status: 200 }));
      await slowPromise;
    });

    await listEpisodes();
    expect(screen.getByTestId("race-token")).toHaveTextContent("tok_2_fast");
    expect(screen.getByTestId("race-user")).toHaveTextContent("fast_user");
    expect(screen.getByTestId("race-role")).toHaveTextContent("reviewer");
    expect(seenTokens[seenTokens.length - 1]).toBe("tok_2_fast");
  });

  it("does not allow in-flight setCredentials(..., true) to resurrect session after concurrent logout()", async () => {
    let resolveSlowMe!: (res: Response) => void;
    const slowMePromise = new Promise<Response>((resolve) => {
      resolveSlowMe = resolve;
    });
    const seenTokens: Array<string | null> = [];
    let meCalls = 0;

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        const headers = new Headers(init?.headers);
        if (url.includes("/api/auth/me")) {
          meCalls += 1;
          return slowMePromise;
        }
        if (url.includes("/api/episodes")) {
          seenTokens.push(headers.get("X-Auth-Token"));
          return new Response(JSON.stringify([]), { status: 200 });
        }
        return new Response(JSON.stringify({}), { status: 200 });
      }),
    );

    let ctxRef: ReturnType<typeof useAuth> | null = null;
    const Harness: React.FC = () => {
      ctxRef = useAuth();
      return (
        <div>
          <span data-testid="logout-race-token">{ctxRef.token}</span>
          <span data-testid="logout-race-role">{ctxRef.role}</span>
        </div>
      );
    };

    render(
      <AuthProvider initialAuth={{ token: "", username: "reviewer_1", role: "reviewer" }}>
        <Harness />
      </AuthProvider>,
    );

    let slowPromise!: Promise<void>;
    act(() => {
      slowPromise = ctxRef!.setCredentials(
        { token: "tok_1_slow", username: "slow_user", role: "lead" },
        true,
      );
    });
    await waitFor(() => expect(meCalls).toBe(1));

    act(() => {
      ctxRef!.logout();
    });
    expect(screen.getByTestId("logout-race-token")).toBeEmptyDOMElement();

    await act(async () => {
      resolveSlowMe(new Response(JSON.stringify({ role: "lead", user: "slow_user" }), { status: 200 }));
      await slowPromise;
    });

    await listEpisodes();
    expect(screen.getByTestId("logout-race-token")).toBeEmptyDOMElement();
    expect(screen.getByTestId("logout-race-role")).toHaveTextContent("reviewer");
    expect(seenTokens[seenTokens.length - 1]).toBeNull();
  });

  it("syncs authRef.current when mount /api/auth/me probe updates role before subsequent setCredentials({ username }, false)", async () => {
    window.history.replaceState({}, "", "/?token=mount-lead-tok");
    let resolveMe!: (res: Response) => void;
    const mePromise = new Promise<Response>((resolve) => {
      resolveMe = resolve;
    });
    let meCalls = 0;

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.includes("/api/auth/me")) {
          meCalls += 1;
          if (meCalls === 1) {
            return mePromise;
          }
          // Subsequent probe for renamed_lead stays pending so role="lead" must come from authRef.current
          return new Promise<Response>(() => {});
        }
        return new Response(JSON.stringify({}), { status: 200 });
      }),
    );

    let ctxRef: ReturnType<typeof useAuth> | null = null;
    const Harness: React.FC = () => {
      ctxRef = useAuth();
      return (
        <div>
          <span data-testid="mount-sync-role">{ctxRef.role}</span>
          <span data-testid="mount-sync-user">{ctxRef.username}</span>
        </div>
      );
    };

    render(
      <AuthProvider>
        <Harness />
      </AuthProvider>,
    );

    await waitFor(() => expect(meCalls).toBe(1));

    await act(async () => {
      resolveMe(new Response(JSON.stringify({ role: "lead", user: "mount_lead" }), { status: 200 }));
      for (let i = 0; i < 10; i += 1) {
        await Promise.resolve();
      }
      await ctxRef!.setCredentials({ username: "renamed_lead" }, false);
    });

    expect(screen.getByTestId("mount-sync-role")).toHaveTextContent("lead");
    expect(screen.getByTestId("mount-sync-user")).toHaveTextContent("renamed_lead");
  });

  it("syncs AuthModal local inputs when opened after mount role probe updates auth state", async () => {
    window.history.replaceState({}, "", "/?token=url-lead-tok&user=alice");
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.includes("/api/auth/me")) {
          return new Response(JSON.stringify({ role: "lead", user: "alice" }), { status: 200 });
        }
        if (url.includes("/api/episodes")) {
          return new Response(JSON.stringify([]), { status: 200 });
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

    await waitFor(() => expect(screen.getByTestId("current-role-badge")).toHaveTextContent("Lead: alice"));

    fireEvent.click(screen.getByTestId("open-auth-modal"));
    const modal = screen.getByTestId("auth-modal");
    const tokenInput = modal.querySelector('input[type="password"]') as HTMLInputElement;
    const userInput = modal.querySelector('input[type="text"]') as HTMLInputElement;
    const roleSelect = modal.querySelector("select") as HTMLSelectElement;

    expect(tokenInput.value).toBe("url-lead-tok");
    expect(userInput.value).toBe("alice");
    expect(roleSelect.value).toBe("lead");
  });

  it("auto-probes /api/auth/me in --no-auth mode with empty token, adopts Lead role, hydrates jobs and connects SSE without AuthModal", async () => {
    const sseUrls: string[] = [];
    class MockEventSource {
      url: string;
      onopen: (() => void) | null = null;
      onmessage: ((evt: MessageEvent) => void) | null = null;
      onerror: (() => void) | null = null;
      close = vi.fn();
      constructor(url: string) {
        this.url = url;
        sseUrls.push(url);
      }
    }
    vi.stubGlobal("EventSource", MockEventSource);

    const requestedUrls: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        requestedUrls.push(url);
        if (url.includes("/api/auth/me")) {
          return new Response(
            JSON.stringify({ role: "lead", user: "reviewer_1", no_auth: true }),
            { status: 200 },
          );
        }
        if (url.startsWith("/api/jobs")) {
          return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200 });
        }
        if (url.includes("/api/episodes")) {
          return new Response(
            JSON.stringify({
              items: [makeEpisodeListItem({ episode_id: "ds/no_auth_ep1", status: "ingested" })],
              total: 1,
              status_counts: { ingested: 1 },
            }),
            { status: 200 },
          );
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

    await waitFor(() =>
      expect(screen.getByTestId("current-role-badge")).toHaveTextContent("Lead: reviewer_1"),
    );
    await waitFor(() =>
      expect(screen.getByTestId("episode-row-ds/no_auth_ep1")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("lead-status-strip")).toBeInTheDocument();
    expect(screen.getByTestId("live-job-pill")).toBeInTheDocument();
    expect(screen.queryByTestId("auth-modal")).toBeNull();
    // Should never send a Reviewer-scoped status=vlm_ready query during boot
    expect(requestedUrls.some((u) => u.includes("status=vlm_ready"))).toBe(false);
    // Should hydrate jobs and subscribe to /api/events with no ?token= param
    expect(requestedUrls.some((u) => u.includes("/api/jobs?limit=20"))).toBe(true);
    expect(sseUrls.filter((u) => u === "/api/events").length).toBeGreaterThanOrEqual(2);
  });

  it("keeps empty token as reviewer without SSE or /api/jobs when /api/auth/me returns 401 or lacks no_auth=true", async () => {
    const sseUrls: string[] = [];
    class MockEventSource {
      url: string;
      onopen: (() => void) | null = null;
      onmessage: ((evt: MessageEvent) => void) | null = null;
      onerror: (() => void) | null = null;
      close = vi.fn();
      constructor(url: string) {
        this.url = url;
        sseUrls.push(url);
      }
    }
    vi.stubGlobal("EventSource", MockEventSource);

    const requestedUrls: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        requestedUrls.push(url);
        if (url.includes("/api/auth/me")) {
          return new Response(JSON.stringify({ detail: "Unauthorized" }), { status: 401 });
        }
        if (url.includes("/api/episodes")) {
          return new Response(JSON.stringify({ detail: "Unauthorized" }), { status: 401 });
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
    expect(screen.getByTestId("current-role-badge")).toHaveTextContent("Reviewer:");
    expect(requestedUrls.some((u) => u.startsWith("/api/jobs"))).toBe(false);
    expect(sseUrls).toHaveLength(0);
  });
});
