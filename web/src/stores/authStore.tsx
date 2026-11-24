import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { type AuthCredentials, configureApiCredentialsProvider, getAuthMe, getLabLabels } from "../api/client";
import type { ReviewRole } from "../api/types";

const STORAGE_KEY = "robo_vlm_qc.auth";

export interface AuthState {
  token: string;
  username: string;
  role: ReviewRole;
}

export interface AuthContextValue extends AuthState {
  isLead: boolean;
  isReviewer: boolean;
  noAuth: boolean;
  authProbing: boolean;
  canApprove: boolean;
  canFreezeGt: boolean;
  canAccessLab: boolean;
  canAccessExports: boolean;
  canViewDiff: boolean;
  canManageJobs: boolean;
  authPromptOpen: boolean;
  setAuthPromptOpen: (open: boolean) => void;
  setCredentials: (next: Partial<AuthState>, probeRole?: boolean) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

function loadInitialAuth(): {
  state: AuthState;
  needsUrlProbe: boolean;
  explicitReviewerRole: boolean;
} {
  const defaultState: AuthState = {
    token: "",
    username: "reviewer_1",
    role: "reviewer",
  };
  let needsUrlProbe = false;
  let explicitReviewerRole = false;

  if (typeof window === "undefined") {
    return { state: defaultState, needsUrlProbe, explicitReviewerRole };
  }

  try {
    let storedToken = "";
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<AuthState>;
      if (parsed.token !== undefined) {
        defaultState.token = parsed.token;
        storedToken = parsed.token;
      }
      if (parsed.username) defaultState.username = parsed.username;
      if (parsed.role === "lead" || parsed.role === "reviewer") defaultState.role = parsed.role;
    }
    const url = new URL(window.location.href);
    const urlToken = url.searchParams.get("token");
    const urlUser = url.searchParams.get("user");
    const urlRole = url.searchParams.get("role");
    if (urlToken) {
      defaultState.token = urlToken;
      if (urlToken !== storedToken) {
        defaultState.role = "reviewer";
        needsUrlProbe = true;
      }
    }
    if (urlUser) {
      defaultState.username = urlUser;
    }
    if (urlRole === "reviewer") {
      defaultState.role = "reviewer";
      explicitReviewerRole = true;
    }
  } catch {
    // Ignore storage/URL parsing errors
  }

  return { state: defaultState, needsUrlProbe, explicitReviewerRole };
}

export interface AuthProviderProps {
  children: React.ReactNode;
  initialAuth?: Partial<AuthState> & { noAuth?: boolean };
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children, initialAuth }) => {
  const initialBoot = useMemo(() => loadInitialAuth(), []);
  const hasExplicitInitialRole = Boolean(initialAuth?.role);
  const [auth, setAuth] = useState<AuthState>(() => ({
    token: initialAuth?.token ?? initialBoot.state.token,
    username: initialAuth?.username ?? initialBoot.state.username,
    role: initialAuth?.role ?? initialBoot.state.role,
  }));
  const [noAuth, setNoAuth] = useState<boolean>(() => Boolean(initialAuth?.noAuth));
  const [authProbing, setAuthProbing] = useState<boolean>(
    () => Boolean(!hasExplicitInitialRole && (initialBoot.needsUrlProbe || !initialBoot.state.token)),
  );
  const [authPromptOpen, setAuthPromptOpen] = useState(false);

  const authRef = useRef<AuthState>(auth);
  authRef.current = auth;
  const authSeqRef = useRef(0);
  const manualNoAuthReviewerRef = useRef<boolean>(initialBoot.explicitReviewerRole);

  const credentialsProviderRef = useRef<() => AuthCredentials>(() => ({
    token: authRef.current.token,
    username: authRef.current.username,
  }));
  configureApiCredentialsProvider(credentialsProviderRef.current);

  useEffect(() => {
    try {
      if (typeof window !== "undefined") {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(auth));
      }
    } catch {
      // Ignore storage quota errors
    }
  }, [auth]);

  useEffect(() => {
    if (hasExplicitInitialRole) {
      setAuthProbing(false);
      return;
    }
    let cancelled = false;
    const seq = authSeqRef.current;
    const probingToken = auth.token;
    const probingUser = auth.username;
    void getAuthMe({ token: probingToken, username: probingUser })
      .then((me) => {
        if (cancelled || seq !== authSeqRef.current) return;
        const isServerNoAuth = Boolean(me?.no_auth);
        setNoAuth(isServerNoAuth);

        if (!probingToken) {
          if (!isServerNoAuth) {
            if (authRef.current.token === "" && authRef.current.role !== "reviewer") {
              authRef.current = { ...authRef.current, role: "reviewer" };
            }
            setAuth((prev) => {
              if (seq !== authSeqRef.current || prev.token !== "" || prev.role === "reviewer") {
                return prev;
              }
              const updated: AuthState = { ...prev, role: "reviewer" };
              authRef.current = updated;
              return updated;
            });
            return;
          }
          const targetRole: ReviewRole = manualNoAuthReviewerRef.current
            ? "reviewer"
            : me.role === "lead" || me.role === "reviewer"
              ? me.role
              : "lead";
          if (authRef.current.token === "" && authRef.current.role !== targetRole) {
            authRef.current = { ...authRef.current, role: targetRole };
          }
          setAuth((prev) => {
            if (seq !== authSeqRef.current || prev.token !== "" || prev.role === targetRole) {
              return prev;
            }
            const updated: AuthState = { ...prev, role: targetRole };
            authRef.current = updated;
            return updated;
          });
          return;
        }

        if (me && (me.role === "lead" || me.role === "reviewer")) {
          if (authRef.current.token === probingToken && authRef.current.role !== me.role) {
            authRef.current = { ...authRef.current, role: me.role };
          }
          setAuth((prev) => {
            if (seq !== authSeqRef.current || prev.token !== probingToken || prev.role === me.role) {
              return prev;
            }
            const updated: AuthState = { ...prev, role: me.role };
            authRef.current = updated;
            return updated;
          });
        }
      })
      .catch(() => {
        if (cancelled || seq !== authSeqRef.current) return;
        setNoAuth(false);
        if (authRef.current.token === probingToken && authRef.current.role !== "reviewer") {
          authRef.current = { ...authRef.current, role: "reviewer" };
        }
        setAuth((prev) => {
          if (seq !== authSeqRef.current || prev.token !== probingToken || prev.role === "reviewer") {
            return prev;
          }
          const updated: AuthState = { ...prev, role: "reviewer" };
          authRef.current = updated;
          return updated;
        });
      })
      .finally(() => {
        if (!cancelled) {
          setAuthProbing(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [auth.token, auth.username, hasExplicitInitialRole]);

  const setCredentials = useCallback(
    async (next: Partial<AuthState>, probeRole = false) => {
      const seq = ++authSeqRef.current;
      const merged: AuthState = {
        ...authRef.current,
        ...next,
      };
      if (!merged.token && next.role === "reviewer") {
        manualNoAuthReviewerRef.current = true;
      } else if (merged.token || next.role === "lead") {
        manualNoAuthReviewerRef.current = false;
      }
      if (probeRole) {
        if (!merged.token) {
          try {
            const me = await getAuthMe({ token: "", username: merged.username });
            if (seq !== authSeqRef.current) return;
            if (me && me.no_auth === true) {
              setNoAuth(true);
              if (next.role === "reviewer") {
                manualNoAuthReviewerRef.current = true;
                merged.role = "reviewer";
              } else {
                manualNoAuthReviewerRef.current = false;
                merged.role =
                  me.role === "lead" || me.role === "reviewer" ? me.role : "lead";
              }
            } else {
              setNoAuth(false);
              merged.role = "reviewer";
            }
          } catch {
            if (seq !== authSeqRef.current) return;
            setNoAuth(false);
            merged.role = "reviewer";
          }
        } else {
          try {
            const me = await getAuthMe({ token: merged.token, username: merged.username });
            if (seq !== authSeqRef.current) return;
            setNoAuth(Boolean(me?.no_auth));
            if (me && (me.role === "lead" || me.role === "reviewer")) {
              merged.role = me.role;
            } else {
              await getLabLabels({ token: merged.token, username: merged.username });
              if (seq !== authSeqRef.current) return;
              merged.role = "lead";
            }
          } catch {
            if (seq !== authSeqRef.current) return;
            merged.role = "reviewer";
          }
        }
        if (seq !== authSeqRef.current) return;
      }
      authRef.current = merged;
      setAuth(merged);
    },
    [],
  );

  const logout = useCallback(() => {
    authSeqRef.current += 1;
    manualNoAuthReviewerRef.current = true;
    const nextAuth: AuthState = { token: "", username: "reviewer_1", role: "reviewer" };
    authRef.current = nextAuth;
    setAuth(nextAuth);
  }, []);

  const value = useMemo<AuthContextValue>(() => {
    const isLead = auth.role === "lead";
    return {
      ...auth,
      isLead,
      isReviewer: !isLead,
      noAuth,
      authProbing,
      canApprove: isLead,
      canFreezeGt: isLead,
      canAccessLab: isLead,
      canAccessExports: isLead,
      canViewDiff: isLead,
      canManageJobs: isLead,
      authPromptOpen,
      setAuthPromptOpen,
      setCredentials,
      logout,
    };
  }, [auth, noAuth, authProbing, authPromptOpen, setCredentials, logout]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used inside AuthProvider");
  }
  return ctx;
}
