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

function loadInitialAuth(): { state: AuthState; needsUrlProbe: boolean } {
  const defaultState: AuthState = {
    token: "",
    username: "reviewer_1",
    role: "reviewer",
  };
  let needsUrlProbe = false;

  if (typeof window === "undefined") return { state: defaultState, needsUrlProbe };

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
    }
  } catch {
    // Ignore storage/URL parsing errors
  }

  return { state: defaultState, needsUrlProbe };
}

export interface AuthProviderProps {
  children: React.ReactNode;
  initialAuth?: Partial<AuthState>;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children, initialAuth }) => {
  const initialBoot = useMemo(() => loadInitialAuth(), []);
  const [auth, setAuth] = useState<AuthState>(() => ({
    ...initialBoot.state,
    ...initialAuth,
  }));
  const [authProbing, setAuthProbing] = useState<boolean>(
    () => Boolean(initialBoot.needsUrlProbe && !initialAuth?.role),
  );
  const [authPromptOpen, setAuthPromptOpen] = useState(false);

  const authRef = useRef<AuthState>(auth);
  authRef.current = auth;
  const authSeqRef = useRef(0);

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
    if (!auth.token || initialAuth?.role) {
      setAuthProbing(false);
      return;
    }
    let cancelled = false;
    const seq = authSeqRef.current;
    void getAuthMe({ token: auth.token, username: auth.username })
      .then((me) => {
        if (cancelled || seq !== authSeqRef.current) return;
        if (me && (me.role === "lead" || me.role === "reviewer")) {
          if (authRef.current.token === auth.token && authRef.current.role !== me.role) {
            authRef.current = { ...authRef.current, role: me.role };
          }
          setAuth((prev) => {
            if (seq !== authSeqRef.current || prev.token !== auth.token || prev.role === me.role) {
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
        if (authRef.current.token === auth.token && authRef.current.role !== "reviewer") {
          authRef.current = { ...authRef.current, role: "reviewer" };
        }
        setAuth((prev) => {
          if (seq !== authSeqRef.current || prev.token !== auth.token || prev.role === "reviewer") {
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
  }, [auth.token, auth.username, initialAuth?.role]);

  const setCredentials = useCallback(
    async (next: Partial<AuthState>, probeRole = false) => {
      const seq = ++authSeqRef.current;
      const merged: AuthState = {
        ...authRef.current,
        ...next,
      };
      if (probeRole) {
        if (!merged.token) {
          merged.role = "reviewer";
        } else {
          try {
            const me = await getAuthMe({ token: merged.token, username: merged.username });
            if (me && (me.role === "lead" || me.role === "reviewer")) {
              merged.role = me.role;
            } else {
              await getLabLabels({ token: merged.token, username: merged.username });
              merged.role = "lead";
            }
          } catch {
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
  }, [auth, authProbing, authPromptOpen, setCredentials, logout]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used inside AuthProvider");
  }
  return ctx;
}
