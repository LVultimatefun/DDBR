import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

export type ThemeMode = "dark" | "light";

const THEME_STORAGE_KEY = "robo_vlm_qc.theme";

function loadInitialTheme(): ThemeMode {
  if (typeof window === "undefined") return "dark";
  try {
    const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (stored === "light" || stored === "dark") {
      return stored;
    }
  } catch {
    // Ignore storage errors in restricted contexts
  }
  return "dark";
}

function applyDocumentTheme(theme: ThemeMode): void {
  if (typeof document === "undefined") return;
  document.documentElement.dataset.theme = theme;
  document.documentElement.style.colorScheme = theme;
}

export interface ThemeContextValue {
  theme: ThemeMode;
  setTheme: (next: ThemeMode) => void;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

export const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [theme, setThemeState] = useState<ThemeMode>(() => loadInitialTheme());

  useEffect(() => {
    applyDocumentTheme(theme);
    try {
      if (typeof window !== "undefined") {
        window.localStorage.setItem(THEME_STORAGE_KEY, theme);
      }
    } catch {
      // Ignore storage errors
    }
  }, [theme]);

  const setTheme = useCallback((next: ThemeMode) => {
    setThemeState(next);
  }, []);

  const toggleTheme = useCallback(() => {
    setThemeState((prev) => (prev === "dark" ? "light" : "dark"));
  }, []);

  const value = useMemo<ThemeContextValue>(
    () => ({ theme, setTheme, toggleTheme }),
    [theme, setTheme, toggleTheme],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
};

export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  const [fallbackTheme, setFallbackTheme] = useState<ThemeMode>(() => loadInitialTheme());

  const setFallback = useCallback((next: ThemeMode) => {
    setFallbackTheme(next);
    applyDocumentTheme(next);
    try {
      if (typeof window !== "undefined") {
        window.localStorage.setItem(THEME_STORAGE_KEY, next);
      }
    } catch {
      // Ignore storage errors
    }
  }, []);

  const toggleFallback = useCallback(() => {
    setFallbackTheme((prev) => {
      const next: ThemeMode = prev === "dark" ? "light" : "dark";
      applyDocumentTheme(next);
      try {
        if (typeof window !== "undefined") {
          window.localStorage.setItem(THEME_STORAGE_KEY, next);
        }
      } catch {
        // Ignore storage errors
      }
      return next;
    });
  }, []);

  if (ctx) return ctx;
  return {
    theme: fallbackTheme,
    setTheme: setFallback,
    toggleTheme: toggleFallback,
  };
}
