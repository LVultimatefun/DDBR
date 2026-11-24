import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { commonEn, commonZh } from "../i18n/common";
import { labEn, labZh } from "../i18n/lab";
import { studioEn, studioZh } from "../i18n/studio";

export type Locale = "en" | "zh";
export type TranslationVars = Record<string, string | number>;
export type TranslateFn = (key: string, vars?: TranslationVars) => string;

const LOCALE_STORAGE_KEY = "robo_vlm_qc.locale";

const EN_DICT: Record<string, string> = {
  ...commonEn,
  ...studioEn,
  ...labEn,
};

const ZH_DICT: Record<string, string> = {
  ...commonZh,
  ...studioZh,
  ...labZh,
};

export function translate(locale: Locale, key: string, vars?: TranslationVars): string {
  const dict = locale === "zh" ? ZH_DICT : EN_DICT;
  const template = dict[key] ?? EN_DICT[key] ?? key;
  if (!vars) return template;
  return template.replace(/\{(\w+)\}/g, (match, name: string) =>
    Object.prototype.hasOwnProperty.call(vars, name) ? String(vars[name]) : match,
  );
}

function loadInitialLocale(): Locale {
  if (typeof window === "undefined") return "en";
  try {
    const stored = window.localStorage.getItem(LOCALE_STORAGE_KEY);
    if (stored === "zh" || stored === "en") {
      return stored;
    }
  } catch {
    // Ignore storage errors in restricted contexts
  }
  return "en";
}

function applyDocumentLang(locale: Locale): void {
  if (typeof document === "undefined") return;
  document.documentElement.lang = locale === "zh" ? "zh-CN" : "en";
}

export interface I18nContextValue {
  locale: Locale;
  setLocale: (next: Locale) => void;
  toggleLocale: () => void;
  t: TranslateFn;
}

const I18nContext = createContext<I18nContextValue | null>(null);

const FALLBACK_I18N: I18nContextValue = {
  locale: "en",
  setLocale: () => {},
  toggleLocale: () => {},
  t: (key: string, vars?: TranslationVars) => translate("en", key, vars),
};

export interface I18nProviderProps {
  children: React.ReactNode;
  initialLocale?: Locale;
}

const I18nProviderState: React.FC<I18nProviderProps> = ({ children, initialLocale }) => {
  const [locale, setLocaleState] = useState<Locale>(() => initialLocale ?? loadInitialLocale());

  useEffect(() => {
    applyDocumentLang(locale);
    try {
      if (typeof window !== "undefined") {
        window.localStorage.setItem(LOCALE_STORAGE_KEY, locale);
      }
    } catch {
      // Ignore storage errors
    }
  }, [locale]);

  const setLocale = useCallback((next: Locale) => {
    setLocaleState(next);
  }, []);

  const toggleLocale = useCallback(() => {
    setLocaleState((prev) => (prev === "en" ? "zh" : "en"));
  }, []);

  const t = useCallback<TranslateFn>(
    (key: string, vars?: TranslationVars) => translate(locale, key, vars),
    [locale],
  );

  const value = useMemo<I18nContextValue>(
    () => ({ locale, setLocale, toggleLocale, t }),
    [locale, setLocale, toggleLocale, t],
  );

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
};

export const I18nProvider: React.FC<I18nProviderProps> = ({ children, initialLocale }) => {
  const parent = useContext(I18nContext);
  if (parent && initialLocale === undefined) {
    return <>{children}</>;
  }
  return <I18nProviderState initialLocale={initialLocale}>{children}</I18nProviderState>;
};

export function useI18n(): I18nContextValue {
  const ctx = useContext(I18nContext);
  return ctx ?? FALLBACK_I18N;
}
