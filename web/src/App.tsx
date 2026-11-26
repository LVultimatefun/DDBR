import React, { Suspense, useEffect, useState } from "react";
import { BrowserRouter, Link, Route, Routes, useLocation } from "react-router-dom";
import {
  Activity,
  Cpu,
  KeyRound,
  Languages,
  Layers,
  ListChecks,
  Lock,
  Moon,
  PackageCheck,
  Search,
  Sun,
} from "lucide-react";
import { BrandMark } from "./components/BrandMark";
import { CommandPalette } from "./components/CommandPalette";
import { JobCenterDrawer } from "./components/JobCenterDrawer";
import { PageSkeleton } from "./components/SkeletonLoader";
import { VlmSettingsModal } from "./components/VlmSettingsModal";
import { AuthProvider, useAuth } from "./stores/authStore";
import { I18nProvider, useI18n } from "./stores/i18nStore";
import { JobProvider, useJobs } from "./stores/jobStore";
import { ThemeProvider, useTheme } from "./stores/themeStore";
import { GlobalToastContainer, ToastProvider } from "./stores/toastStore";
import type { ReviewRole } from "./api/types";

const QueuePage = React.lazy(() =>
  import("./pages/QueuePage").then((m) => ({ default: m.QueuePage })),
);
const ReviewStudioPage = React.lazy(() =>
  import("./pages/ReviewStudioPage").then((m) => ({ default: m.ReviewStudioPage })),
);
const ExportPage = React.lazy(() =>
  import("./pages/ExportPage").then((m) => ({ default: m.ExportPage })),
);
const VlmLabPage = React.lazy(() =>
  import("./pages/VlmLabPage").then((m) => ({ default: m.VlmLabPage })),
);

const AuthModal: React.FC = () => {
  const { authPromptOpen, setAuthPromptOpen, token, username, role, setCredentials } = useAuth();
  const { t } = useI18n();
  const [nextToken, setNextToken] = useState(token);
  const [nextUser, setNextUser] = useState(username);
  const [nextRole, setNextRole] = useState<ReviewRole>(role);
  const [prevOpen, setPrevOpen] = useState(authPromptOpen);

  if (authPromptOpen !== prevOpen) {
    setPrevOpen(authPromptOpen);
    if (authPromptOpen) {
      setNextToken(token);
      setNextUser(username);
      setNextRole(role);
    }
  }

  useEffect(() => {
    if (!authPromptOpen) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        setAuthPromptOpen(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [authPromptOpen, setAuthPromptOpen]);

  if (!authPromptOpen) return null;

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    await setCredentials(
      { token: nextToken.trim(), username: nextUser.trim() || "reviewer_1", role: nextRole },
      true,
    );
    setAuthPromptOpen(false);
  };

  return (
    <div
      className="modal-backdrop"
      data-testid="auth-modal"
      onClick={(e) => {
        if (e.target === e.currentTarget) setAuthPromptOpen(false);
      }}
    >
      <form className="modal-card" onSubmit={(e) => void handleSave(e)}>
        <h3>{t("auth.title")}</h3>
        <label className="field-group">
          <span>{t("auth.apiToken")}</span>
          <input
            type="password"
            value={nextToken}
            onChange={(e) => setNextToken(e.target.value)}
            placeholder={t("auth.apiTokenPlaceholder")}
          />
        </label>
        <label className="field-group">
          <span>{t("auth.username")}</span>
          <input
            type="text"
            value={nextUser}
            onChange={(e) => setNextUser(e.target.value)}
            placeholder="reviewer_1"
          />
        </label>
        <label className="field-group">
          <span>{t("auth.viewRole")}</span>
          <select value={nextRole} onChange={(e) => setNextRole(e.target.value as ReviewRole)}>
            <option value="reviewer">{t("auth.roleReviewerOption")}</option>
            <option value="lead">{t("auth.roleLeadOption")}</option>
          </select>
        </label>
        <div className="modal-actions">
          <button
            type="button"
            className="secondary-btn compact"
            onClick={() => setAuthPromptOpen(false)}
          >
            {t("common.cancel")}
          </button>
          <button type="submit" className="primary-btn compact">
            {t("auth.apply")}
          </button>
        </div>
      </form>
    </div>
  );
};

const TopNav: React.FC = () => {
  const { isLead, username, role, setAuthPromptOpen } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const { locale, toggleLocale, t } = useI18n();
  const { jobs, activeJobs, drawerOpen, setDrawerOpen } = useJobs();
  const loc = useLocation();
  const [vlmSettingsOpen, setVlmSettingsOpen] = useState(false);
  const [cmdPaletteOpen, setCmdPaletteOpen] = useState(false);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setCmdPaletteOpen((prev) => !prev);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const activeDone = activeJobs.reduce((sum, j) => sum + Math.max(0, j.progress_done ?? 0), 0);
  const activeTotal = activeJobs.reduce((sum, j) => sum + Math.max(0, j.progress_total ?? 0), 0);
  const activePct = activeTotal > 0 ? Math.min(100, Math.round((activeDone / activeTotal) * 100)) : 0;

  return (
    <>
      <header className="top-navbar">
        <div className="nav-left">
          <Link className="brand" to="/">
            <BrandMark size={22} />
            <span>robo_vlm_qc</span>
            <span className="brand-tag">Studio</span>
          </Link>
          <nav className="nav-links" aria-label={t("nav.primaryLabel")}>
            <Link className={`nav-link ${loc.pathname === "/" ? "active" : ""}`} to="/">
              <ListChecks size={15} /> {t("nav.workQueue")}
            </Link>
            {isLead ? (
              <>
                <Link
                  className={`nav-link ${loc.pathname === "/export" ? "active" : ""}`}
                  to="/export"
                  data-testid="nav-export"
                >
                  <PackageCheck size={15} /> {t("nav.exportStudio")}
                </Link>
                <Link
                  className={`nav-link ${loc.pathname === "/lab" ? "active" : ""}`}
                  to="/lab"
                  data-testid="nav-lab"
                >
                  <Layers size={15} /> {t("nav.vlmLab")}
                </Link>
              </>
            ) : (
              <button
                type="button"
                className="nav-link nav-link-locked"
                data-testid="nav-export-locked"
                title={t("nav.exportLockedTitle")}
                onClick={() => setAuthPromptOpen(true)}
              >
                <PackageCheck size={15} /> {t("nav.exportStudio")} <Lock size={12} />
              </button>
            )}
          </nav>
        </div>

        <div className="nav-right">
          <button
            type="button"
            className="cmd-trigger-pill"
            onClick={() => setCmdPaletteOpen(true)}
            data-testid="open-cmd-palette-btn"
            aria-label={t("nav.commandPaletteAria")}
            title={t("nav.commandPaletteAria")}
          >
            <Search size={13} />
            <span className="hide-md">{t("nav.commandPalette")}</span>
            <kbd className="cmd-kbd">{t("nav.commandPaletteShortcut")}</kbd>
          </button>

          {isLead && (
            <div className="nav-utility-group">
              <button
                type="button"
                className="mini-btn"
                onClick={() => setVlmSettingsOpen(true)}
                data-testid="open-vlm-settings-btn"
                title={t("nav.modelApiTitle")}
              >
                <Cpu size={13} /> {t("nav.modelApi")}
              </button>
              <button
                type="button"
                className={`job-pill tabular-nums ${activeJobs.length > 0 ? "active" : ""}`}
                onClick={() => setDrawerOpen(!drawerOpen)}
                data-testid="live-job-pill"
                aria-label={t("nav.openJobCenter")}
              >
                {activeJobs.length > 0 ? (
                  <>
                    <span className="job-pill-dot pulsing" />
                    <span>
                      {t("nav.jobsRunning", {
                        done: activeDone,
                        total: activeTotal,
                        pct: activePct,
                      })}
                    </span>
                  </>
                ) : (
                  <>
                    <Activity size={13} />
                    <span>{t("nav.jobsCount", { count: jobs.length })}</span>
                  </>
                )}
              </button>
            </div>
          )}

          <div className="nav-utility-group">
            <button
              type="button"
              className="locale-toggle-btn"
              onClick={toggleLocale}
              data-testid="locale-toggle-btn"
              aria-label={locale === "en" ? t("nav.switchToZh") : t("nav.switchToEn")}
              title={locale === "en" ? t("nav.switchToZh") : t("nav.switchToEn")}
            >
              <Languages size={14} />
              <span>{locale === "en" ? "中文" : "EN"}</span>
            </button>
            <button
              type="button"
              className="theme-toggle-btn"
              onClick={toggleTheme}
              data-testid="theme-toggle-btn"
              aria-label={theme === "dark" ? t("nav.switchToLight") : t("nav.switchToDark")}
              title={theme === "dark" ? t("nav.switchToLight") : t("nav.switchToDark")}
            >
              {theme === "dark" ? <Sun size={14} /> : <Moon size={14} />}
            </button>
            <span className={`role-badge ${role}`} data-testid="current-role-badge">
              {role === "lead" ? t("nav.roleLead") : t("nav.roleReviewer")}: {username}
            </span>
            <button
              type="button"
              className="mini-btn"
              onClick={() => setAuthPromptOpen(true)}
              data-testid="open-auth-modal"
            >
              <KeyRound size={13} /> <span className="hide-sm">{t("nav.switchTokenRole")}</span>
            </button>
          </div>
        </div>
      </header>

      <CommandPalette
        open={cmdPaletteOpen}
        onClose={() => setCmdPaletteOpen(false)}
        onOpenVlmSettings={() => setVlmSettingsOpen(true)}
      />

      {isLead && (
        <VlmSettingsModal open={vlmSettingsOpen} onClose={() => setVlmSettingsOpen(false)} />
      )}
    </>
  );
};

export const AppRoutes: React.FC = () => (
  <I18nProvider>
    <ThemeProvider>
      <ToastProvider>
        <JobProvider>
          <div className="app-shell">
            <TopNav />
            <AuthModal />
            <main className="app-main">
              <Suspense fallback={<PageSkeleton />}>
                <Routes>
                  <Route path="/" element={<QueuePage />} />
                  <Route path="/review/:dataset/:stem" element={<ReviewStudioPage />} />
                  <Route path="/review/*" element={<ReviewStudioPage />} />
                  <Route path="/export" element={<ExportPage />} />
                  <Route path="/lab" element={<VlmLabPage />} />
                  <Route path="*" element={<QueuePage />} />
                </Routes>
              </Suspense>
            </main>
            <JobCenterDrawer />
            <GlobalToastContainer />
          </div>
        </JobProvider>
      </ToastProvider>
    </ThemeProvider>
  </I18nProvider>
);

export const App: React.FC = () => (
  <AuthProvider>
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  </AuthProvider>
);

export default App;
