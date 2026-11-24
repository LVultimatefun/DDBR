import React, { useEffect, useState } from "react";
import { BrowserRouter, Link, Route, Routes, useLocation } from "react-router-dom";
import { Activity, KeyRound, Layers, ListChecks, Moon, PackageCheck, Sun } from "lucide-react";
import { JobCenterDrawer } from "./components/JobCenterDrawer";
import { ExportPage } from "./pages/ExportPage";
import { QueuePage } from "./pages/QueuePage";
import { ReviewStudioPage } from "./pages/ReviewStudioPage";
import { VlmLabPage } from "./pages/VlmLabPage";
import { AuthProvider, useAuth } from "./stores/authStore";
import { JobProvider, useJobs } from "./stores/jobStore";
import { ThemeProvider, useTheme } from "./stores/themeStore";
import { GlobalToastContainer, ToastProvider } from "./stores/toastStore";
import type { ReviewRole } from "./api/types";

const AuthModal: React.FC = () => {
  const { authPromptOpen, setAuthPromptOpen, token, username, role, setCredentials } = useAuth();
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
        <h3>Credentials & Role</h3>
        <label className="field-group">
          <span>API Token (X-Auth-Token)</span>
          <input
            type="password"
            value={nextToken}
            onChange={(e) => setNextToken(e.target.value)}
            placeholder="e.g. rev-token or lead-token"
          />
        </label>
        <label className="field-group">
          <span>Username (X-User)</span>
          <input
            type="text"
            value={nextUser}
            onChange={(e) => setNextUser(e.target.value)}
            placeholder="reviewer_1"
          />
        </label>
        <label className="field-group">
          <span>View Role</span>
          <select value={nextRole} onChange={(e) => setNextRole(e.target.value as ReviewRole)}>
            <option value="reviewer">Reviewer (Annotator)</option>
            <option value="lead">Lead (Engineer)</option>
          </select>
        </label>
        <div className="modal-actions">
          <button
            type="button"
            className="secondary-btn compact"
            onClick={() => setAuthPromptOpen(false)}
          >
            Cancel
          </button>
          <button type="submit" className="primary-btn compact">
            Apply Credentials
          </button>
        </div>
      </form>
    </div>
  );
};

const TopNav: React.FC = () => {
  const { isLead, username, role, setAuthPromptOpen } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const { jobs, activeJobs, drawerOpen, setDrawerOpen } = useJobs();
  const loc = useLocation();

  const activeDone = activeJobs.reduce((sum, j) => sum + Math.max(0, j.progress_done ?? 0), 0);
  const activeTotal = activeJobs.reduce((sum, j) => sum + Math.max(0, j.progress_total ?? 0), 0);
  const activePct = activeTotal > 0 ? Math.min(100, Math.round((activeDone / activeTotal) * 100)) : 0;

  return (
    <header className="top-navbar">
      <div className="nav-left">
        <Link className="brand" to="/">
          robo_vlm_qc <span className="brand-tag">Studio</span>
        </Link>
        <nav className="nav-links" aria-label="Primary navigation">
          <Link className={`nav-link ${loc.pathname === "/" ? "active" : ""}`} to="/">
            <ListChecks size={15} /> Work Queue
          </Link>
          {isLead && (
            <>
              <Link
                className={`nav-link ${loc.pathname === "/export" ? "active" : ""}`}
                to="/export"
                data-testid="nav-export"
              >
                <PackageCheck size={15} /> Export Studio
              </Link>
              <Link
                className={`nav-link ${loc.pathname === "/lab" ? "active" : ""}`}
                to="/lab"
                data-testid="nav-lab"
              >
                <Layers size={15} /> VLM Lab
              </Link>
            </>
          )}
        </nav>
      </div>

      <div className="nav-right">
        {isLead && (
          <button
            type="button"
            className={`job-pill ${activeJobs.length > 0 ? "active" : ""}`}
            onClick={() => setDrawerOpen(!drawerOpen)}
            data-testid="live-job-pill"
            aria-label="Open async job center"
          >
            {activeJobs.length > 0 ? (
              <>
                <span className="job-pill-dot pulsing" />
                <span>
                  Running {activeDone}/{activeTotal} ({activePct}%)
                </span>
              </>
            ) : (
              <>
                <Activity size={13} />
                <span>Jobs ({jobs.length})</span>
              </>
            )}
          </button>
        )}
        <button
          type="button"
          className="theme-toggle-btn"
          onClick={toggleTheme}
          data-testid="theme-toggle-btn"
          aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
          title={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
        >
          {theme === "dark" ? <Sun size={14} /> : <Moon size={14} />}
        </button>
        <span className={`role-badge ${role}`} data-testid="current-role-badge">
          {role === "lead" ? "Lead" : "Reviewer"}: {username}
        </span>
        <button
          type="button"
          className="mini-btn"
          onClick={() => setAuthPromptOpen(true)}
          data-testid="open-auth-modal"
        >
          <KeyRound size={13} /> Switch Token / Role
        </button>
      </div>
    </header>
  );
};

export const AppRoutes: React.FC = () => (
  <ThemeProvider>
    <ToastProvider>
      <JobProvider>
        <div className="app-shell">
          <TopNav />
          <AuthModal />
          <main className="app-main">
            <Routes>
              <Route path="/" element={<QueuePage />} />
              <Route path="/review/:dataset/:stem" element={<ReviewStudioPage />} />
              <Route path="/review/*" element={<ReviewStudioPage />} />
              <Route path="/export" element={<ExportPage />} />
              <Route path="/lab" element={<VlmLabPage />} />
              <Route path="*" element={<QueuePage />} />
            </Routes>
          </main>
          <JobCenterDrawer />
          <GlobalToastContainer />
        </div>
      </JobProvider>
    </ToastProvider>
  </ThemeProvider>
);

export const App: React.FC = () => (
  <AuthProvider>
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  </AuthProvider>
);

export default App;
