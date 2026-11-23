import React, { useEffect } from "react";
import { Activity, Download, RefreshCw, X, XCircle } from "lucide-react";
import { exportDownloadUrl } from "../api/client";
import type { JobStatus } from "../api/types";
import { useAuth } from "../stores/authStore";
import { useJobs } from "../stores/jobStore";

function formatJobStatusLabel(status: JobStatus["status"]): string {
  switch (status) {
    case "queued":
      return "Queued";
    case "running":
      return "Running";
    case "completed":
      return "Completed";
    case "failed":
      return "Failed";
    case "cancelled":
      return "Cancelled";
    default:
      return status;
  }
}

export const JobCenterDrawer: React.FC = () => {
  const { isLead, token } = useAuth();
  const { jobs, drawerOpen, setDrawerOpen, refreshJobs, cancelJob, artifactDownloadUrl } =
    useJobs();

  useEffect(() => {
    if (!drawerOpen) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setDrawerOpen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [drawerOpen, setDrawerOpen]);

  if (!isLead || !drawerOpen) return null;

  return (
    <div
      className="drawer-backdrop"
      data-testid="job-center-drawer-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) setDrawerOpen(false);
      }}
    >
      <aside className="drawer-panel" data-testid="job-center-drawer" aria-label="Async Job Center">
        <div className="drawer-header">
          <h3>
            <Activity size={16} /> Async Job Center ({jobs.length})
          </h3>
          <div className="header-actions">
            <button
              type="button"
              className="secondary-btn compact"
              onClick={() => void refreshJobs()}
              data-testid="refresh-job-drawer-btn"
            >
              <RefreshCw size={13} /> Refresh
            </button>
            <button
              type="button"
              className="mini-btn"
              onClick={() => setDrawerOpen(false)}
              data-testid="close-job-drawer-btn"
              aria-label="Close job drawer"
            >
              <X size={14} /> Close
            </button>
          </div>
        </div>

        <div className="drawer-body">
          {jobs.length === 0 ? (
            <div className="empty-cell">No async jobs recorded yet.</div>
          ) : (
            jobs.map((job) => {
              const total = Math.max(0, job.progress_total ?? 0);
              const done = Math.max(0, job.progress_done ?? 0);
              const pct =
                total > 0
                  ? Math.min(100, Math.round((done / total) * 100))
                  : job.status === "completed"
                    ? 100
                    : 0;
              const isActive = job.status === "queued" || job.status === "running";
              const artifacts = job.summary?.artifacts ?? [];
              const results = job.summary?.results ?? [];
              const errors = job.summary?.errors ?? [];
              const exportId = job.summary?.export_id ?? job.summary?.manifest?.export_id ?? null;

              return (
                <div key={job.job_id} className="job-card" data-testid={`drawer-job-${job.job_id}`}>
                  <div className="section-header-row">
                    <div>
                      <span className="mono small">
                        <strong>{job.job_id}</strong>
                      </span>{" "}
                      <span className="badge-pill">{job.job_type}</span>
                    </div>
                    <span
                      className={`badge-pill ${
                        job.status === "completed"
                          ? "ok"
                          : job.status === "failed" || job.status === "cancelled"
                            ? "warn"
                            : ""
                      }`}
                    >
                      {formatJobStatusLabel(job.status)}
                    </span>
                  </div>

                  <div className="job-progress-bar">
                    <div className="job-progress-fill" style={{ width: `${pct}%` }} />
                  </div>

                  <div className="section-header-row small muted">
                    <span>
                      Progress: {done}/{total} ({pct}%)
                    </span>
                    <div className="header-actions">
                      {isActive && (
                        <button
                          type="button"
                          className="danger-btn compact"
                          data-testid={`drawer-cancel-job-${job.job_id}`}
                          onClick={() => void cancelJob(job.job_id)}
                        >
                          <XCircle size={12} /> Cancel
                        </button>
                      )}
                      {job.status === "completed" &&
                        artifacts.map((art, idx) => (
                          <a
                            key={`${art.name}-${idx}`}
                            className="link-btn small"
                            href={artifactDownloadUrl(job.job_id, idx)}
                            download={art.name || undefined}
                            data-testid={`drawer-download-job-${job.job_id}-${idx}`}
                          >
                            <Download size={12} /> Download {art.name || `Artifact #${idx + 1}`}
                          </a>
                        ))}
                      {job.status === "completed" && artifacts.length === 0 && exportId && (
                        <a
                          className="link-btn small"
                          href={exportDownloadUrl(exportId, token)}
                          download={`${exportId}.zip`}
                          data-testid={`drawer-download-export-${job.job_id}`}
                        >
                          <Download size={12} /> Download {exportId}.zip
                        </a>
                      )}
                    </div>
                  </div>

                  {job.job_type === "ingest" &&
                    typeof job.summary?.discovered === "number" && (
                      <div className="small muted mono">
                        Discovered {job.summary.discovered} · Inserted {job.summary.inserted ?? 0} ·
                        Updated {job.summary.updated ?? 0} · Skipped {job.summary.skipped ?? 0}
                      </div>
                    )}

                  {job.job_type === "eval_gt" &&
                    (job.summary?.eval_run_id || job.summary?.metrics) && (
                      <div className="small muted mono">
                        {job.summary?.eval_run_id ? `Eval: ${job.summary.eval_run_id}` : ""}
                        {job.summary?.metrics
                          ? ` · F1 ${job.summary.metrics.f1.toFixed(3)} · Cache ${(
                              job.summary.metrics.cache_hit_rate * 100
                            ).toFixed(0)}%`
                          : ""}
                      </div>
                    )}

                  {(results.length > 0 || errors.length > 0) && (
                    <details className="diag-raw-json">
                      <summary>
                        Per-item details (succeeded {results.length} · failed {errors.length})
                      </summary>
                      <div style={{ marginTop: 6, display: "flex", flexDirection: "column", gap: 4 }}>
                        {results.map((r, idx) => (
                          <div key={`res-${idx}`} className="mono small">
                            ✓ {String(r.episode_id ?? r.run_id ?? JSON.stringify(r))}
                          </div>
                        ))}
                        {errors.map((err, idx) => (
                          <div key={`err-${idx}`} className="mono small delta-neg">
                            ✗ {err.episode_id ?? "*"}: {err.error ?? "Unknown error"}
                          </div>
                        ))}
                      </div>
                    </details>
                  )}
                </div>
              );
            })
          )}
        </div>
      </aside>
    </div>
  );
};
