import React, { useEffect, useState } from "react";
import { Activity, AlertTriangle, Download, RefreshCw, Trash2, X, XCircle } from "lucide-react";
import { exportDownloadUrl } from "../api/client";
import type { JobClearScope, JobStatus } from "../api/types";
import { useAuth } from "../stores/authStore";
import { useI18n } from "../stores/i18nStore";
import { useJobs } from "../stores/jobStore";

type PendingDeleteTarget =
  | { kind: "single"; jobId: string }
  | { kind: "batch"; scope: JobClearScope; count: number };

export const JobCenterDrawer: React.FC = () => {
  const { isLead, token } = useAuth();
  const { t } = useI18n();
  const {
    jobs,
    totalJobs,
    statusCounts,
    drawerOpen,
    setDrawerOpen,
    refreshJobs,
    cancelJob,
    deleteJob,
    clearJobs,
    artifactDownloadUrl,
  } = useJobs();
  const [cancellingIds, setCancellingIds] = useState<Record<string, boolean>>({});
  const [pendingDelete, setPendingDelete] = useState<PendingDeleteTarget | null>(null);
  const [deleting, setDeleting] = useState(false);

  const formatJobStatusLabel = (status: JobStatus["status"]): string => {
    switch (status) {
      case "queued":
        return t("jobs.status.queued");
      case "running":
        return t("jobs.status.running");
      case "completed":
        return t("jobs.status.completed");
      case "failed":
        return t("jobs.status.failed");
      case "cancelled":
        return t("jobs.status.cancelled");
      default:
        return status;
    }
  };

  const handleCancelClick = async (jobId: string) => {
    if (cancellingIds[jobId]) return;
    setCancellingIds((prev) => ({ ...prev, [jobId]: true }));
    try {
      await cancelJob(jobId);
    } finally {
      setCancellingIds((prev) => {
        const next = { ...prev };
        delete next[jobId];
        return next;
      });
    }
  };

  const handleConfirmDelete = async () => {
    if (!pendingDelete || deleting) return;
    setDeleting(true);
    try {
      if (pendingDelete.kind === "single") {
        await deleteJob(pendingDelete.jobId);
      } else {
        await clearJobs(pendingDelete.scope);
      }
      setPendingDelete(null);
    } finally {
      setDeleting(false);
    }
  };

  useEffect(() => {
    if (!drawerOpen) {
      setPendingDelete(null);
      return;
    }
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (pendingDelete) {
          setPendingDelete(null);
        } else {
          setDrawerOpen(false);
        }
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [drawerOpen, pendingDelete, setDrawerOpen]);

  if (!isLead || !drawerOpen) return null;

  const finishedCount =
    statusCounts.completed + statusCounts.failed + statusCounts.cancelled;
  const failedCancelledCount = statusCounts.failed + statusCounts.cancelled;
  const displayTotal = Math.max(totalJobs, jobs.length);

  return (
    <div
      className="drawer-backdrop"
      data-testid="job-center-drawer-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) setDrawerOpen(false);
      }}
    >
      <aside className="drawer-panel" data-testid="job-center-drawer" aria-label={t("jobs.drawerAria")}>
        <div className="drawer-header">
          <h3>
            <Activity size={16} /> {t("jobs.drawerTitle", { count: displayTotal })}
          </h3>
          <div className="header-actions">
            <button
              type="button"
              className="secondary-btn compact"
              onClick={() => void refreshJobs()}
              data-testid="refresh-job-drawer-btn"
            >
              <RefreshCw size={13} /> {t("common.refresh")}
            </button>
            <button
              type="button"
              className="mini-btn"
              onClick={() => setDrawerOpen(false)}
              data-testid="close-job-drawer-btn"
              aria-label={t("jobs.closeAria")}
            >
              <X size={14} /> {t("common.close")}
            </button>
          </div>
        </div>

        {finishedCount > 0 && (
          <div className="job-drawer-toolbar" data-testid="job-drawer-cleanup-bar">
            <button
              type="button"
              className="secondary-btn compact"
              disabled={deleting}
              data-testid="clear-finished-jobs-btn"
              onClick={() =>
                setPendingDelete({
                  kind: "batch",
                  scope: "finished",
                  count: finishedCount,
                })
              }
            >
              <Trash2 size={12} /> {t("jobs.clearFinished", { count: finishedCount })}
            </button>
            {failedCancelledCount > 0 && (
              <button
                type="button"
                className="secondary-btn compact"
                disabled={deleting}
                data-testid="clear-failed-jobs-btn"
                onClick={() =>
                  setPendingDelete({
                    kind: "batch",
                    scope: "failed_cancelled",
                    count: failedCancelledCount,
                  })
                }
              >
                <Trash2 size={12} />{" "}
                {t("jobs.clearFailedCancelled", { count: failedCancelledCount })}
              </button>
            )}
          </div>
        )}

        {pendingDelete && (
          <div className="job-confirm-box" role="alertdialog" data-testid="job-delete-confirm-box">
            <div className="job-confirm-title">
              <AlertTriangle size={14} />
              <span>
                {pendingDelete.kind === "single"
                  ? t("jobs.confirmDeleteSingle", { id: pendingDelete.jobId })
                  : pendingDelete.scope === "finished"
                    ? t("jobs.confirmClearFinished", { count: pendingDelete.count })
                    : t("jobs.confirmClearFailedCancelled", { count: pendingDelete.count })}
              </span>
            </div>
            <p className="small muted" style={{ margin: 0 }}>
              {t("jobs.deleteSafetyNote")}
            </p>
            <div className="header-actions" style={{ justifyContent: "flex-end" }}>
              <button
                type="button"
                className="secondary-btn compact"
                disabled={deleting}
                data-testid="cancel-delete-jobs-btn"
                onClick={() => setPendingDelete(null)}
              >
                {t("common.cancel")}
              </button>
              <button
                type="button"
                className="danger-btn compact"
                disabled={deleting}
                data-testid="confirm-delete-jobs-btn"
                onClick={() => void handleConfirmDelete()}
              >
                <Trash2 size={12} /> {deleting ? t("jobs.deleting") : t("jobs.confirmDeleteBtn")}
              </button>
            </div>
          </div>
        )}

        <div className="drawer-body">
          {jobs.length === 0 ? (
            <div className="empty-cell">{t("jobs.empty")}</div>
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
                      {t("jobs.progress", { done, total, pct })}
                    </span>
                    <div className="header-actions">
                      {isActive ? (
                        <button
                          type="button"
                          className="danger-btn compact"
                          disabled={Boolean(cancellingIds[job.job_id])}
                          data-testid={`drawer-cancel-job-${job.job_id}`}
                          onClick={() => void handleCancelClick(job.job_id)}
                        >
                          <XCircle size={12} />{" "}
                          {cancellingIds[job.job_id] ? t("jobs.cancelling") : t("common.cancel")}
                        </button>
                      ) : (
                        <button
                          type="button"
                          className="mini-btn"
                          disabled={deleting}
                          data-testid={`drawer-delete-job-${job.job_id}`}
                          onClick={() =>
                            setPendingDelete({ kind: "single", jobId: job.job_id })
                          }
                        >
                          <Trash2 size={12} /> {t("jobs.delete")}
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
                            <Download size={12} />{" "}
                            {t("jobs.downloadItem", {
                              name: art.name || t("queue.artifactFallback", { index: idx + 1 }),
                            })}
                          </a>
                        ))}
                      {job.status === "completed" && artifacts.length === 0 && exportId && (
                        <a
                          className="link-btn small"
                          href={exportDownloadUrl(exportId, token)}
                          download={`${exportId}.zip`}
                          data-testid={`drawer-download-export-${job.job_id}`}
                        >
                          <Download size={12} /> {t("jobs.downloadExport", { id: exportId })}
                        </a>
                      )}
                    </div>
                  </div>

                  {job.summary?.error === "interrupted_by_server_restart" && (
                    <div className="small muted">{t("jobs.interruptedByRestart")}</div>
                  )}

                  {job.job_type === "ingest" &&
                    typeof job.summary?.discovered === "number" && (
                      <div className="small muted mono">
                        {t("jobs.ingestSummary", {
                          discovered: job.summary.discovered,
                          inserted: job.summary.inserted ?? 0,
                          updated: job.summary.updated ?? 0,
                          skipped: job.summary.skipped ?? 0,
                        })}
                      </div>
                    )}

                  {job.job_type === "eval_gt" &&
                    (job.summary?.eval_run_id || job.summary?.metrics) && (
                      <div className="small muted mono">
                        {job.summary?.eval_run_id
                          ? t("jobs.evalPrefix", { id: job.summary.eval_run_id })
                          : ""}
                        {job.summary?.metrics
                          ? t("jobs.evalMetrics", {
                              f1: job.summary.metrics.f1.toFixed(3),
                              cache: (job.summary.metrics.cache_hit_rate * 100).toFixed(0),
                            })
                          : ""}
                      </div>
                    )}

                  {(results.length > 0 || errors.length > 0) && (
                    <details className="diag-raw-json">
                      <summary>
                        {t("jobs.perItemDetails", { ok: results.length, failed: errors.length })}
                      </summary>
                      <div style={{ marginTop: 6, display: "flex", flexDirection: "column", gap: 4 }}>
                        {results.map((r, idx) => (
                          <div key={`res-${idx}`} className="mono small">
                            ✓ {String(r.episode_id ?? r.run_id ?? JSON.stringify(r))}
                          </div>
                        ))}
                        {errors.map((err, idx) => (
                          <div key={`err-${idx}`} className="mono small delta-neg">
                            ✗ {err.episode_id ?? "*"}: {err.error ?? t("common.unknownError")}
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
