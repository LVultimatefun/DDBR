import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Download, Eye, PackageCheck, ShieldOff } from "lucide-react";
import {
  ApiError,
  cancelJob as apiCancelJob,
  createJob,
  dryRunExport,
  exportDownloadUrl,
  getJob,
  jobArtifactUrl,
  listExports,
} from "../api/client";
import type {
  EpisodeStatus,
  ExportFormat,
  ExportManifest,
  ExportRecipeName,
  ExportRequest,
  JobArtifactItem,
  JobStatus,
  SmoothTailMode,
} from "../api/types";
import { useEventStream } from "../api/useEventStream";
import { useAuth } from "../stores/authStore";
import { useJobs } from "../stores/jobStore";
import { useToast } from "../stores/toastStore";

const FORMAT_OPTIONS: Array<{ id: ExportFormat; title: string; desc: string }> = [
  { id: "lerobot_v3", title: "LeRobot v3.0", desc: "Parquet state/action + multi-cam CFR MP4 + subtasks.parquet" },
  { id: "h5_clean", title: "Clean Sliced .h5", desc: "Lossless raw streams + /qc/action_loss_mask & subtask_index" },
  { id: "zero_copy_index", title: "Zero-Copy Index", desc: "qc_index.parquet + manifest.jsonl (<0.1% disk footprint)" },
  { id: "viz_mp4", title: "1080p Studio MP4", desc: "2x2 grid + 2D grounding boxes + four-track timeline overlay" },
];

const RECIPE_OPTIONS: Array<{ id: ExportRecipeName; title: string; desc: string }> = [
  { id: "bc_clean", title: "bc_clean", desc: "Successful episodes only; cut valid cut_clean spans, mask_loss set to 0" },
  { id: "recovery_plus", title: "recovery_plus", desc: "Successful episodes + retain valid recovery segments (mask=1.0)" },
  { id: "badcase_pairs", title: "badcase_pairs", desc: "All badcase / terminal_fail segments with surrounding context window" },
];

const SELECTABLE_STATUSES: EpisodeStatus[] = ["approved", "gt_frozen", "reviewed", "vlm_ready"];

export const ExportPage: React.FC = () => {
  const { isLead, token, authProbing, setAuthPromptOpen } = useAuth();
  const jobStore = useJobs();
  const toast = useToast();

  const [format, setFormat] = useState<ExportFormat>("lerobot_v3");
  const [recipe, setRecipe] = useState<ExportRecipeName>("bc_clean");
  const [includeStatuses, setIncludeStatuses] = useState<EpisodeStatus[]>(["approved", "gt_frozen"]);
  const [datasetFilter, setDatasetFilter] = useState("");
  const [taskFilter, setTaskFilter] = useState("");
  const [smoothTail, setSmoothTail] = useState<SmoothTailMode>("on_jump");
  const [minPieceSText, setMinPieceSText] = useState<string>("0.1");
  const [tailFrames, setTailFrames] = useState<string>("");
  const [chunkSize, setChunkSize] = useState<string>("");
  const [contextWindowSText, setContextWindowSText] = useState<string>("2");

  const [dryRunResult, setDryRunResult] = useState<ExportManifest | null>(null);
  const [history, setHistory] = useState<ExportManifest[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [localExportJob, setLocalExportJob] = useState<JobStatus | null>(null);
  const [completedResult, setCompletedResult] = useState<{
    jobId: string;
    exportId: string | null;
    artifacts: JobArtifactItem[];
  } | null>(null);
  const handledJobIdsRef = useRef<Set<string>>(new Set());

  const refreshHistory = useCallback(async () => {
    if (!isLead || authProbing) return;
    try {
      const items = await listExports();
      setHistory(items);
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        setAuthPromptOpen(true);
      }
    }
  }, [isLead, authProbing, token, setAuthPromptOpen]);

  useEffect(() => {
    void refreshHistory();
  }, [refreshHistory]);

  const activeExportJob = useMemo(() => {
    if (!localExportJob) return null;
    if (jobStore.isProvided) {
      return jobStore.jobs.find((j) => j.job_id === localExportJob.job_id) ?? localExportJob;
    }
    return localExportJob;
  }, [localExportJob, jobStore.isProvided, jobStore.jobs]);

  const finalizeExportJob = useCallback(
    async (job: JobStatus) => {
      if (handledJobIdsRef.current.has(job.job_id)) return;
      if (job.status !== "completed" && job.status !== "failed" && job.status !== "cancelled") {
        return;
      }
      handledJobIdsRef.current.add(job.job_id);
      setLocalExportJob(job);
      setBusy(false);

      if (job.status === "failed") {
        const errText = job.summary?.errors?.[0]?.error || "Async export job failed";
        setError(errText);
        toast.showToast({
          message: errText,
          variant: "error",
          testId: "export-info-banner",
        });
        return;
      }

      if (job.status === "cancelled") {
        toast.showToast({
          message: `Export job ${job.job_id} cancelled`,
          variant: "warning",
          testId: "export-info-banner",
        });
        return;
      }

      const summary = job.summary ?? {};
      const manifest = (summary.manifest as ExportManifest | undefined) ?? null;
      const artifacts = Array.isArray(summary.artifacts) ? summary.artifacts : [];
      const exportId =
        manifest?.export_id ??
        (typeof summary.export_id === "string" ? summary.export_id : null) ??
        artifacts.find((a) => Boolean(a.export_id))?.export_id ??
        null;

      if (manifest) {
        setDryRunResult(manifest);
      }
      setCompletedResult({
        jobId: job.job_id,
        exportId,
        artifacts,
      });

      const detail = manifest
        ? ` (${manifest.pieces_exported} pieces / ${manifest.total_frames} frames)`
        : "";
      toast.showToast({
        message: `Export completed: ${exportId ?? job.job_id}${detail}`,
        variant: "success",
        testId: "export-info-banner",
      });
      await refreshHistory();
    },
    [refreshHistory, toast],
  );

  // Watch jobStore updates when mounted inside AppRoutes
  useEffect(() => {
    if (!activeExportJob) return;
    if (
      activeExportJob.status === "completed" ||
      activeExportJob.status === "failed" ||
      activeExportJob.status === "cancelled"
    ) {
      void finalizeExportJob(activeExportJob);
    }
  }, [activeExportJob, finalizeExportJob]);

  // Fallback per-job SSE when standalone
  useEventStream<JobStatus>({
    url:
      localExportJob && !jobStore.isProvided
        ? `/api/jobs/${encodeURIComponent(localExportJob.job_id)}/events`
        : null,
    enabled:
      Boolean(localExportJob) &&
      !jobStore.isProvided &&
      (localExportJob?.status === "queued" || localExportJob?.status === "running"),
    isTerminal: (j) => j.status === "completed" || j.status === "failed" || j.status === "cancelled",
    onMessage: (jobEvt) => {
      setLocalExportJob(jobEvt);
      if (jobEvt.status === "completed" || jobEvt.status === "failed" || jobEvt.status === "cancelled") {
        void finalizeExportJob(jobEvt);
      }
    },
  });

  // Polling fallback when EventSource is unavailable or SSE lags
  const activeExportJobId = activeExportJob?.job_id ?? null;
  const activeExportJobStatus = activeExportJob?.status ?? null;
  const jobStoreProvided = jobStore.isProvided;
  const trackJobInStore = jobStore.trackJob;

  useEffect(() => {
    if (!activeExportJobId) return;
    if (activeExportJobStatus !== "queued" && activeExportJobStatus !== "running") return;

    let stopped = false;
    const timer = setInterval(() => {
      if (stopped) return;
      void getJob(activeExportJobId)
        .then((latest) => {
          if (stopped) return;
          setLocalExportJob(latest);
          if (jobStoreProvided) {
            trackJobInStore(latest);
          }
          if (
            latest.status === "completed" ||
            latest.status === "failed" ||
            latest.status === "cancelled"
          ) {
            void finalizeExportJob(latest);
          }
        })
        .catch(() => {
          // Ignore transient polling errors
        });
    }, 350);

    return () => {
      stopped = true;
      clearInterval(timer);
    };
  }, [
    activeExportJobId,
    activeExportJobStatus,
    finalizeExportJob,
    jobStoreProvided,
    trackJobInStore,
  ]);

  if (authProbing) {
    return <div className="page-container">Verifying permissions...</div>;
  }

  if (!isLead) {
    return (
      <div className="page-container" data-testid="export-forbidden">
        <div className="forbidden-card">
          <ShieldOff size={32} />
          <h2>403 — Forbidden</h2>
          <p>Export Studio is restricted to Lead users. Reviewers should return to the Work Queue.</p>
        </div>
      </div>
    );
  }

  const validateAndBuildRequest = (): { req?: ExportRequest; validationError?: string } => {
    if (includeStatuses.length === 0) {
      return { validationError: "Select at least one status (include_statuses)" };
    }
    const parsedMinPiece = minPieceSText.trim() === "" ? NaN : Number(minPieceSText);
    if (!Number.isFinite(parsedMinPiece) || parsedMinPiece < 0.01) {
      return { validationError: "Min piece duration (min_piece_s) must be a valid number >= 0.01" };
    }
    const parsedCtx = contextWindowSText.trim() === "" ? NaN : Number(contextWindowSText);
    if (recipe === "badcase_pairs" && (!Number.isFinite(parsedCtx) || parsedCtx < 0)) {
      return { validationError: "Context window (context_window_s) must be a valid number >= 0" };
    }
    const parsedTail = tailFrames.trim() ? Number(tailFrames) : null;
    if (parsedTail !== null && (!Number.isInteger(parsedTail) || parsedTail < 0)) {
      return { validationError: "Tail blend frames (tail_frames) must be an integer >= 0" };
    }
    const parsedChunk = chunkSize.trim() ? Number(chunkSize) : null;
    if (parsedChunk !== null && (!Number.isInteger(parsedChunk) || parsedChunk < 2)) {
      return { validationError: "Chunk size (chunk_size) must be an integer >= 2" };
    }
    return {
      req: {
        format,
        recipe,
        include_statuses: includeStatuses,
        dataset_filter: datasetFilter.trim() || null,
        task_filter: taskFilter.trim() || null,
        smooth_tail: smoothTail,
        min_piece_s: parsedMinPiece,
        tail_frames: parsedTail,
        chunk_size: parsedChunk,
        context_window_s: Number.isFinite(parsedCtx) && parsedCtx >= 0 ? parsedCtx : 2.0,
      },
    };
  };

  const handleDryRun = async () => {
    const { req, validationError } = validateAndBuildRequest();
    if (validationError || !req) {
      setError(validationError ?? "Invalid export parameters");
      return;
    }
    setBusy(true);
    setError(null);
    toast.dismissByTestId("export-info-banner");
    try {
      const res = await dryRunExport(req);
      setDryRunResult(res);
      toast.showToast({
        message: "Dry run preview completed (nothing written to exports/)",
        variant: "info",
        testId: "export-info-banner",
      });
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        setAuthPromptOpen(true);
      }
      setError(err instanceof Error ? err.message : "Dry run preview failed");
    } finally {
      setBusy(false);
    }
  };

  const handleTriggerExport = async () => {
    const { req, validationError } = validateAndBuildRequest();
    if (validationError || !req) {
      setError(validationError ?? "Invalid export parameters");
      return;
    }
    setBusy(true);
    setError(null);
    setCompletedResult(null);
    toast.dismissByTestId("export-info-banner");
    try {
      const job = await createJob({
        job_type: "export",
        export_params: req,
      });
      setLocalExportJob(job);
      jobStore.trackJob(job);
      if (job.status === "completed" || job.status === "failed" || job.status === "cancelled") {
        await finalizeExportJob(job);
      } else {
        toast.showToast({
          message: `Export job ${job.job_id} submitted`,
          variant: "info",
          testId: "export-info-banner",
        });
      }
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        setAuthPromptOpen(true);
      }
      setError(err instanceof Error ? err.message : "Failed to create async export job");
      setBusy(false);
    }
  };

  const handleCancelExportJob = async (jobId: string) => {
    const updated = jobStore.isProvided
      ? await jobStore.cancelJob(jobId)
      : await apiCancelJob(jobId).catch(() => null);
    if (updated) {
      await finalizeExportJob(updated);
    }
  };

  const toggleStatus = (st: EpisodeStatus) => {
    setIncludeStatuses((prev) =>
      prev.includes(st) ? prev.filter((item) => item !== st) : [...prev, st],
    );
  };

  return (
    <div className="page-container" data-testid="export-page">
      <div className="page-header">
        <div>
          <h1>Transactional Dataset Export Studio</h1>
          <p className="subtitle">
            4 Output Formats × 3 Training Recipes · Unified Async Jobs + Atomic _staging → rename Commit
          </p>
        </div>
      </div>

      {/* 4 x 3 Matrix */}
      <div className="export-matrix">
        <div className="matrix-section">
          <h3>1. Select Output Format (4 Formats)</h3>
          <div className="card-grid-4">
            {FORMAT_OPTIONS.map((opt) => (
              <button
                key={opt.id}
                type="button"
                className={`choice-card ${format === opt.id ? "selected" : ""}`}
                onClick={() => setFormat(opt.id)}
                data-testid={`format-card-${opt.id}`}
              >
                <div className="choice-title">{opt.title}</div>
                <div className="choice-desc">{opt.desc}</div>
              </button>
            ))}
          </div>
        </div>

        <div className="matrix-section">
          <h3>2. Select Training Recipe (3 Recipes)</h3>
          <div className="card-grid-3">
            {RECIPE_OPTIONS.map((opt) => (
              <button
                key={opt.id}
                type="button"
                className={`choice-card ${recipe === opt.id ? "selected" : ""}`}
                onClick={() => setRecipe(opt.id)}
                data-testid={`recipe-card-${opt.id}`}
              >
                <div className="choice-title">{opt.title}</div>
                <div className="choice-desc">{opt.desc}</div>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Recipe Parameters */}
      <div className="panel-card">
        <h3>3. Slicing &amp; Filter Parameters</h3>
        <div className="params-grid">
          <div className="field-group">
            <span>Include Statuses (include_statuses)</span>
            <div className="checkbox-row">
              {SELECTABLE_STATUSES.map((st) => (
                <label key={st} className="checkbox-inline">
                  <input
                    type="checkbox"
                    checked={includeStatuses.includes(st)}
                    onChange={() => toggleStatus(st)}
                  />
                  <span>{st}</span>
                </label>
              ))}
            </div>
          </div>

          <label className="field-group">
            <span>Dataset Filter (dataset_filter)</span>
            <input
              type="text"
              value={datasetFilter}
              onChange={(e) => setDatasetFilter(e.target.value)}
              placeholder="Leave blank to export all datasets"
            />
          </label>

          <label className="field-group">
            <span>Task Filter (task_filter)</span>
            <input
              type="text"
              value={taskFilter}
              onChange={(e) => setTaskFilter(e.target.value)}
              placeholder="Leave blank to export all tasks"
            />
          </label>

          <label className="field-group">
            <span>Tail Smoothing (smooth_tail)</span>
            <select
              value={smoothTail}
              onChange={(e) => setSmoothTail(e.target.value as SmoothTailMode)}
            >
              <option value="on_jump">on_jump (smooth on discontinuity)</option>
              <option value="always">always (always smooth)</option>
              <option value="never">never (disabled)</option>
            </select>
          </label>

          <label className="field-group">
            <span>Min Piece Duration (min_piece_s)</span>
            <input
              type="number"
              step={0.05}
              min={0.01}
              value={minPieceSText}
              onChange={(e) => setMinPieceSText(e.target.value)}
            />
          </label>

          <label className="field-group">
            <span>Tail Blend Frames (tail_frames)</span>
            <input
              type="number"
              min={0}
              value={tailFrames}
              onChange={(e) => setTailFrames(e.target.value)}
              placeholder="Auto default"
            />
          </label>

          <label className="field-group">
            <span>Chunk Size (chunk_size)</span>
            <input
              type="number"
              min={2}
              value={chunkSize}
              onChange={(e) => setChunkSize(e.target.value)}
              placeholder="Optional"
            />
          </label>

          {recipe === "badcase_pairs" && (
            <label className="field-group">
              <span>Context Window (context_window_s)</span>
              <input
                type="number"
                step={0.5}
                min={0}
                value={contextWindowSText}
                onChange={(e) => setContextWindowSText(e.target.value)}
              />
            </label>
          )}
        </div>

        <div className="export-actions">
          <button
            type="button"
            className="secondary-btn"
            disabled={busy}
            onClick={() => void handleDryRun()}
            data-testid="dry-run-btn"
          >
            <Eye size={15} /> Dry Run Preview
          </button>
          <button
            type="button"
            className="primary-btn"
            disabled={busy}
            onClick={() => void handleTriggerExport()}
            data-testid="trigger-export-btn"
          >
            <PackageCheck size={15} /> {busy ? "Exporting..." : "Run Transactional Export"}
          </button>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {activeExportJob &&
        (activeExportJob.status === "queued" || activeExportJob.status === "running") && (
          <div className="job-progress-banner" data-testid="export-job-progress">
            <div>
              <strong>Export Job {activeExportJob.job_id}</strong> — {activeExportJob.status} (
              {activeExportJob.progress_done}/{activeExportJob.progress_total})
            </div>
            <button
              type="button"
              className="danger-btn compact"
              onClick={() => void handleCancelExportJob(activeExportJob.job_id)}
            >
              Cancel Export
            </button>
          </div>
        )}

      {completedResult && (
        <div className="panel-card" data-testid="export-job-result">
          <div className="section-header-row">
            <strong>
              Export Completed: {completedResult.exportId ?? completedResult.jobId}
            </strong>
            <div className="header-actions">
              {completedResult.artifacts.length > 0 ? (
                completedResult.artifacts.map((art, idx) => (
                  <a
                    key={`${art.name}-${idx}`}
                    className="link-btn"
                    href={jobArtifactUrl(completedResult.jobId, idx, token)}
                    download={art.name || undefined}
                    data-testid={`export-artifact-download-${idx}`}
                  >
                    <Download size={14} /> Download Artifact ({art.name || `#${idx + 1}`})
                  </a>
                ))
              ) : completedResult.exportId ? (
                <a
                  className="link-btn"
                  href={exportDownloadUrl(completedResult.exportId, token)}
                  download={`${completedResult.exportId}.zip`}
                  data-testid="export-artifact-download-0"
                >
                  <Download size={14} /> Download {completedResult.exportId}.zip
                </a>
              ) : null}
            </div>
          </div>
        </div>
      )}

      {dryRunResult && (
        <div className="panel-card" data-testid="export-manifest-preview">
          <h3>Export Manifest Preview ({dryRunResult.export_id})</h3>
          <div className="metrics-row">
            <div>Episodes Included: {dryRunResult.episodes_included}</div>
            <div>Episodes Skipped: {dryRunResult.episodes_skipped}</div>
            <div>Pieces Exported: {dryRunResult.pieces_exported}</div>
            <div>Pieces Dropped: {dryRunResult.pieces_dropped}</div>
            <div>Total Frames: {dryRunResult.total_frames}</div>
          </div>
          {dryRunResult.episodes.length > 0 && (
            <table className="data-table compact">
              <thead>
                <tr>
                  <th>Episode</th>
                  <th>Status</th>
                  <th>Pieces Exported</th>
                  <th>Pieces Dropped</th>
                  <th>Total Frames</th>
                </tr>
              </thead>
              <tbody>
                {dryRunResult.episodes.map((ep) => (
                  <tr key={ep.episode_id}>
                    <td className="mono">{ep.episode_id}</td>
                    <td>{ep.status}</td>
                    <td>{ep.pieces_exported}</td>
                    <td>{ep.pieces_dropped}</td>
                    <td>{ep.total_frames}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      <div className="panel-card" data-testid="export-history-section">
        <h3>Export History ({history.length})</h3>
        {history.length === 0 ? (
          <div className="empty-hint">No exports recorded yet.</div>
        ) : (
          <table className="data-table compact">
            <thead>
              <tr>
                <th>Export ID</th>
                <th>Format</th>
                <th>Recipe</th>
                <th>Episodes / Pieces / Frames</th>
                <th>Download</th>
              </tr>
            </thead>
            <tbody>
              {history.map((item) => (
                <tr key={item.export_id}>
                  <td className="mono">{item.export_id}</td>
                  <td>{item.format}</td>
                  <td>{item.recipe.name}</td>
                  <td>
                    {item.episodes_included} eps · {item.pieces_exported} pieces · {item.total_frames} frames
                  </td>
                  <td>
                    <a
                      className="link-btn"
                      href={exportDownloadUrl(item.export_id, token)}
                      download={`${item.export_id}.zip`}
                    >
                      <Download size={13} /> Download ZIP
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {toast.StandaloneHost}
    </div>
  );
};
