import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useInRouterContext, useSearchParams } from "react-router-dom";
import { AlertTriangle, Download, Eye, Filter, PackageCheck, ShieldOff, X } from "lucide-react";
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
  StateSchemaMode,
} from "../api/types";
import { useEventStream } from "../api/useEventStream";
import { ALL_EPISODE_STATUSES } from "../constants/labels";
import { useAuth } from "../stores/authStore";
import { useI18n } from "../stores/i18nStore";
import { useJobs } from "../stores/jobStore";
import { useToast } from "../stores/toastStore";

const FORMAT_OPTIONS: Array<{ id: ExportFormat; titleKey: string; descKey: string }> = [
  { id: "lerobot_v3", titleKey: "export.fmt.lerobot_v3.title", descKey: "export.fmt.lerobot_v3.desc" },
  { id: "h5_clean", titleKey: "export.fmt.h5_clean.title", descKey: "export.fmt.h5_clean.desc" },
  {
    id: "zero_copy_index",
    titleKey: "export.fmt.zero_copy_index.title",
    descKey: "export.fmt.zero_copy_index.desc",
  },
  { id: "viz_mp4", titleKey: "export.fmt.viz_mp4.title", descKey: "export.fmt.viz_mp4.desc" },
];

const RECIPE_OPTIONS: Array<{ id: ExportRecipeName; title: string; descKey: string }> = [
  { id: "bc_clean", title: "bc_clean", descKey: "export.recipe.bc_clean.desc" },
  { id: "recovery_plus", title: "recovery_plus", descKey: "export.recipe.recovery_plus.desc" },
  { id: "badcase_pairs", title: "badcase_pairs", descKey: "export.recipe.badcase_pairs.desc" },
];

const SELECTABLE_STATUSES: EpisodeStatus[] = ["approved", "gt_frozen", "reviewed", "vlm_ready"];
const VALID_STATUS_SET = new Set<string>(ALL_EPISODE_STATUSES);

function parseCommaList(raw: string | null): string[] {
  if (!raw) return [];
  const seen = new Set<string>();
  const out: string[] = [];
  for (const part of raw.split(",")) {
    const trimmed = part.trim();
    if (trimmed && !seen.has(trimmed)) {
      seen.add(trimmed);
      out.push(trimmed);
    }
  }
  return out;
}

function parseIncludeStatusesParam(raw: string | null): EpisodeStatus[] | null {
  const parts = parseCommaList(raw).filter((s): s is EpisodeStatus => VALID_STATUS_SET.has(s));
  return parts.length > 0 ? parts : null;
}

interface ExportPageInnerProps {
  searchParams: URLSearchParams;
  onClearEpisodeIdsParam?: () => void;
}

const ExportPageInner: React.FC<ExportPageInnerProps> = ({
  searchParams,
  onClearEpisodeIdsParam,
}) => {
  const { isLead, token, authProbing, setAuthPromptOpen } = useAuth();
  const { t } = useI18n();
  const jobStore = useJobs();
  const toast = useToast();

  const initialSearchKey = searchParams.toString();
  const appliedSearchKeyRef = useRef<string>(initialSearchKey);

  const [format, setFormat] = useState<ExportFormat>("lerobot_v3");
  const [recipe, setRecipe] = useState<ExportRecipeName>("bc_clean");
  const [scopedEpisodeIds, setScopedEpisodeIds] = useState<string[]>(() =>
    parseCommaList(searchParams.get("episode_ids")),
  );
  const [includeStatuses, setIncludeStatuses] = useState<EpisodeStatus[]>(
    () => parseIncludeStatusesParam(searchParams.get("include_status")) ?? ["approved", "gt_frozen"],
  );
  const [datasetFilter, setDatasetFilter] = useState(() => searchParams.get("dataset")?.trim() ?? "");
  const [taskFilter, setTaskFilter] = useState("");
  const [smoothTail, setSmoothTail] = useState<SmoothTailMode>("on_jump");
  const [stateSchemaMode, setStateSchemaMode] = useState<StateSchemaMode>("auto");
  const [splitBySubtask, setSplitBySubtask] = useState<boolean>(false);
  const [minPieceSText, setMinPieceSText] = useState<string>("0.1");
  const [tailFrames, setTailFrames] = useState<string>("");
  const [chunkSize, setChunkSize] = useState<string>("");
  const [contextWindowSText, setContextWindowSText] = useState<string>("2");

  // Re-sync if query string changes externally while staying mounted
  useEffect(() => {
    const key = searchParams.toString();
    if (key === appliedSearchKeyRef.current) return;
    appliedSearchKeyRef.current = key;
    if (searchParams.has("episode_ids")) {
      setScopedEpisodeIds(parseCommaList(searchParams.get("episode_ids")));
    }
    if (searchParams.has("dataset")) {
      setDatasetFilter(searchParams.get("dataset")?.trim() ?? "");
    }
    const parsedStatuses = parseIncludeStatusesParam(searchParams.get("include_status"));
    if (parsedStatuses) {
      setIncludeStatuses(parsedStatuses);
    }
  }, [searchParams]);

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
        const errText = job.summary?.errors?.[0]?.error || t("export.errAsyncFailed");
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
          message: t("export.toastJobCancelled", { jobId: job.job_id }),
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
        ? t("export.toastCompletedDetail", {
            pieces: manifest.pieces_exported,
            frames: manifest.total_frames,
          })
        : "";
      toast.showToast({
        message: t("export.toastCompleted", { id: exportId ?? job.job_id, detail }),
        variant: "success",
        testId: "export-info-banner",
      });
      await refreshHistory();
    },
    [refreshHistory, toast, t],
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

  const visibleStatusOptions = useMemo(() => {
    const extra = includeStatuses.filter((st) => !SELECTABLE_STATUSES.includes(st));
    return [...SELECTABLE_STATUSES, ...extra];
  }, [includeStatuses]);

  if (authProbing) {
    return <div className="page-container">{t("export.verifyingPerms")}</div>;
  }

  if (!isLead) {
    return (
      <div className="page-container" data-testid="export-forbidden">
        <div className="forbidden-card">
          <ShieldOff size={32} />
          <h2>{t("export.forbiddenTitle")}</h2>
          <p>{t("export.forbiddenDesc")}</p>
        </div>
      </div>
    );
  }

  const validateAndBuildRequest = (
    overrideStatuses?: EpisodeStatus[],
  ): { req?: ExportRequest; validationError?: string } => {
    const effStatuses = overrideStatuses ?? includeStatuses;
    if (effStatuses.length === 0) {
      return { validationError: t("export.errNoStatus") };
    }
    const parsedMinPiece = minPieceSText.trim() === "" ? NaN : Number(minPieceSText);
    if (!Number.isFinite(parsedMinPiece) || parsedMinPiece < 0.01) {
      return { validationError: t("export.errMinPiece") };
    }
    const parsedCtx = contextWindowSText.trim() === "" ? NaN : Number(contextWindowSText);
    if (recipe === "badcase_pairs" && (!Number.isFinite(parsedCtx) || parsedCtx < 0)) {
      return { validationError: t("export.errContextWindow") };
    }
    const parsedTail = tailFrames.trim() ? Number(tailFrames) : null;
    if (parsedTail !== null && (!Number.isInteger(parsedTail) || parsedTail < 0)) {
      return { validationError: t("export.errTailFrames") };
    }
    const parsedChunk = chunkSize.trim() ? Number(chunkSize) : null;
    if (parsedChunk !== null && (!Number.isInteger(parsedChunk) || parsedChunk < 2)) {
      return { validationError: t("export.errChunkSize") };
    }
    const req: ExportRequest = {
      format,
      recipe,
      include_statuses: effStatuses,
      dataset_filter: datasetFilter.trim() || null,
      task_filter: taskFilter.trim() || null,
      smooth_tail: smoothTail,
      state_schema_mode: stateSchemaMode,
      split_by_subtask: splitBySubtask,
      min_piece_s: parsedMinPiece,
      tail_frames: parsedTail,
      chunk_size: parsedChunk,
      context_window_s: Number.isFinite(parsedCtx) && parsedCtx >= 0 ? parsedCtx : 2.0,
    };
    if (scopedEpisodeIds.length > 0) {
      req.episode_ids = scopedEpisodeIds;
    }
    return { req };
  };

  const handleDryRun = async (overrideStatuses?: EpisodeStatus[]) => {
    const { req, validationError } = validateAndBuildRequest(overrideStatuses);
    if (validationError || !req) {
      setError(validationError ?? t("export.errInvalidParams"));
      return;
    }
    setBusy(true);
    setError(null);
    toast.dismissByTestId("export-info-banner");
    try {
      const res = await dryRunExport(req);
      setDryRunResult(res);
      toast.showToast({
        message: t("export.toastDryRunDone"),
        variant: "info",
        testId: "export-info-banner",
      });
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        setAuthPromptOpen(true);
      }
      setError(err instanceof Error ? err.message : t("export.errDryRunFailed"));
    } finally {
      setBusy(false);
    }
  };

  const handleClearScopedEpisodes = () => {
    setScopedEpisodeIds([]);
    if (searchParams.has("episode_ids")) {
      const next = new URLSearchParams(searchParams);
      next.delete("episode_ids");
      appliedSearchKeyRef.current = next.toString();
      onClearEpisodeIdsParam?.();
    }
  };

  const handleIncludeReviewedAndRetry = async () => {
    const nextStatuses = Array.from(
      new Set<EpisodeStatus>([...includeStatuses, "reviewed", "vlm_ready"]),
    );
    setIncludeStatuses(nextStatuses);
    await handleDryRun(nextStatuses);
  };

  const showZeroMatchHint =
    dryRunResult !== null &&
    dryRunResult.episodes_included === 0 &&
    (!includeStatuses.includes("reviewed") || !includeStatuses.includes("vlm_ready"));

  const handleTriggerExport = async () => {
    const { req, validationError } = validateAndBuildRequest();
    if (validationError || !req) {
      setError(validationError ?? t("export.errInvalidParams"));
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
          message: t("export.toastJobSubmitted", { jobId: job.job_id }),
          variant: "info",
          testId: "export-info-banner",
        });
      }
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        setAuthPromptOpen(true);
      }
      setError(err instanceof Error ? err.message : t("export.errCreateJobFailed"));
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
          <h1>{t("export.title")}</h1>
          <p className="subtitle">{t("export.subtitle")}</p>
        </div>
      </div>

      {/* 4 x 3 Matrix */}
      <div className="export-matrix">
        <div className="matrix-section">
          <h3>{t("export.section1Title")}</h3>
          <div className="card-grid-4">
            {FORMAT_OPTIONS.map((opt) => (
              <button
                key={opt.id}
                type="button"
                className={`choice-card ${format === opt.id ? "selected" : ""}`}
                onClick={() => setFormat(opt.id)}
                data-testid={`format-card-${opt.id}`}
              >
                <div className="choice-title">{t(opt.titleKey)}</div>
                <div className="choice-desc">{t(opt.descKey)}</div>
              </button>
            ))}
          </div>
        </div>

        <div className="matrix-section">
          <h3>{t("export.section2Title")}</h3>
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
                <div className="choice-desc">{t(opt.descKey)}</div>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Recipe Parameters */}
      <div className="panel-card">
        <h3>{t("export.section3Title")}</h3>

        {scopedEpisodeIds.length > 0 && (
          <div className="scoped-episodes-banner" data-testid="scoped-episodes-banner">
            <div className="scoped-episodes-info">
              <Filter size={14} />
              <strong>{t("export.scopedBanner", { count: scopedEpisodeIds.length })}</strong>
              <div className="scoped-episodes-chips" title={scopedEpisodeIds.join(", ")}>
                {scopedEpisodeIds.slice(0, 3).map((epId) => (
                  <span key={epId} className="scoped-episode-chip mono">
                    {epId}
                  </span>
                ))}
                {scopedEpisodeIds.length > 3 && (
                  <span className="scoped-episode-chip">
                    {t("common.more", { count: scopedEpisodeIds.length - 3 })}
                  </span>
                )}
              </div>
            </div>
            <button
              type="button"
              className="mini-btn"
              onClick={handleClearScopedEpisodes}
              data-testid="clear-scoped-episodes-btn"
            >
              <X size={12} /> {t("common.clear")}
            </button>
          </div>
        )}

        <div className="params-grid">
          <div className="field-group">
            <span>{t("export.includeStatuses")}</span>
            <div className="checkbox-row">
              {visibleStatusOptions.map((st) => (
                <label key={st} className="checkbox-inline">
                  <input
                    type="checkbox"
                    checked={includeStatuses.includes(st)}
                    onChange={() => toggleStatus(st)}
                  />
                  <span>{t(`export.statusOption.${st}`)}</span>
                </label>
              ))}
            </div>
          </div>

          <label className="field-group">
            <span>{t("export.datasetFilter")}</span>
            <input
              type="text"
              value={datasetFilter}
              onChange={(e) => setDatasetFilter(e.target.value)}
              placeholder={t("export.datasetFilterPlaceholder")}
            />
          </label>

          <label className="field-group">
            <span>{t("export.taskFilter")}</span>
            <input
              type="text"
              value={taskFilter}
              onChange={(e) => setTaskFilter(e.target.value)}
              placeholder={t("export.taskFilterPlaceholder")}
            />
          </label>

          <label className="field-group">
            <span>{t("export.smoothTail")}</span>
            <select
              value={smoothTail}
              onChange={(e) => setSmoothTail(e.target.value as SmoothTailMode)}
            >
              <option value="on_jump">{t("export.smoothTail.on_jump")}</option>
              <option value="always">{t("export.smoothTail.always")}</option>
              <option value="never">{t("export.smoothTail.never")}</option>
            </select>
          </label>

          <label className="field-group">
            <span>{t("export.stateSchemaMode")}</span>
            <select
              value={stateSchemaMode}
              data-testid="state-schema-mode-select"
              onChange={(e) => setStateSchemaMode(e.target.value as StateSchemaMode)}
            >
              <option value="auto">{t("export.stateSchema.auto")}</option>
              <option value="preserve_native">{t("export.stateSchema.preserve_native")}</option>
              <option value="canonical_81d">{t("export.stateSchema.canonical_81d")}</option>
            </select>
          </label>

          <label className="field-group">
            <span>{t("export.minPieceS")}</span>
            <input
              type="number"
              step={0.05}
              min={0.01}
              value={minPieceSText}
              onChange={(e) => setMinPieceSText(e.target.value)}
            />
          </label>

          <label className="field-group">
            <span>{t("export.tailFrames")}</span>
            <input
              type="number"
              min={0}
              value={tailFrames}
              onChange={(e) => setTailFrames(e.target.value)}
              placeholder={t("export.tailFramesPlaceholder")}
            />
          </label>

          <label className="field-group">
            <span>{t("export.chunkSize")}</span>
            <input
              type="number"
              min={2}
              value={chunkSize}
              onChange={(e) => setChunkSize(e.target.value)}
              placeholder={t("export.chunkSizePlaceholder")}
            />
          </label>

          {recipe === "badcase_pairs" && (
            <label className="field-group">
              <span>{t("export.contextWindowS")}</span>
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

        <label className="checkbox-feature-card">
          <input
            type="checkbox"
            checked={splitBySubtask}
            data-testid="split-by-subtask-checkbox"
            onChange={(e) => setSplitBySubtask(e.target.checked)}
          />
          <div className="checkbox-feature-text">
            <strong>{t("export.splitBySubtaskTitle")}</strong>
            <span className="muted small">{t("export.splitBySubtaskDesc")}</span>
          </div>
        </label>

        <div className="export-actions">
          <button
            type="button"
            className="secondary-btn"
            disabled={busy}
            onClick={() => void handleDryRun()}
            data-testid="dry-run-btn"
          >
            <Eye size={15} /> {t("export.dryRunBtn")}
          </button>
          <button
            type="button"
            className="primary-btn"
            disabled={busy}
            onClick={() => void handleTriggerExport()}
            data-testid="trigger-export-btn"
          >
            <PackageCheck size={15} /> {busy ? t("export.exporting") : t("export.runExportBtn")}
          </button>
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {activeExportJob &&
        (activeExportJob.status === "queued" || activeExportJob.status === "running") && (
          <div className="job-progress-banner" data-testid="export-job-progress">
            <div>
              <strong>{t("export.jobProgressTitle", { jobId: activeExportJob.job_id })}</strong> —{" "}
              {activeExportJob.status} ({activeExportJob.progress_done}/
              {activeExportJob.progress_total})
            </div>
            <button
              type="button"
              className="danger-btn compact"
              onClick={() => void handleCancelExportJob(activeExportJob.job_id)}
            >
              {t("export.cancelExport")}
            </button>
          </div>
        )}

      {completedResult && (
        <div className="panel-card" data-testid="export-job-result">
          <div className="section-header-row">
            <strong>
              {t("export.completedTitle", {
                id: completedResult.exportId ?? completedResult.jobId,
              })}
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
                    <Download size={14} />{" "}
                    {t("export.downloadArtifact", { name: art.name || `#${idx + 1}` })}
                  </a>
                ))
              ) : completedResult.exportId ? (
                <a
                  className="link-btn"
                  href={exportDownloadUrl(completedResult.exportId, token)}
                  download={`${completedResult.exportId}.zip`}
                  data-testid="export-artifact-download-0"
                >
                  <Download size={14} /> {t("export.downloadZip", { id: completedResult.exportId })}
                </a>
              ) : null}
            </div>
          </div>
        </div>
      )}

      {showZeroMatchHint && (
        <div className="empty-export-banner" data-testid="empty-export-warning-banner">
          <div className="empty-export-banner-copy">
            <AlertTriangle size={16} />
            <div>
              <strong>{t("export.zeroMatchTitle")}</strong>{" "}
              <span className="small">{t("export.zeroMatchDesc")}</span>
            </div>
          </div>
          <button
            type="button"
            className="warning-btn compact"
            disabled={busy}
            data-testid="include-reviewed-vlm-ready-retry-btn"
            onClick={() => void handleIncludeReviewedAndRetry()}
          >
            {t("export.zeroMatchRetryBtn")}
          </button>
        </div>
      )}

      {dryRunResult && (
        <div className="panel-card" data-testid="export-manifest-preview">
          <h3>{t("export.manifestPreviewTitle", { id: dryRunResult.export_id })}</h3>
          <div className="metrics-row">
            <div>{t("export.metricEpisodesIncluded", { count: dryRunResult.episodes_included })}</div>
            <div>{t("export.metricEpisodesSkipped", { count: dryRunResult.episodes_skipped })}</div>
            <div>{t("export.metricPiecesExported", { count: dryRunResult.pieces_exported })}</div>
            <div>{t("export.metricPiecesDropped", { count: dryRunResult.pieces_dropped })}</div>
            <div>{t("export.metricTotalFrames", { count: dryRunResult.total_frames })}</div>
          </div>
          {dryRunResult.episodes.length > 0 && (
            <table className="data-table compact">
              <thead>
                <tr>
                  <th>{t("export.colEpisode")}</th>
                  <th>{t("export.colStatus")}</th>
                  <th>{t("export.colPiecesExported")}</th>
                  <th>{t("export.colPiecesDropped")}</th>
                  <th>{t("export.colTotalFrames")}</th>
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
        <h3>{t("export.historyTitle", { count: history.length })}</h3>
        {history.length === 0 ? (
          <div className="empty-hint">{t("export.historyEmpty")}</div>
        ) : (
          <table className="data-table compact">
            <thead>
              <tr>
                <th>{t("export.colExportId")}</th>
                <th>{t("export.colFormat")}</th>
                <th>{t("export.colRecipe")}</th>
                <th>{t("export.colSummary")}</th>
                <th>{t("export.colDownload")}</th>
              </tr>
            </thead>
            <tbody>
              {history.map((item) => (
                <tr key={item.export_id}>
                  <td className="mono">{item.export_id}</td>
                  <td>{item.format}</td>
                  <td>{item.recipe.name}</td>
                  <td>
                    {t("export.historyRowSummary", {
                      eps: item.episodes_included,
                      pieces: item.pieces_exported,
                      frames: item.total_frames,
                    })}
                  </td>
                  <td>
                    <a
                      className="link-btn"
                      href={exportDownloadUrl(item.export_id, token)}
                      download={`${item.export_id}.zip`}
                    >
                      <Download size={13} /> {t("export.downloadZipBtn")}
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

const ExportPageWithRouter: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const handleClearEpisodeIdsParam = useCallback(() => {
    const next = new URLSearchParams(searchParams);
    next.delete("episode_ids");
    setSearchParams(next, { replace: true });
  }, [searchParams, setSearchParams]);
  return (
    <ExportPageInner
      searchParams={searchParams}
      onClearEpisodeIdsParam={handleClearEpisodeIdsParam}
    />
  );
};

const ExportPageStandalone: React.FC = () => {
  const searchParams = useMemo(
    () => new URLSearchParams(typeof window !== "undefined" ? window.location.search : ""),
    [],
  );
  return <ExportPageInner searchParams={searchParams} />;
};

export const ExportPage: React.FC = () => {
  const inRouter = useInRouterContext();
  return inRouter ? <ExportPageWithRouter /> : <ExportPageStandalone />;
};
