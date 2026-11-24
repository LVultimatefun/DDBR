import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  ArrowDown,
  ArrowUp,
  ArrowUpDown,
  Cpu,
  Download,
  FolderPlus,
  PackageCheck,
  Play,
  RefreshCw,
  Settings,
  ShieldAlert,
  Video,
  X,
} from "lucide-react";
import {
  cancelJob as apiCancelJob,
  claimEpisode,
  createJob,
  encodeEpPath,
  getSettings,
  listEpisodes,
  updateSettings,
} from "../api/client";
import type {
  EpisodeFacets,
  EpisodeListItem,
  EpisodeSortField,
  EpisodeStatus,
  JobStatus,
  SortDir,
} from "../api/types";
import { useEventStream } from "../api/useEventStream";
import { FacetCombobox } from "../components/FacetCombobox";
import { IngestModal } from "../components/IngestModal";
import { VlmSettingsModal } from "../components/VlmSettingsModal";
import {
  ALL_EPISODE_STATUSES,
  EPISODE_STATUS_META,
  getEpisodeStatusLabel,
} from "../constants/labels";
import { useAuth } from "../stores/authStore";
import { useI18n } from "../stores/i18nStore";
import { useJobs } from "../stores/jobStore";
import { useToast } from "../stores/toastStore";

type ReviewerTab = "vlm_ready" | "my_claimed" | "submitted";

const QUEUE_REFRESH_EVENTS = new Set([
  "episode_claimed",
  "episode_released",
  "review_saved",
  "job_update",
  "export_completed",
  "episode_updated",
]);

const PAGE_SIZE = 100;
const KNOWN_ADAPTERS = ["h5_raw_v3", "lerobot", "video_dir"];

function formatUpdatedAt(ts: number): string {
  if (!Number.isFinite(ts) || ts <= 0) return "—";
  const ms = ts > 1e12 ? ts : ts * 1000;
  try {
    const d = new Date(ms);
    const pad = (n: number) => String(n).padStart(2, "0");
    return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  } catch {
    return "—";
  }
}

export const QueuePage: React.FC = () => {
  const { isLead, username, token, noAuth, authProbing, setAuthPromptOpen } = useAuth();
  const { locale, t } = useI18n();
  const tRef = useRef(t);
  tRef.current = t;
  const jobStore = useJobs();
  const toast = useToast();
  const navigate = useNavigate();

  const [episodes, setEpisodes] = useState<EpisodeListItem[]>([]);
  const [totalEpisodes, setTotalEpisodes] = useState(0);
  const [myClaimedCount, setMyClaimedCount] = useState(0);
  const [serverStatusCounts, setServerStatusCounts] = useState<Record<string, number> | null>(null);
  const [facets, setFacets] = useState<EpisodeFacets | null>(null);
  const [page, setPage] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Reviewer state
  const [reviewerTab, setReviewerTab] = useState<ReviewerTab>("vlm_ready");
  const [claimingNext, setClaimingNext] = useState(false);

  // Lead filter & sort state
  const [statusFilter, setStatusFilter] = useState<EpisodeStatus | "">("");
  const [datasetFilter, setDatasetFilter] = useState("");
  const [taskFilter, setTaskFilter] = useState("");
  const [adapterFilter, setAdapterFilter] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [sortBy, setSortBy] = useState<EpisodeSortField>("episode_id");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [passedFilter, setPassedFilter] = useState<"all" | "true" | "false">("all");
  const [downgradedOnly, setDowngradedOnly] = useState(false);
  const queueSeqRef = useRef(0);

  // Debounce search query to prevent spamming backend on fast typing
  useEffect(() => {
    const trimmed = searchQuery.trim();
    if (!trimmed) {
      setDebouncedSearch("");
      return;
    }
    const timer = window.setTimeout(() => {
      setDebouncedSearch(trimmed);
    }, 150);
    return () => window.clearTimeout(timer);
  }, [searchQuery]);

  // Clear batch selection whenever visible filter/page/role scope changes
  useEffect(() => {
    setSelectedIds([]);
  }, [
    isLead,
    page,
    statusFilter,
    datasetFilter,
    taskFilter,
    adapterFilter,
    passedFilter,
    downgradedOnly,
    debouncedSearch,
    sortBy,
    sortDir,
  ]);

  // Lead batch job, ingest & workspace settings state
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [vlmStage, setVlmStage] = useState<string>("");
  const [localActiveJob, setLocalActiveJob] = useState<JobStatus | null>(null);
  const [ingestOpen, setIngestOpen] = useState(false);
  const [vlmModalOpen, setVlmModalOpen] = useState(false);

  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsLoading, setSettingsLoading] = useState(false);
  const [settingsSaving, setSettingsSaving] = useState(false);
  const [settingsError, setSettingsError] = useState<string | null>(null);
  const [minWatchCoverage, setMinWatchCoverage] = useState<number>(0.95);
  const [batchConcurrency, setBatchConcurrency] = useState<number>(4);
  const settingsWrapRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!settingsOpen) return;
    const onPointerDown = (e: MouseEvent) => {
      if (settingsWrapRef.current && !settingsWrapRef.current.contains(e.target as Node)) {
        setSettingsOpen(false);
      }
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setSettingsOpen(false);
    };
    document.addEventListener("mousedown", onPointerDown);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [settingsOpen]);

  const handleToggleSettings = async () => {
    if (settingsOpen) {
      setSettingsOpen(false);
      return;
    }
    setSettingsOpen(true);
    setSettingsLoading(true);
    setSettingsError(null);
    try {
      const cfg = await getSettings();
      setMinWatchCoverage(cfg.min_watch_coverage);
      setBatchConcurrency(cfg.batch_vlm_concurrency);
    } catch (err) {
      setSettingsError(err instanceof Error ? err.message : t("queue.errLoadSettings"));
    } finally {
      setSettingsLoading(false);
    }
  };

  const handleSaveSettings = async (e: React.FormEvent) => {
    e.preventDefault();
    const cov = Number(minWatchCoverage);
    const conc = Math.trunc(Number(batchConcurrency));
    if (!Number.isFinite(cov) || cov < 0 || cov > 1) {
      setSettingsError(t("queue.errMinWatchRange"));
      return;
    }
    if (!Number.isInteger(conc) || conc < 1 || conc > 32) {
      setSettingsError(t("queue.errConcurrencyRange"));
      return;
    }
    setSettingsSaving(true);
    setSettingsError(null);
    try {
      const updated = await updateSettings({
        min_watch_coverage: Number(cov.toFixed(2)),
        batch_vlm_concurrency: conc,
      });
      setMinWatchCoverage(updated.min_watch_coverage);
      setBatchConcurrency(updated.batch_vlm_concurrency);
      setSettingsOpen(false);
      toast.showToast({
        message: t("queue.toastSettingsSaved", {
          pct: Math.round(updated.min_watch_coverage * 100),
          concurrency: updated.batch_vlm_concurrency,
        }),
        variant: "success",
        testId: "queue-settings-toast",
      });
    } catch (err) {
      setSettingsError(err instanceof Error ? err.message : t("queue.errSaveSettings"));
    } finally {
      setSettingsSaving(false);
    }
  };

  const toggleSortColumn = (field: EpisodeSortField) => {
    setPage(0);
    if (sortBy === field) {
      setSortDir((prev) => (prev === "asc" ? "desc" : "asc"));
    } else {
      setSortBy(field);
      setSortDir(field === "episode_id" ? "asc" : "desc");
    }
  };

  const renderSortIcon = (field: EpisodeSortField) => {
    if (sortBy !== field) return <ArrowUpDown size={12} />;
    return sortDir === "asc" ? <ArrowUp size={12} /> : <ArrowDown size={12} />;
  };

  const fetchQueue = useCallback(async () => {
    if (authProbing) return;
    const seq = ++queueSeqRef.current;
    setLoading(true);
    setError(null);
    try {
      const filters = isLead
        ? {
            status: statusFilter,
            dataset: datasetFilter.trim() || undefined,
            task_id: taskFilter.trim() || undefined,
            adapter_type: adapterFilter || undefined,
            passed: passedFilter === "all" ? null : passedFilter === "true",
            guardrail_downgraded: downgradedOnly || undefined,
            search: debouncedSearch || undefined,
            sort_by: sortBy,
            sort_dir: sortDir,
            limit: PAGE_SIZE,
            offset: page * PAGE_SIZE,
          }
        : reviewerTab === "vlm_ready"
          ? { status: "vlm_ready", limit: PAGE_SIZE, offset: page * PAGE_SIZE }
          : reviewerTab === "my_claimed"
            ? {
                status: "claimed",
                claimed_by: username,
                limit: PAGE_SIZE,
                offset: page * PAGE_SIZE,
              }
            : {
                status: "reviewed,flagged",
                limit: PAGE_SIZE,
                offset: page * PAGE_SIZE,
              };

      const res = await listEpisodes(filters);
      if (seq !== queueSeqRef.current) return;
      if (isLead) {
        setServerStatusCounts(res.status_counts ?? null);
        setFacets(res.facets ?? null);
      }
      if (res.items.length === 0 && res.total > 0 && page > 0) {
        const clampedPage = Math.max(0, Math.ceil(res.total / PAGE_SIZE) - 1);
        setEpisodes([]);
        setTotalEpisodes(res.total);
        if (res.my_claimed_count !== undefined) {
          setMyClaimedCount(res.my_claimed_count);
        }
        setPage(clampedPage);
        return;
      }
      setEpisodes(res.items);
      setTotalEpisodes(res.total);
      setMyClaimedCount(
        res.my_claimed_count ??
          res.items.filter((e) => e.status === "claimed" && e.claimed_by === username).length,
      );
    } catch (err) {
      if (seq !== queueSeqRef.current) return;
      const status = (err as { status?: number })?.status;
      if (status === 401) {
        setAuthPromptOpen(true);
      }
      setError(err instanceof Error ? err.message : tRef.current("queue.errLoadQueue"));
    } finally {
      if (seq === queueSeqRef.current) {
        setLoading(false);
      }
    }
  }, [
    authProbing,
    isLead,
    token,
    noAuth,
    username,
    page,
    statusFilter,
    datasetFilter,
    taskFilter,
    adapterFilter,
    passedFilter,
    downgradedOnly,
    debouncedSearch,
    sortBy,
    sortDir,
    reviewerTab,
    setAuthPromptOpen,
  ]);

  useEffect(() => {
    void fetchQueue();
  }, [fetchQueue]);

  // Global SSE notifications (supports both token-auth and --no-auth modes)
  const canStreamEvents = !authProbing && (Boolean(token) || noAuth);
  useEventStream({
    url: canStreamEvents ? "/api/events" : null,
    enabled: canStreamEvents,
    onMessage: (evt: Record<string, unknown>) => {
      const eventName =
        typeof evt.type === "string"
          ? evt.type
          : typeof evt.event === "string"
            ? evt.event
            : "";
      if (QUEUE_REFRESH_EVENTS.has(eventName)) {
        void fetchQueue();
      }
    },
  });

  // Watch jobStore.jobs terminal transitions so cancelling/finishing from JobCenterDrawer refreshes queue immediately
  const prevJobStatusesRef = useRef<Map<string, JobStatus["status"]>>(new Map());
  useEffect(() => {
    if (!jobStore.isProvided) return;
    let shouldRefresh = false;
    const prevMap = prevJobStatusesRef.current;
    for (const job of jobStore.jobs) {
      const prev = prevMap.get(job.job_id);
      prevMap.set(job.job_id, job.status);
      const wasActive = prev === "queued" || prev === "running";
      const isTerminal =
        job.status === "completed" || job.status === "failed" || job.status === "cancelled";
      if (wasActive && isTerminal) {
        shouldRefresh = true;
      }
    }
    if (shouldRefresh) {
      void fetchQueue();
    }
  }, [jobStore.isProvided, jobStore.jobs, fetchQueue]);

  // Fallback per-job SSE stream when QueuePage is mounted standalone without JobProvider
  useEventStream<JobStatus>({
    url:
      localActiveJob && !jobStore.isProvided
        ? `/api/jobs/${encodeURIComponent(localActiveJob.job_id)}/events`
        : null,
    enabled: Boolean(localActiveJob) && !jobStore.isProvided,
    isTerminal: (j) => j.status === "completed" || j.status === "failed" || j.status === "cancelled",
    onMessage: (jobEvt) => {
      setLocalActiveJob(jobEvt);
      if (jobEvt.status === "completed" || jobEvt.status === "failed" || jobEvt.status === "cancelled") {
        void fetchQueue();
      }
    },
  });

  const activeJob = useMemo(() => {
    if (!localActiveJob) return null;
    if (jobStore.isProvided) {
      return jobStore.jobs.find((j) => j.job_id === localActiveJob.job_id) ?? localActiveJob;
    }
    return localActiveJob;
  }, [localActiveJob, jobStore.isProvided, jobStore.jobs]);

  const reviewerFiltered = useMemo(() => {
    if (isLead) return episodes;
    if (reviewerTab === "vlm_ready") {
      return episodes.filter((ep) => ep.status === "vlm_ready");
    }
    if (reviewerTab === "my_claimed") {
      return episodes.filter((ep) => ep.status === "claimed" && ep.claimed_by === username);
    }
    return episodes.filter((ep) => ep.status === "reviewed" || ep.status === "flagged");
  }, [episodes, isLead, reviewerTab, username]);

  const clientStatusCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const st of ALL_EPISODE_STATUSES) counts[st] = 0;
    for (const ep of episodes) {
      counts[ep.status] = (counts[ep.status] ?? 0) + 1;
    }
    return counts;
  }, [episodes]);

  const statusCounts = serverStatusCounts ?? clientStatusCounts;
  const allCount = useMemo(() => {
    if (serverStatusCounts) {
      return Object.values(serverStatusCounts).reduce((sum, val) => sum + (Number(val) || 0), 0);
    }
    return episodes.length;
  }, [serverStatusCounts, episodes.length]);

  const adapterOptions = useMemo(() => {
    const set = new Set<string>(KNOWN_ADAPTERS);
    for (const item of facets?.adapter_types ?? []) {
      if (item) set.add(item);
    }
    return Array.from(set);
  }, [facets]);

  const handleClaimNext = async () => {
    setClaimingNext(true);
    setError(null);
    try {
      const res = await listEpisodes({ status: "vlm_ready", limit: 10 });
      const candidates = (res.items ?? []).filter((ep) => ep.status === "vlm_ready");
      if (candidates.length === 0) {
        setError(t("queue.errNoReadyClaim"));
        return;
      }
      let lastErr: unknown = null;
      for (const candidate of candidates) {
        try {
          await claimEpisode(candidate.episode_id, username);
          navigate(`/review/${encodeEpPath(candidate.episode_id)}`);
          return;
        } catch (err) {
          const status = (err as { status?: number })?.status;
          if (status === 409) {
            lastErr = err;
            continue;
          }
          throw err;
        }
      }
      setError(
        lastErr instanceof Error
          ? t("queue.errClaimConflict", { msg: lastErr.message })
          : t("queue.errNoClaimable"),
      );
      void fetchQueue();
    } catch (err) {
      setError(err instanceof Error ? err.message : t("queue.errClaimFailed"));
    } finally {
      setClaimingNext(false);
    }
  };

  const toggleSelect = (epId: string) => {
    setSelectedIds((prev) =>
      prev.includes(epId) ? prev.filter((id) => id !== epId) : [...prev, epId],
    );
  };

  const handleTrackSubmittedJob = (job: JobStatus) => {
    setLocalActiveJob(job);
    jobStore.trackJob(job);
    toast.showToast({
      message: t("queue.toastJobSubmitted", { jobId: job.job_id, jobType: job.job_type }),
      variant: "info",
      testId: "queue-job-toast",
    });
  };

  const handleTriggerJob = async (jobType: "batch_vlm" | "render_mp4") => {
    if (selectedIds.length === 0) {
      toast.showToast({
        message: t("queue.toastSelectOne"),
        variant: "warning",
        testId: "queue-job-toast",
      });
      return;
    }
    try {
      const job = await createJob({
        job_type: jobType,
        episode_ids: selectedIds,
        stage: jobType === "batch_vlm" && vlmStage ? vlmStage : undefined,
      });
      handleTrackSubmittedJob(job);
      if (jobType === "batch_vlm") {
        void fetchQueue();
      }
    } catch (err) {
      toast.showToast({
        message: err instanceof Error ? err.message : t("queue.toastCreateJobFailed"),
        variant: "error",
        testId: "queue-job-toast",
      });
    }
  };

  const handleExportSelected = () => {
    if (selectedIds.length === 0) {
      toast.showToast({
        message: t("queue.toastSelectOne"),
        variant: "warning",
        testId: "queue-job-toast",
      });
      return;
    }
    const selectedSet = new Set(selectedIds);
    const selectedRows = episodes.filter((ep) => selectedSet.has(ep.episode_id));
    const statuses = Array.from(new Set(selectedRows.map((ep) => ep.status)));
    const datasets = Array.from(new Set(selectedRows.map((ep) => ep.dataset).filter(Boolean)));
    const params = new URLSearchParams();
    params.set("episode_ids", selectedIds.join(","));
    if (statuses.length > 0) {
      params.set("include_status", statuses.join(","));
    }
    if (datasets.length === 1) {
      params.set("dataset", datasets[0]);
    }
    navigate(`/export?${params.toString()}`);
  };

  const handleCancelActiveJob = async (jobId: string) => {
    const updated = jobStore.isProvided
      ? await jobStore.cancelJob(jobId)
      : await apiCancelJob(jobId).catch(() => null);
    if (updated) {
      prevJobStatusesRef.current.set(updated.job_id, updated.status);
      setLocalActiveJob(updated);
      if (!jobStore.isProvided && updated.status === "cancelled") {
        toast.showToast({
          message:
            updated.job_type === "batch_vlm"
              ? t("queue.toastVlmCancelled", { jobId: updated.job_id })
              : t("queue.toastJobCancelled", { jobId: updated.job_id, jobType: updated.job_type }),
          variant: "warning",
          testId: "queue-job-toast",
        });
      }
      void fetchQueue();
    }
  };

  const leadRows = useMemo(() => {
    if (!adapterFilter) return episodes;
    return episodes.filter((ep) => ep.adapter_type === adapterFilter);
  }, [episodes, adapterFilter]);

  const displayRows = isLead ? leadRows : reviewerFiltered;

  return (
    <div className="page-container" data-testid="queue-page">
      <div className="page-header">
        <div>
          <h1>{isLead ? t("queue.titleLead") : t("queue.titleReviewer")}</h1>
          <p className="subtitle">
            {isLead ? t("queue.subtitleLead") : t("queue.subtitleReviewer")}
          </p>
        </div>

        <div className="header-actions">
          {!isLead && (
            <button
              type="button"
              className="primary-btn"
              disabled={claimingNext}
              onClick={handleClaimNext}
              data-testid="claim-next-btn"
            >
              <Play size={15} /> {claimingNext ? t("queue.claiming") : t("queue.claimNext")}
            </button>
          )}
          {isLead && (
            <>
              <button
                type="button"
                className="accent-btn"
                onClick={() => setIngestOpen(true)}
                data-testid="open-ingest-modal-btn"
              >
                <FolderPlus size={15} /> {t("queue.ingestData")}
              </button>
              <div className="settings-popover-anchor" ref={settingsWrapRef}>
                <button
                  type="button"
                  className="secondary-btn"
                  onClick={() => void handleToggleSettings()}
                  data-testid="open-settings-popover-btn"
                  aria-expanded={settingsOpen}
                >
                  <Settings size={14} /> {t("queue.workspaceSettings")}
                </button>
                {settingsOpen && (
                  <form
                    className="settings-popover"
                    data-testid="workspace-settings-popover"
                    onSubmit={(e) => void handleSaveSettings(e)}
                  >
                    <div className="section-header-row">
                      <strong>{t("queue.workspaceSettings")}</strong>
                      <button
                        type="button"
                        className="mini-btn"
                        onClick={() => setSettingsOpen(false)}
                        aria-label={t("queue.closeSettingsAria")}
                      >
                        <X size={13} />
                      </button>
                    </div>
                    {settingsLoading ? (
                      <div className="muted small">{t("queue.loadingSettings")}</div>
                    ) : (
                      <>
                        <label className="field-group">
                          <span>
                            {t("queue.minWatchCoverage", {
                              pct: Math.round(minWatchCoverage * 100),
                            })}
                          </span>
                          <input
                            type="number"
                            min={0}
                            max={1}
                            step={0.05}
                            value={minWatchCoverage}
                            data-testid="setting-min-watch-coverage"
                            onChange={(e) => setMinWatchCoverage(Number(e.target.value))}
                          />
                        </label>
                        <label className="field-group">
                          <span>{t("queue.batchConcurrency")}</span>
                          <input
                            type="number"
                            min={1}
                            max={32}
                            step={1}
                            value={batchConcurrency}
                            data-testid="setting-batch-concurrency"
                            onChange={(e) => setBatchConcurrency(Number(e.target.value))}
                          />
                        </label>
                        {settingsError && <div className="error-banner small">{settingsError}</div>}
                        <div className="modal-actions">
                          <button
                            type="button"
                            className="secondary-btn compact"
                            onClick={() => setSettingsOpen(false)}
                          >
                            {t("common.cancel")}
                          </button>
                          <button
                            type="submit"
                            className="primary-btn compact"
                            disabled={settingsSaving}
                            data-testid="save-settings-btn"
                          >
                            {settingsSaving ? t("common.saving") : t("queue.saveSettings")}
                          </button>
                        </div>
                        <div className="settings-popover-divider" />
                        <button
                          type="button"
                          className="secondary-btn compact"
                          data-testid="settings-open-vlm-modal-btn"
                          onClick={() => {
                            setSettingsOpen(false);
                            setVlmModalOpen(true);
                          }}
                        >
                          <Cpu size={13} /> {t("queue.configureVlm")}
                        </button>
                      </>
                    )}
                  </form>
                )}
              </div>
            </>
          )}
          <button type="button" className="secondary-btn" onClick={() => void fetchQueue()}>
            <RefreshCw size={14} /> {t("common.refresh")}
          </button>
        </div>
      </div>

      {/* Reviewer Role Tabs & Personal Stats */}
      {!isLead && (
        <div className="reviewer-bar" data-testid="reviewer-queue-controls">
          <div className="tab-group">
            <button
              type="button"
              className={`tab-btn ${reviewerTab === "vlm_ready" ? "active" : ""}`}
              onClick={() => {
                setPage(0);
                setReviewerTab("vlm_ready");
              }}
            >
              {t("queue.tabReadyForReview")}
            </button>
            <button
              type="button"
              className={`tab-btn ${reviewerTab === "my_claimed" ? "active" : ""}`}
              onClick={() => {
                setPage(0);
                setReviewerTab("my_claimed");
              }}
            >
              {t("queue.tabMyClaimed", { count: myClaimedCount })}
            </button>
            <button
              type="button"
              className={`tab-btn ${reviewerTab === "submitted" ? "active" : ""}`}
              onClick={() => {
                setPage(0);
                setReviewerTab("submitted");
              }}
            >
              {t("queue.tabSubmittedFlagged")}
            </button>
          </div>
          <div className="reviewer-stats">
            <span>{t("queue.operator", { username })}</span>
          </div>
        </div>
      )}

      {/* Lead 9-Status Summary + Filter Bar + Batch Controls */}
      {isLead && (
        <>
          <div className="status-strip" data-testid="lead-status-strip">
            <button
              type="button"
              className={`status-pill ${statusFilter === "" ? "active" : ""}`}
              onClick={() => {
                setPage(0);
                setStatusFilter("");
              }}
            >
              {t("queue.allCount", { count: allCount })}
            </button>
            {ALL_EPISODE_STATUSES.map((st) => (
              <button
                key={st}
                type="button"
                className={`status-pill ${statusFilter === st ? "active" : ""}`}
                onClick={() => {
                  setPage(0);
                  setStatusFilter(st);
                }}
              >
                {getEpisodeStatusLabel(st, locale)} ({statusCounts[st] ?? 0})
              </button>
            ))}
          </div>

          <div className="filter-toolbar" data-testid="lead-filter-toolbar">
            <input
              type="text"
              aria-label={t("queue.searchAria")}
              placeholder={t("queue.searchPlaceholder")}
              data-testid="queue-search-input"
              value={searchQuery}
              onChange={(e) => {
                setPage(0);
                setSearchQuery(e.target.value);
              }}
            />
            <FacetCombobox
              placeholder={t("queue.filterDatasetPlaceholder")}
              value={datasetFilter}
              onChange={(val) => {
                setPage(0);
                setDatasetFilter(val);
              }}
              options={facets?.datasets ?? []}
            />
            <FacetCombobox
              placeholder={t("queue.filterTaskPlaceholder")}
              value={taskFilter}
              onChange={(val) => {
                setPage(0);
                setTaskFilter(val);
              }}
              options={facets?.task_ids ?? []}
            />
            <select
              aria-label={t("queue.adapterFilterAria")}
              data-testid="adapter-filter-select"
              value={adapterFilter}
              onChange={(e) => {
                setPage(0);
                setAdapterFilter(e.target.value);
              }}
            >
              <option value="">{t("queue.allAdapters")}</option>
              {adapterOptions.map((ad) => (
                <option key={ad} value={ad}>
                  {ad}
                </option>
              ))}
            </select>
            <select
              aria-label={t("queue.outcomeFilterAria")}
              value={passedFilter}
              onChange={(e) => {
                setPage(0);
                setPassedFilter(e.target.value as "all" | "true" | "false");
              }}
            >
              <option value="all">{t("queue.allOutcomes")}</option>
              <option value="true">{t("queue.passedOnly")}</option>
              <option value="false">{t("queue.failedOnly")}</option>
            </select>
            <select
              aria-label={t("queue.sortFieldAria")}
              data-testid="queue-sort-by"
              value={sortBy}
              onChange={(e) => {
                setPage(0);
                setSortBy(e.target.value as EpisodeSortField);
              }}
            >
              <option value="episode_id">{t("queue.sortEpisodeId")}</option>
              <option value="updated_at">{t("queue.sortUpdatedAt")}</option>
              <option value="duration_s">{t("queue.sortDuration")}</option>
              <option value="cut_ratio">{t("queue.sortCutRatio")}</option>
              <option value="mask_ratio">{t("queue.sortMaskRatio")}</option>
              <option value="badcase_count">{t("queue.sortBadcaseCount")}</option>
            </select>
            <select
              aria-label={t("queue.sortDirAria")}
              data-testid="queue-sort-dir"
              value={sortDir}
              onChange={(e) => {
                setPage(0);
                setSortDir(e.target.value as SortDir);
              }}
            >
              <option value="asc">{t("queue.sortAsc")}</option>
              <option value="desc">{t("queue.sortDesc")}</option>
            </select>
            <label className="checkbox-inline">
              <input
                type="checkbox"
                checked={downgradedOnly}
                onChange={(e) => {
                  setPage(0);
                  setDowngradedOnly(e.target.checked);
                }}
              />
              <span>{t("queue.downgradedOnly")}</span>
            </label>
          </div>

          <div className="batch-job-bar" data-testid="lead-batch-controls">
            <span>{t("queue.selectedCount", { count: selectedIds.length })}</span>
            <select
              aria-label={t("queue.rerunStageAria")}
              data-testid="batch-vlm-stage-select"
              value={vlmStage}
              onChange={(e) => setVlmStage(e.target.value)}
            >
              <option value="">{t("queue.stageFullDag")}</option>
              <option value="s1_grounding">{t("queue.stageS1Only")}</option>
              <option value="s3_s4">{t("queue.stageS3S4")}</option>
              <option value="s3_discover">{t("queue.stageS3Only")}</option>
              <option value="s4_verify">{t("queue.stageS4Only")}</option>
              <option value="s5_success">{t("queue.stageS5Only")}</option>
            </select>
            <button
              type="button"
              className="primary-btn compact"
              onClick={() => void handleTriggerJob("batch_vlm")}
              data-testid="batch-vlm-btn"
            >
              <Play size={14} /> {t("queue.runBatchVlm")}
            </button>
            <button
              type="button"
              className="secondary-btn compact"
              onClick={() => void handleTriggerJob("render_mp4")}
              data-testid="batch-render-btn"
            >
              <Video size={14} /> {t("queue.batchRenderMp4")}
            </button>
            <button
              type="button"
              className="accent-btn compact"
              disabled={selectedIds.length === 0}
              onClick={handleExportSelected}
              data-testid="queue-export-selected-btn"
              title={
                selectedIds.length === 0
                  ? t("queue.exportSelectedEmptyTitle")
                  : t("queue.exportSelectedTitle", { count: selectedIds.length })
              }
            >
              <PackageCheck size={14} /> {t("queue.exportSelected", { count: selectedIds.length })}
            </button>
          </div>

          {activeJob && (
            <div className="job-progress-banner" data-testid="active-job-banner">
              <div>
                <strong>
                  {t("queue.jobBannerTitle", { jobId: activeJob.job_id, jobType: activeJob.job_type })}
                </strong>{" "}
                —{" "}
                {t("queue.jobBannerStatus", {
                  status: activeJob.status,
                  done: activeJob.progress_done,
                  total: activeJob.progress_total,
                })}
                {activeJob.summary?.errors && activeJob.summary.errors.length > 0 && (
                  <div className="small delta-neg" data-testid="active-job-error-summary">
                    {t("common.error")}:{" "}
                    {activeJob.summary.errors[0].episode_id
                      ? `${activeJob.summary.errors[0].episode_id}: `
                      : ""}
                    {activeJob.summary.errors[0].error || t("queue.jobExecFailed")}
                    {activeJob.summary.errors.length > 1
                      ? ` (${t("common.more", { count: activeJob.summary.errors.length - 1 })})`
                      : ""}
                  </div>
                )}
              </div>
              <div className="header-actions">
                {(activeJob.status === "queued" || activeJob.status === "running") && (
                  <button
                    type="button"
                    className="danger-btn compact"
                    onClick={() => void handleCancelActiveJob(activeJob.job_id)}
                  >
                    {t("queue.cancelJob")}
                  </button>
                )}
                {activeJob.status === "completed" &&
                  (activeJob.summary?.artifacts ?? []).map((art, idx) => (
                    <a
                      key={`${art.name}-${idx}`}
                      className="link-btn small"
                      href={jobStore.artifactDownloadUrl(activeJob.job_id, idx)}
                      download={art.name || undefined}
                    >
                      <Download size={13} />{" "}
                      {t("queue.downloadArtifact", {
                        name: art.name || t("queue.artifactFallback", { index: idx + 1 }),
                      })}
                    </a>
                  ))}
                {(activeJob.status === "completed" ||
                  activeJob.status === "failed" ||
                  activeJob.status === "cancelled") && (
                  <button
                    type="button"
                    className="mini-btn"
                    data-testid="dismiss-active-job-btn"
                    onClick={() => setLocalActiveJob(null)}
                    aria-label={t("queue.dismissBannerAria")}
                  >
                    <X size={13} /> {t("common.dismiss")}
                  </button>
                )}
              </div>
            </div>
          )}
        </>
      )}

      {error && <div className="error-banner">{error}</div>}

      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              {isLead && (
                <th>
                  <input
                    type="checkbox"
                    aria-label={t("queue.selectAllAria")}
                    checked={
                      displayRows.length > 0 &&
                      displayRows.every((r) => selectedIds.includes(r.episode_id))
                    }
                    onChange={(e) =>
                      setSelectedIds(e.target.checked ? displayRows.map((r) => r.episode_id) : [])
                    }
                  />
                </th>
              )}
              <th>
                {isLead ? (
                  <button
                    type="button"
                    className={`sort-col-btn ${sortBy === "episode_id" ? "active" : ""}`}
                    data-testid="sort-col-episode_id"
                    onClick={() => toggleSortColumn("episode_id")}
                  >
                    {t("queue.colEpisodeId")} {renderSortIcon("episode_id")}
                  </button>
                ) : (
                  t("queue.colEpisodeId")
                )}
              </th>
              <th>{t("queue.colTaskAdapter")}</th>
              <th>{t("queue.colStatus")}</th>
              <th>
                {isLead ? (
                  <button
                    type="button"
                    className={`sort-col-btn ${sortBy === "duration_s" ? "active" : ""}`}
                    data-testid="sort-col-duration_s"
                    onClick={() => toggleSortColumn("duration_s")}
                  >
                    {t("queue.colDurationFps")} {renderSortIcon("duration_s")}
                  </button>
                ) : (
                  t("queue.colDurationFps")
                )}
              </th>
              <th>{t("queue.colOutcome")}</th>
              <th>
                {isLead ? (
                  <div className="sort-col-group">
                    <button
                      type="button"
                      className={`sort-col-btn ${sortBy === "badcase_count" ? "active" : ""}`}
                      data-testid="sort-col-badcase_count"
                      onClick={() => toggleSortColumn("badcase_count")}
                    >
                      {t("queue.colBadcase")} {renderSortIcon("badcase_count")}
                    </button>
                    <span>/</span>
                    <button
                      type="button"
                      className={`sort-col-btn ${sortBy === "cut_ratio" ? "active" : ""}`}
                      data-testid="sort-col-cut_ratio"
                      onClick={() => toggleSortColumn("cut_ratio")}
                    >
                      {t("queue.colCut")} {renderSortIcon("cut_ratio")}
                    </button>
                    <span>/ {t("queue.colMask")}</span>
                  </div>
                ) : (
                  t("queue.colBadcaseCutMask")
                )}
              </th>
              {isLead && (
                <th>
                  <button
                    type="button"
                    className={`sort-col-btn ${sortBy === "updated_at" ? "active" : ""}`}
                    data-testid="sort-col-updated_at"
                    onClick={() => toggleSortColumn("updated_at")}
                  >
                    {t("queue.colUpdated")} {renderSortIcon("updated_at")}
                  </button>
                </th>
              )}
              <th>{t("queue.colClaimedBy")}</th>
              <th>{t("queue.colAction")}</th>
            </tr>
          </thead>
          <tbody>
            {loading && displayRows.length === 0 ? (
              <tr>
                <td colSpan={isLead ? 10 : 8} className="empty-cell">
                  {t("common.loading")}
                </td>
              </tr>
            ) : displayRows.length === 0 ? (
              <tr>
                <td colSpan={isLead ? 10 : 8} className="empty-cell">
                  {t("queue.emptyTable")}
                </td>
              </tr>
            ) : (
              displayRows.map((ep) => {
                const meta = EPISODE_STATUS_META[ep.status];
                const statusLabel = getEpisodeStatusLabel(ep.status, locale);
                const statusColor = meta?.color ?? "#64748b";
                const passedText =
                  ep.resolved_passed === null || ep.resolved_passed === undefined
                    ? t("queue.outcomeUnsure")
                    : ep.resolved_passed
                      ? t("queue.outcomePass")
                      : t("queue.outcomeFail");
                return (
                  <tr key={ep.episode_id} data-testid={`episode-row-${ep.episode_id}`}>
                    {isLead && (
                      <td>
                        <input
                          type="checkbox"
                          checked={selectedIds.includes(ep.episode_id)}
                          onChange={() => toggleSelect(ep.episode_id)}
                        />
                      </td>
                    )}
                    <td className="mono">{ep.episode_id}</td>
                    <td>
                      <div>{ep.task_id}</div>
                      <div className="muted small">{ep.adapter_type}</div>
                    </td>
                    <td>
                      <span className="status-tag" style={{ backgroundColor: statusColor }}>
                        {statusLabel}
                      </span>
                      {ep.guardrail_downgraded && (
                        <span
                          className="inline-warn"
                          title={t("queue.guardrailDowngradedTitle")}
                        >
                          <ShieldAlert size={13} />
                        </span>
                      )}
                    </td>
                    <td>
                      {ep.duration_s.toFixed(1)}s · {ep.fps.toFixed(0)}fps
                    </td>
                    <td>{passedText}</td>
                    <td>
                      {t("queue.rowStats", {
                        segs: ep.badcase_count ?? 0,
                        cut: ((ep.cut_ratio ?? 0) * 100).toFixed(0),
                        mask: ((ep.mask_ratio ?? 0) * 100).toFixed(0),
                      })}
                    </td>
                    {isLead && <td className="mono small muted">{formatUpdatedAt(ep.updated_at)}</td>}
                    <td>{ep.claimed_by ?? "—"}</td>
                    <td>
                      <Link className="link-btn" to={`/review/${encodeEpPath(ep.episode_id)}`}>
                        {t("queue.openStudio")}
                      </Link>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {(page > 0 || totalEpisodes > PAGE_SIZE) && (
        <div className="pagination-bar" data-testid="queue-pagination">
          <button
            type="button"
            className="secondary-btn compact"
            disabled={page === 0}
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            data-testid="queue-prev-page"
          >
            {t("queue.prevPage")}
          </button>
          <span className="muted small">
            {t("queue.pageSummary", {
              current: page + 1,
              pages: Math.max(1, Math.ceil(totalEpisodes / PAGE_SIZE)),
              total: totalEpisodes,
            })}
          </span>
          <button
            type="button"
            className="secondary-btn compact"
            disabled={(page + 1) * PAGE_SIZE >= totalEpisodes}
            onClick={() => setPage((p) => p + 1)}
            data-testid="queue-next-page"
          >
            {t("queue.nextPage")}
          </button>
        </div>
      )}

      {isLead && (
        <>
          <IngestModal
            open={ingestOpen}
            onClose={() => setIngestOpen(false)}
            onJobCreated={handleTrackSubmittedJob}
          />
          <VlmSettingsModal open={vlmModalOpen} onClose={() => setVlmModalOpen(false)} />
        </>
      )}
      {toast.StandaloneHost}
    </div>
  );
};
