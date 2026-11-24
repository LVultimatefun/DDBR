import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  ArrowDown,
  ArrowUp,
  ArrowUpDown,
  Download,
  FolderPlus,
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
import { ALL_EPISODE_STATUSES, EPISODE_STATUS_META } from "../constants/labels";
import { useAuth } from "../stores/authStore";
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
      setSettingsError(err instanceof Error ? err.message : "Failed to load workspace settings");
    } finally {
      setSettingsLoading(false);
    }
  };

  const handleSaveSettings = async (e: React.FormEvent) => {
    e.preventDefault();
    const cov = Number(minWatchCoverage);
    const conc = Math.trunc(Number(batchConcurrency));
    if (!Number.isFinite(cov) || cov < 0 || cov > 1) {
      setSettingsError("min_watch_coverage must be between 0.0 and 1.0");
      return;
    }
    if (!Number.isInteger(conc) || conc < 1 || conc > 32) {
      setSettingsError("batch_vlm_concurrency must be an integer between 1 and 32");
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
        message: `Workspace settings saved (min watch ${Math.round(updated.min_watch_coverage * 100)}%, VLM concurrency ${updated.batch_vlm_concurrency})`,
        variant: "success",
        testId: "queue-settings-toast",
      });
    } catch (err) {
      setSettingsError(err instanceof Error ? err.message : "Failed to save workspace settings");
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
      setError(err instanceof Error ? err.message : "Failed to load queue");
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
        setError("No episodes currently ready for review (vlm_ready)");
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
          ? `Claim conflict: all candidates were already claimed (${lastErr.message})`
          : "No claimable episodes currently available",
      );
      void fetchQueue();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to claim episode");
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
      message: `Submitted job ${job.job_id} (${job.job_type})`,
      variant: "info",
      testId: "queue-job-toast",
    });
  };

  const handleTriggerJob = async (jobType: "batch_vlm" | "render_mp4") => {
    if (selectedIds.length === 0) {
      toast.showToast({
        message: "Select at least one episode first",
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
        message: err instanceof Error ? err.message : "Failed to create batch job",
        variant: "error",
        testId: "queue-job-toast",
      });
    }
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
              ? `Job ${updated.job_id} cancelled; unfinished episode statuses reverted`
              : `Job ${updated.job_id} (${updated.job_type}) cancelled`,
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
          <h1>{isLead ? "Global QC Queue & Batch Dispatch (Lead)" : "Episode Review Queue (Reviewer)"}</h1>
          <p className="subtitle">
            {isLead
              ? "Filter across all statuses, trigger batch VLM / 1080p MP4 jobs, and adjudicate reviews"
              : "Claim episodes, refine 3-layer annotations (QC segments / success / macro stages), and flag edge cases"}
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
              <Play size={15} /> {claimingNext ? "Claiming..." : "Claim Next Episode"}
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
                <FolderPlus size={15} /> Ingest Data
              </button>
              <div className="settings-popover-anchor" ref={settingsWrapRef}>
                <button
                  type="button"
                  className="secondary-btn"
                  onClick={() => void handleToggleSettings()}
                  data-testid="open-settings-popover-btn"
                  aria-expanded={settingsOpen}
                >
                  <Settings size={14} /> Workspace Settings
                </button>
                {settingsOpen && (
                  <form
                    className="settings-popover"
                    data-testid="workspace-settings-popover"
                    onSubmit={(e) => void handleSaveSettings(e)}
                  >
                    <div className="section-header-row">
                      <strong>Workspace Settings</strong>
                      <button
                        type="button"
                        className="mini-btn"
                        onClick={() => setSettingsOpen(false)}
                        aria-label="Close settings popover"
                      >
                        <X size={13} />
                      </button>
                    </div>
                    {settingsLoading ? (
                      <div className="muted small">Loading settings...</div>
                    ) : (
                      <>
                        <label className="field-group">
                          <span>
                            Min Watch Coverage (min_watch_coverage · {Math.round(minWatchCoverage * 100)}%)
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
                          <span>Batch VLM Concurrency (batch_vlm_concurrency · 1–32)</span>
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
                            Cancel
                          </button>
                          <button
                            type="submit"
                            className="primary-btn compact"
                            disabled={settingsSaving}
                            data-testid="save-settings-btn"
                          >
                            {settingsSaving ? "Saving..." : "Save Settings"}
                          </button>
                        </div>
                      </>
                    )}
                  </form>
                )}
              </div>
            </>
          )}
          <button type="button" className="secondary-btn" onClick={() => void fetchQueue()}>
            <RefreshCw size={14} /> Refresh
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
              Ready for Review (vlm_ready)
            </button>
            <button
              type="button"
              className={`tab-btn ${reviewerTab === "my_claimed" ? "active" : ""}`}
              onClick={() => {
                setPage(0);
                setReviewerTab("my_claimed");
              }}
            >
              My Claimed ({myClaimedCount})
            </button>
            <button
              type="button"
              className={`tab-btn ${reviewerTab === "submitted" ? "active" : ""}`}
              onClick={() => {
                setPage(0);
                setReviewerTab("submitted");
              }}
            >
              Submitted / Flagged
            </button>
          </div>
          <div className="reviewer-stats">
            <span>Operator: {username}</span>
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
              All ({allCount})
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
                {EPISODE_STATUS_META[st].label} ({statusCounts[st] ?? 0})
              </button>
            ))}
          </div>

          <div className="filter-toolbar" data-testid="lead-filter-toolbar">
            <input
              type="text"
              aria-label="Search episodes"
              placeholder="Search episode ID..."
              data-testid="queue-search-input"
              value={searchQuery}
              onChange={(e) => {
                setPage(0);
                setSearchQuery(e.target.value);
              }}
            />
            <FacetCombobox
              placeholder="Filter by dataset (dataset)"
              value={datasetFilter}
              onChange={(val) => {
                setPage(0);
                setDatasetFilter(val);
              }}
              options={facets?.datasets ?? []}
            />
            <FacetCombobox
              placeholder="Filter by task (task_id)"
              value={taskFilter}
              onChange={(val) => {
                setPage(0);
                setTaskFilter(val);
              }}
              options={facets?.task_ids ?? []}
            />
            <select
              aria-label="Adapter filter"
              data-testid="adapter-filter-select"
              value={adapterFilter}
              onChange={(e) => {
                setPage(0);
                setAdapterFilter(e.target.value);
              }}
            >
              <option value="">All Adapters</option>
              {adapterOptions.map((ad) => (
                <option key={ad} value={ad}>
                  {ad}
                </option>
              ))}
            </select>
            <select
              aria-label="Outcome filter"
              value={passedFilter}
              onChange={(e) => {
                setPage(0);
                setPassedFilter(e.target.value as "all" | "true" | "false");
              }}
            >
              <option value="all">All Outcomes</option>
              <option value="true">Passed only (passed=true)</option>
              <option value="false">Failed only (passed=false)</option>
            </select>
            <select
              aria-label="Sort field"
              data-testid="queue-sort-by"
              value={sortBy}
              onChange={(e) => {
                setPage(0);
                setSortBy(e.target.value as EpisodeSortField);
              }}
            >
              <option value="episode_id">Sort: Episode ID</option>
              <option value="updated_at">Sort: Updated At</option>
              <option value="duration_s">Sort: Duration</option>
              <option value="cut_ratio">Sort: Cut Ratio</option>
              <option value="mask_ratio">Sort: Mask Ratio</option>
              <option value="badcase_count">Sort: Badcase Count</option>
            </select>
            <select
              aria-label="Sort direction"
              data-testid="queue-sort-dir"
              value={sortDir}
              onChange={(e) => {
                setPage(0);
                setSortDir(e.target.value as SortDir);
              }}
            >
              <option value="asc">Asc</option>
              <option value="desc">Desc</option>
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
              <span>Guardrail downgraded only</span>
            </label>
          </div>

          <div className="batch-job-bar" data-testid="lead-batch-controls">
            <span>Selected: {selectedIds.length}</span>
            <select
              aria-label="Rerun stage"
              data-testid="batch-vlm-stage-select"
              value={vlmStage}
              onChange={(e) => setVlmStage(e.target.value)}
            >
              <option value="">Full DAG (S1–S5)</option>
              <option value="s1_grounding">Rerun S1 only (refresh 2D boxes, reuse S2-S5 cache)</option>
              <option value="s3_s4">Rerun S3/S4 only (Discover + Verify)</option>
              <option value="s3_discover">Rerun S3 only (Discover)</option>
              <option value="s4_verify">Rerun S4 only (Verify)</option>
              <option value="s5_success">Rerun S5 only (Success)</option>
            </select>
            <button
              type="button"
              className="primary-btn compact"
              onClick={() => void handleTriggerJob("batch_vlm")}
              data-testid="batch-vlm-btn"
            >
              <Play size={14} /> Run Batch VLM
            </button>
            <button
              type="button"
              className="secondary-btn compact"
              onClick={() => void handleTriggerJob("render_mp4")}
              data-testid="batch-render-btn"
            >
              <Video size={14} /> Batch Render 1080p MP4
            </button>
          </div>

          {activeJob && (
            <div className="job-progress-banner" data-testid="active-job-banner">
              <div>
                <strong>
                  Job {activeJob.job_id} ({activeJob.job_type})
                </strong>{" "}
                — Status: {activeJob.status} ({activeJob.progress_done}/{activeJob.progress_total})
                {activeJob.summary?.errors && activeJob.summary.errors.length > 0 && (
                  <div className="small delta-neg" data-testid="active-job-error-summary">
                    Error: {activeJob.summary.errors[0].episode_id ? `${activeJob.summary.errors[0].episode_id}: ` : ""}
                    {activeJob.summary.errors[0].error || "Job execution failed"}
                    {activeJob.summary.errors.length > 1
                      ? ` (+${activeJob.summary.errors.length - 1} more)`
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
                    Cancel Job
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
                      <Download size={13} /> Download {art.name || `Artifact #${idx + 1}`}
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
                    aria-label="Dismiss job banner"
                  >
                    <X size={13} /> Dismiss
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
                    aria-label="Select all"
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
                    Episode ID {renderSortIcon("episode_id")}
                  </button>
                ) : (
                  "Episode ID"
                )}
              </th>
              <th>Task / Adapter</th>
              <th>Status</th>
              <th>
                {isLead ? (
                  <button
                    type="button"
                    className={`sort-col-btn ${sortBy === "duration_s" ? "active" : ""}`}
                    data-testid="sort-col-duration_s"
                    onClick={() => toggleSortColumn("duration_s")}
                  >
                    Duration / FPS {renderSortIcon("duration_s")}
                  </button>
                ) : (
                  "Duration / FPS"
                )}
              </th>
              <th>Outcome</th>
              <th>
                {isLead ? (
                  <div className="sort-col-group">
                    <button
                      type="button"
                      className={`sort-col-btn ${sortBy === "badcase_count" ? "active" : ""}`}
                      data-testid="sort-col-badcase_count"
                      onClick={() => toggleSortColumn("badcase_count")}
                    >
                      Badcase {renderSortIcon("badcase_count")}
                    </button>
                    <span>/</span>
                    <button
                      type="button"
                      className={`sort-col-btn ${sortBy === "cut_ratio" ? "active" : ""}`}
                      data-testid="sort-col-cut_ratio"
                      onClick={() => toggleSortColumn("cut_ratio")}
                    >
                      Cut {renderSortIcon("cut_ratio")}
                    </button>
                    <span>/ Mask</span>
                  </div>
                ) : (
                  "Badcase / Cut / Mask"
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
                    Updated {renderSortIcon("updated_at")}
                  </button>
                </th>
              )}
              <th>Claimed By</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {loading && displayRows.length === 0 ? (
              <tr>
                <td colSpan={isLead ? 10 : 8} className="empty-cell">
                  Loading...
                </td>
              </tr>
            ) : displayRows.length === 0 ? (
              <tr>
                <td colSpan={isLead ? 10 : 8} className="empty-cell">
                  No matching episodes found
                </td>
              </tr>
            ) : (
              displayRows.map((ep) => {
                const meta = EPISODE_STATUS_META[ep.status] ?? {
                  label: ep.status,
                  color: "#64748b",
                };
                const passedText =
                  ep.resolved_passed === null || ep.resolved_passed === undefined
                    ? "Unsure"
                    : ep.resolved_passed
                      ? "Pass"
                      : "Fail";
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
                      <span className="status-tag" style={{ backgroundColor: meta.color }}>
                        {meta.label}
                      </span>
                      {ep.guardrail_downgraded && (
                        <span className="inline-warn" title="Contains guardrail-downgraded segments">
                          <ShieldAlert size={13} />
                        </span>
                      )}
                    </td>
                    <td>
                      {ep.duration_s.toFixed(1)}s · {ep.fps.toFixed(0)}fps
                    </td>
                    <td>{passedText}</td>
                    <td>
                      {ep.badcase_count ?? 0} segs · cut {((ep.cut_ratio ?? 0) * 100).toFixed(0)}% ·
                      mask {((ep.mask_ratio ?? 0) * 100).toFixed(0)}%
                    </td>
                    {isLead && <td className="mono small muted">{formatUpdatedAt(ep.updated_at)}</td>}
                    <td>{ep.claimed_by ?? "—"}</td>
                    <td>
                      <Link className="link-btn" to={`/review/${encodeEpPath(ep.episode_id)}`}>
                        Open Studio
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
            Previous
          </button>
          <span className="muted small">
            Page {page + 1} / {Math.max(1, Math.ceil(totalEpisodes / PAGE_SIZE))} ({totalEpisodes}{" "}
            total)
          </span>
          <button
            type="button"
            className="secondary-btn compact"
            disabled={(page + 1) * PAGE_SIZE >= totalEpisodes}
            onClick={() => setPage((p) => p + 1)}
            data-testid="queue-next-page"
          >
            Next
          </button>
        </div>
      )}

      {isLead && (
        <IngestModal
          open={ingestOpen}
          onClose={() => setIngestOpen(false)}
          onJobCreated={handleTrackSubmittedJob}
        />
      )}
      {toast.StandaloneHost}
    </div>
  );
};
