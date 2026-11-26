import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import {
  AlertTriangle,
  ArrowLeft,
  CheckCircle,
  ChevronDown,
  Coins,
  Download,
  Edit3,
  ExternalLink,
  Eye,
  FileJson,
  Flag,
  History,
  Keyboard,
  Lock,
  Magnet,
  Play,
  Redo2,
  RefreshCw,
  Rows3,
  Save,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Tag,
  Undo2,
  Unlock,
  Video,
  X,
  XCircle,
} from "lucide-react";
import {
  ApiError,
  assignEpisodeTask,
  buildReviewWritePayload,
  cancelJob as apiCancelJob,
  claimEpisode,
  createJob,
  encodeEpPath,
  getEpisode,
  getJob,
  getLabLabels,
  getSettings,
  getSharedLabels,
  inferSubtasks,
  listEpisodeRuns,
  listEpisodes,
  listEpisodeTasks,
  listReviews,
  releaseClaim,
  renewClaim,
  submitReview,
} from "../api/client";
import type {
  EpisodeTaskSummary,
  HumanReviewArtifact,
  JobStatus,
  MacroStage,
  ResolvedEpisode,
  ReviewAction,
  SplicePolicy,
  SuccessVerdict,
  TaskSpec,
  UsageSummary,
  VisualLabelDef,
  VlmRunSummary,
  WatchCoverage,
} from "../api/types";
import { DiffOverlay } from "../components/DiffOverlay";
import { FourTrackTimeline, type StageSpanUpdate } from "../components/FourTrackTimeline";
import { SegmentInspector } from "../components/SegmentInspector";
import { ShortcutsModal } from "../components/ShortcutsModal";
import { StudioSkeleton } from "../components/SkeletonLoader";
import { StudioPlayer, type StudioPlayerHandle } from "../components/StudioPlayer";
import { SuccessPanel } from "../components/SuccessPanel";
import { TaskDraftWizardModal } from "../components/TaskDraftWizardModal";
import {
  createDefaultSegment,
  DEFAULT_VISUAL_LABELS,
  EPISODE_STATUS_META,
  normalizeSegmentTriplet,
} from "../constants/labels";
import { useAuth } from "../stores/authStore";
import { useI18n } from "../stores/i18nStore";
import { useJobs } from "../stores/jobStore";
import {
  canReviewerSubmit,
  MIN_WATCH_COVERAGE,
  useReviewHistory,
  WatchCoverageTracker,
} from "../stores/reviewHistoryStore";
import { useToast } from "../stores/toastStore";
import "../styles/studio-nle.css";

type InspectorTab = "segment" | "success" | "diff";

const SPLIT_STORAGE_KEY = "robo_vlm_qc.studio_split_pct";
const AUTO_ADVANCE_STORAGE_KEY = "robo_vlm_qc.auto_advance";
const SNAP_ENABLED_STORAGE_KEY = "robo_vlm_qc.timeline_snap";
const COMPACT_TRACKS_STORAGE_KEY = "robo_vlm_qc.timeline_compact";
const DEFAULT_LEFT_PCT = 61;
const MIN_LEFT_PCT = 50;
const MAX_LEFT_PCT = 75;

const QUICK_SHORTCUT_ITEMS: Array<{ keys: string[]; descKey: string }> = [
  { keys: ["Space"], descKey: "shortcuts.playPause" },
  { keys: ["←", "→"], descKey: "shortcuts.prevNextFrame" },
  { keys: ["J", "L"], descKey: "shortcuts.seekHalf" },
  { keys: ["[", "]"], descKey: "shortcuts.setEdge" },
  { keys: ["N"], descKey: "shortcuts.addSegment" },
  { keys: ["S"], descKey: "shortcuts.splitStage" },
  { keys: ["1", "2", "3"], descKey: "shortcuts.switchPolicy" },
  { keys: ["Ctrl+S"], descKey: "shortcuts.saveDraft" },
];

function formatTokenCount(tokens: number): string {
  if (!Number.isFinite(tokens) || tokens <= 0) return "0";
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(2)}M`;
  if (tokens >= 1_000) return `${(tokens / 1_000).toFixed(1)}k`;
  return String(Math.round(tokens));
}

function formatUsd(cost: number): string {
  if (!Number.isFinite(cost) || cost <= 0) return "$0.00";
  if (cost < 0.001) return `$${cost.toFixed(4)}`;
  if (cost < 1) return `$${cost.toFixed(3)}`;
  return `$${cost.toFixed(2)}`;
}

function loadBoolPref(key: string, defaultValue: boolean): boolean {
  if (typeof window === "undefined") return defaultValue;
  try {
    const raw = window.localStorage.getItem(key);
    if (raw === "false") return false;
    if (raw === "true") return true;
  } catch {
    // Ignore storage read errors
  }
  return defaultValue;
}

const EMPTY_SUCCESS: SuccessVerdict = {
  passed: null,
  verdict: "unsure",
  confidence: 0,
  criteria: [],
  control_check: null,
  summary: "",
};

function loadSplitPct(): number {
  if (typeof window === "undefined") return DEFAULT_LEFT_PCT;
  try {
    const raw = window.localStorage.getItem(SPLIT_STORAGE_KEY);
    if (raw) {
      const n = Number(raw);
      if (Number.isFinite(n)) return Math.max(MIN_LEFT_PCT, Math.min(MAX_LEFT_PCT, n));
    }
  } catch {
    // Ignore storage read errors
  }
  return DEFAULT_LEFT_PCT;
}

function loadAutoAdvance(): boolean {
  if (typeof window === "undefined") return true;
  try {
    const raw = window.localStorage.getItem(AUTO_ADVANCE_STORAGE_KEY);
    if (raw === "false") return false;
    if (raw === "true") return true;
  } catch {
    // Ignore storage read errors
  }
  return true;
}

export const ReviewStudioPage: React.FC = () => {
  const params = useParams<{ dataset?: string; stem?: string; "*"?: string }>();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  const episodeId = useMemo(() => {
    if (params.dataset && params.stem) return `${params.dataset}/${params.stem}`;
    return params["*"] ?? "";
  }, [params]);

  const {
    isLead,
    username,
    token,
    authProbing,
    canApprove,
    canFreezeGt,
    canViewDiff,
    setAuthPromptOpen,
  } = useAuth();
  const jobStore = useJobs();
  const toast = useToast();
  const { t } = useI18n();

  const [episode, setEpisode] = useState<ResolvedEpisode | null>(null);
  const [reviews, setReviews] = useState<HumanReviewArtifact[]>([]);
  const [availableRuns, setAvailableRuns] = useState<VlmRunSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string>("");
  const [labels, setLabels] = useState<VisualLabelDef[]>(DEFAULT_VISUAL_LABELS);
  const [minWatchCoverage, setMinWatchCoverage] = useState<number>(MIN_WATCH_COVERAGE);

  const [taskCatalog, setTaskCatalog] = useState<EpisodeTaskSummary[]>([]);
  const [taskPopoverOpen, setTaskPopoverOpen] = useState(false);
  const [pendingTaskId, setPendingTaskId] = useState("");
  const [updatingTask, setUpdatingTask] = useState(false);
  const [taskWizardOpen, setTaskWizardOpen] = useState(false);
  const [frozenTaskConfirmId, setFrozenTaskConfirmId] = useState<string | null>(null);
  const [taskRerunPrompt, setTaskRerunPrompt] = useState<{ taskId: string } | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [currentTime, setCurrentTime] = useState(0);
  const [selectedSegmentId, setSelectedSegmentId] = useState<string | null>(null);
  const [selectedStageId, setSelectedStageId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<InspectorTab>("segment");
  const [showDiffTrack, setShowDiffTrack] = useState<boolean>(false);

  const [cutSkipEnabled, setCutSkipEnabled] = useState<boolean>(false);
  const [loopSegmentEnabled, setLoopSegmentEnabled] = useState<boolean>(false);
  const [shortcutsOpen, setShortcutsOpen] = useState<boolean>(false);
  const [costPopoverOpen, setCostPopoverOpen] = useState<boolean>(false);
  const [displayPopoverOpen, setDisplayPopoverOpen] = useState<boolean>(false);
  const [snapEnabled, setSnapEnabled] = useState<boolean>(() =>
    loadBoolPref(SNAP_ENABLED_STORAGE_KEY, true),
  );
  const [compactTracks, setCompactTracks] = useState<boolean>(() =>
    loadBoolPref(COMPACT_TRACKS_STORAGE_KEY, false),
  );
  const [rerunMenuOpen, setRerunMenuOpen] = useState<boolean>(false);
  const [exportMenuOpen, setExportMenuOpen] = useState<boolean>(false);
  const [customObjectsModalOpen, setCustomObjectsModalOpen] = useState<boolean>(false);
  const [customObjectsInput, setCustomObjectsInput] = useState<string>("");
  const [inferringSubtasks, setInferringSubtasks] = useState<boolean>(false);
  const [rerunning, setRerunning] = useState<boolean>(false);
  const [cancellingRerun, setCancellingRerun] = useState<boolean>(false);
  const [localRerunJobId, setLocalRerunJobId] = useState<string | null>(null);
  const [rerunCompletedReady, setRerunCompletedReady] = useState<boolean>(false);
  const [rerunCompletedRunId, setRerunCompletedRunId] = useState<string | null>(null);

  const [leftPanePct, setLeftPanePct] = useState<number>(loadSplitPct);
  const [draggingSplit, setDraggingSplit] = useState<boolean>(false);
  const [autoAdvance, setAutoAdvance] = useState<boolean>(loadAutoAdvance);

  const [notes, setNotes] = useState("");
  const initialNotesRef = useRef<string>("");
  const [flagDialogOpen, setFlagDialogOpen] = useState(false);
  const [flaggedReason, setFlaggedReason] = useState("");
  const [saving, setSaving] = useState(false);

  const [claimedBy, setClaimedBy] = useState<string | null>(null);
  const [claimExpiresAt, setClaimExpiresAt] = useState<number | null>(null);

  const trackerRef = useRef<WatchCoverageTracker>(new WatchCoverageTracker(1));
  const trackedEpisodeIdRef = useRef<string | null>(null);
  const [watchCoverage, setWatchCoverage] = useState<WatchCoverage>(() =>
    trackerRef.current.snapshot(),
  );

  const playerCtrlRef = useRef<StudioPlayerHandle | null>(null);
  const splitGridRef = useRef<HTMLDivElement | null>(null);
  const appliedDeepLinkKeyRef = useRef<string>("");
  const rerunMenuWrapRef = useRef<HTMLDivElement | null>(null);
  const exportMenuWrapRef = useRef<HTMLDivElement | null>(null);
  const taskPopoverWrapRef = useRef<HTMLDivElement | null>(null);
  const costPopoverWrapRef = useRef<HTMLDivElement | null>(null);
  const displayPopoverWrapRef = useRef<HTMLDivElement | null>(null);

  const history = useReviewHistory({
    qc_segments: [],
    success: EMPTY_SUCCESS,
    macro_stages: [],
  });

  const searchKey = searchParams.toString();
  const deepLinkRunId = searchParams.get("run_id") ?? "";

  const { showToast } = toast;
  const notify = useCallback(
    (message: string, variant: "success" | "info" | "warning" | "error" = "info") => {
      showToast({
        message,
        variant,
        testId: "studio-info-banner",
      });
    },
    [showToast],
  );

  // Fetch workspace settings (min_watch_coverage)
  useEffect(() => {
    if (authProbing) return;
    let cancelled = false;
    void getSettings()
      .then((res) => {
        if (
          !cancelled &&
          res &&
          typeof res.min_watch_coverage === "number" &&
          Number.isFinite(res.min_watch_coverage) &&
          res.min_watch_coverage >= 0 &&
          res.min_watch_coverage <= 1
        ) {
          setMinWatchCoverage(res.min_watch_coverage);
        }
      })
      .catch(() => {
        // Keep default MIN_WATCH_COVERAGE
      });
    return () => {
      cancelled = true;
    };
  }, [token, authProbing]);

  const loadRunsCatalog = useCallback(async () => {
    if (!episodeId || !isLead || authProbing) {
      setAvailableRuns([]);
      return;
    }
    try {
      const res = await listEpisodeRuns(episodeId);
      if (res && Array.isArray(res.runs)) {
        setAvailableRuns(res.runs);
      }
    } catch {
      setAvailableRuns([]);
    }
  }, [episodeId, isLead, authProbing]);

  useEffect(() => {
    void loadRunsCatalog();
  }, [loadRunsCatalog, token]);

  const loadTaskCatalog = useCallback(async () => {
    if (authProbing) return;
    try {
      const res = await listEpisodeTasks();
      setTaskCatalog(res.tasks ?? []);
    } catch {
      // Optional task catalog
    }
  }, [authProbing]);

  useEffect(() => {
    void loadTaskCatalog();
  }, [loadTaskCatalog, token]);

  const loadSeqRef = useRef(0);
  const selectedRunIdRef = useRef(selectedRunId);
  selectedRunIdRef.current = selectedRunId;
  const minWatchCoverageRef = useRef(minWatchCoverage);
  minWatchCoverageRef.current = minWatchCoverage;

  const loadEpisodeData = useCallback(
    async (runIdOverride?: string) => {
      if (!episodeId) return;
      const seq = ++loadSeqRef.current;
      const targetRunId = runIdOverride !== undefined ? runIdOverride : selectedRunIdRef.current;
      setLoading(true);
      setError(null);
      try {
        const [ep, revs] = await Promise.all([
          getEpisode(episodeId, targetRunId || undefined),
          listReviews(episodeId).catch(() => [] as HumanReviewArtifact[]),
        ]);
        if (seq !== loadSeqRef.current) return;
        setEpisode(ep);
        setReviews(revs);
        const loadedNotes = ep.notes ?? "";
        setNotes(loadedNotes);
        initialNotesRef.current = loadedNotes;
        setFlaggedReason(ep.flagged_reason ?? "");
        setClaimedBy(ep.claimed_by ?? null);
        setClaimExpiresAt(ep.claim_expires_at ?? null);
        history.init({
          qc_segments: ep.qc_segments,
          success: ep.success,
          macro_stages: ep.macro_stages,
        });

        const deepKey = `${episodeId}|${searchKey}`;
        const shouldApplyDeepLink = appliedDeepLinkKeyRef.current !== deepKey;
        let nextSelectedSegId: string | null =
          ep.qc_segments.length > 0 ? ep.qc_segments[0].seg_id : null;
        let nextSelectedStageId: string | null = null;

        if (shouldApplyDeepLink) {
          appliedDeepLinkKeyRef.current = deepKey;
          const qs = new URLSearchParams(searchKey);
          const tParam = qs.get("t");
          const segParam = qs.get("seg");
          const tabParam = qs.get("tab");

          let targetTime: number | null = null;
          if (tParam !== null && tParam.trim() !== "") {
            const parsedT = Number(tParam);
            if (Number.isFinite(parsedT)) {
              targetTime = Math.max(0, Math.min(ep.meta.duration_s, parsedT));
            }
          }

          if (segParam) {
            const matchedSeg = ep.qc_segments.find((s) => s.seg_id === segParam);
            if (matchedSeg) {
              nextSelectedSegId = matchedSeg.seg_id;
              nextSelectedStageId = null;
              if (targetTime === null) targetTime = matchedSeg.t_start;
            } else {
              const matchedStage = ep.macro_stages.find((st) => st.stage_id === segParam);
              if (matchedStage) {
                nextSelectedSegId = null;
                nextSelectedStageId = matchedStage.stage_id;
                if (targetTime === null) targetTime = matchedStage.t_start;
              } else {
                const matchedRaw = ep.raw_vlm_segments.find((s) => s.seg_id === segParam);
                if (matchedRaw && targetTime === null) {
                  targetTime = matchedRaw.t_start;
                }
              }
            }
          }

          if (targetTime !== null) {
            setCurrentTime(targetTime);
          } else if (trackedEpisodeIdRef.current !== ep.episode_id) {
            setCurrentTime(0);
          }

          if (tabParam === "diff" && canViewDiff) {
            setActiveTab("diff");
            setShowDiffTrack(true);
          } else if (tabParam === "success") {
            setActiveTab("success");
          } else if (tabParam === "segment") {
            setActiveTab("segment");
          }
        }

        setSelectedStageId(nextSelectedStageId);
        setSelectedSegmentId(nextSelectedSegId);

        const covThreshold = minWatchCoverageRef.current;
        const preservedIntervals =
          trackedEpisodeIdRef.current === ep.episode_id
            ? trackerRef.current.snapshot(covThreshold).watched_intervals
            : [];
        trackedEpisodeIdRef.current = ep.episode_id;
        trackerRef.current = new WatchCoverageTracker(ep.meta.duration_s, preservedIntervals);
        setWatchCoverage(trackerRef.current.snapshot(covThreshold));
      } catch (err) {
        if (seq !== loadSeqRef.current) return;
        if (err instanceof ApiError && err.status === 401) {
          setAuthPromptOpen(true);
        }
        setError(err instanceof Error ? err.message : t("studio.errLoadEpisode"));
      } finally {
        if (seq === loadSeqRef.current) {
          setLoading(false);
        }
      }
      // eslint-disable-next-line react-hooks/exhaustive-deps
    },
    [episodeId, token, setAuthPromptOpen, searchKey, canViewDiff],
  );

  // Reset run selection when navigating to a new episode
  useEffect(() => {
    setSelectedRunId(isLead && deepLinkRunId ? deepLinkRunId : "");
    setLocalRerunJobId(null);
    setRerunning(false);
    setCancellingRerun(false);
    setRerunCompletedReady(false);
    setRerunCompletedRunId(null);
    setExportMenuOpen(false);
    setCustomObjectsModalOpen(false);
    setCustomObjectsInput("");
    setInferringSubtasks(false);
    setTaskPopoverOpen(false);
    setCostPopoverOpen(false);
    setDisplayPopoverOpen(false);
    setTaskWizardOpen(false);
    setFrozenTaskConfirmId(null);
    setTaskRerunPrompt(null);
  }, [episodeId, isLead, deepLinkRunId]);

  useEffect(() => {
    if (authProbing) return;
    void loadEpisodeData(isLead && deepLinkRunId ? deepLinkRunId : "");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [episodeId, token, searchKey, authProbing]);

  // If ?run_id=... is present and user elevates to Lead without changing token (e.g. --no-auth switch), load that run
  useEffect(() => {
    if (authProbing || !isLead || !deepLinkRunId) return;
    if (episode && episode.selected_run_id !== deepLinkRunId) {
      setSelectedRunId(deepLinkRunId);
      void loadEpisodeData(deepLinkRunId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isLead, deepLinkRunId, authProbing]);

  // Recompute watchCoverage snapshot when dynamic min_watch_coverage setting loads
  useEffect(() => {
    setWatchCoverage(trackerRef.current.snapshot(minWatchCoverage));
  }, [minWatchCoverage]);

  // If ?tab=diff is present and Lead role finishes probing slightly after load, activate diff tab + ghost track
  useEffect(() => {
    if (canViewDiff && searchParams.get("tab") === "diff" && episode) {
      setActiveTab("diff");
      setShowDiffTrack(true);
    }
  }, [canViewDiff, searchParams, episode]);

  const handledRerunJobIdsRef = useRef<Set<string>>(new Set());
  const prevJobStatusRef = useRef<Map<string, JobStatus["status"]>>(new Map());

  const handleRerunTerminalJob = useCallback(
    (tracked: JobStatus) => {
      if (handledRerunJobIdsRef.current.has(tracked.job_id)) return;
      if (tracked.status === "completed") {
        handledRerunJobIdsRef.current.add(tracked.job_id);
        const firstRes = tracked.summary?.results?.[0];
        const newRunId =
          firstRes && typeof firstRes.run_id === "string" ? (firstRes.run_id as string) : null;
        setRerunCompletedRunId(newRunId);
        setRerunCompletedReady(true);
        setRerunning(false);
        setLocalRerunJobId(null);
        void loadRunsCatalog();
      } else if (tracked.status === "failed" || tracked.status === "cancelled") {
        handledRerunJobIdsRef.current.add(tracked.job_id);
        setRerunning(false);
        setLocalRerunJobId(null);
        const errDetail = tracked.summary?.errors?.[0]?.error;
        notify(
          tracked.status === "cancelled"
            ? t("studio.toastRerunCancelled", { jobId: tracked.job_id })
            : errDetail
              ? t("studio.toastRerunFailedDetail", { jobId: tracked.job_id, detail: errDetail })
              : t("studio.toastRerunFailed", { jobId: tracked.job_id }),
          tracked.status === "cancelled" ? "warning" : "error",
        );
        void loadEpisodeData();
      }
    },
    [loadRunsCatalog, loadEpisodeData, notify, t],
  );

  // Watch jobStore for completion/cancellation/failure of our triggered single-episode local VLM rerun
  useEffect(() => {
    if (!localRerunJobId || handledRerunJobIdsRef.current.has(localRerunJobId)) return;
    const tracked = jobStore.jobs.find((j) => j.job_id === localRerunJobId);
    if (!tracked) return;
    handleRerunTerminalJob(tracked);
  }, [jobStore.jobs, localRerunJobId, handleRerunTerminalJob]);

  // Fallback polling for localRerunJobId when mounted standalone without JobProvider
  useEffect(() => {
    if (!localRerunJobId || !rerunning || jobStore.isProvided) return;
    let stopped = false;
    const timer = window.setInterval(() => {
      if (stopped) return;
      void getJob(localRerunJobId)
        .then((latest) => {
          if (stopped) return;
          if (
            latest.status === "completed" ||
            latest.status === "failed" ||
            latest.status === "cancelled"
          ) {
            handleRerunTerminalJob(latest);
          }
        })
        .catch(() => {
          // Ignore transient polling errors
        });
    }, 1500);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [localRerunJobId, rerunning, jobStore.isProvided, handleRerunTerminalJob]);

  // Also refresh episode status if an externally started batch_vlm job targeting this episode is cancelled/failed
  useEffect(() => {
    if (!jobStore.isProvided || !episodeId) return;
    let shouldReload = false;
    const prevMap = prevJobStatusRef.current;
    for (const job of jobStore.jobs) {
      const prev = prevMap.get(job.job_id);
      prevMap.set(job.job_id, job.status);
      if (job.job_id === localRerunJobId || handledRerunJobIdsRef.current.has(job.job_id)) {
        continue;
      }
      const wasActive = prev === "queued" || prev === "running";
      const isCancelledOrFailed = job.status === "cancelled" || job.status === "failed";
      const epIds = Array.isArray(job.params?.episode_ids)
        ? (job.params.episode_ids as unknown[])
        : [];
      if (
        wasActive &&
        isCancelledOrFailed &&
        job.job_type === "batch_vlm" &&
        (epIds.includes(episodeId) ||
          episode?.status === "vlm_queued" ||
          episode?.status === "vlm_running")
      ) {
        shouldReload = true;
      }
    }
    if (shouldReload) {
      void loadEpisodeData();
    }
  }, [
    jobStore.isProvided,
    jobStore.jobs,
    episodeId,
    localRerunJobId,
    episode?.status,
    loadEpisodeData,
  ]);

  // Close rerun split menu when clicking outside
  useEffect(() => {
    if (!rerunMenuOpen) return;
    const onDocClick = (e: MouseEvent) => {
      if (rerunMenuWrapRef.current && !rerunMenuWrapRef.current.contains(e.target as Node)) {
        setRerunMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [rerunMenuOpen]);

  // Close export popover when clicking outside
  useEffect(() => {
    if (!exportMenuOpen) return;
    const onDocClick = (e: MouseEvent) => {
      if (exportMenuWrapRef.current && !exportMenuWrapRef.current.contains(e.target as Node)) {
        setExportMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [exportMenuOpen]);

  // Close task popover when clicking outside
  useEffect(() => {
    if (!taskPopoverOpen) return;
    const onDocClick = (e: MouseEvent) => {
      if (taskPopoverWrapRef.current && !taskPopoverWrapRef.current.contains(e.target as Node)) {
        setTaskPopoverOpen(false);
      }
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [taskPopoverOpen]);

  // Close run cost popover when clicking outside
  useEffect(() => {
    if (!costPopoverOpen) return;
    const onDocClick = (e: MouseEvent) => {
      if (costPopoverWrapRef.current && !costPopoverWrapRef.current.contains(e.target as Node)) {
        setCostPopoverOpen(false);
      }
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [costPopoverOpen]);

  // Close display & shortcuts popover when clicking outside
  useEffect(() => {
    if (!displayPopoverOpen) return;
    const onDocClick = (e: MouseEvent) => {
      if (
        displayPopoverWrapRef.current &&
        !displayPopoverWrapRef.current.contains(e.target as Node)
      ) {
        setDisplayPopoverOpen(false);
      }
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [displayPopoverOpen]);

  // Load label catalog separately so role probe updates do not discard unsaved edits / undo history
  useEffect(() => {
    if (authProbing) return;
    let cancelled = false;
    const req = isLead ? getLabLabels() : getSharedLabels();
    void req
      .then((reg) => {
        if (!cancelled && reg.labels?.length > 0) {
          setLabels(reg.labels.filter((l) => l.status !== "retired"));
        }
      })
      .catch(() => {
        if (!cancelled) {
          setLabels(DEFAULT_VISUAL_LABELS);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [isLead, token, authProbing]);

  // Lease heartbeat renewal every 60s when claimed by current user
  const isClaimedByMe = claimedBy === username;
  const isClaimedByOther = Boolean(claimedBy && claimedBy !== username);
  const isGtFrozen = episode?.status === "gt_frozen";
  const isHistoricalView = Boolean(selectedRunId) || Boolean(episode?.is_historical_run_view);
  const isReadOnly = isGtFrozen || isClaimedByOther || isHistoricalView;
  const hasUnsavedChanges = !isReadOnly && (history.dirty || notes !== initialNotesRef.current);

  useEffect(() => {
    if (!episodeId || !isClaimedByMe || isGtFrozen) return;
    const timer = window.setInterval(() => {
      void renewClaim(episodeId, username)
        .then((res) => {
          setClaimExpiresAt(res.claim_expires_at);
        })
        .catch((err) => {
          if (err instanceof ApiError && err.status === 409) {
            setError(t("studio.errClaimExpired"));
            setClaimedBy(null);
            setClaimExpiresAt(null);
          }
        });
    }, 60_000);
    return () => window.clearInterval(timer);
  }, [episodeId, isClaimedByMe, isGtFrozen, username, t]);

  // Unsaved-changes guard: browser beforeunload
  useEffect(() => {
    if (!hasUnsavedChanges) return;
    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, [hasUnsavedChanges]);

  // Unsaved-changes guard: in-app link navigation confirmation (compatible with BrowserRouter & MemoryRouter)
  useEffect(() => {
    if (!hasUnsavedChanges) return;
    const onDocumentClick = (e: MouseEvent) => {
      if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) {
        return;
      }
      const anchor = (e.target as HTMLElement | null)?.closest("a[href]") as HTMLAnchorElement | null;
      if (!anchor || anchor.target === "_blank" || anchor.hasAttribute("download")) return;
      const href = anchor.getAttribute("href");
      if (!href || href.startsWith("#")) return;
      if (!window.confirm(t("studio.confirmLeaveUnsaved"))) {
        e.preventDefault();
        e.stopPropagation();
      }
    };
    document.addEventListener("click", onDocumentClick, true);
    return () => document.removeEventListener("click", onDocumentClick, true);
  }, [hasUnsavedChanges, t]);

  const handleWatchInterval = useCallback(
    (t0: number, t1: number) => {
      const snap = trackerRef.current.recordInterval(t0, t1, minWatchCoverage);
      setWatchCoverage(snap);
    },
    [minWatchCoverage],
  );

  const selectedSegment = useMemo(
    () => history.state.qc_segments.find((s) => s.seg_id === selectedSegmentId) ?? null,
    [history.state.qc_segments, selectedSegmentId],
  );

  const selectedStage = useMemo(
    () => history.state.macro_stages.find((s) => s.stage_id === selectedStageId) ?? null,
    [history.state.macro_stages, selectedStageId],
  );

  const primaryCamera = useMemo(() => {
    const cams = episode?.meta.cameras ?? [];
    const mainCam = cams.find((c) => c.role === "main" || c.role === "front");
    return mainCam?.name ?? cams[0]?.name ?? "main";
  }, [episode]);

  const handleAddSegmentAt = useCallback(
    (tStart: number, tEnd?: number) => {
      if (!episode || isReadOnly) return;
      const base = createDefaultSegment(tStart, episode.meta.duration_s, labels[2] ?? labels[0]);
      const seg =
        tEnd !== undefined && tEnd > tStart
          ? normalizeSegmentTriplet(
              base,
              {
                t_start: Math.max(0, Number(tStart.toFixed(2))),
                t_end: Math.min(episode.meta.duration_s, Number(tEnd.toFixed(2))),
              },
              labels,
            )
          : base;
      history.applyEdit((prev) => ({
        ...prev,
        qc_segments: [...prev.qc_segments, seg],
      }));
      setSelectedSegmentId(seg.seg_id);
      setSelectedStageId(null);
      setActiveTab("segment");
    },
    [episode, history, isReadOnly, labels],
  );

  const handleSplitMacroStage = useCallback(() => {
    if (!episode || isReadOnly) return;
    const splitT = Number(currentTime.toFixed(2));
    const idx = history.state.macro_stages.findIndex(
      (st) => splitT >= st.t_start + 0.1 && splitT <= st.t_end - 0.1,
    );
    if (idx < 0) {
      notify(t("studio.toastSplitNeedPlayhead"), "warning");
      return;
    }
    const target = history.state.macro_stages[idx];
    const nextId = `M${history.state.macro_stages.length + 1}_${Date.now().toString(36).slice(-3)}`;
    const firstHalf: MacroStage = { ...target, t_end: splitT };
    const secondHalf: MacroStage = {
      ...target,
      stage_id: nextId,
      t_start: splitT,
      t_end: target.t_end,
      label: `${target.label} (2)`,
    };
    history.applyEdit((prev) => {
      const nextStages = [...prev.macro_stages];
      nextStages.splice(idx, 1, firstHalf, secondHalf);
      return {
        ...prev,
        macro_stages: nextStages,
      };
    });
    setSelectedStageId(secondHalf.stage_id);
    setSelectedSegmentId(null);
    setActiveTab("segment");
    notify(t("studio.toastSplitDone", { t: splitT.toFixed(2) }), "info");
  }, [currentTime, episode, history, isReadOnly, notify, t]);

  const handleUpdateStageSpans = useCallback(
    (updates: StageSpanUpdate[]) => {
      if (isReadOnly || updates.length === 0) return;
      const map = new Map(updates.map((u) => [u.stageId, u]));
      history.applyEdit((prev) => ({
        ...prev,
        macro_stages: prev.macro_stages.map((st) => {
          const upd = map.get(st.stage_id);
          return upd ? { ...st, t_start: upd.tStart, t_end: upd.tEnd } : st;
        }),
      }));
    },
    [history, isReadOnly],
  );

  const handleAlignStagesToQc = useCallback(() => {
    if (!episode || isReadOnly) return;
    const cutBoundaries = history.state.qc_segments
      .filter((seg) => seg.splice_policy === "cut_clean")
      .flatMap((seg) => [seg.t_start, seg.t_end]);
    if (cutBoundaries.length === 0) {
      notify(t("studio.toastNoCutClean"), "info");
      return;
    }

    const ALIGN_THRESHOLD_S = 0.25;
    const findNearestCut = (tVal: number): number | null => {
      let best: number | null = null;
      let bestDist = ALIGN_THRESHOLD_S + 1e-9;
      for (const b of cutBoundaries) {
        const d = Math.abs(b - tVal);
        if (d <= bestDist) {
          bestDist = d;
          best = b;
        }
      }
      return best;
    };

    let changedEdges = 0;
    const maxDur = episode.meta.duration_s;
    const nextStages = history.state.macro_stages.map((st) => {
      const snapStart = findNearestCut(st.t_start);
      const snapEnd = findNearestCut(st.t_end);
      let nextStart =
        snapStart !== null ? Math.max(0, Math.min(maxDur, Number(snapStart.toFixed(2)))) : st.t_start;
      let nextEnd =
        snapEnd !== null ? Math.max(0, Math.min(maxDur, Number(snapEnd.toFixed(2)))) : st.t_end;

      if (nextEnd - nextStart < 0.1) {
        // Keep stage valid (minimum 0.1s span); prefer keeping whichever edge moved less
        const dStart = snapStart !== null ? Math.abs(nextStart - st.t_start) : Infinity;
        const dEnd = snapEnd !== null ? Math.abs(nextEnd - st.t_end) : Infinity;
        if (dStart <= dEnd) {
          nextEnd = st.t_end;
          if (nextEnd - nextStart < 0.1) nextStart = st.t_start;
        } else {
          nextStart = st.t_start;
          if (nextEnd - nextStart < 0.1) nextEnd = st.t_end;
        }
      }

      if (Math.abs(nextStart - st.t_start) > 1e-4) changedEdges += 1;
      if (Math.abs(nextEnd - st.t_end) > 1e-4) changedEdges += 1;

      return Math.abs(nextStart - st.t_start) > 1e-4 || Math.abs(nextEnd - st.t_end) > 1e-4
        ? { ...st, t_start: nextStart, t_end: nextEnd }
        : st;
    });

    if (changedEdges === 0) {
      notify(t("studio.toastAlreadyAligned"), "info");
      return;
    }

    history.applyEdit((prev) => ({
      ...prev,
      macro_stages: nextStages,
    }));
    notify(t("studio.toastAlignedEdges", { count: changedEdges }), "success");
  }, [episode, history, isReadOnly, notify, t]);

  const handleInferSubtasks = useCallback(async () => {
    if (!episode || isReadOnly || inferringSubtasks) return;
    setInferringSubtasks(true);
    setError(null);
    try {
      const res = await inferSubtasks(episodeId, {
        qc_segments: history.state.qc_segments,
        macro_stages: history.state.macro_stages,
      });
      if (Array.isArray(res.macro_stages) && res.macro_stages.length > 0) {
        history.applyEdit((prev) => ({
          ...prev,
          macro_stages: res.macro_stages,
        }));
        setSelectedStageId((prev) =>
          prev && res.macro_stages.some((st) => st.stage_id === prev)
            ? prev
            : res.macro_stages[0].stage_id,
        );
        setSelectedSegmentId(null);
        setActiveTab("segment");
        notify(
          t("studio.toastInferredSubtasks", {
            stages: res.macro_stages.length,
            segments: res.used_segments_count,
            grounding: res.used_grounding ? t("studio.toastInferredWithGrounding") : "",
          }),
          "success",
        );
      } else {
        notify(t("studio.toastInferEmpty"), "warning");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : t("studio.errInferSubtasks"));
    } finally {
      setInferringSubtasks(false);
    }
  }, [episode, episodeId, history, inferringSubtasks, isReadOnly, notify, t]);

  const handleDownloadEpisodeJson = useCallback(() => {
    if (!episode) return;
    setExportMenuOpen(false);
    const payload = {
      episode_id: episode.episode_id,
      meta: episode.meta,
      qc_segments: history.state.qc_segments,
      macro_stages: history.state.macro_stages,
      success: history.state.success,
      notes,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const safeName = episode.episode_id.replace(/[^a-zA-Z0-9._-]+/g, "_");
    const link = document.createElement("a");
    link.href = url;
    link.download = `${safeName}_annotations.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    notify(
      t("studio.toastDownloadedJson", { filename: `${safeName}_annotations.json` }),
      "success",
    );
  }, [episode, history.state, notes, notify, t]);

  const handleOpenExportStudio = useCallback(() => {
    if (!episode) return;
    setExportMenuOpen(false);
    if (!isLead) {
      setAuthPromptOpen(true);
      notify(t("studio.toastExportNeedsLead"), "warning");
      return;
    }
    const targetUrl = `/export?episode_ids=${encodeURIComponent(episode.episode_id)}&dataset=${encodeURIComponent(episode.meta.dataset)}&include_status=${encodeURIComponent(episode.status)}`;
    navigate(targetUrl);
  }, [episode, isLead, navigate, notify, setAuthPromptOpen, t]);

  const handleClaim = async () => {
    try {
      const res = await claimEpisode(episodeId, username);
      setClaimedBy(res.claimed_by);
      setClaimExpiresAt(res.claim_expires_at);
      setEpisode((prev) =>
        prev
          ? {
              ...prev,
              status: res.status ?? "claimed",
              claimed_by: res.claimed_by,
              claim_expires_at: res.claim_expires_at,
            }
          : prev,
      );
      notify(t("studio.toastClaimed", { user: res.claimed_by }), "success");
    } catch (err) {
      setError(err instanceof Error ? err.message : t("studio.errClaimFailed"));
    }
  };

  const handleRelease = async () => {
    try {
      const res = await releaseClaim(episodeId, username);
      setClaimedBy(null);
      setClaimExpiresAt(null);
      setEpisode((prev) => (prev ? { ...prev, status: res.status } : prev));
      notify(t("studio.toastReleased"), "info");
    } catch (err) {
      setError(err instanceof Error ? err.message : t("studio.errReleaseFailed"));
    }
  };

  const tryAutoAdvanceNext = useCallback(async (): Promise<boolean> => {
    try {
      if (!isLead) {
        const res = await listEpisodes({ status: "vlm_ready", limit: 20 });
        const candidates = (res.items ?? []).filter(
          (item) => item.episode_id !== episodeId && item.status === "vlm_ready",
        );
        for (const cand of candidates) {
          try {
            await claimEpisode(cand.episode_id, username);
            notify(t("studio.toastAutoClaimed", { id: cand.episode_id }), "success");
            navigate(`/review/${encodeEpPath(cand.episode_id)}`);
            return true;
          } catch {
            // Try next candidate if another reviewer claimed simultaneously
          }
        }
        return false;
      }

      // Lead: look for next reviewed/flagged episode first, then vlm_ready
      const reviewedRes = await listEpisodes({ status: "reviewed,flagged", limit: 20 });
      const nextLead =
        (reviewedRes.items ?? []).find((item) => item.episode_id !== episodeId) ??
        (
          (await listEpisodes({ status: "vlm_ready", limit: 20 })).items ?? []
        ).find((item) => item.episode_id !== episodeId);

      if (nextLead) {
        notify(t("studio.toastAutoAdvanced", { id: nextLead.episode_id }), "success");
        navigate(`/review/${encodeEpPath(nextLead.episode_id)}`);
        return true;
      }
      return false;
    } catch {
      return false;
    }
  }, [episodeId, isLead, navigate, notify, username, t]);

  const handleSubmitAction = useCallback(
    async (action: ReviewAction) => {
      if (!episode || isReadOnly) return;
      if (!episode.active_run_id) {
        setError(t("studio.errNoVlmRun"));
        return;
      }
      if (!isLead && action === "submit_review" && !canReviewerSubmit(watchCoverage, minWatchCoverage)) {
        setError(
          t("studio.errWatchCoverage", {
            pct: (watchCoverage.coverage_ratio * 100).toFixed(1),
            minPct: Math.round(minWatchCoverage * 100),
          }),
        );
        return;
      }
      if (action === "flag" && !flaggedReason.trim()) {
        setFlagDialogOpen(true);
        return;
      }

      setSaving(true);
      setError(null);
      try {
        const sanitizedStages = history.state.macro_stages.map((st, idx) => ({
          ...st,
          label: st.label.trim() || t("studio.stageFallbackLabel", { index: idx + 1 }),
        }));
        const payload = buildReviewWritePayload({
          baseRunId: episode.active_run_id,
          action,
          qcSegments: history.state.qc_segments,
          success: history.state.success,
          macroStages: sanitizedStages,
          watchCoverage,
          notes,
          flaggedReason: action === "flag" ? flaggedReason.trim() : null,
        });
        const saved = await submitReview(episodeId, payload);
        history.markSaved();
        initialNotesRef.current = notes;
        setReviews((prev) => [saved, ...prev]);
        setFlagDialogOpen(false);
        notify(t("studio.toastActionSubmitted", { action, revId: saved.rev_id }), "success");
        if (action === "save_draft") {
          setEpisode((prev) =>
            prev
              ? {
                  ...prev,
                  active_rev_id: saved.rev_id,
                  is_human_reviewed: true,
                  review_diff: saved.diff_summary,
                  notes: saved.notes,
                }
              : prev,
          );
        } else {
          const shouldAdvance =
            autoAdvance &&
            ((!isLead && (action === "submit_review" || action === "flag")) ||
              (isLead &&
                (action === "approve" ||
                  action === "freeze_gt" ||
                  action === "submit_review" ||
                  action === "flag")));
          const advanced = shouldAdvance ? await tryAutoAdvanceNext() : false;
          if (!advanced) {
            await loadEpisodeData();
          }
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : t("studio.errSubmitFailed"));
      } finally {
        setSaving(false);
      }
    },
    [
      episode,
      isReadOnly,
      isLead,
      watchCoverage,
      minWatchCoverage,
      flaggedReason,
      history,
      notes,
      episodeId,
      notify,
      autoAdvance,
      tryAutoAdvanceNext,
      loadEpisodeData,
      t,
    ],
  );

  const handleRenderSingleMp4 = async () => {
    try {
      const job = await createJob({ job_type: "render_mp4", episode_ids: [episodeId] });
      jobStore.trackJob(job);
      notify(t("studio.toastRenderMp4", { jobId: job.job_id }), "info");
    } catch (err) {
      setError(err instanceof Error ? err.message : t("studio.errRenderMp4"));
    }
  };

  const handleLocalRerunVlm = async (
    mode: "s3_s4" | "s1_grounding" | "s1_s5",
    customObjects?: string[],
  ) => {
    if (isGtFrozen) return;
    setRerunMenuOpen(false);
    setRerunning(true);
    setRerunCompletedReady(false);
    setRerunCompletedRunId(null);
    setError(null);
    try {
      const cleanedCustomObjects = customObjects?.map((s) => s.trim()).filter(Boolean);
      const jobPayload = {
        job_type: "batch_vlm" as const,
        episode_ids: [episodeId],
        stage: mode,
        fresh: mode === "s1_s5",
        ...(cleanedCustomObjects && cleanedCustomObjects.length > 0
          ? { custom_objects: cleanedCustomObjects }
          : {}),
      };
      const job = await createJob(jobPayload);
      setLocalRerunJobId(job.job_id);
      jobStore.trackJob(job);
      const label =
        mode === "s3_s4"
          ? t("studio.rerunModeS3S4")
          : mode === "s1_grounding"
            ? cleanedCustomObjects && cleanedCustomObjects.length > 0
              ? t("studio.rerunModeCustomS1", { objects: cleanedCustomObjects.join(", ") })
              : t("studio.rerunModeS1")
            : t("studio.rerunModeFull");
      notify(t("studio.toastRerunSubmitted", { label, jobId: job.job_id }), "info");
      if (job.status === "completed" || job.status === "failed" || job.status === "cancelled") {
        handleRerunTerminalJob(job);
      }
    } catch (err) {
      setRerunning(false);
      setError(err instanceof Error ? err.message : t("studio.errRerunVlm"));
    }
  };

  const parsedCustomObjects = useMemo(
    () =>
      customObjectsInput
        .split(/[,，\n]+/)
        .map((s) => s.trim())
        .filter(Boolean),
    [customObjectsInput],
  );

  const handleSubmitCustomObjectsS1 = async (e: React.FormEvent) => {
    e.preventDefault();
    if (parsedCustomObjects.length === 0 || rerunning || isGtFrozen) return;
    setCustomObjectsModalOpen(false);
    await handleLocalRerunVlm("s1_grounding", parsedCustomObjects);
  };

  const handleCancelLocalRerun = async () => {
    if (!localRerunJobId || cancellingRerun) return;
    const jobId = localRerunJobId;
    setCancellingRerun(true);
    try {
      const updated = jobStore.isProvided
        ? await jobStore.cancelJob(jobId)
        : await apiCancelJob(jobId).catch(() => null);
      if (updated) {
        handledRerunJobIdsRef.current.add(jobId);
        prevJobStatusRef.current.set(updated.job_id, updated.status);
        setRerunning(false);
        setLocalRerunJobId(null);
        // Avoid duplicate cancel toast when JobProvider already emits notifyJobCancelled
        if (!jobStore.isProvided) {
          notify(t("studio.toastRerunCancelled", { jobId }), "warning");
        }
        await loadEpisodeData();
      }
    } finally {
      setCancellingRerun(false);
    }
  };

  const handleSwitchRunVersion = async (nextRunId: string) => {
    if (nextRunId === selectedRunId) return;
    if (nextRunId && hasUnsavedChanges && !window.confirm(t("studio.confirmHistoricalSwitch"))) {
      return;
    }
    setSelectedRunId(nextRunId);
    await loadEpisodeData(nextRunId);
    if (nextRunId) {
      notify(t("studio.toastSwitchedRun", { runId: nextRunId }), "info");
    } else {
      notify(t("studio.toastReturnedActive"), "info");
    }
  };

  const studioTaskOptions = useMemo(() => {
    const list: EpisodeTaskSummary[] = [];
    const seen = new Set<string>();
    for (const item of taskCatalog) {
      if (!item.task_id || seen.has(item.task_id)) continue;
      if (!isLead && item.task_id === "_default") continue;
      seen.add(item.task_id);
      list.push(item);
    }
    if (isLead && episode?.meta.task_id && !seen.has(episode.meta.task_id)) {
      list.push({
        task_id: episode.meta.task_id,
        title: "",
        description: "",
        goal: "",
        match: [],
        aliases: [],
        grounding_objects_count: 0,
      });
    }
    return list;
  }, [taskCatalog, isLead, episode?.meta.task_id]);

  const handleOpenTaskPopover = () => {
    if (!episode) return;
    if (taskPopoverOpen) {
      setTaskPopoverOpen(false);
      return;
    }
    if (isLead) {
      setPendingTaskId(episode.meta.task_id);
    } else {
      setPendingTaskId(studioTaskOptions[0]?.task_id ?? "");
    }
    setTaskPopoverOpen(true);
  };

  const handleApplyEpisodeTask = async (nextTaskId: string, allowFrozen = false) => {
    if (!episode) return;
    const trimmed = nextTaskId.trim();
    if (!trimmed || trimmed === episode.meta.task_id) return;
    if (!isLead && (episode.meta.task_id !== "_default" || isGtFrozen || trimmed === "_default")) {
      return;
    }
    if (isLead && isGtFrozen && !allowFrozen) {
      setTaskPopoverOpen(false);
      setFrozenTaskConfirmId(trimmed);
      return;
    }

    setUpdatingTask(true);
    setError(null);
    try {
      const res = await assignEpisodeTask(episode.episode_id, {
        task_id: trimmed,
        ...(isLead && allowFrozen ? { allow_frozen: true } : {}),
      });
      setEpisode((prev) =>
        prev
          ? {
              ...prev,
              meta: {
                ...prev.meta,
                task_id: res.task_id,
              },
            }
          : prev,
      );
      setTaskPopoverOpen(false);
      setFrozenTaskConfirmId(null);
      notify(
        t("studio.taskUpdatedToast", {
          prev: res.previous_task_id,
          next: res.task_id,
        }),
        "success",
      );
      if (res.re_run_recommended) {
        setTaskRerunPrompt({ taskId: res.task_id });
      }
    } catch (err) {
      if (
        isLead &&
        err instanceof ApiError &&
        err.status === 409 &&
        !allowFrozen &&
        /gt_frozen|allow_frozen/i.test(err.detail)
      ) {
        setTaskPopoverOpen(false);
        setFrozenTaskConfirmId(trimmed);
        return;
      }
      setError(err instanceof Error ? err.message : t("studio.taskErrAssign"));
    } finally {
      setUpdatingTask(false);
    }
  };

  const handleStudioWizardTaskSaved = async (savedSpec: TaskSpec) => {
    const summary: EpisodeTaskSummary = {
      task_id: savedSpec.task_id,
      title: savedSpec.title,
      description: savedSpec.description,
      goal: savedSpec.goal,
      match: savedSpec.match ?? [],
      aliases: savedSpec.aliases ?? [],
      grounding_objects_count: (savedSpec.grounding_objects ?? []).length,
    };
    setTaskCatalog((prev) => [
      summary,
      ...prev.filter((item) => item.task_id !== summary.task_id),
    ]);
    void loadTaskCatalog();
    await handleApplyEpisodeTask(savedSpec.task_id, false);
  };

  const handleRerunVlmAfterTaskChange = async () => {
    if (!episode || !isLead || rerunning) return;
    setTaskRerunPrompt(null);
    setRerunning(true);
    setRerunCompletedReady(false);
    setRerunCompletedRunId(null);
    setError(null);
    try {
      const job = await createJob({
        job_type: "batch_vlm",
        episode_ids: [episode.episode_id],
        fresh: true,
      });
      setLocalRerunJobId(job.job_id);
      jobStore.trackJob(job);
      notify(
        t("studio.toastRerunSubmitted", {
          label: t("studio.rerunModeFull"),
          jobId: job.job_id,
        }),
        "info",
      );
      if (job.status === "completed" || job.status === "failed" || job.status === "cancelled") {
        handleRerunTerminalJob(job);
      }
    } catch (err) {
      setRerunning(false);
      setError(err instanceof Error ? err.message : t("studio.errRerunVlm"));
    }
  };

  // Draggable 6px vertical split resizer (50%–75%, persisted in localStorage)
  const startSplitDrag = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0 || !splitGridRef.current) return;
    e.preventDefault();
    setDraggingSplit(true);
    let latestPct = leftPanePct;

    const onMove = (ev: PointerEvent) => {
      const grid = splitGridRef.current;
      if (!grid) return;
      const rect = grid.getBoundingClientRect();
      if (rect.width <= 0) return;
      const rawPct = ((ev.clientX - rect.left) / rect.width) * 100;
      latestPct = Math.round(Math.max(MIN_LEFT_PCT, Math.min(MAX_LEFT_PCT, rawPct)) * 10) / 10;
      setLeftPanePct(latestPct);
    };

    const onUp = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
      setDraggingSplit(false);
      try {
        window.localStorage.setItem(SPLIT_STORAGE_KEY, String(latestPct));
      } catch {
        // Ignore storage errors
      }
    };

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
  };

  const handleSplitKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const delta = e.key === "ArrowLeft" ? -2 : 2;
    const next = Math.max(MIN_LEFT_PCT, Math.min(MAX_LEFT_PCT, leftPanePct + delta));
    setLeftPanePct(next);
    try {
      window.localStorage.setItem(SPLIT_STORAGE_KEY, String(next));
    } catch {
      // Ignore storage errors
    }
  };

  // Unified Review Studio keyboard shortcut suite (ignored in inputs/textareas/selects and behind overlays)
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (taskWizardOpen) {
          return;
        }
        if (frozenTaskConfirmId) {
          e.preventDefault();
          setFrozenTaskConfirmId(null);
          return;
        }
        if (taskPopoverOpen) {
          e.preventDefault();
          setTaskPopoverOpen(false);
          return;
        }
        if (customObjectsModalOpen) {
          e.preventDefault();
          setCustomObjectsModalOpen(false);
          return;
        }
        if (shortcutsOpen) {
          e.preventDefault();
          setShortcutsOpen(false);
          return;
        }
        if (flagDialogOpen) {
          e.preventDefault();
          setFlagDialogOpen(false);
          return;
        }
        if (rerunMenuOpen) {
          e.preventDefault();
          setRerunMenuOpen(false);
          return;
        }
        if (exportMenuOpen) {
          e.preventDefault();
          setExportMenuOpen(false);
          return;
        }
        if (costPopoverOpen) {
          e.preventDefault();
          setCostPopoverOpen(false);
          return;
        }
        if (displayPopoverOpen) {
          e.preventDefault();
          setDisplayPopoverOpen(false);
          return;
        }
        return;
      }

      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable)
      ) {
        return;
      }

      const key = e.key;
      const lower = key.toLowerCase();
      const mod = e.ctrlKey || e.metaKey;

      // Block global Studio shortcuts while any modal, drawer, or flag dialog is open
      const hasOpenOverlay =
        taskWizardOpen ||
        Boolean(frozenTaskConfirmId) ||
        customObjectsModalOpen ||
        shortcutsOpen ||
        flagDialogOpen ||
        Boolean(document.querySelector(".modal-backdrop, .drawer-backdrop"));
      if (hasOpenOverlay) {
        if (
          shortcutsOpen &&
          !flagDialogOpen &&
          (key === "?" || (key === "/" && e.shiftKey)) &&
          !mod &&
          !e.altKey
        ) {
          e.preventDefault();
          setShortcutsOpen(false);
        }
        return;
      }

      // ? toggles ShortcutsModal
      if ((key === "?" || (key === "/" && e.shiftKey)) && !mod && !e.altKey) {
        e.preventDefault();
        setShortcutsOpen((v) => !v);
        return;
      }

      if (!episode) return;

      // Save draft (Ctrl+S / Cmd+S)
      if (mod && lower === "s") {
        e.preventDefault();
        if (!isReadOnly && !saving) {
          void handleSubmitAction("save_draft");
        }
        return;
      }

      // Undo / Redo (Ctrl+Z / Ctrl+Shift+Z / Ctrl+Y)
      if (mod && lower === "z" && !e.shiftKey) {
        e.preventDefault();
        if (!isReadOnly) history.undo();
        return;
      }
      if (mod && ((lower === "z" && e.shiftKey) || lower === "y")) {
        e.preventDefault();
        if (!isReadOnly) history.redo();
        return;
      }

      if (mod || e.altKey) return;

      const fps = Math.max(1, episode.meta.fps);

      // Playback & navigation shortcuts
      if (e.code === "Space") {
        e.preventDefault();
        playerCtrlRef.current?.togglePlay();
        return;
      }
      if (key === "ArrowLeft") {
        e.preventDefault();
        playerCtrlRef.current?.stepTime(-1 / fps);
        return;
      }
      if (key === "ArrowRight") {
        e.preventDefault();
        playerCtrlRef.current?.stepTime(1 / fps);
        return;
      }
      if (lower === "j") {
        e.preventDefault();
        playerCtrlRef.current?.stepTime(e.shiftKey ? -2.0 : -0.5);
        return;
      }
      if (lower === "l") {
        e.preventDefault();
        playerCtrlRef.current?.stepTime(e.shiftKey ? 2.0 : 0.5);
        return;
      }
      if (lower === "k") {
        e.preventDefault();
        setCutSkipEnabled((v) => !v);
        return;
      }
      if (lower === "r") {
        e.preventDefault();
        if (selectedSegment) {
          setLoopSegmentEnabled((prev) => {
            const next = !prev;
            if (next) {
              const lo = Math.max(0, Number((selectedSegment.t_start - 0.5).toFixed(3)));
              const hi = Math.min(
                episode.meta.duration_s,
                Number((selectedSegment.t_end + 0.5).toFixed(3)),
              );
              if (currentTime < lo || currentTime > hi) {
                playerCtrlRef.current?.seekTo(lo);
              }
            }
            return next;
          });
        } else if (loopSegmentEnabled) {
          setLoopSegmentEnabled(false);
        }
        return;
      }

      // Editing shortcuts (blocked in read-only mode)
      if (isReadOnly) return;

      if (key === "[") {
        e.preventDefault();
        const cur = Number(currentTime.toFixed(2));
        if (selectedSegment) {
          const nextStart = Math.max(0, Math.min(selectedSegment.t_end - 0.05, cur));
          history.applyEdit((prev) => ({
            ...prev,
            qc_segments: prev.qc_segments.map((s) =>
              s.seg_id === selectedSegment.seg_id
                ? normalizeSegmentTriplet(s, { t_start: nextStart }, labels)
                : s,
            ),
          }));
        } else if (selectedStage) {
          const nextStart = Math.max(0, Math.min(selectedStage.t_end - 0.05, cur));
          history.applyEdit((prev) => ({
            ...prev,
            macro_stages: prev.macro_stages.map((st) =>
              st.stage_id === selectedStage.stage_id ? { ...st, t_start: nextStart } : st,
            ),
          }));
        }
        return;
      }

      if (key === "]") {
        e.preventDefault();
        const cur = Number(currentTime.toFixed(2));
        if (selectedSegment) {
          const nextEnd = Math.min(
            episode.meta.duration_s,
            Math.max(selectedSegment.t_start + 0.05, cur),
          );
          history.applyEdit((prev) => ({
            ...prev,
            qc_segments: prev.qc_segments.map((s) =>
              s.seg_id === selectedSegment.seg_id
                ? normalizeSegmentTriplet(s, { t_end: nextEnd }, labels)
                : s,
            ),
          }));
        } else if (selectedStage) {
          const nextEnd = Math.min(
            episode.meta.duration_s,
            Math.max(selectedStage.t_start + 0.05, cur),
          );
          history.applyEdit((prev) => ({
            ...prev,
            macro_stages: prev.macro_stages.map((st) =>
              st.stage_id === selectedStage.stage_id ? { ...st, t_end: nextEnd } : st,
            ),
          }));
        }
        return;
      }

      if (lower === "n") {
        e.preventDefault();
        handleAddSegmentAt(currentTime);
        return;
      }

      if (lower === "s") {
        e.preventDefault();
        handleSplitMacroStage();
        return;
      }

      if ((key === "1" || key === "2" || key === "3") && selectedSegment) {
        e.preventDefault();
        const policyMap: Record<string, SplicePolicy> = {
          "1": "keep_positive",
          "2": "mask_loss",
          "3": "cut_clean",
        };
        const nextPolicy = policyMap[key];
        history.applyEdit((prev) => ({
          ...prev,
          qc_segments: prev.qc_segments.map((s) =>
            s.seg_id === selectedSegment.seg_id
              ? normalizeSegmentTriplet(s, { splice_policy: nextPolicy }, labels)
              : s,
          ),
        }));
        return;
      }

      if (key === "Delete" || key === "Backspace") {
        e.preventDefault();
        if (selectedSegment) {
          const delId = selectedSegment.seg_id;
          history.applyEdit((prev) => ({
            ...prev,
            qc_segments: prev.qc_segments.filter((s) => s.seg_id !== delId),
          }));
          setSelectedSegmentId(null);
        } else if (selectedStage) {
          const delId = selectedStage.stage_id;
          history.applyEdit((prev) => ({
            ...prev,
            macro_stages: prev.macro_stages.filter((st) => st.stage_id !== delId),
          }));
          setSelectedStageId(null);
        }
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [
    taskWizardOpen,
    frozenTaskConfirmId,
    taskPopoverOpen,
    customObjectsModalOpen,
    shortcutsOpen,
    flagDialogOpen,
    rerunMenuOpen,
    exportMenuOpen,
    costPopoverOpen,
    displayPopoverOpen,
    episode,
    isReadOnly,
    saving,
    handleSubmitAction,
    history,
    selectedSegment,
    selectedStage,
    currentTime,
    loopSegmentEnabled,
    labels,
    handleAddSegmentAt,
    handleSplitMacroStage,
  ]);

  if (loading) {
    return (
      <div className="page-container studio-loading-container" data-testid="review-studio-loading">
        <div className="muted small">{t("studio.loading", { id: episodeId })}</div>
        <StudioSkeleton />
      </div>
    );
  }

  if (!episode) {
    return (
      <div className="page-container" data-testid="review-studio-error">
        <div className="error-banner">{error ?? t("studio.notFound")}</div>
        <Link className="secondary-btn" to="/">
          {t("studio.backToQueue")}
        </Link>
        {toast.StandaloneHost}
      </div>
    );
  }

  const statusMeta = EPISODE_STATUS_META[episode.status];
  const watchPct = Math.round(watchCoverage.coverage_ratio * 100);
  const minWatchPct = Math.round(minWatchCoverage * 100);
  const reviewerSubmitBlocked = !isLead && !canReviewerSubmit(watchCoverage, minWatchCoverage);
  const isVlmActive =
    episode.status === "vlm_queued" || episode.status === "vlm_running" || rerunning;
  const isDefaultTask = episode.meta.task_id === "_default";
  const canReassignTask = isLead
    ? !isVlmActive
    : isDefaultTask && !isGtFrozen && !isVlmActive;

  const runUsage: UsageSummary | null = episode.usage ?? null;
  const totalCachedTokens = runUsage
    ? (runUsage.total_cached_tokens ??
      runUsage.stages.reduce((acc, st) => acc + (st.cached_tokens ?? 0), 0))
    : 0;
  const cachedPct =
    runUsage && totalCachedTokens > 0 && runUsage.total_tokens > 0
      ? Math.min(100, Math.max(1, Math.round((totalCachedTokens / runUsage.total_tokens) * 100)))
      : 0;

  const handleToggleSnap = (next: boolean) => {
    setSnapEnabled(next);
    try {
      window.localStorage.setItem(SNAP_ENABLED_STORAGE_KEY, String(next));
    } catch {
      // Ignore storage errors
    }
  };

  const handleToggleCompactTracks = (next: boolean) => {
    setCompactTracks(next);
    try {
      window.localStorage.setItem(COMPACT_TRACKS_STORAGE_KEY, String(next));
    } catch {
      // Ignore storage errors
    }
  };

  return (
    <div className="studio-page" data-testid="review-studio-page">
      {/* Top Studio Header */}
      <div className="studio-header">
        <div className="studio-title-group">
          <Link
            className="icon-link"
            to="/"
            title={t("studio.backToQueueTitle")}
            data-testid="studio-back-link"
          >
            <ArrowLeft size={16} />
          </Link>
          <div>
            <div className="studio-ep-id">
              <span className="mono">{episode.episode_id}</span>
              <span className="status-tag" style={{ backgroundColor: statusMeta.color }}>
                {t(`status.${episode.status}`)}
              </span>
              {canReassignTask ? (
                <div className="task-popover-anchor" ref={taskPopoverWrapRef}>
                  <button
                    type="button"
                    className={`task-badge-btn ${isDefaultTask ? "warn" : ""}`}
                    data-testid="studio-task-badge-btn"
                    title={t("studio.taskEditTitle")}
                    aria-expanded={taskPopoverOpen}
                    onClick={handleOpenTaskPopover}
                  >
                    {isDefaultTask ? <AlertTriangle size={11} /> : <Tag size={11} />}
                    <span>
                      {isDefaultTask
                        ? t("studio.taskUnrecognized")
                        : t("studio.taskBadgeLabel", { task: episode.meta.task_id })}
                    </span>
                    <Edit3 size={10} />
                  </button>
                  {taskPopoverOpen && (
                    <div className="task-popover" data-testid="studio-task-popover">
                      <div className="section-header-row">
                        <strong className="small">{t("studio.taskPopoverTitle")}</strong>
                        <button
                          type="button"
                          className="mini-btn"
                          onClick={() => setTaskPopoverOpen(false)}
                          aria-label={t("common.close")}
                        >
                          <X size={12} />
                        </button>
                      </div>
                      {!isLead && (
                        <div className="muted small">{t("studio.taskReviewerHint")}</div>
                      )}
                      <select
                        aria-label={t("studio.taskSelectAria")}
                        data-testid="studio-task-select"
                        value={pendingTaskId}
                        onChange={(e) => setPendingTaskId(e.target.value)}
                      >
                        {!isLead && (
                          <option value="">{t("studio.taskSelectPlaceholder")}</option>
                        )}
                        {studioTaskOptions.map((item) => (
                          <option key={item.task_id} value={item.task_id}>
                            {item.task_id}
                            {item.title ? ` — ${item.title}` : ""}
                          </option>
                        ))}
                      </select>
                      <div className="modal-actions" style={{ justifyContent: "space-between" }}>
                        {isLead ? (
                          <button
                            type="button"
                            className="accent-btn compact"
                            data-testid="studio-new-task-ai-btn"
                            onClick={() => {
                              setTaskPopoverOpen(false);
                              setTaskWizardOpen(true);
                            }}
                          >
                            <Sparkles size={12} /> {t("studio.taskNewAiBtn")}
                          </button>
                        ) : (
                          <span />
                        )}
                        <button
                          type="button"
                          className="primary-btn compact"
                          data-testid="studio-task-save-btn"
                          disabled={
                            updatingTask ||
                            !pendingTaskId ||
                            pendingTaskId === episode.meta.task_id ||
                            (!isLead && pendingTaskId === "_default")
                          }
                          onClick={() => void handleApplyEpisodeTask(pendingTaskId, false)}
                        >
                          {updatingTask ? t("studio.taskSaving") : t("studio.taskSaveBtn")}
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <span
                  className={`badge-pill ${isDefaultTask ? "warn" : ""}`}
                  data-testid="studio-task-badge-readonly"
                  title={isVlmActive ? t("studio.taskRunningDisabledTitle") : undefined}
                >
                  {isDefaultTask ? <AlertTriangle size={11} /> : <Tag size={11} />}
                  <span>
                    {isDefaultTask
                      ? t("studio.taskUnrecognized")
                      : t("studio.taskBadgeLabel", { task: episode.meta.task_id })}
                  </span>
                </span>
              )}
              {runUsage && (
                <div className="studio-cost-anchor" ref={costPopoverWrapRef}>
                  <button
                    type="button"
                    className={`studio-cost-chip tabular-nums ${costPopoverOpen ? "active" : ""}`}
                    data-testid="studio-run-cost-chip"
                    title={t("studio.costChipTitle")}
                    aria-expanded={costPopoverOpen}
                    onClick={() => setCostPopoverOpen((v) => !v)}
                  >
                    <Coins size={12} />
                    <span>
                      {t("studio.costChipTokens", {
                        tokens: formatTokenCount(runUsage.total_tokens),
                      })}
                    </span>
                    <span className="studio-cost-sep">·</span>
                    <span className="studio-cost-usd">{formatUsd(runUsage.total_cost_usd)}</span>
                    {totalCachedTokens > 0 && (
                      <span
                        className="studio-cost-cached"
                        data-testid="studio-run-cost-cached"
                      >
                        {t("studio.costChipCached", { pct: cachedPct })}
                      </span>
                    )}
                    <ChevronDown size={11} />
                  </button>
                  {costPopoverOpen && (
                    <div
                      className="studio-cost-popover"
                      data-testid="studio-run-cost-popover"
                      role="dialog"
                      aria-label={t("studio.costChipTitle")}
                    >
                      <div className="section-header-row">
                        <div className="studio-cost-popover-title">
                          <Coins size={13} />
                          <strong>{t("studio.costChipTitle")}</strong>
                        </div>
                        <div className="studio-cost-popover-meta tabular-nums">
                          <span className="muted small">
                            {runUsage.total_elapsed_s.toFixed(1)}s
                          </span>
                          <button
                            type="button"
                            className="mini-btn"
                            onClick={() => setCostPopoverOpen(false)}
                            aria-label={t("common.close")}
                          >
                            <X size={12} />
                          </button>
                        </div>
                      </div>
                      <div className="studio-cost-section-label">
                        {t("studio.costStageBreakdown")}
                      </div>
                      <div className="studio-cost-table-wrap">
                        <table className="studio-cost-table tabular-nums">
                          <thead>
                            <tr>
                              <th>Stage</th>
                              <th>Model</th>
                              <th className="num">Tokens</th>
                              <th className="num">Cached</th>
                              <th className="num">Cost</th>
                              <th className="num">Time</th>
                            </tr>
                          </thead>
                          <tbody>
                            {runUsage.stages.map((st, idx) => (
                              <tr key={`${st.stage}-${idx}`}>
                                <td className="mono">{st.stage}</td>
                                <td className="mono muted studio-cost-model" title={st.model}>
                                  {st.model}
                                </td>
                                <td
                                  className="num mono"
                                  title={`${st.prompt_tokens} in / ${st.completion_tokens} out`}
                                >
                                  {formatTokenCount(st.total_tokens)}
                                </td>
                                <td className="num mono">
                                  {(st.cached_tokens ?? 0) > 0
                                    ? formatTokenCount(st.cached_tokens ?? 0)
                                    : "—"}
                                  {st.cache_hit && (
                                    <span className="studio-cache-hit-badge">
                                      {t("studio.costStageCacheHit")}
                                    </span>
                                  )}
                                </td>
                                <td className="num mono">{formatUsd(st.estimated_cost_usd)}</td>
                                <td className="num mono muted">{st.elapsed_s.toFixed(2)}s</td>
                              </tr>
                            ))}
                          </tbody>
                          <tfoot>
                            <tr>
                              <td colSpan={2}>
                                <strong>Total</strong>
                              </td>
                              <td className="num mono">
                                <strong>{formatTokenCount(runUsage.total_tokens)}</strong>
                              </td>
                              <td className="num mono">
                                <strong>
                                  {totalCachedTokens > 0
                                    ? formatTokenCount(totalCachedTokens)
                                    : "—"}
                                </strong>
                              </td>
                              <td className="num mono">
                                <strong>{formatUsd(runUsage.total_cost_usd)}</strong>
                              </td>
                              <td className="num mono">
                                <strong>{runUsage.total_elapsed_s.toFixed(2)}s</strong>
                              </td>
                            </tr>
                          </tfoot>
                        </table>
                      </div>
                    </div>
                  )}
                </div>
              )}
              {hasUnsavedChanges && (
                <span className="dirty-indicator" data-testid="unsaved-indicator">
                  {t("studio.unsaved")}
                </span>
              )}
            </div>
            <div className="muted small">
              {t("studio.metaSummary", {
                task: episode.meta.task_id,
                dur: episode.meta.duration_s.toFixed(2),
                run: episode.selected_run_id ?? episode.active_run_id ?? t("studio.none"),
              })}
            </div>
          </div>
        </div>

        <div className="studio-toolbar">
          {/* Lead-only Multi-Run Version Switcher & Local Re-Run VLM */}
          {isLead && (
            <>
              <div className="run-switcher-wrap" data-testid="run-version-switcher">
                <History size={13} className="muted" />
                <select
                  className="run-switcher-select"
                  aria-label={t("studio.runSwitcherAria")}
                  data-testid="run-version-select"
                  value={selectedRunId}
                  onChange={(e) => void handleSwitchRunVersion(e.target.value)}
                >
                  <option value="">
                    {t("studio.activeViewOption", {
                      run: episode.active_run_id ?? t("studio.noRun"),
                    })}
                  </option>
                  {availableRuns.map((r) => (
                    <option key={r.run_id} value={r.run_id}>
                      {r.run_id}
                      {r.run_id === episode.active_run_id ? t("studio.runTagActive") : ""}
                      {r.is_eval_sandbox ? t("studio.runTagSandbox") : ""} · {r.prompt_version}
                    </option>
                  ))}
                </select>
              </div>

              <div className="rerun-split-group" ref={rerunMenuWrapRef} data-testid="rerun-vlm-control">
                <button
                  type="button"
                  className="mini-btn accent rerun-main-btn"
                  disabled={rerunning || isGtFrozen}
                  onClick={() => void handleLocalRerunVlm("s3_s4")}
                  data-testid="rerun-vlm-s3s4-btn"
                  title={isGtFrozen ? t("studio.rerunDisabledGt") : t("studio.rerunS3S4Title")}
                >
                  <Sparkles size={13} />{" "}
                  {rerunning ? t("studio.rerunning") : t("studio.rerunS3S4")}
                </button>
                <button
                  type="button"
                  className="mini-btn accent rerun-caret-btn"
                  disabled={rerunning || isGtFrozen}
                  aria-label={t("studio.rerunMenuToggleAria")}
                  data-testid="rerun-vlm-menu-toggle"
                  onClick={() => setRerunMenuOpen((v) => !v)}
                >
                  <ChevronDown size={13} />
                </button>
                {rerunMenuOpen && (
                  <div className="rerun-menu" data-testid="rerun-vlm-menu">
                    <button
                      type="button"
                      className="rerun-menu-item"
                      data-testid="rerun-vlm-opt-s3s4"
                      onClick={() => void handleLocalRerunVlm("s3_s4")}
                    >
                      <strong>{t("studio.rerunOptS3S4Title")}</strong>
                      <span className="rerun-menu-desc">{t("studio.rerunOptS3S4Desc")}</span>
                    </button>
                    <button
                      type="button"
                      className="rerun-menu-item"
                      data-testid="rerun-vlm-s1-btn"
                      onClick={() => void handleLocalRerunVlm("s1_grounding")}
                    >
                      <strong>{t("studio.rerunOptS1Title")}</strong>
                      <span className="rerun-menu-desc">{t("studio.rerunOptS1Desc")}</span>
                    </button>
                    <button
                      type="button"
                      className="rerun-menu-item"
                      data-testid="rerun-vlm-custom-objects-btn"
                      onClick={() => {
                        setRerunMenuOpen(false);
                        setCustomObjectsModalOpen(true);
                      }}
                    >
                      <strong>{t("studio.rerunOptCustomTitle")}</strong>
                      <span className="rerun-menu-desc">{t("studio.rerunOptCustomDesc")}</span>
                    </button>
                    <button
                      type="button"
                      className="rerun-menu-item"
                      data-testid="rerun-vlm-all-btn"
                      onClick={() => void handleLocalRerunVlm("s1_s5")}
                    >
                      <strong>{t("studio.rerunOptAllTitle")}</strong>
                      <span className="rerun-menu-desc">{t("studio.rerunOptAllDesc")}</span>
                    </button>
                  </div>
                )}
              </div>

              {rerunning && localRerunJobId && (
                <button
                  type="button"
                  className="danger-btn compact"
                  disabled={cancellingRerun}
                  data-testid="cancel-rerun-vlm-btn"
                  onClick={() => void handleCancelLocalRerun()}
                  title={t("studio.cancelRerunTitle")}
                >
                  <XCircle size={12} />{" "}
                  {cancellingRerun ? t("studio.cancellingRerun") : t("studio.cancelRerun")}
                </button>
              )}

              {rerunCompletedReady && (
                <button
                  type="button"
                  className="mini-btn primary"
                  data-testid="refresh-after-rerun-btn"
                  onClick={() => {
                    const isProtected =
                      episode.status === "claimed" ||
                      episode.status === "reviewed" ||
                      episode.status === "flagged" ||
                      episode.status === "approved" ||
                      episode.status === "gt_frozen";
                    const nextRun = isProtected && rerunCompletedRunId ? rerunCompletedRunId : "";
                    setRerunCompletedReady(false);
                    setSelectedRunId(nextRun);
                    void loadRunsCatalog();
                    void loadEpisodeData(nextRun);
                  }}
                >
                  <RefreshCw size={12} /> {t("studio.loadNewRun")}
                </button>
              )}
            </>
          )}

          <div className="undo-redo-group">
            <button
              type="button"
              className="mini-btn"
              disabled={!history.canUndo || isReadOnly}
              onClick={history.undo}
              data-testid="undo-btn"
              title={t("studio.undoTitle")}
            >
              <Undo2 size={14} /> {t("studio.undo", { count: history.undoCount })}
            </button>
            <button
              type="button"
              className="mini-btn"
              disabled={!history.canRedo || isReadOnly}
              onClick={history.redo}
              data-testid="redo-btn"
              title={t("studio.redoTitle")}
            >
              <Redo2 size={14} /> {t("studio.redo", { count: history.redoCount })}
            </button>
            <button
              type="button"
              className="mini-btn"
              onClick={() => setShortcutsOpen((v) => !v)}
              data-testid="open-shortcuts-btn"
              title={t("studio.shortcutsTitle")}
              aria-label={t("studio.shortcutsAria")}
            >
              <Keyboard size={14} /> ?
            </button>
            <div className="studio-display-anchor" ref={displayPopoverWrapRef}>
              <button
                type="button"
                className={`mini-btn studio-display-btn ${displayPopoverOpen ? "active" : ""}`}
                data-testid="studio-display-popover-btn"
                aria-expanded={displayPopoverOpen}
                onClick={() => setDisplayPopoverOpen((v) => !v)}
              >
                <SlidersHorizontal size={13} />
                <span>{t("studio.displayControls")}</span>
                <ChevronDown size={11} />
              </button>
              {displayPopoverOpen && (
                <div
                  className="studio-display-popover"
                  data-testid="studio-display-popover"
                  role="dialog"
                  aria-label={t("studio.displayControls")}
                >
                  <div className="section-header-row">
                    <strong className="small">{t("studio.displayControls")}</strong>
                    <button
                      type="button"
                      className="mini-btn"
                      onClick={() => setDisplayPopoverOpen(false)}
                      aria-label={t("common.close")}
                    >
                      <X size={12} />
                    </button>
                  </div>

                  <div className="studio-display-toggles">
                    <label className="studio-display-toggle-row">
                      <div className="studio-display-toggle-meta">
                        <span className="studio-display-toggle-label">
                          <Magnet size={13} /> {t("studio.snapToGuides")}
                        </span>
                        <span className="studio-display-toggle-hint">{t("studio.snapHint")}</span>
                      </div>
                      <input
                        type="checkbox"
                        data-testid="toggle-timeline-snap"
                        checked={snapEnabled}
                        onChange={(e) => handleToggleSnap(e.target.checked)}
                      />
                    </label>

                    <label className="studio-display-toggle-row">
                      <div className="studio-display-toggle-meta">
                        <span className="studio-display-toggle-label">
                          <Rows3 size={13} /> {t("studio.compactTracks")}
                        </span>
                      </div>
                      <input
                        type="checkbox"
                        data-testid="toggle-compact-tracks"
                        checked={compactTracks}
                        onChange={(e) => handleToggleCompactTracks(e.target.checked)}
                      />
                    </label>
                  </div>

                  <div className="studio-display-divider" />

                  <div className="studio-display-shortcuts">
                    <div className="section-header-row">
                      <span className="studio-display-section-title">
                        {t("studio.shortcutsHeading")}
                      </span>
                      <button
                        type="button"
                        className="mini-btn"
                        data-testid="display-open-full-shortcuts-btn"
                        onClick={() => {
                          setDisplayPopoverOpen(false);
                          setShortcutsOpen(true);
                        }}
                      >
                        <Keyboard size={12} /> ?
                      </button>
                    </div>
                    <div className="studio-display-shortcuts-list">
                      {QUICK_SHORTCUT_ITEMS.map((item) => (
                        <div key={item.keys.join("/")} className="studio-quick-shortcut-row">
                          <span className="muted">{t(item.descKey)}</span>
                          <span className="shortcut-keys">
                            {item.keys.map((k, idx) => (
                              <React.Fragment key={k}>
                                {idx > 0 && <span className="muted">/</span>}
                                <kbd className="kbd-cap">{k}</kbd>
                              </React.Fragment>
                            ))}
                          </span>
                        </div>
                      ))}
                    </div>
                    <div className="studio-display-tip muted small">{t("studio.zoomTip")}</div>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="claim-controls">
            {isGtFrozen ? (
              <span className="lock-pill frozen" data-testid="gt-frozen-banner">
                <Lock size={13} /> {t("studio.gtFrozenBanner")}
              </span>
            ) : isClaimedByMe ? (
              <button
                type="button"
                className="secondary-btn compact"
                onClick={() => void handleRelease()}
                data-testid="release-claim-btn"
              >
                <Unlock size={14} /> {t("studio.releaseLock", { user: claimedBy ?? "" })}
              </button>
            ) : (
              <button
                type="button"
                className="secondary-btn compact"
                disabled={isClaimedByOther}
                onClick={() => void handleClaim()}
                data-testid="claim-btn"
              >
                <Lock size={14} />{" "}
                {isClaimedByOther
                  ? t("studio.claimedBy", { user: claimedBy ?? "" })
                  : t("studio.claimLock")}
              </button>
            )}
            {claimExpiresAt && (
              <span className="muted small">
                {t("studio.leaseUntil", {
                  time: new Date(claimExpiresAt * 1000).toLocaleTimeString(),
                })}
              </span>
            )}
          </div>
        </div>
      </div>

      {isHistoricalView && (
        <div className="historical-run-banner" data-testid="historical-run-banner">
          <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            <Eye size={14} />
            <span>
              <strong>{t("studio.historicalTitle")}</strong>
              {t("studio.historicalPrefix")}
              <code className="mono">{selectedRunId || episode.selected_run_id}</code>
              {t("studio.historicalSuffix")}
            </span>
          </span>
          <button
            type="button"
            className="mini-btn primary"
            data-testid="exit-historical-run-btn"
            onClick={() => void handleSwitchRunVersion("")}
          >
            {t("studio.returnActiveView")}
          </button>
        </div>
      )}

      {taskRerunPrompt && (
        <div className="info-banner" data-testid="studio-rerun-vlm-banner">
          <span>
            {isLead
              ? t("studio.taskRerunPrompt", { taskId: taskRerunPrompt.taskId })
              : t("studio.taskRerunPromptReviewer", { taskId: taskRerunPrompt.taskId })}
          </span>
          <div className="header-actions">
            {isLead && (
              <button
                type="button"
                className="primary-btn compact"
                data-testid="studio-rerun-vlm-now-btn"
                disabled={rerunning}
                onClick={() => void handleRerunVlmAfterTaskChange()}
              >
                <Play size={13} /> {t("studio.taskRerunNowBtn")}
              </button>
            )}
            <button
              type="button"
              className="mini-btn"
              data-testid="studio-dismiss-rerun-banner-btn"
              onClick={() => setTaskRerunPrompt(null)}
            >
              <X size={13} /> {t("common.dismiss")}
            </button>
          </div>
        </div>
      )}

      {isClaimedByOther && (
        <div className="warning-banner" data-testid="claim-conflict-banner">
          <AlertTriangle size={15} /> {t("studio.claimConflictPrefix")}
          <strong>{claimedBy}</strong>
          {t("studio.claimConflictSuffix")}
        </div>
      )}
      {error && (
        <div className="error-banner" data-testid="studio-error-banner">
          {error}
        </div>
      )}

      {/* Main Draggable Split Workspace */}
      <div
        ref={splitGridRef}
        className="studio-main-grid studio-resizable-grid"
        style={{ "--studio-left-pct": `${leftPanePct}%` } as React.CSSProperties}
        data-testid="studio-split-grid"
      >
        <div className="studio-left-col">
          <StudioPlayer
            episodeId={episode.episode_id}
            cameras={episode.meta.cameras}
            durationS={episode.meta.duration_s}
            fps={episode.meta.fps}
            grounding={episode.grounding}
            currentTime={currentTime}
            onTimeUpdate={setCurrentTime}
            onWatchInterval={handleWatchInterval}
            qcSegments={history.state.qc_segments}
            selectedSegment={selectedSegment}
            cutSkipEnabled={cutSkipEnabled}
            onCutSkipChange={setCutSkipEnabled}
            loopSegmentEnabled={loopSegmentEnabled}
            onLoopSegmentChange={setLoopSegmentEnabled}
            disableInternalShortcuts={true}
            controllerRef={playerCtrlRef}
          />

          <FourTrackTimeline
            durationS={episode.meta.duration_s}
            currentTime={currentTime}
            macroStages={history.state.macro_stages}
            armTimelines={episode.arm_timelines}
            qcSegments={history.state.qc_segments}
            rawVlmSegments={episode.raw_vlm_segments}
            watchCoverage={watchCoverage}
            showDiffTrack={isLead && showDiffTrack}
            readOnly={isReadOnly}
            snapEnabled={snapEnabled}
            compact={compactTracks}
            selectedSegmentId={selectedSegmentId}
            selectedStageId={selectedStageId}
            onSeek={setCurrentTime}
            onSelectSegment={(id) => {
              setSelectedSegmentId(id);
              if (id) setActiveTab("segment");
            }}
            onSelectStage={(id) => {
              setSelectedStageId(id);
              if (id) setActiveTab("segment");
            }}
            onAddSegment={() => handleAddSegmentAt(currentTime)}
            onCreateSegmentSpan={(tStart, tEnd) => handleAddSegmentAt(tStart, tEnd)}
            onSplitMacroStage={handleSplitMacroStage}
            onAddMacroStage={() => {
              const maxDur = Math.max(0.1, episode.meta.duration_s);
              const fallbackSpan = Math.min(2.0, Math.max(0.2, Number((maxDur * 0.2).toFixed(2))));
              const rawT0 = Math.max(0, Number(currentTime.toFixed(2)));
              const t0 = Number(
                (rawT0 + 0.1 > maxDur ? Math.max(0, maxDur - fallbackSpan) : rawT0).toFixed(2),
              );
              const t1 = Number(
                Math.min(maxDur, Math.max(t0 + 0.1, Number((t0 + 2.0).toFixed(2)))).toFixed(2),
              );
              const st: MacroStage = {
                stage_id: `M${history.state.macro_stages.length + 1}_${Date.now().toString(36).slice(-3)}`,
                t_start: t0,
                t_end: t1 > t0 ? t1 : Number(maxDur.toFixed(2)),
                label: t("studio.newSubtaskStageLabel"),
                label_en: "new_stage",
                summary: "",
                instruction: "",
                color_rgb: [59, 130, 246],
              };
              history.applyEdit((prev) => ({
                ...prev,
                macro_stages: [...prev.macro_stages, st],
              }));
              setSelectedStageId(st.stage_id);
              setSelectedSegmentId(null);
              setActiveTab("segment");
            }}
            onUpdateSegmentSpan={(segId, tStart, tEnd) => {
              history.applyEdit((prev) => ({
                ...prev,
                qc_segments: prev.qc_segments.map((s) =>
                  s.seg_id === segId
                    ? normalizeSegmentTriplet(s, { t_start: tStart, t_end: tEnd }, labels)
                    : s,
                ),
              }));
            }}
            onUpdateStageSpan={(stageId, tStart, tEnd) => {
              history.applyEdit((prev) => ({
                ...prev,
                macro_stages: prev.macro_stages.map((st) =>
                  st.stage_id === stageId ? { ...st, t_start: tStart, t_end: tEnd } : st,
                ),
              }));
            }}
            onUpdateStageSpans={handleUpdateStageSpans}
            onAlignStagesToQc={handleAlignStagesToQc}
            onInferSubtasks={() => void handleInferSubtasks()}
            inferringSubtasks={inferringSubtasks}
          />
        </div>

        {/* 6px Draggable Vertical Split Resizer (50%–75%) */}
        <div
          className={`studio-split-resizer ${draggingSplit ? "dragging" : ""}`}
          role="separator"
          aria-orientation="vertical"
          aria-label={t("studio.splitResizerAria")}
          aria-valuenow={leftPanePct}
          aria-valuemin={MIN_LEFT_PCT}
          aria-valuemax={MAX_LEFT_PCT}
          tabIndex={0}
          data-testid="studio-split-resizer"
          title={t("studio.splitResizerTitle")}
          onPointerDown={startSplitDrag}
          onKeyDown={handleSplitKeyDown}
        />

        <div className="studio-right-col">
          <div className="inspector-tabs">
            <button
              type="button"
              className={`tab-btn ${activeTab === "segment" ? "active" : ""}`}
              onClick={() => setActiveTab("segment")}
            >
              {t("studio.tabSegmentStage")}
            </button>
            <button
              type="button"
              className={`tab-btn ${activeTab === "success" ? "active" : ""}`}
              onClick={() => setActiveTab("success")}
            >
              {t("studio.tabSuccess")}
            </button>
            {canViewDiff && (
              <button
                type="button"
                className={`tab-btn ${activeTab === "diff" ? "active" : ""}`}
                onClick={() => setActiveTab("diff")}
                data-testid="diff-tab-btn"
              >
                {t("studio.tabDiff")}
              </button>
            )}
          </div>

          <div className="inspector-body">
            {activeTab === "segment" && (
              <SegmentInspector
                episodeId={episode.episode_id}
                taskId={episode.meta.task_id}
                durationS={episode.meta.duration_s}
                currentTime={currentTime}
                isLead={isLead}
                readOnly={isReadOnly}
                labels={labels}
                primaryCamera={primaryCamera}
                authToken={token}
                selectedSegment={selectedSegment}
                selectedStage={selectedStage}
                onSeek={setCurrentTime}
                onUpdateSegment={(nextSeg) => {
                  history.applyEdit((prev) => ({
                    ...prev,
                    qc_segments: prev.qc_segments.map((s) =>
                      s.seg_id === nextSeg.seg_id ? nextSeg : s,
                    ),
                  }));
                }}
                onDeleteSegment={(segId) => {
                  history.applyEdit((prev) => ({
                    ...prev,
                    qc_segments: prev.qc_segments.filter((s) => s.seg_id !== segId),
                  }));
                  setSelectedSegmentId(null);
                }}
                onUpdateStage={(nextStage) => {
                  history.applyEdit((prev) => ({
                    ...prev,
                    macro_stages: prev.macro_stages.map((st) =>
                      st.stage_id === nextStage.stage_id ? nextStage : st,
                    ),
                  }));
                }}
                onDeleteStage={(stageId) => {
                  history.applyEdit((prev) => ({
                    ...prev,
                    macro_stages: prev.macro_stages.filter((st) => st.stage_id !== stageId),
                  }));
                  setSelectedStageId(null);
                }}
              />
            )}

            {activeTab === "success" && (
              <SuccessPanel
                success={history.state.success}
                currentTime={currentTime}
                durationS={episode.meta.duration_s}
                readOnly={isReadOnly}
                onChange={(nextSuccess) => {
                  history.applyEdit((prev) => ({
                    ...prev,
                    success: nextSuccess,
                  }));
                }}
                onSeek={setCurrentTime}
              />
            )}

            {activeTab === "diff" && canViewDiff && (
              <DiffOverlay
                isLead={isLead}
                diffEnabled={showDiffTrack}
                onToggleDiff={setShowDiffTrack}
                reviewDiff={episode.review_diff}
                rawVlmSegments={episode.raw_vlm_segments}
                currentSegments={history.state.qc_segments}
                rawVlmSuccess={episode.raw_vlm_success}
                currentSuccess={history.state.success}
                rawVlmMacroStages={episode.raw_vlm_macro_stages}
                currentMacroStages={history.state.macro_stages}
                reviews={reviews}
                onRestoreSnapshot={(rev) => {
                  history.restoreSnapshot(rev);
                  notify(t("studio.toastRestoredSnapshot", { revId: rev.rev_id }), "info");
                }}
                onSeek={setCurrentTime}
              />
            )}
          </div>
        </div>
      </div>

      {/* Bottom RBAC-Gated Action Footer */}
      <div className="studio-footer" data-testid="studio-action-footer">
        <div className="footer-left">
          <div className="watch-coverage-badge tabular-nums" data-testid="watch-coverage-badge">
            <span>{t("studio.watchCoverageLabel")}</span>
            <strong>{watchPct}%</strong>
            <span className="muted">{t("studio.watchCoverageReq", { pct: minWatchPct })}</span>
          </div>
          <input
            type="text"
            className="notes-input"
            placeholder={t("studio.notesPlaceholder")}
            disabled={isReadOnly}
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
          />
          <label className="auto-advance-label">
            <input
              type="checkbox"
              data-testid="auto-advance-toggle"
              checked={autoAdvance}
              onChange={(e) => {
                const next = e.target.checked;
                setAutoAdvance(next);
                try {
                  window.localStorage.setItem(AUTO_ADVANCE_STORAGE_KEY, String(next));
                } catch {
                  // Ignore storage errors
                }
              }}
            />
            <span>{t("studio.autoAdvance")}</span>
          </label>
        </div>

        <div className="footer-actions">
          <button
            type="button"
            className="secondary-btn"
            disabled={saving || isReadOnly}
            onClick={() => void handleSubmitAction("save_draft")}
            data-testid="action-save-draft"
          >
            <Save size={14} /> {t("studio.saveDraft")}
          </button>

          <button
            type="button"
            className="primary-btn"
            disabled={saving || isReadOnly || reviewerSubmitBlocked}
            title={
              reviewerSubmitBlocked
                ? t("studio.submitBlockedTitle", { pct: minWatchPct })
                : t("studio.submitTitle")
            }
            onClick={() => void handleSubmitAction("submit_review")}
            data-testid="action-submit-review"
          >
            <CheckCircle size={14} /> {t("studio.submitReview")}
          </button>

          <button
            type="button"
            className="warning-btn"
            disabled={saving || isReadOnly}
            onClick={() => setFlagDialogOpen((v) => !v)}
            data-testid="action-flag-btn"
          >
            <Flag size={14} /> {t("studio.flagIssue")}
          </button>

          {canApprove && (
            <button
              type="button"
              className="success-btn"
              disabled={saving || isReadOnly}
              onClick={() => void handleSubmitAction("approve")}
              data-testid="action-approve"
            >
              <ShieldCheck size={14} /> {t("studio.approve")}
            </button>
          )}

          {canFreezeGt && (
            <button
              type="button"
              className="accent-btn"
              disabled={saving || isReadOnly}
              onClick={() => void handleSubmitAction("freeze_gt")}
              data-testid="action-freeze-gt"
            >
              <Lock size={14} /> {t("studio.freezeGt")}
            </button>
          )}

          {isLead && (
            <button
              type="button"
              className="secondary-btn"
              onClick={() => void handleRenderSingleMp4()}
              data-testid="action-render-mp4"
            >
              <Video size={14} /> {t("studio.renderMp4")}
            </button>
          )}

          <div
            className="export-menu-wrap"
            ref={exportMenuWrapRef}
            data-testid="studio-export-menu-wrap"
          >
            <button
              type="button"
              className="secondary-btn"
              aria-expanded={exportMenuOpen}
              onClick={() => setExportMenuOpen((v) => !v)}
              data-testid="studio-export-btn"
            >
              <Download size={14} /> {t("studio.exportBtn")} <ChevronDown size={13} />
            </button>
            {exportMenuOpen && (
              <div className="export-popover" data-testid="studio-export-menu">
                <button
                  type="button"
                  className="rerun-menu-item"
                  data-testid="export-download-json-btn"
                  onClick={handleDownloadEpisodeJson}
                >
                  <strong style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
                    <FileJson size={13} /> {t("studio.exportJsonTitle")}
                  </strong>
                  <span className="rerun-menu-desc">{t("studio.exportJsonDesc")}</span>
                </button>
                <button
                  type="button"
                  className="rerun-menu-item"
                  data-testid="export-open-studio-btn"
                  onClick={handleOpenExportStudio}
                >
                  <strong style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
                    {isLead ? <ExternalLink size={13} /> : <Lock size={13} />}{" "}
                    {t("studio.exportStudioTitle")}
                  </strong>
                  <span className="rerun-menu-desc">
                    {isLead
                      ? t("studio.exportStudioDescLead")
                      : t("studio.exportStudioDescReviewer")}
                  </span>
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {flagDialogOpen && (
        <div className="flag-modal-bar" data-testid="flag-reason-bar">
          <input
            type="text"
            placeholder={t("studio.flagPlaceholder")}
            value={flaggedReason}
            onChange={(e) => setFlaggedReason(e.target.value)}
          />
          <button
            type="button"
            className="warning-btn compact"
            onClick={() => void handleSubmitAction("flag")}
          >
            {t("studio.confirmFlag")}
          </button>
          <button
            type="button"
            className="mini-btn"
            onClick={() => setFlagDialogOpen(false)}
          >
            {t("common.cancel")}
          </button>
        </div>
      )}

      {customObjectsModalOpen && (
        <div
          className="modal-backdrop"
          data-testid="custom-objects-modal"
          onClick={(e) => {
            if (e.target === e.currentTarget) setCustomObjectsModalOpen(false);
          }}
        >
          <form
            className="modal-card custom-objects-modal-card"
            onSubmit={(e) => void handleSubmitCustomObjectsS1(e)}
          >
            <div className="inspector-header">
              <h3 style={{ margin: 0 }}>{t("studio.customObjTitle")}</h3>
              <button
                type="button"
                className="mini-btn"
                aria-label={t("studio.customObjCloseAria")}
                onClick={() => setCustomObjectsModalOpen(false)}
              >
                <X size={14} />
              </button>
            </div>
            <p className="muted small" style={{ margin: 0 }}>
              {t("studio.customObjDesc")}
            </p>
            <label className="field-group">
              <span>{t("studio.customObjLabel")}</span>
              <input
                type="text"
                autoFocus
                data-testid="custom-objects-input"
                value={customObjectsInput}
                onChange={(e) => setCustomObjectsInput(e.target.value)}
                placeholder={t("studio.customObjPlaceholder")}
              />
            </label>
            {parsedCustomObjects.length > 0 && (
              <div className="custom-objects-tags" data-testid="custom-objects-preview">
                {parsedCustomObjects.map((obj, idx) => (
                  <span key={`${obj}-${idx}`} className="custom-object-tag">
                    {obj}
                  </span>
                ))}
              </div>
            )}
            <div className="footer-actions" style={{ justifyContent: "flex-end" }}>
              <button
                type="button"
                className="secondary-btn compact"
                onClick={() => setCustomObjectsModalOpen(false)}
              >
                {t("common.cancel")}
              </button>
              <button
                type="submit"
                className="primary-btn compact"
                disabled={parsedCustomObjects.length === 0 || rerunning}
                data-testid="submit-custom-objects-btn"
              >
                <Sparkles size={13} />{" "}
                {t("studio.customObjSubmit", { count: parsedCustomObjects.length })}
              </button>
            </div>
          </form>
        </div>
      )}

      {isLead && (
        <>
          <TaskDraftWizardModal
            open={taskWizardOpen}
            onClose={() => setTaskWizardOpen(false)}
            initialSampleEpisodeId={episode.episode_id}
            sampleEpisodes={[
              {
                episode_id: episode.episode_id,
                task_id: episode.meta.task_id,
                dataset: episode.meta.dataset,
              },
            ]}
            onTaskSaved={handleStudioWizardTaskSaved}
          />
          {frozenTaskConfirmId && (
            <div
              className="modal-backdrop modal-backdrop-top"
              data-testid="studio-frozen-task-confirm-modal"
              onClick={(e) => {
                if (e.target === e.currentTarget) setFrozenTaskConfirmId(null);
              }}
            >
              <div className="modal-card">
                <h3 style={{ margin: 0, display: "inline-flex", alignItems: "center", gap: 6 }}>
                  <AlertTriangle size={16} /> {t("studio.taskFrozenConfirmTitle")}
                </h3>
                <p className="muted small" style={{ margin: 0 }}>
                  {t("studio.taskFrozenConfirmDesc", { taskId: frozenTaskConfirmId })}
                </p>
                <div className="modal-actions">
                  <button
                    type="button"
                    className="secondary-btn compact"
                    onClick={() => setFrozenTaskConfirmId(null)}
                  >
                    {t("common.cancel")}
                  </button>
                  <button
                    type="button"
                    className="warning-btn compact"
                    data-testid="studio-confirm-frozen-task-btn"
                    disabled={updatingTask}
                    onClick={() => void handleApplyEpisodeTask(frozenTaskConfirmId, true)}
                  >
                    {t("studio.taskFrozenConfirmBtn")}
                  </button>
                </div>
              </div>
            </div>
          )}
        </>
      )}

      <ShortcutsModal open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />
      {toast.StandaloneHost}
    </div>
  );
};

export default ReviewStudioPage;
