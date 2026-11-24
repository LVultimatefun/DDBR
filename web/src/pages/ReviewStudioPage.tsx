import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import {
  AlertTriangle,
  ArrowLeft,
  CheckCircle,
  ChevronDown,
  Eye,
  Flag,
  History,
  Keyboard,
  Lock,
  Redo2,
  RefreshCw,
  Save,
  ShieldCheck,
  Sparkles,
  Undo2,
  Unlock,
  Video,
  XCircle,
} from "lucide-react";
import {
  ApiError,
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
  listEpisodeRuns,
  listEpisodes,
  listReviews,
  releaseClaim,
  renewClaim,
  submitReview,
} from "../api/client";
import type {
  HumanReviewArtifact,
  JobStatus,
  MacroStage,
  ResolvedEpisode,
  ReviewAction,
  SplicePolicy,
  SuccessVerdict,
  VisualLabelDef,
  VlmRunSummary,
  WatchCoverage,
} from "../api/types";
import { DiffOverlay } from "../components/DiffOverlay";
import { FourTrackTimeline, type StageSpanUpdate } from "../components/FourTrackTimeline";
import { SegmentInspector } from "../components/SegmentInspector";
import { ShortcutsModal } from "../components/ShortcutsModal";
import { StudioPlayer, type StudioPlayerHandle } from "../components/StudioPlayer";
import { SuccessPanel } from "../components/SuccessPanel";
import {
  createDefaultSegment,
  DEFAULT_VISUAL_LABELS,
  EPISODE_STATUS_META,
  normalizeSegmentTriplet,
} from "../constants/labels";
import { useAuth } from "../stores/authStore";
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
const DEFAULT_LEFT_PCT = 61;
const MIN_LEFT_PCT = 50;
const MAX_LEFT_PCT = 75;

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

  const [episode, setEpisode] = useState<ResolvedEpisode | null>(null);
  const [reviews, setReviews] = useState<HumanReviewArtifact[]>([]);
  const [availableRuns, setAvailableRuns] = useState<VlmRunSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string>("");
  const [labels, setLabels] = useState<VisualLabelDef[]>(DEFAULT_VISUAL_LABELS);
  const [minWatchCoverage, setMinWatchCoverage] = useState<number>(MIN_WATCH_COVERAGE);

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
  const [rerunMenuOpen, setRerunMenuOpen] = useState<boolean>(false);
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
        setError(err instanceof Error ? err.message : "Failed to load episode details");
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
            ? `Single-episode VLM rerun job ${tracked.job_id} cancelled; episode status reverted`
            : errDetail
              ? `Single-episode VLM rerun job ${tracked.job_id} failed: ${errDetail}`
              : `Single-episode VLM rerun job ${tracked.job_id} failed`,
          tracked.status === "cancelled" ? "warning" : "error",
        );
        void loadEpisodeData();
      }
    },
    [loadRunsCatalog, loadEpisodeData, notify],
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
            setError("Claim lease expired or was taken by another reviewer. Please re-claim or refresh.");
            setClaimedBy(null);
            setClaimExpiresAt(null);
          }
        });
    }, 60_000);
    return () => window.clearInterval(timer);
  }, [episodeId, isClaimedByMe, isGtFrozen, username]);

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
      if (!window.confirm("You have unsaved review edits. Are you sure you want to leave Review Studio?")) {
        e.preventDefault();
        e.stopPropagation();
      }
    };
    document.addEventListener("click", onDocumentClick, true);
    return () => document.removeEventListener("click", onDocumentClick, true);
  }, [hasUnsavedChanges]);

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
      notify("Move the playhead inside a macro stage (at least 0.1s from edges) to split", "warning");
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
    notify(`Split macro stage at ${splitT.toFixed(2)}s`, "info");
  }, [currentTime, episode, history, isReadOnly, notify]);

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
      notify(`Claimed lock: ${res.claimed_by}`, "success");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to claim episode");
    }
  };

  const handleRelease = async () => {
    try {
      const res = await releaseClaim(episodeId, username);
      setClaimedBy(null);
      setClaimExpiresAt(null);
      setEpisode((prev) => (prev ? { ...prev, status: res.status } : prev));
      notify("Claim lock released", "info");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to release claim");
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
            notify(`Auto-claimed next episode: ${cand.episode_id}`, "success");
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
        notify(`Auto-advanced to next episode: ${nextLead.episode_id}`, "success");
        navigate(`/review/${encodeEpPath(nextLead.episode_id)}`);
        return true;
      }
      return false;
    } catch {
      return false;
    }
  }, [episodeId, isLead, navigate, notify, username]);

  const handleSubmitAction = useCallback(
    async (action: ReviewAction) => {
      if (!episode || isReadOnly) return;
      if (!episode.active_run_id) {
        setError("This episode has no VLM run yet and cannot be reviewed");
        return;
      }
      if (!isLead && action === "submit_review" && !canReviewerSubmit(watchCoverage, minWatchCoverage)) {
        setError(
          `Insufficient watch coverage (${(watchCoverage.coverage_ratio * 100).toFixed(1)}% < ${Math.round(minWatchCoverage * 100)}%). Watch the episode before submitting review.`,
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
          label: st.label.trim() || `Stage ${idx + 1}`,
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
        notify(`Action "${action}" submitted (rev: ${saved.rev_id})`, "success");
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
        setError(err instanceof Error ? err.message : "Failed to submit review");
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
    ],
  );

  const handleRenderSingleMp4 = async () => {
    try {
      const job = await createJob({ job_type: "render_mp4", episode_ids: [episodeId] });
      jobStore.trackJob(job);
      notify(`Triggered 1080p Studio MP4 render job: ${job.job_id}`, "info");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to trigger MP4 render");
    }
  };

  const handleLocalRerunVlm = async (mode: "s3_s4" | "s1_grounding" | "s1_s5") => {
    if (isGtFrozen) return;
    setRerunMenuOpen(false);
    setRerunning(true);
    setRerunCompletedReady(false);
    setRerunCompletedRunId(null);
    setError(null);
    try {
      const job = await createJob({
        job_type: "batch_vlm",
        episode_ids: [episodeId],
        stage: mode,
        fresh: mode === "s1_s5",
      });
      setLocalRerunJobId(job.job_id);
      jobStore.trackJob(job);
      const label =
        mode === "s3_s4"
          ? "S3/S4 rerun (cached)"
          : mode === "s1_grounding"
            ? "S1 only rerun (refresh 2D boxes, reuse S2-S5 cache)"
            : "Full S1-S5 rerun";
      notify(`Submitted single-episode ${label} job: ${job.job_id}`, "info");
      if (job.status === "completed" || job.status === "failed" || job.status === "cancelled") {
        handleRerunTerminalJob(job);
      }
    } catch (err) {
      setRerunning(false);
      setError(err instanceof Error ? err.message : "Failed to trigger local VLM rerun");
    }
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
          notify(
            `Single-episode VLM rerun job ${jobId} cancelled; episode status reverted`,
            "warning",
          );
        }
        await loadEpisodeData();
      }
    } finally {
      setCancellingRerun(false);
    }
  };

  const handleSwitchRunVersion = async (nextRunId: string) => {
    if (nextRunId === selectedRunId) return;
    if (
      nextRunId &&
      hasUnsavedChanges &&
      !window.confirm(
        "You have unsaved edits. Switching to a historical run read-only view will discard unsaved drafts. Continue?",
      )
    ) {
      return;
    }
    setSelectedRunId(nextRunId);
    await loadEpisodeData(nextRunId);
    if (nextRunId) {
      notify(`Switched to read-only historical run: ${nextRunId}`, "info");
    } else {
      notify("Returned to active review view", "info");
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
    shortcutsOpen,
    flagDialogOpen,
    rerunMenuOpen,
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
      <div className="page-container" data-testid="review-studio-loading">
        Loading Review Studio ({episodeId})...
      </div>
    );
  }

  if (!episode) {
    return (
      <div className="page-container" data-testid="review-studio-error">
        <div className="error-banner">{error ?? "Episode not found"}</div>
        <Link className="secondary-btn" to="/">
          Back to Work Queue
        </Link>
        {toast.StandaloneHost}
      </div>
    );
  }

  const statusMeta = EPISODE_STATUS_META[episode.status];
  const watchPct = Math.round(watchCoverage.coverage_ratio * 100);
  const minWatchPct = Math.round(minWatchCoverage * 100);
  const reviewerSubmitBlocked = !isLead && !canReviewerSubmit(watchCoverage, minWatchCoverage);

  return (
    <div className="studio-page" data-testid="review-studio-page">
      {/* Top Studio Header */}
      <div className="studio-header">
        <div className="studio-title-group">
          <Link className="icon-link" to="/" title="Back to queue" data-testid="studio-back-link">
            <ArrowLeft size={16} />
          </Link>
          <div>
            <div className="studio-ep-id">
              <span className="mono">{episode.episode_id}</span>
              <span className="status-tag" style={{ backgroundColor: statusMeta.color }}>
                {statusMeta.label}
              </span>
              {hasUnsavedChanges && (
                <span className="dirty-indicator" data-testid="unsaved-indicator">
                  ● Unsaved
                </span>
              )}
            </div>
            <div className="muted small">
              Task: {episode.meta.task_id} · Duration: {episode.meta.duration_s.toFixed(2)}s · Run:{" "}
              {episode.selected_run_id ?? episode.active_run_id ?? "None"}
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
                  aria-label="Switch VLM run version"
                  data-testid="run-version-select"
                  value={selectedRunId}
                  onChange={(e) => void handleSwitchRunVersion(e.target.value)}
                >
                  <option value="">
                    Active View ({episode.active_run_id ?? "No Run"})
                  </option>
                  {availableRuns.map((r) => (
                    <option key={r.run_id} value={r.run_id}>
                      {r.run_id}
                      {r.run_id === episode.active_run_id ? " [Active]" : ""}
                      {r.is_eval_sandbox ? " [Sandbox]" : ""} · {r.prompt_version}
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
                  title={
                    isGtFrozen
                      ? "GT is frozen; VLM rerun is disabled"
                      : "Reuse S1/S2/S5 cache and quickly rerun S3/S4"
                  }
                >
                  <Sparkles size={13} /> {rerunning ? "Rerunning..." : "Rerun S3/S4"}
                </button>
                <button
                  type="button"
                  className="mini-btn accent rerun-caret-btn"
                  disabled={rerunning || isGtFrozen}
                  aria-label="Expand VLM rerun options"
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
                      <strong>Rerun S3/S4 Only</strong>
                      <span className="rerun-menu-desc">
                        Reuse S1/S2/S5 stage caches and refresh anomaly discovery &amp; verification
                      </span>
                    </button>
                    <button
                      type="button"
                      className="rerun-menu-item"
                      data-testid="rerun-vlm-s1-btn"
                      onClick={() => void handleLocalRerunVlm("s1_grounding")}
                    >
                      <strong>Rerun S1 Only (Refresh 2D Boxes, Reuse S2-S5 Cache)</strong>
                      <span className="rerun-menu-desc">
                        Refresh 2D boxes and frame-level tracking while preserving S2–S5 caches
                      </span>
                    </button>
                    <button
                      type="button"
                      className="rerun-menu-item"
                      data-testid="rerun-vlm-all-btn"
                      onClick={() => void handleLocalRerunVlm("s1_s5")}
                    >
                      <strong>Full Rerun S1–S5</strong>
                      <span className="rerun-menu-desc">
                        Bypass caches and rerun all 5 VLM stages from scratch
                      </span>
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
                  title="Cancel this single-episode VLM rerun and revert episode status immediately"
                >
                  <XCircle size={12} /> {cancellingRerun ? "Cancelling..." : "Cancel Rerun"}
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
                  <RefreshCw size={12} /> Load New Run
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
              title="Undo (Ctrl+Z)"
            >
              <Undo2 size={14} /> Undo ({history.undoCount})
            </button>
            <button
              type="button"
              className="mini-btn"
              disabled={!history.canRedo || isReadOnly}
              onClick={history.redo}
              data-testid="redo-btn"
              title="Redo (Ctrl+Shift+Z)"
            >
              <Redo2 size={14} /> Redo ({history.redoCount})
            </button>
            <button
              type="button"
              className="mini-btn"
              onClick={() => setShortcutsOpen((v) => !v)}
              data-testid="open-shortcuts-btn"
              title="Keyboard shortcuts (?)"
              aria-label="Open keyboard shortcuts"
            >
              <Keyboard size={14} /> ?
            </button>
          </div>

          <div className="claim-controls">
            {isGtFrozen ? (
              <span className="lock-pill frozen" data-testid="gt-frozen-banner">
                <Lock size={13} /> GT Frozen (Read-Only)
              </span>
            ) : isClaimedByMe ? (
              <button
                type="button"
                className="secondary-btn compact"
                onClick={() => void handleRelease()}
                data-testid="release-claim-btn"
              >
                <Unlock size={14} /> Release Lock ({claimedBy})
              </button>
            ) : (
              <button
                type="button"
                className="secondary-btn compact"
                disabled={isClaimedByOther}
                onClick={() => void handleClaim()}
                data-testid="claim-btn"
              >
                <Lock size={14} /> {isClaimedByOther ? `Claimed by ${claimedBy}` : "Claim Lock"}
              </button>
            )}
            {claimExpiresAt && (
              <span className="muted small">
                Lease until {new Date(claimExpiresAt * 1000).toLocaleTimeString()}
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
              <strong>Read-Only Compare Mode</strong>: viewing historical VLM run{" "}
              <code className="mono">{selectedRunId || episode.selected_run_id}</code> (does not
              modify human reviews or frozen GT)
            </span>
          </span>
          <button
            type="button"
            className="mini-btn primary"
            data-testid="exit-historical-run-btn"
            onClick={() => void handleSwitchRunVersion("")}
          >
            Return to Active View
          </button>
        </div>
      )}

      {isClaimedByOther && (
        <div className="warning-banner" data-testid="claim-conflict-banner">
          <AlertTriangle size={15} /> This episode is currently locked by <strong>{claimedBy}</strong>{" "}
          (read-only preview).
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
                label: "New Macro Stage",
                label_en: "new_stage",
                summary: "",
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
          />
        </div>

        {/* 6px Draggable Vertical Split Resizer (50%–75%) */}
        <div
          className={`studio-split-resizer ${draggingSplit ? "dragging" : ""}`}
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize left and right studio panes"
          aria-valuenow={leftPanePct}
          aria-valuemin={MIN_LEFT_PCT}
          aria-valuemax={MAX_LEFT_PCT}
          tabIndex={0}
          data-testid="studio-split-resizer"
          title="Drag horizontally to resize player and inspector panes (50%–75%)"
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
              Segment / Stage
            </button>
            <button
              type="button"
              className={`tab-btn ${activeTab === "success" ? "active" : ""}`}
              onClick={() => setActiveTab("success")}
            >
              Success Verdict
            </button>
            {canViewDiff && (
              <button
                type="button"
                className={`tab-btn ${activeTab === "diff" ? "active" : ""}`}
                onClick={() => setActiveTab("diff")}
                data-testid="diff-tab-btn"
              >
                VLM Diff &amp; History
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
                  notify(`Restored snapshot ${rev.rev_id} to edit stack (save or submit to commit)`, "info");
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
          <div className="watch-coverage-badge" data-testid="watch-coverage-badge">
            <span>Watch Coverage: </span>
            <strong>{watchPct}%</strong>
            <span className="muted"> (Required ≥ {minWatchPct}%)</span>
          </div>
          <input
            type="text"
            className="notes-input"
            placeholder="Review notes (notes)..."
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
            <span>Auto-advance on submit</span>
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
            <Save size={14} /> Save Draft
          </button>

          <button
            type="button"
            className="primary-btn"
            disabled={saving || isReadOnly || reviewerSubmitBlocked}
            title={
              reviewerSubmitBlocked
                ? `Watch coverage must reach ${minWatchPct}% before submitting review`
                : "Submit review"
            }
            onClick={() => void handleSubmitAction("submit_review")}
            data-testid="action-submit-review"
          >
            <CheckCircle size={14} /> Submit Review
          </button>

          <button
            type="button"
            className="warning-btn"
            disabled={saving || isReadOnly}
            onClick={() => setFlagDialogOpen((v) => !v)}
            data-testid="action-flag-btn"
          >
            <Flag size={14} /> Flag Issue
          </button>

          {canApprove && (
            <button
              type="button"
              className="success-btn"
              disabled={saving || isReadOnly}
              onClick={() => void handleSubmitAction("approve")}
              data-testid="action-approve"
            >
              <ShieldCheck size={14} /> Approve
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
              <Lock size={14} /> Freeze as GT
            </button>
          )}

          {isLead && (
            <button
              type="button"
              className="secondary-btn"
              onClick={() => void handleRenderSingleMp4()}
              data-testid="action-render-mp4"
            >
              <Video size={14} /> Render 1080p MP4
            </button>
          )}
        </div>
      </div>

      {flagDialogOpen && (
        <div className="flag-modal-bar" data-testid="flag-reason-bar">
          <input
            type="text"
            placeholder="Describe why this episode is flagged (flagged_reason)..."
            value={flaggedReason}
            onChange={(e) => setFlaggedReason(e.target.value)}
          />
          <button
            type="button"
            className="warning-btn compact"
            onClick={() => void handleSubmitAction("flag")}
          >
            Confirm Flag
          </button>
          <button
            type="button"
            className="mini-btn"
            onClick={() => setFlagDialogOpen(false)}
          >
            Cancel
          </button>
        </div>
      )}

      <ShortcutsModal open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />
      {toast.StandaloneHost}
    </div>
  );
};
