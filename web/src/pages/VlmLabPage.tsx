import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  Bookmark,
  CheckCircle2,
  FlaskConical,
  GitCompare,
  History,
  Layers,
  Plus,
  RefreshCw,
  Rocket,
  ShieldOff,
  Sparkles,
  Wand2,
} from "lucide-react";
import {
  ApiError,
  cancelJob as apiCancelJob,
  compareLabEvals,
  createFewShot,
  createJob,
  encodeEpPath,
  fewShotTileUrl,
  getJob,
  getLabDiagnosis,
  getLabLabels,
  getLabPrompts,
  getLabTasks,
  getLabUnknowns,
  listFewShots,
  promoteLabConfig,
  putLabLabels,
  putLabPrompts,
  putLabTasks,
  requestPromptAssist,
  toggleFewShotEnabled,
  triageUnknown,
} from "../api/client";
import type {
  DiagnosisErrorItem,
  DiagnosisErrorType,
  DiagnosisReport,
  EvalCompareResponse,
  EvalRunReport,
  FewShotItem,
  JobStatus,
  LabelCategory,
  LabelRegistry,
  LabelStatus,
  PromptAssistResponse,
  PromptBundle,
  PromptStageKey,
  SplicePolicy,
  TaskSpec,
  TrainingRole,
  UnknownAnomalyItem,
  VisualLabelDef,
} from "../api/types";
import { useEventStream } from "../api/useEventStream";
import { LabHistoryDrawer } from "../components/LabHistoryDrawer";
import { DEFAULT_VISUAL_LABELS } from "../constants/labels";
import { useAuth } from "../stores/authStore";
import { useJobs } from "../stores/jobStore";
import { useToast } from "../stores/toastStore";

type LabTab = "diagnosis" | "few_shots" | "taxonomy" | "eval";
type TaxonomySubTab = "labels" | "prompts" | "tasks";

const DIAGNOSIS_BUCKETS: Array<{ key: DiagnosisErrorType; label: string }> = [
  { key: "fp_overflag", label: "FP Overflag" },
  { key: "fn_missed", label: "FN Missed" },
  { key: "policy_flip", label: "Policy / Overcut Flip" },
  { key: "boundary", label: "Boundary Drift" },
  { key: "success_flip", label: "Success Flip" },
  { key: "stage", label: "Macro Stage Mismatch" },
];

const PROMPT_STAGES: Array<{ key: PromptStageKey; label: string }> = [
  { key: "s1_grounding", label: "S1 · 2D Grounding" },
  { key: "s2_macro", label: "S2 · Macro Backbone" },
  { key: "s3_discover", label: "S3 · Sliding-Window QC Discovery" },
  { key: "s4_verify", label: "S4 · Start/End Frame Verification" },
  { key: "s5_success", label: "S5 · Tail Success Evaluation" },
];

const LABEL_CATEGORIES: LabelCategory[] = [
  "idle",
  "failure_recovery",
  "inefficiency",
  "failure",
  "necessary_skill",
  "terminal",
  "unknown",
];

function safeJsonStringify(val: unknown): string {
  if (typeof val === "string") return val;
  try {
    return JSON.stringify(val ?? {}, null, 2);
  } catch {
    return "{}";
  }
}

function parseJsonObject(text: string): Record<string, unknown> {
  const trimmed = text.trim();
  if (!trimmed) return {};
  const parsed = JSON.parse(trimmed);
  if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
    return parsed as Record<string, unknown>;
  }
  throw new Error("JSON must be an object");
}

function fmtPct(v: number | undefined | null): string {
  if (v === undefined || v === null || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function fmtNum(v: number | undefined | null, digits = 3): string {
  if (v === undefined || v === null || Number.isNaN(v)) return "—";
  return v.toFixed(digits);
}

function fmtDelta(
  v: number | undefined | null,
  opts: { pct?: boolean; lowerBetter?: boolean; digits?: number } = {},
): { text: string; cls: string } {
  if (v === undefined || v === null || Number.isNaN(v)) {
    return { text: "—", cls: "delta-neutral" };
  }
  const sign = v > 0 ? "+" : "";
  const text = opts.pct
    ? `${sign}${(v * 100).toFixed(1)}%`
    : `${sign}${v.toFixed(opts.digits ?? 3)}`;
  if (Math.abs(v) < 1e-6) return { text, cls: "delta-neutral" };
  const improved = opts.lowerBetter ? v < 0 : v > 0;
  return { text, cls: improved ? "delta-pos" : "delta-neg" };
}

function toJsonDict(val: unknown, fallback: Record<string, unknown>): Record<string, unknown> {
  if (val && typeof val === "object" && !Array.isArray(val)) {
    return val as Record<string, unknown>;
  }
  if (Array.isArray(val)) {
    return { items: val };
  }
  if (val !== null && val !== undefined) {
    return { value: val };
  }
  return fallback;
}

function fmtCompactDiagValue(val: unknown): string {
  if (val === null || val === undefined) return "null";
  if (typeof val === "string") return val;
  if (typeof val === "number" || typeof val === "boolean") return String(val);
  if (typeof val === "object" && !Array.isArray(val)) {
    const rec = val as Record<string, unknown>;
    if (typeof rec.label_id === "string") {
      const policy = typeof rec.splice_policy === "string" ? ` [${rec.splice_policy}]` : "";
      const span =
        typeof rec.t_start === "number" && typeof rec.t_end === "number"
          ? ` (${rec.t_start.toFixed(1)}–${rec.t_end.toFixed(1)}s)`
          : "";
      return `${rec.label_id}${policy}${span}`;
    }
    if (typeof rec.verdict === "string" || typeof rec.passed === "boolean") {
      const verdict = typeof rec.verdict === "string" ? rec.verdict : "";
      const passedTxt =
        rec.passed === true ? "pass" : rec.passed === false ? "fail" : rec.passed === null ? "unsure" : "";
      if (verdict && passedTxt) return `${verdict} (${passedTxt})`;
      return verdict || passedTxt || "—";
    }
  }
  try {
    const raw = JSON.stringify(val);
    return raw.length > 140 ? `${raw.slice(0, 137)}...` : raw;
  } catch {
    return String(val);
  }
}

interface StageLike {
  stage_id?: string;
  t_start?: number;
  t_end?: number;
  label?: string;
  label_en?: string;
}

function summarizeStageDiff(vlmVal: unknown, gtVal: unknown): string[] {
  const vlmArr: StageLike[] = Array.isArray(vlmVal) ? (vlmVal as StageLike[]) : [];
  const gtArr: StageLike[] = Array.isArray(gtVal) ? (gtVal as StageLike[]) : [];
  if (vlmArr.length === 0 && gtArr.length === 0) {
    return ["Macro stage structure differs"];
  }

  const lines: string[] = [];
  const maxLen = Math.max(vlmArr.length, gtArr.length);
  for (let i = 0; i < maxLen; i += 1) {
    const v = vlmArr[i];
    const g = gtArr[i];
    if (v && g) {
      const id = g.stage_id || v.stage_id || `S${i + 1}`;
      const vStart = typeof v.t_start === "number" ? v.t_start : null;
      const vEnd = typeof v.t_end === "number" ? v.t_end : null;
      const gStart = typeof g.t_start === "number" ? g.t_start : null;
      const gEnd = typeof g.t_end === "number" ? g.t_end : null;
      const boundaryChanged =
        vStart !== null &&
        vEnd !== null &&
        gStart !== null &&
        gEnd !== null &&
        (Math.abs(vStart - gStart) > 0.05 || Math.abs(vEnd - gEnd) > 0.05);
      const vName = v.label || v.label_en || "";
      const gName = g.label || g.label_en || "";
      const nameChanged = Boolean(vName || gName) && vName !== gName;

      if (boundaryChanged || nameChanged) {
        const parts: string[] = [];
        if (boundaryChanged && vStart !== null && vEnd !== null && gStart !== null && gEnd !== null) {
          parts.push(
            `${id} boundary shift: ${vStart.toFixed(1)}–${vEnd.toFixed(1)}s → ${gStart.toFixed(1)}–${gEnd.toFixed(1)}s`,
          );
        } else {
          parts.push(`${id}`);
        }
        if (nameChanged) {
          parts.push(`renamed (${vName || "—"} → ${gName || "—"})`);
        }
        lines.push(parts.join(" · "));
      }
    } else if (v && !g) {
      const id = v.stage_id || `S${i + 1}`;
      const span =
        typeof v.t_start === "number" && typeof v.t_end === "number"
          ? ` (${v.t_start.toFixed(1)}–${v.t_end.toFixed(1)}s)`
          : "";
      lines.push(`Extra VLM stage ${id}${span}`);
    } else if (!v && g) {
      const id = g.stage_id || `S${i + 1}`;
      const span =
        typeof g.t_start === "number" && typeof g.t_end === "number"
          ? ` (${g.t_start.toFixed(1)}–${g.t_end.toFixed(1)}s)`
          : "";
      lines.push(`Added GT stage ${id}${span}`);
    }
  }

  return lines.length > 0 ? lines : [`VLM ${vlmArr.length} stages → GT ${gtArr.length} stages`];
}

function buildDiagnosisReviewLink(err: DiagnosisErrorItem): string {
  const qs = new URLSearchParams();
  let tVal: number | null =
    typeof err.t_start === "number" && Number.isFinite(err.t_start) ? err.t_start : null;

  if (tVal === null && err.error_type === "stage") {
    const vlmArr: StageLike[] = Array.isArray(err.vlm_value) ? (err.vlm_value as StageLike[]) : [];
    const gtArr: StageLike[] = Array.isArray(err.gt_value) ? (err.gt_value as StageLike[]) : [];
    const maxLen = Math.max(vlmArr.length, gtArr.length);
    for (let i = 0; i < maxLen; i += 1) {
      const v = vlmArr[i];
      const g = gtArr[i];
      const changed =
        !v ||
        !g ||
        v.t_start !== g.t_start ||
        v.t_end !== g.t_end ||
        v.label !== g.label ||
        v.label_en !== g.label_en;
      if (changed) {
        const cand = g?.t_start ?? v?.t_start;
        if (typeof cand === "number" && Number.isFinite(cand)) {
          tVal = cand;
          break;
        }
      }
    }
  }

  if (tVal !== null) {
    qs.set("t", String(Number(tVal.toFixed(2))));
  }
  const segId = err.gt_seg_id || err.vlm_seg_id;
  if (segId) {
    qs.set("seg", segId);
  }
  qs.set("tab", "diff");
  return `/review/${encodeEpPath(err.episode_id)}?${qs.toString()}`;
}

function fmtCacheHits(
  stages: Record<string, number> | undefined | null,
  rate: number | undefined | null,
): string {
  const entries = Object.entries(stages ?? {}).filter(
    ([, count]) => typeof count === "number" && count > 0,
  );
  const totalHits = entries.reduce((sum, [, count]) => sum + count, 0);
  const breakdown = entries
    .map(([stage, count]) => `${stage.replace(/_.*$/, "")}:${count}`)
    .join(" ");
  if (breakdown) {
    return `${totalHits} (${breakdown}, ${fmtPct(rate)})`;
  }
  return `${totalHits} (${fmtPct(rate)})`;
}

interface TriageDraft {
  mode: "merge" | "promote";
  targetLabelId: string;
  rationale: string;
  newId: string;
  newName: string;
  newCategory: LabelCategory;
  newRole: TrainingRole;
  newPolicy: SplicePolicy;
  newUiTitle: string;
  newUiHint: string;
  newDefinition: string;
}

export const VlmLabPage: React.FC = () => {
  const { isLead, token, authProbing, setAuthPromptOpen } = useAuth();
  const jobStore = useJobs();
  const toast = useToast();
  const [activeTab, setActiveTab] = useState<LabTab>("diagnosis");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [historyOpen, setHistoryOpen] = useState(false);

  const showLabInfo = useCallback(
    (message: string, variant: "info" | "success" | "warning" | "error" = "info") => {
      toast.showToast({ message, variant, testId: "lab-info-banner" });
    },
    [toast],
  );

  const clearLabInfo = useCallback(() => {
    toast.dismissByTestId("lab-info-banner");
  }, [toast]);

  // Shared taxonomy / config state
  const [registry, setRegistry] = useState<LabelRegistry | null>(null);
  const [prompts, setPrompts] = useState<PromptBundle | null>(null);
  const [tasksMap, setTasksMap] = useState<Record<string, TaskSpec>>({});

  // Tab 1: Diagnosis + Unknown Triage
  const [diagEpFilter, setDiagEpFilter] = useState("");
  const [diagTopN, setDiagTopN] = useState(20);
  const [diagnosis, setDiagnosis] = useState<DiagnosisReport | null>(null);
  const [diagLoading, setDiagLoading] = useState(false);

  const [quickFsError, setQuickFsError] = useState<DiagnosisErrorItem | null>(null);
  const [quickFsScope, setQuickFsScope] = useState("global");
  const [quickFsRationale, setQuickFsRationale] = useState("");
  const [quickFsSaving, setQuickFsSaving] = useState(false);

  const [unkDatasetFilter, setUnkDatasetFilter] = useState("");
  const [unknowns, setUnknowns] = useState<UnknownAnomalyItem[]>([]);
  const [unkTotal, setUnkTotal] = useState(0);
  const [triageDrafts, setTriageDrafts] = useState<Record<string, TriageDraft>>({});
  const [triageBusyKey, setTriageBusyKey] = useState<string | null>(null);

  // Tab 2: Few-Shot Mistake Book
  const [fsScopeFilter, setFsScopeFilter] = useState("");
  const [fsTaskFilter, setFsTaskFilter] = useState("");
  const [fsLabelFilter, setFsLabelFilter] = useState("");
  const [fsEnabledOnly, setFsEnabledOnly] = useState(false);
  const [fewShots, setFewShots] = useState<FewShotItem[]>([]);
  const [fsLoading, setFsLoading] = useState(false);

  const [manualFsOpen, setManualFsOpen] = useState(false);
  const [manualFsEpisode, setManualFsEpisode] = useState("");
  const [manualFsStart, setManualFsStart] = useState("0.0");
  const [manualFsEnd, setManualFsEnd] = useState("1.5");
  const [manualFsScope, setManualFsScope] = useState("global");
  const [manualFsTask, setManualFsTask] = useState("");
  const [manualFsLabel, setManualFsLabel] = useState("");
  const [manualFsErrorType, setManualFsErrorType] = useState<DiagnosisErrorType>("fp_overflag");
  const [manualFsWrong, setManualFsWrong] = useState('{"label_id": "hesitation_oscillation"}');
  const [manualFsRight, setManualFsRight] = useState('{"label_id": "precision_fine_adjust"}');
  const [manualFsRationale, setManualFsRationale] = useState("");
  const [manualFsCaptureTile, setManualFsCaptureTile] = useState(true);

  // Tab 3: Taxonomy / Prompts / TaskSpecs
  const [taxSubTab, setTaxSubTab] = useState<TaxonomySubTab>("labels");
  const [selectedLabelId, setSelectedLabelId] = useState<string>("boundary_idle");
  const [newLabelOpen, setNewLabelOpen] = useState(false);
  const [newLabelDraft, setNewLabelDraft] = useState<VisualLabelDef>({
    id: "",
    name: "",
    category: "failure",
    status: "active",
    version: 1,
    default_training_role: "badcase",
    default_splice_policy: "mask_loss",
    collector_ui: { title: "", hint: "" },
    vlm_semantics: { definition: "", counter_examples: [], visual_cues: [] },
  });

  const [selectedStage, setSelectedStage] = useState<PromptStageKey>("s3_discover");
  const [assistTaskId, setAssistTaskId] = useState("");
  const [assistTopN, setAssistTopN] = useState(10);
  const [assistLoading, setAssistLoading] = useState(false);
  const [assistPreview, setAssistPreview] = useState<PromptAssistResponse | null>(null);

  const [selectedTaskId, setSelectedTaskId] = useState<string>("_default");
  const [savingConfig, setSavingConfig] = useState(false);

  // Tab 4: GT Eval Sandbox & Promote
  const [evalDataset, setEvalDataset] = useState("");
  const [evalEpisodeIds, setEvalEpisodeIds] = useState("");
  const [evalFewShot, setEvalFewShot] = useState(true);
  const [evalTopK, setEvalTopK] = useState(4);
  const [evalMaxTokens, setEvalMaxTokens] = useState(4000);
  const [evalRunning, setEvalRunning] = useState(false);
  const [latestEval, setLatestEval] = useState<EvalRunReport | null>(null);
  const [compareData, setCompareData] = useState<EvalCompareResponse | null>(null);
  const [baselineId, setBaselineId] = useState("");
  const [candidateId, setCandidateId] = useState("");
  const [promoteReason, setPromoteReason] = useState("lab_gt_eval_verified");
  const [promoting, setPromoting] = useState(false);
  const [localEvalJob, setLocalEvalJob] = useState<JobStatus | null>(null);
  const handledEvalJobIdsRef = useRef<Set<string>>(new Set());

  const labelsList = useMemo(
    () => registry?.labels ?? DEFAULT_VISUAL_LABELS,
    [registry],
  );

  const loadLabels = useCallback(async () => {
    const reg = await getLabLabels();
    setRegistry(reg);
    if (reg.labels.length > 0) {
      setSelectedLabelId((prev) =>
        reg.labels.some((l) => l.id === prev) ? prev : reg.labels[0].id,
      );
    }
  }, []);

  const loadPrompts = useCallback(async () => {
    const bundle = await getLabPrompts();
    setPrompts(bundle);
  }, []);

  const loadTasks = useCallback(async () => {
    const res = await getLabTasks();
    setTasksMap(res.tasks ?? {});
    const keys = Object.keys(res.tasks ?? {});
    if (keys.length > 0) {
      setSelectedTaskId((prev) => (res.tasks[prev] ? prev : keys[0]));
    }
  }, []);

  const loadDiagnosis = useCallback(async () => {
    setDiagLoading(true);
    try {
      const epList = diagEpFilter
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
      const rep = await getLabDiagnosis({
        episode_ids: epList.length > 0 ? epList : undefined,
        top_n: diagTopN,
      });
      setDiagnosis(rep);
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) setAuthPromptOpen(true);
      setErrorMsg(err instanceof Error ? err.message : "Failed to load diagnosis report");
    } finally {
      setDiagLoading(false);
    }
  }, [diagEpFilter, diagTopN, setAuthPromptOpen]);

  const loadUnknowns = useCallback(async () => {
    try {
      const res = await getLabUnknowns({
        dataset: unkDatasetFilter.trim() || undefined,
      });
      setUnknowns(res.items);
      setUnkTotal(res.total);
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) setAuthPromptOpen(true);
      setErrorMsg(err instanceof Error ? err.message : "Failed to load unknown anomaly pool");
    }
  }, [unkDatasetFilter, setAuthPromptOpen]);

  const loadFewShots = useCallback(async () => {
    setFsLoading(true);
    try {
      const res = await listFewShots({
        scope: fsScopeFilter.trim() || undefined,
        task_id: fsTaskFilter.trim() || undefined,
        label_id: fsLabelFilter.trim() || undefined,
        enabled_only: fsEnabledOnly ? true : undefined,
      });
      setFewShots(res.items);
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) setAuthPromptOpen(true);
      setErrorMsg(err instanceof Error ? err.message : "Failed to load Few-Shot Mistake Book");
    } finally {
      setFsLoading(false);
    }
  }, [fsScopeFilter, fsTaskFilter, fsLabelFilter, fsEnabledOnly, setAuthPromptOpen]);

  const loadEvalCompare = useCallback(
    async (base?: string, cand?: string) => {
      try {
        const res = await compareLabEvals({
          baseline: base || undefined,
          candidate: cand || undefined,
        });
        setCompareData(res);
        if (res.baseline?.eval_id && !base) setBaselineId(res.baseline.eval_id);
        if (res.candidate?.eval_id && !cand) setCandidateId(res.candidate.eval_id);
        if (!res.candidate && res.evals.length > 0 && !cand) {
          setCandidateId(res.evals[res.evals.length - 1].eval_id);
        }
      } catch (err) {
        setErrorMsg(err instanceof Error ? err.message : "Failed to load evaluation comparison");
      }
    },
    [],
  );

  useEffect(() => {
    if (!isLead || authProbing) return;
    const onErr = (err: unknown) => {
      if (err instanceof ApiError && err.status === 401) setAuthPromptOpen(true);
    };
    void loadLabels().catch(onErr);
    void loadPrompts().catch(onErr);
    void loadTasks().catch(onErr);
    void loadDiagnosis();
    void loadUnknowns();
    // Initial mount / credential load only; filter fields refresh via explicit buttons
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isLead, authProbing, token]);

  useEffect(() => {
    if (!isLead || authProbing) return;
    if (activeTab === "few_shots") {
      void loadFewShots();
    } else if (activeTab === "eval" && !compareData) {
      void loadEvalCompare();
    }
  }, [isLead, authProbing, activeTab, loadFewShots, loadEvalCompare, compareData]);

  const activeEvalJob = useMemo(() => {
    if (!localEvalJob) return null;
    if (jobStore.isProvided) {
      return jobStore.jobs.find((j) => j.job_id === localEvalJob.job_id) ?? localEvalJob;
    }
    return localEvalJob;
  }, [localEvalJob, jobStore.isProvided, jobStore.jobs]);

  const finalizeEvalJob = useCallback(
    async (job: JobStatus) => {
      if (handledEvalJobIdsRef.current.has(job.job_id)) return;
      if (job.status !== "completed" && job.status !== "failed" && job.status !== "cancelled") {
        return;
      }
      handledEvalJobIdsRef.current.add(job.job_id);
      setLocalEvalJob(job);
      setEvalRunning(false);

      if (job.status === "failed") {
        const errText = job.summary?.errors?.[0]?.error || "GT regression evaluation failed";
        setErrorMsg(errText);
        return;
      }
      if (job.status === "cancelled") {
        showLabInfo(`Evaluation job ${job.job_id} cancelled`, "warning");
        return;
      }

      const summary = job.summary ?? {};
      const inlineReport =
        ((summary.report ?? summary.eval_report) as EvalRunReport | undefined) ?? null;
      const evalRunId =
        inlineReport?.eval_id ??
        (typeof summary.eval_run_id === "string" ? summary.eval_run_id : null) ??
        (typeof summary.eval_id === "string" ? summary.eval_id : null);

      let resolvedReport: EvalRunReport | null = inlineReport;
      try {
        const cmp = await compareLabEvals({
          baseline: baselineId || undefined,
          candidate: evalRunId || undefined,
        });
        setCompareData(cmp);
        if (cmp.baseline?.eval_id && !baselineId) setBaselineId(cmp.baseline.eval_id);
        const candFromCmp = cmp.candidate ?? (evalRunId ? cmp.evals.find((e) => e.eval_id === evalRunId) : cmp.evals[cmp.evals.length - 1]) ?? null;
        if (candFromCmp) {
          resolvedReport = candFromCmp;
        }
      } catch (err) {
        setErrorMsg(err instanceof Error ? err.message : "Failed to load evaluation comparison");
      }

      if (resolvedReport) {
        setLatestEval(resolvedReport);
        setCandidateId(resolvedReport.eval_id);
        const epCount =
          resolvedReport.metrics?.episodes_evaluated ?? resolvedReport.per_episode.length;
        showLabInfo(
          `Evaluation completed: ${resolvedReport.eval_id} (${epCount} GT episodes, S1/S2/S5 cache hit rate ${fmtPct(resolvedReport.metrics.cache_hit_rate)})`,
          "success",
        );
      } else {
        if (evalRunId) setCandidateId(evalRunId);
        showLabInfo(`Evaluation job completed: ${evalRunId ?? job.job_id}`, "success");
      }
    },
    [baselineId, showLabInfo],
  );

  useEffect(() => {
    if (!activeEvalJob) return;
    if (
      activeEvalJob.status === "completed" ||
      activeEvalJob.status === "failed" ||
      activeEvalJob.status === "cancelled"
    ) {
      void finalizeEvalJob(activeEvalJob);
    }
  }, [activeEvalJob, finalizeEvalJob]);

  useEventStream<JobStatus>({
    url:
      localEvalJob && !jobStore.isProvided
        ? `/api/jobs/${encodeURIComponent(localEvalJob.job_id)}/events`
        : null,
    enabled:
      Boolean(localEvalJob) &&
      !jobStore.isProvided &&
      (localEvalJob?.status === "queued" || localEvalJob?.status === "running"),
    isTerminal: (j) => j.status === "completed" || j.status === "failed" || j.status === "cancelled",
    onMessage: (jobEvt) => {
      setLocalEvalJob(jobEvt);
      if (jobEvt.status === "completed" || jobEvt.status === "failed" || jobEvt.status === "cancelled") {
        void finalizeEvalJob(jobEvt);
      }
    },
  });

  useEffect(() => {
    if (!activeEvalJob) return;
    if (activeEvalJob.status !== "queued" && activeEvalJob.status !== "running") return;

    let stopped = false;
    const jobId = activeEvalJob.job_id;
    const timer = setInterval(() => {
      if (stopped) return;
      void getJob(jobId)
        .then((latest) => {
          if (stopped) return;
          setLocalEvalJob(latest);
          if (jobStore.isProvided) {
            jobStore.trackJob(latest);
          }
          if (
            latest.status === "completed" ||
            latest.status === "failed" ||
            latest.status === "cancelled"
          ) {
            void finalizeEvalJob(latest);
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
  }, [activeEvalJob, finalizeEvalJob, jobStore]);

  if (authProbing) {
    return <div className="page-container">Verifying permissions...</div>;
  }

  if (!isLead) {
    return (
      <div className="page-container" data-testid="lab-forbidden">
        <div className="forbidden-card">
          <ShieldOff size={32} />
          <h2>403 — Forbidden</h2>
          <p>VLM Lab is restricted to Lead users. Reviewers should return to the Work Queue.</p>
        </div>
      </div>
    );
  }

  // Helpers for Unknown Triage drafts
  const getTriageDraft = (item: UnknownAnomalyItem): TriageDraft => {
    const key = `${item.episode_id}::${item.seg_id}`;
    return (
      triageDrafts[key] ?? {
        mode: "merge",
        targetLabelId:
          labelsList.find((l) => l.id !== "unknown_anomaly")?.id ?? "hesitation_oscillation",
        rationale: item.unknown_note || "Merge into existing visual label",
        newId: "",
        newName: "",
        newCategory: "failure",
        newRole: "badcase",
        newPolicy: "mask_loss",
        newUiTitle: "",
        newUiHint: "",
        newDefinition: item.unknown_note || "",
      }
    );
  };

  const patchTriageDraft = (item: UnknownAnomalyItem, patch: Partial<TriageDraft>) => {
    const key = `${item.episode_id}::${item.seg_id}`;
    const current = getTriageDraft(item);
    setTriageDrafts((prev) => ({ ...prev, [key]: { ...current, ...patch } }));
  };

  const handleTriageSubmit = async (item: UnknownAnomalyItem) => {
    const key = `${item.episode_id}::${item.seg_id}`;
    const draft = getTriageDraft(item);
    setTriageBusyKey(key);
    setErrorMsg(null);
    clearLabInfo();
    try {
      if (draft.mode === "merge") {
        const res = await triageUnknown({
          episode_id: item.episode_id,
          seg_id: item.seg_id,
          action: "merge",
          target_label_id: draft.targetLabelId,
          rationale: draft.rationale.trim() || item.unknown_note || "Merge unknown anomaly",
        });
        showLabInfo(
          `Merged ${item.seg_id} into ${draft.targetLabelId} and recorded Few-Shot (${res.few_shot_id ?? "ok"})`,
          "success",
        );
      } else {
        if (!draft.newId.trim() || !draft.newName.trim()) {
          throw new Error("Promoting a new label requires both a label ID and a display name");
        }
        const newLabel: VisualLabelDef = {
          id: draft.newId.trim(),
          name: draft.newName.trim(),
          category: draft.newCategory,
          status: "active",
          version: 1,
          default_training_role: draft.newRole,
          default_splice_policy: draft.newPolicy,
          collector_ui: {
            title: draft.newUiTitle.trim() || draft.newName.trim(),
            hint: draft.newUiHint.trim() || draft.newDefinition.trim(),
          },
          vlm_semantics: {
            definition: draft.newDefinition.trim() || draft.newName.trim(),
            counter_examples: [],
            visual_cues: [],
          },
        };
        const res = await triageUnknown({
          episode_id: item.episode_id,
          seg_id: item.seg_id,
          action: "promote",
          rationale: draft.rationale.trim() || undefined,
          new_label: newLabel,
        });
        showLabInfo(
          `Promoted unknown anomaly to new label ${newLabel.id} (registry v${res.label_version ?? "updated"}, archived to .history/)`,
          "success",
        );
      }
      await Promise.all([loadUnknowns(), loadLabels()]);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Failed to triage unknown anomaly");
    } finally {
      setTriageBusyKey(null);
    }
  };

  // Quick Few-Shot from Diagnosis row
  const openQuickFewShot = (err: DiagnosisErrorItem) => {
    setQuickFsError(err);
    setQuickFsScope(err.task_id ? `task:${err.task_id}` : "global");
    setQuickFsRationale(
      err.summary || `Visual criterion correction for ${err.error_type} (${err.label_id ?? "task"})`,
    );
  };

  const submitQuickFewShot = async () => {
    if (!quickFsError) return;
    if (!quickFsRationale.trim()) {
      setErrorMsg("Please provide a Few-Shot rationale");
      return;
    }
    setQuickFsSaving(true);
    setErrorMsg(null);
    clearLabInfo();
    try {
      const tStart = Math.max(0, quickFsError.t_start ?? 0);
      const rawEnd = quickFsError.t_end ?? tStart + 1.0;
      const tEnd = rawEnd > tStart ? rawEnd : tStart + 0.5;
      const created = await createFewShot({
        scope: quickFsScope,
        task_id: quickFsScope.startsWith("task:")
          ? quickFsScope.slice(5)
          : quickFsError.task_id ?? null,
        label_id: quickFsScope.startsWith("label:")
          ? quickFsScope.slice(6)
          : quickFsError.label_id ?? null,
        error_type: quickFsError.error_type,
        episode_id: quickFsError.episode_id,
        t_start: tStart,
        t_end: tEnd,
        vlm_wrong_json: toJsonDict(quickFsError.vlm_value, {
          raw: quickFsError.error_type,
        }),
        human_right_json: toJsonDict(quickFsError.gt_value, {
          corrected: "human_gt",
        }),
        rationale: quickFsRationale.trim(),
        capture_tile: true,
      });
      showLabInfo(`Added to Few-Shot Mistake Book: ${created.example_id}`, "success");
      setQuickFsError(null);
      setQuickFsRationale("");
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Failed to add Few-Shot example");
    } finally {
      setQuickFsSaving(false);
    }
  };

  // Manual Few-Shot creation
  const handleManualCreateFewShot = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMsg(null);
    clearLabInfo();
    try {
      const tStart = Number(manualFsStart);
      const tEnd = Number(manualFsEnd);
      if (!manualFsEpisode.trim()) throw new Error("Please enter episode_id");
      if (!Number.isFinite(tStart) || !Number.isFinite(tEnd) || tEnd <= tStart) {
        throw new Error("Invalid time span: t_end must be greater than t_start");
      }
      if (!manualFsRationale.trim()) throw new Error("Please enter a rationale");
      const vlmWrong = parseJsonObject(manualFsWrong);
      const humanRight = parseJsonObject(manualFsRight);
      const created = await createFewShot({
        scope: manualFsScope.trim() || "global",
        task_id: manualFsTask.trim() || null,
        label_id: manualFsLabel.trim() || null,
        error_type: manualFsErrorType,
        episode_id: manualFsEpisode.trim(),
        t_start: tStart,
        t_end: tEnd,
        vlm_wrong_json: vlmWrong,
        human_right_json: humanRight,
        rationale: manualFsRationale.trim(),
        capture_tile: manualFsCaptureTile,
      });
      showLabInfo(`Created Few-Shot example: ${created.example_id}`, "success");
      setManualFsOpen(false);
      await loadFewShots();
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Failed to create Few-Shot example");
    }
  };

  const handleToggleFewShot = async (item: FewShotItem) => {
    setErrorMsg(null);
    try {
      const updated = await toggleFewShotEnabled(item.example_id, !item.enabled);
      setFewShots((prev) =>
        prev.map((x) => (x.example_id === item.example_id ? updated : x)),
      );
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Failed to toggle Few-Shot status");
    }
  };

  // Taxonomy Editors
  const selectedLabel = labelsList.find((l) => l.id === selectedLabelId) ?? labelsList[0] ?? null;

  const updateSelectedLabel = (patch: Partial<VisualLabelDef>) => {
    if (!selectedLabel) return;
    const baseReg: LabelRegistry = registry ?? {
      schema_version: "1.0",
      version: 1,
      categories: LABEL_CATEGORIES,
      labels: DEFAULT_VISUAL_LABELS,
    };
    setRegistry({
      ...baseReg,
      labels: baseReg.labels.map((l) =>
        l.id === selectedLabel.id ? { ...l, ...patch } : l,
      ),
    });
  };

  const handleAddNewLabel = () => {
    if (!newLabelDraft.id.trim() || !newLabelDraft.name.trim()) {
      setErrorMsg("New label requires both an ID and a display name");
      return;
    }
    const baseReg: LabelRegistry = registry ?? {
      schema_version: "1.0",
      version: 1,
      categories: LABEL_CATEGORIES,
      labels: DEFAULT_VISUAL_LABELS,
    };
    if (baseReg.labels.some((l) => l.id === newLabelDraft.id.trim())) {
      setErrorMsg(`Label ID "${newLabelDraft.id.trim()}" already exists`);
      return;
    }
    const added: VisualLabelDef = {
      ...newLabelDraft,
      id: newLabelDraft.id.trim(),
      name: newLabelDraft.name.trim(),
    };
    setRegistry({
      ...baseReg,
      version: baseReg.version + 1,
      labels: [...baseReg.labels, added],
    });
    setSelectedLabelId(added.id);
    setNewLabelOpen(false);
    showLabInfo(`Added label draft ${added.id} (click "Save Label Registry" to persist)`);
  };

  const handleSaveLabels = async () => {
    if (!registry) return;
    setSavingConfig(true);
    setErrorMsg(null);
    clearLabInfo();
    try {
      const saved = await putLabLabels(registry);
      setRegistry(saved);
      showLabInfo(`Label registry saved (v${saved.version}); previous version archived to .history/`, "success");
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Failed to save label registry");
    } finally {
      setSavingConfig(false);
    }
  };

  const activePromptTpl = prompts ? prompts[selectedStage] : null;

  const updateActivePrompt = (patch: Record<string, string>) => {
    if (!prompts || !activePromptTpl) return;
    setPrompts({
      ...prompts,
      [selectedStage]: {
        ...activePromptTpl,
        ...patch,
      },
    });
  };

  const handleSavePrompts = async () => {
    if (!prompts) return;
    setSavingConfig(true);
    setErrorMsg(null);
    clearLabInfo();
    try {
      const saved = await putLabPrompts(prompts);
      setPrompts(saved);
      showLabInfo(
        `Prompt bundle saved (${selectedStage} v${saved[selectedStage].version}); previous version archived to .history/`,
        "success",
      );
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Failed to save prompt bundle");
    } finally {
      setSavingConfig(false);
    }
  };

  const handleRequestPromptAssist = async () => {
    setAssistLoading(true);
    setErrorMsg(null);
    try {
      const res = await requestPromptAssist({
        stage: selectedStage,
        task_id: assistTaskId.trim() || undefined,
        top_n: assistTopN,
      });
      setAssistPreview(res);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Failed to generate prompt revision suggestions");
    } finally {
      setAssistLoading(false);
    }
  };

  const applyAssistDraftLocally = () => {
    if (!assistPreview || !prompts || !activePromptTpl) return;
    const patch: Record<string, string> = {};
    const sysPatch = assistPreview.proposed_system_template_patch?.trim();
    const userPatch = assistPreview.proposed_user_template_patch?.trim();
    if (sysPatch) {
      patch.system_template = sysPatch;
    }
    if (userPatch) {
      patch.user_template = userPatch;
    }
    if (!patch.system_template && !patch.user_template && assistPreview.suggested_prompt_diff) {
      patch.user_template = `${activePromptTpl.user_template}\n\n# Suggested revision reference:\n${assistPreview.suggested_prompt_diff}`;
    }
    updateActivePrompt(patch);
    showLabInfo("Applied AI suggestion to local draft (not saved yet; click Save when ready)");
  };

  const activeTaskSpec = tasksMap[selectedTaskId] ?? null;

  const updateActiveTaskSpec = (patch: Partial<TaskSpec>) => {
    if (!activeTaskSpec) return;
    setTasksMap((prev) => ({
      ...prev,
      [selectedTaskId]: { ...activeTaskSpec, ...patch },
    }));
  };

  const handleSaveTaskSpec = async () => {
    if (!activeTaskSpec) return;
    setSavingConfig(true);
    setErrorMsg(null);
    clearLabInfo();
    try {
      const res = await putLabTasks(activeTaskSpec);
      setTasksMap(res.tasks);
      showLabInfo(
        `TaskSpec "${activeTaskSpec.task_id}" (v${activeTaskSpec.version}) saved and backed up to .history/`,
        "success",
      );
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Failed to save TaskSpec");
    } finally {
      setSavingConfig(false);
    }
  };

  // Eval Sandbox & Promote (Async Job)
  const handleRunEval = async () => {
    setEvalRunning(true);
    setErrorMsg(null);
    clearLabInfo();
    try {
      const epIds = evalEpisodeIds
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
      const job = await createJob({
        job_type: "eval_gt",
        episode_ids: epIds,
        eval_params: {
          dataset: evalDataset.trim() || undefined,
          episode_ids: epIds.length > 0 ? epIds : undefined,
          few_shot: evalFewShot,
          top_k: evalTopK,
          max_tokens: evalMaxTokens,
        },
      });
      setLocalEvalJob(job);
      jobStore.trackJob(job);
      if (job.status === "completed" || job.status === "failed" || job.status === "cancelled") {
        await finalizeEvalJob(job);
      } else {
        showLabInfo(`Submitted GT regression evaluation job ${job.job_id}`);
      }
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Failed to submit GT regression eval job");
      setEvalRunning(false);
    }
  };

  const handleCancelEvalJob = async (jobId: string) => {
    const updated = jobStore.isProvided
      ? await jobStore.cancelJob(jobId)
      : await apiCancelJob(jobId).catch(() => null);
    if (updated) {
      await finalizeEvalJob(updated);
    }
  };

  const handleCompareEvals = async () => {
    setErrorMsg(null);
    await loadEvalCompare(baselineId || undefined, candidateId || undefined);
  };

  const handlePromote = async () => {
    const targetEvalId = candidateId || latestEval?.eval_id;
    setPromoting(true);
    setErrorMsg(null);
    clearLabInfo();
    try {
      const res = await promoteLabConfig({
        eval_id: targetEvalId || undefined,
        dataset: evalDataset.trim() || undefined,
        reason: promoteReason.trim() || "lab_promote",
      });
      showLabInfo(
        `Promoted config to active workspace (${res.promoted_files.join(", ") || "configs_override"}); frozen GT and historical benchmarks remain untouched.`,
        "success",
      );
      await Promise.all([loadLabels(), loadPrompts(), loadTasks()]);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Failed to promote config");
    } finally {
      setPromoting(false);
    }
  };

  const errorItems = diagnosis?.top_errors?.length ? diagnosis.top_errors : diagnosis?.errors ?? [];
  const baseEval = compareData?.baseline ?? null;
  const candEval = compareData?.candidate ?? latestEval ?? null;
  const deltas = compareData?.deltas ?? null;

  return (
    <div className="page-container" data-testid="vlm-lab-page">
      <div className="page-header">
        <div>
          <h1>VLM Lab Closed-Loop Iteration Workbench (Lead)</h1>
          <p className="subtitle">
            Pure In-Context Closed-Loop Evolution: Error Diagnosis &amp; Unknown Triage → 2×2 Visual
            Few-Shot Mistake Book → Versioned Labels / Prompts / TaskSpecs → GT Regression Eval
            Sandbox &amp; One-Click Promote
          </p>
        </div>
      </div>

      {errorMsg && (
        <div className="error-banner" data-testid="lab-error-banner">
          <span>{errorMsg}</span>
          <button type="button" className="mini-btn" onClick={() => setErrorMsg(null)}>
            Dismiss
          </button>
        </div>
      )}

      <div className="tab-group">
        <button
          type="button"
          data-testid="tab-diagnosis"
          className={`tab-btn ${activeTab === "diagnosis" ? "active" : ""}`}
          onClick={() => setActiveTab("diagnosis")}
        >
          <Sparkles size={14} /> Diagnosis &amp; Unknown Triage
        </button>
        <button
          type="button"
          data-testid="tab-few-shots"
          className={`tab-btn ${activeTab === "few_shots" ? "active" : ""}`}
          onClick={() => setActiveTab("few_shots")}
        >
          <Bookmark size={14} /> Few-Shot Mistake Book
        </button>
        <button
          type="button"
          data-testid="tab-taxonomy"
          className={`tab-btn ${activeTab === "taxonomy" ? "active" : ""}`}
          onClick={() => setActiveTab("taxonomy")}
        >
          <Layers size={14} /> Labels / Prompts / TaskSpecs
        </button>
        <button
          type="button"
          data-testid="tab-eval"
          className={`tab-btn ${activeTab === "eval" ? "active" : ""}`}
          onClick={() => setActiveTab("eval")}
        >
          <FlaskConical size={14} /> GT Eval Sandbox
        </button>
      </div>

      {/* TAB 1: DIAGNOSIS & UNKNOWN TRIAGE */}
      {activeTab === "diagnosis" && (
        <div className="page-container" data-testid="lab-panel-diagnosis">
          <div className="panel-card">
            <div className="section-header-row">
              <h3>VLM Raw vs Human GT Fine-Grained Error Diagnosis</h3>
              <div className="filter-toolbar">
                <input
                  type="text"
                  aria-label="Filter episode_ids"
                  placeholder="episode_ids (comma-separated, optional)"
                  value={diagEpFilter}
                  onChange={(e) => setDiagEpFilter(e.target.value)}
                />
                <input
                  type="number"
                  aria-label="Top-N"
                  min={1}
                  max={200}
                  style={{ width: 84 }}
                  value={diagTopN}
                  onChange={(e) => setDiagTopN(Math.max(1, Number(e.target.value) || 20))}
                />
                <button
                  type="button"
                  className="primary-btn compact"
                  data-testid="refresh-diagnosis-btn"
                  disabled={diagLoading}
                  onClick={() => void loadDiagnosis()}
                >
                  <RefreshCw size={13} /> {diagLoading ? "Diagnosing..." : "Refresh Diagnosis"}
                </button>
              </div>
            </div>

            <div className="lab-kpi-strip" data-testid="diagnosis-kpi-strip">
              <div className="lab-kpi-card">
                <span className="lab-kpi-val">{diagnosis?.episodes_evaluated ?? 0}</span>
                <span className="lab-kpi-lbl">Evaluated GT Episodes</span>
              </div>
              <div className="lab-kpi-card">
                <span className="lab-kpi-val">{diagnosis?.total_errors ?? 0}</span>
                <span className="lab-kpi-lbl">Total Errors</span>
              </div>
              {DIAGNOSIS_BUCKETS.map((b) => (
                <div key={b.key} className="lab-kpi-card" data-testid={`kpi-bucket-${b.key}`}>
                  <span className="lab-kpi-val">{diagnosis?.counts_by_type?.[b.key] ?? 0}</span>
                  <span className="lab-kpi-lbl">
                    {b.label} ({b.key})
                  </span>
                </div>
              ))}
            </div>

            {quickFsError && (
              <div className="lab-card" data-testid="quick-fewshot-form">
                <div className="section-header-row">
                  <strong>
                    Quick Add Few-Shot Example: {quickFsError.episode_id} ({quickFsError.error_type})
                  </strong>
                  <button
                    type="button"
                    className="mini-btn"
                    onClick={() => setQuickFsError(null)}
                  >
                    Cancel
                  </button>
                </div>
                <div className="two-col">
                  <label className="field-group">
                    <span>Scope (scope)</span>
                    <select value={quickFsScope} onChange={(e) => setQuickFsScope(e.target.value)}>
                      <option value="global">Global (global)</option>
                      {quickFsError.task_id && (
                        <option value={`task:${quickFsError.task_id}`}>
                          Task (task:{quickFsError.task_id})
                        </option>
                      )}
                      {quickFsError.label_id && (
                        <option value={`label:${quickFsError.label_id}`}>
                          Label (label:{quickFsError.label_id})
                        </option>
                      )}
                    </select>
                  </label>
                  <label className="field-group">
                    <span>Time Span</span>
                    <input
                      type="text"
                      readOnly
                      value={`${(quickFsError.t_start ?? 0).toFixed(2)}s - ${(quickFsError.t_end ?? (quickFsError.t_start ?? 0) + 1.0).toFixed(2)}s`}
                    />
                  </label>
                </div>
                <label className="field-group">
                  <span>Rationale (rationale)</span>
                  <textarea
                    rows={2}
                    value={quickFsRationale}
                    onChange={(e) => setQuickFsRationale(e.target.value)}
                    placeholder="Explain the visual criterion and pitfall..."
                  />
                </label>
                <div>
                  <button
                    type="button"
                    className="accent-btn compact"
                    data-testid="submit-quick-fewshot-btn"
                    disabled={quickFsSaving}
                    onClick={() => void submitQuickFewShot()}
                  >
                    {quickFsSaving ? "Capturing & Saving..." : "Confirm & Save (with 2×2 Tile)"}
                  </button>
                </div>
              </div>
            )}

            <div className="table-wrap">
              <table className="data-table" data-testid="diagnosis-errors-table">
                <thead>
                  <tr>
                    <th>Episode</th>
                    <th>Error Type</th>
                    <th>Label / Segment</th>
                    <th>Time Span</th>
                    <th>VLM vs GT Details</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {errorItems.length === 0 ? (
                    <tr>
                      <td colSpan={6} className="empty-cell">
                        No diagnosis errors yet (ensure episodes have human reviews or frozen GT).
                      </td>
                    </tr>
                  ) : (
                    errorItems.map((err, idx) => (
                      <tr key={err.error_id || `${err.episode_id}-${err.vlm_seg_id ?? err.gt_seg_id ?? idx}-${err.error_type}`}>
                        <td className="mono small">
                          <Link
                            className="link-btn"
                            to={buildDiagnosisReviewLink(err)}
                          >
                            {err.episode_id}
                          </Link>
                          {err.task_id && <div className="muted">{err.task_id}</div>}
                        </td>
                        <td>
                          <span className="badge-pill warn">{err.error_type}</span>
                        </td>
                        <td className="mono small">
                          {err.label_id ?? "—"}
                          {err.iou !== null && err.iou !== undefined && (
                            <span className="muted"> (IoU {err.iou.toFixed(2)})</span>
                          )}
                          {(err.vlm_seg_id || err.gt_seg_id) && (
                            <div className="muted">
                              {[err.vlm_seg_id && `VLM:${err.vlm_seg_id}`, err.gt_seg_id && `GT:${err.gt_seg_id}`]
                                .filter(Boolean)
                                .join(" · ")}
                            </div>
                          )}
                        </td>
                        <td className="mono small">
                          {err.t_start !== null && err.t_start !== undefined && err.t_end !== null && err.t_end !== undefined
                            ? `${err.t_start.toFixed(2)}s–${err.t_end.toFixed(2)}s`
                            : "Global"}
                        </td>
                        <td className="small">
                          <div>{err.summary || "VLM prediction diverges from Human GT"}</div>
                          {(err.vlm_value !== undefined || err.gt_value !== undefined) && (
                            err.error_type === "stage" ? (
                              <div className="diag-stage-diff">
                                {summarizeStageDiff(err.vlm_value, err.gt_value).map((line, lineIdx) => (
                                  <div key={lineIdx} className="diag-stage-item">
                                    {line}
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <div className="muted mono">
                                VLM: {fmtCompactDiagValue(err.vlm_value)} → GT: {fmtCompactDiagValue(err.gt_value)}
                              </div>
                            )
                          )}
                          {(err.vlm_value !== undefined || err.gt_value !== undefined) && (
                            <details className="diag-raw-json">
                              <summary>View raw JSON</summary>
                              <pre className="json-box">
                                {safeJsonStringify({
                                  vlm: err.vlm_value ?? null,
                                  gt: err.gt_value ?? null,
                                })}
                              </pre>
                            </details>
                          )}
                        </td>
                        <td>
                          <button
                            type="button"
                            className="mini-btn accent"
                            data-testid={`quick-fewshot-btn-${idx}`}
                            onClick={() => openQuickFewShot(err)}
                          >
                            <Plus size={12} /> Add Few-Shot
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>

          <div className="panel-card" data-testid="unknown-triage-panel">
            <div className="section-header-row">
              <h3>Unknown Anomaly Triage Pool (`unknown_anomaly` · {unkTotal} pending)</h3>
              <div className="filter-toolbar">
                <input
                  type="text"
                  aria-label="Filter unknown anomaly dataset"
                  placeholder="Filter dataset (optional)"
                  value={unkDatasetFilter}
                  onChange={(e) => setUnkDatasetFilter(e.target.value)}
                />
                <button
                  type="button"
                  className="secondary-btn compact"
                  data-testid="refresh-unknowns-btn"
                  onClick={() => void loadUnknowns()}
                >
                  <RefreshCw size={13} /> Refresh Pool
                </button>
              </div>
            </div>

            {unknowns.length === 0 ? (
              <div className="muted">No pending unknown_anomaly segments to triage.</div>
            ) : (
              <div className="lab-card-grid">
                {unknowns.map((item) => {
                  const key = `${item.episode_id}::${item.seg_id}`;
                  const draft = getTriageDraft(item);
                  const busy = triageBusyKey === key;
                  return (
                    <div key={key} className="lab-card" data-testid={`unknown-item-${item.seg_id}`}>
                      <div className="lab-card-header">
                        <Link
                          className="link-btn mono small"
                          to={`/review/${encodeEpPath(item.episode_id)}?t=${Number(item.t_start.toFixed(2))}&seg=${encodeURIComponent(item.seg_id)}&tab=diff`}
                        >
                          {item.episode_id}
                        </Link>
                        <span className="badge-pill">
                          {item.seg_id} · {item.t_start.toFixed(2)}s–{item.t_end.toFixed(2)}s
                        </span>
                      </div>
                      <div className="small">
                        <strong>Observation: </strong>
                        {item.unknown_note || "(No description provided)"}
                      </div>

                      <div className="tab-group">
                        <button
                          type="button"
                          className={`tab-btn ${draft.mode === "merge" ? "active" : ""}`}
                          data-testid={`triage-mode-merge-${item.seg_id}`}
                          onClick={() => patchTriageDraft(item, { mode: "merge" })}
                        >
                          Merge into Existing Label
                        </button>
                        <button
                          type="button"
                          className={`tab-btn ${draft.mode === "promote" ? "active" : ""}`}
                          data-testid={`triage-mode-promote-${item.seg_id}`}
                          onClick={() => patchTriageDraft(item, { mode: "promote" })}
                        >
                          Promote to New Label
                        </button>
                      </div>

                      {draft.mode === "merge" ? (
                        <div className="field-group">
                          <label className="field-group">
                            <span>Target Label</span>
                            <select
                              aria-label={`Target label-${item.seg_id}`}
                              value={draft.targetLabelId}
                              onChange={(e) =>
                                patchTriageDraft(item, { targetLabelId: e.target.value })
                              }
                            >
                              {labelsList
                                .filter((l) => l.id !== "unknown_anomaly")
                                .map((l) => (
                                  <option key={l.id} value={l.id}>
                                    {l.name} ({l.id})
                                  </option>
                                ))}
                            </select>
                          </label>
                          <label className="field-group">
                            <span>Merge Rationale (creates Few-Shot counter-example)</span>
                            <input
                              type="text"
                              value={draft.rationale}
                              onChange={(e) =>
                                patchTriageDraft(item, { rationale: e.target.value })
                              }
                            />
                          </label>
                          <button
                            type="button"
                            className="primary-btn compact"
                            data-testid={`triage-merge-btn-${item.seg_id}`}
                            disabled={busy}
                            onClick={() => void handleTriageSubmit(item)}
                          >
                            {busy ? "Processing..." : "Confirm Merge & Save Few-Shot"}
                          </button>
                        </div>
                      ) : (
                        <div className="field-group">
                          <div className="two-col">
                            <label className="field-group">
                              <span>New Label ID</span>
                              <input
                                type="text"
                                placeholder="e.g. cable_snag"
                                aria-label={`New label ID-${item.seg_id}`}
                                value={draft.newId}
                                onChange={(e) => patchTriageDraft(item, { newId: e.target.value })}
                              />
                            </label>
                            <label className="field-group">
                              <span>Display Name</span>
                              <input
                                type="text"
                                placeholder="e.g. Cable Snag"
                                aria-label={`New label name-${item.seg_id}`}
                                value={draft.newName}
                                onChange={(e) =>
                                  patchTriageDraft(item, { newName: e.target.value })
                                }
                              />
                            </label>
                          </div>
                          <div className="three-col">
                            <label className="field-group">
                              <span>Category</span>
                              <select
                                value={draft.newCategory}
                                onChange={(e) =>
                                  patchTriageDraft(item, {
                                    newCategory: e.target.value as LabelCategory,
                                  })
                                }
                              >
                                {LABEL_CATEGORIES.map((c) => (
                                  <option key={c} value={c}>
                                    {c}
                                  </option>
                                ))}
                              </select>
                            </label>
                            <label className="field-group">
                              <span>Training Role</span>
                              <select
                                value={draft.newRole}
                                onChange={(e) =>
                                  patchTriageDraft(item, {
                                    newRole: e.target.value as TrainingRole,
                                  })
                                }
                              >
                                <option value="badcase">badcase</option>
                                <option value="recovery">recovery</option>
                                <option value="necessary">necessary</option>
                                <option value="terminal_fail">terminal_fail</option>
                              </select>
                            </label>
                            <label className="field-group">
                              <span>Splice Policy</span>
                              <select
                                value={draft.newPolicy}
                                onChange={(e) =>
                                  patchTriageDraft(item, {
                                    newPolicy: e.target.value as SplicePolicy,
                                  })
                                }
                              >
                                <option value="mask_loss">mask_loss</option>
                                <option value="keep_positive">keep_positive</option>
                                <option value="cut_clean">cut_clean</option>
                              </select>
                            </label>
                          </div>
                          <div className="two-col">
                            <label className="field-group">
                              <span>Reviewer UI Title</span>
                              <input
                                type="text"
                                placeholder="e.g. Cable or Gripper Snag"
                                value={draft.newUiTitle}
                                onChange={(e) =>
                                  patchTriageDraft(item, { newUiTitle: e.target.value })
                                }
                              />
                            </label>
                            <label className="field-group">
                              <span>VLM Semantic Definition</span>
                              <input
                                type="text"
                                value={draft.newDefinition}
                                onChange={(e) =>
                                  patchTriageDraft(item, { newDefinition: e.target.value })
                                }
                              />
                            </label>
                          </div>
                          <button
                            type="button"
                            className="accent-btn compact"
                            data-testid={`triage-promote-btn-${item.seg_id}`}
                            disabled={busy}
                            onClick={() => void handleTriageSubmit(item)}
                          >
                            {busy ? "Promoting..." : "Promote to New Label (bumps version & archives .history/)"}
                          </button>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 2: FEW-SHOT MISTAKE BOOK */}
      {activeTab === "few_shots" && (
        <div className="panel-card" data-testid="lab-panel-few-shots">
          <div className="section-header-row">
            <h3>Visual / Text Few-Shot Mistake Book (Auto-injected into S3 / S4)</h3>
            <div className="filter-toolbar">
              <button
                type="button"
                className="accent-btn compact"
                data-testid="open-manual-fewshot-btn"
                onClick={() => setManualFsOpen((v) => !v)}
              >
                <Plus size={13} /> Add Few-Shot Example
              </button>
              <button
                type="button"
                className="secondary-btn compact"
                data-testid="refresh-fewshots-btn"
                onClick={() => void loadFewShots()}
              >
                <RefreshCw size={13} /> Refresh
              </button>
            </div>
          </div>

          <div className="filter-toolbar">
            <select
              aria-label="Filter by scope"
              value={fsScopeFilter}
              onChange={(e) => setFsScopeFilter(e.target.value)}
            >
              <option value="">All Scopes</option>
              <option value="global">global</option>
              {Object.keys(tasksMap).map((tid) => (
                <option key={tid} value={`task:${tid}`}>
                  task:{tid}
                </option>
              ))}
              {labelsList.map((lbl) => (
                <option key={lbl.id} value={`label:${lbl.id}`}>
                  label:{lbl.id}
                </option>
              ))}
            </select>
            <input
              type="text"
              placeholder="Filter by task_id"
              value={fsTaskFilter}
              onChange={(e) => setFsTaskFilter(e.target.value)}
            />
            <input
              type="text"
              placeholder="Filter by label_id"
              value={fsLabelFilter}
              onChange={(e) => setFsLabelFilter(e.target.value)}
            />
            <label className="checkbox-inline">
              <input
                type="checkbox"
                checked={fsEnabledOnly}
                onChange={(e) => setFsEnabledOnly(e.target.checked)}
              />
              <span>Enabled only</span>
            </label>
            <button
              type="button"
              className="primary-btn compact"
              onClick={() => void loadFewShots()}
            >
              Apply Filters
            </button>
          </div>

          {manualFsOpen && (
            <form
              className="lab-card"
              data-testid="manual-fewshot-form"
              onSubmit={(e) => void handleManualCreateFewShot(e)}
            >
              <h4>Add Few-Shot Example (auto-captures 2×2 keyframe comparison tile)</h4>
              <div className="three-col">
                <label className="field-group">
                  <span>episode_id</span>
                  <input
                    type="text"
                    placeholder="demo_ds/ep_001"
                    value={manualFsEpisode}
                    onChange={(e) => setManualFsEpisode(e.target.value)}
                  />
                </label>
                <label className="field-group">
                  <span>t_start (s)</span>
                  <input
                    type="number"
                    step={0.1}
                    value={manualFsStart}
                    onChange={(e) => setManualFsStart(e.target.value)}
                  />
                </label>
                <label className="field-group">
                  <span>t_end (s)</span>
                  <input
                    type="number"
                    step={0.1}
                    value={manualFsEnd}
                    onChange={(e) => setManualFsEnd(e.target.value)}
                  />
                </label>
              </div>
              <div className="three-col">
                <label className="field-group">
                  <span>scope</span>
                  <input
                    type="text"
                    value={manualFsScope}
                    onChange={(e) => setManualFsScope(e.target.value)}
                    placeholder="global / task:<id> / label:<id>"
                  />
                </label>
                <label className="field-group">
                  <span>task_id (optional)</span>
                  <input
                    type="text"
                    value={manualFsTask}
                    onChange={(e) => setManualFsTask(e.target.value)}
                  />
                </label>
                <label className="field-group">
                  <span>Error Type</span>
                  <select
                    value={manualFsErrorType}
                    onChange={(e) =>
                      setManualFsErrorType(e.target.value as DiagnosisErrorType)
                    }
                  >
                    {DIAGNOSIS_BUCKETS.map((b) => (
                      <option key={b.key} value={b.key}>
                        {b.label} ({b.key})
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <div className="two-col">
                <label className="field-group">
                  <span>label_id (optional)</span>
                  <input
                    type="text"
                    value={manualFsLabel}
                    onChange={(e) => setManualFsLabel(e.target.value)}
                  />
                </label>
                <label className="checkbox-inline">
                  <input
                    type="checkbox"
                    checked={manualFsCaptureTile}
                    onChange={(e) => setManualFsCaptureTile(e.target.checked)}
                  />
                  <span>Auto-capture 2×2 keyframe comparison tile (capture_tile)</span>
                </label>
              </div>
              <div className="two-col">
                <label className="field-group">
                  <span>vlm_wrong_json</span>
                  <textarea
                    rows={3}
                    className="code-textarea"
                    value={manualFsWrong}
                    onChange={(e) => setManualFsWrong(e.target.value)}
                  />
                </label>
                <label className="field-group">
                  <span>human_right_json</span>
                  <textarea
                    rows={3}
                    className="code-textarea"
                    value={manualFsRight}
                    onChange={(e) => setManualFsRight(e.target.value)}
                  />
                </label>
              </div>
              <label className="field-group">
                <span>Rationale (rationale)</span>
                <textarea
                  rows={2}
                  value={manualFsRationale}
                  onChange={(e) => setManualFsRationale(e.target.value)}
                  placeholder="Explain why the VLM prediction was wrong and the correct visual criterion..."
                />
              </label>
              <div className="header-actions">
                <button type="submit" className="primary-btn compact">
                  Save Example
                </button>
                <button
                  type="button"
                  className="secondary-btn compact"
                  onClick={() => setManualFsOpen(false)}
                >
                  Cancel
                </button>
              </div>
            </form>
          )}

          {fsLoading ? (
            <div className="muted">Loading Few-Shot examples...</div>
          ) : fewShots.length === 0 ? (
            <div className="muted">No Few-Shot examples recorded yet.</div>
          ) : (
            <div className="lab-card-grid" data-testid="few-shots-list">
              {fewShots.map((fs) => (
                <div key={fs.example_id} className="lab-card" data-testid={`fewshot-card-${fs.example_id}`}>
                  <div className="lab-card-header">
                    <span className="mono small">
                      <strong>{fs.example_id}</strong> · {fs.scope}
                    </span>
                    <label className="checkbox-inline">
                      <input
                        type="checkbox"
                        data-testid={`toggle-fewshot-${fs.example_id}`}
                        checked={fs.enabled}
                        onChange={() => void handleToggleFewShot(fs)}
                      />
                      <span>{fs.enabled ? "Enabled" : "Disabled"}</span>
                    </label>
                  </div>

                  <div className="header-actions small">
                    <span className="badge-pill warn">{fs.error_type}</span>
                    <Link className="link-btn mono" to={`/review/${encodeEpPath(fs.episode_id)}`}>
                      {fs.episode_id}
                    </Link>
                    <span className="muted mono">
                      [{fs.t_start.toFixed(2)}s – {fs.t_end.toFixed(2)}s]
                    </span>
                  </div>

                  {fs.tile_image_path ? (
                    <img
                      src={fewShotTileUrl(fs.example_id)}
                      alt={`2x2 tile ${fs.example_id}`}
                      className="few-shot-tile-img"
                    />
                  ) : (
                    <div className="few-shot-tile-placeholder">Text-only Few-Shot (no tile)</div>
                  )}

                  <div className="two-col">
                    <div className="field-group">
                      <span className="muted">VLM Wrong (vlm_wrong_json)</span>
                      <pre className="json-box">{safeJsonStringify(fs.vlm_wrong_json)}</pre>
                    </div>
                    <div className="field-group">
                      <span className="muted">Human GT (human_right_json)</span>
                      <pre className="json-box">{safeJsonStringify(fs.human_right_json)}</pre>
                    </div>
                  </div>

                  <div className="small">
                    <strong>Rationale: </strong>
                    {fs.rationale}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* TAB 3: TAXONOMY / PROMPTS / TASKS */}
      {activeTab === "taxonomy" && (
        <div className="panel-card" data-testid="lab-panel-taxonomy">
          <div className="section-header-row">
            <div className="tab-group">
              <button
                type="button"
                data-testid="taxonomy-subtab-labels"
                className={`tab-btn ${taxSubTab === "labels" ? "active" : ""}`}
                onClick={() => setTaxSubTab("labels")}
              >
                Visual Labels (labels.yaml)
              </button>
              <button
                type="button"
                data-testid="taxonomy-subtab-prompts"
                className={`tab-btn ${taxSubTab === "prompts" ? "active" : ""}`}
                onClick={() => setTaxSubTab("prompts")}
              >
                Stage Prompts (S1–S5)
              </button>
              <button
                type="button"
                data-testid="taxonomy-subtab-tasks"
                className={`tab-btn ${taxSubTab === "tasks" ? "active" : ""}`}
                onClick={() => setTaxSubTab("tasks")}
              >
                Task Specs (TaskSpec)
              </button>
            </div>
            <div className="header-actions">
              <span className="badge-pill ok">Auto-archives snapshot to .history/ before write</span>
              <button
                type="button"
                className="secondary-btn compact"
                data-testid="open-history-drawer-btn"
                onClick={() => setHistoryOpen(true)}
              >
                <History size={13} /> History &amp; Rollback
              </button>
            </div>
          </div>

          {taxSubTab === "labels" && (
            <div className="field-group" data-testid="labels-editor">
              <div className="section-header-row">
                <div>
                  <strong>
                    Active Label Registry: v{registry?.version ?? 1} · {labelsList.length} visual
                    labels
                  </strong>
                </div>
                <div className="header-actions">
                  <button
                    type="button"
                    className="secondary-btn compact"
                    onClick={() =>
                      registry &&
                      setRegistry({ ...registry, version: registry.version + 1 })
                    }
                  >
                    Bump Version (+1)
                  </button>
                  <button
                    type="button"
                    className="accent-btn compact"
                    data-testid="add-label-btn"
                    onClick={() => setNewLabelOpen((v) => !v)}
                  >
                    <Plus size={13} /> Add Label
                  </button>
                  <button
                    type="button"
                    className="primary-btn compact"
                    data-testid="save-labels-btn"
                    disabled={savingConfig || !registry}
                    onClick={() => void handleSaveLabels()}
                  >
                    {savingConfig ? "Saving..." : "Save Label Registry"}
                  </button>
                </div>
              </div>

              {newLabelOpen && (
                <div className="lab-card" data-testid="new-label-form">
                  <div className="three-col">
                    <label className="field-group">
                      <span>Label ID</span>
                      <input
                        type="text"
                        placeholder="new_visual_label"
                        value={newLabelDraft.id}
                        onChange={(e) =>
                          setNewLabelDraft({ ...newLabelDraft, id: e.target.value })
                        }
                      />
                    </label>
                    <label className="field-group">
                      <span>Display Name</span>
                      <input
                        type="text"
                        value={newLabelDraft.name}
                        onChange={(e) =>
                          setNewLabelDraft({ ...newLabelDraft, name: e.target.value })
                        }
                      />
                    </label>
                    <label className="field-group">
                      <span>Category</span>
                      <select
                        value={newLabelDraft.category}
                        onChange={(e) =>
                          setNewLabelDraft({
                            ...newLabelDraft,
                            category: e.target.value as LabelCategory,
                          })
                        }
                      >
                        {LABEL_CATEGORIES.map((c) => (
                          <option key={c} value={c}>
                            {c}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                  <div className="header-actions">
                    <button
                      type="button"
                      className="primary-btn compact"
                      onClick={handleAddNewLabel}
                    >
                      Add to List
                    </button>
                    <button
                      type="button"
                      className="secondary-btn compact"
                      onClick={() => setNewLabelOpen(false)}
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}

              <div className="two-col">
                <div className="label-chip-grid">
                  {labelsList.map((lbl) => (
                    <button
                      key={lbl.id}
                      type="button"
                      className={`choice-card ${selectedLabel?.id === lbl.id ? "selected" : ""}`}
                      onClick={() => setSelectedLabelId(lbl.id)}
                    >
                      <span className="choice-title">{lbl.name}</span>
                      <span className="mono small">{lbl.id}</span>
                      <span className="choice-desc">
                        {lbl.category} · {lbl.default_training_role} · {lbl.default_splice_policy}
                      </span>
                    </button>
                  ))}
                </div>

                {selectedLabel && (
                  <div className="lab-card" data-testid="selected-label-editor">
                    <h4>
                      Edit Label: <span className="mono">{selectedLabel.id}</span>
                    </h4>
                    <div className="two-col">
                      <label className="field-group">
                        <span>Display Name (name)</span>
                        <input
                          type="text"
                          value={selectedLabel.name}
                          onChange={(e) =>
                            updateSelectedLabel({ name: e.target.value })
                          }
                        />
                      </label>
                      <label className="field-group">
                        <span>Status (status)</span>
                        <select
                          value={selectedLabel.status}
                          onChange={(e) =>
                            updateSelectedLabel({ status: e.target.value as LabelStatus })
                          }
                        >
                          <option value="active">active</option>
                          <option value="proposed">proposed</option>
                          <option value="retired">retired</option>
                        </select>
                      </label>
                    </div>
                    <div className="three-col">
                      <label className="field-group">
                        <span>Category</span>
                        <select
                          value={selectedLabel.category}
                          onChange={(e) =>
                            updateSelectedLabel({
                              category: e.target.value as LabelCategory,
                            })
                          }
                        >
                          {LABEL_CATEGORIES.map((c) => (
                            <option key={c} value={c}>
                              {c}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label className="field-group">
                        <span>Default Training Role</span>
                        <select
                          value={selectedLabel.default_training_role}
                          onChange={(e) =>
                            updateSelectedLabel({
                              default_training_role: e.target.value as TrainingRole,
                            })
                          }
                        >
                          <option value="necessary">necessary</option>
                          <option value="recovery">recovery</option>
                          <option value="badcase">badcase</option>
                          <option value="terminal_fail">terminal_fail</option>
                        </select>
                      </label>
                      <label className="field-group">
                        <span>Default Splice Policy</span>
                        <select
                          value={selectedLabel.default_splice_policy}
                          onChange={(e) =>
                            updateSelectedLabel({
                              default_splice_policy: e.target.value as SplicePolicy,
                            })
                          }
                        >
                          <option value="keep_positive">keep_positive</option>
                          <option value="mask_loss">mask_loss</option>
                          <option value="cut_clean">cut_clean</option>
                        </select>
                      </label>
                    </div>
                    <label className="field-group">
                      <span>Reviewer UI Title (collector_ui.title)</span>
                      <input
                        type="text"
                        value={selectedLabel.collector_ui.title}
                        onChange={(e) =>
                          updateSelectedLabel({
                            collector_ui: {
                              ...selectedLabel.collector_ui,
                              title: e.target.value,
                            },
                          })
                        }
                      />
                    </label>
                    <label className="field-group">
                      <span>Reviewer UI Hint (collector_ui.hint)</span>
                      <input
                        type="text"
                        value={selectedLabel.collector_ui.hint}
                        onChange={(e) =>
                          updateSelectedLabel({
                            collector_ui: {
                              ...selectedLabel.collector_ui,
                              hint: e.target.value,
                            },
                          })
                        }
                      />
                    </label>
                    <label className="field-group">
                      <span>VLM Semantic Definition (vlm_semantics.definition)</span>
                      <textarea
                        rows={3}
                        value={selectedLabel.vlm_semantics.definition}
                        onChange={(e) =>
                          updateSelectedLabel({
                            vlm_semantics: {
                              ...selectedLabel.vlm_semantics,
                              definition: e.target.value,
                            },
                          })
                        }
                      />
                    </label>
                    <label className="field-group">
                      <span>Counter-Examples (one per line, counter_examples)</span>
                      <textarea
                        rows={2}
                        value={selectedLabel.vlm_semantics.counter_examples.join("\n")}
                        onChange={(e) =>
                          updateSelectedLabel({
                            vlm_semantics: {
                              ...selectedLabel.vlm_semantics,
                              counter_examples: e.target.value
                                .split("\n")
                                .map((s) => s.trim())
                                .filter(Boolean),
                            },
                          })
                        }
                      />
                    </label>
                  </div>
                )}
              </div>
            </div>
          )}

          {taxSubTab === "prompts" && (
            <div className="field-group" data-testid="prompts-editor">
              <div className="section-header-row">
                <div className="tab-group">
                  {PROMPT_STAGES.map((st) => (
                    <button
                      key={st.key}
                      type="button"
                      className={`tab-btn ${selectedStage === st.key ? "active" : ""}`}
                      onClick={() => setSelectedStage(st.key)}
                    >
                      {st.label}
                    </button>
                  ))}
                </div>
                <div className="header-actions">
                  <button
                    type="button"
                    className="accent-btn compact"
                    data-testid="prompt-assist-btn"
                    disabled={assistLoading}
                    onClick={() => void handleRequestPromptAssist()}
                  >
                    <Wand2 size={13} /> {assistLoading ? "Generating..." : "AI Prompt Assist"}
                  </button>
                  <button
                    type="button"
                    className="primary-btn compact"
                    data-testid="save-prompts-btn"
                    disabled={savingConfig || !prompts}
                    onClick={() => void handleSavePrompts()}
                  >
                    {savingConfig ? "Saving..." : "Save Prompt Bundle"}
                  </button>
                </div>
              </div>

              <div className="filter-toolbar">
                <label className="field-group">
                  <span>Target Task (optional)</span>
                  <input
                    type="text"
                    placeholder="task_id (optional)"
                    value={assistTaskId}
                    onChange={(e) => setAssistTaskId(e.target.value)}
                  />
                </label>
                <label className="field-group">
                  <span>Top-N Frequent Errors</span>
                  <input
                    type="number"
                    min={1}
                    max={50}
                    style={{ width: 84 }}
                    value={assistTopN}
                    onChange={(e) => setAssistTopN(Math.max(1, Number(e.target.value) || 10))}
                  />
                </label>
              </div>

              {assistPreview && (
                <div className="lab-card" data-testid="prompt-assist-preview">
                  <div className="section-header-row">
                    <strong>
                      AI Revision Suggestion Preview (
                      {assistPreview.advisory_only ? "Advisory only, never auto-saved" : "Draft"}) ·{" "}
                      {assistPreview.stage} (base v{assistPreview.base_version})
                    </strong>
                    <div className="header-actions">
                      <button
                        type="button"
                        className="accent-btn compact"
                        data-testid="apply-assist-draft-btn"
                        onClick={applyAssistDraftLocally}
                      >
                        Apply to Draft Editor
                      </button>
                      <button
                        type="button"
                        className="mini-btn"
                        onClick={() => setAssistPreview(null)}
                      >
                        Close Preview
                      </button>
                    </div>
                  </div>
                  <div className="small">
                    <strong>Rationale: </strong>
                    {assistPreview.rationale}
                  </div>
                  {assistPreview.task_spec_notes && (
                    <div className="small muted">
                      <strong>TaskSpec Notes: </strong>
                      {typeof assistPreview.task_spec_notes === "string"
                        ? assistPreview.task_spec_notes
                        : JSON.stringify(assistPreview.task_spec_notes)}
                    </div>
                  )}
                  <pre className="json-box">{assistPreview.suggested_prompt_diff}</pre>
                </div>
              )}

              {activePromptTpl ? (
                <div className="lab-card">
                  <div className="two-col">
                    <label className="field-group">
                      <span>Stage Version (version)</span>
                      <input
                        type="text"
                        aria-label="Prompt version"
                        value={activePromptTpl.version}
                        onChange={(e) => updateActivePrompt({ version: e.target.value })}
                      />
                    </label>
                    <label className="field-group">
                      <span>Description (description)</span>
                      <input
                        type="text"
                        value={activePromptTpl.description ?? ""}
                        onChange={(e) => updateActivePrompt({ description: e.target.value })}
                      />
                    </label>
                  </div>
                  <label className="field-group">
                    <span>System Template (system_template)</span>
                    <textarea
                      rows={6}
                      className="code-textarea"
                      value={activePromptTpl.system_template}
                      onChange={(e) =>
                        updateActivePrompt({ system_template: e.target.value })
                      }
                    />
                  </label>
                  <label className="field-group">
                    <span>User Template (user_template)</span>
                    <textarea
                      rows={8}
                      className="code-textarea"
                      value={activePromptTpl.user_template}
                      onChange={(e) =>
                        updateActivePrompt({ user_template: e.target.value })
                      }
                    />
                  </label>
                </div>
              ) : (
                <div className="muted">Loading prompt templates...</div>
              )}
            </div>
          )}

          {taxSubTab === "tasks" && (
            <div className="field-group" data-testid="tasks-editor">
              <div className="section-header-row">
                <div className="filter-toolbar">
                  <label className="field-group">
                    <span>Select TaskSpec</span>
                    <select
                      aria-label="Select TaskSpec"
                      value={selectedTaskId}
                      onChange={(e) => setSelectedTaskId(e.target.value)}
                    >
                      {Object.keys(tasksMap).map((tid) => (
                        <option key={tid} value={tid}>
                          {tid} ({tasksMap[tid].title || "Untitled"})
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
                <button
                  type="button"
                  className="primary-btn compact"
                  data-testid="save-task-btn"
                  disabled={savingConfig || !activeTaskSpec}
                  onClick={() => void handleSaveTaskSpec()}
                >
                  {savingConfig ? "Saving..." : "Save TaskSpec"}
                </button>
              </div>

              {activeTaskSpec ? (
                <div className="lab-card">
                  <div className="three-col">
                    <label className="field-group">
                      <span>task_id</span>
                      <input type="text" readOnly value={activeTaskSpec.task_id} />
                    </label>
                    <label className="field-group">
                      <span>version</span>
                      <input
                        type="text"
                        value={activeTaskSpec.version}
                        onChange={(e) => updateActiveTaskSpec({ version: e.target.value })}
                      />
                    </label>
                    <label className="field-group">
                      <span>Title (title)</span>
                      <input
                        type="text"
                        value={activeTaskSpec.title}
                        onChange={(e) => updateActiveTaskSpec({ title: e.target.value })}
                      />
                    </label>
                  </div>
                  <label className="field-group">
                    <span>Goal (goal)</span>
                    <input
                      type="text"
                      value={activeTaskSpec.goal}
                      onChange={(e) => updateActiveTaskSpec({ goal: e.target.value })}
                    />
                  </label>
                  <div className="two-col">
                    <label className="field-group">
                      <span>Completion Criteria (one per line, criteria)</span>
                      <textarea
                        rows={4}
                        value={activeTaskSpec.criteria.join("\n")}
                        onChange={(e) =>
                          updateActiveTaskSpec({
                            criteria: e.target.value
                              .split("\n")
                              .map((s) => s.trim())
                              .filter(Boolean),
                          })
                        }
                      />
                    </label>
                    <label className="field-group">
                      <span>Expected Steps (one per line, steps)</span>
                      <textarea
                        rows={4}
                        value={activeTaskSpec.steps.join("\n")}
                        onChange={(e) =>
                          updateActiveTaskSpec({
                            steps: e.target.value
                              .split("\n")
                              .map((s) => s.trim())
                              .filter(Boolean),
                          })
                        }
                      />
                    </label>
                  </div>
                  <label className="field-group">
                    <span>Task Context Prompt (task_prompt)</span>
                    <textarea
                      rows={4}
                      className="code-textarea"
                      value={activeTaskSpec.task_prompt}
                      onChange={(e) => updateActiveTaskSpec({ task_prompt: e.target.value })}
                    />
                  </label>
                </div>
              ) : (
                <div className="muted">No TaskSpec configurations available.</div>
              )}
            </div>
          )}
        </div>
      )}

      {/* TAB 4: GT REGRESSION EVAL SANDBOX */}
      {activeTab === "eval" && (
        <div className="page-container" data-testid="lab-panel-eval">
          <div className="panel-card" data-testid="eval-runner-controls">
            <div className="section-header-row">
              <h3>GT Regression Eval Sandbox (`is_eval_sandbox=1` · reuses S1/S2/S5 caches)</h3>
              <button
                type="button"
                className="primary-btn compact"
                data-testid="run-eval-btn"
                disabled={evalRunning}
                onClick={() => void handleRunEval()}
              >
                <FlaskConical size={14} /> {evalRunning ? "Evaluating..." : "Run GT Regression Eval"}
              </button>
            </div>

            <div className="params-grid">
              <label className="field-group">
                <span>Dataset Filter (optional)</span>
                <input
                  type="text"
                  placeholder="e.g. demo_lab"
                  value={evalDataset}
                  onChange={(e) => setEvalDataset(e.target.value)}
                />
              </label>
              <label className="field-group">
                <span>episode_ids Filter (comma-separated, optional)</span>
                <input
                  type="text"
                  placeholder="ep1, ep2"
                  value={evalEpisodeIds}
                  onChange={(e) => setEvalEpisodeIds(e.target.value)}
                />
              </label>
              <label className="field-group">
                <span>Few-Shot Top-K</span>
                <input
                  type="number"
                  min={0}
                  max={20}
                  value={evalTopK}
                  onChange={(e) => setEvalTopK(Math.max(0, Number(e.target.value) || 0))}
                />
              </label>
              <label className="field-group">
                <span>Few-Shot Max Tokens</span>
                <input
                  type="number"
                  min={100}
                  step={200}
                  value={evalMaxTokens}
                  onChange={(e) => setEvalMaxTokens(Math.max(100, Number(e.target.value) || 4000))}
                />
              </label>
            </div>

            <label className="checkbox-inline">
              <input
                type="checkbox"
                data-testid="eval-fewshot-toggle"
                checked={evalFewShot}
                onChange={(e) => setEvalFewShot(e.target.checked)}
              />
              <span>Inject enabled Few-Shot Mistake Book (`few_shot=true`)</span>
            </label>

            {activeEvalJob &&
              (activeEvalJob.status === "queued" || activeEvalJob.status === "running") && (
                <div className="job-progress-banner" data-testid="eval-job-progress-banner">
                  <div>
                    <strong>Eval Job {activeEvalJob.job_id}</strong> — {activeEvalJob.status} (
                    {activeEvalJob.progress_done}/{activeEvalJob.progress_total})
                    {activeEvalJob.summary?.cache_hit_stages &&
                      Object.keys(activeEvalJob.summary.cache_hit_stages).length > 0 && (
                        <span className="mono small" style={{ marginLeft: 8 }}>
                          · Cache hits:{" "}
                          {Object.entries(activeEvalJob.summary.cache_hit_stages)
                            .map(([st, cnt]) => `${st.replace(/_.*$/, "")}:${cnt}`)
                            .join(" ")}
                        </span>
                      )}
                  </div>
                  <button
                    type="button"
                    className="danger-btn compact"
                    onClick={() => void handleCancelEvalJob(activeEvalJob.job_id)}
                  >
                    Cancel Eval
                  </button>
                </div>
              )}
          </div>

          <div className="panel-card" data-testid="eval-compare-panel">
            <div className="section-header-row">
              <h3>Version Metric Comparison &amp; One-Click Promote</h3>
              <div className="filter-toolbar">
                <label className="field-group">
                  <span>Baseline</span>
                  <select
                    aria-label="Baseline eval"
                    value={baselineId}
                    onChange={(e) => setBaselineId(e.target.value)}
                  >
                    <option value="">-- Select Baseline --</option>
                    {(compareData?.evals ?? []).map((ev) => (
                      <option key={ev.eval_id} value={ev.eval_id}>
                        {ev.eval_id} (F1 {fmtNum(ev.metrics.f1)})
                      </option>
                    ))}
                  </select>
                </label>
                <label className="field-group">
                  <span>Candidate</span>
                  <select
                    aria-label="Candidate eval"
                    value={candidateId}
                    onChange={(e) => setCandidateId(e.target.value)}
                  >
                    <option value="">-- Select Candidate --</option>
                    {(compareData?.evals ?? []).map((ev) => (
                      <option key={ev.eval_id} value={ev.eval_id}>
                        {ev.eval_id} (F1 {fmtNum(ev.metrics.f1)})
                      </option>
                    ))}
                  </select>
                </label>
                <button
                  type="button"
                  className="secondary-btn compact"
                  data-testid="compare-evals-btn"
                  onClick={() => void handleCompareEvals()}
                >
                  <GitCompare size={13} /> Compare Metrics
                </button>
              </div>
            </div>

            {(candEval || baseEval) && (
              <div className="table-wrap">
                <table className="data-table" data-testid="eval-compare-table">
                  <thead>
                    <tr>
                      <th>Metric</th>
                      <th>Baseline ({baseEval?.eval_id ?? "—"})</th>
                      <th>Candidate ({candEval?.eval_id ?? "—"})</th>
                      <th>Delta (Candidate − Baseline)</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>F1 Score</td>
                      <td className="mono">{fmtNum(baseEval?.metrics.f1)}</td>
                      <td className="mono">{fmtNum(candEval?.metrics.f1)}</td>
                      <td className={`mono ${fmtDelta(deltas?.delta_f1).cls}`}>
                        {fmtDelta(deltas?.delta_f1).text}
                      </td>
                    </tr>
                    <tr>
                      <td>Precision / Recall</td>
                      <td className="mono">
                        {fmtNum(baseEval?.metrics.precision)} / {fmtNum(baseEval?.metrics.recall)}
                      </td>
                      <td className="mono">
                        {fmtNum(candEval?.metrics.precision)} / {fmtNum(candEval?.metrics.recall)}
                      </td>
                      <td className={`mono ${fmtDelta(deltas?.delta_precision).cls}`}>
                        {fmtDelta(deltas?.delta_precision).text} / {fmtDelta(deltas?.delta_recall).text}
                      </td>
                    </tr>
                    <tr>
                      <td>Mean IoU</td>
                      <td className="mono">{fmtNum(baseEval?.metrics.mean_iou)}</td>
                      <td className="mono">{fmtNum(candEval?.metrics.mean_iou)}</td>
                      <td className={`mono ${fmtDelta(deltas?.delta_mean_iou).cls}`}>
                        {fmtDelta(deltas?.delta_mean_iou).text}
                      </td>
                    </tr>
                    <tr>
                      <td>Overcut Rate (overcut_rate, lower is better)</td>
                      <td className="mono">{fmtPct(baseEval?.metrics.overcut_rate)}</td>
                      <td className="mono">{fmtPct(candEval?.metrics.overcut_rate)}</td>
                      <td
                        className={`mono ${
                          fmtDelta(deltas?.delta_overcut_rate, { pct: true, lowerBetter: true }).cls
                        }`}
                      >
                        {fmtDelta(deltas?.delta_overcut_rate, { pct: true, lowerBetter: true }).text}
                      </td>
                    </tr>
                    <tr>
                      <td>Guardrail Downgrade Rate (guardrail_downgrade_rate)</td>
                      <td className="mono">{fmtPct(baseEval?.metrics.guardrail_downgrade_rate)}</td>
                      <td className="mono">{fmtPct(candEval?.metrics.guardrail_downgrade_rate)}</td>
                      <td
                        className={`mono ${
                          fmtDelta(deltas?.delta_guardrail_downgrade_rate, {
                            pct: true,
                            lowerBetter: true,
                          }).cls
                        }`}
                      >
                        {
                          fmtDelta(deltas?.delta_guardrail_downgrade_rate, {
                            pct: true,
                            lowerBetter: true,
                          }).text
                        }
                      </td>
                    </tr>
                    <tr>
                      <td>Success Accuracy (success_accuracy)</td>
                      <td className="mono">{fmtPct(baseEval?.metrics.success_accuracy)}</td>
                      <td className="mono">{fmtPct(candEval?.metrics.success_accuracy)}</td>
                      <td
                        className={`mono ${
                          fmtDelta(deltas?.delta_success_accuracy, { pct: true }).cls
                        }`}
                      >
                        {fmtDelta(deltas?.delta_success_accuracy, { pct: true }).text}
                      </td>
                    </tr>
                    <tr>
                      <td>Total Tokens (total_tokens)</td>
                      <td className="mono">{baseEval?.metrics.total_tokens ?? "—"}</td>
                      <td className="mono">{candEval?.metrics.total_tokens ?? "—"}</td>
                      <td
                        className={`mono ${
                          fmtDelta(deltas?.delta_total_tokens, { lowerBetter: true, digits: 0 }).cls
                        }`}
                      >
                        {fmtDelta(deltas?.delta_total_tokens, { lowerBetter: true, digits: 0 }).text}
                      </td>
                    </tr>
                    <tr>
                      <td>Cache Hit Stages (cache_hit_stages / rate)</td>
                      <td className="mono">
                        {baseEval
                          ? fmtCacheHits(
                              baseEval.metrics.cache_hit_stages,
                              baseEval.metrics.cache_hit_rate,
                            )
                          : "—"}
                      </td>
                      <td className="mono">
                        {candEval
                          ? fmtCacheHits(
                              candEval.metrics.cache_hit_stages,
                              candEval.metrics.cache_hit_rate,
                            )
                          : "—"}
                      </td>
                      <td
                        className={`mono ${
                          deltas
                            ? fmtDelta(deltas.delta_cache_hit_rate, { pct: true }).cls
                            : "muted"
                        }`}
                      >
                        {deltas
                          ? fmtDelta(deltas.delta_cache_hit_rate, { pct: true }).text
                          : "S1/S2/S5 cache reuse"}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            )}

            <div className="section-header-row">
              <div className="filter-toolbar">
                <input
                  type="text"
                  aria-label="Promote reason"
                  placeholder="Promote reason"
                  value={promoteReason}
                  onChange={(e) => setPromoteReason(e.target.value)}
                />
              </div>
              <button
                type="button"
                className="success-btn compact"
                data-testid="promote-btn"
                disabled={promoting}
                onClick={() => void handlePromote()}
              >
                <Rocket size={14} /> {promoting ? "Promoting..." : "Promote as Active Config"}
              </button>
            </div>

            {candEval && candEval.per_episode.length > 0 && (
              <details>
                <summary className="small muted">
                  View Per-Episode Candidate Details ({candEval.per_episode.length} GT episodes)
                </summary>
                <div className="table-wrap" style={{ marginTop: 8 }}>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>episode_id</th>
                        <th>F1</th>
                        <th>Mean IoU</th>
                        <th>Overcut</th>
                        <th>Success Match</th>
                        <th>Tokens</th>
                      </tr>
                    </thead>
                    <tbody>
                      {candEval.per_episode.map((ep) => (
                        <tr key={ep.episode_id}>
                          <td className="mono small">{ep.episode_id}</td>
                          <td className="mono small">{fmtNum(ep.f1)}</td>
                          <td className="mono small">{fmtNum(ep.mean_iou)}</td>
                          <td className="mono small">{fmtPct(ep.overcut_rate)}</td>
                          <td className="small">
                            {ep.success_correct ? (
                              <span className="badge-pill ok">
                                <CheckCircle2 size={11} /> Match
                              </span>
                            ) : (
                              <span className="badge-pill warn">Mismatch</span>
                            )}
                          </td>
                          <td className="mono small">{ep.tokens_total ?? "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            )}
          </div>
        </div>
      )}

      <LabHistoryDrawer
        open={historyOpen}
        target={taxSubTab}
        onClose={() => setHistoryOpen(false)}
        onRolledBack={async (res) => {
          await Promise.all([loadLabels(), loadPrompts(), loadTasks()]);
          showLabInfo(
            `Safely rolled back ${res.relative_path} (sha256: ${res.config_sha256.slice(0, 8)})`,
            "success",
          );
        }}
      />
      {toast.StandaloneHost}
    </div>
  );
};
