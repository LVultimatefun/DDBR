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
import { type TranslateFn, useI18n } from "../stores/i18nStore";
import { useJobs } from "../stores/jobStore";
import { useToast } from "../stores/toastStore";

type LabTab = "diagnosis" | "few_shots" | "taxonomy" | "eval";
type TaxonomySubTab = "labels" | "prompts" | "tasks";

const DIAGNOSIS_BUCKETS: Array<{ key: DiagnosisErrorType; labelKey: string }> = [
  { key: "fp_overflag", labelKey: "lab.bucket.fp_overflag" },
  { key: "fn_missed", labelKey: "lab.bucket.fn_missed" },
  { key: "policy_flip", labelKey: "lab.bucket.policy_flip" },
  { key: "boundary", labelKey: "lab.bucket.boundary" },
  { key: "success_flip", labelKey: "lab.bucket.success_flip" },
  { key: "stage", labelKey: "lab.bucket.stage" },
];

const PROMPT_STAGES: Array<{ key: PromptStageKey; labelKey: string }> = [
  { key: "s1_grounding", labelKey: "lab.stage.s1_grounding" },
  { key: "s2_macro", labelKey: "lab.stage.s2_macro" },
  { key: "s3_discover", labelKey: "lab.stage.s3_discover" },
  { key: "s4_verify", labelKey: "lab.stage.s4_verify" },
  { key: "s5_success", labelKey: "lab.stage.s5_success" },
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

function parseJsonObject(text: string, invalidObjMessage: string): Record<string, unknown> {
  const trimmed = text.trim();
  if (!trimmed) return {};
  const parsed = JSON.parse(trimmed);
  if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
    return parsed as Record<string, unknown>;
  }
  throw new Error(invalidObjMessage);
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

function fmtCompactDiagValue(val: unknown, t: TranslateFn): string {
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
        rec.passed === true
          ? t("lab.diag.verdictPass")
          : rec.passed === false
            ? t("lab.diag.verdictFail")
            : rec.passed === null
              ? t("lab.diag.verdictUnsure")
              : "";
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

function summarizeStageDiff(vlmVal: unknown, gtVal: unknown, t: TranslateFn): string[] {
  const vlmArr: StageLike[] = Array.isArray(vlmVal) ? (vlmVal as StageLike[]) : [];
  const gtArr: StageLike[] = Array.isArray(gtVal) ? (gtVal as StageLike[]) : [];
  if (vlmArr.length === 0 && gtArr.length === 0) {
    return [t("lab.diag.stageStructDiffers")];
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
            t("lab.diag.stageBoundaryShift", {
              id,
              vStart: vStart.toFixed(1),
              vEnd: vEnd.toFixed(1),
              gStart: gStart.toFixed(1),
              gEnd: gEnd.toFixed(1),
            }),
          );
        } else {
          parts.push(`${id}`);
        }
        if (nameChanged) {
          parts.push(
            t("lab.diag.stageRenamed", {
              vName: vName || "—",
              gName: gName || "—",
            }),
          );
        }
        lines.push(parts.join(" · "));
      }
    } else if (v && !g) {
      const id = v.stage_id || `S${i + 1}`;
      const span =
        typeof v.t_start === "number" && typeof v.t_end === "number"
          ? ` (${v.t_start.toFixed(1)}–${v.t_end.toFixed(1)}s)`
          : "";
      lines.push(t("lab.diag.stageExtraVlm", { id, span }));
    } else if (!v && g) {
      const id = g.stage_id || `S${i + 1}`;
      const span =
        typeof g.t_start === "number" && typeof g.t_end === "number"
          ? ` (${g.t_start.toFixed(1)}–${g.t_end.toFixed(1)}s)`
          : "";
      lines.push(t("lab.diag.stageAddedGt", { id, span }));
    }
  }

  return lines.length > 0
    ? lines
    : [t("lab.diag.stageCountDiff", { vlmCount: vlmArr.length, gtCount: gtArr.length })];
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
  const { t } = useI18n();
  const tRef = useRef(t);
  tRef.current = t;
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
      setErrorMsg(err instanceof Error ? err.message : tRef.current("lab.err.loadDiagnosis"));
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
      setErrorMsg(err instanceof Error ? err.message : tRef.current("lab.err.loadUnknowns"));
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
      setErrorMsg(err instanceof Error ? err.message : tRef.current("lab.err.loadFewShots"));
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
          setCandidateId(res.evals[0].eval_id);
        }
      } catch (err) {
        setErrorMsg(err instanceof Error ? err.message : tRef.current("lab.err.loadEvalCompare"));
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
        const errText = job.summary?.errors?.[0]?.error || tRef.current("lab.err.evalFailed");
        setErrorMsg(errText);
        return;
      }
      if (job.status === "cancelled") {
        showLabInfo(tRef.current("lab.toast.evalCancelled", { jobId: job.job_id }), "warning");
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
        const candFromCmp =
          cmp.candidate ??
          (evalRunId ? cmp.evals.find((e) => e.eval_id === evalRunId) : cmp.evals[0]) ??
          null;
        if (candFromCmp) {
          resolvedReport = candFromCmp;
        }
      } catch (err) {
        setErrorMsg(err instanceof Error ? err.message : tRef.current("lab.err.loadEvalCompare"));
      }

      if (resolvedReport) {
        setLatestEval(resolvedReport);
        setCandidateId(resolvedReport.eval_id);
        const epCount =
          resolvedReport.metrics?.episodes_evaluated ?? resolvedReport.per_episode.length;
        showLabInfo(
          tRef.current("lab.toast.evalCompleted", {
            evalId: resolvedReport.eval_id,
            epCount,
            rate: fmtPct(resolvedReport.metrics.cache_hit_rate),
          }),
          "success",
        );
      } else {
        if (evalRunId) setCandidateId(evalRunId);
        showLabInfo(
          tRef.current("lab.toast.evalJobCompleted", { id: evalRunId ?? job.job_id }),
          "success",
        );
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

  const activeEvalJobId = activeEvalJob?.job_id ?? null;
  const activeEvalJobStatus = activeEvalJob?.status ?? null;
  const jobStoreProvided = jobStore.isProvided;
  const trackJobInStore = jobStore.trackJob;

  useEffect(() => {
    if (!activeEvalJobId) return;
    if (activeEvalJobStatus !== "queued" && activeEvalJobStatus !== "running") return;

    let stopped = false;
    const timer = setInterval(() => {
      if (stopped) return;
      void getJob(activeEvalJobId)
        .then((latest) => {
          if (stopped) return;
          setLocalEvalJob(latest);
          if (jobStoreProvided) {
            trackJobInStore(latest);
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
  }, [activeEvalJobId, activeEvalJobStatus, finalizeEvalJob, jobStoreProvided, trackJobInStore]);

  if (authProbing) {
    return <div className="page-container">{t("lab.auth.verifying")}</div>;
  }

  if (!isLead) {
    return (
      <div className="page-container" data-testid="lab-forbidden">
        <div className="forbidden-card">
          <ShieldOff size={32} />
          <h2>{t("lab.auth.forbiddenTitle")}</h2>
          <p>{t("lab.auth.forbiddenDesc")}</p>
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
        rationale: item.unknown_note || t("lab.unknowns.defaultRationale"),
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
          rationale:
            draft.rationale.trim() ||
            item.unknown_note ||
            t("lab.unknowns.fallbackMergeRationale"),
        });
        showLabInfo(
          t("lab.toast.unknownMerged", {
            segId: item.seg_id,
            targetLabelId: draft.targetLabelId,
            fewShotId: res.few_shot_id ?? "ok",
          }),
          "success",
        );
      } else {
        if (!draft.newId.trim() || !draft.newName.trim()) {
          throw new Error(t("lab.err.promoteRequireIdName"));
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
          t("lab.toast.unknownPromoted", {
            labelId: newLabel.id,
            version: res.label_version ?? "updated",
          }),
          "success",
        );
      }
      await Promise.all([loadUnknowns(), loadLabels()]);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : t("lab.err.triageUnknown"));
    } finally {
      setTriageBusyKey(null);
    }
  };

  // Quick Few-Shot from Diagnosis row
  const openQuickFewShot = (err: DiagnosisErrorItem) => {
    setQuickFsError(err);
    setQuickFsScope(err.task_id ? `task:${err.task_id}` : "global");
    setQuickFsRationale(
      err.summary ||
        t("lab.fewShots.defaultQuickRationale", {
          errorType: err.error_type,
          target: err.label_id ?? "task",
        }),
    );
  };

  const submitQuickFewShot = async () => {
    if (!quickFsError) return;
    if (!quickFsRationale.trim()) {
      setErrorMsg(t("lab.err.quickFsRationaleRequired"));
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
      showLabInfo(t("lab.toast.quickFsAdded", { id: created.example_id }), "success");
      setQuickFsError(null);
      setQuickFsRationale("");
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : t("lab.err.addFewShot"));
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
      if (!manualFsEpisode.trim()) throw new Error(t("lab.err.episodeIdRequired"));
      if (!Number.isFinite(tStart) || !Number.isFinite(tEnd) || tEnd <= tStart) {
        throw new Error(t("lab.err.invalidTimeSpan"));
      }
      if (!manualFsRationale.trim()) throw new Error(t("lab.err.rationaleRequired"));
      const vlmWrong = parseJsonObject(manualFsWrong, t("lab.err.jsonMustBeObject"));
      const humanRight = parseJsonObject(manualFsRight, t("lab.err.jsonMustBeObject"));
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
      showLabInfo(t("lab.toast.manualFsCreated", { id: created.example_id }), "success");
      setManualFsOpen(false);
      await loadFewShots();
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : t("lab.err.createFewShot"));
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
      setErrorMsg(err instanceof Error ? err.message : t("lab.err.toggleFewShot"));
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
      setErrorMsg(t("lab.err.newLabelRequireIdName"));
      return;
    }
    const baseReg: LabelRegistry = registry ?? {
      schema_version: "1.0",
      version: 1,
      categories: LABEL_CATEGORIES,
      labels: DEFAULT_VISUAL_LABELS,
    };
    if (baseReg.labels.some((l) => l.id === newLabelDraft.id.trim())) {
      setErrorMsg(t("lab.err.labelIdExists", { id: newLabelDraft.id.trim() }));
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
    showLabInfo(t("lab.toast.labelDraftAdded", { id: added.id }));
  };

  const handleSaveLabels = async () => {
    if (!registry) return;
    setSavingConfig(true);
    setErrorMsg(null);
    clearLabInfo();
    try {
      const sanitized: LabelRegistry = {
        ...registry,
        labels: registry.labels.map((l) => ({
          ...l,
          vlm_semantics: {
            ...l.vlm_semantics,
            counter_examples: l.vlm_semantics.counter_examples
              .map((s) => s.trim())
              .filter(Boolean),
            visual_cues: l.vlm_semantics.visual_cues.map((s) => s.trim()).filter(Boolean),
          },
        })),
      };
      const saved = await putLabLabels(sanitized);
      setRegistry(saved);
      showLabInfo(t("lab.toast.labelsSaved", { version: saved.version }), "success");
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : t("lab.err.saveLabels"));
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
        t("lab.toast.promptsSaved", {
          stage: selectedStage,
          version: saved[selectedStage].version,
        }),
        "success",
      );
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : t("lab.err.savePrompts"));
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
      setErrorMsg(err instanceof Error ? err.message : t("lab.err.promptAssist"));
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
    showLabInfo(t("lab.toast.assistApplied"));
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
      const sanitizedTask: TaskSpec = {
        ...activeTaskSpec,
        criteria: activeTaskSpec.criteria.map((s) => s.trim()).filter(Boolean),
        steps: activeTaskSpec.steps.map((s) => s.trim()).filter(Boolean),
      };
      const res = await putLabTasks(sanitizedTask);
      setTasksMap(res.tasks);
      showLabInfo(
        t("lab.toast.taskSaved", {
          taskId: sanitizedTask.task_id,
          version: sanitizedTask.version,
        }),
        "success",
      );
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : t("lab.err.saveTask"));
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
        showLabInfo(t("lab.toast.evalSubmitted", { jobId: job.job_id }));
      }
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : t("lab.err.submitEval"));
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
        t("lab.toast.promoted", {
          files: res.promoted_files.join(", ") || "configs_override",
        }),
        "success",
      );
      await Promise.all([loadLabels(), loadPrompts(), loadTasks()]);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : t("lab.err.promote"));
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
          <h1>{t("lab.title")}</h1>
          <p className="subtitle">{t("lab.subtitle")}</p>
        </div>
      </div>

      {errorMsg && (
        <div className="error-banner" data-testid="lab-error-banner">
          <span>{errorMsg}</span>
          <button type="button" className="mini-btn" onClick={() => setErrorMsg(null)}>
            {t("lab.common.dismiss")}
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
          <Sparkles size={14} /> {t("lab.tab.diagnosis")}
        </button>
        <button
          type="button"
          data-testid="tab-few-shots"
          className={`tab-btn ${activeTab === "few_shots" ? "active" : ""}`}
          onClick={() => setActiveTab("few_shots")}
        >
          <Bookmark size={14} /> {t("lab.tab.fewShots")}
        </button>
        <button
          type="button"
          data-testid="tab-taxonomy"
          className={`tab-btn ${activeTab === "taxonomy" ? "active" : ""}`}
          onClick={() => setActiveTab("taxonomy")}
        >
          <Layers size={14} /> {t("lab.tab.taxonomy")}
        </button>
        <button
          type="button"
          data-testid="tab-eval"
          className={`tab-btn ${activeTab === "eval" ? "active" : ""}`}
          onClick={() => setActiveTab("eval")}
        >
          <FlaskConical size={14} /> {t("lab.tab.eval")}
        </button>
      </div>

      {/* TAB 1: DIAGNOSIS & UNKNOWN TRIAGE */}
      {activeTab === "diagnosis" && (
        <div className="page-container" data-testid="lab-panel-diagnosis">
          <div className="panel-card">
            <div className="section-header-row">
              <h3>{t("lab.diag.title")}</h3>
              <div className="filter-toolbar">
                <input
                  type="text"
                  aria-label={t("lab.diag.filterEpAria")}
                  placeholder={t("lab.diag.filterEpPlaceholder")}
                  value={diagEpFilter}
                  onChange={(e) => setDiagEpFilter(e.target.value)}
                />
                <input
                  type="number"
                  aria-label={t("lab.diag.topNAria")}
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
                  <RefreshCw size={13} />{" "}
                  {diagLoading ? t("lab.diag.diagnosing") : t("lab.diag.refreshBtn")}
                </button>
              </div>
            </div>

            <div className="lab-kpi-strip" data-testid="diagnosis-kpi-strip">
              <div className="lab-kpi-card">
                <span className="lab-kpi-val">{diagnosis?.episodes_evaluated ?? 0}</span>
                <span className="lab-kpi-lbl">{t("lab.diag.kpiEvaluated")}</span>
              </div>
              <div className="lab-kpi-card">
                <span className="lab-kpi-val">{diagnosis?.total_errors ?? 0}</span>
                <span className="lab-kpi-lbl">{t("lab.diag.kpiTotalErrors")}</span>
              </div>
              {DIAGNOSIS_BUCKETS.map((b) => (
                <div key={b.key} className="lab-kpi-card" data-testid={`kpi-bucket-${b.key}`}>
                  <span className="lab-kpi-val">{diagnosis?.counts_by_type?.[b.key] ?? 0}</span>
                  <span className="lab-kpi-lbl">
                    {t(b.labelKey)} ({b.key})
                  </span>
                </div>
              ))}
            </div>

            {quickFsError && (
              <div className="lab-card" data-testid="quick-fewshot-form">
                <div className="section-header-row">
                  <strong>
                    {t("lab.diag.quickFsTitle", {
                      episodeId: quickFsError.episode_id,
                      errorType: quickFsError.error_type,
                    })}
                  </strong>
                  <button
                    type="button"
                    className="mini-btn"
                    onClick={() => setQuickFsError(null)}
                  >
                    {t("lab.common.cancel")}
                  </button>
                </div>
                <div className="two-col">
                  <label className="field-group">
                    <span>{t("lab.diag.quickFsScope")}</span>
                    <select value={quickFsScope} onChange={(e) => setQuickFsScope(e.target.value)}>
                      <option value="global">{t("lab.diag.scopeGlobal")}</option>
                      {quickFsError.task_id && (
                        <option value={`task:${quickFsError.task_id}`}>
                          {t("lab.diag.scopeTask", { taskId: quickFsError.task_id })}
                        </option>
                      )}
                      {quickFsError.label_id && (
                        <option value={`label:${quickFsError.label_id}`}>
                          {t("lab.diag.scopeLabel", { labelId: quickFsError.label_id })}
                        </option>
                      )}
                    </select>
                  </label>
                  <label className="field-group">
                    <span>{t("lab.diag.timeSpan")}</span>
                    <input
                      type="text"
                      readOnly
                      value={`${(quickFsError.t_start ?? 0).toFixed(2)}s - ${(quickFsError.t_end ?? (quickFsError.t_start ?? 0) + 1.0).toFixed(2)}s`}
                    />
                  </label>
                </div>
                <label className="field-group">
                  <span>{t("lab.fewShots.rationaleLabel")}</span>
                  <textarea
                    rows={2}
                    value={quickFsRationale}
                    onChange={(e) => setQuickFsRationale(e.target.value)}
                    placeholder={t("lab.diag.quickFsRationalePlaceholder")}
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
                    {quickFsSaving ? t("lab.diag.quickFsSaving") : t("lab.diag.quickFsSubmit")}
                  </button>
                </div>
              </div>
            )}

            <div className="table-wrap">
              <table className="data-table" data-testid="diagnosis-errors-table">
                <thead>
                  <tr>
                    <th>{t("lab.diag.colEpisode")}</th>
                    <th>{t("lab.diag.colErrorType")}</th>
                    <th>{t("lab.diag.colLabelSegment")}</th>
                    <th>{t("lab.diag.colTimeSpan")}</th>
                    <th>{t("lab.diag.colDetails")}</th>
                    <th>{t("lab.diag.colAction")}</th>
                  </tr>
                </thead>
                <tbody>
                  {errorItems.length === 0 ? (
                    <tr>
                      <td colSpan={6} className="empty-cell">
                        {t("lab.diag.emptyTable")}
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
                            : t("lab.diag.spanGlobal")}
                        </td>
                        <td className="small">
                          <div>{err.summary || t("lab.diag.defaultSummary")}</div>
                          {(err.vlm_value !== undefined || err.gt_value !== undefined) && (
                            err.error_type === "stage" ? (
                              <div className="diag-stage-diff">
                                {summarizeStageDiff(err.vlm_value, err.gt_value, t).map((line, lineIdx) => (
                                  <div key={lineIdx} className="diag-stage-item">
                                    {line}
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <div className="muted mono">
                                VLM: {fmtCompactDiagValue(err.vlm_value, t)} → GT: {fmtCompactDiagValue(err.gt_value, t)}
                              </div>
                            )
                          )}
                          {(err.vlm_value !== undefined || err.gt_value !== undefined) && (
                            <details className="diag-raw-json">
                              <summary>{t("lab.diag.viewRawJson")}</summary>
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
                            <Plus size={12} /> {t("lab.diag.addFewShotBtn")}
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
              <h3>{t("lab.unknowns.title", { count: unkTotal })}</h3>
              <div className="filter-toolbar">
                <input
                  type="text"
                  aria-label={t("lab.unknowns.filterDatasetAria")}
                  placeholder={t("lab.unknowns.filterDatasetPlaceholder")}
                  value={unkDatasetFilter}
                  onChange={(e) => setUnkDatasetFilter(e.target.value)}
                />
                <button
                  type="button"
                  className="secondary-btn compact"
                  data-testid="refresh-unknowns-btn"
                  onClick={() => void loadUnknowns()}
                >
                  <RefreshCw size={13} /> {t("lab.unknowns.refreshBtn")}
                </button>
              </div>
            </div>

            {unknowns.length === 0 ? (
              <div className="muted">{t("lab.unknowns.empty")}</div>
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
                        <strong>{t("lab.unknowns.observationLabel")}</strong>
                        {item.unknown_note || t("lab.unknowns.noDescription")}
                      </div>

                      <div className="tab-group">
                        <button
                          type="button"
                          className={`tab-btn ${draft.mode === "merge" ? "active" : ""}`}
                          data-testid={`triage-mode-merge-${item.seg_id}`}
                          onClick={() => patchTriageDraft(item, { mode: "merge" })}
                        >
                          {t("lab.unknowns.modeMerge")}
                        </button>
                        <button
                          type="button"
                          className={`tab-btn ${draft.mode === "promote" ? "active" : ""}`}
                          data-testid={`triage-mode-promote-${item.seg_id}`}
                          onClick={() => patchTriageDraft(item, { mode: "promote" })}
                        >
                          {t("lab.unknowns.modePromote")}
                        </button>
                      </div>

                      {draft.mode === "merge" ? (
                        <div className="field-group">
                          <label className="field-group">
                            <span>{t("lab.unknowns.targetLabel")}</span>
                            <select
                              aria-label={t("lab.unknowns.targetLabelAria", { segId: item.seg_id })}
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
                            <span>{t("lab.unknowns.mergeRationale")}</span>
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
                            {busy ? t("lab.unknowns.processing") : t("lab.unknowns.confirmMergeBtn")}
                          </button>
                        </div>
                      ) : (
                        <div className="field-group">
                          <div className="two-col">
                            <label className="field-group">
                              <span>{t("lab.unknowns.newIdLabel")}</span>
                              <input
                                type="text"
                                placeholder={t("lab.unknowns.newIdPlaceholder")}
                                aria-label={t("lab.unknowns.newIdAria", { segId: item.seg_id })}
                                value={draft.newId}
                                onChange={(e) => patchTriageDraft(item, { newId: e.target.value })}
                              />
                            </label>
                            <label className="field-group">
                              <span>{t("lab.unknowns.newNameLabel")}</span>
                              <input
                                type="text"
                                placeholder={t("lab.unknowns.newNamePlaceholder")}
                                aria-label={t("lab.unknowns.newNameAria", { segId: item.seg_id })}
                                value={draft.newName}
                                onChange={(e) =>
                                  patchTriageDraft(item, { newName: e.target.value })
                                }
                              />
                            </label>
                          </div>
                          <div className="three-col">
                            <label className="field-group">
                              <span>{t("lab.taxonomy.category")}</span>
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
                                    {t(`lab.category.${c}`)}
                                  </option>
                                ))}
                              </select>
                            </label>
                            <label className="field-group">
                              <span>{t("lab.unknowns.trainingRole")}</span>
                              <select
                                value={draft.newRole}
                                onChange={(e) =>
                                  patchTriageDraft(item, {
                                    newRole: e.target.value as TrainingRole,
                                  })
                                }
                              >
                                <option value="badcase">{t("lab.role.badcase")}</option>
                                <option value="recovery">{t("lab.role.recovery")}</option>
                                <option value="necessary">{t("lab.role.necessary")}</option>
                                <option value="terminal_fail">{t("lab.role.terminal_fail")}</option>
                              </select>
                            </label>
                            <label className="field-group">
                              <span>{t("lab.unknowns.splicePolicy")}</span>
                              <select
                                value={draft.newPolicy}
                                onChange={(e) =>
                                  patchTriageDraft(item, {
                                    newPolicy: e.target.value as SplicePolicy,
                                  })
                                }
                              >
                                <option value="mask_loss">{t("lab.policy.mask_loss")}</option>
                                <option value="keep_positive">{t("lab.policy.keep_positive")}</option>
                                <option value="cut_clean">{t("lab.policy.cut_clean")}</option>
                              </select>
                            </label>
                          </div>
                          <div className="two-col">
                            <label className="field-group">
                              <span>{t("lab.unknowns.uiTitle")}</span>
                              <input
                                type="text"
                                placeholder={t("lab.unknowns.uiTitlePlaceholder")}
                                value={draft.newUiTitle}
                                onChange={(e) =>
                                  patchTriageDraft(item, { newUiTitle: e.target.value })
                                }
                              />
                            </label>
                            <label className="field-group">
                              <span>{t("lab.unknowns.vlmDefinition")}</span>
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
                            {busy ? t("lab.unknowns.promoting") : t("lab.unknowns.confirmPromoteBtn")}
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
            <h3>{t("lab.fewShots.title")}</h3>
            <div className="filter-toolbar">
              <button
                type="button"
                className="accent-btn compact"
                data-testid="open-manual-fewshot-btn"
                onClick={() => setManualFsOpen((v) => !v)}
              >
                <Plus size={13} /> {t("lab.fewShots.addBtn")}
              </button>
              <button
                type="button"
                className="secondary-btn compact"
                data-testid="refresh-fewshots-btn"
                onClick={() => void loadFewShots()}
              >
                <RefreshCw size={13} /> {t("lab.common.refresh")}
              </button>
            </div>
          </div>

          <div className="filter-toolbar">
            <select
              aria-label={t("lab.fewShots.filterScopeAria")}
              value={fsScopeFilter}
              onChange={(e) => setFsScopeFilter(e.target.value)}
            >
              <option value="">{t("lab.fewShots.allScopes")}</option>
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
              placeholder={t("lab.fewShots.filterTaskPlaceholder")}
              value={fsTaskFilter}
              onChange={(e) => setFsTaskFilter(e.target.value)}
            />
            <input
              type="text"
              placeholder={t("lab.fewShots.filterLabelPlaceholder")}
              value={fsLabelFilter}
              onChange={(e) => setFsLabelFilter(e.target.value)}
            />
            <label className="checkbox-inline">
              <input
                type="checkbox"
                checked={fsEnabledOnly}
                onChange={(e) => setFsEnabledOnly(e.target.checked)}
              />
              <span>{t("lab.fewShots.enabledOnly")}</span>
            </label>
            <button
              type="button"
              className="primary-btn compact"
              onClick={() => void loadFewShots()}
            >
              {t("lab.fewShots.applyFiltersBtn")}
            </button>
          </div>

          {manualFsOpen && (
            <form
              className="lab-card"
              data-testid="manual-fewshot-form"
              onSubmit={(e) => void handleManualCreateFewShot(e)}
            >
              <h4>{t("lab.fewShots.manualFormTitle")}</h4>
              <div className="three-col">
                <label className="field-group">
                  <span>{t("lab.fewShots.episodeIdLabel")}</span>
                  <input
                    type="text"
                    placeholder="demo_ds/ep_001"
                    value={manualFsEpisode}
                    onChange={(e) => setManualFsEpisode(e.target.value)}
                  />
                </label>
                <label className="field-group">
                  <span>{t("lab.fewShots.tStartLabel")}</span>
                  <input
                    type="number"
                    step={0.1}
                    value={manualFsStart}
                    onChange={(e) => setManualFsStart(e.target.value)}
                  />
                </label>
                <label className="field-group">
                  <span>{t("lab.fewShots.tEndLabel")}</span>
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
                  <span>{t("lab.fewShots.scopeLabel")}</span>
                  <input
                    type="text"
                    value={manualFsScope}
                    onChange={(e) => setManualFsScope(e.target.value)}
                    placeholder="global / task:<id> / label:<id>"
                  />
                </label>
                <label className="field-group">
                  <span>{t("lab.fewShots.taskIdOptional")}</span>
                  <input
                    type="text"
                    value={manualFsTask}
                    onChange={(e) => setManualFsTask(e.target.value)}
                  />
                </label>
                <label className="field-group">
                  <span>{t("lab.fewShots.errorTypeLabel")}</span>
                  <select
                    value={manualFsErrorType}
                    onChange={(e) =>
                      setManualFsErrorType(e.target.value as DiagnosisErrorType)
                    }
                  >
                    {DIAGNOSIS_BUCKETS.map((b) => (
                      <option key={b.key} value={b.key}>
                        {t(b.labelKey)} ({b.key})
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <div className="two-col">
                <label className="field-group">
                  <span>{t("lab.fewShots.labelIdOptional")}</span>
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
                  <span>{t("lab.fewShots.captureTileLabel")}</span>
                </label>
              </div>
              <div className="two-col">
                <label className="field-group">
                  <span>{t("lab.fewShots.vlmWrongJsonLabel")}</span>
                  <textarea
                    rows={3}
                    className="code-textarea"
                    value={manualFsWrong}
                    onChange={(e) => setManualFsWrong(e.target.value)}
                  />
                </label>
                <label className="field-group">
                  <span>{t("lab.fewShots.humanRightJsonLabel")}</span>
                  <textarea
                    rows={3}
                    className="code-textarea"
                    value={manualFsRight}
                    onChange={(e) => setManualFsRight(e.target.value)}
                  />
                </label>
              </div>
              <label className="field-group">
                <span>{t("lab.fewShots.rationaleLabel")}</span>
                <textarea
                  rows={2}
                  value={manualFsRationale}
                  onChange={(e) => setManualFsRationale(e.target.value)}
                  placeholder={t("lab.fewShots.manualRationalePlaceholder")}
                />
              </label>
              <div className="header-actions">
                <button type="submit" className="primary-btn compact">
                  {t("lab.fewShots.saveExampleBtn")}
                </button>
                <button
                  type="button"
                  className="secondary-btn compact"
                  onClick={() => setManualFsOpen(false)}
                >
                  {t("lab.common.cancel")}
                </button>
              </div>
            </form>
          )}

          {fsLoading ? (
            <div className="muted">{t("lab.fewShots.loading")}</div>
          ) : fewShots.length === 0 ? (
            <div className="muted">{t("lab.fewShots.empty")}</div>
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
                      <span>{fs.enabled ? t("lab.fewShots.enabled") : t("lab.fewShots.disabled")}</span>
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
                      alt={t("lab.fewShots.tileAlt", { id: fs.example_id })}
                      className="few-shot-tile-img"
                    />
                  ) : (
                    <div className="few-shot-tile-placeholder">{t("lab.fewShots.textOnly")}</div>
                  )}

                  <div className="two-col">
                    <div className="field-group">
                      <span className="muted">{t("lab.fewShots.vlmWrongCardLabel")}</span>
                      <pre className="json-box">{safeJsonStringify(fs.vlm_wrong_json)}</pre>
                    </div>
                    <div className="field-group">
                      <span className="muted">{t("lab.fewShots.humanGtCardLabel")}</span>
                      <pre className="json-box">{safeJsonStringify(fs.human_right_json)}</pre>
                    </div>
                  </div>

                  <div className="small">
                    <strong>{t("lab.fewShots.cardRationale")}</strong>
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
                {t("lab.taxonomy.subtabLabels")}
              </button>
              <button
                type="button"
                data-testid="taxonomy-subtab-prompts"
                className={`tab-btn ${taxSubTab === "prompts" ? "active" : ""}`}
                onClick={() => setTaxSubTab("prompts")}
              >
                {t("lab.taxonomy.subtabPrompts")}
              </button>
              <button
                type="button"
                data-testid="taxonomy-subtab-tasks"
                className={`tab-btn ${taxSubTab === "tasks" ? "active" : ""}`}
                onClick={() => setTaxSubTab("tasks")}
              >
                {t("lab.taxonomy.subtabTasks")}
              </button>
            </div>
            <div className="header-actions">
              <span className="badge-pill ok">{t("lab.taxonomy.autoArchiveBadge")}</span>
              <button
                type="button"
                className="secondary-btn compact"
                data-testid="open-history-drawer-btn"
                onClick={() => setHistoryOpen(true)}
              >
                <History size={13} /> {t("lab.taxonomy.historyBtn")}
              </button>
            </div>
          </div>

          {taxSubTab === "labels" && (
            <div className="field-group" data-testid="labels-editor">
              <div className="section-header-row">
                <div>
                  <strong>
                    {t("lab.taxonomy.labelsSummary", {
                      version: registry?.version ?? 1,
                      count: labelsList.length,
                    })}
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
                    {t("lab.taxonomy.bumpVersionBtn")}
                  </button>
                  <button
                    type="button"
                    className="accent-btn compact"
                    data-testid="add-label-btn"
                    onClick={() => setNewLabelOpen((v) => !v)}
                  >
                    <Plus size={13} /> {t("lab.taxonomy.addLabelBtn")}
                  </button>
                  <button
                    type="button"
                    className="primary-btn compact"
                    data-testid="save-labels-btn"
                    disabled={savingConfig || !registry}
                    onClick={() => void handleSaveLabels()}
                  >
                    {savingConfig ? t("lab.common.saving") : t("lab.taxonomy.saveLabelsBtn")}
                  </button>
                </div>
              </div>

              {newLabelOpen && (
                <div className="lab-card" data-testid="new-label-form">
                  <div className="three-col">
                    <label className="field-group">
                      <span>{t("lab.taxonomy.labelId")}</span>
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
                      <span>{t("lab.taxonomy.displayName")}</span>
                      <input
                        type="text"
                        value={newLabelDraft.name}
                        onChange={(e) =>
                          setNewLabelDraft({ ...newLabelDraft, name: e.target.value })
                        }
                      />
                    </label>
                    <label className="field-group">
                      <span>{t("lab.taxonomy.category")}</span>
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
                            {t(`lab.category.${c}`)}
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
                      {t("lab.taxonomy.addToListBtn")}
                    </button>
                    <button
                      type="button"
                      className="secondary-btn compact"
                      onClick={() => setNewLabelOpen(false)}
                    >
                      {t("lab.common.cancel")}
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
                      {t("lab.taxonomy.editLabelPrefix")}
                      <span className="mono">{selectedLabel.id}</span>
                    </h4>
                    <div className="two-col">
                      <label className="field-group">
                        <span>{t("lab.taxonomy.displayNameField")}</span>
                        <input
                          type="text"
                          value={selectedLabel.name}
                          onChange={(e) =>
                            updateSelectedLabel({ name: e.target.value })
                          }
                        />
                      </label>
                      <label className="field-group">
                        <span>{t("lab.taxonomy.statusField")}</span>
                        <select
                          value={selectedLabel.status}
                          onChange={(e) =>
                            updateSelectedLabel({ status: e.target.value as LabelStatus })
                          }
                        >
                          <option value="active">{t("lab.labelStatus.active")}</option>
                          <option value="proposed">{t("lab.labelStatus.proposed")}</option>
                          <option value="retired">{t("lab.labelStatus.retired")}</option>
                        </select>
                      </label>
                    </div>
                    <div className="three-col">
                      <label className="field-group">
                        <span>{t("lab.taxonomy.category")}</span>
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
                              {t(`lab.category.${c}`)}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label className="field-group">
                        <span>{t("lab.taxonomy.defaultTrainingRole")}</span>
                        <select
                          value={selectedLabel.default_training_role}
                          onChange={(e) =>
                            updateSelectedLabel({
                              default_training_role: e.target.value as TrainingRole,
                            })
                          }
                        >
                          <option value="necessary">{t("lab.role.necessary")}</option>
                          <option value="recovery">{t("lab.role.recovery")}</option>
                          <option value="badcase">{t("lab.role.badcase")}</option>
                          <option value="terminal_fail">{t("lab.role.terminal_fail")}</option>
                        </select>
                      </label>
                      <label className="field-group">
                        <span>{t("lab.taxonomy.defaultSplicePolicy")}</span>
                        <select
                          value={selectedLabel.default_splice_policy}
                          onChange={(e) =>
                            updateSelectedLabel({
                              default_splice_policy: e.target.value as SplicePolicy,
                            })
                          }
                        >
                          <option value="keep_positive">{t("lab.policy.keep_positive")}</option>
                          <option value="mask_loss">{t("lab.policy.mask_loss")}</option>
                          <option value="cut_clean">{t("lab.policy.cut_clean")}</option>
                        </select>
                      </label>
                    </div>
                    <label className="field-group">
                      <span>{t("lab.taxonomy.uiTitleField")}</span>
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
                      <span>{t("lab.taxonomy.uiHintField")}</span>
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
                      <span>{t("lab.taxonomy.vlmDefField")}</span>
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
                      <span>{t("lab.taxonomy.counterExamplesField")}</span>
                      <textarea
                        rows={2}
                        aria-label={t("lab.taxonomy.counterExamplesAria")}
                        value={selectedLabel.vlm_semantics.counter_examples.join("\n")}
                        onChange={(e) =>
                          updateSelectedLabel({
                            vlm_semantics: {
                              ...selectedLabel.vlm_semantics,
                              counter_examples: e.target.value.split("\n"),
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
                      {t(st.labelKey)}
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
                    <Wand2 size={13} />{" "}
                    {assistLoading ? t("lab.prompts.generating") : t("lab.prompts.assistBtn")}
                  </button>
                  <button
                    type="button"
                    className="primary-btn compact"
                    data-testid="save-prompts-btn"
                    disabled={savingConfig || !prompts}
                    onClick={() => void handleSavePrompts()}
                  >
                    {savingConfig ? t("lab.common.saving") : t("lab.prompts.saveBtn")}
                  </button>
                </div>
              </div>

              <div className="filter-toolbar">
                <label className="field-group">
                  <span>{t("lab.prompts.targetTask")}</span>
                  <input
                    type="text"
                    placeholder={t("lab.prompts.targetTaskPlaceholder")}
                    value={assistTaskId}
                    onChange={(e) => setAssistTaskId(e.target.value)}
                  />
                </label>
                <label className="field-group">
                  <span>{t("lab.prompts.topNErrors")}</span>
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
                      {t("lab.prompts.assistPreviewTitle", {
                        mode: assistPreview.advisory_only
                          ? t("lab.prompts.advisoryOnly")
                          : t("lab.prompts.draft"),
                        stage: assistPreview.stage,
                        version: assistPreview.base_version,
                      })}
                    </strong>
                    <div className="header-actions">
                      <button
                        type="button"
                        className="accent-btn compact"
                        data-testid="apply-assist-draft-btn"
                        onClick={applyAssistDraftLocally}
                      >
                        {t("lab.prompts.applyDraftBtn")}
                      </button>
                      <button
                        type="button"
                        className="mini-btn"
                        onClick={() => setAssistPreview(null)}
                      >
                        {t("lab.prompts.closePreviewBtn")}
                      </button>
                    </div>
                  </div>
                  <div className="small">
                    <strong>{t("lab.prompts.assistRationale")}</strong>
                    {assistPreview.rationale}
                  </div>
                  {assistPreview.task_spec_notes && (
                    <div className="small muted">
                      <strong>{t("lab.prompts.taskSpecNotes")}</strong>
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
                      <span>{t("lab.prompts.versionField")}</span>
                      <input
                        type="text"
                        aria-label={t("lab.prompts.versionAria")}
                        value={activePromptTpl.version}
                        onChange={(e) => updateActivePrompt({ version: e.target.value })}
                      />
                    </label>
                    <label className="field-group">
                      <span>{t("lab.prompts.descriptionField")}</span>
                      <input
                        type="text"
                        value={activePromptTpl.description ?? ""}
                        onChange={(e) => updateActivePrompt({ description: e.target.value })}
                      />
                    </label>
                  </div>
                  <label className="field-group">
                    <span>{t("lab.prompts.systemTemplateField")}</span>
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
                    <span>{t("lab.prompts.userTemplateField")}</span>
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
                <div className="muted">{t("lab.prompts.loading")}</div>
              )}
            </div>
          )}

          {taxSubTab === "tasks" && (
            <div className="field-group" data-testid="tasks-editor">
              <div className="section-header-row">
                <div className="filter-toolbar">
                  <label className="field-group">
                    <span>{t("lab.tasks.selectLabel")}</span>
                    <select
                      aria-label={t("lab.tasks.selectAria")}
                      value={selectedTaskId}
                      onChange={(e) => setSelectedTaskId(e.target.value)}
                    >
                      {Object.keys(tasksMap).map((tid) => (
                        <option key={tid} value={tid}>
                          {tid} ({tasksMap[tid].title || t("lab.tasks.untitled")})
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
                  {savingConfig ? t("lab.common.saving") : t("lab.tasks.saveBtn")}
                </button>
              </div>

              {activeTaskSpec ? (
                <div className="lab-card">
                  <div className="three-col">
                    <label className="field-group">
                      <span>{t("lab.tasks.taskIdField")}</span>
                      <input type="text" readOnly value={activeTaskSpec.task_id} />
                    </label>
                    <label className="field-group">
                      <span>{t("lab.tasks.versionField")}</span>
                      <input
                        type="text"
                        value={activeTaskSpec.version}
                        onChange={(e) => updateActiveTaskSpec({ version: e.target.value })}
                      />
                    </label>
                    <label className="field-group">
                      <span>{t("lab.tasks.titleField")}</span>
                      <input
                        type="text"
                        value={activeTaskSpec.title}
                        onChange={(e) => updateActiveTaskSpec({ title: e.target.value })}
                      />
                    </label>
                  </div>
                  <label className="field-group">
                    <span>{t("lab.tasks.goalField")}</span>
                    <input
                      type="text"
                      value={activeTaskSpec.goal}
                      onChange={(e) => updateActiveTaskSpec({ goal: e.target.value })}
                    />
                  </label>
                  <div className="two-col">
                    <label className="field-group">
                      <span>{t("lab.tasks.criteriaField")}</span>
                      <textarea
                        rows={4}
                        aria-label={t("lab.tasks.criteriaAria")}
                        value={activeTaskSpec.criteria.join("\n")}
                        onChange={(e) =>
                          updateActiveTaskSpec({
                            criteria: e.target.value.split("\n"),
                          })
                        }
                      />
                    </label>
                    <label className="field-group">
                      <span>{t("lab.tasks.stepsField")}</span>
                      <textarea
                        rows={4}
                        aria-label={t("lab.tasks.stepsAria")}
                        value={activeTaskSpec.steps.join("\n")}
                        onChange={(e) =>
                          updateActiveTaskSpec({
                            steps: e.target.value.split("\n"),
                          })
                        }
                      />
                    </label>
                  </div>
                  <label className="field-group">
                    <span>{t("lab.tasks.taskPromptField")}</span>
                    <textarea
                      rows={4}
                      className="code-textarea"
                      value={activeTaskSpec.task_prompt}
                      onChange={(e) => updateActiveTaskSpec({ task_prompt: e.target.value })}
                    />
                  </label>
                </div>
              ) : (
                <div className="muted">{t("lab.tasks.empty")}</div>
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
              <h3>{t("lab.eval.sandboxTitle")}</h3>
              <button
                type="button"
                className="primary-btn compact"
                data-testid="run-eval-btn"
                disabled={evalRunning}
                onClick={() => void handleRunEval()}
              >
                <FlaskConical size={14} />{" "}
                {evalRunning ? t("lab.eval.evaluating") : t("lab.eval.runBtn")}
              </button>
            </div>

            <div className="params-grid">
              <label className="field-group">
                <span>{t("lab.eval.datasetFilter")}</span>
                <input
                  type="text"
                  placeholder={t("lab.eval.datasetPlaceholder")}
                  value={evalDataset}
                  onChange={(e) => setEvalDataset(e.target.value)}
                />
              </label>
              <label className="field-group">
                <span>{t("lab.eval.episodeFilter")}</span>
                <input
                  type="text"
                  placeholder={t("lab.eval.episodePlaceholder")}
                  value={evalEpisodeIds}
                  onChange={(e) => setEvalEpisodeIds(e.target.value)}
                />
              </label>
              <label className="field-group">
                <span>{t("lab.eval.topK")}</span>
                <input
                  type="number"
                  min={0}
                  max={20}
                  value={evalTopK}
                  onChange={(e) => setEvalTopK(Math.max(0, Number(e.target.value) || 0))}
                />
              </label>
              <label className="field-group">
                <span>{t("lab.eval.maxTokens")}</span>
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
              <span>{t("lab.eval.injectFewShot")}</span>
            </label>

            {activeEvalJob &&
              (activeEvalJob.status === "queued" || activeEvalJob.status === "running") && (
                <div className="job-progress-banner" data-testid="eval-job-progress-banner">
                  <div>
                    <strong>{t("lab.eval.jobProgressPrefix", { jobId: activeEvalJob.job_id })}</strong> —{" "}
                    {t(`lab.eval.jobStatus.${activeEvalJob.status}`)} (
                    {activeEvalJob.progress_done}/{activeEvalJob.progress_total})
                    {activeEvalJob.summary?.cache_hit_stages &&
                      Object.keys(activeEvalJob.summary.cache_hit_stages).length > 0 && (
                        <span className="mono small" style={{ marginLeft: 8 }}>
                          {t("lab.eval.cacheHitsPrefix")}{" "}
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
                    {t("lab.eval.cancelBtn")}
                  </button>
                </div>
              )}
          </div>

          <div className="panel-card" data-testid="eval-compare-panel">
            <div className="section-header-row">
              <h3>{t("lab.eval.compareTitle")}</h3>
              <div className="filter-toolbar">
                <label className="field-group">
                  <span>{t("lab.eval.baseline")}</span>
                  <select
                    aria-label={t("lab.eval.baselineAria")}
                    value={baselineId}
                    onChange={(e) => setBaselineId(e.target.value)}
                  >
                    <option value="">{t("lab.eval.selectBaseline")}</option>
                    {(compareData?.evals ?? []).map((ev) => (
                      <option key={ev.eval_id} value={ev.eval_id}>
                        {ev.eval_id} (F1 {fmtNum(ev.metrics.f1)})
                      </option>
                    ))}
                  </select>
                </label>
                <label className="field-group">
                  <span>{t("lab.eval.candidate")}</span>
                  <select
                    aria-label={t("lab.eval.candidateAria")}
                    value={candidateId}
                    onChange={(e) => setCandidateId(e.target.value)}
                  >
                    <option value="">{t("lab.eval.selectCandidate")}</option>
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
                  <GitCompare size={13} /> {t("lab.eval.compareBtn")}
                </button>
              </div>
            </div>

            {(candEval || baseEval) && (
              <div className="table-wrap">
                <table className="data-table" data-testid="eval-compare-table">
                  <thead>
                    <tr>
                      <th>{t("lab.eval.colMetric")}</th>
                      <th>{t("lab.eval.colBaseline", { id: baseEval?.eval_id ?? "—" })}</th>
                      <th>{t("lab.eval.colCandidate", { id: candEval?.eval_id ?? "—" })}</th>
                      <th>{t("lab.eval.colDelta")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>{t("lab.eval.metricF1")}</td>
                      <td className="mono">{fmtNum(baseEval?.metrics.f1)}</td>
                      <td className="mono">{fmtNum(candEval?.metrics.f1)}</td>
                      <td className={`mono ${fmtDelta(deltas?.delta_f1).cls}`}>
                        {fmtDelta(deltas?.delta_f1).text}
                      </td>
                    </tr>
                    <tr>
                      <td>{t("lab.eval.metricPrecRec")}</td>
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
                      <td>{t("lab.eval.metricMeanIou")}</td>
                      <td className="mono">{fmtNum(baseEval?.metrics.mean_iou)}</td>
                      <td className="mono">{fmtNum(candEval?.metrics.mean_iou)}</td>
                      <td className={`mono ${fmtDelta(deltas?.delta_mean_iou).cls}`}>
                        {fmtDelta(deltas?.delta_mean_iou).text}
                      </td>
                    </tr>
                    <tr>
                      <td>{t("lab.eval.metricOvercut")}</td>
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
                      <td>{t("lab.eval.metricGuardrail")}</td>
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
                      <td>{t("lab.eval.metricSuccessAcc")}</td>
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
                      <td>{t("lab.eval.metricTokens")}</td>
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
                      <td>{t("lab.eval.metricCacheHit")}</td>
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
                          : t("lab.eval.cacheReuseFallback")}
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
                  aria-label={t("lab.eval.promoteReasonAria")}
                  placeholder={t("lab.eval.promoteReasonPlaceholder")}
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
                <Rocket size={14} />{" "}
                {promoting ? t("lab.eval.promoting") : t("lab.eval.promoteBtn")}
              </button>
            </div>

            {candEval && candEval.per_episode.length > 0 && (
              <details>
                <summary className="small muted">
                  {t("lab.eval.perEpSummary", { count: candEval.per_episode.length })}
                </summary>
                <div className="table-wrap" style={{ marginTop: 8 }}>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>{t("lab.eval.epColId")}</th>
                        <th>{t("lab.eval.epColF1")}</th>
                        <th>{t("lab.eval.epColIou")}</th>
                        <th>{t("lab.eval.epColOvercut")}</th>
                        <th>{t("lab.eval.epColSuccess")}</th>
                        <th>{t("lab.eval.epColTokens")}</th>
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
                                <CheckCircle2 size={11} /> {t("lab.eval.match")}
                              </span>
                            ) : (
                              <span className="badge-pill warn">{t("lab.eval.mismatch")}</span>
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
            t("lab.toast.rolledBack", {
              path: res.relative_path,
              hash: res.config_sha256.slice(0, 8),
            }),
            "success",
          );
        }}
      />
      {toast.StandaloneHost}
    </div>
  );
};
