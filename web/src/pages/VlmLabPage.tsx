import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Bookmark,
  FlaskConical,
  Layers,
  ShieldOff,
  Sparkles,
} from "lucide-react";
import {
  ApiError,
  cancelJob as apiCancelJob,
  compareLabEvals,
  createJob,
  getJob,
  getLabDiagnosis,
  getLabLabels,
  getLabPrompts,
  getLabTasks,
  getLabUnknowns,
  listFewShots,
  promoteLabConfig,
} from "../api/client";
import type {
  DiagnosisReport,
  EpisodeListItem,
  EvalCompareResponse,
  EvalRunReport,
  FewShotItem,
  JobStatus,
  LabelRegistry,
  PromptBundle,
  TaskSpec,
  UnknownAnomalyItem,
} from "../api/types";
import { useEventStream } from "../api/useEventStream";
import { LabHistoryDrawer } from "../components/LabHistoryDrawer";
import { TaskDraftWizardModal } from "../components/TaskDraftWizardModal";
import { DEFAULT_VISUAL_LABELS } from "../constants/labels";
import { useAuth } from "../stores/authStore";
import { useI18n } from "../stores/i18nStore";
import { useJobs } from "../stores/jobStore";
import { useToast } from "../stores/toastStore";
import { DiagnosisTriageTab } from "./lab/DiagnosisTriageTab";
import { FewShotBookTab } from "./lab/FewShotBookTab";
import { GtEvalSandboxTab } from "./lab/GtEvalSandboxTab";
import { fmtPct, type LabTab, type TaxonomySubTab } from "./lab/labUtils";
import { TaxonomyConfigTab } from "./lab/TaxonomyConfigTab";

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
  const [taxSubTab, setTaxSubTab] = useState<TaxonomySubTab>("labels");
  const [selectedLabelId, setSelectedLabelId] = useState<string>("boundary_idle");
  const [selectedTaskId, setSelectedTaskId] = useState<string>("_default");
  const [taskWizardOpen, setTaskWizardOpen] = useState(false);
  const [wizardEpisodes, setWizardEpisodes] = useState<EpisodeListItem[]>([]);

  // Tab 1: Diagnosis + Unknown Triage
  const [diagEpFilter, setDiagEpFilter] = useState("");
  const [diagTopN, setDiagTopN] = useState(20);
  const [diagnosis, setDiagnosis] = useState<DiagnosisReport | null>(null);
  const [diagLoading, setDiagLoading] = useState(false);
  const [unkDatasetFilter, setUnkDatasetFilter] = useState("");
  const [unknowns, setUnknowns] = useState<UnknownAnomalyItem[]>([]);
  const [unkTotal, setUnkTotal] = useState(0);

  // Tab 2: Few-Shot Mistake Book
  const [fsScopeFilter, setFsScopeFilter] = useState("");
  const [fsTaskFilter, setFsTaskFilter] = useState("");
  const [fsLabelFilter, setFsLabelFilter] = useState("");
  const [fsEnabledOnly, setFsEnabledOnly] = useState(false);
  const [fewShots, setFewShots] = useState<FewShotItem[]>([]);
  const [fsLoading, setFsLoading] = useState(false);

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

  const labelsList = useMemo(() => registry?.labels ?? DEFAULT_VISUAL_LABELS, [registry]);

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

  const loadEvalCompare = useCallback(async (base?: string, cand?: string) => {
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
  }, []);

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
      if (
        jobEvt.status === "completed" ||
        jobEvt.status === "failed" ||
        jobEvt.status === "cancelled"
      ) {
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

      {activeTab === "diagnosis" && (
        <DiagnosisTriageTab
          diagnosis={diagnosis}
          diagLoading={diagLoading}
          diagEpFilter={diagEpFilter}
          diagTopN={diagTopN}
          onDiagEpFilterChange={setDiagEpFilter}
          onDiagTopNChange={setDiagTopN}
          onRefreshDiagnosis={loadDiagnosis}
          unknowns={unknowns}
          unkTotal={unkTotal}
          unkDatasetFilter={unkDatasetFilter}
          onUnkDatasetFilterChange={setUnkDatasetFilter}
          onRefreshUnknowns={loadUnknowns}
          onReloadLabels={loadLabels}
          labelsList={labelsList}
          setErrorMsg={setErrorMsg}
          showLabInfo={showLabInfo}
          clearLabInfo={clearLabInfo}
        />
      )}

      {activeTab === "few_shots" && (
        <FewShotBookTab
          fewShots={fewShots}
          setFewShots={setFewShots}
          fsLoading={fsLoading}
          fsScopeFilter={fsScopeFilter}
          fsTaskFilter={fsTaskFilter}
          fsLabelFilter={fsLabelFilter}
          fsEnabledOnly={fsEnabledOnly}
          onFsScopeFilterChange={setFsScopeFilter}
          onFsTaskFilterChange={setFsTaskFilter}
          onFsLabelFilterChange={setFsLabelFilter}
          onFsEnabledOnlyChange={setFsEnabledOnly}
          onRefreshFewShots={loadFewShots}
          tasksMap={tasksMap}
          labelsList={labelsList}
          setErrorMsg={setErrorMsg}
          showLabInfo={showLabInfo}
          clearLabInfo={clearLabInfo}
        />
      )}

      {activeTab === "taxonomy" && (
        <TaxonomyConfigTab
          taxSubTab={taxSubTab}
          onTaxSubTabChange={setTaxSubTab}
          onOpenHistory={() => setHistoryOpen(true)}
          registry={registry}
          setRegistry={setRegistry}
          labelsList={labelsList}
          selectedLabelId={selectedLabelId}
          setSelectedLabelId={setSelectedLabelId}
          prompts={prompts}
          setPrompts={setPrompts}
          tasksMap={tasksMap}
          setTasksMap={setTasksMap}
          selectedTaskId={selectedTaskId}
          setSelectedTaskId={setSelectedTaskId}
          onOpenTaskWizard={(eps) => {
            setTaskWizardOpen(true);
            if (eps.length > 0) setWizardEpisodes(eps);
          }}
          setErrorMsg={setErrorMsg}
          showLabInfo={showLabInfo}
          clearLabInfo={clearLabInfo}
        />
      )}

      {activeTab === "eval" && (
        <GtEvalSandboxTab
          evalDataset={evalDataset}
          setEvalDataset={setEvalDataset}
          evalEpisodeIds={evalEpisodeIds}
          setEvalEpisodeIds={setEvalEpisodeIds}
          evalFewShot={evalFewShot}
          setEvalFewShot={setEvalFewShot}
          evalTopK={evalTopK}
          setEvalTopK={setEvalTopK}
          evalMaxTokens={evalMaxTokens}
          setEvalMaxTokens={setEvalMaxTokens}
          evalRunning={evalRunning}
          onRunEval={handleRunEval}
          activeEvalJob={activeEvalJob}
          onCancelEvalJob={handleCancelEvalJob}
          compareData={compareData}
          latestEval={latestEval}
          baselineId={baselineId}
          setBaselineId={setBaselineId}
          candidateId={candidateId}
          setCandidateId={setCandidateId}
          onCompareEvals={handleCompareEvals}
          promoteReason={promoteReason}
          setPromoteReason={setPromoteReason}
          promoting={promoting}
          onPromote={handlePromote}
        />
      )}

      <TaskDraftWizardModal
        open={taskWizardOpen}
        onClose={() => setTaskWizardOpen(false)}
        sampleEpisodes={wizardEpisodes}
        onTaskSaved={async (savedSpec) => {
          setTasksMap((prev) => ({ ...prev, [savedSpec.task_id]: savedSpec }));
          setSelectedTaskId(savedSpec.task_id);
          void loadTasks().catch(() => {});
          showLabInfo(
            t("lab.toast.taskSaved", {
              taskId: savedSpec.task_id,
              version: savedSpec.version,
            }),
            "success",
          );
        }}
      />

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

export default VlmLabPage;
