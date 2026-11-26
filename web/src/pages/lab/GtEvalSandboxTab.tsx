import React from "react";
import { CheckCircle2, FlaskConical, GitCompare, Rocket } from "lucide-react";
import type { EvalCompareResponse, EvalRunReport, JobStatus } from "../../api/types";
import { useI18n } from "../../stores/i18nStore";
import { fmtCacheHits, fmtDelta, fmtNum, fmtPct } from "./labUtils";

export interface GtEvalSandboxTabProps {
  evalDataset: string;
  setEvalDataset: (val: string) => void;
  evalEpisodeIds: string;
  setEvalEpisodeIds: (val: string) => void;
  evalFewShot: boolean;
  setEvalFewShot: (val: boolean) => void;
  evalTopK: number;
  setEvalTopK: (val: number) => void;
  evalMaxTokens: number;
  setEvalMaxTokens: (val: number) => void;
  evalRunning: boolean;
  onRunEval: () => Promise<void>;
  activeEvalJob: JobStatus | null;
  onCancelEvalJob: (jobId: string) => Promise<void>;
  compareData: EvalCompareResponse | null;
  latestEval: EvalRunReport | null;
  baselineId: string;
  setBaselineId: (val: string) => void;
  candidateId: string;
  setCandidateId: (val: string) => void;
  onCompareEvals: () => Promise<void>;
  promoteReason: string;
  setPromoteReason: (val: string) => void;
  promoting: boolean;
  onPromote: () => Promise<void>;
}

export const GtEvalSandboxTab: React.FC<GtEvalSandboxTabProps> = ({
  evalDataset,
  setEvalDataset,
  evalEpisodeIds,
  setEvalEpisodeIds,
  evalFewShot,
  setEvalFewShot,
  evalTopK,
  setEvalTopK,
  evalMaxTokens,
  setEvalMaxTokens,
  evalRunning,
  onRunEval,
  activeEvalJob,
  onCancelEvalJob,
  compareData,
  latestEval,
  baselineId,
  setBaselineId,
  candidateId,
  setCandidateId,
  onCompareEvals,
  promoteReason,
  setPromoteReason,
  promoting,
  onPromote,
}) => {
  const { t } = useI18n();

  const baseEval = compareData?.baseline ?? null;
  const candEval = compareData?.candidate ?? latestEval ?? null;
  const deltas = compareData?.deltas ?? null;

  return (
    <div className="page-container" data-testid="lab-panel-eval">
      <div className="panel-card" data-testid="eval-runner-controls">
        <div className="section-header-row">
          <h3>{t("lab.eval.sandboxTitle")}</h3>
          <button
            type="button"
            className="primary-btn compact"
            data-testid="run-eval-btn"
            disabled={evalRunning}
            onClick={() => void onRunEval()}
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
                <strong>{t("lab.eval.jobProgressPrefix", { jobId: activeEvalJob.job_id })}</strong>{" "}
                — {t(`lab.eval.jobStatus.${activeEvalJob.status}`)} (
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
                onClick={() => void onCancelEvalJob(activeEvalJob.job_id)}
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
              onClick={() => void onCompareEvals()}
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
                    {fmtDelta(deltas?.delta_precision).text} /{" "}
                    {fmtDelta(deltas?.delta_recall).text}
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
            onClick={() => void onPromote()}
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
  );
};
