import React, { useEffect, useMemo, useRef } from "react";
import {
  Check,
  CheckCircle2,
  Clock,
  HelpCircle,
  Plus,
  RefreshCw,
  X,
  XCircle,
} from "lucide-react";
import type { CriterionResult, SuccessVerdict } from "../api/types";
import { useI18n } from "../stores/i18nStore";

export interface SuccessPanelProps {
  success: SuccessVerdict;
  currentTime?: number;
  durationS?: number;
  readOnly?: boolean;
  onChange: (next: SuccessVerdict) => void;
  onSeek: (tS: number) => void;
}

export function deriveVerdictFromCriteria(
  criteria: CriterionResult[],
): SuccessVerdict["verdict"] | null {
  if (criteria.length === 0) return null;
  if (criteria.every((c) => c.met === true)) return "success";
  if (criteria.some((c) => c.met === false)) return "failure";
  return "unsure";
}

interface AutoResizeTextareaProps
  extends Omit<React.TextareaHTMLAttributes<HTMLTextAreaElement>, "value"> {
  value: string;
}

const AutoResizeTextarea: React.FC<AutoResizeTextareaProps> = ({ value, style, ...rest }) => {
  const ref = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    if (el.scrollHeight > 0) {
      el.style.height = `${el.scrollHeight}px`;
    }
  }, [value]);

  return <textarea ref={ref} value={value} style={style} {...rest} />;
};

function clampTime(t: number, durationS?: number): number {
  const nonNeg = Math.max(0, Number.isFinite(t) ? t : 0);
  const clamped =
    durationS !== undefined && Number.isFinite(durationS) && durationS > 0
      ? Math.min(durationS, nonNeg)
      : nonNeg;
  return Number(clamped.toFixed(2));
}

export const SuccessPanel: React.FC<SuccessPanelProps> = ({
  success,
  currentTime = 0,
  durationS,
  readOnly = false,
  onChange,
  onSeek,
}) => {
  const { t } = useI18n();
  const clampedNow = clampTime(currentTime, durationS);

  const verdictLabel = (v: SuccessVerdict["verdict"]) => t(`success.verdict.${v}`);

  const handleVerdict = (verdict: SuccessVerdict["verdict"]) => {
    const passed = verdict === "success" ? true : verdict === "failure" ? false : null;
    onChange({ ...success, verdict, passed });
  };

  const updateCriterion = (index: number, patch: Partial<CriterionResult>) => {
    const nextCriteria = success.criteria.map((c, idx) =>
      idx === index ? { ...c, ...patch } : c,
    );
    onChange({ ...success, criteria: nextCriteria });
  };

  const handleAddCriterion = () => {
    if (readOnly) return;
    const nextCriterion: CriterionResult = {
      criterion: t("success.defaultCriterionName", { index: success.criteria.length + 1 }),
      met: null,
      satisfied_at_s: null,
      evidence: "",
    };
    onChange({
      ...success,
      criteria: [...success.criteria, nextCriterion],
    });
  };

  const handleDeleteCriterion = (index: number) => {
    if (readOnly) return;
    onChange({
      ...success,
      criteria: success.criteria.filter((_, idx) => idx !== index),
    });
  };

  const suggestedVerdict = useMemo(
    () => deriveVerdictFromCriteria(success.criteria),
    [success.criteria],
  );
  const showSyncBar = suggestedVerdict !== null && suggestedVerdict !== success.verdict;

  return (
    <div className="success-panel" data-testid="success-panel">
      <div className="verdict-section">
        <div className="section-label">{t("success.verdictSectionTitle")}</div>
        <div className="verdict-buttons">
          <button
            type="button"
            disabled={readOnly}
            aria-pressed={success.verdict === "success"}
            className={`verdict-btn success ${success.verdict === "success" ? "active" : ""}`}
            onClick={() => handleVerdict("success")}
            data-testid="verdict-success-btn"
          >
            <CheckCircle2 size={15} /> {t("success.verdict.success")}
          </button>
          <button
            type="button"
            disabled={readOnly}
            aria-pressed={success.verdict === "failure"}
            className={`verdict-btn failure ${success.verdict === "failure" ? "active" : ""}`}
            onClick={() => handleVerdict("failure")}
            data-testid="verdict-failure-btn"
          >
            <XCircle size={15} /> {t("success.verdict.failure")}
          </button>
          <button
            type="button"
            disabled={readOnly}
            aria-pressed={success.verdict === "unsure"}
            className={`verdict-btn unsure ${success.verdict === "unsure" ? "active" : ""}`}
            onClick={() => handleVerdict("unsure")}
            data-testid="verdict-unsure-btn"
          >
            <HelpCircle size={15} /> {t("success.verdict.unsure")}
          </button>
        </div>
      </div>

      {showSyncBar && suggestedVerdict && (
        <div
          className={`verdict-sync-bar suggest-${suggestedVerdict}`}
          data-testid="verdict-sync-bar"
          role="status"
        >
          <span className="verdict-sync-text">
            {t("success.syncSuggest")} <strong>{verdictLabel(suggestedVerdict)}</strong>
            <span className="verdict-sync-current">
              {t("success.syncCurrent", { verdict: verdictLabel(success.verdict) })}
            </span>
          </span>
          {!readOnly && (
            <button
              type="button"
              className="mini-btn verdict-sync-btn"
              data-testid="sync-verdict-btn"
              onClick={() => handleVerdict(suggestedVerdict)}
            >
              <RefreshCw size={12} /> {t("success.syncBtn")}
            </button>
          )}
        </div>
      )}

      <label className="field-group">
        <span>{t("success.summaryLabel")}</span>
        <AutoResizeTextarea
          rows={3}
          className="verdict-summary-textarea"
          data-testid="verdict-summary-textarea"
          disabled={readOnly}
          value={success.summary}
          onChange={(e) => onChange({ ...success, summary: e.target.value })}
          placeholder={t("success.summaryPlaceholder")}
        />
      </label>

      <div className="criteria-section">
        <div className="criteria-header">
          <div className="section-label">
            {t("success.criteriaHeading", { count: success.criteria.length })}
          </div>
          {!readOnly && (
            <button
              type="button"
              className="mini-btn add-criterion-btn"
              data-testid="add-criterion-btn"
              onClick={handleAddCriterion}
            >
              <Plus size={13} /> {t("success.addCriterion")}
            </button>
          )}
        </div>
        {success.criteria.length === 0 ? (
          <div className="empty-hint">{t("success.emptyCriteria")}</div>
        ) : (
          <div className="criteria-list">
            {success.criteria.map((item, idx) => (
              <div key={idx} className="criterion-card" data-testid={`criterion-card-${idx}`}>
                <div className="criterion-top">
                  <AutoResizeTextarea
                    rows={1}
                    className="criterion-title-input"
                    aria-label={t("success.criterionTitleAria", { index: idx + 1 })}
                    data-testid={`criterion-title-${idx}`}
                    disabled={readOnly}
                    value={item.criterion}
                    placeholder={t("success.criterionTitlePlaceholder")}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault();
                        e.currentTarget.blur();
                      }
                    }}
                    onChange={(e) => updateCriterion(idx, { criterion: e.target.value })}
                  />
                  {!readOnly && (
                    <button
                      type="button"
                      className="criterion-delete-btn"
                      data-testid={`delete-criterion-${idx}`}
                      aria-label={t("success.deleteCriterionAria", { index: idx + 1 })}
                      title={t("success.deleteCriterionTitle")}
                      onClick={() => handleDeleteCriterion(idx)}
                    >
                      <X size={13} />
                    </button>
                  )}
                </div>

                <div className="criterion-controls-row">
                  <div
                    className="tri-state-group"
                    role="group"
                    aria-label={t("success.statusGroupAria", {
                      name: item.criterion || `#${idx + 1}`,
                    })}
                    data-testid={`criterion-status-group-${idx}`}
                  >
                    <button
                      type="button"
                      disabled={readOnly}
                      aria-pressed={item.met === true}
                      aria-label={t("success.statusMetAria", { index: idx + 1 })}
                      data-testid={`criterion-${idx}-met`}
                      className={`tri-state-btn met ${item.met === true ? "active" : ""}`}
                      onClick={() => updateCriterion(idx, { met: true })}
                    >
                      <Check size={12} /> {t("success.statusMet")}
                    </button>
                    <button
                      type="button"
                      disabled={readOnly}
                      aria-pressed={item.met === false}
                      aria-label={t("success.statusNotMetAria", { index: idx + 1 })}
                      data-testid={`criterion-${idx}-not-met`}
                      className={`tri-state-btn not-met ${item.met === false ? "active" : ""}`}
                      onClick={() => updateCriterion(idx, { met: false })}
                    >
                      <X size={12} /> {t("success.statusNotMet")}
                    </button>
                    <button
                      type="button"
                      disabled={readOnly}
                      aria-pressed={item.met === null}
                      aria-label={t("success.statusUnsureAria", { index: idx + 1 })}
                      data-testid={`criterion-${idx}-unsure`}
                      className={`tri-state-btn unsure ${item.met === null ? "active" : ""}`}
                      onClick={() => updateCriterion(idx, { met: null })}
                    >
                      <HelpCircle size={12} /> {t("success.statusUnsure")}
                    </button>
                  </div>

                  <div className="criterion-time-row" data-testid={`criterion-time-row-${idx}`}>
                    {item.satisfied_at_s !== null ? (
                      <>
                        <button
                          type="button"
                          className="seek-pill"
                          data-testid={`criterion-seek-${idx}`}
                          onClick={() => onSeek(item.satisfied_at_s!)}
                          title={t("success.seekTitle")}
                        >
                          <Clock size={12} /> {item.satisfied_at_s.toFixed(2)}s
                        </button>
                        {!readOnly && (
                          <>
                            <input
                              type="number"
                              className="criterion-time-input"
                              aria-label={t("success.timestampAria", { index: idx + 1 })}
                              data-testid={`criterion-time-input-${idx}`}
                              step={0.05}
                              min={0}
                              max={durationS}
                              value={item.satisfied_at_s}
                              onChange={(e) => {
                                const raw = e.target.value;
                                if (raw.trim() === "") return;
                                const parsed = Number(raw);
                                if (!Number.isFinite(parsed)) return;
                                updateCriterion(idx, {
                                  satisfied_at_s: clampTime(parsed, durationS),
                                });
                              }}
                            />
                            <button
                              type="button"
                              className="criterion-time-btn"
                              data-testid={`criterion-now-${idx}`}
                              title={t("success.nowTitle", { time: clampedNow.toFixed(2) })}
                              onClick={() =>
                                updateCriterion(idx, { satisfied_at_s: clampedNow })
                              }
                            >
                              {t("success.nowBtn")}
                            </button>
                            <button
                              type="button"
                              className="criterion-time-clear"
                              data-testid={`criterion-clear-time-${idx}`}
                              aria-label={t("success.clearTimeAria", { index: idx + 1 })}
                              title={t("success.clearTimeTitle")}
                              onClick={() => updateCriterion(idx, { satisfied_at_s: null })}
                            >
                              ×
                            </button>
                          </>
                        )}
                      </>
                    ) : !readOnly ? (
                      <button
                        type="button"
                        className="criterion-set-now-btn"
                        data-testid={`criterion-set-now-${idx}`}
                        title={t("success.setAtNowTitle")}
                        onClick={() => updateCriterion(idx, { satisfied_at_s: clampedNow })}
                      >
                        <Clock size={12} />{" "}
                        {t("success.setAtNow", { time: clampedNow.toFixed(2) })}
                      </button>
                    ) : (
                      <span className="muted small">{t("success.noTimestamp")}</span>
                    )}
                  </div>
                </div>

                <AutoResizeTextarea
                  rows={2}
                  className="criterion-evidence"
                  aria-label={t("success.evidenceAria", { index: idx + 1 })}
                  data-testid={`criterion-evidence-${idx}`}
                  disabled={readOnly}
                  value={item.evidence}
                  placeholder={t("success.evidencePlaceholder")}
                  onChange={(e) => updateCriterion(idx, { evidence: e.target.value })}
                />
              </div>
            ))}
          </div>
        )}
      </div>

      {success.control_check && (
        <div className="control-check-card" data-testid="control-check-readonly">
          <div className="section-label">{t("success.controlCheckTitle")}</div>
          <div className="diag-grid">
            <span>{t("success.controlCut", { cut: success.control_check.cut_s.toFixed(2) })}</span>
            <span>
              {t("success.cutRatio", {
                ratio: (success.control_check.cut_ratio * 100).toFixed(0),
              })}
            </span>
            <span>
              {t("success.controlVerdict", {
                verdict: verdictLabel(success.control_check.control_verdict),
              })}
            </span>
            <span>
              {t("success.antiFpCheck", {
                result: success.control_check.passed ? t("success.passed") : t("success.failed"),
              })}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
