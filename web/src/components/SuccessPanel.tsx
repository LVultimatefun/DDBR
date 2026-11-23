import React from "react";
import { CheckCircle2, Clock, HelpCircle, XCircle } from "lucide-react";
import type { CriterionResult, SuccessVerdict } from "../api/types";

export interface SuccessPanelProps {
  success: SuccessVerdict;
  readOnly?: boolean;
  onChange: (next: SuccessVerdict) => void;
  onSeek: (tS: number) => void;
}

export const SuccessPanel: React.FC<SuccessPanelProps> = ({
  success,
  readOnly = false,
  onChange,
  onSeek,
}) => {
  const handleVerdict = (verdict: SuccessVerdict["verdict"]) => {
    const passed = verdict === "success" ? true : verdict === "failure" ? false : null;
    onChange({ ...success, verdict, passed });
  };

  const updateCriterion = (index: number, patch: Partial<CriterionResult>) => {
    const nextCriteria = success.criteria.map((c, idx) => (idx === index ? { ...c, ...patch } : c));
    onChange({ ...success, criteria: nextCriteria });
  };

  return (
    <div className="success-panel" data-testid="success-panel">
      <div className="verdict-section">
        <div className="section-label">Task Success Verdict (Layer 2)</div>
        <div className="verdict-buttons">
          <button
            type="button"
            disabled={readOnly}
            className={`verdict-btn success ${success.verdict === "success" ? "active" : ""}`}
            onClick={() => handleVerdict("success")}
            data-testid="verdict-success-btn"
          >
            <CheckCircle2 size={15} /> Success
          </button>
          <button
            type="button"
            disabled={readOnly}
            className={`verdict-btn failure ${success.verdict === "failure" ? "active" : ""}`}
            onClick={() => handleVerdict("failure")}
            data-testid="verdict-failure-btn"
          >
            <XCircle size={15} /> Failure
          </button>
          <button
            type="button"
            disabled={readOnly}
            className={`verdict-btn unsure ${success.verdict === "unsure" ? "active" : ""}`}
            onClick={() => handleVerdict("unsure")}
            data-testid="verdict-unsure-btn"
          >
            <HelpCircle size={15} /> Unsure
          </button>
        </div>
      </div>

      <label className="field-group">
        <span>Verdict Summary (summary)</span>
        <textarea
          rows={2}
          disabled={readOnly}
          value={success.summary}
          onChange={(e) => onChange({ ...success, summary: e.target.value })}
          placeholder="Summarize task completion or failure cause..."
        />
      </label>

      <div className="criteria-section">
        <div className="section-label">Completion Criteria ({success.criteria.length})</div>
        {success.criteria.length === 0 ? (
          <div className="empty-hint">No completion criteria recorded.</div>
        ) : (
          <div className="criteria-list">
            {success.criteria.map((item, idx) => (
              <div key={idx} className="criterion-card">
                <div className="criterion-top">
                  <span className="criterion-title">{item.criterion}</span>
                  {item.satisfied_at_s !== null && (
                    <button
                      type="button"
                      className="seek-pill"
                      onClick={() => onSeek(item.satisfied_at_s!)}
                      title="Seek to timestamp where criterion was satisfied"
                    >
                      <Clock size={12} /> {item.satisfied_at_s.toFixed(2)}s
                    </button>
                  )}
                </div>
                <div className="criterion-controls">
                  <select
                    aria-label={`Criterion ${item.criterion} status`}
                    disabled={readOnly}
                    value={item.met === null ? "null" : String(item.met)}
                    onChange={(e) => {
                      const val = e.target.value;
                      updateCriterion(idx, {
                        met: val === "true" ? true : val === "false" ? false : null,
                      });
                    }}
                  >
                    <option value="true">Met</option>
                    <option value="false">Not Met</option>
                    <option value="null">Unsure</option>
                  </select>
                  <input
                    type="text"
                    disabled={readOnly}
                    value={item.evidence}
                    placeholder="Visual evidence description..."
                    onChange={(e) => updateCriterion(idx, { evidence: e.target.value })}
                  />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {success.control_check && (
        <div className="control-check-card" data-testid="control-check-readonly">
          <div className="section-label">S5 Prefix Control Check (Read-Only)</div>
          <div className="diag-grid">
            <span>Control Cut: {success.control_check.cut_s.toFixed(2)}s</span>
            <span>Cut Ratio: {(success.control_check.cut_ratio * 100).toFixed(0)}%</span>
            <span>Control Verdict: {success.control_check.control_verdict}</span>
            <span>Anti-FP Check: {success.control_check.passed ? "Passed" : "Failed"}</span>
          </div>
        </div>
      )}
    </div>
  );
};
