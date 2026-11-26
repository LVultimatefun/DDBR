import React, { useState } from "react";
import { Link } from "react-router-dom";
import { Plus, RefreshCw } from "lucide-react";
import { createFewShot, encodeEpPath, triageUnknown } from "../../api/client";
import type {
  DiagnosisErrorItem,
  DiagnosisReport,
  LabelCategory,
  SplicePolicy,
  TrainingRole,
  UnknownAnomalyItem,
  VisualLabelDef,
} from "../../api/types";
import { useI18n } from "../../stores/i18nStore";
import {
  buildDiagnosisReviewLink,
  DIAGNOSIS_BUCKETS,
  fmtCompactDiagValue,
  LABEL_CATEGORIES,
  safeJsonStringify,
  summarizeStageDiff,
  toJsonDict,
  type TriageDraft,
} from "./labUtils";

export interface DiagnosisTriageTabProps {
  diagnosis: DiagnosisReport | null;
  diagLoading: boolean;
  diagEpFilter: string;
  diagTopN: number;
  onDiagEpFilterChange: (val: string) => void;
  onDiagTopNChange: (val: number) => void;
  onRefreshDiagnosis: () => Promise<void>;
  unknowns: UnknownAnomalyItem[];
  unkTotal: number;
  unkDatasetFilter: string;
  onUnkDatasetFilterChange: (val: string) => void;
  onRefreshUnknowns: () => Promise<void>;
  onReloadLabels: () => Promise<void>;
  labelsList: VisualLabelDef[];
  setErrorMsg: (msg: string | null) => void;
  showLabInfo: (message: string, variant?: "info" | "success" | "warning" | "error") => void;
  clearLabInfo: () => void;
}

export const DiagnosisTriageTab: React.FC<DiagnosisTriageTabProps> = ({
  diagnosis,
  diagLoading,
  diagEpFilter,
  diagTopN,
  onDiagEpFilterChange,
  onDiagTopNChange,
  onRefreshDiagnosis,
  unknowns,
  unkTotal,
  unkDatasetFilter,
  onUnkDatasetFilterChange,
  onRefreshUnknowns,
  onReloadLabels,
  labelsList,
  setErrorMsg,
  showLabInfo,
  clearLabInfo,
}) => {
  const { t } = useI18n();

  const [quickFsError, setQuickFsError] = useState<DiagnosisErrorItem | null>(null);
  const [quickFsScope, setQuickFsScope] = useState("global");
  const [quickFsRationale, setQuickFsRationale] = useState("");
  const [quickFsSaving, setQuickFsSaving] = useState(false);

  const [triageDrafts, setTriageDrafts] = useState<Record<string, TriageDraft>>({});
  const [triageBusyKey, setTriageBusyKey] = useState<string | null>(null);

  const errorItems = diagnosis?.top_errors?.length
    ? diagnosis.top_errors
    : diagnosis?.errors ?? [];

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
      await Promise.all([onRefreshUnknowns(), onReloadLabels()]);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : t("lab.err.triageUnknown"));
    } finally {
      setTriageBusyKey(null);
    }
  };

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

  return (
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
              onChange={(e) => onDiagEpFilterChange(e.target.value)}
            />
            <input
              type="number"
              aria-label={t("lab.diag.topNAria")}
              min={1}
              max={200}
              style={{ width: 84 }}
              value={diagTopN}
              onChange={(e) => onDiagTopNChange(Math.max(1, Number(e.target.value) || 20))}
            />
            <button
              type="button"
              className="primary-btn compact"
              data-testid="refresh-diagnosis-btn"
              disabled={diagLoading}
              onClick={() => void onRefreshDiagnosis()}
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
                  <tr
                    key={
                      err.error_id ||
                      `${err.episode_id}-${err.vlm_seg_id ?? err.gt_seg_id ?? idx}-${err.error_type}`
                    }
                  >
                    <td className="mono small">
                      <Link className="link-btn" to={buildDiagnosisReviewLink(err)}>
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
                          {[
                            err.vlm_seg_id && `VLM:${err.vlm_seg_id}`,
                            err.gt_seg_id && `GT:${err.gt_seg_id}`,
                          ]
                            .filter(Boolean)
                            .join(" · ")}
                        </div>
                      )}
                    </td>
                    <td className="mono small">
                      {err.t_start !== null &&
                      err.t_start !== undefined &&
                      err.t_end !== null &&
                      err.t_end !== undefined
                        ? `${err.t_start.toFixed(2)}s–${err.t_end.toFixed(2)}s`
                        : t("lab.diag.spanGlobal")}
                    </td>
                    <td className="small">
                      <div>{err.summary || t("lab.diag.defaultSummary")}</div>
                      {(err.vlm_value !== undefined || err.gt_value !== undefined) &&
                        (err.error_type === "stage" ? (
                          <div className="diag-stage-diff">
                            {summarizeStageDiff(err.vlm_value, err.gt_value, t).map(
                              (line, lineIdx) => (
                                <div key={lineIdx} className="diag-stage-item">
                                  {line}
                                </div>
                              ),
                            )}
                          </div>
                        ) : (
                          <div className="muted mono">
                            VLM: {fmtCompactDiagValue(err.vlm_value, t)} → GT:{" "}
                            {fmtCompactDiagValue(err.gt_value, t)}
                          </div>
                        ))}
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
              onChange={(e) => onUnkDatasetFilterChange(e.target.value)}
            />
            <button
              type="button"
              className="secondary-btn compact"
              data-testid="refresh-unknowns-btn"
              onClick={() => void onRefreshUnknowns()}
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
                            onChange={(e) => patchTriageDraft(item, { newName: e.target.value })}
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
  );
};
