import React, { useState } from "react";
import { Link } from "react-router-dom";
import { Plus, RefreshCw } from "lucide-react";
import {
  createFewShot,
  encodeEpPath,
  fewShotTileUrl,
  toggleFewShotEnabled,
} from "../../api/client";
import type {
  DiagnosisErrorType,
  FewShotItem,
  TaskSpec,
  VisualLabelDef,
} from "../../api/types";
import { useI18n } from "../../stores/i18nStore";
import { DIAGNOSIS_BUCKETS, parseJsonObject, safeJsonStringify } from "./labUtils";

export interface FewShotBookTabProps {
  fewShots: FewShotItem[];
  setFewShots: React.Dispatch<React.SetStateAction<FewShotItem[]>>;
  fsLoading: boolean;
  fsScopeFilter: string;
  fsTaskFilter: string;
  fsLabelFilter: string;
  fsEnabledOnly: boolean;
  onFsScopeFilterChange: (val: string) => void;
  onFsTaskFilterChange: (val: string) => void;
  onFsLabelFilterChange: (val: string) => void;
  onFsEnabledOnlyChange: (val: boolean) => void;
  onRefreshFewShots: () => Promise<void>;
  tasksMap: Record<string, TaskSpec>;
  labelsList: VisualLabelDef[];
  setErrorMsg: (msg: string | null) => void;
  showLabInfo: (message: string, variant?: "info" | "success" | "warning" | "error") => void;
  clearLabInfo: () => void;
}

export const FewShotBookTab: React.FC<FewShotBookTabProps> = ({
  fewShots,
  setFewShots,
  fsLoading,
  fsScopeFilter,
  fsTaskFilter,
  fsLabelFilter,
  fsEnabledOnly,
  onFsScopeFilterChange,
  onFsTaskFilterChange,
  onFsLabelFilterChange,
  onFsEnabledOnlyChange,
  onRefreshFewShots,
  tasksMap,
  labelsList,
  setErrorMsg,
  showLabInfo,
  clearLabInfo,
}) => {
  const { t } = useI18n();

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
      await onRefreshFewShots();
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : t("lab.err.createFewShot"));
    }
  };

  const handleToggleFewShot = async (item: FewShotItem) => {
    setErrorMsg(null);
    try {
      const updated = await toggleFewShotEnabled(item.example_id, !item.enabled);
      setFewShots((prev) => prev.map((x) => (x.example_id === item.example_id ? updated : x)));
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : t("lab.err.toggleFewShot"));
    }
  };

  return (
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
            onClick={() => void onRefreshFewShots()}
          >
            <RefreshCw size={13} /> {t("lab.common.refresh")}
          </button>
        </div>
      </div>

      <div className="filter-toolbar">
        <select
          aria-label={t("lab.fewShots.filterScopeAria")}
          value={fsScopeFilter}
          onChange={(e) => onFsScopeFilterChange(e.target.value)}
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
          onChange={(e) => onFsTaskFilterChange(e.target.value)}
        />
        <input
          type="text"
          placeholder={t("lab.fewShots.filterLabelPlaceholder")}
          value={fsLabelFilter}
          onChange={(e) => onFsLabelFilterChange(e.target.value)}
        />
        <label className="checkbox-inline">
          <input
            type="checkbox"
            checked={fsEnabledOnly}
            onChange={(e) => onFsEnabledOnlyChange(e.target.checked)}
          />
          <span>{t("lab.fewShots.enabledOnly")}</span>
        </label>
        <button
          type="button"
          className="primary-btn compact"
          onClick={() => void onRefreshFewShots()}
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
                onChange={(e) => setManualFsErrorType(e.target.value as DiagnosisErrorType)}
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
  );
};
