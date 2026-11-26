import React, { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  ArrowLeft,
  FilePlus2,
  Plus,
  Save,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";
import { draftLabTask, listEpisodes, putLabTasks } from "../api/client";
import type { GroundingObjectSpec, TaskDraftResponse, TaskSpec } from "../api/types";
import { useI18n } from "../stores/i18nStore";

export interface SampleEpisodeOption {
  episode_id: string;
  task_id?: string;
  dataset?: string;
}

export interface TaskDraftWizardModalProps {
  open: boolean;
  onClose: () => void;
  onTaskSaved: (savedSpec: TaskSpec) => void | Promise<void>;
  initialSampleEpisodeId?: string | null;
  initialTaskId?: string;
  sampleEpisodes?: SampleEpisodeOption[];
}

const SLUG_RE = /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/;

function validateTaskIdSlug(raw: string, t: (key: string) => string): string | null {
  const trimmed = raw.trim();
  if (!trimmed) return t("taskWizard.errTaskIdRequired");
  if (trimmed === "_default") return t("taskWizard.errTaskIdDefault");
  if (!SLUG_RE.test(trimmed)) return t("taskWizard.errTaskIdSlug");
  return null;
}

function splitLines(text: string): string[] {
  return text
    .split("\n")
    .map((s) => s.trim())
    .filter(Boolean);
}

function splitTokens(text: string): string[] {
  return text
    .split(/[,\n]+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

function makeBlankSpec(taskId: string, title: string, description: string): TaskSpec {
  return {
    task_id: taskId,
    version: "1.0.0",
    match: [],
    aliases: [],
    title: title || taskId,
    description,
    goal: "",
    criteria: [],
    steps: [],
    task_prompt: "",
    grounding_objects: [],
    hand_mapping: null,
    camera_roles_override: null,
  };
}

export const TaskDraftWizardModal: React.FC<TaskDraftWizardModalProps> = ({
  open,
  onClose,
  onTaskSaved,
  initialSampleEpisodeId = null,
  initialTaskId = "",
  sampleEpisodes,
}) => {
  const { t } = useI18n();

  const [step, setStep] = useState<"input" | "review">("input");
  const [taskIdInput, setTaskIdInput] = useState("");
  const [titleInput, setTitleInput] = useState("");
  const [descInput, setDescInput] = useState("");
  const [sampleEpisodeId, setSampleEpisodeId] = useState("");
  const [numFrames, setNumFrames] = useState(5);

  const [fetchedEpisodes, setFetchedEpisodes] = useState<SampleEpisodeOption[]>([]);
  const [generating, setGenerating] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [draftMeta, setDraftMeta] = useState<TaskDraftResponse | null>(null);
  const [spec, setSpec] = useState<TaskSpec | null>(null);
  const [stepsText, setStepsText] = useState("");
  const [criteriaText, setCriteriaText] = useState("");
  const [matchText, setMatchText] = useState("");
  const [aliasesText, setAliasesText] = useState("");

  useEffect(() => {
    if (!open) return;
    setStep("input");
    setTaskIdInput(initialTaskId ?? "");
    setTitleInput("");
    setDescInput("");
    setSampleEpisodeId(initialSampleEpisodeId ?? "");
    setNumFrames(5);
    setDraftMeta(null);
    setSpec(null);
    setStepsText("");
    setCriteriaText("");
    setMatchText("");
    setAliasesText("");
    setError(null);
    setGenerating(false);
    setSaving(false);
  }, [open, initialSampleEpisodeId, initialTaskId]);

  useEffect(() => {
    if (!open) return;
    if (sampleEpisodes && sampleEpisodes.length > 0) return;
    let cancelled = false;
    void listEpisodes({ limit: 100 })
      .then((res) => {
        if (!cancelled && Array.isArray(res.items)) {
          setFetchedEpisodes(
            res.items.map((ep) => ({
              episode_id: ep.episode_id,
              task_id: ep.task_id,
              dataset: ep.dataset,
            })),
          );
        }
      })
      .catch(() => {
        // Optional sample list; ignore errors
      });
    return () => {
      cancelled = true;
    };
  }, [open, sampleEpisodes]);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        onClose();
      }
    };
    window.addEventListener("keydown", onKeyDown, true);
    return () => window.removeEventListener("keydown", onKeyDown, true);
  }, [open, onClose]);

  const episodeOptions = useMemo(() => {
    const base =
      sampleEpisodes && sampleEpisodes.length > 0 ? sampleEpisodes : fetchedEpisodes;
    const seen = new Set<string>();
    const list: SampleEpisodeOption[] = [];
    if (initialSampleEpisodeId && initialSampleEpisodeId.trim()) {
      const initId = initialSampleEpisodeId.trim();
      const matched = base.find((e) => e.episode_id === initId);
      list.push(matched ?? { episode_id: initId });
      seen.add(initId);
    }
    for (const item of base) {
      if (!item.episode_id || seen.has(item.episode_id)) continue;
      seen.add(item.episode_id);
      list.push(item);
    }
    return list;
  }, [sampleEpisodes, fetchedEpisodes, initialSampleEpisodeId]);

  if (!open) return null;

  const hydrateReviewStep = (nextSpec: TaskSpec, meta: TaskDraftResponse | null) => {
    setDraftMeta(meta);
    setSpec({
      ...nextSpec,
      match: Array.isArray(nextSpec.match) ? nextSpec.match : [],
      aliases: Array.isArray(nextSpec.aliases) ? nextSpec.aliases : [],
      steps: Array.isArray(nextSpec.steps) ? nextSpec.steps : [],
      criteria: Array.isArray(nextSpec.criteria) ? nextSpec.criteria : [],
      grounding_objects: Array.isArray(nextSpec.grounding_objects)
        ? nextSpec.grounding_objects
        : [],
    });
    setStepsText((nextSpec.steps ?? []).join("\n"));
    setCriteriaText((nextSpec.criteria ?? []).join("\n"));
    setMatchText((nextSpec.match ?? []).join(", "));
    setAliasesText((nextSpec.aliases ?? []).join(", "));
    setStep("review");
  };

  const handleGenerateDraft = async (e: React.FormEvent) => {
    e.preventDefault();
    const idErr = validateTaskIdSlug(taskIdInput, t);
    if (idErr) {
      setError(idErr);
      return;
    }
    const trimmedDesc = descInput.trim();
    if (trimmedDesc.length < 3) {
      setError(t("taskWizard.errDescTooShort"));
      return;
    }

    const trimmedId = taskIdInput.trim();
    const trimmedTitle = titleInput.trim();
    const trimmedSample = sampleEpisodeId.trim();

    setGenerating(true);
    setError(null);
    try {
      const res = await draftLabTask({
        task_id: trimmedId,
        description: trimmedDesc,
        title: trimmedTitle || null,
        sample_episode_id: trimmedSample || null,
        num_frames: trimmedSample ? Math.max(4, Math.min(6, numFrames)) : undefined,
      });
      const resolvedSpec: TaskSpec = {
        ...res.task_spec,
        task_id: res.task_spec.task_id || trimmedId,
      };
      hydrateReviewStep(resolvedSpec, res);
    } catch (err) {
      setError(err instanceof Error ? err.message : t("taskWizard.errDraftFailed"));
    } finally {
      setGenerating(false);
    }
  };

  const handleStartBlankDraft = () => {
    const idErr = validateTaskIdSlug(taskIdInput, t);
    if (idErr) {
      setError(idErr);
      return;
    }
    setError(null);
    const trimmedId = taskIdInput.trim();
    hydrateReviewStep(makeBlankSpec(trimmedId, titleInput.trim(), descInput.trim()), null);
  };

  const updateSpec = (patch: Partial<TaskSpec>) => {
    setSpec((prev) => (prev ? { ...prev, ...patch } : prev));
  };

  const handleAddObject = () => {
    if (!spec) return;
    const nextIdx = spec.grounding_objects.length + 1;
    const nextObj: GroundingObjectSpec = {
      id: `obj_${nextIdx}`,
      name: "",
      desc: "",
      tag_pos: "top",
      is_static: false,
    };
    updateSpec({ grounding_objects: [...spec.grounding_objects, nextObj] });
  };

  const handleUpdateObject = (index: number, patch: Partial<GroundingObjectSpec>) => {
    if (!spec) return;
    updateSpec({
      grounding_objects: spec.grounding_objects.map((obj, idx) =>
        idx === index ? { ...obj, ...patch } : obj,
      ),
    });
  };

  const handleRemoveObject = (index: number) => {
    if (!spec) return;
    updateSpec({
      grounding_objects: spec.grounding_objects.filter((_, idx) => idx !== index),
    });
  };

  const handleSaveSpec = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!spec) return;
    const idErr = validateTaskIdSlug(spec.task_id, t);
    if (idErr) {
      setError(idErr);
      return;
    }
    const cleanId = spec.task_id.trim();
    const cleanedObjects: GroundingObjectSpec[] = spec.grounding_objects
      .map((obj) => ({
        id: obj.id.trim(),
        name: obj.name.trim(),
        desc: obj.desc.trim(),
        tag_pos: (obj.tag_pos === "bottom" ? "bottom" : "top") as "top" | "bottom",
        is_static: Boolean(obj.is_static),
      }))
      .filter((obj) => Boolean(obj.id || obj.name));

    const payload: TaskSpec = {
      ...spec,
      task_id: cleanId,
      version: spec.version.trim() || "1.0.0",
      title: spec.title.trim() || cleanId,
      description: spec.description.trim(),
      goal: spec.goal.trim(),
      steps: splitLines(stepsText),
      criteria: splitLines(criteriaText),
      task_prompt: spec.task_prompt.trim(),
      grounding_objects: cleanedObjects,
      match: splitTokens(matchText),
      aliases: splitTokens(aliasesText),
    };

    setSaving(true);
    setError(null);
    try {
      const res = await putLabTasks(payload);
      const savedSpec = res?.tasks?.[cleanId] ?? payload;
      await onTaskSaved(savedSpec);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : t("taskWizard.errSaveFailed"));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div
      className="modal-backdrop modal-backdrop-top"
      data-testid="task-draft-wizard-modal"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="modal-card modal-card-xl" role="dialog" aria-modal="true" aria-label={t("taskWizard.title")}>
        <div className="section-header-row">
          <h3 style={{ margin: 0, display: "inline-flex", alignItems: "center", gap: 6 }}>
            <Sparkles size={17} /> {t("taskWizard.title")}
          </h3>
          <button
            type="button"
            className="mini-btn"
            onClick={onClose}
            aria-label={t("taskWizard.closeAria")}
            data-testid="close-task-wizard-btn"
          >
            <X size={14} /> {t("common.close")}
          </button>
        </div>

        <div className="wizard-step-bar">
          <span className={`wizard-step-chip ${step === "input" ? "active" : ""}`}>
            {t("taskWizard.step1Label")}
          </span>
          <span className={`wizard-step-chip ${step === "review" ? "active" : ""}`}>
            {t("taskWizard.step2Label")}
          </span>
        </div>

        {step === "input" && (
          <form className="field-group" style={{ gap: 10 }} onSubmit={(e) => void handleGenerateDraft(e)}>
            <p className="muted small" style={{ margin: 0 }}>
              {t("taskWizard.subtitle")}
            </p>

            <div className="two-col">
              <label className="field-group">
                <span>{t("taskWizard.taskIdLabel")}</span>
                <input
                  type="text"
                  data-testid="wizard-task-id-input"
                  value={taskIdInput}
                  onChange={(e) => setTaskIdInput(e.target.value)}
                  placeholder={t("taskWizard.taskIdPlaceholder")}
                  autoFocus
                />
                <span className="muted small">{t("taskWizard.taskIdHint")}</span>
              </label>
              <label className="field-group">
                <span>{t("taskWizard.titleLabel")}</span>
                <input
                  type="text"
                  data-testid="wizard-title-input"
                  value={titleInput}
                  onChange={(e) => setTitleInput(e.target.value)}
                  placeholder={t("taskWizard.titlePlaceholder")}
                />
              </label>
            </div>

            <label className="field-group">
              <span>{t("taskWizard.descLabel")}</span>
              <textarea
                rows={4}
                data-testid="wizard-description-input"
                value={descInput}
                onChange={(e) => setDescInput(e.target.value)}
                placeholder={t("taskWizard.descPlaceholder")}
              />
              <span className="muted small">{t("taskWizard.descHint")}</span>
            </label>

            <div className="two-col">
              <label className="field-group">
                <span>{t("taskWizard.sampleEpLabel")}</span>
                <select
                  data-testid="wizard-sample-episode-select"
                  value={sampleEpisodeId}
                  onChange={(e) => setSampleEpisodeId(e.target.value)}
                >
                  <option value="">{t("taskWizard.sampleEpNone")}</option>
                  {episodeOptions.map((ep) => (
                    <option key={ep.episode_id} value={ep.episode_id}>
                      {ep.episode_id}
                      {ep.task_id ? ` (${ep.task_id})` : ""}
                    </option>
                  ))}
                </select>
                <span className="muted small">{t("taskWizard.sampleEpHint")}</span>
              </label>

              {sampleEpisodeId && (
                <label className="field-group">
                  <span>{t("taskWizard.numFramesLabel")}</span>
                  <input
                    type="number"
                    min={4}
                    max={6}
                    step={1}
                    data-testid="wizard-num-frames-input"
                    value={numFrames}
                    onChange={(e) =>
                      setNumFrames(Math.max(4, Math.min(6, Math.trunc(Number(e.target.value) || 5))))
                    }
                  />
                </label>
              )}
            </div>

            {error && (
              <div className="error-banner" data-testid="wizard-error-banner">
                {error}
              </div>
            )}

            <div className="vlm-modal-footer">
              <button
                type="button"
                className="secondary-btn compact"
                data-testid="wizard-blank-draft-btn"
                disabled={generating}
                onClick={handleStartBlankDraft}
              >
                <FilePlus2 size={13} /> {t("taskWizard.blankDraftBtn")}
              </button>
              <div className="modal-actions">
                <button
                  type="button"
                  className="secondary-btn compact"
                  onClick={onClose}
                  disabled={generating}
                >
                  {t("common.cancel")}
                </button>
                <button
                  type="submit"
                  className="primary-btn compact"
                  data-testid="wizard-generate-draft-btn"
                  disabled={generating}
                >
                  <Sparkles size={13} />{" "}
                  {generating ? t("taskWizard.generating") : t("taskWizard.generateBtn")}
                </button>
              </div>
            </div>
          </form>
        )}

        {step === "review" && spec && (
          <form className="field-group" style={{ gap: 10 }} onSubmit={(e) => void handleSaveSpec(e)}>
            {draftMeta && (
              <div className="lab-card" data-testid="wizard-draft-meta">
                <div className="header-actions">
                  <span className="badge-pill" data-testid="wizard-advisory-badge">
                    {t("taskWizard.advisoryBadge")}
                  </span>
                  {draftMeta.sampled_frames_count > 0 && (
                    <span className="badge-pill ok" data-testid="wizard-sampled-frames-badge">
                      {t("taskWizard.sampledFramesBadge", {
                        count: draftMeta.sampled_frames_count,
                        episodeId: draftMeta.used_sample_episode_id ?? sampleEpisodeId,
                      })}
                    </span>
                  )}
                </div>
                {draftMeta.rationale && (
                  <div className="small" data-testid="wizard-draft-rationale">
                    <strong>{t("taskWizard.rationaleLabel")}</strong> {draftMeta.rationale}
                  </div>
                )}
              </div>
            )}

            {draftMeta && draftMeta.warnings.length > 0 && (
              <div className="warning-banner" data-testid="wizard-draft-warnings">
                <div>
                  <div style={{ display: "inline-flex", alignItems: "center", gap: 5, fontWeight: 600 }}>
                    <AlertTriangle size={14} /> {t("taskWizard.warningsTitle")}
                  </div>
                  <ul style={{ margin: "4px 0 0 18px", padding: 0 }} className="small">
                    {draftMeta.warnings.map((w, idx) => (
                      <li key={idx}>{w}</li>
                    ))}
                  </ul>
                </div>
              </div>
            )}

            <div className="three-col">
              <label className="field-group">
                <span>{t("taskWizard.taskIdLabel")}</span>
                <input
                  type="text"
                  data-testid="wizard-review-task-id"
                  value={spec.task_id}
                  onChange={(e) => updateSpec({ task_id: e.target.value })}
                />
              </label>
              <label className="field-group">
                <span>{t("taskWizard.versionLabel")}</span>
                <input
                  type="text"
                  data-testid="wizard-field-version"
                  value={spec.version}
                  onChange={(e) => updateSpec({ version: e.target.value })}
                />
              </label>
              <label className="field-group">
                <span>{t("taskWizard.reviewTitleLabel")}</span>
                <input
                  type="text"
                  data-testid="wizard-field-title"
                  value={spec.title}
                  onChange={(e) => updateSpec({ title: e.target.value })}
                />
              </label>
            </div>

            <label className="field-group">
              <span>{t("taskWizard.reviewDescLabel")}</span>
              <textarea
                rows={2}
                data-testid="wizard-field-description"
                value={spec.description}
                onChange={(e) => updateSpec({ description: e.target.value })}
              />
            </label>

            <label className="field-group">
              <span>{t("taskWizard.goalLabel")}</span>
              <input
                type="text"
                data-testid="wizard-field-goal"
                value={spec.goal}
                onChange={(e) => updateSpec({ goal: e.target.value })}
              />
            </label>

            <div className="two-col">
              <label className="field-group">
                <span>{t("taskWizard.stepsLabel")}</span>
                <textarea
                  rows={4}
                  data-testid="wizard-field-steps"
                  value={stepsText}
                  onChange={(e) => setStepsText(e.target.value)}
                />
              </label>
              <label className="field-group">
                <span>{t("taskWizard.criteriaLabel")}</span>
                <textarea
                  rows={4}
                  data-testid="wizard-field-criteria"
                  value={criteriaText}
                  onChange={(e) => setCriteriaText(e.target.value)}
                />
              </label>
            </div>

            <label className="field-group">
              <span>{t("taskWizard.taskPromptLabel")}</span>
              <textarea
                rows={3}
                className="code-textarea"
                data-testid="wizard-field-task-prompt"
                value={spec.task_prompt}
                onChange={(e) => updateSpec({ task_prompt: e.target.value })}
              />
            </label>

            <div className="field-group" data-testid="wizard-grounding-objects">
              <div className="section-header-row">
                <span>{t("taskWizard.groundingObjectsLabel")}</span>
                <button
                  type="button"
                  className="mini-btn"
                  data-testid="wizard-add-object-btn"
                  onClick={handleAddObject}
                >
                  <Plus size={12} /> {t("taskWizard.addObjectBtn")}
                </button>
              </div>
              {spec.grounding_objects.length === 0 ? (
                <div className="muted small">{t("taskWizard.noObjectsHint")}</div>
              ) : (
                <div className="wizard-objects-list">
                  {spec.grounding_objects.map((obj, idx) => (
                    <div key={idx} className="wizard-object-row" data-testid={`wizard-object-row-${idx}`}>
                      <div className="wizard-object-row-head">
                        <label className="field-group">
                          <span>{t("taskWizard.objId")}</span>
                          <input
                            type="text"
                            data-testid={`wizard-obj-id-${idx}`}
                            value={obj.id}
                            onChange={(e) => handleUpdateObject(idx, { id: e.target.value })}
                          />
                        </label>
                        <label className="field-group">
                          <span>{t("taskWizard.objName")}</span>
                          <input
                            type="text"
                            data-testid={`wizard-obj-name-${idx}`}
                            value={obj.name}
                            onChange={(e) => handleUpdateObject(idx, { name: e.target.value })}
                          />
                        </label>
                        <label className="field-group">
                          <span>{t("taskWizard.objTagPos")}</span>
                          <select
                            data-testid={`wizard-obj-tag-${idx}`}
                            value={obj.tag_pos}
                            onChange={(e) =>
                              handleUpdateObject(idx, {
                                tag_pos: e.target.value === "bottom" ? "bottom" : "top",
                              })
                            }
                          >
                            <option value="top">{t("taskWizard.objTagTop")}</option>
                            <option value="bottom">{t("taskWizard.objTagBottom")}</option>
                          </select>
                        </label>
                        <button
                          type="button"
                          className="mini-btn"
                          data-testid={`wizard-remove-object-${idx}`}
                          aria-label={t("taskWizard.removeObjectAria", { index: idx + 1 })}
                          onClick={() => handleRemoveObject(idx)}
                        >
                          <Trash2 size={12} />
                        </button>
                      </div>
                      <div className="two-col" style={{ alignItems: "center" }}>
                        <label className="field-group">
                          <span>{t("taskWizard.objDesc")}</span>
                          <input
                            type="text"
                            data-testid={`wizard-obj-desc-${idx}`}
                            value={obj.desc}
                            onChange={(e) => handleUpdateObject(idx, { desc: e.target.value })}
                          />
                        </label>
                        <label className="checkbox-inline" style={{ marginTop: 14 }}>
                          <input
                            type="checkbox"
                            data-testid={`wizard-obj-static-${idx}`}
                            checked={obj.is_static}
                            onChange={(e) =>
                              handleUpdateObject(idx, { is_static: e.target.checked })
                            }
                          />
                          <span>{t("taskWizard.objStatic")}</span>
                        </label>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="two-col">
              <label className="field-group">
                <span>{t("taskWizard.matchLabel")}</span>
                <input
                  type="text"
                  data-testid="wizard-field-match"
                  value={matchText}
                  onChange={(e) => setMatchText(e.target.value)}
                  placeholder={t("taskWizard.matchPlaceholder")}
                />
              </label>
              <label className="field-group">
                <span>{t("taskWizard.aliasesLabel")}</span>
                <input
                  type="text"
                  data-testid="wizard-field-aliases"
                  value={aliasesText}
                  onChange={(e) => setAliasesText(e.target.value)}
                  placeholder={t("taskWizard.aliasesPlaceholder")}
                />
              </label>
            </div>

            {error && (
              <div className="error-banner" data-testid="wizard-error-banner">
                {error}
              </div>
            )}

            <div className="vlm-modal-footer">
              <button
                type="button"
                className="secondary-btn compact"
                data-testid="wizard-back-btn"
                disabled={saving}
                onClick={() => {
                  setError(null);
                  setStep("input");
                }}
              >
                <ArrowLeft size={13} /> {t("taskWizard.backBtn")}
              </button>
              <div className="modal-actions">
                <button
                  type="button"
                  className="secondary-btn compact"
                  onClick={onClose}
                  disabled={saving}
                >
                  {t("common.cancel")}
                </button>
                <button
                  type="submit"
                  className="primary-btn compact"
                  data-testid="wizard-save-task-btn"
                  disabled={saving}
                >
                  <Save size={13} /> {saving ? t("taskWizard.saving") : t("taskWizard.saveBtn")}
                </button>
              </div>
            </div>
          </form>
        )}
      </div>
    </div>
  );
};
