import React, { useState } from "react";
import { History, Plus, Sparkles, Wand2 } from "lucide-react";
import {
  listEpisodes,
  putLabLabels,
  putLabPrompts,
  putLabTasks,
  requestPromptAssist,
} from "../../api/client";
import type {
  EpisodeListItem,
  LabelCategory,
  LabelRegistry,
  LabelStatus,
  PromptAssistResponse,
  PromptBundle,
  PromptStageKey,
  SplicePolicy,
  TaskSpec,
  TrainingRole,
  VisualLabelDef,
} from "../../api/types";
import { DEFAULT_VISUAL_LABELS } from "../../constants/labels";
import { useI18n } from "../../stores/i18nStore";
import { LABEL_CATEGORIES, PROMPT_STAGES, type TaxonomySubTab } from "./labUtils";

export interface TaxonomyConfigTabProps {
  taxSubTab: TaxonomySubTab;
  onTaxSubTabChange: (tab: TaxonomySubTab) => void;
  onOpenHistory: () => void;
  registry: LabelRegistry | null;
  setRegistry: React.Dispatch<React.SetStateAction<LabelRegistry | null>>;
  labelsList: VisualLabelDef[];
  selectedLabelId: string;
  setSelectedLabelId: React.Dispatch<React.SetStateAction<string>>;
  prompts: PromptBundle | null;
  setPrompts: React.Dispatch<React.SetStateAction<PromptBundle | null>>;
  tasksMap: Record<string, TaskSpec>;
  setTasksMap: React.Dispatch<React.SetStateAction<Record<string, TaskSpec>>>;
  selectedTaskId: string;
  setSelectedTaskId: React.Dispatch<React.SetStateAction<string>>;
  onOpenTaskWizard: (episodes: EpisodeListItem[]) => void;
  setErrorMsg: (msg: string | null) => void;
  showLabInfo: (message: string, variant?: "info" | "success" | "warning" | "error") => void;
  clearLabInfo: () => void;
}

export const TaxonomyConfigTab: React.FC<TaxonomyConfigTabProps> = ({
  taxSubTab,
  onTaxSubTabChange,
  onOpenHistory,
  registry,
  setRegistry,
  labelsList,
  selectedLabelId,
  setSelectedLabelId,
  prompts,
  setPrompts,
  tasksMap,
  setTasksMap,
  selectedTaskId,
  setSelectedTaskId,
  onOpenTaskWizard,
  setErrorMsg,
  showLabInfo,
  clearLabInfo,
}) => {
  const { t } = useI18n();

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
  const [savingConfig, setSavingConfig] = useState(false);

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
      labels: baseReg.labels.map((l) => (l.id === selectedLabel.id ? { ...l, ...patch } : l)),
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

  return (
    <div className="panel-card" data-testid="lab-panel-taxonomy">
      <div className="section-header-row">
        <div className="tab-group">
          <button
            type="button"
            data-testid="taxonomy-subtab-labels"
            className={`tab-btn ${taxSubTab === "labels" ? "active" : ""}`}
            onClick={() => onTaxSubTabChange("labels")}
          >
            {t("lab.taxonomy.subtabLabels")}
          </button>
          <button
            type="button"
            data-testid="taxonomy-subtab-prompts"
            className={`tab-btn ${taxSubTab === "prompts" ? "active" : ""}`}
            onClick={() => onTaxSubTabChange("prompts")}
          >
            {t("lab.taxonomy.subtabPrompts")}
          </button>
          <button
            type="button"
            data-testid="taxonomy-subtab-tasks"
            className={`tab-btn ${taxSubTab === "tasks" ? "active" : ""}`}
            onClick={() => onTaxSubTabChange("tasks")}
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
            onClick={onOpenHistory}
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
                  registry && setRegistry({ ...registry, version: registry.version + 1 })
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
                    onChange={(e) => setNewLabelDraft({ ...newLabelDraft, id: e.target.value })}
                  />
                </label>
                <label className="field-group">
                  <span>{t("lab.taxonomy.displayName")}</span>
                  <input
                    type="text"
                    value={newLabelDraft.name}
                    onChange={(e) => setNewLabelDraft({ ...newLabelDraft, name: e.target.value })}
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
                <button type="button" className="primary-btn compact" onClick={handleAddNewLabel}>
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
                      onChange={(e) => updateSelectedLabel({ name: e.target.value })}
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
                  onChange={(e) => updateActivePrompt({ system_template: e.target.value })}
                />
              </label>
              <label className="field-group">
                <span>{t("lab.prompts.userTemplateField")}</span>
                <textarea
                  rows={8}
                  className="code-textarea"
                  value={activePromptTpl.user_template}
                  onChange={(e) => updateActivePrompt({ user_template: e.target.value })}
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
            <div className="header-actions">
              <button
                type="button"
                className="accent-btn compact"
                data-testid="lab-new-task-ai-btn"
                onClick={() => {
                  onOpenTaskWizard([]);
                  void listEpisodes({ limit: 100 })
                    .then((res) => onOpenTaskWizard(res.items ?? []))
                    .catch(() => {});
                }}
              >
                <Sparkles size={13} /> {t("lab.tasks.newTaskAiBtn")}
              </button>
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
  );
};
