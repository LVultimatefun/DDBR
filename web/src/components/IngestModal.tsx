import React, { useCallback, useEffect, useRef, useState } from "react";
import { ArrowUp, FolderOpen, Sparkles, Upload, X } from "lucide-react";
import { browseIngestDir, createJob, listEpisodeTasks } from "../api/client";
import type {
  EpisodeTaskSummary,
  IngestBrowseResponse,
  JobStatus,
  TaskSpec,
} from "../api/types";
import { useI18n } from "../stores/i18nStore";
import { TaskDraftWizardModal } from "./TaskDraftWizardModal";

export interface IngestModalProps {
  open: boolean;
  onClose: () => void;
  onJobCreated: (job: JobStatus) => void;
}

export const IngestModal: React.FC<IngestModalProps> = ({ open, onClose, onJobCreated }) => {
  const { t } = useI18n();
  const tRef = useRef(t);
  tRef.current = t;
  const [pathInput, setPathInput] = useState("");
  const [browseData, setBrowseData] = useState<IngestBrowseResponse | null>(null);
  const [adapter, setAdapter] = useState<string>("auto");
  const [datasetOverride, setDatasetOverride] = useState("");
  const [taskOverride, setTaskOverride] = useState<string>("auto");
  const [taskCatalog, setTaskCatalog] = useState<EpisodeTaskSummary[]>([]);
  const [wizardOpen, setWizardOpen] = useState(false);
  const [loadingBrowse, setLoadingBrowse] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadDirectory = useCallback(async (targetPath?: string) => {
    setLoadingBrowse(true);
    setError(null);
    try {
      const res = await browseIngestDir(targetPath?.trim() || undefined);
      setBrowseData(res);
      setPathInput(res.current_path || targetPath || "");
    } catch (err) {
      setError(err instanceof Error ? err.message : tRef.current("ingest.errBrowse"));
    } finally {
      setLoadingBrowse(false);
    }
  }, []);

  const loadTasks = useCallback(async () => {
    try {
      const res = await listEpisodeTasks();
      setTaskCatalog(res.tasks ?? []);
    } catch {
      // Keep existing catalog if endpoint unavailable
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    setAdapter("auto");
    setDatasetOverride("");
    setTaskOverride("auto");
    setWizardOpen(false);
    setError(null);
    void loadDirectory();
    void loadTasks();
  }, [open, loadDirectory, loadTasks]);

  useEffect(() => {
    if (!open || wizardOpen) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, wizardOpen, onClose]);

  if (!open) return null;

  const handleTaskCreated = async (savedSpec: TaskSpec) => {
    const summary: EpisodeTaskSummary = {
      task_id: savedSpec.task_id,
      title: savedSpec.title,
      description: savedSpec.description,
      goal: savedSpec.goal,
      match: savedSpec.match ?? [],
      aliases: savedSpec.aliases ?? [],
      grounding_objects_count: (savedSpec.grounding_objects ?? []).length,
    };
    setTaskCatalog((prev) => {
      const filtered = prev.filter((item) => item.task_id !== summary.task_id);
      return [summary, ...filtered];
    });
    setTaskOverride(savedSpec.task_id);
    try {
      const res = await listEpisodeTasks();
      if (res.tasks.length > 0) {
        setTaskCatalog(
          res.tasks.some((item) => item.task_id === summary.task_id)
            ? res.tasks
            : [summary, ...res.tasks.filter((item) => item.task_id !== summary.task_id)],
        );
      }
    } catch {
      // Keep optimistically inserted task
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const targetPath = pathInput.trim();
    if (!targetPath) {
      setError(t("ingest.errNoPath"));
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const job = await createJob({
        job_type: "ingest",
        ingest_params: {
          path: targetPath,
          adapter: adapter || "auto",
          dataset: datasetOverride.trim() || null,
          task_id: taskOverride === "auto" ? null : taskOverride,
        },
      });
      onJobCreated(job);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : t("ingest.errSubmit"));
    } finally {
      setSubmitting(false);
    }
  };

  const dirs = (browseData?.entries ?? []).filter((entry) => entry.is_dir);

  return (
    <>
      <div
        className="modal-backdrop"
        data-testid="ingest-modal"
        onClick={(e) => {
          if (e.target === e.currentTarget && !wizardOpen) onClose();
        }}
      >
        <form className="modal-card modal-card-wide" onSubmit={(e) => void handleSubmit(e)}>
          <div className="section-header-row">
            <h3 style={{ margin: 0, display: "inline-flex", alignItems: "center", gap: 6 }}>
              <FolderOpen size={17} /> {t("ingest.title")}
            </h3>
            <button
              type="button"
              className="mini-btn"
              onClick={onClose}
              aria-label={t("ingest.closeAria")}
              data-testid="close-ingest-modal-btn"
            >
              <X size={14} /> {t("common.close")}
            </button>
          </div>

          <p className="muted small" style={{ margin: 0 }}>
            {t("ingest.desc")}
          </p>

          {browseData && browseData.allowed_roots.length > 0 && (
            <div className="field-group">
              <span>{t("ingest.allowedRoots")}</span>
              <div className="header-actions">
                {browseData.allowed_roots.map((root, idx) => (
                  <button
                    key={root}
                    type="button"
                    className="mini-btn"
                    data-testid={`ingest-root-chip-${idx}`}
                    onClick={() => void loadDirectory(root)}
                  >
                    {root}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="field-group">
            <span>{t("ingest.targetPath")}</span>
            <div className="input-with-btn">
              <input
                type="text"
                data-testid="ingest-path-input"
                value={pathInput}
                onChange={(e) => setPathInput(e.target.value)}
                placeholder="/path/to/dataset"
              />
              <button
                type="button"
                className="secondary-btn compact"
                data-testid="ingest-browse-btn"
                disabled={loadingBrowse}
                onClick={() => void loadDirectory(pathInput)}
              >
                {loadingBrowse ? t("ingest.scanning") : t("ingest.browseBtn")}
              </button>
            </div>
          </div>

          {browseData && (
            <div className="field-group">
              <div className="section-header-row">
                <span>{t("ingest.dirBrowser", { path: browseData.current_path })}</span>
                {browseData.parent_path && (
                  <button
                    type="button"
                    className="mini-btn"
                    data-testid="ingest-parent-btn"
                    onClick={() => void loadDirectory(browseData.parent_path!)}
                  >
                    <ArrowUp size={12} /> {t("ingest.parentDir")}
                  </button>
                )}
              </div>
              <div className="ingest-dir-list" data-testid="ingest-dir-list">
                {dirs.length === 0 ? (
                  <div className="small muted" style={{ padding: "6px 8px" }}>
                    {t("ingest.noSubdirs")}
                  </div>
                ) : (
                  dirs.map((entry) => (
                    <button
                      key={entry.path}
                      type="button"
                      className="ingest-dir-item mono"
                      onClick={() => void loadDirectory(entry.path)}
                    >
                      <FolderOpen size={13} /> {entry.name}
                    </button>
                  ))
                )}
              </div>
            </div>
          )}

          {browseData && browseData.detected_adapters.length > 0 && (
            <div className="field-group" data-testid="ingest-detected-adapters">
              <span>{t("ingest.detectedAdapters")}</span>
              <div className="header-actions">
                {browseData.detected_adapters.map((item) => (
                  <span key={item.adapter_type} className="badge-pill ok">
                    {t("ingest.adapterEpisodes", {
                      adapter: item.adapter_type,
                      count: item.episode_count,
                    })}
                  </span>
                ))}
              </div>
            </div>
          )}

          <div className="two-col">
            <label className="field-group">
              <span>{t("ingest.inputAdapter")}</span>
              <select
                value={adapter}
                onChange={(e) => setAdapter(e.target.value)}
                data-testid="ingest-adapter-select"
              >
                <option value="auto">{t("ingest.adapterAuto")}</option>
                <option value="h5_raw_v3">h5_raw_v3</option>
                <option value="lerobot">lerobot</option>
                <option value="video_dir">video_dir</option>
              </select>
            </label>
            <label className="field-group">
              <span>{t("ingest.datasetOverride")}</span>
              <input
                type="text"
                value={datasetOverride}
                onChange={(e) => setDatasetOverride(e.target.value)}
                placeholder={t("ingest.datasetOverridePlaceholder")}
                data-testid="ingest-dataset-input"
              />
            </label>
          </div>

          <div className="field-group">
            <span>{t("ingest.taskLabel")}</span>
            <div className="input-with-btn">
              <select
                value={taskOverride}
                onChange={(e) => setTaskOverride(e.target.value)}
                data-testid="ingest-task-select"
                style={{ flex: "1 1 auto", minWidth: 0 }}
              >
                <option value="auto">{t("ingest.taskAuto")}</option>
                {taskCatalog.map((task) => (
                  <option key={task.task_id} value={task.task_id}>
                    {task.task_id}
                    {task.title ? ` — ${task.title}` : ""}
                  </option>
                ))}
              </select>
              <button
                type="button"
                className="accent-btn compact"
                data-testid="ingest-new-task-btn"
                onClick={() => setWizardOpen(true)}
              >
                <Sparkles size={13} /> {t("ingest.newTaskAiBtn")}
              </button>
            </div>
          </div>

          {error && <div className="error-banner">{error}</div>}

          <div className="modal-actions">
            <button type="button" className="secondary-btn compact" onClick={onClose}>
              {t("common.cancel")}
            </button>
            <button
              type="submit"
              className="primary-btn compact"
              disabled={submitting}
              data-testid="submit-ingest-btn"
            >
              <Upload size={13} /> {submitting ? t("ingest.submitting") : t("ingest.startBtn")}
            </button>
          </div>
        </form>
      </div>

      <TaskDraftWizardModal
        open={wizardOpen}
        onClose={() => setWizardOpen(false)}
        onTaskSaved={handleTaskCreated}
      />
    </>
  );
};
