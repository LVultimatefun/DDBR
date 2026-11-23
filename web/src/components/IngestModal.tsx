import React, { useCallback, useEffect, useState } from "react";
import { ArrowUp, FolderOpen, Upload, X } from "lucide-react";
import { browseIngestDir, createJob } from "../api/client";
import type { IngestBrowseResponse, JobStatus } from "../api/types";

export interface IngestModalProps {
  open: boolean;
  onClose: () => void;
  onJobCreated: (job: JobStatus) => void;
}

export const IngestModal: React.FC<IngestModalProps> = ({ open, onClose, onJobCreated }) => {
  const [pathInput, setPathInput] = useState("");
  const [browseData, setBrowseData] = useState<IngestBrowseResponse | null>(null);
  const [adapter, setAdapter] = useState<string>("auto");
  const [datasetOverride, setDatasetOverride] = useState("");
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
      setError(err instanceof Error ? err.message : "Failed to browse directory");
    } finally {
      setLoadingBrowse(false);
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    setAdapter("auto");
    setDatasetOverride("");
    setError(null);
    void loadDirectory();
  }, [open, loadDirectory]);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const targetPath = pathInput.trim();
    if (!targetPath) {
      setError("Please enter or select a dataset directory path to ingest");
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
        },
      });
      onJobCreated(job);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to submit ingest job");
    } finally {
      setSubmitting(false);
    }
  };

  const dirs = (browseData?.entries ?? []).filter((entry) => entry.is_dir);

  return (
    <div
      className="modal-backdrop"
      data-testid="ingest-modal"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <form className="modal-card modal-card-wide" onSubmit={(e) => void handleSubmit(e)}>
        <div className="section-header-row">
          <h3 style={{ margin: 0, display: "inline-flex", alignItems: "center", gap: 6 }}>
            <FolderOpen size={17} /> Safe Data Ingest (Web Ingest)
          </h3>
          <button
            type="button"
            className="mini-btn"
            onClick={onClose}
            aria-label="Close ingest modal"
            data-testid="close-ingest-modal-btn"
          >
            <X size={14} /> Close
          </button>
        </div>

        <p className="muted small" style={{ margin: 0 }}>
          Read-only scan of allowlisted data directories into SQLite; raw dataset files are never
          modified.
        </p>

        {browseData && browseData.allowed_roots.length > 0 && (
          <div className="field-group">
            <span>Allowlisted Roots (allowed_roots)</span>
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
          <span>Target Data Path</span>
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
              {loadingBrowse ? "Scanning..." : "Browse / Probe"}
            </button>
          </div>
        </div>

        {browseData && (
          <div className="field-group">
            <div className="section-header-row">
              <span>Directory Browser ({browseData.current_path})</span>
              {browseData.parent_path && (
                <button
                  type="button"
                  className="mini-btn"
                  data-testid="ingest-parent-btn"
                  onClick={() => void loadDirectory(browseData.parent_path!)}
                >
                  <ArrowUp size={12} /> Parent Directory
                </button>
              )}
            </div>
            <div className="ingest-dir-list" data-testid="ingest-dir-list">
              {dirs.length === 0 ? (
                <div className="small muted" style={{ padding: "6px 8px" }}>
                  No subdirectories in current folder
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
            <span>Detected Adapters</span>
            <div className="header-actions">
              {browseData.detected_adapters.map((item) => (
                <span key={item.adapter_type} className="badge-pill ok">
                  {item.adapter_type} · {item.episode_count} episodes
                </span>
              ))}
            </div>
          </div>
        )}

        <div className="two-col">
          <label className="field-group">
            <span>Input Adapter (adapter)</span>
            <select
              value={adapter}
              onChange={(e) => setAdapter(e.target.value)}
              data-testid="ingest-adapter-select"
            >
              <option value="auto">auto (Auto-detect)</option>
              <option value="h5_raw_v3">h5_raw_v3</option>
              <option value="lerobot">lerobot</option>
              <option value="video_dir">video_dir</option>
            </select>
          </label>
          <label className="field-group">
            <span>Dataset Name Override (optional)</span>
            <input
              type="text"
              value={datasetOverride}
              onChange={(e) => setDatasetOverride(e.target.value)}
              placeholder="Leave blank to use folder name"
              data-testid="ingest-dataset-input"
            />
          </label>
        </div>

        {error && <div className="error-banner">{error}</div>}

        <div className="modal-actions">
          <button type="button" className="secondary-btn compact" onClick={onClose}>
            Cancel
          </button>
          <button
            type="submit"
            className="primary-btn compact"
            disabled={submitting}
            data-testid="submit-ingest-btn"
          >
            <Upload size={13} /> {submitting ? "Submitting..." : "Start Async Ingest"}
          </button>
        </div>
      </form>
    </div>
  );
};
