import React, { useCallback, useEffect, useState } from "react";
import { History, RefreshCw, RotateCcw, X } from "lucide-react";
import { getLabHistoryDiff, listLabHistory, rollbackLabHistory } from "../api/client";
import type {
  LabHistoryDiffResponse,
  LabHistoryItem,
  LabHistoryRollbackResponse,
  LabHistoryTarget,
} from "../api/types";

export interface LabHistoryDrawerProps {
  open: boolean;
  target: LabHistoryTarget;
  onClose: () => void;
  onRolledBack: (res: LabHistoryRollbackResponse) => void | Promise<void>;
}

const TARGET_TABS: Array<{ key: LabHistoryTarget; label: string }> = [
  { key: "labels", label: "Labels (labels)" },
  { key: "prompts", label: "Prompt Templates" },
  { key: "tasks", label: "TaskSpec" },
];

function fmtTimestamp(ts: number): string {
  if (!Number.isFinite(ts) || ts <= 0) return "—";
  const ms = ts > 1e12 ? ts : ts * 1000;
  try {
    return new Date(ms).toLocaleString();
  } catch {
    return String(ts);
  }
}

export const LabHistoryDrawer: React.FC<LabHistoryDrawerProps> = ({
  open,
  target,
  onClose,
  onRolledBack,
}) => {
  const [activeTarget, setActiveTarget] = useState<LabHistoryTarget>(target);
  const [items, setItems] = useState<LabHistoryItem[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [diff, setDiff] = useState<LabHistoryDiffResponse | null>(null);
  const [loadingList, setLoadingList] = useState(false);
  const [loadingDiff, setLoadingDiff] = useState(false);
  const [rollingBack, setRollingBack] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setActiveTarget(target);
    }
  }, [open, target]);

  const loadList = useCallback(async (tgt: LabHistoryTarget) => {
    setLoadingList(true);
    setError(null);
    setSelectedId(null);
    setDiff(null);
    try {
      const res = await listLabHistory({ target: tgt });
      setItems(res.items);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load .history/ snapshots");
    } finally {
      setLoadingList(false);
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    void loadList(activeTarget);
  }, [open, activeTarget, loadList]);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  const handleSelectSnapshot = async (snapshotId: string) => {
    setSelectedId(snapshotId);
    setLoadingDiff(true);
    setError(null);
    try {
      const res = await getLabHistoryDiff(snapshotId);
      setDiff(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load snapshot diff");
    } finally {
      setLoadingDiff(false);
    }
  };

  const handleRollback = async () => {
    if (!selectedId) return;
    setRollingBack(true);
    setError(null);
    try {
      const res = await rollbackLabHistory(selectedId);
      await onRolledBack(res);
      await loadList(activeTarget);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to roll back snapshot");
    } finally {
      setRollingBack(false);
    }
  };

  if (!open) return null;

  return (
    <div
      className="drawer-backdrop"
      data-testid="lab-history-drawer-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <aside
        className="drawer-panel drawer-wide"
        data-testid="lab-history-drawer"
        aria-label=".history/ version browser and rollback"
      >
        <div className="drawer-header">
          <h3>
            <History size={16} /> .history/ Config Snapshots &amp; Rollback
          </h3>
          <div className="header-actions">
            <button
              type="button"
              className="secondary-btn compact"
              onClick={() => void loadList(activeTarget)}
              disabled={loadingList}
            >
              <RefreshCw size={13} /> Refresh
            </button>
            <button
              type="button"
              className="mini-btn"
              onClick={onClose}
              data-testid="close-history-drawer-btn"
            >
              <X size={14} /> Close
            </button>
          </div>
        </div>

        <div className="drawer-body">
          <div className="tab-group">
            {TARGET_TABS.map((t) => (
              <button
                key={t.key}
                type="button"
                className={`tab-btn ${activeTarget === t.key ? "active" : ""}`}
                onClick={() => setActiveTarget(t.key)}
                data-testid={`history-target-${t.key}`}
              >
                {t.label}
              </button>
            ))}
          </div>

          {error && <div className="error-banner">{error}</div>}

          <div className="field-group">
            <strong>Archived Snapshots ({items.length})</strong>
            {loadingList ? (
              <div className="muted small">Loading snapshots...</div>
            ) : items.length === 0 ? (
              <div className="empty-cell">No .history/ snapshots archived for this target.</div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {items.map((item) => (
                  <button
                    key={item.snapshot_id}
                    type="button"
                    className={`choice-card ${selectedId === item.snapshot_id ? "selected" : ""}`}
                    data-testid={`history-snapshot-${item.snapshot_id}`}
                    onClick={() => void handleSelectSnapshot(item.snapshot_id)}
                  >
                    <div className="section-header-row" style={{ width: "100%" }}>
                      <span className="mono small">
                        <strong>{item.relative_path || item.name}</strong>
                      </span>
                      <span className="badge-pill">{item.reason || "snapshot"}</span>
                    </div>
                    <div className="muted small">
                      {fmtTimestamp(item.created_at)} · {item.size_bytes} B · ID:{" "}
                      <span className="mono">{item.snapshot_id}</span>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>

          {selectedId && (
            <div className="panel-card" data-testid="history-diff-panel">
              <div className="section-header-row">
                <div>
                  <strong>Line-by-Line Diff vs Active Config</strong>
                  {diff && (
                    <div className="muted small mono">
                      {diff.relative_path} · {diff.reason}
                    </div>
                  )}
                </div>
                <button
                  type="button"
                  className="warning-btn compact"
                  disabled={rollingBack || loadingDiff}
                  onClick={() => void handleRollback()}
                  data-testid="history-rollback-btn"
                >
                  <RotateCcw size={13} />{" "}
                  {rollingBack ? "Rolling back..." : "Rollback to This Version"}
                </button>
              </div>

              {loadingDiff ? (
                <div className="muted small">Computing diff...</div>
              ) : diff ? (
                diff.lines && diff.lines.length > 0 ? (
                  <div className="diff-viewer-lines" data-testid="history-diff-viewer">
                    {diff.lines.map((line, idx) => {
                      const displayText =
                        line.type === "add" && !line.text.startsWith("+")
                          ? `+${line.text}`
                          : line.type === "remove" && !line.text.startsWith("-")
                            ? `-${line.text}`
                            : line.text;
                      return (
                        <div key={idx} className={`diff-line diff-line-${line.type}`}>
                          <span className="diff-line-num">{line.old_line ?? ""}</span>
                          <span className="diff-line-num">{line.new_line ?? ""}</span>
                          <span>{displayText}</span>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <pre className="json-box" data-testid="history-diff-viewer">
                    {diff.unified_diff || "(Identical to active config, no differences)"}
                  </pre>
                )
              ) : null}
            </div>
          )}
        </div>
      </aside>
    </div>
  );
};
