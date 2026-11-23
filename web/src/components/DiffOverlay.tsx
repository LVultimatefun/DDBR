import React from "react";
import { GitCompare, History, RotateCcw } from "lucide-react";
import type {
  HumanReviewArtifact,
  MacroStage,
  QcSegment,
  ReviewDiff,
  SuccessVerdict,
} from "../api/types";

export interface DiffOverlayProps {
  isLead: boolean;
  diffEnabled: boolean;
  onToggleDiff: (enabled: boolean) => void;
  reviewDiff: ReviewDiff | null;
  rawVlmSegments: QcSegment[];
  currentSegments: QcSegment[];
  rawVlmSuccess: SuccessVerdict | null;
  currentSuccess: SuccessVerdict;
  rawVlmMacroStages: MacroStage[];
  currentMacroStages: MacroStage[];
  reviews: HumanReviewArtifact[];
  onRestoreSnapshot: (rev: HumanReviewArtifact) => void;
  onSeek?: (tS: number) => void;
}

function segmentIou(a: QcSegment, b: QcSegment): number {
  const inter = Math.max(0, Math.min(a.t_end, b.t_end) - Math.max(a.t_start, b.t_start));
  if (inter <= 0) return 0;
  const union = Math.max(a.t_end, b.t_end) - Math.min(a.t_start, b.t_start);
  return union > 0 ? inter / union : 0;
}

function computeLiveDiff(params: {
  rawVlmSegments: QcSegment[];
  currentSegments: QcSegment[];
  rawVlmSuccess: SuccessVerdict | null;
  currentSuccess: SuccessVerdict;
  rawVlmMacroStages: MacroStage[];
  currentMacroStages: MacroStage[];
}): {
  diff: ReviewDiff;
  matchedByRawIdx: Map<number, QcSegment>;
  matchedCurIdxs: Set<number>;
} {
  const {
    rawVlmSegments,
    currentSegments,
    rawVlmSuccess,
    currentSuccess,
    rawVlmMacroStages,
    currentMacroStages,
  } = params;

  const candidates: Array<{ iou: number; sameId: number; vi: number; hi: number }> = [];
  for (let vi = 0; vi < rawVlmSegments.length; vi += 1) {
    for (let hi = 0; hi < currentSegments.length; hi += 1) {
      const iou = segmentIou(rawVlmSegments[vi], currentSegments[hi]);
      if (iou >= 0.5) {
        candidates.push({
          iou,
          sameId: rawVlmSegments[vi].seg_id === currentSegments[hi].seg_id ? 1 : 0,
          vi,
          hi,
        });
      }
    }
  }
  candidates.sort((a, b) => b.iou - a.iou || b.sameId - a.sameId || a.vi - b.vi || a.hi - b.hi);

  const matchedV = new Set<number>();
  const matchedH = new Set<number>();
  const matchedByRawIdx = new Map<number, QcSegment>();
  const pairs: Array<[QcSegment, QcSegment]> = [];
  for (const c of candidates) {
    if (!matchedV.has(c.vi) && !matchedH.has(c.hi)) {
      matchedV.add(c.vi);
      matchedH.add(c.hi);
      matchedByRawIdx.set(c.vi, currentSegments[c.hi]);
      pairs.push([rawVlmSegments[c.vi], currentSegments[c.hi]]);
    }
  }

  const fp_deleted = rawVlmSegments.length - matchedV.size;
  const fn_added = currentSegments.length - matchedH.size;
  const policy_flips = pairs.filter(([v, h]) => v.splice_policy !== h.splice_policy).length;
  const label_changes = pairs.filter(([v, h]) => v.label_id !== h.label_id).length;
  const boundary_nudges = pairs.filter(
    ([v, h]) =>
      Math.abs(v.t_start - h.t_start) > 0.05 + 1e-9 ||
      Math.abs(v.t_end - h.t_end) > 0.05 + 1e-9,
  ).length;

  let macro_stage_edits = Math.abs(currentMacroStages.length - rawVlmMacroStages.length);
  const minStages = Math.min(rawVlmMacroStages.length, currentMacroStages.length);
  for (let i = 0; i < minStages; i += 1) {
    const vs = rawVlmMacroStages[i];
    const hs = currentMacroStages[i];
    if (
      vs.label !== hs.label ||
      Math.abs(vs.t_start - hs.t_start) > 0.05 + 1e-9 ||
      Math.abs(vs.t_end - hs.t_end) > 0.05 + 1e-9
    ) {
      macro_stage_edits += 1;
    }
  }

  const success_flipped =
    rawVlmSuccess !== null &&
    (rawVlmSuccess.passed !== currentSuccess.passed ||
      rawVlmSuccess.verdict !== currentSuccess.verdict);

  const total_edits =
    fp_deleted +
    fn_added +
    policy_flips +
    label_changes +
    boundary_nudges +
    macro_stage_edits +
    (success_flipped ? 1 : 0);

  return {
    diff: {
      fp_deleted,
      fn_added,
      policy_flips,
      label_changes,
      boundary_nudges,
      macro_stage_edits,
      success_flipped,
      total_edits,
    },
    matchedByRawIdx,
    matchedCurIdxs: matchedH,
  };
}

export const DiffOverlay: React.FC<DiffOverlayProps> = ({
  isLead,
  diffEnabled,
  onToggleDiff,
  reviewDiff: _reviewDiff,
  rawVlmSegments,
  currentSegments,
  rawVlmSuccess,
  currentSuccess,
  rawVlmMacroStages,
  currentMacroStages,
  reviews,
  onRestoreSnapshot,
  onSeek,
}) => {
  if (!isLead) {
    return null;
  }

  const { diff, matchedByRawIdx, matchedCurIdxs } = computeLiveDiff({
    rawVlmSegments,
    currentSegments,
    rawVlmSuccess,
    currentSuccess,
    rawVlmMacroStages,
    currentMacroStages,
  });

  return (
    <div className="diff-overlay-panel" data-testid="diff-overlay">
      <div className="diff-header">
        <div className="diff-title">
          <GitCompare size={15} />
          <span>VLM Raw vs Human Revision (Lead View)</span>
        </div>
        <label className="checkbox-inline">
          <input
            type="checkbox"
            checked={diffEnabled}
            onChange={(e) => onToggleDiff(e.target.checked)}
            data-testid="toggle-diff-track"
          />
          <span>Show ghost comparison track on timeline</span>
        </label>
      </div>

      <div className="diff-metrics-grid" data-testid="diff-metrics">
        <div className="diff-stat">
          <span className="stat-val">{diff.fp_deleted}</span>
          <span className="stat-lbl">FP Deleted (fp_deleted)</span>
        </div>
        <div className="diff-stat">
          <span className="stat-val">{diff.fn_added}</span>
          <span className="stat-lbl">FN Added (fn_added)</span>
        </div>
        <div className="diff-stat">
          <span className="stat-val">{diff.policy_flips}</span>
          <span className="stat-lbl">Policy Flips (policy_flips)</span>
        </div>
        <div className="diff-stat">
          <span className="stat-val">{diff.label_changes}</span>
          <span className="stat-lbl">Label Changes (label_changes)</span>
        </div>
        <div className="diff-stat">
          <span className="stat-val">{diff.boundary_nudges}</span>
          <span className="stat-lbl">Boundary Nudges (boundary_nudges)</span>
        </div>
        <div className="diff-stat">
          <span className="stat-val">{diff.macro_stage_edits}</span>
          <span className="stat-lbl">Stage Edits (macro_stage_edits)</span>
        </div>
        <div className={`diff-stat ${diff.success_flipped ? "highlight" : ""}`}>
          <span className="stat-val">{diff.success_flipped ? "Yes" : "No"}</span>
          <span className="stat-lbl">Success Flipped (success_flipped)</span>
        </div>
      </div>

      <div className="diff-table-section">
        <div className="section-label">Segment Comparison Details</div>
        <div className="diff-list">
          {rawVlmSegments.map((raw, idx) => {
            const cur = matchedByRawIdx.get(idx);
            const statusText = !cur
              ? "Deleted (FP)"
              : cur.splice_policy !== raw.splice_policy || cur.label_id !== raw.label_id
                ? `Modified (${raw.label_id}/${raw.splice_policy} → ${cur.label_id}/${cur.splice_policy})`
                : Math.abs(cur.t_start - raw.t_start) > 0.05 + 1e-9 ||
                    Math.abs(cur.t_end - raw.t_end) > 0.05 + 1e-9
                  ? `Boundary Nudged (${raw.t_start.toFixed(2)}-${raw.t_end.toFixed(2)} → ${cur.t_start.toFixed(2)}-${cur.t_end.toFixed(2)})`
                  : "Unchanged";
            return (
              <div
                key={`raw-${raw.seg_id}-${idx}`}
                className="diff-row-item"
                onClick={() => onSeek?.(raw.t_start)}
              >
                <span>
                  [VLM] {raw.name || raw.label_id} ({raw.t_start.toFixed(2)}s-{raw.t_end.toFixed(2)}s)
                </span>
                <span className="diff-status-badge">{statusText}</span>
              </div>
            );
          })}
          {currentSegments
            .filter((_, idx) => !matchedCurIdxs.has(idx))
            .map((added) => (
              <div
                key={`fn-${added.seg_id}`}
                className="diff-row-item added"
                onClick={() => onSeek?.(added.t_start)}
              >
                <span>
                  [Manual FN] {added.name || added.label_id} ({added.t_start.toFixed(2)}s-
                  {added.t_end.toFixed(2)}s)
                </span>
                <span className="diff-status-badge">{added.splice_policy}</span>
              </div>
            ))}
        </div>
      </div>

      {rawVlmSuccess && (
        <div className="diff-success-compare">
          <strong>Success Comparison: </strong>VLM={rawVlmSuccess.verdict} → Human=
          {currentSuccess.verdict}
        </div>
      )}

      <div className="revision-history-drawer" data-testid="revision-history-drawer">
        <div className="section-label">
          <History size={13} /> Immutable Review Snapshots ({reviews.length})
        </div>
        {reviews.length === 0 ? (
          <div className="empty-hint">No review snapshots recorded yet.</div>
        ) : (
          <div className="revision-list">
            {reviews.map((rev) => (
              <div key={rev.rev_id} className="revision-item" data-testid={`rev-${rev.rev_id}`}>
                <div className="revision-meta">
                  <strong>{rev.action}</strong> · {rev.author} ({rev.role})
                  <span className="muted">
                    {new Date(rev.created_at * 1000).toLocaleString()}
                  </span>
                </div>
                {rev.notes && <div className="revision-notes">Notes: {rev.notes}</div>}
                <button
                  type="button"
                  className="mini-btn"
                  onClick={() => onRestoreSnapshot(rev)}
                  data-testid={`restore-rev-${rev.rev_id}`}
                >
                  <RotateCcw size={12} /> Restore Snapshot to Edit Stack
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
