import React, { useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  BookmarkPlus,
  CheckCircle2,
  Crosshair,
  Loader2,
  Maximize2,
  ShieldAlert,
  Trash2,
  X,
} from "lucide-react";
import { checkSplice, createFewShot, frameUrl } from "../api/client";
import type {
  CheckSpliceResponse,
  DiagnosisErrorType,
  MacroStage,
  MaskSide,
  QcSegment,
  SplicePolicy,
  TrainingRole,
  VisualLabelDef,
} from "../api/types";
import { DEFAULT_VISUAL_LABELS, normalizeSegmentTriplet } from "../constants/labels";

export interface SegmentInspectorProps {
  episodeId: string;
  taskId: string;
  durationS: number;
  currentTime: number;
  isLead: boolean;
  readOnly?: boolean;
  labels?: VisualLabelDef[];
  primaryCamera?: string;
  authToken?: string;
  selectedSegment: QcSegment | null;
  selectedStage: MacroStage | null;
  onSeek?: (tS: number) => void;
  onUpdateSegment: (next: QcSegment) => void;
  onDeleteSegment: (segId: string) => void;
  onUpdateStage: (next: MacroStage) => void;
  onDeleteStage: (stageId: string) => void;
}

const SPLICE_DEBOUNCE_MS = 250;

export const SegmentInspector: React.FC<SegmentInspectorProps> = ({
  episodeId,
  taskId,
  durationS,
  currentTime,
  isLead,
  readOnly = false,
  labels = DEFAULT_VISUAL_LABELS,
  primaryCamera = "main",
  authToken,
  selectedSegment,
  selectedStage,
  onSeek,
  onUpdateSegment,
  onDeleteSegment,
  onUpdateStage,
  onDeleteStage,
}) => {
  const [fewShotOpen, setFewShotOpen] = useState(false);
  const [fsScope, setFsScope] = useState<string>("global");
  const [fsErrorType, setFsErrorType] = useState<DiagnosisErrorType>("fp_overflag");
  const [fsRationale, setFsRationale] = useState<string>("");
  const [fsStatusMsg, setFsStatusMsg] = useState<string | null>(null);
  const [fsSaving, setFsSaving] = useState(false);

  const [frameLightboxOpen, setFrameLightboxOpen] = useState(false);
  const [liveCheck, setLiveCheck] = useState<CheckSpliceResponse | null>(null);
  const [liveCheckSegId, setLiveCheckSegId] = useState<string | null>(null);
  const [checkingSplice, setCheckingSplice] = useState(false);

  const checkSeqRef = useRef(0);
  const prevTrackedRef = useRef<{
    segId: string;
    tStart: number;
    tEnd: number;
    labelId: string;
    policy: SplicePolicy;
    arm: MaskSide;
  } | null>(null);

  const hasSelectedSeg = Boolean(selectedSegment);
  const segId = selectedSegment?.seg_id ?? null;
  const segStart = selectedSegment?.t_start ?? 0;
  const segEnd = selectedSegment?.t_end ?? 0;
  const segLabelId = selectedSegment?.label_id ?? "";
  const segPolicy = selectedSegment?.splice_policy ?? "keep_positive";
  const segArm = selectedSegment?.affected_arm ?? "both";

  const stageId = selectedStage?.stage_id ?? null;
  const stageStart = selectedStage?.t_start ?? 0;
  const stageEnd = selectedStage?.t_end ?? 0;

  const [segStartDraft, setSegStartDraft] = useState<string>(String(segStart));
  const [segEndDraft, setSegEndDraft] = useState<string>(String(segEnd));
  const [stageStartDraft, setStageStartDraft] = useState<string>(String(stageStart));
  const [stageEndDraft, setStageEndDraft] = useState<string>(String(stageEnd));

  useEffect(() => {
    setSegStartDraft(String(segStart));
  }, [segId, segStart]);

  useEffect(() => {
    setSegEndDraft(String(segEnd));
  }, [segId, segEnd]);

  useEffect(() => {
    setStageStartDraft(String(stageStart));
  }, [stageId, stageStart]);

  useEffect(() => {
    setStageEndDraft(String(stageEnd));
  }, [stageId, stageEnd]);

  // Reset Few-Shot form & frame lightbox when switching segment or episode
  useEffect(() => {
    setFewShotOpen(false);
    setFsScope("global");
    setFsErrorType("fp_overflag");
    setFsRationale("");
    setFsStatusMsg(null);
    setFrameLightboxOpen(false);
  }, [segId, episodeId]);

  // Close frame compare lightbox on Escape
  useEffect(() => {
    if (!frameLightboxOpen) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        setFrameLightboxOpen(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [frameLightboxOpen]);

  // Debounced 250ms live checkSplice backend call on cut_clean or boundary edits
  useEffect(() => {
    if (!hasSelectedSeg || !segId || segEnd <= segStart) {
      checkSeqRef.current += 1;
      prevTrackedRef.current = null;
      setLiveCheck(null);
      setLiveCheckSegId(null);
      setCheckingSplice(false);
      return;
    }

    const prev = prevTrackedRef.current;
    const isSameSeg = prev?.segId === segId;
    const boundaryOrPolicyEdited =
      isSameSeg &&
      (prev.tStart !== segStart ||
        prev.tEnd !== segEnd ||
        prev.policy !== segPolicy ||
        prev.labelId !== segLabelId ||
        prev.arm !== segArm);

    prevTrackedRef.current = {
      segId,
      tStart: segStart,
      tEnd: segEnd,
      labelId: segLabelId,
      policy: segPolicy,
      arm: segArm,
    };

    if (!isSameSeg || segPolicy === "keep_positive") {
      checkSeqRef.current += 1;
      setLiveCheck(null);
      setLiveCheckSegId(null);
    }

    const shouldCheck =
      segPolicy === "cut_clean" || (boundaryOrPolicyEdited && segPolicy !== "keep_positive");

    if (!shouldCheck) {
      setCheckingSplice(false);
      return;
    }

    const seq = ++checkSeqRef.current;
    setCheckingSplice(true);

    const timer = window.setTimeout(() => {
      void checkSplice(episodeId, {
        t_start: segStart,
        t_end: segEnd,
        label_id: segLabelId,
        splice_policy: segPolicy,
        affected_arm: segArm,
      })
        .then((res) => {
          if (seq !== checkSeqRef.current) return;
          if (res && typeof res.passed === "boolean") {
            setLiveCheck(res);
            setLiveCheckSegId(segId);
          }
        })
        .catch(() => {
          if (seq !== checkSeqRef.current) return;
          setLiveCheck(null);
          setLiveCheckSegId(null);
        })
        .finally(() => {
          if (seq === checkSeqRef.current) {
            setCheckingSplice(false);
          }
        });
    }, SPLICE_DEBOUNCE_MS);

    return () => {
      window.clearTimeout(timer);
    };
  }, [episodeId, hasSelectedSeg, segId, segStart, segEnd, segLabelId, segPolicy, segArm]);

  if (!selectedSegment && !selectedStage) {
    return (
      <div className="inspector-empty" data-testid="segment-inspector-empty">
        Click any <strong>QC Segment</strong> or <strong>Macro Stage</strong> on the four-track
        timeline to edit its properties.
      </div>
    );
  }

  if (selectedStage) {
    return (
      <div className="inspector-panel" data-testid="macro-stage-inspector">
        <div className="inspector-header">
          <h4>Macro Subtask Stage ({selectedStage.stage_id})</h4>
          {!readOnly && (
            <button
              type="button"
              className="danger-btn compact"
              onClick={() => onDeleteStage(selectedStage.stage_id)}
            >
              <Trash2 size={13} /> Delete Stage
            </button>
          )}
        </div>

        <label className="field-group">
          <span>Stage Label (label)</span>
          <input
            type="text"
            disabled={readOnly}
            value={selectedStage.label}
            onChange={(e) => onUpdateStage({ ...selectedStage, label: e.target.value })}
            onBlur={() => {
              if (!selectedStage.label.trim()) {
                onUpdateStage({ ...selectedStage, label: "Untitled Stage" });
              }
            }}
          />
        </label>

        <label className="field-group">
          <span>English Slug (label_en)</span>
          <input
            type="text"
            disabled={readOnly}
            value={selectedStage.label_en}
            onChange={(e) => onUpdateStage({ ...selectedStage, label_en: e.target.value })}
          />
        </label>

        <div className="two-col">
          <label className="field-group">
            <span>Start (s)</span>
            <input
              type="number"
              aria-label="Stage start"
              step={0.05}
              min={0}
              max={durationS}
              disabled={readOnly}
              value={stageStartDraft}
              onChange={(e) => {
                const raw = e.target.value;
                setStageStartDraft(raw);
                if (raw.trim() === "") return;
                const parsed = Number(raw);
                if (!Number.isFinite(parsed)) return;
                const v = Math.max(0, Math.min(selectedStage.t_end - 0.05, parsed));
                onUpdateStage({ ...selectedStage, t_start: Number(v.toFixed(2)) });
              }}
              onBlur={() => setStageStartDraft(String(selectedStage.t_start))}
            />
          </label>
          <label className="field-group">
            <span>End (s)</span>
            <input
              type="number"
              aria-label="Stage end"
              step={0.05}
              min={0.05}
              max={durationS}
              disabled={readOnly}
              value={stageEndDraft}
              onChange={(e) => {
                const raw = e.target.value;
                setStageEndDraft(raw);
                if (raw.trim() === "") return;
                const parsed = Number(raw);
                if (!Number.isFinite(parsed)) return;
                const v = Math.min(durationS, Math.max(selectedStage.t_start + 0.05, parsed));
                onUpdateStage({ ...selectedStage, t_end: Number(v.toFixed(2)) });
              }}
              onBlur={() => setStageEndDraft(String(selectedStage.t_end))}
            />
          </label>
        </div>

        <label className="field-group">
          <span>Stage Summary (summary)</span>
          <textarea
            rows={3}
            disabled={readOnly}
            value={selectedStage.summary}
            onChange={(e) => onUpdateStage({ ...selectedStage, summary: e.target.value })}
          />
        </label>
      </div>
    );
  }

  const seg = selectedSegment!;
  const labelDef = labels.find((l) => l.id === seg.label_id);
  const activeLiveCheck = liveCheck && liveCheckSegId === seg.seg_id ? liveCheck : null;

  const optStart = activeLiveCheck?.opt_t_start ?? seg.guardrail.opt_t_start;
  const optEnd = activeLiveCheck?.opt_t_end ?? seg.guardrail.opt_t_end;
  const canSnapOpt = optStart !== null && optEnd !== null && optEnd > optStart;

  const isHallucinationOverride =
    Boolean(activeLiveCheck) &&
    activeLiveCheck!.effective_splice_policy === "keep_positive" &&
    seg.splice_policy !== "keep_positive";
  const effectivePassed = activeLiveCheck ? activeLiveCheck.passed : seg.guardrail.passed;
  const effectiveDowngraded = activeLiveCheck ? activeLiveCheck.downgraded : seg.guardrail.downgraded;
  const effectiveReason = activeLiveCheck?.downgrade_reason ?? seg.guardrail.downgrade_reason;
  const liveEefPosM =
    activeLiveCheck?.metrics.eef_pos_m ??
    (seg.guardrail.gate_type === "splice_gate" ? seg.guardrail.metrics.eef_pos_m : null);
  const liveVisualSsim = activeLiveCheck?.metrics.visual_ssim ?? seg.guardrail.metrics.visual_ssim;
  const metricSummary =
    liveEefPosM !== null
      ? `${(liveEefPosM * 100).toFixed(2)}cm`
      : liveVisualSsim !== null
        ? `SSIM ${liveVisualSsim.toFixed(2)}`
        : "";

  const showLiveSpliceBar =
    checkingSplice ||
    seg.splice_policy === "cut_clean" ||
    (activeLiveCheck !== null && activeLiveCheck.mode !== "skipped") ||
    seg.guardrail.downgraded ||
    seg.guardrail.gate_type === "splice_gate";

  const startThumbUrl = frameUrl(episodeId, primaryCamera, seg.t_start, authToken);
  const endThumbUrl = frameUrl(episodeId, primaryCamera, seg.t_end, authToken);

  const applyPatch = (patch: Partial<QcSegment>) => {
    onUpdateSegment(normalizeSegmentTriplet(seg, patch, labels));
  };

  const handleSaveFewShot = async () => {
    if (!fsRationale.trim()) {
      setFsStatusMsg("Please provide a brief rationale explaining the visual criterion.");
      return;
    }
    setFsSaving(true);
    setFsStatusMsg(null);
    try {
      await createFewShot({
        scope: fsScope,
        task_id: fsScope.startsWith("task:") ? taskId : null,
        label_id: fsScope.startsWith("label:") ? seg.label_id : null,
        error_type: fsErrorType,
        episode_id: episodeId,
        t_start: seg.t_start,
        t_end: seg.t_end,
        vlm_wrong_json: {
          label_id: seg.label_id,
          vlm_raw_policy: seg.vlm_raw_policy,
          guardrail: seg.guardrail,
        },
        human_right_json: {
          label_id: seg.label_id,
          training_role: seg.training_role,
          splice_policy: seg.splice_policy,
          mask_side: seg.mask_side,
        },
        rationale: fsRationale.trim(),
        capture_tile: true,
      });
      setFsStatusMsg("Saved to Few-Shot Mistake Book");
      setFewShotOpen(false);
      setFsRationale("");
    } catch (err) {
      setFsStatusMsg(err instanceof Error ? err.message : "Failed to save Few-Shot example");
    } finally {
      setFsSaving(false);
    }
  };

  const diagEefPosM = activeLiveCheck?.metrics.eef_pos_m ?? seg.guardrail.metrics.eef_pos_m;
  const diagEefRotRad = activeLiveCheck?.metrics.eef_rot_rad ?? seg.guardrail.metrics.eef_rot_rad;
  const diagHandRad = activeLiveCheck?.metrics.hand_rad ?? seg.guardrail.metrics.hand_rad;
  const diagOtherArmM =
    activeLiveCheck?.metrics.other_arm_disp_m ?? seg.guardrail.metrics.other_arm_disp_m;
  const diagCoupled =
    activeLiveCheck?.metrics.bimanual_coupled ?? seg.guardrail.metrics.bimanual_coupled;

  return (
    <div className="inspector-panel" data-testid="segment-inspector">
      <div className="inspector-header">
        <h4>
          QC Segment ({seg.seg_id}){" "}
          <span className="muted small mono">Δ{(seg.t_end - seg.t_start).toFixed(2)}s</span>
        </h4>
        {!readOnly && (
          <button
            type="button"
            className="danger-btn compact"
            onClick={() => onDeleteSegment(seg.seg_id)}
            data-testid="delete-segment-btn"
          >
            <Trash2 size={13} /> Delete
          </button>
        )}
      </div>

      {/* Compact Start vs End Side-by-Side Frame Comparison */}
      <div className="frame-compare-strip" data-testid="segment-frame-compare">
        <div
          className="frame-thumb-card"
          data-testid="segment-start-thumb"
          role="button"
          tabIndex={0}
          title={`Click to seek to start frame ${seg.t_start.toFixed(2)}s`}
          onClick={() => onSeek?.(seg.t_start)}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              onSeek?.(seg.t_start);
            }
          }}
        >
          <img
            className="frame-thumb-img"
            src={startThumbUrl}
            alt={`Start frame ${seg.t_start.toFixed(2)}s`}
          />
          <span className="frame-thumb-badge">Start {seg.t_start.toFixed(2)}s</span>
          <button
            type="button"
            className="frame-thumb-zoom"
            data-testid="open-frame-lightbox"
            aria-label="Enlarge start/end frame comparison"
            title="Enlarge side-by-side start/end frame comparison"
            onClick={(e) => {
              e.stopPropagation();
              setFrameLightboxOpen(true);
            }}
          >
            <Maximize2 size={11} />
          </button>
        </div>

        <div
          className="frame-thumb-card"
          data-testid="segment-end-thumb"
          role="button"
          tabIndex={0}
          title={`Click to seek to end frame ${seg.t_end.toFixed(2)}s`}
          onClick={() => onSeek?.(seg.t_end)}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              onSeek?.(seg.t_end);
            }
          }}
        >
          <img
            className="frame-thumb-img"
            src={endThumbUrl}
            alt={`End frame ${seg.t_end.toFixed(2)}s`}
          />
          <span className="frame-thumb-badge">End {seg.t_end.toFixed(2)}s</span>
        </div>
      </div>

      {/* Live Splice Guardrail Bar */}
      {showLiveSpliceBar && (
        <div
          className={`live-splice-bar ${
            checkingSplice
              ? "checking"
              : isHallucinationOverride
                ? "warn"
                : effectivePassed && !effectiveDowngraded
                  ? "pass"
                  : "warn"
          }`}
          data-testid="live-splice-badge"
        >
          <span className="live-splice-status">
            {checkingSplice ? (
              <>
                <Loader2 size={13} /> Recalculating splice guardrail...
              </>
            ) : isHallucinationOverride ? (
              <>
                <AlertTriangle size={13} /> Hallucination override: protected as keep_positive
              </>
            ) : effectivePassed && !effectiveDowngraded ? (
              <>
                <CheckCircle2 size={13} /> {metricSummary ? `${metricSummary} ` : ""}Seamless cut OK
                {activeLiveCheck && (
                  <span className="muted small">({activeLiveCheck.compute_ms.toFixed(0)}ms)</span>
                )}
              </>
            ) : (
              <>
                <AlertTriangle size={13} /> {metricSummary ? `${metricSummary} ` : ""}Jump exceeds threshold
              </>
            )}
          </span>
          {!readOnly && canSnapOpt && (
            <button
              type="button"
              className="mini-btn accent"
              data-testid="snap-opt-cut-btn"
              onClick={() =>
                applyPatch({
                  t_start: optStart!,
                  t_end: optEnd!,
                })
              }
            >
              <Crosshair size={12} /> Snap ({optStart!.toFixed(2)}–{optEnd!.toFixed(2)}s)
            </button>
          )}
        </div>
      )}

      <label className="field-group">
        <span>QC Label (label_id)</span>
        <select
          aria-label="QC Label"
          disabled={readOnly}
          value={seg.label_id}
          onChange={(e) => applyPatch({ label_id: e.target.value })}
        >
          {labels.map((lbl) => (
            <option key={lbl.id} value={lbl.id}>
              {lbl.name} ({lbl.id})
            </option>
          ))}
        </select>
      </label>

      {labelDef && (
        <div className="label-hint-box" data-testid="collector-ui-hint">
          <div className="hint-title">{labelDef.collector_ui.title}</div>
          <div className="hint-desc">{labelDef.collector_ui.hint}</div>
          {isLead && labelDef.vlm_semantics.definition && (
            <details className="vlm-semantics-details">
              <summary>VLM Semantics &amp; Counter-Examples</summary>
              <p>{labelDef.vlm_semantics.definition}</p>
              {labelDef.vlm_semantics.counter_examples.length > 0 && (
                <ul>
                  {labelDef.vlm_semantics.counter_examples.map((ex, i) => (
                    <li key={i}>Counter-example: {ex}</li>
                  ))}
                </ul>
              )}
            </details>
          )}
        </div>
      )}

      <div className="two-col">
        <div className="field-group">
          <span>Start (t_start, s)</span>
          <div className="input-with-btn">
            <input
              type="number"
              aria-label="Segment start"
              step={0.05}
              min={0}
              max={durationS}
              disabled={readOnly}
              value={segStartDraft}
              onChange={(e) => {
                const raw = e.target.value;
                setSegStartDraft(raw);
                if (raw.trim() === "") return;
                const parsed = Number(raw);
                if (!Number.isFinite(parsed)) return;
                const v = Math.max(0, Math.min(seg.t_end - 0.05, parsed));
                applyPatch({ t_start: Number(v.toFixed(2)) });
              }}
              onBlur={() => setSegStartDraft(String(seg.t_start))}
            />
            {!readOnly && (
              <button
                type="button"
                className="mini-btn"
                title="Set to current playhead ([)"
                onClick={() => {
                  const next = Math.min(
                    seg.t_end - 0.05,
                    Math.max(0, Number(currentTime.toFixed(2))),
                  );
                  applyPatch({ t_start: next });
                }}
              >
                Now
              </button>
            )}
          </div>
        </div>

        <div className="field-group">
          <span>End (t_end, s)</span>
          <div className="input-with-btn">
            <input
              type="number"
              aria-label="Segment end"
              step={0.05}
              min={0.05}
              max={durationS}
              disabled={readOnly}
              value={segEndDraft}
              onChange={(e) => {
                const raw = e.target.value;
                setSegEndDraft(raw);
                if (raw.trim() === "") return;
                const parsed = Number(raw);
                if (!Number.isFinite(parsed)) return;
                const v = Math.min(durationS, Math.max(seg.t_start + 0.05, parsed));
                applyPatch({ t_end: Number(v.toFixed(2)) });
              }}
              onBlur={() => setSegEndDraft(String(seg.t_end))}
            />
            {!readOnly && (
              <button
                type="button"
                className="mini-btn"
                title="Set to current playhead (])"
                onClick={() => {
                  const next = Math.max(
                    seg.t_start + 0.05,
                    Math.min(durationS, Number(currentTime.toFixed(2))),
                  );
                  applyPatch({ t_end: next });
                }}
              >
                Now
              </button>
            )}
          </div>
        </div>
      </div>

      <div className="three-col">
        <label className="field-group">
          <span>Training Role</span>
          <select
            aria-label="Training Role"
            disabled={readOnly}
            value={seg.training_role}
            onChange={(e) => applyPatch({ training_role: e.target.value as TrainingRole })}
          >
            <option value="necessary">necessary</option>
            <option value="recovery">recovery</option>
            <option value="badcase">badcase</option>
            <option value="terminal_fail">terminal_fail</option>
          </select>
        </label>

        <label className="field-group">
          <span>Splice Policy</span>
          <select
            aria-label="Splice Policy"
            disabled={readOnly}
            value={seg.splice_policy}
            onChange={(e) => applyPatch({ splice_policy: e.target.value as SplicePolicy })}
          >
            <option value="keep_positive">keep_positive</option>
            <option value="mask_loss">mask_loss</option>
            <option value="cut_clean">cut_clean</option>
          </select>
        </label>

        <label className="field-group">
          <span>Mask Side</span>
          <select
            aria-label="Mask Side"
            disabled={readOnly || seg.splice_policy !== "mask_loss"}
            value={seg.mask_side ?? ""}
            onChange={(e) =>
              applyPatch({
                mask_side: (e.target.value || null) as MaskSide | null,
              })
            }
          >
            {seg.splice_policy !== "mask_loss" && <option value="">None (null)</option>}
            <option value="left">left (Left Arm)</option>
            <option value="right">right (Right Arm)</option>
            <option value="both">both (Both Arms)</option>
          </select>
        </label>
      </div>

      <div className="evidence-card">
        <div className="card-subtitle">Visual Evidence (VLM / Manual)</div>
        <p>
          <strong>Start:</strong> {seg.visual_evidence.initial_state || "—"} ·{" "}
          <strong>Process:</strong> {seg.visual_evidence.process_change || "—"} ·{" "}
          <strong>End:</strong> {seg.visual_evidence.end_state || "—"}
        </p>
      </div>

      <div className="unknown-box">
        <label className="checkbox-inline">
          <input
            type="checkbox"
            disabled={readOnly}
            checked={seg.is_unknown}
            onChange={(e) => applyPatch({ is_unknown: e.target.checked })}
          />
          <span>Flag as unknown anomaly (send to VLM Lab triage)</span>
        </label>
        {seg.is_unknown && (
          <input
            type="text"
            placeholder="Describe the undefined anomaly..."
            disabled={readOnly}
            value={seg.unknown_note}
            onChange={(e) => applyPatch({ unknown_note: e.target.value })}
          />
        )}
      </div>

      {/* Lead-only Guardrail Diagnostics & Few-Shot Bookmark */}
      {isLead && (
        <div className="lead-diagnostics" data-testid="lead-guardrail-diagnostics">
          <div className="diag-header">
            <ShieldAlert size={13} />
            <span>Guardrail Physics Diagnostics (Lead Only)</span>
          </div>
          <div className="diag-grid">
            <span>Gate: {activeLiveCheck ? activeLiveCheck.mode : seg.guardrail.gate_type}</span>
            <span>VLM Raw: {seg.vlm_raw_policy}</span>
            <span>Downgraded: {effectiveDowngraded ? "Yes" : "No"}</span>
            <span>Bimanual Coupled: {diagCoupled ? "Yes" : "No"}</span>
            <span>EEF Disp: {(diagEefPosM * 100).toFixed(2)} cm</span>
            <span>EEF Rot: {diagEefRotRad.toFixed(3)} rad</span>
            <span>Hand Joint: {diagHandRad.toFixed(3)} rad</span>
            <span>Other Arm: {(diagOtherArmM * 100).toFixed(2)} cm</span>
          </div>
          {effectiveReason && <div className="diag-reason">{effectiveReason}</div>}

          <div className="few-shot-actions">
            <button
              type="button"
              className="mini-btn accent"
              data-testid="add-few-shot-btn"
              onClick={() => setFewShotOpen((v) => !v)}
            >
              <BookmarkPlus size={13} /> Add to Mistake Book (Few-Shot)
            </button>
          </div>

          {fewShotOpen && (
            <div className="few-shot-popover" data-testid="few-shot-form">
              <div className="two-col">
                <label className="field-group">
                  <span>Scope</span>
                  <select value={fsScope} onChange={(e) => setFsScope(e.target.value)}>
                    <option value="global">Global (global)</option>
                    <option value={`task:${taskId}`}>Current Task (task:{taskId})</option>
                    <option value={`label:${seg.label_id}`}>Current Label (label:{seg.label_id})</option>
                  </select>
                </label>
                <label className="field-group">
                  <span>Error Type</span>
                  <select
                    value={fsErrorType}
                    onChange={(e) => setFsErrorType(e.target.value as DiagnosisErrorType)}
                  >
                    <option value="fp_overflag">FP Overflag</option>
                    <option value="fn_missed">FN Missed</option>
                    <option value="policy_flip">Policy / Overcut Flip</option>
                    <option value="boundary">Boundary Drift</option>
                    <option value="success_flip">Success Flip</option>
                  </select>
                </label>
              </div>
              <label className="field-group">
                <span>Rationale (rationale)</span>
                <textarea
                  rows={2}
                  value={fsRationale}
                  onChange={(e) => setFsRationale(e.target.value)}
                  placeholder="Explain why the VLM prediction was wrong and the correct visual criterion..."
                />
              </label>
              <button
                type="button"
                className="primary-btn compact"
                disabled={fsSaving}
                onClick={handleSaveFewShot}
              >
                {fsSaving ? "Saving..." : "Save Few-Shot (with 2x2 Tile)"}
              </button>
            </div>
          )}

          {fsStatusMsg && <div className="status-note">{fsStatusMsg}</div>}
        </div>
      )}

      {/* Enlarged Side-by-Side Start/End Lightbox Modal */}
      {frameLightboxOpen && (
        <div
          className="modal-backdrop"
          data-testid="frame-compare-lightbox"
          onClick={(e) => {
            if (e.target === e.currentTarget) setFrameLightboxOpen(false);
          }}
        >
          <div className="modal-card modal-card-wide">
            <div className="inspector-header">
              <h3 style={{ margin: 0 }}>
                Start vs End Frame Comparison · {seg.name || seg.label_id} ({seg.t_start.toFixed(2)}s
                vs {seg.t_end.toFixed(2)}s)
              </h3>
              <button
                type="button"
                className="mini-btn"
                onClick={() => setFrameLightboxOpen(false)}
              >
                <X size={14} /> Close
              </button>
            </div>
            <div className="frame-lightbox-grid">
              <div className="frame-lightbox-pane">
                <strong>Start Frame t_start = {seg.t_start.toFixed(2)}s</strong>
                <img src={startThumbUrl} alt={`HD start frame ${seg.t_start.toFixed(2)}s`} />
              </div>
              <div className="frame-lightbox-pane">
                <strong>End Frame t_end = {seg.t_end.toFixed(2)}s</strong>
                <img src={endThumbUrl} alt={`HD end frame ${seg.t_end.toFixed(2)}s`} />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
