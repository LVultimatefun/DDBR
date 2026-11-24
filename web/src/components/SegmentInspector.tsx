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
import { useI18n } from "../stores/i18nStore";

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
  const { t } = useI18n();
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
        {t("inspector.emptyPrefix")}
        <strong>{t("inspector.emptyQcSegment")}</strong>
        {t("inspector.emptyMiddle")}
        <strong>{t("inspector.emptyMacroStage")}</strong>
        {t("inspector.emptySuffix")}
      </div>
    );
  }

  if (selectedStage) {
    return (
      <div className="inspector-panel" data-testid="macro-stage-inspector">
        <div className="inspector-header">
          <h4>{t("inspector.stageTitle", { id: selectedStage.stage_id })}</h4>
          {!readOnly && (
            <button
              type="button"
              className="danger-btn compact"
              onClick={() => onDeleteStage(selectedStage.stage_id)}
            >
              <Trash2 size={13} /> {t("inspector.deleteStage")}
            </button>
          )}
        </div>

        <label className="field-group">
          <span>{t("inspector.stageLabel")}</span>
          <input
            type="text"
            disabled={readOnly}
            value={selectedStage.label}
            onChange={(e) => onUpdateStage({ ...selectedStage, label: e.target.value })}
            onBlur={() => {
              if (!selectedStage.label.trim()) {
                onUpdateStage({ ...selectedStage, label: t("inspector.untitledStage") });
              }
            }}
          />
        </label>

        <label className="field-group">
          <span>{t("inspector.stageLabelEn")}</span>
          <input
            type="text"
            disabled={readOnly}
            value={selectedStage.label_en}
            onChange={(e) => onUpdateStage({ ...selectedStage, label_en: e.target.value })}
          />
        </label>

        <div className="two-col">
          <label className="field-group">
            <span>{t("inspector.startS")}</span>
            <input
              type="number"
              aria-label={t("inspector.stageStartAria")}
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
            <span>{t("inspector.endS")}</span>
            <input
              type="number"
              aria-label={t("inspector.stageEndAria")}
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
          <span>{t("inspector.stageInstructionLabel")}</span>
          <textarea
            rows={2}
            aria-label={t("inspector.stageInstructionAria")}
            data-testid="stage-instruction-input"
            disabled={readOnly}
            value={selectedStage.instruction ?? ""}
            onChange={(e) => onUpdateStage({ ...selectedStage, instruction: e.target.value })}
            placeholder={t("inspector.stageInstructionPlaceholder")}
          />
        </label>

        <label className="field-group">
          <span>{t("inspector.stageSummaryLabel")}</span>
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
      setFsStatusMsg(t("inspector.fsNeedRationale"));
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
      setFsStatusMsg(t("inspector.fsSavedMsg"));
      setFewShotOpen(false);
      setFsRationale("");
    } catch (err) {
      setFsStatusMsg(err instanceof Error ? err.message : t("inspector.fsSaveFailed"));
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
          {t("inspector.qcSegmentTitle", { id: seg.seg_id })}{" "}
          <span className="muted small mono">Δ{(seg.t_end - seg.t_start).toFixed(2)}s</span>
        </h4>
        {!readOnly && (
          <button
            type="button"
            className="danger-btn compact"
            onClick={() => onDeleteSegment(seg.seg_id)}
            data-testid="delete-segment-btn"
          >
            <Trash2 size={13} /> {t("inspector.deleteBtn")}
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
          title={t("inspector.startThumbTitle", { t: seg.t_start.toFixed(2) })}
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
            alt={t("inspector.startFrameAlt", { t: seg.t_start.toFixed(2) })}
          />
          <span className="frame-thumb-badge">
            {t("inspector.startBadge", { t: seg.t_start.toFixed(2) })}
          </span>
          <button
            type="button"
            className="frame-thumb-zoom"
            data-testid="open-frame-lightbox"
            aria-label={t("inspector.enlargeCompareAria")}
            title={t("inspector.enlargeCompareTitle")}
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
          title={t("inspector.endThumbTitle", { t: seg.t_end.toFixed(2) })}
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
            alt={t("inspector.endFrameAlt", { t: seg.t_end.toFixed(2) })}
          />
          <span className="frame-thumb-badge">
            {t("inspector.endBadge", { t: seg.t_end.toFixed(2) })}
          </span>
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
                <Loader2 size={13} /> {t("inspector.spliceChecking")}
              </>
            ) : isHallucinationOverride ? (
              <>
                <AlertTriangle size={13} /> {t("inspector.spliceHallucination")}
              </>
            ) : effectivePassed && !effectiveDowngraded ? (
              <>
                <CheckCircle2 size={13} />{" "}
                {t("inspector.splicePass", { metric: metricSummary ? `${metricSummary} ` : "" })}
                {activeLiveCheck && (
                  <span className="muted small">({activeLiveCheck.compute_ms.toFixed(0)}ms)</span>
                )}
              </>
            ) : (
              <>
                <AlertTriangle size={13} />{" "}
                {t("inspector.spliceWarn", { metric: metricSummary ? `${metricSummary} ` : "" })}
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
              <Crosshair size={12} />{" "}
              {t("inspector.snapBtn", {
                start: optStart!.toFixed(2),
                end: optEnd!.toFixed(2),
              })}
            </button>
          )}
        </div>
      )}

      <label className="field-group">
        <span>{t("inspector.qcLabel")}</span>
        <select
          aria-label={t("inspector.qcLabelAria")}
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
              <summary>{t("inspector.vlmSemanticsSummary")}</summary>
              <p>{labelDef.vlm_semantics.definition}</p>
              {labelDef.vlm_semantics.counter_examples.length > 0 && (
                <ul>
                  {labelDef.vlm_semantics.counter_examples.map((ex, i) => (
                    <li key={i}>{t("inspector.counterExamplePrefix", { ex })}</li>
                  ))}
                </ul>
              )}
            </details>
          )}
        </div>
      )}

      <div className="two-col">
        <div className="field-group">
          <span>{t("inspector.segStartLabel")}</span>
          <div className="input-with-btn">
            <input
              type="number"
              aria-label={t("inspector.segStartAria")}
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
                title={t("inspector.segStartNowTitle")}
                onClick={() => {
                  const next = Math.min(
                    seg.t_end - 0.05,
                    Math.max(0, Number(currentTime.toFixed(2))),
                  );
                  applyPatch({ t_start: next });
                }}
              >
                {t("inspector.nowBtn")}
              </button>
            )}
          </div>
        </div>

        <div className="field-group">
          <span>{t("inspector.segEndLabel")}</span>
          <div className="input-with-btn">
            <input
              type="number"
              aria-label={t("inspector.segEndAria")}
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
                title={t("inspector.segEndNowTitle")}
                onClick={() => {
                  const next = Math.max(
                    seg.t_start + 0.05,
                    Math.min(durationS, Number(currentTime.toFixed(2))),
                  );
                  applyPatch({ t_end: next });
                }}
              >
                {t("inspector.nowBtn")}
              </button>
            )}
          </div>
        </div>
      </div>

      <div className="three-col">
        <label className="field-group">
          <span>{t("inspector.trainingRole")}</span>
          <select
            aria-label={t("inspector.trainingRole")}
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
          <span>{t("inspector.splicePolicy")}</span>
          <select
            aria-label={t("inspector.splicePolicy")}
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
          <span>{t("inspector.maskSide")}</span>
          <select
            aria-label={t("inspector.maskSide")}
            disabled={readOnly || seg.splice_policy !== "mask_loss"}
            value={seg.mask_side ?? ""}
            onChange={(e) =>
              applyPatch({
                mask_side: (e.target.value || null) as MaskSide | null,
              })
            }
          >
            {seg.splice_policy !== "mask_loss" && (
              <option value="">{t("inspector.maskSideNone")}</option>
            )}
            <option value="left">{t("inspector.maskSideLeft")}</option>
            <option value="right">{t("inspector.maskSideRight")}</option>
            <option value="both">{t("inspector.maskSideBoth")}</option>
          </select>
        </label>
      </div>

      <div className="evidence-card">
        <div className="card-subtitle">{t("inspector.visualEvidenceTitle")}</div>
        <p>
          <strong>{t("inspector.evidenceStart")}</strong>{" "}
          {seg.visual_evidence.initial_state || "—"} ·{" "}
          <strong>{t("inspector.evidenceProcess")}</strong>{" "}
          {seg.visual_evidence.process_change || "—"} ·{" "}
          <strong>{t("inspector.evidenceEnd")}</strong> {seg.visual_evidence.end_state || "—"}
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
          <span>{t("inspector.flagUnknown")}</span>
        </label>
        {seg.is_unknown && (
          <input
            type="text"
            placeholder={t("inspector.unknownPlaceholder")}
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
            <span>{t("inspector.leadDiagTitle")}</span>
          </div>
          <div className="diag-grid">
            <span>
              {t("inspector.diagGate", {
                value: activeLiveCheck ? activeLiveCheck.mode : seg.guardrail.gate_type,
              })}
            </span>
            <span>{t("inspector.diagVlmRaw", { value: seg.vlm_raw_policy })}</span>
            <span>
              {t("inspector.diagDowngraded", {
                value: effectiveDowngraded ? t("inspector.yes") : t("inspector.no"),
              })}
            </span>
            <span>
              {t("inspector.diagBimanualCoupled", {
                value: diagCoupled ? t("inspector.yes") : t("inspector.no"),
              })}
            </span>
            <span>{t("inspector.diagEefDisp", { value: (diagEefPosM * 100).toFixed(2) })}</span>
            <span>{t("inspector.diagEefRot", { value: diagEefRotRad.toFixed(3) })}</span>
            <span>{t("inspector.diagHandJoint", { value: diagHandRad.toFixed(3) })}</span>
            <span>{t("inspector.diagOtherArm", { value: (diagOtherArmM * 100).toFixed(2) })}</span>
          </div>
          {effectiveReason && <div className="diag-reason">{effectiveReason}</div>}

          <div className="few-shot-actions">
            <button
              type="button"
              className="mini-btn accent"
              data-testid="add-few-shot-btn"
              onClick={() => setFewShotOpen((v) => !v)}
            >
              <BookmarkPlus size={13} /> {t("inspector.addFewShotBtn")}
            </button>
          </div>

          {fewShotOpen && (
            <div className="few-shot-popover" data-testid="few-shot-form">
              <div className="two-col">
                <label className="field-group">
                  <span>{t("inspector.fsScope")}</span>
                  <select value={fsScope} onChange={(e) => setFsScope(e.target.value)}>
                    <option value="global">{t("inspector.fsScopeGlobal")}</option>
                    <option value={`task:${taskId}`}>
                      {t("inspector.fsScopeTask", { taskId })}
                    </option>
                    <option value={`label:${seg.label_id}`}>
                      {t("inspector.fsScopeLabel", { labelId: seg.label_id })}
                    </option>
                  </select>
                </label>
                <label className="field-group">
                  <span>{t("inspector.fsErrorType")}</span>
                  <select
                    value={fsErrorType}
                    onChange={(e) => setFsErrorType(e.target.value as DiagnosisErrorType)}
                  >
                    <option value="fp_overflag">{t("inspector.fsErrFpOverflag")}</option>
                    <option value="fn_missed">{t("inspector.fsErrFnMissed")}</option>
                    <option value="policy_flip">{t("inspector.fsErrPolicyFlip")}</option>
                    <option value="boundary">{t("inspector.fsErrBoundary")}</option>
                    <option value="success_flip">{t("inspector.fsErrSuccessFlip")}</option>
                  </select>
                </label>
              </div>
              <label className="field-group">
                <span>{t("inspector.fsRationale")}</span>
                <textarea
                  rows={2}
                  value={fsRationale}
                  onChange={(e) => setFsRationale(e.target.value)}
                  placeholder={t("inspector.fsRationalePlaceholder")}
                />
              </label>
              <button
                type="button"
                className="primary-btn compact"
                disabled={fsSaving}
                onClick={handleSaveFewShot}
              >
                {fsSaving ? t("inspector.fsSaving") : t("inspector.fsSaveBtn")}
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
                {t("inspector.lightboxTitle", {
                  name: seg.name || seg.label_id,
                  start: seg.t_start.toFixed(2),
                  end: seg.t_end.toFixed(2),
                })}
              </h3>
              <button
                type="button"
                className="mini-btn"
                onClick={() => setFrameLightboxOpen(false)}
              >
                <X size={14} /> {t("common.close")}
              </button>
            </div>
            <div className="frame-lightbox-grid">
              <div className="frame-lightbox-pane">
                <strong>{t("inspector.lightboxStartLabel", { t: seg.t_start.toFixed(2) })}</strong>
                <img
                  src={startThumbUrl}
                  alt={t("inspector.lightboxStartAlt", { t: seg.t_start.toFixed(2) })}
                />
              </div>
              <div className="frame-lightbox-pane">
                <strong>{t("inspector.lightboxEndLabel", { t: seg.t_end.toFixed(2) })}</strong>
                <img
                  src={endThumbUrl}
                  alt={t("inspector.lightboxEndAlt", { t: seg.t_end.toFixed(2) })}
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
