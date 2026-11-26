import React, { useEffect, useRef, useState } from "react";
import { Calculator, Play, Sparkles, X } from "lucide-react";
import { estimateJobCost } from "../api/client";
import type { JobEstimateResponse } from "../api/types";
import { useI18n } from "../stores/i18nStore";

export interface CostEstimatePopoverProps {
  open: boolean;
  episodeIds: string[];
  stage?: string;
  fresh?: boolean;
  onClose: () => void;
  onConfirmRun: () => Promise<void> | void;
}

export function formatTokenCount(tokens: number): string {
  if (!Number.isFinite(tokens) || tokens <= 0) return "0";
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(2)}M`;
  if (tokens >= 10_000) return `${(tokens / 1_000).toFixed(1)}k`;
  if (tokens >= 1_000) return `${(tokens / 1_000).toFixed(2)}k`;
  return String(Math.round(tokens));
}

export function formatUsdCost(usd: number): string {
  if (!Number.isFinite(usd) || usd <= 0) return "$0.00";
  if (usd < 0.01) return `$${usd.toFixed(4)}`;
  return `$${usd.toFixed(3)}`;
}

export const CostEstimatePopover: React.FC<CostEstimatePopoverProps> = ({
  open,
  episodeIds,
  stage,
  fresh = false,
  onClose,
  onConfirmRun,
}) => {
  const { t } = useI18n();
  const tRef = useRef(t);
  tRef.current = t;

  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [estimate, setEstimate] = useState<JobEstimateResponse | null>(null);

  const idsKey = episodeIds.join("\u0000");

  useEffect(() => {
    if (!open || episodeIds.length === 0) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    setEstimate(null);

    void estimateJobCost({
      episode_ids: episodeIds,
      stage: stage || undefined,
      fresh,
    })
      .then((res) => {
        if (!cancelled) setEstimate(res);
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : tRef.current("estimate.errLoad"));
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [open, idsKey, stage, fresh, episodeIds]);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  const activeSavings = estimate?.token_saving_active
    ? ([
        estimate.token_saving_active.low_detail_stages && t("estimate.savingLowDetail"),
        estimate.token_saving_active.static_frame_dedup && t("estimate.savingStaticDedup"),
        estimate.token_saving_active.adaptive_s3_step && t("estimate.savingAdaptiveS3"),
        estimate.token_saving_active.short_episode_s1_merge && t("estimate.savingShortS1Merge"),
        estimate.token_saving_active.s4b_dynamic_frames && t("estimate.savingDynamicS4b"),
      ].filter(Boolean) as string[])
    : [];

  const handleConfirm = async () => {
    if (submitting) return;
    setSubmitting(true);
    try {
      await onConfirmRun();
      onClose();
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div
      className="cost-estimate-popover"
      role="dialog"
      aria-label={t("estimate.title")}
      data-testid="cost-estimate-popover"
    >
      <div className="section-header-row">
        <strong style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <Calculator size={15} /> {t("estimate.title")}
        </strong>
        <button
          type="button"
          className="mini-btn"
          onClick={onClose}
          aria-label={t("estimate.closeAria")}
          data-testid="close-cost-estimate-btn"
        >
          <X size={13} />
        </button>
      </div>

      {loading ? (
        <div className="muted small" style={{ padding: "12px 0" }}>
          {t("estimate.loading")}
        </div>
      ) : error ? (
        <div className="error-banner small">{error}</div>
      ) : estimate ? (
        <>
          <div className="small muted tabular-nums">
            {t("estimate.summaryEpisodes", {
              count: estimate.episode_count,
              dur: estimate.total_duration_s.toFixed(1),
            })}
          </div>

          <div className="cost-estimate-metrics">
            <div className="cost-metric-box">
              <span className="cost-metric-lbl">{t("estimate.totalTokens")}</span>
              <strong className="cost-metric-val tabular-nums" data-testid="estimate-total-tokens">
                {formatTokenCount(estimate.estimated_total_tokens)}
              </strong>
            </div>
            <div className="cost-metric-box">
              <span className="cost-metric-lbl">{t("estimate.totalCost")}</span>
              <strong className="cost-metric-val tabular-nums" data-testid="estimate-total-cost">
                {formatUsdCost(estimate.estimated_cost_usd)}
              </strong>
            </div>
            <div className="cost-metric-box">
              <span className="cost-metric-lbl">{t("estimate.cacheHits")}</span>
              <strong className="cost-metric-val tabular-nums">
                {estimate.estimated_cache_hits}
              </strong>
            </div>
          </div>

          {estimate.per_stage.length > 0 && (
            <div className="cost-estimate-table-wrap">
              <table className="cost-estimate-table" data-testid="cost-estimate-stage-table">
                <thead>
                  <tr>
                    <th>{t("estimate.colStage")}</th>
                    <th>{t("estimate.colModel")}</th>
                    <th>{t("estimate.colDetail")}</th>
                    <th>{t("estimate.colTokens")}</th>
                    <th>{t("estimate.colCost")}</th>
                    <th>{t("estimate.colCache")}</th>
                  </tr>
                </thead>
                <tbody>
                  {estimate.per_stage.map((st) => (
                    <tr key={st.stage}>
                      <td className="mono">{st.stage}</td>
                      <td className="mono small">{st.model}</td>
                      <td>
                        <span className="badge-pill">{st.image_detail}</span>
                      </td>
                      <td className="tabular-nums">{formatTokenCount(st.estimated_tokens)}</td>
                      <td className="tabular-nums">{formatUsdCost(st.estimated_cost_usd)}</td>
                      <td className="tabular-nums">{st.cached_episodes}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {activeSavings.length > 0 && (
            <div className="cost-saving-section">
              <div className="small muted" style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                <Sparkles size={12} /> {t("estimate.savingHeading")}
              </div>
              <div className="cost-saving-pills" data-testid="cost-estimate-saving-pills">
                {activeSavings.map((label) => (
                  <span key={label} className="badge-pill ok">
                    {label}
                  </span>
                ))}
              </div>
            </div>
          )}

          <div className="modal-actions">
            <button type="button" className="secondary-btn compact" onClick={onClose}>
              {t("common.cancel")}
            </button>
            <button
              type="button"
              className="primary-btn compact"
              disabled={submitting}
              data-testid="cost-estimate-confirm-btn"
              onClick={() => void handleConfirm()}
            >
              <Play size={13} />
              {submitting ? t("estimate.submitting") : t("estimate.confirmRunBtn")}
            </button>
          </div>
        </>
      ) : null}
    </div>
  );
};
