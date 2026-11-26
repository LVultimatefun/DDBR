import React, { useState } from "react";
import { Check, Copy, Cpu, FolderPlus, Sparkles, Terminal } from "lucide-react";
import { useI18n } from "../stores/i18nStore";

export const DEMO_QUICKSTART_COMMAND =
  "python scripts/generate_demo_dataset.py --out ./demo_data --with-sample-run --workspace ./workspace";

export interface OnboardingEmptyStateProps {
  isLead: boolean;
  onOpenIngest?: () => void;
  onOpenVlmSettings?: () => void;
}

export const OnboardingEmptyState: React.FC<OnboardingEmptyStateProps> = ({
  isLead,
  onOpenIngest,
  onOpenVlmSettings,
}) => {
  const { t } = useI18n();
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(DEMO_QUICKSTART_COMMAND);
      }
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    }
  };

  return (
    <section
      className="onboarding-empty-card"
      data-testid="onboarding-empty-state"
      aria-label={t("onboarding.title")}
    >
      <div className="onboarding-header">
        <span className="badge-pill ok">
          <Sparkles size={12} /> {t("onboarding.badge")}
        </span>
        <h2>{t("onboarding.title")}</h2>
        <p className="muted">{t("onboarding.subtitle")}</p>
      </div>

      <div className="onboarding-steps-grid">
        <div className="onboarding-step-card onboarding-step-wide">
          <div className="onboarding-step-title">
            <Terminal size={15} />
            <strong>{t("onboarding.step1Title")}</strong>
          </div>
          <p className="small muted">{t("onboarding.step1Desc")}</p>
          <div className="onboarding-cmd-box">
            <code className="mono small" data-testid="onboarding-demo-cmd">
              {DEMO_QUICKSTART_COMMAND}
            </code>
            <button
              type="button"
              className="secondary-btn compact"
              data-testid="onboarding-copy-cmd-btn"
              aria-label={t("onboarding.copyAria")}
              onClick={() => void handleCopy()}
            >
              {copied ? <Check size={13} /> : <Copy size={13} />}
              {copied ? t("onboarding.copiedBtn") : t("onboarding.copyCmdBtn")}
            </button>
          </div>
        </div>

        <div className="onboarding-step-card">
          <div className="onboarding-step-title">
            <FolderPlus size={15} />
            <strong>{t("onboarding.step2Title")}</strong>
          </div>
          <p className="small muted">{t("onboarding.step2Desc")}</p>
          {isLead && onOpenIngest ? (
            <button
              type="button"
              className="accent-btn compact"
              data-testid="onboarding-ingest-btn"
              onClick={onOpenIngest}
            >
              <FolderPlus size={14} /> {t("onboarding.ingestBtn")}
            </button>
          ) : (
            <p className="small muted">{t("onboarding.reviewerHint")}</p>
          )}
        </div>

        <div className="onboarding-step-card">
          <div className="onboarding-step-title">
            <Cpu size={15} />
            <strong>{t("onboarding.step3Title")}</strong>
          </div>
          <p className="small muted">{t("onboarding.step3Desc")}</p>
          {isLead && onOpenVlmSettings ? (
            <button
              type="button"
              className="secondary-btn compact"
              data-testid="onboarding-vlm-btn"
              onClick={onOpenVlmSettings}
            >
              <Cpu size={14} /> {t("onboarding.configureVlmBtn")}
            </button>
          ) : (
            <p className="small muted">{t("onboarding.reviewerHint")}</p>
          )}
        </div>
      </div>
    </section>
  );
};
