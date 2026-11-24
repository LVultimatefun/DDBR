import React from "react";
import { Keyboard, X } from "lucide-react";
import { useI18n } from "../stores/i18nStore";

export interface ShortcutsModalProps {
  open: boolean;
  onClose: () => void;
}

interface ShortcutEntry {
  keys: string[];
  descKey: string;
}

const SHORTCUT_ITEMS: ShortcutEntry[] = [
  { keys: ["Space"], descKey: "shortcuts.playPause" },
  { keys: ["←", "→"], descKey: "shortcuts.prevNextFrame" },
  { keys: ["J", "L"], descKey: "shortcuts.seekHalf" },
  { keys: ["Shift+J", "Shift+L"], descKey: "shortcuts.seekTwo" },
  { keys: ["[", "]"], descKey: "shortcuts.setEdge" },
  { keys: ["N"], descKey: "shortcuts.addSegment" },
  { keys: ["S"], descKey: "shortcuts.splitStage" },
  { keys: ["R"], descKey: "shortcuts.loopSeg" },
  { keys: ["K"], descKey: "shortcuts.toggleCutSkip" },
  { keys: ["1", "2", "3"], descKey: "shortcuts.switchPolicy" },
  { keys: ["Delete", "Backspace"], descKey: "shortcuts.deleteItem" },
  { keys: ["Ctrl+Z", "Ctrl+Shift+Z"], descKey: "shortcuts.undoRedo" },
  { keys: ["Ctrl+S"], descKey: "shortcuts.saveDraft" },
  { keys: ["?"], descKey: "shortcuts.toggleSheet" },
];

export const ShortcutsModal: React.FC<ShortcutsModalProps> = ({ open, onClose }) => {
  const { t } = useI18n();
  if (!open) return null;

  return (
    <div
      className="modal-backdrop"
      data-testid="shortcuts-modal"
      role="dialog"
      aria-modal="true"
      aria-label={t("shortcuts.dialogAria")}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="modal-card modal-card-wide shortcuts-modal-card">
        <div className="inspector-header">
          <h3 style={{ margin: 0, display: "inline-flex", alignItems: "center", gap: 6 }}>
            <Keyboard size={16} /> {t("shortcuts.title")}
          </h3>
          <button
            type="button"
            className="mini-btn"
            onClick={onClose}
            data-testid="close-shortcuts-modal"
            aria-label={t("shortcuts.closeAria")}
          >
            <X size={14} /> {t("shortcuts.closeBtn")}
          </button>
        </div>

        <div className="shortcuts-grid">
          {SHORTCUT_ITEMS.map((item) => (
            <div key={item.keys.join("/")} className="shortcut-row">
              <span>{t(item.descKey)}</span>
              <span className="shortcut-keys">
                {item.keys.map((k, idx) => (
                  <React.Fragment key={k}>
                    {idx > 0 && <span className="muted">/</span>}
                    <kbd className="kbd-cap">{k}</kbd>
                  </React.Fragment>
                ))}
              </span>
            </div>
          ))}
        </div>

        <div className="shortcuts-tips">
          <strong>{t("shortcuts.tipsTitle")}</strong>
          <span>{t("shortcuts.tip1")}</span>
          <span>{t("shortcuts.tip2")}</span>
          <span>{t("shortcuts.tip3")}</span>
          <span>{t("shortcuts.tip4")}</span>
        </div>
      </div>
    </div>
  );
};
