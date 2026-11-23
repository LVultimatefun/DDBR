import React from "react";
import { Keyboard, X } from "lucide-react";

export interface ShortcutsModalProps {
  open: boolean;
  onClose: () => void;
}

interface ShortcutEntry {
  keys: string[];
  desc: string;
}

const SHORTCUT_ITEMS: ShortcutEntry[] = [
  { keys: ["Space"], desc: "Play / Pause" },
  { keys: ["←", "→"], desc: "Previous / Next frame" },
  { keys: ["J", "L"], desc: "Seek -0.5s / +0.5s" },
  { keys: ["Shift+J", "Shift+L"], desc: "Seek -2.0s / +2.0s" },
  { keys: ["[", "]"], desc: "Set selected segment/stage start or end to playhead" },
  { keys: ["N"], desc: "Add QC segment at playhead" },
  { keys: ["S"], desc: "Split macro stage at playhead" },
  { keys: ["R"], desc: "Loop selected segment (±0.5s padding)" },
  { keys: ["K"], desc: "Toggle Cut-Skip preview mode" },
  { keys: ["1", "2", "3"], desc: "Switch policy (keep / mask / cut)" },
  { keys: ["Delete", "Backspace"], desc: "Delete selected segment or stage" },
  { keys: ["Ctrl+Z", "Ctrl+Shift+Z"], desc: "Undo / Redo" },
  { keys: ["Ctrl+S"], desc: "Save draft" },
  { keys: ["?"], desc: "Open / close shortcuts cheat sheet" },
];

export const ShortcutsModal: React.FC<ShortcutsModalProps> = ({ open, onClose }) => {
  if (!open) return null;

  return (
    <div
      className="modal-backdrop"
      data-testid="shortcuts-modal"
      role="dialog"
      aria-modal="true"
      aria-label="Review Studio keyboard shortcuts"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="modal-card modal-card-wide shortcuts-modal-card">
        <div className="inspector-header">
          <h3 style={{ margin: 0, display: "inline-flex", alignItems: "center", gap: 6 }}>
            <Keyboard size={16} /> Review Studio Keyboard Shortcuts
          </h3>
          <button
            type="button"
            className="mini-btn"
            onClick={onClose}
            data-testid="close-shortcuts-modal"
            aria-label="Close shortcuts modal"
          >
            <X size={14} /> Close (Esc)
          </button>
        </div>

        <div className="shortcuts-grid">
          {SHORTCUT_ITEMS.map((item) => (
            <div key={item.keys.join("/")} className="shortcut-row">
              <span>{item.desc}</span>
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
          <strong>Direct Timeline Manipulation Tips:</strong>
          <span>
            • Hover or select a QC segment / macro stage and drag the 8px left/right edge handles to
            scrub boundaries live; releasing commits a single undo step.
          </span>
          <span>
            • Click and drag horizontally (≥ 0.2s) on empty space in Track 3 (QC Segments) to
            brush-create and auto-select a new segment.
          </span>
          <span>
            • Dragging a shared boundary between adjacent macro stages (gap &lt; 0.05s) snaps both
            edges magnetically; hold Alt to detach and move a single edge.
          </span>
          <span>
            • Click any red unwatched gap on the 4px Watch Map bar below the ruler to jump directly
            to unwatched footage.
          </span>
        </div>
      </div>
    </div>
  );
};
