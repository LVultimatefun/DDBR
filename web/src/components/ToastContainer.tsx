import React from "react";
import { AlertTriangle, CheckCircle2, Info, X, XCircle } from "lucide-react";

export type ToastVariant = "success" | "info" | "warning" | "error";

export interface ToastItem {
  id: string;
  message: string;
  variant: ToastVariant;
  testId?: string;
  durationMs?: number;
}

export interface ToastContainerProps {
  toasts: ToastItem[];
  onDismiss: (id: string) => void;
}

const ICON_MAP: Record<ToastVariant, React.ReactNode> = {
  success: <CheckCircle2 size={15} />,
  info: <Info size={15} />,
  warning: <AlertTriangle size={15} />,
  error: <XCircle size={15} />,
};

export const ToastContainer: React.FC<ToastContainerProps> = ({ toasts, onDismiss }) => {
  if (toasts.length === 0) return null;

  return (
    <div className="toast-container" data-testid="toast-container" aria-live="polite">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={`toast-item toast-${toast.variant}`}
          data-testid={toast.testId}
          role="status"
        >
          <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
            {ICON_MAP[toast.variant]}
            <span>{toast.message}</span>
          </span>
          <button
            type="button"
            className="toast-close"
            aria-label="Dismiss notification"
            onClick={() => onDismiss(toast.id)}
          >
            <X size={14} />
          </button>
        </div>
      ))}
    </div>
  );
};
