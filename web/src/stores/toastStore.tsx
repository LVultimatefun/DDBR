import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  ToastContainer,
  type ToastItem,
  type ToastVariant,
} from "../components/ToastContainer";

export type { ToastItem, ToastVariant };

export interface ShowToastInput {
  message: string;
  variant?: ToastVariant;
  testId?: string;
  durationMs?: number;
}

export interface ToastContextValue {
  toasts: ToastItem[];
  isProvided: boolean;
  showToast: (input: ShowToastInput) => string;
  dismissToast: (id: string) => void;
  dismissByTestId: (testId: string) => void;
  StandaloneHost: React.ReactNode;
}

const ToastContext = createContext<ToastContextValue | null>(null);

function useToastStateManager() {
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const timersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());
  const seqRef = useRef(0);

  const clearTimer = useCallback((id: string) => {
    const timer = timersRef.current.get(id);
    if (timer !== undefined) {
      clearTimeout(timer);
      timersRef.current.delete(id);
    }
  }, []);

  useEffect(() => {
    const timers = timersRef.current;
    return () => {
      for (const timer of timers.values()) {
        clearTimeout(timer);
      }
      timers.clear();
    };
  }, []);

  const dismissToast = useCallback(
    (id: string) => {
      clearTimer(id);
      setToasts((prev) => prev.filter((item) => item.id !== id));
    },
    [clearTimer],
  );

  const dismissByTestId = useCallback(
    (testId: string) => {
      setToasts((prev) => {
        for (const item of prev) {
          if (item.testId === testId) {
            clearTimer(item.id);
          }
        }
        return prev.filter((item) => item.testId !== testId);
      });
    },
    [clearTimer],
  );

  const showToast = useCallback(
    (input: ShowToastInput): string => {
      seqRef.current += 1;
      const id = `toast_${Date.now().toString(36)}_${seqRef.current}`;
      const durationMs = input.durationMs ?? 4500;
      const item: ToastItem = {
        id,
        message: input.message,
        variant: input.variant ?? "info",
        testId: input.testId,
        durationMs,
      };

      setToasts((prev) => {
        let next = prev;
        if (input.testId) {
          for (const existing of prev) {
            if (existing.testId === input.testId) {
              clearTimer(existing.id);
            }
          }
          next = prev.filter((existing) => existing.testId !== input.testId);
        }
        return [...next, item];
      });

      if (durationMs > 0) {
        const timer = setTimeout(() => {
          timersRef.current.delete(id);
          setToasts((prev) => prev.filter((t) => t.id !== id));
        }, durationMs);
        timersRef.current.set(id, timer);
      }
      return id;
    },
    [clearTimer],
  );

  return { toasts, showToast, dismissToast, dismissByTestId };
}

export const ToastProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const mgr = useToastStateManager();

  const value = useMemo<ToastContextValue>(
    () => ({
      ...mgr,
      isProvided: true,
      StandaloneHost: null,
    }),
    [mgr],
  );

  return <ToastContext.Provider value={value}>{children}</ToastContext.Provider>;
};

export const GlobalToastContainer: React.FC = () => {
  const ctx = useContext(ToastContext);
  if (!ctx) return null;
  return <ToastContainer toasts={ctx.toasts} onDismiss={ctx.dismissToast} />;
};

export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext);
  const localMgr = useToastStateManager();

  const standaloneHost = useMemo(() => {
    if (ctx) return null;
    return <ToastContainer toasts={localMgr.toasts} onDismiss={localMgr.dismissToast} />;
  }, [ctx, localMgr.toasts, localMgr.dismissToast]);

  if (ctx) return ctx;
  return {
    ...localMgr,
    isProvided: false,
    StandaloneHost: standaloneHost,
  };
}
