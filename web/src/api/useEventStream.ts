import { useEffect, useRef, useState } from "react";
import { useAuth } from "../stores/authStore";

export interface UseEventStreamOptions<T> {
  url: string | null;
  enabled?: boolean;
  onMessage?: (data: T) => void;
  isTerminal?: (data: T) => boolean;
}

export interface UseEventStreamResult<T> {
  lastEvent: T | null;
  connected: boolean;
  error: string | null;
}

export function useEventStream<T = Record<string, unknown>>(
  options: UseEventStreamOptions<T>,
): UseEventStreamResult<T> {
  const { url, enabled = true, onMessage, isTerminal } = options;
  const { token } = useAuth();
  const [lastEvent, setLastEvent] = useState<T | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;
  const isTerminalRef = useRef(isTerminal);
  isTerminalRef.current = isTerminal;

  useEffect(() => {
    if (!enabled || !url || typeof EventSource === "undefined") {
      setConnected(false);
      return;
    }

    let closed = false;
    let source: EventSource | null = null;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let attempt = 0;

    const connect = () => {
      if (closed) return;
      const fullUrl = token
        ? `${url}${url.includes("?") ? "&" : "?"}token=${encodeURIComponent(token)}`
        : url;

      source = new EventSource(fullUrl);
      source.onopen = () => {
        attempt = 0;
        setConnected(true);
        setError(null);
      };

      source.onmessage = (evt) => {
        if (!evt.data) return;
        try {
          const parsed = JSON.parse(evt.data) as T;
          setLastEvent(parsed);
          onMessageRef.current?.(parsed);
          if (isTerminalRef.current?.(parsed)) {
            closed = true;
            source?.close();
            setConnected(false);
          }
        } catch {
          // Ignore keepalive / non-JSON SSE pings
        }
      };

      source.onerror = () => {
        setConnected(false);
        source?.close();
        if (closed) return;
        attempt += 1;
        setError("SSE connection interrupted, reconnecting...");
        const delayMs = Math.min(8000, 500 * Math.pow(2, Math.min(attempt, 4)));
        retryTimer = setTimeout(connect, delayMs);
      };
    };

    connect();

    return () => {
      closed = true;
      if (retryTimer) clearTimeout(retryTimer);
      source?.close();
      setConnected(false);
    };
  }, [url, enabled, token]);

  return { lastEvent, connected, error };
}
