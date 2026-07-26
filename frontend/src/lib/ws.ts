import type { LivePointMessage } from "./types";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

export interface LiveSocketOptions {
  token?: string | null;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (ev: Event) => void;
  /** Reconnect with exponential backoff (default true). */
  reconnect?: boolean;
  maxBackoffMs?: number;
}

export interface LiveSocketHandle {
  close: () => void;
}

/** Connect to NEXT_PUBLIC_WS_URL/ws/live/{streamKey} with optional reconnect. */
export function connectLiveSocket(
  streamKey: string,
  onMessage: (msg: LivePointMessage) => void,
  options: LiveSocketOptions = {},
): LiveSocketHandle {
  const params = new URLSearchParams();
  if (options.token) params.set("token", options.token);
  const qs = params.toString();
  const url = `${WS_URL}/ws/live/${encodeURIComponent(streamKey)}${qs ? `?${qs}` : ""}`;

  let closed = false;
  let ws: WebSocket | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let attempt = 0;
  const maxBackoff = options.maxBackoffMs ?? 15_000;
  const shouldReconnect = options.reconnect !== false;

  function clearTimer() {
    if (timer) {
      clearTimeout(timer);
      timer = null;
    }
  }

  function connect() {
    if (closed) return;
    clearTimer();
    ws = new WebSocket(url);

    ws.onopen = () => {
      attempt = 0;
      options.onOpen?.();
    };

    ws.onclose = () => {
      options.onClose?.();
      if (!closed && shouldReconnect) {
        const delay = Math.min(1000 * 2 ** attempt, maxBackoff);
        attempt += 1;
        timer = setTimeout(connect, delay);
      }
    };

    ws.onerror = (ev) => options.onError?.(ev);

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(String(ev.data)) as LivePointMessage;
        onMessage(msg);
      } catch {
        onMessage({ type: "error", message: "Invalid WebSocket payload" });
      }
    };
  }

  connect();

  return {
    close: () => {
      closed = true;
      clearTimer();
      if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        ws.close();
      }
      ws = null;
    },
  };
}
