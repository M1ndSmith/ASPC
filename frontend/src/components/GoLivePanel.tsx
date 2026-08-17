"use client";

import { useId, useState } from "react";
import { Button, Panel, Term, TextInput } from "@/components/ui";
import { ApiError, api } from "@/lib/api";

const API_KEY_STORAGE = "aspc_api_key";

export function GoLivePanel({
  limitsVersion,
  streamKey,
  onStreamKeyChange,
  disabled,
  onSuccess,
  onOpenLive,
  onLimitsChange,
  title = "Go live",
}: {
  limitsVersion?: string;
  streamKey: string;
  onStreamKeyChange: (value: string) => void;
  disabled?: boolean;
  onSuccess?: (streamKey: string, limitsVersion: string) => void;
  onOpenLive?: (streamKey: string, limitsVersion: string) => void;
  onLimitsChange?: (value: string) => void;
  title?: string;
}) {
  const [apiKey, setApiKey] = useState(() =>
    typeof window !== "undefined" ? localStorage.getItem(API_KEY_STORAGE) || "" : "",
  );
  const [busy, setBusy] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [done, setDone] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const uid = useId();

  async function activate() {
    if (!limitsVersion) {
      setError("No frozen limits version — Phase I must freeze before go-live");
      return;
    }
    if (!apiKey) {
      setError("API key required to register a stream");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      localStorage.setItem(API_KEY_STORAGE, apiKey);
      await api.registerStream({ stream_key: streamKey }, apiKey);
      await api.goLive(streamKey, { limits_version: limitsVersion }, apiKey);
      setDone(true);
      setConfirming(false);
      onSuccess?.(streamKey, limitsVersion);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : (err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Panel title={title}>
      <p className="mb-4 text-sm leading-relaxed text-aspc-muted">
        <Term k="go-live" /> attaches this <Term k="limits-version" /> to a named{" "}
        <Term k="stream-key" /> so new measurements are judged against locked limits.
      </p>
      {error && <p className="mb-3 text-sm text-aspc-accent">{error}</p>}
      {done && (
        <p className="mb-3 text-sm font-medium text-aspc-ok">
          Stream {streamKey} is live against frozen limits.
        </p>
      )}
      <div className="grid gap-4 md:grid-cols-2">
        <TextInput
          id={`${uid}-stream`}
          label="Stream key"
          value={streamKey}
          onChange={(e) => onStreamKeyChange(e.target.value)}
        />
        {onLimitsChange && (
          <TextInput
            id={`${uid}-limits`}
            label="Frozen limits version"
            value={limitsVersion || ""}
            onChange={(e) => onLimitsChange(e.target.value)}
            placeholder="from Analyze result"
          />
        )}
        <TextInput
          id={`${uid}-key`}
          label="API key"
          type="password"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
        />
      </div>
      <p className="mt-3 text-xs text-aspc-muted">
        Requires <Term k="timescale" /> for the stream registry. Limits version:{" "}
        <span className="font-mono">{limitsVersion || "—"}</span>
      </p>
      <div className="mt-4 flex flex-wrap items-center gap-2">
        {!confirming ? (
            <Button
              type="button"
              disabled={busy || disabled || !limitsVersion || done}
              onClick={() => setConfirming(true)}
              tip="Turn this analysis into a live watch on a named line. You will be asked to confirm."
            >
            Register + go live
          </Button>
        ) : (
          <>
            <Button type="button" disabled={busy} onClick={activate} tip="Yes: start watching this line against the locked limits.">
              {busy ? "Activating…" : "Confirm go-live"}
            </Button>
            <Button type="button" variant="secondary" disabled={busy} onClick={() => setConfirming(false)} tip="Do not start the live watch yet.">
              Cancel
            </Button>
          </>
        )}
        {onOpenLive && limitsVersion && (
          <Button type="button" variant="ghost" onClick={() => onOpenLive(streamKey, limitsVersion)} tip="Open the Live page without registering again.">
            Open Live only
          </Button>
        )}
      </div>
    </Panel>
  );
}
