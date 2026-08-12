"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ControlChart } from "@/components/ControlChart";
import { ErrorBanner, PageHeader, Panel, PrimaryButton, SelectInput, Spinner } from "@/components/ui";
import { api, getToken } from "@/lib/api";
import { formatTimestamp } from "@/lib/format";
import type { LivePointMessage, Signal } from "@/lib/types";
import { connectLiveSocket, type LiveSocketHandle } from "@/lib/ws";

const MAX_POINTS = 200;

interface AlertItem extends Signal {
  ts?: string;
}

export default function LivePage() {
  const streamsQ = useQuery({
    queryKey: ["streams"],
    queryFn: api.listStreams,
    retry: false,
  });

  const [streamKey, setStreamKey] = useState("");
  const [manualKey, setManualKey] = useState("line-1");
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [values, setValues] = useState<number[]>([]);
  const [ooc, setOoc] = useState<number[]>([]);
  const [streamIndices, setStreamIndices] = useState<number[]>([]);
  const [ucl, setUcl] = useState<number>(0);
  const [cl, setCl] = useState<number>(0);
  const [lcl, setLcl] = useState<number>(0);
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const sockRef = useRef<LiveSocketHandle | null>(null);

  const activeKey = streamKey || manualKey;

  const onMessage = useCallback((msg: LivePointMessage) => {
    if (msg.type === "limits" && msg.limits) {
      setUcl(Array.isArray(msg.limits.ucl) ? msg.limits.ucl[0] : msg.limits.ucl);
      setCl(msg.limits.center);
      setLcl(Array.isArray(msg.limits.lcl) ? msg.limits.lcl[0] : msg.limits.lcl);
      return;
    }

    if (msg.type === "point" || msg.value !== undefined) {
      const v = msg.value;
      if (typeof v === "number") {
        setValues((prev) => [...prev, v].slice(-MAX_POINTS));
        setStreamIndices((prev) => {
          const streamIdx = typeof msg.index === "number" ? msg.index : (prev.at(-1) ?? -1) + 1;
          return [...prev, streamIdx].slice(-MAX_POINTS);
        });
        if (typeof msg.ucl === "number") setUcl(msg.ucl);
        if (typeof msg.center === "number") setCl(msg.center);
        if (typeof msg.lcl === "number") setLcl(msg.lcl);
      }
    }

    const sigs = msg.signals || (msg.signal ? [msg.signal] : []);
    if (sigs.length || msg.type === "alert") {
      // Map server stream index → chart array index within the sliding window
      setStreamIndices((indices) => {
        const streamIdx = typeof msg.index === "number" ? msg.index : indices.at(-1);
        if (typeof streamIdx === "number") {
          const chartIdx = indices.lastIndexOf(streamIdx);
          if (chartIdx >= 0) {
            setOoc((prev) => [...prev, chartIdx].slice(-MAX_POINTS));
          }
        }
        return indices;
      });
      setAlerts((prev) => {
        const added = sigs.map((s) => ({ ...s, ts: msg.ts }));
        if (!added.length && msg.type === "alert") {
          added.push({
            rule_id: "alert",
            rule_name: "Alert",
            index: msg.index ?? 0,
            value: msg.value ?? 0,
            description: msg.message || "Out of control",
            ts: msg.ts,
          });
        }
        return [...added, ...prev].slice(0, 50);
      });
    }

    if (msg.type === "error") {
      setError(msg.message || "WebSocket error");
    }
  }, []);

  function disconnect() {
    sockRef.current?.close();
    sockRef.current = null;
    setConnected(false);
  }

  function connect() {
    setError(null);
    disconnect();
    setValues([]);
    setOoc([]);
    setStreamIndices([]);
    setAlerts([]);
    const handle = connectLiveSocket(activeKey, onMessage, {
      token: getToken(),
      reconnect: true,
      onOpen: () => setConnected(true),
      onClose: () => setConnected(false),
      onError: () => setError("WebSocket connection failed"),
    });
    sockRef.current = handle;
  }

  useEffect(() => () => disconnect(), []);

  const streamOptions = useMemo(
    () => streamsQ.data?.streams?.filter((s) => s.active) ?? [],
    [streamsQ.data],
  );

  return (
    <div>
      <PageHeader
        title="Live Monitoring"
        hideTitle
        subtitle="Phase II stream against frozen limits — WebSocket chart + alert feed"
        actions={
          <span
            className={`rounded-pill border px-3 py-1 text-[10px] font-semibold uppercase tracking-wider ${
              connected
                ? "border-aspc-ok/40 bg-aspc-ok/15 text-aspc-ok"
                : "border-aspc-border text-aspc-muted"
            }`}
          >
            {connected ? "Connected" : "Disconnected"}
          </span>
        }
      />

      {error && <ErrorBanner message={error} />}

      <Panel title="Stream" className="mb-6">
        <div className="grid gap-4 md:grid-cols-3">
          <SelectInput
            id="stream_select"
            label="Registered stream"
            value={streamKey}
            onChange={(e) => setStreamKey(e.target.value)}
          >
            <option value="">— Manual key —</option>
            {streamOptions.map((s) => (
              <option key={s.stream_key} value={s.stream_key}>
                {s.stream_key}
                {s.chart_type ? ` (${s.chart_type})` : ""}
              </option>
            ))}
          </SelectInput>
          <div>
            <label
              htmlFor="manual_key"
              className="mb-1.5 block text-xs uppercase tracking-wider text-aspc-muted"
            >
              Stream key
            </label>
            <input
              id="manual_key"
              value={streamKey || manualKey}
              disabled={!!streamKey}
              onChange={(e) => setManualKey(e.target.value)}
              className="w-full rounded-2xl border border-aspc-border bg-aspc-elevated px-3 py-2.5 text-sm outline-none focus:border-aspc-accent/50 disabled:opacity-50"
            />
          </div>
          <div className="flex items-end gap-2">
            <PrimaryButton onClick={connect} disabled={!activeKey}>
              Connect
            </PrimaryButton>
            <button
              type="button"
              onClick={disconnect}
              className="rounded-pill border border-aspc-border px-4 py-2.5 text-sm text-aspc-muted hover:text-aspc-text"
            >
              Disconnect
            </button>
          </div>
        </div>
        {streamsQ.isError && (
          <p className="mt-3 text-xs text-aspc-muted">
            Could not list streams — enter a stream key manually.
          </p>
        )}
      </Panel>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <ControlChart
            title={`Live · ${activeKey || "—"}`}
            values={values}
            ucl={ucl}
            cl={cl}
            lcl={lcl}
            oocIndices={ooc}
          />
          {!connected && values.length === 0 && (
            <p className="mt-3 text-sm text-aspc-muted">Connect to a stream to receive points.</p>
          )}
        </div>
        <Panel title="Alert Feed">
          {alerts.length === 0 && <p className="text-sm text-aspc-muted">No alerts yet.</p>}
          <ul className="max-h-[22rem] space-y-2 overflow-y-auto">
            {alerts.map((a, i) => (
              <li
                key={`${a.rule_id}-${a.index}-${i}`}
                className="rounded-2xl border border-aspc-stop/30 bg-aspc-stop/10 px-3 py-2 text-sm"
              >
                <div className="font-mono text-xs text-aspc-stop">
                  [{a.rule_id}] idx {a.index}
                </div>
                <p className="mt-0.5 text-aspc-text">{a.description || a.rule_name}</p>
                <div className="mt-1 text-[11px] text-aspc-muted">
                  value={a.value}
                  {a.ts ? ` · ${formatTimestamp(a.ts)}` : ""}
                </div>
              </li>
            ))}
          </ul>
          {streamsQ.isLoading && <Spinner label="Loading streams…" />}
        </Panel>
      </div>
    </div>
  );
}
