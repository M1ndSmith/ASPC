"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useSearchParams } from "next/navigation";
import { ControlChart } from "@/components/ControlChart";
import { ErrorBanner, PageHeader, Panel, PrimaryButton, SelectInput, Spinner } from "@/components/ui";
import { api, getToken } from "@/lib/api";
import { formatTimestamp } from "@/lib/format";
import type { LivePointMessage, Signal } from "@/lib/types";
import { connectLiveSocket, type LiveSocketHandle } from "@/lib/ws";

const MAX_POINTS = 200;
const API_KEY_STORAGE = "aspc_api_key";

interface AlertItem extends Signal {
  ts?: string;
  explanation?: string;
}

export default function LivePageInner() {
  const search = useSearchParams();
  const streamsQ = useQuery({
    queryKey: ["streams"],
    queryFn: api.listStreams,
    retry: false,
  });

  const [streamKey, setStreamKey] = useState(search.get("stream") || "");
  const [manualKey, setManualKey] = useState(search.get("stream") || "line-1");
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [values, setValues] = useState<number[]>([]);
  const [ooc, setOoc] = useState<number[]>([]);
  const [streamIndices, setStreamIndices] = useState<number[]>([]);
  const [ucl, setUcl] = useState<number>(0);
  const [cl, setCl] = useState<number>(0);
  const [lcl, setLcl] = useState<number>(0);
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [apiKey, setApiKey] = useState(() =>
    typeof window !== "undefined" ? localStorage.getItem(API_KEY_STORAGE) || "" : "",
  );
  const [limitsVersion, setLimitsVersion] = useState(search.get("limits") || "");
  const [goLiveBusy, setGoLiveBusy] = useState(false);
  const [explain, setExplain] = useState<Record<string, unknown> | null>(null);
  const [sensorHealth, setSensorHealth] = useState<{
    window?: number;
    rates?: Record<string, number>;
  } | null>(null);
  const sockRef = useRef<LiveSocketHandle | null>(null);

  const activeKey = streamKey || manualKey;

  const onMessage = useCallback(
    (
      msg: LivePointMessage & {
        explanations?: { operator_summary?: string }[];
        sensor_health?: { window?: number; rates?: Record<string, number> };
      },
    ) => {
      if (msg.type === "limits" && msg.limits) {
        setUcl(Array.isArray(msg.limits.ucl) ? msg.limits.ucl[0] : msg.limits.ucl);
        setCl(msg.limits.center);
        setLcl(Array.isArray(msg.limits.lcl) ? msg.limits.lcl[0] : msg.limits.lcl);
        return;
      }

      if (msg.sensor_health) {
        setSensorHealth(msg.sensor_health);
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
          const added = sigs.map((s, i) => ({
            ...s,
            ts: msg.ts,
            explanation: msg.explanations?.[i]?.operator_summary,
          }));
          if (!added.length && msg.type === "alert") {
            added.push({
              rule_id: "alert",
              rule_name: "Alert",
              index: msg.index ?? 0,
              value: msg.value ?? 0,
              description: msg.message || "Out of control",
              ts: msg.ts,
              explanation: undefined,
            });
          }
          return [...added, ...prev].slice(0, 50);
        });
      }

      if (msg.type === "error") {
        setError(msg.message || "WebSocket error");
      }
    },
    [],
  );

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

  async function onGoLive() {
    if (!activeKey || !limitsVersion || !apiKey) {
      setError("Stream key, limits version, and API key are required to go live");
      return;
    }
    setGoLiveBusy(true);
    setError(null);
    try {
      localStorage.setItem(API_KEY_STORAGE, apiKey);
      await api.registerStream({ stream_key: activeKey }, apiKey);
      await api.goLive(activeKey, { limits_version: limitsVersion }, apiKey);
      await streamsQ.refetch();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setGoLiveBusy(false);
    }
  }

  async function openExplain(a: AlertItem) {
    try {
      const body = await api.explain({
        signal: {
          rule_id: a.rule_id,
          rule_name: a.rule_name,
          index: a.index,
          value: a.value,
          description: a.description,
          side: a.side,
        },
        limits_version: limitsVersion || undefined,
      });
      setExplain(body);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  useEffect(() => () => disconnect(), []);

  const streamOptions = useMemo(
    () => streamsQ.data?.streams?.filter((s) => s.active) ?? [],
    [streamsQ.data],
  );

  const healthRates = sensorHealth?.rates || {};

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
      </Panel>

      <Panel title="Go live" className="mb-6">
        <div className="grid gap-4 md:grid-cols-3">
          <div>
            <label
              htmlFor="limits_version"
              className="mb-1.5 block text-xs uppercase tracking-wider text-aspc-muted"
            >
              Frozen limits version
            </label>
            <input
              id="limits_version"
              value={limitsVersion}
              onChange={(e) => setLimitsVersion(e.target.value)}
              className="w-full rounded-2xl border border-aspc-border bg-aspc-elevated px-3 py-2.5 text-sm outline-none focus:border-aspc-accent/50"
              placeholder="from Analyze result"
            />
          </div>
          <div>
            <label
              htmlFor="api_key"
              className="mb-1.5 block text-xs uppercase tracking-wider text-aspc-muted"
            >
              X-API-Key
            </label>
            <input
              id="api_key"
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              className="w-full rounded-2xl border border-aspc-border bg-aspc-elevated px-3 py-2.5 text-sm outline-none focus:border-aspc-accent/50"
              placeholder="stream mutation key"
            />
          </div>
          <div className="flex items-end">
            <PrimaryButton onClick={onGoLive} disabled={goLiveBusy || !activeKey}>
              {goLiveBusy ? "Activating…" : "Register + go live"}
            </PrimaryButton>
          </div>
        </div>
      </Panel>

      {sensorHealth && (sensorHealth.window ?? 0) > 0 && (
        <Panel title="Sensor health" className="mb-6">
          <p className="mb-2 text-xs text-aspc-muted">
            Rolling QualityFlag rates (window={sensorHealth.window})
          </p>
          <div className="flex flex-wrap gap-3 text-sm">
            {Object.entries(healthRates).map(([k, v]) => (
              <span key={k} className="rounded-pill border border-aspc-border px-3 py-1 font-mono">
                {k}: {(v * 100).toFixed(1)}%
              </span>
            ))}
            {Object.keys(healthRates).length === 0 && (
              <span className="text-aspc-muted">No quality flags yet</span>
            )}
          </div>
        </Panel>
      )}

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
                {a.explanation && (
                  <p className="mt-1 text-[11px] text-aspc-muted">{a.explanation}</p>
                )}
                <div className="mt-1 flex items-center justify-between text-[11px] text-aspc-muted">
                  <span>
                    value={a.value}
                    {a.ts ? ` · ${formatTimestamp(a.ts)}` : ""}
                  </span>
                  <button
                    type="button"
                    className="text-aspc-accent hover:underline"
                    onClick={() => openExplain(a)}
                  >
                    Explain
                  </button>
                </div>
              </li>
            ))}
          </ul>
          {streamsQ.isLoading && <Spinner label="Loading streams…" />}
        </Panel>
      </div>

      {explain && (
        <Panel title="Explainable SPC Copilot" className="mt-6">
          <pre className="overflow-x-auto whitespace-pre-wrap rounded-2xl bg-aspc-elevated p-4 text-xs text-aspc-text">
            {JSON.stringify(explain, null, 2)}
          </pre>
          <button
            type="button"
            className="mt-3 text-sm text-aspc-muted hover:text-aspc-text"
            onClick={() => setExplain(null)}
          >
            Close
          </button>
        </Panel>
      )}
    </div>
  );
}
