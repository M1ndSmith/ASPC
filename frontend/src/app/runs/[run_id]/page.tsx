"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Checklist } from "@/components/Checklist";
import { ControlChart } from "@/components/ControlChart";
import { GateList } from "@/components/GateList";
import { ErrorBanner, PageHeader, Panel, PrimaryButton, Spinner, TextInput } from "@/components/ui";
import { ApiError, api, getToken, reportUrl } from "@/lib/api";
import { formatLimits, formatTimestamp, shortId } from "@/lib/format";
import type { Gate, Phase1Checklist, SPCReport } from "@/lib/types";

const API_KEY_STORAGE = "aspc_api_key";

function asSpcReport(raw: unknown): SPCReport | null {
  if (!raw || typeof raw !== "object") return null;
  if ("plotted_values" in raw && "limits" in raw) return raw as SPCReport;
  return null;
}

export default function RunDetailPage() {
  const params = useParams();
  const router = useRouter();
  const runId = String(params.run_id ?? "");
  const [streamKey, setStreamKey] = useState("line-1");
  const [apiKey, setApiKey] = useState(() =>
    typeof window !== "undefined" ? localStorage.getItem(API_KEY_STORAGE) || "" : "",
  );
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [diffOther, setDiffOther] = useState("");
  const [diffResult, setDiffResult] = useState<Record<string, unknown> | null>(null);

  const { data, isLoading, error } = useQuery({
    queryKey: ["run", runId],
    queryFn: () => api.getRun(runId),
    enabled: !!runId,
  });

  const report = data ? asSpcReport(data.report) : null;
  const primary = report?.limits?.components
    ? Object.values(report.limits.components)[0]
    : null;
  const oocIndices = report?.signals?.map((s) => s.index) ?? [];
  const gates = (report?.gates ?? (data?.report as { gates?: Gate[] })?.gates) as Gate[] | undefined;
  const checklist = (report?.checklist ??
    (data?.report as { checklist?: Phase1Checklist })?.checklist) as Phase1Checklist | undefined;
  const limitsVersion = data?.limits_version || report?.limits?.version;

  const capResult =
    data?.report && typeof data.report === "object" && "result" in data.report
      ? (data.report as { result: Record<string, unknown> }).result
      : null;

  const msaResult =
    data?.report && typeof data.report === "object" && "result" in data.report
      ? (data.report as { study_type?: string; result: Record<string, unknown> })
      : null;

  async function goLive() {
    if (!limitsVersion || !apiKey) {
      setActionError("Limits version and API key required");
      return;
    }
    setBusy(true);
    setActionError(null);
    try {
      localStorage.setItem(API_KEY_STORAGE, apiKey);
      await api.registerStream({ stream_key: streamKey }, apiKey);
      await api.goLive(streamKey, { limits_version: limitsVersion }, apiKey);
      router.push(
        `/live?stream=${encodeURIComponent(streamKey)}&limits=${encodeURIComponent(limitsVersion)}`,
      );
    } catch (err) {
      setActionError(err instanceof ApiError ? err.message : (err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function runDiff() {
    if (!limitsVersion || !diffOther) return;
    setActionError(null);
    try {
      setDiffResult(await api.limitsDiff(limitsVersion, diffOther));
    } catch (err) {
      setActionError(err instanceof ApiError ? err.message : (err as Error).message);
    }
  }

  function exportXlsx() {
    const token = getToken();
    const url = api.exportRunXlsxUrl(runId);
    void (async () => {
      try {
        const res = await fetch(url, {
          headers: token ? { Authorization: `Bearer ${token}` } : {},
        });
        if (!res.ok) throw new Error(await res.text());
        const blob = await res.blob();
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `${runId}.xlsx`;
        a.click();
        URL.revokeObjectURL(a.href);
      } catch (err) {
        setActionError(err instanceof Error ? err.message : String(err));
      }
    })();
  }

  return (
    <div>
      <PageHeader
        title="Run Report"
        hideTitle
        subtitle={runId ? shortId(runId, 20) : "—"}
        actions={
          <Link
            href="/runs"
            className="rounded-pill border border-aspc-border px-3 py-1.5 text-xs text-aspc-muted hover:text-aspc-text"
          >
            ← All runs
          </Link>
        }
      />

      {error && <ErrorBanner message={(error as Error).message} />}
      {actionError && <ErrorBanner message={actionError} />}
      {isLoading && <Spinner />}

      {data && (
        <div className="space-y-6">
          <Panel title="Summary">
            <dl className="grid gap-3 text-sm sm:grid-cols-2">
              <div>
                <dt className="text-aspc-muted">Run ID</dt>
                <dd className="font-mono text-aspc-accent">{data.run_id}</dd>
              </div>
              <div>
                <dt className="text-aspc-muted">Type</dt>
                <dd className="font-mono">{data.analysis_type}</dd>
              </div>
              <div>
                <dt className="text-aspc-muted">Created</dt>
                <dd>{formatTimestamp(data.created_at)}</dd>
              </div>
              <div>
                <dt className="text-aspc-muted">Source</dt>
                <dd className="truncate">{data.source_file || "—"}</dd>
              </div>
              {limitsVersion && (
                <div>
                  <dt className="text-aspc-muted">Limits version</dt>
                  <dd className="font-mono text-xs">{limitsVersion}</dd>
                </div>
              )}
            </dl>
            <div className="mt-4 flex flex-wrap gap-3">
              <a
                href={reportUrl(runId)}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-aspc-accent hover:underline"
              >
                Open HTML report ↗
              </a>
              {report && (
                <button
                  type="button"
                  onClick={exportXlsx}
                  className="text-sm text-aspc-accent hover:underline"
                >
                  Export Excel
                </button>
              )}
            </div>
          </Panel>

          {limitsVersion && (
            <Panel title="One-click go-live">
              <div className="grid gap-4 md:grid-cols-3">
                <TextInput
                  id="rk"
                  label="Stream key"
                  value={streamKey}
                  onChange={(e) => setStreamKey(e.target.value)}
                />
                <TextInput
                  id="ak"
                  label="X-API-Key"
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                />
                <div className="flex items-end">
                  <PrimaryButton type="button" onClick={goLive} disabled={busy}>
                    {busy ? "Activating…" : "Go live → Live"}
                  </PrimaryButton>
                </div>
              </div>
            </Panel>
          )}

          {limitsVersion && (
            <Panel title="Limit diff / freeze time travel">
              <div className="grid gap-4 md:grid-cols-2">
                <TextInput
                  id="diff_b"
                  label="Compare to limits version"
                  value={diffOther}
                  onChange={(e) => setDiffOther(e.target.value)}
                />
                <div className="flex items-end">
                  <PrimaryButton type="button" onClick={runDiff} disabled={!diffOther}>
                    Diff
                  </PrimaryButton>
                </div>
              </div>
              {diffResult && (
                <pre className="mt-4 overflow-x-auto rounded-2xl bg-aspc-elevated p-3 text-xs">
                  {JSON.stringify(diffResult, null, 2)}
                </pre>
              )}
            </Panel>
          )}

          {report && primary && (
            <>
              <ControlChart
                title={`${report.chart_type} · Phase ${report.phase || "I"}`}
                values={report.plotted_values}
                ucl={primary.ucl}
                cl={primary.center}
                lcl={primary.lcl}
                oocIndices={oocIndices}
              />
              <div className="text-sm text-aspc-muted">{formatLimits(primary)}</div>
              <div className="grid gap-6 lg:grid-cols-2">
                <Panel title="Gates">
                  <GateList gates={gates ?? []} />
                </Panel>
                <Panel title="Phase I Checklist">
                  <Checklist checklist={checklist} />
                </Panel>
              </div>
            </>
          )}

          {capResult && (
            <Panel title="Capability indices">
              <div className="overflow-x-auto">
                <table className="w-full min-w-[20rem] text-left text-sm">
                  <thead>
                    <tr className="border-b border-aspc-border text-[11px] uppercase tracking-wider text-aspc-muted">
                      <th className="pb-2 pr-4 font-medium">Index</th>
                      <th className="pb-2 font-medium">Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {["cp", "cpk", "pp", "ppk", "sigma_level", "method"].map((k) =>
                      capResult[k] !== undefined && capResult[k] !== null ? (
                        <tr key={k} className="border-b border-aspc-border/50">
                          <td className="py-2 pr-4 font-mono uppercase text-aspc-muted">{k}</td>
                          <td className="py-2 font-mono text-aspc-accent">
                            {typeof capResult[k] === "number"
                              ? (capResult[k] as number).toFixed(4)
                              : String(capResult[k])}
                          </td>
                        </tr>
                      ) : null,
                    )}
                  </tbody>
                </table>
              </div>
            </Panel>
          )}

          {msaResult?.result && (
            <Panel title={`MSA · ${msaResult.study_type ?? "study"}`}>
              <div className="overflow-x-auto">
                <table className="w-full min-w-[24rem] text-left text-sm">
                  <thead>
                    <tr className="border-b border-aspc-border text-[11px] uppercase tracking-wider text-aspc-muted">
                      <th className="pb-2 pr-4 font-medium">Metric</th>
                      <th className="pb-2 font-medium">Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(msaResult.result)
                      .filter(
                        ([, v]) =>
                          typeof v === "number" || typeof v === "string" || typeof v === "boolean",
                      )
                      .map(([k, v]) => (
                        <tr key={k} className="border-b border-aspc-border/50">
                          <td className="py-2 pr-4 font-mono text-aspc-muted">{k}</td>
                          <td className="py-2 font-mono">
                            {typeof v === "number" ? v.toFixed(4) : String(v)}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </Panel>
          )}

          {!report && !capResult && !msaResult?.result && (
            <Panel title="Report JSON">
              <pre className="max-h-[32rem] overflow-auto rounded-2xl bg-aspc-elevated p-3 text-xs text-aspc-muted">
                {JSON.stringify(data.report, null, 2)}
              </pre>
            </Panel>
          )}
        </div>
      )}
    </div>
  );
}
