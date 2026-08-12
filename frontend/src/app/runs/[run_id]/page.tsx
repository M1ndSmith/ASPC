"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { Checklist } from "@/components/Checklist";
import { ControlChart } from "@/components/ControlChart";
import { GateList } from "@/components/GateList";
import { ErrorBanner, PageHeader, Panel, Spinner } from "@/components/ui";
import { api, reportUrl } from "@/lib/api";
import { formatLimits, formatTimestamp, shortId } from "@/lib/format";
import type { Gate, Phase1Checklist, SPCReport } from "@/lib/types";

function asSpcReport(raw: unknown): SPCReport | null {
  if (!raw || typeof raw !== "object") return null;
  if ("plotted_values" in raw && "limits" in raw) return raw as SPCReport;
  return null;
}

export default function RunDetailPage() {
  const params = useParams();
  const runId = String(params.run_id ?? "");

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

  const capResult =
    data?.report && typeof data.report === "object" && "result" in data.report
      ? (data.report as { result: Record<string, unknown> }).result
      : null;

  const msaResult =
    data?.report && typeof data.report === "object" && "result" in data.report
      ? (data.report as { study_type?: string; result: Record<string, unknown> })
      : null;

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
              {data.limits_version && (
                <div>
                  <dt className="text-aspc-muted">Limits version</dt>
                  <dd className="font-mono text-xs">{data.limits_version}</dd>
                </div>
              )}
            </dl>
            <div className="mt-4">
              <a
                href={reportUrl(runId)}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-aspc-accent hover:underline"
              >
                Open HTML report ↗
              </a>
            </div>
          </Panel>

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
              <div className="text-sm text-aspc-muted">
                {formatLimits(primary)}
              </div>
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
                      .filter(([, v]) => typeof v === "number" || typeof v === "string" || typeof v === "boolean")
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
