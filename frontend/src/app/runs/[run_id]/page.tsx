"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Checklist } from "@/components/Checklist";
import { ControlChart } from "@/components/ControlChart";
import { GateList } from "@/components/GateList";
import { GoLivePanel } from "@/components/GoLivePanel";
import {
  Button,
  DataTable,
  ErrorBanner,
  JsonBlock,
  PageHeader,
  Panel,
  Spinner,
  Td,
  Term,
  TextInput,
} from "@/components/ui";
import { ApiError, api, getToken, reportUrl } from "@/lib/api";
import { formatLimits, formatTimestamp, shortId } from "@/lib/format";
import type { Gate, Phase1Checklist, SPCReport } from "@/lib/types";

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
  const [actionError, setActionError] = useState<string | null>(null);
  const [diffOther, setDiffOther] = useState("");
  const [diffResult, setDiffResult] = useState<Record<string, unknown> | null>(null);

  const { data, isLoading, error } = useQuery({
    queryKey: ["run", runId],
    queryFn: () => api.getRun(runId),
    enabled: !!runId,
  });

  const report = data ? asSpcReport(data.report) : null;
  const primary = report?.limits?.components ? Object.values(report.limits.components)[0] : null;
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

  function goToLive(key: string, limits: string) {
    router.push(`/live?stream=${encodeURIComponent(key)}&limits=${encodeURIComponent(limits)}`);
  }

  return (
    <div>
      <PageHeader
        title="Run Report"
        hideTitle
        subtitle={runId ? shortId(runId, 20) : "—"}
        actions={
          <Link href="/runs">
            <Button variant="secondary" tip="Go back to the full list of analyses.">All runs</Button>
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
                  <dt className="text-aspc-muted">
                    <Term k="limits-version" />
                  </dt>
                  <dd className="font-mono text-xs">{limitsVersion}</dd>
                </div>
              )}
            </dl>
            <div className="mt-4 flex flex-wrap gap-3">
              <a
                href={reportUrl(runId)}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm font-bold uppercase tracking-wide text-aspc-accent hover:underline"
              >
                Open HTML report
              </a>
              {report && (
                <button
                  type="button"
                  onClick={exportXlsx}
                  className="text-sm font-bold uppercase tracking-wide text-aspc-accent hover:underline"
                >
                  Export Excel
                </button>
              )}
            </div>
          </Panel>

          {limitsVersion && (
            <GoLivePanel
              title="One-click go-live"
              limitsVersion={limitsVersion}
              streamKey={streamKey}
              onStreamKeyChange={setStreamKey}
              onSuccess={goToLive}
            />
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
                  <Button type="button" onClick={runDiff} disabled={!diffOther} tip="Show what changed between two locked limit versions.">
                    Diff
                  </Button>
                </div>
              </div>
              {diffResult && (
                <div className="mt-4">
                  <JsonBlock value={diffResult} />
                </div>
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
              <DataTable headers={["Index", "Value"]}>
                {["cp", "cpk", "pp", "ppk", "sigma_level", "method"].map((k) =>
                  capResult[k] !== undefined && capResult[k] !== null ? (
                    <tr key={k}>
                      <Td className="font-mono uppercase text-aspc-muted">{k}</Td>
                      <Td className="font-mono text-aspc-accent">
                        {typeof capResult[k] === "number"
                          ? (capResult[k] as number).toFixed(4)
                          : String(capResult[k])}
                      </Td>
                    </tr>
                  ) : null,
                )}
              </DataTable>
            </Panel>
          )}

          {msaResult?.result && (
            <Panel title={`MSA · ${msaResult.study_type ?? "study"}`}>
              <DataTable headers={["Metric", "Value"]}>
                {Object.entries(msaResult.result)
                  .filter(
                    ([, v]) => typeof v === "number" || typeof v === "string" || typeof v === "boolean",
                  )
                  .map(([k, v]) => (
                    <tr key={k}>
                      <Td className="font-mono text-aspc-muted">{k}</Td>
                      <Td className="font-mono">{typeof v === "number" ? v.toFixed(4) : String(v)}</Td>
                    </tr>
                  ))}
              </DataTable>
            </Panel>
          )}

          {!report && !capResult && !msaResult?.result && (
            <Panel title="Report JSON">
              <JsonBlock value={data.report} maxHeight="32rem" />
            </Panel>
          )}
        </div>
      )}
    </div>
  );
}
