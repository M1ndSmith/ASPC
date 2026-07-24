"use client";

import { FormEvent, useMemo, useState } from "react";
import { Checklist } from "@/components/Checklist";
import { ControlChart } from "@/components/ControlChart";
import { GateList } from "@/components/GateList";
import {
  ErrorBanner,
  FileField,
  PageHeader,
  Panel,
  PrimaryButton,
  SelectInput,
  Spinner,
} from "@/components/ui";
import { ApiError, api } from "@/lib/api";
import { formatLimits } from "@/lib/format";
import type { AnalyzeResponse, Gate, Phase1Checklist, SPCReport } from "@/lib/types";

function asReport(raw: AnalyzeResponse["report"]): SPCReport | null {
  if (!raw || typeof raw !== "object") return null;
  if ("plotted_values" in raw && "limits" in raw) return raw as SPCReport;
  return null;
}

export default function AnalyzePage() {
  const [file, setFile] = useState<File | null>(null);
  const [chartType, setChartType] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (!file) {
      setError("Choose a data file");
      return;
    }
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      if (chartType) fd.append("chart_type", chartType);
      const res = await api.analyzeControlChart(fd);
      setResult(res);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : (err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const report = useMemo(() => (result ? asReport(result.report) : null), [result]);
  const gates = (report?.gates ?? (result?.report as { gates?: Gate[] })?.gates) as Gate[] | undefined;
  const checklist = (report?.checklist ??
    (result?.report as { checklist?: Phase1Checklist })?.checklist) as Phase1Checklist | undefined;

  const primary = report?.limits?.components
    ? Object.values(report.limits.components)[0]
    : null;
  const oocIndices = report?.signals?.map((s) => s.index) ?? [];

  return (
    <div>
      <PageHeader
        title="Control Chart Analysis"
        subtitle="Upload CSV → POST /analyze/control-chart → gates, checklist, Plotly chart"
      />

      {error && <ErrorBanner message={error} />}

      <Panel title="Upload" className="mb-6">
        <form onSubmit={onSubmit} className="grid gap-4 md:grid-cols-2">
          <div className="md:col-span-2">
            <FileField id="file" label="Data file (CSV)" onChange={setFile} required />
          </div>

          <SelectInput
            id="chart_type"
            label="Chart type (optional)"
            value={chartType}
            onChange={(e) => setChartType(e.target.value)}
          >
            <option value="">Auto-detect</option>
            <option value="I-MR">I-MR</option>
            <option value="Xbar-R">Xbar-R</option>
            <option value="Xbar-S">Xbar-S</option>
            <option value="EWMA">EWMA</option>
            <option value="CUSUM">CUSUM</option>
            <option value="P">P</option>
            <option value="NP">NP</option>
            <option value="C">C</option>
            <option value="U">U</option>
          </SelectInput>

          <div className="flex items-end gap-3">
            <PrimaryButton type="submit" disabled={busy || !file}>
              {busy ? "Running…" : "Run analysis"}
            </PrimaryButton>
            {busy && <Spinner />}
          </div>
        </form>
      </Panel>

      {result && (
        <div className="space-y-6">
          <Panel title="Result">
            <div className="flex flex-wrap gap-4 text-sm">
              <div>
                <span className="text-aspc-muted">Run ID </span>
                <span className="font-mono text-aspc-cyan">{result.run_id}</span>
              </div>
              <div>
                <span className="text-aspc-muted">Type </span>
                <span className="font-mono">{result.analysis_type}</span>
              </div>
              {primary && (
                <div>
                  <span className="text-aspc-muted">Limits </span>
                  <span className="font-mono text-xs">{formatLimits(primary)}</span>
                </div>
              )}
            </div>
          </Panel>

          {report && primary && (
            <ControlChart
              title={`${report.chart_type} · Phase ${report.phase || "I"}`}
              values={report.plotted_values}
              ucl={primary.ucl}
              cl={primary.center}
              lcl={primary.lcl}
              oocIndices={oocIndices}
            />
          )}

          <div className="grid gap-6 lg:grid-cols-2">
            <Panel title="Gates">
              <GateList gates={gates ?? []} />
            </Panel>
            <Panel title="Phase I Checklist">
              <Checklist checklist={checklist} />
            </Panel>
          </div>

          {!report && (
            <Panel title="Report JSON">
              <pre className="max-h-96 overflow-auto rounded-lg bg-aspc-bg p-3 text-xs text-aspc-muted">
                {JSON.stringify(result.report, null, 2)}
              </pre>
            </Panel>
          )}
        </div>
      )}
    </div>
  );
}
