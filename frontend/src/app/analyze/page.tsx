"use client";

import { FormEvent, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
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
  TextInput,
} from "@/components/ui";
import { ApiError, api } from "@/lib/api";
import { formatLimits } from "@/lib/format";
import type { AnalyzeResponse, Gate, Phase1Checklist, SPCReport } from "@/lib/types";

const API_KEY_STORAGE = "aspc_api_key";

function asReport(raw: AnalyzeResponse["report"]): SPCReport | null {
  if (!raw || typeof raw !== "object") return null;
  if ("plotted_values" in raw && "limits" in raw) return raw as SPCReport;
  return null;
}

export default function AnalyzePage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [msaFile, setMsaFile] = useState<File | null>(null);
  const [chartType, setChartType] = useState("");
  const [ruleset, setRuleset] = useState("nelson");
  const [validMin, setValidMin] = useState("");
  const [validMax, setValidMax] = useState("");
  const [msaTolerance, setMsaTolerance] = useState("");
  const [busy, setBusy] = useState(false);
  const [goLiveBusy, setGoLiveBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [streamKey, setStreamKey] = useState("line-1");
  const [apiKey, setApiKey] = useState(() =>
    typeof window !== "undefined" ? localStorage.getItem(API_KEY_STORAGE) || "" : "",
  );

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
      if (ruleset) fd.append("ruleset", ruleset);
      if (validMin !== "" && validMax !== "") {
        fd.append("valid_range_min", validMin);
        fd.append("valid_range_max", validMax);
      }
      if (msaFile) {
        fd.append("msa_file", msaFile);
        if (msaTolerance !== "") fd.append("msa_tolerance", msaTolerance);
      }
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
    result?.checklist ??
    (result?.report as { checklist?: Phase1Checklist })?.checklist) as Phase1Checklist | undefined;

  const primary = report?.limits?.components
    ? Object.values(report.limits.components)[0]
    : null;
  const oocIndices = report?.signals?.map((s) => s.index) ?? [];
  const limitsVersion = report?.limits?.version;
  const checklistOk = checklist?.passed !== false;

  async function oneClickGoLive() {
    if (!limitsVersion) {
      setError("No frozen limits version on this result");
      return;
    }
    if (!apiKey) {
      setError("Enter X-API-Key for stream mutations");
      return;
    }
    setGoLiveBusy(true);
    setError(null);
    try {
      localStorage.setItem(API_KEY_STORAGE, apiKey);
      await api.registerStream({ stream_key: streamKey }, apiKey);
      await api.goLive(streamKey, { limits_version: limitsVersion }, apiKey);
      router.push(
        `/live?stream=${encodeURIComponent(streamKey)}&limits=${encodeURIComponent(limitsVersion)}`,
      );
    } catch (err) {
      setError(err instanceof ApiError ? err.message : (err as Error).message);
    } finally {
      setGoLiveBusy(false);
    }
  }

  return (
    <div>
      <PageHeader
        title="Control Chart Analysis"
        hideTitle
        subtitle="Upload CSV → POST /analyze/control-chart → gates, checklist, Plotly chart"
        actions={
          <Link
            href="/onboarding"
            className="rounded-pill border border-aspc-accent/40 px-3 py-1.5 text-xs text-aspc-accent hover:bg-aspc-accent-soft"
          >
            Onboarding wizard
          </Link>
        }
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
            <option value="">Auto</option>
            <option value="I_MR">I-MR</option>
            <option value="XBAR_R">Xbar-R</option>
            <option value="XBAR_S">Xbar-S</option>
            <option value="P">p</option>
            <option value="NP">np</option>
            <option value="C">c</option>
            <option value="U">u</option>
            <option value="EWMA">EWMA</option>
            <option value="CUSUM">CUSUM</option>
          </SelectInput>

          <SelectInput
            id="ruleset"
            label="Ruleset"
            value={ruleset}
            onChange={(e) => setRuleset(e.target.value)}
          >
            <option value="nelson">Nelson</option>
            <option value="western_electric">Western Electric</option>
            <option value="wheeler">Wheeler</option>
          </SelectInput>

          <TextInput
            id="valid_min"
            label="Valid range min (optional)"
            type="number"
            step="any"
            value={validMin}
            onChange={(e) => setValidMin(e.target.value)}
          />
          <TextInput
            id="valid_max"
            label="Valid range max (optional)"
            type="number"
            step="any"
            value={validMax}
            onChange={(e) => setValidMax(e.target.value)}
          />

          <div className="md:col-span-2">
            <FileField id="msa_file" label="MSA file (optional)" onChange={setMsaFile} />
          </div>
          <TextInput
            id="msa_tolerance"
            label="MSA tolerance (optional)"
            type="number"
            step="any"
            value={msaTolerance}
            onChange={(e) => setMsaTolerance(e.target.value)}
          />

          <div className="flex items-end gap-3 md:col-span-2">
            <PrimaryButton type="submit" disabled={busy || !file}>
              {busy ? "Running…" : "Run analysis"}
            </PrimaryButton>
            {busy && <Spinner />}
          </div>
        </form>
        <p className="mt-3 text-xs text-aspc-muted">
          Freeze can succeed with MSA warn; go-live still requires a passing checklist
          (including Gage R&amp;R / NDC when study data is supplied).
        </p>
      </Panel>

      {result && (
        <div className="space-y-6">
          <Panel title="Result">
            <div className="flex flex-wrap gap-4 text-sm">
              <div>
                <span className="text-aspc-muted">Run ID </span>
                <span className="font-mono text-aspc-accent">{result.run_id}</span>
              </div>
              <div>
                <span className="text-aspc-muted">Type </span>
                <span className="font-mono">{result.analysis_type}</span>
              </div>
              {limitsVersion && (
                <div>
                  <span className="text-aspc-muted">Limits version </span>
                  <span className="font-mono text-aspc-accent">{limitsVersion}</span>
                </div>
              )}
              {primary && (
                <div>
                  <span className="text-aspc-muted">Limits </span>
                  <span className="font-mono">{formatLimits(primary)}</span>
                </div>
              )}
            </div>
          </Panel>

          {gates && <GateList gates={gates} />}
          {checklist && <Checklist checklist={checklist} />}

          {limitsVersion && (
            <Panel title="One-click go-live">
              <div className="grid gap-4 md:grid-cols-3">
                <TextInput
                  id="golive_stream"
                  label="Stream key"
                  value={streamKey}
                  onChange={(e) => setStreamKey(e.target.value)}
                />
                <TextInput
                  id="golive_key"
                  label="X-API-Key"
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                />
                <div className="flex items-end gap-2">
                  <PrimaryButton
                    type="button"
                    onClick={oneClickGoLive}
                    disabled={goLiveBusy || !checklistOk}
                  >
                    {goLiveBusy ? "Activating…" : "Go live → Live"}
                  </PrimaryButton>
                  <Link
                    href={`/live?stream=${encodeURIComponent(streamKey)}&limits=${encodeURIComponent(limitsVersion)}`}
                    className="rounded-pill border border-aspc-border px-3 py-2.5 text-sm text-aspc-muted hover:text-aspc-text"
                  >
                    Open Live only
                  </Link>
                </div>
              </div>
              {!checklistOk && (
                <p className="mt-2 text-xs text-aspc-stop">
                  Checklist did not pass — go-live will be rejected by the API until resolved.
                </p>
              )}
            </Panel>
          )}

          {report?.plotted_values && primary && (
            <Panel title="Control chart">
              <ControlChart
                values={report.plotted_values}
                ucl={typeof primary.ucl === "number" ? primary.ucl : Number(primary.ucl?.[0] ?? 0)}
                cl={primary.center}
                lcl={typeof primary.lcl === "number" ? primary.lcl : Number(primary.lcl?.[0] ?? 0)}
                oocIndices={oocIndices}
              />
            </Panel>
          )}
        </div>
      )}
    </div>
  );
}
