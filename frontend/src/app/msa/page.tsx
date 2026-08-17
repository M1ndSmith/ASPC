"use client";

import { FormEvent, useState } from "react";
import {
  Button,
  DataTable,
  ErrorBanner,
  FileField,
  JsonBlock,
  PageHeader,
  Panel,
  SelectInput,
  Spinner,
  Td,
  Term,
  TextInput,
} from "@/components/ui";
import { ApiError, api } from "@/lib/api";
import type { AnalyzeResponse } from "@/lib/types";

export default function MsaPage() {
  const [file, setFile] = useState<File | null>(null);
  const [studyType, setStudyType] = useState("gage_rr");
  const [method, setMethod] = useState("anova");
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
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("study_type", studyType);
      fd.append("method", method);
      const res = await api.analyzeMsa(fd);
      setResult(res);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : (err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const msa =
    result && typeof result.report === "object" && result.report
      ? (result.report as { study_type?: string; result?: Record<string, unknown> })
      : null;

  return (
    <div>
      <PageHeader
        title="MSA"
        hideTitle
        subtitle={
          <>
            <Term k="gage-rr" />, bias, linearity, stability — plus <Term k="ndc" /> and continuous MSA drift
          </>
        }
      />

      {error && <ErrorBanner message={error} />}

      <Panel title="Study upload" className="mb-6">
        <form onSubmit={onSubmit} className="grid gap-4 md:grid-cols-2">
          <div className="md:col-span-2">
            <FileField id="msa_file" label="MSA data file" onChange={setFile} required />
          </div>
          <SelectInput
            id="study_type"
            label="Study type"
            value={studyType}
            onChange={(e) => setStudyType(e.target.value)}
          >
            <option value="gage_rr">Gage R&R</option>
            <option value="bias">Bias</option>
            <option value="linearity">Linearity</option>
            <option value="stability">Stability</option>
          </SelectInput>
          <SelectInput
            id="method"
            label="Gage R&R method"
            value={method}
            onChange={(e) => setMethod(e.target.value)}
            disabled={studyType !== "gage_rr"}
          >
            <option value="anova">ANOVA</option>
            <option value="range">Range</option>
          </SelectInput>
          <div className="flex items-end gap-3 md:col-span-2">
            <Button type="submit" disabled={busy || !file} tip="Check whether the measuring tool is trustworthy.">
              {busy ? "Running…" : "Run MSA"}
            </Button>
            {busy && <Spinner />}
          </div>
        </form>
      </Panel>

      {result && (
        <Panel title="Batch MSA result" className="mb-6">
          <div className="mb-3 text-sm">
            <span className="text-aspc-muted">Run </span>
            <span className="font-mono text-aspc-accent">{result.run_id}</span>
            {msa?.study_type && (
              <>
                <span className="mx-2 text-aspc-muted">·</span>
                <span className="font-mono">{msa.study_type}</span>
              </>
            )}
          </div>
          {msa?.result ? (
            <DataTable headers={["Metric", "Value"]}>
              {Object.entries(msa.result)
                .filter(([, v]) => typeof v === "number" || typeof v === "string" || typeof v === "boolean")
                .map(([k, v]) => (
                  <tr key={k}>
                    <Td className="font-mono text-aspc-muted">{k}</Td>
                    <Td className="font-mono">{typeof v === "number" ? v.toFixed(4) : String(v)}</Td>
                  </tr>
                ))}
            </DataTable>
          ) : (
            <JsonBlock value={result.report} />
          )}
        </Panel>
      )}

      <Panel title="Continuous MSA drift">
        <ContinuousMsaPanel />
      </Panel>
    </div>
  );
}

function ContinuousMsaPanel() {
  const [measured, setMeasured] = useState("10.1,10.2,10.0,10.4,10.5");
  const [reference, setReference] = useState("10,10,10,10,10");
  const [tolerance, setTolerance] = useState("1");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [summary, setSummary] = useState<Record<string, unknown> | null>(null);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      const m = measured.split(",").map((s) => Number(s.trim()));
      const r = reference.split(",").map((s) => Number(s.trim()));
      const body = await api.analyzeMsaContinuous({
        measured: m,
        reference: r,
        tolerance: Number(tolerance) || 1,
      });
      setSummary(body.summary);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-4">
      <p className="text-sm text-aspc-muted">
        Evaluate reference-standard injections with <Term k="ewma" /> bias (α=0.2), rolling R, and
        calibration alerts.
      </p>
      {error && <ErrorBanner message={error} />}
      <div className="grid gap-3 md:grid-cols-3">
        <TextInput
          id="msa_measured"
          label="Measured"
          value={measured}
          onChange={(e) => setMeasured(e.target.value)}
        />
        <TextInput
          id="msa_reference"
          label="Reference"
          value={reference}
          onChange={(e) => setReference(e.target.value)}
        />
        <TextInput
          id="msa_tolerance"
          label="Tolerance"
          value={tolerance}
          onChange={(e) => setTolerance(e.target.value)}
        />
      </div>
      <Button onClick={run} disabled={busy} tip="Compare measured values to a known reference over time.">
        {busy ? "Evaluating…" : "Run continuous MSA"}
      </Button>
      {summary && <JsonBlock value={summary} />}
    </div>
  );
}
