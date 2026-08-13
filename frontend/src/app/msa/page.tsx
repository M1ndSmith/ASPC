"use client";

import { FormEvent, useState } from "react";
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
        subtitle="Gage R&R, bias, linearity, stability — plus continuous MSA drift"
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
            <PrimaryButton type="submit" disabled={busy || !file}>
              {busy ? "Running…" : "Run MSA"}
            </PrimaryButton>
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
            <div className="overflow-x-auto">
              <table className="w-full min-w-[24rem] text-left text-sm">
                <thead>
                  <tr className="border-b border-aspc-border text-[11px] uppercase tracking-wider text-aspc-muted">
                    <th className="pb-2 pr-4 font-medium">Metric</th>
                    <th className="pb-2 font-medium">Value</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(msa.result)
                    .filter(([, v]) => typeof v === "number" || typeof v === "string" || typeof v === "boolean")
                    .map(([k, v]) => (
                      <tr key={k} className="border-b border-aspc-border/50">
                        <td className="py-2.5 pr-4 font-mono text-aspc-muted">{k}</td>
                        <td className="py-2.5 font-mono text-aspc-text">
                          {typeof v === "number" ? v.toFixed(4) : String(v)}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          ) : (
            <pre className="max-h-80 overflow-auto rounded-2xl bg-aspc-elevated p-3 text-xs text-aspc-muted">
              {JSON.stringify(result.report, null, 2)}
            </pre>
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
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/analyze/msa-continuous`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(typeof window !== "undefined" && localStorage.getItem("aspc_token")
              ? { Authorization: `Bearer ${localStorage.getItem("aspc_token")}` }
              : {}),
          },
          body: JSON.stringify({
            measured: m,
            reference: r,
            tolerance: Number(tolerance) || 1,
          }),
        },
      );
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || res.statusText);
      }
      const body = (await res.json()) as { summary: Record<string, unknown> };
      setSummary(body.summary);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-4">
      <p className="text-sm text-aspc-muted">
        Evaluate reference-standard injections with{" "}
        <code className="font-mono text-xs">ContinuousMSA</code> (EWMA bias α=0.2, rolling R,
        calibration alerts).
      </p>
      {error && <ErrorBanner message={error} />}
      <div className="grid gap-3 md:grid-cols-3">
        <label className="text-xs text-aspc-muted">
          Measured
          <input
            className="mt-1 w-full rounded-2xl border border-aspc-border bg-aspc-elevated px-3 py-2 font-mono text-sm"
            value={measured}
            onChange={(e) => setMeasured(e.target.value)}
          />
        </label>
        <label className="text-xs text-aspc-muted">
          Reference
          <input
            className="mt-1 w-full rounded-2xl border border-aspc-border bg-aspc-elevated px-3 py-2 font-mono text-sm"
            value={reference}
            onChange={(e) => setReference(e.target.value)}
          />
        </label>
        <label className="text-xs text-aspc-muted">
          Tolerance
          <input
            className="mt-1 w-full rounded-2xl border border-aspc-border bg-aspc-elevated px-3 py-2 font-mono text-sm"
            value={tolerance}
            onChange={(e) => setTolerance(e.target.value)}
          />
        </label>
      </div>
      <PrimaryButton onClick={run} disabled={busy}>
        {busy ? "Evaluating…" : "Run continuous MSA"}
      </PrimaryButton>
      {summary && (
        <pre className="max-h-64 overflow-auto rounded-2xl bg-aspc-elevated p-3 text-xs text-aspc-muted">
          {JSON.stringify(summary, null, 2)}
        </pre>
      )}
    </div>
  );
}
