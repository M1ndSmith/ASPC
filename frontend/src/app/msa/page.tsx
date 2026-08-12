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
        subtitle="Gage R&R, bias, linearity, stability — plus continuous MSA drift placeholder"
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
        <div className="rounded-2xl border border-dashed border-aspc-border bg-aspc-elevated/40 px-4 py-10 text-center">
          <p className="text-xs font-medium uppercase tracking-widest text-aspc-accent">Coming online</p>
          <p className="mx-auto mt-2 max-w-md text-sm text-aspc-muted">
            Live bias EWMA (α=0.2), rolling R, and calibration alerts from ContinuousMSA will render
            here once the stream engine publishes drift metrics on the selected stream.
          </p>
          <div className="mx-auto mt-6 h-24 max-w-lg rounded-2xl bg-gradient-to-r from-aspc-border/20 via-aspc-accent/15 to-aspc-border/20" />
        </div>
      </Panel>
    </div>
  );
}
