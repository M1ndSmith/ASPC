"use client";

import { FormEvent, useState } from "react";
import {
  ErrorBanner,
  FileField,
  PageHeader,
  Panel,
  PrimaryButton,
  Spinner,
  TextInput,
} from "@/components/ui";
import { ApiError, api } from "@/lib/api";
import type { AnalyzeResponse } from "@/lib/types";

export default function CapabilityPage() {
  const [file, setFile] = useState<File | null>(null);
  const [usl, setUsl] = useState("10.5");
  const [lsl, setLsl] = useState("9.5");
  const [target, setTarget] = useState("");
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
      fd.append("usl", usl);
      fd.append("lsl", lsl);
      if (target) fd.append("target", target);
      const res = await api.analyzeCapability(fd);
      setResult(res);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : (err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const cap =
    result && typeof result.report === "object" && result.report && "result" in result.report
      ? (result.report as { result: Record<string, unknown> }).result
      : null;

  return (
    <div>
      <PageHeader
        title="Capability Study"
        hideTitle
        subtitle="Cp / Cpk / Pp / Ppk against specification limits"
      />

      {error && <ErrorBanner message={error} />}

      <Panel title="Study setup" className="mb-6">
        <form onSubmit={onSubmit} className="grid gap-4 md:grid-cols-2">
          <div className="md:col-span-2">
            <FileField id="cap_file" label="Measurement file" onChange={setFile} required />
          </div>
          <TextInput
            id="usl"
            label="USL"
            type="number"
            step="any"
            value={usl}
            onChange={(e) => setUsl(e.target.value)}
            required
          />
          <TextInput
            id="lsl"
            label="LSL"
            type="number"
            step="any"
            value={lsl}
            onChange={(e) => setLsl(e.target.value)}
            required
          />
          <TextInput
            id="target"
            label="Target (optional)"
            type="number"
            step="any"
            value={target}
            onChange={(e) => setTarget(e.target.value)}
          />
          <div className="flex items-end gap-3">
            <PrimaryButton type="submit" disabled={busy || !file}>
              {busy ? "Computing…" : "Run capability"}
            </PrimaryButton>
            {busy && <Spinner />}
          </div>
        </form>
      </Panel>

      {result && (
        <Panel title="Results">
          <div className="mb-4 text-sm">
            <span className="text-aspc-muted">Run </span>
            <span className="font-mono text-aspc-accent">{result.run_id}</span>
          </div>
          {cap ? (
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
                    cap[k] !== undefined && cap[k] !== null ? (
                      <tr key={k} className="border-b border-aspc-border/50">
                        <td className="py-2.5 pr-4 font-mono uppercase text-aspc-muted">{k}</td>
                        <td className="py-2.5 font-mono text-lg text-aspc-accent">
                          {typeof cap[k] === "number" ? (cap[k] as number).toFixed(4) : String(cap[k])}
                        </td>
                      </tr>
                    ) : null,
                  )}
                </tbody>
              </table>
            </div>
          ) : (
            <pre className="max-h-96 overflow-auto rounded-2xl bg-aspc-elevated p-3 text-xs text-aspc-muted">
              {JSON.stringify(result.report, null, 2)}
            </pre>
          )}
        </Panel>
      )}
    </div>
  );
}
