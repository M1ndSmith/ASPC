"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Button, ErrorBanner, JsonBlock, PageHeader, Panel, Spinner } from "@/components/ui";
import { ApiError, api } from "@/lib/api";

function ReachabilityHint({ message }: { message: string }) {
  if (!message.includes("Cannot reach API")) return null;
  return (
    <div className="mb-4 rounded-lg bg-aspc-elevated px-4 py-3 text-sm text-aspc-muted shadow-recessed">
      <p className="mb-2 font-medium text-aspc-text">API unreachable from the browser</p>
      <ol className="list-decimal space-y-1 pl-5">
        <li>
          Confirm API: <code className="font-mono text-aspc-accent">curl -sS http://127.0.0.1:8000/health</code>
        </li>
        <li>
          Confirm UI proxy:{" "}
          <code className="font-mono text-aspc-accent">curl -sS http://127.0.0.1:3000/backend/health</code>
        </li>
        <li>
          Compose should use <code className="font-mono text-aspc-accent">NEXT_PUBLIC_API_URL=/backend</code>{" "}
          (rebuild frontend after changing build args).
        </li>
        <li>Log in (Lab requires an authenticated analyst/admin session).</li>
      </ol>
    </div>
  );
}

export default function LabPage() {
  const casesQ = useQuery({ queryKey: ["lab-cases"], queryFn: api.labCases });
  const [running, setRunning] = useState<string | null>(null);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function runCase(id: string) {
    setRunning(id);
    setError(null);
    setResult(null);
    try {
      const r = await api.labRunCase(id);
      setResult(r);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : (err as Error).message);
    } finally {
      setRunning(null);
    }
  }

  const listError = casesQ.isError ? (casesQ.error as Error).message : null;

  return (
    <div>
      <PageHeader
        title="Resilience Lab"
        hideTitle
        subtitle="Browse catalog cases and compare expected vs actual judgments"
      />
      {error && <ErrorBanner message={error} />}
      {error && <ReachabilityHint message={error} />}
      {listError && <ErrorBanner message={listError} />}
      {listError && <ReachabilityHint message={listError} />}
      {casesQ.isLoading && <Spinner />}

      <Panel title="Cases" className="mb-6">
        <ul className="max-h-96 space-y-2 overflow-y-auto">
          {(casesQ.data?.cases || []).map((c) => (
            <li
              key={c.id}
              className="flex flex-wrap items-center justify-between gap-2 rounded-md bg-aspc-elevated px-3 py-2 shadow-recessed"
            >
              <div>
                <div className="font-mono text-sm text-aspc-accent">{c.id}</div>
                <div className="text-xs text-aspc-muted">
                  {c.category || "—"} · {c.description || c.entry || ""}
                </div>
              </div>
              <Button type="button" onClick={() => runCase(c.id)} disabled={running === c.id} tip="Run this example and compare expected vs actual.">
                {running === c.id ? "Running…" : "Run"}
              </Button>
            </li>
          ))}
        </ul>
      </Panel>

      {result && (
        <Panel title="Result">
          <JsonBlock value={result} />
        </Panel>
      )}
    </div>
  );
}
