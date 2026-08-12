"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { formatTimestamp, shortId } from "@/lib/format";
import { ErrorBanner, PageHeader, Panel, Spinner } from "@/components/ui";

function RunsPageContent() {
  const searchParams = useSearchParams();
  const q = (searchParams.get("q") || "").trim().toLowerCase();

  const { data, isLoading, error, refetch, isFetching } = useQuery({
    queryKey: ["runs"],
    queryFn: () => api.listRuns({ limit: 100 }),
  });

  const runs = useMemo(() => {
    const all = data?.runs ?? [];
    if (!q) return all;
    return all.filter(
      (r) =>
        r.run_id.toLowerCase().includes(q) ||
        r.analysis_type.toLowerCase().includes(q) ||
        (r.source_file || "").toLowerCase().includes(q),
    );
  }, [data?.runs, q]);

  return (
    <div>
      <PageHeader
        title="Run History"
        hideTitle
        subtitle={q ? `Filtered by “${q}”` : "Audit trail of batch analyses from GET /runs"}
        actions={
          <button
            type="button"
            onClick={() => refetch()}
            className="rounded-pill border border-aspc-border px-4 py-1.5 text-xs text-aspc-muted hover:text-aspc-text"
          >
            {isFetching ? "Refreshing…" : "Refresh"}
          </button>
        }
      />

      {error && <ErrorBanner message={(error as Error).message} />}

      <Panel>
        {isLoading && <Spinner />}
        {!isLoading && runs.length === 0 && (
          <p className="text-sm text-aspc-muted">
            {q ? `No runs matching “${q}”.` : "No analysis runs stored yet."}
          </p>
        )}
        {runs.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[40rem] text-left text-sm">
              <thead>
                <tr className="border-b border-aspc-border text-[11px] uppercase tracking-wider text-aspc-muted">
                  <th className="pb-2 pr-3 font-medium">Run ID</th>
                  <th className="pb-2 pr-3 font-medium">Type</th>
                  <th className="pb-2 pr-3 font-medium">Source</th>
                  <th className="pb-2 pr-3 font-medium">Limits</th>
                  <th className="pb-2 font-medium">Created</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => (
                  <tr key={r.run_id} className="border-b border-aspc-border/50 hover:bg-aspc-elevated/50">
                    <td className="py-2.5 pr-3">
                      <Link
                        href={`/runs/${r.run_id}`}
                        className="font-mono text-aspc-accent hover:underline"
                      >
                        {shortId(r.run_id, 14)}
                      </Link>
                    </td>
                    <td className="py-2.5 pr-3">{r.analysis_type}</td>
                    <td className="max-w-[12rem] truncate py-2.5 pr-3 text-aspc-muted" title={r.source_file || ""}>
                      {r.source_file ? r.source_file.split(/[/\\]/).pop() : "—"}
                    </td>
                    <td className="py-2.5 pr-3 font-mono text-xs text-aspc-muted">
                      {r.limits_version ? shortId(r.limits_version) : "—"}
                    </td>
                    <td className="py-2.5 text-aspc-muted">{formatTimestamp(r.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  );
}

export default function RunsPage() {
  return (
    <Suspense fallback={<Spinner label="Loading runs…" />}>
      <RunsPageContent />
    </Suspense>
  );
}
