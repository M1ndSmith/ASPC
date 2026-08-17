"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { formatTimestamp, shortId } from "@/lib/format";
import { Button, DataTable, EmptyState, ErrorBanner, PageHeader, Panel, Spinner, Td, TextInput } from "@/components/ui";

function RunsPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const q = (searchParams.get("q") || "").trim().toLowerCase();
  const [draft, setDraft] = useState(searchParams.get("q") || "");

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

  function applySearch() {
    const next = draft.trim();
    router.push(next ? `/runs?q=${encodeURIComponent(next)}` : "/runs");
  }

  return (
    <div>
      <PageHeader
        title="Run History"
        hideTitle
        subtitle={q ? `Filtered by “${q}”` : "Audit trail of batch analyses from GET /runs"}
        actions={
          <Button type="button" variant="secondary" onClick={() => refetch()} tip="Reload the list of past analyses.">
            {isFetching ? "Refreshing…" : "Refresh"}
          </Button>
        }
      />

      {error && <ErrorBanner message={(error as Error).message} />}

      <Panel className="mb-6">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
          <div className="flex-1">
            <TextInput
              id="runs_search"
              label="Search runs"
              type="search"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") applySearch();
              }}
              placeholder="Run ID, type, or file name"
            />
          </div>
          <Button type="button" onClick={applySearch} tip="Show only runs that match this text.">
            Search
          </Button>
        </div>
      </Panel>

      <Panel>
        {isLoading && <Spinner />}
        {!isLoading && runs.length === 0 && (
          <EmptyState>{q ? `No runs matching “${q}”.` : "No analysis runs stored yet."}</EmptyState>
        )}
        {runs.length > 0 && (
          <DataTable headers={["Run ID", "Type", "Source", "Limits", "Created"]}>
            {runs.map((r) => (
              <tr key={r.run_id}>
                <Td>
                  <Link href={`/runs/${r.run_id}`} className="font-mono text-aspc-accent hover:underline">
                    {shortId(r.run_id, 14)}
                  </Link>
                </Td>
                <Td>{r.analysis_type}</Td>
                <Td className="max-w-[12rem] truncate text-aspc-muted" title={r.source_file || ""}>
                  {r.source_file ? r.source_file.split(/[/\\]/).pop() : "—"}
                </Td>
                <Td className="font-mono text-xs text-aspc-muted">
                  {r.limits_version ? shortId(r.limits_version) : "—"}
                </Td>
                <Td className="text-aspc-muted">{formatTimestamp(r.created_at)}</Td>
              </tr>
            ))}
          </DataTable>
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
