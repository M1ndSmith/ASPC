"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { formatTimestamp, shortId } from "@/lib/format";
import { ErrorBanner, PageHeader, Panel, Spinner, StatCard } from "@/components/ui";

export default function DashboardPage() {
  const health = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 30_000 });
  const runs = useQuery({ queryKey: ["runs", 8], queryFn: () => api.listRuns({ limit: 8 }) });
  const streams = useQuery({
    queryKey: ["streams"],
    queryFn: api.listStreams,
    retry: false,
  });

  const healthy = health.data?.status === "healthy" || health.data?.status === "ok";
  const liveCount = streams.data?.streams?.filter((s) => s.active).length ?? null;
  const recent = runs.data?.runs ?? [];

  return (
    <div>
      <PageHeader
        title="Dashboard"
        subtitle="Plant-wide SPC health, recent analyses, and live streams"
      />

      {(health.isError || runs.isError) && (
        <ErrorBanner
          message={
            (health.error as Error)?.message ||
            (runs.error as Error)?.message ||
            "API unreachable — check NEXT_PUBLIC_API_URL"
          }
        />
      )}

      <div className="mb-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="API Health"
          value={health.isLoading ? "…" : healthy ? "Healthy" : health.isError ? "Down" : health.data?.status || "—"}
          hint={health.data?.version ? `v${health.data.version}` : undefined}
          tone={healthy ? "ok" : health.isError ? "stop" : "warn"}
        />
        <StatCard
          label="Live Streams"
          value={streams.isLoading ? "…" : liveCount === null ? "—" : liveCount}
          hint={streams.isError ? "Streams endpoint unavailable" : "Active Phase II monitors"}
          tone="cyan"
        />
        <StatCard
          label="Recent Runs"
          value={runs.isLoading ? "…" : recent.length}
          hint="Last batch analyses"
        />
        <StatCard
          label="Quick Actions"
          value={
            <span className="flex flex-wrap gap-2 text-sm font-sans font-medium">
              <Link href="/analyze" className="text-aspc-cyan hover:underline">
                Analyze
              </Link>
              <span className="text-aspc-muted">·</span>
              <Link href="/live" className="text-aspc-cyan hover:underline">
                Live
              </Link>
            </span>
          }
        />
      </div>

      <Panel title="Recent Runs">
        {runs.isLoading && <Spinner />}
        {!runs.isLoading && recent.length === 0 && (
          <p className="text-sm text-aspc-muted">No runs yet. Upload a file on Analyze.</p>
        )}
        {recent.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[32rem] text-left text-sm">
              <thead>
                <tr className="border-b border-aspc-border text-[11px] uppercase tracking-wider text-aspc-muted">
                  <th className="pb-2 pr-3 font-medium">Run</th>
                  <th className="pb-2 pr-3 font-medium">Type</th>
                  <th className="pb-2 pr-3 font-medium">Created</th>
                  <th className="pb-2 font-medium">Limits</th>
                </tr>
              </thead>
              <tbody>
                {recent.map((r) => (
                  <tr key={r.run_id} className="border-b border-aspc-border/60 hover:bg-aspc-bg/40">
                    <td className="py-2.5 pr-3">
                      <Link
                        href={`/runs/${r.run_id}`}
                        className="font-mono text-aspc-cyan hover:underline"
                      >
                        {shortId(r.run_id, 12)}
                      </Link>
                    </td>
                    <td className="py-2.5 pr-3 text-aspc-text">{r.analysis_type}</td>
                    <td className="py-2.5 pr-3 text-aspc-muted">{formatTimestamp(r.created_at)}</td>
                    <td className="py-2.5 font-mono text-xs text-aspc-muted">
                      {r.limits_version ? shortId(r.limits_version) : "—"}
                    </td>
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
