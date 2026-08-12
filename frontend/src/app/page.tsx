"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { ControlChart } from "@/components/ControlChart";
import { DeltaChip, ErrorBanner, Panel, Spinner } from "@/components/ui";
import { api } from "@/lib/api";
import { formatTimestamp, shortId } from "@/lib/format";
import type { SPCReport } from "@/lib/types";

function asSpcReport(raw: unknown): SPCReport | null {
  if (!raw || typeof raw !== "object") return null;
  if ("plotted_values" in raw && "limits" in raw) return raw as SPCReport;
  return null;
}

const ACTIONS = [
  {
    href: "/analyze",
    title: "Analyze",
    ticker: "SPC",
    blurb: "Batch control charts",
    tone: "ok" as const,
  },
  {
    href: "/live",
    title: "Live",
    ticker: "WS",
    blurb: "Phase II streams",
    tone: "accent" as const,
  },
  {
    href: "/capability",
    title: "Capability",
    ticker: "Cpk",
    blurb: "Process capability",
    tone: "ok" as const,
  },
  {
    href: "/msa",
    title: "MSA",
    ticker: "GRR",
    blurb: "Measurement systems",
    tone: "warn" as const,
  },
];

export default function DashboardPage() {
  const health = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 30_000 });
  const runs = useQuery({ queryKey: ["runs", 8], queryFn: () => api.listRuns({ limit: 8 }) });
  const streams = useQuery({
    queryKey: ["streams"],
    queryFn: api.listStreams,
    retry: false,
  });

  const recent = runs.data?.runs ?? [];
  const latestId = recent[0]?.run_id;
  const latest = useQuery({
    queryKey: ["run", latestId],
    queryFn: () => api.getRun(latestId!),
    enabled: !!latestId,
  });

  const healthy = health.data?.status === "healthy" || health.data?.status === "ok";
  const liveCount = streams.data?.streams?.filter((s) => s.active).length ?? 0;
  const report = latest.data ? asSpcReport(latest.data.report) : null;
  const primary = report?.limits?.components ? Object.values(report.limits.components)[0] : null;
  const oocIndices = report?.signals?.map((s) => s.index) ?? [];

  return (
    <div className="space-y-6">
      {(health.isError || runs.isError) && (
        <ErrorBanner
          message={
            (health.error as Error)?.message ||
            (runs.error as Error)?.message ||
            "API unreachable — check NEXT_PUBLIC_API_URL"
          }
        />
      )}

      {/* Summary strip */}
      <section className="flex flex-wrap items-end justify-between gap-6 rounded-card border border-aspc-border bg-aspc-panel px-6 py-6 shadow-card">
        <div>
          <div className="text-[11px] font-medium uppercase tracking-widest text-aspc-muted">
            Platform status
          </div>
          <div className="mt-2 font-mono text-4xl font-semibold tracking-tight text-aspc-text md:text-5xl">
            {health.isLoading ? "…" : healthy ? "Online" : health.isError ? "Down" : health.data?.status || "—"}
          </div>
          <p className="mt-2 text-sm text-aspc-muted">
            {health.data?.version ? `API v${health.data.version}` : "SPC operator console"}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-6 md:gap-8">
          <DeltaChip
            label="Health"
            value={healthy ? "OK" : "Check"}
            positive={healthy ? true : health.isError ? false : null}
          />
          <div className="hidden h-10 w-px bg-aspc-border sm:block" />
          <DeltaChip label="Live" value={streams.isLoading ? "…" : String(liveCount)} positive={liveCount > 0 ? true : null} />
          <div className="hidden h-10 w-px bg-aspc-border sm:block" />
          <DeltaChip
            label="Runs"
            value={runs.isLoading ? "…" : String(recent.length)}
            positive={recent.length > 0 ? true : null}
          />
        </div>
      </section>

      {/* Action cards */}
      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {ACTIONS.map((a) => (
          <Link
            key={a.href}
            href={a.href}
            className="group rounded-card border border-aspc-border bg-aspc-panel p-5 shadow-card transition hover:border-aspc-accent/40 hover:shadow-glow"
          >
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="text-base font-semibold text-aspc-text">{a.title}</div>
                <div className="text-xs uppercase tracking-wider text-aspc-muted">{a.ticker}</div>
              </div>
              <span
                className={`flex h-9 w-9 items-center justify-center rounded-full ${
                  a.tone === "ok"
                    ? "bg-aspc-ok/15 text-aspc-ok"
                    : a.tone === "warn"
                      ? "bg-aspc-warn/15 text-aspc-warn"
                      : "bg-aspc-accent-soft text-aspc-accent"
                }`}
              >
                ↗
              </span>
            </div>
            <p className="mt-6 text-2xl font-semibold tracking-tight text-aspc-text group-hover:text-aspc-accent">
              Open
            </p>
            <p className="mt-1 text-sm text-aspc-muted">{a.blurb}</p>
            <div className="mt-5 h-10 overflow-hidden rounded-xl bg-gradient-to-r from-aspc-elevated via-aspc-accent/20 to-aspc-elevated opacity-80" />
          </Link>
        ))}
      </section>

      {/* Portfolio + Chart */}
      <section className="grid gap-4 lg:grid-cols-5">
        <Panel
          title="Recent runs"
          variant="accent"
          className="lg:col-span-2"
          action={
            <Link href="/runs" className="text-xs font-medium text-aspc-bg/70 hover:text-aspc-bg">
              View all
            </Link>
          }
        >
          {runs.isLoading && <Spinner label="Loading runs…" />}
          {!runs.isLoading && recent.length === 0 && (
            <p className="text-sm text-aspc-bg/70">No runs yet. Upload a file on Analyze.</p>
          )}
          <ul className="space-y-2">
            {recent.map((r) => (
              <li key={r.run_id}>
                <Link
                  href={`/runs/${r.run_id}`}
                  className="flex items-center gap-3 rounded-2xl bg-aspc-bg/10 px-3 py-2.5 transition hover:bg-aspc-bg/20"
                >
                  <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-aspc-bg/20 text-xs font-bold">
                    {r.analysis_type.slice(0, 2).toUpperCase()}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm font-semibold">{shortId(r.run_id, 14)}</div>
                    <div className="text-xs text-aspc-bg/65">{r.analysis_type}</div>
                  </div>
                  <span className="rounded-pill bg-aspc-bg/15 px-2.5 py-1 text-[11px] font-medium">
                    {formatTimestamp(r.created_at).split(",")[0]}
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        </Panel>

        <Panel
          title="Chart"
          className="lg:col-span-3"
          action={
            latestId ? (
              <Link href={`/runs/${latestId}`} className="text-xs text-aspc-muted hover:text-aspc-accent">
                Latest run
              </Link>
            ) : null
          }
        >
          {latest.isLoading && <Spinner label="Loading chart…" />}
          {!latestId && !runs.isLoading && (
            <div className="flex h-72 items-center justify-center rounded-2xl border border-dashed border-aspc-border text-sm text-aspc-muted">
              Run a control-chart analysis to populate this panel
            </div>
          )}
          {report && primary && (
            <div>
              <div className="mb-3 flex flex-wrap items-end justify-between gap-2">
                <div>
                  <div className="text-xs uppercase tracking-wider text-aspc-muted">
                    {report.chart_type}
                    {report.phase ? ` · ${report.phase}` : ""}
                  </div>
                  <div className="mt-1 font-mono text-2xl font-semibold text-aspc-text">
                    CL {primary.center.toPrecision(5)}
                  </div>
                </div>
              </div>
              <ControlChart
                values={report.plotted_values}
                ucl={primary.ucl}
                cl={primary.center}
                lcl={primary.lcl}
                oocIndices={oocIndices}
                title=""
                height={300}
              />
            </div>
          )}
          {latestId && latest.isSuccess && !report && (
            <div className="flex h-72 items-center justify-center rounded-2xl border border-dashed border-aspc-border text-sm text-aspc-muted">
              Latest run has no plotted control-chart series
            </div>
          )}
        </Panel>
      </section>
    </div>
  );
}
