"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { ArrowUpRight, Beaker, LineChart, Radio, Sparkles } from "lucide-react";
import { ControlChart } from "@/components/ControlChart";
import {
  DataTable,
  DeltaChip,
  EmptyState,
  ErrorBanner,
  Led,
  Panel,
  Spinner,
  Td,
  Tooltip,
} from "@/components/ui";
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
    href: "/onboarding",
    title: "Onboarding",
    blurb: "Use a sample file; we’ll walk you through it.",
    icon: Sparkles,
  },
  {
    href: "/analyze",
    title: "Analyze",
    blurb: "Upload a spreadsheet of measurements.",
    icon: LineChart,
  },
  {
    href: "/live",
    title: "Live",
    blurb: "Watch measurements as they arrive.",
    icon: Radio,
  },
  {
    href: "/lab",
    title: "Lab",
    blurb: "Try known good and bad examples.",
    icon: Beaker,
  },
];

export default function OverviewPage() {
  const health = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 30_000 });
  const runs = useQuery({ queryKey: ["runs", 8], queryFn: () => api.listRuns({ limit: 8 }) });
  const streams = useQuery({ queryKey: ["streams"], queryFn: api.listStreams, retry: false });
  const ops = useQuery({ queryKey: ["ops-summary"], queryFn: api.opsSummary, retry: false });

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
    <div className="space-y-8">
      {(health.isError || runs.isError) && (
        <ErrorBanner
          message={
            (health.error as Error)?.message ||
            (runs.error as Error)?.message ||
            "Can't reach the server. Check that it is running, then refresh."
          }
        />
      )}

      <section className="relative overflow-hidden rounded-xl bg-aspc-dark px-6 py-8 text-white shadow-sharp md:px-10">
        <div
          className="pointer-events-none absolute inset-0 opacity-20 mix-blend-overlay"
          style={{
            backgroundImage:
              "repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(0,0,0,0.35) 3px, rgba(0,0,0,0.35) 4px)",
          }}
          aria-hidden
        />
        <div className="relative flex flex-wrap items-end justify-between gap-6">
          <div>
            <div className="mb-3">
              <Led on={healthy} tone={healthy ? "ok" : "accent"} label="System" inverted />
            </div>
            <div className="font-mono text-xs font-bold uppercase tracking-[0.08em] text-white/60">
              Is everything working?
            </div>
            <div className="mt-2 font-mono text-4xl font-semibold tracking-tight md:text-5xl">
              {health.isLoading ? "…" : healthy ? "Online" : health.isError ? "Down" : health.data?.status || "—"}
            </div>
            <p className="mt-2 text-sm text-white/70">
              {health.data?.version ? `Server version ${health.data.version}` : healthy ? "Server connected" : ""}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-6 md:gap-8">
            <DeltaChip
              inverted
              label="Health"
              value={healthy ? "OK" : "Check"}
              positive={healthy ? true : health.isError ? false : null}
              tip="Whether the server that stores results is reachable."
            />
            <DeltaChip
              inverted
              label="Live"
              value={streams.isLoading ? "…" : String(liveCount)}
              positive={liveCount > 0 ? true : null}
              tip="How many production lines are being watched right now."
            />
            <DeltaChip
              inverted
              label="Runs"
              value={runs.isLoading ? "…" : String(recent.length)}
              positive={recent.length > 0 ? true : null}
              tip="How many recent analyses are on this home page."
            />
            <DeltaChip
              inverted
              label="Issues"
              value={ops.isLoading ? "…" : String(ops.data?.checklist_debt ?? 0)}
              positive={(ops.data?.checklist_debt ?? 0) === 0 ? true : false}
              tip="Items that still block going live."
            />
          </div>
        </div>
      </section>

      {ops.data && ops.data.streams.length > 0 && (
        <Panel title="Lines being watched">
          <DataTable headers={["Line", "On", "Chart", "Limits"]}>
            {ops.data.streams.map((s) => (
              <tr key={s.stream_key}>
                <Td className="font-mono text-aspc-accent">{s.stream_key}</Td>
                <Td>{s.active ? "yes" : "no"}</Td>
                <Td className="font-mono">{s.chart_type || "—"}</Td>
                <Td className="font-mono text-xs">{s.limits_version || "—"}</Td>
              </tr>
            ))}
          </DataTable>
        </Panel>
      )}

      <section className="grid gap-6 sm:grid-cols-2 xl:grid-cols-4">
        {ACTIONS.map((a) => {
          const Icon = a.icon;
          return (
            <Tooltip key={a.href} content={a.blurb} className="w-full">
              <Link
                href={a.href}
                title={a.blurb}
                className="group block w-full rounded-lg bg-aspc-bg p-6 shadow-card transition duration-300 ease-mech hover:-translate-y-1 hover:shadow-floating"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="text-base font-semibold text-aspc-text">{a.title}</div>
                  <span className="flex h-14 w-14 items-center justify-center rounded-full bg-aspc-bg text-aspc-accent shadow-floating">
                    <Icon className="h-7 w-7 transition duration-200 group-hover:rotate-12 group-hover:scale-110" strokeWidth={1.5} />
                  </span>
                </div>
                <p className="mt-6 flex items-center gap-1 text-2xl font-semibold tracking-tight text-aspc-text group-hover:text-aspc-accent">
                  Open <ArrowUpRight className="h-5 w-5" />
                </p>
                <p className="mt-1 text-sm text-aspc-muted">{a.blurb}</p>
              </Link>
            </Tooltip>
          );
        })}
      </section>

      <section className="grid gap-6 lg:grid-cols-5">
        <Panel
          title="Recent runs"
          className="lg:col-span-2"
          action={
            <Link href="/runs" className="text-xs font-bold uppercase tracking-wide text-aspc-accent hover:underline" title="See every past analysis">
              View all
            </Link>
          }
        >
          {runs.isLoading && <Spinner label="Loading runs…" />}
          {!runs.isLoading && recent.length === 0 && (
            <EmptyState>No runs yet. Upload a file on Analyze.</EmptyState>
          )}
          <ul className="space-y-2">
            {recent.map((r) => (
              <li key={r.run_id}>
                <Link
                  href={`/runs/${r.run_id}`}
                  title="Open this analysis"
                  className="flex items-center gap-3 rounded-md bg-aspc-elevated px-3 py-2.5 shadow-recessed transition hover:shadow-card"
                >
                  <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-aspc-bg font-mono text-xs font-bold shadow-card">
                    {r.analysis_type.slice(0, 2).toUpperCase()}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm font-semibold">{shortId(r.run_id, 14)}</div>
                    <div className="text-xs text-aspc-muted">{r.analysis_type}</div>
                  </div>
                  <span className="rounded-pill bg-aspc-bg px-2.5 py-1 font-mono text-xs font-medium shadow-card">
                    {formatTimestamp(r.created_at).split(",")[0]}
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        </Panel>

        <Panel
          title="Latest chart"
          className="lg:col-span-3"
          action={
            latestId ? (
              <Link href={`/runs/${latestId}`} className="text-xs text-aspc-muted hover:text-aspc-accent" title="Open the newest analysis">
                Latest run
              </Link>
            ) : null
          }
        >
          {latest.isLoading && <Spinner label="Loading chart…" />}
          {!latestId && !runs.isLoading && (
            <EmptyState>Upload a measurements file on Analyze to see a chart here.</EmptyState>
          )}
          {report && primary && (
            <div>
              <div className="mb-3">
                <div className="font-mono text-xs font-bold uppercase tracking-wider text-aspc-muted">
                  {report.chart_type}
                  {report.phase ? ` · ${report.phase}` : ""}
                </div>
                <div className="mt-1 font-mono text-2xl font-semibold">Typical {primary.center.toPrecision(5)}</div>
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
            <EmptyState>The newest run has no chart to draw.</EmptyState>
          )}
        </Panel>
      </section>
    </div>
  );
}
