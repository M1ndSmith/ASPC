"use client";

import type { Gate, GateStatus } from "@/lib/types";

const styles: Record<GateStatus, string> = {
  ok: "bg-aspc-ok/15 text-aspc-ok border-aspc-ok/40",
  warn: "bg-aspc-warn/15 text-aspc-warn border-aspc-warn/40",
  stop: "bg-aspc-stop/15 text-aspc-stop border-aspc-stop/40",
};

function Badge({ status }: { status: GateStatus }) {
  return (
    <span
      className={`inline-flex min-w-[3.25rem] justify-center rounded border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${styles[status] || styles.warn}`}
    >
      {status}
    </span>
  );
}

export function GateList({ gates }: { gates: Gate[] }) {
  if (!gates?.length) {
    return (
      <p className="text-sm text-aspc-muted">No pipeline gates returned for this run.</p>
    );
  }

  return (
    <ul className="space-y-2">
      {gates.map((g, i) => (
        <li
          key={`${g.step}-${i}`}
          className="flex items-start gap-3 rounded-2xl border border-aspc-border bg-aspc-elevated/60 px-3 py-2.5"
        >
          <Badge status={g.status} />
          <div className="min-w-0 flex-1">
            <div className="font-mono text-xs font-medium text-aspc-accent">{g.step}</div>
            <p className="mt-0.5 text-sm text-aspc-text/90">{g.reason}</p>
          </div>
        </li>
      ))}
    </ul>
  );
}
