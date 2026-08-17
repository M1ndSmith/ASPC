"use client";

import type { Gate, GateStatus } from "@/lib/types";
import { Term } from "@/components/ui";

const styles: Record<GateStatus, string> = {
  ok: "bg-aspc-ok/15 text-aspc-ok",
  warn: "bg-aspc-warn/15 text-aspc-warn",
  stop: "bg-aspc-accent-soft text-aspc-accent",
};

function Badge({ status }: { status: GateStatus }) {
  return (
    <span
      className={`inline-flex min-w-[3.25rem] justify-center rounded-sm px-2 py-0.5 font-mono text-[10px] font-bold uppercase tracking-wider ${styles[status] || styles.warn}`}
    >
      {status}
    </span>
  );
}

export function GateList({ gates }: { gates: Gate[] }) {
  if (!gates?.length) {
    return (
      <p className="text-sm text-aspc-muted">
        No pipeline <Term k="gate">gates</Term> returned for this run.
      </p>
    );
  }

  return (
    <ul className="space-y-2">
      {gates.map((g, i) => (
        <li
          key={`${g.step}-${i}`}
          className="flex items-start gap-3 rounded-md bg-aspc-elevated px-3 py-2.5 shadow-recessed"
        >
          <Badge status={g.status} />
          <div className="min-w-0 flex-1">
            <div className="font-mono text-xs font-medium text-aspc-accent">{g.step}</div>
            <p className="mt-0.5 text-sm text-aspc-text">{g.reason}</p>
          </div>
        </li>
      ))}
    </ul>
  );
}
