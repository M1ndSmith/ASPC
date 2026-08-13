"use client";

import type { ChecklistItem, Phase1Checklist } from "@/lib/types";

function ItemRow({ item }: { item: ChecklistItem }) {
  return (
    <li className="flex items-start gap-3 rounded-2xl border border-aspc-border bg-aspc-elevated/60 px-3 py-2.5">
      <span
        className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[10px] font-bold ${
          item.passed
            ? "bg-aspc-ok/20 text-aspc-ok"
            : "bg-aspc-stop/20 text-aspc-stop"
        }`}
        aria-label={item.passed ? "passed" : "failed"}
      >
        {item.passed ? "✓" : "✕"}
      </span>
      <div className="min-w-0 flex-1">
        <div className="font-mono text-xs font-medium text-aspc-text">
          {item.item.replace(/_/g, " ")}
        </div>
        <p className="mt-0.5 text-sm text-aspc-muted">{item.reason}</p>
      </div>
    </li>
  );
}

export function Checklist({
  checklist,
  items,
}: {
  checklist?: Phase1Checklist | null;
  items?: ChecklistItem[];
}) {
  const list = items ?? checklist?.items ?? [];
  const allPass = checklist?.passed ?? (list.length > 0 && list.every((i) => i.passed));

  if (!list.length) {
    return (
      <p className="text-sm text-aspc-muted">Phase I checklist not available for this analysis.</p>
    );
  }

  return (
    <div>
      <div className="mb-3 flex items-center justify-between gap-2">
        <p className="text-xs uppercase tracking-widest text-aspc-muted">
          Phase I · 10-item go-live
        </p>
        <span
          className={`rounded border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${
            allPass
              ? "border-aspc-ok/40 bg-aspc-ok/15 text-aspc-ok"
              : "border-aspc-warn/40 bg-aspc-warn/15 text-aspc-warn"
          }`}
        >
          {allPass ? "Ready" : "Blocked"}
        </span>
      </div>
      <ul className="space-y-2">
        {list.map((item, i) => (
          <ItemRow key={`${item.item}-${i}`} item={item} />
        ))}
      </ul>
    </div>
  );
}
