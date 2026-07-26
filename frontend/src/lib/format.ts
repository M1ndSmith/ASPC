/** Display helpers for limits, ids, and timestamps. */

export interface LimitLike {
  center: number;
  ucl: number | number[];
  lcl: number | number[];
}

function fmt(n: number, digits: number): string {
  return n.toFixed(digits);
}

function limitRange(v: number | number[], digits: number): string {
  if (Array.isArray(v)) {
    if (v.length === 0) return "—";
    const lo = Math.min(...v);
    const hi = Math.max(...v);
    return lo === hi ? fmt(lo, digits) : `${fmt(lo, digits)}…${fmt(hi, digits)}`;
  }
  return fmt(v, digits);
}

/** Format UCL / CL / LCL for display (handles variable limits). */
export function formatLimits(limits: LimitLike, digits = 2): string {
  return `UCL ${limitRange(limits.ucl, digits)} · CL ${fmt(limits.center, digits)} · LCL ${limitRange(limits.lcl, digits)}`;
}

/** Resolve scalar or per-index limit value. */
export function limitAt(limit: number | number[] | undefined, index: number): number | undefined {
  if (limit === undefined) return undefined;
  if (Array.isArray(limit)) return limit[index] ?? limit[limit.length - 1];
  return limit;
}

/** Truncate long UUIDs for table cells. */
export function shortId(id: string, max = 10): string {
  if (id.length <= max) return id;
  return `${id.slice(0, max)}…`;
}

/** Human-readable timestamp from ISO string. */
export function formatTimestamp(iso: string | undefined | null): string {
  if (!iso) return "—";
  try {
    return new Intl.DateTimeFormat(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
    }).format(new Date(iso));
  } catch {
    return iso;
  }
}
