"""Standalone judgment report for the resilience catalog.

Writes ``resilience_data/JUDGMENT.md`` without leaving pytest with a dirty tree.

Usage:
  .venv/bin/python scripts/resilience_report.py
"""
from __future__ import annotations

import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resilience_data import ROOT as DATA_ROOT  # noqa: E402
from resilience_data import load_manifest  # noqa: E402
from resilience_data.runner import run_case  # noqa: E402

OUT = DATA_ROOT / "JUDGMENT.md"


def main() -> int:
    cases = load_manifest()
    rows = [run_case(spec) for spec in cases]
    counts = Counter(r["status"] for r in rows)

    lines = [
        "# Resilience judgment report",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        f"Cases: {len(rows)}",
        "",
        "| Status | Count |",
        "|--------|------:|",
    ]
    for status in ("PASS", "FAIL", "ERROR", "XFAIL"):
        lines.append(f"| {status} | {counts.get(status, 0)} |")
    lines += ["", "## Per-case results", ""]
    lines += ["| id | status | notes |", "|----|--------|-------|"]
    for r in rows:
        notes = ""
        if r["status"] != "PASS":
            notes = "; ".join(r.get("mismatches") or [])[:120]
            if r.get("xfail_reason"):
                notes = f"xfail: {r['xfail_reason']}; {notes}"
        lines.append(f"| `{r['id']}` | {r['status']} | {notes} |")

    fails = [r for r in rows if r["status"] in ("FAIL", "ERROR")]
    if fails:
        lines += ["", "## Failures detail", ""]
        for r in fails:
            lines.append(f"### `{r['id']}` — {r['status']}")
            lines.append("")
            for m in r.get("mismatches") or []:
                lines.append(f"- {m}")
            lines.append("")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} — {dict(counts)}")
    return 0 if counts.get("FAIL", 0) == 0 and counts.get("ERROR", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
