"""Empirically dump observed spc_core outcomes for each resilience case.

Usage:
  PYTHONPATH=. python3 scripts/calibrate_resilience.py > /tmp/calibrate.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from resilience_data import load_manifest  # noqa: E402
from resilience_data.runner import run_case  # noqa: E402


def main() -> int:
    cases = load_manifest()
    rows = []
    for spec in cases:
        # Strip expect so we always capture raw observation (raises become ERROR).
        probe = {**spec, "expect": {}}
        result = run_case(probe)
        rows.append(
            {
                "id": spec["id"],
                "entry": spec["entry"],
                "status": result["status"],
                "observed": result.get("observed"),
                "mismatches": result.get("mismatches"),
            }
        )
        print(f"# {spec['id']}: {result['status']}", file=sys.stderr)
        if result.get("mismatches"):
            print(f"  {result['mismatches']}", file=sys.stderr)
    json.dump(rows, sys.stdout, indent=2, default=str)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
