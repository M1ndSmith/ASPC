"""Standards-mapped resilience catalog for spc_core judgment.

CSV encoding: empty cells are missing (None); never the literal string \"None\".
"""
from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "MANIFEST.json"
CASES_DIR = ROOT / "cases"

Columns = dict[str, list[Any]]


def load_manifest(path: Path | None = None) -> list[dict[str, Any]]:
    """Load MANIFEST.json and return the list of case specs."""
    p = path or MANIFEST_PATH
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "cases" in data:
        return list(data["cases"])
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected MANIFEST shape in {p}")


def write_csv(cols: Mapping[str, list[Any]], path: str | Path) -> Path:
    """Write a column dict to CSV. None/NaN become empty cells."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(cols.keys())
    if not keys:
        raise ValueError("empty columns")
    n = len(cols[keys[0]])
    if any(len(cols[k]) != n for k in keys):
        raise ValueError("column length mismatch")

    def _cell(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return ""
        return str(v)

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(keys)
        for i in range(n):
            writer.writerow([_cell(cols[k][i]) for k in keys])
    return path


def read_csv(path: str | Path) -> Columns:
    """Read CSV into a column dict. Empty cells become None."""
    path = Path(path)
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return {}
        cols: Columns = {k: [] for k in reader.fieldnames}
        for row in reader:
            for k in reader.fieldnames:
                raw = row.get(k, "")
                if raw is None or raw == "":
                    cols[k].append(None)
                else:
                    cols[k].append(_coerce(raw))
    return cols


def _coerce(raw: str) -> Any:
    """Best-effort numeric coerce; leave strings for Part/Operator/reason."""
    try:
        if raw.isdigit() or (raw.startswith("-") and raw[1:].isdigit()):
            return int(raw)
        return float(raw)
    except ValueError:
        return raw


def case_csv_path(rel: str) -> Path:
    """Resolve a MANIFEST path relative to the package root."""
    return ROOT / rel
