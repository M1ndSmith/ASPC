"""Excel bridge — export SPC reports and import simple measurement sheets."""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any


def spc_report_to_xlsx_bytes(report: dict[str, Any]) -> bytes:
    """Build a minimal XLSX workbook from an SPC report dict.

    Uses openpyxl when available; otherwise raises ImportError with install hint.
    """
    try:
        from openpyxl import Workbook
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "openpyxl is required for Excel export. Install with: pip install openpyxl"
        ) from exc

    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    limits = report.get("limits") or {}
    comps = limits.get("components") or {}
    primary = next(iter(comps.values()), {}) if comps else {}
    rows = [
        ("chart_type", report.get("chart_type") or limits.get("chart_type")),
        ("limits_version", limits.get("version")),
        ("center", primary.get("center") if isinstance(primary, dict) else None),
        ("ucl", primary.get("ucl") if isinstance(primary, dict) else None),
        ("lcl", primary.get("lcl") if isinstance(primary, dict) else None),
        ("phase", report.get("phase")),
        ("n_points", len(report.get("plotted_values") or [])),
        ("n_signals", len(report.get("signals") or [])),
    ]
    ws.append(["field", "value"])
    for k, v in rows:
        ws.append([k, v if not isinstance(v, list) else (v[0] if v else None)])

    ws2 = wb.create_sheet("Points")
    ws2.append(["index", "value", "ooc"])
    plotted = report.get("plotted_values") or []
    ooc = {int(s.get("index")) for s in (report.get("signals") or []) if isinstance(s, dict)}
    for i, v in enumerate(plotted):
        ws2.append([i, float(v), 1 if i in ooc else 0])

    ws3 = wb.create_sheet("Signals")
    ws3.append(["rule_id", "rule_name", "index", "value", "description", "side"])
    for s in report.get("signals") or []:
        if not isinstance(s, dict):
            continue
        ws3.append(
            [
                s.get("rule_id"),
                s.get("rule_name"),
                s.get("index"),
                s.get("value"),
                s.get("description"),
                s.get("side"),
            ]
        )

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def import_sheet_to_csv_bytes(path: Path | str) -> bytes:
    """Read first sheet of an xlsx/xls and emit CSV bytes (header + rows).

    Falls back to reading CSV/text as-is.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return p.read_bytes()
    if suffix not in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        raise ValueError(f"Unsupported spreadsheet type: {suffix}")
    try:
        from openpyxl import load_workbook
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "openpyxl is required for Excel import. Install with: pip install openpyxl"
        ) from exc
    wb = load_workbook(p, read_only=True, data_only=True)
    ws = wb.active
    out = io.StringIO()
    writer = csv.writer(out)
    for row in ws.iter_rows(values_only=True):
        if row is None or all(c is None for c in row):
            continue
        writer.writerow(["" if c is None else c for c in row])
    return out.getvalue().encode("utf-8")
