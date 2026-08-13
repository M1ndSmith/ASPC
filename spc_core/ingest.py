"""Column auto-detection and frame validation (deduped from the three legacy pipelines).

Works on plain dicts of column -> values so the core stays free of pandas/polars.
Adapters convert CSV/Parquet into this shape before calling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

SKIP_PATTERNS = ("id", "subgroup", "batch", "sample", "group", "lot", "serial", "number",
                 "part", "operator", "trial", "appraiser")
MEASUREMENT_PRIORITY = ("measurement", "value", "measure", "reading", "result", "data",
                        "defect", "count", "defective")


@dataclass
class ColumnMap:
    value_col: str | None = None
    subgroup_col: str | None = None
    sample_size_col: str | None = None
    opportunity_col: str | None = None
    part_col: str | None = None
    operator_col: str | None = None
    trial_col: str | None = None
    reference_col: str | None = None
    date_col: str | None = None
    issues: list[str] = field(default_factory=list)


@dataclass
class IngestedFrame:
    columns: dict[str, list[Any]]
    column_map: ColumnMap
    n_rows: int


def _numeric_columns(columns: dict[str, list[Any]]) -> list[str]:
    out = []
    for name, vals in columns.items():
        try:
            arr = np.asarray(vals, dtype=float)
            if arr.size and not np.all(np.isnan(arr)):
                out.append(name)
        except (TypeError, ValueError):
            continue
    return out


def _pick_measurement(numeric_cols: list[str]) -> str | None:
    filtered = [c for c in numeric_cols
                if not any(p in c.lower() for p in SKIP_PATTERNS)]
    candidates = filtered or numeric_cols
    if not candidates:
        return None
    for priority in MEASUREMENT_PRIORITY:
        matching = [c for c in candidates if priority in c.lower()]
        if matching:
            return matching[0]
    return candidates[0]


def _find_by_names(columns: dict[str, list[Any]], names: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for name in names:
        if name in lower:
            return lower[name]
    for col in columns:
        for name in names:
            if name in col.lower():
                return col
    return None


def detect_columns(columns: dict[str, list[Any]],
                   value_col: str | None = None,
                   subgroup_col: str | None = None,
                   sample_size_col: str | None = None,
                   opportunity_col: str | None = None,
                   part_col: str | None = None,
                   operator_col: str | None = None,
                   trial_col: str | None = None,
                   reference_col: str | None = None,
                   date_col: str | None = None) -> ColumnMap:
    """Auto-detect standard SPC/MSA column roles from a column dict."""
    cmap = ColumnMap(
        value_col=value_col,
        subgroup_col=subgroup_col,
        sample_size_col=sample_size_col,
        opportunity_col=opportunity_col,
        part_col=part_col,
        operator_col=operator_col,
        trial_col=trial_col,
        reference_col=reference_col,
        date_col=date_col,
    )
    numeric = _numeric_columns(columns)

    if cmap.value_col is None:
        cmap.value_col = _pick_measurement(numeric)
        if cmap.value_col is None:
            cmap.issues.append("ERROR: No numeric columns found for analysis")

    if cmap.subgroup_col is None:
        cmap.subgroup_col = _find_by_names(columns, ("subgroup", "batch", "group", "lot"))
    if cmap.sample_size_col is None:
        cmap.sample_size_col = _find_by_names(
            columns, ("sample_size", "inspected", "n_inspected", "samplesize")
        )
    if cmap.opportunity_col is None:
        cmap.opportunity_col = _find_by_names(
            columns, ("opportunity", "units", "area", "opportunity_area")
        )
    if cmap.part_col is None:
        cmap.part_col = _find_by_names(columns, ("part",))
    if cmap.operator_col is None:
        cmap.operator_col = _find_by_names(columns, ("operator", "appraiser"))
    if cmap.trial_col is None:
        cmap.trial_col = _find_by_names(columns, ("trial", "repeat", "rep"))
    if cmap.reference_col is None:
        cmap.reference_col = _find_by_names(columns, ("reference", "standard", "master"))
    if cmap.date_col is None:
        cmap.date_col = _find_by_names(columns, ("date", "time", "timestamp", "datetime"))

    return cmap


def validate_frame(columns: dict[str, list[Any]], cmap: ColumnMap,
                   min_points: int = 25) -> list[str]:
    """Basic quality checks shared by control charts, MSA, and capability."""
    issues = list(cmap.issues)
    if cmap.value_col is None or cmap.value_col not in columns:
        issues.append("ERROR: Measurement/value column not found")
        return issues

    vals = columns[cmap.value_col]
    try:
        arr = np.asarray(vals, dtype=float)
    except (TypeError, ValueError):
        issues.append(f"ERROR: Column '{cmap.value_col}' is not numeric")
        return issues

    missing = int(np.isnan(arr).sum())
    clean = arr[~np.isnan(arr)]
    if missing > 0:
        issues.append(f"WARNING: {missing} missing values in '{cmap.value_col}'")
    if clean.size < min_points:
        issues.append(
            f"WARNING: Only {clean.size} points. Minimum {min_points} recommended"
        )
    if clean.size > 0 and np.unique(clean).size == 1:
        issues.append(f"WARNING: All values are identical ({clean[0]})")
    error_codes = clean[np.isin(clean, [999, 9999, -999, -9999])]
    if error_codes.size > 0:
        issues.append("WARNING: Potential error codes (999/-999) detected")
    return issues


def ingest(columns: dict[str, list[Any]], **overrides) -> IngestedFrame:
    """Detect columns, validate, and return a typed ingest result."""
    cmap = detect_columns(columns, **overrides)
    issues = validate_frame(columns, cmap)
    cmap.issues = issues
    n_rows = len(next(iter(columns.values()))) if columns else 0
    return IngestedFrame(columns=columns, column_map=cmap, n_rows=n_rows)
