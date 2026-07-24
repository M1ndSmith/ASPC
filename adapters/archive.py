"""Cold-archive export of raw measurements to Parquet/CSV."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_COLUMNS = (
    "stream_key",
    "ts",
    "value",
    "quality_flag",
    "machine_id",
    "gage_id",
    "limits_version",
)


def export_parquet(rows: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]],
                   path: str | Path) -> Path:
    """Write observation rows to Parquet (polars) or CSV fallback.

    Each row should be a mapping with at least ``stream_key``, ``ts``, ``value``.
    Extra keys are preserved when using polars; CSV fallback writes known columns.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    materialised = [dict(r) for r in rows]

    try:
        import polars as pl
    except ImportError:
        pl = None  # type: ignore[assignment]

    if pl is not None:
        df = pl.DataFrame(materialised) if materialised else pl.DataFrame({c: [] for c in _COLUMNS})
        if path.suffix.lower() == ".csv":
            df.write_csv(path)
        else:
            # Ensure .parquet suffix for clarity when caller omitted it
            if path.suffix.lower() not in (".parquet", ".pq", ".csv"):
                path = path.with_suffix(".parquet")
            df.write_parquet(path)
        return path

    # CSV fallback when polars is unavailable
    out = path if path.suffix.lower() == ".csv" else path.with_suffix(".csv")
    fieldnames = list(_COLUMNS)
    for row in materialised:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in materialised:
            serialised = {}
            for k, v in row.items():
                if hasattr(v, "isoformat"):
                    serialised[k] = v.isoformat()
                else:
                    serialised[k] = v
            writer.writerow(serialised)
    return out


def export_raw_to_parquet(
    repo_or_engine: Any,
    stream_key: str,
    start: datetime,
    end: datetime,
    path: str | Path,
) -> Path:
    """Dump ``raw_measurements`` for ``stream_key`` in ``[start, end]`` via :func:`export_parquet`.

    Accepts a :class:`TimescaleDBRepository` (preferred) or a SQLAlchemy ``Engine``.
    """
    rows = _fetch_rows(repo_or_engine, stream_key, start, end)
    return export_parquet(rows, path)


def _fetch_rows(
    repo_or_engine: Any,
    stream_key: str,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    if hasattr(repo_or_engine, "query_raw_measurements"):
        return list(repo_or_engine.query_raw_measurements(stream_key, start, end))

    try:
        from sqlalchemy import select
        from sqlalchemy.orm import Session
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Engine-based export requires the tsdb extra. "
            "Install with: pip install 'aspc[tsdb]'"
        ) from exc

    from adapters.db_models import RawMeasurementRow

    engine = repo_or_engine
    with Session(engine) as session:
        stmt = (
            select(RawMeasurementRow)
            .where(
                RawMeasurementRow.stream_key == stream_key,
                RawMeasurementRow.ts >= start,
                RawMeasurementRow.ts <= end,
            )
            .order_by(RawMeasurementRow.ts)
        )
        return [
            {
                "id": r.id,
                "stream_key": r.stream_key,
                "ts": r.ts,
                "value": r.value,
                "quality_flag": r.quality_flag,
                "machine_id": r.machine_id,
                "gage_id": r.gage_id,
                "limits_version": r.limits_version,
            }
            for r in session.scalars(stmt).all()
        ]
