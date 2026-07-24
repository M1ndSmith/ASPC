"""File I/O adapters — CSV/Parquet readers with safe path handling.

Converts files into the plain column-dict shape that ``spc_core.ingest`` expects.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Optional


class FileReadError(Exception):
    """Raised when a data file cannot be read or is empty."""


_SAFE_NAME = re.compile(r"^[\w.\- ]+$")


def safe_filename(name: str) -> str:
    """Reject path traversal and unsafe characters; return basename only."""
    if name is None or not str(name).strip():
        raise FileReadError(f"Unsafe filename: {name!r}")
    raw = str(name)
    # Reject any path separators or parent-dir tokens before taking basename.
    if "/" in raw or "\\" in raw or ".." in raw:
        raise FileReadError(f"Unsafe filename: {name!r}")
    base = Path(raw).name
    if not base or base in (".", "..") or not _SAFE_NAME.match(base):
        raise FileReadError(f"Unsafe filename: {name!r}")
    return base


def read_csv(path: str | Path, encoding: str = "utf-8") -> dict[str, list[Any]]:
    """Read a CSV into ``{column: [values...]}``.

    Raises ``FileReadError`` for missing/empty files (replaces the legacy
    ``pd.errors.EmptyDataData`` typo that raised AttributeError).
    """
    p = Path(path)
    if not p.exists():
        raise FileReadError(f"File not found: {path}")
    if p.stat().st_size == 0:
        raise FileReadError(f"File is empty: {path}")

    with p.open(newline="", encoding=encoding) as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise FileReadError(f"No header row in CSV: {path}")
        columns: dict[str, list[Any]] = {name: [] for name in reader.fieldnames}
        row_count = 0
        for row in reader:
            row_count += 1
            for name in reader.fieldnames:
                raw = row.get(name, "")
                columns[name].append(_coerce(raw))
        if row_count == 0:
            raise FileReadError(f"CSV has headers but no data rows: {path}")
    return columns


def read_parquet(path: str | Path) -> dict[str, list[Any]]:
    """Read a Parquet file via Polars (optional dependency)."""
    try:
        import polars as pl
    except ImportError as exc:
        raise FileReadError(
            "polars is required for Parquet support. Install with: pip install aspc[data]"
        ) from exc
    p = Path(path)
    if not p.exists():
        raise FileReadError(f"File not found: {path}")
    df = pl.read_parquet(p)
    if df.height == 0:
        raise FileReadError(f"Parquet file is empty: {path}")
    return {col: df[col].to_list() for col in df.columns}


def load_columns(path: str | Path) -> dict[str, list[Any]]:
    """Dispatch on extension: .csv / .parquet / .pq."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return read_csv(p)
    if suffix in (".parquet", ".pq"):
        return read_parquet(p)
    raise FileReadError(f"Unsupported file type '{suffix}'. Use .csv or .parquet")


def save_upload(content: bytes, dest_dir: str | Path, filename: str,
                max_bytes: Optional[int] = None,
                allowed_extensions: Optional[list[str]] = None) -> Path:
    """Write an uploaded file safely into dest_dir."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    name = safe_filename(filename)
    if allowed_extensions:
        ext = Path(name).suffix.lower()
        if ext not in {e.lower() if e.startswith(".") else f".{e.lower()}"
                       for e in allowed_extensions}:
            raise FileReadError(f"Extension '{ext}' not allowed. Allowed: {allowed_extensions}")
    if max_bytes is not None and len(content) > max_bytes:
        raise FileReadError(f"File exceeds max size ({max_bytes} bytes)")
    target = dest / name
    target.write_bytes(content)
    return target


def _coerce(raw: str) -> Any:
    if raw is None or raw == "":
        return None
    try:
        if "." in raw or "e" in raw.lower():
            return float(raw)
        return int(raw)
    except ValueError:
        return raw
