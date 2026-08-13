"""File I/O adapters — CSV/Parquet readers with safe path handling.

Converts files into the plain column-dict shape that ``spc_core.ingest`` expects.
"""
from __future__ import annotations

import csv
import re
import uuid
from pathlib import Path
from typing import Any


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
                max_bytes: int | None = None,
                allowed_extensions: list[str] | None = None) -> Path:
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
    target = dest / f"{uuid.uuid4().hex}_{name}"
    target.write_bytes(content)
    return target


def save_upload_stream(
    stream,
    dest_dir: str | Path,
    filename: str,
    *,
    max_bytes: int | None = None,
    allowed_extensions: list[str] | None = None,
    chunk_size: int = 64 * 1024,
) -> Path:
    """Stream an upload to disk with a running size check (avoids reading whole body first)."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    name = safe_filename(filename)
    if allowed_extensions:
        ext = Path(name).suffix.lower()
        if ext not in {e.lower() if e.startswith(".") else f".{e.lower()}"
                       for e in allowed_extensions}:
            raise FileReadError(f"Extension '{ext}' not allowed. Allowed: {allowed_extensions}")
    target = dest / f"{uuid.uuid4().hex}_{name}"
    written = 0
    try:
        with target.open("wb") as out:
            while True:
                chunk = stream.read(chunk_size)
                if not chunk:
                    break
                written += len(chunk)
                if max_bytes is not None and written > max_bytes:
                    raise FileReadError(f"File exceeds max size ({max_bytes} bytes)")
                out.write(chunk)
    except Exception:
        if target.exists():
            target.unlink(missing_ok=True)
        raise
    if written == 0:
        target.unlink(missing_ok=True)
        raise FileReadError("Uploaded file is empty")
    return target


def resolve_under(base: str | Path, user_path: str) -> Path:
    """Resolve ``user_path`` and ensure it stays inside ``base`` (no traversal)."""
    root = Path(base).resolve()
    candidate = (root / user_path).resolve() if not Path(user_path).is_absolute() else Path(user_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise FileReadError(f"Path escapes allowed directory: {user_path!r}") from exc
    return candidate


def _coerce(raw: str) -> Any:
    if raw is None or raw == "":
        return None
    try:
        if "." in raw or "e" in raw.lower():
            return float(raw)
        return int(raw)
    except ValueError:
        return raw
