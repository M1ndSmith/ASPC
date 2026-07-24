"""Repository factory — select SQLite or TimescaleDB backend."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from adapters.persistence import Repository, SQLiteRepository


def get_repository(
    cfg: Any = None,
    backend: Optional[str] = None,
    *,
    sqlite_path: str | Path | None = None,
    timescale_dsn: Optional[str] = None,
) -> Repository:
    """Return a :class:`Repository` for the requested backend.

    Parameters
    ----------
    cfg:
        Optional :class:`apps.config.Config` (or duck-typed object with
        ``persistence_backend``, ``sqlite_path``, ``timescale_dsn``).
        When provided, backend/paths are taken from config unless overridden.
    backend:
        ``"sqlite"`` (default) or ``"timescale"`` / ``"timescaledb"`` / ``"postgres"``.
    sqlite_path:
        Path for the SQLite file (default ``aspc.db``).
    timescale_dsn:
        SQLAlchemy DSN for Timescale/Postgres. Async DSNs (``+asyncpg``) are
        normalized to sync ``+psycopg``.
    """
    if cfg is not None:
        backend = backend or getattr(cfg, "persistence_backend", None) or "sqlite"
        if sqlite_path is None:
            sqlite_path = getattr(cfg, "sqlite_path", None)
        if timescale_dsn is None:
            timescale_dsn = getattr(cfg, "timescale_dsn", None)

    key = (backend or "sqlite").strip().lower()
    if key in ("sqlite", "sqllite", "file"):
        return SQLiteRepository(sqlite_path or "aspc.db")
    if key in ("timescale", "timescaledb", "postgres", "postgresql", "tsdb"):
        if not timescale_dsn:
            raise ValueError("timescale_dsn is required for the TimescaleDB backend")
        from adapters.persistence_tsdb import TimescaleDBRepository

        return TimescaleDBRepository(timescale_dsn)
    raise ValueError(
        f"Unknown persistence backend '{backend}'. "
        "Use 'sqlite' or 'timescale'."
    )
