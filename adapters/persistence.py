"""Persistence adapters — repository interface with SQLite (and TimescaleDB).

Stores provenance: which analysis ran, on what data, with which frozen limits version,
at what time. This is the audit trail the legacy ephemeral-upload approach lacked.

TimescaleDB lives in ``adapters.persistence_tsdb`` (optional ``aspc[tsdb]`` extra).
"""
from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


class Repository(ABC):
    """Abstract store for SPC analysis runs and frozen control limits."""

    @abstractmethod
    def save_limits(self, limits_payload: dict[str, Any], version: str,
                    chart_type: str, meta: dict | None = None) -> str:
        ...

    @abstractmethod
    def get_limits(self, version: str) -> dict[str, Any] | None:
        ...

    @abstractmethod
    def save_run(self, analysis_type: str, report: dict[str, Any],
                 limits_version: str | None = None,
                 source_file: str | None = None,
                 user_id: str | None = None) -> str:
        ...

    @abstractmethod
    def get_run(self, run_id: str) -> dict[str, Any] | None:
        ...

    @abstractmethod
    def list_runs(self, analysis_type: str | None = None, limit: int = 50) -> list[dict]:
        ...

    @abstractmethod
    def save_audit(self, event: str, detail: dict[str, Any],
                   user_id: str | None = None) -> str:
        ...


class SQLiteRepository(Repository):
    """File-backed SQLite store — fine for prototyping and single-node deployment."""

    def __init__(self, db_path: str | Path = "aspc.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS control_limits (
                    version TEXT PRIMARY KEY,
                    chart_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    meta TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS analysis_runs (
                    run_id TEXT PRIMARY KEY,
                    analysis_type TEXT NOT NULL,
                    limits_version TEXT,
                    source_file TEXT,
                    user_id TEXT,
                    report TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_log (
                    event_id TEXT PRIMARY KEY,
                    event TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    user_id TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )

    def save_limits(self, limits_payload: dict[str, Any], version: str,
                    chart_type: str, meta: dict | None = None) -> str:
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO control_limits "
                "(version, chart_type, payload, meta, created_at) VALUES (?,?,?,?,?)",
                (version, chart_type, json.dumps(limits_payload, default=str),
                 json.dumps(meta or {}), now),
            )
        return version

    def get_limits(self, version: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM control_limits WHERE version = ?", (version,)
            ).fetchone()
        if not row:
            return None
        return {
            "version": row["version"],
            "chart_type": row["chart_type"],
            "payload": json.loads(row["payload"]),
            "meta": json.loads(row["meta"] or "{}"),
            "created_at": row["created_at"],
        }

    def save_run(self, analysis_type: str, report: dict[str, Any],
                 limits_version: str | None = None,
                 source_file: str | None = None,
                 user_id: str | None = None) -> str:
        run_id = str(uuid4())
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO analysis_runs "
                "(run_id, analysis_type, limits_version, source_file, user_id, report, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (run_id, analysis_type, limits_version, source_file, user_id,
                 json.dumps(report, default=str), now),
            )
        self.save_audit(
            "analysis_run",
            {"run_id": run_id, "analysis_type": analysis_type,
             "limits_version": limits_version, "source_file": source_file},
            user_id=user_id,
        )
        return run_id

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM analysis_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if not row:
            return None
        return {
            "run_id": row["run_id"],
            "analysis_type": row["analysis_type"],
            "limits_version": row["limits_version"],
            "source_file": row["source_file"],
            "user_id": row["user_id"],
            "report": json.loads(row["report"]),
            "created_at": row["created_at"],
        }

    def list_runs(self, analysis_type: str | None = None, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            if analysis_type:
                rows = conn.execute(
                    "SELECT run_id, analysis_type, limits_version, source_file, "
                    "user_id, created_at FROM analysis_runs "
                    "WHERE analysis_type = ? ORDER BY created_at DESC LIMIT ?",
                    (analysis_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT run_id, analysis_type, limits_version, source_file, "
                    "user_id, created_at FROM analysis_runs "
                    "ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def save_audit(self, event: str, detail: dict[str, Any],
                   user_id: str | None = None) -> str:
        event_id = str(uuid4())
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO audit_log (event_id, event, detail, user_id, created_at) "
                "VALUES (?,?,?,?,?)",
                (event_id, event, json.dumps(detail, default=str), user_id, now),
            )
        return event_id
