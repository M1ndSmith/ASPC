"""TimescaleDB / PostgreSQL repository (optional ``aspc[tsdb]`` extra).

Uses a SQLAlchemy sync engine with the ``psycopg`` driver. Also works against
SQLite for local tests (tables only — no hypertables or retention policies).
"""
from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

try:
    from sqlalchemy import create_engine, func, select, text
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from sqlalchemy.engine import Engine
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.orm import Session, sessionmaker
except ImportError as exc:  # pragma: no cover - exercised when extra missing
    raise ImportError(
        "TimescaleDB persistence requires the tsdb extra. "
        "Install with: pip install 'aspc[tsdb]'"
    ) from exc

logger = logging.getLogger(__name__)

from adapters.db_models import (
    AnalysisRunRow,
    AuditLogRow,
    Base,
    CapabilityHistoryRow,
    ControlLimitRow,
    OocEventRow,
    RawMeasurementRow,
    StreamRegistryRow,
)
from adapters.persistence import Repository


def normalize_sync_dsn(dsn: str) -> str:
    """Convert common ASPC DSNs to a SQLAlchemy sync ``psycopg`` URL."""
    if dsn.startswith("postgresql+psycopg"):
        return dsn
    for prefix in (
        "postgresql+asyncpg://",
        "postgresql://",
        "postgres://",
    ):
        if dsn.startswith(prefix):
            return "postgresql+psycopg://" + dsn[len(prefix) :]
    return dsn


def init_schema(engine: Engine) -> None:
    """Create tables; on PostgreSQL also enable Timescale hypertables + retention."""
    Base.metadata.create_all(engine)
    if engine.dialect.name != "postgresql":
        return
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"))
        conn.execute(
            text(
                "SELECT create_hypertable("
                "'raw_measurements', 'ts', if_not_exists => TRUE)"
            )
        )
        # Retention may already exist on re-init; log (don't swallow silently).
        try:
            conn.execute(
                text(
                    """
                    SELECT add_retention_policy(
                        'raw_measurements',
                        INTERVAL '90 days',
                        if_not_exists => TRUE
                    );
                    """
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Timescale retention policy not applied on raw_measurements: %s",
                exc,
            )


class TimescaleDBRepository(Repository):
    """SQLAlchemy-backed repository for PostgreSQL/TimescaleDB (or SQLite)."""

    def __init__(self, dsn: str | Engine, *, echo: bool = False, init: bool = True):
        if isinstance(dsn, Engine):
            self.engine = dsn
        else:
            self.engine = create_engine(normalize_sync_dsn(dsn), echo=echo, future=True)
        self._Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)
        # Compose runs Alembic in a migrate service; set ASPC_TSDB_INIT=0 to skip
        # concurrent create_all races from api + stream-engine.
        env_init = os.getenv("ASPC_TSDB_INIT", "1").lower() not in ("0", "false", "no")
        if init and env_init:
            init_schema(self.engine)

    def _session(self) -> Session:
        return self._Session()

    # --- Repository ABC -----------------------------------------------------

    def save_limits(
        self,
        limits_payload: dict[str, Any],
        version: str,
        chart_type: str,
        meta: Optional[dict] = None,
    ) -> str:
        now = datetime.now(timezone.utc)
        with self._session() as session:
            row = session.get(ControlLimitRow, version)
            if row is None:
                row = ControlLimitRow(
                    version=version,
                    chart_type=chart_type,
                    payload=limits_payload,
                    meta=meta or {},
                    created_at=now,
                )
                session.add(row)
            else:
                row.chart_type = chart_type
                row.payload = limits_payload
                row.meta = meta or {}
            session.commit()
        return version

    def get_limits(self, version: str) -> Optional[dict[str, Any]]:
        with self._session() as session:
            row = session.get(ControlLimitRow, version)
            if row is None:
                return None
            return {
                "version": row.version,
                "chart_type": row.chart_type,
                "payload": row.payload,
                "meta": row.meta or {},
                "created_at": row.created_at.isoformat()
                if isinstance(row.created_at, datetime)
                else row.created_at,
            }

    def save_run(
        self,
        analysis_type: str,
        report: dict[str, Any],
        limits_version: Optional[str] = None,
        source_file: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        run_id = str(uuid4())
        now = datetime.now(timezone.utc)
        with self._session() as session:
            session.add(
                AnalysisRunRow(
                    run_id=run_id,
                    analysis_type=analysis_type,
                    limits_version=limits_version,
                    source_file=source_file,
                    user_id=user_id,
                    report=report,
                    created_at=now,
                )
            )
            session.commit()
        self.save_audit(
            "analysis_run",
            {
                "run_id": run_id,
                "analysis_type": analysis_type,
                "limits_version": limits_version,
                "source_file": source_file,
            },
            user_id=user_id,
        )
        return run_id

    def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
        with self._session() as session:
            row = session.get(AnalysisRunRow, run_id)
            if row is None:
                return None
            return {
                "run_id": row.run_id,
                "analysis_type": row.analysis_type,
                "limits_version": row.limits_version,
                "source_file": row.source_file,
                "user_id": row.user_id,
                "report": row.report,
                "created_at": row.created_at.isoformat()
                if isinstance(row.created_at, datetime)
                else row.created_at,
            }

    def list_runs(
        self, analysis_type: Optional[str] = None, limit: int = 50
    ) -> list[dict]:
        with self._session() as session:
            stmt = select(AnalysisRunRow).order_by(AnalysisRunRow.created_at.desc()).limit(
                limit
            )
            if analysis_type:
                stmt = stmt.where(AnalysisRunRow.analysis_type == analysis_type)
            rows = session.scalars(stmt).all()
            return [
                {
                    "run_id": r.run_id,
                    "analysis_type": r.analysis_type,
                    "limits_version": r.limits_version,
                    "source_file": r.source_file,
                    "user_id": r.user_id,
                    "created_at": r.created_at.isoformat()
                    if isinstance(r.created_at, datetime)
                    else r.created_at,
                }
                for r in rows
            ]

    def save_audit(
        self,
        event: str,
        detail: dict[str, Any],
        user_id: Optional[str] = None,
    ) -> str:
        event_id = str(uuid4())
        now = datetime.now(timezone.utc)
        with self._session() as session:
            session.add(
                AuditLogRow(
                    event_id=event_id,
                    event=event,
                    detail=detail,
                    user_id=user_id,
                    created_at=now,
                )
            )
            session.commit()
        return event_id

    # --- Streaming / Tier-1 / Tier-2 ----------------------------------------

    @staticmethod
    def _measurement_id(stream_key: str, ts: datetime, value: float) -> int:
        """Deterministic id so Kafka redelivery is a no-op on the PK (id, ts)."""
        blob = f"{stream_key}|{ts.isoformat()}|{value:.12g}".encode()
        return int(hashlib.sha256(blob).hexdigest()[:15], 16)

    def save_raw_measurement(
        self,
        stream_key: str,
        ts: datetime,
        value: float,
        **meta: Any,
    ) -> None:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        row_id = int(meta["id"]) if meta.get("id") is not None else self._measurement_id(
            stream_key, ts, float(value)
        )
        values = dict(
            id=row_id,
            stream_key=stream_key,
            ts=ts,
            value=float(value),
            quality_flag=meta.get("quality_flag"),
            machine_id=meta.get("machine_id"),
            gage_id=meta.get("gage_id"),
            limits_version=meta.get("limits_version"),
        )
        with self._session() as session:
            if self.engine.dialect.name == "postgresql":
                stmt = (
                    pg_insert(RawMeasurementRow)
                    .values(**values)
                    .on_conflict_do_nothing(constraint="pk_raw_measurements")
                )
                session.execute(stmt)
                session.commit()
                return
            try:
                session.add(RawMeasurementRow(**values))
                session.commit()
            except IntegrityError:
                session.rollback()

    def count_raw_measurements(self, stream_key: str) -> int:
        with self._session() as session:
            return int(
                session.scalar(
                    select(func.count())
                    .select_from(RawMeasurementRow)
                    .where(RawMeasurementRow.stream_key == stream_key)
                )
                or 0
            )

    def recent_raw_measurements(
        self,
        stream_key: str,
        *,
        limit: int = 15,
    ) -> list[dict[str, Any]]:
        with self._session() as session:
            stmt = (
                select(RawMeasurementRow)
                .where(RawMeasurementRow.stream_key == stream_key)
                .order_by(RawMeasurementRow.ts.desc())
                .limit(limit)
            )
            rows = list(reversed(session.scalars(stmt).all()))
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
                for r in rows
            ]

    def save_ooc_event(
        self,
        stream_key: str,
        ts: datetime,
        *,
        limits_version: Optional[str],
        index: int,
        value: float,
        rule_id: str,
        rule_name: str,
        description: str,
        side: Optional[str] = None,
    ) -> bool:
        """Insert an OOC event. Returns True if inserted, False if duplicate."""
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        with self._session() as session:
            if self.engine.dialect.name == "postgresql":
                stmt = (
                    pg_insert(OocEventRow)
                    .values(
                        stream_key=stream_key,
                        limits_version=limits_version,
                        index=index,
                        value=float(value),
                        rule_id=rule_id,
                        rule_name=rule_name,
                        description=description,
                        side=side,
                        ts=ts,
                        acked=False,
                    )
                    .on_conflict_do_nothing(
                        constraint="uq_ooc_stream_ts_rule"
                    )
                )
                result = session.execute(stmt)
                session.commit()
                return (result.rowcount or 0) > 0

            existing = session.execute(
                select(OocEventRow).where(
                    OocEventRow.stream_key == stream_key,
                    OocEventRow.ts == ts,
                    OocEventRow.rule_id == rule_id,
                )
            ).scalar_one_or_none()
            if existing is not None:
                return False
            session.add(
                OocEventRow(
                    stream_key=stream_key,
                    limits_version=limits_version,
                    index=index,
                    value=float(value),
                    rule_id=rule_id,
                    rule_name=rule_name,
                    description=description,
                    side=side,
                    ts=ts,
                    acked=False,
                )
            )
            session.commit()
            return True

    def list_ooc_events(
        self,
        stream_key: Optional[str] = None,
        *,
        unacked_only: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._session() as session:
            stmt = select(OocEventRow).order_by(OocEventRow.ts.desc()).limit(limit)
            if stream_key:
                stmt = stmt.where(OocEventRow.stream_key == stream_key)
            if unacked_only:
                stmt = stmt.where(OocEventRow.acked.is_(False))
            return [_ooc_to_dict(r) for r in session.scalars(stmt).all()]

    def ack_alert(
        self,
        event_id: int,
        *,
        acked_by: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Acknowledge an OOC alert by id. Returns the updated row or None if missing."""
        with self._session() as session:
            row = session.get(OocEventRow, event_id)
            if row is None:
                return None
            row.acked = True
            row.acked_at = datetime.now(timezone.utc)
            row.acked_by = acked_by
            session.commit()
            session.refresh(row)
            return _ooc_to_dict(row)

    def save_capability(
        self,
        run_id: str,
        *,
        cpk: Optional[float] = None,
        ppk: Optional[float] = None,
        sigma_level: Optional[float] = None,
    ) -> int:
        with self._session() as session:
            row = CapabilityHistoryRow(
                run_id=run_id,
                cpk=cpk,
                ppk=ppk,
                sigma_level=sigma_level,
                created_at=datetime.now(timezone.utc),
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return int(row.id)

    def register_stream(
        self,
        stream_key: str,
        *,
        topic: Optional[str] = None,
        limits_version: Optional[str] = None,
        chart_type: Optional[str] = None,
        ruleset: str = "nelson",
        active: bool = True,
        meta: Optional[dict] = None,
    ) -> str:
        now = datetime.now(timezone.utc)
        with self._session() as session:
            row = session.get(StreamRegistryRow, stream_key)
            if row is None:
                session.add(
                    StreamRegistryRow(
                        stream_key=stream_key,
                        topic=topic,
                        limits_version=limits_version,
                        chart_type=chart_type,
                        ruleset=ruleset,
                        active=active,
                        meta=meta or {},
                        created_at=now,
                    )
                )
            else:
                if topic is not None:
                    row.topic = topic
                if limits_version is not None:
                    row.limits_version = limits_version
                if chart_type is not None:
                    row.chart_type = chart_type
                row.ruleset = ruleset
                row.active = active
                if meta is not None:
                    row.meta = meta
            session.commit()
        return stream_key

    def get_stream(self, stream_key: str) -> Optional[dict[str, Any]]:
        with self._session() as session:
            row = session.get(StreamRegistryRow, stream_key)
            if row is None:
                return None
            return _stream_to_dict(row)

    def list_streams(self, active_only: bool = False) -> list[dict[str, Any]]:
        with self._session() as session:
            stmt = select(StreamRegistryRow).order_by(StreamRegistryRow.stream_key)
            if active_only:
                stmt = stmt.where(StreamRegistryRow.active.is_(True))
            return [_stream_to_dict(r) for r in session.scalars(stmt).all()]

    def set_stream_active(self, stream_key: str, active: bool) -> None:
        with self._session() as session:
            row = session.get(StreamRegistryRow, stream_key)
            if row is None:
                raise KeyError(f"Unknown stream_key: {stream_key}")
            row.active = active
            session.commit()

    def query_raw_measurements(
        self,
        stream_key: str,
        start: datetime,
        end: datetime,
    ) -> list[dict[str, Any]]:
        with self._session() as session:
            stmt = (
                select(RawMeasurementRow)
                .where(
                    RawMeasurementRow.stream_key == stream_key,
                    RawMeasurementRow.ts >= start,
                    RawMeasurementRow.ts <= end,
                )
                .order_by(RawMeasurementRow.ts)
            )
            rows = session.scalars(stmt).all()
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
                for r in rows
            ]


def _stream_to_dict(row: StreamRegistryRow) -> dict[str, Any]:
    return {
        "stream_key": row.stream_key,
        "topic": row.topic,
        "limits_version": row.limits_version,
        "chart_type": row.chart_type,
        "ruleset": row.ruleset,
        "active": row.active,
        "meta": row.meta or {},
        "created_at": row.created_at.isoformat()
        if isinstance(row.created_at, datetime)
        else row.created_at,
    }


def _ooc_to_dict(row: OocEventRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "stream_key": row.stream_key,
        "limits_version": row.limits_version,
        "index": row.index,
        "value": row.value,
        "rule_id": row.rule_id,
        "rule_name": row.rule_name,
        "description": row.description,
        "side": row.side,
        "ts": row.ts.isoformat() if isinstance(row.ts, datetime) else row.ts,
        "acked": bool(row.acked),
        "acked_at": row.acked_at.isoformat()
        if isinstance(row.acked_at, datetime)
        else row.acked_at,
        "acked_by": row.acked_by,
    }
