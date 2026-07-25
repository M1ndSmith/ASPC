"""SQLAlchemy 2.0 declarative models for ASPC persistence (SQLite + TimescaleDB)."""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    PrimaryKeyConstraint,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Shared declarative base for all ASPC tables."""


class ControlLimitRow(Base):
    __tablename__ = "control_limits"

    version: Mapped[str] = mapped_column(String(64), primary_key=True)
    chart_type: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    meta: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class AnalysisRunRow(Base):
    __tablename__ = "analysis_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    analysis_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    limits_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_file: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    report: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, index=True
    )


class AuditLogRow(Base):
    __tablename__ = "audit_log"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    event: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    detail: Mapped[dict] = mapped_column(JSON, nullable=False)
    user_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class OocEventRow(Base):
    __tablename__ = "ooc_events"
    __table_args__ = (
        UniqueConstraint("stream_key", "ts", "rule_id", name="uq_ooc_stream_ts_rule"),
        Index("ix_ooc_stream_ts", "stream_key", "ts"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stream_key: Mapped[str] = mapped_column(String(256), nullable=False)
    limits_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    index: Mapped[int] = mapped_column(Integer, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    rule_id: Mapped[str] = mapped_column(String(32), nullable=False)
    rule_name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    side: Mapped[str | None] = mapped_column(String(16), nullable=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    acked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    acked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    acked_by: Mapped[str | None] = mapped_column(String(128), nullable=True)


class CapabilityHistoryRow(Base):
    __tablename__ = "capability_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    cpk: Mapped[float | None] = mapped_column(Float, nullable=True)
    ppk: Mapped[float | None] = mapped_column(Float, nullable=True)
    sigma_level: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class RawMeasurementRow(Base):
    """Hot-path observations; becomes a Timescale hypertable on PostgreSQL.

    ``id`` is assigned in application code (not DB autoincrement) so the composite
    primary key ``(id, ts)`` works on both SQLite and TimescaleDB (hypertables
    require the partition column ``ts`` in every unique constraint).
    """

    __tablename__ = "raw_measurements"
    __table_args__ = (
        PrimaryKeyConstraint("id", "ts", name="pk_raw_measurements"),
        Index("ix_raw_stream_ts", "stream_key", "ts"),
    )

    id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    stream_key: Mapped[str] = mapped_column(String(256), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    quality_flag: Mapped[str | None] = mapped_column(String(64), nullable=True)
    machine_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    gage_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    limits_version: Mapped[str | None] = mapped_column(String(64), nullable=True)


class StreamRegistryRow(Base):
    __tablename__ = "stream_registry"

    stream_key: Mapped[str] = mapped_column(String(256), primary_key=True)
    topic: Mapped[str | None] = mapped_column(String(512), nullable=True)
    limits_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    chart_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    ruleset: Mapped[str] = mapped_column(String(64), nullable=False, default="nelson")
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    meta: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


# Canonical names requested by the platform schema (aliases of *Row models).
ControlLimit = ControlLimitRow
AnalysisRun = AnalysisRunRow
AuditLog = AuditLogRow
OocEvent = OocEventRow
CapabilityHistory = CapabilityHistoryRow
RawMeasurement = RawMeasurementRow
StreamRegistration = StreamRegistryRow
