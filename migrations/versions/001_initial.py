"""Initial ASPC schema — control limits, runs, audit, OOC, capability, raw, streams."""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "control_limits",
        sa.Column("version", sa.String(64), primary_key=True),
        sa.Column("chart_type", sa.String(32), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("meta", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "analysis_runs",
        sa.Column("run_id", sa.String(64), primary_key=True),
        sa.Column("analysis_type", sa.String(64), nullable=False),
        sa.Column("limits_version", sa.String(64), nullable=True),
        sa.Column("source_file", sa.Text(), nullable=True),
        sa.Column("user_id", sa.String(128), nullable=True),
        sa.Column("report", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_analysis_runs_analysis_type", "analysis_runs", ["analysis_type"])
    op.create_index("ix_analysis_runs_created_at", "analysis_runs", ["created_at"])

    op.create_table(
        "audit_log",
        sa.Column("event_id", sa.String(64), primary_key=True),
        sa.Column("event", sa.String(128), nullable=False),
        sa.Column("detail", sa.JSON(), nullable=False),
        sa.Column("user_id", sa.String(128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_audit_log_event", "audit_log", ["event"])

    op.create_table(
        "ooc_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("stream_key", sa.String(256), nullable=False),
        sa.Column("limits_version", sa.String(64), nullable=True),
        sa.Column("index", sa.Integer(), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("rule_id", sa.String(32), nullable=False),
        sa.Column("rule_name", sa.String(128), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("side", sa.String(16), nullable=True),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("acked", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("acked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("acked_by", sa.String(128), nullable=True),
        sa.UniqueConstraint("stream_key", "ts", "rule_id", name="uq_ooc_stream_ts_rule"),
    )
    op.create_index("ix_ooc_stream_ts", "ooc_events", ["stream_key", "ts"])

    op.create_table(
        "capability_history",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("cpk", sa.Float(), nullable=True),
        sa.Column("ppk", sa.Float(), nullable=True),
        sa.Column("sigma_level", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_capability_history_run_id", "capability_history", ["run_id"])

    op.create_table(
        "raw_measurements",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("stream_key", sa.String(256), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("quality_flag", sa.String(64), nullable=True),
        sa.Column("machine_id", sa.String(128), nullable=True),
        sa.Column("gage_id", sa.String(128), nullable=True),
        sa.Column("limits_version", sa.String(64), nullable=True),
        sa.PrimaryKeyConstraint("id", "ts", name="pk_raw_measurements"),
    )
    op.create_index("ix_raw_stream_ts", "raw_measurements", ["stream_key", "ts"])

    op.create_table(
        "stream_registry",
        sa.Column("stream_key", sa.String(256), primary_key=True),
        sa.Column("topic", sa.String(512), nullable=True),
        sa.Column("limits_version", sa.String(64), nullable=True),
        sa.Column("chart_type", sa.String(32), nullable=True),
        sa.Column("ruleset", sa.String(64), nullable=False, server_default="nelson"),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("meta", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    # Timescale-specific: only when running against PostgreSQL with the extension.
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
        op.execute(
            "SELECT create_hypertable('raw_measurements', 'ts', if_not_exists => TRUE)"
        )
        op.execute(
            """
            DO $$
            BEGIN
                PERFORM add_retention_policy(
                    'raw_measurements', INTERVAL '90 days', if_not_exists => TRUE
                );
            EXCEPTION WHEN OTHERS THEN
                NULL;
            END $$;
            """
        )


def downgrade() -> None:
    op.drop_table("stream_registry")
    op.drop_index("ix_raw_stream_ts", table_name="raw_measurements")
    op.drop_table("raw_measurements")
    op.drop_index("ix_capability_history_run_id", table_name="capability_history")
    op.drop_table("capability_history")
    op.drop_index("ix_ooc_stream_ts", table_name="ooc_events")
    op.drop_table("ooc_events")
    op.drop_index("ix_audit_log_event", table_name="audit_log")
    op.drop_table("audit_log")
    op.drop_index("ix_analysis_runs_created_at", table_name="analysis_runs")
    op.drop_index("ix_analysis_runs_analysis_type", table_name="analysis_runs")
    op.drop_table("analysis_runs")
    op.drop_table("control_limits")
