"""Add tenant_id scaffolding, ooc acked index, stream measurement_count watermark.

Revision ID: 002_tenant_watermark
Revises: 001_initial
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "002_tenant_watermark"
down_revision = "001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("control_limits", sa.Column("tenant_id", sa.String(length=128), nullable=True))
    op.create_index("ix_control_limits_tenant_id", "control_limits", ["tenant_id"])

    op.add_column("analysis_runs", sa.Column("tenant_id", sa.String(length=128), nullable=True))
    op.create_index("ix_analysis_runs_tenant_id", "analysis_runs", ["tenant_id"])

    op.add_column("ooc_events", sa.Column("tenant_id", sa.String(length=128), nullable=True))
    op.create_index("ix_ooc_events_tenant_id", "ooc_events", ["tenant_id"])
    op.create_index("ix_ooc_acked", "ooc_events", ["acked"])

    op.add_column("raw_measurements", sa.Column("tenant_id", sa.String(length=128), nullable=True))
    op.create_index("ix_raw_measurements_tenant_id", "raw_measurements", ["tenant_id"])

    op.add_column(
        "stream_registry",
        sa.Column("measurement_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column("stream_registry", sa.Column("tenant_id", sa.String(length=128), nullable=True))
    op.create_index("ix_stream_registry_tenant_id", "stream_registry", ["tenant_id"])


def downgrade() -> None:
    op.drop_index("ix_stream_registry_tenant_id", table_name="stream_registry")
    op.drop_column("stream_registry", "tenant_id")
    op.drop_column("stream_registry", "measurement_count")

    op.drop_index("ix_raw_measurements_tenant_id", table_name="raw_measurements")
    op.drop_column("raw_measurements", "tenant_id")

    op.drop_index("ix_ooc_acked", table_name="ooc_events")
    op.drop_index("ix_ooc_events_tenant_id", table_name="ooc_events")
    op.drop_column("ooc_events", "tenant_id")

    op.drop_index("ix_analysis_runs_tenant_id", table_name="analysis_runs")
    op.drop_column("analysis_runs", "tenant_id")

    op.drop_index("ix_control_limits_tenant_id", table_name="control_limits")
    op.drop_column("control_limits", "tenant_id")
