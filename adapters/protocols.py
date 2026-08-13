"""Typed repository protocols for streaming and API capability checks."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StreamRepository(Protocol):
    """Minimal persistence surface required by :class:`StreamEngine`."""

    def save_raw_measurement(
        self, stream_key: str, ts: datetime, value: float, **meta: Any
    ) -> None: ...

    def save_ooc_event(
        self,
        stream_key: str,
        ts: datetime,
        *,
        limits_version: str | None,
        index: int,
        value: float,
        rule_id: str,
        rule_name: str,
        description: str,
        side: str | None = None,
    ) -> bool: ...

    def get_limits(self, version: str) -> dict[str, Any] | None: ...


@runtime_checkable
class StreamingOpsRepository(Protocol):
    """Timescale-backed ops used by stream registry / alerts API routes."""

    def register_stream(
        self,
        stream_key: str,
        *,
        topic: str | None = None,
        limits_version: str | None = None,
        chart_type: str | None = None,
        ruleset: str = "nelson",
        active: bool = True,
        meta: dict[str, Any] | None = None,
    ) -> str: ...

    def list_streams(self, active_only: bool = False) -> list[dict[str, Any]]: ...

    def get_stream(self, stream_key: str) -> dict[str, Any] | None: ...

    def ack_alert(self, event_id: int, *, acked_by: str | None = None) -> dict[str, Any] | None: ...

    def get_limits(self, version: str) -> dict[str, Any] | None: ...

    def save_audit(
        self, event: str, detail: dict[str, Any], *, user_id: str | None = None
    ) -> str: ...
