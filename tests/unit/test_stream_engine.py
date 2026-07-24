"""Unit tests for StreamEngine OOC detection with a fake in-memory repository."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from adapters.stream_engine import StreamEngine
from spc_core.limits import imr_limits


class FakeStreamRepo:
    """In-memory stand-in implementing the StreamRepository protocol."""

    def __init__(self):
        self.raw: list[dict[str, Any]] = []
        self.ooc: list[dict[str, Any]] = []

    def save_raw_measurement(
        self, stream_key: str, ts: datetime, value: float, **meta: Any
    ) -> None:
        self.raw.append(
            {"stream_key": stream_key, "ts": ts, "value": float(value), **meta}
        )

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
        key = (stream_key, ts.isoformat(), rule_id)
        if any(
            (e["stream_key"], e["ts"].isoformat(), e["rule_id"]) == key for e in self.ooc
        ):
            return False
        self.ooc.append(
            {
                "stream_key": stream_key,
                "ts": ts,
                "limits_version": limits_version,
                "index": index,
                "value": float(value),
                "rule_id": rule_id,
                "rule_name": rule_name,
                "description": description,
                "side": side,
            }
        )
        return True


def test_handle_observation_detects_ooc_and_persists():
    limits = imr_limits([10.0, 10.1, 9.9, 10.05, 10.0, 10.02, 9.98, 10.01] * 4)
    repo = FakeStreamRepo()
    engine = StreamEngine(repo)
    engine.register("line-a", limits, ruleset="nelson")

    ts = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    # In-control point
    signals = engine.handle_observation("line-a", 10.0, ts)
    assert signals == []
    assert len(repo.raw) == 1
    assert repo.raw[0]["limits_version"] == limits.version
    assert repo.ooc == []

    # Beyond UCL — must fire rule 1
    ooc_ts = datetime(2026, 1, 15, 12, 0, 1, tzinfo=timezone.utc)
    spike = float(limits.primary.ucl) + 10.0
    signals = engine.handle_observation("line-a", spike, ooc_ts)
    assert any(s.rule_id == "1" for s in signals)
    assert len(repo.raw) == 2
    assert len(repo.ooc) >= 1
    assert all(e["limits_version"] == limits.version for e in repo.ooc)
    assert any(e["rule_id"] == "1" and e["value"] == spike for e in repo.ooc)


def test_ooc_writes_are_idempotent():
    limits = imr_limits([10.0, 10.1, 9.9, 10.05, 10.0, 10.02, 9.98, 10.01] * 4)
    repo = FakeStreamRepo()
    engine = StreamEngine(repo)
    engine.register("line-b", limits)

    ts = datetime(2026, 1, 15, 13, 0, 0, tzinfo=timezone.utc)
    spike = float(limits.primary.ucl) + 5.0
    s1 = engine.handle_observation("line-b", spike, ts)
    n_ooc = len(repo.ooc)
    assert n_ooc >= 1
    assert any(s.rule_id == "1" for s in s1)

    # Same stream/ts/rule again via direct repo call must not duplicate
    first = repo.ooc[0]
    inserted = repo.save_ooc_event(
        first["stream_key"],
        first["ts"],
        limits_version=first["limits_version"],
        index=first["index"],
        value=first["value"],
        rule_id=first["rule_id"],
        rule_name=first["rule_name"],
        description=first["description"],
        side=first["side"],
    )
    assert inserted is False
    assert len(repo.ooc) == n_ooc


def test_unregistered_stream_raises():
    repo = FakeStreamRepo()
    engine = StreamEngine(repo)
    try:
        engine.handle_observation("missing", 1.0)
        assert False, "expected KeyError"
    except KeyError:
        pass
