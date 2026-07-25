"""Service-gated integration tests — skip when brokers/DB are unavailable."""
from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("ASPC_INTEGRATION") != "1",
        reason="Set ASPC_INTEGRATION=1 with compose stack up to run",
    ),
]


def test_timescale_roundtrip():
    dsn = os.environ.get("ASPC_TIMESCALE_DSN")
    if not dsn:
        pytest.skip("ASPC_TIMESCALE_DSN not set")
    from adapters.persistence_tsdb import TimescaleDBRepository

    repo = TimescaleDBRepository(dsn)
    ver = repo.save_limits({"chart_type": "I-MR"}, "testver", "I-MR")
    assert repo.get_limits(ver) is not None


def test_redis_publish_smoke():
    url = os.environ.get("ASPC_REDIS_URL", "redis://localhost:6379/0")
    redis = pytest.importorskip("redis")
    try:
        r = redis.Redis.from_url(url, socket_connect_timeout=1)
        r.ping()
    except Exception:
        pytest.skip("Redis not reachable")
    r.publish("spc:live:test", '{"value": 1.0}')
