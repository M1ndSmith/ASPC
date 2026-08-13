"""Missing-value / range / reason generators."""
from __future__ import annotations

from typing import Any

import numpy as np

Columns = dict[str, list[Any]]


def _base(n: int = 40, seed: int = 201) -> list[float]:
    rng = np.random.default_rng(seed)
    return [float(v) for v in 100.0 + rng.normal(0.0, 1.0, n)]


def gap_short_locf(*, seed: int = 201) -> Columns:
    values: list[float | None] = _base(40, seed)
    values[10] = None
    values[11] = None
    return {"measurement": values, "reason": [None] * len(values)}


def gap_long_hold(*, seed: int = 202) -> Columns:
    values: list[float | None] = _base(40, seed)
    for i in range(10, 15):
        values[i] = None
    return {"measurement": values, "reason": [None] * len(values)}


def reason_maintenance(*, seed: int = 203) -> Columns:
    values: list[float | None] = _base(30, seed)
    reasons: list[str | None] = [None] * len(values)
    values[12] = None
    reasons[12] = "maintenance"
    return {"measurement": values, "reason": reasons}


def reason_human(*, seed: int = 204) -> Columns:
    values: list[float | None] = _base(30, seed)
    reasons: list[str | None] = [None] * len(values)
    values[8] = None
    reasons[8] = "human"
    return {"measurement": values, "reason": reasons}


def reason_backup(*, seed: int = 205) -> Columns:
    values: list[float | None] = _base(30, seed)
    reasons: list[str | None] = [None] * len(values)
    reasons[15] = "backup"
    return {"measurement": values, "reason": reasons}


def reason_incomplete(*, seed: int = 206) -> Columns:
    values: list[float | None] = _base(30, seed)
    reasons: list[str | None] = [None] * len(values)
    values[20] = None
    reasons[20] = "incomplete"
    return {"measurement": values, "reason": reasons}


def sensor_sentinel_range(*, seed: int = 207) -> Columns:
    values = _base(30, seed)
    values[5] = -999.0
    values[18] = -999.0
    return {"measurement": values}


def sensor_sentinel_long(*, n: int = 80, seed: int = 208) -> Columns:
    """Sentinels in a long in-control series, for the full pipeline rather than range_check.

    Longer than ``sensor_sentinel_range`` on purpose: at n=30, dropping two points moves
    the lag-1 ACF estimate enough to trip the autocorrelation gate and divert the case to
    EWMA, which would confound the thing being asserted (that a -999 sentinel never
    reaches the chart as an out-of-control signal).
    """
    values = _base(n, seed)
    values[17] = -999.0
    values[52] = -999.0
    return {"measurement": values}
