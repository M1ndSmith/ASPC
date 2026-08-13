"""Capability generators."""
from __future__ import annotations

from typing import Any

import numpy as np

Columns = dict[str, list[Any]]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def capability(*, kind: str = "excellent", n: int = 200, seed: int = 401) -> Columns:
    """Capability datasets.

    Specs used by the suite: USL=10.5, LSL=9.5 unless noted otherwise.
    """
    rng = _rng(seed)
    if kind == "excellent":
        values = 10.0 + rng.normal(0.0, 0.10, n)
    elif kind == "off_center":
        values = 10.25 + rng.normal(0.0, 0.12, n)
    elif kind == "high_variation":
        values = 10.0 + rng.normal(0.0, 0.45, n)
    elif kind == "skewed":
        values = rng.exponential(2.0, n) + 5.0
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return {"measurement": [float(v) for v in values]}
