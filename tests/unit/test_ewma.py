"""Unit tests for EWMA control chart."""
from __future__ import annotations

import numpy as np
import pytest

from spc_core.ewma import ewma_chart


def test_ewma_basic_in_control():
    rng = np.random.default_rng(0)
    values = 10.0 + rng.normal(0, 0.1, size=40)
    result = ewma_chart(values, lam=0.2, L=3.0)
    assert result.lam == 0.2
    assert result.L == 3.0
    assert len(result.z) == 40
    assert len(result.ucl) == 40
    assert len(result.lcl) == 40
    assert result.limits is not None
    assert result.limits.chart_type.value == "EWMA"
    assert result.limits.version


def test_ewma_detects_shift():
    # Stable then large mean shift — EWMA should fire
    values = [10.0] * 20 + [12.0] * 20
    result = ewma_chart(values, lam=0.2, L=3.0, target=10.0, sigma=0.2)
    assert any(s.side == "upper" for s in result.signals)
    assert result.z[-1] > result.z[0]


def test_ewma_rejects_bad_lambda():
    with pytest.raises(ValueError):
        ewma_chart([1.0, 2.0, 3.0], lam=0.0)
    with pytest.raises(ValueError):
        ewma_chart([1.0, 2.0, 3.0], lam=1.5)


def test_ewma_requires_two_points():
    with pytest.raises(ValueError):
        ewma_chart([1.0])
