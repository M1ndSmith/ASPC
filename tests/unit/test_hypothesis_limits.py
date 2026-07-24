"""Hypothesis property tests for limit/rule invariants."""
from __future__ import annotations

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings
from hypothesis import strategies as st

from spc_core.charts import analyze_control_chart
from spc_core.constants import A2, D3, D4, c4, d2
from spc_core.rules import RuleEngine


@given(st.integers(min_value=2, max_value=40))
@settings(max_examples=30)
def test_shewhart_constants_positive(n):
    assert d2(n) > 0
    assert c4(n) > 0
    assert A2(n) > 0
    assert D4(n) > D3(n) >= 0


@given(st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False), min_size=10, max_size=80))
@settings(max_examples=20)
def test_imr_limits_bracket_center(values):
    # Need some spread
    if np.std(values) < 1e-9:
        return
    result = analyze_control_chart(values)
    primary = result.limits.primary
    assert primary.lcl <= primary.center <= primary.ucl


@given(st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False))
@settings(max_examples=20)
def test_point_on_ucl_fires_rule1(center_offset):
    center = 0.0
    sigma = 1.0
    eng = RuleEngine(center=center, sigma=sigma, ruleset="nelson")
    # Exactly 3 sigma above center
    signals = eng.add(center + 3.0 * sigma)
    assert any(s.rule_id == "1" for s in signals)
