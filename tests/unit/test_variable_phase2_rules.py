"""Unit tests for Phase II variable-limit rule extensions."""
from __future__ import annotations

from spc_core.evaluator import Phase2Evaluator
from spc_core.models import ChartType, ControlLimits, LimitSet


def test_variable_limits_fires_zone_rules():
    limits = ControlLimits(
        chart_type=ChartType.P,
        subgroup_size=1,
        components={
            "p": LimitSet(
                center=0.1,
                ucl=[0.4, 0.4, 0.4, 0.4],
                lcl=[0.0, 0.0, 0.0, 0.0],
            )
        },
        sigma=0.1,
    )
    ev = Phase2Evaluator(limits, ruleset="nelson")
    # Two points beyond 2σ (~0.1 + 2*0.1 = 0.3) on the upper side.
    s1 = ev.observe(0.35)
    s2 = ev.observe(0.36)
    rule_ids = {s.rule_id for s in s1 + s2}
    assert "5" in rule_ids or "1" in rule_ids
