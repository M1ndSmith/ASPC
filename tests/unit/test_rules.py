"""Stateful Nelson / Western Electric run-rule engine."""
from __future__ import annotations

import pytest

from spc_core.rules import RuleEngine, evaluate_series


def test_rule1_beyond_3sigma():
    engine = RuleEngine(center=0.0, sigma=1.0, ruleset="nelson")
    # Points inside: no signal
    for v in [0.0, 0.5, -0.5, 1.0]:
        assert engine.add(v) == []
    # Beyond 3 sigma
    signals = engine.add(3.5)
    assert any(s.rule_id == "1" for s in signals)
    assert signals[0].side == "upper"


def test_rule2_nine_on_one_side():
    engine = RuleEngine(center=0.0, sigma=1.0)
    signals = []
    for _ in range(8):
        signals.extend(engine.add(0.5))  # all above center, within 1 sigma
    assert not any(s.rule_id == "2" for s in signals)
    signals = engine.add(0.5)  # 9th
    assert any(s.rule_id == "2" for s in signals)


def test_rule3_trend():
    engine = RuleEngine(center=10.0, sigma=2.0)
    signals = []
    for v in [1, 2, 3, 4, 5, 6]:
        signals.extend(engine.add(float(v)))
    assert any(s.rule_id == "3" for s in signals)


def test_western_electric_rule4_eight_on_side():
    engine = RuleEngine(center=0.0, sigma=1.0, ruleset="western_electric")
    for _ in range(7):
        assert not any(s.rule_id == "WE4" for s in engine.add(-0.3))
    signals = engine.add(-0.3)
    assert any(s.rule_id == "WE4" for s in signals)


def test_evaluate_series_batch_matches_incremental():
    values = [0.1, 0.2, -0.1, 0.0, 0.3, 3.5, 0.1]
    batch = evaluate_series(values, center=0.0, sigma=1.0)
    engine = RuleEngine(center=0.0, sigma=1.0)
    incr = []
    for v in values:
        incr.extend(engine.add(v))
    assert [s.rule_id for s in batch] == [s.rule_id for s in incr]
    assert [s.index for s in batch] == [s.index for s in incr]


def test_rule5_two_of_three_beyond_2sigma():
    engine = RuleEngine(center=0.0, sigma=1.0)
    engine.add(0.0)
    engine.add(2.5)  # beyond 2σ
    signals = engine.add(2.6)  # 2 of last 3 beyond 2σ
    assert any(s.rule_id == "5" for s in signals)
