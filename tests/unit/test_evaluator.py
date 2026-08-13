"""Phase I / Phase II evaluator — frozen limits, incremental observation."""
from __future__ import annotations

import numpy as np
import pytest

from spc_core.evaluator import Phase2Evaluator, evaluate_batch
from spc_core.limits import imr_limits
from spc_core.models import ChartType


def test_phase2_does_not_recompute_limits():
    rng = np.random.default_rng(0)
    phase1 = 100 + rng.normal(0, 1, 40)
    limits = imr_limits(phase1)
    version = limits.version

    ev = Phase2Evaluator(limits)
    # Stream Phase II points — limits object must stay the same version
    for v in (100 + rng.normal(0, 1, 20)).tolist():
        ev.observe(float(v))
    assert limits.version == version
    assert ev.limits is limits


def test_phase2_detects_ooc():
    limits = imr_limits([10.0, 10.1, 9.9, 10.05, 10.0, 10.02, 9.98, 10.01] * 4)
    ev = Phase2Evaluator(limits)
    # A huge spike must trigger rule 1
    signals = ev.observe(limits.primary.ucl + 10)
    assert any(s.rule_id == "1" for s in signals)


def test_evaluate_batch():
    limits = imr_limits([1.0, 1.1, 0.9, 1.05, 1.0, 0.95, 1.02, 0.98] * 3)
    plotted = [1.0] * 5 + [limits.primary.ucl + 5]
    signals = evaluate_batch(limits, plotted)
    assert any(s.rule_id == "1" for s in signals)


def test_subgroup_chart_requires_observe_subgroup():
    from spc_core.limits import xbar_r_limits
    subs = [np.array([1.0, 1.1, 0.9, 1.05, 1.0]) for _ in range(20)]
    limits = xbar_r_limits(subs)
    assert limits.chart_type == ChartType.XBAR_R
    ev = Phase2Evaluator(limits)
    with pytest.raises(ValueError):
        ev.observe(1.0)
    signals = ev.observe_subgroup([1.0, 1.0, 1.0, 1.0, 1.0])
    assert isinstance(signals, list)
