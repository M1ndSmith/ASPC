"""Input guards and data-type routing in the Phase I pipeline.

These cover failure modes that the resilience catalog previously accepted as "handled"
because it only asserted the exception *type*: a domain guard and an internal crash both
surface as ValueError.
"""
from __future__ import annotations

import numpy as np
import pytest

from spc_core import ChartType, establish


def test_empty_input_raises_domain_error():
    with pytest.raises(ValueError, match="at least 2 observations"):
        establish([])


def test_single_point_raises_domain_error():
    with pytest.raises(ValueError, match="at least 2 observations"):
        establish([42.0])


def test_all_missing_raises_domain_error():
    """Previously crashed inside scipy.stats.boxcox with 'not enough values to unpack'."""
    with pytest.raises(ValueError, match="usable observations"):
        establish([None, None, None, None])


def test_attribute_chart_skips_distribution_gates():
    """Count data is discrete and binomial/Poisson-distributed.

    Applying a Gaussian normality test or Hartigan's dip test to it is invalid — the
    tie-heavy ECDF of a handful of distinct integers reads as multimodal and used to
    STOP in-control attribute studies.
    """
    rng = np.random.default_rng(7)
    defects = [int(v) for v in rng.poisson(3.0, 25)]

    pipe = establish(defects, chart_type=ChartType.C)

    assert pipe.stopped is False
    assert pipe.frozen is True
    assert pipe.gate_status("multimodal") is None
    normality = next(g for g in pipe.gates if g.step == "normality")
    assert normality.status == "ok"
    assert normality.detail.get("skipped") is True


def test_out_of_range_sentinel_is_not_an_ooc_signal():
    """A -999 disconnected-sensor reading is a measurement failure, not process variation.

    Left in the data it fires Nelson rule 1 and drags the distribution non-normal,
    diverting the whole study to the Wheeler route.
    """
    rng = np.random.default_rng(11)
    values = [float(v) for v in 100.0 + rng.normal(0.0, 1.0, 80)]
    values[17] = -999.0
    values[52] = -999.0

    unguarded = establish(values)
    assert "1" in {s.rule_id for s in unguarded.chart.signals}

    guarded = establish(values, valid_range=(0.0, 200.0))
    assert "1" not in {s.rule_id for s in guarded.chart.signals}
    assert guarded.gate_status("range") == "warn"
    assert guarded.gate_status("missing") == "warn"
    assert guarded.chart_route == "shewhart"
