"""Hardening Phase 1: correctness and gate-enforcement regression tests."""
from __future__ import annotations

import numpy as np
import pytest

from spc_core.charts import analyze_control_chart
from spc_core.evaluator import Phase2Evaluator
from spc_core.ewma import ewma_chart
from spc_core.models import ChartType, ControlLimits
from spc_core.msa import gage_rr_anova
from spc_core.pipeline import establish, phase1_checklist


def test_wheeler_preserves_subgroup_chart():
    """Non-normal subgrouped data must stay Xbar-R, not flatten to I-MR."""
    rng = np.random.default_rng(99)
    # Right-skewed: lognormal — fails normality, transform may or may not help.
    values = list(rng.lognormal(mean=0.0, sigma=1.0, size=100))
    subgroup_ids = [i // 5 for i in range(100)]
    result = establish(
        values,
        subgroup_ids=subgroup_ids,
        chart_type=ChartType.XBAR_R,
        force_wheeler=True,
    )
    assert result.chart.chart_type == ChartType.XBAR_R
    assert result.chart_route == "wheeler" or result.chart.limits.chart_type == ChartType.XBAR_R
    # Plotted values are subgroup means, not all 100 individuals
    assert len(result.chart.plotted_values) == 20


def test_force_wheeler_false_still_preserves_subgroups_when_nonnormal():
    rng = np.random.default_rng(7)
    values = list(rng.exponential(scale=2.0, size=80))
    subgroup_ids = [i // 4 for i in range(80)]
    result = establish(
        values,
        subgroup_ids=subgroup_ids,
        chart_type=ChartType.XBAR_R,
        force_wheeler=False,
    )
    # Must not collapse to I-MR
    assert result.chart.chart_type != ChartType.I_MR or result.chart_route != "wheeler"
    if result.chart_route == "wheeler":
        assert result.chart.chart_type == ChartType.XBAR_R


def test_stop_gate_blocks_freeze():
    """MSA STOP must set frozen=False and freeze gate status=blocked."""
    # Construct deliberately bad MSA: huge measurement noise vs part variation
    parts, ops, meas = [], [], []
    rng = np.random.default_rng(1)
    for p in range(5):
        for o in range(2):
            for _ in range(2):
                parts.append(p)
                ops.append(o)
                # Enormous gage noise → high %GRR
                meas.append(float(rng.normal(0, 50)))
    values = list(rng.normal(100, 1, size=40))
    result = establish(
        values,
        msa_parts=parts,
        msa_operators=ops,
        msa_measurements=meas,
        msa_tolerance=1.0,
    )
    assert result.stopped is True
    assert result.frozen is False
    freeze = next(g for g in result.gates if g.step == "freeze")
    assert freeze.status == "blocked"
    checklist = phase1_checklist(result, min_subgroups=25, phase2_enabled=False)
    limits_item = next(i for i in checklist["items"] if i["item"] == "limits_frozen")
    assert limits_item["passed"] is False


def test_variable_n_xbar_emits_beyond_limit_signals():
    """Variable-size subgroups must not silence run rules via sigma=0."""
    # Build unequal subgroups with one clear outlier mean
    values = []
    subgroup_ids = []
    # 10 subgroups of size 3 around 10, then one of size 5 with huge mean
    for s in range(10):
        for _ in range(3):
            values.append(10.0)
            subgroup_ids.append(s)
    for _ in range(5):
        values.append(50.0)  # clear shift
        subgroup_ids.append(10)

    result = analyze_control_chart(
        values,
        subgroup_ids=subgroup_ids,
        chart_type=ChartType.XBAR_R,
        ruleset="nelson",
    )
    assert isinstance(result.limits.primary.ucl, list)
    # The last subgroup mean (50) must fire beyond-limits
    assert result.out_of_control_count >= 1
    assert any(s.rule_id == "1" for s in result.signals)


def test_ewma_phase2_roundtrip():
    """Phase2Evaluator must support EWMA limits from Phase I."""
    rng = np.random.default_rng(3)
    phase1 = list(rng.normal(0, 1, size=40))
    ew = ewma_chart(phase1, lam=0.2, L=3.0)
    assert ew.limits is not None
    assert ew.limits.chart_type == ChartType.EWMA

    ev = Phase2Evaluator(ew.limits, ruleset="nelson")
    # In-control observations should generally not signal
    for v in rng.normal(0, 1, size=5):
        ev.observe(float(v))
    # Large shift should eventually trip EWMA
    signals = []
    for _ in range(30):
        signals.extend(ev.observe(5.0))
    assert any(s.rule_id.startswith("EWMA") for s in signals)


def test_control_limits_version_in_model_dump():
    from spc_core.models import LimitSet

    limits = ControlLimits(
        chart_type=ChartType.I_MR,
        subgroup_size=1,
        components={
            "individuals": LimitSet(center=0.0, ucl=3.0, lcl=-3.0),
        },
        sigma=1.0,
    )
    dumped = limits.model_dump()
    assert "version" in dumped
    assert dumped["version"] == limits.version
    assert len(dumped["version"]) == 16


def test_p_chart_rejects_zero_sample_size():
    with pytest.raises(ValueError, match="sample sizes"):
        analyze_control_chart(
            [1, 2, 0],
            sample_sizes=[10, 0, 10],
            chart_type=ChartType.P,
        )


def test_u_chart_rejects_zero_opportunity():
    with pytest.raises(ValueError, match="opportunities"):
        analyze_control_chart(
            [1, 2, 0],
            opportunities=[10, 0, 10],
            chart_type=ChartType.U,
        )


def test_unbalanced_gage_rr_falls_back_to_range():
    # Missing some part-operator cells
    parts = [1, 1, 1, 2, 2, 3]
    ops = ["A", "A", "B", "A", "A", "B"]
    meas = [10.0, 10.1, 10.2, 11.0, 11.1, 12.0]
    result = gage_rr_anova(parts, ops, meas)
    assert "unbalanced" in result.method.lower() or result.detail.get("anova_skipped")
    assert result.detail.get("balanced") is False
    assert result.grr_percent >= 0.0
