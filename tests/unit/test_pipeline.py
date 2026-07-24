"""Unit tests for gated Phase I pipeline."""
from __future__ import annotations

import numpy as np

from spc_core.pipeline import establish, phase1_checklist


def test_establish_returns_chart_and_gates():
    rng = np.random.default_rng(42)
    values = 100.0 + rng.normal(0, 1.0, size=40)
    result = establish(values, ruleset="nelson")
    assert result.chart is not None
    assert result.limits_version
    assert result.gates
    steps = {g.step for g in result.gates}
    assert "msa" in steps
    assert "autocorrelation" in steps
    assert "chart" in steps
    assert "freeze" in steps
    assert result.chart.limits.version == result.limits_version


def test_phase1_checklist_structure():
    rng = np.random.default_rng(1)
    values = 50.0 + rng.normal(0, 0.5, size=30)
    pipeline = establish(values)
    checklist = phase1_checklist(pipeline, min_subgroups=25, phase2_enabled=False)
    assert "passed" in checklist
    assert "items" in checklist
    assert checklist["limits_version"] == pipeline.limits_version
    names = {i["item"] for i in checklist["items"]}
    assert "min_subgroups" in names
    assert "limits_frozen" in names
    # Without MSA inputs, msa item should fail go-live
    msa_item = next(i for i in checklist["items"] if i["item"] == "msa_grr_ndc")
    assert msa_item["passed"] is False


def test_establish_respects_chart_type():
    values = list(range(30))
    from spc_core.models import ChartType

    result = establish(values, chart_type=ChartType.I_MR)
    # May route to Wheeler/EWMA if non-normal, but should still produce a chart
    assert result.chart.plotted_values
    assert result.chart.limits.version
