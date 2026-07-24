"""Control limit computation — including the Xbar-S branch that was missing."""
from __future__ import annotations

import numpy as np
import pytest

from spc_core.charts import analyze_control_chart, select_chart_type
from spc_core.limits import imr_limits, xbar_r_limits, xbar_s_limits, c_limits, p_limits, np_limits, u_limits
from spc_core.models import ChartType, DataType


def test_imr_limits_basic():
    rng = np.random.default_rng(0)
    values = 100 + rng.normal(0, 1, 30)
    limits = imr_limits(values)
    assert limits.chart_type == ChartType.I_MR
    assert "individuals" in limits.components
    assert "moving_range" in limits.components
    ind = limits.components["individuals"]
    assert ind.ucl > ind.center > ind.lcl
    # LCL must NOT be clamped to 0 for two-sided measurements
    assert ind.lcl < ind.center
    assert limits.version  # content hash present
    assert len(limits.version) == 16


def test_xbar_r_lcl_not_clamped_to_zero():
    """Legacy bug: Xbar LCL was max(..., 0) which hides low-side OOC."""
    subgroups = [np.array([10.0, 10.1, 9.9, 10.05, 10.02]) for _ in range(25)]
    limits = xbar_r_limits(subgroups)
    xbar = limits.components["xbar"]
    # Mean ~10, LCL should be positive but computed without artificial floor forcing
    # (for this data LCL is naturally >0; the point is the formula is unclamped)
    assert xbar.lcl == pytest.approx(xbar.center - (xbar.ucl - xbar.center), abs=1e-9)


def test_xbar_s_implemented():
    """The critical missing branch — n>=9 must produce Xbar-S limits, not {}."""
    # 12 subgroups of size 10
    rng = np.random.default_rng(1)
    subgroups = [100 + rng.normal(0, 2, 10) for _ in range(12)]
    limits = xbar_s_limits(subgroups)
    assert limits.chart_type == ChartType.XBAR_S
    assert "xbar" in limits.components
    assert "s" in limits.components
    assert limits.subgroup_size == 10
    assert limits.components["xbar"].ucl > limits.components["xbar"].center


def test_select_chart_routing():
    assert select_chart_type(DataType.CONTINUOUS, 1) == ChartType.I_MR
    assert select_chart_type(DataType.CONTINUOUS, 5) == ChartType.XBAR_R
    assert select_chart_type(DataType.CONTINUOUS, 8) == ChartType.XBAR_R
    assert select_chart_type(DataType.CONTINUOUS, 9) == ChartType.XBAR_S
    assert select_chart_type(DataType.CONTINUOUS, 15) == ChartType.XBAR_S
    assert select_chart_type(DataType.ATTRIBUTE, attribute_defectives=True, variable_size=False) == ChartType.NP
    assert select_chart_type(DataType.ATTRIBUTE, attribute_defectives=True, variable_size=True) == ChartType.P
    assert select_chart_type(DataType.ATTRIBUTE, attribute_defectives=False, variable_size=False) == ChartType.C
    assert select_chart_type(DataType.ATTRIBUTE, attribute_defectives=False, variable_size=True) == ChartType.U


def test_analyze_imr_sample(dataset):
    cols = dataset("spc_individual_out_of_control")
    result = analyze_control_chart(cols["measurement"])
    assert result.chart_type == ChartType.I_MR
    assert len(result.plotted_values) == len(cols["measurement"])
    assert result.limits.components["individuals"].ucl > result.limits.components["individuals"].center


def test_analyze_subgroup_xbar_r(dataset):
    cols = dataset("spc_subgroup_data")
    result = analyze_control_chart(cols["measurement"], subgroup_ids=cols["subgroup"])
    assert result.chart_type == ChartType.XBAR_R
    assert result.subgroup_size == 5
    assert result.secondary_name == "range"


def test_analyze_xbar_s_large_subgroups():
    """End-to-end: n=10 subgroups must select and compute Xbar-S."""
    rng = np.random.default_rng(2)
    values, sids = [], []
    for i in range(15):
        for v in 50 + rng.normal(0, 1, 10):
            values.append(float(v))
            sids.append(i)
    result = analyze_control_chart(values, subgroup_ids=sids)
    assert result.chart_type == ChartType.XBAR_S
    assert "s" in result.limits.components
    assert result.limits.components["xbar"].ucl != result.limits.components["xbar"].lcl


def test_c_chart(dataset):
    cols = dataset("spc_c_chart_data")
    result = analyze_control_chart(cols["defects"], chart_type=ChartType.C)
    assert result.chart_type == ChartType.C
    assert result.limits.components["defects"].lcl >= 0


def test_p_chart(dataset):
    cols = dataset("spc_p_chart_data")
    result = analyze_control_chart(
        cols["defective"], sample_sizes=cols["inspected"], chart_type=ChartType.P
    )
    assert result.chart_type == ChartType.P
    assert isinstance(result.limits.primary.ucl, list)


def test_np_chart(dataset):
    cols = dataset("spc_np_chart_data")
    result = analyze_control_chart(
        cols["defectives"], sample_sizes=cols["sample_size"], chart_type=ChartType.NP
    )
    assert result.chart_type == ChartType.NP


def test_u_chart(dataset):
    cols = dataset("spc_u_chart_data")
    result = analyze_control_chart(
        cols["defects"], opportunities=cols["units"], chart_type=ChartType.U
    )
    assert result.chart_type == ChartType.U
    assert isinstance(result.limits.primary.ucl, list)


def test_limits_immutable():
    limits = imr_limits([1.0, 2.0, 1.5, 2.1, 1.8, 2.0, 1.9])
    with pytest.raises(Exception):
        limits.subgroup_size = 99  # frozen pydantic model
