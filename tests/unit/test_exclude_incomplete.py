"""exclude_incomplete must drop incomplete subgroups from plotted values."""
from __future__ import annotations

from spc_core import ChartType, analyze_control_chart


def test_exclude_incomplete_drops_ragged_subgroup():
    measurements = []
    subgroups = []
    for sid in range(1, 25):
        for _ in range(5):
            measurements.append(50.0)
            subgroups.append(sid)
    # Ragged last subgroup of size 2.
    measurements.extend([50.0, 51.0])
    subgroups.extend([25, 25])

    keep = analyze_control_chart(
        measurements,
        subgroup_ids=subgroups,
        chart_type=ChartType.XBAR_R,
        exclude_incomplete=False,
    )
    drop = analyze_control_chart(
        measurements,
        subgroup_ids=subgroups,
        chart_type=ChartType.XBAR_R,
        exclude_incomplete=True,
    )
    assert len(keep.plotted_values) == 25
    assert len(drop.plotted_values) == 24
