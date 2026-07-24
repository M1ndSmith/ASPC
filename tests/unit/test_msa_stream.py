"""Unit tests for continuous / streaming MSA."""
from __future__ import annotations

from spc_core.msa_stream import ContinuousMSA


def test_continuous_msa_healthy_when_on_target():
    msa = ContinuousMSA(tolerance=1.0, alpha=0.2, alert_fraction=0.10)
    for _ in range(10):
        alert = msa.observe_reference(measured=10.01, reference=10.0)
        assert alert is None
    assert msa.state.gage_healthy is True
    summary = msa.summary()
    assert summary["n_references"] == 10
    assert summary["n_alerts"] == 0
    assert abs(summary["ewma_bias"]) < summary["threshold"]


def test_continuous_msa_alerts_on_drift():
    msa = ContinuousMSA(tolerance=1.0, alpha=0.5, alert_fraction=0.10)
    # Bias of 0.5 exceeds 10% of tolerance (0.1)
    alert = None
    for _ in range(5):
        alert = msa.observe_reference(measured=10.5, reference=10.0)
    assert alert is not None
    assert "Calibration alert" in alert.message
    assert msa.state.gage_healthy is False
    assert msa.summary()["n_alerts"] >= 1


def test_rolling_r_chart_after_refs():
    msa = ContinuousMSA(tolerance=2.0)
    for i in range(5):
        msa.observe_reference(measured=5.0 + 0.01 * i, reference=5.0)
    chart = msa.rolling_r_chart()
    assert chart["n"] >= 2
    assert chart["r_bar"] >= 0.0
    assert "ucl" in chart
