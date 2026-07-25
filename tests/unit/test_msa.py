"""MSA — Gage R&R exposes grr_percent at the top level (legacy key-mismatch fix)."""
from __future__ import annotations

from spc_core.msa import bias_study, gage_rr_anova, gage_rr_range, linearity_study, stability_study
from spc_core.report import MSAReport


def test_gage_rr_anova_excellent(dataset):
    cols = dataset("msa_gage_rr_excellent")
    result = gage_rr_anova(cols["Part"], cols["Operator"], cols["Measurement"])
    # Flat key — this is what the broken msa_tools expected
    assert hasattr(result, "grr_percent")
    assert isinstance(result.grr_percent, float)
    assert result.ndc >= 0
    assert result.acceptability in ("Excellent", "Acceptable", "Unacceptable")
    # Excellent fixture should have low GRR
    assert result.grr_percent < 30


def test_gage_rr_report_has_grr_percent(dataset):
    cols = dataset("msa_gage_rr_excellent")
    result = gage_rr_anova(cols["Part"], cols["Operator"], cols["Measurement"])
    report = MSAReport.from_gage_rr(result)
    d = report.model_dump()
    assert "grr_percent" in d["result"]
    assert d["result"]["grr_percent"] == result.grr_percent


def test_gage_rr_poor(dataset):
    cols = dataset("msa_gage_rr_poor")
    result = gage_rr_anova(cols["Part"], cols["Operator"], cols["Measurement"])
    assert isinstance(result.grr_percent, float)
    assert result.grr_percent >= 10.0  # poor fixture is intentionally high GRR
    assert result.acceptability in ("Acceptable", "Unacceptable")


def test_gage_rr_range_method(dataset):
    cols = dataset("msa_gage_rr_excellent")
    result = gage_rr_range(cols["Part"], cols["Operator"], cols["Measurement"])
    assert result.method == "Range"
    assert "grr_percent" in result.__dataclass_fields__


def test_bias_study(dataset):
    cols = dataset("msa_bias_study")
    result = bias_study(cols["Measurement"], cols["Reference"])
    assert result.n > 0
    assert isinstance(result.is_significant, bool)


def test_linearity_study(dataset):
    cols = dataset("msa_linearity_study")
    result = linearity_study(cols["Measurement"], cols["Reference"])
    assert isinstance(result.r_squared, float)


def test_stability_study(dataset):
    cols = dataset("msa_stability_study")
    result = stability_study(cols["Measurement"])
    assert result.ucl > result.lcl
    assert isinstance(result.is_stable, bool)
