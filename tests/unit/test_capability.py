"""Capability analysis — analytic DPMO/sigma, non-parametric path, flat contract keys."""
from __future__ import annotations

import numpy as np
import pytest

from spc_core.capability import (
    capability_analysis,
    dpmo_to_sigma,
    nonparametric_capability,
    parametric_capability,
    sigma_to_dpmo,
)
from spc_core.normality import check_normality
from spc_core.report import CapabilityReport


def test_dpmo_sigma_analytic_not_bucketed():
    """Legacy used buckets (691462→1σ etc). We use norm.ppf."""
    # ~3.4 DPMO at 6σ short-term (with 1.5 shift => Z_bench≈4.5)
    sigma = dpmo_to_sigma(3.4)
    assert sigma == pytest.approx(6.0, abs=0.1)
    # Round-trip
    z_bench = 3.0
    dpmo = sigma_to_dpmo(z_bench)
    assert dpmo == pytest.approx(1350, abs=50)


def test_parametric_capability_centered(dataset):
    cols = dataset("capability_excellent")
    values = cols["measurement"]
    # Specs wide enough for excellent data around 10
    result = parametric_capability(values, usl=10.5, lsl=9.5, target=10.0)
    assert result.method == "parametric"
    assert result.cp is not None and result.cpk is not None
    assert result.cpk > 1.0
    assert result.sigma_level is not None


def test_capability_report_exposes_flat_normality_keys(dataset):
    """Fixes the legacy tool/pipeline key mismatch (is_normal / shapiro_p / anderson_stat)."""
    cols = dataset("capability_excellent")
    values = [float(v) for v in cols["measurement"]]
    normality = check_normality(values)
    result = capability_analysis(values, usl=10.5, lsl=9.5)
    report = CapabilityReport.from_capability(result, normality=normality)
    d = report.model_dump()
    assert "is_normal" in d["normality"]
    assert "shapiro_p" in d["normality"]
    assert "anderson_stat" in d["normality"]
    # These are the keys the broken tool expected at the top level of normality
    assert isinstance(d["normality"]["is_normal"], bool)


def test_nonparametric_for_skewed(dataset):
    cols = dataset("capability_skewed_data")
    values = [float(v) for v in cols["measurement"]]
    # Force non-parametric
    result = nonparametric_capability(values, usl=max(values) * 1.2, lsl=min(values) * 0.8)
    assert result.method == "nonparametric"
    assert result.cp is None  # parametric indices withheld
    assert result.ppk is not None


def test_auto_routes_nonnormal_to_nonparametric_or_transformed():
    rng = np.random.default_rng(0)
    # Strongly right-skewed
    values = rng.exponential(2.0, 200)
    result = capability_analysis(values, usl=float(np.percentile(values, 99)),
                                 lsl=float(np.percentile(values, 1)))
    assert result.method in ("nonparametric", "parametric", "transformed")
    # If Shapiro fails (almost always for exponential), expect nonparametric OR
    # transformed (when Box-Cox/YJ restores normality and specs transform).
    if not check_normality(values).is_normal:
        assert result.method in ("nonparametric", "transformed")


def test_transform_spec_failure_recorded_in_notes(monkeypatch):
    """Spec transform failures must not silent-fallthrough without a reason."""
    from types import SimpleNamespace

    import spc_core.capability as cap_mod
    import spc_core.normality as norm_mod

    values = list(np.random.default_rng(0).normal(10, 1, 80))
    fake_tr = SimpleNamespace(
        became_normal=True,
        applied="LOG",
        label="log(x)",
        lam=None,
        values=np.asarray(values),
    )
    monkeypatch.setattr(norm_mod, "apply_transform", lambda *a, **k: fake_tr)
    monkeypatch.setattr(
        norm_mod, "check_normality", lambda *a, **k: SimpleNamespace(is_normal=False)
    )
    monkeypatch.setattr(
        cap_mod,
        "_transform_specs",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("bad specs")),
    )
    result = capability_analysis(values, usl=12.0, lsl=8.0)
    assert result.method == "nonparametric"
    notes = result.notes["normality"]
    assert notes["path"] == "nonparametric_after_transform_error"
    assert "bad specs" in notes["transform_error"]
