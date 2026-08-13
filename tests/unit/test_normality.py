"""Normality, autocorrelation gate, transforms."""
from __future__ import annotations

import numpy as np

from spc_core.normality import apply_transform, check_autocorrelation, check_normality


def test_normal_data_passes():
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, 200)
    result = check_normality(values)
    assert result.is_normal is True
    assert result.shapiro_p is not None
    assert result.anderson_stat is not None


def test_skewed_data_fails():
    rng = np.random.default_rng(0)
    values = rng.exponential(2, 200)
    result = check_normality(values)
    assert result.is_normal is False
    assert abs(result.skewness) > 1.0


def test_autocorrelation_gate():
    # Independent
    rng = np.random.default_rng(0)
    ind = rng.normal(0, 1, 100)
    r = check_autocorrelation(ind, threshold=0.2)
    assert r.is_autocorrelated is False

    # Strong AR(1)
    ar = [0.0]
    for _ in range(99):
        ar.append(0.9 * ar[-1] + rng.normal(0, 0.3))
    r2 = check_autocorrelation(ar, threshold=0.2)
    assert r2.is_autocorrelated is True
    assert abs(r2.lag1) > 0.2


def test_boxcox_transform_positive():
    rng = np.random.default_rng(0)
    values = rng.exponential(2, 100) + 0.1
    tr = apply_transform(values, method="boxcox")
    assert tr.applied == "BOXCOX"
    assert tr.lam is not None
    assert len(tr.values) == len(values)


def test_yeojohnson_for_negatives():
    values = np.array([-1.0, 0.0, 1.0, 2.0, -0.5, 3.0] * 20)
    tr = apply_transform(values, method="auto")
    assert tr.applied == "YEO-JOHNSON"
