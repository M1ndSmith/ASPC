"""SPC cleaning — missing-value classifier (never silent impute)."""
from __future__ import annotations

import math

from spc_core.cleaning import classify_missing, range_check
from spc_core.models import QualityFlag


def test_original_values_pass_through():
    result = classify_missing([1.0, 2.0, 3.0])
    assert all(f == QualityFlag.ORIGINAL for f in result.flags)
    assert all(result.usable)


def test_short_gap_locf():
    result = classify_missing([1.0, None, None, 4.0])
    assert result.flags[1] == QualityFlag.IMPUTED_LOCF
    assert result.flags[2] == QualityFlag.IMPUTED_LOCF
    assert result.values[1] == 1.0
    assert result.values[2] == 1.0
    assert result.usable[1] and result.usable[2]


def test_long_gap_held_for_investigation():
    result = classify_missing([1.0, None, None, None, None, 6.0], locf_max=3)
    assert result.flags[1] == QualityFlag.MISSING_SENSOR
    assert result.values[1] is None
    assert not result.usable[1]


def test_explicit_maintenance_reason():
    result = classify_missing(
        [1.0, None, 3.0],
        reasons=[None, "maintenance", None],
    )
    assert result.flags[1] == QualityFlag.EXCLUDED_MAINTENANCE
    assert not result.usable[1]


def test_nan_treated_as_missing():
    result = classify_missing([1.0, float("nan"), 3.0])
    assert result.flags[1] == QualityFlag.IMPUTED_LOCF


def test_range_check_sensor_failure():
    valid = range_check([10.0, -999.0, 11.0], low=0.0, high=100.0)
    assert valid == [True, False, True]
