"""Golden-value tests for Shewhart chart constants."""
from __future__ import annotations

import math

import pytest

from spc_core import constants as k


def test_d2_known_values():
    assert k.d2(2) == pytest.approx(1.128, abs=0.001)
    assert k.d2(5) == pytest.approx(2.326, abs=0.001)
    assert k.d2(10) == pytest.approx(3.078, abs=0.001)


def test_c4_formula():
    # c4(2) = sqrt(2/pi) ≈ 0.7979
    assert k.c4(2) == pytest.approx(math.sqrt(2 / math.pi), abs=1e-6)
    # c4(5) ≈ 0.9400
    assert k.c4(5) == pytest.approx(0.9400, abs=0.001)


def test_A2_A3_derived():
    # A2(5) = 3/(d2*sqrt(5)) ≈ 0.577
    assert k.A2(5) == pytest.approx(0.577, abs=0.01)
    # A3(5) = 3/(c4*sqrt(5)) ≈ 1.427
    assert k.A3(5) == pytest.approx(1.427, abs=0.01)


def test_D3_D4_B3_B4():
    assert k.D3(5) == pytest.approx(0.0, abs=0.01)  # D3(n<=6) == 0
    assert k.D4(5) == pytest.approx(2.114, abs=0.02)
    assert k.B3(5) == pytest.approx(0.0, abs=0.05)
    assert k.B4(5) == pytest.approx(2.089, abs=0.05)


def test_imr_constants():
    assert k.E2_MR == pytest.approx(2.66, abs=0.01)
    assert k.D4_MR == pytest.approx(3.267, abs=0.01)


def test_constants_beyond_n9():
    """Legacy code stopped at n=9; these must still work for Xbar-S."""
    assert k.A3(10) > 0
    assert k.B4(10) > 1
    assert k.d2(15) > k.d2(10)


def test_n_too_small_raises():
    with pytest.raises(ValueError):
        k.d2(1)
    with pytest.raises(ValueError):
        k.c4(1)
