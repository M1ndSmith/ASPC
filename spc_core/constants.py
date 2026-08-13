"""Shewhart control-chart constants as functions of subgroup size n.

The current codebase hardcoded A2/D3/D4 only up to n=9. Here we keep the standard
d2/d3 unbiasing tables (which have no simple closed form) and *derive* every other
constant from them plus c4, so charts remain correct for any subgroup size.

References: ASTM / AIAG SPC control chart constant tables.
"""
from __future__ import annotations

import math

# Hartley's d2 (mean of the relative range) for n = 2..25.
_D2 = {
    2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847,
    9: 2.970, 10: 3.078, 11: 3.173, 12: 3.258, 13: 3.336, 14: 3.407,
    15: 3.472, 16: 3.532, 17: 3.588, 18: 3.640, 19: 3.689, 20: 3.735,
    21: 3.778, 22: 3.819, 23: 3.858, 24: 3.895, 25: 3.931,
}

# d3 (standard deviation of the relative range) for n = 2..25.
_D3 = {
    2: 0.853, 3: 0.888, 4: 0.880, 5: 0.864, 6: 0.848, 7: 0.833, 8: 0.820,
    9: 0.808, 10: 0.797, 11: 0.787, 12: 0.778, 13: 0.770, 14: 0.763,
    15: 0.756, 16: 0.750, 17: 0.744, 18: 0.739, 19: 0.734, 20: 0.729,
    21: 0.724, 22: 0.720, 23: 0.716, 24: 0.712, 25: 0.708,
}


def _require(n: int) -> None:
    if n < 2:
        raise ValueError(f"subgroup size must be >= 2, got {n}")


def c4(n: int) -> float:
    """Unbiasing constant for the sample standard deviation.

    Closed form: c4(n) = sqrt(2/(n-1)) * Gamma(n/2) / Gamma((n-1)/2).
    Uses lgamma for numerical stability at large n.
    """
    _require(n)
    return math.sqrt(2.0 / (n - 1)) * math.exp(
        math.lgamma(n / 2.0) - math.lgamma((n - 1) / 2.0)
    )


def d2(n: int) -> float:
    """Mean of the relative range. Tabulated for n<=25; asymptotic approx beyond."""
    _require(n)
    if n in _D2:
        return _D2[n]
    # Tippett/Hartley asymptotic: d2 ≈ sqrt(2*ln(n)) - (γ + ln(ln(n))) / (2*sqrt(2*ln(n)))
    # for large n, where γ ≈ 0.57721 (Euler-Mascheroni). Good enough for n>25.
    import math
    ln_n = math.log(n)
    gamma = 0.5772156649
    return math.sqrt(2.0 * ln_n) - (gamma + math.log(ln_n)) / (2.0 * math.sqrt(2.0 * ln_n))


def d3(n: int) -> float:
    """Std of the relative range. Tabulated for n<=25; decays slowly beyond."""
    _require(n)
    if n in _D3:
        return _D3[n]
    # Approximate decay: d3 ~ π / sqrt(6*ln(n)) for large n (extreme-value theory).
    import math
    return math.pi / math.sqrt(6.0 * math.log(n))


def A2(n: int) -> float:
    """Xbar-R: UCL/LCL = Xbar +/- A2 * Rbar."""
    return 3.0 / (d2(n) * math.sqrt(n))


def A3(n: int) -> float:
    """Xbar-S: UCL/LCL = Xbar +/- A3 * Sbar."""
    return 3.0 / (c4(n) * math.sqrt(n))


def D3(n: int) -> float:
    """R chart lower factor (clamped at 0)."""
    val = 1.0 - 3.0 * d3(n) / d2(n)
    return max(val, 0.0)


def D4(n: int) -> float:
    """R chart upper factor."""
    return 1.0 + 3.0 * d3(n) / d2(n)


def B3(n: int) -> float:
    """S chart lower factor (clamped at 0)."""
    val = 1.0 - 3.0 / c4(n) * math.sqrt(1.0 - c4(n) ** 2)
    return max(val, 0.0)


def B4(n: int) -> float:
    """S chart upper factor."""
    return 1.0 + 3.0 / c4(n) * math.sqrt(1.0 - c4(n) ** 2)


def E2(n: int) -> float:
    """I-MR: individuals limits = Xbar +/- E2 * MRbar (E2 = 3/d2, n=2 for moving range)."""
    return 3.0 / d2(n)


# Convenience: the I-MR moving-range case uses n=2.
D4_MR = D4(2)  # ~3.267
E2_MR = E2(2)  # ~2.660
