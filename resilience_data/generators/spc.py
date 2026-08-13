"""SPC chart / OOC / distribution / Phase I size generators."""
from __future__ import annotations

from typing import Any

import numpy as np

Columns = dict[str, list[Any]]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def imr_in_control(*, n: int = 50, seed: int = 100) -> Columns:
    rng = _rng(seed)
    values = 100.0 + rng.normal(0.0, 1.0, n)
    return {"measurement": [float(v) for v in values]}


def xbar_r(*, n_subgroups: int = 25, size: int = 5, seed: int = 101) -> Columns:
    rng = _rng(seed)
    measurements: list[float] = []
    subgroups: list[int] = []
    for sid in range(1, n_subgroups + 1):
        for v in 50.0 + rng.normal(0.0, 1.2, size):
            measurements.append(float(v))
            subgroups.append(sid)
    return {"measurement": measurements, "subgroup": subgroups}


def xbar_s(*, n_subgroups: int = 25, size: int = 10, seed: int = 100) -> Columns:
    return xbar_r(n_subgroups=n_subgroups, size=size, seed=seed)


def attribute_p(*, n: int = 25, seed: int = 104) -> Columns:
    rng = _rng(seed)
    inspected = [int(x) for x in rng.integers(80, 120, n)]
    defective = [int(rng.binomial(k, 0.05)) for k in inspected]
    return {"defective": defective, "inspected": inspected}


def attribute_np(*, n: int = 25, sample_size: int = 100, seed: int = 105) -> Columns:
    rng = _rng(seed)
    defectives = [int(x) for x in rng.binomial(sample_size, 0.04, n)]
    return {"defectives": defectives, "sample_size": [sample_size] * n}


def attribute_c(*, n: int = 25, seed: int = 106) -> Columns:
    rng = _rng(seed)
    return {"defects": [int(x) for x in rng.poisson(3.0, n)]}


def attribute_u(*, n: int = 25, seed: int = 107) -> Columns:
    rng = _rng(seed)
    units = [int(x) for x in rng.integers(5, 15, n)]
    defects = [int(rng.poisson(0.6 * u)) for u in units]
    return {"defects": defects, "units": units}


def imr_mean_shift(*, n: int = 50, seed: int = 111) -> Columns:
    rng = _rng(seed)
    values = 100.0 + rng.normal(0.0, 1.0, n)
    values[n // 2 :] += 4.0
    return {"measurement": [float(v) for v in values]}


def imr_single_spike(*, n: int = 50, seed: int = 112) -> Columns:
    rng = _rng(seed)
    values = 100.0 + rng.normal(0.0, 1.0, n)
    values[n // 2] = 100.0 + 8.0
    return {"measurement": [float(v) for v in values]}


def imr_trend(*, n: int = 50, seed: int = 113) -> Columns:
    rng = _rng(seed)
    t = np.arange(n, dtype=float)
    values = 100.0 + 0.15 * t + rng.normal(0.0, 0.3, n)
    return {"measurement": [float(v) for v in values]}


def xbar_r_variance_increase(
    *, n_subgroups: int = 30, size: int = 5, seed: int = 114
) -> Columns:
    rng = _rng(seed)
    measurements: list[float] = []
    subgroups: list[int] = []
    for sid in range(1, n_subgroups + 1):
        sd = 1.2 if sid <= 20 else 4.0
        for v in 50.0 + rng.normal(0.0, sd, size):
            measurements.append(float(v))
            subgroups.append(sid)
    return {"measurement": measurements, "subgroup": subgroups}


def imr_sustained_small_shift(
    *, n: int = 60, shift: float = 0.8, seed: int = 163
) -> Columns:
    """A sustained sub-sigma shift: too small for rule 1, big enough for the run rules.

    A large step shift induces enough lag-1 autocorrelation that ``establish`` routes it
    to EWMA (see ``imr_mean_shift``), which means the Shewhart run rules never run. A
    ~0.8-sigma shift keeps lag-1 ACF under the 0.2 gate threshold, so this case stays on
    the Shewhart route and exercises Nelson rule 2 — the classic reason run rules exist.
    """
    rng = _rng(seed)
    values = 100.0 + rng.normal(0.0, 1.0, n)
    values[n // 2 :] += shift
    return {"measurement": [float(v) for v in values]}


def imr_trend_nelson3(*, n: int = 30, slope: float = 0.5, seed: int = 141) -> Columns:
    """Steady drift with tiny noise so six-in-a-row monotonic (Nelson 3) fires.

    Charted directly via ``analyze_control_chart`` rather than ``establish``: a drift is
    autocorrelated by construction, so the pipeline correctly diverts it to EWMA. The
    Shewhart trend rule still has to work, so it is asserted at the chart layer.
    """
    rng = _rng(seed)
    t = np.arange(n, dtype=float)
    values = 100.0 + slope * t + rng.normal(0.0, 0.05, n)
    return {"measurement": [float(v) for v in values]}


def imr_alternating_nelson4(*, n: int = 40, swing: float = 2.0, seed: int = 142) -> Columns:
    """Sawtooth from operator over-adjustment (tampering) — Nelson rule 4, 14 alternating."""
    rng = _rng(seed)
    base = np.array([swing if i % 2 else -swing for i in range(n)], dtype=float)
    values = 100.0 + base + rng.normal(0.0, 0.1, n)
    return {"measurement": [float(v) for v in values]}


def normal_path(*, n: int = 60, seed: int = 122) -> Columns:
    return imr_in_control(n=n, seed=seed)


def skewed_boxcox(*, n: int = 80, seed: int = 122) -> Columns:
    rng = _rng(seed)
    # Lognormal — right-skewed, typically unimodal, Box-Cox / log recoverable.
    values = np.exp(rng.normal(0.0, 0.8, n))
    return {"measurement": [float(v) for v in values]}


def heavy_tail_wheeler(*, n: int = 80, seed: int = 123) -> Columns:
    rng = _rng(seed)
    # Cauchy — heavy tails that resist Box-Cox / Yeo-Johnson normalization.
    values = 10.0 + rng.standard_cauchy(n)
    return {"measurement": [float(v) for v in values]}


def heavy_tail_wheeler_subgroup(
    *, n_subgroups: int = 25, size: int = 5, seed: int = 124
) -> Columns:
    rng = _rng(seed)
    measurements: list[float] = []
    subgroups: list[int] = []
    for sid in range(1, n_subgroups + 1):
        for v in 10.0 + rng.standard_cauchy(size):
            measurements.append(float(v))
            subgroups.append(sid)
    return {"measurement": measurements, "subgroup": subgroups}


def autocorrelated_ewma(*, n: int = 80, seed: int = 125) -> Columns:
    rng = _rng(seed)
    # AR(1) with phi=0.8 → strong lag-1 ACF.
    phi = 0.8
    eps = rng.normal(0.0, 1.0, n)
    x = np.zeros(n)
    x[0] = eps[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return {"measurement": [float(v) for v in (100.0 + x)]}


def multimodal_stop(*, n: int = 100, seed: int = 126) -> Columns:
    rng = _rng(seed)
    # Two well-separated clusters so the no-diptest heuristic fires.
    a = rng.normal(0.0, 0.5, n // 2)
    b = rng.normal(20.0, 0.5, n - n // 2)
    values = np.concatenate([a, b])
    rng.shuffle(values)
    return {"measurement": [float(v) for v in values]}


def too_few_points(*, n: int = 10, seed: int = 131) -> Columns:
    return imr_in_control(n=n, seed=seed)


def incomplete_subgroup(*, n_complete: int = 24, size: int = 5, seed: int = 132) -> Columns:
    """24 full subgroups of 5, then a ragged last subgroup of 2."""
    base = xbar_r(n_subgroups=n_complete, size=size, seed=seed)
    rng = _rng(seed + 1)
    last_id = n_complete + 1
    for v in 50.0 + rng.normal(0.0, 1.2, 2):
        base["measurement"].append(float(v))
        base["subgroup"].append(last_id)
    return base


def constant_series(*, n: int = 40) -> Columns:
    return {"measurement": [10.0] * n}


def p_zero_n() -> Columns:
    return {"defective": [1, 2, 0], "inspected": [100, 0, 80]}


def u_zero_opportunity() -> Columns:
    return {"defects": [1, 2, 0], "units": [10, 0, 8]}


def empty_series() -> Columns:
    return {"measurement": []}


def all_nan(*, n: int = 10) -> Columns:
    return {"measurement": [None] * n}
