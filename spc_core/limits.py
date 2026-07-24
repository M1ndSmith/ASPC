"""Phase I control-limit computation.

Every function returns an immutable :class:`ControlLimits`. Key correctness points vs
the legacy code:

* Xbar-S is implemented (was routed to but missing -> produced empty limits).
* X / Xbar center-line limits are NOT clamped at zero. Measurements are two-sided;
  clamping the lower limit to 0 was a bug that hid low-side out-of-control points.
* Range/S/attribute lower limits ARE clamped at 0 (a range or count cannot be negative).
* Chart constants come from :mod:`spc_core.constants` as functions of n (not a table
  that stopped at n=9).
"""
from __future__ import annotations

import numpy as np

from . import constants as k
from .models import ChartType, ControlLimits, LimitSet


def _as_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr


def build_subgroups(
    values,
    subgroup_ids,
    *,
    expected_n: int | None = None,
    exclude_incomplete: bool = False,
) -> list[np.ndarray]:
    """Group flat values into subgroups, preserving first-seen subgroup order.

    If ``expected_n`` is set and ``exclude_incomplete`` is True, subgroups whose
    size differs from ``expected_n`` are dropped (flagged incomplete). Otherwise
    all subgroups are kept and callers should use per-subgroup constants.
    """
    values = _as_array(values)
    order: list = []
    buckets: dict = {}
    for val, sid in zip(values, subgroup_ids):
        if sid not in buckets:
            buckets[sid] = []
            order.append(sid)
        buckets[sid].append(val)
    groups = [np.asarray(buckets[sid], dtype=float) for sid in order]
    if expected_n is not None and exclude_incomplete:
        groups = [g for g in groups if len(g) == expected_n]
    return groups


def imr_limits(values) -> ControlLimits:
    x = _as_array(values)
    x = x[~np.isnan(x)]
    if x.size < 2:
        raise ValueError("I-MR requires at least 2 observations")
    moving_ranges = np.abs(np.diff(x))
    mr_bar = float(moving_ranges.mean())
    x_bar = float(x.mean())
    sigma = mr_bar / k.d2(2) if mr_bar > 0 else 0.0

    individuals = LimitSet(
        center=x_bar,
        ucl=x_bar + k.E2_MR * mr_bar,
        lcl=x_bar - k.E2_MR * mr_bar,
    )
    moving_range = LimitSet(center=mr_bar, ucl=k.D4_MR * mr_bar, lcl=0.0)
    return ControlLimits(
        chart_type=ChartType.I_MR,
        subgroup_size=1,
        components={"individuals": individuals, "moving_range": moving_range},
        sigma=sigma,
        source_n_points=int(x.size),
    )


def _subgroup_sizes(subgroups: list[np.ndarray]) -> tuple[int, bool]:
    """Return (nominal n, sizes_vary). Nominal n is the mode (most common size)."""
    sizes = [len(s) for s in subgroups]
    if not sizes:
        raise ValueError("No subgroups provided")
    # Mode; tie-break toward median.
    vals, counts = np.unique(sizes, return_counts=True)
    n = int(vals[int(np.argmax(counts))])
    vary = len(set(sizes)) > 1
    return n, vary


def xbar_r_limits(subgroups: list[np.ndarray], *, exclude_incomplete: bool = False) -> ControlLimits:
    n, sizes_vary = _subgroup_sizes(subgroups)
    if exclude_incomplete and sizes_vary:
        subgroups = [s for s in subgroups if len(s) == n]
        n, sizes_vary = _subgroup_sizes(subgroups)
    if n < 2:
        raise ValueError("Xbar-R requires subgroups of size >= 2")
    if not subgroups:
        raise ValueError("No complete subgroups remain after filtering")

    means = np.array([s.mean() for s in subgroups], dtype=float)
    ranges = np.array([s.max() - s.min() for s in subgroups], dtype=float)
    x_bar = float(means.mean())
    r_bar = float(ranges.mean())
    sigma_within = r_bar / k.d2(n) if r_bar > 0 else 0.0

    # If sizes vary and we did not exclude, use per-point Xbar limits with each
    # subgroup's own A2(n_i); R chart still uses the nominal n.
    if sizes_vary and not exclude_incomplete:
        ucl = [float(m + k.A2(len(s)) * r_bar) for m, s in zip(means, subgroups)]
        lcl = [float(m - k.A2(len(s)) * r_bar) for m, s in zip(means, subgroups)]
        # Recompute as limits around x_bar with per-n A2 (standard practice).
        ucl = [float(x_bar + k.A2(len(s)) * r_bar) for s in subgroups]
        lcl = [float(x_bar - k.A2(len(s)) * r_bar) for s in subgroups]
        xbar = LimitSet(center=x_bar, ucl=ucl, lcl=lcl)
    else:
        xbar = LimitSet(
            center=x_bar,
            ucl=x_bar + k.A2(n) * r_bar,
            lcl=x_bar - k.A2(n) * r_bar,
        )
    rng = LimitSet(center=r_bar, ucl=k.D4(n) * r_bar, lcl=k.D3(n) * r_bar)
    return ControlLimits(
        chart_type=ChartType.XBAR_R,
        subgroup_size=n,
        components={"xbar": xbar, "range": rng},
        sigma=sigma_within / np.sqrt(n) if sigma_within else 0.0,
        source_n_points=len(subgroups),
        notes={
            "sigma_within": sigma_within,
            "r_bar": r_bar,
            "sizes_vary": float(sizes_vary),
        },
    )


def xbar_s_limits(subgroups: list[np.ndarray], *, exclude_incomplete: bool = False) -> ControlLimits:
    """Xbar-S: preferred for subgroup sizes > ~8 (uses the subgroup std, not range)."""
    n, sizes_vary = _subgroup_sizes(subgroups)
    if exclude_incomplete and sizes_vary:
        subgroups = [s for s in subgroups if len(s) == n]
        n, sizes_vary = _subgroup_sizes(subgroups)
    if n < 2:
        raise ValueError("Xbar-S requires subgroups of size >= 2")
    if not subgroups:
        raise ValueError("No complete subgroups remain after filtering")

    means = np.array([s.mean() for s in subgroups], dtype=float)
    stds = np.array([s.std(ddof=1) for s in subgroups], dtype=float)
    x_bar = float(means.mean())
    s_bar = float(stds.mean())
    sigma_within = s_bar / k.c4(n) if s_bar > 0 else 0.0

    if sizes_vary and not exclude_incomplete:
        ucl = [float(x_bar + k.A3(len(s)) * s_bar) for s in subgroups]
        lcl = [float(x_bar - k.A3(len(s)) * s_bar) for s in subgroups]
        xbar = LimitSet(center=x_bar, ucl=ucl, lcl=lcl)
    else:
        xbar = LimitSet(
            center=x_bar,
            ucl=x_bar + k.A3(n) * s_bar,
            lcl=x_bar - k.A3(n) * s_bar,
        )
    s = LimitSet(center=s_bar, ucl=k.B4(n) * s_bar, lcl=k.B3(n) * s_bar)
    return ControlLimits(
        chart_type=ChartType.XBAR_S,
        subgroup_size=n,
        components={"xbar": xbar, "s": s},
        sigma=sigma_within / np.sqrt(n) if sigma_within else 0.0,
        source_n_points=len(subgroups),
        notes={
            "sigma_within": sigma_within,
            "s_bar": s_bar,
            "sizes_vary": float(sizes_vary),
        },
    )


def p_limits(defectives, sample_sizes) -> ControlLimits:
    """P chart: plotted statistic is the proportion defective; limits vary with n_i."""
    d = _as_array(defectives)
    n = _as_array(sample_sizes)
    p_bar = float(d.sum() / n.sum())
    ucl, lcl = [], []
    for ni in n:
        spread = 3.0 * np.sqrt(p_bar * (1 - p_bar) / ni) if ni > 0 else 0.0
        ucl.append(min(p_bar + spread, 1.0))
        lcl.append(max(p_bar - spread, 0.0))
    proportion = LimitSet(center=p_bar, ucl=ucl, lcl=lcl)
    return ControlLimits(
        chart_type=ChartType.P, subgroup_size=1,
        components={"proportion": proportion}, source_n_points=int(d.size),
        notes={"p_bar": p_bar},
    )


def np_limits(defectives, n: float) -> ControlLimits:
    """NP chart: constant sample size n; plotted statistic is the count defective."""
    d = _as_array(defectives)
    np_bar = float(d.mean())
    p_bar = np_bar / n if n else 0.0
    spread = 3.0 * np.sqrt(np_bar * (1 - p_bar)) if np_bar > 0 else 0.0
    comp = LimitSet(center=np_bar, ucl=np_bar + spread, lcl=max(np_bar - spread, 0.0))
    return ControlLimits(
        chart_type=ChartType.NP, subgroup_size=int(n) if n else 1,
        components={"np": comp}, source_n_points=int(d.size), notes={"p_bar": p_bar},
    )


def c_limits(counts) -> ControlLimits:
    """C chart: defect counts over a constant area of opportunity."""
    c = _as_array(counts)
    c_bar = float(c.mean())
    spread = 3.0 * np.sqrt(c_bar) if c_bar > 0 else 0.0
    comp = LimitSet(center=c_bar, ucl=c_bar + spread, lcl=max(c_bar - spread, 0.0))
    return ControlLimits(
        chart_type=ChartType.C, subgroup_size=1,
        components={"defects": comp}, source_n_points=int(c.size), notes={"c_bar": c_bar},
    )


def u_limits(counts, opportunities) -> ControlLimits:
    """U chart: defects per unit; plotted statistic is counts_i / opportunity_i."""
    c = _as_array(counts)
    o = _as_array(opportunities)
    u_bar = float(c.sum() / o.sum())
    ucl, lcl = [], []
    for oi in o:
        spread = 3.0 * np.sqrt(u_bar / oi) if oi > 0 else 0.0
        ucl.append(u_bar + spread)
        lcl.append(max(u_bar - spread, 0.0))
    comp = LimitSet(center=u_bar, ucl=ucl, lcl=lcl)
    return ControlLimits(
        chart_type=ChartType.U, subgroup_size=1,
        components={"defects_per_unit": comp}, source_n_points=int(c.size),
        notes={"u_bar": u_bar},
    )
