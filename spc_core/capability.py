"""Process capability / performance.

Corrections vs the legacy code:
* DPMO <-> sigma level is computed analytically with the normal quantile function, not
  read off a hardcoded bucket table.
* When data is non-normal and cannot be transformed, a percentile-based (ISO 22514
  "Cnpk") path is provided instead of silently reporting parametric Cp/Cpk.
* Pure numpy/scipy; subgroup within-variation uses the same constants as the charts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from . import constants as k


def _clean(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[~np.isnan(arr)]


def dpmo_to_sigma(dpmo: float, shift: float = 1.5) -> float:
    """Convert DPMO to a (short-term) sigma level analytically.

    Z_bench (long-term) = Phi^-1(1 - dpmo/1e6). The conventional "process sigma level"
    adds the 1.5-sigma shift.
    """
    dpmo = max(min(dpmo, 999999.0), 0.0)
    if dpmo <= 0:
        z = 6.0
    else:
        z = float(stats.norm.ppf(1.0 - dpmo / 1_000_000.0))
    return z + shift


def sigma_to_dpmo(z_bench: float) -> float:
    """Inverse of the long-term relationship: DPMO expected for a given Z_bench."""
    return float((1.0 - stats.norm.cdf(z_bench)) * 1_000_000.0)


def _sigma_within(values: np.ndarray, subgroups: list[np.ndarray] | None) -> float:
    if subgroups:
        n = int(np.median([len(s) for s in subgroups]))
        if n >= 2:
            if n <= 8:
                r_bar = float(np.mean([s.max() - s.min() for s in subgroups]))
                return r_bar / k.d2(n)
            s_bar = float(np.mean([s.std(ddof=1) for s in subgroups]))
            return s_bar / k.c4(n)
    return float(values.std(ddof=1))


@dataclass
class CapabilityResult:
    n: int
    mean: float
    usl: float
    lsl: float
    target: float
    method: str                       # "parametric" | "nonparametric" | "transformed"
    sigma_within: float
    sigma_overall: float
    cp: float | None = None
    cpk: float | None = None
    cpu: float | None = None
    cpl: float | None = None
    cpm: float | None = None
    pp: float | None = None
    ppk: float | None = None
    ppu: float | None = None
    ppl: float | None = None
    observed_dpmo: float = 0.0
    expected_dpmo: float | None = None
    z_bench: float | None = None
    sigma_level: float | None = None
    yield_pct: float = 100.0
    is_centered: bool = True
    offset_from_target: float = 0.0
    rating: str = ""
    notes: dict = field(default_factory=dict)


def _rate(cpk: float | None) -> str:
    if cpk is None:
        return "Unknown"
    if cpk >= 1.67:
        return "World-class (>=1.67)"
    if cpk >= 1.33:
        return "Capable (>=1.33)"
    if cpk >= 1.0:
        return "Marginal (>=1.00)"
    return "Not capable (<1.00)"


def parametric_capability(values, usl, lsl, target=None, subgroups=None) -> CapabilityResult:
    arr = _clean(values)
    if usl is None or lsl is None or usl <= lsl:
        raise ValueError("Valid USL > LSL required for capability analysis")
    if target is None:
        target = (usl + lsl) / 2.0

    mean = float(arr.mean())
    sig_w = _sigma_within(arr, subgroups)
    sig_o = float(arr.std(ddof=1))
    spec_width = usl - lsl

    def _indices(sigma):
        cp = spec_width / (6 * sigma) if sigma > 0 else None
        cpu = (usl - mean) / (3 * sigma) if sigma > 0 else None
        cpl = (mean - lsl) / (3 * sigma) if sigma > 0 else None
        cpk = min(cpu, cpl) if sigma > 0 and cpu is not None and cpl is not None else None
        return cp, cpk, cpu, cpl

    cp, cpk, cpu, cpl = _indices(sig_w)
    pp, ppk, ppu, ppl = _indices(sig_o)

    cpm = None
    if sig_w > 0:
        tau = np.sqrt(sig_w ** 2 + (mean - target) ** 2)
        cpm = spec_width / (6 * tau) if tau > 0 else None

    return _finish(arr, usl, lsl, target, mean, sig_w, sig_o, "parametric",
                   cp, cpk, cpu, cpl, pp, ppk, ppu, ppl, cpm)


def nonparametric_capability(values, usl, lsl, target=None, subgroups=None) -> CapabilityResult:
    """Percentile (ISO 22514 / Cnpk) capability for non-normal data.

    Uses the 0.135 / 50 / 99.865 percentiles so the spread matches the +/-3 sigma
    coverage of a normal distribution without assuming normality.
    """
    arr = _clean(values)
    if usl is None or lsl is None or usl <= lsl:
        raise ValueError("Valid USL > LSL required for capability analysis")
    if target is None:
        target = (usl + lsl) / 2.0

    p00135, p50, p99865 = np.percentile(arr, [0.135, 50.0, 99.865])
    spread = p99865 - p00135
    mean = float(arr.mean())
    sig_o = float(arr.std(ddof=1))
    sig_w = _sigma_within(arr, subgroups)

    pp = (usl - lsl) / spread if spread > 0 else None
    ppu = (usl - p50) / (p99865 - p50) if (p99865 - p50) > 0 else None
    ppl = (p50 - lsl) / (p50 - p00135) if (p50 - p00135) > 0 else None
    ppk = min(ppu, ppl) if (ppu is not None and ppl is not None) else None

    res = _finish(arr, usl, lsl, target, mean, sig_w, sig_o, "nonparametric",
                  None, None, None, None, pp, ppk, ppu, ppl, None)
    res.notes["percentiles"] = {"p0.135": float(p00135), "median": float(p50), "p99.865": float(p99865)}
    return res


def _finish(arr, usl, lsl, target, mean, sig_w, sig_o, method,
            cp, cpk, cpu, cpl, pp, ppk, ppu, ppl, cpm) -> CapabilityResult:
    n = int(arr.size)
    above = int(np.sum(arr > usl))
    below = int(np.sum(arr < lsl))
    defects = above + below
    observed_dpmo = (defects / n) * 1_000_000.0 if n else 0.0
    yield_pct = ((n - defects) / n) * 100.0 if n else 100.0

    z_bench = None
    expected_dpmo = None
    sigma_level = None
    if sig_o > 0:
        z_usl = (usl - mean) / sig_o
        z_lsl = (mean - lsl) / sig_o
        z_bench = float(min(z_usl, z_lsl))
        expected_dpmo = sigma_to_dpmo(z_bench)
        # Sigma level from expected (parametric) DPMO — consistent with z_bench.
        # Guard zero-defect inflation: when observed_dpmo==0 on small n, do not claim 7.5σ.
        if n >= 30 or defects > 0:
            sigma_level = dpmo_to_sigma(expected_dpmo)
        else:
            sigma_level = z_bench + 1.5
            # Cap optimistic claims on tiny samples with zero defects.
            if n < 30 and defects == 0:
                sigma_level = min(sigma_level, 4.5)
    elif observed_dpmo > 0:
        sigma_level = dpmo_to_sigma(observed_dpmo)
    offset = mean - target
    key_cpk = cpk if cpk is not None else ppk
    is_centered = (abs(cp - cpk) < 0.1) if (cp is not None and cpk is not None) else abs(offset) < (
        (usl - lsl) * 0.125
    )

    return CapabilityResult(
        n=n, mean=mean, usl=usl, lsl=lsl, target=target, method=method,
        sigma_within=sig_w, sigma_overall=sig_o,
        cp=cp, cpk=cpk, cpu=cpu, cpl=cpl, cpm=cpm,
        pp=pp, ppk=ppk, ppu=ppu, ppl=ppl,
        observed_dpmo=observed_dpmo, expected_dpmo=expected_dpmo, z_bench=z_bench,
        sigma_level=sigma_level, yield_pct=yield_pct,
        is_centered=bool(is_centered), offset_from_target=float(offset),
        rating=_rate(key_cpk),
    )


def capability_analysis(values, usl, lsl, target=None, subgroups=None,
                        force_method: str | None = None) -> CapabilityResult:
    """Full capability decision: normality -> transform -> parametric or non-parametric.

    Returns a :class:`CapabilityResult`; the chosen ``method`` records the path taken.
    """
    from .normality import apply_transform, check_normality

    arr = _clean(values)
    if force_method == "nonparametric":
        return nonparametric_capability(arr, usl, lsl, target, subgroups)
    if force_method == "parametric":
        return parametric_capability(arr, usl, lsl, target, subgroups)

    norm = check_normality(arr)
    if norm.is_normal:
        res = parametric_capability(arr, usl, lsl, target, subgroups)
        res.notes["normality"] = {"is_normal": True, "path": "raw"}
        return res

    # Try to normalize; if successful, compute parametric capability on the
    # transformed scale with correspondingly transformed specification limits.
    tr = apply_transform(arr, method="auto")
    transform_error: str | None = None
    if tr.became_normal and tr.applied != "NONE":
        try:
            usl_t, lsl_t, target_t = _transform_specs(usl, lsl, target, tr)
            res = parametric_capability(tr.values, usl_t, lsl_t, target_t, subgroups=None)
            res.method = "transformed"
            res.notes["normality"] = {
                "is_normal": False,
                "transform_applied": tr.applied,
                "transform_label": tr.label,
                "transform_lambda": tr.lam,
                "path": "transformed_parametric",
            }
            return res
        except Exception as exc:  # noqa: BLE001 — fall through to nonparametric
            transform_error = f"{type(exc).__name__}: {exc}"

    res = nonparametric_capability(arr, usl, lsl, target, subgroups)
    notes: dict = {
        "is_normal": False,
        "transform_tried": tr.applied,
        "transform_became_normal": tr.became_normal,
        "path": "nonparametric",
    }
    if transform_error is not None:
        notes["transform_error"] = transform_error
        notes["path"] = "nonparametric_after_transform_error"
    res.notes["normality"] = notes
    return res


def _transform_specs(usl, lsl, target, tr):
    """Apply the same transform used on data to the specification limits."""

    if tr.applied == "LOG":
        if min(usl, lsl, target if target is not None else usl) <= 0:
            raise ValueError("Log transform requires positive specs")
        if "log(x+1)" in (tr.label or ""):
            return np.log1p(usl), np.log1p(lsl), np.log1p(target) if target is not None else None
        return np.log(usl), np.log(lsl), np.log(target) if target is not None else None

    if tr.applied == "BOXCOX":

        lam = tr.lam
        def _bc(x):
            if abs(lam) < 1e-12:
                return np.log(x)
            return (x ** lam - 1.0) / lam
        return float(_bc(usl)), float(_bc(lsl)), float(_bc(target)) if target is not None else None

    if tr.applied == "YEO-JOHNSON":

        # yeojohnson on a scalar needs the fitted lambda.
        def _yj(x, lam):
            x = float(x)
            if lam == 0 and x >= 0:
                return np.log1p(x)
            if x >= 0:
                return ((x + 1) ** lam - 1) / lam
            if lam == 2:
                return -np.log1p(-x)
            return -((1 - x) ** (2 - lam) - 1) / (2 - lam)
        return (
            float(_yj(usl, tr.lam)),
            float(_yj(lsl, tr.lam)),
            float(_yj(target, tr.lam)) if target is not None else None,
        )

    raise ValueError(f"Cannot transform specs for applied={tr.applied}")
