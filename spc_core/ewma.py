"""EWMA (Exponentially Weighted Moving Average) control chart.

Statistic: z_i = λ x_i + (1-λ) z_{i-1}, with z_0 = target (or process mean).
Time-varying control limits (Montgomery):

    σ_{z_i} = σ √( λ/(2-λ) · [1 - (1-λ)^{2i}] )
    UCL_i = target + L · σ_{z_i}
    LCL_i = target - L · σ_{z_i}

Defaults: λ=0.2, L=3.0. Sigma estimated from MR/d2 when not supplied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import constants as k
from .models import ChartType, ControlLimits, LimitSet, Signal


@dataclass
class EWMAResult:
    lam: float
    L: float
    target: float
    sigma: float
    z: list[float]
    ucl: list[float]
    lcl: list[float]
    center: float
    signals: list[Signal] = field(default_factory=list)
    limits: Optional[ControlLimits] = None


def _estimate_sigma_mr(values: np.ndarray) -> float:
    if values.size < 2:
        return float(values.std(ddof=1)) if values.size > 1 else 0.0
    mr = np.abs(np.diff(values))
    return float(mr.mean() / k.d2(2)) if mr.size else 0.0


def ewma_chart(
    values,
    lam: float = 0.2,
    L: float = 3.0,
    target: Optional[float] = None,
    sigma: Optional[float] = None,
) -> EWMAResult:
    """Compute EWMA statistic series with time-varying control limits."""
    if not (0.0 < lam <= 1.0):
        raise ValueError(f"EWMA lambda must be in (0, 1], got {lam}")
    if L <= 0:
        raise ValueError(f"EWMA L must be > 0, got {L}")

    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        raise ValueError("EWMA requires at least 2 observations")

    tgt = float(target) if target is not None else float(arr.mean())
    sig = float(sigma) if sigma is not None and sigma > 0 else _estimate_sigma_mr(arr)
    if sig <= 0:
        raise ValueError("EWMA requires a positive process sigma")

    z: list[float] = []
    ucl: list[float] = []
    lcl: list[float] = []
    prev = tgt
    factor = lam / (2.0 - lam)

    for i, x in enumerate(arr, start=1):
        zi = lam * float(x) + (1.0 - lam) * prev
        z.append(zi)
        prev = zi
        var_factor = factor * (1.0 - (1.0 - lam) ** (2 * i))
        sigma_z = sig * np.sqrt(max(var_factor, 0.0))
        ucl.append(tgt + L * sigma_z)
        lcl.append(tgt - L * sigma_z)

    # Steady-state sigma for the frozen ControlLimits summary.
    sigma_z_ss = sig * np.sqrt(factor) if sig > 0 else 0.0
    limits = ControlLimits(
        chart_type=ChartType.EWMA,
        subgroup_size=1,
        components={
            "ewma": LimitSet(center=tgt, ucl=ucl, lcl=lcl),
        },
        sigma=sigma_z_ss,
        source_n_points=int(arr.size),
        notes={"lambda": lam, "L": L, "sigma_process": sig},
    )

    signals: list[Signal] = []
    for i, (zi, u, lo) in enumerate(zip(z, ucl, lcl)):
        if zi >= u:
            signals.append(Signal(
                rule_id="EWMA1", rule_name="Beyond EWMA UCL", index=i, value=zi,
                description=f"EWMA statistic beyond UCL (λ={lam})", side="upper",
            ))
        elif zi <= lo:
            signals.append(Signal(
                rule_id="EWMA1", rule_name="Beyond EWMA LCL", index=i, value=zi,
                description=f"EWMA statistic beyond LCL (λ={lam})", side="lower",
            ))

    return EWMAResult(
        lam=lam, L=L, target=tgt, sigma=sig, z=z, ucl=ucl, lcl=lcl,
        center=tgt, signals=signals, limits=limits,
    )
