"""Tabular CUSUM (Cumulative Sum) control chart.

Two-sided tabular CUSUM (Montgomery):

    C+_i = max(0, x_i - (target + k·σ) + C+_{i-1})
    C-_i = max(0, (target - k·σ) - x_i + C-_{i-1})

Signal when C+ > h·σ or C- > h·σ.

Defaults: k=0.5, h=5.0 (in units of sigma).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import constants as k
from .models import ChartType, ControlLimits, LimitSet, Signal


@dataclass
class CUSUMResult:
    k: float
    h: float
    target: float
    sigma: float
    c_plus: list[float]
    c_minus: list[float]
    decision_interval: float
    signals: list[Signal] = field(default_factory=list)
    limits: ControlLimits | None = None


def _estimate_sigma_mr(values: np.ndarray) -> float:
    if values.size < 2:
        return float(values.std(ddof=1)) if values.size > 1 else 0.0
    mr = np.abs(np.diff(values))
    return float(mr.mean() / k.d2(2)) if mr.size else 0.0


def cusum_chart(
    values,
    k: float = 0.5,
    h: float = 5.0,
    target: float | None = None,
    sigma: float | None = None,
) -> CUSUMResult:
    """Compute tabular two-sided CUSUM with decision interval h·σ."""
    if k <= 0:
        raise ValueError(f"CUSUM k must be > 0, got {k}")
    if h <= 0:
        raise ValueError(f"CUSUM h must be > 0, got {h}")

    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        raise ValueError("CUSUM requires at least 2 observations")

    tgt = float(target) if target is not None else float(arr.mean())
    sig = float(sigma) if sigma is not None and sigma > 0 else _estimate_sigma_mr(arr)
    if sig <= 0:
        raise ValueError("CUSUM requires a positive process sigma")

    k_abs = k * sig
    h_abs = h * sig
    c_plus: list[float] = []
    c_minus: list[float] = []
    cp = cm = 0.0
    signals: list[Signal] = []

    for i, x in enumerate(arr):
        cp = max(0.0, float(x) - (tgt + k_abs) + cp)
        cm = max(0.0, (tgt - k_abs) - float(x) + cm)
        c_plus.append(cp)
        c_minus.append(cm)
        if cp >= h_abs:
            signals.append(Signal(
                rule_id="CUSUM+", rule_name="CUSUM upper shift", index=i, value=cp,
                description=f"C+ exceeded h·σ={h_abs:.4g} (k={k}, h={h})", side="upper",
            ))
            cp = 0.0  # optional restart after signal
            c_plus[-1] = 0.0
        if cm >= h_abs:
            signals.append(Signal(
                rule_id="CUSUM-", rule_name="CUSUM lower shift", index=i, value=cm,
                description=f"C- exceeded h·σ={h_abs:.4g} (k={k}, h={h})", side="lower",
            ))
            cm = 0.0
            c_minus[-1] = 0.0

    # Represent decision interval as a LimitSet on the C+ / C- scale.
    limits = ControlLimits(
        chart_type=ChartType.CUSUM,
        subgroup_size=1,
        components={
            "cusum_plus": LimitSet(center=0.0, ucl=h_abs, lcl=0.0),
            "cusum_minus": LimitSet(center=0.0, ucl=h_abs, lcl=0.0),
        },
        sigma=sig,
        source_n_points=int(arr.size),
        notes={"k": k, "h": h, "k_abs": k_abs, "h_abs": h_abs, "target": tgt},
    )

    return CUSUMResult(
        k=k, h=h, target=tgt, sigma=sig,
        c_plus=c_plus, c_minus=c_minus,
        decision_interval=h_abs, signals=signals, limits=limits,
    )
