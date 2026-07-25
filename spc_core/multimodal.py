"""Multimodality detection (Hartigan dip test) for stratification STOP gate."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MultimodalResult:
    is_multimodal: bool
    dip_statistic: float
    p_value: float
    recommendation: str


def _dip_statistic(x: np.ndarray) -> float:
    """Hartigan dip statistic approximation via modal-interval grid search."""
    x = np.sort(x)
    n = x.size
    if n < 8:
        return 0.0

    ecdf = np.arange(1, n + 1) / n
    step = max(1, n // 40)
    idxs = list(range(0, n, step))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)

    best = 0.0
    for a in idxs:
        for b in idxs:
            if b <= a:
                continue
            um = np.empty(n)
            if a > 0:
                um[: a + 1] = np.linspace(0.0, ecdf[a], a + 1)
            else:
                um[0] = ecdf[0]
            mid_mass = ecdf[b] - ecdf[a]
            um[a: b + 1] = ecdf[a] + np.linspace(0.0, mid_mass, b - a + 1)
            if b < n - 1:
                um[b:] = np.linspace(ecdf[b], 1.0, n - b)
            else:
                um[-1] = 1.0
            dip = float(np.max(np.abs(ecdf - um)))
            if dip > best:
                best = dip
    return best


def check_multimodal(values, alpha: float = 0.05) -> MultimodalResult:
    """Detect multimodality for the stratification STOP gate.

    Uses the ``diptest`` package when installed; otherwise a dip approximation
    plus a histogram peak heuristic.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 8:
        return MultimodalResult(
            is_multimodal=False, dip_statistic=0.0, p_value=1.0,
            recommendation="Insufficient data for multimodality test (n < 8).",
        )

    try:
        from diptest import diptest as _diptest

        dip, p, _ = _diptest(arr, is_data_sorted=False)
        dip, p = float(dip), float(p)
    except ImportError:
        dip = _dip_statistic(arr)
        thresh = 1.0 / (2.0 * np.sqrt(arr.size))
        p = float(np.exp(-((dip / max(thresh, 1e-9)) ** 2)))
        p = min(max(p, 0.0), 1.0)

    hist, _ = np.histogram(arr, bins=min(20, max(5, arr.size // 5)), density=True)
    peaks = 0
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] > hist.mean():
            peaks += 1

    is_mm = (p < alpha and dip > 0) or (peaks >= 2 and p < 0.15)
    if is_mm:
        rec = (
            "Multimodal distribution detected. STOP — stratify by machine/shift/lot "
            "and run separate charts per stratum. Mixing streams invalidates limits."
        )
    else:
        rec = "No significant multimodality detected."

    return MultimodalResult(
        is_multimodal=bool(is_mm),
        dip_statistic=float(dip),
        p_value=float(p),
        recommendation=rec,
    )
