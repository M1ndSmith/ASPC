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

        dip, p = _diptest(arr)
        dip, p = float(dip), float(p)
    except ImportError:
        dip = _dip_statistic(arr)
        thresh = 1.0 / (2.0 * np.sqrt(arr.size))
        p = float(np.exp(-((dip / max(thresh, 1e-9)) ** 2)))
        p = min(max(p, 0.0), 1.0)

    hist, _ = np.histogram(arr, bins=min(20, max(5, arr.size // 5)), density=True)
    # Local maxima including edge bins, prominent relative to the tallest bin.
    mx = float(hist.max()) if hist.size else 0.0
    peak_idxs: list[int] = []
    for i in range(len(hist)):
        left = hist[i - 1] if i > 0 else -np.inf
        right = hist[i + 1] if i < len(hist) - 1 else -np.inf
        if hist[i] > left and hist[i] > right and mx > 0 and hist[i] >= 0.4 * mx:
            peak_idxs.append(i)

    # A genuine mixture is two populated modes separated by a near-EMPTY region. A merely
    # *shallow* dip is not evidence: normal data at n < 100 routinely shows two or three
    # noise peaks whose valley sits at 0.2-0.3 of the mode height, which would false-STOP
    # an in-control process. Require an essentially empty gap at least two bins wide with
    # substantial mass on both sides of it.
    clear_bimodal = False
    if len(peak_idxs) >= 2 and mx > 0:
        empty = 0.05 * mx
        widest: tuple[int, int] | None = None
        start: int | None = None
        for i in range(peak_idxs[0] + 1, peak_idxs[-1]):
            if hist[i] <= empty:
                if start is None:
                    start = i
                if widest is None or (i - start) > (widest[1] - widest[0]):
                    widest = (start, i)
            else:
                start = None
        if widest is not None and (widest[1] - widest[0] + 1) >= 2:
            total = float(hist.sum())
            left_mass = float(hist[: widest[0]].sum()) / total
            right_mass = float(hist[widest[1] + 1 :].sum()) / total
            clear_bimodal = min(left_mass, right_mass) >= 0.15

    # ``diptest`` is a declared dependency, so ``p`` is normally the real Hartigan
    # p-value. The histogram check stays as a fallback for installs where the optional
    # C extension is unavailable and ``_dip_statistic``'s approximate p is unreliable.
    is_mm = (p < alpha and dip > 0) or clear_bimodal
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
