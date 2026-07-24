"""Continuous / real-time MSA: reference-standard injection and bias drift.

Insert certified reference standards into the measurement stream at fixed intervals.
Track EWMA of bias (α=0.2) and a rolling R chart; raise a calibration alert when
EWMA bias exceeds ±10% of tolerance.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import constants as k


@dataclass
class CalibrationAlert:
    index: int
    ewma_bias: float
    threshold: float
    message: str


@dataclass
class ContinuousMSAState:
    alpha: float = 0.2
    tolerance: float = 1.0
    alert_fraction: float = 0.10  # ±10% of tolerance
    ewma_bias: float = 0.0
    initialized: bool = False
    reference_biases: list[float] = field(default_factory=list)
    rolling_ranges: deque = field(default_factory=lambda: deque(maxlen=50))
    alerts: list[CalibrationAlert] = field(default_factory=list)
    _last_ref: Optional[float] = None
    _n: int = 0

    @property
    def threshold(self) -> float:
        return abs(self.tolerance) * self.alert_fraction

    @property
    def gage_healthy(self) -> bool:
        return abs(self.ewma_bias) <= self.threshold


class ContinuousMSA:
    """Incremental continuous MSA monitor."""

    def __init__(
        self,
        tolerance: float,
        alpha: float = 0.2,
        alert_fraction: float = 0.10,
    ):
        if tolerance <= 0:
            raise ValueError("tolerance must be > 0")
        self.state = ContinuousMSAState(
            alpha=alpha, tolerance=tolerance, alert_fraction=alert_fraction,
        )

    def observe_reference(self, measured: float, reference: float) -> Optional[CalibrationAlert]:
        """Record a reference-standard measurement; return alert if drift detected."""
        bias = float(measured) - float(reference)
        st = self.state
        st._n += 1
        st.reference_biases.append(bias)

        if not st.initialized:
            st.ewma_bias = bias
            st.initialized = True
        else:
            st.ewma_bias = st.alpha * bias + (1.0 - st.alpha) * st.ewma_bias

        if st._last_ref is not None:
            st.rolling_ranges.append(abs(bias - st._last_ref))
        st._last_ref = bias

        if abs(st.ewma_bias) > st.threshold:
            alert = CalibrationAlert(
                index=st._n - 1,
                ewma_bias=st.ewma_bias,
                threshold=st.threshold,
                message=(
                    f"Calibration alert: EWMA bias={st.ewma_bias:.4g} exceeds "
                    f"±{st.threshold:.4g} ({st.alert_fraction:.0%} of tolerance). "
                    "Suspend measurements and recalibrate."
                ),
            )
            st.alerts.append(alert)
            return alert
        return None

    def rolling_r_chart(self) -> dict:
        """Summary of the rolling range chart on consecutive reference biases."""
        st = self.state
        if len(st.rolling_ranges) < 2:
            return {"n": 0, "r_bar": 0.0, "ucl": 0.0, "lcl": 0.0}
        ranges = np.asarray(list(st.rolling_ranges), dtype=float)
        r_bar = float(ranges.mean())
        return {
            "n": int(ranges.size),
            "r_bar": r_bar,
            "ucl": k.D4_MR * r_bar,
            "lcl": 0.0,
            "values": ranges.tolist(),
        }

    def summary(self) -> dict:
        st = self.state
        return {
            "ewma_bias": st.ewma_bias,
            "threshold": st.threshold,
            "gage_healthy": st.gage_healthy,
            "n_references": st._n,
            "n_alerts": len(st.alerts),
            "rolling_r": self.rolling_r_chart(),
            "alerts": [
                {"index": a.index, "ewma_bias": a.ewma_bias, "message": a.message}
                for a in st.alerts
            ],
        }
