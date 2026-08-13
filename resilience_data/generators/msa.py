"""MSA generators — Gage R&R, bias, linearity, stability, NDC/resolution stress."""
from __future__ import annotations

from typing import Any

import numpy as np

Columns = dict[str, list[Any]]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def gage_rr(
    *,
    quality: str = "excellent",
    n_parts: int = 10,
    n_operators: int = 3,
    n_trials: int = 3,
    seed: int = 301,
) -> Columns:
    """Balanced Gage R&R.

    excellent: %GRR typically < 10 of tolerance 10
    marginal:  %GRR typically 10–30
    poor:      %GRR typically > 30
    ndc_fail:  very low part variation → NDC < 5
    """
    rng = _rng(seed)
    if quality == "excellent":
        part_sd, op_sd, repeat_sd = 2.0, 0.05, 0.08
    elif quality == "marginal":
        # Tuned so %GRR of tolerance 10 lands in the AIAG 10–30 conditional band.
        part_sd, op_sd, repeat_sd = 1.2, 0.15, 0.3
    elif quality == "poor":
        part_sd, op_sd, repeat_sd = 0.3, 0.8, 1.2
    elif quality == "ndc_fail":
        part_sd, op_sd, repeat_sd = 0.05, 0.4, 0.5
    else:
        raise ValueError(f"unknown quality {quality!r}")

    part_effects = rng.normal(0.0, part_sd, n_parts)
    op_effects = rng.normal(0.0, op_sd, n_operators)
    parts: list[str] = []
    operators: list[str] = []
    measurements: list[float] = []
    for pi in range(n_parts):
        for oi in range(n_operators):
            for _ in range(n_trials):
                noise = float(rng.normal(0.0, repeat_sd))
                y = 10.0 + part_effects[pi] + op_effects[oi] + noise
                parts.append(f"P{pi + 1}")
                operators.append(f"Op{oi + 1}")
                measurements.append(float(y))
    return {"Part": parts, "Operator": operators, "Measurement": measurements}


def gage_rr_unbalanced(*, seed: int = 305) -> Columns:
    """Deliberately unbalanced cells so ANOVA falls back to range."""
    rng = _rng(seed)
    parts: list[str] = []
    operators: list[str] = []
    measurements: list[float] = []
    plan = {
        ("P1", "Op1"): 3,
        ("P1", "Op2"): 1,
        ("P2", "Op1"): 2,
        ("P2", "Op2"): 3,
        ("P3", "Op1"): 1,
        ("P3", "Op2"): 2,
    }
    for (part, op), n in plan.items():
        for _ in range(n):
            parts.append(part)
            operators.append(op)
            measurements.append(float(10.0 + rng.normal(0.0, 0.5)))
    return {"Part": parts, "Operator": operators, "Measurement": measurements}


def bias_significant(*, n: int = 30, reference: float = 10.0, seed: int = 311) -> Columns:
    rng = _rng(seed)
    meas = [float(reference + 0.8 + rng.normal(0.0, 0.1)) for _ in range(n)]
    return {"Measurement": meas, "Reference": [reference] * n}


def linearity_ok(*, n_refs: int = 5, n_reps: int = 6, seed: int = 312) -> Columns:
    rng = _rng(seed)
    refs_levels = np.linspace(5.0, 15.0, n_refs)
    measurements: list[float] = []
    references: list[float] = []
    for ref in refs_levels:
        for _ in range(n_reps):
            bias = 0.02 * (ref - 10.0)
            measurements.append(float(ref + bias + rng.normal(0.0, 0.08)))
            references.append(float(ref))
    return {"Measurement": measurements, "Reference": references}


def stability_ok(*, n: int = 40, seed: int = 313) -> Columns:
    rng = _rng(seed)
    values = 10.0 + rng.normal(0.0, 0.15, n)
    return {"Measurement": [float(v) for v in values]}
