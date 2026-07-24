"""Deterministic synthetic SPC / MSA / capability datasets.

Used by pytest fixtures and ``python -m sample_data --out examples/data``.
All generators are numpy-seeded so outputs are reproducible across runs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

Columns = dict[str, list[Any]]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def spc_individual(*, in_control: bool = True, n: int = 50, seed: int = 42) -> Columns:
    """I-MR series around 100. Out-of-control injects a late mean shift + spike."""
    rng = _rng(seed if in_control else seed + 1)
    values = 100.0 + rng.normal(0.0, 1.0, n)
    if not in_control:
        values[n // 2 :] += 4.0
        values[-3] = 100.0 + 8.0
    return {"measurement": [float(v) for v in values]}


def spc_subgroup(*, n_subgroups: int = 25, size: int = 5, seed: int = 7) -> Columns:
    """Xbar-R data: ``measurement`` + ``subgroup`` with fixed subgroup size."""
    rng = _rng(seed)
    measurements: list[float] = []
    subgroups: list[int] = []
    for sid in range(1, n_subgroups + 1):
        for v in 50.0 + rng.normal(0.0, 1.2, size):
            measurements.append(float(v))
            subgroups.append(sid)
    return {"measurement": measurements, "subgroup": subgroups}


def attribute_c(*, n: int = 25, seed: int = 11) -> Columns:
    rng = _rng(seed)
    return {"defects": [int(x) for x in rng.poisson(3.0, n)]}


def attribute_p(*, n: int = 25, seed: int = 12) -> Columns:
    rng = _rng(seed)
    inspected = [int(x) for x in rng.integers(80, 120, n)]
    defective = [int(rng.binomial(k, 0.05)) for k in inspected]
    return {"defective": defective, "inspected": inspected}


def attribute_np(*, n: int = 25, sample_size: int = 100, seed: int = 13) -> Columns:
    rng = _rng(seed)
    defectives = [int(x) for x in rng.binomial(sample_size, 0.04, n)]
    return {"defectives": defectives, "sample_size": [sample_size] * n}


def attribute_u(*, n: int = 25, seed: int = 14) -> Columns:
    rng = _rng(seed)
    units = [int(x) for x in rng.integers(5, 15, n)]
    defects = [int(rng.poisson(0.6 * u)) for u in units]
    return {"defects": defects, "units": units}


def msa_gage_rr(
    *,
    quality: str = "excellent",
    n_parts: int = 10,
    n_operators: int = 3,
    n_trials: int = 3,
    seed: int = 21,
) -> Columns:
    """Balanced Gage R&R study.

    excellent: high part variation, low gage noise → grr_percent < 30
    poor: gage noise dominates part variation
    """
    if quality not in ("excellent", "poor"):
        raise ValueError("quality must be 'excellent' or 'poor'")
    rng = _rng(seed if quality == "excellent" else seed + 99)

    if quality == "excellent":
        part_sd, op_sd, repeat_sd = 2.0, 0.05, 0.08
    else:
        part_sd, op_sd, repeat_sd = 0.3, 0.8, 1.2

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


def msa_bias(*, n: int = 30, reference: float = 10.0, seed: int = 31) -> Columns:
    rng = _rng(seed)
    refs = [reference] * n
    # Small positive bias + noise
    meas = [float(reference + 0.05 + rng.normal(0.0, 0.1)) for _ in range(n)]
    return {"Measurement": meas, "Reference": refs}


def msa_linearity(*, n_refs: int = 5, n_reps: int = 6, seed: int = 32) -> Columns:
    rng = _rng(seed)
    refs_levels = np.linspace(5.0, 15.0, n_refs)
    measurements: list[float] = []
    references: list[float] = []
    for ref in refs_levels:
        for _ in range(n_reps):
            # Near-zero slope bias with small noise → high R^2 linearity residual model
            bias = 0.02 * (ref - 10.0)
            measurements.append(float(ref + bias + rng.normal(0.0, 0.08)))
            references.append(float(ref))
    return {"Measurement": measurements, "Reference": references}


def msa_stability(*, n: int = 40, seed: int = 33) -> Columns:
    rng = _rng(seed)
    values = 10.0 + rng.normal(0.0, 0.15, n)
    return {"Measurement": [float(v) for v in values]}


def capability(*, kind: str = "excellent", n: int = 200, seed: int = 41) -> Columns:
    """Capability datasets keyed by kind.

    excellent: mean≈10, σ≈0.1 → Cpk > 1.0 for USL=10.5 / LSL=9.5
    skewed: exponential (non-normal)
    off_center: mean shifted toward USL
    high_variation: large σ relative to specs
    """
    rng = _rng(seed)
    if kind == "excellent":
        values = 10.0 + rng.normal(0.0, 0.10, n)
    elif kind == "skewed":
        values = rng.exponential(2.0, n) + 5.0
    elif kind == "off_center":
        values = 10.25 + rng.normal(0.0, 0.12, n)
    elif kind == "high_variation":
        values = 10.0 + rng.normal(0.0, 0.45, n)
    else:
        raise ValueError(
            f"Unknown capability kind '{kind}'. "
            "Use excellent|skewed|off_center|high_variation"
        )
    return {"measurement": [float(v) for v in values]}


# Canonical names used by CLI demos and ``python -m sample_data``
DATASET_CATALOG: dict[str, Columns] = {
    "spc_individual_in_control": spc_individual(in_control=True),
    "spc_individual_out_of_control": spc_individual(in_control=False),
    "spc_subgroup_data": spc_subgroup(),
    "spc_c_chart_data": attribute_c(),
    "spc_p_chart_data": attribute_p(),
    "spc_np_chart_data": attribute_np(),
    "spc_u_chart_data": attribute_u(),
    "msa_gage_rr_excellent": msa_gage_rr(quality="excellent"),
    "msa_gage_rr_poor": msa_gage_rr(quality="poor"),
    "msa_bias_study": msa_bias(),
    "msa_linearity_study": msa_linearity(),
    "msa_stability_study": msa_stability(),
    "capability_excellent": capability(kind="excellent"),
    "capability_skewed_data": capability(kind="skewed"),
    "capability_off_center": capability(kind="off_center"),
    "capability_high_variation": capability(kind="high_variation"),
}


def get_dataset(name: str) -> Columns:
    """Return a fresh copy of a named dataset from the catalog."""
    if name not in DATASET_CATALOG:
        # Rebuild on demand for names that map to generators
        builders = {
            "spc_individual_in_control": lambda: spc_individual(in_control=True),
            "spc_individual_out_of_control": lambda: spc_individual(in_control=False),
            "spc_subgroup_data": spc_subgroup,
            "spc_c_chart_data": attribute_c,
            "spc_p_chart_data": attribute_p,
            "spc_np_chart_data": attribute_np,
            "spc_u_chart_data": attribute_u,
            "msa_gage_rr_excellent": lambda: msa_gage_rr(quality="excellent"),
            "msa_gage_rr_poor": lambda: msa_gage_rr(quality="poor"),
            "msa_bias_study": msa_bias,
            "msa_linearity_study": msa_linearity,
            "msa_stability_study": msa_stability,
            "capability_excellent": lambda: capability(kind="excellent"),
            "capability_skewed_data": lambda: capability(kind="skewed"),
            "capability_off_center": lambda: capability(kind="off_center"),
            "capability_high_variation": lambda: capability(kind="high_variation"),
        }
        if name not in builders:
            raise KeyError(f"Unknown dataset '{name}'. Available: {list(builders)}")
        cols = builders[name]()
    else:
        cols = DATASET_CATALOG[name]
    return {k: list(v) for k, v in cols.items()}


def write_csv(cols: Mapping[str, list[Any]], path: str | Path) -> Path:
    """Write a column dict to CSV and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(cols.keys())
    if not keys:
        raise ValueError("empty columns")
    n = len(cols[keys[0]])
    if any(len(cols[k]) != n for k in keys):
        raise ValueError("column length mismatch")
    lines = [",".join(keys)]
    for i in range(n):
        row = []
        for k in keys:
            v = cols[k][i]
            row.append(str(v))
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_all(out_dir: str | Path) -> list[Path]:
    """Write every catalog dataset as ``{name}.csv`` under ``out_dir``."""
    out = Path(out_dir)
    written: list[Path] = []
    for name in sorted(
        {
            "spc_individual_in_control",
            "spc_individual_out_of_control",
            "spc_subgroup_data",
            "spc_c_chart_data",
            "spc_p_chart_data",
            "spc_np_chart_data",
            "spc_u_chart_data",
            "msa_gage_rr_excellent",
            "msa_gage_rr_poor",
            "msa_bias_study",
            "msa_linearity_study",
            "msa_stability_study",
            "capability_excellent",
            "capability_skewed_data",
            "capability_off_center",
            "capability_high_variation",
        }
    ):
        written.append(write_csv(get_dataset(name), out / f"{name}.csv"))
    return written
