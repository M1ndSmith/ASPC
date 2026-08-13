"""Measurement System Analysis: Gage R&R (ANOVA + range), bias, linearity, stability.

Pure numpy/scipy. Results are dataclasses with an unambiguous ``grr_percent`` field so
the acceptance decision cannot be silently mis-keyed (a legacy bug read ``grr_percent``
while the pipeline emitted a nested ``percent_study_variation['gage_rr']``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from . import constants as k


def _acceptability(grr_percent: float, ndc: int = 0, *, require_ndc: bool = True) -> str:
    """AIAG MSA-4: %GRR < 10 excellent, 10-30 conditional, >30 unacceptable.
    NDC >= 5 is also required for acceptability when ``require_ndc`` is True.
    """
    if require_ndc and ndc < 5:
        return "Unacceptable"
    if grr_percent < 10:
        return "Excellent"
    if grr_percent < 30:
        return "Acceptable"
    return "Unacceptable"


def ndc_gate(ndc: int, minimum: int = 5) -> tuple[bool, str]:
    """NDC must be >= 5 for the gage to discriminate parts for SPC."""
    ok = ndc >= minimum
    reason = (
        f"NDC={ndc} meets minimum {minimum}."
        if ok
        else f"NDC={ndc} < {minimum}: gage cannot discriminate parts sufficiently for SPC."
    )
    return ok, reason


def gage_resolution_gate(resolution: float, tolerance: float, ratio: float = 10.0) -> tuple[bool, str]:
    """10:1 rule — gage resolution must be <= tolerance / ratio."""
    if tolerance <= 0:
        return False, "Tolerance must be positive for the 10:1 resolution rule."
    max_res = tolerance / ratio
    ok = resolution <= max_res
    reason = (
        f"Resolution {resolution} <= tolerance/{ratio:g} = {max_res}."
        if ok
        else f"Resolution {resolution} > tolerance/{ratio:g} = {max_res} (fails 10:1 rule)."
    )
    return ok, reason


def _cell_counts(parts, operators) -> dict[tuple, int]:
    counts: dict[tuple, int] = {}
    for p, o in zip(parts, operators):
        key = (p, o)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _is_balanced(parts, operators, n_parts: int, n_ops: int) -> tuple[bool, int]:
    """Return (balanced, trials_per_cell). Unbalanced if any cell missing or unequal."""
    counts = _cell_counts(parts, operators)
    expected_cells = n_parts * n_ops
    if len(counts) != expected_cells:
        return False, 0
    vals = list(counts.values())
    if len(set(vals)) != 1:
        return False, 0
    return True, vals[0]


def _groups(keys, values) -> dict:
    out: dict = {}
    for kk, v in zip(keys, values):
        out.setdefault(kk, []).append(v)
    return {kk: np.asarray(v, dtype=float) for kk, v in out.items()}


@dataclass
class GageRRResult:
    method: str
    n_parts: int
    n_operators: int
    n_trials: int
    var_repeatability: float
    var_reproducibility: float
    var_gage_rr: float
    var_part: float
    var_total: float
    grr_percent: float                 # % study variation attributable to Gage R&R
    part_percent: float
    ndc: int
    acceptability: str
    grr_percent_tolerance: float | None = None
    detail: dict = field(default_factory=dict)


def gage_rr_anova(parts, operators, measurements, tolerance: float | None = None) -> GageRRResult:
    parts = np.asarray(list(parts))
    operators = np.asarray(list(operators))
    y = np.asarray(list(measurements), dtype=float)

    uparts = list(dict.fromkeys(parts.tolist()))
    uops = list(dict.fromkeys(operators.tolist()))
    n_parts, n_ops = len(uparts), len(uops)
    balanced, n_trials = _is_balanced(parts, operators, n_parts, n_ops)
    if not balanced:
        # Balanced ANOVA formulas are invalid on unbalanced cells — fall back to
        # the range method rather than returning silently wrong variance components.
        result = gage_rr_range(parts, operators, measurements, tolerance=tolerance)
        result.detail = {
            **(result.detail or {}),
            "balanced": False,
            "anova_skipped": True,
            "reason": "Unbalanced design; ANOVA formulas not applied — used range method.",
        }
        result.method = "Range (unbalanced fallback)"
        return result
    if n_trials < 1:
        raise ValueError("Gage R&R requires at least one trial per part-operator cell")

    grand = y.mean()
    part_means = _group_means(parts, y)
    op_means = _group_means(operators, y)

    ss_total = float(np.sum((y - grand) ** 2))
    ss_part = n_ops * n_trials * float(np.sum((np.array([part_means[p] for p in uparts]) - grand) ** 2))
    ss_op = n_parts * n_trials * float(np.sum((np.array([op_means[o] for o in uops]) - grand) ** 2))

    cell_means: dict[tuple, list[float]] = {}
    for p, o, v in zip(parts, operators, y):
        cell_means.setdefault((p, o), []).append(v)
    cell_mean_arr = np.array([np.mean(cell_means[(p, o)]) for p in uparts for o in uops
                              if (p, o) in cell_means])
    ss_cells = n_trials * float(np.sum((cell_mean_arr - grand) ** 2))
    ss_interaction = ss_cells - ss_part - ss_op
    ss_equip = ss_total - ss_part - ss_op - ss_interaction

    df_part = n_parts - 1
    df_op = n_ops - 1
    df_int = df_part * df_op
    df_equip = n_parts * n_ops * (n_trials - 1)

    ms_part = ss_part / df_part if df_part > 0 else 0.0
    ms_op = ss_op / df_op if df_op > 0 else 0.0
    ms_int = ss_interaction / df_int if df_int > 0 else 0.0
    ms_equip = ss_equip / df_equip if df_equip > 0 else 0.0

    var_rep = max(ms_equip, 0.0)                                            # repeatability
    var_reprod = max((ms_op - ms_int) / (n_parts * n_trials), 0.0)
    var_int = max((ms_int - ms_equip) / n_trials, 0.0)
    var_reprod_total = var_reprod + var_int
    var_grr = var_rep + var_reprod_total
    var_part = max((ms_part - ms_int) / (n_ops * n_trials), 0.0)
    var_total = var_grr + var_part

    std_grr = np.sqrt(var_grr)
    std_part = np.sqrt(var_part)
    std_total = np.sqrt(var_total)

    grr_percent = float(std_grr / std_total * 100) if std_total > 0 else 0.0
    part_percent = float(std_part / std_total * 100) if std_total > 0 else 0.0
    ndc = int(np.floor(np.sqrt(2) * std_part / std_grr)) if std_grr > 0 else 0

    grr_pct_tol = float(6 * std_grr / tolerance * 100) if tolerance else None

    return GageRRResult(
        method="ANOVA", n_parts=n_parts, n_operators=n_ops, n_trials=n_trials,
        var_repeatability=float(var_rep), var_reproducibility=float(var_reprod_total),
        var_gage_rr=float(var_grr), var_part=float(var_part), var_total=float(var_total),
        grr_percent=grr_percent, part_percent=part_percent, ndc=ndc,
        acceptability=_acceptability(grr_percent, ndc), grr_percent_tolerance=grr_pct_tol,
        detail={
            "pct_contribution_grr": float(var_grr / var_total * 100) if var_total > 0 else 0.0,
            "std_gage_rr": float(std_grr), "std_part": float(std_part), "std_total": float(std_total),
            "balanced": balanced,
            "ndc_ok": ndc >= 5,
        },
    )


def gage_rr_range(parts, operators, measurements, tolerance: float | None = None) -> GageRRResult:
    parts = np.asarray(list(parts))
    operators = np.asarray(list(operators))
    y = np.asarray(list(measurements), dtype=float)
    uparts = list(dict.fromkeys(parts.tolist()))
    uops = list(dict.fromkeys(operators.tolist()))
    n_parts, n_ops = len(uparts), len(uops)
    n_trials = len(y) // (n_parts * n_ops) if (n_parts * n_ops) else 0

    ranges = []
    for p in uparts:
        for o in uops:
            cell = y[(parts == p) & (operators == o)]
            if cell.size > 1:
                ranges.append(cell.max() - cell.min())
    r_bar = float(np.mean(ranges)) if ranges else 0.0
    d2 = k.d2(max(n_trials, 2))
    ev = r_bar / d2 if d2 else 0.0

    op_means = _group_means(operators, y)
    r_ops = max(op_means.values()) - min(op_means.values()) if op_means else 0.0
    av = np.sqrt(max((r_ops / d2) ** 2 - (ev ** 2 / (n_parts * max(n_trials, 1))), 0.0))

    part_means = _group_means(parts, y)
    r_parts = max(part_means.values()) - min(part_means.values()) if part_means else 0.0
    pv = r_parts / d2 if d2 else 0.0

    grr = np.sqrt(ev ** 2 + av ** 2)
    tv = np.sqrt(grr ** 2 + pv ** 2)
    grr_percent = float(grr / tv * 100) if tv > 0 else 0.0
    part_percent = float(pv / tv * 100) if tv > 0 else 0.0
    ndc = int(np.floor(np.sqrt(2) * pv / grr)) if grr > 0 else 0

    return GageRRResult(
        method="Range", n_parts=n_parts, n_operators=n_ops, n_trials=n_trials,
        var_repeatability=float(ev ** 2), var_reproducibility=float(av ** 2),
        var_gage_rr=float(grr ** 2), var_part=float(pv ** 2), var_total=float(tv ** 2),
        grr_percent=grr_percent, part_percent=part_percent, ndc=ndc,
        acceptability=_acceptability(grr_percent, ndc),
        grr_percent_tolerance=float(6 * grr / tolerance * 100) if tolerance else None,
        detail={"EV": float(ev), "AV": float(av), "PV": float(pv), "ndc_ok": ndc >= 5},
    )


@dataclass
class BiasResult:
    mean_bias: float
    std_bias: float
    percent_bias: float
    t_statistic: float
    p_value: float
    is_significant: bool
    n: int


def bias_study(measurements, references) -> BiasResult:
    m = np.asarray(list(measurements), dtype=float)
    r = np.asarray(list(references), dtype=float)
    bias = m - r
    mean_bias = float(bias.mean())
    std_bias = float(bias.std(ddof=1))
    t_stat, p = stats.ttest_1samp(bias, 0.0)
    ref_mean = float(r.mean())
    return BiasResult(
        mean_bias=mean_bias, std_bias=std_bias,
        percent_bias=float(mean_bias / ref_mean * 100) if ref_mean else 0.0,
        t_statistic=float(t_stat), p_value=float(p), is_significant=bool(p < 0.05), n=int(m.size),
    )


@dataclass
class LinearityResult:
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    std_error: float
    is_linear: bool


def linearity_study(measurements, references) -> LinearityResult:
    m = np.asarray(list(measurements), dtype=float)
    r = np.asarray(list(references), dtype=float)
    slope, intercept, rval, p, se = stats.linregress(r, m - r)
    return LinearityResult(
        slope=float(slope), intercept=float(intercept), r_squared=float(rval ** 2),
        p_value=float(p), std_error=float(se), is_linear=bool(abs(slope) < 0.1),
    )


@dataclass
class StabilityResult:
    mean: float
    std_dev: float
    ucl: float
    lcl: float
    out_of_control_points: int
    has_trend: bool
    is_stable: bool
    n: int


def stability_study(measurements) -> StabilityResult:
    m = np.asarray(list(measurements), dtype=float)
    mean = float(m.mean())
    std = float(m.std(ddof=1))
    mr_bar = float(np.mean(np.abs(np.diff(m)))) if m.size > 1 else 0.0
    ucl = mean + k.E2_MR * mr_bar
    lcl = mean - k.E2_MR * mr_bar
    ooc = int(np.sum((m > ucl) | (m < lcl)))

    n = m.size
    s = 0
    for i in range(n - 1):
        s += int(np.sum(np.sign(m[i + 1:] - m[i])))
    has_trend = abs(s) > (n * (n - 1) / 4)

    return StabilityResult(
        mean=mean, std_dev=std, ucl=ucl, lcl=lcl, out_of_control_points=ooc,
        has_trend=bool(has_trend), is_stable=bool(ooc == 0 and not has_trend), n=int(n),
    )


def _group_means(keys, values) -> dict:
    groups = _groups(keys.tolist() if hasattr(keys, "tolist") else list(keys), values)
    return {kk: float(v.mean()) for kk, v in groups.items()}
