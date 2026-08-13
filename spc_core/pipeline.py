"""Gated Phase I master pipeline (MVP S1 decision flow).

Order: MSA gate → ACF → normality → multimodal STOP → transform/Wheeler →
outlier classification → missing-value handling → chart → freeze/version limits.

Returns gates with status ok | warn | stop so callers can enforce go-live.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .charts import ControlChartResult, analyze_control_chart
from .cleaning import classify_missing, range_check
from .models import ChartType, DistributionFlag, Phase, QualityFlag
from .msa import GageRRResult, gage_resolution_gate, gage_rr_anova, ndc_gate
from .multimodal import MultimodalResult, check_multimodal
from .normality import (
    AutocorrelationResult,
    NormalityResult,
    TransformResult,
    apply_transform,
    check_autocorrelation,
    check_normality,
)

# Hard floor for computing any control limits at all (a moving range needs two points).
# This is *not* the Phase I adequacy threshold — that is phase1_checklist's job.
MIN_ESTABLISH_POINTS = 2


@dataclass
class Gate:
    step: str
    status: str  # "ok" | "warn" | "stop"
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    chart: ControlChartResult
    gates: list[Gate]
    normality: NormalityResult | None = None
    autocorrelation: AutocorrelationResult | None = None
    multimodal: MultimodalResult | None = None
    transform: TransformResult | None = None
    msa: GageRRResult | None = None
    stopped: bool = False
    frozen: bool = True
    chart_route: str = "shewhart"

    @property
    def limits_version(self) -> str:
        return self.chart.limits.version

    def gate_status(self, step: str) -> str | None:
        for g in self.gates:
            if g.step == step:
                return g.status
        return None


def establish(
    values,
    *,
    subgroup_ids=None,
    sample_sizes=None,
    opportunities=None,
    chart_type: ChartType | None = None,
    ruleset: str = "nelson",
    acf_threshold: float = 0.2,
    # Optional MSA inputs — when provided, MSA gate runs.
    msa_parts=None,
    msa_operators=None,
    msa_measurements=None,
    msa_tolerance: float | None = None,
    gage_resolution: float | None = None,
    # Missing-value reasons aligned with values (optional).
    missing_reasons=None,
    # Physical (low, high) measurement range; out-of-range points are measurement failures.
    valid_range: tuple[float, float] | None = None,
    # Prefer EWMA over CUSUM when autocorrelated.
    autocorrelated_chart: str = "EWMA",
    force_wheeler: bool = False,
) -> PipelineResult:
    """Run the gated Phase I pipeline and return chart + gates."""
    gates: list[Gate] = []
    arr = np.asarray(values, dtype=float)
    if arr.size < MIN_ESTABLISH_POINTS:
        raise ValueError(
            f"establish() requires at least {MIN_ESTABLISH_POINTS} observations to "
            f"compute control limits; received {arr.size}."
        )
    working = arr.copy()
    dist_flag = DistributionFlag.NORMAL
    transform_applied = None
    transform: TransformResult | None = None
    normality: NormalityResult | None = None
    acf: AutocorrelationResult | None = None
    multimodal: MultimodalResult | None = None
    msa_result: GageRRResult | None = None
    route = "shewhart"
    active_ruleset = ruleset
    active_chart = chart_type
    # Determined up front from the caller's intent, because active_chart is mutated below.
    # Attribute (count) data is binomial/Poisson, so the continuous-data machinery —
    # Gaussian normality, Hartigan's dip test, and the EWMA/CUSUM reroute — does not
    # apply to it.
    is_attribute = (
        chart_type in (ChartType.P, ChartType.NP, ChartType.C, ChartType.U)
        or sample_sizes is not None
        or opportunities is not None
    )

    # ---- 1. MSA gate ----
    if msa_measurements is not None and msa_parts is not None and msa_operators is not None:
        msa_result = gage_rr_anova(
            msa_parts, msa_operators, msa_measurements, tolerance=msa_tolerance,
        )
        ndc_ok, ndc_reason = ndc_gate(msa_result.ndc)
        if msa_result.grr_percent > 30 or not ndc_ok:
            gates.append(Gate(
                step="msa", status="stop",
                reason=(
                    f"MSA unacceptable: %GRR={msa_result.grr_percent:.1f}, "
                    f"NDC={msa_result.ndc}. {ndc_reason}"
                ),
                detail={"grr_percent": msa_result.grr_percent, "ndc": msa_result.ndc},
            ))
            # Still produce a chart for diagnostics, but mark stopped.
        elif msa_result.grr_percent >= 10:
            gates.append(Gate(
                step="msa", status="warn",
                reason=f"MSA conditional: %GRR={msa_result.grr_percent:.1f} (10–30%). Document risk.",
                detail={"grr_percent": msa_result.grr_percent, "ndc": msa_result.ndc},
            ))
        else:
            gates.append(Gate(
                step="msa", status="ok",
                reason=f"MSA acceptable: %GRR={msa_result.grr_percent:.1f}, NDC={msa_result.ndc}.",
                detail={"grr_percent": msa_result.grr_percent, "ndc": msa_result.ndc},
            ))
        if gage_resolution is not None and msa_tolerance is not None:
            ok, reason = gage_resolution_gate(gage_resolution, msa_tolerance)
            gates.append(Gate(
                step="gage_resolution", status="ok" if ok else "stop", reason=reason,
            ))
    else:
        gates.append(Gate(
            step="msa", status="warn",
            reason="MSA inputs not provided; proceeding without measurement-system gate.",
        ))

    # ---- 1b. Physical range gate ----
    # An out-of-range reading (e.g. a -999 disconnected-sensor sentinel) is a measurement
    # failure, not process variation. Blank it here so it cannot be charted as an OOC
    # signal; the missing-value step below then classifies it as MISSING_SENSOR.
    if valid_range is not None:
        low, high = float(valid_range[0]), float(valid_range[1])
        out_of_range = [
            i for i, ok in enumerate(range_check(working, low, high))
            if not ok and not np.isnan(working[i])
        ]
        if out_of_range:
            reasons = (
                list(missing_reasons) if missing_reasons is not None
                else [None] * int(working.size)
            )
            for i in out_of_range:
                working[i] = np.nan
                reasons[i] = "sensor"
            missing_reasons = reasons
        gates.append(Gate(
            step="range",
            status="warn" if out_of_range else "ok",
            reason=(
                f"{len(out_of_range)} reading(s) outside physical range "
                f"[{low}, {high}] blanked as measurement failures."
                if out_of_range
                else f"All readings within physical range [{low}, {high}]."
            ),
            detail={"n_out_of_range": len(out_of_range), "low": low, "high": high},
        ))

    # ---- 2. Missing-value classification ----
    if missing_reasons is not None or np.any(np.isnan(working)):
        result = classify_missing(working, reasons=missing_reasons)
        flags = result.flags
        n_imputed = sum(1 for f in flags if f == QualityFlag.IMPUTED_LOCF)
        n_excluded = sum(
            1 for f in flags
            if f in (QualityFlag.EXCLUDED_MAINTENANCE, QualityFlag.EXCLUDED_INCOMPLETE,
                     QualityFlag.MISSING_SENSOR, QualityFlag.MISSING_HUMAN)
        )
        working = np.asarray(
            [float(v) if v is not None else np.nan for v in result.values],
            dtype=float,
        )
        gates.append(Gate(
            step="missing", status="ok" if n_excluded == 0 else "warn",
            reason=f"Missing classified: imputed={n_imputed}, excluded={n_excluded}.",
            detail={"imputed": n_imputed, "excluded": n_excluded},
        ))
    else:
        gates.append(Gate(step="missing", status="ok", reason="No missing values."))

    clean = working[~np.isnan(working)]
    if clean.size < MIN_ESTABLISH_POINTS:
        raise ValueError(
            f"establish() requires at least {MIN_ESTABLISH_POINTS} usable observations "
            f"to compute control limits; only {clean.size} of {arr.size} values remain "
            "after missing-value classification. Whether there are *enough* points for "
            "Phase I is judged separately by phase1_checklist()."
        )

    # ---- 3. Autocorrelation ----
    acf = check_autocorrelation(clean, threshold=acf_threshold)
    if acf.is_autocorrelated and is_attribute:
        # Flag it, but keep the attribute chart: an EWMA on raw counts would throw away
        # the per-point binomial/Poisson limits that P and U charts depend on.
        gates.append(Gate(
            step="autocorrelation", status="warn",
            reason=(
                f"Autocorrelation detected (lag-1={acf.lag1:.3f}) in attribute data. "
                "Investigate serial dependence in the count process; the chart stays "
                "on binomial/Poisson limits rather than rerouting to EWMA/CUSUM."
            ),
            detail={"lag1": acf.lag1, "route": "attribute", "reroute_suppressed": True},
        ))
    elif acf.is_autocorrelated:
        route = autocorrelated_chart.upper()
        active_chart = ChartType.EWMA if route == "EWMA" else ChartType.CUSUM
        gates.append(Gate(
            step="autocorrelation", status="warn",
            reason=acf.recommendation,
            detail={"lag1": acf.lag1, "route": route},
        ))
    else:
        gates.append(Gate(
            step="autocorrelation", status="ok", reason=acf.recommendation,
            detail={"lag1": acf.lag1},
        ))

    # ---- 4. Normality + multimodal (only if not already routed to EWMA/CUSUM) ----
    if is_attribute:
        gates.append(Gate(
            step="normality", status="ok",
            reason=(
                "Attribute (count) data: binomial/Poisson limits apply, so the normality "
                "and dip tests are not applicable and were skipped."
            ),
            detail={"skipped": True, "reason_code": "attribute_data"},
        ))
    elif active_chart not in (ChartType.EWMA, ChartType.CUSUM):
        multimodal = check_multimodal(clean)
        if multimodal.is_multimodal:
            gates.append(Gate(
                step="multimodal", status="stop", reason=multimodal.recommendation,
                detail={"dip": multimodal.dip_statistic, "p": multimodal.p_value},
            ))
        else:
            gates.append(Gate(
                step="multimodal", status="ok", reason=multimodal.recommendation,
            ))

        normality = check_normality(clean)
        if normality.is_normal:
            gates.append(Gate(
                step="normality", status="ok", reason=normality.recommendation,
            ))
            dist_flag = DistributionFlag.NORMAL
        else:
            transform = apply_transform(clean, method="auto")
            if transform.became_normal:
                clean = transform.values
                working = transform.values
                # Transform may drop NaNs / change length — clear parallel
                # subgroup metadata so analyze_control_chart lengths match.
                if subgroup_ids is not None and len(subgroup_ids) != len(clean):
                    subgroup_ids = None
                if sample_sizes is not None and len(sample_sizes) != len(clean):
                    sample_sizes = None
                if opportunities is not None and len(opportunities) != len(clean):
                    opportunities = None
                dist_flag = DistributionFlag.TRANSFORMED
                transform_applied = transform.label
                gates.append(Gate(
                    step="normality", status="ok",
                    reason=f"Non-normal; transform {transform.label} restored normality.",
                    detail={"transform": transform.applied, "lambda": transform.lam},
                ))
            else:
                # Non-normal after transform: Wheeler robust path (points-outside only).
                # Never flatten subgrouped Phase I data to I-MR — that destroys
                # within-subgroup variance structure. Keep Xbar-R/S and only switch
                # the ruleset. force_wheeler forces I-MR only when there are no subgroups.
                active_ruleset = "wheeler"
                dist_flag = DistributionFlag.NON_NORMAL_RAW
                route = "wheeler"
                if subgroup_ids is not None:
                    chart_note = "subgroup chart preserved; Wheeler ruleset only"
                elif force_wheeler or active_chart is None:
                    active_chart = ChartType.I_MR
                    chart_note = "I-MR, points-outside-limits only"
                else:
                    chart_note = f"{active_chart.value}, Wheeler ruleset"
                gates.append(Gate(
                    step="normality", status="warn",
                    reason=(
                        "Non-normal after transform; using Wheeler robust path "
                        f"({chart_note})."
                    ),
                    detail={
                        "transform_tried": transform.applied if transform else None,
                        "force_wheeler": force_wheeler,
                        "subgroup_preserved": subgroup_ids is not None,
                    },
                ))

    # ---- 5. Chart (always produced for diagnostics, even when STOP gates fired) ----
    chart = analyze_control_chart(
        clean if dist_flag == DistributionFlag.TRANSFORMED else working[~np.isnan(working)],
        subgroup_ids=subgroup_ids,
        sample_sizes=sample_sizes,
        opportunities=opportunities,
        chart_type=active_chart,
        ruleset=active_ruleset,
    )
    chart.distribution_flag = dist_flag
    chart.transform_applied = transform_applied
    chart.phase = Phase.PHASE_I

    gates.append(Gate(
        step="chart", status="ok",
        reason=f"Chart {chart.chart_type.value} established; limits version {chart.limits.version}.",
        detail={
            "chart_type": chart.chart_type.value,
            "limits_version": chart.limits.version,
            "n_points": len(chart.plotted_values),
            "ooc": chart.out_of_control_count,
            "route": route,
        },
    ))

    # ---- 6. Freeze gate — STOP blocks freezing ----
    stopped = any(g.status == "stop" for g in gates)
    frozen = not stopped
    if frozen:
        gates.append(Gate(
            step="freeze", status="ok",
            reason=f"Phase I limits frozen with version hash {chart.limits.version}.",
            detail={"limits_version": chart.limits.version, "frozen": True},
        ))
    else:
        stop_steps = [g.step for g in gates if g.status == "stop"]
        gates.append(Gate(
            step="freeze", status="blocked",
            reason=(
                "Phase I limits NOT frozen: STOP gate(s) fired "
                f"({', '.join(stop_steps)}). Chart is diagnostic only."
            ),
            detail={
                "limits_version": chart.limits.version,
                "frozen": False,
                "stop_steps": stop_steps,
            },
        ))

    return PipelineResult(
        chart=chart, gates=gates, normality=normality, autocorrelation=acf,
        multimodal=multimodal, transform=transform, msa=msa_result,
        stopped=stopped, frozen=frozen, chart_route=route,
    )


def phase1_checklist(
    pipeline: PipelineResult,
    *,
    min_subgroups: int = 25,
    gage_calibration_ok: bool = True,
    phase2_enabled: bool = False,
    outliers_investigated: bool = True,
) -> dict[str, Any]:
    """MVP S8 Phase I go-live checklist — 10 items, pass/fail with reasons."""
    items: list[dict[str, Any]] = []

    def _add(name: str, passed: bool, reason: str) -> None:
        items.append({"item": name, "passed": passed, "reason": reason})

    # 1. MSA
    msa_gate = next((g for g in pipeline.gates if g.step == "msa"), None)
    msa_ok = msa_gate is not None and msa_gate.status == "ok"
    if pipeline.msa is not None:
        _add(
            "msa_grr_ndc",
            pipeline.msa.grr_percent < 10 and pipeline.msa.ndc >= 5,
            f"%GRR={pipeline.msa.grr_percent:.1f}, NDC={pipeline.msa.ndc}",
        )
    else:
        _add("msa_grr_ndc", False, "MSA not run — required before go-live.")

    # 2. Normality / distribution path
    norm_gate = next((g for g in pipeline.gates if g.step == "normality"), None)
    _add(
        "normality_path",
        norm_gate is None or norm_gate.status in ("ok", "warn"),
        norm_gate.reason if norm_gate else "Skipped (EWMA/CUSUM route).",
    )

    # 3. ACF
    acf_gate = next((g for g in pipeline.gates if g.step == "autocorrelation"), None)
    _add(
        "autocorrelation",
        acf_gate is not None and acf_gate.status in ("ok", "warn"),
        acf_gate.reason if acf_gate else "ACF not checked.",
    )

    # 4. Outliers investigated
    _add(
        "outliers_investigated",
        outliers_investigated,
        "Outliers investigated with root-cause notes."
        if outliers_investigated else "Outliers not yet investigated.",
    )

    # 5. Missing data classified
    miss_gate = next((g for g in pipeline.gates if g.step == "missing"), None)
    _add(
        "missing_classified",
        miss_gate is not None and miss_gate.status in ("ok", "warn"),
        miss_gate.reason if miss_gate else "Missing-value step not run.",
    )

    # 6. Min 25 subgroups / points
    n_pts = len(pipeline.chart.plotted_values)
    _add(
        "min_subgroups",
        n_pts >= min_subgroups,
        f"{n_pts} points/subgroups (need >= {min_subgroups}).",
    )

    # 7. Limits frozen with version hash (blocked when STOP gates fired)
    _add(
        "limits_frozen",
        pipeline.frozen and bool(pipeline.chart.limits.version),
        (
            f"Limits version {pipeline.chart.limits.version}."
            if pipeline.frozen
            else "Limits not frozen — STOP gate(s) blocked Phase I freeze."
        ),
    )

    # 8. Transform documented
    if pipeline.chart.distribution_flag.value == "TRANSFORMED":
        _add(
            "transform_documented",
            bool(pipeline.chart.transform_applied),
            f"Transform: {pipeline.chart.transform_applied}",
        )
    else:
        _add("transform_documented", True, "No transform applied (or Wheeler/EWMA path).")

    # 9. Gage calibration
    _add(
        "gage_calibration",
        gage_calibration_ok,
        "Gage calibration within active interval."
        if gage_calibration_ok else "Gage calibration expired or unknown.",
    )

    # 10. Phase II monitoring enabled
    _add(
        "phase2_enabled",
        phase2_enabled,
        "Phase II monitoring enabled."
        if phase2_enabled else "Phase II not yet enabled — freeze & go-live required.",
    )

    # Multimodal stop
    mm = next((g for g in pipeline.gates if g.step == "multimodal"), None)
    if mm and mm.status == "stop":
        _add("stratification", False, mm.reason)

    all_pass = all(i["passed"] for i in items)
    return {
        "passed": all_pass,
        "items": items,
        "limits_version": pipeline.chart.limits.version,
        "stopped": pipeline.stopped,
        "msa_ok": msa_ok,
    }


def checklist_ready_for_golive(checklist: dict[str, Any]) -> bool:
    """True when checklist items other than ``phase2_enabled`` pass.

    Analyze always runs the checklist with ``phase2_enabled=False``, so the
    aggregate ``passed`` flag stays False until go-live. Use this for the
    persisted go-live gate instead.
    """
    items = checklist.get("items") or []
    return all(
        bool(i.get("passed"))
        for i in items
        if i.get("item") != "phase2_enabled"
    )
