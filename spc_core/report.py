"""Pure data report models — no HTML, no I/O.

Adapters (render_plotly, persistence) consume these. Keeping reports as data means
the same analysis result can be rendered as HTML, JSON, or stored without re-running
the statistics.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from .capability import CapabilityResult
from .charts import ControlChartResult
from .models import ChartType, ControlLimits, Phase, Signal
from .msa import BiasResult, GageRRResult, LinearityResult, StabilityResult
from .normality import AutocorrelationResult, NormalityResult


class SPCReport(BaseModel):
    """Serializable SPC analysis report."""

    analysis_type: str = "control_chart"
    chart_type: ChartType
    phase: Phase = Phase.PHASE_I
    limits: ControlLimits
    plotted_values: list[float]
    secondary_values: Optional[list[float]] = None
    secondary_name: Optional[str] = None
    signals: list[Signal] = Field(default_factory=list)
    subgroup_size: int = 1
    summary: dict[str, Any] = Field(default_factory=dict)
    normality: Optional[dict[str, Any]] = None
    autocorrelation: Optional[dict[str, Any]] = None
    gates: Optional[list[dict[str, Any]]] = None
    checklist: Optional[dict[str, Any]] = None
    records: Optional[list[dict[str, Any]]] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_file: Optional[str] = None

    @classmethod
    def from_chart_result(
        cls,
        result: ControlChartResult,
        normality: Optional[NormalityResult] = None,
        autocorrelation: Optional[AutocorrelationResult] = None,
        source_file: Optional[str] = None,
        phase: Phase = Phase.PHASE_I,
        gates: Optional[list] = None,
        checklist: Optional[dict] = None,
        include_records: bool = False,
    ) -> "SPCReport":
        records = None
        if include_records:
            records = [r.model_dump(mode="json") for r in result.to_records()]
        return cls(
            chart_type=result.chart_type,
            phase=phase,
            limits=result.limits,
            plotted_values=result.plotted_values,
            secondary_values=result.secondary_values,
            secondary_name=result.secondary_name,
            signals=result.signals,
            subgroup_size=result.subgroup_size,
            summary={
                **result.summary,
                "out_of_control_count": result.out_of_control_count,
                "data_type": result.data_type.value,
                "distribution_flag": result.distribution_flag.value,
                "transform_applied": result.transform_applied,
            },
            normality=_normality_dict(normality),
            autocorrelation=_acf_dict(autocorrelation),
            gates=[
                {"step": g.step, "status": g.status, "reason": g.reason, "detail": g.detail}
                if hasattr(g, "step") else g
                for g in (gates or [])
            ] or None,
            checklist=checklist,
            records=records,
            source_file=source_file,
        )


class CapabilityReport(BaseModel):
    analysis_type: str = "capability"
    result: dict[str, Any]
    normality: Optional[dict[str, Any]] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_file: Optional[str] = None

    @classmethod
    def from_capability(
        cls,
        result: CapabilityResult,
        normality: Optional[NormalityResult] = None,
        source_file: Optional[str] = None,
    ) -> "CapabilityReport":
        return cls(
            result=_capability_dict(result),
            normality=_normality_dict(normality),
            source_file=source_file,
        )


class MSAReport(BaseModel):
    analysis_type: str = "msa"
    study_type: str
    result: dict[str, Any]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_file: Optional[str] = None

    @classmethod
    def from_gage_rr(cls, result: GageRRResult, source_file: Optional[str] = None) -> "MSAReport":
        return cls(
            study_type="Gage R&R",
            result={
                "method": result.method,
                "n_parts": result.n_parts,
                "n_operators": result.n_operators,
                "n_trials": result.n_trials,
                "grr_percent": result.grr_percent,
                "part_percent": result.part_percent,
                "ndc": result.ndc,
                "acceptability": result.acceptability,
                "grr_percent_tolerance": result.grr_percent_tolerance,
                "var_repeatability": result.var_repeatability,
                "var_reproducibility": result.var_reproducibility,
                "var_gage_rr": result.var_gage_rr,
                "var_part": result.var_part,
                "var_total": result.var_total,
                "detail": result.detail,
            },
            source_file=source_file,
        )

    @classmethod
    def from_bias(cls, result: BiasResult, source_file: Optional[str] = None) -> "MSAReport":
        return cls(
            study_type="Bias",
            result={
                "mean_bias": result.mean_bias,
                "std_bias": result.std_bias,
                "percent_bias": result.percent_bias,
                "t_statistic": result.t_statistic,
                "p_value": result.p_value,
                "is_significant": result.is_significant,
                "n": result.n,
            },
            source_file=source_file,
        )

    @classmethod
    def from_linearity(cls, result: LinearityResult, source_file: Optional[str] = None) -> "MSAReport":
        return cls(
            study_type="Linearity",
            result={
                "slope": result.slope,
                "intercept": result.intercept,
                "r_squared": result.r_squared,
                "p_value": result.p_value,
                "std_error": result.std_error,
                "is_linear": result.is_linear,
            },
            source_file=source_file,
        )

    @classmethod
    def from_stability(cls, result: StabilityResult, source_file: Optional[str] = None) -> "MSAReport":
        return cls(
            study_type="Stability",
            result={
                "mean": result.mean,
                "std_dev": result.std_dev,
                "ucl": result.ucl,
                "lcl": result.lcl,
                "out_of_control_points": result.out_of_control_points,
                "has_trend": result.has_trend,
                "is_stable": result.is_stable,
                "n": result.n,
            },
            source_file=source_file,
        )


def _normality_dict(n: Optional[NormalityResult]) -> Optional[dict[str, Any]]:
    if n is None:
        return None
    return {
        "is_normal": n.is_normal,
        "tests_passed": n.tests_passed,
        "total_tests": n.total_tests,
        "confidence": n.confidence,
        "shapiro_p": n.shapiro_p,
        "anderson_stat": n.anderson_stat,
        "skewness": n.skewness,
        "kurtosis": n.kurtosis,
        "recommendation": n.recommendation,
    }


def _acf_dict(a: Optional[AutocorrelationResult]) -> Optional[dict[str, Any]]:
    if a is None:
        return None
    return {
        "lag1": a.lag1,
        "threshold": a.threshold,
        "is_autocorrelated": a.is_autocorrelated,
        "recommendation": a.recommendation,
    }


def _capability_dict(r: CapabilityResult) -> dict[str, Any]:
    return {
        "n": r.n,
        "mean": r.mean,
        "usl": r.usl,
        "lsl": r.lsl,
        "target": r.target,
        "method": r.method,
        "sigma_within": r.sigma_within,
        "sigma_overall": r.sigma_overall,
        "Cp": r.cp,
        "Cpk": r.cpk,
        "Cpu": r.cpu,
        "Cpl": r.cpl,
        "Cpm": r.cpm,
        "Pp": r.pp,
        "Ppk": r.ppk,
        "Ppu": r.ppu,
        "Ppl": r.ppl,
        "observed_dpmo": r.observed_dpmo,
        "expected_dpmo": r.expected_dpmo,
        "z_bench": r.z_bench,
        "sigma_level": r.sigma_level,
        "yield_pct": r.yield_pct,
        "is_centered": r.is_centered,
        "offset_from_target": r.offset_from_target,
        "rating": r.rating,
        "notes": r.notes,
    }
