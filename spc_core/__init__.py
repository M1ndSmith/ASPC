"""ASPC statistical process control core library.

Pure computation — no I/O, no FastAPI, no LLM. Import what you need:

    from spc_core import analyze_control_chart, capability_analysis, gage_rr_anova
    from spc_core import establish, phase1_checklist
"""
from .capability import (
    CapabilityResult,
    capability_analysis,
    dpmo_to_sigma,
    nonparametric_capability,
    parametric_capability,
    sigma_to_dpmo,
)
from .charts import ControlChartResult, analyze_control_chart, select_chart_type
from .cleaning import CleaningResult, classify_missing, range_check
from .cusum import CUSUMResult, cusum_chart
from .evaluator import Phase2Evaluator, evaluate_batch
from .ewma import EWMAResult, ewma_chart
from .ingest import ColumnMap, IngestedFrame, detect_columns, ingest, validate_frame
from .limits import (
    c_limits,
    imr_limits,
    np_limits,
    p_limits,
    u_limits,
    xbar_r_limits,
    xbar_s_limits,
)
from .models import (
    ChartType,
    ControlLimits,
    DataType,
    DistributionFlag,
    LimitSet,
    Phase,
    QualityFlag,
    Signal,
    SPCRecord,
)
from .msa import (
    BiasResult,
    GageRRResult,
    LinearityResult,
    StabilityResult,
    bias_study,
    gage_resolution_gate,
    gage_rr_anova,
    gage_rr_range,
    linearity_study,
    ndc_gate,
    stability_study,
)
from .msa_stream import CalibrationAlert, ContinuousMSA, ContinuousMSAState
from .multimodal import MultimodalResult, check_multimodal
from .normality import (
    AutocorrelationResult,
    NormalityResult,
    TransformResult,
    apply_transform,
    check_autocorrelation,
    check_normality,
)
from .pipeline import Gate, PipelineResult, checklist_ready_for_golive, establish, phase1_checklist
from .report import CapabilityReport, MSAReport, SPCReport
from .rules import RuleEngine, evaluate_series

__version__ = "2.0.0"

__all__ = [
    "AutocorrelationResult",
    "BiasResult",
    "CUSUMResult",
    "CalibrationAlert",
    "CapabilityReport",
    "CapabilityResult",
    "ChartType",
    "CleaningResult",
    "ColumnMap",
    "ContinuousMSA",
    "ContinuousMSAState",
    "ControlChartResult",
    "ControlLimits",
    "DataType",
    "DistributionFlag",
    "EWMAResult",
    "Gate",
    "GageRRResult",
    "IngestedFrame",
    "LimitSet",
    "LinearityResult",
    "MSAReport",
    "MultimodalResult",
    "NormalityResult",
    "Phase",
    "Phase2Evaluator",
    "PipelineResult",
    "QualityFlag",
    "RuleEngine",
    "SPCRecord",
    "SPCReport",
    "Signal",
    "StabilityResult",
    "TransformResult",
    "analyze_control_chart",
    "apply_transform",
    "bias_study",
    "c_limits",
    "capability_analysis",
    "check_autocorrelation",
    "check_multimodal",
    "check_normality",
    "classify_missing",
    "cusum_chart",
    "detect_columns",
    "dpmo_to_sigma",
    "establish",
    "evaluate_batch",
    "evaluate_series",
    "ewma_chart",
    "gage_resolution_gate",
    "gage_rr_anova",
    "gage_rr_range",
    "imr_limits",
    "ingest",
    "linearity_study",
    "ndc_gate",
    "nonparametric_capability",
    "np_limits",
    "p_limits",
    "parametric_capability",
    "phase1_checklist",
    "checklist_ready_for_golive",
    "range_check",
    "select_chart_type",
    "sigma_to_dpmo",
    "stability_study",
    "u_limits",
    "validate_frame",
    "xbar_r_limits",
    "xbar_s_limits",
]
