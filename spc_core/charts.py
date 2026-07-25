"""Control-chart selection and the batch analysis orchestrator.

Selection matrix (continuous data):
    n == 1        -> I-MR
    2 <= n <= 8   -> Xbar-R
    n >= 9        -> Xbar-S

Attribute data:
    defectives, variable n   -> P
    defectives, fixed n      -> NP
    counts, variable area    -> U
    counts, fixed area       -> C

Also supports EWMA and CUSUM (explicit chart_type or pipeline routing).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import limits as L
from . import rules as R
from .cusum import cusum_chart
from .evaluator import Phase2Evaluator
from .ewma import ewma_chart
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


def detect_data_type(values, sample_size_col_present=False, opportunity_col_present=False,
                     subgroup_present=False) -> DataType:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        raise ValueError("No numeric values to analyze")

    is_binary = bool(np.all(np.isin(arr, [0, 1])))
    is_integer = bool(np.all(arr == np.floor(arr)))
    is_non_negative = bool(np.all(arr >= 0))

    if sample_size_col_present or opportunity_col_present or is_binary:
        return DataType.ATTRIBUTE
    if is_integer and is_non_negative and arr.max() < 50 and not subgroup_present:
        return DataType.ATTRIBUTE
    if not is_integer:
        return DataType.CONTINUOUS
    unique_ratio = len(np.unique(arr)) / arr.size
    return DataType.CONTINUOUS if unique_ratio > 0.3 else DataType.ATTRIBUTE


def select_chart_type(data_type: DataType, subgroup_size: int = 1,
                      attribute_defectives: bool = True,
                      variable_size: bool = False) -> ChartType:
    if data_type == DataType.CONTINUOUS:
        if subgroup_size == 1:
            return ChartType.I_MR
        if subgroup_size <= 8:
            return ChartType.XBAR_R
        return ChartType.XBAR_S
    if attribute_defectives:
        return ChartType.P if variable_size else ChartType.NP
    return ChartType.U if variable_size else ChartType.C


@dataclass
class ControlChartResult:
    chart_type: ChartType
    data_type: DataType
    subgroup_size: int
    limits: ControlLimits
    plotted_values: list[float]
    secondary_values: list[float] | None = None
    secondary_name: str | None = None
    signals: list[Signal] = field(default_factory=list)
    secondary_signals: list[Signal] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    distribution_flag: DistributionFlag = DistributionFlag.NORMAL
    transform_applied: str | None = None
    phase: Phase = Phase.PHASE_I

    @property
    def out_of_control_count(self) -> int:
        idxs = {s.index for s in self.signals} | {s.index for s in self.secondary_signals}
        return len(idxs)

    def to_records(
        self,
        *,
        timestamps=None,
        subgroup_ids=None,
        quality_flags=None,
        gage_id: str | None = None,
        machine_id: str | None = None,
    ) -> list[SPCRecord]:
        """Emit one SPCRecord per plotted point with limits, flags, and signals."""
        primary = self.limits.primary
        by_idx: dict[int, list[Signal]] = {}
        for s in self.signals:
            by_idx.setdefault(s.index, []).append(s)

        records: list[SPCRecord] = []
        for i, v in enumerate(self.plotted_values):
            ucl = primary.ucl_at(i) if hasattr(primary, "ucl_at") else (
                primary.ucl[i] if isinstance(primary.ucl, list) else primary.ucl
            )
            lcl = primary.lcl_at(i) if hasattr(primary, "lcl_at") else (
                primary.lcl[i] if isinstance(primary.lcl, list) else primary.lcl
            )
            qf = QualityFlag.ORIGINAL
            if quality_flags is not None and i < len(quality_flags):
                raw = quality_flags[i]
                qf = raw if isinstance(raw, QualityFlag) else QualityFlag(raw)
            ts = timestamps[i] if timestamps is not None and i < len(timestamps) else None
            sid = subgroup_ids[i] if subgroup_ids is not None and i < len(subgroup_ids) else i
            records.append(SPCRecord(
                timestamp=ts,
                subgroup_id=int(sid) if sid is not None else i,
                measurement_value=float(v),
                data_quality_flag=qf,
                distribution_flag=self.distribution_flag,
                transform_applied=self.transform_applied,
                phase=self.phase,
                ucl=float(ucl) if ucl is not None else None,
                lcl=float(lcl) if lcl is not None else None,
                centerline=float(primary.center),
                gage_id=gage_id,
                machine_id=machine_id,
                signals=by_idx.get(i, []),
            ))
        return records


def _secondary_signals(secondary_values, secondary_limits: LimitSet | None, ruleset: str) -> list[Signal]:
    if secondary_values is None or secondary_limits is None:
        return []
    center = secondary_limits.center
    ucl = secondary_limits.ucl if not isinstance(secondary_limits.ucl, list) else None
    if ucl is None:
        return []
    sigma = (float(ucl) - center) / 3.0
    # Wheeler / points-outside for secondary panels (R/MR/S zone tests rarely used).
    rs = "wheeler" if ruleset == "wheeler" else ruleset
    return R.evaluate_series(secondary_values, center=center, sigma=max(sigma, 0.0), ruleset=rs)


def analyze_control_chart(
    values,
    subgroup_ids=None,
    sample_sizes=None,
    opportunities=None,
    chart_type: ChartType | None = None,
    ruleset: str = "nelson",
    ewma_lambda: float = 0.2,
    ewma_L: float = 3.0,
    cusum_k: float = 0.5,
    cusum_h: float = 5.0,
    exclude_incomplete: bool = False,
) -> ControlChartResult:
    """Compute Phase I limits, plotted statistics, and run-rule signals for a batch."""
    arr = np.asarray(values, dtype=float)

    # ---- determine chart type ----
    if chart_type is None:
        dtype = detect_data_type(
            arr,
            sample_size_col_present=sample_sizes is not None,
            opportunity_col_present=opportunities is not None,
            subgroup_present=subgroup_ids is not None,
        )
        if dtype == DataType.CONTINUOUS:
            if subgroup_ids is not None:
                subs = L.build_subgroups(arr, subgroup_ids)
                n = int(np.median([len(s) for s in subs]))
            else:
                n = 1
            chart_type = select_chart_type(dtype, subgroup_size=n)
        else:
            if sample_sizes is not None:
                variable = len(set(np.asarray(sample_sizes).tolist())) > 1
                chart_type = select_chart_type(dtype, attribute_defectives=True, variable_size=variable)
            elif opportunities is not None:
                variable = len(set(np.asarray(opportunities).tolist())) > 1
                chart_type = select_chart_type(dtype, attribute_defectives=False, variable_size=variable)
            else:
                chart_type = select_chart_type(dtype, attribute_defectives=False, variable_size=False)

    secondary = None
    secondary_name = None
    secondary_limits = None
    subgroup_size = 1
    signals: list[Signal] = []

    # ---- EWMA / CUSUM ----
    if chart_type == ChartType.EWMA:
        clean = arr[~np.isnan(arr)]
        ew = ewma_chart(clean, lam=ewma_lambda, L=ewma_L)
        limits = ew.limits
        plotted = ew.z
        signals = ew.signals
        dtype = DataType.CONTINUOUS

    elif chart_type == ChartType.CUSUM:
        clean = arr[~np.isnan(arr)]
        cu = cusum_chart(clean, k=cusum_k, h=cusum_h)
        limits = cu.limits
        plotted = cu.c_plus  # primary panel; C- in secondary
        secondary = cu.c_minus
        secondary_name = "cusum_minus"
        signals = cu.signals
        dtype = DataType.CONTINUOUS

    elif chart_type == ChartType.I_MR:
        clean = arr[~np.isnan(arr)]
        limits = L.imr_limits(clean)
        plotted = clean.tolist()
        secondary = np.abs(np.diff(clean)).tolist()
        secondary_name = "moving_range"
        secondary_limits = limits.components.get("moving_range")
        dtype = DataType.CONTINUOUS

    elif chart_type in (ChartType.XBAR_R, ChartType.XBAR_S):
        subs = L.build_subgroups(arr, subgroup_ids)
        subgroup_size = int(np.median([len(s) for s in subs]))
        if chart_type == ChartType.XBAR_R:
            limits = L.xbar_r_limits(subs, exclude_incomplete=exclude_incomplete)
            secondary = [float(s.max() - s.min()) for s in subs]
            secondary_name = "range"
            secondary_limits = limits.components.get("range")
        else:
            limits = L.xbar_s_limits(subs, exclude_incomplete=exclude_incomplete)
            secondary = [float(s.std(ddof=1)) for s in subs]
            secondary_name = "s"
            secondary_limits = limits.components.get("s")
        plotted = [float(s.mean()) for s in subs]
        dtype = DataType.CONTINUOUS

    elif chart_type == ChartType.P:
        n_arr = np.asarray(sample_sizes, dtype=float)
        if np.any(n_arr <= 0):
            raise ValueError("P chart sample sizes must be > 0")
        limits = L.p_limits(arr, n_arr)
        plotted = (arr / n_arr).tolist()
        dtype = DataType.ATTRIBUTE

    elif chart_type == ChartType.NP:
        n_np = float(np.asarray(sample_sizes)[0]) if sample_sizes is not None else float(arr.size)
        limits = L.np_limits(arr, n_np)
        plotted = arr.tolist()
        subgroup_size = int(n_np)
        dtype = DataType.ATTRIBUTE

    elif chart_type == ChartType.C:
        limits = L.c_limits(arr)
        plotted = arr.tolist()
        dtype = DataType.ATTRIBUTE

    elif chart_type == ChartType.U:
        o = np.asarray(opportunities, dtype=float)
        if np.any(o <= 0):
            raise ValueError("U chart opportunities must be > 0")
        limits = L.u_limits(arr, o)
        plotted = (arr / o).tolist()
        dtype = DataType.ATTRIBUTE

    else:  # pragma: no cover
        raise ValueError(f"Unsupported chart type: {chart_type}")

    if limits is None:
        raise RuntimeError(f"Limits not established for chart type {chart_type}")

    # ---- run-rule signals (Shewhart paths) ----
    if chart_type not in (ChartType.EWMA, ChartType.CUSUM):
        primary = limits.primary
        # Variable-limit charts (P/U, variable-n Xbar): use Phase2Evaluator so
        # per-point UCL/LCL are honoured. Fixed-limit charts use RuleEngine.
        if (
            chart_type in (ChartType.P, ChartType.U)
            or isinstance(primary.ucl, list)
            or isinstance(primary.lcl, list)
        ):
            ev = Phase2Evaluator(limits, ruleset=ruleset)
            signals = []
            for v in plotted:
                signals.extend(ev.observe(float(v)))
        else:
            center = primary.center
            sigma = limits.sigma if (limits.sigma and limits.sigma > 0) else (
                (float(primary.ucl) - center) / 3.0
            )
            signals = R.evaluate_series(plotted, center=center, sigma=sigma, ruleset=ruleset)

    sec_signals = _secondary_signals(secondary, secondary_limits, ruleset)

    summary = {
        "n_points": len(plotted),
        "mean": float(np.mean(plotted)) if plotted else 0.0,
        "limits_version": limits.version,
    }

    return ControlChartResult(
        chart_type=chart_type, data_type=dtype,
        subgroup_size=subgroup_size, limits=limits, plotted_values=plotted,
        secondary_values=secondary, secondary_name=secondary_name, signals=signals,
        secondary_signals=sec_signals, summary=summary,
    )
