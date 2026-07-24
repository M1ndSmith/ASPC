"""Phase II evaluation: apply frozen Phase I limits to new observations.

The evaluator never recomputes limits. It consumes an immutable :class:`ControlLimits`
and streams observations through the stateful :class:`RuleEngine`. The same object is
used for a live stream or a replayed batch; batch is just "replay every row".

Chart dispatch:
* I-MR / NP / C  -> ``observe(value)`` (scalar plotted statistic, fixed limits).
* Xbar-R / Xbar-S -> ``observe_subgroup(values)`` (plots the subgroup mean).
* P / U          -> ``observe(value)`` where value is the proportion / rate; limits
  vary per point, so only the beyond-limit and one-sided-run rules apply.
"""
from __future__ import annotations

from .models import ChartType, ControlLimits, Signal
from .rules import RuleEngine

_SUBGROUP_CHARTS = {ChartType.XBAR_R, ChartType.XBAR_S}
_VARIABLE_LIMIT_CHARTS = {ChartType.P, ChartType.U}


class Phase2Evaluator:
    def __init__(self, limits: ControlLimits, ruleset: str = "nelson"):
        self.limits = limits
        self.ruleset = ruleset
        self._i = -1
        self._prev_side = 0
        self._side_run = 0

        primary = limits.primary
        center = primary.center
        # Prefer the stored sigma; else derive from the (unclamped upper) 3-sigma limit.
        if limits.sigma and limits.sigma > 0:
            sigma = limits.sigma
        else:
            ucl = primary.ucl if not isinstance(primary.ucl, list) else primary.ucl[0]
            sigma = (ucl - center) / 3.0 if ucl is not None else 0.0
        self._center = center
        self._sigma = sigma
        self._variable = limits.chart_type in _VARIABLE_LIMIT_CHARTS
        self._engine = None if self._variable else RuleEngine(center, sigma, ruleset)

    @property
    def is_subgroup_chart(self) -> bool:
        return self.limits.chart_type in _SUBGROUP_CHARTS

    def observe(self, value: float) -> list[Signal]:
        if self.is_subgroup_chart:
            raise ValueError(
                f"{self.limits.chart_type.value} plots subgroup means; use observe_subgroup()."
            )
        self._i += 1
        if self._variable:
            return self._observe_variable(value)
        return self._engine.add(float(value))

    def observe_subgroup(self, values) -> list[Signal]:
        if not self.is_subgroup_chart:
            raise ValueError(
                f"{self.limits.chart_type.value} is not a subgroup chart; use observe()."
            )
        self._i += 1
        mean = float(sum(values) / len(values))
        return self._engine.add(mean)

    def _observe_variable(self, value: float) -> list[Signal]:
        """P/U charts: per-point limits, so evaluate rule 1 and one-sided runs only."""
        comp = self.limits.primary
        i = self._i
        ucl = comp.ucl_at(i) if isinstance(comp.ucl, list) and i < len(comp.ucl) else (
            comp.ucl if not isinstance(comp.ucl, list) else comp.ucl[-1]
        )
        lcl = comp.lcl_at(i) if isinstance(comp.lcl, list) and i < len(comp.lcl) else (
            comp.lcl if not isinstance(comp.lcl, list) else comp.lcl[-1]
        )
        out: list[Signal] = []
        # Inclusive: a point exactly on UCL/LCL is out of control.
        if value >= ucl:
            out.append(Signal(rule_id="1", rule_name="Beyond control limits", index=i,
                              value=float(value), description="Point above the upper control limit",
                              side="upper"))
        elif value <= lcl:
            out.append(Signal(rule_id="1", rule_name="Beyond control limits", index=i,
                              value=float(value), description="Point below the lower control limit",
                              side="lower"))

        side = 1 if value > comp.center else -1 if value < comp.center else 0
        if side != 0 and side == self._prev_side:
            self._side_run += 1
        else:
            self._side_run = 1 if side != 0 else 0
        self._prev_side = side
        if self._side_run >= 9:
            out.append(Signal(rule_id="2", rule_name="Run on one side", index=i,
                              value=float(value),
                              description="Nine points in a row on the same side of the center line"))
        return out


def evaluate_batch(limits: ControlLimits, plotted_values, ruleset: str = "nelson") -> list[Signal]:
    """Replay already-plotted statistics (individuals, means, proportions) through Phase II."""
    ev = Phase2Evaluator(limits, ruleset=ruleset)
    signals: list[Signal] = []
    if ev.is_subgroup_chart:
        raise ValueError("Use evaluate_batch only for scalar-plotted charts.")
    for v in plotted_values:
        signals.extend(ev.observe(float(v)))
    return signals
