"""Phase II evaluation: apply frozen Phase I limits to new observations.

The evaluator never recomputes limits. It consumes an immutable :class:`ControlLimits`
and streams observations through the stateful :class:`RuleEngine`. The same object is
used for a live stream or a replayed batch; batch is just "replay every row".

Chart dispatch:
* I-MR / NP / C  -> ``observe(value)`` (scalar plotted statistic, fixed limits).
* Xbar-R / Xbar-S -> ``observe_subgroup(values)`` (plots the subgroup mean).
* P / U / variable-n Xbar -> ``observe(value)`` with per-point limits via ``ucl_at``.
* EWMA -> ``observe(value)`` updates the EWMA statistic and checks time-varying limits.
* CUSUM -> ``observe(value)`` updates tabular C+/C- against the decision interval.
"""
from __future__ import annotations

from typing import Optional

from .models import ChartType, ControlLimits, Signal
from .rules import RuleEngine

_SUBGROUP_CHARTS = {ChartType.XBAR_R, ChartType.XBAR_S}


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

        # Variable limits: P/U, or any chart whose UCL/LCL is a per-point list
        # (including variable-n Xbar and EWMA time-varying limits).
        self._variable = (
            limits.chart_type in {ChartType.P, ChartType.U}
            or isinstance(primary.ucl, list)
            or isinstance(primary.lcl, list)
        )
        self._ewma = limits.chart_type == ChartType.EWMA
        self._cusum = limits.chart_type == ChartType.CUSUM

        # EWMA state
        notes = limits.notes or {}
        self._lam = float(notes.get("lambda", 0.2)) if self._ewma else 0.2
        self._ewma_z = self._center  # z_0 = target
        self._sigma_process = float(notes.get("sigma_process", sigma)) if self._ewma else sigma
        self._L = float(notes.get("L", 3.0)) if self._ewma else 3.0

        # CUSUM state
        self._cusum_k = float(notes.get("k_abs", notes.get("k", 0.5) * (sigma or 1.0))) if self._cusum else 0.0
        self._cusum_h = float(notes.get("h_abs", primary.ucl if not isinstance(primary.ucl, list) else 0.0)) if self._cusum else 0.0
        self._cusum_target = float(notes.get("target", center)) if self._cusum else center
        self._c_plus = 0.0
        self._c_minus = 0.0

        if self._ewma or self._cusum or self._variable:
            self._engine: Optional[RuleEngine] = None
        else:
            self._engine = RuleEngine(center, sigma, ruleset)

    @property
    def index(self) -> int:
        """Current 0-based observation index (-1 before first observe)."""
        return self._i

    @property
    def is_subgroup_chart(self) -> bool:
        return self.limits.chart_type in _SUBGROUP_CHARTS and not self._variable

    def seed_state(
        self,
        *,
        index: int = -1,
        values: Optional[list[float]] = None,
    ) -> None:
        """Restore evaluator continuity after restart.

        ``index`` is the last observation index already seen (so the next
        ``observe`` continues at ``index + 1``). ``values`` are recent plotted
        statistics used to warm the rule buffer / EWMA / CUSUM state without
        emitting signals.
        """
        self._i = int(index)
        if not values:
            return
        # Warm internal state silently (no signal collection).
        saved_i = self._i
        # Replay from a temporary index so RuleEngine buffer fills correctly.
        if self._engine is not None:
            # Reset engine and replay
            self._engine = RuleEngine(self._center, self._sigma, self.ruleset)
            for v in values:
                self._engine.add(float(v))
            self._i = saved_i
        elif self._ewma:
            z = self._center
            for v in values:
                z = self._lam * float(v) + (1.0 - self._lam) * z
            self._ewma_z = z
        elif self._cusum:
            cp = cm = 0.0
            k_abs = self._cusum_k
            tgt = self._cusum_target
            for v in values:
                x = float(v)
                cp = max(0.0, x - (tgt + k_abs) + cp)
                cm = max(0.0, (tgt - k_abs) - x + cm)
            self._c_plus = cp
            self._c_minus = cm
        elif self._variable:
            # Warm one-sided run counter only
            for v in values:
                side = 1 if float(v) > self._center else -1 if float(v) < self._center else 0
                if side != 0 and side == self._prev_side:
                    self._side_run += 1
                else:
                    self._side_run = 1 if side != 0 else 0
                self._prev_side = side

    def observe(self, value: float) -> list[Signal]:
        if self.is_subgroup_chart:
            raise ValueError(
                f"{self.limits.chart_type.value} plots subgroup means; use observe_subgroup()."
            )
        self._i += 1
        if self._ewma:
            return self._observe_ewma(float(value))
        if self._cusum:
            return self._observe_cusum(float(value))
        if self._variable:
            return self._observe_variable(float(value))
        assert self._engine is not None
        return self._engine.add(float(value))

    def observe_subgroup(self, values) -> list[Signal]:
        if self.limits.chart_type not in _SUBGROUP_CHARTS:
            raise ValueError(
                f"{self.limits.chart_type.value} is not a subgroup chart; use observe()."
            )
        if not values:
            raise ValueError("observe_subgroup requires a non-empty subgroup")
        self._i += 1
        mean = float(sum(values) / len(values))
        if self._variable:
            return self._observe_variable(mean)
        assert self._engine is not None
        return self._engine.add(mean)

    def _observe_variable(self, value: float) -> list[Signal]:
        """Per-point limits (P/U, variable-n Xbar, EWMA list limits)."""
        comp = self.limits.primary
        i = self._i
        if isinstance(comp.ucl, list):
            ucl = comp.ucl_at(i) if i < len(comp.ucl) else comp.ucl[-1]
        else:
            ucl = comp.ucl
        if isinstance(comp.lcl, list):
            lcl = comp.lcl_at(i) if i < len(comp.lcl) else comp.lcl[-1]
        else:
            lcl = comp.lcl
        out: list[Signal] = []
        # Inclusive: a point exactly on UCL/LCL is out of control.
        if ucl is not None and value >= ucl:
            out.append(Signal(
                rule_id="1", rule_name="Beyond control limits", index=i,
                value=float(value), description="Point above the upper control limit",
                side="upper",
            ))
        elif lcl is not None and value <= lcl:
            out.append(Signal(
                rule_id="1", rule_name="Beyond control limits", index=i,
                value=float(value), description="Point below the lower control limit",
                side="lower",
            ))

        side = 1 if value > comp.center else -1 if value < comp.center else 0
        if side != 0 and side == self._prev_side:
            self._side_run += 1
        else:
            self._side_run = 1 if side != 0 else 0
        self._prev_side = side
        if self.ruleset != "wheeler" and self._side_run >= 9:
            out.append(Signal(
                rule_id="2", rule_name="Run on one side", index=i,
                value=float(value),
                description="Nine points in a row on the same side of the center line",
            ))
        return out

    def _observe_ewma(self, value: float) -> list[Signal]:
        """Update EWMA statistic and check against time-varying (or steady) limits."""
        i = self._i
        z = self._lam * value + (1.0 - self._lam) * self._ewma_z
        self._ewma_z = z

        primary = self.limits.primary
        if isinstance(primary.ucl, list) and i < len(primary.ucl):
            ucl = primary.ucl[i]
            lcl = primary.lcl[i] if isinstance(primary.lcl, list) else primary.lcl
        else:
            # Steady-state fallback using stored EWMA sigma
            import math
            factor = self._lam / (2.0 - self._lam)
            var_factor = factor * (1.0 - (1.0 - self._lam) ** (2 * (i + 1)))
            sigma_z = self._sigma_process * math.sqrt(max(var_factor, 0.0))
            ucl = self._center + self._L * sigma_z
            lcl = self._center - self._L * sigma_z

        out: list[Signal] = []
        if z >= ucl:
            out.append(Signal(
                rule_id="EWMA1", rule_name="Beyond EWMA UCL", index=i, value=z,
                description=f"EWMA statistic beyond UCL (λ={self._lam})", side="upper",
            ))
        elif z <= lcl:
            out.append(Signal(
                rule_id="EWMA1", rule_name="Beyond EWMA LCL", index=i, value=z,
                description=f"EWMA statistic beyond LCL (λ={self._lam})", side="lower",
            ))
        return out

    def _observe_cusum(self, value: float) -> list[Signal]:
        """Update tabular C+/C- against the frozen decision interval."""
        i = self._i
        k_abs = self._cusum_k
        h_abs = self._cusum_h
        tgt = self._cusum_target
        self._c_plus = max(0.0, value - (tgt + k_abs) + self._c_plus)
        self._c_minus = max(0.0, (tgt - k_abs) - value + self._c_minus)
        out: list[Signal] = []
        if self._c_plus >= h_abs:
            out.append(Signal(
                rule_id="CUSUM+", rule_name="CUSUM upper shift", index=i,
                value=self._c_plus,
                description=f"C+ exceeded h·σ={h_abs:.4g}", side="upper",
            ))
            self._c_plus = 0.0
        if self._c_minus >= h_abs:
            out.append(Signal(
                rule_id="CUSUM-", rule_name="CUSUM lower shift", index=i,
                value=self._c_minus,
                description=f"C- exceeded h·σ={h_abs:.4g}", side="lower",
            ))
            self._c_minus = 0.0
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
