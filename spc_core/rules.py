"""Stateful Nelson / Western Electric run-rule engine.

The legacy engine only flagged points outside the control limits (Nelson rule 1). Real
SPC needs the pattern rules, which require *history*. This engine keeps a bounded ring
buffer of recent points and evaluates every rule against the window ending at the newest
point, so it works identically for a streamed point or a replayed batch.

Rule severities/windows follow the standard Nelson set (1-8). Western Electric is the
classic subset (rules 1, 5, 6, and a "n-in-a-row on one side" run test).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .models import Signal

# Nelson rule catalog: id -> (name, window length, description).
NELSON = {
    "1": ("Beyond 3-sigma", 1, "One point beyond zone A (>3 sigma from center)"),
    "2": ("Run on one side", 9, "Nine points in a row on the same side of the center line"),
    "3": ("Trend", 6, "Six points in a row steadily increasing or decreasing"),
    "4": ("Alternating", 14, "Fourteen points in a row alternating up and down"),
    "5": ("2 of 3 beyond 2-sigma", 3, "Two out of three consecutive points beyond 2 sigma (same side)"),
    "6": ("4 of 5 beyond 1-sigma", 5, "Four out of five consecutive points beyond 1 sigma (same side)"),
    "7": ("Stratification", 15, "Fifteen points in a row within 1 sigma (both sides)"),
    "8": ("Mixture", 8, "Eight points in a row beyond 1 sigma, none within zone C"),
}

WESTERN_ELECTRIC = {
    "WE1": ("Beyond 3-sigma", 1, "One point beyond 3 sigma"),
    "WE2": ("2 of 3 beyond 2-sigma", 3, "Two of three consecutive points beyond 2 sigma (same side)"),
    "WE3": ("4 of 5 beyond 1-sigma", 5, "Four of five consecutive points beyond 1 sigma (same side)"),
    "WE4": ("8 on one side", 8, "Eight points in a row on the same side of the center line"),
}


@dataclass
class _Pt:
    index: int
    value: float
    side: int          # +1 above center, -1 below, 0 on center
    zone: int          # number of sigmas away, floored (0,1,2,3+) using abs distance


class RuleEngine:
    """Incremental run-rule evaluator.

    Parameters
    ----------
    center, sigma : the frozen center line and 1-sigma width of the plotted statistic.
    ruleset : "nelson" (default) or "western_electric".
    """

    def __init__(self, center: float, sigma: float, ruleset: str = "nelson"):
        self.center = center
        self.sigma = sigma if sigma and sigma > 0 else 0.0
        # "nelson" | "western_electric" | "wheeler" (points-outside-limits only)
        self.ruleset = ruleset
        self._buf: deque[_Pt] = deque(maxlen=15)
        self._i = -1

    def _classify(self, value: float) -> _Pt:
        self._i += 1
        if value > self.center:
            side = 1
        elif value < self.center:
            side = -1
        else:
            side = 0
        if self.sigma > 0:
            z = abs(value - self.center) / self.sigma
        else:
            z = 0.0
        # Inclusive zone boundaries: a point exactly on 3σ is zone 3 (fires rule 1).
        zone = 3 if z >= 3 else 2 if z >= 2 else 1 if z >= 1 else 0
        return _Pt(index=self._i, value=value, side=side, zone=zone)

    def add(self, value: float) -> list[Signal]:
        pt = self._classify(value)
        self._buf.append(pt)
        if self.ruleset == "western_electric":
            return self._eval_we(pt)
        if self.ruleset == "wheeler":
            return self._eval_wheeler(pt)
        return self._eval_nelson(pt)

    def _eval_wheeler(self, pt: _Pt) -> list[Signal]:
        """Wheeler robust path: only points beyond 3σ (skip all zone/run tests)."""
        if pt.zone >= 3 and self.sigma > 0:
            return [self._sig("1", pt, side="upper" if pt.side > 0 else "lower")]
        return []

    # ---- Nelson -----------------------------------------------------------------
    def _eval_nelson(self, pt: _Pt) -> list[Signal]:
        b = list(self._buf)
        out: list[Signal] = []

        # Rule 1: beyond 3 sigma.
        if pt.zone >= 3 and self.sigma > 0:
            out.append(self._sig("1", pt, side="upper" if pt.side > 0 else "lower"))

        # Rule 2: 9 in a row same side.
        if self._run_same_side(b, 9):
            out.append(self._sig("2", pt))

        # Rule 3: 6 monotonic.
        if self._monotonic(b, 6):
            out.append(self._sig("3", pt))

        # Rule 4: 14 alternating.
        if self._alternating(b, 14):
            out.append(self._sig("4", pt))

        # Rule 5: 2 of 3 beyond 2 sigma, same side (window ends at current point).
        if self.sigma > 0 and self._k_of_m_beyond(b, k=2, m=3, zone=2):
            out.append(self._sig("5", pt))

        # Rule 6: 4 of 5 beyond 1 sigma, same side.
        if self.sigma > 0 and self._k_of_m_beyond(b, k=4, m=5, zone=1):
            out.append(self._sig("6", pt))

        # Rule 7: 15 within 1 sigma.
        if self.sigma > 0 and self._run_within(b, 15, zone_lt=1):
            out.append(self._sig("7", pt))

        # Rule 8: 8 in a row beyond 1 sigma (either side, none within zone C).
        if self.sigma > 0 and self._run_beyond(b, 8, zone_ge=1):
            out.append(self._sig("8", pt))

        return out

    # ---- Western Electric -------------------------------------------------------
    def _eval_we(self, pt: _Pt) -> list[Signal]:
        b = list(self._buf)
        out: list[Signal] = []
        if pt.zone >= 3 and self.sigma > 0:
            out.append(self._sig("WE1", pt, side="upper" if pt.side > 0 else "lower",
                                 catalog=WESTERN_ELECTRIC))
        if self.sigma > 0 and self._k_of_m_beyond(b, k=2, m=3, zone=2):
            out.append(self._sig("WE2", pt, catalog=WESTERN_ELECTRIC))
        if self.sigma > 0 and self._k_of_m_beyond(b, k=4, m=5, zone=1):
            out.append(self._sig("WE3", pt, catalog=WESTERN_ELECTRIC))
        if self._run_same_side(b, 8):
            out.append(self._sig("WE4", pt, catalog=WESTERN_ELECTRIC))
        return out

    # ---- window predicates ------------------------------------------------------
    @staticmethod
    def _run_same_side(b: list[_Pt], length: int) -> bool:
        if len(b) < length:
            return False
        tail = b[-length:]
        first = tail[0].side
        return first != 0 and all(p.side == first for p in tail)

    @staticmethod
    def _monotonic(b: list[_Pt], length: int) -> bool:
        if len(b) < length:
            return False
        tail = [p.value for p in b[-length:]]
        inc = all(tail[i] < tail[i + 1] for i in range(len(tail) - 1))
        dec = all(tail[i] > tail[i + 1] for i in range(len(tail) - 1))
        return inc or dec

    @staticmethod
    def _alternating(b: list[_Pt], length: int) -> bool:
        if len(b) < length:
            return False
        tail = [p.value for p in b[-length:]]
        diffs = [tail[i + 1] - tail[i] for i in range(len(tail) - 1)]
        if any(d == 0 for d in diffs):
            return False
        return all((diffs[i] > 0) != (diffs[i + 1] > 0) for i in range(len(diffs) - 1))

    @staticmethod
    def _k_of_m_beyond(b: list[_Pt], k: int, m: int, zone: int) -> bool:
        """k of the last m points beyond `zone` sigma on the SAME side, current point included."""
        if len(b) < m:
            return False
        tail = b[-m:]
        if tail[-1].zone < zone:
            return False  # attribute to the current point only when it participates
        for side in (1, -1):
            cnt = sum(1 for p in tail if p.side == side and p.zone >= zone)
            if cnt >= k and tail[-1].side == side:
                return True
        return False

    @staticmethod
    def _run_within(b: list[_Pt], length: int, zone_lt: int) -> bool:
        if len(b) < length:
            return False
        return all(p.zone < zone_lt for p in b[-length:])

    @staticmethod
    def _run_beyond(b: list[_Pt], length: int, zone_ge: int) -> bool:
        if len(b) < length:
            return False
        return all(p.zone >= zone_ge for p in b[-length:])

    def _sig(self, rule_id: str, pt: _Pt, side: str | None = None, catalog=None) -> Signal:
        catalog = catalog or NELSON
        name, _, desc = catalog[rule_id]
        return Signal(rule_id=rule_id, rule_name=name, index=pt.index,
                      value=pt.value, description=desc, side=side)


def evaluate_series(values, center: float, sigma: float, ruleset: str = "nelson") -> list[Signal]:
    """Replay a whole series through the stateful engine (batch convenience)."""
    engine = RuleEngine(center=center, sigma=sigma, ruleset=ruleset)
    signals: list[Signal] = []
    for v in values:
        signals.extend(engine.add(float(v)))
    return signals
