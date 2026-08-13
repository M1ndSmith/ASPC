"""Parity helpers for batch vs stream Phase II signals."""
from __future__ import annotations

from collections import Counter
from typing import Any


def signal_multiset(signals: list[Any]) -> list[tuple[str, int]]:
    """Sorted (rule_id, count) pairs for comparison."""
    c: Counter[str] = Counter()
    for s in signals:
        rid = getattr(s, "rule_id", None) or (s.get("rule_id") if isinstance(s, dict) else None)
        if rid is not None:
            c[str(rid)] += 1
    return sorted(c.items())


def signals_equal(a: list[Any], b: list[Any]) -> bool:
    return signal_multiset(a) == signal_multiset(b)
