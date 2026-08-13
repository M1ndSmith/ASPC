"""Adversarial streaming event emitter for Phase2Evaluator."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Literal

EventKind = Literal["observe", "observe_subgroup", "seed"]


@dataclass
class StreamEvent:
    kind: EventKind
    value: float | list[float] | None = None
    index_hint: int | None = None
    seed_index: int | None = None
    seed_values: list[float] | None = None
    delay_s: float = 0.0


def events_from_plotted(
    plotted: list[float],
    *,
    subgroup_size: int | None = None,
    raw_subgroups: list[list[float]] | None = None,
) -> list[StreamEvent]:
    """Build canonical in-order observe events."""
    if raw_subgroups:
        return [StreamEvent(kind="observe_subgroup", value=list(g), index_hint=i) for i, g in enumerate(raw_subgroups)]
    return [StreamEvent(kind="observe", value=float(v), index_hint=i) for i, v in enumerate(plotted)]


def apply_adversarial(
    events: list[StreamEvent],
    adversarial: str,
    *,
    seed: int = 0,
) -> tuple[list[StreamEvent], str]:
    """Return transformed events + behavior tag."""
    if adversarial in ("none", "", None):
        return events, "canonical"

    rng = random.Random(seed)
    tag = adversarial

    if adversarial == "out_of_order":
        out = list(events)
        if len(out) > 3:
            rng.shuffle(out)
        return out, "order_sensitive"

    if adversarial == "jitter_delay":
        out = []
        for e in events:
            out.append(
                StreamEvent(
                    kind=e.kind,
                    value=e.value,
                    index_hint=e.index_hint,
                    delay_s=rng.uniform(0.0, 0.002),
                )
            )
        return out, "timing_noop"

    if adversarial == "duplicate_index":
        out = list(events)
        if out:
            mid = len(out) // 2
            out.insert(mid, StreamEvent(kind=out[mid].kind, value=out[mid].value, index_hint=out[mid].index_hint))
        return out, "duplicate_observe"

    if adversarial == "watermark_jump":
        seed_ev = StreamEvent(kind="seed", seed_index=10_000, seed_values=[])
        return [seed_ev, *events], "watermark_jump"

    if adversarial == "split_brain_seed":
        warm = [float(e.value) for e in events[:5] if isinstance(e.value, (int, float))]
        # Conflicting warm path: seed with reversed warm values at wrong index
        seed_ev = StreamEvent(kind="seed", seed_index=4, seed_values=list(reversed(warm)) if warm else [0.0])
        return [seed_ev, *events], "split_brain_seed"

    return events, tag


def play_events(evaluator: Any, events: list[StreamEvent], *, apply_delays: bool = True) -> list[Any]:
    """Drive Phase2Evaluator with events; return flat Signal list."""
    signals: list[Any] = []
    for ev in events:
        if apply_delays and ev.delay_s > 0:
            time.sleep(ev.delay_s)
        if ev.kind == "seed":
            evaluator.seed_state(index=ev.seed_index if ev.seed_index is not None else -1, values=ev.seed_values)
            continue
        if ev.kind == "observe_subgroup":
            signals.extend(evaluator.observe_subgroup(ev.value))
        else:
            signals.extend(evaluator.observe(float(ev.value)))  # type: ignore[arg-type]
    return signals
