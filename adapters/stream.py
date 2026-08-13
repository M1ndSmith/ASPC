"""Streaming adapters — file replay and live observation sources.

Batch = replay a column through Phase2Evaluator. The same evaluator is used for
live streams; only the source changes. Kafka/MQTT sources live in
``adapters.stream_sources`` (optional extras) so the core adapter stays
dependency-light.
"""
from __future__ import annotations

import csv
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from pathlib import Path

from spc_core.evaluator import Phase2Evaluator
from spc_core.models import ControlLimits, Signal


class ObservationSource(ABC):
    """Yields scalar observations (or subgroup lists) for the Phase II evaluator."""

    @abstractmethod
    def __iter__(self) -> Iterator[float | list[float]]:
        ...


class FileReplaySource(ObservationSource):
    """Replay a CSV column as a stream (optionally with a delay for demos)."""

    def __init__(self, path: str | Path, value_col: str, delay_s: float = 0.0):
        self.path = Path(path)
        self.value_col = value_col
        self.delay_s = delay_s

    def __iter__(self) -> Iterator[float]:
        with self.path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if self.value_col not in (reader.fieldnames or []):
                raise ValueError(f"Column '{self.value_col}' not in {self.path}")
            for row in reader:
                raw = row[self.value_col]
                if raw == "" or raw is None:
                    continue
                if self.delay_s > 0:
                    time.sleep(self.delay_s)
                yield float(raw)


def stream_evaluate(
    source: ObservationSource,
    limits: ControlLimits,
    ruleset: str = "nelson",
    on_signal: Callable[[Signal], None] | None = None,
) -> list[Signal]:
    """Feed a source into Phase2Evaluator; optionally call ``on_signal`` for each hit.

    Returns the full list of signals collected during the stream.
    """
    ev = Phase2Evaluator(limits, ruleset=ruleset)
    all_signals: list[Signal] = []
    for obs in source:
        if isinstance(obs, (list, tuple)):
            signals = ev.observe_subgroup(obs)
        else:
            signals = ev.observe(float(obs))
        if signals:
            all_signals.extend(signals)
            if on_signal:
                for s in signals:
                    on_signal(s)
    return all_signals
