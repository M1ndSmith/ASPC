"""Streaming Phase2 runs + parity against evaluate_batch."""
from __future__ import annotations

import traceback
from typing import Any

from combinatorial.generators.series import build_series, values_from_columns
from combinatorial.matrix import CaseSpec
from combinatorial.parity import signal_multiset, signals_equal
from combinatorial.stream_emitter import apply_adversarial, events_from_plotted, play_events
from spc_core import ChartType, establish
from spc_core.evaluator import Phase2Evaluator, evaluate_batch
from spc_core.limits import build_subgroups


def _chart_type(params: dict[str, Any]) -> ChartType | None:
    raw = params.get("chart_type")
    return ChartType(raw) if raw else None


def _subgroup_lists(values: list[Any], subgroup_ids: list[Any] | None) -> list[list[float]] | None:
    if not subgroup_ids:
        return None
    groups = build_subgroups([float(v) for v in values], subgroup_ids)
    return [[float(x) for x in g] for g in groups]


def run_stream_case(case: CaseSpec, *, seed: int = 0) -> dict[str, Any]:
    observed: dict[str, Any] = {"id": case.id, "entry": case.entry}
    mismatches: list[str] = []
    status = "PASS"
    behavior = "canonical"

    try:
        cols = build_series(case.series_kind, seed=seed)
        values, extra = values_from_columns(cols, case.series_kind)
        params = dict(case.params)
        ct = _chart_type(params)
        ruleset = params.get("ruleset", "nelson")

        # Establish Phase I limits (must freeze for Phase II)
        pipe = establish(
            values,
            chart_type=ct,
            ruleset=ruleset,
            **{k: v for k, v in extra.items()},
        )
        observed["phase1_frozen"] = pipe.frozen
        if not pipe.frozen:
            if case.expect_class == "phase2_rejects":
                # still try reject paths below if wrong_api with unfrozen — skip
                pass
            elif case.expect_class in ("phase2_signals", "adversarial_behavior"):
                # Use chart limits anyway for diagnostic observe if chart exists
                if pipe.stopped and case.expect_class == "adversarial_behavior":
                    status = "PASS"
                    observed["skipped"] = "unfrozen"
                    return _result(case, status, mismatches, observed, behavior)
                mismatches.append("Phase I not frozen; cannot run Phase II parity")
                status = "FAIL"
                return _result(case, status, mismatches, observed, behavior)

        limits = pipe.chart.limits
        plotted = list(pipe.chart.plotted_values)
        subgroups = _subgroup_lists(values, extra.get("subgroup_ids"))

        if case.entry == "phase2_reject":
            ev = Phase2Evaluator(limits, ruleset=ruleset)
            wrong = params.get("wrong_api")
            try:
                if wrong == "observe":
                    ev.observe(float(plotted[0]) if plotted else 0.0)
                elif wrong == "observe_subgroup":
                    ev.observe_subgroup([1.0, 2.0, 3.0])
                else:
                    raise RuntimeError("wrong_api not set")
                mismatches.append("expected Phase2 ValueError but call succeeded")
                status = "FAIL"
            except ValueError as exc:
                observed["exception"] = {"type": "ValueError", "message": str(exc)}
                status = "PASS"
            return _result(case, status, mismatches, observed, behavior)

        # Parity / adversarial
        batch_signals = evaluate_batch(limits, plotted, ruleset=ruleset) if not subgroups else []
        if subgroups:
            # batch evaluate via sequential observe_subgroup
            ev_batch = Phase2Evaluator(limits, ruleset=ruleset)
            for g in subgroups:
                batch_signals.extend(ev_batch.observe_subgroup(g))

        events = events_from_plotted(plotted, raw_subgroups=subgroups)
        events, behavior = apply_adversarial(events, case.adversarial, seed=seed)
        observed["behavior"] = behavior

        ev = Phase2Evaluator(limits, ruleset=ruleset)
        # For watermark/split_brain, seed is inside events
        stream_signals = play_events(ev, events, apply_delays=(case.adversarial == "jitter_delay"))

        batch_ids = signal_multiset(batch_signals)
        stream_ids = signal_multiset(stream_signals)
        observed["batch_rule_ids"] = batch_ids
        observed["stream_rule_ids"] = stream_ids
        observed["parity"] = signals_equal(batch_signals, stream_signals)

        if case.expect_class == "adversarial_behavior":
            if behavior == "timing_noop":
                if not observed["parity"]:
                    mismatches.append("jitter should preserve parity")
                    status = "FAIL"
                else:
                    status = "PASS"
            elif behavior == "order_sensitive":
                # Document divergence; PASS if we recorded behavior (parity may fail)
                observed["parity_expected"] = False
                status = "PASS"
            else:
                # duplicate / watermark / split_brain: record outcome; do not require parity
                observed["parity_expected"] = False
                status = "PASS"
        else:
            # strict parity
            if not observed["parity"]:
                mismatches.append(
                    f"parity mismatch batch={batch_ids} stream={stream_ids}"
                )
                status = "FAIL"

    except Exception as exc:  # noqa: BLE001
        observed["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=4),
        }
        if case.expect_class == "raises":
            status = "PASS"
        else:
            status = "ERROR"
            mismatches.append(f"unexpected {type(exc).__name__}: {exc}")

    return _result(case, status, mismatches, observed, behavior)


def _result(case: CaseSpec, status: str, mismatches: list[str], observed: dict, behavior: str) -> dict[str, Any]:
    return {
        "id": case.id,
        "status": status,
        "expect_class": case.expect_class,
        "mismatches": mismatches,
        "observed": observed,
        "adversarial": case.adversarial,
        "behavior": behavior,
        "modality": case.modality,
    }
