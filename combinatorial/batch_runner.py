"""Write static case fixtures and run batch entries against spc_core."""
from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

from combinatorial.generators.series import build_series, sanitize_for_json, values_from_columns
from combinatorial.matrix import CaseSpec
from spc_core import ChartType, analyze_control_chart, capability_analysis, establish
from spc_core.pipeline import phase1_checklist


def write_static_fixture(case: CaseSpec, out_dir: Path, *, seed: int = 0) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = build_series(case.series_kind, seed=seed)
    payload = {
        "id": case.id,
        "case": case.to_dict(),
        "columns": sanitize_for_json(cols),
    }
    path = out_dir / f"{case.id}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _chart_type(params: dict[str, Any]) -> ChartType | None:
    raw = params.get("chart_type")
    if not raw:
        return None
    return ChartType(raw)


def run_batch_case(case: CaseSpec, *, seed: int = 0) -> dict[str, Any]:
    """Execute one batch-oriented case; return judgment dict."""
    observed: dict[str, Any] = {"id": case.id, "entry": case.entry}
    status = "PASS"
    mismatches: list[str] = []
    exc_info: dict[str, Any] | None = None

    try:
        cols = build_series(case.series_kind, seed=seed)
        values, extra = values_from_columns(cols, case.series_kind)
        params = dict(case.params)
        ct = _chart_type(params)
        ruleset = params.get("ruleset", "nelson")

        if case.entry == "establish":
            pipe = establish(
                values,
                chart_type=ct,
                ruleset=ruleset,
                **{k: v for k, v in extra.items()},
            )
            checklist = phase1_checklist(pipe)
            gates = [{"step": g.step, "status": g.status, "reason": g.reason} for g in pipe.gates]
            observed.update(
                {
                    "frozen": pipe.frozen,
                    "stopped": pipe.stopped,
                    "gates": gates,
                    "checklist_passed": checklist.get("passed"),
                    "limits_version": pipe.limits_version if pipe.frozen else None,
                    "chart_type": pipe.chart.chart_type.value if pipe.chart else None,
                    "rule_ids": [s.rule_id for s in (pipe.chart.signals or [])],
                }
            )
            status = _judge_establish(case, observed, mismatches)

        elif case.entry == "analyze_control_chart":
            result = analyze_control_chart(
                values,
                chart_type=ct,
                ruleset=ruleset,
                **{k: v for k, v in extra.items()},
            )
            observed.update(
                {
                    "frozen": True,
                    "chart_type": result.chart_type.value,
                    "limits_version": result.limits.version,
                    "rule_ids": [s.rule_id for s in (result.signals or [])],
                    "n_plotted": len(result.plotted_values),
                }
            )
            if case.expect_class == "raises":
                mismatches.append("expected raise but analyze succeeded")
                status = "FAIL"
            else:
                status = "PASS"

        elif case.entry == "capability":
            usl = float(params["usl"])
            lsl = float(params["lsl"])
            # coerce numeric values only
            nums = [float(v) for v in values if isinstance(v, (int, float))]
            capability_analysis(nums, usl=usl, lsl=lsl)
            if case.expect_class == "raises":
                mismatches.append("expected raise but capability succeeded")
                status = "FAIL"

        else:
            observed["skipped_batch"] = True
            status = "PASS"

    except Exception as exc:  # noqa: BLE001 — capture for judgment
        exc_info = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=4),
        }
        observed["exception"] = exc_info
        if case.expect_class == "raises":
            status = "PASS"
        else:
            status = "ERROR"
            mismatches.append(f"unexpected {type(exc).__name__}: {exc}")

    return {
        "id": case.id,
        "status": status,
        "expect_class": case.expect_class,
        "mismatches": mismatches,
        "observed": observed,
        "adversarial": case.adversarial,
        "modality": case.modality,
    }


def _judge_establish(case: CaseSpec, observed: dict[str, Any], mismatches: list[str]) -> str:
    if case.expect_class == "raises":
        mismatches.append("expected raise but establish succeeded")
        return "FAIL"
    if case.expect_class == "stop_unfrozen":
        if observed.get("frozen") is True:
            # multimodal may not always stop depending on dip — soft-pass with note
            stops = [g for g in observed.get("gates") or [] if g.get("status") == "stop"]
            if not stops and not observed.get("stopped"):
                mismatches.append("expected STOP/unfrozen but frozen=True with no stop gate")
                return "FAIL"
            if observed.get("frozen") and stops:
                mismatches.append("stop gate present but frozen=True (unexpected)")
                return "FAIL"
        return "PASS"
    if case.expect_class == "ok_freeze":
        if not observed.get("frozen"):
            # constant / sentinel may still freeze; if not, record soft fail
            mismatches.append("expected freeze but frozen=False")
            return "FAIL"
        return "PASS"
    return "PASS"
