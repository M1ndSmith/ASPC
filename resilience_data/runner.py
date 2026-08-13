"""Shared runner: execute one MANIFEST case against spc_core and compare expect."""
from __future__ import annotations

from typing import Any

from resilience_data import case_csv_path, read_csv
from spc_core import (
    ChartType,
    Phase2Evaluator,
    analyze_control_chart,
    bias_study,
    capability_analysis,
    classify_missing,
    establish,
    gage_rr_anova,
    linearity_study,
    ndc_gate,
    phase1_checklist,
    range_check,
    stability_study,
)
from spc_core.ewma import ewma_chart
from spc_core.msa import gage_resolution_gate


def _col(cols: dict[str, list], name: str | None) -> list | None:
    if name is None:
        return None
    return cols[name]


def _as_float_list(values: list) -> list[float | None]:
    out: list[float | None] = []
    for v in values:
        if v is None:
            out.append(None)
        else:
            out.append(float(v))
    return out


def _chart_type(raw: str | None) -> ChartType | None:
    if raw is None:
        return None
    return ChartType(raw)


def _rule_ids(signals) -> list[str]:
    """Sorted unique rule ids, so a case can assert *which* rule fired, not just how many."""
    return sorted({s.rule_id for s in signals})


def _chart_observed(result) -> dict[str, Any]:
    """Signal/rule facts shared by every chart-producing entry point."""
    return {
        "chart_type": result.chart_type.value,
        "ruleset_applied": result.ruleset_applied,
        "n_plotted": len(result.plotted_values),
        "n_signals": len(result.signals),
        "rule_ids": _rule_ids(result.signals),
        "n_secondary_signals": len(result.secondary_signals),
        "secondary_rule_ids": _rule_ids(result.secondary_signals),
        "secondary_name": result.secondary_name,
    }


def run_case(spec: dict[str, Any]) -> dict[str, Any]:
    """Execute one case. Returns a judgment dict with status and details."""
    case_id = spec["id"]
    entry = spec["entry"]
    expect = spec.get("expect") or {}
    cols_map = spec.get("columns") or {}
    params = dict(spec.get("params") or {})
    xfail = bool(expect.get("xfail", False))

    try:
        cols = read_csv(case_csv_path(spec["path"])) if spec.get("path") else {}
        observed = _dispatch(entry, cols, cols_map, params)
        mismatches = _compare(expect, observed)
        if mismatches:
            status = "XFAIL" if xfail else "FAIL"
            return {
                "id": case_id,
                "status": status,
                "mismatches": mismatches,
                "observed": observed,
                "xfail_reason": expect.get("xfail_reason"),
            }
        if xfail:
            return {
                "id": case_id,
                "status": "FAIL",
                "mismatches": ["marked xfail but all expects matched"],
                "observed": observed,
            }
        return {"id": case_id, "status": "PASS", "observed": observed, "mismatches": []}
    except Exception as exc:  # noqa: BLE001 — judgment must catch all
        expected_raises = expect.get("raises")
        if expected_raises and type(exc).__name__ == expected_raises:
            observed = {"raises": type(exc).__name__, "message": str(exc)}
            # An exception *type* alone is too weak: a domain guard and an internal
            # crash both surface as ValueError. Require the message to prove which.
            wanted_msg = expect.get("raises_match")
            if wanted_msg and wanted_msg not in str(exc):
                return {
                    "id": case_id,
                    "status": "XFAIL" if xfail else "FAIL",
                    "mismatches": [
                        f"raises_match: {wanted_msg!r} not in message {str(exc)!r}"
                    ],
                    "observed": observed,
                    "xfail_reason": expect.get("xfail_reason"),
                }
            return {
                "id": case_id,
                "status": "PASS",
                "observed": observed,
                "mismatches": [],
            }
        if xfail and expected_raises is None:
            return {
                "id": case_id,
                "status": "XFAIL",
                "mismatches": [f"unexpected {type(exc).__name__}: {exc}"],
                "observed": {"raises": type(exc).__name__, "message": str(exc)},
                "xfail_reason": expect.get("xfail_reason"),
            }
        return {
            "id": case_id,
            "status": "ERROR",
            "mismatches": [f"{type(exc).__name__}: {exc}"],
            "observed": {"raises": type(exc).__name__, "message": str(exc)},
        }


def _dispatch(
    entry: str,
    cols: dict[str, list],
    cols_map: dict[str, str],
    params: dict[str, Any],
) -> dict[str, Any]:
    if entry == "establish":
        values = _as_float_list(_col(cols, cols_map.get("values", "measurement")) or [])
        subgroup_ids = _col(cols, cols_map.get("subgroup_ids"))
        sample_sizes = _col(cols, cols_map.get("sample_sizes"))
        opportunities = _col(cols, cols_map.get("opportunities"))
        reasons = _col(cols, cols_map.get("missing_reasons"))
        msa_parts = _col(cols, cols_map.get("msa_parts"))
        msa_operators = _col(cols, cols_map.get("msa_operators"))
        msa_measurements = _col(cols, cols_map.get("msa_measurements"))

        # Optional companion MSA CSV columns living in the same file.
        if msa_measurements is None and "Measurement" in cols and "Part" in cols:
            msa_parts = cols["Part"]
            msa_operators = cols["Operator"]
            msa_measurements = cols["Measurement"]

        pipe = establish(
            values,
            subgroup_ids=subgroup_ids,
            sample_sizes=sample_sizes,
            opportunities=opportunities,
            chart_type=_chart_type(params.get("chart_type")),
            ruleset=params.get("ruleset", "nelson"),
            missing_reasons=reasons,
            msa_parts=msa_parts,
            msa_operators=msa_operators,
            msa_measurements=msa_measurements,
            msa_tolerance=params.get("msa_tolerance"),
            gage_resolution=params.get("gage_resolution"),
            autocorrelated_chart=params.get("autocorrelated_chart", "EWMA"),
            force_wheeler=bool(params.get("force_wheeler", False)),
            valid_range=(
                tuple(params["valid_range"]) if params.get("valid_range") else None
            ),
        )
        checklist = phase1_checklist(
            pipe,
            min_subgroups=int(params.get("min_subgroups", 25)),
            phase2_enabled=bool(params.get("phase2_enabled", False)),
        )
        gates = {g.step: g.status for g in pipe.gates}
        return {
            **_chart_observed(pipe.chart),
            "stopped": pipe.stopped,
            "frozen": pipe.frozen,
            "gates": gates,
            "chart_route": pipe.chart_route,
            "distribution_flag": pipe.chart.distribution_flag.value,
            "checklist_passed": checklist["passed"],
            "checklist_items": {
                i["item"]: i["passed"] for i in checklist["items"]
            },
            "msa_grr_percent": (
                pipe.msa.grr_percent if pipe.msa is not None else None
            ),
            "msa_ndc": pipe.msa.ndc if pipe.msa is not None else None,
            "raises": None,
        }

    if entry == "analyze_control_chart":
        values = _as_float_list(_col(cols, cols_map.get("values", "measurement")) or [])
        subgroup_ids = _col(cols, cols_map.get("subgroup_ids"))
        sample_sizes = _col(cols, cols_map.get("sample_sizes"))
        opportunities = _col(cols, cols_map.get("opportunities"))
        result = analyze_control_chart(
            [v for v in values if v is not None],
            subgroup_ids=subgroup_ids,
            sample_sizes=sample_sizes,
            opportunities=opportunities,
            chart_type=_chart_type(params.get("chart_type")),
            ruleset=params.get("ruleset", "nelson"),
            exclude_incomplete=bool(params.get("exclude_incomplete", False)),
        )
        return {**_chart_observed(result), "raises": None}

    if entry == "classify_missing":
        values = _as_float_list(_col(cols, cols_map.get("values", "measurement")) or [])
        reasons = _col(cols, cols_map.get("reasons", "reason"))
        result = classify_missing(values, reasons=reasons)
        flag_counts: dict[str, int] = {}
        for f in result.flags:
            flag_counts[f.value] = flag_counts.get(f.value, 0) + 1
        return {
            "flag_counts": flag_counts,
            "n_usable": sum(1 for u in result.usable if u),
            "n_unusable": sum(1 for u in result.usable if not u),
            "raises": None,
        }

    if entry == "range_check":
        values = _as_float_list(_col(cols, cols_map.get("values", "measurement")) or [])
        low = float(params["low"])
        high = float(params["high"])
        valid = range_check(values, low, high)
        return {
            "n_invalid": sum(1 for v in valid if not v),
            "n_valid": sum(1 for v in valid if v),
            "raises": None,
        }

    if entry == "gage_rr_anova":
        parts = _col(cols, cols_map.get("parts", "Part")) or []
        operators = _col(cols, cols_map.get("operators", "Operator")) or []
        measurements = _col(cols, cols_map.get("measurements", "Measurement")) or []
        tolerance = params.get("tolerance", 10.0)
        result = gage_rr_anova(parts, operators, measurements, tolerance=tolerance)
        ndc_ok, _ = ndc_gate(result.ndc)
        detail = result.detail or {}
        return {
            "grr_percent": result.grr_percent,
            "ndc": result.ndc,
            "ndc_ok": ndc_ok,
            "method": result.method,
            "anova_fallback": bool(detail.get("anova_skipped")),
            "raises": None,
        }

    if entry == "gage_resolution_gate":
        ok, reason = gage_resolution_gate(
            float(params["resolution"]), float(params["tolerance"])
        )
        return {"ok": ok, "reason": reason, "raises": None}

    if entry == "bias_study":
        meas = _col(cols, cols_map.get("measurements", "Measurement")) or []
        refs = _col(cols, cols_map.get("references", "Reference")) or []
        result = bias_study(meas, refs)
        return {
            "mean_bias": result.mean_bias,
            "is_significant": result.is_significant,
            "raises": None,
        }

    if entry == "linearity_study":
        meas = _col(cols, cols_map.get("measurements", "Measurement")) or []
        refs = _col(cols, cols_map.get("references", "Reference")) or []
        result = linearity_study(meas, refs)
        return {
            "slope": result.slope,
            "is_linear": result.is_linear,
            "raises": None,
        }

    if entry == "stability_study":
        meas = _col(cols, cols_map.get("measurements", "Measurement")) or []
        result = stability_study(meas)
        return {
            "out_of_control_points": result.out_of_control_points,
            "is_stable": result.is_stable,
            "raises": None,
        }

    if entry == "capability_analysis":
        values = _as_float_list(_col(cols, cols_map.get("values", "measurement")) or [])
        result = capability_analysis(
            [v for v in values if v is not None],
            usl=float(params["usl"]),
            lsl=float(params["lsl"]),
            target=params.get("target"),
            force_method=params.get("force_method"),
        )
        return {
            "method": result.method,
            "cp": result.cp,
            "cpk": result.cpk,
            "pp": result.pp,
            "ppk": result.ppk,
            # Relative form, so an off-centre case asserts "Cpk penalised vs Cp"
            # instead of hardcoding an observed float.
            "cpk_lt_cp": (
                None
                if result.cp is None or result.cpk is None
                else bool(result.cpk < result.cp)
            ),
            "raises": None,
        }

    if entry == "ewma_chart":
        values = _as_float_list(_col(cols, cols_map.get("values", "measurement")) or [])
        result = ewma_chart([v for v in values if v is not None])
        return {
            "chart_type": result.limits.chart_type.value,
            "raises": None,
        }

    if entry == "phase2_detect":
        # Establish on first half, evaluate second half.
        values = [
            v
            for v in _as_float_list(_col(cols, cols_map.get("values", "measurement")) or [])
            if v is not None
        ]
        split = int(params.get("split", len(values) // 2))
        pipe = establish(values[:split], chart_type=_chart_type(params.get("chart_type")))
        ruleset = "wheeler" if pipe.chart_route == "wheeler" else "nelson"
        ev = Phase2Evaluator(pipe.chart.limits, ruleset=ruleset)
        signals = []
        for v in values[split:]:
            signals.extend(ev.observe(v))
        return {
            "n_signals": len(signals),
            "rule_ids": _rule_ids(signals),
            "ruleset_applied": ruleset,
            "frozen": pipe.frozen,
            "raises": None,
        }

    raise ValueError(f"Unknown entry {entry!r}")


def _compare(expect: dict[str, Any], observed: dict[str, Any]) -> list[str]:
    """Return list of mismatch strings. Ignores meta keys."""
    mismatches: list[str] = []
    skip = {"xfail", "xfail_reason", "raises", "raises_match", "notes"}
    list_ops = ("includes", "excludes", "subset_of")
    if expect.get("raises"):
        # Handled in run_case exception path; if we got here, no raise occurred.
        mismatches.append(
            f"expected raises={expect['raises']!r} but got result {observed}"
        )
        return mismatches

    for key, wanted in expect.items():
        if key in skip:
            continue
        actual = observed.get(key)
        if isinstance(wanted, dict) and any(k in wanted for k in list_ops):
            actual_list = list(actual or [])
            for item in wanted.get("includes", []):
                if item not in actual_list:
                    mismatches.append(
                        f"{key}: expected to include {item!r}, got {actual_list!r}"
                    )
            for item in wanted.get("excludes", []):
                if item in actual_list:
                    mismatches.append(
                        f"{key}: expected NOT to include {item!r}, got {actual_list!r}"
                    )
            if "subset_of" in wanted:
                allowed = set(wanted["subset_of"])
                extra = sorted(x for x in actual_list if x not in allowed)
                if extra:
                    mismatches.append(
                        f"{key}: {extra!r} not allowed; expected subset of "
                        f"{sorted(allowed)!r}"
                    )
        elif isinstance(wanted, dict) and any(
            k in wanted for k in ("lt", "lte", "gt", "gte", "eq")
        ):
            if actual is None:
                mismatches.append(f"{key}: expected bound {wanted}, got None")
                continue
            try:
                val = float(actual)
            except (TypeError, ValueError):
                mismatches.append(f"{key}: expected numeric, got {actual!r}")
                continue
            if "lt" in wanted and not (val < float(wanted["lt"])):
                mismatches.append(f"{key}: {val} not < {wanted['lt']}")
            if "lte" in wanted and not (val <= float(wanted["lte"])):
                mismatches.append(f"{key}: {val} not <= {wanted['lte']}")
            if "gt" in wanted and not (val > float(wanted["gt"])):
                mismatches.append(f"{key}: {val} not > {wanted['gt']}")
            if "gte" in wanted and not (val >= float(wanted["gte"])):
                mismatches.append(f"{key}: {val} not >= {wanted['gte']}")
            if "eq" in wanted and val != float(wanted["eq"]):
                mismatches.append(f"{key}: {val} != {wanted['eq']}")
        elif isinstance(wanted, dict) and key == "gates":
            actual_gates = actual or {}
            for step, status in wanted.items():
                if status == "__absent__":
                    if step in actual_gates:
                        mismatches.append(
                            f"gates.{step}: expected absent, got {actual_gates[step]!r}"
                        )
                elif actual_gates.get(step) != status:
                    mismatches.append(
                        f"gates.{step}: expected {status!r}, got {actual_gates.get(step)!r}"
                    )
        elif isinstance(wanted, dict) and key in ("flag_counts", "checklist_items"):
            actual_map = actual or {}
            for k, v in wanted.items():
                if actual_map.get(k) != v:
                    mismatches.append(
                        f"{key}.{k}: expected {v!r}, got {actual_map.get(k)!r}"
                    )
        elif wanted != actual:
            mismatches.append(f"{key}: expected {wanted!r}, got {actual!r}")
    return mismatches
