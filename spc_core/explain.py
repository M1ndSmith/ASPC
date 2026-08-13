"""Deterministic Explainable SPC Copilot — structured "why" for OOC signals.

Never invents rules. Optional LLM phrasing is out of scope for this module;
callers may rewrite the returned ``operator_summary`` only.
"""
from __future__ import annotations

from typing import Any

from .models import ControlLimits, Signal
from .rules import NELSON, WESTERN_ELECTRIC

# Extra rule ids used by evaluator / EWMA / CUSUM (beyond Nelson catalog keys).
_EXTRA: dict[str, tuple[str, str]] = {
    "EWMA1": (
        "Beyond EWMA limits",
        "The EWMA statistic crossed its control limit — a sustained mean shift is likely.",
    ),
    "CUSUM+": (
        "CUSUM upper shift",
        "The upper CUSUM crossed its decision interval — evidence of an upward mean shift.",
    ),
    "CUSUM-": (
        "CUSUM lower shift",
        "The lower CUSUM crossed its decision interval — evidence of a downward mean shift.",
    ),
    "WE1": ("Beyond 3-sigma", WESTERN_ELECTRIC["WE1"][2]),
    "WE2": ("2 of 3 beyond 2-sigma", WESTERN_ELECTRIC["WE2"][2]),
    "WE3": ("4 of 5 beyond 1-sigma", WESTERN_ELECTRIC["WE3"][2]),
    "WE4": ("8 on one side", WESTERN_ELECTRIC["WE4"][2]),
}


def _lookup(rule_id: str) -> tuple[str, str]:
    if rule_id in NELSON:
        name, _, desc = NELSON[rule_id]
        return name, desc
    if rule_id in WESTERN_ELECTRIC:
        name, _, desc = WESTERN_ELECTRIC[rule_id]
        return name, desc
    if rule_id in _EXTRA:
        return _EXTRA[rule_id]
    return (f"Rule {rule_id}", f"Rule {rule_id} fired (see signal description).")


def explain_signal(
    signal: Signal | dict[str, Any],
    *,
    limits_version: str | None = None,
    limits: ControlLimits | dict[str, Any] | None = None,
    gates: list[dict[str, Any]] | None = None,
    checklist: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a structured explanation for one OOC signal.

    Returns a JSON-serializable dict suitable for Live drawers and ``POST /analyze/explain``.
    """
    if isinstance(signal, Signal):
        rule_id = signal.rule_id
        rule_name = signal.rule_name
        description = signal.description
        index = signal.index
        value = signal.value
        side = signal.side
    else:
        rule_id = str(signal.get("rule_id") or "")
        rule_name = str(signal.get("rule_name") or "")
        description = str(signal.get("description") or "")
        index = int(signal.get("index") or 0)
        value = float(signal.get("value") or 0.0)
        side = signal.get("side")

    catalog_name, catalog_desc = _lookup(rule_id)
    display_name = rule_name or catalog_name
    why = description or catalog_desc

    limit_snapshot: dict[str, Any] | None = None
    chart_type = None
    if isinstance(limits, ControlLimits):
        chart_type = limits.chart_type.value
        primary = limits.primary
        limit_snapshot = {
            "center": primary.center,
            "ucl": primary.ucl if not isinstance(primary.ucl, list) else primary.ucl[0],
            "lcl": primary.lcl if not isinstance(primary.lcl, list) else primary.lcl[0],
            "version": limits.version,
        }
        limits_version = limits_version or limits.version
    elif isinstance(limits, dict):
        chart_type = limits.get("chart_type")
        comps = limits.get("components") or {}
        primary = next(iter(comps.values()), None) if comps else None
        if isinstance(primary, dict):
            limit_snapshot = {
                "center": primary.get("center"),
                "ucl": primary.get("ucl"),
                "lcl": primary.get("lcl"),
                "version": limits.get("version") or limits_version,
            }
        limits_version = limits_version or limits.get("version")

    gate_summary = None
    if gates:
        gate_summary = [
            {"step": g.get("step"), "status": g.get("status"), "reason": g.get("reason")}
            for g in gates
            if isinstance(g, dict)
        ]

    checklist_passed = None
    if checklist and isinstance(checklist, dict):
        checklist_passed = checklist.get("passed")

    operator_summary = (
        f"Point {index} (value={value}) triggered {display_name} "
        f"(rule {rule_id}): {why}"
    )
    if side:
        operator_summary += f" Side: {side}."
    if limits_version:
        operator_summary += f" Frozen limits version: {limits_version}."

    return {
        "rule_id": rule_id,
        "rule_name": display_name,
        "catalog_description": catalog_desc,
        "why": why,
        "index": index,
        "value": value,
        "side": side,
        "limits_version": limits_version,
        "chart_type": chart_type,
        "limit_snapshot": limit_snapshot,
        "gates": gate_summary,
        "checklist_passed": checklist_passed,
        "operator_summary": operator_summary,
        "auditable": True,
        "llm_required": False,
    }


def explain_signals(
    signals: list[Signal | dict[str, Any]],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    return [explain_signal(s, **kwargs) for s in signals]


def diff_limits(
    a: dict[str, Any],
    b: dict[str, Any],
) -> dict[str, Any]:
    """Compare two stored limits payloads (from persistence ``get_limits``)."""
    va = a.get("version") or a.get("limits_version")
    vb = b.get("version") or b.get("limits_version")
    ca = a.get("chart_type")
    cb = b.get("chart_type")
    comps_a = a.get("components") or {}
    comps_b = b.get("components") or {}
    names = sorted(set(comps_a) | set(comps_b))
    component_diffs: list[dict[str, Any]] = []
    for name in names:
        pa = comps_a.get(name) or {}
        pb = comps_b.get(name) or {}
        if not isinstance(pa, dict):
            pa = {}
        if not isinstance(pb, dict):
            pb = {}

        def _num(x: Any) -> float | None:
            if x is None:
                return None
            if isinstance(x, list):
                return float(x[0]) if x else None
            return float(x)

        ua, ub = _num(pa.get("ucl")), _num(pb.get("ucl"))
        la, lb = _num(pa.get("lcl")), _num(pb.get("lcl"))
        cta, ctb = _num(pa.get("center")), _num(pb.get("center"))
        component_diffs.append(
            {
                "component": name,
                "a": {"center": cta, "ucl": ua, "lcl": la},
                "b": {"center": ctb, "ucl": ub, "lcl": lb},
                "delta": {
                    "center": None if cta is None or ctb is None else ctb - cta,
                    "ucl": None if ua is None or ub is None else ub - ua,
                    "lcl": None if la is None or lb is None else lb - la,
                },
            }
        )
    return {
        "version_a": va,
        "version_b": vb,
        "chart_type_a": ca,
        "chart_type_b": cb,
        "same_chart_type": ca == cb,
        "components": component_diffs,
        "notes_a": a.get("notes") or {},
        "notes_b": b.get("notes") or {},
    }
