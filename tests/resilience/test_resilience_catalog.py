"""Judgment suite: every MANIFEST case must match its expect block."""
from __future__ import annotations

import pytest

from resilience_data import load_manifest
from resilience_data.runner import run_case

CASES = load_manifest()
CASE_IDS = [c["id"] for c in CASES]


@pytest.mark.parametrize("case_id", CASE_IDS, ids=CASE_IDS)
def test_resilience_case(case_id: str):
    spec = next(c for c in CASES if c["id"] == case_id)
    result = run_case(spec)
    assert result["status"] == "PASS", (
        f"{case_id} → {result['status']}: {result.get('mismatches')}\n"
        f"observed={result.get('observed')}"
    )


def test_manifest_covers_all_chart_types_and_quality_flags():
    """Sanity: catalog exercises the surface area claimed in the README."""
    from spc_core.models import ChartType, QualityFlag

    chart_types: set[str] = set()
    quality_flags: set[str] = set()
    gate_steps: set[str] = set()
    for spec in CASES:
        exp = spec.get("expect") or {}
        if isinstance(exp.get("chart_type"), str):
            chart_types.add(exp["chart_type"])
        ct = (spec.get("params") or {}).get("chart_type")
        if ct:
            chart_types.add(ct)
        quality_flags.update((exp.get("flag_counts") or {}).keys())
        gate_steps.update((exp.get("gates") or {}).keys())

    for member in ChartType:
        assert member.value in chart_types, f"ChartType {member.value} missing from catalog"

    for flag in QualityFlag:
        if flag == QualityFlag.ORIGINAL:
            continue  # present on every clean series; not asserted via flag_counts
        assert flag.value in quality_flags, f"QualityFlag {flag.value} not exercised"

    for step in ("msa", "autocorrelation", "multimodal", "normality", "range", "freeze"):
        assert step in gate_steps, f"gate {step} missing from expects"
    assert any(c["entry"] == "gage_resolution_gate" for c in CASES)
    assert any(c["entry"] == "classify_missing" for c in CASES)


def test_manifest_asserts_every_run_rule():
    """Counting signals is not enough — a case must pin down *which* rule fired.

    Without this, a trend case would pass on an unrelated beyond-3-sigma point, and a
    whole ruleset could regress unnoticed.
    """
    from spc_core.rules import NELSON, WESTERN_ELECTRIC

    asserted: set[str] = set()
    rulesets: set[str] = set()
    for spec in CASES:
        exp = spec.get("expect") or {}
        wanted = exp.get("rule_ids")
        if isinstance(wanted, dict):
            asserted.update(wanted.get("includes") or [])
        secondary = exp.get("secondary_rule_ids")
        if isinstance(secondary, dict):
            asserted.update(secondary.get("includes") or [])
        if isinstance(exp.get("ruleset_applied"), str):
            rulesets.add(exp["ruleset_applied"])

    missing_nelson = sorted(set(NELSON) - asserted)
    assert not missing_nelson, f"Nelson rules never asserted by any case: {missing_nelson}"

    missing_we = sorted(set(WESTERN_ELECTRIC) - asserted)
    assert not missing_we, f"Western Electric rules never asserted: {missing_we}"

    for ruleset in ("nelson", "western_electric", "wheeler"):
        assert ruleset in rulesets, f"ruleset {ruleset!r} never asserted as applied"
