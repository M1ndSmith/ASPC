"""Unit tests for deterministic Explainable SPC Copilot."""
from __future__ import annotations

from spc_core.explain import diff_limits, explain_signal
from spc_core.models import Signal


def test_explain_nelson_rule_1():
    sig = Signal(
        rule_id="1",
        rule_name="Beyond 3-sigma",
        index=12,
        value=110.0,
        description="One point beyond zone A",
        side="above",
    )
    out = explain_signal(sig, limits_version="abc123")
    assert out["rule_id"] == "1"
    assert out["auditable"] is True
    assert out["llm_required"] is False
    assert "abc123" in out["operator_summary"]
    assert "Beyond" in out["catalog_description"] or "zone" in out["catalog_description"].lower()


def test_diff_limits_delta():
    a = {
        "version": "v1",
        "chart_type": "I_MR",
        "components": {"I": {"center": 100.0, "ucl": 103.0, "lcl": 97.0}},
    }
    b = {
        "version": "v2",
        "chart_type": "I_MR",
        "components": {"I": {"center": 101.0, "ucl": 104.0, "lcl": 98.0}},
    }
    d = diff_limits(a, b)
    assert d["same_chart_type"] is True
    assert d["components"][0]["delta"]["center"] == 1.0
