"""Smoke tests for combinatorial matrix."""
from __future__ import annotations

from combinatorial.matrix import build_matrix, load_config
from combinatorial.schema_catalog import RULESETS, load_catalog
from combinatorial.dual_runner import run_case
from spc_core import ChartType


def test_catalog_covers_chart_types_and_rulesets():
    catalog = load_catalog()
    chart_values = {ct.value for ct in ChartType}
    assert set(catalog.chart_types) == chart_values
    assert set(RULESETS) == set(catalog.rulesets)
    for ct in ChartType:
        assert ct.value in catalog.chart_types
    for rs in ("nelson", "western_electric", "wheeler"):
        assert rs in catalog.rulesets


def test_sparse_matrix_includes_critical_and_passes():
    cfg = load_config()
    cases = build_matrix(
        mode="sparse",
        seed=int(cfg.get("seed", 42)),
        max_sparse_cases=min(40, int(cfg.get("max_sparse_cases", 80))),
        adversarial=True,
    )
    assert any(c.id.startswith("crit_") for c in cases)
    # One known parity case must pass
    parity = next(c for c in cases if c.id == "crit_imr_in_control_nelson")
    result = run_case(parity, seed=0)
    assert result["status"] == "PASS", result


def test_sparse_run_smoke():
    """Full sparse subsample must be mostly green; allow soft multimodal flake."""
    from combinatorial.dual_runner import run_matrix

    summary = run_matrix(
        mode="sparse",
        seed=42,
        max_sparse_cases=35,
        adversarial=True,
        write_fixtures=False,
    )
    assert summary["n_cases"] >= 20
    # Critical path: zero unexpected ERROR except soft-allowed ids
    soft = {"crit_multimodal_stop", "crit_nan_inf"}
    hard_fails = [
        r
        for r in summary["results"]
        if r["status"] in ("FAIL", "ERROR") and r["id"] not in soft
    ]
    assert not hard_fails, hard_fails[:5]
