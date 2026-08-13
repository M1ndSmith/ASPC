"""Schema catalog for combinatorial probing of spc_core contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spc_core.models import ChartType, QualityFlag
from spc_core.rules import NELSON, WESTERN_ELECTRIC

RULESETS = ("nelson", "western_electric", "wheeler")

GATE_STEPS = (
    "msa",
    "gage_resolution",
    "valid_range",
    "missing",
    "autocorrelation",
    "multimodal",
    "normality",
    "transform",
    "chart",
    "freeze",
)

EXPECT_CLASSES = (
    "ok_freeze",
    "stop_unfrozen",
    "raises",
    "phase2_signals",
    "phase2_rejects",
    "adversarial_behavior",
)

# Documented hard raises / STOP edges (kept in sync via meta-tests).
DOCUMENTED_EDGES: list[dict[str, Any]] = [
    {"id": "establish_n0", "raises": "ValueError", "note": "<2 points"},
    {"id": "establish_n1", "raises": "ValueError", "note": "single point"},
    {"id": "p_zero_n", "raises": "ValueError", "note": "P chart n<=0"},
    {"id": "u_zero_opp", "raises": "ValueError", "note": "U opportunities<=0"},
    {"id": "phase2_scalar_on_xbar", "raises": "ValueError", "note": "observe on Xbar"},
    {"id": "phase2_subgroup_on_imr", "raises": "ValueError", "note": "observe_subgroup on I-MR"},
    {"id": "msa_stop", "expect_class": "stop_unfrozen", "note": "GRR>30 or NDC fail"},
    {"id": "multimodal_stop", "expect_class": "stop_unfrozen", "note": "Hartigan dip"},
    {"id": "usl_le_lsl", "raises": "ValueError", "note": "capability specs"},
]


@dataclass(frozen=True)
class SchemaCatalog:
    chart_types: tuple[str, ...] = tuple(c.value for c in ChartType)
    quality_flags: tuple[str, ...] = tuple(q.value for q in QualityFlag)
    rulesets: tuple[str, ...] = RULESETS
    nelson_ids: tuple[str, ...] = tuple(NELSON.keys())
    we_ids: tuple[str, ...] = tuple(WESTERN_ELECTRIC.keys())
    gate_steps: tuple[str, ...] = GATE_STEPS
    expect_classes: tuple[str, ...] = EXPECT_CLASSES
    documented_edges: tuple[dict[str, Any], ...] = tuple(DOCUMENTED_EDGES)
    series_kinds: tuple[str, ...] = (
        "imr_in_control",
        "imr_mean_shift",
        "imr_trend",
        "imr_alternating",
        "imr_constant",
        "imr_empty",
        "imr_n1",
        "imr_n2",
        "imr_nan_inf",
        "imr_sentinel",
        "imr_overflow",
        "xbar_r",
        "attribute_p",
        "attribute_p_zero_n",
        "attribute_np",
        "attribute_c",
        "attribute_u",
        "attribute_u_zero_opp",
        "multimodal",
        "heavy_tail",
        "type_mismatch",
    )
    adversarial_kinds: tuple[str, ...] = (
        "none",
        "out_of_order",
        "jitter_delay",
        "duplicate_index",
        "split_brain_seed",
        "watermark_jump",
    )


def load_catalog() -> SchemaCatalog:
    return SchemaCatalog()


def axis_coverage_targets(catalog: SchemaCatalog | None = None) -> dict[str, set[str]]:
    c = catalog or load_catalog()
    return {
        "chart_types": set(c.chart_types),
        "rulesets": set(c.rulesets),
        "quality_flags": set(c.quality_flags),
        "nelson_ids": set(c.nelson_ids),
        "we_ids": set(c.we_ids),
        "gate_steps": set(c.gate_steps),
        "expect_classes": set(c.expect_classes),
        "series_kinds": set(c.series_kinds),
        "adversarial_kinds": set(c.adversarial_kinds),
    }
