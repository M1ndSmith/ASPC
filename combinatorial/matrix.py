"""CaseSpec definitions and matrix builders (exhaustive / sparse)."""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from combinatorial.schema_catalog import RULESETS, SchemaCatalog, load_catalog

Modality = Literal["batch", "stream", "both"]
ExpectClass = Literal[
    "ok_freeze",
    "stop_unfrozen",
    "raises",
    "phase2_signals",
    "phase2_rejects",
    "adversarial_behavior",
]


@dataclass
class CaseSpec:
    id: str
    modality: Modality
    entry: str  # establish | analyze_control_chart | phase2_parity | phase2_reject | capability
    series_kind: str
    params: dict[str, Any] = field(default_factory=dict)
    expect_class: ExpectClass = "ok_freeze"
    adversarial: str = "none"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _critical_cases() -> list[CaseSpec]:
    """Always-included critical probes."""
    return [
        CaseSpec(
            id="crit_establish_empty",
            modality="batch",
            entry="establish",
            series_kind="imr_empty",
            expect_class="raises",
            notes="n=0 ValueError",
        ),
        CaseSpec(
            id="crit_establish_n1",
            modality="batch",
            entry="establish",
            series_kind="imr_n1",
            expect_class="raises",
            notes="n=1 ValueError",
        ),
        CaseSpec(
            id="crit_establish_n2_freeze",
            modality="batch",
            entry="establish",
            series_kind="imr_n2",
            params={"ruleset": "nelson"},
            expect_class="ok_freeze",
            notes="minimum establish points",
        ),
        CaseSpec(
            id="crit_p_zero_n",
            modality="batch",
            entry="analyze_control_chart",
            series_kind="attribute_p_zero_n",
            params={"chart_type": "P"},
            expect_class="raises",
        ),
        CaseSpec(
            id="crit_u_zero_opp",
            modality="batch",
            entry="analyze_control_chart",
            series_kind="attribute_u_zero_opp",
            params={"chart_type": "U"},
            expect_class="raises",
        ),
        CaseSpec(
            id="crit_imr_in_control_nelson",
            modality="both",
            entry="phase2_parity",
            series_kind="imr_in_control",
            params={"ruleset": "nelson", "chart_type": "I-MR"},
            expect_class="phase2_signals",
            adversarial="none",
        ),
        CaseSpec(
            id="crit_imr_mean_shift_parity",
            modality="both",
            entry="phase2_parity",
            series_kind="imr_mean_shift",
            params={"ruleset": "nelson", "chart_type": "I-MR"},
            expect_class="phase2_signals",
            adversarial="none",
        ),
        CaseSpec(
            id="crit_xbar_subgroup_reject_scalar",
            modality="stream",
            entry="phase2_reject",
            series_kind="xbar_r",
            params={"chart_type": "Xbar-R", "ruleset": "nelson", "wrong_api": "observe"},
            expect_class="phase2_rejects",
        ),
        CaseSpec(
            id="crit_imr_reject_subgroup",
            modality="stream",
            entry="phase2_reject",
            series_kind="imr_in_control",
            params={"chart_type": "I-MR", "ruleset": "nelson", "wrong_api": "observe_subgroup"},
            expect_class="phase2_rejects",
        ),
        CaseSpec(
            id="crit_multimodal_stop",
            modality="batch",
            entry="establish",
            series_kind="multimodal",
            params={"ruleset": "nelson"},
            expect_class="stop_unfrozen",
        ),
        CaseSpec(
            id="crit_constant_series",
            modality="batch",
            entry="establish",
            series_kind="imr_constant",
            params={"ruleset": "nelson"},
            expect_class="raises",
            notes="constant series → Box-Cox ValueError in establish transform",
        ),
        CaseSpec(
            id="crit_sentinel",
            modality="batch",
            entry="establish",
            series_kind="imr_sentinel",
            params={"ruleset": "nelson"},
            expect_class="ok_freeze",
        ),
        CaseSpec(
            id="crit_nan_inf",
            modality="batch",
            entry="establish",
            series_kind="imr_nan_inf",
            params={"ruleset": "nelson"},
            expect_class="raises",
            notes="non-finite often fails usable-points check",
        ),
        CaseSpec(
            id="crit_type_mismatch",
            modality="batch",
            entry="establish",
            series_kind="type_mismatch",
            expect_class="raises",
        ),
        CaseSpec(
            id="crit_capability_bad_specs",
            modality="batch",
            entry="capability",
            series_kind="imr_in_control",
            params={"usl": 90.0, "lsl": 110.0},
            expect_class="raises",
        ),
        CaseSpec(
            id="crit_adv_ooo",
            modality="stream",
            entry="phase2_parity",
            series_kind="imr_mean_shift",
            params={"ruleset": "nelson", "chart_type": "I-MR"},
            expect_class="adversarial_behavior",
            adversarial="out_of_order",
            notes="order_sensitive",
        ),
        CaseSpec(
            id="crit_adv_dup",
            modality="stream",
            entry="phase2_parity",
            series_kind="imr_in_control",
            params={"ruleset": "nelson", "chart_type": "I-MR"},
            expect_class="adversarial_behavior",
            adversarial="duplicate_index",
        ),
        CaseSpec(
            id="crit_adv_jitter",
            modality="stream",
            entry="phase2_parity",
            series_kind="imr_in_control",
            params={"ruleset": "nelson", "chart_type": "I-MR"},
            expect_class="adversarial_behavior",
            adversarial="jitter_delay",
            notes="timing no-op for state",
        ),
        CaseSpec(
            id="crit_adv_watermark",
            modality="stream",
            entry="phase2_parity",
            series_kind="imr_in_control",
            params={"ruleset": "nelson", "chart_type": "I-MR"},
            expect_class="adversarial_behavior",
            adversarial="watermark_jump",
        ),
        CaseSpec(
            id="crit_adv_split_brain",
            modality="stream",
            entry="phase2_parity",
            series_kind="imr_in_control",
            params={"ruleset": "nelson", "chart_type": "I-MR"},
            expect_class="adversarial_behavior",
            adversarial="split_brain_seed",
        ),
    ]


def _cartesian_batch(catalog: SchemaCatalog) -> list[CaseSpec]:
    """Finite Cartesian over ruleset × core series kinds (not every ChartType combo)."""
    cases: list[CaseSpec] = []
    continuous_kinds = (
        "imr_in_control",
        "imr_mean_shift",
        "imr_constant",
        "heavy_tail",
        "imr_overflow",
    )
    for ruleset in RULESETS:
        for kind in continuous_kinds:
            # Constant → Box-Cox ValueError; overflow → multimodal histogram ValueError
            expect: ExpectClass = (
                "raises" if kind in ("imr_constant", "imr_overflow") else "ok_freeze"
            )
            cases.append(
                CaseSpec(
                    id=f"batch_{kind}_{ruleset}",
                    modality="batch",
                    entry="establish",
                    series_kind=kind,
                    params={"ruleset": ruleset, "chart_type": "I-MR"},
                    expect_class=expect,
                )
            )
    for kind, ct in (
        ("xbar_r", "Xbar-R"),
        ("attribute_p", "P"),
        ("attribute_np", "NP"),
        ("attribute_c", "C"),
        ("attribute_u", "U"),
    ):
        cases.append(
            CaseSpec(
                id=f"batch_{kind}_nelson",
                modality="batch",
                entry="establish",
                series_kind=kind,
                params={"ruleset": "nelson", "chart_type": ct},
                expect_class="ok_freeze",
            )
        )
    # Phase2 parity for each ruleset on I-MR in-control
    for ruleset in RULESETS:
        cases.append(
            CaseSpec(
                id=f"parity_imr_{ruleset}",
                modality="both",
                entry="phase2_parity",
                series_kind="imr_in_control",
                params={"ruleset": ruleset, "chart_type": "I-MR"},
                expect_class="phase2_signals",
                adversarial="none",
            )
        )
    cases.append(
        CaseSpec(
            id="parity_xbar_r_nelson",
            modality="both",
            entry="phase2_parity",
            series_kind="xbar_r",
            params={"ruleset": "nelson", "chart_type": "Xbar-R"},
            expect_class="phase2_signals",
            adversarial="none",
        )
    )
    # Fire Western Electric + Nelson pattern rules for coverage
    cases.append(
        CaseSpec(
            id="rules_we_mean_shift",
            modality="batch",
            entry="analyze_control_chart",
            series_kind="imr_mean_shift",
            params={"ruleset": "western_electric", "chart_type": "I-MR"},
            expect_class="ok_freeze",
        )
    )
    cases.append(
        CaseSpec(
            id="rules_nelson_trend",
            modality="batch",
            entry="analyze_control_chart",
            series_kind="imr_trend",
            params={"ruleset": "nelson", "chart_type": "I-MR"},
            expect_class="ok_freeze",
        )
    )
    cases.append(
        CaseSpec(
            id="rules_nelson_alternating",
            modality="batch",
            entry="analyze_control_chart",
            series_kind="imr_alternating",
            params={"ruleset": "nelson", "chart_type": "I-MR"},
            expect_class="ok_freeze",
        )
    )
    # ChartType coverage via analyze
    for ct in catalog.chart_types:
        kind = {
            "I-MR": "imr_in_control",
            "Xbar-R": "xbar_r",
            "Xbar-S": "xbar_r",
            "P": "attribute_p",
            "NP": "attribute_np",
            "C": "attribute_c",
            "U": "attribute_u",
            "EWMA": "imr_in_control",
            "CUSUM": "imr_in_control",
        }.get(ct, "imr_in_control")
        cases.append(
            CaseSpec(
                id=f"chartcov_{ct.replace('-', '_').replace('/', '_')}",
                modality="batch",
                entry="analyze_control_chart",
                series_kind=kind,
                params={"chart_type": ct, "ruleset": "nelson"},
                expect_class="ok_freeze",
            )
        )
    return cases


def build_matrix(
    *,
    mode: str = "sparse",
    seed: int = 42,
    max_sparse_cases: int = 80,
    adversarial: bool = True,
    catalog: SchemaCatalog | None = None,
) -> list[CaseSpec]:
    catalog = catalog or load_catalog()
    critical = _critical_cases()
    if not adversarial:
        critical = [c for c in critical if c.adversarial == "none"]
    full = critical + _cartesian_batch(catalog)
    # de-dupe by id
    by_id: dict[str, CaseSpec] = {}
    for c in full:
        by_id[c.id] = c
    cases = list(by_id.values())

    if mode == "exhaustive":
        return sorted(cases, key=lambda c: c.id)

    # sparse: keep all critical, sample the rest
    crit_ids = {c.id for c in critical}
    rest = [c for c in cases if c.id not in crit_ids]
    rng = random.Random(seed)
    budget = max(0, max_sparse_cases - len(critical))
    picked = rng.sample(rest, k=min(budget, len(rest))) if rest else []
    out = critical + picked
    return sorted(out, key=lambda c: c.id)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    cfg_path = Path(path) if path else root / "config.yaml"
    defaults: dict[str, Any] = {
        "mode": "sparse",
        "seed": 42,
        "max_sparse_cases": 80,
        "adversarial": True,
        "output_dir": str(root / "out"),
    }
    if not cfg_path.exists():
        return defaults
    try:
        import yaml
    except ImportError:
        return defaults
    with cfg_path.open() as f:
        data = yaml.safe_load(f) or {}
    defaults.update(data)
    out = Path(str(defaults["output_dir"]))
    if not out.is_absolute():
        # Resolve relative to repo root (parent of combinatorial/)
        defaults["output_dir"] = str((root.parent / out).resolve())
    return defaults
