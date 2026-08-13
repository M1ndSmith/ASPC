"""Coverage metrics over schema axes vs executed cases."""
from __future__ import annotations

from typing import Any

from combinatorial.matrix import CaseSpec
from combinatorial.schema_catalog import axis_coverage_targets, load_catalog


def compute_coverage(cases: list[CaseSpec], results: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    catalog = load_catalog()
    targets = axis_coverage_targets(catalog)

    hit: dict[str, set[str]] = {k: set() for k in targets}
    for c in cases:
        hit["series_kinds"].add(c.series_kind)
        hit["expect_classes"].add(c.expect_class)
        hit["adversarial_kinds"].add(c.adversarial or "none")
        rs = c.params.get("ruleset")
        if rs:
            hit["rulesets"].add(str(rs))
        ct = c.params.get("chart_type")
        if ct:
            hit["chart_types"].add(str(ct))

    if results:
        for r in results:
            obs = r.get("observed") or {}
            # flatten nested
            blobs = [obs]
            if "batch" in obs and isinstance(obs["batch"], dict):
                blobs.append(obs["batch"])
            if "stream" in obs and isinstance(obs["stream"], dict):
                blobs.append(obs["stream"])
            for blob in blobs:
                for g in blob.get("gates") or []:
                    if isinstance(g, dict) and g.get("step"):
                        hit["gate_steps"].add(str(g["step"]))
                for rid in blob.get("rule_ids") or []:
                    if str(rid) in targets["nelson_ids"]:
                        hit["nelson_ids"].add(str(rid))
                    if str(rid) in targets["we_ids"]:
                        hit["we_ids"].add(str(rid))
                for pair in blob.get("batch_rule_ids") or []:
                    rid = pair[0] if isinstance(pair, (list, tuple)) else pair
                    if str(rid) in targets["nelson_ids"]:
                        hit["nelson_ids"].add(str(rid))
                for pair in blob.get("stream_rule_ids") or []:
                    rid = pair[0] if isinstance(pair, (list, tuple)) else pair
                    if str(rid) in targets["nelson_ids"]:
                        hit["nelson_ids"].add(str(rid))

    axes: dict[str, Any] = {}
    for name, target in targets.items():
        got = hit[name] & target
        pct = (100.0 * len(got) / len(target)) if target else 100.0
        axes[name] = {
            "target": len(target),
            "hit": len(got),
            "percent": round(pct, 1),
            "missing": sorted(target - got),
        }

    overall = sum(a["percent"] for a in axes.values()) / max(1, len(axes))
    return {"overall_percent": round(overall, 1), "axes": axes, "n_cases": len(cases)}
