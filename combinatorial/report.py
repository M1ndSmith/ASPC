"""Judgment markdown + engine behavior report writers."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from combinatorial.coverage import compute_coverage
from combinatorial.dual_runner import run_matrix
from combinatorial.matrix import build_matrix, load_config


def write_judgment(summary: dict[str, Any], coverage: dict[str, Any], out_dir: Path) -> Path:
    counts = summary.get("counts") or {}
    lines = [
        "# Combinatorial judgment report",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        f"Mode: `{summary.get('mode')}` · seed={summary.get('seed')} · cases={summary.get('n_cases')}",
        "",
        "| Status | Count |",
        "|--------|------:|",
    ]
    for st in ("PASS", "FAIL", "ERROR"):
        lines.append(f"| {st} | {counts.get(st, 0)} |")
    lines += [
        "",
        f"## Coverage (overall {coverage.get('overall_percent')}%)",
        "",
        "| Axis | Hit | Target | % | Missing |",
        "|------|----:|-------:|--:|---------|",
    ]
    for name, ax in (coverage.get("axes") or {}).items():
        missing = ", ".join(f"`{m}`" for m in (ax.get("missing") or [])[:8])
        if len(ax.get("missing") or []) > 8:
            missing += ", …"
        lines.append(
            f"| {name} | {ax['hit']} | {ax['target']} | {ax['percent']} | {missing or '—'} |"
        )

    lines += ["", "## Per-case results", "", "| id | status | expect | adversarial | notes |", "|----|--------|--------|-------------|-------|"]
    for r in summary.get("results") or []:
        notes = "; ".join(r.get("mismatches") or [])[:100]
        lines.append(
            f"| `{r['id']}` | {r['status']} | {r.get('expect_class')} | {r.get('adversarial')} | {notes} |"
        )

    fails = [r for r in summary.get("results") or [] if r["status"] in ("FAIL", "ERROR")]
    if fails:
        lines += ["", "## Failures detail", ""]
        for r in fails:
            lines.append(f"### `{r['id']}` — {r['status']}")
            for m in r.get("mismatches") or []:
                lines.append(f"- {m}")
            exc = (r.get("observed") or {}).get("exception")
            if exc:
                lines.append(f"- exception: `{exc.get('type')}`: {exc.get('message')}")
            lines.append("")

    path = out_dir / "JUDGMENT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "COVERAGE.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    return path


def write_engine_report(summary: dict[str, Any], coverage: dict[str, Any], out_dir: Path) -> Path:
    """Narrative report of how spc_core behaves under the matrix."""
    results = summary.get("results") or []
    by_expect: dict[str, list] = defaultdict(list)
    for r in results:
        by_expect[str(r.get("expect_class"))].append(r)

    parity = [r for r in results if r.get("expect_class") == "phase2_signals"]
    parity_ok = sum(1 for r in parity if r["status"] == "PASS")
    adv = [r for r in results if r.get("expect_class") == "adversarial_behavior"]
    raises = [r for r in results if r.get("expect_class") == "raises"]
    stops = [r for r in results if r.get("expect_class") == "stop_unfrozen"]

    gate_counter: Counter[str] = Counter()
    for r in results:
        obs = r.get("observed") or {}
        for blob in (obs, obs.get("batch") or {}, obs.get("stream") or {}):
            if not isinstance(blob, dict):
                continue
            for g in blob.get("gates") or []:
                if isinstance(g, dict):
                    gate_counter[f"{g.get('step')}:{g.get('status')}"] += 1

    lines = [
        "# SPC Core Engine Behavior Report",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        f"Matrix mode: `{summary.get('mode')}` · cases={summary.get('n_cases')} · "
        f"PASS={summary.get('counts', {}).get('PASS', 0)} "
        f"FAIL={summary.get('counts', {}).get('FAIL', 0)} "
        f"ERROR={summary.get('counts', {}).get('ERROR', 0)}",
        "",
        "This report is produced by the combinatorial dual-mode pipeline "
        "(`python -m combinatorial report`). It summarizes how `spc_core` "
        "handles Phase I establishment, Phase II evaluation, and adversarial streams.",
        "",
        "## 1. Phase I gate order and freeze semantics",
        "",
        "The `establish()` pipeline runs gated checks (MSA → range → missing → ACF → "
        "multimodal/normality → chart → freeze). Any gate with status `stop` sets "
        "`frozen=False`; warnings still allow freeze when no STOP fired.",
        "",
        "Observed gate status counts in this run:",
        "",
    ]
    for k, v in sorted(gate_counter.items()):
        lines.append(f"- `{k}`: {v}")
    if not gate_counter:
        lines.append("- (no gate payloads recorded in this slice)")

    lines += [
        "",
        f"STOP/unfrozen expect class: {len(stops)} cases "
        f"({sum(1 for r in stops if r['status']=='PASS')} PASS).",
        "",
        "## 2. Chart auto-selection vs forced ChartType",
        "",
        "Forced `chart_type` in matrix params exercises each `ChartType` enum value via "
        "`analyze_control_chart` / `establish`. Attribute charts require "
        "`sample_sizes` / `opportunities`; subgroup charts require `subgroup_ids`.",
        "",
        f"Chart-type axis coverage: "
        f"{coverage.get('axes', {}).get('chart_types', {}).get('percent', '?')}%.",
        "",
        "## 3. Ruleset differences",
        "",
        "- `nelson`: full Nelson 1–8 pattern rules on Shewhart charts.",
        "- `western_electric`: WE1–WE4 subset.",
        "- `wheeler`: beyond-limits only (used on non-normal / forced Wheeler paths).",
        "",
        f"Ruleset coverage: {coverage.get('axes', {}).get('rulesets', {}).get('percent', '?')}%.",
        "",
        "## 4. Phase II: frozen limits and batch↔stream parity",
        "",
        "`Phase2Evaluator` never recomputes limits. For `adversarial=none` parity cases, "
        "`evaluate_batch` (or sequential `observe_subgroup`) must match streamed observes "
        "on the **rule_id multiset**.",
        "",
        f"Parity cases: {parity_ok}/{len(parity)} PASS.",
        "",
    ]
    for r in parity:
        obs = r.get("observed") or {}
        lines.append(
            f"- `{r['id']}`: {r['status']} parity={obs.get('parity')} "
            f"batch={obs.get('batch_rule_ids')} stream={obs.get('stream_rule_ids')}"
        )

    lines += [
        "",
        "## 5. Cleaning / capability / malformed edges",
        "",
        f"Raise-expect cases: {sum(1 for r in raises if r['status']=='PASS')}/{len(raises)} PASS.",
        "",
    ]
    for r in raises:
        exc = (r.get("observed") or {}).get("exception") or {}
        lines.append(
            f"- `{r['id']}`: {r['status']} → {exc.get('type', '—')}: {exc.get('message', '')}"
        )

    lines += [
        "",
        "## 6. Adversarial streaming behavior",
        "",
        "| Case | Adversarial | Behavior tag | Status | Notes |",
        "|------|-------------|--------------|--------|-------|",
    ]
    for r in adv:
        obs = r.get("observed") or {}
        lines.append(
            f"| `{r['id']}` | {r.get('adversarial')} | {obs.get('behavior') or r.get('behavior')} | "
            f"{r['status']} | parity={obs.get('parity')} |"
        )
    lines += [
        "",
        "Interpretation:",
        "",
        "- `jitter_delay`: timing only — state must match canonical parity.",
        "- `out_of_order`: Phase II is order-sensitive; divergence is expected and recorded.",
        "- `duplicate_index` / `watermark_jump` / `split_brain_seed`: probe restart and "
        "idempotency-adjacent behavior; parity not required.",
        "",
        "## 7. Coverage and blind spots",
        "",
        f"Overall axis coverage: **{coverage.get('overall_percent')}%**.",
        "",
    ]
    for name, ax in (coverage.get("axes") or {}).items():
        miss = ax.get("missing") or []
        if miss:
            lines.append(f"- `{name}` missing: {', '.join(f'`{m}`' for m in miss[:12])}")
    lines += [
        "",
        "## Regenerate",
        "",
        "```bash",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m combinatorial report --mode sparse",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m combinatorial report --mode exhaustive",
        "```",
        "",
    ]
    path = out_dir / "ENGINE_BEHAVIOR_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def full_report(*, mode: str | None = None) -> dict[str, Any]:
    cfg = load_config()
    mode = mode or cfg.get("mode", "sparse")
    summary = run_matrix(mode=mode, write_fixtures=True)
    cases = build_matrix(
        mode=mode,
        seed=int(summary["seed"]),
        max_sparse_cases=int(cfg.get("max_sparse_cases", 80)),
        adversarial=bool(cfg.get("adversarial", True)),
    )
    coverage = compute_coverage(cases, summary.get("results"))
    out = Path(cfg["output_dir"])
    write_judgment(summary, coverage, out)
    write_engine_report(summary, coverage, out)
    return {"summary": summary, "coverage": coverage, "out": str(out)}
