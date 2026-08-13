# Resilience data catalog

Standards-mapped CSV corpus for judging `spc_core` behavior.

## Standards

| Standard | Encoded as |
|----------|------------|
| AIAG SPC | ≥25 Phase I points/subgroups; chart matrix; Nelson 1–8, Western Electric WE1–WE4, Wheeler |
| AIAG MSA-4 | %GRR &lt;10 / 10–30 / &gt;30; NDC ≥ 5; bias / linearity / stability |
| ISO 7870 | Variable + attribute chart coverage |
| Wheeler / Burr | Non-normal robust path (beyond-3σ only; subgroups preserved) |
| Six Sigma | Cp/Cpk/Pp/Ppk, nonparametric when transform fails |

## Layout

```
resilience_data/
  MANIFEST.json          # case specs + expect blocks
  generators/            # seeded NumPy builders
  cases/{spc,cleaning,msa,capability}/
scripts/resilience_report.py
tests/resilience/test_resilience_catalog.py
```

## Regenerate CSVs

```bash
python -m resilience_data --force
```

CSV encoding: empty cells are missing (`None`). Never the literal string `"None"`.

## Related: combinatorial dual-mode matrix

Finite batch + in-process Phase II streaming coverage of `spc_core` contracts lives in [`combinatorial/`](../combinatorial/) (sparse for CI, exhaustive locally):

```bash
python -m combinatorial report --mode sparse
# → combinatorial/out/JUDGMENT.md, COVERAGE.json, ENGINE_BEHAVIOR_REPORT.md
```

## Run judgment

```bash
# Assert every MANIFEST expect block
pytest tests/resilience -q

# Human-readable summary (writes resilience_data/JUDGMENT.md)
python scripts/resilience_report.py
```

## Adding a case

1. Add a builder in `generators/` and register it in `generators/__init__.py` `CASE_BUILDERS`.
2. Add a MANIFEST entry with `id`, `path`, `entry`, `columns`, `params`, and an empty `expect`.
3. `python -m resilience_data --force` to write the CSV.
4. Calibrate: `python scripts/calibrate_resilience.py` dumps the observed output for every case.
   Check it against the standards table, then freeze *bounds* into `expect` — see the two rules below.
5. Re-run `pytest tests/resilience`.

A case charting a mean shift or drift belongs at the `analyze_control_chart` entry, not `establish`:
both patterns are autocorrelated by construction, so the pipeline diverts them to EWMA and the
Shewhart run rules never execute. `imr_sustained_small_shift` is the exception — a 0.8σ shift stays
under the autocorrelation gate, so it exercises the run rules through the full pipeline.

## Expect schema

Judgment outcomes: `PASS` | `FAIL` | `ERROR` | `XFAIL`.

| Form | Meaning |
|------|---------|
| `"chart_type": "I-MR"` | exact match on an observed scalar |
| `{"cpk": {"lt": 1.33}}` | numeric bound — `lt` / `lte` / `gt` / `gte` / `eq` |
| `{"gates": {"normality": "warn"}}` | gate status; `"__absent__"` asserts the gate never ran |
| `{"rule_ids": {"includes": ["2"]}}` | the named run rule must have fired |
| `{"rule_ids": {"excludes": ["1"]}}` | the named rule must **not** have fired |
| `{"rule_ids": {"subset_of": ["1"]}}` | no rule outside this set may fire |
| `{"flag_counts": {"MISSING_SENSOR": 5}}` | exact count per `QualityFlag` |
| `"raises": "ValueError"` | the case must raise this exception type |
| `"raises_match": "at least 2 observations"` | substring the exception message must contain |

Two rules keep the corpus honest, both learned from cases that passed while asserting nothing:

**Bound, never freeze an observed float.** `{"cpk": {"lt": 1.282522727241782}}` is a recalibration
trap, not an assertion. Express the intent instead: a standard threshold (`lt: 1.33`) or a relation
(`cpk_lt_cp: true`).

**Assert the mechanism, not just the count.** `n_signals >= 1` passes when the wrong rule fires, and
`raises: ValueError` passes when the engine crashes internally instead of rejecting bad input
cleanly. Pin the rule id and the message. Where a signal belongs on the dispersion chart — an R/S
chart variance shift — assert `secondary_rule_ids`, since the primary chart barely moves.

`tests/resilience/test_resilience_catalog.py` enforces coverage: every `ChartType`, every
`QualityFlag`, every Nelson and Western Electric rule id, and all three rulesets must be asserted
by at least one case.

## Known sensitivity limits

`check_multimodal` uses Hartigan's dip test via the `diptest` dependency. If that package is
unavailable the code falls back to a histogram heuristic that only catches well-separated mixtures
(roughly ≥6σ apart), versus ~4σ for the real test. The fallback is deliberately conservative: this
gate issues a hard STOP that blocks go-live, so a false alarm on an in-control process is worse
than missing a subtle mixture.

Neither test runs on attribute (count) data. The dip test assumes a continuous distribution, and
the tie-heavy ECDF of a handful of distinct integers reads as multimodal — in-control Poisson
counts produced p = 0.003. The `p/np/c/u_in_control` cases assert `"multimodal": "__absent__"` to
hold that line.
