# Benchmarking

Quantitative evidence for ASPC: **performance** (in-process), **statistical accuracy** (formula / synthetic truth), and **robustness** (resilience catalog). Reproduce with [benchmarks/README.md](../../benchmarks/README.md).

Product narrative: [problem and solution](problem-and-solution.md). Feature map: [capabilities](capabilities.md).

**CI:** `benchmarks/accuracy.py` is a blocking CI step. Performance is **not** CI-gated (host noise); re-run locally and refresh [RESULTS.md](../../benchmarks/RESULTS.md) before citing numbers in a proposal.

## How to read these numbers

| Layer | What it proves | What it does not prove |
|-------|----------------|-------------------------|
| Performance | Core math + Phase II eval speed on one host | Full-stack MQTT→UI latency; multi-process / multi-host load |
| Accuracy | Published SPC formulas and rule engines on known inputs | AIAG MSA-4 worked examples; every Minitab/JMP dialog option; Box-Cox Cpk analytic parity |
| Resilience | Sad-path / gate / rule-id behavior under a fixed corpus | Your plant’s sensor network or MES quirks |

## Standards alignment

| Standard | What ASPC encodes | Where proven |
|----------|-------------------|--------------|
| AIAG SPC | ≥25 Phase I points/subgroups (checklist); chart matrix; Nelson 1–8, Western Electric WE1–WE4, Wheeler | [resilience_data/](../../resilience_data/), `spc_core/rules.py` |
| AIAG MSA-4 | %GRR bands (&lt;10 / 10–30 / ≥30), NDC ≥ 5, bias / linearity / stability, resolution 10:1 | Resilience MSA cases; pipeline STOP when study inputs supplied; **not** AIAG Table 6.1 byte-match |
| ISO 7870 | Variable + attribute chart coverage | Resilience + `ChartType` surface |
| Wheeler | Non-normal robust path (beyond-3σ only; subgroups preserved) | Resilience `heavy_tail_wheeler*`; `ruleset_applied: wheeler` |
| Six Sigma | Cp/Cpk/Pp/Ppk; transformed / nonparametric routing | Resilience capability cases; `capability_analysis` |

## A. Performance

Command:

```bash
.venv/bin/python benchmarks/performance.py
```

Committed snapshot (exact table, host, UTC date): **[benchmarks/RESULTS.md](../../benchmarks/RESULTS.md)**. Re-run on your hardware before proposals — do not treat the snapshot as an SLA.

### Interpretation

- **Phase I** (`establish`) includes optional MSA path, missing classification, ACF, normality / multimodal (`diptest`), chart, and freeze logic.
- **Phase II** (`Phase2Evaluator.observe`) is the in-process hot path. Network and broker overhead dominate in production; budget those separately.
- Not measured: Compose-stack end-to-end alert latency.

## B. Formula and synthetic-truth checks (28)

Command (also run in CI):

```bash
.venv/bin/python benchmarks/accuracy.py
```

**28/28 checks** in the current suite cover:

| Area | What is checked |
|------|-----------------|
| Constants | `d2`, `E2`, `A2`, `D3`, `D4` vs handbook targets |
| I-MR | Center / UCL / LCL / MR UCL vs hand formulas |
| Xbar-R | `X̄̄ ± A2·R̄`, `D3/D4·R̄` on constructed subgroups |
| P chart | `p̄ ± 3√(p̄(1-p̄)/n)` |
| Run rules | Nelson 1, 2, 3 and Western Electric WE1 on constructed series |
| MSA | Synthetic excellent gage %GRR &lt; 10; poor gage %GRR &gt; 30; ordering |
| Capability | Large normal sample Cp near (USL−LSL)/(6σ); centered Cpk ≈ Cp |
| Guards | Empty `establish([])` raises domain `ValueError` |

### Explicitly out of scope for this suite

- AIAG MSA-4 Table 6.1 (or other published worked-example) numerical parity
- JMP / Minitab dialog option bit-identity
- Box-Cox / Yeo-Johnson Cpk vs a closed-form analytic reference on transformed specs
- Nelson 4–8 / full WE set (those are covered in the **resilience** catalog, not `accuracy.py`)

This is the right bar for an open engine: **formula fidelity and directional MSA/capability behavior**, not “identical to Minitab.”

## C. Robustness (resilience catalog)

Source of truth: [`resilience_data/`](../../resilience_data/) (55 MANIFEST cases) and `tests/resilience/`.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/resilience -q
.venv/bin/python scripts/resilience_report.py   # writes resilience_data/JUDGMENT.md (gitignored)
```

### Coverage (asserted, not merely present)

- All nine `ChartType` values
- Quality flags used in cleaning (LOCF, sensor, maintenance, human, backup, incomplete, …)
- Nelson 1–8 and Western Electric WE1–WE4 via `rule_ids` / `includes` / `subset_of`
- Rulesets `nelson`, `western_electric`, `wheeler` (Wheeler asserts zone rules stay suppressed)
- Gates including MSA, autocorrelation, multimodal, normality, **range** (when `valid_range` set), freeze
- Raises with `raises_match` (message, not only exception type)

### Highlighted outcomes from hardening

| Topic | Result |
|-------|--------|
| Multimodal false positives (pure normal, with `diptest`) | ~0.25% in a large sweep; clear 5σ mixtures detected |
| Attribute (count) data | Dip / Gaussian normality skipped — in-control P/NP/C/U no longer false-STOP |
| Sensor sentinels | With `valid_range`, `-999` does not fire Nelson 1 as process OOC |
| Empty / all-NaN | Clear domain errors (no SciPy unpack crash) |
| R-chart variance shift | Asserted on `secondary_rule_ids`, not a vacuous `n_signals ≥ 0` |

Mutation checks (breaking R secondary signals, Wheeler→Nelson leak, removing input guards, ignoring `valid_range`) make the corresponding cases **FAIL**.

## D. Comparison matrix (feature posture)

Architecture comparison — not a declaration that ASPC wins every statistician’s preferred dialog. Commercial seat pricing changes often; verify before procurement.

| Concern | ASPC | Typical desktop SPC (e.g. Minitab / JMP) | Typical minimal Python chart snippet | R `qcc`-class packages |
|---------|------|------------------------------------------|--------------------------------------|-------------------------|
| Streaming Phase II | Yes (Kafka/MQTT path) | Desktop / project workflow | No | No |
| Frozen versioned limits | Yes | Analyst-managed | Rare | Rare |
| MSA STOP/WARN | When MSA inputs provided; else warn only | Separate study UI | Usually absent | Partial |
| REST + WebSocket | Yes | No (desktop) | No | No |
| License | MIT | Commercial seats | Often MIT | Often GPL |
| Open formula benches | `benchmarks/` + resilience | Proprietary | Usually none | Varies |
| Multi-tenant API | **No** (single deployment; JWT / API-key auth only) | N/A | N/A | N/A |

## Reproducing for a report

1. Record host (`uname -a`, Python version) and UTC time.
2. Run `benchmarks/performance.py`; update or attach [RESULTS.md](../../benchmarks/RESULTS.md).
3. Run `benchmarks/accuracy.py` (must exit 0; also CI).
4. Run resilience pytest / `resilience_report.py`; attach PASS counts.
5. State clearly if Compose e2e latency was or was not measured.

## Related docs

- [benchmarks/README.md](../../benchmarks/README.md)
- [benchmarks/RESULTS.md](../../benchmarks/RESULTS.md)
- [resilience_data/README.md](../../resilience_data/README.md)
- [development.md](../development.md)
- [capabilities.md](capabilities.md) — Known limitations
