# Capabilities

What ASPC actually does, mapped to manufacturing work — not a feature checklist alone. Math detail: [concepts](../concepts.md). Pipeline detail: [pipeline](../pipeline.md).

## Statistical capabilities

### Variable charts

| Chart | When | Notes |
|-------|------|--------|
| **I-MR** | Continuous, one-at-a-time | Individuals + moving range; secondary MR panel also gets run rules |
| **Xbar-R** | Subgroup size 2–8 | Mean chart + range chart; dispersion shifts show on **R**, not only on Xbar |
| **Xbar-S** | Subgroup size ≥ 9 | Mean + standard deviation |

Rulesets on Shewhart charts:

- `nelson` — Nelson 1–8 (default)
- `western_electric` — WE1–WE4
- `wheeler` — beyond 3σ only (non-normal robust path)

A sustained **sub-sigma** mean shift that never breaches 3σ is caught by run rules (e.g. Nelson 2 / 6) when the series stays on the Shewhart route. Large steps and long drifts often trip autocorrelation and correctly divert to EWMA — those patterns are still asserted at the chart layer in the resilience catalog.

### Attribute charts

| Chart | Data |
|-------|------|
| **NP** | Defectives, fixed sample size |
| **P** | Defectives, variable sample size |
| **C** | Defect counts, fixed opportunity |
| **U** | Defect counts, variable opportunity |

Limits use binomial / Poisson formulas. Distribution gates that assume continuous data are **skipped** so in-control count processes are not false-STOPped by Hartigan’s dip test.

### Small-shift / autocorrelated series

| Chart | Role |
|-------|------|
| **EWMA** | λ and L configurable; time-varying limits |
| **CUSUM** | Tabular two-sided C+ / C− |

Selected automatically when Phase I detects significant lag-1 autocorrelation on continuous data (`establish(..., autocorrelated_chart="EWMA"|"CUSUM")`).

### Measurement System Analysis

| Study | Purpose |
|-------|---------|
| Gage R&R ANOVA | %GRR, NDC, acceptability; range fallback when design is unbalanced |
| Bias | Mean bias vs reference, significance |
| Linearity | Slope / R² across reference range |
| Stability | Time stability of the measurement system |
| Resolution gate | 10:1 rule vs tolerance |
| Continuous MSA | Streaming bias / R tracking with calibration alerts |

AIAG MSA-4 style bands on the **computed** Gage R&R when study inputs are passed into `establish` (or called directly): %GRR &lt; 10 excellent (with NDC ≥ 5), 10–30 conditional, ≥ 30 or NDC &lt; 5 unacceptable — and that can **STOP** Phase I. If MSA inputs are omitted, the MSA gate is `warn` and limits may still freeze.

### Process capability

`capability_analysis` routes:

- Parametric Cp / Cpk / Pp / Ppk (and Cpm with target)
- Transformed path (Box-Cox / Yeo-Johnson / log) when normality fails and transform restores it
- Nonparametric percentile-based indices when transform fails

DPMO and sigma level helpers are analytic, not lookup tables.

## Operational capabilities

### Batch

- **Python library** — `from spc_core import establish, analyze_control_chart, …`
- **CLI** — `aspc control-chart`, capability, MSA commands ([cli](../cli.md))
- **CSV / frames** — ingest helpers with column detection

### Streaming

- Kafka / Redpanda consumer in `services/stream_engine`
- MQTT → Redpanda bridge
- Per-stream keys, register → go-live after Phase I freeze
- Redis pub/sub for live alert fan-out
- WebSocket + SSE on the API ([api](../api.md))

### Persistence and apps

| Layer | Options |
|-------|---------|
| Storage | SQLite (embedded) or TimescaleDB (hypertables + analysis tables) |
| Auth | JWT for users, API keys for services |
| UI | Next.js operator dashboard (charts, signals, stream status) |
| Ops | Docker Compose, Alembic migrations, Grafana |

## Data quality capabilities

SPC fails silently when dirty data looks like process signals. ASPC treats measurement failure as a first-class concern:

| Mechanism | Behavior |
|-----------|----------|
| **classify_missing** | Short gaps ≤ LOCF max may be forward-filled and flagged `IMPUTED_LOCF`; longer gaps become `MISSING_SENSOR` and are excluded from chart math |
| **Reason codes** | maintenance / human / incomplete / backup map to explicit `QualityFlag`s |
| **valid_range on establish** | When `valid_range` is passed: out-of-physical-range readings (e.g. `-999`) are blanked as measurement failures — they must **not** fire Nelson rule 1 as “OOC process.” Opt-in; see Known limitations. |
| **Multimodal STOP** | Hartigan dip test (`diptest`); clear mixtures block go-live so you stratify first |
| **Input guards** | Empty / all-missing series raise clear `ValueError`s, not SciPy unpack crashes |
| **Incomplete subgroups** | Optional exclude of ragged last subgroups on Xbar charts |

## Manufacturing patterns

See [use-cases](use-cases.md) for stamping (streaming I-MR), tablet weight (batch Xbar-R + capability), and vision flash (P chart) integration patterns.

## Known limitations

Black-belt / practitioner caveats — not marketing footnotes:

| Topic | Behavior |
|-------|----------|
| **MSA opt-in** | MSA is **not** required to freeze. Absence → `warn`. STOP only when study data is supplied and %GRR / NDC / resolution fail. |
| **`valid_range` opt-in** | Without `valid_range`, sensor sentinels can still enter the chart and fire rule 1 as if they were process signals. |
| **Autocorrelation routing** | Continuous series with \|lag-1 ACF\| above the threshold (including **negative** ACF) route to EWMA/CUSUM. Oscillation / operator tampering often needs Shewhart run rules (e.g. Nelson 4 / 7), not smoothing — assert those at the chart layer or raise the ACF threshold deliberately. |
| **Multimodal sensitivity** | Hartigan dip (`diptest`) plus a conservative histogram fallback. Clear, well-separated mixtures STOP; subtle ~2–3σ mixtures may be missed. |
| **Not a CSV / Part 11 pack** | Reports are HTML/JSON via CLI/API. Regulated computer-system validation is out of product scope. |
| **Not a substitute for MSA planning** | Parts × operators × trials design remains an engineering decision. |

Also: not every plant IT stack is turnkey; not LLM-driven SPC — the engine is deterministic statistics. Measured limits: [benchmarking](benchmarking.md).
