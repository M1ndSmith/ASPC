# SPC concepts (as implemented)

This page describes the statistics ASPC actually computes. Source of truth: `spc_core/`.

## Chart types

`ChartType` in [`spc_core/models.py`](../spc_core/models.py):

| Value | Use when |
|-------|----------|
| `I-MR` | Continuous data, subgroup size 1 (individuals + moving range) |
| `Xbar-R` | Continuous, subgroup size 2–8 |
| `Xbar-S` | Continuous, subgroup size ≥ 9 |
| `NP` | Defectives, constant sample size |
| `P` | Defectives, variable sample size |
| `C` | Defects (counts), constant opportunity |
| `U` | Defects, variable opportunity |
| `EWMA` | Small shifts / autocorrelated series (λ, L) |
| `CUSUM` | Small persistent shifts (tabular two-sided C+/C−) |

Auto-selection (`select_chart_type` / `analyze_control_chart`):

- Continuous + no subgroups → `I-MR`
- Continuous + median subgroup size ≤ 8 → `Xbar-R`; else `Xbar-S`
- Attribute with `sample_sizes` → `P` if sizes vary, else `NP`
- Attribute with `opportunities` → `U` if sizes vary, else `C`

You can override with an explicit `chart_type`.

## Run rules

[`spc_core/rules.py`](../spc_core/rules.py) — `RuleEngine(ruleset=...)`:

| Ruleset | Behavior |
|---------|----------|
| `nelson` (default) | Nelson rules 1–8 (zones use inclusive boundaries) |
| `western_electric` | Western Electric subset |
| `wheeler` | Points-outside-limits only (robust path for non-normal data) |

Primary and secondary panels (R / MR / S) are both evaluated when present (`secondary_signals` on `ControlChartResult`).

## Measurement System Analysis (MSA)

[`spc_core/msa.py`](../spc_core/msa.py) and continuous tracking in [`spc_core/msa_stream.py`](../spc_core/msa_stream.py).

| Study | Function | Key outputs |
|-------|----------|-------------|
| Gage R&R (ANOVA) | `gage_rr_anova` | `grr_percent`, `ndc`, `acceptability` |
| Gage R&R (Range) | `gage_rr_range` | Same flat `grr_percent` contract |
| Bias | `bias_study` | mean bias, t-test, `is_significant` |
| Linearity | `linearity_study` | slope, R², `is_linear` |
| Stability | `stability_study` | I-MR style limits, `is_stable` |

Acceptability (AIAG MSA-4 style):

- `%GRR < 10` → Excellent (also requires NDC ≥ 5)
- `10 ≤ %GRR < 30` → Acceptable / conditional
- `%GRR ≥ 30` or NDC < 5 → Unacceptable

Gates:

- `ndc_gate(ndc, minimum=5)` — discrimination for SPC
- `gage_resolution_gate(resolution, tolerance, ratio=10.0)` — 10:1 rule

Continuous MSA (`ContinuousMSA`) tracks EWMA bias drift and a rolling R chart on reference injections; emits `CalibrationAlert` when bias exceeds threshold.

## Process capability

[`spc_core/capability.py`](../spc_core/capability.py):

- Parametric: Cp, Cpk, Pp, Ppk, Cpm (when target given), sigma level, DPMO
- Nonparametric: percentile-based Ppk when normality fails and transform does not restore it
- Transformed path: Box-Cox / Yeo-Johnson / log when transform restores normality; specs are transformed consistently
- Helpers: `dpmo_to_sigma`, `sigma_to_dpmo` (analytic via `norm.ppf`, not bucket tables)

`capability_analysis(..., method="auto")` routes by normality and transform success.

## Normality, transforms, multimodality

[`spc_core/normality.py`](../spc_core/normality.py), [`spc_core/multimodal.py`](../spc_core/multimodal.py):

- Normality: Anderson-Darling, Shapiro-Wilk, Kolmogorov-Smirnov (Lilliefors when `statsmodels` is available), skewness / kurtosis
- Autocorrelation: lag-1 ACF vs configurable threshold (default 0.2); recommends EWMA / CUSUM when elevated
- Transforms: `apply_transform(..., method="auto")`
- Multimodal: Hartigan dip test (+ histogram peak heuristic); recommendation to **stratify** (pipeline STOP)

## Quality and distribution flags

From [`spc_core/models.py`](../spc_core/models.py):

**`QualityFlag`** (per measurement provenance — no silent imputation):

`ORIGINAL`, `IMPUTED_LOCF`, `MISSING_SENSOR`, `EXCLUDED_MAINTENANCE`, `EXCLUDED_INCOMPLETE`, `RESTORED_FROM_BACKUP`, `MISSING_HUMAN`

**`DistributionFlag`**:

`NORMAL`, `TRANSFORMED`, `NON_NORMAL_RAW` (Wheeler path)

**`Phase`**: `PHASE_I` | `PHASE_II`

## Limits and versioning

`ControlLimits` is a frozen Pydantic model. `version` is a 16-character SHA-256 content hash of chart type, subgroup size, components, and sigma. Phase II and the stream registry must reference this version explicitly.

See [pipeline.md](pipeline.md) for how these pieces are orchestrated.
