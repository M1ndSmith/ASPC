# SPC Core Engine Behavior Report

Generated: 2026-08-12T16:04:59.339917+00:00
Matrix mode: `exhaustive` · cases=56 · PASS=56 FAIL=0 ERROR=0

This report is produced by the combinatorial dual-mode pipeline (`python -m combinatorial report`). It summarizes how `spc_core` handles Phase I establishment, Phase II evaluation, and adversarial streams.

## 1. Phase I gate order and freeze semantics

The `establish()` pipeline runs gated checks (MSA → range → missing → ACF → multimodal/normality → chart → freeze). Any gate with status `stop` sets `frozen=False`; warnings still allow freeze when no STOP fired.

Observed gate status counts in this run:

- `autocorrelation:ok`: 9
- `autocorrelation:warn`: 8
- `chart:ok`: 17
- `freeze:blocked`: 1
- `freeze:ok`: 16
- `missing:ok`: 17
- `msa:warn`: 17
- `multimodal:ok`: 6
- `multimodal:stop`: 1
- `normality:ok`: 9
- `normality:warn`: 2

STOP/unfrozen expect class: 1 cases (1 PASS).

## 2. Chart auto-selection vs forced ChartType

Forced `chart_type` in matrix params exercises each `ChartType` enum value via `analyze_control_chart` / `establish`. Attribute charts require `sample_sizes` / `opportunities`; subgroup charts require `subgroup_ids`.

Chart-type axis coverage: 100.0%.

## 3. Ruleset differences

- `nelson`: full Nelson 1–8 pattern rules on Shewhart charts.
- `western_electric`: WE1–WE4 subset.
- `wheeler`: beyond-limits only (used on non-normal / forced Wheeler paths).

Ruleset coverage: 100.0%.

## 4. Phase II: frozen limits and batch↔stream parity

`Phase2Evaluator` never recomputes limits. For `adversarial=none` parity cases, `evaluate_batch` (or sequential `observe_subgroup`) must match streamed observes on the **rule_id multiset**.

Parity cases: 6/6 PASS.

- `crit_imr_in_control_nelson`: PASS parity=True batch=[] stream=[]
- `crit_imr_mean_shift_parity`: PASS parity=True batch=[('EWMA1', 39)] stream=[('EWMA1', 39)]
- `parity_imr_nelson`: PASS parity=True batch=[] stream=[]
- `parity_imr_western_electric`: PASS parity=True batch=[('WE2', 1), ('WE4', 1)] stream=[('WE2', 1), ('WE4', 1)]
- `parity_imr_wheeler`: PASS parity=True batch=[] stream=[]
- `parity_xbar_r_nelson`: PASS parity=True batch=[] stream=[]

## 5. Cleaning / capability / malformed edges

Raise-expect cases: 14/14 PASS.

- `batch_imr_constant_nelson`: PASS → ValueError: Data must not be constant.
- `batch_imr_constant_western_electric`: PASS → ValueError: Data must not be constant.
- `batch_imr_constant_wheeler`: PASS → ValueError: Data must not be constant.
- `batch_imr_overflow_nelson`: PASS → ValueError: Too many bins for data range. Cannot create 6 finite-sized bins.
- `batch_imr_overflow_western_electric`: PASS → ValueError: Too many bins for data range. Cannot create 6 finite-sized bins.
- `batch_imr_overflow_wheeler`: PASS → ValueError: Too many bins for data range. Cannot create 6 finite-sized bins.
- `crit_capability_bad_specs`: PASS → ValueError: Valid USL > LSL required for capability analysis
- `crit_constant_series`: PASS → ValueError: Data must not be constant.
- `crit_establish_empty`: PASS → ValueError: establish() requires at least 2 observations to compute control limits; received 0.
- `crit_establish_n1`: PASS → ValueError: establish() requires at least 2 observations to compute control limits; received 1.
- `crit_nan_inf`: PASS → ValueError: autodetected range of [100.0, inf] is not finite
- `crit_p_zero_n`: PASS → ValueError: P chart sample sizes must be > 0
- `crit_type_mismatch`: PASS → ValueError: could not convert string to float: 'not-a-number'
- `crit_u_zero_opp`: PASS → ValueError: U chart opportunities must be > 0

## 6. Adversarial streaming behavior

| Case | Adversarial | Behavior tag | Status | Notes |
|------|-------------|--------------|--------|-------|
| `crit_adv_dup` | duplicate_index | duplicate_observe | PASS | parity=True |
| `crit_adv_jitter` | jitter_delay | timing_noop | PASS | parity=True |
| `crit_adv_ooo` | out_of_order | order_sensitive | PASS | parity=False |
| `crit_adv_split_brain` | split_brain_seed | split_brain_seed | PASS | parity=True |
| `crit_adv_watermark` | watermark_jump | watermark_jump | PASS | parity=True |

Interpretation:

- `jitter_delay`: timing only — state must match canonical parity.
- `out_of_order`: Phase II is order-sensitive; divergence is expected and recorded.
- `duplicate_index` / `watermark_jump` / `split_brain_seed`: probe restart and idempotency-adjacent behavior; parity not required.

## 7. Coverage and blind spots

Overall axis coverage: **85.6%**.

- `quality_flags` missing: `EXCLUDED_INCOMPLETE`, `EXCLUDED_MAINTENANCE`, `IMPUTED_LOCF`, `MISSING_HUMAN`, `MISSING_SENSOR`, `ORIGINAL`, `RESTORED_FROM_BACKUP`
- `gate_steps` missing: `gage_resolution`, `transform`, `valid_range`

## Regenerate

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m combinatorial report --mode sparse
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m combinatorial report --mode exhaustive
```

