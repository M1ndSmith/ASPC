# Combinatorial judgment report

Generated: 2026-08-12T16:04:59.339527+00:00
Mode: `exhaustive` · seed=42 · cases=56

| Status | Count |
|--------|------:|
| PASS | 56 |
| FAIL | 0 |
| ERROR | 0 |

## Coverage (overall 85.6%)

| Axis | Hit | Target | % | Missing |
|------|----:|-------:|--:|---------|
| chart_types | 9 | 9 | 100.0 | — |
| rulesets | 3 | 3 | 100.0 | — |
| quality_flags | 0 | 7 | 0.0 | `EXCLUDED_INCOMPLETE`, `EXCLUDED_MAINTENANCE`, `IMPUTED_LOCF`, `MISSING_HUMAN`, `MISSING_SENSOR`, `ORIGINAL`, `RESTORED_FROM_BACKUP` |
| nelson_ids | 8 | 8 | 100.0 | — |
| we_ids | 4 | 4 | 100.0 | — |
| gate_steps | 7 | 10 | 70.0 | `gage_resolution`, `transform`, `valid_range` |
| expect_classes | 6 | 6 | 100.0 | — |
| series_kinds | 21 | 21 | 100.0 | — |
| adversarial_kinds | 6 | 6 | 100.0 | — |

## Per-case results

| id | status | expect | adversarial | notes |
|----|--------|--------|-------------|-------|
| `batch_attribute_c_nelson` | PASS | ok_freeze | none |  |
| `batch_attribute_np_nelson` | PASS | ok_freeze | none |  |
| `batch_attribute_p_nelson` | PASS | ok_freeze | none |  |
| `batch_attribute_u_nelson` | PASS | ok_freeze | none |  |
| `batch_heavy_tail_nelson` | PASS | ok_freeze | none |  |
| `batch_heavy_tail_western_electric` | PASS | ok_freeze | none |  |
| `batch_heavy_tail_wheeler` | PASS | ok_freeze | none |  |
| `batch_imr_constant_nelson` | PASS | raises | none |  |
| `batch_imr_constant_western_electric` | PASS | raises | none |  |
| `batch_imr_constant_wheeler` | PASS | raises | none |  |
| `batch_imr_in_control_nelson` | PASS | ok_freeze | none |  |
| `batch_imr_in_control_western_electric` | PASS | ok_freeze | none |  |
| `batch_imr_in_control_wheeler` | PASS | ok_freeze | none |  |
| `batch_imr_mean_shift_nelson` | PASS | ok_freeze | none |  |
| `batch_imr_mean_shift_western_electric` | PASS | ok_freeze | none |  |
| `batch_imr_mean_shift_wheeler` | PASS | ok_freeze | none |  |
| `batch_imr_overflow_nelson` | PASS | raises | none |  |
| `batch_imr_overflow_western_electric` | PASS | raises | none |  |
| `batch_imr_overflow_wheeler` | PASS | raises | none |  |
| `batch_xbar_r_nelson` | PASS | ok_freeze | none |  |
| `chartcov_C` | PASS | ok_freeze | none |  |
| `chartcov_CUSUM` | PASS | ok_freeze | none |  |
| `chartcov_EWMA` | PASS | ok_freeze | none |  |
| `chartcov_I_MR` | PASS | ok_freeze | none |  |
| `chartcov_NP` | PASS | ok_freeze | none |  |
| `chartcov_P` | PASS | ok_freeze | none |  |
| `chartcov_U` | PASS | ok_freeze | none |  |
| `chartcov_Xbar_R` | PASS | ok_freeze | none |  |
| `chartcov_Xbar_S` | PASS | ok_freeze | none |  |
| `crit_adv_dup` | PASS | adversarial_behavior | duplicate_index |  |
| `crit_adv_jitter` | PASS | adversarial_behavior | jitter_delay |  |
| `crit_adv_ooo` | PASS | adversarial_behavior | out_of_order |  |
| `crit_adv_split_brain` | PASS | adversarial_behavior | split_brain_seed |  |
| `crit_adv_watermark` | PASS | adversarial_behavior | watermark_jump |  |
| `crit_capability_bad_specs` | PASS | raises | none |  |
| `crit_constant_series` | PASS | raises | none |  |
| `crit_establish_empty` | PASS | raises | none |  |
| `crit_establish_n1` | PASS | raises | none |  |
| `crit_establish_n2_freeze` | PASS | ok_freeze | none |  |
| `crit_imr_in_control_nelson` | PASS | phase2_signals | none |  |
| `crit_imr_mean_shift_parity` | PASS | phase2_signals | none |  |
| `crit_imr_reject_subgroup` | PASS | phase2_rejects | none |  |
| `crit_multimodal_stop` | PASS | stop_unfrozen | none |  |
| `crit_nan_inf` | PASS | raises | none |  |
| `crit_p_zero_n` | PASS | raises | none |  |
| `crit_sentinel` | PASS | ok_freeze | none |  |
| `crit_type_mismatch` | PASS | raises | none |  |
| `crit_u_zero_opp` | PASS | raises | none |  |
| `crit_xbar_subgroup_reject_scalar` | PASS | phase2_rejects | none |  |
| `parity_imr_nelson` | PASS | phase2_signals | none |  |
| `parity_imr_western_electric` | PASS | phase2_signals | none |  |
| `parity_imr_wheeler` | PASS | phase2_signals | none |  |
| `parity_xbar_r_nelson` | PASS | phase2_signals | none |  |
| `rules_nelson_alternating` | PASS | ok_freeze | none |  |
| `rules_nelson_trend` | PASS | ok_freeze | none |  |
| `rules_we_mean_shift` | PASS | ok_freeze | none |  |
