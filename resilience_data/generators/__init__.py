"""Catalog of case builders keyed by MANIFEST id."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import capability as cap
from . import cleaning as cln
from . import msa, spc

Columns = dict[str, list[Any]]
Builder = Callable[[], Columns]

# id → (relative path under cases/, builder)
CASE_BUILDERS: dict[str, tuple[str, Builder]] = {
    # A. Happy path
    "imr_in_control_n50": ("spc/imr_in_control_n50.csv", spc.imr_in_control),
    "xbar_r_25x5": ("spc/xbar_r_25x5.csv", spc.xbar_r),
    "xbar_s_25x10": ("spc/xbar_s_25x10.csv", lambda: spc.xbar_s()),
    "p_in_control": ("spc/p_in_control.csv", spc.attribute_p),
    "np_in_control": ("spc/np_in_control.csv", spc.attribute_np),
    "c_in_control": ("spc/c_in_control.csv", spc.attribute_c),
    "u_in_control": ("spc/u_in_control.csv", spc.attribute_u),
    # B. OOC
    "imr_mean_shift": ("spc/imr_mean_shift.csv", spc.imr_mean_shift),
    "imr_single_spike": ("spc/imr_single_spike.csv", spc.imr_single_spike),
    "imr_trend": ("spc/imr_trend.csv", spc.imr_trend),
    "xbar_r_variance_increase": (
        "spc/xbar_r_variance_increase.csv",
        spc.xbar_r_variance_increase,
    ),
    # B2. Run-rule patterns (Nelson 2-8 / Western Electric)
    "imr_sustained_small_shift": (
        "spc/imr_sustained_small_shift.csv",
        spc.imr_sustained_small_shift,
    ),
    "imr_trend_nelson3": ("spc/imr_trend_nelson3.csv", spc.imr_trend_nelson3),
    "imr_alternating_nelson4": (
        "spc/imr_alternating_nelson4.csv",
        spc.imr_alternating_nelson4,
    ),
    # C. Distribution routes
    "normal_path": ("spc/normal_path.csv", spc.normal_path),
    "skewed_boxcox": ("spc/skewed_boxcox.csv", spc.skewed_boxcox),
    "heavy_tail_wheeler": ("spc/heavy_tail_wheeler.csv", spc.heavy_tail_wheeler),
    "heavy_tail_wheeler_subgroup": (
        "spc/heavy_tail_wheeler_subgroup.csv",
        spc.heavy_tail_wheeler_subgroup,
    ),
    "autocorrelated_ewma": ("spc/autocorrelated_ewma.csv", spc.autocorrelated_ewma),
    "multimodal_stop": ("spc/multimodal_stop.csv", spc.multimodal_stop),
    # D. Cleaning
    "gap_short_locf": ("cleaning/gap_short_locf.csv", cln.gap_short_locf),
    "gap_long_hold": ("cleaning/gap_long_hold.csv", cln.gap_long_hold),
    "reason_maintenance": ("cleaning/reason_maintenance.csv", cln.reason_maintenance),
    "reason_human": ("cleaning/reason_human.csv", cln.reason_human),
    "reason_backup": ("cleaning/reason_backup.csv", cln.reason_backup),
    "reason_incomplete": ("cleaning/reason_incomplete.csv", cln.reason_incomplete),
    "sensor_sentinel_range": (
        "cleaning/sensor_sentinel_range.csv",
        cln.sensor_sentinel_range,
    ),
    "sensor_sentinel_long": (
        "cleaning/sensor_sentinel_long.csv",
        cln.sensor_sentinel_long,
    ),
    "incomplete_subgroup": ("spc/incomplete_subgroup.csv", spc.incomplete_subgroup),
    "empty_series": ("spc/empty_series.csv", spc.empty_series),
    "all_nan": ("spc/all_nan.csv", spc.all_nan),
    "constant_series": ("spc/constant_series.csv", spc.constant_series),
    "p_zero_n": ("spc/p_zero_n.csv", spc.p_zero_n),
    "u_zero_opportunity": ("spc/u_zero_opportunity.csv", spc.u_zero_opportunity),
    # E. MSA
    "gage_rr_excellent": (
        "msa/gage_rr_excellent.csv",
        lambda: msa.gage_rr(quality="excellent"),
    ),
    "gage_rr_marginal": (
        "msa/gage_rr_marginal.csv",
        lambda: msa.gage_rr(quality="marginal", seed=11),
    ),
    "gage_rr_poor": (
        "msa/gage_rr_poor.csv",
        lambda: msa.gage_rr(quality="poor", seed=303),
    ),
    "gage_rr_unbalanced": ("msa/gage_rr_unbalanced.csv", msa.gage_rr_unbalanced),
    "ndc_fail": (
        "msa/ndc_fail.csv",
        lambda: msa.gage_rr(quality="ndc_fail", seed=304),
    ),
    "bias_significant": ("msa/bias_significant.csv", msa.bias_significant),
    "linearity_ok": ("msa/linearity_ok.csv", msa.linearity_ok),
    "stability_ok": ("msa/stability_ok.csv", msa.stability_ok),
    # F. Capability
    "cap_excellent": (
        "capability/cap_excellent.csv",
        lambda: cap.capability(kind="excellent"),
    ),
    "cap_off_center": (
        "capability/cap_off_center.csv",
        lambda: cap.capability(kind="off_center", seed=402),
    ),
    "cap_high_variation": (
        "capability/cap_high_variation.csv",
        lambda: cap.capability(kind="high_variation", seed=403),
    ),
    "cap_skewed": (
        "capability/cap_skewed.csv",
        lambda: cap.capability(kind="skewed", seed=404),
    ),
    # G. Phase I insufficiency
    "too_few_points_n10": ("spc/too_few_points_n10.csv", spc.too_few_points),
}
