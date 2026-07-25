"""Normality assessment, autocorrelation gate, and transformations.

Pure numpy/scipy. No pandas, no I/O. The autocorrelation check is implemented with
numpy directly so statsmodels is not a required dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class NormalityResult:
    n: int
    is_normal: bool
    tests_passed: int
    total_tests: int
    confidence: str
    shapiro_stat: float | None = None
    shapiro_p: float | None = None
    anderson_stat: float | None = None
    anderson_critical: float | None = None
    ks_stat: float | None = None
    ks_p: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    recommendation: str = ""
    detail: dict = field(default_factory=dict)


def _clean(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[~np.isnan(arr)]


def check_normality(values, alpha: float = 0.05) -> NormalityResult:
    """Run AD + SW + KS + skew/kurt and combine into a majority verdict.

    Thresholds follow the MVP design doc: |skew| < 1.0, |excess kurt| < 3.5.
    """
    arr = _clean(values)
    n = int(arr.size)
    if n < 3:
        return NormalityResult(
            n=n, is_normal=False, tests_passed=0, total_tests=0,
            confidence="None", recommendation="Insufficient data for normality test (n < 3).",
        )

    passed = 0
    total = 0
    detail: dict = {}

    # Anderson-Darling (index 2 == 5% significance level).
    # Prefer method='interpolate' on SciPy >=1.17 to avoid FutureWarning; fall back.
    ad_stat = ad_crit = None
    try:
        try:
            ad = stats.anderson(arr, dist="norm", method="interpolate")
            ad_stat = float(ad.statistic)
            # With method=, result exposes pvalue instead of critical_values.
            ad_p = float(getattr(ad, "pvalue", 0.0) or 0.0)
            ad_pass = ad_p > alpha if hasattr(ad, "pvalue") else False
            ad_crit = None
            detail["anderson_darling"] = {
                "stat": ad_stat, "p": ad_p, "critical": ad_crit, "passes": ad_pass
            }
        except TypeError:
            ad = stats.anderson(arr, dist="norm")
            idx = 2 if len(ad.critical_values) > 2 else -1
            ad_stat = float(ad.statistic)
            ad_crit = float(ad.critical_values[idx])
            ad_pass = ad_stat < ad_crit
            detail["anderson_darling"] = {
                "stat": ad_stat, "critical": ad_crit, "passes": ad_pass
            }
        passed += int(ad_pass)
        total += 1
    except Exception as exc:  # pragma: no cover - defensive
        detail["anderson_darling"] = {"error": str(exc)}

    # Shapiro-Wilk (valid for 3 <= n <= 5000).
    sw_stat = sw_p = None
    if 3 <= n <= 5000:
        try:
            sw_stat, sw_p = (float(x) for x in stats.shapiro(arr))
            sw_pass = sw_p > alpha
            detail["shapiro_wilk"] = {"stat": sw_stat, "p": sw_p, "passes": sw_pass}
            passed += int(sw_pass)
            total += 1
        except Exception as exc:  # pragma: no cover - defensive
            detail["shapiro_wilk"] = {"error": str(exc)}

    # Kolmogorov-Smirnov with Lilliefors correction for estimated parameters.
    # Plain KS against a fitted normal is anticonservative; use scipy's
    # lilliefors when available, otherwise skip KS from the majority vote.
    ks_stat = ks_p = None
    try:
        mean, std = float(arr.mean()), float(arr.std(ddof=1))
        if std > 0:
            try:
                from statsmodels.stats.diagnostic import lilliefors as _lilliefors
                ks_stat, ks_p = (float(x) for x in _lilliefors(arr, dist="norm"))
                ks_pass = ks_p > alpha
                detail["kolmogorov_smirnov"] = {
                    "stat": ks_stat, "p": ks_p, "passes": ks_pass, "method": "lilliefors",
                }
                passed += int(ks_pass)
                total += 1
            except ImportError:
                # Without statsmodels: report raw KS for diagnostics but do NOT
                # count it toward the majority vote (parameters estimated from data).
                ks_stat, ks_p = (float(x) for x in stats.kstest(arr, "norm", args=(mean, std)))
                detail["kolmogorov_smirnov"] = {
                    "stat": ks_stat, "p": ks_p, "passes": None,
                    "method": "raw_kstest_not_voted",
                    "note": "KS skipped from vote; install statsmodels for Lilliefors.",
                }
    except Exception as exc:  # pragma: no cover - defensive
        detail["kolmogorov_smirnov"] = {"error": str(exc)}

    # Shape: skewness and excess kurtosis.
    skew = float(stats.skew(arr))
    kurt = float(stats.kurtosis(arr))  # excess kurtosis (0 == normal)
    skew_ok = abs(skew) < 1.0
    kurt_ok = abs(kurt) < 3.5
    detail["shape"] = {"skewness": skew, "kurtosis": kurt,
                       "skew_ok": skew_ok, "kurt_ok": kurt_ok}
    passed += int(skew_ok) + int(kurt_ok)
    total += 2

    is_normal = total > 0 and passed >= (total / 2)
    confidence = "High" if passed >= total - 1 else "Medium" if passed >= total / 2 else "Low"

    if is_normal:
        rec = "Data appears normally distributed. Standard Cp/Cpk and all zone tests are valid."
    elif abs(skew) > 1.0:
        rec = ("Data is skewed. Try Box-Cox/log transformation; if that fails, use the "
               "Wheeler individuals path (points-outside-limits only) with percentile Cpk.")
    else:
        rec = "Data is mildly non-normal. Interpret capability with caution or transform."

    return NormalityResult(
        n=n, is_normal=is_normal, tests_passed=passed, total_tests=total,
        confidence=confidence, shapiro_stat=sw_stat, shapiro_p=sw_p,
        anderson_stat=ad_stat, anderson_critical=ad_crit, ks_stat=ks_stat, ks_p=ks_p,
        skewness=skew, kurtosis=kurt, recommendation=rec, detail=detail,
    )


def autocorrelation(values, max_lag: int = 1) -> list[float]:
    """Sample autocorrelation function (ACF) up to ``max_lag``, computed in numpy."""
    arr = _clean(values)
    n = arr.size
    if n < 2:
        return [1.0] + [0.0] * max_lag
    arr = arr - arr.mean()
    denom = float(np.sum(arr ** 2))
    if denom == 0:
        return [1.0] + [0.0] * max_lag
    acf = []
    for lag in range(0, max_lag + 1):
        num = float(np.sum(arr[: n - lag] * arr[lag:]))
        acf.append(num / denom)
    return acf


@dataclass
class AutocorrelationResult:
    lag1: float
    threshold: float
    is_autocorrelated: bool
    recommendation: str


def check_autocorrelation(values, threshold: float = 0.2) -> AutocorrelationResult:
    """Gate before Shewhart charting. |lag-1 ACF| > threshold => use EWMA/CUSUM."""
    acf = autocorrelation(values, max_lag=1)
    lag1 = acf[1] if len(acf) > 1 else 0.0
    is_ac = abs(lag1) > threshold
    rec = (
        "Autocorrelation detected. Standard Shewhart limits are invalid; "
        "route to EWMA (lambda=0.2) or CUSUM (k=0.5, h=5)."
        if is_ac
        else "No significant autocorrelation. Shewhart charts are appropriate."
    )
    return AutocorrelationResult(lag1=lag1, threshold=threshold, is_autocorrelated=is_ac, recommendation=rec)


@dataclass
class TransformResult:
    applied: str            # "NONE" | "LOG" | "BOXCOX" | "YEO-JOHNSON"
    label: str              # human-readable, includes lambda where relevant
    values: np.ndarray
    lam: float | None = None
    became_normal: bool | None = None


def apply_transform(values, method: str = "auto", alpha: float = 0.05) -> TransformResult:
    """Attempt a normalizing transform.

    method: "auto" (choose based on positivity), "log", "boxcox", "yeo-johnson", "none".
    """
    arr = _clean(values)
    method = method.lower()

    if method == "none":
        return TransformResult(applied="NONE", label="No transform", values=arr, became_normal=None)

    all_positive = bool(np.all(arr > 0))

    if method == "auto":
        method = "boxcox" if all_positive else "yeo-johnson"

    if method == "log":
        if not all_positive:
            # log1p keeps zeros usable; negatives are undefined.
            if np.any(arr < 0):
                return TransformResult(applied="NONE", label="Log requires non-negative data",
                                       values=arr, became_normal=None)
            out = np.log1p(arr)
            label = "log(x+1)"
        else:
            out = np.log(arr)
            label = "log(x)"
        lam = None
    elif method == "boxcox":
        if not all_positive:
            method = "yeo-johnson"
        else:
            out, lam = stats.boxcox(arr)
            out = np.asarray(out, dtype=float)
            label = f"Box-Cox(lambda={lam:.3f})"
    if method == "yeo-johnson":
        out, lam = stats.yeojohnson(arr)
        out = np.asarray(out, dtype=float)
        label = f"Yeo-Johnson(lambda={lam:.3f})"

    applied = {"log": "LOG", "boxcox": "BOXCOX", "yeo-johnson": "YEO-JOHNSON"}[method]
    became_normal = check_normality(out, alpha=alpha).is_normal
    return TransformResult(applied=applied, label=label, values=out, lam=lam, became_normal=became_normal)
