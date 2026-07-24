"""SPC-specific data cleaning.

SPC cleaning is the inverse of ML cleaning: preserve the process as it actually ran.
Never mean/median impute control-chart data; never silently drop. Every missing value
is classified and flagged. This module implements the missing-value decision tree from
the MVP design document (section 6).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .models import QualityFlag

# Explicit root-cause reason -> flag mapping (caller supplies reasons when known).
_REASON_TO_FLAG = {
    "sensor": QualityFlag.MISSING_SENSOR,
    "maintenance": QualityFlag.EXCLUDED_MAINTENANCE,
    "incomplete": QualityFlag.EXCLUDED_INCOMPLETE,
    "backup": QualityFlag.RESTORED_FROM_BACKUP,
    "human": QualityFlag.MISSING_HUMAN,
    "comms": None,  # decided by gap length below
}


@dataclass
class CleaningResult:
    values: list[Optional[float]]        # cleaned values (LOCF may fill a few points)
    flags: list[QualityFlag]
    usable: list[bool]                   # True if the point may enter control-chart math
    notes: dict = field(default_factory=dict)


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def range_check(values, low: float, high: float) -> list[bool]:
    """Flag measurement-system failures (out of physical range) vs process variation.

    Returns a boolean "is_valid_measurement" per point. A disconnected-sensor sentinel
    (e.g. -999) fails the range check and is a measurement failure, NOT an OOC signal.
    """
    out = []
    for v in values:
        if _is_missing(v):
            out.append(False)
        else:
            out.append(low <= v <= high)
    return out


def classify_missing(values, reasons: Optional[list[Optional[str]]] = None,
                     locf_max: int = 3) -> CleaningResult:
    """Classify and handle missing values without silent imputation.

    Parameters
    ----------
    values : sequence with possible None/NaN entries.
    reasons : optional per-index root-cause hint (see ``_REASON_TO_FLAG``). When absent,
        a short consecutive gap (<= ``locf_max``) is forward-filled and flagged
        IMPUTED_LOCF; a longer gap is held as MISSING_SENSOR for investigation.
    locf_max : maximum consecutive gap that may be forward-filled.
    """
    values = list(values)
    reasons = reasons or [None] * len(values)
    flags: list[QualityFlag] = []
    cleaned: list[Optional[float]] = []
    usable: list[bool] = []
    counts: dict[str, int] = {}

    # Pre-compute consecutive missing run lengths.
    run_len = [0] * len(values)
    i = 0
    while i < len(values):
        if _is_missing(values[i]):
            j = i
            while j < len(values) and _is_missing(values[j]):
                j += 1
            for m in range(i, j):
                run_len[m] = j - i
            i = j
        else:
            i += 1

    last_valid: Optional[float] = None
    for idx, v in enumerate(values):
        reason = (reasons[idx] or "").lower() if reasons[idx] else None

        if not _is_missing(v):
            last_valid = float(v)
            cleaned.append(float(v))
            flag = _REASON_TO_FLAG.get(reason) if reason == "backup" else QualityFlag.ORIGINAL
            flag = flag or QualityFlag.ORIGINAL
            flags.append(flag)
            usable.append(True)
            counts[flag.value] = counts.get(flag.value, 0) + 1
            continue

        # Missing value: decide by explicit reason, else by gap length.
        if reason and reason in _REASON_TO_FLAG and _REASON_TO_FLAG[reason] is not None:
            flag = _REASON_TO_FLAG[reason]
        elif reason == "comms":
            flag = QualityFlag.IMPUTED_LOCF if run_len[idx] <= locf_max else QualityFlag.MISSING_SENSOR
        else:
            flag = QualityFlag.IMPUTED_LOCF if run_len[idx] <= locf_max else QualityFlag.MISSING_SENSOR

        if flag == QualityFlag.IMPUTED_LOCF and last_valid is not None:
            cleaned.append(last_valid)
            usable.append(True)
        else:
            cleaned.append(None)
            usable.append(False)
        flags.append(flag)
        counts[flag.value] = counts.get(flag.value, 0) + 1

    return CleaningResult(values=cleaned, flags=flags, usable=usable,
                          notes={"flag_counts": counts, "locf_max": locf_max})
