"""Seeded series builders for combinatorial cases."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

Columns = dict[str, list[Any]]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def build_series(kind: str, *, seed: int = 0) -> Columns:
    """Return column dict for ``kind`` (resilience-style empty = None elsewhere)."""
    if kind == "imr_in_control":
        try:
            from resilience_data.generators.spc import imr_in_control

            return imr_in_control(n=40, seed=100 + seed)
        except ImportError:
            rng = _rng(100 + seed)
            return {"measurement": [float(v) for v in 100.0 + rng.normal(0, 1, 40)]}

    if kind == "imr_mean_shift":
        try:
            from resilience_data.generators.spc import imr_mean_shift

            return imr_mean_shift(n=50, seed=110 + seed)
        except ImportError:
            rng = _rng(110 + seed)
            vals = 100.0 + rng.normal(0, 1, 50)
            vals[25:] += 4.0
            return {"measurement": [float(v) for v in vals]}

    if kind == "imr_trend":
        try:
            from resilience_data.generators.spc import imr_trend

            return imr_trend(n=50, seed=113 + seed)
        except ImportError:
            rng = _rng(113 + seed)
            vals = 100.0 + rng.normal(0, 0.5, 50) + np.linspace(0, 8, 50)
            return {"measurement": [float(v) for v in vals]}

    if kind == "imr_alternating":
        try:
            from resilience_data.generators.spc import imr_alternating_nelson4

            return imr_alternating_nelson4(n=40, seed=142 + seed)
        except ImportError:
            swing = 2.0
            return {"measurement": [100.0 + ((-1) ** i) * swing for i in range(40)]}

    if kind == "imr_constant":
        return {"measurement": [100.0] * 30}

    if kind == "imr_empty":
        return {"measurement": []}

    if kind == "imr_n1":
        return {"measurement": [100.0]}

    if kind == "imr_n2":
        return {"measurement": [100.0, 101.0]}

    if kind == "imr_nan_inf":
        return {"measurement": [100.0, float("nan"), float("inf"), 101.0] + [100.0] * 20}

    if kind == "imr_sentinel":
        vals = [100.0 + i * 0.01 for i in range(30)]
        vals[5] = 999.0
        vals[6] = -999.0
        return {"measurement": vals}

    if kind == "imr_overflow":
        return {"measurement": [1e308, -1e308] + [100.0] * 28}

    if kind == "xbar_r":
        try:
            from resilience_data.generators.spc import xbar_r

            return xbar_r(n_subgroups=25, size=5, seed=120 + seed)
        except ImportError:
            rng = _rng(120 + seed)
            m, s = [], []
            for sid in range(1, 26):
                for v in 50.0 + rng.normal(0, 1.2, 5):
                    m.append(float(v))
                    s.append(sid)
            return {"measurement": m, "subgroup": s}

    if kind == "attribute_p":
        try:
            from resilience_data.generators.spc import attribute_p

            return attribute_p(n=25, seed=130 + seed)
        except ImportError:
            return {"defective": [2] * 25, "inspected": [100] * 25}

    if kind == "attribute_p_zero_n":
        return {"defective": [0, 1], "inspected": [0, 100]}

    if kind == "attribute_np":
        try:
            from resilience_data.generators.spc import attribute_np

            return attribute_np(n=25, seed=131 + seed)
        except ImportError:
            return {"defectives": [3] * 25, "sample_size": [100] * 25}

    if kind == "attribute_c":
        try:
            from resilience_data.generators.spc import attribute_c

            return attribute_c(n=25, seed=132 + seed)
        except ImportError:
            return {"defects": [2] * 25}

    if kind == "attribute_u":
        try:
            from resilience_data.generators.spc import attribute_u

            return attribute_u(n=25, seed=133 + seed)
        except ImportError:
            return {"defects": [3] * 25, "units": [10] * 25}

    if kind == "attribute_u_zero_opp":
        return {"defects": [1, 2], "units": [0, 10]}

    if kind == "multimodal":
        try:
            from resilience_data.generators.spc import multimodal_stop

            return multimodal_stop(seed=140 + seed)
        except ImportError:
            rng = _rng(140 + seed)
            a = rng.normal(90, 0.5, 40)
            b = rng.normal(110, 0.5, 40)
            return {"measurement": [float(v) for v in np.concatenate([a, b])]}

    if kind == "heavy_tail":
        try:
            from resilience_data.generators.spc import heavy_tail_wheeler

            return heavy_tail_wheeler(seed=150 + seed)
        except ImportError:
            rng = _rng(150 + seed)
            return {"measurement": [float(v) for v in 100.0 + rng.standard_t(2.5, 50)]}

    if kind == "type_mismatch":
        return {"measurement": ["not-a-number", "also-bad", 1, 2, 3]}

    raise KeyError(f"Unknown series kind: {kind}")


def values_from_columns(cols: Columns, kind: str) -> tuple[list[Any], dict[str, Any]]:
    """Extract establish/analyze kwargs from columns."""
    extra: dict[str, Any] = {}
    if "subgroup" in cols:
        return list(cols["measurement"]), {"subgroup_ids": list(cols["subgroup"])}
    if kind.startswith("attribute_p") or ("defective" in cols and "inspected" in cols):
        return list(cols["defective"]), {"sample_sizes": list(cols["inspected"])}
    if "defectives" in cols and "sample_size" in cols:
        return list(cols["defectives"]), {"sample_sizes": list(cols["sample_size"])}
    if kind.startswith("attribute_u") or ("units" in cols and "defects" in cols and "inspected" not in cols):
        if "units" in cols:
            return list(cols["defects"]), {"opportunities": list(cols["units"])}
    if "defects" in cols and "measurement" not in cols:
        return list(cols["defects"]), {}
    if "measurement" in cols:
        return list(cols["measurement"]), extra
    # fallback first column
    key = next(iter(cols))
    return list(cols[key]), extra


def sanitize_for_json(cols: Columns) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for k, vals in cols.items():
        row = []
        for v in vals:
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                row.append(str(v))
            else:
                row.append(v)
        out[k] = row
    return out
