"""Transform path must not pass misaligned subgroup metadata into charting."""
from __future__ import annotations

import numpy as np

from spc_core.pipeline import establish


def test_transform_clears_mismatched_subgroup_ids(monkeypatch):
    values = np.concatenate([
        np.random.lognormal(mean=0.0, sigma=1.0, size=40),
        [np.nan, np.nan],
    ])
    subgroup_ids = list(range(len(values)))

    class _FakeTransform:
        became_normal = True
        values = np.linspace(0, 1, 30)  # shorter than subgroup_ids
        label = "boxcox"
        applied = "boxcox"
        lam = 0.0

    monkeypatch.setattr("spc_core.pipeline.check_multimodal", lambda *_a, **_k: type(
        "M", (), {"is_multimodal": False, "recommendation": "ok", "dip_statistic": 0, "p_value": 1}
    )())
    monkeypatch.setattr(
        "spc_core.pipeline.check_normality",
        lambda *_a, **_k: type(
            "N", (), {"is_normal": False, "recommendation": "non-normal"}
        )(),
    )
    monkeypatch.setattr("spc_core.pipeline.apply_transform", lambda *_a, **_k: _FakeTransform())

    # Should not raise due to length mismatch.
    result = establish(values, subgroup_ids=subgroup_ids, chart_type=None)
    assert result.chart is not None
