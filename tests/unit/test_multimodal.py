"""Multimodality STOP gate — clear bimodality vs normal false positives."""
from __future__ import annotations

import numpy as np
import pytest

from spc_core.multimodal import check_multimodal


def test_clear_bimodal_stops():
    rng = np.random.default_rng(0)
    values = np.concatenate(
        [rng.normal(0.0, 0.4, 50), rng.normal(20.0, 0.4, 50)]
    )
    result = check_multimodal(values)
    assert result.is_multimodal is True


def test_normal_does_not_false_stop():
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 1.0, 200)
    result = check_multimodal(values)
    assert result.is_multimodal is False


def test_small_normal_sample_does_not_false_stop():
    """At n<100 a normal histogram shows several noise peaks with a shallow dip.

    Treating a shallow dip as bimodality STOPped in-control studies (the gate blocks
    go-live), so the separation must be near-empty, not merely lower than the modes.
    """
    for seed in range(25):
        rng = np.random.default_rng(seed)
        for n in (30, 40, 60):
            result = check_multimodal(rng.normal(100.0, 1.0, n))
            assert result.is_multimodal is False, f"false STOP at seed={seed} n={n}"


def test_diptest_backend_is_wired_correctly():
    """``diptest`` is a declared dependency, so the real Hartigan p-value must be used.

    The integration was previously written against a wrong signature, and because the
    call sat under ``except ImportError`` the resulting TypeError was not caught — the
    gate crashed as soon as the package was present.
    """
    pytest.importorskip("diptest")
    rng = np.random.default_rng(1)
    separated = np.concatenate([rng.normal(0.0, 1.0, 60), rng.normal(6.0, 1.0, 60)])
    result = check_multimodal(separated)
    assert result.is_multimodal is True
    # The fallback approximation returns a near-1.0 p even for obvious mixtures; the
    # real test must produce a significant one.
    assert result.p_value < 0.05
