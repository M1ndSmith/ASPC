"""Shared fixtures — deterministic synthetic datasets (no committed CSVs)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sample_data import get_dataset, write_csv


@pytest.fixture
def dataset():
    """Factory: ``dataset("spc_individual_in_control")`` → column dict."""

    def _get(name: str) -> dict[str, list[Any]]:
        return get_dataset(name)

    return _get


@pytest.fixture
def write_dataset(tmp_path: Path):
    """Factory: write a named dataset to tmp_path and return the CSV Path."""

    def _write(name: str, filename: str | None = None) -> Path:
        cols = get_dataset(name)
        path = tmp_path / (filename or f"{name}.csv")
        return write_csv(cols, path)

    return _write
