"""Ingest / column auto-detection (deduped from 3 legacy pipelines)."""
from __future__ import annotations

import pytest

from adapters.io_files import FileReadError, read_csv, safe_filename
from spc_core.ingest import detect_columns, ingest


def test_detect_measurement_prefers_named_column():
    cols = {"subgroup_id": [1, 1, 2], "measurement": [10.0, 10.1, 10.2], "batch": [1, 1, 2]}
    cmap = detect_columns(cols)
    assert cmap.value_col == "measurement"
    assert cmap.subgroup_col in ("subgroup_id", "batch")


def test_ingest_sample(dataset):
    cols = dataset("spc_subgroup_data")
    frame = ingest(cols)
    assert frame.column_map.value_col == "measurement"
    assert frame.column_map.subgroup_col == "subgroup"
    assert frame.n_rows > 0


def test_msa_column_detect(dataset):
    cols = dataset("msa_gage_rr_excellent")
    frame = ingest(cols)
    assert frame.column_map.part_col == "Part"
    assert frame.column_map.operator_col == "Operator"
    assert frame.column_map.value_col == "Measurement"


def test_safe_filename_rejects_traversal():
    with pytest.raises(FileReadError):
        safe_filename("../etc/passwd")
    with pytest.raises(FileReadError):
        safe_filename("foo/bar.csv")
    assert safe_filename("data.csv") == "data.csv"


def test_empty_csv_raises(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("")
    with pytest.raises(FileReadError, match="empty"):
        read_csv(p)


def test_missing_file_raises():
    with pytest.raises(FileReadError, match="not found"):
        read_csv("/tmp/definitely_does_not_exist_aspc_xyz.csv")
