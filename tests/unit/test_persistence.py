"""SQLite persistence + audit trail."""
from __future__ import annotations

from adapters.persistence import SQLiteRepository
from spc_core.limits import imr_limits


def test_save_and_get_limits(tmp_path):
    repo = SQLiteRepository(tmp_path / "test.db")
    limits = imr_limits([1.0, 1.1, 0.9, 1.05, 1.0, 0.95, 1.02, 0.98] * 3)
    payload = limits.model_dump(mode="json")
    version = repo.save_limits(payload, limits.version, limits.chart_type.value)
    stored = repo.get_limits(version)
    assert stored is not None
    assert stored["version"] == limits.version
    assert stored["chart_type"] == "I-MR"


def test_save_run_and_audit(tmp_path):
    repo = SQLiteRepository(tmp_path / "test.db")
    run_id = repo.save_run(
        "control_chart",
        {"chart_type": "I-MR", "signals": []},
        limits_version="abc123",
        source_file="foo.csv",
        user_id="tester",
    )
    run = repo.get_run(run_id)
    assert run is not None
    assert run["analysis_type"] == "control_chart"
    assert run["user_id"] == "tester"
    runs = repo.list_runs(analysis_type="control_chart")
    assert any(r["run_id"] == run_id for r in runs)
