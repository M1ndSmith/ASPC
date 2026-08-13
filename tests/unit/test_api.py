"""API smoke tests (httpx / TestClient). Auth disabled via env for unit speed."""
from __future__ import annotations

import os

import pytest

# Disable auth before app import side-effects in dependent fixtures.
os.environ.setdefault("ASPC_AUTH_ENABLED", "false")
os.environ.setdefault("ASPC_API_KEYS", "")
os.environ.setdefault("ASPC_DEV_INSECURE", "1")
os.environ.setdefault("ASPC_JWT_SECRET", "unit-test-secret")

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db = tmp_path_factory.mktemp("api") / "test.db"
    os.environ["ASPC_PERSISTENCE_BACKEND"] = "sqlite"
    os.environ["ASPC_SQLITE_PATH"] = str(db)
    # Re-import config/repo cleanly
    import apps.api.main as api_main
    import apps.config as config_mod

    config_mod._config = None
    api_main.cfg = config_mod.get_config()
    from adapters.factory import get_repository

    api_main.repo = get_repository(api_main.cfg)
    return TestClient(api_main.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_analyze_control_chart(client, write_dataset):
    path = write_dataset("spc_individual_in_control", "spc.csv")
    with path.open("rb") as f:
        r = client.post(
            "/analyze/control-chart",
            files={"file": ("spc.csv", f, "text/csv")},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "run_id" in body
    report = body["report"]
    assert "limits" in report
    assert "gates" in report or "summary" in report


def test_auth_token_when_enabled(client):
    # Token endpoint should still respond
    r = client.post("/auth/token", data={"username": "op", "password": "admin"})
    # May be 200 with token or 401 depending on config; must not 500
    assert r.status_code in (200, 401, 403)
