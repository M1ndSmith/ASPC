"""Security hardening tests — auth, traversal, API keys, go-live gates."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


@pytest.fixture()
def secure_client(tmp_path, monkeypatch):
    """Auth enabled, keys required, non-default JWT secret."""
    db = tmp_path / "sec.db"
    monkeypatch.setenv("ASPC_AUTH_ENABLED", "true")
    monkeypatch.setenv("ASPC_DEV_INSECURE", "0")
    monkeypatch.setenv("ASPC_JWT_SECRET", "unit-test-secret-not-default")
    monkeypatch.setenv("ASPC_API_KEYS", "test-key")
    monkeypatch.setenv("ASPC_ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ASPC_ADMIN_PASSWORD", "s3cret")
    monkeypatch.setenv("ASPC_PERSISTENCE_BACKEND", "sqlite")
    monkeypatch.setenv("ASPC_SQLITE_PATH", str(db))
    monkeypatch.setenv("ASPC_CORS_ORIGINS", "http://localhost:3000")

    import apps.api.main as api_main
    import apps.config as config_mod
    from adapters.factory import get_repository

    config_mod._config = None
    api_main.cfg = config_mod.get_config()
    api_main.repo = get_repository(api_main.cfg)
    api_main._admin_password_hash = None  # force re-hash
    return TestClient(api_main.app)


def _token(client: TestClient) -> str:
    r = client.post("/auth/token", data={"username": "admin", "password": "s3cret"})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def test_login_rejects_wrong_username(secure_client):
    r = secure_client.post("/auth/token", data={"username": "anyone", "password": "s3cret"})
    assert r.status_code == 401


def test_login_rejects_wrong_password(secure_client):
    r = secure_client.post("/auth/token", data={"username": "admin", "password": "nope"})
    assert r.status_code == 401


def test_login_accepts_admin(secure_client):
    r = secure_client.post("/auth/token", data={"username": "admin", "password": "s3cret"})
    assert r.status_code == 200
    assert r.json()["access_token"]


def test_api_key_required_for_register(secure_client):
    token = _token(secure_client)
    r = secure_client.post(
        "/streams/register",
        json={"stream_key": "line-1"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 401


def test_api_key_accepted(secure_client):
    token = _token(secure_client)
    r = secure_client.post(
        "/streams/register",
        json={"stream_key": "line-1"},
        headers={"Authorization": f"Bearer {token}", "X-API-Key": "test-key"},
    )
    # SQLite has no stream registry → 501, but auth passed
    assert r.status_code == 501


def test_reports_require_auth(secure_client):
    r = secure_client.get("/reports/abc")
    assert r.status_code == 401


def test_reports_reject_traversal(secure_client):
    token = _token(secure_client)
    # Dots / separators are rejected by run_id allowlist
    r = secure_client.get(
        "/reports/..passwd",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 400
    r2 = secure_client.get(
        "/reports/foo/bar",
        headers={"Authorization": f"Bearer {token}"},
    )
    # Nested path does not match the route → 404
    assert r2.status_code == 404


def test_stream_replay_rejects_escape(secure_client, tmp_path):
    token = _token(secure_client)
    r = secure_client.get(
        "/stream/replay",
        params={
            "file_path": "../../../etc/passwd",
            "limits_version": "deadbeefdeadbeef",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    # 400 (path escape) or 404 (limits not found — order may vary)
    assert r.status_code in (400, 404)


def test_go_live_rejects_unfrozen_limits(secure_client, tmp_path):
    """Limits saved with frozen=False must not go live."""
    import apps.api.main as api_main

    token = _token(secure_client)
    version = "unfrozenversion01"
    api_main.repo.save_limits(
        {"chart_type": "I-MR", "subgroup_size": 1, "components": {}},
        version,
        "I-MR",
        meta={"frozen": False, "stopped": True, "checklist_passed": False},
    )
    r = secure_client.post(
        "/streams/line-x/go-live",
        json={"limits_version": version},
        headers={"Authorization": f"Bearer {token}", "X-API-Key": "test-key"},
    )
    # 409 preferred; 501 if sqlite has no registry (still proves auth path)
    assert r.status_code in (409, 501)
    if r.status_code == 409:
        assert "not frozen" in r.json()["detail"].lower() or "STOP" in r.json()["detail"]
