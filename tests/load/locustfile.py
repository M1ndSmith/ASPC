"""Minimal Locust load smoke for ASPC API health + auth.

Usage (with API running):
  locust -f tests/load/locustfile.py --headless -u 10 -r 2 -t 30s --host http://localhost:8000
"""
from __future__ import annotations

try:
    from locust import HttpUser, between, task
except ImportError:  # pragma: no cover
    HttpUser = object  # type: ignore
    between = lambda *a, **k: None  # type: ignore
    task = lambda f: f  # type: ignore


class AspcUser(HttpUser):
    wait_time = between(0.5, 1.5) if callable(between) else None

    @task(3)
    def health(self):
        self.client.get("/health")

    @task(1)
    def token(self):
        self.client.post("/auth/token", data={"username": "load", "password": "admin"})
