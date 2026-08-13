"""Webhook helper tests."""
from __future__ import annotations

from adapters.webhooks import resolve_webhook_url


def test_resolve_prefers_stream_meta():
    assert (
        resolve_webhook_url({"webhook_url": "https://a.example/hook"}, global_url="https://b.example")
        == "https://a.example/hook"
    )


def test_resolve_falls_back_global():
    assert resolve_webhook_url({}, global_url="https://b.example/hook") == "https://b.example/hook"
