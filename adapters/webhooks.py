"""Outbound signed webhooks for OOC events."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


def _sign(body: bytes, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def deliver_webhook(
    url: str,
    payload: dict[str, Any],
    *,
    secret: str | None = None,
    timeout_s: float = 5.0,
) -> bool:
    """POST JSON payload to ``url``. Returns True on 2xx.

    When ``secret`` is set, sends header ``X-ASPC-Signature: sha256=<hex>``.
    Failures are logged; never raises to callers (stream eval must continue).
    """
    if not url:
        return False
    body = json.dumps(payload, default=str).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "ASPC-Webhook/1.0",
    }
    secret = secret or os.getenv("ASPC_WEBHOOK_SECRET") or ""
    if secret:
        headers["X-ASPC-Signature"] = f"sha256={_sign(body, secret)}"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 — operator-configured URL
            ok = 200 <= getattr(resp, "status", 200) < 300
            if not ok:
                logger.warning("Webhook %s returned status %s", url, getattr(resp, "status", "?"))
            return ok
    except urllib.error.HTTPError as exc:
        logger.warning("Webhook HTTP error %s for %s: %s", exc.code, url, exc.reason)
        return False
    except Exception:  # noqa: BLE001
        logger.exception("Webhook delivery failed for %s", url)
        return False


def resolve_webhook_url(
    stream_meta: dict[str, Any] | None,
    *,
    global_url: str | None = None,
) -> str | None:
    """Prefer per-stream ``meta.webhook_url``, else global config/env."""
    if stream_meta:
        u = stream_meta.get("webhook_url")
        if isinstance(u, str) and u.strip():
            return u.strip()
    if global_url and str(global_url).strip():
        return str(global_url).strip()
    env = os.getenv("ASPC_WEBHOOK_URL", "").strip()
    return env or None
