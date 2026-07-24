#!/usr/bin/env python3
"""End-to-end manufacturing live protocol against the Compose stack.

Assumes Docker Compose services are up (API :8000, Mosquitto :1883, Redis :6379).
Uses Compose credentials (API key demokey), not the host sqlite .env.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "var" / "reports" / "live_protocol_result.md"
SAMPLES = ROOT / "examples" / "data"
CSV = SAMPLES / "spc_individual_in_control.csv"

API = os.getenv("ASPC_PROTOCOL_API", "http://localhost:8000")
API_KEY = os.getenv("ASPC_PROTOCOL_API_KEY", "demokey")
PASSWORD = os.getenv("ASPC_PROTOCOL_PASSWORD", "admin")
STREAM_KEY = os.getenv("ASPC_PROTOCOL_STREAM", "line-1")
MQTT_HOST = os.getenv("ASPC_PROTOCOL_MQTT_HOST", "localhost")
REDIS_URL = os.getenv("ASPC_PROTOCOL_REDIS_URL", "redis://localhost:6379/0")


def _log(msg: str) -> None:
    print(msg, flush=True)


def wait_health(timeout: float = 180.0) -> None:
    import urllib.request

    deadline = time.time() + timeout
    last_err = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{API}/health", timeout=3) as r:
                body = json.loads(r.read().decode())
                if body.get("status") == "healthy":
                    _log(f"API healthy at {API}")
                    return
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
        time.sleep(2)
    raise RuntimeError(f"API not healthy within {timeout}s: {last_err}")


def get_token() -> str:
    import urllib.parse
    import urllib.request

    data = urllib.parse.urlencode({"username": "operator", "password": PASSWORD}).encode()
    req = urllib.request.Request(
        f"{API}/auth/token",
        data=data,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        body = json.loads(r.read().decode())
    token = body.get("access_token")
    if not token:
        raise RuntimeError(f"No access_token: {body}")
    return token


def multipart_analyze(token: str, path: Path) -> dict:
    """POST /analyze/control-chart with file upload (stdlib multipart)."""
    import uuid
    import urllib.request

    boundary = f"----aspc{uuid.uuid4().hex}"
    file_bytes = path.read_bytes()
    parts: list[bytes] = []
    parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
            f"Content-Type: text/csv\r\n\r\n"
        ).encode()
        + file_bytes
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(
        f"{API}/analyze/control-chart",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def api_json(method: str, path: str, token: str, payload: dict | None = None, api_key: bool = False) -> dict:
    import urllib.request

    data = None if payload is None else json.dumps(payload).encode()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    if api_key:
        headers["X-API-Key"] = API_KEY
    req = urllib.request.Request(f"{API}{path}", data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read().decode()
        return json.loads(raw) if raw else {}


class RedisCollector:
    def __init__(self, stream_key: str):
        self.channel = f"spc:live:{stream_key}"
        self.messages: list[dict] = []
        self.ooc_count = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        try:
            import redis
        except ImportError:
            self.error = "redis package not installed"
            return
        try:
            client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
            pubsub = client.pubsub(ignore_subscribe_messages=True)
            pubsub.subscribe(self.channel)
            _log(f"Subscribed Redis {self.channel}")
            while not self._stop.is_set():
                msg = pubsub.get_message(timeout=1.0)
                if not msg or msg.get("type") != "message":
                    continue
                try:
                    payload = json.loads(msg["data"])
                except json.JSONDecodeError:
                    continue
                self.messages.append(payload)
                if payload.get("ooc") or payload.get("signals"):
                    self.ooc_count += 1
                    _log(
                        f"OOC redis: value={payload.get('value')} "
                        f"signals={payload.get('signals')}"
                    )
            pubsub.unsubscribe(self.channel)
            pubsub.close()
            client.close()
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)


def ensure_samples() -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    if CSV.exists():
        return
    _log("Generating sample_data CSVs…")
    subprocess.check_call(
        [sys.executable, "-m", "sample_data", "--out", str(SAMPLES)],
        cwd=str(ROOT),
    )


def run_simulator() -> None:
    sim = ROOT / "scripts" / "manufacturing_sim.py"
    cmd = [
        sys.executable,
        str(sim),
        "--host",
        MQTT_HOST,
        "--stream-key",
        STREAM_KEY,
        "--rate",
        "2.0",
        "--warmup",
        "15",
        "--production",
        "25",
        "--shift",
        "20",
    ]
    _log("Starting manufacturing simulator…")
    subprocess.check_call(cmd, cwd=str(ROOT))


def write_report(result: dict) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    status = "PASS" if result["passed"] else "FAIL"
    lines = [
        f"# Live manufacturing protocol — {status}",
        "",
        f"- Time (UTC): `{result['finished_at']}`",
        f"- API: `{API}`",
        f"- Stream key: `{STREAM_KEY}`",
        f"- Limits version: `{result.get('limits_version')}`",
        f"- Phase I run_id: `{result.get('run_id')}`",
        f"- Redis messages: `{result.get('redis_messages')}`",
        f"- Redis OOC events: `{result.get('redis_ooc')}`",
        f"- Stream active: `{result.get('stream_active')}`",
        f"- Sample limits in live payload: `{result.get('limits_in_payload')}`",
        "",
        "## Notes",
        "",
        result.get("notes", ""),
        "",
        "## Watch",
        "",
        "1. Open http://localhost:3000/login (password `admin`)",
        "2. Go to **Live** → stream key `line-1` → Connect",
        "3. Re-run simulator: "
        "`python scripts/manufacturing_sim.py --host localhost --stream-key line-1`",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    _log(f"Wrote {REPORT}")


def main() -> int:
    ensure_samples()
    wait_health()
    token = get_token()
    _log("Phase I analyze…")
    analyze = multipart_analyze(token, CSV)
    report = analyze.get("report") or {}
    limits = report.get("limits") or {}
    limits_version = (
        (analyze.get("checklist") or {}).get("limits_version")
        or limits.get("version")
    )
    if not limits_version:
        run = api_json("GET", f"/runs/{analyze['run_id']}", token)
        limits_version = run.get("limits_version")
    if not limits_version:
        raise RuntimeError(
            f"Could not find limits_version in analyze response. "
            f"keys={list(analyze.keys())} checklist={analyze.get('checklist')}"
        )

    _log(f"limits_version={limits_version} run_id={analyze.get('run_id')}")

    _log("Register stream…")
    api_json(
        "POST",
        "/streams/register",
        token,
        {"stream_key": STREAM_KEY, "chart_type": "I-MR", "ruleset": "nelson"},
        api_key=True,
    )
    _log("Go-live…")
    api_json(
        "POST",
        f"/streams/{STREAM_KEY}/go-live",
        token,
        {"limits_version": limits_version, "ruleset": "nelson"},
        api_key=True,
    )

    streams = api_json("GET", "/streams?active_only=true", token)
    active = any(s.get("stream_key") == STREAM_KEY for s in streams.get("streams") or [])
    _log(f"Active streams: {streams.get('streams')}")

    collector = RedisCollector(STREAM_KEY)
    collector.start()
    time.sleep(1.0)

    try:
        run_simulator()
        # Allow bridge/engine lag
        time.sleep(5.0)
    finally:
        collector.stop()

    limits_in_payload = False
    for m in collector.messages:
        if m.get("ucl") is not None and m.get("center") is not None:
            limits_in_payload = True
            break

    passed = (
        active
        and collector.error is None
        and len(collector.messages) >= 10
        and collector.ooc_count >= 1
        and limits_in_payload
    )
    notes = []
    if collector.error:
        notes.append(f"Redis collector error: {collector.error}")
    if len(collector.messages) < 10:
        notes.append(f"Too few Redis messages: {len(collector.messages)}")
    if collector.ooc_count < 1:
        notes.append("No OOC observed on Redis — check mqtt-bridge / stream-engine logs")
    if not limits_in_payload:
        notes.append("Live payload missing ucl/center — rebuild stream-engine image")
    if not active:
        notes.append("Stream not listed as active")

    result = {
        "passed": passed,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "limits_version": limits_version,
        "run_id": analyze.get("run_id"),
        "redis_messages": len(collector.messages),
        "redis_ooc": collector.ooc_count,
        "stream_active": active,
        "limits_in_payload": limits_in_payload,
        "notes": "; ".join(notes) if notes else "All checks passed.",
    }
    write_report(result)
    _log(f"PROTOCOL {'PASS' if passed else 'FAIL'}: {result['notes']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
