#!/usr/bin/env python3
"""Manufacturing-like MQTT publisher for ASPC Phase II live testing.

Publishes JSON observations to ``sensors/{stream_key}`` for the mqtt-bridge:

    {"key": "line-1", "value": 100.2, "timestamp": "...", "machine_id": "CNC-01"}

Phases (default):
  warmup     — stable process mean~100 σ~1
  production — continued in-control
  shift      — mean +4 (process shift)
  spike      — one extreme outlier

Example:
  python scripts/manufacturing_sim.py --host localhost --stream-key line-1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from typing import Iterator

import numpy as np


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_phases(
    *,
    seed: int = 42,
    mean: float = 100.0,
    sigma: float = 1.0,
    warmup: int = 20,
    production: int = 40,
    shift: int = 25,
    shift_delta: float = 4.0,
    spike: bool = True,
) -> Iterator[tuple[str, float]]:
    """Yield (phase_name, value) for a manufacturing-style sequence."""
    rng = np.random.default_rng(seed)
    for _ in range(warmup):
        yield "warmup", float(mean + rng.normal(0.0, sigma))
    for _ in range(production):
        yield "production", float(mean + rng.normal(0.0, sigma))
    for _ in range(shift):
        yield "shift", float(mean + shift_delta + rng.normal(0.0, sigma))
    if spike:
        yield "spike", float(mean + 8.0 * sigma)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ASPC manufacturing MQTT stream simulator")
    parser.add_argument("--host", default="localhost", help="MQTT broker host")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--stream-key", default="line-1")
    parser.add_argument("--topic-prefix", default="sensors")
    parser.add_argument("--rate", type=float, default=2.0, help="Points per second")
    parser.add_argument("--mean", type=float, default=100.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--production", type=int, default=40)
    parser.add_argument("--shift", type=int, default=25)
    parser.add_argument("--shift-delta", type=float, default=4.0)
    parser.add_argument("--no-spike", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--machine-id", default="CNC-01")
    parser.add_argument("--dry-run", action="store_true", help="Print only, do not publish")
    args = parser.parse_args(argv)

    topic = f"{args.topic_prefix.rstrip('/')}/{args.stream_key}"
    delay = 1.0 / args.rate if args.rate > 0 else 0.0

    client = None
    if not args.dry_run:
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            print(
                "paho-mqtt required. Install with: uv pip install paho-mqtt",
                file=sys.stderr,
            )
            return 1
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except AttributeError:
            client = mqtt.Client()
        client.connect(args.host, args.port, keepalive=60)
        client.loop_start()

    counts: dict[str, int] = {}
    total = 0
    print(
        f"Publishing to mqtt://{args.host}:{args.port}/{topic} "
        f"at {args.rate} Hz (stream_key={args.stream_key})",
        flush=True,
    )

    try:
        for phase, value in generate_phases(
            seed=args.seed,
            mean=args.mean,
            sigma=args.sigma,
            warmup=args.warmup,
            production=args.production,
            shift=args.shift,
            shift_delta=args.shift_delta,
            spike=not args.no_spike,
        ):
            payload = {
                "key": args.stream_key,
                "value": value,
                "timestamp": _now_iso(),
                "machine_id": args.machine_id,
                "phase": phase,
            }
            body = json.dumps(payload)
            if args.dry_run:
                print(f"[{phase}] {body}")
            else:
                assert client is not None
                client.publish(topic, body, qos=0)
                print(f"[{phase}] value={value:.3f}", flush=True)
            counts[phase] = counts.get(phase, 0) + 1
            total += 1
            if delay:
                time.sleep(delay)
    finally:
        if client is not None:
            client.loop_stop()
            client.disconnect()

    print(f"Done. published={total} by_phase={counts}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
