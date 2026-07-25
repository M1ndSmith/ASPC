"""CLI entry: MQTT → Kafka bridge for ASPC live measurements."""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger("aspc.mqtt_bridge")


def _produce_kafka(bootstrap: str, topic: str, messages):
    """Lazy-import aiokafka producer wrapped for sync use, or kafka-python."""
    try:
        from kafka import KafkaProducer  # type: ignore[import-untyped]

        producer = KafkaProducer(
            bootstrap_servers=bootstrap.split(","),
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            key_serializer=lambda v: v.encode("utf-8") if v else None,
        )

        def send(key: str, payload: dict[str, Any]) -> None:
            producer.send(topic, key=key, value=payload)
            producer.flush()

        def close() -> None:
            producer.close()

        return send, close
    except ImportError:
        pass

    try:
        import asyncio

        from aiokafka import AIOKafkaProducer
    except ImportError as exc:
        raise ImportError(
            "mqtt_bridge requires kafka-python or aiokafka. "
            "Install with: pip install 'aspc[stream]'  (or pip install kafka-python)"
        ) from exc

    loop = asyncio.new_event_loop()
    producer = AIOKafkaProducer(bootstrap_servers=bootstrap)
    loop.run_until_complete(producer.start())

    def send(key: str, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, default=str).encode("utf-8")
        loop.run_until_complete(
            producer.send_and_wait(topic, value=data, key=key.encode("utf-8"))
        )

    def close() -> None:
        loop.run_until_complete(producer.stop())
        loop.close()

    return send, close


def _normalise(msg: dict[str, Any]) -> dict[str, Any]:
    key = str(msg.get("key") or msg.get("stream_key") or "default")
    value = msg.get("value")
    if value is None:
        raise ValueError(f"MQTT payload missing value: {msg!r}")
    ts = msg.get("timestamp") or msg.get("ts")
    if isinstance(ts, datetime):
        ts = ts.isoformat()
    elif ts is None:

        ts = datetime.now(UTC).isoformat()
    return {"key": key, "value": float(value), "timestamp": ts}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aspc-mqtt-bridge",
        description="Bridge MQTT sensor topics into the ASPC Kafka measurement topic",
    )
    parser.add_argument("--mqtt-host", default=os.getenv("ASPC_MQTT_HOST", "localhost"))
    parser.add_argument("--mqtt-port", type=int, default=int(os.getenv("ASPC_MQTT_PORT", "1883")))
    parser.add_argument("--mqtt-topic", default=os.getenv("ASPC_MQTT_TOPIC", "sensors/#"))
    parser.add_argument("--mqtt-user", default=os.getenv("ASPC_MQTT_USER"))
    parser.add_argument("--mqtt-password", default=os.getenv("ASPC_MQTT_PASSWORD"))
    parser.add_argument(
        "--bootstrap",
        default=os.getenv("ASPC_KAFKA_BOOTSTRAP", "localhost:9092"),
    )
    parser.add_argument(
        "--kafka-topic",
        default=os.getenv("ASPC_KAFKA_TOPIC", "spc.measurements"),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from adapters.stream_sources import MQTTSource

    stop = False

    def _stop(*_a) -> None:
        nonlocal stop
        stop = True
        logger.info("Shutdown requested")

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        send, close = _produce_kafka(args.bootstrap, args.kafka_topic, None)
    except ImportError as exc:
        logger.error("%s", exc)
        return 1

    source = MQTTSource(
        args.mqtt_host,
        args.mqtt_topic,
        port=args.mqtt_port,
        username=args.mqtt_user,
        password=args.mqtt_password,
    )

    logger.info(
        "Bridging MQTT %s:%s/%s → Kafka %s/%s",
        args.mqtt_host,
        args.mqtt_port,
        args.mqtt_topic,
        args.bootstrap,
        args.kafka_topic,
    )
    try:
        for msg in source.iter_sync():
            if stop:
                break
            try:
                payload = _normalise(msg)
                send(payload["key"], payload)
                logger.debug("Forwarded %s", payload)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to forward MQTT message %s", msg)
    except ImportError as exc:
        logger.error("%s", exc)
        return 1
    finally:
        close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
