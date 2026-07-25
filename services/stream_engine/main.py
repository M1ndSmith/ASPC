"""CLI entry: consume Kafka measurements and run the Phase II StreamEngine."""
from __future__ import annotations

import argparse
import logging
import signal
import sys

from adapters.factory import get_repository, require_streaming_repository
from adapters.stream_engine import StreamEngine
from adapters.stream_sources import KafkaSource
from apps.config import get_config

logger = logging.getLogger("aspc.stream_engine")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aspc-stream-engine",
        description="ASPC Phase II stream engine — Kafka → evaluate → Timescale/Redis",
    )
    parser.add_argument("--bootstrap", default=None, help="Kafka bootstrap servers")
    parser.add_argument("--topic", default=None, help="Kafka topic")
    parser.add_argument("--group-id", default="aspc-stream-engine")
    parser.add_argument("--redis-url", default=None)
    parser.add_argument(
        "--preload",
        action="append",
        default=[],
        metavar="STREAM_KEY:LIMITS_VERSION",
        help="Pre-register stream_key with frozen limits version (repeatable)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = get_config()
    bootstrap = args.bootstrap or cfg.kafka_bootstrap
    topic = args.topic or cfg.kafka_topic
    redis_url = args.redis_url or cfg.redis_url

    backend = cfg.persistence_backend
    if backend == "sqlite" and cfg.timescale_dsn:
        backend = "timescale"
    repo = get_repository(cfg, backend=backend if backend != "sqlite" else cfg.persistence_backend)
    try:
        require_streaming_repository(repo)
    except TypeError as exc:
        logger.error("%s", exc)
        return 2

    engine = StreamEngine(repo, redis_url=redis_url)

    # Preload from CLI and/or active stream registry
    for spec in args.preload:
        if ":" not in spec:
            logger.error("Invalid --preload %r (expected STREAM_KEY:LIMITS_VERSION)", spec)
            return 2
        key, version = spec.split(":", 1)
        engine.load_limits(key.strip(), version.strip(), ruleset=cfg.ruleset)
        logger.info("Preloaded stream %s @ limits %s", key, version)

    if hasattr(repo, "list_streams"):
        for row in repo.list_streams(active_only=True):
            key = row["stream_key"]
            version = row.get("limits_version")
            if not version:
                continue
            if key in engine.registered_keys():
                continue
            try:
                engine.load_limits(key, version, ruleset=row.get("ruleset") or cfg.ruleset)
                logger.info("Loaded registered stream %s @ %s", key, version)
            except KeyError as exc:
                logger.warning("Skip stream %s: %s", key, exc)

    source = KafkaSource(
        bootstrap,
        topic,
        group_id=args.group_id,
    )

    stop = False

    def _stop(*_args) -> None:
        nonlocal stop
        stop = True
        logger.info("Shutdown requested")

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    logger.info("Consuming %s from %s", topic, bootstrap)
    try:
        for msg in source.iter_sync():
            if stop:
                break
            key = str(msg.get("key") or "")
            if key and key not in engine.registered_keys():
                # Lazily attach from registry if present
                if hasattr(repo, "get_stream"):
                    row = repo.get_stream(key)
                    if row and row.get("limits_version"):
                        try:
                            engine.load_limits(
                                key,
                                row["limits_version"],
                                ruleset=row.get("ruleset") or cfg.ruleset,
                            )
                        except KeyError:
                            logger.warning("No limits for stream %s; dropping observation", key)
                            continue
                    else:
                        logger.debug("Unregistered stream %s; dropping", key)
                        continue
                else:
                    logger.debug("Unregistered stream %s; dropping", key)
                    continue
            try:
                signals = engine.handle_message(msg)
                if signals:
                    logger.warning(
                        "OOC %s value=%s rules=%s",
                        key,
                        msg.get("value"),
                        [s.rule_id for s in signals],
                    )
            except Exception:  # noqa: BLE001
                logger.exception("Failed handling message %s", msg)
    except ImportError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
