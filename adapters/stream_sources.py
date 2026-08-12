"""Live observation sources — Kafka and MQTT (optional ``aspc[stream]`` extra).

Provides async iterators plus a sync ``iter_sync()`` that yields dicts
``{key, value, timestamp}`` for the stream engine.
"""
from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from adapters.stream import ObservationSource

logger = logging.getLogger(__name__)


@dataclass
class Observation:
    """Normalized measurement from a live source.

    ``value`` is a scalar for I-MR / attribute / EWMA / CUSUM streams, or a list
    of floats for Xbar-R / Xbar-S subgroup payloads.
    """

    key: str
    ts: datetime
    value: float | list[float]
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "timestamp": self.ts,
            **{k: v for k, v in self.raw.items() if k not in ("key", "value", "timestamp", "ts")},
        }


def _parse_payload(payload: bytes | str | dict, *, default_key: str = "default") -> Observation:
    if isinstance(payload, dict):
        data = payload
    else:
        text = payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else str(payload)
        text = text.strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Bare numeric payload
            return Observation(
                key=default_key,
                ts=datetime.now(UTC),
                value=float(text),
                raw={"value": float(text)},
            )

    if not isinstance(data, dict):
        return Observation(
            key=default_key,
            ts=datetime.now(UTC),
            value=float(data),
            raw={"value": float(data)},
        )

    key = str(data.get("key") or data.get("stream_key") or data.get("topic") or default_key)
    raw_ts = data.get("ts") or data.get("timestamp") or data.get("time")
    if raw_ts is None:
        ts = datetime.now(UTC)
    elif isinstance(raw_ts, datetime):
        ts = raw_ts if raw_ts.tzinfo else raw_ts.replace(tzinfo=UTC)
    elif isinstance(raw_ts, (int, float)):
        # Treat large numbers as ms epoch
        epoch = float(raw_ts)
        if epoch > 1e12:
            epoch /= 1000.0
        ts = datetime.fromtimestamp(epoch, tz=UTC)
    else:
        ts = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))

    value = data.get("value")
    if value is None:
        value = data.get("measurement") or data.get("v")
    if value is None:
        raise ValueError(f"Observation payload missing value: {data!r}")
    if isinstance(value, (list, tuple)):
        parsed: float | list[float] = [float(v) for v in value]
        if not parsed:
            raise ValueError(f"Observation payload has empty subgroup: {data!r}")
    else:
        parsed = float(value)
    return Observation(key=key, ts=ts, value=parsed, raw=dict(data))


class _AsyncSourceBase:
    """Mixin: sync iteration via background asyncio loop, yielding dicts."""

    async def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        raise TypeError(f"{type(self).__name__} must implement async __aiter__")

    def iter_sync(self, *, timeout: float | None = None) -> Iterator[dict[str, Any]]:
        """Yield ``{key, value, timestamp}`` dicts from a background async consumer."""
        q: queue.Queue[dict[str, Any] | BaseException | None] = queue.Queue(maxsize=256)
        stop = threading.Event()

        async def _pump() -> None:
            try:
                async for msg in self:  # type: ignore[attr-defined]
                    if stop.is_set():
                        break
                    q.put(msg)
            except BaseException as exc:  # noqa: BLE001 — forward to consumer
                q.put(exc)
            finally:
                q.put(None)

        def _runner() -> None:
            asyncio.run(_pump())

        thread = threading.Thread(target=_runner, name=type(self).__name__, daemon=True)
        thread.start()
        try:
            while True:
                item = q.get(timeout=timeout) if timeout else q.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop.set()


class KafkaSource(_AsyncSourceBase, ObservationSource):
    """Consume measurements from a Kafka / Redpanda topic via aiokafka.

    Raises ``ImportError`` (with install hint) if aiokafka is not installed.
    Sync iteration yields dicts ``{key, value, timestamp}``.
    """

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        *,
        group_id: str = "aspc-stream-engine",
        default_key: str = "default",
        auto_offset_reset: str = "latest",
    ):
        try:
            import aiokafka  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "KafkaSource requires aiokafka. "
                "Install with: pip install 'aspc[stream]'  (or pip install aiokafka)"
            ) from exc
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.default_key = default_key
        self.auto_offset_reset = auto_offset_reset
        self._consumer = None

    def __iter__(self) -> Iterator[float]:
        for msg in self.iter_sync():
            yield float(msg["value"])

    async def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        from aiokafka import AIOKafkaConsumer

        backoff = 1.0
        while True:
            consumer = AIOKafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset=self.auto_offset_reset,
                enable_auto_commit=False,
            )
            self._consumer = consumer
            try:
                await consumer.start()
                backoff = 1.0
                async for msg in consumer:
                    key_hint = (
                        msg.key.decode("utf-8")
                        if isinstance(msg.key, (bytes, bytearray))
                        else (str(msg.key) if msg.key is not None else self.default_key)
                    )
                    try:
                        obs = _parse_payload(
                            msg.value or b"", default_key=key_hint or self.default_key
                        )
                    except Exception:
                        # Poison message: commit past it so it is not retried forever
                        await consumer.commit()
                        raise
                    yield obs.as_dict()
                    await consumer.commit()
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
            finally:
                try:
                    await consumer.stop()
                except Exception:
                    pass
                self._consumer = None


class MQTTSource(_AsyncSourceBase, ObservationSource):
    """Subscribe to an MQTT topic via aiomqtt (or paho-mqtt sync fallback).

    Raises ``ImportError`` with install hint if neither client is available.
    Sync iteration yields dicts ``{key, value, timestamp}``.
    """

    def __init__(
        self,
        host: str,
        topic: str,
        *,
        port: int = 1883,
        username: str | None = None,
        password: str | None = None,
        default_key: str | None = None,
    ):
        self._backend: str
        try:
            import aiomqtt  # noqa: F401
            self._backend = "aiomqtt"
        except ImportError:
            try:
                import paho.mqtt.client as mqtt  # noqa: F401
                self._backend = "paho"
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "MQTTSource requires aiomqtt or paho-mqtt. "
                    "Install with: pip install 'aspc[stream]'  (or pip install aiomqtt)"
                ) from exc
        self.host = host
        self.port = port
        self.topic = topic
        self.username = username
        self.password = password
        self.default_key = default_key

    def __iter__(self) -> Iterator[float]:
        for msg in self.iter_sync():
            yield float(msg["value"])

    def iter_sync(self, *, timeout: float | None = None) -> Iterator[dict[str, Any]]:
        if self._backend == "paho":
            yield from self._iter_paho(timeout=timeout)
            return
        yield from super().iter_sync(timeout=timeout)

    def _iter_paho(self, *, timeout: float | None = None) -> Iterator[dict[str, Any]]:
        import time as _time

        import paho.mqtt.client as mqtt

        q: queue.Queue[dict[str, Any] | BaseException | None] = queue.Queue(maxsize=256)
        stop = threading.Event()

        def _on_message(_client, _userdata, message) -> None:
            try:
                topic_str = str(message.topic)
                default_key = self.default_key or topic_str
                obs = _parse_payload(message.payload, default_key=default_key)
                if obs.key == default_key and self.default_key is None:
                    obs = Observation(key=topic_str, ts=obs.ts, value=obs.value, raw=obs.raw)
                try:
                    q.put(obs.as_dict(), timeout=5.0)
                except queue.Full:
                    # Drop under backpressure rather than block the network thread,
                    # but surface the loss so Phase II ARL claims are not silently wrong.
                    logger.warning(
                        "MQTT backpressure: dropped observation key=%s topic=%s (queue full)",
                        obs.key,
                        topic_str,
                    )
            except BaseException as exc:  # noqa: BLE001
                q.put(exc)

        def _on_disconnect(client, _userdata, _flags, reason_code, _properties=None):
            if stop.is_set():
                return
            # paho VERSION2 signature; reconnect with backoff in a helper thread
            delay = 1.0
            while not stop.is_set():
                try:
                    client.reconnect()
                    client.subscribe(self.topic)
                    return
                except Exception:
                    _time.sleep(delay)
                    delay = min(delay * 2, 60.0)

        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except AttributeError:
            client = mqtt.Client()
        if self.username is not None:
            client.username_pw_set(self.username, self.password)
        client.on_message = _on_message
        try:
            client.on_disconnect = _on_disconnect
        except Exception:
            pass
        backoff = 1.0
        while True:
            try:
                client.connect(self.host, self.port)
                client.subscribe(self.topic)
                client.loop_start()
                backoff = 1.0
                break
            except Exception:
                _time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
        try:
            while True:
                item = q.get(timeout=timeout) if timeout else q.get()
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop.set()
            client.loop_stop()
            try:
                client.disconnect()
            except Exception:
                pass

    async def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        if self._backend != "aiomqtt":
            # Drive paho from a thread via iter_sync for async consumers.
            loop = asyncio.get_running_loop()
            q: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=256)
            stop = threading.Event()

            def _runner() -> None:
                try:
                    for msg in self._iter_paho():
                        if stop.is_set():
                            break
                        asyncio.run_coroutine_threadsafe(q.put(msg), loop).result()
                finally:
                    asyncio.run_coroutine_threadsafe(q.put(None), loop).result()

            thread = threading.Thread(target=_runner, daemon=True)
            thread.start()
            try:
                while True:
                    item = await q.get()
                    if item is None:
                        break
                    yield item
            finally:
                stop.set()
            return

        import aiomqtt

        kwargs: dict[str, Any] = {"hostname": self.host, "port": self.port}
        if self.username is not None:
            kwargs["username"] = self.username
        if self.password is not None:
            kwargs["password"] = self.password

        backoff = 1.0
        while True:
            try:
                async with aiomqtt.Client(**kwargs) as client:
                    await client.subscribe(self.topic)
                    backoff = 1.0
                    async for message in client.messages:
                        topic_str = str(message.topic)
                        default_key = self.default_key or topic_str
                        payload = message.payload
                        obs = _parse_payload(payload, default_key=default_key)
                        if obs.key == default_key and self.default_key is None:
                            obs = Observation(
                                key=topic_str, ts=obs.ts, value=obs.value, raw=obs.raw
                            )
                        yield obs.as_dict()
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
