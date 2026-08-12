"""Keyed Phase II stream engine — frozen limits, Tier-1 raw + Tier-2 OOC + Redis live."""
from __future__ import annotations

import json
import logging
from collections import OrderedDict
from datetime import UTC, datetime
from typing import Any, Protocol

from spc_core.evaluator import Phase2Evaluator
from spc_core.models import ControlLimits, Signal

logger = logging.getLogger(__name__)

# Warm this many prior points into the rule buffer after restart.
_RULE_WARMUP = 15
# Soft cap on in-memory stream state (LRU eviction of inactive keys).
_DEFAULT_MAX_STREAMS = 10_000


class StreamRepository(Protocol):
    """Minimal persistence surface required by :class:`StreamEngine`."""

    def save_raw_measurement(
        self, stream_key: str, ts: datetime, value: float, **meta: Any
    ) -> None: ...

    def save_ooc_event(
        self,
        stream_key: str,
        ts: datetime,
        *,
        limits_version: str | None,
        index: int,
        value: float,
        rule_id: str,
        rule_name: str,
        description: str,
        side: str | None = None,
    ) -> bool: ...

    def get_limits(self, version: str) -> dict[str, Any] | None: ...


class StreamEngine:
    """Per-stream :class:`Phase2Evaluator` state machine.

    Limits are frozen at ``register`` / ``load_limits`` time and never recomputed.
    Each observation is persisted as a Tier-1 raw measurement; OOC signals are
    written idempotently to Tier-2 and published as JSON to Redis channel
    ``spc:live:{key}``.
    """

    def __init__(
        self,
        repo: StreamRepository,
        *,
        redis_client: Any = None,
        redis_url: str | None = None,
        max_streams: int = _DEFAULT_MAX_STREAMS,
        restore_state: bool = True,
    ):
        if not hasattr(repo, "save_raw_measurement"):
            raise TypeError(
                f"{type(repo).__name__} does not support streaming "
                "(missing save_raw_measurement). Use the TimescaleDB backend."
            )
        self.repo = repo
        self._evaluators: OrderedDict[str, Phase2Evaluator] = OrderedDict()
        self._limits: dict[str, ControlLimits] = {}
        self._limits_versions: dict[str, str] = {}
        self._rulesets: dict[str, str] = {}
        self._redis = redis_client
        self._redis_url = redis_url
        self._max_streams = max(1, int(max_streams))
        self._restore_state = restore_state

    def _get_redis(self) -> Any:
        if self._redis is not None:
            return self._redis
        if not self._redis_url:
            return None
        try:
            import redis as redis_lib
        except ImportError:
            logger.warning("redis package not installed; live publish disabled")
            return None
        self._redis = redis_lib.Redis.from_url(self._redis_url, decode_responses=True)
        return self._redis

    def register(
        self,
        stream_key: str,
        limits: ControlLimits,
        ruleset: str = "nelson",
    ) -> None:
        """Attach a frozen Phase I limit set to ``stream_key``.

        When the repository can supply recent measurements, the evaluator index
        and rule buffer are restored so a process restart does not reset Phase II.
        """
        if stream_key in self._evaluators:
            self.unregister(stream_key)

        while len(self._evaluators) >= self._max_streams:
            oldest, _ = self._evaluators.popitem(last=False)
            self._limits.pop(oldest, None)
            self._limits_versions.pop(oldest, None)
            self._rulesets.pop(oldest, None)
            logger.warning("Evicted stream %s (max_streams=%s)", oldest, self._max_streams)

        ev = Phase2Evaluator(limits, ruleset=ruleset)
        if self._restore_state:
            self._restore_evaluator(stream_key, ev)
        self._evaluators[stream_key] = ev
        self._limits[stream_key] = limits
        self._limits_versions[stream_key] = limits.version
        self._rulesets[stream_key] = ruleset

    def _restore_evaluator(self, stream_key: str, ev: Phase2Evaluator) -> None:
        count_fn = getattr(self.repo, "count_raw_measurements", None)
        recent_fn = getattr(self.repo, "recent_raw_measurements", None)
        if not callable(count_fn) or not callable(recent_fn):
            return
        try:
            count = int(count_fn(stream_key))
            recent = recent_fn(stream_key, limit=_RULE_WARMUP)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to restore evaluator state for %s", stream_key)
            return
        if count <= 0:
            return
        values = [float(r["value"]) for r in recent]
        ev.seed_state(index=count - 1, values=values)
        logger.info(
            "Restored stream %s evaluator at index=%s (warmed %s points)",
            stream_key,
            count - 1,
            len(values),
        )

    def unregister(self, stream_key: str) -> None:
        """Drop in-memory evaluator state for a deactivated stream."""
        self._evaluators.pop(stream_key, None)
        self._limits.pop(stream_key, None)
        self._limits_versions.pop(stream_key, None)
        self._rulesets.pop(stream_key, None)

    def load_limits(
        self,
        stream_key: str,
        limits_version: str,
        *,
        ruleset: str = "nelson",
    ) -> ControlLimits:
        """Load frozen limits from the repository and register the stream."""
        stored = self.repo.get_limits(limits_version)
        if not stored:
            raise KeyError(f"Limits version not found: {limits_version}")
        limits = _limits_from_payload(stored["payload"])
        self.register(stream_key, limits, ruleset=ruleset)
        return limits

    def registered_keys(self) -> list[str]:
        return sorted(self._evaluators)

    def handle_observation(
        self,
        stream_key: str,
        value: float | list[float],
        ts: datetime | None = None,
        **meta: Any,
    ) -> list[Signal]:
        """Evaluate one observation; write raw + any OOC events; publish live; return signals.

        Scalar charts (I-MR, EWMA, …) expect a float. Xbar-R / Xbar-S expect a
        non-empty list of subgroup members; the subgroup mean is persisted and plotted.
        """
        ev = self._evaluators.get(stream_key)
        if ev is None:
            raise KeyError(f"Stream '{stream_key}' is not registered")
        # Touch LRU order
        self._evaluators.move_to_end(stream_key)

        if ts is None:
            ts = datetime.now(UTC)
        elif ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)

        if isinstance(value, (list, tuple)):
            subgroup = [float(v) for v in value]
            if not subgroup:
                raise ValueError("Empty subgroup observation")
            if not ev.is_subgroup_chart:
                raise ValueError(
                    f"{ev.limits.chart_type.value} expects scalar observations; "
                    f"got subgroup of size {len(subgroup)}"
                )
            plotted = float(sum(subgroup) / len(subgroup))
        else:
            if ev.is_subgroup_chart:
                raise ValueError(
                    f"{ev.limits.chart_type.value} plots subgroup means; "
                    "send value as a JSON list of subgroup observations"
                )
            subgroup = None
            plotted = float(value)

        limits_version = self._limits_versions[stream_key]
        self.repo.save_raw_measurement(
            stream_key,
            ts,
            plotted,
            limits_version=limits_version,
            quality_flag=meta.get("quality_flag"),
            machine_id=meta.get("machine_id"),
            gage_id=meta.get("gage_id"),
        )

        if subgroup is not None:
            signals = ev.observe_subgroup(subgroup)
        else:
            signals = ev.observe(plotted)

        for sig in signals:
            inserted = self.repo.save_ooc_event(
                stream_key,
                ts,
                limits_version=limits_version,
                index=sig.index,
                value=sig.value,
                rule_id=sig.rule_id,
                rule_name=sig.rule_name,
                description=sig.description,
                side=sig.side,
            )
            if not inserted:
                logger.debug(
                    "Duplicate OOC suppressed %s rule=%s ts=%s",
                    stream_key,
                    sig.rule_id,
                    ts,
                )

        primary = self._limits[stream_key].primary
        ucl = primary.ucl_at(0) if isinstance(primary.ucl, list) else primary.ucl
        lcl = primary.lcl_at(0) if isinstance(primary.lcl, list) else primary.lcl
        # Always publish the stream index so the dashboard can map markers.
        payload = {
            "type": "point",
            "stream_key": stream_key,
            "value": plotted,
            "timestamp": ts.isoformat(),
            "ts": ts.isoformat(),
            "index": ev.index,
            "ucl": float(ucl) if ucl is not None else None,
            "center": float(primary.center),
            "lcl": float(lcl) if lcl is not None else None,
            "limits_version": limits_version,
            "signals": [
                {
                    "rule_id": s.rule_id,
                    "rule_name": s.rule_name,
                    "index": s.index,
                    "value": s.value,
                    "description": s.description,
                    "side": s.side,
                }
                for s in signals
            ],
            "ooc": bool(signals),
        }
        self._publish(stream_key, payload)
        return signals

    def handle_message(self, msg: dict[str, Any]) -> list[Signal]:
        """Handle a source dict ``{key, value, timestamp}`` (plus optional meta).

        ``value`` may be a scalar or a list (Xbar subgroup).
        """
        key = str(msg.get("key") or msg.get("stream_key") or "")
        if not key:
            raise ValueError(f"Message missing key: {msg!r}")
        value = msg.get("value")
        if value is None:
            raise ValueError(f"Message missing value: {msg!r}")
        raw_ts = msg.get("timestamp") or msg.get("ts")
        ts: datetime | None
        if raw_ts is None:
            ts = None
        elif isinstance(raw_ts, datetime):
            ts = raw_ts
        else:
            ts = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
        meta = {
            k: v
            for k, v in msg.items()
            if k not in ("key", "stream_key", "value", "timestamp", "ts")
        }
        if isinstance(value, (list, tuple)):
            return self.handle_observation(key, [float(v) for v in value], ts, **meta)
        return self.handle_observation(key, float(value), ts, **meta)

    def _publish(self, stream_key: str, payload: dict[str, Any]) -> None:
        client = self._get_redis()
        if client is None:
            return
        channel = f"spc:live:{stream_key}"
        try:
            client.publish(channel, json.dumps(payload, default=str))
        except Exception:  # noqa: BLE001 — live publish must not break evaluation
            logger.exception("Failed to publish to Redis channel %s", channel)

    def close(self) -> None:
        if self._redis is not None:
            try:
                self._redis.close()
            except Exception:  # noqa: BLE001
                pass
            self._redis = None


def _limits_from_payload(payload: dict[str, Any]) -> ControlLimits:
    from spc_core.models import ChartType, ControlLimits, LimitSet

    components = {
        name: LimitSet(**comp) for name, comp in payload["components"].items()
    }
    return ControlLimits(
        chart_type=ChartType(payload["chart_type"]),
        subgroup_size=payload["subgroup_size"],
        components=components,
        sigma=payload.get("sigma"),
        source_n_points=payload.get("source_n_points"),
        notes=payload.get("notes") or {},
    )
