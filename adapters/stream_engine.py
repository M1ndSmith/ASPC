"""Keyed Phase II stream engine — frozen limits, Tier-1 raw + Tier-2 OOC + Redis live."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

from spc_core.evaluator import Phase2Evaluator
from spc_core.models import ControlLimits, Signal

logger = logging.getLogger(__name__)


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
        limits_version: Optional[str],
        index: int,
        value: float,
        rule_id: str,
        rule_name: str,
        description: str,
        side: Optional[str] = None,
    ) -> bool: ...

    def get_limits(self, version: str) -> Optional[dict[str, Any]]: ...


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
        redis_url: Optional[str] = None,
    ):
        self.repo = repo
        self._evaluators: dict[str, Phase2Evaluator] = {}
        self._limits: dict[str, ControlLimits] = {}
        self._limits_versions: dict[str, str] = {}
        self._rulesets: dict[str, str] = {}
        self._redis = redis_client
        self._redis_url = redis_url

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
        """Attach a frozen Phase I limit set to ``stream_key``."""
        self._evaluators[stream_key] = Phase2Evaluator(limits, ruleset=ruleset)
        self._limits[stream_key] = limits
        self._limits_versions[stream_key] = limits.version
        self._rulesets[stream_key] = ruleset

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
        value: float,
        ts: datetime | None = None,
        **meta: Any,
    ) -> list[Signal]:
        """Evaluate one observation; write raw + any OOC events; publish live; return signals."""
        ev = self._evaluators.get(stream_key)
        if ev is None:
            raise KeyError(f"Stream '{stream_key}' is not registered")
        if ts is None:
            ts = datetime.now(timezone.utc)
        elif ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        limits_version = self._limits_versions[stream_key]
        self.repo.save_raw_measurement(
            stream_key,
            ts,
            float(value),
            limits_version=limits_version,
            quality_flag=meta.get("quality_flag"),
            machine_id=meta.get("machine_id"),
            gage_id=meta.get("gage_id"),
        )

        signals = ev.observe(float(value))
        for sig in signals:
            self.repo.save_ooc_event(
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

        primary = self._limits[stream_key].primary
        # Scalar limits for the Live dashboard (variable P/U charts use first/index 0).
        ucl = primary.ucl_at(0) if isinstance(primary.ucl, list) else primary.ucl
        lcl = primary.lcl_at(0) if isinstance(primary.lcl, list) else primary.lcl
        payload = {
            "type": "point",
            "stream_key": stream_key,
            "value": float(value),
            "timestamp": ts.isoformat(),
            "ts": ts.isoformat(),
            "index": signals[0].index if signals else None,
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
        """Handle a source dict ``{key, value, timestamp}`` (plus optional meta)."""
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
