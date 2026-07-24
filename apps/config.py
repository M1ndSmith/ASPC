"""Application config — YAML + env overrides."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from dotenv import load_dotenv


DEFAULTS: dict[str, Any] = {
    "api": {
        "host": "0.0.0.0",
        "port": 8000,
        "cors_origins": ["*"],
    },
    "auth": {
        "enabled": True,
        "jwt_secret": "change-me-in-production",
        "jwt_algorithm": "HS256",
        "jwt_expire_minutes": 60,
        "admin_password": "admin",
        "api_keys": [],
    },
    "uploads": {
        "temp_directory": "var/uploads",
        "max_file_size_mb": 10,
        "allowed_extensions": [".csv", ".parquet", ".pq"],
    },
    "reports": {
        "auto_generate": True,
        "output_directory": "var/reports",
        "include_plots": True,
    },
    "persistence": {
        "backend": "sqlite",
        "sqlite_path": "aspc.db",
        "timescale_dsn": None,
    },
    "redis": {
        "url": "redis://localhost:6379/0",
    },
    "kafka": {
        "bootstrap": "localhost:9092",
        "topic": "spc.measurements",
    },
    "spc": {
        "ruleset": "nelson",
        "min_phase1_points": 25,
        "acf_threshold": 0.2,
    },
}


class Config:
    def __init__(self, config_path: Optional[str | Path] = None):
        root = Path(__file__).resolve().parents[1]  # repo root (apps/ -> ASPC/)
        load_dotenv(root / ".env")

        if config_path is None:
            candidates = [
                Path(__file__).parent / "config.yaml",
                root / "config" / "config.yaml",
            ]
            config_path = next((p for p in candidates if p.exists()), candidates[0])

        self.config_path = Path(config_path)
        self.config = self._deep_merge(DEFAULTS, self._load_yaml())
        self._apply_env_overrides()

    def _load_yaml(self) -> dict:
        if not self.config_path.exists() or yaml is None:
            return {}
        with self.config_path.open() as f:
            data = yaml.safe_load(f) or {}
        return data

    def _apply_env_overrides(self) -> None:
        """Environment variables win over YAML for deployment wiring."""
        p = self.config.setdefault("persistence", {})
        if os.getenv("ASPC_PERSISTENCE_BACKEND"):
            p["backend"] = os.environ["ASPC_PERSISTENCE_BACKEND"]
        if os.getenv("ASPC_SQLITE_PATH"):
            p["sqlite_path"] = os.environ["ASPC_SQLITE_PATH"]
        if os.getenv("ASPC_TIMESCALE_DSN") or os.getenv("DATABASE_URL"):
            p["timescale_dsn"] = os.getenv("ASPC_TIMESCALE_DSN") or os.getenv("DATABASE_URL")

        api = self.config.setdefault("api", {})
        if os.getenv("ASPC_CORS_ORIGINS"):
            raw = os.environ["ASPC_CORS_ORIGINS"]
            api["cors_origins"] = [o.strip() for o in raw.split(",") if o.strip()]
        if os.getenv("ASPC_API_HOST"):
            api["host"] = os.environ["ASPC_API_HOST"]
        if os.getenv("ASPC_API_PORT"):
            api["port"] = int(os.environ["ASPC_API_PORT"])

        auth = self.config.setdefault("auth", {})
        if os.getenv("ASPC_JWT_SECRET"):
            auth["jwt_secret"] = os.environ["ASPC_JWT_SECRET"]
        if os.getenv("ASPC_ADMIN_PASSWORD"):
            auth["admin_password"] = os.environ["ASPC_ADMIN_PASSWORD"]
        if os.getenv("ASPC_API_KEYS"):
            auth["api_keys"] = [k.strip() for k in os.environ["ASPC_API_KEYS"].split(",") if k.strip()]
        if os.getenv("ASPC_AUTH_ENABLED") is not None:
            auth["enabled"] = os.environ["ASPC_AUTH_ENABLED"].lower() in ("1", "true", "yes")

        redis = self.config.setdefault("redis", {})
        if os.getenv("ASPC_REDIS_URL"):
            redis["url"] = os.environ["ASPC_REDIS_URL"]

        kafka = self.config.setdefault("kafka", {})
        if os.getenv("ASPC_KAFKA_BOOTSTRAP"):
            kafka["bootstrap"] = os.environ["ASPC_KAFKA_BOOTSTRAP"]
        if os.getenv("ASPC_KAFKA_TOPIC"):
            kafka["topic"] = os.environ["ASPC_KAFKA_TOPIC"]

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        out = dict(base)
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = Config._deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    @property
    def api_host(self) -> str:
        return self.config["api"]["host"]

    @property
    def api_port(self) -> int:
        return int(self.config["api"]["port"])

    @property
    def cors_origins(self) -> list[str]:
        return list(self.config["api"]["cors_origins"])

    @property
    def temp_upload_dir(self) -> str:
        return self.config["uploads"]["temp_directory"]

    @property
    def max_file_size_bytes(self) -> int:
        return int(self.config["uploads"]["max_file_size_mb"]) * 1024 * 1024

    @property
    def allowed_extensions(self) -> list[str]:
        return list(self.config["uploads"]["allowed_extensions"])

    @property
    def report_dir(self) -> str:
        return self.config["reports"]["output_directory"]

    @property
    def persistence_backend(self) -> str:
        return str(self.config["persistence"]["backend"])

    @property
    def sqlite_path(self) -> str:
        return self.config["persistence"]["sqlite_path"]

    @property
    def timescale_dsn(self) -> Optional[str]:
        return self.config["persistence"].get("timescale_dsn")

    @property
    def redis_url(self) -> str:
        return str(self.config["redis"]["url"])

    @property
    def kafka_bootstrap(self) -> str:
        return str(self.config["kafka"]["bootstrap"])

    @property
    def kafka_topic(self) -> str:
        return str(self.config["kafka"].get("topic") or "spc.measurements")

    @property
    def jwt_secret(self) -> str:
        return str(self.config["auth"]["jwt_secret"])

    @property
    def jwt_algorithm(self) -> str:
        return str(self.config["auth"].get("jwt_algorithm") or "HS256")

    @property
    def jwt_expire_minutes(self) -> int:
        return int(self.config["auth"].get("jwt_expire_minutes") or 60)

    @property
    def admin_password(self) -> str:
        return str(self.config["auth"].get("admin_password") or "admin")

    @property
    def api_keys(self) -> list[str]:
        return list(self.config["auth"].get("api_keys") or [])

    @property
    def auth_enabled(self) -> bool:
        return bool(self.config["auth"].get("enabled", True))

    @property
    def ruleset(self) -> str:
        return self.config["spc"]["ruleset"]

    @property
    def acf_threshold(self) -> float:
        return float(self.config["spc"]["acf_threshold"])

    @property
    def min_phase1_points(self) -> int:
        return int(self.config["spc"].get("min_phase1_points") or 25)


_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
