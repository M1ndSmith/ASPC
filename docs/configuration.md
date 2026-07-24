# Configuration

Application config is YAML defaults merged with environment overrides. Sources:

- [`apps/config.yaml`](../apps/config.yaml) — checked-in defaults
- [`apps/config.py`](../apps/config.py) — load + merge + `ASPC_*` overrides
- [`env.example`](../env.example) — copy to `.env` at repo root

**Precedence:** environment variables win over YAML.

## YAML sections

### `api`

| Key | Default | Meaning |
|-----|---------|---------|
| `host` | `0.0.0.0` | Bind address |
| `port` | `8000` | HTTP port |
| `cors_origins` | `["*"]` | CORS allow list |

### `auth`

| Key | Default | Meaning |
|-----|---------|---------|
| `enabled` | `true` | Require JWT on protected routes |
| `jwt_secret` | `change-me-in-production` | HS256 secret |
| `jwt_algorithm` | `HS256` | JWT algorithm |
| `jwt_expire_minutes` | `60` | Token lifetime |
| `admin_password` | `admin` | Password accepted by `/auth/token` |
| `api_keys` | `[]` | Allowed `X-API-Key` values |

### `uploads` / `reports`

| Key | Default |
|-----|---------|
| `uploads.temp_directory` | `var/uploads` |
| `uploads.max_file_size_mb` | `10` |
| `uploads.allowed_extensions` | `.csv`, `.parquet`, `.pq` |
| `reports.auto_generate` | `true` |
| `reports.output_directory` | `var/reports` |
| `reports.include_plots` | `true` |

Directories are created on write. Both under `var/` are gitignored.

### `persistence`

| Key | Default | Meaning |
|-----|---------|---------|
| `backend` | `sqlite` | `sqlite` or `timescale` |
| `sqlite_path` | `aspc.db` | Local DB file |
| `timescale_dsn` | `null` | SQLAlchemy/asyncpg DSN |

Factory: [`adapters/factory.py`](../adapters/factory.py) → `SQLiteRepository` or `TimescaleDBRepository`.

### `redis` / `kafka`

| Key | Default |
|-----|---------|
| `redis.url` | `redis://localhost:6379/0` |
| `kafka.bootstrap` | `localhost:9092` |
| `kafka.topic` | `spc.measurements` |

### `spc`

| Key | Default | Meaning |
|-----|---------|---------|
| `ruleset` | `nelson` | Default run-rule set |
| `min_phase1_points` | `25` | Checklist minimum plotted points |
| `acf_threshold` | `0.2` | Lag-1 ACF gate |

## Environment overrides

| Variable | Maps to |
|----------|---------|
| `ASPC_AUTH_ENABLED` | `auth.enabled` (`1`/`true`/`yes`) |
| `ASPC_JWT_SECRET` | `auth.jwt_secret` |
| `ASPC_ADMIN_PASSWORD` | `auth.admin_password` |
| `ASPC_API_KEYS` | `auth.api_keys` (comma-separated) |
| `ASPC_API_HOST` | `api.host` |
| `ASPC_API_PORT` | `api.port` |
| `ASPC_CORS_ORIGINS` | `api.cors_origins` (comma-separated) |
| `ASPC_PERSISTENCE_BACKEND` | `persistence.backend` |
| `ASPC_SQLITE_PATH` | `persistence.sqlite_path` |
| `ASPC_TIMESCALE_DSN` / `DATABASE_URL` | `persistence.timescale_dsn` |
| `ASPC_REDIS_URL` | `redis.url` |
| `ASPC_KAFKA_BOOTSTRAP` | `kafka.bootstrap` |
| `ASPC_KAFKA_TOPIC` | `kafka.topic` |

MQTT bridge also reads `ASPC_MQTT_HOST`, `ASPC_MQTT_PORT`, `ASPC_MQTT_TOPIC` (service-level, not in `Config`).

## Local setup

```bash
cp env.example .env
# edit secrets
uv pip install -e ".[dev]"
aspc serve
```

For the full stack, prefer Compose env (see [deployment.md](deployment.md)) over a local `.env`.
