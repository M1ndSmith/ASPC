# Deployment

## Docker Compose (full stack)

Compose file: [`deploy/compose/docker-compose.yml`](../deploy/compose/docker-compose.yml).

```bash
docker compose -f deploy/compose/docker-compose.yml up -d --build
```

| Service | Role | Host port |
|---------|------|-----------|
| `api` | FastAPI | 8000 |
| `frontend` | Next.js dashboard | 3000 |
| `redpanda` | Kafka-compatible broker | 19092 (external), 9644 admin |
| `mosquitto` | MQTT broker | 1883 |
| `timescaledb` | Tier-1/Tier-2 persistence | 5433 (host → container 5432) |
| `redis` | Live alert pub/sub | 6379 |
| `stream-engine` | Kafka → Phase II eval → DB + Redis | — |
| `mqtt-bridge` | MQTT → Redpanda | — |
| `grafana` | Ops dashboards (`--profile ops`) | 3001 |

Default API env in Compose:

- `ASPC_PERSISTENCE_BACKEND=timescale`
- `ASPC_TIMESCALE_DSN=postgresql+asyncpg://aspc:aspc@timescaledb:5432/aspc`
- `ASPC_REDIS_URL=redis://redis:6379/0`
- `ASPC_KAFKA_BOOTSTRAP=redpanda:9092`
- `ASPC_API_KEYS=demokey`
- `ASPC_CORS_ORIGINS=http://localhost:3000`

Frontend: `NEXT_PUBLIC_API_URL=http://localhost:8000`, `NEXT_PUBLIC_WS_URL=ws://localhost:8000`.

Ops profile:

```bash
docker compose -f deploy/compose/docker-compose.yml --profile ops up -d
```

## Docker images

Python images use [uv](https://docs.astral.sh/uv/) (`COPY --from=ghcr.io/astral-sh/uv:latest`):

- [`deploy/docker/Dockerfile.api`](../deploy/docker/Dockerfile.api) → `aspc-api`
- [`deploy/docker/Dockerfile.stream_engine`](../deploy/docker/Dockerfile.stream_engine) → `aspc-stream-engine`
- [`deploy/docker/Dockerfile.mqtt_bridge`](../deploy/docker/Dockerfile.mqtt_bridge) → `aspc-mqtt-bridge`
- [`deploy/docker/Dockerfile.frontend`](../deploy/docker/Dockerfile.frontend) → Next.js (npm)

## Data path (real-time)

```mermaid
flowchart LR
  edge[Sensors / MQTT] --> mosquitto
  mosquitto --> bridge[mqtt-bridge]
  bridge --> redpanda
  redpanda --> engine[stream-engine]
  engine --> tsdb[TimescaleDB Tier1+Tier2]
  engine --> redis
  redis --> api[API WS fan-out]
  api --> ui[Next.js Live page]
```

1. Edge publishes to Mosquitto (`sensors/#` by default).
2. `mqtt-bridge` forwards to Kafka topic `spc.measurements`.
3. `stream-engine` evaluates each keyed stream against **frozen** limits, writes raw + OOC events, publishes to Redis `spc:live:{stream_key}`.
4. API WebSocket `/ws/live/{stream_key}` fans out to the dashboard.

Go-live requires Timescale stream registry: Phase I analysis → `limits_version` → `POST /streams/{key}/go-live`.

## Migrations (TimescaleDB)

Alembic:

- Config: [`alembic.ini`](../alembic.ini)
- Env: [`migrations/env.py`](../migrations/env.py)
- Initial: [`migrations/versions/001_initial.py`](../migrations/versions/001_initial.py) (hypertables + analysis tables)

Point `ASPC_TIMESCALE_DSN` / Alembic URL at the database and run:

```bash
uv pip install -e ".[tsdb]"
alembic upgrade head
```

(Compose may rely on repository bootstrap depending on your ops practice; prefer explicit migrations in production.)

## Observability

- `GET /metrics` — Prometheus counters/histograms when `prometheus-client` is installed
- Grafana optional profile on port 3001
- Structured logging via `structlog` in the apps extras

## Minimal local (no Compose)

SQLite + API + optional frontend:

```bash
uv pip install -e ".[dev]"
aspc serve --port 8000
# another shell:
cd frontend && cp .env.example .env.local && npm install && npm run dev
```

Streaming features need Redis / Kafka / Timescale as above.
