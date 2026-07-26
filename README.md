# ASPC — Production Statistical Process Control

Correct, tested SPC for **batch** and **real-time** work: Shewhart / EWMA / CUSUM charts, MSA, capability, TimescaleDB, Redpanda/MQTT streaming, and a Next.js operator dashboard.

Phase I establishes and freezes versioned limits; Phase II evaluates new data against those limits. No stubs — every advertised `ChartType` has real math.

## Why ASPC?

Manufacturing needs SPC that is **correct**, **gated before go-live**, and **usable in real time** against frozen limits — not only a desktop chart after the shift. See:

| Doc | Contents |
|-----|----------|
| [docs/overview/problem-and-solution.md](docs/overview/problem-and-solution.md) | Problem, solution, Phase I → freeze → Phase II |
| [docs/overview/capabilities.md](docs/overview/capabilities.md) | Statistical + operational capabilities |
| [docs/overview/use-cases.md](docs/overview/use-cases.md) | Stamping / pharma / molding integration patterns |
| [docs/overview/benchmarking.md](docs/overview/benchmarking.md) | Performance, accuracy, resilience evidence |

Reproduce benches: `python benchmarks/performance.py` and `python benchmarks/accuracy.py` ([benchmarks/README.md](benchmarks/README.md)). Claims are scoped in [docs/overview/benchmarking.md](docs/overview/benchmarking.md): formula fidelity ≠ Minitab/JMP parity; MSA and `valid_range` gates are **opt-in** (absence warns / skips, does not always STOP).

## Features

- Gated Phase I pipeline (`establish`) with ok / warn / stop gates and a 10-item go-live checklist
- Charts: I-MR, Xbar-R, Xbar-S, P, NP, C, U, EWMA, CUSUM
- MSA: Gage R&R (ANOVA / range), bias, linearity, stability, NDC and 10:1 resolution gates
- Capability: Cp/Cpk/Pp/Ppk, DPMO / sigma level, parametric · transformed · nonparametric routing
- FastAPI + CLI, JWT / API-key auth, WebSocket live alerts, SSE replay
- Docker Compose stack: Redpanda, Mosquitto, TimescaleDB, Redis, stream engine, MQTT bridge, UI

## Architecture

```
spc_core/                 Pure statistics (Shewhart, EWMA, CUSUM, MSA, capability, gated pipeline)
adapters/                 I/O, SQLite/TimescaleDB, Plotly, Kafka/MQTT sources, stream engine
apps/api/                 FastAPI — JWT + API-key auth, REST, SSE replay, WebSocket live
apps/cli/                 aspc CLI
services/stream_engine/   Kafka consumer → Phase II eval → Tier1/Tier2 + Redis
services/mqtt_bridge/     MQTT → Redpanda bridge
frontend/                 Next.js operator dashboard
deploy/compose/           Full stack orchestration
migrations/               Alembic (Timescale hypertables + analysis tables)
sample_data/              Deterministic synthetic datasets for tests and demos
docs/                     Full documentation
```

## Quick start

Install [uv](https://docs.astral.sh/uv/), then:

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

uv run python -m sample_data --out examples/data
aspc control-chart -f examples/data/spc_individual_out_of_control.csv --json

aspc serve --port 8000
```

Dashboard:

```bash
cd frontend && cp .env.example .env.local && npm install && npm run dev
# http://localhost:3000
```

Full stack:

```bash
docker compose -f deploy/compose/docker-compose.yml up -d --build
```

| Service | Port |
|---------|------|
| API | 8000 |
| Frontend | 3000 |
| Redpanda | 19092 |
| Mosquitto | 1883 |
| TimescaleDB | 5433 |
| Redis | 6379 |
| Grafana (ops) | 3001 |

## Documentation

| Guide | Description |
|-------|-------------|
| [docs/overview/problem-and-solution.md](docs/overview/problem-and-solution.md) | Why ASPC — problem, solution, architecture |
| [docs/overview/benchmarking.md](docs/overview/benchmarking.md) | Performance, accuracy, robustness |
| [docs/index.md](docs/index.md) | Doc map and Phase I → freeze → Phase II model |
| [docs/concepts.md](docs/concepts.md) | Charts, rules, MSA, capability, flags |
| [docs/pipeline.md](docs/pipeline.md) | Gated `establish()`, checklist, `SPCRecord` |
| [docs/cli.md](docs/cli.md) | `aspc` command reference |
| [docs/api.md](docs/api.md) | REST, auth, WebSocket, SSE |
| [docs/python-api.md](docs/python-api.md) | Library usage and extras |
| [docs/configuration.md](docs/configuration.md) | YAML + `ASPC_*` env |
| [docs/deployment.md](docs/deployment.md) | Compose, images, migrations |
| [docs/development.md](docs/development.md) | Tests, lint, `sample_data`, CI |

Interactive OpenAPI: `http://localhost:8000/docs` when the API is running.

## Standards

AIAG MSA-4 · AIAG SPC · ISO 7870 · Six Sigma DMAIC

## License

See [LICENSE](LICENSE).
