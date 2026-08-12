# Development

## Tooling

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) for installs and lockfile (`uv.lock`)
- Node 20+ for `frontend/`
- Optional: Docker for integration / Compose

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp env.example .env
```

Regenerate lock after dependency changes:

```bash
uv lock
```

## Layout

| Path | Role |
|------|------|
| `spc_core/` | Pure statistics |
| `adapters/` | I/O, persistence, Plotly, stream sources |
| `apps/` | FastAPI + CLI + config |
| `services/` | stream-engine, mqtt-bridge |
| `sample_data/` | Deterministic synthetic datasets |
| `resilience_data/` | Standards-mapped sad/happy path corpus |
| `benchmarks/` | Performance + accuracy harnesses |
| `frontend/` | Next.js operator UI |
| `tests/` | unit + integration + resilience |
| `docs/` | This documentation |
| `deploy/` | Docker + Compose |
| `migrations/` | Alembic |

## Tests

```bash
# Prefer disabling ROS/launch pytest plugins if present on the machine
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q

# Frontend
cd frontend && npm test
```

Notes:

- Unit tests use in-memory `sample_data` via `dataset` / `write_dataset` fixtures ([`tests/conftest.py`](../tests/conftest.py)) — **no committed CSVs**.
- Hypothesis property tests live under `tests/unit/test_hypothesis_limits.py`.
- Integration tests that need Timescale/Redis skip unless `ASPC_TIMESCALE_DSN` / `ASPC_REDIS_URL` are set (`tests/integration/test_realtime_gated.py`).

Acceptance-style smoke (also in CI):

```bash
python - <<'PY'
from spc_core import ChartType, analyze_control_chart, establish
from spc_core.ewma import ewma_chart
from spc_core.cusum import cusum_chart
import numpy as np
x = np.random.default_rng(0).normal(10, 1, 40)
assert analyze_control_chart(x).chart_type == ChartType.I_MR
assert ewma_chart(x).limits.chart_type == ChartType.EWMA
assert cusum_chart(x).limits.chart_type == ChartType.CUSUM
pipe = establish(x)
assert len(pipe.chart.to_records()) == len(pipe.chart.plotted_values)
print("ok")
PY
```

## Lint / types

Configured in [`pyproject.toml`](../pyproject.toml):

```bash
ruff check spc_core adapters apps services sample_data resilience_data scripts tests benchmarks
mypy spc_core   # tool.mypy.packages = ["spc_core"] only
```

CI runs **ruff as a blocking check** and **mypy on `spc_core` only** (adapters/apps/services are not type-gated yet).

## Benchmarks

In-process performance and formula accuracy (see [overview/benchmarking.md](overview/benchmarking.md)):

```bash
.venv/bin/python benchmarks/performance.py
.venv/bin/python benchmarks/accuracy.py   # exit 1 if any check fails
```

## Synthetic data (`sample_data`)

Package: [`sample_data/`](../sample_data/). Generators are numpy-seeded and honor column-name contracts used by ingest/tests.

```bash
# Write demo CSVs (gitignored under examples/data/)
uv run python -m sample_data --out examples/data

# Or import in Python / tests
from sample_data import spc_individual, msa_gage_rr, capability, get_dataset, write_csv
```

Catalog names include `spc_individual_in_control`, `msa_gage_rr_excellent`, `capability_excellent`, attribute chart sets, etc. See `DATASET_CATALOG` / `get_dataset` in the package.

## CI

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml):

1. **python** (3.11 / 3.12) — `uv pip install -e ".[dev]"`, blocking ruff, mypy `spc_core`, pytest (`not integration`, includes `tests/resilience`), `benchmarks/accuracy.py`, acceptance script
2. **integration** — Timescale + Redis services; `tests/integration`
3. **frontend** — npm ci/lint/test/build
4. **docker** — build API + frontend images

## Conventions

- Do not put I/O or LLM imports in `spc_core`.
- Prefer frozen `ControlLimits` + `limits.version` over recomputing Phase I for live data.
- No `NotImplementedError` stubs for advertised features — optional deps should raise `ImportError` with an install hint.
