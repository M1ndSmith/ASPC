# CLI reference (`aspc`)

Entry point: [`apps/cli/main.py`](../apps/cli/main.py) → console script `aspc`.

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
uv run python -m sample_data --out examples/data
```

## Commands

### `aspc control-chart`

Runs the gated Phase I pipeline (`establish`) and persists limits + run to SQLite.

| Flag | Required | Description |
|------|----------|-------------|
| `-f` / `--file` | yes | CSV or Parquet path |
| `--value-col` | no | Measurement column (auto-detected if omitted) |
| `--subgroup-col` | no | Subgroup id column |
| `--sample-size-col` | no | For P / NP charts |
| `--opportunity-col` | no | For U charts |
| `--chart-type` | no | One of `ChartType` values (`I-MR`, `Xbar-R`, …) |
| `--ruleset` | no | Default from config (`nelson`) |
| `--html PATH` | no | Write Plotly HTML report |
| `--json` | no | Print full JSON report to stdout |

Example:

```bash
aspc control-chart -f examples/data/spc_individual_out_of_control.csv --json
aspc control-chart -f examples/data/spc_subgroup_data.csv --html /tmp/xbar.html
```

Human output includes chart type, limits version, signal count, checklist pass flag, and up to 10 signals.

### `aspc capability`

| Flag | Required | Description |
|------|----------|-------------|
| `-f` / `--file` | yes | Input file |
| `--usl` | yes | Upper spec limit |
| `--lsl` | yes | Lower spec limit |
| `--target` | no | Target for Cpm |
| `--value-col` | no | Measurement column |
| `--subgroup-col` | no | Optional subgroups for short-term sigma |
| `--html` | no | HTML report path |
| `--json` | no | JSON report |

```bash
aspc capability -f examples/data/capability_excellent.csv --usl 10.5 --lsl 9.5
```

### `aspc msa`

| Flag | Required | Description |
|------|----------|-------------|
| `-f` / `--file` | yes | Input file |
| `--study-type` | no | `gage_rr` \| `bias` \| `linearity` \| `stability` (auto if omitted) |
| `--method` | no | `anova` (default) or `range` for Gage R&R |
| `--tolerance` | no | Spec tolerance for %GRR of tolerance |
| `--part-col` | no | Part column |
| `--operator-col` | no | Operator column |
| `--measurement-col` | no | Measurement column |
| `--reference-col` | no | Reference for bias / linearity |
| `--html` | no | HTML report |
| `--json` | no | JSON report |

Auto study selection:

- Part + operator columns → `gage_rr`
- Reference column with >1 unique value → `linearity`; else `bias`
- Otherwise → `stability`

```bash
aspc msa -f examples/data/msa_gage_rr_excellent.csv --study-type gage_rr
aspc msa -f examples/data/msa_bias_study.csv --study-type bias
```

### `aspc serve`

Starts the FastAPI app via uvicorn.

| Flag | Default |
|------|---------|
| `--host` | config `api.host` (`0.0.0.0`) |
| `--port` | config `api.port` (`8000`) |

```bash
aspc serve --port 8000
# equivalent entry point:
aspc-api
```

## Column auto-detection

[`spc_core.ingest`](../spc_core/ingest.py) maps common header names (`measurement`, `Part`, `Operator`, `subgroup`, `defective`, `inspected`, …). Pass explicit `--*-col` flags when headers are nonstandard.

## Persistence

CLI writes to the configured SQLite path (`ASPC_SQLITE_PATH` / `persistence.sqlite_path`, default `aspc.db`): frozen limits and analysis runs. Reports HTML goes to `--html` if provided.

See [configuration.md](configuration.md) and [api.md](api.md).
