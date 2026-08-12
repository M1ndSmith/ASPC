# HTTP API reference

Source: [`apps/api/main.py`](../apps/api/main.py). Interactive docs: `GET /docs` (Swagger) when the server is running.

```bash
aspc serve --port 8000
# or: aspc-api
```

## Authentication

### JWT (`Authorization: Bearer …`)

Most analyze / history endpoints depend on `get_current_user`:

- If `auth.enabled` / `ASPC_AUTH_ENABLED` is **false**, requests proceed as anonymous.
- If enabled, a valid Bearer JWT is required.

Obtain a token (password must match `ASPC_ADMIN_PASSWORD`; username must match
`ASPC_ADMIN_USERNAME`, default `admin`). Startup refuses the default password
`admin` and an empty API-key list unless `ASPC_DEV_INSECURE=1`:

```bash
curl -s -X POST http://localhost:8000/auth/token \
  -d 'username=admin&password=$ASPC_ADMIN_PASSWORD'
# → {"access_token":"…","token_type":"bearer"}
```

### API key (`X-API-Key`)

Required for stream mutations and alert ack (`require_api_key`):

- Keys from config `auth.api_keys` or `ASPC_API_KEYS` (comma-separated)
- If **no keys configured**, requests fail closed unless `ASPC_DEV_INSECURE=1`

```bash
curl -H "X-API-Key: $ASPC_API_KEY" -H "Authorization: Bearer $TOKEN" …
```

## Common response shape

Analyze endpoints return `AnalyzeResponse`:

```json
{
  "status": "success",
  "run_id": "…",
  "analysis_type": "control_chart",
  "report": { },
  "html_report": "var/reports/<run_id>_control_chart.html",
  "checklist": { "passed": false, "items": [ ] }
}
```

`checklist` is present for control-chart analysis (Phase I go-live items).

## Endpoints

### Health and metrics

| Method | Path | Auth | Notes |
|--------|------|------|-------|
| `GET` | `/` | no | Endpoint map |
| `GET` | `/health` | no | `{status, version}` |
| `GET` | `/metrics` | no | Prometheus text (if `prometheus-client` installed) |

### Auth

| Method | Path | Body |
|--------|------|------|
| `POST` | `/auth/token` | OAuth2 password form: `username`, `password` |

### Analyze (multipart form + file)

All require JWT when auth is enabled. Upload CSV/Parquet (`file`).

#### `POST /analyze/control-chart`

Form fields: `value_col`, `subgroup_col`, `sample_size_col`, `opportunity_col`, `chart_type`, `ruleset`, `user_id`, `include_records` (bool).

Runs `establish` + `phase1_checklist`, saves limits and run, optional HTML under `var/reports/`.

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/auth/token \
  -d 'username=op&password=admin' | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])')

curl -s -X POST http://localhost:8000/analyze/control-chart \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@examples/data/spc_individual_in_control.csv" \
  -F "include_records=false"
```

#### `POST /analyze/capability`

Form: `usl` (required), `lsl` (required), `target`, `value_col`, `subgroup_col`, `user_id`. Rejects `usl <= lsl`.

#### `POST /analyze/msa`

Form: `study_type`, `method` (`anova`\|`range`), `tolerance`, `part_col`, `operator_col`, `measurement_col`, `trial_col`, `reference_col`, `user_id`.

### History

| Method | Path | Notes |
|--------|------|-------|
| `GET` | `/runs` | Query: `analysis_type`, `limit` (≤500) |
| `GET` | `/runs/{run_id}` | Full stored run |
| `GET` | `/reports/{run_id}` | HTML if generated, else JSON fallback (no auth) |

### Streams (TimescaleDB backend)

These call repository methods available on the Timescale adapter. With SQLite they return **501** (register / go-live / ack) or an empty list (list).

| Method | Path | Auth | Body / notes |
|--------|------|------|----------------|
| `POST` | `/streams/register` | JWT + API key | JSON: `stream_key`, optional `topic`, `chart_type`, `ruleset`, `meta` |
| `POST` | `/streams/{stream_key}/go-live` | JWT + API key | JSON: `limits_version`, optional `ruleset` — freezes go-live against stored limits |
| `GET` | `/streams` | JWT | Query: `active_only` |
| `POST` | `/alerts/{event_id}/ack` | JWT + API key | Acknowledge OOC event |

```bash
curl -X POST http://localhost:8000/streams/register \
  -H "Authorization: Bearer $TOKEN" -H "X-API-Key: demokey" \
  -H "Content-Type: application/json" \
  -d '{"stream_key":"line-a","chart_type":"I-MR","ruleset":"nelson"}'

curl -X POST http://localhost:8000/streams/line-a/go-live \
  -H "Authorization: Bearer $TOKEN" -H "X-API-Key: demokey" \
  -H "Content-Type: application/json" \
  -d '{"limits_version":"<16-char-hash>"}'
```

### Live WebSocket

`WS /ws/live/{stream_key}`

Subscribes to Redis channel `spc:live:{stream_key}` and forwards JSON messages. Requires `redis` package and a reachable `ASPC_REDIS_URL`. First message: `{"event":"subscribed","channel":"…"}`.

### SSE replay

`GET /stream/replay?file_path=…&value_col=measurement&limits_version=…`

Server-side CSV replay against frozen limits; emits SSE `data:` lines of `Signal` JSON, then `{"event":"done"}`.

## CORS

Configured via `api.cors_origins` / `ASPC_CORS_ORIGINS` (comma-separated). Compose defaults to `http://localhost:3000` for the dashboard.

See [configuration.md](configuration.md) and [deployment.md](deployment.md).
