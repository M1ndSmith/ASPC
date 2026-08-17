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
| `GET` | `/health` | no | `{status, version, checks}` |
| `GET` | `/metrics` | JWT | Prometheus text (if `prometheus-client` installed) |

### Auth

| Method | Path | Body |
|--------|------|------|
| `POST` | `/auth/token` | OAuth2 password form: `username`, `password` |
| `GET` | `/auth/me` | JWT — `{username, role, tenant_id}` |

### Analyze (multipart form + file)

All require JWT when auth is enabled. Upload CSV/Parquet (`file`).

#### `POST /analyze/control-chart`

Form fields: `value_col`, `subgroup_col`, `sample_size_col`, `opportunity_col`, `chart_type`, `ruleset`, `valid_range_min` / `valid_range_max`, optional `msa_file` + `msa_tolerance`, `include_records` (bool). Identity comes from the JWT, not a spoofable Form `user_id`.

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

Form: `study_type`, `method` (`anova`\|`range`), `tolerance`, `part_col`, `operator_col`, `measurement_col`, `trial_col`, `reference_col`.

#### `POST /analyze/msa-continuous`

JSON body for streaming MSA drift (`spc_core.msa_stream`). JWT. Used by the MSA page “Continuous MSA” panel.

#### `POST /analyze/explain`

JSON: one `Signal` (or list) plus optional `limits_version` / limits snapshot. Returns structured `operator_summary` from [`spc_core.explain`](../spc_core/explain.py) — no LLM.

#### `POST /analyze/counterfactual`

Sandbox re-`establish` on posted values without mutating stored frozen limits.

### Onboarding, Lab, ops

| Method | Path | Auth | Notes |
|--------|------|------|-------|
| `GET` | `/onboarding/sample` | JWT | Query `dataset=` — CSV body from `sample_data` |
| `POST` | `/onboarding/demo-stream` | JWT + API key | Register + go-live helper for the wizard |
| `GET` | `/lab/cases` | JWT | Resilience catalog ids |
| `POST` | `/lab/cases/{case_id}/run` | JWT (analyst/admin) | Run one judgment case |
| `GET` | `/ops/summary` | JWT | Compact ops counts for Overview |
| `GET` | `/limits/{version_a}/diff/{version_b}` | JWT | Limit-version diff |
| `GET` | `/runs/{run_id}/export.xlsx` | JWT | Excel export of a stored run |

Compose API sets `PYTHONPATH=/app` so `/lab/cases` can import `resilience_data`.

### History

| Method | Path | Notes |
|--------|------|-------|
| `GET` | `/runs` | Query: `analysis_type`, `limit` (≤500) |
| `GET` | `/runs/{run_id}` | Full stored run |
| `GET` | `/reports/{run_id}` | HTML if generated, else JSON fallback (JWT) |

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

Subscribes to Redis channel `spc:live:{stream_key}` and forwards JSON messages. Requires `redis` package and a reachable `ASPC_REDIS_URL`.

When auth is enabled, after the handshake the client must send a first text frame:

```json
{"type": "auth", "token": "<jwt>"}
```

Do **not** put the JWT in the query string. First server message after auth: `{"event":"subscribed","channel":"…"}`.

### SSE replay

`GET /stream/replay?file_path=…&value_col=measurement&limits_version=…`

Server-side CSV replay against frozen limits; emits SSE `data:` lines of `Signal` JSON, then `{"event":"done"}`.

## CORS

Configured via `api.cors_origins` / `ASPC_CORS_ORIGINS` (comma-separated). Compose `.env.example` allows `http://localhost:3000`, `http://127.0.0.1:3000`, and the same hosts on `:3001` (Next falls back to 3001 when 3000 is taken).

The operator UI talks to the API via same-origin **`/backend/*`** (Next.js rewrite to the API). Browsers on `:3000` do not need a cross-origin call to `:8000`. Direct `curl` to `:8000` still works.

See [configuration.md](configuration.md) and [deployment.md](deployment.md).
