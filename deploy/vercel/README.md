# Slim deploy: frontend + API only (Vercel)

This folder is **additive** — it does not change Compose or the streaming stack.

**Only two things:** UI (like `:3000`) and API (like `:8000`). No MQTT/Kafka/Redis/DB service.

| Included | Not included |
|----------|----------------|
| Next.js UI (`frontend/`) | Mosquitto / Kafka / Redis |
| FastAPI SPC core (Analyze, MSA, capability, auth) | stream-engine, Live + sim |
| SQLite inside the API (`/tmp/aspc.db`, ephemeral) | |

## API project (`aspc` → aspc-plum.vercel.app)

1. Framework: **FastAPI**
2. Root Directory: **empty / `.`** (never set this to a Dockerfile path)
3. Install Command: `pip install -e ".[apps]"` (skip `render`/plotly — it blows the 225 MB function limit; Analyze JSON still works)
4. Env (at least):

```text
ASPC_AUTH_ENABLED=true
ASPC_JWT_SECRET=some-long-random-string
ASPC_ADMIN_USERNAME=admin
ASPC_ADMIN_PASSWORD=admin
ASPC_API_KEYS=demokey
ASPC_PERSISTENCE_BACKEND=sqlite
ASPC_SQLITE_PATH=/tmp/aspc.db
ASPC_UPLOAD_DIR=/tmp/aspc-uploads
ASPC_REPORT_DIR=/tmp/aspc-reports
ASPC_CORS_ORIGINS=https://aspc-web.vercel.app
ASPC_DEV_INSECURE=1
VERCEL_SUPPORT_LARGE_FUNCTIONS=1
```

Uploads/reports default to `/tmp/...` automatically when `VERCEL=1` is present.

`VERCEL_SUPPORT_LARGE_FUNCTIONS=1` is required so numpy/scipy fit past the default ~225 MB function limit.

Do **not** put a root `Dockerfile.vercel` unless Container Images are enabled for the team — otherwise deploys can become empty 2s “Ready” no-ops.

5. Check:

```bash
curl -sS https://aspc-plum.vercel.app/health
```

## Frontend project (`aspc-web`)

Root Directory: **`frontend`**. Env:

```text
NEXT_PUBLIC_API_URL=https://aspc-plum.vercel.app
NEXT_PUBLIC_WS_URL=wss://aspc-plum.vercel.app
```

## Local slim image (optional)

```bash
docker build -f deploy/vercel/Dockerfile.api -t aspc-api:slim .
```
