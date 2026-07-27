# Slim deploy: frontend + API only (Vercel)

This folder is **additive** — it does not change Compose or the streaming stack.

**Only two things:** UI (like `:3000`) and API (like `:8000`). No MQTT/Kafka/Redis/DB service.

| Included | Not included |
|----------|----------------|
| Next.js UI (`frontend/`) | Mosquitto / Kafka / Redis |
| FastAPI SPC core (Analyze, MSA, capability, auth) | stream-engine, Live + sim |
| SQLite inside the API (`/tmp/aspc.db`, ephemeral) | |

## Preferred: container image (`Dockerfile.vercel`)

Native FastAPI serverless packaging pulls in numpy/scipy and exceeds Vercel’s default ~225 MB function limit. The root [`Dockerfile.vercel`](../../Dockerfile.vercel) deploys the API as a container function instead.

### 1. API project (`aspc-api` / …)

1. Framework: **Other** (or leave auto-detect — root `Dockerfile.vercel` is enough)
2. Root Directory: **`.`** (repo root)
3. Env:

```text
ASPC_AUTH_ENABLED=true
ASPC_JWT_SECRET=some-long-random-string
ASPC_ADMIN_USERNAME=admin
ASPC_ADMIN_PASSWORD=admin
ASPC_API_KEYS=demokey
ASPC_PERSISTENCE_BACKEND=sqlite
ASPC_SQLITE_PATH=/tmp/aspc.db
ASPC_CORS_ORIGINS=https://aspc-web.vercel.app
ASPC_DEV_INSECURE=1
VERCEL_SUPPORT_LARGE_FUNCTIONS=1
```

`ASPC_DEV_INSECURE=1` is required if you use password `admin` or a default JWT secret (startup refuses those otherwise). Demo only.

`VERCEL_SUPPORT_LARGE_FUNCTIONS=1` opts existing projects into Fluid large functions (needed for scientific Python deps).

4. Redeploy from latest `aspc-refactor`.
5. Check:

```bash
curl -sS https://YOUR-API.vercel.app/health
curl -sS https://YOUR-API.vercel.app/
```

Expect JSON, not `NOT_FOUND`.

### 2. Frontend project (`aspc-web`)

1. Root Directory: **`frontend`**
2. Framework: **Next.js**
3. Env (must match the **working** API URL):

```text
NEXT_PUBLIC_API_URL=https://YOUR-API.vercel.app
NEXT_PUBLIC_WS_URL=wss://YOUR-API.vercel.app
```

4. Redeploy frontend after changing these (baked at build time).

### 3. Login

Use `ASPC_ADMIN_USERNAME` / `ASPC_ADMIN_PASSWORD` from the API env (e.g. `admin` / `admin` if set as above).

## Local slim image

```bash
docker build -f Dockerfile.vercel -t aspc-api:vercel .
# or: docker build -f deploy/vercel/Dockerfile.api -t aspc-api:slim .
```

## Brother demo choice

| Goal | Use |
|------|-----|
| Analyze in the cloud | This slim Vercel path |
| Live + manufacturing sim | Compose on your PC + ngrok |
