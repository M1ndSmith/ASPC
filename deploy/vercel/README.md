# Slim deploy: frontend + API only (Vercel)

This folder is **additive** — it does not change Compose or the streaming stack.

**Only two things:** UI (like `:3000`) and API (like `:8000`). No MQTT/Kafka/Redis/DB service.

| Included | Not included |
|----------|----------------|
| Next.js UI (`frontend/`) | Mosquitto / Kafka / Redis |
| FastAPI SPC core (Analyze, MSA, capability, auth) | stream-engine, Live + sim |
| SQLite inside the API (`/tmp/aspc.db`, ephemeral) | |

## Preferred: native FastAPI (not Docker)

Vercel Framework Preset: **FastAPI** (not “Other”, not Docker).

### 0. One-time in the repo (already done if you pulled latest)

`pyproject.toml` contains:

```toml
[tool.vercel]
entrypoint = "apps.api.main:app"
```

If a root `Dockerfile.vercel` exists, **rename or delete it** for this project so Vercel does not try Docker instead:

```bash
git rm -f Dockerfile.vercel   # or: mv Dockerfile.vercel Dockerfile.vercel.bak
git push
```

For the API project, copy the install helper to the repo root (API root = `.`):

```bash
cp deploy/vercel/vercel.json ./vercel.json
git add vercel.json pyproject.toml
git commit -m "Configure Vercel FastAPI entrypoint and install extras"
git push
```

(`frontend/` is a separate Vercel project with Root Directory `frontend`, so this root `vercel.json` does not affect the UI.)

### 1. API project (`aspc-api` / `aspc-plum` / …)

1. Framework: **FastAPI**
2. Root Directory: **`.`** (repo root)
3. Install Command (if not using root `vercel.json`):  
   `pip install -e ".[apps,data,render]"`
4. Env:

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
```

`ASPC_DEV_INSECURE=1` is required if you use password `admin` or a default JWT secret (startup refuses those otherwise). Use only for a private demo.

5. Redeploy.
6. Check:

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

## Optional: Docker API image

[`Dockerfile.api`](Dockerfile.api) remains for local slim image tests or if you prefer Docker later. Native FastAPI is simpler on Vercel.

## Brother demo choice

| Goal | Use |
|------|-----|
| Analyze in the cloud | This slim Vercel path |
| Live + manufacturing sim | Compose on your PC + ngrok |
