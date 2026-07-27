# Slim deploy: frontend + API only (Vercel)

This folder is **additive** — it does not change Compose, `spc_core`, or the full streaming stack.

**Only two things:** the UI (like local `:3000`) and the API (like local `:8000`). No extra DB service, no Supabase/Neon, no MQTT/Kafka.

| Included | Not included |
|----------|----------------|
| Next.js UI (`frontend/`) | Mosquitto / MQTT / Kafka / Redis |
| FastAPI SPC core (Analyze, MSA, capability, auth) | stream-engine, mqtt-bridge |
| SQLite inside the API container | Live Monitoring + manufacturing sim |

SQLite on Vercel is **ephemeral** (cold starts can wipe saved runs). Fine for a brother demo of Analyze; not for permanent history.

For Live + sim, keep using local Compose (+ ngrok). See [docs/deployment.md](../../docs/deployment.md).

## Layout

| File | Role |
|------|------|
| [`Dockerfile.api`](Dockerfile.api) | API image; build context = **repo root** |
| [`.env.example`](.env.example) | Env vars for API + frontend |

## 1. API on Vercel (Docker)

1. Push this repo to GitHub.
2. [Vercel](https://vercel.com) → **Add New Project** → import the repo.
3. Project name e.g. `aspc-api`.
4. **Root Directory:** repository root (`.`).
5. Use this Dockerfile — Vercel looks for `Dockerfile.vercel` at the repo root:

   ```bash
   # from repo root, only for this deploy (optional; do not have to commit)
   cp deploy/vercel/Dockerfile.api Dockerfile.vercel
   vercel --prod
   ```

   See [Vercel Docker docs](https://vercel.com/kb/guide/does-vercel-support-docker-deployments).

6. Set env vars from [`.env.example`](.env.example) (API section). Leave `ASPC_CORS_ORIGINS` until the frontend URL exists, then update and redeploy.

7. Deploy → note the API URL, e.g. `https://aspc-api-xxx.vercel.app`.

```bash
curl -sS https://aspc-api-xxx.vercel.app/health
```

## 2. Frontend on Vercel (Next.js)

1. **Add New Project** again → same repo.
2. Project name e.g. `aspc-web`.
3. **Root Directory:** `frontend`.
4. Env:

   ```text
   NEXT_PUBLIC_API_URL=https://aspc-api-xxx.vercel.app
   NEXT_PUBLIC_WS_URL=wss://aspc-api-xxx.vercel.app
   ```

5. Deploy → note the UI URL.
6. On **aspc-api** set `ASPC_CORS_ORIGINS=https://aspc-web-xxx.vercel.app` → redeploy API.

Open the UI → login → **Analyze**. **Live** will not work here.

## 3. Local test of the slim API image

```bash
cd /path/to/ASPC
docker build -f deploy/vercel/Dockerfile.api -t aspc-api:slim .
docker run --rm -p 8000:8000 \
  -e ASPC_DEV_INSECURE=1 \
  -e ASPC_JWT_SECRET=dev \
  -e ASPC_ADMIN_PASSWORD=admin \
  -e ASPC_API_KEYS=devkey \
  aspc-api:slim
```

## Brother demo choice

| Goal | Use |
|------|-----|
| Analyze in the cloud (UI + API only) | This slim Vercel path |
| Live + manufacturing sim | Compose on your PC + ngrok |
