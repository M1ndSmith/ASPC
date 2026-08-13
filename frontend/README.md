# ASPC Operator Console

Next.js dashboard for ASPC — batch analysis, Phase I checklist, limits go-live, and live WebSocket monitoring.

## Setup

```bash
cp .env.example .env.local
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). REST defaults to same-origin **`/backend`** (Next rewrites to `http://127.0.0.1:8000` in local dev, `http://api:8000` in Compose). WebSocket still uses `ws://localhost:8000`.

Routes: `/` Overview · `/onboarding` · `/live` · `/analyze` · `/capability` · `/msa` · `/runs` · `/lab`. Sign in at `/login`.

## Scripts

| Script | Purpose |
|--------|---------|
| `npm run dev` | Dev server |
| `npm run build` | Production build |
| `npm start` | Serve production build |
| `npm run lint` | ESLint |
| `npm test` | Vitest unit tests |
| `npm run test:e2e` | Playwright (login + all operator pages; Compose must be up) |

```bash
# UI + API already running (Compose)
E2E_USERNAME=admin E2E_PASSWORD='change-me-admin-password' npm run test:e2e
```

`home.spec.ts` skips if `:3000` is down. Other specs fail if the stack is unreachable. Auth cookies live in `e2e/.auth/` (gitignored).

## Env

- `NEXT_PUBLIC_API_URL` — REST base (default `/backend`)
- `NEXT_PUBLIC_WS_URL` — WebSocket base (default `ws://localhost:8000`)
- `ASPC_API_PROXY_TARGET` — rewrite target for `/backend` (dev: `http://127.0.0.1:8000`; Compose build: `http://api:8000`)
