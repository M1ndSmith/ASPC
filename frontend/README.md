# ASPC Operator Console

Next.js dashboard for ASPC — batch analysis, Phase I checklist, limits go-live, and live WebSocket monitoring.

## Setup

```bash
cp .env.example .env.local
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). API defaults to `http://localhost:8000`.

## Scripts

| Script | Purpose |
|--------|---------|
| `npm run dev` | Dev server |
| `npm run build` | Production build |
| `npm start` | Serve production build |
| `npm run lint` | ESLint |
| `npm test` | Vitest unit tests |
| `npm run test:e2e` | Playwright smoke (skips if server down) |

## Env

- `NEXT_PUBLIC_API_URL` — REST base (default `http://localhost:8000`)
- `NEXT_PUBLIC_WS_URL` — WebSocket base (default `ws://localhost:8000`)
