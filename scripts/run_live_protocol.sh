#!/usr/bin/env bash
# Bring up Compose (if needed) and run the manufacturing live protocol.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_FILE="$ROOT/deploy/compose/docker-compose.yml"
cd "$ROOT"

echo "==> Docker Compose up (build)"
docker compose -f "$COMPOSE_FILE" up -d --build

echo "==> Waiting for service health"
for i in $(seq 1 90); do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo "API is up"
    break
  fi
  if [[ "$i" -eq 90 ]]; then
    echo "API failed to become healthy" >&2
    docker compose -f "$COMPOSE_FILE" ps
    docker compose -f "$COMPOSE_FILE" logs --tail=80 api stream-engine mqtt-bridge || true
    exit 1
  fi
  sleep 2
done

# Host-side deps for simulator + redis verify
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi
python3 -c "import paho.mqtt.client" 2>/dev/null || uv pip install paho-mqtt
python3 -c "import redis" 2>/dev/null || uv pip install redis

export ASPC_PROTOCOL_API="${ASPC_PROTOCOL_API:-http://localhost:8000}"
export ASPC_PROTOCOL_API_KEY="${ASPC_PROTOCOL_API_KEY:-demokey}"
export ASPC_PROTOCOL_PASSWORD="${ASPC_PROTOCOL_PASSWORD:-admin}"
export ASPC_PROTOCOL_STREAM="${ASPC_PROTOCOL_STREAM:-line-1}"
export ASPC_PROTOCOL_MQTT_HOST="${ASPC_PROTOCOL_MQTT_HOST:-localhost}"
export ASPC_PROTOCOL_REDIS_URL="${ASPC_PROTOCOL_REDIS_URL:-redis://localhost:6379/0}"

echo "==> Running live protocol"
python3 "$ROOT/scripts/run_live_protocol.py"
status=$?

echo ""
echo "============================================================"
echo " Stack left running for you to watch:"
echo "   Dashboard:  http://localhost:3000/login  (password: admin)"
echo "   Live page:  http://localhost:3000/live   → Connect 'line-1'"
echo "   API docs:   http://localhost:8000/docs"
echo "   Report:     $ROOT/var/reports/live_protocol_result.md"
echo " Re-sim:       python scripts/manufacturing_sim.py --host localhost --stream-key line-1"
echo "============================================================"
exit "$status"
