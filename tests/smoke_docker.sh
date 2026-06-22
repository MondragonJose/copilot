#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────
# Smoke test for Research Copilot Docker stack.
# Waits for all services (db, redis, grobid, api, frontend)
# to report healthy.  Exits 0 on success, 1 on timeout/failure.
# ────────────────────────────────────────────────────────────────────
set -euo pipefail

MAX_WAIT_SECONDS=180
POLL_INTERVAL=5
SERVICES=("db" "redis" "grobid" "api" "frontend")

echo "=== Smoke: Research Copilot Docker stack ==="
echo "Waiting up to ${MAX_WAIT_SECONDS}s for services: ${SERVICES[*]}"
echo

# Resolve container names
COMPOSE_CONTAINERS=()
for SERVICE in "${SERVICES[@]}"; do
    NAME=$(docker compose ps --format json "$SERVICE" 2>/dev/null \
      | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['Name'])" 2>/dev/null || echo "")
    if [ -z "$NAME" ]; then
        NAME="${PWD##*/}-${SERVICE}-1"
    fi
    COMPOSE_CONTAINERS+=("$NAME")
done

ELAPSED=0
ALL_HEALTHY=false

while [ $ELAPSED -lt $MAX_WAIT_SECONDS ]; do
    ALL_HEALTHY=true
    for i in "${!SERVICES[@]}"; do
        SERVICE="${SERVICES[$i]}"
        CONTAINER="${COMPOSE_CONTAINERS[$i]}"
        STATUS=$(docker inspect --format '{{json .State.Health.Status}}' "$CONTAINER" 2>/dev/null || echo "\"unhealthy\"")
        STATUS=${STATUS//\"/}
        if [ "$STATUS" != "healthy" ]; then
            ALL_HEALTHY=false
            echo "  [WAIT] ${SERVICE} = ${STATUS}"
        fi
    done

    if [ "$ALL_HEALTHY" = true ]; then
        echo
        echo "All services healthy after ${ELAPSED}s."
        echo
        echo "  Frontend : http://localhost:5173"
        echo "  API      : http://localhost:8000/docs"
        echo
        exit 0
    fi

    sleep "$POLL_INTERVAL"
    ELAPSED=$((ELAPSED + POLL_INTERVAL))
done

echo
echo "TIMEOUT: Not all services healthy within ${MAX_WAIT_SECONDS}s."
echo "Run 'docker ps' and 'docker inspect <container>' for details."
exit 1
