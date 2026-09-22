#!/usr/bin/env bash
# Run the ZenkAI FastAPI backend with dev-friendly defaults.
#
# Why the extra flags:
#   --reload-exclude 'data/*'
#       The DuckDB file lives under data/ and is written on every vocab
#       action. Without the exclusion, uvicorn's watcher respawns the
#       server mid-request.
#
#   --timeout-keep-alive 120
#       Long-lived browser connections survive idle page reading without
#       socket resets on the next fetch.

set -euo pipefail

cd "$(dirname "$0")/.."

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

exec uvicorn backend.main:app \
  --host "$HOST" \
  --port "$PORT" \
  --reload \
  --reload-exclude 'data/*' \
  --reload-exclude '*.duckdb*' \
  --timeout-keep-alive 120 \
  "$@"
