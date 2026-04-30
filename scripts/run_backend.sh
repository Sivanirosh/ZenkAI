#!/usr/bin/env bash
# Run the ZenkAI FastAPI backend with dev-friendly defaults.
#
# Why the extra flags:
#   --timeout-keep-alive 120
#       Uvicorn's default of 5 s closes idle HTTP keep-alive sockets before
#       the Next.js dev rewrite (undici) releases them from its pool. The
#       next reused socket then dies with ECONNRESET and the browser shows
#       "TypeError: Failed to fetch". 120 s is comfortably larger than
#       undici's default 4 s keep-alive.
#
#   --reload-exclude 'data/*'
#       The DuckDB file lives under data/ and is written every time a user
#       marks a word seen/opened. Without the exclusion, uvicorn's watcher
#       respawns the server mid-request, producing the same symptom.

set -euo pipefail

cd "$(dirname "$0")/.."

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

exec uvicorn backend.main:app \
  --host "$HOST" \
  --port "$PORT" \
  --reload \
  --reload-exclude 'data/*' \
  --reload-exclude 'frontend/*' \
  --reload-exclude '*.duckdb*' \
  --timeout-keep-alive 120 \
  "$@"
