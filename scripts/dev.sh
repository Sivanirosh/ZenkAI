#!/usr/bin/env bash
# One-command dev launcher for ZenkAI.
#
# Starts the FastAPI backend (which also serves the static frontend),
# streams the log into this terminal, opens the browser once the app
# answers, and kills the child on Ctrl+C or when the terminal closes.
#
# Usage:  ./scripts/dev.sh
#
# Env overrides:
#   LM_HOST        (default 127.0.0.1)  backend bind host
#   LM_PORT        (default 8000)       app port
#   LM_NO_BROWSER  (default 0)          set to 1 to skip auto-open
#   LM_CONDA_ENV   (default medvlm-base) conda env for the backend

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

BACKEND_HOST="${LM_HOST:-127.0.0.1}"
BACKEND_PORT="${LM_PORT:-8000}"
CONDA_ENV="${LM_CONDA_ENV:-medvlm-base}"

C_DEV='\033[1;36m'; C_WARN='\033[1;33m'; C_API='\033[34m'; C_OFF='\033[0m'

log()  { printf "${C_DEV}[dev]${C_OFF} %s\n"  "$*" >&2; }
warn() { printf "${C_WARN}[dev]${C_OFF} %s\n" "$*" >&2; }

# ── Conda (for the backend) ────────────────────────────────────────────

if [ -z "${CONDA_EXE:-}" ] && [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  . "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
if [ -n "${CONDA_EXE:-}" ] && conda env list 2>/dev/null | grep -q "^${CONDA_ENV} "; then
  log "Activating conda env ${CONDA_ENV}"
  conda activate "$CONDA_ENV"
else
  log "Conda env '${CONDA_ENV}' not found — using current environment"
fi

if ! command -v uvicorn >/dev/null 2>&1; then
  warn "uvicorn not found. Install deps with:"
  warn "  uv pip install -e \".[dev]\""
  exit 1
fi

# ── Start backend ──────────────────────────────────────────────────────

BACKEND_PID=""

cleanup() {
  [ -n "$BACKEND_PID" ] && kill "$BACKEND_PID" 2>/dev/null
}
trap cleanup EXIT INT TERM

log "Starting backend on http://${BACKEND_HOST}:${BACKEND_PORT}"
uvicorn backend.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" \
  --reload-exclude 'data/*' --reload-exclude '*.duckdb*' \
  --timeout-keep-alive 120 &
BACKEND_PID=$!

# ── Wait for the app, then open the browser ────────────────────────────

if [ "${LM_NO_BROWSER:-0}" != "1" ]; then
  (
    URL="http://${BACKEND_HOST}:${BACKEND_PORT}"
    if [ "$BACKEND_HOST" = "0.0.0.0" ]; then URL="http://localhost:${BACKEND_PORT}"; fi
    for _ in $(seq 1 60); do
      curl -sf -o /dev/null "$URL/health" && break
      sleep 0.5
    done
    sleep 0.5
    command -v xdg-open >/dev/null 2>&1 && xdg-open "$URL" >/dev/null 2>&1
  ) &
fi

log "Ctrl+C to stop"
wait
