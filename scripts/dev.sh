#!/usr/bin/env bash
# One-command dev launcher for LinguaMate.
#
# Starts the FastAPI backend and the Next.js frontend side-by-side,
# streams both logs into this terminal with `[api]` / `[web]` prefixes,
# opens the browser once the frontend is reachable, and kills every
# child (including Next.js workers) on Ctrl+C or when the terminal
# closes.
#
# Usage:  ./scripts/dev.sh
#
# Env overrides:
#   LM_HOST        (default 127.0.0.1)   backend bind host
#   LM_PORT        (default 8000)        backend port
#   LM_FRONT_PORT  (default 3000)        Next.js port
#   LM_NO_BROWSER  (default 0)           set to 1 to skip auto-open
#   LM_CONDA_ENV   (default medvlm-base) conda env for the backend
#
# We deliberately avoid `set -e` / `set -o pipefail`: those interact
# badly with the `| sed` log pipelines and can suppress trap handlers
# on signal. Each command below is either idempotent or checks its
# own return status.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

BACKEND_HOST="${LM_HOST:-127.0.0.1}"
BACKEND_PORT="${LM_PORT:-8000}"
FRONTEND_PORT="${LM_FRONT_PORT:-3000}"
CONDA_ENV="${LM_CONDA_ENV:-medvlm-base}"

C_DEV='\033[1;36m'; C_WARN='\033[1;33m'; C_API='\033[34m'; C_WEB='\033[35m'; C_OFF='\033[0m'

log()  { printf "${C_DEV}[dev]${C_OFF} %s\n"  "$*" >&2; }
warn() { printf "${C_WARN}[dev]${C_OFF} %s\n" "$*" >&2; }

# ── Conda (for the backend) ────────────────────────────────────────────

if [ -z "${CONDA_EXE:-}" ] && [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

if command -v conda >/dev/null 2>&1; then
  if conda activate "$CONDA_ENV" 2>/dev/null; then
    log "conda env active: $CONDA_ENV"
  else
    warn "Could not activate conda env '$CONDA_ENV'; falling back to current shell."
  fi
else
  warn "conda not found — using the system 'python' / 'uvicorn'."
fi

# ── Pre-flight checks ──────────────────────────────────────────────────

if [ ! -d frontend/node_modules ]; then
  log "frontend/node_modules missing — running 'npm install' (one-time)…"
  ( cd frontend && npm install )
fi

if command -v curl >/dev/null 2>&1; then
  if ! curl -s --max-time 1 http://localhost:11434/api/tags >/dev/null 2>&1; then
    warn "Ollama is not reachable at :11434 — chat will use the offline fallback."
    warn "  Start it in another shell with:  ollama serve"
  fi
fi

if ! command -v uvicorn >/dev/null 2>&1; then
  warn "'uvicorn' is not on PATH — backend will fail to start."
fi

# ── Clear stale listeners on our ports (e.g. after a crash) ───────────

for port in "$BACKEND_PORT" "$FRONTEND_PORT"; do
  if command -v fuser >/dev/null 2>&1; then
    if fuser -s -n tcp "$port" 2>/dev/null; then
      warn "port $port was still bound — terminating previous listener."
      fuser -k -TERM -n tcp "$port" 2>/dev/null
      sleep 1
    fi
  fi
done

# ── Cleanup (kills every descendant of this script) ───────────────────

stopped=0

kill_tree() {
  local pid="$1" sig="${2:-TERM}"
  [ -z "$pid" ] && return
  local kids
  kids=$(pgrep -P "$pid" 2>/dev/null)
  for k in $kids; do
    kill_tree "$k" "$sig"
  done
  kill "-$sig" "$pid" 2>/dev/null
}

cleanup() {
  # Re-entrancy guard: the script may receive INT and EXIT back-to-back.
  [ "$stopped" -eq 1 ] && return
  stopped=1
  trap - EXIT INT TERM HUP
  echo >&2
  log "stopping backend + frontend…"
  for pid in "$BROWSER_PID" "$FRONTEND_PID" "$BACKEND_PID"; do
    [ -n "$pid" ] && kill_tree "$pid" TERM
  done
  # Grace period — then force.
  for _ in 1 2 3 4 5 6; do
    local alive=0
    for pid in "$BACKEND_PID" "$FRONTEND_PID"; do
      [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && alive=1
    done
    [ "$alive" -eq 0 ] && break
    sleep 0.25
  done
  for pid in "$BROWSER_PID" "$FRONTEND_PID" "$BACKEND_PID"; do
    [ -n "$pid" ] && kill_tree "$pid" KILL
  done
  # Belt-and-braces: anything still listening on our ports after a crash.
  if command -v fuser >/dev/null 2>&1; then
    fuser -k -KILL -n tcp "$BACKEND_PORT" "$FRONTEND_PORT" 2>/dev/null
  fi
  log "stopped."
}

# EXIT trap runs on normal exit AND when the script is terminated by a
# signal handler that ends with `exit`. Route signals → exit to force
# the EXIT trap to run reliably even when we're blocked in `wait`.
trap 'exit 130' INT
trap 'exit 143' TERM HUP
trap cleanup EXIT

# ── Launch ─────────────────────────────────────────────────────────────

log "starting backend  → http://${BACKEND_HOST}:${BACKEND_PORT}"
log "starting frontend → http://localhost:${FRONTEND_PORT}"
[ "${LM_NO_BROWSER:-0}" = "1" ] || log "browser will open automatically when ready"

# Backend — uvicorn via run_backend.sh, prefixed.
(
  HOST="$BACKEND_HOST" PORT="$BACKEND_PORT" \
    bash scripts/run_backend.sh 2>&1 \
    | sed -u "s/^/$(printf "${C_API}[api]${C_OFF} ")/"
) &
BACKEND_PID=$!

# Frontend — call `next` directly (avoids `npm run dev` duplicating -p).
(
  cd frontend && ./node_modules/.bin/next dev -p "$FRONTEND_PORT" 2>&1 \
    | sed -u "s/^/$(printf "${C_WEB}[web]${C_OFF} ")/"
) &
FRONTEND_PID=$!

# Browser auto-open (non-blocking; exits on its own when done).
(
  [ "${LM_NO_BROWSER:-0}" = "1" ] && exit 0
  command -v xdg-open >/dev/null 2>&1 || exit 0
  url="http://localhost:${FRONTEND_PORT}"
  for _ in $(seq 1 60); do
    if curl -s --max-time 1 -o /dev/null "$url" 2>/dev/null; then
      printf "${C_DEV}[dev]${C_OFF} opening %s\n" "$url" >&2
      xdg-open "$url" >/dev/null 2>&1
      exit 0
    fi
    sleep 0.5
  done
  printf "${C_WARN}[dev]${C_OFF} frontend did not answer on %s — skipping browser open.\n" "$url" >&2
) &
BROWSER_PID=$!

# If either server dies on its own, bring the whole thing down.
wait -n "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null
