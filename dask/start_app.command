#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

APP_HOST="${CLUSTER_APP_HOST:-0.0.0.0}"
APP_PORT="${CLUSTER_APP_PORT:-18080}"
APP_OPEN_HOST="${CLUSTER_APP_OPEN_HOST:-127.0.0.1}"
APP_URL="http://${APP_OPEN_HOST}:${APP_PORT}"
CONFIG_FILE="${CLUSTER_APP_CONFIG:-config.yaml}"
VENV_DIR="${CLUSTER_APP_VENV:-.venv}"
VENV_PYTHON="${VENV_DIR}/bin/python"
INSTALL_STAMP="${VENV_DIR}/.cluster_app_install_stamp"

log() {
  printf "\n[%s] %s\n" "$(date '+%H:%M:%S')" "$*"
}

fail() {
  printf "\nERROR: %s\n" "$*" >&2
  printf "Press Enter to close this window..."
  read -r _
  exit 1
}

is_compatible_python() {
  "$1" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if (3, 12) <= sys.version_info[:2] < (3, 14) else 1)
PY
}

find_python() {
  if [ -n "${PYTHON:-}" ] && command -v "$PYTHON" >/dev/null 2>&1 && is_compatible_python "$PYTHON"; then
    printf "%s" "$PYTHON"
    return 0
  fi

  for candidate in python3.12 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && is_compatible_python "$candidate"; then
      printf "%s" "$candidate"
      return 0
    fi
  done

  return 1
}

app_is_running() {
  curl -fsS --max-time 2 "${APP_URL}/api/metrics/status" >/dev/null 2>&1
}

wait_for_app() {
  for _ in $(seq 1 60); do
    if app_is_running; then
      return 0
    fi
    if ! kill -0 "$1" >/dev/null 2>&1; then
      return 1
    fi
    sleep 1
  done
  return 1
}

log "Dask Cluster App launcher"
log "Project: ${ROOT_DIR}"
log "Bind: ${APP_HOST}:${APP_PORT}"
log "URL: ${APP_URL}"

if app_is_running; then
  log "Application is already running."
  echo ""
  echo "  [1] Open in browser"
  echo "  [2] Restart"
  echo "  [3] Stop & close"
  echo "  [4] Cancel"
  echo ""
  printf "Choose an option (1-4): "
  read -r choice
  case "$choice" in
    1)
      open "$APP_URL" >/dev/null 2>&1 || true
      exit 0
      ;;
    2)
      log "Stopping current instance..."
      PID=$(curl -fsS --max-time 3 "${APP_URL}/api/admin/scheduler/stop" >/dev/null 2>&1; lsof -ti "TCP:${APP_PORT}" 2>/dev/null || true)
      if [ -n "$PID" ]; then
        kill "$PID" 2>/dev/null || true
        for _ in $(seq 1 30); do
          if ! kill -0 "$PID" 2>/dev/null; then
            break
          fi
          sleep 1
        done
      fi
      log "Restarting..."
      ;;
    3)
      log "Stopping scheduler..."
      curl -fsS --max-time 3 "${APP_URL}/api/admin/scheduler/stop" >/dev/null 2>&1 || true
      PID=$(lsof -ti "TCP:${APP_PORT}" 2>/dev/null || true)
      if [ -n "$PID" ]; then
        log "Stopping application (PID ${PID})..."
        kill "$PID" 2>/dev/null || true
      fi
      log "Application stopped."
      exit 0
      ;;
    *)
      exit 0
      ;;
  esac
fi

if [ ! -x "$VENV_PYTHON" ]; then
  PYTHON_BIN="$(find_python)" || fail "Python 3.12 or 3.13 is required. Install Python 3.12 and run this file again."
  log "Creating virtual environment with ${PYTHON_BIN}"
  "$PYTHON_BIN" -m venv "$VENV_DIR" || fail "Could not create virtual environment."
fi

if ! is_compatible_python "$VENV_PYTHON"; then
  fail "Existing ${VENV_DIR} is not Python 3.12/3.13. Remove it or set CLUSTER_APP_VENV to another folder."
fi

NEEDS_INSTALL=0
if [ ! -f "$INSTALL_STAMP" ]; then
  NEEDS_INSTALL=1
elif [ "pyproject.toml" -nt "$INSTALL_STAMP" ]; then
  NEEDS_INSTALL=1
elif ! "$VENV_PYTHON" - <<'PY' >/dev/null 2>&1; then
import cluster_app, fastapi, distributed, uvicorn, zeroconf, cryptography
PY
  NEEDS_INSTALL=1
fi

if [ "$NEEDS_INSTALL" = "1" ]; then
  log "Installing or updating dependencies"
  "$VENV_PYTHON" -m pip install --upgrade pip || fail "Could not upgrade pip."
  "$VENV_PYTHON" -m pip install -e . || fail "Could not install project dependencies."
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "$INSTALL_STAMP"
else
  log "Virtual environment is ready"
fi

if [ ! -f "$CONFIG_FILE" ]; then
  log "Creating default ${CONFIG_FILE}"
  "$VENV_PYTHON" -m cluster_app.main --config "$CONFIG_FILE" init || fail "Could not create config file."
fi

log "Starting application"
"$VENV_PYTHON" -m cluster_app.main --config "$CONFIG_FILE" start --host "$APP_HOST" --port "$APP_PORT" &
APP_PID=$!

if wait_for_app "$APP_PID"; then
  log "Application started at ${APP_URL}"
  open "$APP_URL" >/dev/null 2>&1 || true
  wait "$APP_PID"
else
  fail "Application did not start. Check the terminal output above."
fi