#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PIP_INSTALL_LOG="${ROOT_DIR}/.venv/start_app_pip_install.log"

run_pip_install_requirements() {
  mkdir -p "$(dirname "$PIP_INSTALL_LOG")"

  if python -m pip install -q -r "requirements.txt" >"$PIP_INSTALL_LOG" 2>&1; then
    return 0
  fi

  echo "Dependency installation failed. Last pip log lines:"
  tail -n 40 "$PIP_INSTALL_LOG" || true
  echo "Full pip log: $PIP_INSTALL_LOG"
  return 1
}

repair_macos_hidden_python_metadata() {
  if [ "$(uname -s)" != "Darwin" ] || [ ! -d ".venv" ]; then
    return
  fi

  chflags nohidden ".venv" ".venv/lib" >/dev/null 2>&1 || true
  local python_dir
  for python_dir in .venv/lib/python*; do
    [ -d "$python_dir" ] || continue
    local site_packages_dir="${python_dir}/site-packages"
    chflags nohidden "$python_dir" "$site_packages_dir" >/dev/null 2>&1 || true
    if [ -d "$site_packages_dir" ] && compgen -G "${site_packages_dir}/*.pth" >/dev/null; then
      chflags nohidden "$site_packages_dir"/*.pth >/dev/null 2>&1 || true
    fi
  done
}

create_env() {
  local py_cmd=""
  if command -v python3 >/dev/null 2>&1; then
    py_cmd="python3"
  elif command -v python >/dev/null 2>&1; then
    py_cmd="python"
  else
    echo "Python not found. Install Python 3 to continue."
    exit 1
  fi

  echo "Creating virtual environment in .venv..."
  "$py_cmd" -m venv ".venv"
  # shellcheck disable=SC1091
  source ".venv/bin/activate"

  if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    run_pip_install_requirements
    repair_macos_hidden_python_metadata
  else
    echo "requirements.txt not found. Cannot install dependencies."
    exit 1
  fi
}

venv_is_stale() {
  if [ ! -d ".venv" ] || [ ! -f "requirements.txt" ]; then
    return 0
  fi

  .venv/bin/python - <<'PY'
from pathlib import Path

venv = Path(".venv")
requirements = Path("requirements.txt")
raise SystemExit(0 if venv.stat().st_mtime < requirements.stat().st_mtime else 1)
PY
}

dask_runtime_audit() {
  local output=""
  if output="$(
    .venv/bin/python - <<'PY' 2>&1
import importlib
import json

missing = []
for module_name in (
    "cluster_app",
    "distributed",
    "dask",
    "fastapi",
    "uvicorn",
    "zeroconf",
    "cryptography",
):
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        missing.append({"module": module_name, "error": f"{type(exc).__name__}: {exc}"})

if missing:
    print(json.dumps({"missing": missing}, ensure_ascii=False))
    raise SystemExit(1)

print("dask_runtime_ok")
PY
  )"; then
    DASK_RUNTIME_AUDIT_OUTPUT="$output"
    return 0
  fi

  DASK_RUNTIME_AUDIT_OUTPUT="$output"
  return 1
}

repair_dask_runtime_if_needed() {
  local needs_repair=0

  repair_macos_hidden_python_metadata

  if venv_is_stale; then
    needs_repair=1
  fi

  if ! dask_runtime_audit; then
    needs_repair=1
  fi

  if [ "$needs_repair" -eq 0 ]; then
    return
  fi

  if [ ! -f "requirements.txt" ]; then
    echo "requirements.txt not found. Cannot repair Dask runtime."
    exit 1
  fi

  echo "Preparing local Python environment..."
  run_pip_install_requirements
  repair_macos_hidden_python_metadata
  touch ".venv"

  if ! dask_runtime_audit; then
    echo "Dask runtime audit failed after reinstall. Streamlit will not start."
    if [ -n "${DASK_RUNTIME_AUDIT_OUTPUT:-}" ]; then
      echo "$DASK_RUNTIME_AUDIT_OUTPUT"
    fi
    echo "Run 'pip install -r requirements.txt' manually and verify the missing Dask extras."
    exit 1
  fi
}

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
else
  create_env
fi

repair_dask_runtime_if_needed

load_env_file() {
  local env_file="$1"
  if [ -f "$env_file" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$env_file"
    set +a
  fi
}

load_env_file ".env"
load_env_file ".env.local"

setup_python_warning_filters() {
  local sklearn_parallel_filter="ignore::UserWarning:sklearn.utils.parallel"
  if [ -n "${PYTHONWARNINGS:-}" ]; then
    export PYTHONWARNINGS="${sklearn_parallel_filter},${PYTHONWARNINGS}"
  else
    export PYTHONWARNINGS="${sklearn_parallel_filter}"
  fi
}

setup_python_warning_filters

setup_crash_reporter_suppression() {
  if [ "$(uname -s)" != "Darwin" ]; then
    return
  fi
  if [ "${SUMO_SUPPRESS_CRASH_DIALOGS:-1}" != "1" ]; then
    return
  fi

  if CRASH_DIALOGTYPE_PREV="$(defaults read com.apple.CrashReporter DialogType 2>/dev/null)"; then
    CRASH_DIALOGTYPE_WAS_SET=1
  else
    CRASH_DIALOGTYPE_PREV=""
    CRASH_DIALOGTYPE_WAS_SET=0
  fi
  defaults write com.apple.CrashReporter DialogType none >/dev/null 2>&1 || true

  REPORTCRASH_AGENT_PLIST="/System/Library/LaunchAgents/com.apple.ReportCrash.plist"
  REPORTCRASH_AGENT_SVC="gui/$(id -u)/com.apple.ReportCrash"
  if launchctl print "$REPORTCRASH_AGENT_SVC" >/dev/null 2>&1; then
    REPORTCRASH_AGENT_WAS_PRESENT=1
  else
    REPORTCRASH_AGENT_WAS_PRESENT=0
  fi
  if [ "$REPORTCRASH_AGENT_WAS_PRESENT" -eq 1 ]; then
    launchctl unload -w "$REPORTCRASH_AGENT_PLIST" >/dev/null 2>&1 || true
  fi
}

restore_crash_reporter_setting() {
  if [ "$(uname -s)" != "Darwin" ]; then
    return
  fi
  if [ "${SUMO_SUPPRESS_CRASH_DIALOGS:-1}" != "1" ]; then
    return
  fi

  if [ "${CRASH_DIALOGTYPE_WAS_SET:-0}" -eq 1 ]; then
    defaults write com.apple.CrashReporter DialogType "$CRASH_DIALOGTYPE_PREV" >/dev/null 2>&1 || true
  else
    defaults delete com.apple.CrashReporter DialogType >/dev/null 2>&1 || true
  fi

  if [ "${REPORTCRASH_AGENT_WAS_PRESENT:-0}" -eq 1 ]; then
    launchctl load -w "/System/Library/LaunchAgents/com.apple.ReportCrash.plist" >/dev/null 2>&1 || true
  fi
}

setup_crash_reporter_suppression
trap restore_crash_reporter_setting EXIT INT TERM

streamlit run streamlit_main.py
