#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

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
    pip install -r "requirements.txt"
  else
    echo "requirements.txt not found. Cannot install dependencies."
    exit 1
  fi
}

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
elif [ -f "venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "venv/bin/activate"
else
  create_env
fi

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
