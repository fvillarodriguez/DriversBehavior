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

exec streamlit run streamlit_main.py
