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
    read -n 1 -s -r -p "Press any key to exit..."
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
    read -n 1 -s -r -p "Press any key to exit..."
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

echo "Ejecutando sincronización con GitHub..."
python src/git_sync.py

echo ""
echo "Operación finalizada."
read -n 1 -s -r -p "Presiona cualquier tecla para cerrar esta ventana..."
