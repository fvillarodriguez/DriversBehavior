#!/bin/bash
# Bootstrap the Tesis Python environment on Rockfish with CUDA-enabled PyTorch + PyG.
#
# Run this ONCE inside an interactive GPU job (login nodes prohibit heavy installs):
#   srun --partition=ica100 --account=<PI>_gpu --qos=qos_gpu \
#        --gres=gpu:1 --cpus-per-task=4 --time=01:00:00 --pty bash
#   cd ~/Tesis && bash scripts/slurm/setup_rockfish_env.sh
#
# Idempotent: re-running upgrades pip and re-checks every wheel.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Tesis}"
VENV_DIR="${VENV_DIR:-$REPO_DIR/venv_gpu}"
TORCH_VERSION="${TORCH_VERSION:-2.4.1}"
CUDA_TAG="${CUDA_TAG:-cu121}"
PYG_VERSION="${PYG_VERSION:-2.7.0}"
PYG_LIB_VERSION="${PYG_LIB_VERSION:-0.4.0}"
PYG_TORCH_TAG="${PYG_TORCH_TAG:-torch-2.4.0+${CUDA_TAG}}"
PYTHON_MODULE="${PYTHON_MODULE:-anaconda}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.1}"

echo ">> Loading modules: $PYTHON_MODULE  $CUDA_MODULE"
module purge
module load "$PYTHON_MODULE"
module load "$CUDA_MODULE"

if [ ! -d "$REPO_DIR" ]; then
    echo "ERROR: repo dir $REPO_DIR does not exist. rsync the project first." >&2
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo ">> Creating venv at $VENV_DIR"
    python -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools

echo ">> Installing PyTorch ${TORCH_VERSION} + ${CUDA_TAG}"
pip install --no-cache-dir \
    "torch==${TORCH_VERSION}" \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo ">> Installing torch-geometric ${PYG_VERSION}"
pip install --no-cache-dir "torch-geometric==${PYG_VERSION}"

echo ">> Installing PyG CUDA wheels (pyg-lib, scatter, sparse, cluster, spline-conv)"
pip install --no-cache-dir \
    "pyg-lib==${PYG_LIB_VERSION}" \
    "torch-scatter==2.1.2" \
    "torch-sparse==0.6.18" \
    "torch-cluster==1.6.3" \
    "torch-spline-conv==1.2.2" \
    -f "https://data.pyg.org/whl/${PYG_TORCH_TAG}.html"

# Install the rest of requirements.txt minus the torch/PyG lines (already pinned).
REQ_FILE="$REPO_DIR/requirements.txt"
if [ -f "$REQ_FILE" ]; then
    echo ">> Installing remaining requirements (filtered)"
    FILTERED="$(mktemp)"
    grep -vE '^(torch(==|$)|torch-(scatter|sparse|cluster|spline-conv)|torch-geometric|pyg-lib|-f https://data\.pyg\.org)' "$REQ_FILE" \
        | grep -vE '^\s*#' \
        | grep -vE '^\s*$' \
        > "$FILTERED" || true
    pip install --no-cache-dir -r "$FILTERED" || {
        echo "!! Some requirements failed. Inspect $FILTERED and re-run pip manually." >&2
    }
    rm -f "$FILTERED"
fi

echo ">> Verifying torch + CUDA"
python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

echo ">> Setup complete. Activate with: source $VENV_DIR/bin/activate"
