#!/usr/bin/env bash
# Creates/updates the venv and installs the right torch build for this machine.
# Usage: bash scripts/setup_env.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="$REPO_ROOT/.venv"

echo "[setup_env] Using $("$PYTHON_BIN" --version 2>&1) at $(command -v "$PYTHON_BIN")"

if [ ! -d "$VENV_DIR" ]; then
    echo "[setup_env] Creating virtualenv at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "[setup_env] Reusing existing virtualenv at $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --upgrade pip

echo "[setup_env] Installing base dependencies"
pip install -r "$REPO_ROOT/requirements-base.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[setup_env] nvidia-smi found - installing CUDA build of torch"
    echo "[setup_env] (edit requirements-cuda.txt's cu124 tag first if your driver needs a different CUDA version)"
    pip install -r "$REPO_ROOT/requirements-cuda.txt"
else
    echo "[setup_env] No NVIDIA GPU detected - installing CPU-only torch"
    pip install -r "$REPO_ROOT/requirements-cpu.txt"
fi

echo "[setup_env] Installing clone-voice in editable mode"
pip install -e "$REPO_ROOT"

echo ""
echo "[setup_env] Done. Next steps:"
echo "  source $VENV_DIR/bin/activate"
echo "  python scripts/check_gpu.py"
echo "  pytest"
