#!/usr/bin/env bash
# Creates/updates the venv and installs the right torch build for this machine.
# If a conda env (or venv) is already active, installs into it directly instead
# of creating a separate .venv - activate your env first, then run this.
# Usage: bash scripts/setup_env.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -n "${CONDA_PREFIX:-}" ] || [ -n "${VIRTUAL_ENV:-}" ]; then
    USING_ACTIVE_ENV=true
    PYTHON_BIN="${PYTHON_BIN:-python}"
    ENV_LABEL="${CONDA_DEFAULT_ENV:-${VIRTUAL_ENV:-active environment}}"
    echo "[setup_env] Detected an active environment ($ENV_LABEL) - installing into it directly"
else
    USING_ACTIVE_ENV=false
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

echo "[setup_env] Using $("$PYTHON_BIN" --version 2>&1) at $(command -v "$PYTHON_BIN")"

PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "$PY_VERSION" in
    3.10|3.11) ;;
    *)
        echo "[setup_env] WARNING: detected Python $PY_VERSION."
        echo "[setup_env]          clone-voice targets 3.10 (matches GPT-SoVITS's own tested/recommended version)."
        echo "[setup_env]          Continuing anyway, but installs may fail later, especially from Batch 3 onward."
        if [ "$USING_ACTIVE_ENV" = true ]; then
            echo "[setup_env]          Fix: activate a 3.10 env first (e.g. 'conda activate <your-3.10-env>'), then re-run."
        else
            echo "[setup_env]          Fix: install Python 3.10, then re-run: PYTHON_BIN=python3.10 bash scripts/setup_env.sh"
        fi
        ;;
esac

if [ "$USING_ACTIVE_ENV" = false ]; then
    VENV_DIR="$REPO_ROOT/.venv"
    if [ ! -d "$VENV_DIR" ]; then
        echo "[setup_env] Creating virtualenv at $VENV_DIR"
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    else
        echo "[setup_env] Reusing existing virtualenv at $VENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    PYTHON_BIN="python"
fi

"$PYTHON_BIN" -m pip install --upgrade pip

echo "[setup_env] Installing base dependencies"
"$PYTHON_BIN" -m pip install -r "$REPO_ROOT/requirements-base.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[setup_env] nvidia-smi found - installing CUDA build of torch"
    echo "[setup_env] (edit requirements-cuda.txt's cu126 tag first if your driver needs a different CUDA version)"
    "$PYTHON_BIN" -m pip install -r "$REPO_ROOT/requirements-cuda.txt"
else
    echo "[setup_env] No NVIDIA GPU detected - installing CPU-only torch"
    "$PYTHON_BIN" -m pip install -r "$REPO_ROOT/requirements-cpu.txt"
fi

echo "[setup_env] Installing clone-voice in editable mode"
"$PYTHON_BIN" -m pip install -e "$REPO_ROOT"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[setup_env] WARNING: ffmpeg not found on PATH. core/audio/io.py shells out to it"
    echo "[setup_env]          to transcode input audio - install it (e.g. 'sudo apt install ffmpeg')"
    echo "[setup_env]          before running anything that loads real audio."
fi

echo ""
if [ "$USING_ACTIVE_ENV" = true ]; then
    echo "[setup_env] Done. Your active environment now has everything installed. Next steps:"
    echo "  python scripts/check_gpu.py"
    echo "  pytest"
else
    echo "[setup_env] Done. Next steps:"
    echo "  source $VENV_DIR/bin/activate"
    echo "  python scripts/check_gpu.py"
    echo "  pytest"
fi
