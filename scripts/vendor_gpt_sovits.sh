#!/usr/bin/env bash
# Clones GPT-SoVITS at a pinned commit and downloads its pretrained checkpoints
# (~5.2GB). Safe to re-run. Usage: bash scripts/vendor_gpt_sovits.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$REPO_ROOT/vendor/GPT-SoVITS"
PINNED_COMMIT="48b1a0169a28582a8984402f82cf438d3bfa6aca"

if [ ! -d "$VENDOR_DIR" ]; then
    echo "[vendor_gpt_sovits] Cloning GPT-SoVITS into $VENDOR_DIR"
    git clone https://github.com/RVC-Boss/GPT-SoVITS.git "$VENDOR_DIR"
fi

cd "$VENDOR_DIR"
echo "[vendor_gpt_sovits] Checking out pinned commit $PINNED_COMMIT"
echo "[vendor_gpt_sovits] (pinned, not latest main - our adapter's script names/args were"
echo "[vendor_gpt_sovits] confirmed against this exact commit; moving ahead risks drift)"
git fetch origin
git checkout "$PINNED_COMMIT"

echo "[vendor_gpt_sovits] Running GPT-SoVITS's own installer."
echo "[vendor_gpt_sovits] WARNING: this installs GPT-SoVITS's Python dependencies into"
echo "[vendor_gpt_sovits] your currently-active environment (pytorch_lightning, peft,"
echo "[vendor_gpt_sovits] funasr, etc.) and may touch already-installed packages,"
echo "[vendor_gpt_sovits] possibly including torch itself. If anything about your existing"
echo "[vendor_gpt_sovits] setup changes unexpectedly, report the exact output."
bash install.sh --device CU126 --source HF

echo ""
echo "[vendor_gpt_sovits] Done. Verify the pretrained checkpoints landed:"
echo "  ls $VENDOR_DIR/GPT_SoVITS/pretrained_models"
