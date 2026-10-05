# Clones GPT-SoVITS at a pinned commit and downloads its pretrained checkpoints
# (~5.2GB). Safe to re-run. Usage: powershell -File scripts/vendor_gpt_sovits.ps1
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VendorDir = Join-Path $RepoRoot "vendor\GPT-SoVITS"
$PinnedCommit = "48b1a0169a28582a8984402f82cf438d3bfa6aca"

if (-not (Test-Path $VendorDir)) {
    Write-Host "[vendor_gpt_sovits] Cloning GPT-SoVITS into $VendorDir"
    git clone https://github.com/RVC-Boss/GPT-SoVITS.git $VendorDir
}

Set-Location $VendorDir
Write-Host "[vendor_gpt_sovits] Checking out pinned commit $PinnedCommit"
Write-Host "[vendor_gpt_sovits] (pinned, not latest main - our adapter's script names/args"
Write-Host "[vendor_gpt_sovits] were confirmed against this exact commit; moving ahead risks drift)"
git fetch origin
git checkout $PinnedCommit

Write-Host "[vendor_gpt_sovits] Running GPT-SoVITS's own installer."
Write-Host "[vendor_gpt_sovits] WARNING: this installs GPT-SoVITS's Python dependencies into"
Write-Host "[vendor_gpt_sovits] your currently-active environment (pytorch_lightning, peft,"
Write-Host "[vendor_gpt_sovits] funasr, etc.) and may touch already-installed packages,"
Write-Host "[vendor_gpt_sovits] possibly including torch itself. If anything about your"
Write-Host "[vendor_gpt_sovits] existing setup changes unexpectedly, report the exact output."
powershell -File install.ps1 -Device CU126 -Source HF

Write-Host ""
Write-Host "[vendor_gpt_sovits] Done. Verify the pretrained checkpoints landed:"
Write-Host "  ls $VendorDir\GPT_SoVITS\pretrained_models"
