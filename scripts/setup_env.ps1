# Creates/updates the venv and installs the right torch build for this machine.
# Usage: powershell -File scripts/setup_env.ps1
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
$VenvDir = Join-Path $RepoRoot ".venv"

Write-Host "[setup_env] Using $(& $PythonBin --version)"

if (-not (Test-Path $VenvDir)) {
    Write-Host "[setup_env] Creating virtualenv at $VenvDir"
    & $PythonBin -m venv $VenvDir
} else {
    Write-Host "[setup_env] Reusing existing virtualenv at $VenvDir"
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

& $VenvPython -m pip install --upgrade pip

Write-Host "[setup_env] Installing base dependencies"
& $VenvPython -m pip install -r (Join-Path $RepoRoot "requirements-base.txt")

$HasNvidiaGpu = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($HasNvidiaGpu) {
    Write-Host "[setup_env] nvidia-smi found - installing CUDA build of torch"
    Write-Host "[setup_env] (edit requirements-cuda.txt's cu124 tag first if your driver needs a different CUDA version)"
    & $VenvPython -m pip install -r (Join-Path $RepoRoot "requirements-cuda.txt")
} else {
    Write-Host "[setup_env] No NVIDIA GPU detected - installing CPU-only torch"
    & $VenvPython -m pip install -r (Join-Path $RepoRoot "requirements-cpu.txt")
}

Write-Host "[setup_env] Installing clone-voice in editable mode"
& $VenvPython -m pip install -e $RepoRoot

Write-Host ""
Write-Host "[setup_env] Done. Next steps:"
Write-Host "  $VenvDir\Scripts\Activate.ps1"
Write-Host "  python scripts\check_gpu.py"
Write-Host "  pytest"
