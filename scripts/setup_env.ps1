# Creates/updates the venv and installs the right torch build for this machine.
# If a conda env (or venv) is already active, installs into it directly instead
# of creating a separate .venv - activate your env first, then run this.
# Usage: powershell -File scripts/setup_env.ps1
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$UsingActiveEnv = [bool]($env:CONDA_PREFIX -or $env:VIRTUAL_ENV)

$PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }

if ($UsingActiveEnv) {
    $EnvLabel = if ($env:CONDA_DEFAULT_ENV) { $env:CONDA_DEFAULT_ENV } elseif ($env:VIRTUAL_ENV) { $env:VIRTUAL_ENV } else { "active environment" }
    Write-Host "[setup_env] Detected an active environment ($EnvLabel) - installing into it directly"
}

Write-Host "[setup_env] Using $(& $PythonBin --version)"

$PyVersion = & $PythonBin -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($PyVersion -notin @("3.10", "3.11")) {
    Write-Host "[setup_env] WARNING: detected Python $PyVersion."
    Write-Host "[setup_env]          clone-voice targets 3.10 (matches GPT-SoVITS's own tested/recommended version)."
    Write-Host "[setup_env]          Continuing anyway, but installs may fail later, especially from Batch 3 onward."
    if ($UsingActiveEnv) {
        Write-Host "[setup_env]          Fix: activate a 3.10 env first (e.g. 'conda activate <your-3.10-env>'), then re-run."
    } else {
        Write-Host "[setup_env]          Fix: install Python 3.10, then re-run with `$env:PYTHON_BIN set to it."
    }
}

if (-not $UsingActiveEnv) {
    $VenvDir = Join-Path $RepoRoot ".venv"
    if (-not (Test-Path $VenvDir)) {
        Write-Host "[setup_env] Creating virtualenv at $VenvDir"
        & $PythonBin -m venv $VenvDir
    } else {
        Write-Host "[setup_env] Reusing existing virtualenv at $VenvDir"
    }
    $PythonBin = Join-Path $VenvDir "Scripts\python.exe"
}

& $PythonBin -m pip install --upgrade pip

Write-Host "[setup_env] Installing base dependencies"
& $PythonBin -m pip install -r (Join-Path $RepoRoot "requirements-base.txt")

$HasNvidiaGpu = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($HasNvidiaGpu) {
    Write-Host "[setup_env] nvidia-smi found - installing CUDA build of torch"
    Write-Host "[setup_env] (edit requirements-cuda.txt's cu126 tag first if your driver needs a different CUDA version)"
    & $PythonBin -m pip install -r (Join-Path $RepoRoot "requirements-cuda.txt")
} else {
    Write-Host "[setup_env] No NVIDIA GPU detected - installing CPU-only torch"
    & $PythonBin -m pip install -r (Join-Path $RepoRoot "requirements-cpu.txt")
}

Write-Host "[setup_env] Installing clone-voice in editable mode"
& $PythonBin -m pip install -e $RepoRoot

Write-Host ""
if ($UsingActiveEnv) {
    Write-Host "[setup_env] Done. Your active environment now has everything installed. Next steps:"
    Write-Host "  python scripts\check_gpu.py"
    Write-Host "  pytest"
} else {
    Write-Host "[setup_env] Done. Next steps:"
    Write-Host "  $VenvDir\Scripts\Activate.ps1"
    Write-Host "  python scripts\check_gpu.py"
    Write-Host "  pytest"
}
