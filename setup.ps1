# NVTranscriber — Phase 1 environment setup
# Run from the project root: .\setup.ps1
# Requires: Python 3.11+, CUDA toolkit, ffmpeg in PATH

param(
    [string]$CudaVersion = "cu124"   # Change to match your CUDA installation (cu118, cu121, cu124 …)
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "=== NVTranscriber Setup ===" -ForegroundColor Cyan

# 1. Create virtual environment
if (-Not (Test-Path "venv")) {
    Write-Host "[1/4] Creating virtual environment …" -ForegroundColor Yellow
    python -m venv venv
} else {
    Write-Host "[1/4] Virtual environment already exists — skipping." -ForegroundColor Gray
}

# 2. Activate venv
Write-Host "[2/4] Activating virtual environment …" -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# 3. Install PyTorch with CUDA support (must come before requirements.txt)
Write-Host "[3/4] Installing PyTorch ($CudaVersion) + torchaudio …" -ForegroundColor Yellow
pip install torch torchaudio --index-url "https://download.pytorch.org/whl/$CudaVersion"

# 4. Install remaining backend dependencies
Write-Host "[4/4] Installing backend requirements …" -ForegroundColor Yellow
pip install -r backend\requirements.txt

Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Green
Write-Host "Start the server with:" -ForegroundColor Cyan
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000" -ForegroundColor White
