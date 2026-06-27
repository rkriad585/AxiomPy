# AxiomPy installer for Windows
# Usage: .\installer.ps1 [-Dev]

param([switch]$Dev)

$ErrorActionPreference = "Stop"

if (Get-Command uv -ErrorAction SilentlyContinue) {
    Write-Host "==> Installing with uv..." -ForegroundColor Green
    if ($Dev) { uv sync --dev } else { uv sync }
} else {
    Write-Host "==> uv not found, falling back to pip..." -ForegroundColor Yellow
    python -m pip install --upgrade pip
    if ($Dev) { python -m pip install -e . } else { python -m pip install -e . }
}

Write-Host "==> AxiomPy installed." -ForegroundColor Green
