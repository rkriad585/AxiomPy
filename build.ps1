Write-Host "==> Installing dependencies..." -ForegroundColor Green
uv sync

Write-Host "==> Building wheel..." -ForegroundColor Green
uv build

Write-Host "==> Done. Wheel is in dist/" -ForegroundColor Green
