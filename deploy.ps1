# ============================================================
# One-click deploy for the FL server (fl-server)
# Usage:  .\deploy.ps1
# Steps:  check docker -> ensure .env -> up --build -d -> health check
# ============================================================
$ErrorActionPreference = "Stop"

Write-Host "[1/4] Checking Docker..." -ForegroundColor Cyan
docker --version *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker not found. Install & start Docker Desktop first." -ForegroundColor Red
    exit 1
}

Write-Host "[2/4] Checking .env..." -ForegroundColor Cyan
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "Created .env from .env.example. Edit JWT_SECRET inside it before production." -ForegroundColor Yellow
}

Write-Host "[3/4] Building and starting services..." -ForegroundColor Cyan
docker compose up --build -d
if ($LASTEXITCODE -ne 0) {
    Write-Host "docker compose failed. See logs:  docker compose logs -f fl-server" -ForegroundColor Red
    exit 1
}

Write-Host "[4/4] Health check..." -ForegroundColor Cyan
Start-Sleep -Seconds 3
$resp = curl.exe -s http://localhost:8000/api/health
if ($resp -match '"ok"') {
    Write-Host "Deployed OK!  REST: http://localhost:8000  Docs: http://localhost:8000/docs" -ForegroundColor Green
} else {
    Write-Host "Health check failed (got: $resp). Check logs:  docker compose logs -f fl-server" -ForegroundColor Red
}
