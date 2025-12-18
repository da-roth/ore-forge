# Build custom Docker image with patch and ninja-build pre-installed
# Run this once, then use test-build-fast.ps1

$ErrorActionPreference = "Stop"

Write-Host "======================================"
Write-Host "Building ORE-Forge Docker Image"
Write-Host "======================================"
Write-Host ""
Write-Host "This will:"
Write-Host "  1. Pull ghcr.io/lballabio/quantlib-devenv:rolling"
Write-Host "  2. Add patch and ninja-build on top"
Write-Host "  3. Tag as 'ore-forge-builder'"
Write-Host ""
Write-Host "This takes ~1-2 minutes and only needs to be done ONCE."
Write-Host ""

docker build -t ore-forge-builder .

if ($LASTEXITCODE -eq 0) {
  Write-Host ""
  Write-Host "======================================"
  Write-Host "Docker image built successfully!"
  Write-Host "======================================"
  Write-Host ""
  Write-Host "Image details:"
  docker images ore-forge-builder
  Write-Host ""
  Write-Host "You can now run builds without 'apt-get install':"
  Write-Host "  .\test-build-fast.ps1"
  Write-Host ""
  exit 0
} else {
  Write-Host ""
  Write-Host "======================================"
  Write-Host "Docker image build FAILED"
  Write-Host "======================================"
  exit 1
}
