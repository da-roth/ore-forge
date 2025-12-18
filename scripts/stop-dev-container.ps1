# Stop and remove the persistent dev container

$ErrorActionPreference = "Stop"
$CONTAINER_NAME = "ore-forge-dev"

Write-Host "Stopping and removing container: $CONTAINER_NAME"

docker stop $CONTAINER_NAME | Out-Null
docker rm $CONTAINER_NAME | Out-Null

Write-Host "Container removed successfully"
Write-Host ""
Write-Host "Run .\start-dev-container.ps1 to start a new dev session"
