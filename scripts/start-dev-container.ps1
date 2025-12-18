# Start a persistent development container for incremental builds
# Run this once, then use rebuild.ps1 for fast iteration

$ErrorActionPreference = "Stop"

Write-Host "======================================"
Write-Host "Starting ORE-Forge Dev Container"
Write-Host "======================================"
Write-Host ""

# Configuration
$ORE_REPO = if ($env:ORE_REPO) { $env:ORE_REPO } else { "OpenSourceRisk/Engine" }
$ORE_BRANCH = if ($env:ORE_BRANCH) { $env:ORE_BRANCH } else { "v1.8.14.0" }
$XAD_REPO = if ($env:XAD_REPO) { $env:XAD_REPO } else { "da-roth/xad-jit" }
$XAD_BRANCH = if ($env:XAD_BRANCH) { $env:XAD_BRANCH } else { "main" }
$FORGE_REPO = if ($env:FORGE_REPO) { $env:FORGE_REPO } else { "da-roth/forge" }
$FORGE_BRANCH = if ($env:FORGE_BRANCH) { $env:FORGE_BRANCH } else { "main" }
$BUILD_TYPE = if ($env:BUILD_TYPE) { $env:BUILD_TYPE } else { "Release" }
$CAPI = if ($env:CAPI) { $env:CAPI } else { "off" }
$CONTAINER_NAME = "ore-forge-dev"

# Check if container already exists
$existing = docker ps -a -q -f name=$CONTAINER_NAME
if ($existing) {
    Write-Host "Container '$CONTAINER_NAME' already exists. Removing it..."
    docker rm -f $CONTAINER_NAME | Out-Null
}

Write-Host "Starting persistent dev container: $CONTAINER_NAME"
Write-Host ""

# Start container in detached mode
docker run -d --name $CONTAINER_NAME `
  -v "${PWD}:/workspace" `
  -w /workspace `
  -e "ORE_REPO=$ORE_REPO" `
  -e "ORE_BRANCH=$ORE_BRANCH" `
  -e "XAD_REPO=$XAD_REPO" `
  -e "XAD_BRANCH=$XAD_BRANCH" `
  -e "FORGE_REPO=$FORGE_REPO" `
  -e "FORGE_BRANCH=$FORGE_BRANCH" `
  -e "BUILD_TYPE=$BUILD_TYPE" `
  -e "CAPI=$CAPI" `
  ore-forge-builder sleep infinity

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to start container"
    exit 1
}

Write-Host "Container started successfully!"
Write-Host ""
Write-Host "Running initial setup (clone repos, apply patches, configure cmake)..."
Write-Host ""

# Initial setup script
$setupScript = @"
set -e
set -x

cd /tmp

# Clone repositories
git clone --depth 1 --branch "$ORE_BRANCH" https://github.com/$ORE_REPO.git Engine
cd Engine
git submodule update --init QuantLib
cd /tmp
# Copy (not move) QuantLib so we can reset from Engine/QuantLib later
cp -r Engine/QuantLib QuantLib

git clone --depth 1 --branch "$XAD_BRANCH" https://github.com/$XAD_REPO.git xad-jit

# Apply patches
cd /tmp/QuantLib
echo "Applying ORE-Forge patches..."
for patch in /workspace/patches/*.patch; do
  if [ -f "`$patch" ]; then
    echo "Applying: `$patch"
    patch -p1 < "`$patch"
  fi
done

# Configure CMake
mkdir -p build
cd build

if [ "$CAPI" = "off" ]; then
  FORGE_PREFIX="/opt/forge"
  echo "Configuring for C++ API"
  cmake -G Ninja -DBOOST_ROOT=/usr \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DXAD_WARNINGS_PARANOID=OFF \
    -DCMAKE_PREFIX_PATH="`$FORGE_PREFIX" \
    -DQL_EXTERNAL_SUBDIRECTORIES="/tmp/xad-jit;/workspace" \
    -DQL_EXTRA_LINK_LIBRARIES=QuantLib-Risks \
    -DQL_NULL_AS_FUNCTIONS=ON \
    -DQL_BUILD_TEST_SUITE=OFF \
    -DQL_BUILD_EXAMPLES=OFF \
    -DQL_BUILD_BENCHMARK=OFF \
    -DQLRISKS_DISABLE_AAD=OFF \
    -DQLRISKS_BUILD_TEST_SUITE=ON \
    -DQLRISKS_ENABLE_FORGE_TESTS=ON \
    ..
else
  FORGE_PREFIX="/opt/forge-capi"
  echo "Configuring for C API"
  cmake -G Ninja -DBOOST_ROOT=/usr \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DXAD_WARNINGS_PARANOID=OFF \
    -DCMAKE_PREFIX_PATH="`$FORGE_PREFIX" \
    -DQL_EXTERNAL_SUBDIRECTORIES="/tmp/xad-jit;/workspace" \
    -DQL_EXTRA_LINK_LIBRARIES=QuantLib-Risks \
    -DQL_NULL_AS_FUNCTIONS=ON \
    -DQL_BUILD_TEST_SUITE=OFF \
    -DQL_BUILD_EXAMPLES=OFF \
    -DQL_BUILD_BENCHMARK=OFF \
    -DQLRISKS_DISABLE_AAD=OFF \
    -DQLRISKS_USE_FORGE_CAPI=ON \
    -DQLRISKS_BUILD_TEST_SUITE=ON \
    -DQLRISKS_ENABLE_FORGE_TESTS=ON \
    ..
fi

echo ""
echo "======================================"
echo "Initial setup complete!"
echo "======================================"
echo "Build directory ready at: /tmp/QuantLib/build"
echo ""
echo "Use rebuild.ps1 to build incrementally"
"@

# Run setup
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
$scriptWithLF = $setupScript -replace "`r`n", "`n"
[System.IO.File]::WriteAllText("$PWD\setup-script.sh", $scriptWithLF, $utf8NoBom)

docker exec $CONTAINER_NAME bash /workspace/setup-script.sh

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "======================================"
    Write-Host "Dev container ready!"
    Write-Host "======================================"
    Write-Host ""
    Write-Host "Container: $CONTAINER_NAME"
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "  1. Run: .\rebuild.ps1          # Build (incremental)"
    Write-Host "  2. Edit patches in patches/"
    Write-Host "  3. Run: .\rebuild.ps1 -Patch   # Re-apply patches and rebuild"
    Write-Host "  4. Run: .\stop-dev-container.ps1  # When done"
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "Setup failed!"
    exit 1
}
