# Fast local build test using custom ORE-Forge Docker image
# Run build-docker-image.ps1 first to create the image

$ErrorActionPreference = "Stop"

Write-Host "======================================"
Write-Host "ORE-Forge Local Build Test"
Write-Host "======================================"
Write-Host ""

# Configuration matching the GitHub workflow defaults
$ORE_REPO = if ($env:ORE_REPO) { $env:ORE_REPO } else { "OpenSourceRisk/Engine" }
$ORE_BRANCH = if ($env:ORE_BRANCH) { $env:ORE_BRANCH } else { "v1.8.14.0" }
$XAD_REPO = if ($env:XAD_REPO) { $env:XAD_REPO } else { "da-roth/xad-jit" }
$XAD_BRANCH = if ($env:XAD_BRANCH) { $env:XAD_BRANCH } else { "main" }
$FORGE_REPO = if ($env:FORGE_REPO) { $env:FORGE_REPO } else { "da-roth/forge" }
$FORGE_BRANCH = if ($env:FORGE_BRANCH) { $env:FORGE_BRANCH } else { "main" }
$BUILD_TYPE = if ($env:BUILD_TYPE) { $env:BUILD_TYPE } else { "Release" }
$CAPI = if ($env:CAPI) { $env:CAPI } else { "off" }

# Check if custom image exists, otherwise use base image
$imageExists = docker images -q ore-forge-builder 2>$null
if ($imageExists) {
    $DOCKER_IMAGE = "ore-forge-builder"
    $skipInstall = $true
    Write-Host "Using custom ORE-Forge image (dependencies pre-installed)"
} else {
    $DOCKER_IMAGE = "ghcr.io/lballabio/quantlib-devenv:rolling"
    $skipInstall = $false
    Write-Host "Using base QuantLib image"
    Write-Host "TIP: Run .\build-docker-image.ps1 once to skip 'apt-get install' step"
}

Write-Host ""
Write-Host "Configuration (matching GitHub Actions):"
Write-Host "  Container: $DOCKER_IMAGE"
Write-Host "  ORE: $ORE_REPO @ $ORE_BRANCH"
Write-Host "  XAD-JIT: $XAD_REPO @ $XAD_BRANCH"
Write-Host "  Forge: $FORGE_REPO @ $FORGE_BRANCH"
Write-Host "  Build Type: $BUILD_TYPE"
Write-Host "  C API: $CAPI"
Write-Host ""

if (-not $skipInstall) {
    Write-Host "Pulling QuantLib dev environment container..."
    docker pull $DOCKER_IMAGE | Out-Null
}

Write-Host "Starting Docker container..."
Write-Host ""

# Create the build script
if ($skipInstall) {
    $installStep = '# Dependencies already in custom image'
    $forgeInstallPath = '/opt/forge'     # Pre-built Forge location
    $forgeCapiInstallPath = '/opt/forge-capi'
    $cloneForge = '# Forge already in Docker image'
    $buildForge = 'echo "Using pre-built Forge from Docker image"'
} else {
    $installStep = 'apt-get update && apt-get install -y patch ninja-build'
    $forgeInstallPath = '/tmp/install'
    $forgeCapiInstallPath = '/tmp/install'
    $cloneForge = 'git clone --depth 1 --branch "$FORGE_BRANCH" https://github.com/$FORGE_REPO.git forge'
    $buildForge = @'
if [ "$CAPI" = "off" ]; then
  echo "Building Forge (C++ API)"
  cd /tmp/forge
  cmake -B build -S tools/packaging -G Ninja -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DCMAKE_INSTALL_PREFIX=/tmp/install
  cmake --build build --config "$BUILD_TYPE"
  cmake --install build --config "$BUILD_TYPE"
else
  echo "Building Forge (C API)"
  cd /tmp/forge
  cmake -B build -S tools/capi -G Ninja -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DCMAKE_INSTALL_PREFIX=/tmp/install
  cmake --build build --config "$BUILD_TYPE"
  cmake --install build --config "$BUILD_TYPE"
fi
'@
}

$buildScript = @"
set -e
set -x

echo "======================================"
echo "Installing additional dependencies..."
echo "======================================"

$installStep

echo ""
echo "======================================"
echo "Cloning repositories..."
echo "======================================"

cd /tmp

git clone --depth 1 --branch "`$ORE_BRANCH" https://github.com/`$ORE_REPO.git Engine

cd Engine
git submodule update --init QuantLib

cd /tmp
mv Engine/QuantLib QuantLib

git clone --depth 1 --branch "`$XAD_BRANCH" https://github.com/`$XAD_REPO.git xad-jit

$cloneForge

echo ""
echo "======================================"
echo "Applying ORE-Forge patches..."
echo "======================================"

cd /tmp/QuantLib

echo "Applying ORE-Forge patches..."
patch_count=0
for patch in /workspace/patches/*.patch; do
  if [ -f "`$patch" ]; then
    echo "Applying: `$patch"
    patch -p1 < "`$patch"
    patch_count=`$((patch_count + 1))
  fi
done
echo "Patches applied. Verifying:"
grep -n "const Real tolerance" ql/math/comparison.hpp | head -2 || true

echo ""
echo "======================================"
echo "Building Forge..."
echo "======================================"

$buildForge

# Set the correct Forge install path based on C API setting
if [ "`$CAPI" = "off" ]; then
  FORGE_PREFIX="$forgeInstallPath"
else
  FORGE_PREFIX="$forgeCapiInstallPath"
fi

echo ""
echo "======================================"
echo "Configuring QuantLib with XAD-JIT + Forge..."
echo "======================================"

cd /tmp/QuantLib
mkdir build
cd build

if [ "`$CAPI" = "off" ]; then
  echo "Configuring for C++ API"
  cmake -G Ninja -DBOOST_ROOT=/usr \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_BUILD_TYPE="`$BUILD_TYPE" \
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
  echo "Configuring for C API"
  cmake -G Ninja -DBOOST_ROOT=/usr \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_BUILD_TYPE="`$BUILD_TYPE" \
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
echo "Building QuantLib..."
echo "======================================"
echo "Using -k 0 flag to show all compilation errors"
echo ""

if cmake --build . -- -k 0; then
  echo ""
  echo "======================================"
  echo "BUILD SUCCESSFUL!"
  echo "======================================"

  echo ""
  echo "======================================"
  echo "Running Tests..."
  echo "======================================"

  ./ORE-Forge/test-suite/quantlib-risks-test-suite --log_level=message

  echo ""
  echo "======================================"
  echo "TESTS PASSED!"
  echo "======================================"
  exit 0
else
  echo ""
  echo "======================================"
  echo "BUILD FAILED - See errors above"
  echo "======================================"
  exit 1
fi
"@

# Debug: Save the script to see what's being generated
$buildScript | Out-File -FilePath "debug-build-script.sh" -Encoding ASCII

# Run Docker - Convert CRLF to LF and use UTF8 encoding without BOM
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
$scriptWithLF = $buildScript -replace "`r`n", "`n"
[System.IO.File]::WriteAllText("$PWD\build-script-unix.sh", $scriptWithLF, $utf8NoBom)

# Run the script by mounting and executing it
docker run --rm `
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
  $DOCKER_IMAGE bash /workspace/build-script-unix.sh

if ($LASTEXITCODE -eq 0) {
  Write-Host ""
  Write-Host "======================================"
  Write-Host "Local build test PASSED"
  Write-Host "======================================"
  exit 0
} else {
  Write-Host ""
  Write-Host "======================================"
  Write-Host "Local build test FAILED"
  Write-Host "======================================"
  exit 1
}
