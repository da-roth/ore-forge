# Local build test for full ORE with XAD-JIT and Forge
# Mirrors the ci-ore.yaml GitHub workflow
# Run build-docker-image.ps1 first to create the image (optional, speeds up builds)

$ErrorActionPreference = "Stop"

Write-Host "======================================"
Write-Host "ORE + XAD-JIT + Forge Local Build Test"
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
$RUN_TESTS = if ($env:RUN_TESTS) { $env:RUN_TESTS } else { "off" }

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
Write-Host "Configuration (matching ci-ore.yaml GitHub Actions):"
Write-Host "  Container: $DOCKER_IMAGE"
Write-Host "  ORE: $ORE_REPO @ $ORE_BRANCH"
Write-Host "  XAD-JIT: $XAD_REPO @ $XAD_BRANCH"
Write-Host "  Forge: $FORGE_REPO @ $FORGE_BRANCH"
Write-Host "  Build Type: $BUILD_TYPE"
Write-Host "  Run Tests: $RUN_TESTS"
Write-Host ""

if (-not $skipInstall) {
    Write-Host "Pulling QuantLib dev environment container..."
    docker pull $DOCKER_IMAGE
}

Write-Host "Starting Docker container..."
Write-Host ""

# Create the build script
if ($skipInstall) {
    $installStep = '# Dependencies already in custom image'
    $forgeInstallPath = '/opt/forge'
    $cloneForge = '# Forge already in Docker image'
    $buildForge = 'echo "Using pre-built Forge from Docker image"'
} else {
    $installStep = 'apt-get update && apt-get install -y patch ninja-build'
    $forgeInstallPath = '/tmp/install'
    $cloneForge = 'git clone --depth 1 --branch "$FORGE_BRANCH" https://github.com/$FORGE_REPO.git forge'
    $buildForge = @'
echo "Building Forge (C++ API)"
cd /tmp/forge
cmake -B build -S tools/packaging -G Ninja -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DCMAKE_INSTALL_PREFIX=/tmp/install
cmake --build build --config "$BUILD_TYPE"
cmake --install build --config "$BUILD_TYPE"
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

# Clone full ORE Engine with QuantLib submodule
git clone --depth 1 --branch "`$ORE_BRANCH" https://github.com/`$ORE_REPO.git Engine
cd Engine
git submodule update --init --recursive
cd /tmp

# Clone XAD-JIT
git clone --depth 1 --branch "`$XAD_BRANCH" https://github.com/`$XAD_REPO.git xad-jit

# Clone Forge
$cloneForge

echo ""
echo "======================================"
echo "Applying ORE-Forge patches to QuantLib..."
echo "======================================"

cd /tmp/Engine/QuantLib

patch_count=0
for patch in /workspace/patches/*.patch; do
  if [ -f "`$patch" ]; then
    echo "Applying: `$patch"
    patch -p1 < "`$patch"
    patch_count=`$((patch_count + 1))
  fi
done
echo "Applied `$patch_count patches to QuantLib"

# Apply QuantExt patches if they exist
if ls /workspace/patches-quantext/*.patch 1>/dev/null 2>&1; then
  echo ""
  echo "Applying ORE-Forge patches to QuantExt..."
  cd /tmp/Engine/QuantExt
  for patch in /workspace/patches-quantext/*.patch; do
    if [ -f "`$patch" ]; then
      echo "Applying: `$patch"
      patch -p1 < "`$patch"
    fi
  done
fi

echo ""
echo "======================================"
echo "Building Forge..."
echo "======================================"

$buildForge

echo ""
echo "======================================"
echo "Configuring ORE with XAD-JIT + Forge..."
echo "======================================"

cd /tmp/Engine

cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE="`$BUILD_TYPE" \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations" \
  -DBOOST_ROOT=/usr \
  -DCMAKE_PREFIX_PATH="$forgeInstallPath" \
  -DQL_EXTERNAL_SUBDIRECTORIES="/tmp/xad-jit;/workspace" \
  -DQL_EXTRA_LINK_LIBRARIES=QuantLib-Risks \
  -DQL_NULL_AS_FUNCTIONS=ON \
  -DXAD_WARNINGS_PARANOID=OFF \
  -DQL_BUILD_TEST_SUITE=OFF \
  -DQL_BUILD_EXAMPLES=OFF \
  -DQL_BUILD_BENCHMARK=OFF \
  -DQLRISKS_DISABLE_AAD=OFF \
  -DQLRISKS_BUILD_TEST_SUITE=OFF \
  -DQLRISKS_ENABLE_FORGE_TESTS=OFF \
  -DQLRISKS_BUILD_BENCHMARK=OFF \
  -DORE_BUILD_DOC=OFF \
  -DORE_BUILD_EXAMPLES=ON \
  -DORE_BUILD_TESTS=OFF \
  -DORE_BUILD_SWIG=OFF

echo ""
echo "======================================"
echo "Building ORE..."
echo "======================================"
echo "Using -k 0 flag to show all compilation errors"
echo ""

cd /tmp/Engine/build

if cmake --build . -- -k 0; then
  echo ""
  echo "======================================"
  echo "BUILD SUCCESSFUL!"
  echo "======================================"

  # List what was built
  echo ""
  echo "Built libraries:"
  find . -name "*.so" -o -name "*.a" 2>/dev/null | head -20 || true

  echo ""
  echo "Built executables:"
  find . -name "ore" -type f -executable 2>/dev/null || true

  if [ "`$RUN_TESTS" = "on" ]; then
    echo ""
    echo "======================================"
    echo "Running Tests..."
    echo "======================================"

    ctest -R quantext --output-on-failure || true
    ctest -R oredata --output-on-failure || true
    ctest -R oreanalytics --output-on-failure || true
  fi

  echo ""
  echo "======================================"
  echo "ORE BUILD COMPLETE!"
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

# Get the ore-forge root directory (parent of scripts/)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$oreForgeRoot = Split-Path -Parent $scriptDir

Write-Host "ORE-Forge root: $oreForgeRoot"

# Debug: Save the script to see what's being generated
$buildScript | Out-File -FilePath "$scriptDir\debug-build-ore-script.sh" -Encoding ASCII

# Run Docker - Convert CRLF to LF and use UTF8 encoding without BOM
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
$scriptWithLF = $buildScript -replace "`r`n", "`n"
[System.IO.File]::WriteAllText("$scriptDir\build-ore-script-unix.sh", $scriptWithLF, $utf8NoBom)

# Run the script by mounting ore-forge root to /workspace
docker run --rm `
  -v "${oreForgeRoot}:/workspace" `
  -w /workspace `
  -e "ORE_REPO=$ORE_REPO" `
  -e "ORE_BRANCH=$ORE_BRANCH" `
  -e "XAD_REPO=$XAD_REPO" `
  -e "XAD_BRANCH=$XAD_BRANCH" `
  -e "FORGE_REPO=$FORGE_REPO" `
  -e "FORGE_BRANCH=$FORGE_BRANCH" `
  -e "BUILD_TYPE=$BUILD_TYPE" `
  -e "RUN_TESTS=$RUN_TESTS" `
  $DOCKER_IMAGE bash /workspace/scripts/build-ore-script-unix.sh

if ($LASTEXITCODE -eq 0) {
  Write-Host ""
  Write-Host "======================================"
  Write-Host "ORE local build test PASSED"
  Write-Host "======================================"
  exit 0
} else {
  Write-Host ""
  Write-Host "======================================"
  Write-Host "ORE local build test FAILED"
  Write-Host "======================================"
  exit 1
}
