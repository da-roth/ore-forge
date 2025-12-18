# Incremental rebuild in the persistent dev container
# Much faster than full rebuild - only compiles changed files

param(
    [switch]$Patch,      # Re-apply patches before building
    [switch]$Test        # Run tests after successful build
)

$ErrorActionPreference = "Stop"
$CONTAINER_NAME = "ore-forge-dev"

# Check if container exists and is running
$running = docker ps -q -f name=$CONTAINER_NAME
if (-not $running) {
    Write-Host "ERROR: Container '$CONTAINER_NAME' is not running"
    Write-Host "Run: .\start-dev-container.ps1 first"
    exit 1
}

Write-Host "======================================"
Write-Host "Incremental Rebuild"
Write-Host "======================================"
Write-Host ""

if ($Patch) {
    Write-Host "Checking and applying patches (smart mode)..."
    $patchScript = @"
set -e
cd /tmp/QuantLib

# Check and apply each patch only if needed
for patch in /workspace/patches/*.patch; do
  if [ -f "`$patch" ]; then
    # Check if patch is already applied
    if patch -p1 -R --dry-run -s < "`$patch" > /dev/null 2>&1; then
      echo "SKIP: `$(basename `$patch) already applied"
    else
      # Check if patch can be applied (not already applied)
      if patch -p1 --dry-run -s < "`$patch" > /dev/null 2>&1; then
        echo "APPLY: `$(basename `$patch)"
        patch -p1 < "`$patch"
      else
        echo "ERROR: `$(basename `$patch) cannot be applied (conflicts?)"
        exit 1
      fi
    fi
  fi
done
echo "Patches check complete"
"@
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    $scriptWithLF = $patchScript -replace "`r`n", "`n"
    [System.IO.File]::WriteAllText("$PWD\repatch-script.sh", $scriptWithLF, $utf8NoBom)

    docker exec $CONTAINER_NAME bash /workspace/repatch-script.sh
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to apply patches"
        exit 1
    }
    Write-Host ""
}

Write-Host "Building (incremental - only changed files)..."
Write-Host ""

$buildScript = @"
set -e
# Check if build directory exists, if not create it
if [ ! -d /tmp/QuantLib/build ]; then
  echo "Build directory doesn't exist, running first-time build..."
  mkdir -p /tmp/QuantLib/build
  cd /tmp/QuantLib/build

  # Configure cmake (same as initial setup)
  if [ "$CAPI" = "off" ]; then
    FORGE_PREFIX="/opt/forge"
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
fi

cd /tmp/QuantLib/build
echo "Running incremental build with -k 0 (show all errors)..."
# Get number of CPU cores for parallel build
NPROC=`$(nproc)
echo "Building with `$NPROC parallel jobs..."
if cmake --build . -- -k 0 -j`$NPROC; then
  echo ""
  echo "======================================"
  echo "BUILD SUCCESSFUL!"
  echo "======================================"
  exit 0
else
  echo ""
  echo "======================================"
  echo "BUILD FAILED"
  echo "======================================"
  exit 1
fi
"@

$utf8NoBom = New-Object System.Text.UTF8Encoding $false
$scriptWithLF = $buildScript -replace "`r`n", "`n"
[System.IO.File]::WriteAllText("$PWD\rebuild-script.sh", $scriptWithLF, $utf8NoBom)

docker exec $CONTAINER_NAME bash /workspace/rebuild-script.sh

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "======================================"
    Write-Host "Build succeeded!"
    Write-Host "======================================"

    if ($Test) {
        Write-Host ""
        Write-Host "Running tests..."
        docker exec $CONTAINER_NAME bash -c "cd /tmp/QuantLib/build && ./ORE-Forge/test-suite/quantlib-risks-test-suite --log_level=message"

        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "======================================"
            Write-Host "TESTS PASSED!"
            Write-Host "======================================"
        } else {
            Write-Host ""
            Write-Host "Tests failed"
            exit 1
        }
    }
    exit 0
} else {
    Write-Host ""
    Write-Host "======================================"
    Write-Host "Build failed - see errors above"
    Write-Host "======================================"
    exit 1
}
