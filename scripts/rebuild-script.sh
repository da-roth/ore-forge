set -e
# Check if build directory exists, if not create it
if [ ! -d /tmp/QuantLib/build ]; then
  echo "Build directory doesn't exist, running first-time build..."
  mkdir -p /tmp/QuantLib/build
  cd /tmp/QuantLib/build

  # Configure cmake (same as initial setup)
  if [ "" = "off" ]; then
    FORGE_PREFIX="/opt/forge"
    cmake -G Ninja -DBOOST_ROOT=/usr \
      -DCMAKE_CXX_STANDARD=17 \
      -DCMAKE_BUILD_TYPE="" \
      -DXAD_WARNINGS_PARANOID=OFF \
      -DCMAKE_PREFIX_PATH="$FORGE_PREFIX" \
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
      -DCMAKE_BUILD_TYPE="" \
      -DXAD_WARNINGS_PARANOID=OFF \
      -DCMAKE_PREFIX_PATH="$FORGE_PREFIX" \
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
NPROC=$(nproc)
echo "Building with $NPROC parallel jobs..."
if cmake --build . -- -k 0 -j$NPROC; then
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