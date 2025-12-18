set -e
set -x

echo "======================================"
echo "Installing additional dependencies..."
echo "======================================"

apt-get update && apt-get install -y patch ninja-build

echo ""
echo "======================================"
echo "Cloning repositories..."
echo "======================================"

cd /tmp

git clone --depth 1 --branch "$ORE_BRANCH" https://github.com/$ORE_REPO.git Engine

cd Engine
git submodule update --init QuantLib

cd /tmp
mv Engine/QuantLib QuantLib

git clone --depth 1 --branch "$XAD_BRANCH" https://github.com/$XAD_REPO.git xad-jit

git clone --depth 1 --branch "$FORGE_BRANCH" https://github.com/$FORGE_REPO.git forge

echo ""
echo "======================================"
echo "Applying ORE-Forge patches..."
echo "======================================"

cd /tmp/QuantLib

echo "Applying ORE-Forge patches..."
patch_count=0
for patch in /workspace/patches/*.patch; do
  if [ -f "$patch" ]; then
    echo "Applying: $patch"
    patch -p1 < "$patch"
    patch_count=$((patch_count + 1))
  fi
done
echo "Patches applied. Verifying:"
grep -n "const Real tolerance" ql/math/comparison.hpp | head -2 || true

echo ""
echo "======================================"
echo "Building Forge..."
echo "======================================"

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

# Set the correct Forge install path based on C API setting
if [ "$CAPI" = "off" ]; then
  FORGE_PREFIX="/tmp/install"
else
  FORGE_PREFIX="/tmp/install"
fi

echo ""
echo "======================================"
echo "Configuring QuantLib with XAD-JIT + Forge..."
echo "======================================"

cd /tmp/QuantLib
mkdir build
cd build

if [ "$CAPI" = "off" ]; then
  echo "Configuring for C++ API"
  cmake -G Ninja -DBOOST_ROOT=/usr \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
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
  echo "Configuring for C API"
  cmake -G Ninja -DBOOST_ROOT=/usr \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
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