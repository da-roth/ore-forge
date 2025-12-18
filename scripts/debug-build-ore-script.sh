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

# Clone full ORE Engine with QuantLib submodule
git clone --depth 1 --branch "$ORE_BRANCH" https://github.com/$ORE_REPO.git Engine
cd Engine
git submodule update --init --recursive
cd /tmp

# Clone XAD-JIT
git clone --depth 1 --branch "$XAD_BRANCH" https://github.com/$XAD_REPO.git xad-jit

# Clone Forge
git clone --depth 1 --branch "$FORGE_BRANCH" https://github.com/$FORGE_REPO.git forge

echo ""
echo "======================================"
echo "Applying ORE-Forge patches to QuantLib..."
echo "======================================"

cd /tmp/Engine/QuantLib

patch_count=0
for patch in /workspace/patches/*.patch; do
  if [ -f "$patch" ]; then
    echo "Applying: $patch"
    patch -p1 < "$patch"
    patch_count=$((patch_count + 1))
  fi
done
echo "Applied $patch_count patches to QuantLib"

# Apply QuantExt patches if they exist
if ls /workspace/patches-quantext/*.patch 1>/dev/null 2>&1; then
  echo ""
  echo "Applying ORE-Forge patches to QuantExt..."
  cd /tmp/Engine/QuantExt
  for patch in /workspace/patches-quantext/*.patch; do
    if [ -f "$patch" ]; then
      echo "Applying: $patch"
      patch -p1 < "$patch"
    fi
  done
fi

echo ""
echo "======================================"
echo "Building Forge..."
echo "======================================"

echo "Building Forge (C++ API)"
cd /tmp/forge
cmake -B build -S tools/packaging -G Ninja -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DCMAKE_INSTALL_PREFIX=/tmp/install
cmake --build build --config "$BUILD_TYPE"
cmake --install build --config "$BUILD_TYPE"

echo ""
echo "======================================"
echo "Configuring ORE with XAD-JIT + Forge..."
echo "======================================"

cd /tmp/Engine

cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations" \
  -DBOOST_ROOT=/usr \
  -DCMAKE_PREFIX_PATH="/tmp/install" \
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

  if [ "$RUN_TESTS" = "on" ]; then
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
