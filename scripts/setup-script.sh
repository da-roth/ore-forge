set -e
set -x

cd /tmp

# Clone repositories
git clone --depth 1 --branch "v1.8.14.0" https://github.com/OpenSourceRisk/Engine.git Engine
cd Engine
git submodule update --init QuantLib
cd /tmp
# Copy (not move) QuantLib so we can reset from Engine/QuantLib later
cp -r Engine/QuantLib QuantLib

git clone --depth 1 --branch "main" https://github.com/da-roth/xad-jit.git xad-jit

# Apply patches
cd /tmp/QuantLib
echo "Applying ORE-Forge patches..."
for patch in /workspace/patches/*.patch; do
  if [ -f "$patch" ]; then
    echo "Applying: $patch"
    patch -p1 < "$patch"
  fi
done

# Configure CMake
mkdir -p build
cd build

if [ "off" = "off" ]; then
  FORGE_PREFIX="/opt/forge"
  echo "Configuring for C++ API"
  cmake -G Ninja -DBOOST_ROOT=/usr \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_BUILD_TYPE="Release" \
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
  echo "Configuring for C API"
  cmake -G Ninja -DBOOST_ROOT=/usr \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_BUILD_TYPE="Release" \
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
echo "Initial setup complete!"
echo "======================================"
echo "Build directory ready at: /tmp/QuantLib/build"
echo ""
echo "Use rebuild.ps1 to build incrementally"