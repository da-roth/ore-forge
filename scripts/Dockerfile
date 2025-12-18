# Custom Docker image for ORE-Forge builds
# Pre-installs dependencies and builds Forge to save time

FROM ghcr.io/lballabio/quantlib-devenv:rolling

# Install additional dependencies needed for ORE-Forge builds
RUN apt-get update && \
    apt-get install -y patch ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Pre-build Forge (C++ API) since it rarely changes
# This saves ~2-3 minutes on each build
RUN cd /tmp && \
    git clone --depth 1 --branch main https://github.com/da-roth/forge.git && \
    cd forge && \
    cmake -B build -S tools/packaging \
      -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/forge && \
    cmake --build build --config Release && \
    cmake --install build --config Release && \
    rm -rf /tmp/forge

# Also build Forge C API version
RUN cd /tmp && \
    git clone --depth 1 --branch main https://github.com/da-roth/forge.git && \
    cd forge && \
    cmake -B build -S tools/capi \
      -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/forge-capi && \
    cmake --build build --config Release && \
    cmake --install build --config Release && \
    rm -rf /tmp/forge

WORKDIR /workspace
