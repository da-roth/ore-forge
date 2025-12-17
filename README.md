# ORE-Forge

Integration of [ORE (Open Source Risk Engine)](https://github.com/OpenSourceRisk/Engine) with [XAD-JIT](https://github.com/da-roth/xad-jit) and [Forge](https://github.com/da-roth/forge) for accelerated pricing and AAD-based sensitivities.

## Overview

This repository provides CI workflows and integration scripts to build ORE with:

- **XAD-JIT**: Automatic differentiation with JIT compilation support
- **Forge**: JIT compiler for mathematical expression graphs, generating native x86-64 code
- **QuantLib-Risks-Cpp-Forge**: Adapter connecting QuantLib with XAD and Forge

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  ORE (Open Source Risk Engine)                                      │
│  - OREAnalytics: XVA calculations, simulation                       │
│  - OREData: Trade parsing, market data                              │
│  - QuantExt: QuantLib extensions                                    │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  QuantLib (OpenSourceRisk fork, submodule)                          │
│  Templated with xad::AReal<double> via QuantLib-Risks               │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  XAD-JIT                                                            │
│  JITCompiler with ForgeBackend                                      │
│  - registerInput() / registerOutput()                               │
│  - newRecording() / compile() / forward()                           │
│  - computeAdjoints() for AAD sensitivities                          │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Forge                                                              │
│  JIT compiler: Graph → Native x86-64 code                           │
│  AVX2 SIMD: 4 scenarios per kernel execution                        │
└─────────────────────────────────────────────────────────────────────┘
```

## Benefits

| Aspect | Standard ORE | With XAD-JIT + Forge |
|--------|--------------|----------------------|
| Trade pricing (per call) | ~1-10 ms | ~1-10 μs |
| Sensitivities | Bump-and-revalue | AAD (single pass) |
| 100 sensitivities | ~44 sec | ~1 sec |

## CI Workflow

The CI workflow (`ci.yaml`) builds ORE with all components:

1. **Checkout** ORE Engine (with QuantLib submodule), XAD-JIT, Forge, QuantLib-Risks-Cpp-Forge
2. **Build Forge** as pre-built package (isolates AVX2 compiler flags)
3. **Configure ORE** with XAD integration via `QL_EXTERNAL_SUBDIRECTORIES`
4. **Build** all ORE components
5. **Run tests** for QuantExt, OREData, OREAnalytics

### Key CMake Flags

```bash
-DQL_EXTERNAL_SUBDIRECTORIES="path/to/xad-jit;path/to/QuantLib-Risks-Cpp-Forge"
-DQL_EXTRA_LINK_LIBRARIES=QuantLib-Risks
-DCMAKE_PREFIX_PATH=path/to/forge/install
```

## Repository References

| Repository | Purpose |
|------------|---------|
| [OpenSourceRisk/Engine](https://github.com/OpenSourceRisk/Engine) | ORE source code |
| [da-roth/xad-jit](https://github.com/da-roth/xad-jit) | XAD with JIT compilation |
| [da-roth/forge](https://github.com/da-roth/forge) | JIT compiler for math graphs |
| [da-roth/QuantLib-Risks-Cpp-Forge](https://github.com/da-roth/QuantLib-Risks-Cpp-Forge) | QuantLib + XAD + Forge integration |

## Manual Build

```bash
# Clone repositories
git clone --recursive https://github.com/OpenSourceRisk/Engine
git clone https://github.com/da-roth/xad-jit
git clone https://github.com/da-roth/forge
git clone https://github.com/da-roth/QuantLib-Risks-Cpp-Forge

# Build Forge (pre-built to isolate AVX2 flags)
cd forge
cmake -B build -S tools/packaging -DCMAKE_INSTALL_PREFIX=../install
cmake --build build
cmake --install build
cd ..

# Configure and build ORE
cd Engine
cmake -B build \
  -DCMAKE_PREFIX_PATH=$(pwd)/../install \
  -DQL_EXTERNAL_SUBDIRECTORIES="$(pwd)/../xad-jit;$(pwd)/../QuantLib-Risks-Cpp-Forge" \
  -DQL_EXTRA_LINK_LIBRARIES=QuantLib-Risks \
  -DQLRISKS_DISABLE_AAD=OFF
cmake --build build
```

## License

AGPL-3.0-or-later
