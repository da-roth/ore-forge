##############################################################################
#
#  ORE-Forge Example 3 CVA Benchmarks
#
#  Three benchmark executables for comparing CVA performance:
#  - example3_benchmark_original: Vanilla ORE (double precision)
#  - example3_benchmark_xad: XAD tape-based AAD
#  - example3_benchmark_forge: XAD JIT with Forge
#
#  Copyright (C) 2025 The ORE-Forge Authors
#
#  SPDX-License-Identifier: AGPL-3.0-or-later
#
##############################################################################

option(ORE_BUILD_BENCHMARK "Build ORE Example 3 CVA benchmarks" OFF)

if(ORE_BUILD_BENCHMARK)
    message(STATUS "ORE-Forge: Building Example 3 CVA benchmarks")

    # Common settings
    set(BENCHMARK_COMMON_LIBS
        OREAnalytics
        OREData
        QuantExt
        ql_library
    )

    # Benchmark 1: Original (no AAD)
    add_executable(example3_benchmark_original
        example3_benchmark_original.cpp
    )
    target_link_libraries(example3_benchmark_original PRIVATE
        ${BENCHMARK_COMMON_LIBS}
    )
    if(NOT Boost_USE_STATIC_LIBS)
        target_compile_definitions(example3_benchmark_original PRIVATE BOOST_ALL_DYN_LINK)
    endif()

    # Benchmark 2: XAD Tape (requires XAD-enabled QuantLib)
    if(NOT QLRISKS_DISABLE_AAD)
        add_executable(example3_benchmark_xad
            example3_benchmark_xad.cpp
        )
        target_link_libraries(example3_benchmark_xad PRIVATE
            ${BENCHMARK_COMMON_LIBS}
            XAD::xad
        )
        target_compile_definitions(example3_benchmark_xad PRIVATE QLRISKS_ENABLE_AAD=1)
        if(NOT Boost_USE_STATIC_LIBS)
            target_compile_definitions(example3_benchmark_xad PRIVATE BOOST_ALL_DYN_LINK)
        endif()
    else()
        message(STATUS "ORE-Forge: Skipping example3_benchmark_xad (QLRISKS_DISABLE_AAD=ON)")
    endif()

    # Benchmark 3: Forge JIT (requires XAD-JIT + Forge)
    if(NOT QLRISKS_DISABLE_AAD AND TARGET QLRisks::forge)
        add_executable(example3_benchmark_forge
            example3_benchmark_forge.cpp
        )
        target_link_libraries(example3_benchmark_forge PRIVATE
            ${BENCHMARK_COMMON_LIBS}
            QLRisks::forge
        )
        target_compile_definitions(example3_benchmark_forge PRIVATE
            QLRISKS_ENABLE_AAD=1
            QLRISKS_HAS_FORGE=1
        )
        if(NOT Boost_USE_STATIC_LIBS)
            target_compile_definitions(example3_benchmark_forge PRIVATE BOOST_ALL_DYN_LINK)
        endif()
    else()
        message(STATUS "ORE-Forge: Skipping example3_benchmark_forge (Forge not available)")
    endif()

    # Install benchmarks
    install(TARGETS example3_benchmark_original RUNTIME DESTINATION bin OPTIONAL)
    if(TARGET example3_benchmark_xad)
        install(TARGETS example3_benchmark_xad RUNTIME DESTINATION bin OPTIONAL)
    endif()
    if(TARGET example3_benchmark_forge)
        install(TARGETS example3_benchmark_forge RUNTIME DESTINATION bin OPTIONAL)
    endif()

endif()
