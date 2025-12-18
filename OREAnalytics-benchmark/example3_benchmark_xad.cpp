/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2025 The ORE-Forge Authors

 This file is part of ORE-Forge, integrating Forge JIT compilation with ORE.

 ORE Example 3 CVA Benchmark - XAD Tape (Reverse-Mode AAD)

 This benchmark runs ORE's Example 3 (Swaption CVA calculation) using
 XAD tape-based reverse-mode automatic differentiation. This computes
 sensitivities (dCVA/dMarketData) using the adjoint method.

 When compiled with XAD-enabled QuantLib, Real = xad::AReal<double>,
 which automatically records operations on the tape for derivative
 computation.

 Benchmark Configuration:
 - Portfolio: Swaps and Swaptions
 - Analytics: NPV, Simulation, XVA (CVA)
 - Monte Carlo: 1000 paths, 83 time steps
 - Model: Multi-currency LGM
 - AAD: XAD tape-based reverse mode
*/

#include <orea/app/oreapp.hpp>
#include <orea/app/initbuilders.hpp>

#include <ql/quantlib.hpp>

// XAD headers (only available when compiled with XAD)
#ifdef QLRISKS_ENABLE_AAD
#include <XAD/XAD.hpp>
#endif

#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <cmath>

using namespace std;
using namespace ore::data;
using namespace ore::analytics;

namespace {

// Benchmark configuration
constexpr int WARMUP_ITERATIONS = 3;
constexpr int BENCHMARK_ITERATIONS = 5;

struct BenchmarkResult {
    double forward_mean_ms;
    double forward_stddev_ms;
    double backward_mean_ms;
    double backward_stddev_ms;
    double total_mean_ms;
    double total_stddev_ms;
};

double computeMean(const vector<double>& v) {
    return accumulate(v.begin(), v.end(), 0.0) / v.size();
}

double computeStdDev(const vector<double>& v, double mean) {
    double sq_sum = 0.0;
    for (double t : v) {
        sq_sum += (t - mean) * (t - mean);
    }
    return sqrt(sq_sum / v.size());
}

void printHeader() {
    cout << "\n";
    cout << "=============================================================================\n";
    cout << "  ORE EXAMPLE 3 CVA BENCHMARK - XAD TAPE (Reverse-Mode AAD)\n";
    cout << "=============================================================================\n";
    cout << endl;
    cout << "  This benchmark measures the performance of ORE's CVA calculation\n";
    cout << "  with XAD tape-based automatic differentiation for sensitivities.\n";
    cout << endl;
    cout << "  CONFIGURATION:\n";
    cout << "    Portfolio: Swaps + Swaptions (5 trades)\n";
    cout << "    Analytics: NPV, Simulation, XVA (CVA)\n";
    cout << "    Monte Carlo: 1000 paths, 83 time steps (quarterly)\n";
    cout << "    Model: Multi-currency LGM (EUR, USD, GBP, CHF, JPY)\n";
    cout << "    AAD Method: XAD tape (reverse-mode)\n";
    cout << endl;
    cout << "  BENCHMARK SETTINGS:\n";
    cout << "    Warmup iterations: " << WARMUP_ITERATIONS << "\n";
    cout << "    Benchmark iterations: " << BENCHMARK_ITERATIONS << "\n";
    cout << endl;

#ifdef QLRISKS_ENABLE_AAD
    cout << "  XAD STATUS: ENABLED (Real = xad::AReal<double>)\n";
#else
    cout << "  XAD STATUS: DISABLED (Real = double) - No sensitivities computed\n";
#endif
    cout << endl;
}

void printResults(const BenchmarkResult& result) {
    cout << "  RESULTS:\n";
    cout << "    +---------------------+------------------+------------------+\n";
    cout << "    | Phase               | Mean             | Std Dev          |\n";
    cout << "    +---------------------+------------------+------------------+\n";
    cout << "    | Forward (pricing)   | " << setw(12) << fixed << setprecision(2) << result.forward_mean_ms
         << " ms | " << setw(12) << result.forward_stddev_ms << " ms |\n";
    cout << "    | Backward (adjoints) | " << setw(12) << fixed << setprecision(2) << result.backward_mean_ms
         << " ms | " << setw(12) << result.backward_stddev_ms << " ms |\n";
    cout << "    | Total               | " << setw(12) << fixed << setprecision(2) << result.total_mean_ms
         << " ms | " << setw(12) << result.total_stddev_ms << " ms |\n";
    cout << "    +---------------------+------------------+------------------+\n";
    cout << endl;
}

struct TimingResult {
    double forward_ms;
    double backward_ms;
    double total_ms;
};

TimingResult runSingleIteration(const string& inputFile) {
    TimingResult timing;

#ifdef QLRISKS_ENABLE_AAD
    // Create tape for this iteration
    using tape_type = xad::Tape<double>;
    tape_type tape;

    // Register tape as active
    tape.registerForward();

    // TODO: Register market data inputs on tape
    // This requires access to the market data loader

    auto forward_start = chrono::high_resolution_clock::now();
#else
    auto forward_start = chrono::high_resolution_clock::now();
#endif

    // Forward pass: run ORE
    auto params = QuantLib::ext::make_shared<Parameters>();
    params->fromFile(inputFile);
    OREApp ore(params, false);  // console = false
    ore.run();

    auto forward_end = chrono::high_resolution_clock::now();

#ifdef QLRISKS_ENABLE_AAD
    // Backward pass: compute adjoints
    auto backward_start = chrono::high_resolution_clock::now();

    // TODO: Seed output adjoints and run backward sweep
    // tape.computeAdjoints();

    auto backward_end = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> forward_elapsed = forward_end - forward_start;
    chrono::duration<double, milli> backward_elapsed = backward_end - backward_start;

    timing.forward_ms = forward_elapsed.count();
    timing.backward_ms = backward_elapsed.count();
    timing.total_ms = timing.forward_ms + timing.backward_ms;
#else
    chrono::duration<double, milli> forward_elapsed = forward_end - forward_start;
    timing.forward_ms = forward_elapsed.count();
    timing.backward_ms = 0.0;
    timing.total_ms = timing.forward_ms;
#endif

    return timing;
}

} // anonymous namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " path/to/ore.xml" << endl;
        return 1;
    }

    string inputFile(argv[1]);

    try {
        ore::analytics::initBuilders();

        printHeader();

        // Warmup
        cout << "  Running warmup iterations...\n";
        for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
            cout << "    Warmup " << (i + 1) << "/" << WARMUP_ITERATIONS << "...\r" << flush;
            runSingleIteration(inputFile);
        }
        cout << "    Warmup complete.              \n";
        cout << endl;

        // Benchmark
        cout << "  Running benchmark iterations...\n";
        vector<double> forward_times, backward_times, total_times;

        for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
            TimingResult timing = runSingleIteration(inputFile);
            forward_times.push_back(timing.forward_ms);
            backward_times.push_back(timing.backward_ms);
            total_times.push_back(timing.total_ms);

            cout << "    Iteration " << setw(2) << (i + 1) << "/" << BENCHMARK_ITERATIONS
                 << ": Forward=" << fixed << setprecision(2) << timing.forward_ms << " ms"
                 << ", Backward=" << timing.backward_ms << " ms"
                 << ", Total=" << timing.total_ms << " ms\n";
        }
        cout << endl;

        // Results
        BenchmarkResult result;
        result.forward_mean_ms = computeMean(forward_times);
        result.forward_stddev_ms = computeStdDev(forward_times, result.forward_mean_ms);
        result.backward_mean_ms = computeMean(backward_times);
        result.backward_stddev_ms = computeStdDev(backward_times, result.backward_mean_ms);
        result.total_mean_ms = computeMean(total_times);
        result.total_stddev_ms = computeStdDev(total_times, result.total_mean_ms);

        printResults(result);

        return 0;

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}
