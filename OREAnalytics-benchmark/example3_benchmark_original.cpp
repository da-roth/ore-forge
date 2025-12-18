/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2025 The ORE-Forge Authors

 This file is part of ORE-Forge, integrating Forge JIT compilation with ORE.

 ORE Example 3 CVA Benchmark - Original (No AAD)

 This benchmark runs ORE's Example 3 (Swaption CVA calculation) using
 vanilla QuantLib with double precision. This serves as the baseline
 for performance comparison with XAD and Forge-accelerated versions.

 Benchmark Configuration:
 - Portfolio: Swaps and Swaptions
 - Analytics: NPV, Simulation, XVA (CVA)
 - Monte Carlo: 1000 paths, 83 time steps
 - Model: Multi-currency LGM
*/

#include <orea/app/oreapp.hpp>
#include <orea/app/initbuilders.hpp>

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
constexpr int WARMUP_ITERATIONS = 5;
constexpr int BENCHMARK_ITERATIONS = 10;

struct BenchmarkResult {
    double mean_ms;
    double stddev_ms;
    double min_ms;
    double max_ms;
};

BenchmarkResult computeStats(const vector<double>& times_ms) {
    BenchmarkResult result;

    result.mean_ms = accumulate(times_ms.begin(), times_ms.end(), 0.0) / times_ms.size();
    result.min_ms = *min_element(times_ms.begin(), times_ms.end());
    result.max_ms = *max_element(times_ms.begin(), times_ms.end());

    double sq_sum = 0.0;
    for (double t : times_ms) {
        sq_sum += (t - result.mean_ms) * (t - result.mean_ms);
    }
    result.stddev_ms = sqrt(sq_sum / times_ms.size());

    return result;
}

void printHeader() {
    cout << "\n";
    cout << "=============================================================================\n";
    cout << "  ORE EXAMPLE 3 CVA BENCHMARK - ORIGINAL (No AAD)\n";
    cout << "=============================================================================\n";
    cout << endl;
    cout << "  This benchmark measures the performance of ORE's CVA calculation\n";
    cout << "  using vanilla QuantLib with double precision (no AAD).\n";
    cout << endl;
    cout << "  CONFIGURATION:\n";
    cout << "    Portfolio: Swaps + Swaptions (5 trades)\n";
    cout << "    Analytics: NPV, Simulation, XVA (CVA)\n";
    cout << "    Monte Carlo: 1000 paths, 83 time steps (quarterly)\n";
    cout << "    Model: Multi-currency LGM (EUR, USD, GBP, CHF, JPY)\n";
    cout << endl;
    cout << "  BENCHMARK SETTINGS:\n";
    cout << "    Warmup iterations: " << WARMUP_ITERATIONS << "\n";
    cout << "    Benchmark iterations: " << BENCHMARK_ITERATIONS << "\n";
    cout << endl;
}

void printResults(const BenchmarkResult& result) {
    cout << "  RESULTS:\n";
    cout << "    +-----------------+------------------+\n";
    cout << "    | Metric          | Value            |\n";
    cout << "    +-----------------+------------------+\n";
    cout << "    | Mean            | " << setw(12) << fixed << setprecision(2) << result.mean_ms << " ms |\n";
    cout << "    | Std Dev         | " << setw(12) << fixed << setprecision(2) << result.stddev_ms << " ms |\n";
    cout << "    | Min             | " << setw(12) << fixed << setprecision(2) << result.min_ms << " ms |\n";
    cout << "    | Max             | " << setw(12) << fixed << setprecision(2) << result.max_ms << " ms |\n";
    cout << "    +-----------------+------------------+\n";
    cout << endl;
}

double runSingleIteration(const string& inputFile) {
    auto start = chrono::high_resolution_clock::now();

    auto params = QuantLib::ext::make_shared<Parameters>();
    params->fromFile(inputFile);

    // Suppress output during benchmark
    OREApp ore(params, false);  // console = false
    ore.run();

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed = end - start;
    return elapsed.count();
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
        vector<double> times_ms;
        times_ms.reserve(BENCHMARK_ITERATIONS);

        for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
            double elapsed = runSingleIteration(inputFile);
            times_ms.push_back(elapsed);
            cout << "    Iteration " << setw(2) << (i + 1) << "/" << BENCHMARK_ITERATIONS
                 << ": " << fixed << setprecision(2) << elapsed << " ms\n";
        }
        cout << endl;

        // Results
        BenchmarkResult result = computeStats(times_ms);
        printResults(result);

        return 0;

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}
