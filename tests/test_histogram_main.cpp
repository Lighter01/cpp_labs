#include <iostream>
#include <string>

#include "tests/test_histogram.hpp"

using alpha_hist::Impl;

static Impl parse_impl(const std::string& s) {
    if (s == "scalar") return Impl::Scalar;
    if (s == "simd")   return Impl::SIMD;
    throw std::runtime_error("Unknown impl: " + s);
}

int main(int argc, char** argv) {
    // Defaults
    std::string input_dir  = "data/png";
    std::string output_dir = "results/hist_results";
    Impl impl = Impl::Scalar;
    int iterations = 1;

    // Usage:
    // ./test_histogram [input_dir] [output_dir] [impl] [iterations]
    // Example:
    // ./test_histogram data/png data/tests/hist_results simd 10
    try {
        if (argc >= 2) input_dir  = argv[1];
        if (argc >= 3) output_dir = argv[2];
        if (argc >= 4) impl       = parse_impl(argv[3]);
        if (argc >= 5) iterations = std::stoi(argv[4]);
        
        bool first_iter = true;
        return alpha_hist::run_histogram_tests(input_dir, output_dir, impl, iterations, first_iter);
        first_iter = false;
    } catch (const std::exception& e) {
        std::cerr << "test_histogram failed: " << e.what() << "\n";
        std::cerr << "Usage:\n"
                  << "  ./test_histogram [input_dir] [output_dir] [impl] [iterations]\n"
                  << "  impl: scalar|simd\n"
                  << "  iterations: positive integer\n";
        return 1;
    }
}
