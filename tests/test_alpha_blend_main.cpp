#include <iostream>
#include <string>

#include "tests/test_alpha_blend.hpp"

using alpha_hist::BlendMode;
using alpha_hist::Impl;

static BlendMode parse_mode(const std::string& s) {
    if (s == "over") return BlendMode::Over;
    if (s == "in")   return BlendMode::In;
    if (s == "out")  return BlendMode::Out;
    if (s == "atop") return BlendMode::Atop;
    if (s == "xor")  return BlendMode::Xor;
    throw std::runtime_error("Unknown blend mode: " + s);
}

static Impl parse_impl(const std::string& s) {
    if (s == "scalar") return Impl::Scalar;
    if (s == "simd")   return Impl::SIMD;
    throw std::runtime_error("Unknown impl: " + s);
}

int main(int argc, char** argv) {
    // Defaults
    std::string input_dir  = "data/png";
    std::string output_dir = "results/blend_results";
    float opacity = 1.0f;
    BlendMode mode = BlendMode::Over;
    Impl impl = Impl::Scalar;
    int iterations = 1;
    bool save_results = false;

    // Usage:
    // ./test_alpha_blend [input_dir] [output_dir] [opacity] [mode] [impl] [iterations] [save_results]
    // Example:
    // ./test_alpha_blend data/png data/results 1.0 over scalar 10
    try {
        if (argc >= 2) input_dir  = argv[1];
        if (argc >= 3) output_dir = argv[2];
        if (argc >= 4) opacity    = std::stof(argv[3]);
        if (argc >= 5) mode       = parse_mode(argv[4]);
        if (argc >= 6) impl       = parse_impl(argv[5]);
        if (argc >= 7) iterations = std::stoi(argv[6]);
        if (argc >= 8) save_results = std::stoi(argv[7]) != 0;

        return alpha_hist::run_alpha_blend_tests(input_dir, output_dir, opacity, mode, impl, iterations, save_results);
    } catch (const std::exception& e) {
        std::cerr << "test_alpha_blend failed: " << e.what() << "\n";
        std::cerr << "Usage:\n"
                  << "  ./test_alpha_blend [input_dir] [output_dir] [opacity] [mode] [impl] [iterations] [save_results]\n"
                  << "  mode: over|in|out|atop|xor\n"
                  << "  impl: scalar|simd\n"
                  << "  iterations: positive integer\n"
                  << "  save_results: non-negative integer\n";
        return 1;
    }
}
