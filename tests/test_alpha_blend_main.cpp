#include <algorithm>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "tests/test_alpha_blend.hpp"

using alpha_hist::BlendMode;
using alpha_hist::ExecMode;
using alpha_hist::Impl;

// =============================== Parsing Helpers =============================== //

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

static ExecMode parse_exec_mode(const std::string& s) {
    if (s == "seq" || s == "sequential") return ExecMode::Seq;
    if (s == "par" || s == "parallel") return ExecMode::Par;
    throw std::runtime_error("Unknown exec mode: " + s);
}

static size_t parse_positive_size(const std::string& s) {
    size_t value = std::stoul(s);
    if (value < 1) {
        throw std::runtime_error("value must be >= 1");
    }
    return value;
}

static std::vector<size_t> parse_thread_list(const std::string& s) {
    std::vector<size_t> out;
    // Split by commas, then expand any A-B ranges.
    size_t start = 0;
    while (start <= s.size()) {
        size_t comma = s.find(',', start);
        size_t end = (comma == std::string::npos) ? s.size() : comma;
        std::string token = s.substr(start, end - start);
        if (token.empty()) {
            throw std::runtime_error("empty entry in --num-threads list");
        }

        // Accept either a single value or a range like "2-8".
        size_t dash = token.find('-');
        if (dash != std::string::npos) {
            size_t a = parse_positive_size(token.substr(0, dash));
            size_t b = parse_positive_size(token.substr(dash + 1));
            if (a > b) {
                std::swap(a, b);
            }
            // Expand range inclusively.
            for (size_t v = a; v <= b; ++v) {
                out.push_back(v);
            }
        } else {
            out.push_back(parse_positive_size(token));
        }

        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }

    if (out.empty()) {
        throw std::runtime_error("no values provided for --num-threads");
    }
    return out;
}

int main(int argc, char** argv) {
    // ============================ Defaults / CLI State ============================ //

    // Defaults
    std::string input_dir  = "data/png";
    std::string output_dir = "results/blend_results";
    float opacity = 1.0f;
    BlendMode mode = BlendMode::Over;
    Impl impl = Impl::Scalar;
    ExecMode exec_mode = ExecMode::Seq;
    const size_t default_threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<size_t> num_threads_list{default_threads};
    size_t grain = 0;
    int iterations = 1;
    bool save_results = false;

    // Usage:
    // ./test_alpha_blend [input_dir] [output_dir] [opacity] [mode] [impl] [iterations] [save_results]
    //                  [--exec-mode seq|par] [--num-threads N|A-B|N1,N2,...] [--grain N]
    // Example:
    // ./test_alpha_blend data/png data/results 1.0 over scalar 10 --exec-mode par --num-threads 4
    // ./test_alpha_blend data/png data/results 1.0 over simd 10 --exec-mode par --num-threads 2-8
    // ./test_alpha_blend data/png data/results 1.0 over simd 10 --exec-mode par --num-threads 1,2,4,8
    // ./test_alpha_blend data/png data/results 1.0 over simd 10 --exec-mode par --num-threads 4 --grain 4096
    try {
        // ============================= Argument Parsing ============================= //
        std::vector<std::string> positional;
        positional.reserve(static_cast<size_t>(argc));

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--exec-mode") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("--exec-mode requires a value");
                }
                exec_mode = parse_exec_mode(argv[++i]);
                continue;
            }
            if (arg == "--num-threads") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("--num-threads requires a value");
                }
                num_threads_list = parse_thread_list(argv[++i]);
                continue;
            }
            if (arg == "--grain") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("--grain requires a value");
                }
                grain = parse_positive_size(argv[++i]);
                continue;
            }
            positional.push_back(std::move(arg));
        }

        if (positional.size() >= 1) input_dir  = positional[0];
        if (positional.size() >= 2) output_dir = positional[1];
        if (positional.size() >= 3) opacity    = std::stof(positional[2]);
        if (positional.size() >= 4) mode       = parse_mode(positional[3]);
        if (positional.size() >= 5) impl       = parse_impl(positional[4]);
        if (positional.size() >= 6) iterations = std::stoi(positional[5]);
        if (positional.size() >= 7) save_results = std::stoi(positional[6]) != 0;

        // ============================== Test Execution ============================== //
        int rc = 0;
        bool first_iter = true;
        for (size_t num_threads : num_threads_list) {
            rc = alpha_hist::run_alpha_blend_tests(
                input_dir,
                output_dir,
                opacity,
                mode,
                impl,
                exec_mode,
                num_threads,
                grain,
                iterations,
                save_results,
                first_iter
            );
            if (rc != 0) {
                return rc;
            }
            first_iter = false;
        }
        return rc;
    } catch (const std::exception& e) {
        std::cerr << "test_alpha_blend failed: " << e.what() << "\n";
        std::cerr << "Usage:\n"
                  << "  ./test_alpha_blend [input_dir] [output_dir] [opacity] [mode] [impl] [iterations] [save_results]\n"
                  << "                  [--exec-mode seq|par] [--num-threads N|A-B|N1,N2,...] [--grain N]\n"
                  << "  mode: over|in|out|atop|xor\n"
                  << "  impl: scalar|simd\n"
                  << "  exec-mode: seq|par\n"
                  << "  num-threads: positive integer or range/list (required for exec-mode=par)\n"
                  << "  grain: positive integer (optional; 0 uses default)\n"
                  << "  iterations: positive integer\n"
                  << "  save_results: non-negative integer\n";
        return 1;
    }
}
