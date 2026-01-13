#include "tests/test_histogram.hpp"

#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <immintrin.h>

#include "alpha_hist/io.hpp"

namespace alpha_hist {
namespace fs = std::filesystem;

static std::string impl_name(Impl impl) {
    return (impl == Impl::Scalar) ? "scalar" : "simd";
}

static std::string lowercase(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

void run_histogram_case(const ImageGray8& img,
                        const std::string& out_path,
                        Impl impl,
                        std::chrono::high_resolution_clock::duration& timing,
                        std::uint64_t& cycles)
{
    std::array<std::uint32_t, 256> hist{};
    if (impl == Impl::Scalar) {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::uint64_t c0 = static_cast<std::uint64_t>(_rdtsc());
        histogram_scalar(img, hist);
        std::uint64_t c1 = static_cast<std::uint64_t>(_rdtsc());
        auto t1 = std::chrono::high_resolution_clock::now();
        timing = t1 - t0;
        cycles = c1 - c0;
    } else {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::uint64_t c0 = static_cast<std::uint64_t>(_rdtsc());
        histogram_simd(img, hist);
        std::uint64_t c1 = static_cast<std::uint64_t>(_rdtsc());
        auto t1 = std::chrono::high_resolution_clock::now();
        timing = t1 - t0;
        cycles = c1 - c0;
    }

    std::ofstream out(out_path);
    if (!out) {
        throw std::runtime_error("Failed to open output file: '" + out_path + "'");
    }

    out << "value,count\n";
    for (size_t i = 0; i < hist.size(); ++i) {
        out << i << "," << hist[i] << "\n";
    }
}

int run_histogram_tests(const std::string& input_dir,
                        const std::string& output_dir,
                        Impl impl,
                        int iterations)
{
    fs::create_directories(output_dir);

    const std::string impl_tag = impl_name(impl);
    const std::string perf_name = "histogram_performance_" + impl_tag + ".csv";
    const std::string perf_path = (fs::path(output_dir) / perf_name).string();
    std::ofstream perf_out(perf_path, std::ios::out | std::ios::app); // trunc
    if (!perf_out) {
        throw std::runtime_error("Failed to open output file: '" + perf_path + "'");
    }
    perf_out << "out_path,impl,timing_ns,cycles\n";
    size_t processed = 0;

    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const fs::path path = entry.path();
        const std::string ext = lowercase(path.extension().string());
        if ((ext != ".png") && (ext != ".jpg") && (ext != ".jpeg")) {
            continue;
        }

        const std::string stem = path.stem().string();
        const std::string out_name = stem + "_hist_" + impl_tag + ".csv";
        const std::string out_path = (fs::path(output_dir) / out_name).string();

        std::cout << "Histogram:\n"
                  << "  IN:  " << path.string() << "\n"
                  << "  OUT: " << out_path << "\n";

        // Read image once
        ImageGray8 img = load_gray8(path.string());

        std::chrono::high_resolution_clock::duration timing{};
        std::uint64_t cycles = 0;
        for (int iter = 0; iter < iterations + 1; ++iter) {
            std::cout << "Iteration " << iter << "\n";

            auto t0 = std::chrono::high_resolution_clock::now();

            run_histogram_case(img, out_path, impl, timing, cycles);

            // First iteration is dropped before cache warm-up
            if (iter) {
                const auto timing_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(timing).count();
                perf_out << out_path << ","
                         << impl_tag << ","
                         << timing_ns << ","
                         << cycles << "\n";
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            const auto iter_time_sec = 
                std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count();
            const auto iter_time_millisec = 
                std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            std::cout << "Iteration finished. Elapsed time: " << iter_time_sec << " sec. ("
                      << iter_time_millisec << " ms.)\n";
            std::cout <<  std::string(50, '=') << '\n';
        }
        ++processed;
    }

    std::cout << "Done. " << processed << " file(s) processed. Results saved to: "
              << output_dir << "\n";
    return 0;
}

} // namespace alpha_hist
