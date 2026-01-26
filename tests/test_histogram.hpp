#pragma once

#include <chrono>
#include <cstdint>
#include <string>

#include "alpha_hist/histogram.hpp"

namespace alpha_hist {

// Runs a single histogram test: loads image, computes histogram, saves to CSV.
void run_histogram_case(const ImageGray8& img,
                        const std::string& out_path,
                        Impl impl,
                        std::chrono::high_resolution_clock::duration& timing,
                        std::uint64_t& cycles);

// Runs histogram tests for all PNGs in input_dir.
int run_histogram_tests(const std::string& input_dir,
                        const std::string& output_dir,
                        Impl impl,
                        int iterations,
                        bool first_iter);

} // namespace alpha_hist
