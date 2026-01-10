#pragma once

#include <string>

#include "alpha_hist/histogram.hpp"

namespace alpha_hist {

// Runs a single histogram test: loads image, computes histogram, saves to CSV.
void run_histogram_case(const std::string& in_path,
                        const std::string& out_path,
                        Impl impl);

// Runs histogram tests for all PNGs in input_dir.
int run_histogram_tests(const std::string& input_dir,
                        const std::string& output_dir,
                        Impl impl);

} // namespace alpha_hist
