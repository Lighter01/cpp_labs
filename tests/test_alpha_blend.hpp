#pragma once

#include <string>
#include "alpha_hist/alpha_blend.hpp"

namespace alpha_hist {

// Runs a single blend test: loads fg/bg, blends, saves into out_path
void run_blend_case(const std::string& fg_path,
                    const std::string& bg_path,
                    const std::string& out_path,
                    float global_opacity,
                    BlendMode mode,
                    Impl impl,
                    BlendStageTiming& timing,
                    BlendStageCycles& cycles);

// Runs your required test set: A over B, C over D (and saves to data/results)
int run_alpha_blend_tests(const std::string& input_dir,
                          const std::string& output_dir,
                          float global_opacity,
                          BlendMode mode,
                          Impl impl,
                          int iterations);

} // namespace alpha_hist
