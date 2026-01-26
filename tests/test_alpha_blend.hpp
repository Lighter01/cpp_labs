#pragma once

#include <cstddef>
#include <string>
#include "alpha_hist/alpha_blend.hpp"

namespace alpha_hist {

// ============================ Public Test Interface ============================ //

// Runs a single blend test: loads fg/bg, blends, saves into out_path
void run_blend_case(const ImageRGBA8& fg,
                    const ImageRGBA8& bg,
                    ImageRGBA8& out,
                    float global_opacity,
                    BlendMode mode,
                    Impl impl,
                    ExecMode exec_mode,
                    ThreadPool* pool,
                    size_t grain,
                    BlendStageTiming& timing,
                    BlendStageCycles& cycles);

// Runs required test set: A over B, C over D (and saves to data/results)
int run_alpha_blend_tests(const std::string& input_dir,
                          const std::string& output_dir,
                          float global_opacity,
                          BlendMode mode,
                          Impl impl,
                          ExecMode exec_mode,
                          size_t num_threads,
                          size_t grain,
                          int iterations,
                          bool save_results,
                          bool first_iter);

} // namespace alpha_hist
