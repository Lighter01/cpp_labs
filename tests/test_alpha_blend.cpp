#include "tests/test_alpha_blend.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "alpha_hist/io.hpp"
#include "alpha_hist/thread_pool.hpp"
#include "alpha_hist/utils.hpp"

namespace alpha_hist {
namespace fs = std::filesystem;

// =============================== Formatting Helpers =============================== //

static std::string mode_name(BlendMode mode) {
    switch (mode) {
        case BlendMode::Over: return "over";
        case BlendMode::In:   return "in";
        case BlendMode::Out:  return "out";
        case BlendMode::Atop: return "atop";
        case BlendMode::Xor:  return "xor";
    }
    return "unknown";
}

static std::string impl_name(Impl impl) {
    return (impl == Impl::Scalar) ? "scalar" : "simd";
}

static std::string exec_mode_name(ExecMode exec_mode) {
    return (exec_mode == ExecMode::Seq) ? "seq" : "par";
}

static std::string opacity_tag(float opacity) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << opacity;
    return oss.str();
}

static std::string lowercase(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

// ============================= Single-Case Execution ============================= //

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
                    BlendStageCycles& cycles
                    )
{
    if (exec_mode == ExecMode::Par && !pool) {
        throw std::runtime_error("ThreadPool is required for parallel execution");
    }

    if (impl == Impl::Scalar) {
        if (exec_mode == ExecMode::Seq) {
            out = alpha_blend_pipeline_templ<Impl::Scalar, ExecMode::Seq>(
                fg, bg, global_opacity, mode, timing, cycles, nullptr, grain
            );
        } else {
            out = alpha_blend_pipeline_templ<Impl::Scalar, ExecMode::Par>(
                fg, bg, global_opacity, mode, timing, cycles, pool, grain
            );
        }
    } else {
        if (exec_mode == ExecMode::Seq) {
            out = alpha_blend_pipeline_templ<Impl::SIMD, ExecMode::Seq>(
                fg, bg, global_opacity, mode, timing, cycles, nullptr, grain
            );
        } else {
            out = alpha_blend_pipeline_templ<Impl::SIMD, ExecMode::Par>(
                fg, bg, global_opacity, mode, timing, cycles, pool, grain
            );
        }
    }
}

// ============================= Batch Test Orchestration ============================= //

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
                          bool first_iter)
{
    // ========== Validation & Setup ========== //
    if (exec_mode == ExecMode::Par && num_threads == 0) {
        throw std::runtime_error("num_threads must be >= 1");
    }
    if (grain == 0) {
        // Zero means "use default grain" inside the pipeline.
    }

    // ThreadPool lifetime spans all iterations to avoid per-iteration overhead.
    std::unique_ptr<ThreadPool> pool;
    if (exec_mode == ExecMode::Par) {
        pool = std::make_unique<ThreadPool>(num_threads);
    }
    ThreadPool* pool_ptr = pool ? pool.get() : nullptr;
    const size_t threads_used = (exec_mode == ExecMode::Par) ? num_threads : 1;

    // ========== Output Setup ========== //
    fs::path out_root = output_dir;
    fs::path csv_dir = out_root / "csv";
    fs::path img_dir = out_root / "images";
    fs::create_directories(csv_dir);
    fs::create_directories(img_dir);

    const std::string suffix = "_" + impl_name(impl) + "_" + mode_name(mode)
                             + "_" + exec_mode_name(exec_mode)
                            //  + "_t" + std::to_string(num_threads)
                             + "_g" + std::to_string(grain)
                             + "_a" + opacity_tag(global_opacity);
    const std::string perf_name = "blend_performance_" + suffix + ".csv";
    const std::string perf_path = (csv_dir / perf_name).string();
    std::ofstream perf_out(perf_path, std::ios::out | (first_iter ? std::ios::trunc : std::ios::app));
    if (!perf_out) {
        throw std::runtime_error("Failed to open output file: '" + perf_path + "'");
    }
    if (first_iter) {
        perf_out << "out_size,out_channels,impl,mode,exec_mode,num_threads,grain,preprocess_ns,blend_ns,postprocess_ns,total_ns,"
                 << "preprocess_cycles,blend_cycles,postprocess_cycles,total_cycles\n";
    }

    // ========== Input Discovery ========== //
    // Discover input images (paired in sorted order).
    std::vector<fs::path> images_pth;
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const fs::path path = entry.path();
        const std::string ext = lowercase(path.extension().string());
        if ((ext != ".png") && (ext != ".jpg") && (ext != ".jpeg")) {
            continue;
        }

        images_pth.push_back(path);
    }

    std::sort(images_pth.begin(), images_pth.end());

    // Process pairs: (0,1), (2,3), ...
    size_t processed = 0;
    for (size_t i = 0; i + 1 < images_pth.size(); i += 2) {
        // ========== Pair Preparation ========== //
        const fs::path fg_path = images_pth[i];
        const fs::path bg_path = images_pth[i + 1];
        const std::string out_name = fg_path.stem().string() + "_" + bg_path.stem().string()
                                   + suffix + ".png";
        const std::string out_path = (img_dir / out_name).string();

        // Read image once
        ImageRGBA8 fg = load_rgba8(fg_path);
        ImageRGBA8 bg = load_rgba8(bg_path);
        ImageRGBA8 out;

        // Resize once before running iterations
        if (fg.width != bg.width || fg.height != bg.height) {
            const std::uint64_t fg_area = static_cast<std::uint64_t>(fg.width)
                                        * static_cast<std::uint64_t>(fg.height);
            const std::uint64_t bg_area = static_cast<std::uint64_t>(bg.width)
                                        * static_cast<std::uint64_t>(bg.height);
            if (fg_area < bg_area) {
                fg.resize(bg.width, bg.height);
            } else {
                bg.resize(fg.width, fg.height);
            }
        }

        std::cout << "Blending:\n"
                  << "  FG: " << fg_path.string() << "\n"
                  << "  BG: " << bg_path.string() << "\n"
                  << "  ->  " << out_path << "\n";
        BlendStageTiming timing;
        BlendStageCycles cycles;
        for (int iter = 0; iter < iterations + 1; ++iter) {
            // ========== Timed Iteration ========== //
            std::cout << "Iteration " << iter << "\n";

            auto t0 = std::chrono::high_resolution_clock::now();

            run_blend_case(fg, bg, out, global_opacity, mode, impl,
                           exec_mode, pool_ptr, grain, timing, cycles);
            
            // First iteration is dropped before cache warm-up
            if (iter) {
                // ------------------------------ CSV Write ------------------------------ //
                const auto preprocess_ns =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(timing.preprocess).count();
                const auto blend_ns =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(timing.blend).count();
                const auto postprocess_ns =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(timing.postprocess).count();
                const auto total_ns = preprocess_ns + blend_ns + postprocess_ns;
                const std::uint64_t total_cycles = cycles.preprocess + cycles.blend + cycles.postprocess;
                
                perf_out << out.data.size() / out.channels << ","
                        << ImageRGBA8::channels << ","
                        << impl_name(impl) << ","
                        << mode_name(mode) << ","
                        << exec_mode_name(exec_mode) << ","
                        << threads_used << ","
                        << grain << ","
                        << preprocess_ns << ","
                        << blend_ns << ","
                        << postprocess_ns << ","
                        << total_ns << ","
                        << cycles.preprocess << ","
                        << cycles.blend << ","
                        << cycles.postprocess << ","
                        << total_cycles << "\n";
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

        if (save_results) {
            // Save a background-composited image for inspection.
            save_png(out_path, alpha_hist::get_background(out));
        }

        ++processed;
    }

    // ========== Completion ========== //
    if (images_pth.size() % 2 != 0) {
        std::cout << "Warning: odd number of PNG files in input directory. "
                  << "Last file is skipped: " << images_pth.back().string() << "\n";
    }

    std::cout << "Done. " << processed << " pair(s) processed. Results saved to: "
              << out_root.string() << "\n";
    return 0;
}

} // namespace alpha_hist
