#include "tests/test_alpha_blend.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "alpha_hist/io.hpp"
#include "alpha_hist/utils.hpp"

namespace alpha_hist {
namespace fs = std::filesystem;

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

void run_blend_case(const std::string& fg_path,
                    const std::string& bg_path,
                    const std::string& out_path,
                    float global_opacity,
                    BlendMode mode,
                    Impl impl)
{
    ImageRGBA8 fg = load_rgba8(fg_path);
    ImageRGBA8 bg = load_rgba8(bg_path);

    if (fg.width != bg.width || fg.height != bg.height) {
        throw std::runtime_error("Input images must have same size: '" + fg_path + "' and '" + bg_path + "'");
    }

    ImageRGBA8 out;
    if (impl == Impl::Scalar) {
        out = alpha_blend_pipeline_templ<Impl::Scalar>(fg, bg, global_opacity, mode);
    } else {
        out = alpha_blend_pipeline_templ<Impl::SIMD>(fg, bg, global_opacity, mode);
    }

    save_png(out_path, alpha_hist::get_background(out));
}

int run_alpha_blend_tests(const std::string& input_dir,
                          const std::string& output_dir,
                          float global_opacity,
                          BlendMode mode,
                          Impl impl)
{
    fs::create_directories(output_dir);

    const std::string suffix = "_" + impl_name(impl) + "_" + mode_name(mode)
                             + "_a" + std::to_string(global_opacity);

    // Required test pairs:
    // A with B, C with D
    const std::string A = (fs::path(input_dir) / "A.png").string();
    const std::string B = (fs::path(input_dir) / "B.png").string();
    const std::string C = (fs::path(input_dir) / "C.png").string();
    const std::string D = (fs::path(input_dir) / "D.png").string();
    const std::string sayori = (fs::path(input_dir) / "sayori.png").string();
    const std::string monika = (fs::path(input_dir) / "monika_end.png").string();
    const std::string eva1 = (fs::path(input_dir) / "eva_1.png").string();
    const std::string eva2 = (fs::path(input_dir) / "eva_2.png").string();

    const std::string outAB = (fs::path(output_dir) / ("AB" + suffix + ".png")).string();
    const std::string outCD = (fs::path(output_dir) / ("CD" + suffix + ".png")).string();
    const std::string outSM = (fs::path(output_dir) / ("SayoriMonika" + suffix + ".png")).string();
    const std::string outEVA = (fs::path(output_dir) / ("eva_12" + suffix + ".png")).string();

    std::cout << "Blending:\n"
              << "  FG: " << A << "\n"
              << "  BG: " << B << "\n"
              << "  ->  " << outAB << "\n";
    run_blend_case(A, B, outAB, global_opacity, mode, impl);

    std::cout << "Blending:\n"
              << "  FG: " << C << "\n"
              << "  BG: " << D << "\n"
              << "  ->  " << outCD << "\n";
    run_blend_case(C, D, outCD, global_opacity, mode, impl);

    std::cout << "Blending:\n"
              << "  FG: " << sayori << "\n"
              << "  BG: " << monika << "\n"
              << "  ->  " << outSM << "\n";
    run_blend_case(sayori, monika, outSM, global_opacity, mode, impl);

    std::cout << "Blending:\n"
              << "  FG: " << eva1 << "\n"
              << "  BG: " << eva2 << "\n"
              << "  ->  " << outEVA << "\n";
    run_blend_case(eva1, eva2, outEVA, global_opacity, mode, impl);

    std::cout << "Done. Results saved to: " << output_dir << "\n";
    return 0;
}

} // namespace alpha_hist
