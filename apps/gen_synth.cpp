#include <iostream>
#include <string>
#include <filesystem>

#include "alpha_hist/io.hpp"
#include "alpha_hist/utils.hpp"

int main(int argc, char** argv) {
    namespace fs = std::filesystem;

    // Default output directory
    std::string out_dir = "data/png";
    std::string results_dir = "data/results";
    if (argc >= 2) out_dir = argv[1];
    if (argc >= 3) results_dir = argv[2];

    try {
        fs::create_directories(out_dir);
        fs::create_directories(results_dir);

        constexpr size_t W = 300;
        constexpr size_t H = 300;

        // Generate synthetic RGBA images (with alpha)
        alpha_hist::ImageRGBA8 A = alpha_hist::generate_test_image_rgba(W, H, alpha_hist::GenMode::Stripe45);
        alpha_hist::ImageRGBA8 B = alpha_hist::generate_test_image_rgba(W, H, alpha_hist::GenMode::Stripe135);
        alpha_hist::ImageRGBA8 C = alpha_hist::generate_test_image_rgba(W, H, alpha_hist::GenMode::Rectangle);
        alpha_hist::ImageRGBA8 D = alpha_hist::generate_test_image_rgba(W, H, alpha_hist::GenMode::Circle);

        alpha_hist::save_png(out_dir + "/A.png", A);
        alpha_hist::save_png(out_dir + "/B.png", B);
        alpha_hist::save_png(out_dir + "/C.png", C);
        alpha_hist::save_png(out_dir + "/D.png", D);
        // Composite onto checkerboard background (opaque) and save
        alpha_hist::save_png(results_dir + "/A_preview.png", alpha_hist::get_background(A));
        alpha_hist::save_png(results_dir + "/B_preview.png", alpha_hist::get_background(B));
        alpha_hist::save_png(results_dir + "/C_preview.png", alpha_hist::get_background(C));
        alpha_hist::save_png(results_dir + "/D_preview.png", alpha_hist::get_background(D));

        std::cout << "Saved:\n"
                  << "  " << (out_dir + "/A_preview.png") << "\n"
                  << "  " << (out_dir + "/B_preview.png") << "\n"
                  << "  " << (out_dir + "/C_preview.png") << "\n"
                  << "  " << (out_dir + "/D_preview.png") << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "gen_synth failed: " << e.what() << "\n";
        return 1;
    }
}
