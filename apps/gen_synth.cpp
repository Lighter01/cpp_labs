#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "alpha_hist/io.hpp"
#include "alpha_hist/utils.hpp"

namespace {

    struct PixelSpec {
        bool is_range = false;
        std::uint64_t start = 0;
        std::uint64_t end = 0;
    };

    struct ImageSize {
        std::uint32_t width = 0;
        std::uint32_t height = 0;
    };

    void print_usage(const char* prog) {
        std::cout << "Usage: " << prog << " <ratio a:b> <pixels|start-end> [--step N]\n"
                  << "  ratio: height:width (e.g., 1:2, 2:1, 1:1.618)\n"
                  << "  pixels: total pixel count (e.g., 4096) or range (e.g., 4096-16384)\n"
                  << "  --step: multiplier for ranges (default 2)\n";
    }

    double parse_double(const std::string& s) {
        std::size_t idx = 0;
        double val = std::stod(s, &idx);
        if (idx != s.size() || !std::isfinite(val)) {
            throw std::invalid_argument("invalid number: " + s);
        }
        return val;
    }

    std::uint64_t parse_u64(const std::string& s) {
        std::size_t idx = 0;
        unsigned long long val = std::stoull(s, &idx);
        if (idx != s.size()) {
            throw std::invalid_argument("invalid integer: " + s);
        }
        return static_cast<std::uint64_t>(val);
    }

    bool parse_ratio(const std::string& s, double& a, double& b) {
        std::size_t sep = s.find(':');
        if (sep == std::string::npos) {
            return false;
        }
        std::string a_str = s.substr(0, sep);
        std::string b_str = s.substr(sep + 1);
        if (a_str.empty() || b_str.empty()) {
            return false;
        }
        a = parse_double(a_str);
        b = parse_double(b_str);
        return a > 0.0 && b > 0.0;
    }

    PixelSpec parse_pixels(const std::string& s) {
        std::size_t sep = s.find('-');
        PixelSpec spec;
        if (sep == std::string::npos) {
            spec.start = parse_u64(s);
            spec.end = spec.start;
            spec.is_range = false;
        } else {
            if (sep == 0 || sep + 1 >= s.size()) {
                throw std::invalid_argument("invalid range: " + s);
            }
            spec.start = parse_u64(s.substr(0, sep));
            spec.end = parse_u64(s.substr(sep + 1));
            spec.is_range = true;
        }
        if (spec.start == 0 || spec.end == 0 || spec.start > spec.end) {
            throw std::invalid_argument("invalid pixel range: " + s);
        }
        return spec;
    }

    std::vector<std::uint64_t> expand_sizes(const PixelSpec& spec, double step) {
        std::vector<std::uint64_t> sizes;
        if (!spec.is_range) {
            sizes.push_back(spec.start);
            return sizes;
        }
        if (!(step > 1.0)) {
            throw std::invalid_argument("step must be > 1 for ranges");
        }
        std::uint64_t size = spec.start;
        while (size <= spec.end) {
            sizes.push_back(size);
            double next_f = std::ceil(static_cast<double>(size) * step);
            std::uint64_t next = static_cast<std::uint64_t>(next_f);
            if (next <= size) {
                throw std::invalid_argument("step does not increase size");
            }
            size = next;
        }
        return sizes;
    }

    ImageSize estimate_size(std::uint64_t pixels, double a, double b) {
        double height_f = std::ceil(std::sqrt(static_cast<double>(pixels) * a / b));
        double width_f = std::ceil(std::sqrt(static_cast<double>(pixels) * b / a));
        std::uint32_t height = static_cast<std::uint32_t>(std::max(1.0, height_f));
        std::uint32_t width = static_cast<std::uint32_t>(std::max(1.0, width_f));
        return {width, height};
    }

    struct PatternSpec {
        alpha_hist::GenMode mode;
        const char* suffix;
    };

} // namespace

int main(int argc, char** argv) {
    namespace fs = std::filesystem;

    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        std::string ratio_arg = argv[1];
        std::string pixels_arg = argv[2];
        double step = 2.0;
        bool step_set = false;

        for (int i = 3; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--step") {
                if (i + 1 >= argc) {
                    std::cerr << "Missing value for --step\n";
                    return 1;
                }
                step = parse_double(argv[++i]);
                step_set = true;
            } else {
                std::cerr << "Unknown argument: " << arg << "\n";
                print_usage(argv[0]);
                return 1;
            }
        }

        double ratio_a = 0.0;
        double ratio_b = 0.0;
        if (!parse_ratio(ratio_arg, ratio_a, ratio_b)) {
            throw std::invalid_argument("invalid ratio: " + ratio_arg);
        }

        PixelSpec pixels = parse_pixels(pixels_arg);
        if (!pixels.is_range && step_set) {
            std::cerr << "Warning: --step is ignored for single sizes\n";
        }

        std::vector<std::uint64_t> sizes = expand_sizes(pixels, step);
        if (sizes.empty()) {
            throw std::invalid_argument("no sizes generated");
        }

        std::string out_dir = "data/png_test";
        fs::create_directories(out_dir);

        const std::array<PatternSpec, 12> patterns = {{
            {alpha_hist::GenMode::Stripe45, "stripe45"},
            {alpha_hist::GenMode::Stripe135, "stripe135"},
            {alpha_hist::GenMode::Rectangle, "rect"},
            {alpha_hist::GenMode::Circle, "circle"},
            {alpha_hist::GenMode::SineBands, "sine"},
            {alpha_hist::GenMode::Lissajous, "lissajous"},
            {alpha_hist::GenMode::Spiral, "spiral"},
            {alpha_hist::GenMode::RoundedRects, "roundrect"},
            {alpha_hist::GenMode::Concentric, "concentric"},
            {alpha_hist::GenMode::Blobs, "blobs"},
            {alpha_hist::GenMode::RandomForms, "forms"},
            {alpha_hist::GenMode::GaussianNoise, "gauss"},
        }};

        std::size_t total = sizes.size() * patterns.size();
        int index_width = static_cast<int>(
            std::max<std::size_t>(4, std::to_string(total).size())
        );

        std::size_t index = 0;
        for (std::uint64_t pixels_count : sizes) {
            ImageSize dims = estimate_size(pixels_count, ratio_a, ratio_b);
            for (const auto& pattern : patterns) {
                alpha_hist::ImageRGBA8 img = alpha_hist::generate_test_image_rgba(
                    dims.width, dims.height, pattern.mode);
                std::ostringstream name;
                name << std::setw(index_width) << std::setfill('0') << index++ << "_"
                     << pattern.suffix << ".png";
                alpha_hist::save_png(out_dir + "/" + name.str(), img);
            }
        }

        std::cout << "Saved " << total << " images to " << out_dir << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "gen_synth failed: " << e.what() << "\n";
        return 1;
    }
}
