#include "tests/test_histogram.hpp"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

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

void run_histogram_case(const std::string& in_path,
                        const std::string& out_path,
                        Impl impl)
{
    ImageGray8 img = load_gray8(in_path);

    std::array<std::uint32_t, 256> hist{};
    if (impl == Impl::Scalar) {
        histogram_scalar(img, hist);
    } else {
        histogram_simd(img, hist);
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
                        Impl impl)
{
    fs::create_directories(output_dir);

    const std::string impl_tag = impl_name(impl);
    size_t processed = 0;

    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const fs::path path = entry.path();
        const std::string ext = lowercase(path.extension().string());
        if (ext != ".png") {
            continue;
        }

        const std::string stem = path.stem().string();
        const std::string out_name = stem + "_hist_" + impl_tag + ".csv";
        const std::string out_path = (fs::path(output_dir) / out_name).string();

        std::cout << "Histogram:\n"
                  << "  IN:  " << path.string() << "\n"
                  << "  OUT: " << out_path << "\n";

        run_histogram_case(path.string(), out_path, impl);
        ++processed;
    }

    std::cout << "Done. " << processed << " file(s) processed. Results saved to: "
              << output_dir << "\n";
    return 0;
}

} // namespace alpha_hist
