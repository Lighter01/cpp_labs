#include "alpha_hist/color_lut.hpp"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>

namespace alpha_hist {

    float _srgb_to_linear_exact(float cs) {
        cs = std::ranges::clamp(cs, 0.0f, 1.0f);
        if (cs <= 0.04045f) return cs / 12.92f;
        return std::pow((cs + 0.055f) / 1.055f, 2.4f);
    }

    float _linear_to_srgb_exact(float cl) {
        cl = std::ranges::clamp(cl, 0.0f, 1.0f);
        if (cl <= 0.0031308f) return 12.92f * cl;
        return 1.055f * std::pow(cl, 1.0f / 2.4f) - 0.055f;
    }

    float _srgb_to_linear_approx(float cs, float gamma) {
        cs = std::ranges::clamp(cs, 0.0f, 1.0f);
        return std::pow(cs, gamma);
    }

    float _linear_to_srgb_approx(float cl, float gamma) {
        cl = std::ranges::clamp(cl, 0.0f, 1.0f);
        return std::pow(cl, 1.0f / gamma);
    }


    float u8_to_float32(std::uint8_t x) {
        constexpr float inv255 = 1.0f / 255.0f;
        return x * inv255;
    }

    // std::clamp or custom clamp
    std::uint8_t float32_to_u8(float x) {
        x = std::ranges::clamp(x, 0.0f, 1.0f);
        std::int32_t v = static_cast<std::int32_t>(x * 256.0f);
        return static_cast<std::uint8_t>(std::ranges::clamp(v, 0, 255));
    }

    static void build_lut(ColorLUT& lut, int linear_table_size, bool use_exect_srgb) {
        if (linear_table_size < 2) throw std::invalid_argument("linear_table_size must be >= 2");

        // build sRGB -> linear (256 bins)
        for (int i = 0; i < ColorLUT::kSrgbSize; ++i) {
            float cs = u8_to_float32(i);
            lut.srgb_to_linear[i] = use_exect_srgb ? _srgb_to_linear_exact(cs) : _srgb_to_linear_approx(cs);
        }

        lut.linear_size = linear_table_size;
        lut.linear_to_srgb.resize(static_cast<size_t>(linear_table_size));

        for (int j = 0; j < linear_table_size; ++j) {
            float cl = static_cast<float>(j) / static_cast<float>(linear_table_size - 1);
            float cs = use_exect_srgb ? _linear_to_srgb_exact(cl) : _linear_to_srgb_approx(cl);
            // Конвертирую, потому что меняю логику в linear_to_srgb для потокобезопасного кода
            lut.linear_to_srgb[static_cast<size_t>(j)] = static_cast<std::uint32_t>(float32_to_u8(cs));
        }
    }

    const ColorLUT& get_color_lut(int linear_table_size, bool use_exact_srgb) {
        static std::once_flag init_flag;
        static int init_size = 0;
        static bool init_exact = false;

        static ColorLUT lut;

        std::call_once(init_flag, [=] {
            init_size = linear_table_size;
            init_exact = use_exact_srgb;
            build_lut(lut, linear_table_size, use_exact_srgb);
        });

        if (linear_table_size != init_size || use_exact_srgb != init_exact) {
            throw std::runtime_error(
                "get_color_lut(): LUT configuration changed at runtime. "
                "Expected size=" + std::to_string(init_size) +
                ", exact=" + std::to_string(init_exact) +
                " but got size=" + std::to_string(linear_table_size) +
                ", exact=" + std::to_string(use_exact_srgb)
            );
        }

        return lut;
    }

}