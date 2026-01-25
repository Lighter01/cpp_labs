#pragma once
#include <array>
#include <cstdint>
#include <vector>


namespace alpha_hist {

    struct ColorLUT {
        static constexpr int kSrgbSize = 256;

        std::array<float, kSrgbSize> srgb_to_linear{};

        std::vector<std::uint32_t> linear_to_srgb;
        int linear_size = 0;
    };

    float _srgb_to_linear_exact(float cs);

    float _linear_to_srgb_exact(float cl);

    float _srgb_to_linear_approx(float cs, float gamma = 2.2f);

    float _linear_to_srgb_approx(float cl, float gamma = 2.2f);

    float u8_to_float32(std::uint8_t x);

    std::uint8_t float32_to_u8(float x);

    const ColorLUT& get_color_lut(int linear_table_size = 4096, bool use_exact_srgb = true);

} // namespace alpha_hist