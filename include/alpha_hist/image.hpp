#pragma once

#include <cstdint>
#include <vector>
#include <stdexcept>

namespace alpha_hist {

    template <typename T, int Channels>
    struct Image {
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::vector<T> data;
        static constexpr int channels = Channels;
    };

    using ImageRGBA8 = Image<std::uint8_t, 4>;
    // using ImageRGB8  = Image8<3>;
    using ImageGray8 = Image<std::uint8_t, 1>;
    using ImageRGBAf = Image<float, 4>;
    
} // namespace alpha_hist