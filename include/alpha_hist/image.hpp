#pragma once

#include <cstdint>
#include <vector>
#include <stdexcept>
#include <type_traits>

#include "stb/stb_image_resize2.h"

namespace alpha_hist {

    template <typename T, int Channels>
    struct Image {
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::vector<T> data;
        static constexpr int channels = Channels;
        void resize(std::uint32_t target_width, std::uint32_t target_height);
    };

    using ImageRGBA8 = Image<std::uint8_t, 4>;
    // using ImageRGB8  = Image8<3>;
    using ImageGray8 = Image<std::uint8_t, 1>;
    using ImageRGBAf = Image<float, 4>;
    
} // namespace alpha_hist

//====================================================================================================//

template <typename T, int Channels>
void alpha_hist::Image<T, Channels>::resize(std::uint32_t target_width, std::uint32_t target_height) {
    static_assert(std::is_same_v<T, std::uint8_t>,
                  "Image::resize is only supported for uint8_t images.");
    static_assert(Channels == 1 || Channels == 4,
                  "Image::resize supports only 1 or 4 channel images.");

    if (target_width == width && target_height == height) {
        return;
    }

    const size_t out_size = static_cast<size_t>(target_width)
                          * static_cast<size_t>(target_height)
                          * static_cast<size_t>(Channels);
    std::vector<T> out(out_size);

    if (width == 0 || height == 0 || data.empty()) {
        width = target_width;
        height = target_height;
        data = std::move(out);
        return;
    }

    const int in_stride = static_cast<int>(width * Channels * sizeof(T));
    const int out_stride = static_cast<int>(target_width * Channels * sizeof(T));
    const stbir_pixel_layout layout = (Channels == 4) ? STBIR_RGBA : STBIR_1CHANNEL;

    unsigned char* resized = stbir_resize_uint8_srgb(
        reinterpret_cast<const unsigned char*>(data.data()),
        static_cast<int>(width),
        static_cast<int>(height),
        in_stride,
        reinterpret_cast<unsigned char*>(out.data()),
        static_cast<int>(target_width),
        static_cast<int>(target_height),
        out_stride,
        layout
    );

    if (!resized) {
        throw std::runtime_error("stbir_resize_uint8_srgb failed");
    }

    width = target_width;
    height = target_height;
    data = std::move(out);
}
