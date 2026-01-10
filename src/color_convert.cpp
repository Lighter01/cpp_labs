#include "alpha_hist/color_convert.hpp"

#include <cstdint>
#include <algorithm>

namespace alpha_hist {

    void srgb_to_linear(const ImageRGBA8& in, ImageRGBAf& out, bool use_exact_srgb) {
        size_t img_size = static_cast<size_t>(in.width * in.height);
        out.width = in.width;
        out.height = in.height;
        out.data.resize(in.data.size());

        const ColorLUT& lut = get_color_lut(4096, use_exact_srgb);

        for (size_t i = 0; i < img_size; ++i) {
            size_t idx = i * 4;
            
            for (size_t c = 0; c < 3; ++c) {
                out.data[idx + c] = lut.srgb_to_linear[in.data[idx + c]];
            }

            out.data[idx + 3] = u8_to_float32(in.data[idx + 3]);
        }
    }

    void linear_to_srgb(const ImageRGBAf& in, ImageRGBA8& out, bool use_exact_srgb) {
        size_t img_size = static_cast<size_t>(in.width * in.height);
        out.width = in.width;
        out.height = in.height;
        out.data.resize(in.data.size());

        const ColorLUT& lut = get_color_lut(4096, use_exact_srgb);
        int N = lut.linear_size;

        for (size_t i = 0; i < img_size; ++i) {
            size_t idx = i * 4;
            
            for (size_t c = 0; c < 3; ++c) {
                float cl = std::ranges::clamp(in.data[idx + c], 0.0f, 1.0f);
                int idx_lut = static_cast<int>(cl * (N - 1) + 0.5f);
                out.data[idx + c] = lut.linear_to_srgb[idx_lut];
            }

            out.data[idx + 3] = float32_to_u8(in.data[idx + 3]);
        }
    }


    void premultiply_inplace(ImageRGBAf& in) {
        size_t N = static_cast<size_t>(in.height) * static_cast<size_t>(in.width);

        for (size_t i = 0; i < N; ++i) {
            size_t idx = i * 4;

            float alpha = in.data[idx + 3];
            alpha = std::ranges::clamp(alpha, 0.0f, 1.0f);
            in.data[idx]     *= alpha;
            in.data[idx + 1] *= alpha;
            in.data[idx + 2] *= alpha;
        }
    }

    void revert_premultiply_inplace(ImageRGBAf& in) {
        constexpr float EPS = 1e-6f;
        size_t N = static_cast<size_t>(in.height) * static_cast<size_t>(in.width);

        for (size_t i = 0; i < N; ++i) {
            size_t idx = i * 4;

            float alpha = in.data[idx + 3];
            alpha = std::ranges::clamp(alpha, 0.0f, 1.0f);
            if (alpha <= EPS) {
                in.data[idx]     = 0.0f;
                in.data[idx + 1] = 0.0f;
                in.data[idx + 2] = 0.0f;
                continue;
            }
            float inv_alpha = 1.0f / alpha;
            in.data[idx]     *= inv_alpha;
            in.data[idx + 1] *= inv_alpha;
            in.data[idx + 2] *= inv_alpha;
        }
    }

}