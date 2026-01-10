#include "alpha_hist/utils.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <algorithm>

namespace alpha_hist {

    std::vector<std::uint8_t> gen_checkerboard_rgba(size_t w, size_t h, std::uint8_t value, size_t checker_size) {
        if (checker_size == 0) throw std::invalid_argument(
            "gen_checkerboard_rgb: checker_size should be greater than 0. Got '" 
            + std::to_string(checker_size) + "' instead."
        );

        std::vector<std::uint8_t> checkerboard(w * h * 4, 255);

        for (size_t y = 0; y < h; y += checker_size) {
            for (size_t x = 0; x < w; x += checker_size) {

                if ((y / checker_size + x / checker_size) % 2 == 0) {

                    size_t y_end = std::min(y + checker_size, h);
                    size_t x_end = std::min(x + checker_size, w);

                    for (size_t i = y; i < y_end; i++) {
                        for (size_t j = x; j < x_end; j++) {
                            size_t p = (i * w + j) * 4;
                            checkerboard[p]     = value;
                            checkerboard[p + 1] = value;
                            checkerboard[p + 2] = value;
                            checkerboard[p + 3] = static_cast<std::uint8_t>(255);
                        }
                    }
                } 
            }
        }

        return checkerboard;
    }

    ImageRGBA8 get_background(const ImageRGBA8& img)
    {
        const ColorLUT& lut = get_color_lut();
        int N = lut.linear_size;

        // RGBA - 4 channels
        std::vector<std::uint8_t> bg = gen_checkerboard_rgba(img.width, img.height, 235, 16);

        ImageRGBA8 out;
        out.width = img.width;
        out.height = img.height;
        out.data.resize(img.data.size());
        
        size_t pixels = static_cast<size_t>(img.width) * static_cast<size_t>(img.height);

        for (size_t i = 0; i < pixels; ++i) {
            size_t idx = i * 4;
            
            float a = u8_to_float32(img.data[idx + 3]);
            float inva = 1.0f - a;

            for (size_t c = 0; c < 3; ++c) {
                float fg_lin = lut.srgb_to_linear[img.data[idx + c]];
                float bg_lin = lut.srgb_to_linear[bg[idx + c]];

                float out_lin = std::ranges::clamp(fg_lin + inva * bg_lin, 0.0f, 1.0f);

                int li = static_cast<int>(out_lin * (N - 1) + 0.5f);
                li = std::ranges::clamp(li, 0, N - 1);

                out.data[idx + c] = lut.linear_to_srgb[li];
            }
            out.data[idx + 3] = static_cast<std::uint8_t>(255);
        }

        return out;
    }

    static std::uint8_t linear01_to_srgb_u8(float lin, const ColorLUT& lut) {
        int N = lut.linear_size;
        lin = std::ranges::clamp(lin, 0.0f, 1.0f);
        int idx = static_cast<int>(lin * (N - 1) + 0.5f);
        idx = std::ranges::clamp(idx, 0, N - 1);
        return lut.linear_to_srgb[idx];
    }

    ImageRGBA8 generate_test_image_rgba(size_t w, size_t h, GenMode mode) {
        const auto& lut = get_color_lut();
        ImageRGBA8 out;
        out.width = static_cast<int>(w);
        out.height = static_cast<int>(h);
        out.data.assign(w * h * 4, 0);

        const float f = 0.1f;

        auto set_px = [&](size_t x, size_t y,
                        std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a)
        {
            size_t idx = (y * w + x) * 4;
            out.data[idx + 0] = r;
            out.data[idx + 1] = g;
            out.data[idx + 2] = b;
            out.data[idx + 3] = a;
        };

        if (mode == GenMode::Stripe45 || mode == GenMode::Stripe135) {
            for (size_t y = 0; y < h; ++y) {
                for (size_t x = 0; x < w; ++x) {
                    float alpha = 0.0f;
                    if (mode == GenMode::Stripe45) {
                        alpha = (1.0f + std::sin(f * (static_cast<float>(y) - static_cast<float>(x)) / 2.0f)) / 2.0f;
                        // A: R = srgb(alpha), G=0, B=0.5*srgb(alpha)
                        std::uint8_t r = linear01_to_srgb_u8(alpha, lut);
                        std::uint8_t b = linear01_to_srgb_u8(alpha * (128.0f/255.0f), lut);
                        std::uint8_t a = float32_to_u8(alpha);
                        set_px(x, y, r, 0, b, a);
                    } else {
                        alpha = (1.0f + std::sin(f * (static_cast<float>(y) + static_cast<float>(x)) / 2.0f)) / 2.0f;
                        // B: R = 0, G = srgb(alpha), B=0.5*srgb(alpha)
                        std::uint8_t g = linear01_to_srgb_u8(alpha, lut);
                        std::uint8_t b = linear01_to_srgb_u8(alpha * (128.0f/255.0f), lut);
                        std::uint8_t a = float32_to_u8(alpha);
                        set_px(x, y, 0, g, b, a);
                    }
                }
            }
            return out;
        }

        if (mode == GenMode::Rectangle) {
            float alpha = 0.7f;
            std::uint8_t r = linear01_to_srgb_u8(alpha, lut);
            std::uint8_t b = linear01_to_srgb_u8(alpha * (128.0f/255.0f), lut);
            std::uint8_t a = float32_to_u8(alpha);

            size_t y0 = h / 3;
            size_t y1 = h - 10;
            size_t x0 = 10;
            size_t x1 = w - (w / 3);

            for (size_t y = y0; y < y1; ++y) {
                for (size_t x = x0; x < x1; ++x) {
                    set_px(x, y, r, 0, b, a);
                }
            }
            return out;
        }

        // Circle
        {
            float alpha = 0.7f;
            std::uint8_t g = linear01_to_srgb_u8(alpha, lut);
            std::uint8_t b = linear01_to_srgb_u8(alpha * (128.0f/255.0f), lut);
            std::uint8_t a_on = float32_to_u8(alpha);

            // center c = (w - w//3, h//3), r = 70
            int cx = static_cast<int>(w - (w / 3));
            int cy = static_cast<int>(h / 3);
            int r  = 70;
            int r2 = r * r;

            for (size_t y = 0; y < h; ++y) {
                for (size_t x = 0; x < w; ++x) {
                    int dx = static_cast<int>(x) - cx;
                    int dy = static_cast<int>(y) - cy;
                    int dist2 = dx*dx + dy*dy;
                    if (dist2 < r2) {
                        set_px(x, y, 0, g, b, a_on);
                    }
                }
            }
            return out;
        }
    }
}
