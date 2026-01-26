#include "alpha_hist/utils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

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

    namespace {
        constexpr float kPi = 3.14159265358979323846f;

        std::uint8_t half_alpha_u8() {
            return float32_to_u8(0.5f);
        }

        void set_px(ImageRGBA8& img, size_t w, size_t x, size_t y,
                    std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a) {
            size_t idx = (y * w + x) * 4;
            img.data[idx + 0] = r;
            img.data[idx + 1] = g;
            img.data[idx + 2] = b;
            img.data[idx + 3] = a;
        }

        void draw_disk(ImageRGBA8& img, size_t w, size_t h, int cx, int cy, int radius,
                       std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a) {
            if (w == 0 || h == 0 || radius <= 0) {
                return;
            }
            int max_x = static_cast<int>(w) - 1;
            int max_y = static_cast<int>(h) - 1;
            int x0 = std::max(0, cx - radius);
            int x1 = std::min(max_x, cx + radius);
            int y0 = std::max(0, cy - radius);
            int y1 = std::min(max_y, cy + radius);

            int r2 = radius * radius;
            for (int y = y0; y <= y1; ++y) {
                for (int x = x0; x <= x1; ++x) {
                    int dx = x - cx;
                    int dy = y - cy;
                    if (dx * dx + dy * dy <= r2) {
                        set_px(img, w, static_cast<size_t>(x), static_cast<size_t>(y), r, g, b, a);
                    }
                }
            }
        }

        void draw_ellipse(ImageRGBA8& img, size_t w, size_t h, int cx, int cy, int rx, int ry,
                          std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a) {
            if (w == 0 || h == 0 || rx <= 0 || ry <= 0) {
                return;
            }
            int max_x = static_cast<int>(w) - 1;
            int max_y = static_cast<int>(h) - 1;
            int x0 = std::max(0, cx - rx);
            int x1 = std::min(max_x, cx + rx);
            int y0 = std::max(0, cy - ry);
            int y1 = std::min(max_y, cy + ry);

            float inv_rx2 = 1.0f / static_cast<float>(rx * rx);
            float inv_ry2 = 1.0f / static_cast<float>(ry * ry);
            for (int y = y0; y <= y1; ++y) {
                for (int x = x0; x <= x1; ++x) {
                    float dx = static_cast<float>(x - cx);
                    float dy = static_cast<float>(y - cy);
                    float v = dx * dx * inv_rx2 + dy * dy * inv_ry2;
                    if (v <= 1.0f) {
                        set_px(img, w, static_cast<size_t>(x), static_cast<size_t>(y), r, g, b, a);
                    }
                }
            }
        }

        float edge_fn(float ax, float ay, float bx, float by, float cx, float cy) {
            return (cx - ax) * (by - ay) - (cy - ay) * (bx - ax);
        }

        void draw_triangle(ImageRGBA8& img, size_t w, size_t h,
                           float ax, float ay, float bx, float by, float cx, float cy,
                           std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a) {
            if (w == 0 || h == 0) {
                return;
            }
            float min_xf = std::min({ax, bx, cx});
            float max_xf = std::max({ax, bx, cx});
            float min_yf = std::min({ay, by, cy});
            float max_yf = std::max({ay, by, cy});

            int min_x = std::max(0, static_cast<int>(std::floor(min_xf)));
            int max_x = std::min(static_cast<int>(w) - 1, static_cast<int>(std::ceil(max_xf)));
            int min_y = std::max(0, static_cast<int>(std::floor(min_yf)));
            int max_y = std::min(static_cast<int>(h) - 1, static_cast<int>(std::ceil(max_yf)));

            float area = edge_fn(ax, ay, bx, by, cx, cy);
            if (std::abs(area) < 1e-6f) {
                return;
            }

            for (int y = min_y; y <= max_y; ++y) {
                for (int x = min_x; x <= max_x; ++x) {
                    float px = static_cast<float>(x) + 0.5f;
                    float py = static_cast<float>(y) + 0.5f;
                    float w0 = edge_fn(bx, by, cx, cy, px, py);
                    float w1 = edge_fn(cx, cy, ax, ay, px, py);
                    float w2 = edge_fn(ax, ay, bx, by, px, py);

                    bool has_neg = (w0 < 0.0f) || (w1 < 0.0f) || (w2 < 0.0f);
                    bool has_pos = (w0 > 0.0f) || (w1 > 0.0f) || (w2 > 0.0f);
                    if (!(has_neg && has_pos)) {
                        set_px(img, w, static_cast<size_t>(x), static_cast<size_t>(y), r, g, b, a);
                    }
                }
            }
        }

        bool inside_rounded_rect(int x, int y, int x0, int y0, int x1, int y1, int radius) {
            if (x < x0 || x >= x1 || y < y0 || y >= y1) {
                return false;
            }
            if (radius <= 0) {
                return true;
            }
            int inner_x0 = x0 + radius;
            int inner_x1 = x1 - radius - 1;
            int inner_y0 = y0 + radius;
            int inner_y1 = y1 - radius - 1;

            int cx = std::clamp(x, inner_x0, inner_x1);
            int cy = std::clamp(y, inner_y0, inner_y1);
            int dx = x - cx;
            int dy = y - cy;
            return dx * dx + dy * dy <= radius * radius;
        }

        void draw_rounded_rect(ImageRGBA8& img, size_t w, size_t h,
                               int x0, int y0, int x1, int y1, int radius,
                               std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a) {
            if (w == 0 || h == 0) {
                return;
            }
            int max_x = static_cast<int>(w);
            int max_y = static_cast<int>(h);
            x0 = std::clamp(x0, 0, max_x);
            y0 = std::clamp(y0, 0, max_y);
            x1 = std::clamp(x1, 0, max_x);
            y1 = std::clamp(y1, 0, max_y);
            if (x1 <= x0 || y1 <= y0) {
                return;
            }

            int w_rect = x1 - x0;
            int h_rect = y1 - y0;
            int max_radius = std::max(0, std::min(w_rect, h_rect) / 2 - 1);
            int r_use = std::min(radius, max_radius);

            for (int y = y0; y < y1; ++y) {
                for (int x = x0; x < x1; ++x) {
                    if (inside_rounded_rect(x, y, x0, y0, x1, y1, r_use)) {
                        set_px(img, w, static_cast<size_t>(x), static_cast<size_t>(y), r, g, b, a);
                    }
                }
            }
        }

        std::mt19937 make_rng(size_t w, size_t h, std::uint32_t salt) {
            std::seed_seq seq{
                static_cast<std::uint32_t>(w),
                static_cast<std::uint32_t>(h),
                salt,
                0x9E3779B9u
            };
            return std::mt19937(seq);
        }

        void fill_stripe45(ImageRGBA8& out, const ColorLUT& lut) {
            size_t w = out.width;
            size_t h = out.height;
            float f = 0.12f;
            std::uint8_t a = half_alpha_u8();
            for (size_t y = 0; y < h; ++y) {
                for (size_t x = 0; x < w; ++x) {
                    float t = 0.5f + 0.5f * std::sin(f * (static_cast<float>(y) - static_cast<float>(x)));
                    std::uint8_t r = linear01_to_srgb_u8(t, lut);
                    std::uint8_t b = linear01_to_srgb_u8(t * 0.5f, lut);
                    set_px(out, w, x, y, r, 0, b, a);
                }
            }
        }

        void fill_stripe135(ImageRGBA8& out, const ColorLUT& lut) {
            size_t w = out.width;
            size_t h = out.height;
            float f = 0.12f;
            std::uint8_t a = half_alpha_u8();
            for (size_t y = 0; y < h; ++y) {
                for (size_t x = 0; x < w; ++x) {
                    float t = 0.5f + 0.5f * std::sin(f * (static_cast<float>(y) + static_cast<float>(x)));
                    std::uint8_t g = linear01_to_srgb_u8(t, lut);
                    std::uint8_t b = linear01_to_srgb_u8(t * 0.5f, lut);
                    set_px(out, w, x, y, 0, g, b, a);
                }
            }
        }

        void fill_rectangle(ImageRGBA8& out, const ColorLUT& lut) {
            size_t w = out.width;
            size_t h = out.height;
            float intensity = 0.85f;
            std::uint8_t a = half_alpha_u8();
            std::uint8_t r = linear01_to_srgb_u8(intensity, lut);
            std::uint8_t b = linear01_to_srgb_u8(intensity * 0.5f, lut);

            size_t margin = std::max<size_t>(1, std::min(w, h) / 12);
            size_t x0 = margin;
            size_t y0 = h / 3;
            size_t x1 = (w > margin) ? w - w / 4 : w;
            size_t y1 = (h > margin) ? h - margin : h;

            for (size_t y = y0; y < y1; ++y) {
                for (size_t x = x0; x < x1; ++x) {
                    set_px(out, w, x, y, r, 0, b, a);
                }
            }
        }

        void fill_circle(ImageRGBA8& out, const ColorLUT& lut) {
            size_t w = out.width;
            size_t h = out.height;
            std::uint8_t a = half_alpha_u8();
            std::uint8_t g = linear01_to_srgb_u8(0.85f, lut);
            std::uint8_t b = linear01_to_srgb_u8(0.45f, lut);

            int cx = static_cast<int>(w) - static_cast<int>(w / 3);
            int cy = static_cast<int>(h / 3);
            int r = static_cast<int>(std::max<size_t>(1, std::min(w, h) / 5));
            draw_disk(out, w, h, cx, cy, r, 0, g, b, a);
        }

        void fill_sine_bands(ImageRGBA8& out, const ColorLUT& lut) {
            size_t w = out.width;
            size_t h = out.height;
            if (w == 0 || h == 0) {
                return;
            }
            std::uint8_t a = half_alpha_u8();
            float amplitude = 0.25f * static_cast<float>(h);
            float period = static_cast<float>(w) * 0.9f + 1.0f;
            int thickness = static_cast<int>(std::max<size_t>(1, std::min(w, h) / 60));

            for (size_t x = 0; x < w; ++x) {
                float phase = 2.0f * kPi * static_cast<float>(x) / period;
                float y_center = static_cast<float>(h) * 0.5f + amplitude * std::sin(phase);
                int y0 = static_cast<int>(std::lround(y_center));
                for (int dy = -thickness; dy <= thickness; ++dy) {
                    int y = y0 + dy;
                    if (y < 0 || y >= static_cast<int>(h)) {
                        continue;
                    }
                    float t = 1.0f - std::abs(static_cast<float>(dy)) / static_cast<float>(thickness + 1);
                    std::uint8_t r = linear01_to_srgb_u8(t, lut);
                    std::uint8_t b = linear01_to_srgb_u8(t * 0.5f, lut);
                    set_px(out, w, x, static_cast<size_t>(y), r, 0, b, a);
                }
            }

            float amplitude2 = amplitude * 0.6f;
            float period2 = static_cast<float>(w) * 0.6f + 1.0f;
            for (size_t x = 0; x < w; ++x) {
                float phase = 2.0f * kPi * static_cast<float>(x) / period2 + 1.2f;
                float y_center = static_cast<float>(h) * 0.5f + amplitude2 * std::sin(phase);
                int y0 = static_cast<int>(std::lround(y_center));
                for (int dy = -thickness; dy <= thickness; ++dy) {
                    int y = y0 + dy;
                    if (y < 0 || y >= static_cast<int>(h)) {
                        continue;
                    }
                    float t = 1.0f - std::abs(static_cast<float>(dy)) / static_cast<float>(thickness + 1);
                    std::uint8_t g = linear01_to_srgb_u8(t, lut);
                    std::uint8_t b = linear01_to_srgb_u8(t * 0.4f, lut);
                    set_px(out, w, x, static_cast<size_t>(y), 0, g, b, a);
                }
            }
        }

        void fill_lissajous(ImageRGBA8& out, const ColorLUT& lut) {
            size_t w = out.width;
            size_t h = out.height;
            if (w == 0 || h == 0) {
                return;
            }
            std::uint8_t a = half_alpha_u8();
            std::uint8_t g = linear01_to_srgb_u8(0.8f, lut);
            std::uint8_t b = linear01_to_srgb_u8(0.6f, lut);
            int radius = std::max<int>(1, static_cast<int>(std::min(w, h) / 80));
            int samples = std::max<int>(200, static_cast<int>(std::min(w, h) * 6));

            for (int i = 0; i < samples; ++i) {
                float t = static_cast<float>(i) / static_cast<float>(samples - 1);
                float ang = t * 2.0f * kPi;
                float x = 0.5f + 0.45f * std::sin(3.0f * ang + 0.4f);
                float y = 0.5f + 0.45f * std::sin(2.0f * ang);
                int px = static_cast<int>(x * static_cast<float>(w - 1));
                int py = static_cast<int>(y * static_cast<float>(h - 1));
                draw_disk(out, w, h, px, py, radius, 0, g, b, a);
            }
        }

        void fill_spiral(ImageRGBA8& out, const ColorLUT& lut) {
            size_t w = out.width;
            size_t h = out.height;
            if (w == 0 || h == 0) {
                return;
            }
            std::uint8_t a = half_alpha_u8();
            std::uint8_t r = linear01_to_srgb_u8(0.9f, lut);
            std::uint8_t g = linear01_to_srgb_u8(0.7f, lut);
            int radius = std::max<int>(1, static_cast<int>(std::min(w, h) / 90));

            float cx = static_cast<float>(w - 1) * 0.5f;
            float cy = static_cast<float>(h - 1) * 0.5f;
            int samples = std::max<int>(300, static_cast<int>(std::min(w, h) * 8));
            float max_r = 0.45f * static_cast<float>(std::min(w, h));
            float max_t = 4.0f * kPi;

            for (int i = 0; i < samples; ++i) {
                float t = static_cast<float>(i) / static_cast<float>(samples - 1);
                float ang = t * max_t;
                float rad = t * max_r;
                int px = static_cast<int>(cx + rad * std::cos(ang));
                int py = static_cast<int>(cy + rad * std::sin(ang));
                draw_disk(out, w, h, px, py, radius, r, g, 0, a);
            }
        }

        void fill_rounded_rects(ImageRGBA8& out, const ColorLUT& lut) {
            size_t w = out.width;
            size_t h = out.height;
            int radius = static_cast<int>(std::max<size_t>(1, std::min(w, h) / 12));
            std::uint8_t a = half_alpha_u8();

            std::array<std::array<float, 4>, 3> rects = {{
                {0.08f, 0.10f, 0.62f, 0.38f},
                {0.30f, 0.30f, 0.92f, 0.68f},
                {0.18f, 0.60f, 0.74f, 0.90f},
            }};
            std::array<std::array<float, 3>, 3> colors = {{
                {0.85f, 0.20f, 0.30f},
                {0.20f, 0.80f, 0.35f},
                {0.20f, 0.45f, 0.85f},
            }};

            for (size_t i = 0; i < rects.size(); ++i) {
                int x0 = static_cast<int>(rects[i][0] * static_cast<float>(w));
                int y0 = static_cast<int>(rects[i][1] * static_cast<float>(h));
                int x1 = static_cast<int>(rects[i][2] * static_cast<float>(w));
                int y1 = static_cast<int>(rects[i][3] * static_cast<float>(h));
                std::uint8_t r = linear01_to_srgb_u8(colors[i][0], lut);
                std::uint8_t g = linear01_to_srgb_u8(colors[i][1], lut);
                std::uint8_t b = linear01_to_srgb_u8(colors[i][2], lut);
                draw_rounded_rect(out, w, h, x0, y0, x1, y1, radius, r, g, b, a);
            }
        }

        void fill_concentric(ImageRGBA8& out, const ColorLUT& lut) {
            size_t w = out.width;
            size_t h = out.height;
            if (w == 0 || h == 0) {
                return;
            }
            float cx = static_cast<float>(w - 1) * 0.5f;
            float cy = static_cast<float>(h - 1) * 0.5f;
            float spacing = std::max(6.0f, static_cast<float>(std::min(w, h)) / 10.0f);
            float thickness = spacing * 0.3f;
            std::uint8_t a = half_alpha_u8();

            for (size_t y = 0; y < h; ++y) {
                for (size_t x = 0; x < w; ++x) {
                    float dx = static_cast<float>(x) - cx;
                    float dy = static_cast<float>(y) - cy;
                    float dist = std::sqrt(dx * dx + dy * dy);
                    float mod = std::fmod(dist, spacing);
                    if (mod < thickness) {
                        int ring = static_cast<int>(dist / spacing);
                        int color_id = ring % 3;
                        std::uint8_t r = 0;
                        std::uint8_t g = 0;
                        std::uint8_t b = 0;
                        if (color_id == 0) {
                            r = linear01_to_srgb_u8(0.9f, lut);
                        } else if (color_id == 1) {
                            g = linear01_to_srgb_u8(0.85f, lut);
                        } else {
                            b = linear01_to_srgb_u8(0.8f, lut);
                        }
                        set_px(out, w, x, y, r, g, b, a);
                    }
                }
            }
        }

        void fill_blobs(ImageRGBA8& out, const ColorLUT& lut) {
            size_t w = out.width;
            size_t h = out.height;
            if (w == 0 || h == 0) {
                return;
            }
            std::uint8_t a = half_alpha_u8();
            auto rng = make_rng(w, h, 0xB10B5u);
            std::uniform_int_distribution<int> x_dist(0, static_cast<int>(w - 1));
            std::uniform_int_distribution<int> y_dist(0, static_cast<int>(h - 1));
            int min_r = std::max(1, static_cast<int>(std::min(w, h) / 18));
            int max_r = std::max(min_r, static_cast<int>(std::min(w, h) / 6));
            std::uniform_int_distribution<int> r_dist(min_r, max_r);
            std::uniform_real_distribution<float> c_dist(0.2f, 1.0f);

            int blobs = 10;
            for (int i = 0; i < blobs; ++i) {
                int cx = x_dist(rng);
                int cy = y_dist(rng);
                int r = r_dist(rng);
                std::uint8_t cr = linear01_to_srgb_u8(c_dist(rng), lut);
                std::uint8_t cg = linear01_to_srgb_u8(c_dist(rng), lut);
                std::uint8_t cb = linear01_to_srgb_u8(c_dist(rng), lut);
                draw_disk(out, w, h, cx, cy, r, cr, cg, cb, a);
            }
        }

        void fill_random_forms(ImageRGBA8& out, const ColorLUT& lut) {
            size_t w = out.width;
            size_t h = out.height;
            if (w == 0 || h == 0) {
                return;
            }
            std::uint8_t a = half_alpha_u8();
            auto rng = make_rng(w, h, 0xF04F5u);
            std::uniform_real_distribution<float> xf(0.08f, 0.92f);
            std::uniform_real_distribution<float> yf(0.08f, 0.92f);
            std::uniform_real_distribution<float> rf(0.05f, 0.25f);
            std::uniform_real_distribution<float> cf(0.2f, 1.0f);

            int ellipses = 4;
            for (int i = 0; i < ellipses; ++i) {
                float cx = xf(rng) * static_cast<float>(w - 1);
                float cy = yf(rng) * static_cast<float>(h - 1);
                float rx = rf(rng) * static_cast<float>(w);
                float ry = rf(rng) * static_cast<float>(h);
                std::uint8_t r = linear01_to_srgb_u8(cf(rng), lut);
                std::uint8_t g = linear01_to_srgb_u8(cf(rng), lut);
                std::uint8_t b = linear01_to_srgb_u8(cf(rng), lut);
                draw_ellipse(out, w, h, static_cast<int>(cx), static_cast<int>(cy),
                             static_cast<int>(rx), static_cast<int>(ry), r, g, b, a);
            }

            int triangles = 3;
            for (int i = 0; i < triangles; ++i) {
                float ax = xf(rng) * static_cast<float>(w - 1);
                float ay = yf(rng) * static_cast<float>(h - 1);
                float bx = xf(rng) * static_cast<float>(w - 1);
                float by = yf(rng) * static_cast<float>(h - 1);
                float cx = xf(rng) * static_cast<float>(w - 1);
                float cy = yf(rng) * static_cast<float>(h - 1);
                std::uint8_t r = linear01_to_srgb_u8(cf(rng), lut);
                std::uint8_t g = linear01_to_srgb_u8(cf(rng), lut);
                std::uint8_t b = linear01_to_srgb_u8(cf(rng), lut);
                draw_triangle(out, w, h, ax, ay, bx, by, cx, cy, r, g, b, a);
            }
        }

        void fill_gaussian_noise(ImageRGBA8& out, const ColorLUT& lut) {
            size_t w = out.width;
            size_t h = out.height;
            if (w == 0 || h == 0) {
                return;
            }
            std::uint8_t a = half_alpha_u8();
            auto rng = make_rng(w, h, 0x6A550u);
            std::normal_distribution<float> dist(0.5f, 0.18f);

            for (size_t y = 0; y < h; ++y) {
                for (size_t x = 0; x < w; ++x) {
                    float r_lin = std::clamp(dist(rng), 0.0f, 1.0f);
                    float g_lin = std::clamp(dist(rng), 0.0f, 1.0f);
                    float b_lin = std::clamp(dist(rng), 0.0f, 1.0f);
                    std::uint8_t r = linear01_to_srgb_u8(r_lin, lut);
                    std::uint8_t g = linear01_to_srgb_u8(g_lin, lut);
                    std::uint8_t b = linear01_to_srgb_u8(b_lin, lut);
                    set_px(out, w, x, y, r, g, b, a);
                }
            }
        }
    } // namespace

    ImageRGBA8 generate_test_image_rgba(size_t w, size_t h, GenMode mode) {
        const auto& lut = get_color_lut();
        ImageRGBA8 out;
        out.width = static_cast<std::uint32_t>(w);
        out.height = static_cast<std::uint32_t>(h);
        out.data.assign(w * h * 4, 0);

        switch (mode) {
            case GenMode::Stripe45:
                fill_stripe45(out, lut);
                break;
            case GenMode::Stripe135:
                fill_stripe135(out, lut);
                break;
            case GenMode::Rectangle:
                fill_rectangle(out, lut);
                break;
            case GenMode::Circle:
                fill_circle(out, lut);
                break;
            case GenMode::SineBands:
                fill_sine_bands(out, lut);
                break;
            case GenMode::Lissajous:
                fill_lissajous(out, lut);
                break;
            case GenMode::Spiral:
                fill_spiral(out, lut);
                break;
            case GenMode::RoundedRects:
                fill_rounded_rects(out, lut);
                break;
            case GenMode::Concentric:
                fill_concentric(out, lut);
                break;
            case GenMode::Blobs:
                fill_blobs(out, lut);
                break;
            case GenMode::RandomForms:
                fill_random_forms(out, lut);
                break;
            case GenMode::GaussianNoise:
                fill_gaussian_noise(out, lut);
                break;
            default:
                throw std::invalid_argument("generate_test_image_rgba: unknown GenMode");
        }

        return out;
    }
}
