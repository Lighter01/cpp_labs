#include "alpha_hist/color_convert.hpp"

#include <cstdint>
#include <algorithm>
#include <vector>
#include <immintrin.h>

namespace alpha_hist {

    //============================ srgb_to_linear =================================//

    void srgb_to_linear_scalar(const ImageRGBA8& in, ImageRGBAf& out, bool use_exact_srgb) {
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

    void srgb_to_linear_simd(const ImageRGBA8& in, ImageRGBAf& out, bool use_exact_srgb) {
        size_t img_size = static_cast<size_t>(in.width * in.height);
        out.width = in.width;
        out.height = in.height;
        out.data.resize(in.data.size());

        const ColorLUT& lut = get_color_lut(4096, use_exact_srgb);
        const float* lut_srgb = lut.srgb_to_linear.data();

        const std::uint8_t* in_p = in.data.data();
        float* out_p = out.data.data();

        const __m128 inv255 = _mm_set1_ps(1.0f / 255.0f);
        const __m128i shuffle_r = _mm_setr_epi8(
            0, 4, 8, 12, static_cast<char>(0x80), static_cast<char>(0x80), static_cast<char>(0x80), static_cast<char>(0x80),
            static_cast<char>(0x80), static_cast<char>(0x80), static_cast<char>(0x80), static_cast<char>(0x80),
            static_cast<char>(0x80), static_cast<char>(0x80), static_cast<char>(0x80), static_cast<char>(0x80));
        const __m128i shuffle_g = _mm_setr_epi8(
            1, 5, 9, 13, static_cast<char>(0x80), static_cast<char>(0x80), static_cast<char>(0x80), static_cast<char>(0x80),
            static_cast<char>(0x80), static_cast<char>(0x80), static_cast<char>(0x80), static_cast<char>(0x80),
            static_cast<char>(0x80), static_cast<char>(0x80), static_cast<char>(0x80), static_cast<char>(0x80));
        const __m128i shuffle_b = _mm_setr_epi8(
            2, 6, 10, 14, static_cast<char>(0x80), static_cast<char>(0x80), static_cast<char>(0x80), static_cast<char>(0x80),
            static_cast<char>(0x80), static_cast<char>(0x80), static_cast<char>(0x80), static_cast<char>(0x80),
            static_cast<char>(0x80), static_cast<char>(0x80), static_cast<char>(0x80), static_cast<char>(0x80));
        const __m128i shuffle_a = _mm_setr_epi8(
            3, 7, 11, 15, static_cast<char>(0x80), static_cast<char>(0x80), static_cast<char>(0x80), static_cast<char>(0x80),
            static_cast<char>(0x80), static_cast<char>(0x80), static_cast<char>(0x80), static_cast<char>(0x80),
            static_cast<char>(0x80), static_cast<char>(0x80), static_cast<char>(0x80), static_cast<char>(0x80));

        alignas(16) float r_buf[4];
        alignas(16) float g_buf[4];
        alignas(16) float b_buf[4];
        alignas(16) float a_buf[4];

        size_t i = 0;
        for (; i + 3 < img_size; i += 4) {
            size_t idx = i * 4;
            __m128i px = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in_p + idx));

            __m128i r8 = _mm_shuffle_epi8(px, shuffle_r);
            __m128i g8 = _mm_shuffle_epi8(px, shuffle_g);
            __m128i b8 = _mm_shuffle_epi8(px, shuffle_b);
            __m128i a8 = _mm_shuffle_epi8(px, shuffle_a);

            __m128i r32 = _mm_cvtepu8_epi32(r8);
            __m128i g32 = _mm_cvtepu8_epi32(g8);
            __m128i b32 = _mm_cvtepu8_epi32(b8);
            __m128i a32 = _mm_cvtepu8_epi32(a8);

            __m128 r_f = _mm_i32gather_ps(lut_srgb, r32, 4);
            __m128 g_f = _mm_i32gather_ps(lut_srgb, g32, 4);
            __m128 b_f = _mm_i32gather_ps(lut_srgb, b32, 4);
            __m128 a_f = _mm_mul_ps(_mm_cvtepi32_ps(a32), inv255);

            _mm_storeu_ps(r_buf, r_f);
            _mm_storeu_ps(g_buf, g_f);
            _mm_storeu_ps(b_buf, b_f);
            _mm_storeu_ps(a_buf, a_f);

            for (int p = 0; p < 4; ++p) {
                size_t o = idx + static_cast<size_t>(p) * 4;
                out_p[o]     = r_buf[p];
                out_p[o + 1] = g_buf[p];
                out_p[o + 2] = b_buf[p];
                out_p[o + 3] = a_buf[p];
            }
        }

        for (; i < img_size; ++i) {
            size_t idx = i * 4;
            for (size_t c = 0; c < 3; ++c) {
                out_p[idx + c] = lut_srgb[in_p[idx + c]];
            }
            out_p[idx + 3] = u8_to_float32(in_p[idx + 3]);
        }
    }

    //=============================================================================//

    //============================ linear_to_srgb =================================//

    void linear_to_srgb_scalar(const ImageRGBAf& in, ImageRGBA8& out, bool use_exact_srgb) {
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
                out.data[idx + c] = lut.linear_to_srgb[static_cast<size_t>(idx_lut)];
            }

            out.data[idx + 3] = float32_to_u8(in.data[idx + 3]);
        }
    }

    void linear_to_srgb_simd(const ImageRGBAf& in, ImageRGBA8& out, bool use_exact_srgb) {
        size_t img_size = static_cast<size_t>(in.width * in.height);
        out.width = in.width;
        out.height = in.height;
        out.data.resize(in.data.size());

        const ColorLUT& lut = get_color_lut(4096, use_exact_srgb);
        const int lut_size = lut.linear_size;

        static std::vector<std::uint32_t> lut_u32;
        static int last_size = 0;
        static bool last_exact = true;
        if (lut_u32.empty() || last_size != lut_size || last_exact != use_exact_srgb) {
            lut_u32.resize(static_cast<size_t>(lut_size));
            for (int i = 0; i < lut_size; ++i) {
                lut_u32[static_cast<size_t>(i)] = lut.linear_to_srgb[static_cast<size_t>(i)];
            }
            last_size = lut_size;
            last_exact = use_exact_srgb;
        }
        const std::uint32_t* lut32 = lut_u32.data();

        const float* in_p = in.data.data();
        std::uint8_t* out_p = out.data.data();

        const __m128 zero = _mm_setzero_ps();
        const __m128 one = _mm_set1_ps(1.0f);
        const __m128 scale = _mm_set1_ps(static_cast<float>(lut_size - 1));
        const __m128 half = _mm_set1_ps(0.5f);
        const __m128 alpha_scale = _mm_set1_ps(256.0f);
        const __m128i zero_i = _mm_setzero_si128();
        const __m128i max_i = _mm_set1_epi32(255);

        alignas(16) std::uint32_t r_buf[4];
        alignas(16) std::uint32_t g_buf[4];
        alignas(16) std::uint32_t b_buf[4];
        alignas(16) std::uint32_t a_buf[4];

        size_t i = 0;
        for (; i + 3 < img_size; i += 4) {
            size_t idx = i * 4;
            __m128 p0 = _mm_loadu_ps(in_p + idx);
            __m128 p1 = _mm_loadu_ps(in_p + idx + 4);
            __m128 p2 = _mm_loadu_ps(in_p + idx + 8);
            __m128 p3 = _mm_loadu_ps(in_p + idx + 12);

            _MM_TRANSPOSE4_PS(p0, p1, p2, p3);

            __m128 r = _mm_min_ps(_mm_max_ps(p0, zero), one);
            __m128 g = _mm_min_ps(_mm_max_ps(p1, zero), one);
            __m128 b = _mm_min_ps(_mm_max_ps(p2, zero), one);

            __m128 r_idx_f = _mm_add_ps(_mm_mul_ps(r, scale), half);
            __m128 g_idx_f = _mm_add_ps(_mm_mul_ps(g, scale), half);
            __m128 b_idx_f = _mm_add_ps(_mm_mul_ps(b, scale), half);

            __m128i r_idx = _mm_cvttps_epi32(r_idx_f);
            __m128i g_idx = _mm_cvttps_epi32(g_idx_f);
            __m128i b_idx = _mm_cvttps_epi32(b_idx_f);

            __m128i r_u32 = _mm_i32gather_epi32(reinterpret_cast<const int*>(lut32), r_idx, 4);
            __m128i g_u32 = _mm_i32gather_epi32(reinterpret_cast<const int*>(lut32), g_idx, 4);
            __m128i b_u32 = _mm_i32gather_epi32(reinterpret_cast<const int*>(lut32), b_idx, 4);

            __m128 a = _mm_min_ps(_mm_max_ps(p3, zero), one);
            __m128 a_scaled = _mm_mul_ps(a, alpha_scale);
            __m128i a_i = _mm_cvttps_epi32(a_scaled);
            a_i = _mm_min_epi32(_mm_max_epi32(a_i, zero_i), max_i);

            _mm_storeu_si128(reinterpret_cast<__m128i*>(r_buf), r_u32);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(g_buf), g_u32);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(b_buf), b_u32);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(a_buf), a_i);

            for (int p = 0; p < 4; ++p) {
                size_t o = idx + static_cast<size_t>(p) * 4;
                out_p[o]     = static_cast<std::uint8_t>(r_buf[p]);
                out_p[o + 1] = static_cast<std::uint8_t>(g_buf[p]);
                out_p[o + 2] = static_cast<std::uint8_t>(b_buf[p]);
                out_p[o + 3] = static_cast<std::uint8_t>(a_buf[p]);
            }
        }

        for (; i < img_size; ++i) {
            size_t idx = i * 4;
            for (size_t c = 0; c < 3; ++c) {
                float cl = std::ranges::clamp(in_p[idx + c], 0.0f, 1.0f);
                int idx_lut = static_cast<int>(cl * (lut_size - 1) + 0.5f);
                out_p[idx + c] = lut.linear_to_srgb[static_cast<size_t>(idx_lut)];
            }
            out_p[idx + 3] = float32_to_u8(in_p[idx + 3]);
        }
    }

    //=============================================================================//

    //========================= premultiply_inplace_scalar ========================//

    void premultiply_inplace_scalar(ImageRGBAf& in) {
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

    void premultiply_inplace_simd(ImageRGBAf& in) {
        size_t N = static_cast<size_t>(in.height) * static_cast<size_t>(in.width);
        float* p = in.data.data();

        const __m256 zero = _mm256_setzero_ps();
        const __m256 one = _mm256_set1_ps(1.0f);
        const __m256i idx_alpha_rep = _mm256_setr_epi32(3, 3, 3, 3, 7, 7, 7, 7);
        constexpr int kAlphaMask = 0x88;

        size_t i = 0;
        for (; i + 1 < N; i += 2) {
            size_t idx = i * 4;
            __m256 v = _mm256_loadu_ps(p + idx);
            __m256 a = _mm256_permutevar8x32_ps(v, idx_alpha_rep);
            __m256 a_clamped = _mm256_min_ps(_mm256_max_ps(a, zero), one);
            __m256 rgb_scaled = _mm256_mul_ps(v, a_clamped);
            __m256 out_v = _mm256_blend_ps(rgb_scaled, v, kAlphaMask);
            _mm256_storeu_ps(p + idx, out_v);
        }

        for (; i < N; ++i) {
            size_t idx = i * 4;
            float alpha = p[idx + 3];
            alpha = std::ranges::clamp(alpha, 0.0f, 1.0f);
            p[idx]     *= alpha;
            p[idx + 1] *= alpha;
            p[idx + 2] *= alpha;
        }
    }

    //=============================================================================//

    //=================== revert_premultiply_inplace_scalar =======================//

    void revert_premultiply_inplace_scalar(ImageRGBAf& in) {
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

    void revert_premultiply_inplace_simd(ImageRGBAf& in) {
        constexpr float EPS = 1e-6f;
        size_t N = static_cast<size_t>(in.height) * static_cast<size_t>(in.width);
        float* p = in.data.data();

        const __m256 zero = _mm256_setzero_ps();
        const __m256 one = _mm256_set1_ps(1.0f);
        const __m256 eps = _mm256_set1_ps(EPS);
        const __m256i idx_alpha_rep = _mm256_setr_epi32(3, 3, 3, 3, 7, 7, 7, 7);
        constexpr int kAlphaMask = 0x88;

        size_t i = 0;
        for (; i + 1 < N; i += 2) {
            size_t idx = i * 4;
            __m256 v = _mm256_loadu_ps(p + idx);
            __m256 a = _mm256_permutevar8x32_ps(v, idx_alpha_rep);
            __m256 a_clamped = _mm256_min_ps(_mm256_max_ps(a, zero), one);
            __m256 a_safe = _mm256_max_ps(a_clamped, eps);
            __m256 inv_a = _mm256_div_ps(one, a_safe);

            __m256 rgb_scaled = _mm256_mul_ps(v, inv_a);
            __m256 zeroed = _mm256_blendv_ps(rgb_scaled, zero, _mm256_cmp_ps(a_clamped, eps, _CMP_LE_OQ));
            __m256 out_v = _mm256_blend_ps(zeroed, v, kAlphaMask);
            _mm256_storeu_ps(p + idx, out_v);
        }

        for (; i < N; ++i) {
            size_t idx = i * 4;
            float alpha = p[idx + 3];
            alpha = std::ranges::clamp(alpha, 0.0f, 1.0f);
            if (alpha <= EPS) {
                p[idx]     = 0.0f;
                p[idx + 1] = 0.0f;
                p[idx + 2] = 0.0f;
                continue;
            }
            float inv_alpha = 1.0f / alpha;
            p[idx]     *= inv_alpha;
            p[idx + 1] *= inv_alpha;
            p[idx + 2] *= inv_alpha;
        }
    }

    //=============================================================================//

} // namespace alpha_hist
