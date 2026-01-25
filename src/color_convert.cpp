#include "alpha_hist/color_convert.hpp"

#include <cstdint>
#include <algorithm>
#include <vector>
#include <immintrin.h>

namespace alpha_hist {

    //============================ srgb_to_linear =================================//

    __attribute__((optimize("no-tree-vectorize"), noinline))
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

    __attribute__((target("avx2,fma"), optimize("no-tree-vectorize"), noinline))
    void srgb_to_linear_simd(const ImageRGBA8& in, ImageRGBAf& out, bool use_exact_srgb) {
        const size_t img_size = static_cast<size_t>(in.width) * static_cast<size_t>(in.height);

        out.width  = in.width;
        out.height = in.height;
        out.data.resize(img_size * 4); // explicit

        const ColorLUT& lut = get_color_lut(4096, use_exact_srgb);
        const float* lut_srgb = lut.srgb_to_linear.data();

        const std::uint8_t* in_p = in.data.data();
        float* out_p = out.data.data();

        const __m256 inv255 = _mm256_set1_ps(1.0f / 255.0f);
        const __m256i mask_ff = _mm256_set1_epi32(0xFF);

        size_t i = 0;
        for (; i + 7 < img_size; i += 8) {
            const size_t byte_idx  = i * 4; // input index in bytes
            const size_t float_idx = i * 4; // output index in floats

            // Load 8 pixels (32 bytes). Each 32-bit lane is one pixel in little-endian:
            // lane = (A<<24)|(B<<16)|(G<<8)|R
            __m256i px = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in_p + byte_idx));

            // Extract 0..255 indices per lane
            __m256i r_idx = _mm256_and_si256(px, mask_ff);
            __m256i g_idx = _mm256_and_si256(_mm256_srli_epi32(px, 8),  mask_ff);
            __m256i b_idx = _mm256_and_si256(_mm256_srli_epi32(px, 16), mask_ff);
            __m256i a_idx = _mm256_and_si256(_mm256_srli_epi32(px, 24), mask_ff);

            // LUT gather for RGB
            __m256 r = _mm256_i32gather_ps(lut_srgb, r_idx, 4);
            __m256 g = _mm256_i32gather_ps(lut_srgb, g_idx, 4);
            __m256 b = _mm256_i32gather_ps(lut_srgb, b_idx, 4);

            // Alpha: u8 -> float [0,1]
            __m256 a = _mm256_mul_ps(_mm256_cvtepi32_ps(a_idx), inv255);

            // Store AoS RGBA floats: do two 4x4 transposes (low and high halves)
            __m128 r0 = _mm256_castps256_ps128(r);
            __m128 g0 = _mm256_castps256_ps128(g);
            __m128 b0 = _mm256_castps256_ps128(b);
            __m128 a0 = _mm256_castps256_ps128(a);

            __m128 r1 = _mm256_extractf128_ps(r, 1);
            __m128 g1 = _mm256_extractf128_ps(g, 1);
            __m128 b1 = _mm256_extractf128_ps(b, 1);
            __m128 a1 = _mm256_extractf128_ps(a, 1);

            _MM_TRANSPOSE4_PS(r0, g0, b0, a0); // pixels i+0..i+3
            _MM_TRANSPOSE4_PS(r1, g1, b1, a1); // pixels i+4..i+7

            _mm_storeu_ps(out_p + float_idx +  0, r0);
            _mm_storeu_ps(out_p + float_idx +  4, g0);
            _mm_storeu_ps(out_p + float_idx +  8, b0);
            _mm_storeu_ps(out_p + float_idx + 12, a0);

            _mm_storeu_ps(out_p + float_idx + 16, r1);
            _mm_storeu_ps(out_p + float_idx + 20, g1);
            _mm_storeu_ps(out_p + float_idx + 24, b1);
            _mm_storeu_ps(out_p + float_idx + 28, a1);
        }

        // Scalar tail
        for (; i < img_size; ++i) {
            const size_t idx = i * 4;
            out_p[idx + 0] = lut_srgb[in_p[idx + 0]];
            out_p[idx + 1] = lut_srgb[in_p[idx + 1]];
            out_p[idx + 2] = lut_srgb[in_p[idx + 2]];
            out_p[idx + 3] = u8_to_float32(in_p[idx + 3]);
        }

        // Avoid AVX->SSE transition penalty if surrounding code uses legacy SSE
        _mm256_zeroupper();
    }

    //=============================================================================//

    //============================ linear_to_srgb =================================//

    __attribute__((optimize("no-tree-vectorize"), noinline))
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
                size_t idx_lut = static_cast<size_t>(cl * (N - 1) + 0.5f);
                out.data[idx + c] = static_cast<std::uint8_t>(lut.linear_to_srgb[idx_lut]);
            }

            out.data[idx + 3] = float32_to_u8(in.data[idx + 3]);
        }
    }

    __attribute__((target("avx2,fma"), optimize("no-tree-vectorize"), noinline))
    void linear_to_srgb_simd(const ImageRGBAf& in, ImageRGBA8& out, bool use_exact_srgb) {
        size_t img_size = static_cast<size_t>(in.width * in.height);
        out.width = in.width;
        out.height = in.height;
        out.data.resize(in.data.size());

        const ColorLUT& lut = get_color_lut(4096, use_exact_srgb);
        const size_t lut_size = static_cast<size_t>(lut.linear_size);

        const std::uint32_t* lut32 = lut.linear_to_srgb.data(); // only for AVX2 _mm_i32gather_epi32

        const float* in_p = in.data.data();
        std::uint8_t* out_p = out.data.data();

        const __m128 zero        = _mm_setzero_ps();
        const __m128 one         = _mm_set1_ps(1.0f);
        const __m128 scale       = _mm_set1_ps(static_cast<float>(lut_size - 1));
        const __m128 half        = _mm_set1_ps(0.5f);
        const __m128 alpha_scale = _mm_set1_ps(256.0f);
        const __m128i zero_i     = _mm_setzero_si128();
        const __m128i max_i      = _mm_set1_epi32(255);

        auto process4px = [&](size_t idx) {
            __m128 p0 = _mm_loadu_ps(in_p + idx);
            __m128 p1 = _mm_loadu_ps(in_p + idx + 4);
            __m128 p2 = _mm_loadu_ps(in_p + idx + 8);
            __m128 p3 = _mm_loadu_ps(in_p + idx + 12);

            _MM_TRANSPOSE4_PS(p0, p1, p2, p3);

            // clamp
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

            __m128 a        = _mm_min_ps(_mm_max_ps(p3, zero), one); // clamp
            __m128 a_scaled = _mm_mul_ps(a, alpha_scale);
            __m128i a_i     = _mm_cvttps_epi32(a_scaled);
            a_i = _mm_min_epi32(_mm_max_epi32(a_i, zero_i), max_i); //clamp

            // Pack 4 pixels into 4x uint32: r | (g<<8) | (b<<16) | (a<<24)
            __m128i px = _mm_or_si128(
                _mm_or_si128(r_u32, _mm_slli_epi32(g_u32, 8)),
                _mm_or_si128(_mm_slli_epi32(b_u32, 16), _mm_slli_epi32(a_i, 24))
            );

            // Store 16 bytes = 4 pixels RGBA
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out_p + idx), px);
        };

        size_t i = 0;
        for (; i + 7 < img_size; i += 8) {
            size_t idx = i * 4;
            process4px(idx);
            process4px(idx + 16);
        }
        for (; i + 3 < img_size; i += 4) {
            process4px(i * 4);
        }

        for (; i < img_size; ++i) {
            size_t idx = i * 4;
            for (size_t c = 0; c < 3; ++c) {
                float cl = std::ranges::clamp(in_p[idx + c], 0.0f, 1.0f);
                size_t idx_lut = static_cast<size_t>(cl * (lut_size - 1) + 0.5f);
                out_p[idx + c] = lut.linear_to_srgb[idx_lut];
            }
            out_p[idx + 3] = float32_to_u8(in_p[idx + 3]);
        }

        _mm256_zeroupper();
    }

    //=============================================================================//

    //========================= premultiply_inplace_scalar ========================//

    __attribute__((optimize("no-tree-vectorize"), noinline))
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

    __attribute__((target("avx2,fma"), optimize("no-tree-vectorize"), noinline))
    void premultiply_inplace_simd(ImageRGBAf& in) {
        size_t N = static_cast<size_t>(in.height) * in.width;
        float* p = in.data.data();

        const __m256 zero = _mm256_setzero_ps();
        const __m256 one = _mm256_set1_ps(1.0f);
        const __m256i idx_alpha_rep = _mm256_setr_epi32(3, 3, 3, 3, 7, 7, 7, 7);
        constexpr int kAlphaMask = 0x88;

        auto process2px = [&](size_t idx) {
            __m256 v = _mm256_loadu_ps(p + idx);
            __m256 a = _mm256_permutevar8x32_ps(v, idx_alpha_rep);
            __m256 a_clamped = _mm256_min_ps(_mm256_max_ps(a, zero), one);
            __m256 rgb_scaled = _mm256_mul_ps(v, a_clamped);
            __m256 out_v = _mm256_blend_ps(rgb_scaled, v, kAlphaMask);
            _mm256_storeu_ps(p + idx, out_v);
        };

        size_t i = 0;
        for (; i + 3 < N; i += 4) {
            size_t idx = i * 4;
            process2px(idx);
            process2px(idx + 8);
        }
        for (; i + 1 < N; i += 2) {
            process2px(i * 4);
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

    __attribute__((optimize("no-tree-vectorize"), noinline))
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

    __attribute__((target("avx2,fma"), optimize("no-tree-vectorize"), noinline))
    void revert_premultiply_inplace_simd(ImageRGBAf& in) {
        constexpr float EPS = 1e-6f;
        size_t N = static_cast<size_t>(in.height) * in.width;
        float* p = in.data.data();

        const __m256 zero = _mm256_setzero_ps();
        const __m256 one  = _mm256_set1_ps(1.0f);
        const __m256 eps  = _mm256_set1_ps(EPS);
        const __m256i idx_alpha_rep = _mm256_setr_epi32(3, 3, 3, 3, 7, 7, 7, 7);
        constexpr int kAlphaMask = 0x88;

        auto process2px = [&](size_t idx) {
            __m256 v = _mm256_loadu_ps(p + idx);
            __m256 a = _mm256_permutevar8x32_ps(v, idx_alpha_rep);
            
            __m256 a_clamped = _mm256_min_ps(_mm256_max_ps(a, zero), one);
            __m256 a_safe = _mm256_max_ps(a_clamped, eps);
            __m256 inv_a = _mm256_div_ps(one, a_safe);
            __m256 rgb_scaled = _mm256_mul_ps(v, inv_a);

            __m256 zeroed = _mm256_blendv_ps(rgb_scaled, zero, _mm256_cmp_ps(a_clamped, eps, _CMP_LE_OQ));
            __m256 out_v = _mm256_blend_ps(zeroed, v, kAlphaMask);
            _mm256_storeu_ps(p + idx, out_v);
        };

        size_t i = 0;
        for (; i + 3 < N; i += 4) {
            size_t idx = i * 4;
            process2px(idx);
            process2px(idx + 8);
        }
        for (; i + 1 < N; i += 2) {
            process2px(i * 4);
        }

        for (; i < N; ++i) {
            size_t idx = i * 4;
            float alpha = p[idx + 3];
            alpha = std::ranges::clamp(alpha, 0.0f, 1.0f);
            if (alpha <= EPS) {
                p[idx]     = 0.0f;
                p[idx + 1] = 0.0f;
                p[idx + 2] = 0.0f;
            } else {
                float inv_alpha = 1.0f / alpha;
                p[idx]     *= inv_alpha;
                p[idx + 1] *= inv_alpha;
                p[idx + 2] *= inv_alpha;
            }
        }

        _mm256_zeroupper();
    }

    //=============================================================================//

} // namespace alpha_hist
