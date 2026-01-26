#include "alpha_hist/color_convert.hpp"
#include "alpha_hist/parallel_exec.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <vector>

namespace alpha_hist {
namespace {

constexpr size_t kDefaultGrain = 1 << 14;

size_t resolve_grain(size_t grain) {
    return (grain == 0) ? kDefaultGrain : grain;
}

__attribute__((optimize("no-tree-vectorize"), noinline))
void srgb_to_linear_scalar_chunk(const std::uint8_t* in_p,
                                 float* out_p,
                                 size_t begin,
                                 size_t end,
                                 const float* lut_srgb)
{
    for (size_t i = begin; i < end; ++i) {
        size_t idx = i * 4;

        out_p[idx + 0] = lut_srgb[in_p[idx + 0]];
        out_p[idx + 1] = lut_srgb[in_p[idx + 1]];
        out_p[idx + 2] = lut_srgb[in_p[idx + 2]];
        out_p[idx + 3] = u8_to_float32(in_p[idx + 3]);
    }
}

__attribute__((target("avx2,fma"), optimize("no-tree-vectorize"), noinline))
void srgb_to_linear_simd_chunk(const std::uint8_t* in_p,
                               float* out_p,
                               size_t begin,
                               size_t end,
                               const float* lut_srgb)
{
    const __m256 inv255 = _mm256_set1_ps(1.0f / 255.0f);
    const __m256i mask_ff = _mm256_set1_epi32(0xFF);

    size_t i = begin;
    for (; i + 7 < end; i += 8) {
        const size_t byte_idx  = i * 4;
        const size_t float_idx = i * 4;

        __m256i px = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in_p + byte_idx));

        __m256i r_idx = _mm256_and_si256(px, mask_ff);
        __m256i g_idx = _mm256_and_si256(_mm256_srli_epi32(px, 8),  mask_ff);
        __m256i b_idx = _mm256_and_si256(_mm256_srli_epi32(px, 16), mask_ff);
        __m256i a_idx = _mm256_and_si256(_mm256_srli_epi32(px, 24), mask_ff);

        __m256 r = _mm256_i32gather_ps(lut_srgb, r_idx, 4);
        __m256 g = _mm256_i32gather_ps(lut_srgb, g_idx, 4);
        __m256 b = _mm256_i32gather_ps(lut_srgb, b_idx, 4);

        __m256 a = _mm256_mul_ps(_mm256_cvtepi32_ps(a_idx), inv255);

        __m128 r0 = _mm256_castps256_ps128(r);
        __m128 g0 = _mm256_castps256_ps128(g);
        __m128 b0 = _mm256_castps256_ps128(b);
        __m128 a0 = _mm256_castps256_ps128(a);

        __m128 r1 = _mm256_extractf128_ps(r, 1);
        __m128 g1 = _mm256_extractf128_ps(g, 1);
        __m128 b1 = _mm256_extractf128_ps(b, 1);
        __m128 a1 = _mm256_extractf128_ps(a, 1);

        _MM_TRANSPOSE4_PS(r0, g0, b0, a0);
        _MM_TRANSPOSE4_PS(r1, g1, b1, a1);

        _mm_storeu_ps(out_p + float_idx +  0, r0);
        _mm_storeu_ps(out_p + float_idx +  4, g0);
        _mm_storeu_ps(out_p + float_idx +  8, b0);
        _mm_storeu_ps(out_p + float_idx + 12, a0);

        _mm_storeu_ps(out_p + float_idx + 16, r1);
        _mm_storeu_ps(out_p + float_idx + 20, g1);
        _mm_storeu_ps(out_p + float_idx + 24, b1);
        _mm_storeu_ps(out_p + float_idx + 28, a1);
    }

    for (; i < end; ++i) {
        size_t idx = i * 4;
        out_p[idx + 0] = lut_srgb[in_p[idx + 0]];
        out_p[idx + 1] = lut_srgb[in_p[idx + 1]];
        out_p[idx + 2] = lut_srgb[in_p[idx + 2]];
        out_p[idx + 3] = u8_to_float32(in_p[idx + 3]);
    }

    _mm256_zeroupper();
}

__attribute__((optimize("no-tree-vectorize"), noinline))
void linear_to_srgb_scalar_chunk(const float* in_p,
                                 std::uint8_t* out_p,
                                 size_t begin,
                                 size_t end,
                                 const std::uint32_t* lut_linear,
                                 size_t lut_size)
{
    const size_t lut_max = lut_size - 1;
    for (size_t i = begin; i < end; ++i) {
        size_t idx = i * 4;

        for (size_t c = 0; c < 3; ++c) {
            float cl = std::ranges::clamp(in_p[idx + c], 0.0f, 1.0f);
            size_t idx_lut = static_cast<size_t>(cl * static_cast<float>(lut_max) + 0.5f);
            out_p[idx + c] = static_cast<std::uint8_t>(lut_linear[idx_lut]);
        }

        out_p[idx + 3] = float32_to_u8(in_p[idx + 3]);
    }
}

__attribute__((target("avx2,fma"), optimize("no-tree-vectorize"), noinline))
void linear_to_srgb_simd_chunk(const float* in_p,
                               std::uint8_t* out_p,
                               size_t begin,
                               size_t end,
                               const std::uint32_t* lut32,
                               size_t lut_size)
{
    const float lut_scale = static_cast<float>(lut_size - 1);

    const __m128 zero        = _mm_setzero_ps();
    const __m128 one         = _mm_set1_ps(1.0f);
    const __m128 scale       = _mm_set1_ps(lut_scale);
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

        __m128 a        = _mm_min_ps(_mm_max_ps(p3, zero), one);
        __m128 a_scaled = _mm_mul_ps(a, alpha_scale);
        __m128i a_i     = _mm_cvttps_epi32(a_scaled);
        a_i = _mm_min_epi32(_mm_max_epi32(a_i, zero_i), max_i);

        __m128i px = _mm_or_si128(
            _mm_or_si128(r_u32, _mm_slli_epi32(g_u32, 8)),
            _mm_or_si128(_mm_slli_epi32(b_u32, 16), _mm_slli_epi32(a_i, 24))
        );

        _mm_storeu_si128(reinterpret_cast<__m128i*>(out_p + idx), px);
    };

    size_t i = begin;
    for (; i + 3 < end; i += 4) {
        process4px(i * 4);
    }

    for (; i < end; ++i) {
        size_t idx = i * 4;
        for (size_t c = 0; c < 3; ++c) {
            float cl = std::ranges::clamp(in_p[idx + c], 0.0f, 1.0f);
            size_t idx_lut = static_cast<size_t>(cl * (lut_size - 1) + 0.5f);
            out_p[idx + c] = static_cast<std::uint8_t>(lut32[idx_lut]);
        }
        out_p[idx + 3] = float32_to_u8(in_p[idx + 3]);
    }

    _mm256_zeroupper();
}

__attribute__((optimize("no-tree-vectorize"), noinline))
void premultiply_inplace_scalar_chunk(float* p, size_t begin, size_t end)
{
    for (size_t i = begin; i < end; ++i) {
        size_t idx = i * 4;

        float alpha = p[idx + 3];
        alpha = std::ranges::clamp(alpha, 0.0f, 1.0f);
        p[idx + 0] *= alpha;
        p[idx + 1] *= alpha;
        p[idx + 2] *= alpha;
    }
}

__attribute__((target("avx2,fma"), optimize("no-tree-vectorize"), noinline))
void premultiply_inplace_simd_chunk(float* p, size_t begin, size_t end)
{
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

    size_t i = begin;
    for (; i + 1 < end; i += 2) {
        process2px(i * 4);
    }

    for (; i < end; ++i) {
        size_t idx = i * 4;
        float alpha = p[idx + 3];
        alpha = std::ranges::clamp(alpha, 0.0f, 1.0f);
        p[idx + 0] *= alpha;
        p[idx + 1] *= alpha;
        p[idx + 2] *= alpha;
    }

    _mm256_zeroupper();
}

__attribute__((optimize("no-tree-vectorize"), noinline))
void revert_premultiply_inplace_scalar_chunk(float* p, size_t begin, size_t end)
{
    constexpr float EPS = 1e-6f;
    for (size_t i = begin; i < end; ++i) {
        size_t idx = i * 4;

        float alpha = p[idx + 3];
        alpha = std::ranges::clamp(alpha, 0.0f, 1.0f);
        if (alpha <= EPS) {
            p[idx + 0] = 0.0f;
            p[idx + 1] = 0.0f;
            p[idx + 2] = 0.0f;
            continue;
        }
        float inv_alpha = 1.0f / alpha;
        p[idx + 0] *= inv_alpha;
        p[idx + 1] *= inv_alpha;
        p[idx + 2] *= inv_alpha;
    }
}

__attribute__((target("avx2,fma"), optimize("no-tree-vectorize"), noinline))
void revert_premultiply_inplace_simd_chunk(float* p, size_t begin, size_t end)
{
    constexpr float EPS = 1e-6f;
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

        __m256 zeroed = _mm256_blendv_ps(rgb_scaled, zero,
                                        _mm256_cmp_ps(a_clamped, eps, _CMP_LE_OQ));
        __m256 out_v = _mm256_blend_ps(zeroed, v, kAlphaMask);
        _mm256_storeu_ps(p + idx, out_v);
    };

    size_t i = begin;
    for (; i + 1 < end; i += 2) {
        process2px(i * 4);
    }

    for (; i < end; ++i) {
        size_t idx = i * 4;
        float alpha = p[idx + 3];
        alpha = std::ranges::clamp(alpha, 0.0f, 1.0f);
        if (alpha <= EPS) {
            p[idx + 0] = 0.0f;
            p[idx + 1] = 0.0f;
            p[idx + 2] = 0.0f;
        } else {
            float inv_alpha = 1.0f / alpha;
            p[idx + 0] *= inv_alpha;
            p[idx + 1] *= inv_alpha;
            p[idx + 2] *= inv_alpha;
        }
    }

    _mm256_zeroupper();
}

template <class Exec>
void srgb_to_linear_scalar_exec(const ImageRGBA8& in,
                                ImageRGBAf& out,
                                bool use_exact_srgb,
                                const Exec& exec)
{
    const size_t img_size = static_cast<size_t>(in.width) * static_cast<size_t>(in.height);
    out.width = in.width;
    out.height = in.height;
    out.data.resize(in.data.size());

    const ColorLUT& lut = get_color_lut(4096, use_exact_srgb);
    const float* lut_srgb = lut.srgb_to_linear.data();
    const std::uint8_t* in_p = in.data.data();
    float* out_p = out.data.data();

    exec.for_pixels(img_size, [&](size_t begin, size_t end) {
        srgb_to_linear_scalar_chunk(in_p, out_p, begin, end, lut_srgb);
    });
}

template <class Exec>
void srgb_to_linear_simd_exec(const ImageRGBA8& in,
                              ImageRGBAf& out,
                              bool use_exact_srgb,
                              const Exec& exec)
{
    const size_t img_size = static_cast<size_t>(in.width) * static_cast<size_t>(in.height);
    out.width = in.width;
    out.height = in.height;
    out.data.resize(in.data.size());

    const ColorLUT& lut = get_color_lut(4096, use_exact_srgb);
    const float* lut_srgb = lut.srgb_to_linear.data();
    const std::uint8_t* in_p = in.data.data();
    float* out_p = out.data.data();

    exec.for_pixels(img_size, [&](size_t begin, size_t end) {
        srgb_to_linear_simd_chunk(in_p, out_p, begin, end, lut_srgb);
    });
}

template <class Exec>
void linear_to_srgb_scalar_exec(const ImageRGBAf& in,
                                ImageRGBA8& out,
                                bool use_exact_srgb,
                                const Exec& exec)
{
    const size_t img_size = static_cast<size_t>(in.width) * static_cast<size_t>(in.height);
    out.width = in.width;
    out.height = in.height;
    out.data.resize(in.data.size());

    const ColorLUT& lut = get_color_lut(4096, use_exact_srgb);
    const size_t lut_size = static_cast<size_t>(lut.linear_size);
    const std::uint32_t* lut_linear = lut.linear_to_srgb.data();
    const float* in_p = in.data.data();
    std::uint8_t* out_p = out.data.data();

    exec.for_pixels(img_size, [&](size_t begin, size_t end) {
        linear_to_srgb_scalar_chunk(in_p, out_p, begin, end, lut_linear, lut_size);
    });
}

template <class Exec>
void linear_to_srgb_simd_exec(const ImageRGBAf& in,
                              ImageRGBA8& out,
                              bool use_exact_srgb,
                              const Exec& exec)
{
    const size_t img_size = static_cast<size_t>(in.width) * static_cast<size_t>(in.height);
    out.width = in.width;
    out.height = in.height;
    out.data.resize(in.data.size());

    const ColorLUT& lut = get_color_lut(4096, use_exact_srgb);
    const size_t lut_size = static_cast<size_t>(lut.linear_size);

    std::vector<std::uint32_t> lut32(lut_size);
    for (size_t i = 0; i < lut_size; ++i) {
        lut32[i] = lut.linear_to_srgb[i];
    }

    const float* in_p = in.data.data();
    std::uint8_t* out_p = out.data.data();
    const std::uint32_t* lut_p = lut32.data();

    exec.for_pixels(img_size, [&](size_t begin, size_t end) {
        linear_to_srgb_simd_chunk(in_p, out_p, begin, end, lut_p, lut_size);
    });
}

template <class Exec>
void premultiply_inplace_scalar_exec(ImageRGBAf& in, const Exec& exec)
{
    const size_t img_size = static_cast<size_t>(in.width) * static_cast<size_t>(in.height);
    float* p = in.data.data();

    exec.for_pixels(img_size, [&](size_t begin, size_t end) {
        premultiply_inplace_scalar_chunk(p, begin, end);
    });
}

template <class Exec>
void premultiply_inplace_simd_exec(ImageRGBAf& in, const Exec& exec)
{
    const size_t img_size = static_cast<size_t>(in.width) * static_cast<size_t>(in.height);
    float* p = in.data.data();

    exec.for_pixels(img_size, [&](size_t begin, size_t end) {
        premultiply_inplace_simd_chunk(p, begin, end);
    });
}

template <class Exec>
void revert_premultiply_inplace_scalar_exec(ImageRGBAf& in, const Exec& exec)
{
    const size_t img_size = static_cast<size_t>(in.width) * static_cast<size_t>(in.height);
    float* p = in.data.data();

    exec.for_pixels(img_size, [&](size_t begin, size_t end) {
        revert_premultiply_inplace_scalar_chunk(p, begin, end);
    });
}

template <class Exec>
void revert_premultiply_inplace_simd_exec(ImageRGBAf& in, const Exec& exec)
{
    const size_t img_size = static_cast<size_t>(in.width) * static_cast<size_t>(in.height);
    float* p = in.data.data();

    exec.for_pixels(img_size, [&](size_t begin, size_t end) {
        revert_premultiply_inplace_simd_chunk(p, begin, end);
    });
}

} // namespace

void srgb_to_linear_scalar(const ImageRGBA8& in, ImageRGBAf& out, bool use_exact_srgb)
{
    srgb_to_linear_scalar_exec(in, out, use_exact_srgb, SeqExec{});
}

void srgb_to_linear_simd(const ImageRGBA8& in, ImageRGBAf& out, bool use_exact_srgb)
{
    srgb_to_linear_simd_exec(in, out, use_exact_srgb, SeqExec{});
}

void srgb_to_linear_scalar_par(ThreadPool& pool,
                               const ImageRGBA8& in,
                               ImageRGBAf& out,
                               bool use_exact_srgb,
                               size_t grain)
{
    ParExec exec{pool, resolve_grain(grain), 1};
    srgb_to_linear_scalar_exec(in, out, use_exact_srgb, exec);
}

void srgb_to_linear_simd_par(ThreadPool& pool,
                             const ImageRGBA8& in,
                             ImageRGBAf& out,
                             bool use_exact_srgb,
                             size_t grain)
{
    ParExec exec{pool, resolve_grain(grain), 8};
    srgb_to_linear_simd_exec(in, out, use_exact_srgb, exec);
}

void linear_to_srgb_scalar(const ImageRGBAf& in, ImageRGBA8& out, bool use_exact_srgb)
{
    linear_to_srgb_scalar_exec(in, out, use_exact_srgb, SeqExec{});
}

void linear_to_srgb_simd(const ImageRGBAf& in, ImageRGBA8& out, bool use_exact_srgb)
{
    linear_to_srgb_simd_exec(in, out, use_exact_srgb, SeqExec{});
}

void linear_to_srgb_scalar_par(ThreadPool& pool,
                               const ImageRGBAf& in,
                               ImageRGBA8& out,
                               bool use_exact_srgb,
                               size_t grain)
{
    ParExec exec{pool, resolve_grain(grain), 1};
    linear_to_srgb_scalar_exec(in, out, use_exact_srgb, exec);
}

void linear_to_srgb_simd_par(ThreadPool& pool,
                             const ImageRGBAf& in,
                             ImageRGBA8& out,
                             bool use_exact_srgb,
                             size_t grain)
{
    ParExec exec{pool, resolve_grain(grain), 4};
    linear_to_srgb_simd_exec(in, out, use_exact_srgb, exec);
}

void premultiply_inplace_scalar(ImageRGBAf& in)
{
    premultiply_inplace_scalar_exec(in, SeqExec{});
}

void premultiply_inplace_simd(ImageRGBAf& in)
{
    premultiply_inplace_simd_exec(in, SeqExec{});
}

void premultiply_inplace_scalar_par(ThreadPool& pool, ImageRGBAf& in, size_t grain)
{
    ParExec exec{pool, resolve_grain(grain), 1};
    premultiply_inplace_scalar_exec(in, exec);
}

void premultiply_inplace_simd_par(ThreadPool& pool, ImageRGBAf& in, size_t grain)
{
    ParExec exec{pool, resolve_grain(grain), 2};
    premultiply_inplace_simd_exec(in, exec);
}

void revert_premultiply_inplace_scalar(ImageRGBAf& in)
{
    revert_premultiply_inplace_scalar_exec(in, SeqExec{});
}

void revert_premultiply_inplace_simd(ImageRGBAf& in)
{
    revert_premultiply_inplace_simd_exec(in, SeqExec{});
}

void revert_premultiply_inplace_scalar_par(ThreadPool& pool, ImageRGBAf& in, size_t grain)
{
    ParExec exec{pool, resolve_grain(grain), 1};
    revert_premultiply_inplace_scalar_exec(in, exec);
}

void revert_premultiply_inplace_simd_par(ThreadPool& pool, ImageRGBAf& in, size_t grain)
{
    ParExec exec{pool, resolve_grain(grain), 2};
    revert_premultiply_inplace_simd_exec(in, exec);
}

} // namespace alpha_hist
