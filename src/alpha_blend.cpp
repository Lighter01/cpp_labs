#include "alpha_hist/alpha_blend.hpp"
#include "alpha_hist/color_convert.hpp"
#include "alpha_hist/parallel_exec.hpp"

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <immintrin.h>
#include <stdexcept>
#include <string>

namespace alpha_hist {
namespace {

static void ensure_ok(bool ok, const std::string& msg) {
    if (!ok) {
        throw std::runtime_error(msg);
    }
}

ThreadPool& require_pool(ThreadPool* pool) {
    if (!pool) {
        throw std::runtime_error("alpha_blend_pipeline_templ: ThreadPool is required for parallel execution");
    }
    return *pool;
}


template <BlendMode M>
__attribute__((optimize("no-tree-vectorize"), noinline))
void blend_scalar_chunk(const float* fg_p,
                        const float* bg_p,
                        float* out_p,
                        size_t begin,
                        size_t end,
                        float global_opacity)
{
    for (size_t i = begin; i < end; ++i) {
        size_t idx = i * 4;

        float fg_alpha = fg_p[idx + 3] * global_opacity;
        float bg_alpha = bg_p[idx + 3];

        float F_a = 0.0f;
        float F_b = 0.0f;

        if constexpr (M == BlendMode::Over) {
            F_a = 1.0f;
            F_b = 1.0f - fg_alpha;
        } else if constexpr (M == BlendMode::In) {
            F_a = bg_alpha;
            F_b = 0.0f;
        } else if constexpr (M == BlendMode::Out) {
            F_a = 1.0f - bg_alpha;
            F_b = 0.0f;
        } else if constexpr (M == BlendMode::Atop) {
            F_a = bg_alpha;
            F_b = 1.0f - fg_alpha;
        } else if constexpr (M == BlendMode::Xor) {
            F_a = 1.0f - bg_alpha;
            F_b = 1.0f - fg_alpha;
        } else {
            throw std::runtime_error("Got unknown blend mode");
        }

        out_p[idx + 3] = F_a * fg_alpha + F_b * bg_alpha;
        out_p[idx + 0] = F_a * global_opacity * fg_p[idx + 0] + F_b * bg_p[idx + 0];
        out_p[idx + 1] = F_a * global_opacity * fg_p[idx + 1] + F_b * bg_p[idx + 1];
        out_p[idx + 2] = F_a * global_opacity * fg_p[idx + 2] + F_b * bg_p[idx + 2];
    }
}

template <BlendMode M>
__attribute__((target("avx2,fma"), optimize("no-tree-vectorize"), noinline))
void blend_simd_chunk(const float* fg_p,
                      const float* bg_p,
                      float* out_p,
                      size_t begin,
                      size_t end,
                      float global_opacity)
{
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 glob_op = _mm256_set1_ps(global_opacity);

    size_t i = begin;
    for (; i + 7 < end; i += 8) {
        const size_t idx = i * 4;

        __m256 fg_v0 = _mm256_loadu_ps(fg_p + idx + 0 * 8);
        __m256 fg_v1 = _mm256_loadu_ps(fg_p + idx + 1 * 8);
        __m256 fg_v2 = _mm256_loadu_ps(fg_p + idx + 2 * 8);
        __m256 fg_v3 = _mm256_loadu_ps(fg_p + idx + 3 * 8);

        __m256 bg_v0 = _mm256_loadu_ps(bg_p + idx + 0 * 8);
        __m256 bg_v1 = _mm256_loadu_ps(bg_p + idx + 1 * 8);
        __m256 bg_v2 = _mm256_loadu_ps(bg_p + idx + 2 * 8);
        __m256 bg_v3 = _mm256_loadu_ps(bg_p + idx + 3 * 8);

        __m256 fg_scaled_0 = _mm256_mul_ps(fg_v0, glob_op);
        __m256 fg_scaled_1 = _mm256_mul_ps(fg_v1, glob_op);
        __m256 fg_scaled_2 = _mm256_mul_ps(fg_v2, glob_op);
        __m256 fg_scaled_3 = _mm256_mul_ps(fg_v3, glob_op);

        __m256 rgba_out_0;
        __m256 rgba_out_1;
        __m256 rgba_out_2;
        __m256 rgba_out_3;

        if constexpr (M == BlendMode::Over) {
            __m256 fg_a_scaled_0 = _mm256_shuffle_ps(fg_scaled_0, fg_scaled_0, 255);
            __m256 fg_a_scaled_1 = _mm256_shuffle_ps(fg_scaled_1, fg_scaled_1, 255);
            __m256 fg_a_scaled_2 = _mm256_shuffle_ps(fg_scaled_2, fg_scaled_2, 255);
            __m256 fg_a_scaled_3 = _mm256_shuffle_ps(fg_scaled_3, fg_scaled_3, 255);

            __m256 F_b0 = _mm256_sub_ps(one, fg_a_scaled_0);
            __m256 F_b1 = _mm256_sub_ps(one, fg_a_scaled_1);
            __m256 F_b2 = _mm256_sub_ps(one, fg_a_scaled_2);
            __m256 F_b3 = _mm256_sub_ps(one, fg_a_scaled_3);

            rgba_out_0 = _mm256_fmadd_ps(F_b0, bg_v0, fg_scaled_0);
            rgba_out_1 = _mm256_fmadd_ps(F_b1, bg_v1, fg_scaled_1);
            rgba_out_2 = _mm256_fmadd_ps(F_b2, bg_v2, fg_scaled_2);
            rgba_out_3 = _mm256_fmadd_ps(F_b3, bg_v3, fg_scaled_3);
        } else if constexpr (M == BlendMode::In) {
            __m256 bg_a0 = _mm256_shuffle_ps(bg_v0, bg_v0, 255);
            __m256 bg_a1 = _mm256_shuffle_ps(bg_v1, bg_v1, 255);
            __m256 bg_a2 = _mm256_shuffle_ps(bg_v2, bg_v2, 255);
            __m256 bg_a3 = _mm256_shuffle_ps(bg_v3, bg_v3, 255);

            rgba_out_0 = _mm256_mul_ps(bg_a0, fg_scaled_0);
            rgba_out_1 = _mm256_mul_ps(bg_a1, fg_scaled_1);
            rgba_out_2 = _mm256_mul_ps(bg_a2, fg_scaled_2);
            rgba_out_3 = _mm256_mul_ps(bg_a3, fg_scaled_3);
        } else if constexpr (M == BlendMode::Out) {
            __m256 bg_a0 = _mm256_shuffle_ps(bg_v0, bg_v0, 255);
            __m256 bg_a1 = _mm256_shuffle_ps(bg_v1, bg_v1, 255);
            __m256 bg_a2 = _mm256_shuffle_ps(bg_v2, bg_v2, 255);
            __m256 bg_a3 = _mm256_shuffle_ps(bg_v3, bg_v3, 255);

            __m256 F_a0 = _mm256_sub_ps(one, bg_a0);
            __m256 F_a1 = _mm256_sub_ps(one, bg_a1);
            __m256 F_a2 = _mm256_sub_ps(one, bg_a2);
            __m256 F_a3 = _mm256_sub_ps(one, bg_a3);

            rgba_out_0 = _mm256_mul_ps(F_a0, fg_scaled_0);
            rgba_out_1 = _mm256_mul_ps(F_a1, fg_scaled_1);
            rgba_out_2 = _mm256_mul_ps(F_a2, fg_scaled_2);
            rgba_out_3 = _mm256_mul_ps(F_a3, fg_scaled_3);
        } else if constexpr (M == BlendMode::Atop) {
            __m256 bg_a0 = _mm256_shuffle_ps(bg_v0, bg_v0, 255);
            __m256 bg_a1 = _mm256_shuffle_ps(bg_v1, bg_v1, 255);
            __m256 bg_a2 = _mm256_shuffle_ps(bg_v2, bg_v2, 255);
            __m256 bg_a3 = _mm256_shuffle_ps(bg_v3, bg_v3, 255);

            __m256 fg_a_scaled_0 = _mm256_shuffle_ps(fg_scaled_0, fg_scaled_0, 255);
            __m256 fg_a_scaled_1 = _mm256_shuffle_ps(fg_scaled_1, fg_scaled_1, 255);
            __m256 fg_a_scaled_2 = _mm256_shuffle_ps(fg_scaled_2, fg_scaled_2, 255);
            __m256 fg_a_scaled_3 = _mm256_shuffle_ps(fg_scaled_3, fg_scaled_3, 255);

            __m256 F_b0 = _mm256_sub_ps(one, fg_a_scaled_0);
            __m256 F_b1 = _mm256_sub_ps(one, fg_a_scaled_1);
            __m256 F_b2 = _mm256_sub_ps(one, fg_a_scaled_2);
            __m256 F_b3 = _mm256_sub_ps(one, fg_a_scaled_3);

            rgba_out_0 = _mm256_fmadd_ps(bg_a0, fg_scaled_0, _mm256_mul_ps(F_b0, bg_v0));
            rgba_out_1 = _mm256_fmadd_ps(bg_a1, fg_scaled_1, _mm256_mul_ps(F_b1, bg_v1));
            rgba_out_2 = _mm256_fmadd_ps(bg_a2, fg_scaled_2, _mm256_mul_ps(F_b2, bg_v2));
            rgba_out_3 = _mm256_fmadd_ps(bg_a3, fg_scaled_3, _mm256_mul_ps(F_b3, bg_v3));
        } else if constexpr (M == BlendMode::Xor) {
            __m256 bg_a0 = _mm256_shuffle_ps(bg_v0, bg_v0, 255);
            __m256 bg_a1 = _mm256_shuffle_ps(bg_v1, bg_v1, 255);
            __m256 bg_a2 = _mm256_shuffle_ps(bg_v2, bg_v2, 255);
            __m256 bg_a3 = _mm256_shuffle_ps(bg_v3, bg_v3, 255);

            __m256 F_a0 = _mm256_sub_ps(one, bg_a0);
            __m256 F_a1 = _mm256_sub_ps(one, bg_a1);
            __m256 F_a2 = _mm256_sub_ps(one, bg_a2);
            __m256 F_a3 = _mm256_sub_ps(one, bg_a3);

            __m256 fg_a_scaled_0 = _mm256_shuffle_ps(fg_scaled_0, fg_scaled_0, 255);
            __m256 fg_a_scaled_1 = _mm256_shuffle_ps(fg_scaled_1, fg_scaled_1, 255);
            __m256 fg_a_scaled_2 = _mm256_shuffle_ps(fg_scaled_2, fg_scaled_2, 255);
            __m256 fg_a_scaled_3 = _mm256_shuffle_ps(fg_scaled_3, fg_scaled_3, 255);

            __m256 F_b0 = _mm256_sub_ps(one, fg_a_scaled_0);
            __m256 F_b1 = _mm256_sub_ps(one, fg_a_scaled_1);
            __m256 F_b2 = _mm256_sub_ps(one, fg_a_scaled_2);
            __m256 F_b3 = _mm256_sub_ps(one, fg_a_scaled_3);

            rgba_out_0 = _mm256_fmadd_ps(F_a0, fg_scaled_0, _mm256_mul_ps(F_b0, bg_v0));
            rgba_out_1 = _mm256_fmadd_ps(F_a1, fg_scaled_1, _mm256_mul_ps(F_b1, bg_v1));
            rgba_out_2 = _mm256_fmadd_ps(F_a2, fg_scaled_2, _mm256_mul_ps(F_b2, bg_v2));
            rgba_out_3 = _mm256_fmadd_ps(F_a3, fg_scaled_3, _mm256_mul_ps(F_b3, bg_v3));
        } else {
            throw std::runtime_error("Got unknown blend mode in SIMD kernel");
        }

        _mm256_storeu_ps(out_p + idx + 0 * 8, rgba_out_0);
        _mm256_storeu_ps(out_p + idx + 1 * 8, rgba_out_1);
        _mm256_storeu_ps(out_p + idx + 2 * 8, rgba_out_2);
        _mm256_storeu_ps(out_p + idx + 3 * 8, rgba_out_3);
    }

    for (; i < end; ++i) {
        const size_t idx = i * 4;

        float fg_alpha = fg_p[idx + 3] * global_opacity;
        float bg_alpha = bg_p[idx + 3];

        float F_a = 0.0f;
        float F_b = 0.0f;

        if constexpr (M == BlendMode::Over) {
            F_a = 1.0f;
            F_b = 1.0f - fg_alpha;
        } else if constexpr (M == BlendMode::In) {
            F_a = bg_alpha;
            F_b = 0.0f;
        } else if constexpr (M == BlendMode::Out) {
            F_a = 1.0f - bg_alpha;
            F_b = 0.0f;
        } else if constexpr (M == BlendMode::Atop) {
            F_a = bg_alpha;
            F_b = 1.0f - fg_alpha;
        } else if constexpr (M == BlendMode::Xor) {
            F_a = 1.0f - bg_alpha;
            F_b = 1.0f - fg_alpha;
        } else {
            throw std::runtime_error("Unknown blend mode in SIMD tail");
        }

        out_p[idx + 3] = F_a * fg_alpha + F_b * bg_alpha;
        out_p[idx + 0] = F_a * global_opacity * fg_p[idx + 0] + F_b * bg_p[idx + 0];
        out_p[idx + 1] = F_a * global_opacity * fg_p[idx + 1] + F_b * bg_p[idx + 1];
        out_p[idx + 2] = F_a * global_opacity * fg_p[idx + 2] + F_b * bg_p[idx + 2];
    }

    _mm256_zeroupper();
}

template <class Exec>
void blend_scalar_exec(const ImageRGBAf& fg,
                       const ImageRGBAf& bg,
                       ImageRGBAf& out,
                       float global_opacity,
                       BlendMode mode,
                       const Exec& exec)
{
    const size_t N = static_cast<size_t>(fg.width) * static_cast<size_t>(fg.height);

    out.width = fg.width;
    out.height = fg.height;
    out.data.resize(fg.data.size());

    const float* fg_p = fg.data.data();
    const float* bg_p = bg.data.data();
    float* out_p = out.data.data();

    switch (mode) {
        case BlendMode::Over:
            exec.for_pixels(N, [&](size_t begin, size_t end) {
                blend_scalar_chunk<BlendMode::Over>(fg_p, bg_p, out_p, begin, end, global_opacity);
            });
            break;
        case BlendMode::In:
            exec.for_pixels(N, [&](size_t begin, size_t end) {
                blend_scalar_chunk<BlendMode::In>(fg_p, bg_p, out_p, begin, end, global_opacity);
            });
            break;
        case BlendMode::Out:
            exec.for_pixels(N, [&](size_t begin, size_t end) {
                blend_scalar_chunk<BlendMode::Out>(fg_p, bg_p, out_p, begin, end, global_opacity);
            });
            break;
        case BlendMode::Atop:
            exec.for_pixels(N, [&](size_t begin, size_t end) {
                blend_scalar_chunk<BlendMode::Atop>(fg_p, bg_p, out_p, begin, end, global_opacity);
            });
            break;
        case BlendMode::Xor:
            exec.for_pixels(N, [&](size_t begin, size_t end) {
                blend_scalar_chunk<BlendMode::Xor>(fg_p, bg_p, out_p, begin, end, global_opacity);
            });
            break;
        default:
            throw std::runtime_error("Got unknown blend mode " + std::to_string(static_cast<int>(mode)));
    }
}

template <class Exec>
void blend_simd_exec(const ImageRGBAf& fg,
                     const ImageRGBAf& bg,
                     ImageRGBAf& out,
                     float global_opacity,
                     BlendMode mode,
                     const Exec& exec)
{
    const size_t N = static_cast<size_t>(fg.width) * static_cast<size_t>(fg.height);

    out.width = fg.width;
    out.height = fg.height;
    out.data.resize(fg.data.size());

    const float* fg_p = fg.data.data();
    const float* bg_p = bg.data.data();
    float* out_p = out.data.data();

    switch (mode) {
        case BlendMode::Over:
            exec.for_pixels(N, [&](size_t begin, size_t end) {
                blend_simd_chunk<BlendMode::Over>(fg_p, bg_p, out_p, begin, end, global_opacity);
            });
            break;
        case BlendMode::In:
            exec.for_pixels(N, [&](size_t begin, size_t end) {
                blend_simd_chunk<BlendMode::In>(fg_p, bg_p, out_p, begin, end, global_opacity);
            });
            break;
        case BlendMode::Out:
            exec.for_pixels(N, [&](size_t begin, size_t end) {
                blend_simd_chunk<BlendMode::Out>(fg_p, bg_p, out_p, begin, end, global_opacity);
            });
            break;
        case BlendMode::Atop:
            exec.for_pixels(N, [&](size_t begin, size_t end) {
                blend_simd_chunk<BlendMode::Atop>(fg_p, bg_p, out_p, begin, end, global_opacity);
            });
            break;
        case BlendMode::Xor:
            exec.for_pixels(N, [&](size_t begin, size_t end) {
                blend_simd_chunk<BlendMode::Xor>(fg_p, bg_p, out_p, begin, end, global_opacity);
            });
            break;
        default:
            throw std::runtime_error("Got unknown blend mode " + std::to_string(static_cast<int>(mode)));
    }
}

} // namespace

void blend_scalar(const ImageRGBAf& fg,
                  const ImageRGBAf& bg,
                  ImageRGBAf& out,
                  float global_opacity,
                  BlendMode mode)
{
    ensure_ok(global_opacity >= 0.0f && global_opacity <= 1.0f,
              "global_opacity should be in range [0, 1]. Got '" + std::to_string(global_opacity) + "' instead.");
    ensure_ok(fg.width == bg.width && fg.height == bg.height,
              "Images should be of the same size. Got:\n\tforeground image shape: ("
              + std::to_string(fg.width) + ", " + std::to_string(fg.height) + ")"
              + "background image shape: ("
              + std::to_string(bg.width) + ", " + std::to_string(bg.height) + ")");

    blend_scalar_exec(fg, bg, out, global_opacity, mode, SeqExec{});
}

void blend_simd(const ImageRGBAf& fg,
                const ImageRGBAf& bg,
                ImageRGBAf& out,
                float global_opacity,
                BlendMode mode)
{
    ensure_ok(global_opacity >= 0.0f && global_opacity <= 1.0f,
              "global_opacity should be in range [0, 1]. Got '" + std::to_string(global_opacity) + "' instead.");
    ensure_ok(fg.width == bg.width && fg.height == bg.height,
              "Images should be of the same size. Got:\n\tforeground image shape: ("
              + std::to_string(fg.width) + ", " + std::to_string(fg.height) + ")"
              + "background image shape: ("
              + std::to_string(bg.width) + ", " + std::to_string(bg.height) + ")");

    blend_simd_exec(fg, bg, out, global_opacity, mode, SeqExec{});
}

void blend_scalar_par(ThreadPool& pool,
                      const ImageRGBAf& fg,
                      const ImageRGBAf& bg,
                      ImageRGBAf& out,
                      float global_opacity,
                      BlendMode mode,
                      size_t grain)
{
    ensure_ok(global_opacity >= 0.0f && global_opacity <= 1.0f,
              "global_opacity should be in range [0, 1]. Got '" + std::to_string(global_opacity) + "' instead.");
    ensure_ok(fg.width == bg.width && fg.height == bg.height,
              "Images should be of the same size. Got:\n\tforeground image shape: ("
              + std::to_string(fg.width) + ", " + std::to_string(fg.height) + ")"
              + "background image shape: ("
              + std::to_string(bg.width) + ", " + std::to_string(bg.height) + ")");

    ParExec exec{pool, grain, 1};
    blend_scalar_exec(fg, bg, out, global_opacity, mode, exec);
}

void blend_simd_par(ThreadPool& pool,
                    const ImageRGBAf& fg,
                    const ImageRGBAf& bg,
                    ImageRGBAf& out,
                    float global_opacity,
                    BlendMode mode,
                    size_t grain)
{
    ensure_ok(global_opacity >= 0.0f && global_opacity <= 1.0f,
              "global_opacity should be in range [0, 1]. Got '" + std::to_string(global_opacity) + "' instead.");
    ensure_ok(fg.width == bg.width && fg.height == bg.height,
              "Images should be of the same size. Got:\n\tforeground image shape: ("
              + std::to_string(fg.width) + ", " + std::to_string(fg.height) + ")"
              + "background image shape: ("
              + std::to_string(bg.width) + ", " + std::to_string(bg.height) + ")");

    ParExec exec{pool, grain, 8};
    blend_simd_exec(fg, bg, out, global_opacity, mode, exec);
}


//=============================================================================//

//=============== Wrappers for preprocessing and postprocessing ===============//

void preprocess_images_scalar(const ImageRGBA8& fg_srgb,
                              const ImageRGBA8& bg_srgb,
                              ImageRGBAf& fg_linear,
                              ImageRGBAf& bg_linear)
{
    srgb_to_linear_scalar(fg_srgb, fg_linear);
    srgb_to_linear_scalar(bg_srgb, bg_linear);
    premultiply_inplace_scalar(fg_linear);
    premultiply_inplace_scalar(bg_linear);
}

void preprocess_images_simd(const ImageRGBA8& fg_srgb,
                            const ImageRGBA8& bg_srgb,
                            ImageRGBAf& fg_linear,
                            ImageRGBAf& bg_linear)
{
    srgb_to_linear_simd(fg_srgb, fg_linear);
    srgb_to_linear_simd(bg_srgb, bg_linear);
    premultiply_inplace_simd(fg_linear);
    premultiply_inplace_simd(bg_linear);
}

void preprocess_images_scalar_par(ThreadPool& pool,
                                  const ImageRGBA8& fg_srgb,
                                  const ImageRGBA8& bg_srgb,
                                  ImageRGBAf& fg_linear,
                                  ImageRGBAf& bg_linear,
                                  size_t grain)
{
    srgb_to_linear_scalar_par(pool, fg_srgb, fg_linear, true, grain);
    srgb_to_linear_scalar_par(pool, bg_srgb, bg_linear, true, grain);
    premultiply_inplace_scalar_par(pool, fg_linear, grain);
    premultiply_inplace_scalar_par(pool, bg_linear, grain);
}

void preprocess_images_simd_par(ThreadPool& pool,
                                const ImageRGBA8& fg_srgb,
                                const ImageRGBA8& bg_srgb,
                                ImageRGBAf& fg_linear,
                                ImageRGBAf& bg_linear,
                                size_t grain)
{
    srgb_to_linear_simd_par(pool, fg_srgb, fg_linear, true, grain);
    srgb_to_linear_simd_par(pool, bg_srgb, bg_linear, true, grain);
    premultiply_inplace_simd_par(pool, fg_linear, grain);
    premultiply_inplace_simd_par(pool, bg_linear, grain);
}

void postprocess_image_scalar(ImageRGBAf& out_linear, ImageRGBA8& out_srgb)
{
    revert_premultiply_inplace_scalar(out_linear);
    linear_to_srgb_scalar(out_linear, out_srgb);
}

void postprocess_image_simd(ImageRGBAf& out_linear, ImageRGBA8& out_srgb)
{
    revert_premultiply_inplace_simd(out_linear);
    linear_to_srgb_simd(out_linear, out_srgb);
}

void postprocess_image_scalar_par(ThreadPool& pool,
                                  ImageRGBAf& out_linear,
                                  ImageRGBA8& out_srgb,
                                  size_t grain)
{
    revert_premultiply_inplace_scalar_par(pool, out_linear, grain);
    linear_to_srgb_scalar_par(pool, out_linear, out_srgb, true, grain);
}

void postprocess_image_simd_par(ThreadPool& pool,
                                ImageRGBAf& out_linear,
                                ImageRGBA8& out_srgb,
                                size_t grain)
{
    revert_premultiply_inplace_simd_par(pool, out_linear, grain);
    linear_to_srgb_simd_par(pool, out_linear, out_srgb, true, grain);
}

//=============================================================================//

template <Impl I, ExecMode E>
ImageRGBA8 alpha_blend_pipeline_templ(
    const ImageRGBA8& fg_srgb,
    const ImageRGBA8& bg_srgb,
    float global_opacity,
    BlendMode mode,
    BlendStageTiming& timing,
    BlendStageCycles& cycles,
    ThreadPool* pool,
    size_t grain
) {
    ImageRGBAf fg_linear, bg_linear, out_linear;
    ImageRGBA8 out_srgb;
    std::uint32_t aux;

    if constexpr (I == Impl::Scalar) {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::uint64_t c0 = static_cast<std::uint64_t>(_rdtsc());
        if constexpr (E == ExecMode::Seq) {
            preprocess_images_scalar(fg_srgb, bg_srgb, fg_linear, bg_linear);
        } else {
            preprocess_images_scalar_par(require_pool(pool), fg_srgb, bg_srgb, fg_linear, bg_linear, grain);
        }
        std::uint64_t c1 = static_cast<std::uint64_t>(_rdtscp(&aux));
        auto t1 = std::chrono::high_resolution_clock::now();
        timing.preprocess = t1 - t0;
        cycles.preprocess = c1 - c0;

        t0 = std::chrono::high_resolution_clock::now();
        c0 = static_cast<std::uint64_t>(_rdtsc());
        if constexpr (E == ExecMode::Seq) {
            blend_scalar(fg_linear, bg_linear, out_linear, global_opacity, mode);
        } else {
            blend_scalar_par(require_pool(pool), fg_linear, bg_linear, out_linear, global_opacity, mode, grain);
        }
        c1 = static_cast<std::uint64_t>(_rdtscp(&aux));
        t1 = std::chrono::high_resolution_clock::now();
        timing.blend = t1 - t0;
        cycles.blend = c1 - c0;

        t0 = std::chrono::high_resolution_clock::now();
        c0 = static_cast<std::uint64_t>(_rdtsc());
        if constexpr (E == ExecMode::Seq) {
            postprocess_image_scalar(out_linear, out_srgb);
        } else {
            postprocess_image_scalar_par(require_pool(pool), out_linear, out_srgb, grain);
        }
        c1 = static_cast<std::uint64_t>(_rdtscp(&aux));
        t1 = std::chrono::high_resolution_clock::now();
        timing.postprocess = t1 - t0;
        cycles.postprocess = c1 - c0;
    } else {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::uint64_t c0 = static_cast<std::uint64_t>(_rdtsc());
        if constexpr (E == ExecMode::Seq) {
            preprocess_images_simd(fg_srgb, bg_srgb, fg_linear, bg_linear);
        } else {
            preprocess_images_simd_par(require_pool(pool), fg_srgb, bg_srgb, fg_linear, bg_linear, grain);
        }
        std::uint64_t c1 = static_cast<std::uint64_t>(_rdtscp(&aux));
        auto t1 = std::chrono::high_resolution_clock::now();
        timing.preprocess = t1 - t0;
        cycles.preprocess = c1 - c0;

        t0 = std::chrono::high_resolution_clock::now();
        c0 = static_cast<std::uint64_t>(_rdtsc());
        if constexpr (E == ExecMode::Seq) {
            blend_simd(fg_linear, bg_linear, out_linear, global_opacity, mode);
        } else {
            blend_simd_par(require_pool(pool), fg_linear, bg_linear, out_linear, global_opacity, mode, grain);
        }
        c1 = static_cast<std::uint64_t>(_rdtscp(&aux));
        t1 = std::chrono::high_resolution_clock::now();
        timing.blend = t1 - t0;
        cycles.blend = c1 - c0;

        t0 = std::chrono::high_resolution_clock::now();
        c0 = static_cast<std::uint64_t>(_rdtsc());
        if constexpr (E == ExecMode::Seq) {
            postprocess_image_simd(out_linear, out_srgb);
        } else {
            postprocess_image_simd_par(require_pool(pool), out_linear, out_srgb, grain);
        }
        c1 = static_cast<std::uint64_t>(_rdtscp(&aux));
        t1 = std::chrono::high_resolution_clock::now();
        timing.postprocess = t1 - t0;
        cycles.postprocess = c1 - c0;
    }

    return out_srgb;
}

template ImageRGBA8 alpha_blend_pipeline_templ<Impl::Scalar, ExecMode::Seq>(
    const ImageRGBA8&, const ImageRGBA8&, float, BlendMode, BlendStageTiming&, BlendStageCycles&, ThreadPool*, size_t);

template ImageRGBA8 alpha_blend_pipeline_templ<Impl::SIMD, ExecMode::Seq>(
    const ImageRGBA8&, const ImageRGBA8&, float, BlendMode, BlendStageTiming&, BlendStageCycles&, ThreadPool*, size_t);

template ImageRGBA8 alpha_blend_pipeline_templ<Impl::Scalar, ExecMode::Par>(
    const ImageRGBA8&, const ImageRGBA8&, float, BlendMode, BlendStageTiming&, BlendStageCycles&, ThreadPool*, size_t);

template ImageRGBA8 alpha_blend_pipeline_templ<Impl::SIMD, ExecMode::Par>(
    const ImageRGBA8&, const ImageRGBA8&, float, BlendMode, BlendStageTiming&, BlendStageCycles&, ThreadPool*, size_t);

} // namespace alpha_hist
