#include "alpha_hist/alpha_blend.hpp"
#include "alpha_hist/color_convert.hpp"

#include <cstdlib>
#include <chrono>
#include <string>
#include <tuple>
#include <stdexcept>
#include <immintrin.h>


namespace alpha_hist {

    static void ensure_ok(bool ok, const std::string& msg) {
        if (!ok) throw std::runtime_error(msg);
    }

    template <BlendMode M>
    void blend_kernel_scalar(const ImageRGBAf& fg, const ImageRGBAf& bg, ImageRGBAf& out, float global_opacity)
    {
        size_t N = static_cast<size_t>(fg.width) * static_cast<size_t>(fg.height);
        out.width = fg.width;
        out.height = fg.height;
        out.data.resize(fg.data.size());

        for (size_t i = 0; i < N; ++i) {
            size_t idx = i * 4;

            float fg_alpha = fg.data[idx + 3] * global_opacity;
            float bg_alpha = bg.data[idx + 3];

            float F_a, F_b;

            if constexpr (M == BlendMode::Over) {
                F_a = 1.0f;
                F_b = 1.0f - fg_alpha;
            }
            else if constexpr (M == BlendMode::In) {
                F_a = bg_alpha;
                F_b = 0.0f;
            }
            else if constexpr (M == BlendMode::Out) {
                F_a = 1 - bg_alpha;
                F_b = 0.0f;
            }
            else if constexpr (M == BlendMode::Atop) {
                F_a = bg_alpha;
                F_b = 1 - fg_alpha;
            }
            else if constexpr (M == BlendMode::Xor) {
                F_a = 1 - bg_alpha;
                F_b = 1 - fg_alpha;
            }
            else {
                throw std::runtime_error("Got uknown blend mode");
            }

            out.data[idx + 3] = F_a * fg_alpha + F_b * bg_alpha;

            out.data[idx]     = F_a * global_opacity * fg.data[idx]     + F_b * bg.data[idx];
            out.data[idx + 1] = F_a * global_opacity * fg.data[idx + 1] + F_b * bg.data[idx + 1];
            out.data[idx + 2] = F_a * global_opacity * fg.data[idx + 2] + F_b * bg.data[idx + 2];
        }
    }

    template <BlendMode M>
    void blend_kernel_simd(const ImageRGBAf& fg, const ImageRGBAf& bg, ImageRGBAf& out, float global_opacity)
    {
        size_t N = static_cast<size_t>(fg.width) * static_cast<size_t>(fg.height);
        out.width = fg.width;
        out.height = fg.height;
        out.data.resize(fg.data.size());

        const float* fg_p = fg.data.data();
        const float* bg_p = bg.data.data();
        float* out_p = out.data.data();

        const __m256 one = _mm256_set1_ps(1.0f);
        const __m256 zero = _mm256_setzero_ps();
        const __m256 glob_op = _mm256_set1_ps(global_opacity);

        const __m256i idx_alpha_rep = _mm256_setr_epi32(3,3,3,3, 7,7,7,7);

        constexpr int kAlphaBlendMask = 0x88; // 0b10001000

        size_t i = 0;
        for (; i + 1 < N; i += 2) {
            const size_t idx = i * 4;

            // Loading 2 pixels (RGBA X2)
            __m256 fg_v = _mm256_loadu_ps(fg_p + idx);
            __m256 bg_v = _mm256_loadu_ps(bg_p + idx);

            // replicating alpha lanes
            __m256 fg_a = _mm256_permutevar8x32_ps(fg_v, idx_alpha_rep);
            __m256 bg_a = _mm256_permutevar8x32_ps(bg_v, idx_alpha_rep);

            __m256 fg_a_scaled = _mm256_mul_ps(fg_a, glob_op);

            __m256 F_a, F_b;

            if constexpr (M == BlendMode::Over) {
                F_a = one;
                F_b = _mm256_sub_ps(one, fg_a_scaled);
            }
            else if constexpr (M == BlendMode::In) {
                F_a = bg_a;
                F_b = zero;
            }
            else if constexpr (M == BlendMode::Out) {
                F_a = _mm256_sub_ps(one, bg_a);
                F_b = zero;
            }
            else if constexpr (M == BlendMode::Atop) {
                F_a = bg_a;
                F_b = _mm256_sub_ps(one, fg_a_scaled);
            }
            else if constexpr (M == BlendMode::Xor) {
                F_a = _mm256_sub_ps(one, bg_a);
                F_b = _mm256_sub_ps(one, fg_a_scaled);
            }
            else {
                throw std::runtime_error("Got uknown blend mode in SIMD kernel");
            }

            // RGB
            __m256 fg_scaled = _mm256_mul_ps(fg_v, glob_op); // scales both RGB and alpha
            __m256 rgb_out = _mm256_fmadd_ps(F_a, fg_scaled, _mm256_mul_ps(F_b, bg_v));
            
            // alpha
            __m256 a_out_rep = _mm256_fmadd_ps(F_a, fg_a_scaled, _mm256_mul_ps(F_b, bg_a));

            // Put correct alpha only into lanes 3 and 7
            __m256 out_v = _mm256_blend_ps(rgb_out, a_out_rep, kAlphaBlendMask);

            _mm256_storeu_ps(out_p + idx, out_v);
        }

        // tail (1 pixel if N odd)
        for (; i < N; ++i) {
            const size_t idx = i * 4;

            float fg_alpha = fg_p[idx + 3] * global_opacity;
            float bg_alpha = bg_p[idx + 3];

            float F_a, F_b;

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
    }

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
        
        switch (mode) {
            case BlendMode::Over:
                blend_kernel_scalar<BlendMode::Over>(fg, bg, out, global_opacity);
                break;
            case BlendMode::In:
                blend_kernel_scalar<BlendMode::In>(fg, bg, out, global_opacity);
                break;
            case BlendMode::Out:
                blend_kernel_scalar<BlendMode::Out>(fg, bg, out, global_opacity);
                break;
            case BlendMode::Atop:
                blend_kernel_scalar<BlendMode::Atop>(fg, bg, out, global_opacity);
                break;
            case BlendMode::Xor:
                blend_kernel_scalar<BlendMode::Xor>(fg, bg, out, global_opacity);
                break;
            default:
                throw std::runtime_error("Got uknown blend mode " + std::to_string(static_cast<int>(mode)));
        }
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
        
        switch (mode) {
            case BlendMode::Over:
                blend_kernel_simd<BlendMode::Over>(fg, bg, out, global_opacity);
                break;
            case BlendMode::In:
                blend_kernel_simd<BlendMode::In>(fg, bg, out, global_opacity);
                break;
            case BlendMode::Out:
                blend_kernel_simd<BlendMode::Out>(fg, bg, out, global_opacity);
                break;
            case BlendMode::Atop:
                blend_kernel_simd<BlendMode::Atop>(fg, bg, out, global_opacity);
                break;
            case BlendMode::Xor:
                blend_kernel_simd<BlendMode::Xor>(fg, bg, out, global_opacity);
                break;
            default:
                throw std::runtime_error("Got uknown blend mode " + std::to_string(static_cast<int>(mode)));
        }
    }


    template <Impl I>
    ImageRGBA8 alpha_blend_pipeline_templ(
        const ImageRGBA8& fg_srgb,
        const ImageRGBA8& bg_srgb,
        float global_opacity,
        BlendMode mode,
        BlendStageTiming& timing,
        BlendStageCycles& cycles
    ) {
        ImageRGBAf fg_linear, bg_linear, out_linear;
        ImageRGBA8 out_srgb;

        if constexpr (I == Impl::Scalar) {
            auto t0 = std::chrono::high_resolution_clock::now();
            std::uint64_t c0 = static_cast<std::uint64_t>(_rdtsc());
            preprocess_images_scalar(fg_srgb, bg_srgb, fg_linear, bg_linear);
            std::uint64_t c1 = static_cast<std::uint64_t>(_rdtsc());
            auto t1 = std::chrono::high_resolution_clock::now();
            timing.preprocess = t1 - t0;
            cycles.preprocess = c1 - c0;

            t0 = std::chrono::high_resolution_clock::now();
            c0 = static_cast<std::uint64_t>(_rdtsc());
            blend_scalar(fg_linear, bg_linear, out_linear, global_opacity, mode);
            c1 = static_cast<std::uint64_t>(_rdtsc());
            t1 = std::chrono::high_resolution_clock::now();
            timing.blend = t1 - t0;
            cycles.blend = c1 - c0;

            t0 = std::chrono::high_resolution_clock::now();
            c0 = static_cast<std::uint64_t>(_rdtsc());
            postprocess_image_scalar(out_linear, out_srgb);
            c1 = static_cast<std::uint64_t>(_rdtsc());
            t1 = std::chrono::high_resolution_clock::now();
            timing.postprocess = t1 - t0;
            cycles.postprocess = c1 - c0;
        } else {
            auto t0 = std::chrono::high_resolution_clock::now();
            std::uint64_t c0 = static_cast<std::uint64_t>(_rdtsc());
            preprocess_images_simd(fg_srgb, bg_srgb, fg_linear, bg_linear);
            std::uint64_t c1 = static_cast<std::uint64_t>(_rdtsc());
            auto t1 = std::chrono::high_resolution_clock::now();
            timing.preprocess = t1 - t0;
            cycles.preprocess = c1 - c0;

            t0 = std::chrono::high_resolution_clock::now();
            c0 = static_cast<std::uint64_t>(_rdtsc());
            blend_simd(fg_linear, bg_linear, out_linear, global_opacity, mode);
            c1 = static_cast<std::uint64_t>(_rdtsc());
            t1 = std::chrono::high_resolution_clock::now();
            timing.blend = t1 - t0;
            cycles.blend = c1 - c0;

            t0 = std::chrono::high_resolution_clock::now();
            c0 = static_cast<std::uint64_t>(_rdtsc());
            postprocess_image_simd(out_linear, out_srgb);
            c1 = static_cast<std::uint64_t>(_rdtsc());
            t1 = std::chrono::high_resolution_clock::now();
            timing.postprocess = t1 - t0;
            cycles.postprocess = c1 - c0;
        }

        return out_srgb;
    }

    template ImageRGBA8 alpha_blend_pipeline_templ<Impl::Scalar>(
        const ImageRGBA8&, const ImageRGBA8&, float, BlendMode, BlendStageTiming&, BlendStageCycles&);

    template ImageRGBA8 alpha_blend_pipeline_templ<Impl::SIMD>(
        const ImageRGBA8&, const ImageRGBA8&, float, BlendMode, BlendStageTiming&, BlendStageCycles&);
    
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

}
