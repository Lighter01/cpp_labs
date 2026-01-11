#include "alpha_hist/image.hpp"

#include <chrono>
#include <cstdint>
namespace alpha_hist {

    enum class Impl {
        Scalar,
        SIMD
    };

    enum class BlendMode {
        Over,
        In,
        Out,
        Atop,
        Xor
    };

    template <BlendMode M>
    void blend_kernel_scalar(const ImageRGBAf& fg, const ImageRGBAf& bg, ImageRGBAf& out, float global_opacity);

    template <BlendMode M>
    void blend_kernel_simd(const ImageRGBAf& fg, const ImageRGBAf& bg, ImageRGBAf& out, float global_opacity);

    void blend_scalar(const ImageRGBAf& fg, const ImageRGBAf& bg, ImageRGBAf& out, 
                      float global_opacity = 1.0f, BlendMode mode = BlendMode::Over);

    void blend_simd(const ImageRGBAf& fg, const ImageRGBAf& bg, ImageRGBAf& out, 
                    float global_opacity = 1.0f, BlendMode mode = BlendMode::Over);

    struct BlendStageTiming {
        std::chrono::high_resolution_clock::duration preprocess{};
        std::chrono::high_resolution_clock::duration blend{};
        std::chrono::high_resolution_clock::duration postprocess{};
    };

    struct BlendStageCycles {
        std::uint64_t preprocess = 0;
        std::uint64_t blend = 0;
        std::uint64_t postprocess = 0;
    };

    template <Impl I>
    ImageRGBA8 alpha_blend_pipeline_templ(
        const ImageRGBA8& fg_srgb,
        const ImageRGBA8& bg_srgb,
        float global_opacity,
        BlendMode mode,
        BlendStageTiming& timing,
        BlendStageCycles& cycles
    );

    //=============================================================================//

    //=============== Wrappers for preprocessing and postprocessing ===============//

    void preprocess_images_scalar(const ImageRGBA8& fg_srgb,
                                  const ImageRGBA8& bg_srgb,
                                  ImageRGBAf& fg_linear,
                                  ImageRGBAf& bg_linear);

    void preprocess_images_simd(const ImageRGBA8& fg_srgb,
                                const ImageRGBA8& bg_srgb,
                                ImageRGBAf& fg_linear,
                                ImageRGBAf& bg_linear);

    void postprocess_image_scalar(ImageRGBAf& out_linear, ImageRGBA8& out_srgb);

    void postprocess_image_simd(ImageRGBAf& out_linear, ImageRGBA8& out_srgb);
}
