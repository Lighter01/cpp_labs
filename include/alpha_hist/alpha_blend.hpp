#include "alpha_hist/image.hpp"

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

    template <Impl I>
    ImageRGBA8 alpha_blend_pipeline_templ(
        const ImageRGBA8& fg_srgb,
        const ImageRGBA8& bg_srgb,
        float global_opacity,
        BlendMode mode = BlendMode::Over
    );

}