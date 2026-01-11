#include "alpha_hist/image.hpp"
#include "alpha_hist/color_lut.hpp"

namespace alpha_hist {

    void srgb_to_linear_scalar(const ImageRGBA8& in, ImageRGBAf& out, bool use_exact_srgb = true);
    void srgb_to_linear_simd(const ImageRGBA8& in, ImageRGBAf& out, bool use_exact_srgb = true);

    void linear_to_srgb_scalar(const ImageRGBAf& in, ImageRGBA8& out, bool use_exact_srgb = true);
    void linear_to_srgb_simd(const ImageRGBAf& in, ImageRGBA8& out, bool use_exact_srgb = true);

    void premultiply_inplace_scalar(ImageRGBAf& in);
    void premultiply_inplace_simd(ImageRGBAf& in);

    void revert_premultiply_inplace_scalar(ImageRGBAf& in);
    void revert_premultiply_inplace_simd(ImageRGBAf& in);

}
