#include "alpha_hist/image.hpp"
#include "alpha_hist/color_lut.hpp"

namespace alpha_hist {

    void srgb_to_linear(const ImageRGBA8& in, ImageRGBAf& out, bool use_exact_srgb = true);

    void linear_to_srgb(const ImageRGBAf& in, ImageRGBA8& out, bool use_exact_srgb = true);

    void premultiply_inplace(ImageRGBAf& in);

    void revert_premultiply_inplace(ImageRGBAf& in);

}