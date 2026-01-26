#include "alpha_hist/image.hpp"
#include "alpha_hist/color_lut.hpp"

#include <cstddef>

class ThreadPool;

namespace alpha_hist {

    void srgb_to_linear_scalar(const ImageRGBA8& in, ImageRGBAf& out, bool use_exact_srgb = true);
    void srgb_to_linear_simd(const ImageRGBA8& in, ImageRGBAf& out, bool use_exact_srgb = true);

    void srgb_to_linear_scalar_par(ThreadPool& pool,
                                   const ImageRGBA8& in,
                                   ImageRGBAf& out,
                                   bool use_exact_srgb = true,
                                   size_t grain = 0);

    void srgb_to_linear_simd_par(ThreadPool& pool,
                                 const ImageRGBA8& in,
                                 ImageRGBAf& out,
                                 bool use_exact_srgb = true,
                                 size_t grain = 0);

    void linear_to_srgb_scalar(const ImageRGBAf& in, ImageRGBA8& out, bool use_exact_srgb = true);
    void linear_to_srgb_simd(const ImageRGBAf& in, ImageRGBA8& out, bool use_exact_srgb = true);

    void linear_to_srgb_scalar_par(ThreadPool& pool,
                                   const ImageRGBAf& in,
                                   ImageRGBA8& out,
                                   bool use_exact_srgb = true,
                                   size_t grain = 0);

    void linear_to_srgb_simd_par(ThreadPool& pool,
                                 const ImageRGBAf& in,
                                 ImageRGBA8& out,
                                 bool use_exact_srgb = true,
                                 size_t grain = 0);

    void premultiply_inplace_scalar(ImageRGBAf& in);
    void premultiply_inplace_simd(ImageRGBAf& in);

    void premultiply_inplace_scalar_par(ThreadPool& pool, ImageRGBAf& in, size_t grain = 0);
    void premultiply_inplace_simd_par(ThreadPool& pool, ImageRGBAf& in, size_t grain = 0);

    void revert_premultiply_inplace_scalar(ImageRGBAf& in);
    void revert_premultiply_inplace_simd(ImageRGBAf& in);

    void revert_premultiply_inplace_scalar_par(ThreadPool& pool, ImageRGBAf& in, size_t grain = 0);
    void revert_premultiply_inplace_simd_par(ThreadPool& pool, ImageRGBAf& in, size_t grain = 0);

}
