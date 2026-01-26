# pragma once

#include "alpha_hist/image.hpp"
#include "alpha_hist/color_lut.hpp"
#include "alpha_hist/io.hpp"

#include <cstdint>
#include <vector>
#include <cmath>


namespace alpha_hist {

    ImageRGBA8 get_background(const ImageRGBA8& img);

    enum class GenMode {
        Stripe45,
        Stripe135,
        Rectangle,
        Circle,
        SineBands,
        Lissajous,
        Spiral,
        RoundedRects,
        Concentric,
        Blobs,
        RandomForms,
        GaussianNoise,
    };

    ImageRGBA8 generate_test_image_rgba(size_t w, size_t h, GenMode img_type);

}
