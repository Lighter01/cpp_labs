#include "alpha_hist/image.hpp"
#include "alpha_hist/alpha_blend.hpp"

#include <array>
#include <cstdint>

namespace alpha_hist {

    void histogram_scalar(const ImageGray8& in, std::array<std::uint32_t, 256>& out);

    void histogram_simd(const ImageGray8& in, std::array<std::uint32_t, 256>& out);

    template <Impl I>
    std::array<std::uint32_t, 256> histogram(const ImageGray8& in);

}