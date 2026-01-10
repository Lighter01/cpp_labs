#include "alpha_hist/histogram.hpp"

#include <stdexcept>
#include <cstdlib>
#include <immintrin.h>

namespace alpha_hist {

    void histogram_scalar(const ImageGray8& in, std::array<std::uint32_t, 256>& out)
    {   
        const std::uint8_t* pixels = in.data.data();
        size_t N = static_cast<size_t>(in.height) * in.width;

        for (size_t i = 0; i < N; ++i) {
            ++out[pixels[i]];
        }
    }
    
    // CODEX TO IMPLEMENT
    void histogram_simd(const ImageGray8& in, std::array<std::uint32_t, 256>& out)
    {
        const std::uint8_t* pixels = in.data.data();
        size_t N = static_cast<size_t>(in.height) * in.width;

        std::array<std::uint32_t, 256> h0{};
        std::array<std::uint32_t, 256> h1{};
        std::array<std::uint32_t, 256> h2{};
        std::array<std::uint32_t, 256> h3{};

        size_t i = 0;
        alignas(32) std::uint32_t idx_buf[8];

        // Process 32 pixels per iteration and accumulate into 4 independent histograms.
        for (; i + 32 <= N; i += 32) {
            __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pixels + i));

            __m128i lo = _mm256_castsi256_si128(v);
            __m128i hi = _mm256_extracti128_si256(v, 1);

            __m256i idx0 = _mm256_cvtepu8_epi32(lo);
            __m256i idx1 = _mm256_cvtepu8_epi32(_mm_srli_si128(lo, 8));
            __m256i idx2 = _mm256_cvtepu8_epi32(hi);
            __m256i idx3 = _mm256_cvtepu8_epi32(_mm_srli_si128(hi, 8));

            _mm256_store_si256(reinterpret_cast<__m256i*>(idx_buf), idx0);
            ++h0[idx_buf[0]]; ++h0[idx_buf[1]]; ++h0[idx_buf[2]]; ++h0[idx_buf[3]];
            ++h0[idx_buf[4]]; ++h0[idx_buf[5]]; ++h0[idx_buf[6]]; ++h0[idx_buf[7]];

            _mm256_store_si256(reinterpret_cast<__m256i*>(idx_buf), idx1);
            ++h1[idx_buf[0]]; ++h1[idx_buf[1]]; ++h1[idx_buf[2]]; ++h1[idx_buf[3]];
            ++h1[idx_buf[4]]; ++h1[idx_buf[5]]; ++h1[idx_buf[6]]; ++h1[idx_buf[7]];

            _mm256_store_si256(reinterpret_cast<__m256i*>(idx_buf), idx2);
            ++h2[idx_buf[0]]; ++h2[idx_buf[1]]; ++h2[idx_buf[2]]; ++h2[idx_buf[3]];
            ++h2[idx_buf[4]]; ++h2[idx_buf[5]]; ++h2[idx_buf[6]]; ++h2[idx_buf[7]];

            _mm256_store_si256(reinterpret_cast<__m256i*>(idx_buf), idx3);
            ++h3[idx_buf[0]]; ++h3[idx_buf[1]]; ++h3[idx_buf[2]]; ++h3[idx_buf[3]];
            ++h3[idx_buf[4]]; ++h3[idx_buf[5]]; ++h3[idx_buf[6]]; ++h3[idx_buf[7]];
        }

        for (size_t b = 0; b < out.size(); ++b) {
            out[b] += h0[b] + h1[b] + h2[b] + h3[b];
        }

        for (; i < N; ++i) {
            ++out[pixels[i]];
        }
    }

    template <Impl I>
    std::array<std::uint32_t, 256> histogram(const ImageGray8& in)
    {
        std::array<std::uint32_t, 256> out{};

        if constexpr (I == Impl::Scalar) {
            histogram_scalar(in, out);
        } else {
            histogram_simd(in, out);
        }

        return out;
    }

}
