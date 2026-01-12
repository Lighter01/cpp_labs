#include "alpha_hist/histogram.hpp"

#include <stdexcept>
#include <cstdlib>
#include <immintrin.h>

#define CSIZE (256 + 8)

static void inline __attribute__((always_inline)) histend4(
    const std::array<std::array<std::uint32_t, CSIZE>, 4>& h,  
    std::array<std::uint32_t, 256>& out
) {
    std::uint32_t* __restrict pOut = out.data();

    for (size_t i = 0; i != 256; i+=8) {
        __m256i sv =                  _mm256_load_si256(reinterpret_cast<const __m256i*>(&h[0][i]));
                sv = _mm256_add_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(&h[1][i])), sv);
                sv = _mm256_add_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(&h[2][i])), sv);
                sv = _mm256_add_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(&h[3][i])), sv);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&pOut[i]), sv);
    }
}

static void inline __attribute__((always_inline)) histend8(
    const std::array<std::array<std::uint32_t, CSIZE>, 8>& h,  
    std::array<std::uint32_t, 256>& out
) {
    std::uint32_t* __restrict pOut = out.data();

    for (size_t i = 0; i != 256; i+=8) {
        __m256i v0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&h[0][i]));
        __m256i v1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&h[1][i]));
        __m256i s0 = _mm256_add_epi32(v0, v1);
                v0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&h[2][i]));
                v1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&h[3][i]));
        __m256i s1 = _mm256_add_epi32(v0, v1);
                s0 = _mm256_add_epi32(s0, s1);

                v0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&h[4][i]));
                v1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&h[5][i]));
                s0 = _mm256_add_epi32(v0, v1);
                v0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&h[6][i]));
                v1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&h[7][i]));
                s1 = _mm256_add_epi32(v0, v1);
                s0 = _mm256_add_epi32(s0, s1);

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&pOut[i]), _mm256_add_epi32(s0, s1));
    }
}

namespace alpha_hist {

    void histogram_scalar(const ImageGray8& in, std::array<std::uint32_t, 256>& out)
    {   
        const std::uint8_t* pixels = in.data.data();
        size_t N = static_cast<size_t>(in.height) * in.width;

        std::uint32_t h0[256] = {};
        std::uint32_t h1[256] = {};
        std::uint32_t h2[256] = {};
        std::uint32_t h3[256] = {};

        size_t i = 0;
        for (; i + 4 <= N; i += 4) {
            ++h0[pixels[i]];
            ++h1[pixels[i+1]];
            ++h2[pixels[i+2]];
            ++h3[pixels[i+3]];
        }
        for (; i < N; ++i) ++h0[pixels[i]];

        for (size_t b = 0; b < 256; ++b) {
            out[b] += h0[b] + h1[b] + h2[b] + h3[b];
        }

    }
    
    void histogram_simd(const ImageGray8& in, std::array<std::uint32_t, 256>& out)
    {
        const std::uint8_t* pixels = in.data.data();
        size_t N = static_cast<size_t>(in.height) * in.width;

        alignas(32) std::array<std::array<std::uint32_t, CSIZE>, 4> h{};
        if (N >= (128 + 64)) {
            __m256i u0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pixels));
            __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pixels+32));

            for (; pixels <= (in.data.data()+((N-(64+128)) & ~127)); pixels += 128) {
                __m256i u1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pixels+64));
                __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pixels+64+32));

                h[0][_mm256_extract_epi8(u0, 0)]++;
                h[1][_mm256_extract_epi8(v0, 0)]++;
                h[2][_mm256_extract_epi8(u0, 1)]++;
                h[3][_mm256_extract_epi8(v0, 1)]++;
                h[0][_mm256_extract_epi8(u0, 2)]++;
                h[1][_mm256_extract_epi8(v0, 2)]++;
                h[2][_mm256_extract_epi8(u0, 3)]++;
                h[3][_mm256_extract_epi8(v0, 3)]++;
                h[0][_mm256_extract_epi8(u0, 4)]++;
                h[1][_mm256_extract_epi8(v0, 4)]++;
                h[2][_mm256_extract_epi8(u0, 5)]++;
                h[3][_mm256_extract_epi8(v0, 5)]++;
                h[0][_mm256_extract_epi8(u0, 6)]++;
                h[1][_mm256_extract_epi8(v0, 6)]++;
                h[2][_mm256_extract_epi8(u0, 7)]++;
                h[3][_mm256_extract_epi8(v0, 7)]++;
                h[0][_mm256_extract_epi8(u0, 8)]++;
                h[1][_mm256_extract_epi8(v0, 8)]++;
                h[2][_mm256_extract_epi8(u0, 9)]++;
                h[3][_mm256_extract_epi8(v0, 9)]++;
                h[0][_mm256_extract_epi8(u0, 10)]++;
                h[1][_mm256_extract_epi8(v0, 10)]++;
                h[2][_mm256_extract_epi8(u0, 11)]++;
                h[3][_mm256_extract_epi8(v0, 11)]++;
                h[0][_mm256_extract_epi8(u0, 12)]++;
                h[1][_mm256_extract_epi8(v0, 12)]++;
                h[2][_mm256_extract_epi8(u0, 13)]++;
                h[3][_mm256_extract_epi8(v0, 13)]++;
                h[0][_mm256_extract_epi8(u0, 14)]++;
                h[1][_mm256_extract_epi8(v0, 14)]++;
                h[2][_mm256_extract_epi8(u0, 15)]++;
                h[3][_mm256_extract_epi8(v0, 15)]++;
                h[0][_mm256_extract_epi8(u0, 16)]++;
                h[1][_mm256_extract_epi8(v0, 16)]++;
                h[2][_mm256_extract_epi8(u0, 17)]++;
                h[3][_mm256_extract_epi8(v0, 17)]++;
                h[0][_mm256_extract_epi8(u0, 18)]++;
                h[1][_mm256_extract_epi8(v0, 18)]++;
                h[2][_mm256_extract_epi8(u0, 19)]++;
                h[3][_mm256_extract_epi8(v0, 19)]++;
                h[0][_mm256_extract_epi8(u0, 20)]++;
                h[1][_mm256_extract_epi8(v0, 20)]++;
                h[2][_mm256_extract_epi8(u0, 21)]++;
                h[3][_mm256_extract_epi8(v0, 21)]++;
                h[0][_mm256_extract_epi8(u0, 22)]++;
                h[1][_mm256_extract_epi8(v0, 22)]++;
                h[2][_mm256_extract_epi8(u0, 23)]++;
                h[3][_mm256_extract_epi8(v0, 23)]++;
                h[0][_mm256_extract_epi8(u0, 24)]++;
                h[1][_mm256_extract_epi8(v0, 24)]++;
                h[2][_mm256_extract_epi8(u0, 25)]++;
                h[3][_mm256_extract_epi8(v0, 25)]++;
                h[0][_mm256_extract_epi8(u0, 26)]++;
                h[1][_mm256_extract_epi8(v0, 26)]++;
                h[2][_mm256_extract_epi8(u0, 27)]++;
                h[3][_mm256_extract_epi8(v0, 27)]++;
                h[0][_mm256_extract_epi8(u0, 28)]++;
                h[1][_mm256_extract_epi8(v0, 28)]++;
                h[2][_mm256_extract_epi8(u0, 29)]++;
                h[3][_mm256_extract_epi8(v0, 29)]++;
                h[0][_mm256_extract_epi8(u0, 30)]++;
                h[1][_mm256_extract_epi8(v0, 30)]++;
                h[2][_mm256_extract_epi8(u0, 31)]++;
                h[3][_mm256_extract_epi8(v0, 31)]++;

                u0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pixels+64+64));
                v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pixels+64+96));
                h[0][_mm256_extract_epi8(u1, 0)]++;
                h[1][_mm256_extract_epi8(v1, 0)]++;
                h[2][_mm256_extract_epi8(u1, 1)]++;
                h[3][_mm256_extract_epi8(v1, 1)]++;
                h[0][_mm256_extract_epi8(u1, 2)]++;
                h[1][_mm256_extract_epi8(v1, 2)]++;
                h[2][_mm256_extract_epi8(u1, 3)]++;
                h[3][_mm256_extract_epi8(v1, 3)]++;
                h[0][_mm256_extract_epi8(u1, 4)]++;
                h[1][_mm256_extract_epi8(v1, 4)]++;
                h[2][_mm256_extract_epi8(u1, 5)]++;
                h[3][_mm256_extract_epi8(v1, 5)]++;
                h[0][_mm256_extract_epi8(u1, 6)]++;
                h[1][_mm256_extract_epi8(v1, 6)]++;
                h[2][_mm256_extract_epi8(u1, 7)]++;
                h[3][_mm256_extract_epi8(v1, 7)]++;
                h[0][_mm256_extract_epi8(u1, 8)]++;
                h[1][_mm256_extract_epi8(v1, 8)]++;
                h[2][_mm256_extract_epi8(u1, 9)]++;
                h[3][_mm256_extract_epi8(v1, 9)]++;
                h[0][_mm256_extract_epi8(u1, 10)]++;
                h[1][_mm256_extract_epi8(v1, 10)]++;
                h[2][_mm256_extract_epi8(u1, 11)]++;
                h[3][_mm256_extract_epi8(v1, 11)]++;
                h[0][_mm256_extract_epi8(u1, 12)]++;
                h[1][_mm256_extract_epi8(v1, 12)]++;
                h[2][_mm256_extract_epi8(u1, 13)]++;
                h[3][_mm256_extract_epi8(v1, 13)]++;
                h[0][_mm256_extract_epi8(u1, 14)]++;
                h[1][_mm256_extract_epi8(v1, 14)]++;
                h[2][_mm256_extract_epi8(u1, 15)]++;
                h[3][_mm256_extract_epi8(v1, 15)]++;
                h[0][_mm256_extract_epi8(u1, 16)]++;
                h[1][_mm256_extract_epi8(v1, 16)]++;
                h[2][_mm256_extract_epi8(u1, 17)]++;
                h[3][_mm256_extract_epi8(v1, 17)]++;
                h[0][_mm256_extract_epi8(u1, 18)]++;
                h[1][_mm256_extract_epi8(v1, 18)]++;
                h[2][_mm256_extract_epi8(u1, 19)]++;
                h[3][_mm256_extract_epi8(v1, 19)]++;
                h[0][_mm256_extract_epi8(u1, 20)]++;
                h[1][_mm256_extract_epi8(v1, 20)]++;
                h[2][_mm256_extract_epi8(u1, 21)]++;
                h[3][_mm256_extract_epi8(v1, 21)]++;
                h[0][_mm256_extract_epi8(u1, 22)]++;
                h[1][_mm256_extract_epi8(v1, 22)]++;
                h[2][_mm256_extract_epi8(u1, 23)]++;
                h[3][_mm256_extract_epi8(v1, 23)]++;
                h[0][_mm256_extract_epi8(u1, 24)]++;
                h[1][_mm256_extract_epi8(v1, 24)]++;
                h[2][_mm256_extract_epi8(u1, 25)]++;
                h[3][_mm256_extract_epi8(v1, 25)]++;
                h[0][_mm256_extract_epi8(u1, 26)]++;
                h[1][_mm256_extract_epi8(v1, 26)]++;
                h[2][_mm256_extract_epi8(u1, 27)]++;
                h[3][_mm256_extract_epi8(v1, 27)]++;
                h[0][_mm256_extract_epi8(u1, 28)]++;
                h[1][_mm256_extract_epi8(v1, 28)]++;
                h[2][_mm256_extract_epi8(u1, 29)]++;
                h[3][_mm256_extract_epi8(v1, 29)]++;
                h[0][_mm256_extract_epi8(u1, 30)]++;
                h[1][_mm256_extract_epi8(v1, 30)]++;
                h[2][_mm256_extract_epi8(u1, 31)]++;
                h[3][_mm256_extract_epi8(v1, 31)]++;

                _mm_prefetch(reinterpret_cast<const std::uint8_t*>(pixels+512), 0);
            }
        }
        while (pixels < in.data.data() + N) h[0][*pixels++]++;
        histend4(h, out);
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
