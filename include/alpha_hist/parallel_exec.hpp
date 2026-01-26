#pragma once

#include "alpha_hist/thread_pool.hpp"

#include <algorithm>
#include <cstddef>
#include <future>
#include <vector>

namespace alpha_hist {

    struct SeqExec {
        template <typename Fn>
        void for_pixels(size_t n, Fn&& fn) const {
            if (n == 0) {
                return;
            }
            fn(0, n);
        }
    };

    struct ParExec {
        ThreadPool& pool;
        size_t grain;
        size_t align;

        template <class Fn>
        void for_pixels(size_t n, Fn&& fn) const {
            if (n == 0) {
                return;
            }

            const size_t use_align = (align == 0) ? 1 : align;
            const size_t use_grain = (grain == 0) ? use_align : grain;
            const size_t step = round_up(use_grain, use_align);

            if (n <= step) {
                fn(0, n);
                return;
            }

            std::vector<std::future<void>> futs;
            futs.reserve((n + step - 1) / step);

            for (size_t b = 0; b < n; b += step) {
                const size_t e = std::min(n, b + step);
                futs.emplace_back(pool.submit([&fn, b, e] { fn(b, e); }));
            }

            for (auto& f : futs) {
                f.get();
            }
        }

    private:
        static size_t round_up(size_t value, size_t multiple) {
            if (multiple == 0) {
                return value;
            }
            const size_t rem = value % multiple;
            return (rem == 0) ? value : (value + multiple - rem);
        }
    };

} // namespace alpha_hist
