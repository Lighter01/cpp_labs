#pragma once

#include "alpha_hist/thread_pool.hpp"

#include <algorithm>
#include <cstddef>
#include <future>
#include <vector>
#include <latch>
#include <iostream>

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
    size_t grain = 0; // optional override
    size_t align = 1;

    template <class Fn>
        void for_pixels(size_t n, Fn&& fn) const {
            if (n == 0) return;

            const size_t threads = std::max<size_t>(1, pool.thread_count());
            const size_t use_align = (align == 0) ? 1 : align;

            const size_t target_chunks = threads * 2;

            size_t step;
            // step = (n + target_chunks - 1) / target_chunks;
            // step = round_up(step, use_align);

            if (grain != 0) {
                step = round_up(grain, use_align);
            } else {
                // dynamic step based on n and threads
                size_t base = (n + target_chunks - 1) / target_chunks;
                // also enforce a minimum so chunks aren’t tiny
                base = std::max(base, size_t(256 * 1024)); // 256k pixels minimum
                step = round_up(base, use_align);
                // std::cerr << "Base: " << base << "\n";
            }

        // #ifndef NDEBUG
        //     std::cerr << "[ParExec] n=" << n
        //             << " step=" << step
        //             << " grain=" << grain
        //             << " tasks=" << ((n + step - 1) / step)
        //             << " threads=" << pool.thread_count()
        //             << "\n";
        // #endif

            if (n <= step || threads == 1) {
                fn(0, n);
                return;
            }

            const size_t num_tasks = (n + step - 1) / step;
            std::latch done(num_tasks);

            std::exception_ptr ep;
            std::mutex ep_m;

            for (size_t b = 0; b < n; b += step) {
                const size_t e = std::min(n, b + step);
                pool.enqueue([&, b, e] {
                    try {
                        fn(b, e);
                    } catch (...) {
                        std::lock_guard<std::mutex> lk(ep_m);
                        if (!ep) ep = std::current_exception();
                    }
                    done.count_down();
                });
            }

            done.wait();
            if (ep) std::rethrow_exception(ep);
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
