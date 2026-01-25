#pragma once

#include <vector>
#include <thread>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <stop_token>

class ThreadPool {
private:
    std::mutex mutex_;
    std::condition_variable_any cv_;
    std::atomic_bool accepting_{true};

    std::vector<std::jthread> workers_;
    std::queue<std::function<void()>> tasks_;

    void worker_loop(std::stop_token st);

public:
    ThreadPool(std::size_t thread_count = std::thread::hardware_concurrency());

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    template <typename Fn, typename... Args>
    auto submit(Fn&& f, Args&&... args) -> std::future<std::invoke_result_t<Fn, Args...>>;
    
    void shutdown();

    ~ThreadPool();
};

//==================================================================

//========== Private methods ==========//

void ThreadPool::worker_loop(std::stop_token st) {
    for (;;) {
        std::function<void()> task;
        
        {
            std::unique_lock<std::mutex> lock(mutex_);
            // Wait until there is work OR stop is requested
            cv_.wait(lock, st, [this]{ return !tasks_.empty(); });
            
            // If woken by stop request and no tasks left -> exit
            if (tasks_.empty()) {
                return;
            }

            task = std::move(tasks_.front());
            tasks_.pop();
        }

        task();
    }
}

//========== Public Interface ==========//

ThreadPool::ThreadPool(std::size_t thread_count) {
    if (thread_count == 0) thread_count = 1;

    for (std::size_t i = 0; i < thread_count; ++i) {
        workers_.emplace_back([this](std::stop_token st) { worker_loop(st); });
    }
}

template <typename Fn, typename... Args>
auto ThreadPool::submit(Fn&& f, Args&&... args) -> std::future<std::invoke_result_t<Fn, Args...>> 
{
    using result_type = std::invoke_result_t<Fn, Args...>;

    auto task = std::make_shared<std::packaged_task<result_type()>> (
        std::bind(std::forward<Fn>(f), std::forward<Args>(args)...)
    );
    
    std::future<result_type> result = task->get_future();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!accepting_.load(std::memory_order_relaxed)) {
            throw std::runtime_error("ThreadPool is stopping");
        }
        tasks_.emplace([task]() { (*task)(); });
    }

    cv_.notify_one();
    return result;
}

void ThreadPool::shutdown() {
    // run once
    if (!accepting_.exchange(false)) return;

    // Ask all workers to stop (they will still drain queued tasks)
    for (auto& w : workers_) {
        w.request_stop();
    }

    cv_.notify_all();

    // jthread joins on destruction; clearing forces join now
    workers_.clear();
}

ThreadPool::~ThreadPool() {
    shutdown();
}


// class ThreadPool {
// private:
//     std::mutex mutex_;
//     std::condition_variable cv_;
//     bool stop_{false};
//     std::atomic_bool is_active_{true};

//     std::vector<std::thread> workers_;
//     std::queue<std::function<void()>> tasks_;

//     void worker_loop();
//     void request_stop();

// public:
//     ThreadPool(std::size_t thread_count = std::thread::hardware_concurrency());

//     ThreadPool(const ThreadPool&) = delete;
//     ThreadPool& operator=(const ThreadPool&) = delete;

//     ThreadPool(ThreadPool&&) = delete;
//     ThreadPool& operator=(ThreadPool&&) = delete;

//     template <typename Fn, typename... Args>
//     auto submit(Fn&& f, Args&&... args) -> std::future<std::invoke_result_t<Fn, Args...>>;
    
//     void shutdown();

//     ~ThreadPool();
// };

// //==================================================================

// //========== Private methods ==========//

// void ThreadPool::worker_loop() {
//     for (;;) {
//         std::function<void()> task;
        
//         {
//             std::unique_lock<std::mutex> lock(mutex_);
//             cv_.wait(lock, [this]{
//                 return stop_ || !tasks_.empty();
//             });

//             if (stop_ && tasks_.empty()) {
//                 return;
//             }

//             task = std::move(tasks_.front());
//             tasks_.pop();
//         }

//         task();
//     }
// }

// void ThreadPool::request_stop() {
//     {
//         std::lock_guard<std::mutex> lock(mutex_);
//         stop_ = true;
//     }
//     cv_.notify_all();
// }

// //========== Public Interface ==========//

// ThreadPool::ThreadPool(std::size_t thread_count) {
//     if (thread_count == 0) thread_count = 1;

//     try {
//         for (std::size_t i = 0; i < thread_count; ++i) {
//             workers_.emplace_back(&ThreadPool::worker_loop, this);
//         }
//     } catch (...) {
//         request_stop();
//         for (auto& w : workers_) {
//             if (w.joinable()) {
//                 w.join();
//             }
//         }
//         throw;
//     }
// }

// template <typename Fn, typename... Args>
// auto ThreadPool::submit(Fn&& f, Args&&... args) -> std::future<std::invoke_result_t<Fn, Args...>> 
// {
//     using result_type = std::invoke_result_t<Fn, Args...>;

//     auto task = 
//         std::make_shared<std::packaged_task<result_type()>>(
//             std::bind(std::forward<Fn>(f), std::forward<Args>(args)...)
//         );
    
//     std::future<result_type> result = task->get_future();

//     {
//         std::lock_guard<std::mutex> lock(mutex_);
//         if (stop_) {
//             throw std::runtime_error("ThreadPool is stopping");
//         }
//         tasks_.emplace([task]() { (*task)(); });
//     }

//     cv_.notify_one();
//     return result;
// }

// void ThreadPool::shutdown() {
//     bool expected = true;
//     if (!is_active_.compare_exchange_strong(expected, false)) {
//         // already stopping
//         return;
//     }

//     request_stop();

//     for (auto& worker: workers_) {
//         if (worker.joinable()) {
//             worker.join();
//         }
//     }
// }

// ThreadPool::~ThreadPool() {
//     shutdown();
// }