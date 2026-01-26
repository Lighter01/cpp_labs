#include "alpha_hist/thread_pool.hpp"

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

size_t ThreadPool::thread_count() const noexcept {
    return workers_.size();
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