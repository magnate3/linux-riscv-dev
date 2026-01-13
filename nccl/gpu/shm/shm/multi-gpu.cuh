#ifndef MULTI_GPU_CUH
#define MULTI_GPU_CUH

// C
#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <assert.h>

// C++
#include <iostream>
#include <memory>
#include <chrono>

// CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda_profiler_api.h"

// CUDA driver API
#define CUCHECK(cmd) do {                                     \
    CUresult err = cmd;                                       \
    if (err != CUDA_SUCCESS) {                                \
        const char *errStr;                                   \
        cuGetErrorString(err, &errStr);                       \
        fprintf(stderr, "Failed: CUDA error %s:%d '%s'\n",    \
            __FILE__, __LINE__, errStr);                      \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

// CUDA runtime API
#define CUDACHECK(cmd) do {                                   \
    cudaError_t err = cmd;                                    \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "Failed: CUDA error %s:%d '%s'\n",    \
            __FILE__, __LINE__, cudaGetErrorString(err));     \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)
#if 0
// NCCL
#define NCCLCHECK(cmd) do {                                   \
    ncclResult_t res = cmd;                                   \
if (res != ncclSuccess) {                                     \
    fprintf(stderr, "Failed, NCCL error %s:%d '%s'\n",        \
        __FILE__, __LINE__, ncclGetErrorString(res));         \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

// NVML
#define NVMLCHECK(cmd) do {                                   \
    nvmlReturn_t res = cmd;                                   \
    if (res != NVML_SUCCESS) {                                \
        fprintf(stderr, "Failed, NVML error %s:%d '%s'\n",    \
            __FILE__, __LINE__, nvmlErrorString(res));        \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)
#endif

/* Threadpool (from 29-basic-threadpoo.c, used in 30-multimem-reduce-opt-tp.cu)*/

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

#include <iostream> // for testing
#include <chrono> // for testing

/*
    Example usage

ThreadGang gang(NUM_DEVICES); // threadpool
gang.execute([](int dev_idx) { // set devices
    CUDACHECK(cudaSetDevice(dev_idx));
});
gang.execute([](int dev_idx) { // check devices (sanity check for myself)
    int dev;
    CUDACHECK(cudaGetDevice(&dev));
    if (dev != dev_idx) {
        fprintf(stderr, "Device mismatch: expected %d, got %d\n", dev_idx, dev);
        exit(1);
    }
});

*/
class ThreadGang {
public:
    ThreadGang(size_t num_threads);
    ~ThreadGang();

    void execute(std::function<void(int)> task); // also waits for all threads to finish

private:
    // Condition indicators
    bool stop;
    std::vector<bool> task_available;
    int n_task_done;

    // Threadpool
    std::vector<std::thread> workers;
    
    // Main entry point for each thread
    void worker(int thread_id);

    // Used to dispatch work to all threads
    std::function<void(int)> current_task;

    // Synchronization
    std::mutex mutex;
    std::condition_variable cond_task_available;
    std::condition_variable cond_task_done;
};

ThreadGang::ThreadGang(size_t num_threads) : stop(false), n_task_done(0) {
    for (size_t i = 0; i < num_threads; ++i) {
        task_available.push_back(false);
        workers.emplace_back([this, i] { worker(i); });
    }
}

ThreadGang::~ThreadGang() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        stop = true;
    }
    cond_task_available.notify_all();
    for (std::thread &worker : workers) {
        worker.join();
    }
}

void ThreadGang::execute(std::function<void(int)> task) {
    {
        std::lock_guard<std::mutex> lock(mutex);
        current_task = task;
        for (size_t i = 0; i < task_available.size(); ++i)
            task_available[i] = true;
    }
    cond_task_available.notify_all();
    {
        std::unique_lock<std::mutex> lock(mutex);
        cond_task_done.wait(lock, [this] { return n_task_done == workers.size(); });
        n_task_done = 0;
    }
}

void ThreadGang::worker(int thread_id) {
    while (true) {
        std::function<void(int)> task;
        {
            std::unique_lock<std::mutex> lock(mutex);
            cond_task_available.wait(lock, [this, thread_id] { return stop || task_available[thread_id]; });

            if (stop)
                return;

            task = current_task;
            task_available[thread_id] = false;
        }
        task(thread_id);
        {
            std::lock_guard<std::mutex> lock(mutex); // adds 10 microseconds overhead
            ++n_task_done;
            if (n_task_done == workers.size())
                cond_task_done.notify_one();
        }
    }
}

/*
    Example usage:

    benchmark("Sorting a large vector", [] {
        std::vector<int> v(1'000'000);
        std::iota(v.begin(), v.end(), 0);  // Fill with 0,1,2,...999999
        std::sort(v.begin(), v.end(), std::greater<>()); // Sort in descending order
    });
*/
template <typename Func>
void benchmark(const char* message, Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << message << ": " << elapsed.count() << " ms" << std::endl;
}

#endif // MULTI_GPU_CUH
