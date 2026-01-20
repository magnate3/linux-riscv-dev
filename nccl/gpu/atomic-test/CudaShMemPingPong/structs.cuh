#ifndef STRUCTS_CUH
#define STRUCTS_CUH

#include <cuda/atomic>
#include "cpu_utils.hpp"

#define ITERATIONS 1


enum CachelineType {
    SAME,
    DIFF_CPU,
    DIFF_GPU
};

enum Scope {
    THREAD,
    BLOCK,
    DEVICE,
    SYSTEM
};

enum MemOrder {
    RELAXED,
    ACQ_REL
};

enum Allocator {
    CUDA_MALLOC_HOST,
    MALLOC,
    CUDA_MALLOC,
    UM,
};

enum ProducerConsumerTypes {
    CPU,
    GPU
};

struct alignedDataSameCacheline_thread {
    alignas(cpu_cacheline) cuda::atomic<int, cuda::thread_scope_thread> flag;
    uint32_t data;
};

struct alignedDataDiffCPUCacheline_thread {
    alignas(cpu_cacheline) cuda::atomic<int, cuda::thread_scope_thread> flag;
    alignas(cpu_cacheline) uint32_t data;
};

struct alignedDataDiffGPUCacheline_thread {
    alignas(gpu_cacheline) cuda::atomic<int, cuda::thread_scope_thread> flag;
    alignas(gpu_cacheline) uint32_t data;
};

struct alignedDataSameCacheline_block {
    alignas(cpu_cacheline) cuda::atomic<int, cuda::thread_scope_block> flag;
    uint32_t data;
};

struct alignedDataDiffCPUCacheline_block {
    alignas(cpu_cacheline) cuda::atomic<int, cuda::thread_scope_block> flag;
    alignas(cpu_cacheline) uint32_t data;
};

struct alignedDataDiffGPUCacheline_block {
    alignas(gpu_cacheline) cuda::atomic<int, cuda::thread_scope_block> flag;
    alignas(gpu_cacheline) uint32_t data;
};


struct alignedDataSameCacheline_gpu {
    alignas(cpu_cacheline) cuda::atomic<int, cuda::thread_scope_device> flag;
    uint32_t data;
};

struct alignedDataDiffCPUCacheline_gpu {
    alignas(cpu_cacheline) cuda::atomic<int, cuda::thread_scope_device> flag;
    alignas(cpu_cacheline) uint32_t data;
};

struct alignedDataDiffGPUCacheline_gpu {
    alignas(gpu_cacheline) cuda::atomic<int, cuda::thread_scope_device> flag;
    alignas(gpu_cacheline) uint32_t data;
};

struct alignedDataSameCacheline_sys {
    alignas(cpu_cacheline) cuda::atomic<int, cuda::thread_scope_system> flag;
    uint32_t data;
};

struct alignedDataDiffCPUCacheline_sys {
    alignas(cpu_cacheline) cuda::atomic<int, cuda::thread_scope_system> flag;
    alignas(cpu_cacheline) uint32_t data;
};

struct alignedDataDiffGPUCacheline_sys {
    alignas(gpu_cacheline) cuda::atomic<int, cuda::thread_scope_system> flag;
    alignas(gpu_cacheline) uint32_t data;
};

#endif // STRUCTS_CUH
