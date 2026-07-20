#ifndef CPU_UTILS_H
#define CPU_UTILS_H

#include <iostream>

constexpr size_t cpu_cacheline = 64;
constexpr size_t gpu_cacheline = 128;

__attribute__((always_inline)) inline uint64_t get_cpu_clock() {
    uint64_t tsc;

    asm volatile("isb" : : : "memory");
    asm volatile("mrs %0, cntvct_el0" : "=r"(tsc) :: "memory"); // alternative is cntpct_el0

    return tsc;
}

__attribute__((always_inline)) inline uint64_t get_cpu_freq() {
    uint64_t freq;

    asm volatile("mrs %0, cntfrq_el0" : "=r"(freq) :: "memory");

    return freq;
}

__attribute__((always_inline)) __device__ inline clock_t get_gpu_clock() {
    uint64_t tsc;

    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tsc));

    return tsc;
}

int get_gpu_freq() {
    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, 0);

    return deviceProperties.clockRate;
}

#endif