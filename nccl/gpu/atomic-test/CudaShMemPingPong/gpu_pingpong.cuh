#ifndef GPU_PINGPONG_CUH
#define GPU_PINGPONG_CUH

// #include "gpu_data_functions.cuh"
#include "structs.cuh"

#define PING 1
#define PONG 0
#define PANG 2

template <typename T, typename S>
__global__ void device_fetch_add_relaxed(T *flag, S *sig, clock_t *time) {
    // sig->store(PING);

    sig->fetch_add(PING);
    while(sig->load() != PANG);

    clock_t start = clock64();
    for (size_t i = 0; i < 10000; ++i) {
        flag->fetch_add(1, cuda::memory_order_relaxed);
    }
    clock_t end = clock64();

    *time = end - start;
}

template <typename T, typename S>
__global__ void device_fetch_add_acqrel(T *flag, S *sig, clock_t *time) {
    // sig->store(PING);

    sig->fetch_add(PING);
    while(sig->load() != PANG);

    clock_t start = clock64();
    for (size_t i = 0; i < 10000; ++i) {
        flag->fetch_add(1, cuda::memory_order_acq_rel);
    }
    clock_t end = clock64();

    *time = end - start;
}

template <typename T, typename S>
__global__ void device_fetch_add_seqcst(T *flag, S *sig, clock_t *time) {
    // sig->store(PING);

    sig->fetch_add(PING);
    while(sig->load() != PANG);

    clock_t start = clock64();
    for (size_t i = 0; i < 10000; ++i) {
        flag->fetch_add(1, cuda::memory_order_seq_cst);
    }
    clock_t end = clock64();

    *time = end - start;
}
template <typename T, typename S>
__global__ void device_fetch_add_relaxed_store(T *flag, S *sig, clock_t *time) {
    sig->store(PING);

    // sig->fetch_add(PING);
    // while(sig->load() != PANG);

    clock_t start = clock64();
    for (size_t i = 0; i < 10000; ++i) {
        flag->fetch_add(1, cuda::memory_order_relaxed);
    }
    clock_t end = clock64();

    *time = end - start;
}

template <typename T, typename S>
__global__ void device_fetch_add_acqrel_store(T *flag, S *sig, clock_t *time) {
    sig->store(PING);

    // sig->fetch_add(PING);
    // while(sig->load() != PANG);

    clock_t start = clock64();
    for (size_t i = 0; i < 10000; ++i) {
        flag->fetch_add(1, cuda::memory_order_acq_rel);
    }
    clock_t end = clock64();

    *time = end - start;
}

template <typename T, typename S>
__global__ void device_fetch_add_seqcst_store(T *flag, S *sig, clock_t *time) {
    sig->store(PING);

    // sig->fetch_add(PING);
    // while(sig->load() != PANG);

    clock_t start = clock64();
    for (size_t i = 0; i < 10000; ++i) {
        flag->fetch_add(1, cuda::memory_order_seq_cst);
    }
    clock_t end = clock64();

    *time = end - start;
}
template <typename T, typename S>
__global__ void device_fetch_add_relaxed_wait(T *flag, S *sig, clock_t *time) {
    // sig->store(PING);

    while(sig->load() != PING);

    // sig->fetch_add(PING);
    // while(sig->load() != PANG);

    clock_t start = clock64();
    for (size_t i = 0; i < 10000; ++i) {
        flag->fetch_add(1, cuda::memory_order_relaxed);
    }
    clock_t end = clock64();

    *time = end - start;
}

template <typename T, typename S>
__global__ void device_fetch_add_acqrel_wait(T *flag, S *sig, clock_t *time) {
    // sig->store(PING);

    while(sig->load() != PING);

    // sig->fetch_add(PING);
    // while(sig->load() != PANG);

    clock_t start = clock64();
    for (size_t i = 0; i < 10000; ++i) {
        flag->fetch_add(1, cuda::memory_order_acq_rel);
    }
    clock_t end = clock64();

    *time = end - start;
}

template <typename T, typename S>
__global__ void device_fetch_add_seqcst_wait(T *flag, S *sig, clock_t *time) {
    // sig->store(PING);

    while(sig->load() != PING);

    // sig->fetch_add(PING);
    // while(sig->load() != PANG);

    clock_t start = clock64();
    for (size_t i = 0; i < 10000; ++i) {
        flag->fetch_add(1, cuda::memory_order_seq_cst);
    }
    clock_t end = clock64();

    *time = end - start;
}

// change pong to ping
template <typename T>
__global__ void device_pong_kernel_relaxed_base(T *flag) {
    flag->store(PING, cuda::memory_order_relaxed);
    uint16_t expected = PONG;
    for (size_t i = 0; i < 10000; ++i) {
        while( !flag->compare_exchange_strong(expected, PING, cuda::std::memory_order_relaxed, cuda::std::memory_order_relaxed)) {
            expected = PONG;
        }
    }
}

// https://stackoverflow.com/questions/60624189/atomic-compare-exchange-strong-explicit-what-do-the-various-combinations
template <typename T>
__global__ void device_pong_kernel_acqrel_base(T *flag) {
    flag->store(PING, cuda::memory_order_relaxed);
    uint16_t expected = PONG;
    for (size_t i = 0; i < 10000; ++i) {
        while( !flag->compare_exchange_strong(expected, PING, cuda::std::memory_order_acq_rel, cuda::std::memory_order_acquire)) {
            expected = PONG;
        }
    }
}

template <typename T>
__global__ void device_pong_kernel_relaxed_decoupled(T *flag) {
    flag->store(PING, cuda::memory_order_relaxed);
    uint16_t expected = PONG;
    for (size_t i = 0; i < 10000; ++i) {
        while (flag->load(cuda::memory_order_relaxed) != expected) {
            expected = PONG;
        }
        // *data = i * 32;
        flag->store(PING, cuda::memory_order_relaxed);
    }
}

template <typename T>
__global__ void device_pong_kernel_acqrel_decoupled(T* flag) {
    flag->store(PING, cuda::memory_order_relaxed);
    uint16_t expected = PONG;
    for (size_t i = 0; i < 10000; ++i) {
        while (flag->load(cuda::memory_order_acquire) != expected) {
            expected = PONG;
        }
        // *data = i * 32;
        flag->store(PING, cuda::memory_order_release);
    }
}


template <typename T>
__global__ void device_ping_kernel_relaxed_base(T *flag, clock_t *time) {
    uint16_t expected = PING;
    while (flag->load(cuda::memory_order_relaxed) == PONG);

    clock_t start = clock64();
    for (size_t i = 0; i < 10000; ++i) {
        while (!flag->compare_exchange_strong(expected, PONG, cuda::std::memory_order_relaxed, cuda::std::memory_order_relaxed)) {
            expected = PING;
        }
    }
    clock_t end = clock64();
    *time = end - start;
}

template <typename T>
__global__ void device_ping_kernel_acqrel_base(T *flag, clock_t *time) {
    uint16_t expected = PING;
    while (flag->load(cuda::memory_order_relaxed) == PONG); // S

    clock_t start = clock64();
    for (size_t i = 0; i < 10000; ++i) {
        while (!flag->compare_exchange_strong(expected, PONG, cuda::std::memory_order_acq_rel, cuda::std::memory_order_acquire)) {
            expected = PING;
        }
    }
    clock_t end = clock64();
    *time = end - start;
}

template <typename T>
__global__ void device_ping_kernel_relaxed_decoupled(T* flag, clock_t *time) {
    uint16_t expected = PING;
    while (flag->load(cuda::memory_order_relaxed) == PONG);

    clock_t start = clock64();
    for (size_t i = 0; i < 10000; ++i) {
        while (flag->load(cuda::memory_order_relaxed) != expected) {
            expected = PING;
        }
        // *data = i * 32;
        flag->store(PONG, cuda::memory_order_relaxed);
    }
    clock_t end = clock64();
    *time = end - start;
}

template <typename T>
__global__ void device_ping_kernel_acqrel_decoupled(T *flag, clock_t *time) {
    uint16_t expected = PING;
    while (flag->load(cuda::memory_order_relaxed) == PONG);

    clock_t start = clock64();
    for (size_t i = 0; i < 10000; ++i) {
        while (flag->load(cuda::memory_order_acquire) != expected) {
            expected = PING;
        }
        // *data = i * 32;
        flag->store(PONG, cuda::memory_order_release);
    }
    clock_t end = clock64();
    *time = end - start;
}

#endif // GPU_PINGPONG_CUH  