#ifndef CPU_PINGPONG_HPP
#define CPU_PINGPONG_HPP

#include "gpu_pingpong.cuh"

void host_fetch_add_relaxed(std::atomic<uint16_t> *flag, std::atomic<uint16_t> *sig, uint64_t *time) {
    // while (sig->load() == PONG);

    sig->fetch_add(PING);
    while(sig->load() != PANG);

    uint64_t start = get_cpu_clock();
    for (size_t i = 0; i < 10000; ++i) {
        flag->fetch_add(1, std::memory_order_relaxed);
    }
    uint64_t end = get_cpu_clock();
    *time = end - start;
}

void host_fetch_add_acqrel(std::atomic<uint16_t> *flag, std::atomic<uint16_t> *sig, uint64_t *time) {
    // while (sig->load() == PONG);

    sig->fetch_add(PING);
    while(sig->load() != PANG);

    uint64_t start = get_cpu_clock();
    for (size_t i = 0; i < 10000; ++i) {
        flag->fetch_add(1, std::memory_order_acq_rel);
    }
    uint64_t end = get_cpu_clock();
    *time = end - start;
}

void host_fetch_add_seqcst(std::atomic<uint16_t> *flag, std::atomic<uint16_t> *sig, uint64_t *time) {
    // while (sig->load() == PONG);

    sig->fetch_add(PING);
    while(sig->load() != PANG);

    uint64_t start = get_cpu_clock();
    for (size_t i = 0; i < 10000; ++i) {
        flag->fetch_add(1, std::memory_order_seq_cst);
    }
    uint64_t end = get_cpu_clock();
    *time = end - start;
}

// change ping to pong
void host_ping_function_relaxed_base(std::atomic<uint16_t> *flag, uint64_t *time) {
    while (flag->load() == PONG);
    uint16_t expected = PING;

    uint64_t start = get_cpu_clock();
    for (size_t i = 0; i < 10000; ++i) {
        while (!flag->compare_exchange_strong(expected, PONG, std::memory_order_relaxed, std::memory_order_relaxed)) {
            expected = PING;
        }
        std::cout << i << std::endl;
    }
    uint64_t end = get_cpu_clock();
    *time = end - start;
}


void host_ping_function_acqrel_base(std::atomic<uint16_t> *flag, uint64_t *time) {
    while (flag->load() == PONG);
    uint16_t expected = PING;

    uint64_t start = get_cpu_clock();
    for (size_t i = 0; i < 10000; ++i) {
        while (!flag->compare_exchange_strong(expected, PONG, std::memory_order_acq_rel, std::memory_order_acquire)) {
            expected = PING;
        }
    }
    uint64_t end = get_cpu_clock();
    *time = end - start;
}

void host_ping_function_relaxed_decoupled(std::atomic<uint16_t> *flag, uint64_t *time) {
    while (flag->load() == PONG);
    uint16_t expected = PING;

    uint64_t start = get_cpu_clock();
    for (size_t i = 0; i < 10000; ++i) {
        while (flag->load(std::memory_order_relaxed) != expected) {
            expected = PING;
        }
        flag->store(PONG, std::memory_order_relaxed);
    }
    uint64_t end = get_cpu_clock();
    *time = end - start;
}

void host_ping_function_acqrel_decoupled(std::atomic<uint16_t> *flag, uint64_t *time) {
    while (flag->load() == PONG);
    uint16_t expected = PING;

    uint64_t start = get_cpu_clock();
    for (size_t i = 0; i < 10000; ++i) {
        while (flag->load(std::memory_order_acquire) != expected) {
            expected = PING;
        }
        flag->store(PONG, std::memory_order_release);
    }
    uint64_t end = get_cpu_clock();
    *time = end - start;
}


void host_pong_function_relaxed_base(std::atomic<uint16_t> *flag) {
    uint16_t expected = PONG;
    flag->store(PING);
    for (size_t i = 0; i < 10000; ++i) {
        while (!flag->compare_exchange_strong(expected, PING, std::memory_order_relaxed, std::memory_order_relaxed)) {
            expected = PONG;
        }
    }
}

void host_pong_function_relaxed_decoupled(std::atomic<uint16_t> *flag) {
    uint16_t expected = PONG;
    flag->store(PING);
    for (size_t i = 0; i < 10000; ++i) {
        while (flag->load(std::memory_order_relaxed) != expected) {
            expected = PONG;
        }
        // std::cout << i * 1000000. << std::endl;
        flag->store(PING, std::memory_order_relaxed);
    }
}


void host_pong_function_acqrel_base(std::atomic<uint16_t> *flag) {
    uint16_t expected = PONG;
    flag->store(PING);
    for (size_t i = 0; i < 10000; ++i) {
        while (!flag->compare_exchange_strong(expected, PING, std::memory_order_acq_rel, std::memory_order_acquire)) {
            expected = PONG;
        }
    }
}

void host_pong_function_acqrel_decoupled(std::atomic<uint16_t> *flag) {
    uint16_t expected = PONG;
    flag->store(PING);
    for (size_t i = 0; i < 10000; ++i) {
        while (flag->load(std::memory_order_acquire) != expected) {
            expected = PONG;
        }
        // std::cout << i * 1000000. << std::endl;
        flag->store(PING, std::memory_order_release);
    }
}

void device_device_fetch_add(Allocator allocator) {
    cuda::atomic<uint16_t, cuda::thread_scope_thread> *flag_thread_relaxed;
    cuda::atomic<uint16_t, cuda::thread_scope_device> *flag_device_relaxed;
    cuda::atomic<uint16_t, cuda::thread_scope_system> *flag_system_relaxed;

    cuda::atomic<uint16_t, cuda::thread_scope_system> *sig;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&sig, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == MALLOC) {
        sig = (cuda::atomic<uint16_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == UM) {
        cudaMallocManaged(&sig, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == CUDA_MALLOC) {
        cudaMalloc(&sig, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    }

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&flag_thread_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocHost(&flag_device_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocHost(&flag_system_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == MALLOC) {
        flag_thread_relaxed = (cuda::atomic<uint16_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        flag_device_relaxed = (cuda::atomic<uint16_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        flag_system_relaxed = (cuda::atomic<uint16_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == UM) {
        cudaMallocManaged(&flag_thread_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocManaged(&flag_device_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocManaged(&flag_system_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == CUDA_MALLOC) {
        cudaMalloc(&flag_thread_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMalloc(&flag_device_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMalloc(&flag_system_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    }

    if (allocator == CUDA_MALLOC) {
        // clear flag and sig
        cudaMemset(flag_thread_relaxed, 0, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMemset(flag_device_relaxed, 0, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMemset(flag_system_relaxed, 0, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
        cudaMemset(sig, 0, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else {
        sig->store(PONG);
    }

    clock_t *gpu_time_system_relaxed_store;
    clock_t *gpu_time_system_relaxed_wait;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&gpu_time_system_relaxed_store, sizeof(clock_t));
        cudaMallocHost(&gpu_time_system_relaxed_wait, sizeof(clock_t));
    } else if (allocator == MALLOC) {
        gpu_time_system_relaxed_store = (clock_t *) malloc(sizeof(clock_t));
        gpu_time_system_relaxed_wait = (clock_t *) malloc(sizeof(clock_t));
    } else if (allocator == UM) {
        cudaMallocManaged(&gpu_time_system_relaxed_store, sizeof(clock_t));
        cudaMallocManaged(&gpu_time_system_relaxed_wait, sizeof(clock_t));
    } else if (allocator == CUDA_MALLOC) {
        cudaMalloc(&gpu_time_system_relaxed_store, sizeof(clock_t));
        cudaMalloc(&gpu_time_system_relaxed_wait, sizeof(clock_t));
    }

    cudaStream_t stream_system_relaxed_store;
    cudaStream_t stream_system_relaxed_wait;

    cudaStreamCreate(&stream_system_relaxed_store);
    cudaStreamCreate(&stream_system_relaxed_wait);

    device_fetch_add_relaxed<<<1,1,0, stream_system_relaxed_store>>>(flag_system_relaxed, sig, gpu_time_system_relaxed_store);
    device_fetch_add_relaxed<<<1,1,0, stream_system_relaxed_wait>>>(flag_system_relaxed, sig, gpu_time_system_relaxed_wait);

    cudaStreamSynchronize(stream_system_relaxed_store);
    cudaStreamSynchronize(stream_system_relaxed_wait);

    cudaDeviceSynchronize();

    if (allocator == CUDA_MALLOC) {
        int flag;
        clock_t gpu_clock_store;
        clock_t gpu_clock_wait;

        cudaMemcpy(&flag, flag_system_relaxed, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&gpu_clock_store, gpu_time_system_relaxed_store, sizeof(clock_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&gpu_clock_wait, gpu_time_system_relaxed_wait, sizeof(clock_t), cudaMemcpyDeviceToHost);

        std::cout << "Device-Fetch-Add Device-Fetch-Add (System, Relaxed) | Value : " << flag << " | Store : " << ((double) gpu_clock_store / 10000) / ((double) get_gpu_freq()) * 1000000. << " | Wait : " << ((double) gpu_clock_wait / 10000) / ((double) get_gpu_freq()) * 1000000. << std::endl;
    } else {
        std::cout << "Device-Fetch-Add Device-Fetch-Add (System, Relaxed) | Value:  " << *flag_system_relaxed << " | Store : " << ((double) (*gpu_time_system_relaxed_store / 10000)) / ((double) get_gpu_freq()) * 1000000. << " | Wait : " << ((double) (*gpu_time_system_relaxed_wait / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;
    }

    cudaStreamDestroy(stream_system_relaxed_store);
    cudaStreamDestroy(stream_system_relaxed_wait);

    if (allocator == CUDA_MALLOC) {
        cudaMemset(sig, 0, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else {
        sig->store(PONG);
    }
    
    clock_t *gpu_time_device_relaxed_store;
    clock_t *gpu_time_device_relaxed_wait;
    
    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&gpu_time_device_relaxed_store, sizeof(clock_t));
        cudaMallocHost(&gpu_time_device_relaxed_wait, sizeof(clock_t));
    } else if (allocator == MALLOC) {
        gpu_time_device_relaxed_store = (clock_t *) malloc(sizeof(clock_t));
        gpu_time_device_relaxed_wait = (clock_t *) malloc(sizeof(clock_t));
    } else if (allocator == UM) {
        cudaMallocManaged(&gpu_time_device_relaxed_store, sizeof(clock_t));
        cudaMallocManaged(&gpu_time_device_relaxed_wait, sizeof(clock_t));
    } else if (allocator == CUDA_MALLOC) {
        cudaMalloc(&gpu_time_device_relaxed_store, sizeof(clock_t));
        cudaMalloc(&gpu_time_device_relaxed_wait, sizeof(clock_t));
    }

    cudaStream_t stream_device_relaxed_store;
    cudaStream_t stream_device_relaxed_wait;
    
    cudaStreamCreate(&stream_device_relaxed_store);
    cudaStreamCreate(&stream_device_relaxed_wait);

    device_fetch_add_relaxed<<<1,1,0, stream_device_relaxed_store>>>(flag_device_relaxed, sig, gpu_time_device_relaxed_store);
    device_fetch_add_relaxed<<<1,1,0, stream_device_relaxed_wait>>>(flag_device_relaxed, sig, gpu_time_device_relaxed_wait);
    
    cudaStreamSynchronize(stream_device_relaxed_store);
    cudaStreamSynchronize(stream_device_relaxed_wait);
    
    cudaDeviceSynchronize();
    
    if (allocator == CUDA_MALLOC) {
        int flag;
        clock_t gpu_clock_store;
        clock_t gpu_clock_wait;
        
        cudaMemcpy(&flag, flag_device_relaxed, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&gpu_clock_store, gpu_time_device_relaxed_store, sizeof(clock_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&gpu_clock_wait, gpu_time_device_relaxed_wait, sizeof(clock_t), cudaMemcpyDeviceToHost);
        
        std::cout << "Device-Fetch-Add Device-Fetch-Add (Device, Relaxed) | Value : " << flag << " | Store : " << ((double) gpu_clock_store / 10000) / ((double) get_gpu_freq()) * 1000000. << " | Wait : " << ((double) gpu_clock_wait / 10000) / ((double) get_gpu_freq()) * 1000000. << std::endl;
    } else {
        std::cout << "Device-Fetch-Add Device-Fetch-Add (Device, Relaxed) | Value : " << *flag_device_relaxed << " | Store : " << ((double) (*gpu_time_device_relaxed_store / 10000)) / ((double) get_gpu_freq()) * 1000000. << " | Wait : " << ((double) (*gpu_time_device_relaxed_wait / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;
    }
    
    cudaStreamDestroy(stream_device_relaxed_store);
    cudaStreamDestroy(stream_device_relaxed_wait);

    if (allocator == CUDA_MALLOC) {
        cudaMemset(sig, 0, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else {
        sig->store(PONG);
    }
    
    clock_t *gpu_time_thread_relaxed_store;
    clock_t *gpu_time_thread_relaxed_wait;
    
    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&gpu_time_thread_relaxed_store, sizeof(clock_t));
        cudaMallocHost(&gpu_time_thread_relaxed_wait, sizeof(clock_t));
    } else if (allocator == MALLOC) {
        gpu_time_thread_relaxed_store = (clock_t *) malloc(sizeof(clock_t));
        gpu_time_thread_relaxed_wait = (clock_t *) malloc(sizeof(clock_t));
    } else if (allocator == UM) {
        cudaMallocManaged(&gpu_time_thread_relaxed_store, sizeof(clock_t));
        cudaMallocManaged(&gpu_time_thread_relaxed_wait, sizeof(clock_t));
    } else if (allocator == CUDA_MALLOC) {
        cudaMalloc(&gpu_time_thread_relaxed_store, sizeof(clock_t));
        cudaMalloc(&gpu_time_thread_relaxed_wait, sizeof(clock_t));
    }

    cudaStream_t stream_thread_relaxed_store;
    cudaStream_t stream_thread_relaxed_wait;
    
    cudaStreamCreate(&stream_thread_relaxed_store);
    cudaStreamCreate(&stream_thread_relaxed_wait);
    
    device_fetch_add_relaxed<<<1,1,0, stream_thread_relaxed_store>>>(flag_thread_relaxed, sig, gpu_time_thread_relaxed_store);
    device_fetch_add_relaxed<<<1,1,0, stream_thread_relaxed_wait>>>(flag_thread_relaxed, sig, gpu_time_thread_relaxed_wait);
    
    cudaStreamSynchronize(stream_thread_relaxed_store);
    cudaStreamSynchronize(stream_thread_relaxed_wait);
    
    cudaDeviceSynchronize();
    
    if (allocator == CUDA_MALLOC) {
        int flag;
        clock_t gpu_clock_store;
        clock_t gpu_clock_wait;
        
        cudaMemcpy(&flag, flag_thread_relaxed, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&gpu_clock_store, gpu_time_thread_relaxed_store, sizeof(clock_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&gpu_clock_wait, gpu_time_thread_relaxed_wait, sizeof(clock_t), cudaMemcpyDeviceToHost);
        
        std::cout << "Device-Fetch-Add Device-Fetch-Add (Thread, Relaxed) | Value : " << flag << " | Store : " << ((double) gpu_clock_store / 10000) / ((double) get_gpu_freq()) * 1000000. << " | Wait : " << ((double) gpu_clock_wait / 10000) / ((double) get_gpu_freq()) * 1000000. << std::endl;
    } else {
        std::cout << "Device-Fetch-Add Device-Fetch-Add (Thread, Relaxed) | Value : " << *flag_thread_relaxed << " | Store : " << ((double) (*gpu_time_thread_relaxed_store / 10000)) / ((double) get_gpu_freq()) * 1000000. << " | Wait : " << ((double) (*gpu_time_thread_relaxed_wait / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;
    }
    
    cuda::atomic<uint16_t, cuda::thread_scope_thread> *flag_thread_acqrel;
    cuda::atomic<uint16_t, cuda::thread_scope_device> *flag_device_acqrel;
    cuda::atomic<uint16_t, cuda::thread_scope_system> *flag_system_acqrel;
    
    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&flag_thread_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocHost(&flag_device_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocHost(&flag_system_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == MALLOC) {
        flag_thread_acqrel = (cuda::atomic<uint16_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        flag_device_acqrel = (cuda::atomic<uint16_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        flag_system_acqrel = (cuda::atomic<uint16_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == UM) {
        cudaMallocManaged(&flag_system_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
        cudaMallocManaged(&flag_device_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocManaged(&flag_thread_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
    } else if (allocator == CUDA_MALLOC) {
        cudaMalloc(&flag_thread_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMalloc(&flag_device_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMalloc(&flag_system_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    }

    if (allocator == CUDA_MALLOC) {
        // clear flag and sig
        cudaMemset(flag_thread_acqrel, 0, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMemset(flag_device_acqrel, 0, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMemset(flag_system_acqrel, 0, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
        cudaMemset(sig, 0, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else {
        sig->store(PONG);
    }

    clock_t *gpu_time_system_acqrel_store;
    clock_t *gpu_time_system_acqrel_wait;
    
    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&gpu_time_system_acqrel_store, sizeof(clock_t));
        cudaMallocHost(&gpu_time_system_acqrel_wait, sizeof(clock_t));
    } else if (allocator == MALLOC) {
        gpu_time_system_acqrel_store = (clock_t *) malloc(sizeof(clock_t));
        gpu_time_system_acqrel_wait = (clock_t *) malloc(sizeof(clock_t));
    } else if (allocator == UM) {
        cudaMallocManaged(&gpu_time_system_acqrel_store, sizeof(clock_t));
        cudaMallocManaged(&gpu_time_system_acqrel_wait, sizeof(clock_t));
    } else if (allocator == CUDA_MALLOC) {
        cudaMalloc(&gpu_time_system_acqrel_store, sizeof(clock_t));
        cudaMalloc(&gpu_time_system_acqrel_wait, sizeof(clock_t));
    }

    cudaStream_t stream_system_acqrel_store;
    cudaStream_t stream_system_acqrel_wait;
    
    cudaStreamCreate(&stream_system_acqrel_store);
    cudaStreamCreate(&stream_system_acqrel_wait);
    
    device_fetch_add_acqrel<<<1,1,0, stream_system_acqrel_store>>>(flag_system_acqrel, sig, gpu_time_system_acqrel_store);
    device_fetch_add_acqrel<<<1,1,0, stream_system_acqrel_wait>>>(flag_system_acqrel, sig, gpu_time_system_acqrel_wait);
    
    cudaStreamSynchronize(stream_system_acqrel_store);
    cudaStreamSynchronize(stream_system_acqrel_wait);

    cudaDeviceSynchronize();
    
    if (allocator == CUDA_MALLOC) {
        int flag;
        clock_t gpu_clock_store;
        clock_t gpu_clock_wait;

        cudaMemcpy(&flag, flag_system_acqrel, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&gpu_clock_store, gpu_time_system_acqrel_store, sizeof(clock_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&gpu_clock_wait, gpu_time_system_acqrel_wait, sizeof(clock_t), cudaMemcpyDeviceToHost);

        std::cout << "Device-Fetch-Add Device-Fetch-Add (System, Acq-Rel) | Value : " << flag << " | Store : " << ((double) gpu_clock_store / 10000) / ((double) get_gpu_freq()) * 1000000. << " | Wait : " << ((double) gpu_clock_wait / 10000) / ((double) get_gpu_freq()) * 1000000. << std::endl;
    } else {
        std::cout << "Device-Fetch-Add Device-Fetch-Add (System, Acq-Rel) | Value : " << *flag_system_acqrel << " | Store : " << ((double) (*gpu_time_system_acqrel_store / 10000)) / ((double) get_gpu_freq()) * 1000000. << " | Wait : " << ((double) (*gpu_time_system_acqrel_wait / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;
    }

    cudaStreamDestroy(stream_system_acqrel_store);
    cudaStreamDestroy(stream_system_acqrel_wait);

    if (allocator == CUDA_MALLOC) {
        cudaMemset(sig, 0, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else {
        sig->store(PONG);
    }
    
    clock_t *gpu_time_device_acqrel_store;
    clock_t *gpu_time_device_acqrel_wait;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&gpu_time_device_acqrel_store, sizeof(clock_t));
        cudaMallocHost(&gpu_time_device_acqrel_wait, sizeof(clock_t));
    } else if (allocator == MALLOC) {
        gpu_time_device_acqrel_store = (clock_t *) malloc(sizeof(clock_t));
        gpu_time_device_acqrel_wait = (clock_t *) malloc(sizeof(clock_t));
    } else if (allocator == UM) {
        cudaMallocManaged(&gpu_time_device_acqrel_store, sizeof(clock_t));
        cudaMallocManaged(&gpu_time_device_acqrel_wait, sizeof(clock_t));
    } else if (allocator == CUDA_MALLOC) {
        cudaMalloc(&gpu_time_device_acqrel_store, sizeof(clock_t));
        cudaMalloc(&gpu_time_device_acqrel_wait, sizeof(clock_t));
    }

    cudaStream_t stream_device_acqrel_store;
    cudaStream_t stream_device_acqrel_wait;
    
    cudaStreamCreate(&stream_device_acqrel_store);
    cudaStreamCreate(&stream_device_acqrel_wait);
    
    device_fetch_add_acqrel<<<1,1,0, stream_device_acqrel_store>>>(flag_device_acqrel, sig, gpu_time_device_acqrel_store);
    device_fetch_add_acqrel<<<1,1,0, stream_device_acqrel_wait>>>(flag_device_acqrel, sig, gpu_time_device_acqrel_wait);
    
    cudaStreamSynchronize(stream_device_acqrel_store);
    cudaStreamSynchronize(stream_device_acqrel_wait);
    
    cudaDeviceSynchronize();
    
    if (allocator == CUDA_MALLOC) {
        int flag;
        clock_t gpu_clock_store;
        clock_t gpu_clock_wait;
        
        cudaMemcpy(&flag, flag_device_acqrel, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&gpu_clock_store, gpu_time_device_acqrel_store, sizeof(clock_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&gpu_clock_wait, gpu_time_device_acqrel_wait, sizeof(clock_t), cudaMemcpyDeviceToHost);

        std::cout << "Device-Fetch-Add Device-Fetch-Add (Device, Acq-Rel) | Value : " << flag << " | Store : " << ((double) gpu_clock_store / 10000) / ((double) get_gpu_freq()) * 1000000. << " | Wait : " << ((double) gpu_clock_wait / 10000) / ((double) get_gpu_freq()) * 1000000. << std::endl;
    } else {
        std::cout << "Device-Fetch-Add Device-Fetch-Add (Device, Acq-Rel) | Value : " << *flag_device_acqrel << " | Store : " << ((double) (*gpu_time_device_acqrel_store / 10000)) / ((double) get_gpu_freq()) * 1000000. << " | Wait : " << ((double) (*gpu_time_device_acqrel_wait / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;
    }
    
    cudaStreamDestroy(stream_device_acqrel_store);
    cudaStreamDestroy(stream_device_acqrel_wait);

    if (allocator == CUDA_MALLOC) {
        cudaMemset(sig, 0, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else {
        sig->store(PONG);
    }
    
    clock_t *gpu_time_thread_acqrel_store;
    clock_t *gpu_time_thread_acqrel_wait;
    
    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&gpu_time_thread_acqrel_store, sizeof(clock_t));
        cudaMallocHost(&gpu_time_thread_acqrel_wait, sizeof(clock_t));
    } else if (allocator == MALLOC) {
        gpu_time_thread_acqrel_store = (clock_t *) malloc(sizeof(clock_t));
        gpu_time_thread_acqrel_wait = (clock_t *) malloc(sizeof(clock_t));
    } else if (allocator == UM) {
        cudaMallocManaged(&gpu_time_thread_acqrel_store, sizeof(clock_t));
        cudaMallocManaged(&gpu_time_thread_acqrel_wait, sizeof(clock_t));
    } else if (allocator == CUDA_MALLOC) {
        cudaMalloc(&gpu_time_thread_acqrel_store, sizeof(clock_t));
        cudaMalloc(&gpu_time_thread_acqrel_wait, sizeof(clock_t));
    }

    cudaStream_t stream_thread_acqrel_store;
    cudaStream_t stream_thread_acqrel_wait;
    
    
    cudaStreamCreate(&stream_thread_acqrel_store);
    cudaStreamCreate(&stream_thread_acqrel_wait);

    device_fetch_add_acqrel<<<1,1,0, stream_thread_acqrel_store>>>(flag_thread_acqrel, sig, gpu_time_thread_acqrel_store);
    device_fetch_add_acqrel<<<1,1,0, stream_thread_acqrel_wait>>>(flag_thread_acqrel, sig, gpu_time_thread_acqrel_wait);

    cudaStreamSynchronize(stream_thread_acqrel_store);
    cudaStreamSynchronize(stream_thread_acqrel_wait);

    cudaDeviceSynchronize();
    
    if (allocator == CUDA_MALLOC) {
        int flag;
        clock_t gpu_clock_store;
        clock_t gpu_clock_wait;
        
        cudaMemcpy(&flag, flag_thread_acqrel, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&gpu_clock_store, gpu_time_thread_acqrel_store, sizeof(clock_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&gpu_clock_wait, gpu_time_thread_acqrel_wait, sizeof(clock_t), cudaMemcpyDeviceToHost);
        
        std::cout << "Device-Fetch-Add Device-Fetch-Add (Thread, Acq-Rel) | Value : " << flag << " | Store : " << ((double) gpu_clock_store / 10000) / ((double) get_gpu_freq()) * 1000000. << " | Wait : " << ((double) gpu_clock_wait / 10000) / ((double) get_gpu_freq()) * 1000000. << std::endl;
    } else {
        std::cout << "Device-Fetch-Add Device-Fetch-Add (Thread, Acq-Rel) | Value : " << *flag_thread_acqrel << " | Store : " << ((double) (*gpu_time_thread_acqrel_store / 10000)) / ((double) get_gpu_freq()) * 1000000. << " | Wait : " << ((double) (*gpu_time_thread_acqrel_wait / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;
    }
    
    cuda::atomic<uint16_t, cuda::thread_scope_thread> *flag_thread_seqcst;
    cuda::atomic<uint16_t, cuda::thread_scope_device> *flag_device_seqcst;
    cuda::atomic<uint16_t, cuda::thread_scope_system> *flag_system_seqcst;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&flag_thread_seqcst, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocHost(&flag_device_seqcst, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocHost(&flag_system_seqcst, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == MALLOC) {
        flag_thread_seqcst = (cuda::atomic<uint16_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        flag_device_seqcst = (cuda::atomic<uint16_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        flag_system_seqcst = (cuda::atomic<uint16_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == UM) {
        cudaMallocManaged(&flag_thread_seqcst, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocManaged(&flag_device_seqcst, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocManaged(&flag_system_seqcst, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == CUDA_MALLOC) {
        cudaMalloc(&flag_thread_seqcst, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMalloc(&flag_device_seqcst, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMalloc(&flag_system_seqcst, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    }

    if (allocator == CUDA_MALLOC) {
        // clear flag and sig
        cudaMemset(flag_thread_seqcst, 0, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMemset(flag_device_seqcst, 0, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMemset(flag_system_seqcst, 0, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
        cudaMemset(sig, 0, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else {
        sig->store(PONG);
    }
    
    clock_t *gpu_time_system_seqcst_store;
    clock_t *gpu_time_system_seqcst_wait;
    
    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&gpu_time_system_seqcst_store, sizeof(clock_t));
        cudaMallocHost(&gpu_time_system_seqcst_wait, sizeof(clock_t));
    } else if (allocator == MALLOC) {
        gpu_time_system_seqcst_store = (clock_t *) malloc(sizeof(clock_t));
        gpu_time_system_seqcst_wait = (clock_t *) malloc(sizeof(clock_t));
    } else if (allocator == UM) {
        cudaMallocManaged(&gpu_time_system_seqcst_store, sizeof(clock_t));
        cudaMallocManaged(&gpu_time_system_seqcst_wait, sizeof(clock_t));
    } else if (allocator == CUDA_MALLOC) {
        cudaMalloc(&gpu_time_system_seqcst_store, sizeof(clock_t));
        cudaMalloc(&gpu_time_system_seqcst_wait, sizeof(clock_t));
    }
    
    cudaStream_t stream_system_seqcst_store;
    cudaStream_t stream_system_seqcst_wait;
    
    cudaStreamCreate(&stream_system_seqcst_store);
    cudaStreamCreate(&stream_system_seqcst_wait);

    device_fetch_add_seqcst<<<1,1,0, stream_system_seqcst_store>>>(flag_system_seqcst, sig, gpu_time_system_seqcst_store);
    device_fetch_add_seqcst<<<1,1,0, stream_system_seqcst_wait>>>(flag_system_seqcst, sig, gpu_time_system_seqcst_wait);
    
    cudaStreamSynchronize(stream_system_seqcst_store);
    cudaStreamSynchronize(stream_system_seqcst_wait);
    
    cudaDeviceSynchronize();

    if (allocator == CUDA_MALLOC) {
        int flag;
        clock_t gpu_clock_store;
        clock_t gpu_clock_wait;
        
        cudaMemcpy(&flag, flag_system_seqcst, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&gpu_clock_store, gpu_time_system_seqcst_store, sizeof(clock_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&gpu_clock_wait, gpu_time_system_seqcst_wait, sizeof(clock_t), cudaMemcpyDeviceToHost);
        
        std::cout << "Device-Fetch-Add Device-Fetch-Add (System, Seq-Cst) | Value : " << flag << " | Store : " << ((double) gpu_clock_store / 10000) / ((double) get_gpu_freq()) * 1000000. << " | Wait : " << ((double) gpu_clock_wait / 10000) / ((double) get_gpu_freq()) * 1000000. << std::endl;
    } else {
        std::cout << "Device-Fetch-Add Device-Fetch-Add (System, Seq-Cst) | Value : " << *flag_system_seqcst << " | Store : " << ((double) (*gpu_time_system_seqcst_store / 10000)) / ((double) get_gpu_freq()) * 1000000. << " | Wait : " << ((double) (*gpu_time_system_seqcst_wait / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;
    }
    
    cudaStreamDestroy(stream_system_seqcst_store);
    cudaStreamDestroy(stream_system_seqcst_wait);

    if (allocator == CUDA_MALLOC) {
        cudaMemset(sig, 0, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else {
        sig->store(PONG);
    }
    
    clock_t *gpu_time_device_seqcst_store;
    clock_t *gpu_time_device_seqcst_wait;
    
    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&gpu_time_device_seqcst_store, sizeof(clock_t));
        cudaMallocHost(&gpu_time_device_seqcst_wait, sizeof(clock_t));
    } else if (allocator == MALLOC) {
        gpu_time_device_seqcst_store = (clock_t *) malloc(sizeof(clock_t));
        gpu_time_device_seqcst_wait = (clock_t *) malloc(sizeof(clock_t));
    } else if (allocator == UM) {
        cudaMallocManaged(&gpu_time_device_seqcst_store, sizeof(clock_t));
        cudaMallocManaged(&gpu_time_device_seqcst_wait, sizeof(clock_t));
    } else if (allocator == CUDA_MALLOC) {
        cudaMalloc(&gpu_time_device_seqcst_store, sizeof(clock_t));
        cudaMalloc(&gpu_time_device_seqcst_wait, sizeof(clock_t));
    }
    
    cudaStream_t stream_device_seqcst_store;
    cudaStream_t stream_device_seqcst_wait;
    
    cudaStreamCreate(&stream_device_seqcst_store);
    cudaStreamCreate(&stream_device_seqcst_wait);
    
    device_fetch_add_seqcst<<<1,1,0, stream_device_seqcst_store>>>(flag_device_seqcst, sig, gpu_time_device_seqcst_store);
    device_fetch_add_seqcst<<<1,1,0, stream_device_seqcst_wait>>>(flag_device_seqcst, sig, gpu_time_device_seqcst_wait);
    
    cudaStreamSynchronize(stream_device_seqcst_store);
    cudaStreamSynchronize(stream_device_seqcst_wait);
    
    cudaDeviceSynchronize();
    
    if (allocator == CUDA_MALLOC) {
        int flag;
        clock_t gpu_clock_store;
        clock_t gpu_clock_wait;
        
        cudaMemcpy(&flag, flag_device_seqcst, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&gpu_clock_store, gpu_time_device_seqcst_store, sizeof(clock_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&gpu_clock_wait, gpu_time_device_seqcst_wait, sizeof(clock_t), cudaMemcpyDeviceToHost);
        
        std::cout << "Device-Fetch-Add Device-Fetch-Add (Device, Seq-Cst) | Value : " << flag << " | Store : " << ((double) gpu_clock_store / 10000) / ((double) get_gpu_freq()) * 1000000. << " | Wait : " << ((double) gpu_clock_wait / 10000) / ((double) get_gpu_freq()) * 1000000. << std::endl;
    } else {
        std::cout << "Device-Fetch-Add Device-Fetch-Add (Device, Seq-Cst) | Value : " << *flag_device_seqcst << " | Store : " << ((double) (*gpu_time_device_seqcst_store / 10000)) / ((double) get_gpu_freq()) * 1000000. << " | Wait : " << ((double) (*gpu_time_device_seqcst_wait / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;
    }

    cudaStreamDestroy(stream_device_seqcst_store);
    cudaStreamDestroy(stream_device_seqcst_wait);

    if (allocator == CUDA_MALLOC) {
        cudaMemset(sig, 0, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else {
        sig->store(PONG);
    }
    
    clock_t *gpu_time_thread_seqcst_store;
    clock_t *gpu_time_thread_seqcst_wait;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&gpu_time_thread_seqcst_store, sizeof(clock_t));
        cudaMallocHost(&gpu_time_thread_seqcst_wait, sizeof(clock_t));
    } else if (allocator == MALLOC) {
        gpu_time_thread_seqcst_store = (clock_t *) malloc(sizeof(clock_t));
        gpu_time_thread_seqcst_wait = (clock_t *) malloc(sizeof(clock_t));
    } else if (allocator == UM) {
        cudaMallocManaged(&gpu_time_thread_seqcst_store, sizeof(clock_t));
        cudaMallocManaged(&gpu_time_thread_seqcst_wait, sizeof(clock_t));
    } else if (allocator == CUDA_MALLOC) {
        cudaMalloc(&gpu_time_thread_seqcst_store, sizeof(clock_t));
        cudaMalloc(&gpu_time_thread_seqcst_wait, sizeof(clock_t));
    }

    cudaStream_t stream_thread_seqcst_store;
    cudaStream_t stream_thread_seqcst_wait;

    cudaStreamCreate(&stream_thread_seqcst_store);
    cudaStreamCreate(&stream_thread_seqcst_wait);

    device_fetch_add_seqcst<<<1,1,0, stream_thread_seqcst_store>>>(flag_thread_seqcst, sig, gpu_time_thread_seqcst_store);
    device_fetch_add_seqcst<<<1,1,0, stream_thread_seqcst_wait>>>(flag_thread_seqcst, sig, gpu_time_thread_seqcst_wait);

    cudaStreamSynchronize(stream_thread_seqcst_store);
    cudaStreamSynchronize(stream_thread_seqcst_wait);

    cudaDeviceSynchronize();

    if (allocator == CUDA_MALLOC) {
        int flag;
        clock_t gpu_clock_store;
        clock_t gpu_clock_wait;

        cudaMemcpy(&flag, flag_thread_seqcst, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&gpu_clock_store, gpu_time_thread_seqcst_store, sizeof(clock_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&gpu_clock_wait, gpu_time_thread_seqcst_wait, sizeof(clock_t), cudaMemcpyDeviceToHost);

        std::cout << "Device-Fetch-Add Device-Fetch-Add (Thread, Seq-Cst) | Value : " << flag << " | Store : " << ((double) gpu_clock_store / 10000) / ((double) get_gpu_freq()) * 1000000. << " | Wait : " << ((double) gpu_clock_wait / 10000) / ((double) get_gpu_freq()) * 1000000. << std::endl;
    } else {
        std::cout << "Device-Fetch-Add Device-Fetch-Add (Thread, Seq-Cst) | Value : " << *flag_thread_seqcst << " | Store : " << ((double) (*gpu_time_thread_seqcst_store / 10000)) / ((double) get_gpu_freq()) * 1000000. << " | Wait : " << ((double) (*gpu_time_thread_seqcst_wait / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;
    }

    cudaStreamDestroy(stream_thread_seqcst_store);
    cudaStreamDestroy(stream_thread_seqcst_wait);

    if (allocator == CUDA_MALLOC_HOST) {
        cudaFreeHost(flag_thread_relaxed);
        cudaFreeHost(flag_device_relaxed);
        cudaFreeHost(flag_system_relaxed);
        cudaFreeHost(flag_system_acqrel);
        cudaFreeHost(flag_device_acqrel);
        cudaFreeHost(flag_thread_acqrel);
        cudaFreeHost(flag_system_seqcst);
        cudaFreeHost(flag_device_seqcst);
        cudaFreeHost(flag_thread_seqcst);
        cudaFreeHost(sig);
        cudaFreeHost(gpu_time_system_relaxed_store);
        cudaFreeHost(gpu_time_system_relaxed_wait);
        cudaFreeHost(gpu_time_device_relaxed_store);
        cudaFreeHost(gpu_time_device_relaxed_wait);
        cudaFreeHost(gpu_time_thread_relaxed_store);
        cudaFreeHost(gpu_time_thread_relaxed_wait);
        cudaFreeHost(gpu_time_device_relaxed_store);
        cudaFreeHost(gpu_time_device_relaxed_wait);
        cudaFreeHost(gpu_time_system_acqrel_store);
        cudaFreeHost(gpu_time_system_acqrel_wait);
        cudaFreeHost(gpu_time_device_acqrel_store);
        cudaFreeHost(gpu_time_device_acqrel_wait);
        cudaFreeHost(gpu_time_thread_acqrel_store);
        cudaFreeHost(gpu_time_thread_acqrel_wait);
        cudaFreeHost(gpu_time_system_seqcst_store);
        cudaFreeHost(gpu_time_system_seqcst_wait);
        cudaFreeHost(gpu_time_device_seqcst_store);
        cudaFreeHost(gpu_time_device_seqcst_wait);
        cudaFreeHost(gpu_time_thread_seqcst_store);
        cudaFreeHost(gpu_time_thread_seqcst_wait);
    } else if (allocator == MALLOC) {
        free(flag_thread_relaxed);
        free(flag_device_relaxed);
        free(flag_system_relaxed);
        free(flag_system_acqrel);
        free(flag_device_acqrel);
        free(flag_thread_acqrel);
        free(flag_system_seqcst);
        free(flag_device_seqcst);
        free(flag_thread_seqcst);
        free(sig);
        free(gpu_time_system_relaxed_store);
        free(gpu_time_system_relaxed_wait);
        free(gpu_time_device_relaxed_store);
        free(gpu_time_device_relaxed_wait);        
        free(gpu_time_thread_relaxed_store);
        free(gpu_time_thread_relaxed_wait);
        free(gpu_time_device_relaxed_store);
        free(gpu_time_device_relaxed_wait);
        free(gpu_time_system_acqrel_store);
        free(gpu_time_system_acqrel_wait);
        free(gpu_time_device_acqrel_store);
        free(gpu_time_device_acqrel_wait);
        free(gpu_time_thread_acqrel_store);
        free(gpu_time_thread_acqrel_wait);
        free(gpu_time_system_seqcst_store);
        free(gpu_time_system_seqcst_wait);
        free(gpu_time_device_seqcst_store);
        free(gpu_time_device_seqcst_wait);
        free(gpu_time_thread_seqcst_store);
        free(gpu_time_thread_seqcst_wait);
    } else if (allocator == UM || allocator == CUDA_MALLOC) {
        cudaFree(flag_thread_relaxed);
        cudaFree(flag_device_relaxed);
        cudaFree(flag_system_relaxed);
        cudaFree(flag_system_acqrel);
        cudaFree(flag_device_acqrel);
        cudaFree(flag_thread_acqrel);
        cudaFree(flag_system_seqcst);
        cudaFree(flag_device_seqcst);
        cudaFree(flag_thread_seqcst);
        cudaFree(sig);
        cudaFree(gpu_time_system_relaxed_store);
        cudaFree(gpu_time_system_relaxed_wait);
        cudaFree(gpu_time_thread_relaxed_store);
        cudaFree(gpu_time_thread_relaxed_wait);
        cudaFree(gpu_time_device_relaxed_store);
        cudaFree(gpu_time_device_relaxed_wait);
        cudaFree(gpu_time_system_acqrel_store);
        cudaFree(gpu_time_system_acqrel_wait);
        cudaFree(gpu_time_device_acqrel_store);
        cudaFree(gpu_time_device_acqrel_wait);
        cudaFree(gpu_time_thread_acqrel_store);
        cudaFree(gpu_time_thread_acqrel_wait);
        cudaFree(gpu_time_system_seqcst_store);
        cudaFree(gpu_time_system_seqcst_wait);
        cudaFree(gpu_time_device_seqcst_store);
        cudaFree(gpu_time_device_seqcst_wait);
        cudaFree(gpu_time_thread_seqcst_store);
        cudaFree(gpu_time_thread_seqcst_wait);
    }

    // cuda::atomic<uint16_t, cuda::thread_scope_thread> *flag_thread_relaxed_store_wait;
    // cuda::atomic<uint16_t, cuda::thread_scope_device> *flag_device_relaxed_store_wait;
    // cuda::atomic<uint16_t, cuda::thread_scope_system> *flag_system_relaxed_store_wait;

    // if (allocator)
}

void host_device_fetch_add(Allocator allocator) {
    cuda::atomic<uint16_t, cuda::thread_scope_thread> *flag_thread_relaxed;
    cuda::atomic<uint16_t, cuda::thread_scope_device> *flag_device_relaxed;
    cuda::atomic<uint16_t, cuda::thread_scope_system> *flag_system_relaxed;

    cuda::atomic<uint16_t, cuda::thread_scope_system> *sig;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&sig, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == MALLOC) {
        sig = (cuda::atomic<uint16_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == UM) {
        cudaMallocManaged(&sig, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    }

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&flag_thread_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocHost(&flag_device_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocHost(&flag_system_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == MALLOC) {
        flag_thread_relaxed = (cuda::atomic<uint16_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        flag_device_relaxed = (cuda::atomic<uint16_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        flag_system_relaxed = (cuda::atomic<uint16_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == UM) {
        cudaMallocManaged(&flag_thread_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocManaged(&flag_device_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocManaged(&flag_system_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    }
    
    cpu_set_t cpuset;
    
    sig->store(PONG);
    uint64_t cpu_time_system_relaxed;
    std::thread t_system_relaxed(host_fetch_add_relaxed, (std::atomic<uint16_t> *) flag_system_relaxed, (std::atomic<uint16_t> *) sig, &cpu_time_system_relaxed);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_system_relaxed.native_handle(), sizeof(cpu_set_t), &cpuset);
    
    // clock_t gpu_time_system_relaxed;
    clock_t * gpu_time_system_relaxed;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&gpu_time_system_relaxed, sizeof(clock_t));
    } else if (allocator == MALLOC) {
        gpu_time_system_relaxed = (clock_t *) malloc(sizeof(clock_t));
    } else if (allocator == UM) {
        cudaMallocManaged(&gpu_time_system_relaxed, sizeof(clock_t));
    }

    device_fetch_add_relaxed<<<1,1>>>(flag_system_relaxed, sig, gpu_time_system_relaxed);
    
    cudaDeviceSynchronize();
    t_system_relaxed.join();

    std::cout << "Host-Fetch-Add Device-Fetch-Add (System, Relaxed) | Value : " << *flag_system_relaxed << " | Host : " << ((double) (cpu_time_system_relaxed / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << " | Device : " << ((double) (*gpu_time_system_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;

    sig->store(PONG);
    uint64_t cpu_time_device_relaxed;
    std::thread t_device_relaxed(host_fetch_add_relaxed, (std::atomic<uint16_t> *) flag_device_relaxed, (std::atomic<uint16_t> *) sig, &cpu_time_device_relaxed);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_device_relaxed.native_handle(), sizeof(cpu_set_t), &cpuset);
    
    // clock_t gpu_time_device_relaxed;

    clock_t * gpu_time_device_relaxed;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&gpu_time_device_relaxed, sizeof(clock_t));
    } else if (allocator == MALLOC) {
        gpu_time_device_relaxed = (clock_t *) malloc(sizeof(clock_t));
    } else if (allocator == UM) {
        cudaMallocManaged(&gpu_time_device_relaxed, sizeof(clock_t));
    }

    device_fetch_add_relaxed<<<1,1>>>(flag_device_relaxed, sig, gpu_time_device_relaxed);
    
    cudaDeviceSynchronize();
    t_device_relaxed.join();
    
    std::cout << "Host-Fetch-Add Device-Fetch-Add (Device, Relaxed) | Value : " << *flag_device_relaxed << " | Host : " << ((double) (cpu_time_device_relaxed / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << " | Device : " << ((double) (*gpu_time_device_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;
    
    sig->store(PONG);
    uint64_t cpu_time_thread_relaxed;
    std::thread t_thread_relaxed(host_fetch_add_relaxed, (std::atomic<uint16_t> *) flag_thread_relaxed, (std::atomic<uint16_t> *) sig, &cpu_time_thread_relaxed);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_thread_relaxed.native_handle(), sizeof(cpu_set_t), &cpuset);
    
    // clock_t gpu_time_thread_relaxed;

    clock_t * gpu_time_thread_relaxed;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&gpu_time_thread_relaxed, sizeof(clock_t));
    } else if (allocator == MALLOC) {
        gpu_time_thread_relaxed = (clock_t *) malloc(sizeof(clock_t));
    } else if (allocator == UM) {
        cudaMallocManaged(&gpu_time_thread_relaxed, sizeof(clock_t));
    }

    device_fetch_add_relaxed<<<1,1>>>(flag_thread_relaxed, sig, gpu_time_thread_relaxed);
    
    cudaDeviceSynchronize();
    t_thread_relaxed.join();
    
    std::cout << "Host-Fetch-Add Device-Fetch-Add (Thread, Relaxed) | Value : " << *flag_thread_relaxed << " | Host : " << ((double) (cpu_time_thread_relaxed / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << " | Device : " << ((double) (*gpu_time_thread_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;
    
    cuda::atomic<uint16_t, cuda::thread_scope_thread> *flag_thread_acqrel;
    cuda::atomic<uint16_t, cuda::thread_scope_device> *flag_device_acqrel;
    cuda::atomic<uint16_t, cuda::thread_scope_system> *flag_system_acqrel;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&flag_thread_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocHost(&flag_device_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocHost(&flag_system_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == MALLOC) {
        flag_thread_acqrel = (cuda::atomic<uint16_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        flag_device_acqrel = (cuda::atomic<uint16_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        flag_system_acqrel = (cuda::atomic<uint16_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == UM) {
        cudaMallocManaged(&flag_system_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
        cudaMallocManaged(&flag_device_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocManaged(&flag_thread_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
    }
    
    sig->store(PONG);
    uint64_t cpu_time_system_acqrel;
    std::thread t_system_acqrel(host_fetch_add_acqrel, (std::atomic<uint16_t> *) flag_system_acqrel, (std::atomic<uint16_t> *) sig, &cpu_time_system_acqrel);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_system_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);
    
    // clock_t gpu_time_system_acqrel;

    clock_t * gpu_time_system_acqrel;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&gpu_time_system_acqrel, sizeof(clock_t));
    } else if (allocator == MALLOC) {
        gpu_time_system_acqrel = (clock_t *) malloc(sizeof(clock_t));
    } else if (allocator == UM) {
        cudaMallocManaged(&gpu_time_system_acqrel, sizeof(clock_t));
    }

    device_fetch_add_acqrel<<<1,1>>>(flag_system_acqrel, sig, gpu_time_system_acqrel);
    
    t_system_acqrel.join();
    cudaDeviceSynchronize();
    
    std::cout << "Host-Fetch-Add Device-Fetch-Add (System, Acq-Rel) | Value : " << *flag_system_acqrel << " | Host : " << ((double) (cpu_time_system_acqrel / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << " | Device : " << ((double) (*gpu_time_system_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;
    
    sig->store(PONG);
    uint64_t cpu_time_device_acqrel;
    std::thread t_device_acqrel(host_fetch_add_acqrel, (std::atomic<uint16_t> *) flag_device_acqrel, (std::atomic<uint16_t> *) sig, &cpu_time_device_acqrel);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_device_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);
    
    // clock_t gpu_time_device_acqrel;

    clock_t * gpu_time_device_acqrel;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&gpu_time_device_acqrel, sizeof(clock_t));
    } else if (allocator == MALLOC) {
        gpu_time_device_acqrel = (clock_t *) malloc(sizeof(clock_t));
    } else if (allocator == UM) {
        cudaMallocManaged(&gpu_time_device_acqrel, sizeof(clock_t));
    }

    device_fetch_add_acqrel<<<1,1>>>(flag_device_acqrel, sig, gpu_time_device_acqrel);
    
    t_device_acqrel.join();
    cudaDeviceSynchronize();
    
    std::cout << "Host-Fetch-Add Device-Fetch-Add (Device, Acq-Rel) | Value : " << *flag_device_acqrel << " | Host : " << ((double) (cpu_time_device_acqrel / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << " | Device : " << ((double) (*gpu_time_device_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;
    
    sig->store(PONG);
    uint64_t cpu_time_thread_acqrel;
    std::thread t_thread_acqrel(host_fetch_add_acqrel, (std::atomic<uint16_t> *) flag_thread_acqrel, (std::atomic<uint16_t> *) sig, &cpu_time_thread_acqrel);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_thread_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);
    
    // clock_t gpu_time_thread_acqrel;

    clock_t * gpu_time_thread_acqrel;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&gpu_time_thread_acqrel, sizeof(clock_t));
    } else if (allocator == MALLOC) {
        gpu_time_thread_acqrel = (clock_t *) malloc(sizeof(clock_t));
    } else if (allocator == UM) {
        cudaMallocManaged(&gpu_time_thread_acqrel, sizeof(clock_t));
    }

    device_fetch_add_acqrel<<<1,1>>>(flag_thread_acqrel, sig, gpu_time_thread_acqrel);
    
    t_thread_acqrel.join();
    cudaDeviceSynchronize();
    
    std::cout << "Host-Fetch-Add Device-Fetch-Add (Thread, Acq-Rel) | Value : " << *flag_thread_acqrel << " | Host : " << ((double) (cpu_time_thread_acqrel / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << " | Device : " << ((double) (*gpu_time_thread_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;
    
    cuda::atomic<uint16_t, cuda::thread_scope_thread> *flag_thread_seqcst;
    cuda::atomic<uint16_t, cuda::thread_scope_device> *flag_device_seqcst;
    cuda::atomic<uint16_t, cuda::thread_scope_system> *flag_system_seqcst;
    
    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&flag_thread_seqcst, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocHost(&flag_device_seqcst, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocHost(&flag_system_seqcst, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == MALLOC) {
        flag_thread_seqcst = (cuda::atomic<uint16_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        flag_device_seqcst = (cuda::atomic<uint16_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        flag_system_seqcst = (cuda::atomic<uint16_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == UM) {
        cudaMallocManaged(&flag_thread_seqcst, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocManaged(&flag_device_seqcst, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocManaged(&flag_system_seqcst, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    }
    
    sig->store(PONG);
    uint64_t cpu_time_system_seqcst;
    std::thread t_system_seqcst(host_fetch_add_seqcst, (std::atomic<uint16_t> *) flag_system_seqcst, (std::atomic<uint16_t> *) sig, &cpu_time_system_seqcst);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_system_seqcst.native_handle(), sizeof(cpu_set_t), &cpuset);
    
    // clock_t gpu_time_system_seqcst;

    clock_t * gpu_time_system_seqcst;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&gpu_time_system_seqcst, sizeof(clock_t));
    } else if (allocator == MALLOC) {
        gpu_time_system_seqcst = (clock_t *) malloc(sizeof(clock_t));
    } else if (allocator == UM) {
        cudaMallocManaged(&gpu_time_system_seqcst, sizeof(clock_t));
    }

    device_fetch_add_seqcst<<<1,1>>>(flag_system_seqcst, sig, gpu_time_system_seqcst);
    
    t_system_seqcst.join();
    cudaDeviceSynchronize();
    
    std::cout << "Host-Fetch-Add Device-Fetch-Add (System, Seq-Cst) | Value : " << *flag_system_seqcst << " | Host : " << ((double) (cpu_time_system_seqcst / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << " | Device : " << ((double) (*gpu_time_system_seqcst / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;
    
    sig->store(PONG);
    uint64_t cpu_time_device_seqcst;
    std::thread t_device_seqcst(host_fetch_add_seqcst, (std::atomic<uint16_t> *) flag_device_seqcst, (std::atomic<uint16_t> *) sig, &cpu_time_device_seqcst);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_device_seqcst.native_handle(), sizeof(cpu_set_t), &cpuset);
    
    // clock_t gpu_time_device_seqcst;

    clock_t * gpu_time_device_seqcst;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&gpu_time_device_seqcst, sizeof(clock_t));
    } else if (allocator == MALLOC) {
        gpu_time_device_seqcst = (clock_t *) malloc(sizeof(clock_t));
    } else if (allocator == UM) {
        cudaMallocManaged(&gpu_time_device_seqcst, sizeof(clock_t));
    }

    device_fetch_add_seqcst<<<1,1>>>(flag_device_seqcst, sig, gpu_time_device_seqcst);
    
    t_device_seqcst.join();
    cudaDeviceSynchronize();
    
    std::cout << "Host-Fetch-Add Device-Fetch-Add (Device, Seq-Cst) | Value : " << *flag_device_seqcst << " | Host : " << ((double) (cpu_time_device_seqcst / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << " | Device : " << ((double) (*gpu_time_device_seqcst / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;
    
    sig->store(PONG);
    uint64_t cpu_time_thread_seqcst;
    std::thread t_thread_seqcst(host_fetch_add_seqcst, (std::atomic<uint16_t> *) flag_thread_seqcst, (std::atomic<uint16_t> *) sig, &cpu_time_thread_seqcst);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_thread_seqcst.native_handle(), sizeof(cpu_set_t), &cpuset);

    // clock_t gpu_time_thread_seqcst;

    clock_t * gpu_time_thread_seqcst;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&gpu_time_thread_seqcst, sizeof(clock_t));
    } else if (allocator == MALLOC) {
        gpu_time_thread_seqcst = (clock_t *) malloc(sizeof(clock_t));
    } else if (allocator == UM) {
        cudaMallocManaged(&gpu_time_thread_seqcst, sizeof(clock_t));
    }

    device_fetch_add_seqcst<<<1,1>>>(flag_thread_seqcst, sig, gpu_time_thread_seqcst);

    t_thread_seqcst.join();
    cudaDeviceSynchronize();

    std::cout << "Host-Fetch-Add Device-Fetch-Add (Thread, Seq-Cst) | Value : " << *flag_thread_seqcst << " | Host : " << ((double) (cpu_time_thread_seqcst / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << " | Device : " << ((double) (*gpu_time_thread_seqcst / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;

    if (allocator == MALLOC) {
        free(flag_thread_relaxed);
        free(flag_device_relaxed);
        free(flag_system_relaxed);

        free(flag_thread_acqrel);
        free(flag_device_acqrel);
        free(flag_system_acqrel);

        free(flag_thread_seqcst);
        free(flag_device_seqcst);
        free(flag_system_seqcst);
    } else if (allocator == CUDA_MALLOC_HOST) {
        cudaFreeHost(flag_thread_relaxed);
        cudaFreeHost(flag_device_relaxed);
        cudaFreeHost(flag_system_relaxed);

        cudaFreeHost(flag_thread_acqrel);
        cudaFreeHost(flag_device_acqrel);
        cudaFreeHost(flag_system_acqrel);

        cudaFreeHost(flag_thread_seqcst);
        cudaFreeHost(flag_device_seqcst);
        cudaFreeHost(flag_system_seqcst);
    } else if (allocator == UM) {
        cudaFree(flag_thread_relaxed);
        cudaFree(flag_device_relaxed);
        cudaFree(flag_system_relaxed);

        cudaFree(flag_thread_acqrel);
        cudaFree(flag_device_acqrel);
        cudaFree(flag_system_acqrel);

        cudaFree(flag_thread_seqcst);
        cudaFree(flag_device_seqcst);
        cudaFree(flag_system_seqcst);
    }

    if (allocator == MALLOC) {
        free(sig);
    } else if (allocator == CUDA_MALLOC_HOST) {
        cudaFreeHost(sig);
    } else if (allocator == UM) {
        cudaFree(sig);
    }
}

void host_ping_device_pong_assymetric(Allocator allocator) {

    cuda::atomic<uint16_t, cuda::thread_scope_thread> *flag_thread_relaxed;
    cuda::atomic<uint16_t, cuda::thread_scope_device> *flag_device_relaxed;
    cuda::atomic<uint16_t, cuda::thread_scope_system> *flag_system_relaxed;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&flag_thread_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocHost(&flag_device_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocHost(&flag_system_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == MALLOC) {
        flag_thread_relaxed = (cuda::atomic<uint16_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        flag_device_relaxed = (cuda::atomic<uint16_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        flag_system_relaxed = (cuda::atomic<uint16_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == UM) {
        cudaMallocManaged(&flag_thread_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocManaged(&flag_device_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocManaged(&flag_system_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    }

    cpu_set_t cpuset;

    uint64_t cpu_time_system;
    std::thread t_system(host_ping_function_relaxed_base, (std::atomic<uint16_t> *) flag_system_relaxed, &cpu_time_system);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_system.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_relaxed_decoupled<<<1,1>>>(flag_system_relaxed);
    t_system.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (System, Relaxed, CPU-CAS GPU-Decoupled) | Host : " << ((double) (cpu_time_system / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    uint64_t cpu_time_device;
    std::thread t_device(host_ping_function_relaxed_base, (std::atomic<uint16_t> *) flag_device_relaxed, &cpu_time_device);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_device.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_relaxed_decoupled<<<1,1>>>(flag_device_relaxed);
    t_device.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (Device, Relaxed, CPU-CAS GPU-Decoupled) | Host : " << ((double) (cpu_time_device / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    // uint64_t cpu_time_thread;
    // std::thread t_thread(host_ping_function_relaxed_base, (std::atomic<uint16_t> *) flag_thread_relaxed, &cpu_time_thread);
    // CPU_ZERO(&cpuset);
    // CPU_SET(0, &cpuset);
    // pthread_setaffinity_np(t_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    // device_pong_kernel_relaxed_decoupled<<<1,1>>>(flag_thread_relaxed);
    // t_thread.join();
    // cudaDeviceSynchronize();

    // std::cout << "Host-PING Device-PONG (Thread, Relaxed, CPU-CAS GPU-Decoupled) | Host : " << ((double) (cpu_time_thread / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    cuda::atomic<uint16_t, cuda::thread_scope_thread> *flag_thread_acqrel;
    cuda::atomic<uint16_t, cuda::thread_scope_device> *flag_device_acqrel;
    cuda::atomic<uint16_t, cuda::thread_scope_system> *flag_system_acqrel;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&flag_thread_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocHost(&flag_device_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocHost(&flag_system_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == MALLOC) {
        flag_thread_acqrel = (cuda::atomic<uint16_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        flag_device_acqrel = (cuda::atomic<uint16_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        flag_system_acqrel = (cuda::atomic<uint16_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == UM) {
        cudaMallocManaged(&flag_thread_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocManaged(&flag_device_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocManaged(&flag_system_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    }

    uint64_t cpu_time_system_acqrel;

    std::thread t_system_acqrel(host_ping_function_acqrel_base, (std::atomic<uint16_t> *) flag_system_acqrel, &cpu_time_system_acqrel);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_system_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_acqrel_decoupled<<<1,1>>>(flag_system_acqrel);
    t_system_acqrel.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (System, Acq-Rel, CPU-CAS GPU-Decoupled) | Host : " << ((double) (cpu_time_system_acqrel / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    uint64_t cpu_time_device_acqrel;
    std::thread t_device_acqrel(host_ping_function_acqrel_base, (std::atomic<uint16_t> *) flag_device_acqrel, &cpu_time_device_acqrel);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_device_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_acqrel_decoupled<<<1,1>>>(flag_device_acqrel);
    t_device_acqrel.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (Device, Acq-Rel, CPU-CAS GPU-Decoupled) | Host : " << ((double) (cpu_time_device_acqrel / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    // uint64_t cpu_time_thread_acqrel;
    // std::thread t_thread_acqrel(host_ping_function_acqrel_base, (std::atomic<uint16_t> *) flag_thread_acqrel, &cpu_time_thread_acqrel);
    // CPU_ZERO(&cpuset);
    // CPU_SET(0, &cpuset);
    // pthread_setaffinity_np(t_thread_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);
    // device_pong_kernel_acqrel_decoupled<<<1,1>>>(flag_thread_acqrel);
    // t_thread_acqrel.join();
    // cudaDeviceSynchronize();

    // std::cout << "Host-PING Device-PONG (Thread, Acq-Rel, CPU-CAS GPU-Decoupled) | Host : " << ((double) (cpu_time_thread_acqrel / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    cuda::atomic<uint16_t, cuda::thread_scope_thread> *flag_relaxed_thread;
    cuda::atomic<uint16_t, cuda::thread_scope_device> *flag_relaxed_device;
    cuda::atomic<uint16_t, cuda::thread_scope_system> *flag_relaxed_system;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&flag_relaxed_thread, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocHost(&flag_relaxed_device, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocHost(&flag_relaxed_system, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == MALLOC) {
        flag_relaxed_thread = (cuda::atomic<uint16_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        flag_relaxed_device = (cuda::atomic<uint16_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        flag_relaxed_system = (cuda::atomic<uint16_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == UM) {
        cudaMallocManaged(&flag_relaxed_thread, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocManaged(&flag_relaxed_device, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocManaged(&flag_relaxed_system, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    }

    uint64_t cpu_time_relaxed_system;

    std::thread t_system_relaxed(host_ping_function_relaxed_decoupled, (std::atomic<uint16_t> *) flag_relaxed_system, &cpu_time_relaxed_system);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_system_relaxed.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_relaxed_base<<<1,1>>>(flag_relaxed_system);
    t_system_relaxed.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (System, Relaxed, CPU-Decoupled GPU-CAS) | Host : " << ((double) (cpu_time_relaxed_system / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    uint64_t cpu_time_relaxed_device;
    std::thread t_device_relaxed(host_ping_function_relaxed_decoupled, (std::atomic<uint16_t> *) flag_relaxed_device, &cpu_time_relaxed_device);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_device_relaxed.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_relaxed_base<<<1,1>>>(flag_relaxed_device);
    t_device_relaxed.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (Device, Relaxed, CPU-Decoupled GPU-CAS) | Host : " << ((double) (cpu_time_relaxed_device / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    // uint64_t cpu_time_relaxed_thread;
    // std::thread t_thread_relaxed(host_ping_function_relaxed_decoupled, (std::atomic<uint16_t> *) flag_relaxed_thread, &cpu_time_relaxed_thread);
    // CPU_ZERO(&cpuset);
    // CPU_SET(0, &cpuset);
    // pthread_setaffinity_np(t_thread_relaxed.native_handle(), sizeof(cpu_set_t), &cpuset);
    // device_pong_kernel_relaxed_base<<<1,1>>>(flag_relaxed_thread);
    // t_thread_relaxed.join();
    // cudaDeviceSynchronize();

    // std::cout << "Host-PING Device-PONG (Thread, Relaxed, CPU-Decoupled GPU-CAS) | Host : " << ((double) (cpu_time_relaxed_thread / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    cuda::atomic<uint16_t, cuda::thread_scope_thread> *flag_acqrel_thread;
    cuda::atomic<uint16_t, cuda::thread_scope_device> *flag_acqrel_device;
    cuda::atomic<uint16_t, cuda::thread_scope_system> *flag_acqrel_system;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&flag_acqrel_thread, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocHost(&flag_acqrel_device, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocHost(&flag_acqrel_system, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == MALLOC) {
        flag_acqrel_thread = (cuda::atomic<uint16_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        flag_acqrel_device = (cuda::atomic<uint16_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        flag_acqrel_system = (cuda::atomic<uint16_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == UM) {
        cudaMallocManaged(&flag_acqrel_thread, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocManaged(&flag_acqrel_device, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocManaged(&flag_acqrel_system, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    }

    uint64_t cpu_time_acqrel_system;

    std::thread t_acqrel_system(host_ping_function_acqrel_decoupled, (std::atomic<uint16_t> *) flag_acqrel_system, &cpu_time_acqrel_system);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_acqrel_system.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_acqrel_base<<<1,1>>>(flag_acqrel_system);
    t_acqrel_system.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (System, Acq-Rel, CPU-Decoupled GPU-CAS) | Host : " << ((double) (cpu_time_acqrel_system / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    uint64_t cpu_time_acqrel_device;
    std::thread t_acqrel_device(host_ping_function_acqrel_decoupled, (std::atomic<uint16_t> *) flag_acqrel_device, &cpu_time_acqrel_device);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_acqrel_device.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_acqrel_base<<<1,1>>>(flag_acqrel_device);
    t_acqrel_device.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (Device, Acq-Rel, CPU-Decoupled GPU-CAS) | Host : " << ((double) (cpu_time_acqrel_device / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    // uint64_t cpu_time_acqrel_thread;

    // std::thread t_acqrel_thread(host_ping_function_acqrel_decoupled, (std::atomic<uint16_t> *) flag_acqrel_thread, &cpu_time_acqrel_thread);
    // CPU_ZERO(&cpuset);
    // CPU_SET(0, &cpuset);
    // pthread_setaffinity_np(t_acqrel_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    // device_pong_kernel_acqrel_base<<<1,1>>>(flag_acqrel_thread);
    // t_acqrel_thread.join();
    // cudaDeviceSynchronize();

    // std::cout << "Host-PING Device-PONG (Thread, Acq-Rel, CPU-Decoupled GPU-CAS) | Host : " << ((double) (cpu_time_acqrel_thread / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    if (allocator == MALLOC) {
        free(flag_thread_relaxed);
        free(flag_device_relaxed);
        free(flag_system_relaxed);

        free(flag_thread_acqrel);
        free(flag_device_acqrel);
        free(flag_system_acqrel);

        free(flag_relaxed_thread);
        free(flag_relaxed_device);
        free(flag_relaxed_system);

        free(flag_acqrel_thread);
        free(flag_acqrel_device);
        free(flag_acqrel_system);
    } else if (allocator == CUDA_MALLOC_HOST) {
        cudaFreeHost(flag_thread_relaxed);
        cudaFreeHost(flag_device_relaxed);
        cudaFreeHost(flag_system_relaxed);

        cudaFreeHost(flag_thread_acqrel);
        cudaFreeHost(flag_device_acqrel);
        cudaFreeHost(flag_system_acqrel);

        cudaFreeHost(flag_relaxed_thread);
        cudaFreeHost(flag_relaxed_device);
        cudaFreeHost(flag_relaxed_system);

        cudaFreeHost(flag_acqrel_thread);
        cudaFreeHost(flag_acqrel_device);
        cudaFreeHost(flag_acqrel_system);
    } else if (allocator == UM) {
        cudaFree(flag_thread_relaxed);
        cudaFree(flag_device_relaxed);
        cudaFree(flag_system_relaxed);

        cudaFree(flag_thread_acqrel);
        cudaFree(flag_device_acqrel);
        cudaFree(flag_system_acqrel);

        cudaFree(flag_relaxed_thread);
        cudaFree(flag_relaxed_device);
        cudaFree(flag_relaxed_system);

        cudaFree(flag_acqrel_thread);
        cudaFree(flag_acqrel_device);
        cudaFree(flag_acqrel_system);
    }
}

void device_ping_host_pong_assymetric(Allocator allocator) {
    cuda::atomic<uint16_t, cuda::thread_scope_thread> *flag_thread_relaxed;
    cuda::atomic<uint16_t, cuda::thread_scope_device> *flag_device_relaxed;
    cuda::atomic<uint16_t, cuda::thread_scope_system> *flag_system_relaxed;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&flag_thread_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocHost(&flag_device_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocHost(&flag_system_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == MALLOC) {
        flag_thread_relaxed = (cuda::atomic<uint16_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        flag_device_relaxed = (cuda::atomic<uint16_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        flag_system_relaxed = (cuda::atomic<uint16_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == UM) {
        cudaMallocManaged(&flag_thread_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocManaged(&flag_device_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocManaged(&flag_system_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    }

    cpu_set_t cpuset;

    std::thread t_system(host_pong_function_relaxed_base, (std::atomic<uint16_t> *) flag_system_relaxed);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_system.native_handle(), sizeof(cpu_set_t), &cpuset);

    clock_t time_system_relaxed;
    device_ping_kernel_relaxed_decoupled<<<1,1>>>(flag_system_relaxed, &time_system_relaxed);

    t_system.join();
    cudaDeviceSynchronize();

    std::cout << "Device-PING Host-PONG (System, Relaxed, CPU-CAS GPU-Decoupled) | Device : " << ((double) (time_system_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    std::thread t_device(host_pong_function_relaxed_base, (std::atomic<uint16_t> *) flag_device_relaxed);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_device.native_handle(), sizeof(cpu_set_t), &cpuset);

    clock_t time_device_relaxed;
    device_ping_kernel_relaxed_decoupled<<<1,1>>>(flag_device_relaxed, &time_device_relaxed);

    t_device.join();
    cudaDeviceSynchronize();

    std::cout << "Device-PING Host-PONG (Device, Relaxed, CPU-CAS GPU-Decoupled) | Device : " << ((double) (time_device_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    // std::thread t_thread(host_pong_function_relaxed_base, (std::atomic<uint16_t> *) flag_thread_relaxed);
    // CPU_ZERO(&cpuset);
    // CPU_SET(0, &cpuset);
    // pthread_setaffinity_np(t_thread.native_handle(), sizeof(cpu_set_t), &cpuset);

    // clock_t time_thread_relaxed;
    // device_ping_kernel_relaxed_decoupled<<<1,1>>>(flag_thread_relaxed, &time_thread_relaxed);

    // t_thread.join();
    // cudaDeviceSynchronize();

    // std::cout << "Device-PING Host-PONG (Thread, Relaxed, CPU-CAS GPU-Decoupled) | Device : " << ((double) (time_thread_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;

    cuda::atomic<uint16_t, cuda::thread_scope_thread> *flag_thread_acqrel;
    cuda::atomic<uint16_t, cuda::thread_scope_device> *flag_device_acqrel;
    cuda::atomic<uint16_t, cuda::thread_scope_system> *flag_system_acqrel;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&flag_thread_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocHost(&flag_device_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocHost(&flag_system_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == MALLOC) {
        flag_thread_acqrel = (cuda::atomic<uint16_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        flag_device_acqrel = (cuda::atomic<uint16_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        flag_system_acqrel = (cuda::atomic<uint16_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == UM) {
        cudaMallocManaged(&flag_thread_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocManaged(&flag_device_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocManaged(&flag_system_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    }

    std::thread t_system_acqrel(host_pong_function_acqrel_base, (std::atomic<uint16_t> *) flag_system_acqrel);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_system_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);

    clock_t time_system_acqrel;
    device_ping_kernel_acqrel_decoupled<<<1,1>>>(flag_system_acqrel, &time_system_acqrel);

    t_system_acqrel.join();
    cudaDeviceSynchronize();

    std::cout << "Device-PING Host-PONG (System, Acq-Rel, CPU-CAS GPU-Decoupled) | Device : " << ((double) (time_system_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;

    std::thread t_device_acqrel(host_pong_function_acqrel_base, (std::atomic<uint16_t> *) flag_device_acqrel);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_device_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);

    clock_t time_device_acqrel;
    device_ping_kernel_acqrel_decoupled<<<1,1>>>(flag_device_acqrel, &time_device_acqrel);

    t_device_acqrel.join();
    cudaDeviceSynchronize();

    std::cout << "Device-PING Host-PONG (Device, Acq-Rel, CPU-CAS GPU-Decoupled) | Device : " << ((double) (time_device_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    // std::thread t_thread_acqrel(host_pong_function_acqrel_base, (std::atomic<uint16_t> *) flag_thread_acqrel);
    // CPU_ZERO(&cpuset);
    // CPU_SET(0, &cpuset);
    // pthread_setaffinity_np(t_thread_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);

    // clock_t time_thread_acqrel;
    // device_ping_kernel_acqrel_decoupled<<<1,1>>>(flag_thread_acqrel, &time_thread_acqrel);

    // t_thread_acqrel.join();
    // cudaDeviceSynchronize();

    // std::cout << "Device-PING Host-PONG (Thread, Acq-Rel, CPU-CAS GPU-Decoupled) | Device : " << ((double) (time_thread_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;

    cuda::atomic<uint16_t, cuda::thread_scope_thread> *flag_relaxed_thread;
    cuda::atomic<uint16_t, cuda::thread_scope_device> *flag_relaxed_device;
    cuda::atomic<uint16_t, cuda::thread_scope_system> *flag_relaxed_system;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&flag_relaxed_thread, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocHost(&flag_relaxed_device, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocHost(&flag_relaxed_system, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == MALLOC) {
        flag_relaxed_thread = (cuda::atomic<uint16_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        flag_relaxed_device = (cuda::atomic<uint16_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        flag_relaxed_system = (cuda::atomic<uint16_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == UM) {
        cudaMallocManaged(&flag_relaxed_thread, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocManaged(&flag_relaxed_device, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocManaged(&flag_relaxed_system, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    }

    std::thread t_system_relaxed(host_pong_function_relaxed_decoupled, (std::atomic<uint16_t> *) flag_relaxed_system);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_system_relaxed.native_handle(), sizeof(cpu_set_t), &cpuset);

    clock_t time_relaxed_system;
    device_ping_kernel_relaxed_base<<<1,1>>>(flag_relaxed_system, &time_relaxed_system);

    t_system_relaxed.join();
    cudaDeviceSynchronize();

    std::cout << "Device-PING Host-PONG (System, Relaxed, CPU-Decoupled GPU-CAS) | Device : " << ((double) (time_relaxed_system / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    std::thread t_device_relaxed(host_pong_function_relaxed_decoupled, (std::atomic<uint16_t> *) flag_relaxed_device);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_device_relaxed.native_handle(), sizeof(cpu_set_t), &cpuset);

    clock_t time_relaxed_device;
    device_ping_kernel_relaxed_base<<<1,1>>>(flag_relaxed_device, &time_relaxed_device);

    t_device_relaxed.join();
    cudaDeviceSynchronize();

    std::cout << "Device-PING Host-PONG (Device, Relaxed, CPU-Decoupled GPU-CAS) | Device : " << ((double) (time_relaxed_device / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    // std::thread t_thread_relaxed(host_pong_function_relaxed_decoupled, (std::atomic<uint16_t> *) flag_relaxed_thread);
    // CPU_ZERO(&cpuset);
    // CPU_SET(0, &cpuset);
    // pthread_setaffinity_np(t_thread_relaxed.native_handle(), sizeof(cpu_set_t), &cpuset);

    // clock_t time_relaxed_thread;
    // device_ping_kernel_relaxed_base<<<1,1>>>(flag_relaxed_thread, &time_relaxed_thread);
    
    // t_thread_relaxed.join();
    // cudaDeviceSynchronize();

    // std::cout << "Device-PING Host-PONG (Thread, Relaxed, CPU-Decoupled GPU-CAS) | Device : " << ((double) (time_relaxed_thread / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;

    cuda::atomic<uint16_t, cuda::thread_scope_thread> *flag_acqrel_thread;
    cuda::atomic<uint16_t, cuda::thread_scope_device> *flag_acqrel_device;
    cuda::atomic<uint16_t, cuda::thread_scope_system> *flag_acqrel_system;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&flag_acqrel_thread, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocHost(&flag_acqrel_device, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocHost(&flag_acqrel_system, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == MALLOC) {
        flag_acqrel_thread = (cuda::atomic<uint16_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        flag_acqrel_device = (cuda::atomic<uint16_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        flag_acqrel_system = (cuda::atomic<uint16_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == UM) {
        cudaMallocManaged(&flag_acqrel_thread, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocManaged(&flag_acqrel_device, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocManaged(&flag_acqrel_system, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    }

    std::thread t_acqrel_system(host_pong_function_acqrel_decoupled, (std::atomic<uint16_t> *) flag_acqrel_system);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_acqrel_system.native_handle(), sizeof(cpu_set_t), &cpuset);

    clock_t time_acqrel_system;
    device_ping_kernel_acqrel_base<<<1,1>>>(flag_acqrel_system, &time_acqrel_system);

    t_acqrel_system.join();
    cudaDeviceSynchronize();

    std::cout << "Device-PING Host-PONG (System, Acq-Rel, CPU-Decoupled GPU-CAS) | Device : " << ((double) (time_acqrel_system / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    std::thread t_acqrel_device(host_pong_function_acqrel_decoupled, (std::atomic<uint16_t> *) flag_acqrel_device);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_acqrel_device.native_handle(), sizeof(cpu_set_t), &cpuset);

    clock_t time_acqrel_device;
    device_ping_kernel_acqrel_base<<<1,1>>>(flag_acqrel_device, &time_acqrel_device);

    t_acqrel_device.join();
    cudaDeviceSynchronize();

    std::cout << "Device-PING Host-PONG (Device, Acq-Rel, CPU-Decoupled GPU-CAS) | Device : " << ((double) (time_acqrel_device / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    // std::thread t_acqrel_thread(host_pong_function_acqrel_decoupled, (std::atomic<uint16_t> *) flag_acqrel_thread);
    // CPU_ZERO(&cpuset);
    // CPU_SET(0, &cpuset);
    // pthread_setaffinity_np(t_acqrel_thread.native_handle(), sizeof(cpu_set_t), &cpuset);

    // clock_t time_acqrel_thread;
    // device_ping_kernel_acqrel_base<<<1,1>>>(flag_acqrel_thread, &time_acqrel_thread);

    // t_acqrel_thread.join();
    // cudaDeviceSynchronize();

    // std::cout << "Device-PING Host-PONG (Thread, Acq-Rel, CPU-Decoupled GPU-CAS) | Device : " << ((double) (time_acqrel_thread / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;

    if (allocator == MALLOC) {
        free(flag_thread_relaxed);
        free(flag_device_relaxed);
        free(flag_system_relaxed);

        free(flag_thread_acqrel);
        free(flag_device_acqrel);
        free(flag_system_acqrel);

        free(flag_relaxed_thread);
        free(flag_relaxed_device);
        free(flag_relaxed_system);

        free(flag_acqrel_thread);
        free(flag_acqrel_device);
        free(flag_acqrel_system);
    } else if (allocator == CUDA_MALLOC_HOST) {
        cudaFreeHost(flag_thread_relaxed);
        cudaFreeHost(flag_device_relaxed);
        cudaFreeHost(flag_system_relaxed);

        cudaFreeHost(flag_thread_acqrel);
        cudaFreeHost(flag_device_acqrel);
        cudaFreeHost(flag_system_acqrel);

        cudaFreeHost(flag_relaxed_thread);
        cudaFreeHost(flag_relaxed_device);
        cudaFreeHost(flag_relaxed_system);

        cudaFreeHost(flag_acqrel_thread);
        cudaFreeHost(flag_acqrel_device);
        cudaFreeHost(flag_acqrel_system);
    } else if (allocator == UM) {
        cudaFree(flag_thread_relaxed);
        cudaFree(flag_device_relaxed);
        cudaFree(flag_system_relaxed);

        cudaFree(flag_thread_acqrel);
        cudaFree(flag_device_acqrel);
        cudaFree(flag_system_acqrel);

        cudaFree(flag_relaxed_thread);
        cudaFree(flag_relaxed_device);
        cudaFree(flag_relaxed_system);

        cudaFree(flag_acqrel_thread);
        cudaFree(flag_acqrel_device);
        cudaFree(flag_acqrel_system);
    }
}

void host_ping_device_pong_base(Allocator allocator) {
    cuda::atomic<uint16_t, cuda::thread_scope_thread> *flag_thread_relaxed;
    cuda::atomic<uint16_t, cuda::thread_scope_device> *flag_device_relaxed;
    cuda::atomic<uint16_t, cuda::thread_scope_system> *flag_system_relaxed;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&flag_thread_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocHost(&flag_device_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocHost(&flag_system_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == MALLOC) {
        flag_thread_relaxed = (cuda::atomic<uint16_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        flag_device_relaxed = (cuda::atomic<uint16_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        flag_system_relaxed = (cuda::atomic<uint16_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == UM) {
        cudaMallocManaged(&flag_thread_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocManaged(&flag_device_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocManaged(&flag_system_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } 

    cpu_set_t cpuset;
    
    uint64_t cpu_time_system;
    std::thread t_system(host_ping_function_relaxed_base, (std::atomic<uint16_t> *) flag_system_relaxed, &cpu_time_system);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_system.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_relaxed_base<<<1,1>>>(flag_system_relaxed);
    t_system.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (System, Relaxed) | Host : " << ((double) (cpu_time_system / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;


    uint64_t cpu_time_device;
    std::thread t_device(host_ping_function_relaxed_base, (std::atomic<uint16_t> *) flag_device_relaxed, &cpu_time_device);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_device.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_relaxed_base<<<1,1>>>(flag_device_relaxed);
    t_device.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (Device, Relaxed) | Host : " << ((double) (cpu_time_device / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;


    uint64_t cpu_time_thread;
    std::thread t_thread(host_ping_function_relaxed_base, (std::atomic<uint16_t> *) flag_thread_relaxed, &cpu_time_thread);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_relaxed_base<<<1,1>>>(flag_thread_relaxed);
    t_thread.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (Thread, Relaxed) | Host : " << ((double) (cpu_time_thread / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;
    
    cuda::atomic<uint16_t, cuda::thread_scope_thread> *flag_thread_acqrel;
    cuda::atomic<uint16_t, cuda::thread_scope_device> *flag_device_acqrel;
    cuda::atomic<uint16_t, cuda::thread_scope_system> *flag_system_acqrel;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&flag_thread_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocHost(&flag_device_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocHost(&flag_system_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == MALLOC) {
        flag_thread_acqrel = (cuda::atomic<uint16_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        flag_device_acqrel = (cuda::atomic<uint16_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        flag_system_acqrel = (cuda::atomic<uint16_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == UM) {
        cudaMallocManaged(&flag_thread_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocManaged(&flag_device_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocManaged(&flag_system_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    }

    uint64_t cpu_time_system_acqrel;
    std::thread t_system_acqrel(host_ping_function_acqrel_base, (std::atomic<uint16_t> *) flag_system_acqrel, &cpu_time_system_acqrel);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_system_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_acqrel_base<<<1,1>>>(flag_system_acqrel);
    t_system_acqrel.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (System, Acq-Rel) | Host : " << ((double) (cpu_time_system_acqrel / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    uint64_t cpu_time_device_acqrel;
    std::thread t_device_acqrel(host_ping_function_acqrel_base, (std::atomic<uint16_t> *) flag_device_acqrel, &cpu_time_device_acqrel);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_device_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_acqrel_base<<<1,1>>>(flag_device_acqrel);
    t_device_acqrel.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (Device, Acq-Rel) | Host : " << ((double) (cpu_time_device_acqrel / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    uint64_t cpu_time_thread_acqrel;
    std::thread t_thread_acqrel(host_ping_function_acqrel_base, (std::atomic<uint16_t> *) flag_thread_acqrel, &cpu_time_thread_acqrel);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_thread_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_acqrel_base<<<1,1>>>(flag_thread_acqrel);
    t_thread_acqrel.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (Thread, Acq-Rel) | Host : " << ((double) (cpu_time_thread_acqrel / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    if (allocator == MALLOC) {
        free(flag_thread_relaxed);
        free(flag_device_relaxed);
        free(flag_system_relaxed);

        free(flag_thread_acqrel);
        free(flag_device_acqrel);
        free(flag_system_acqrel);
    } else if (allocator == CUDA_MALLOC_HOST) {
        cudaFreeHost(flag_thread_relaxed);
        cudaFreeHost(flag_device_relaxed);
        cudaFreeHost(flag_system_relaxed);

        cudaFreeHost(flag_thread_acqrel);
        cudaFreeHost(flag_device_acqrel);
        cudaFreeHost(flag_system_acqrel);
    } else if (allocator == UM) {
        cudaFree(flag_thread_relaxed);
        cudaFree(flag_device_relaxed);
        cudaFree(flag_system_relaxed);

        cudaFree(flag_thread_acqrel);
        cudaFree(flag_device_acqrel);
        cudaFree(flag_system_acqrel);
    }
}

void host_ping_device_pong_decoupled(Allocator allocator) {
    cuda::atomic<uint16_t, cuda::thread_scope_thread> *flag_thread_relaxed;
    cuda::atomic<uint16_t, cuda::thread_scope_device> *flag_device_relaxed;
    cuda::atomic<uint16_t, cuda::thread_scope_system> *flag_system_relaxed;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&flag_thread_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocHost(&flag_device_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocHost(&flag_system_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == MALLOC) {
        flag_thread_relaxed = (cuda::atomic<uint16_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        flag_device_relaxed = (cuda::atomic<uint16_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        flag_system_relaxed = (cuda::atomic<uint16_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == UM) {
        cudaMallocManaged(&flag_thread_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocManaged(&flag_device_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocManaged(&flag_system_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    }

    cpu_set_t cpuset;
    
    uint64_t cpu_time_system;
    std::thread t_system(host_ping_function_relaxed_decoupled, (std::atomic<uint16_t> *) flag_system_relaxed, &cpu_time_system);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_system.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_relaxed_decoupled<<<1,1>>>(flag_system_relaxed);
    t_system.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (System, Relaxed, Decoupled) | Host : " << ((double) (cpu_time_system / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    uint64_t cpu_time_device;
    std::thread t_device(host_ping_function_relaxed_decoupled, (std::atomic<uint16_t> *) flag_device_relaxed, &cpu_time_device);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_device.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_relaxed_decoupled<<<1,1>>>(flag_device_relaxed);
    t_device.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (Device, Relaxed, Decoupled) | Host : " << ((double) (cpu_time_device / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    // uint64_t cpu_time_thread;
    // std::thread t_thread(host_ping_function_relaxed_decoupled, (std::atomic<uint16_t> *) flag_thread_relaxed, &cpu_time_thread);
    // CPU_ZERO(&cpuset);
    // CPU_SET(0, &cpuset);
    // pthread_setaffinity_np(t_thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    // device_pong_kernel_relaxed_decoupled<<<1,1>>>(flag_thread_relaxed);
    // t_thread.join();
    // cudaDeviceSynchronize();

    // std::cout << "Host-PING Device-PONG (Thread, Relaxed, Decoupled) | Host : " << ((double) (cpu_time_thread / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;
    

    cuda::atomic<uint16_t, cuda::thread_scope_thread> *flag_thread_acqrel;
    cuda::atomic<uint16_t, cuda::thread_scope_device> *flag_device_acqrel;
    cuda::atomic<uint16_t, cuda::thread_scope_system> *flag_system_acqrel;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&flag_thread_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocHost(&flag_device_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocHost(&flag_system_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == MALLOC) {
        flag_thread_acqrel = (cuda::atomic<uint16_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        flag_device_acqrel = (cuda::atomic<uint16_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        flag_system_acqrel = (cuda::atomic<uint16_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == UM) {
        cudaMallocManaged(&flag_thread_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocManaged(&flag_device_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocManaged(&flag_system_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    }

    uint64_t cpu_time_system_acqrel;
    std::thread t_system_acqrel(host_ping_function_acqrel_decoupled, (std::atomic<uint16_t> *) flag_system_acqrel, &cpu_time_system_acqrel);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_system_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);
    device_pong_kernel_acqrel_decoupled<<<1,1>>>(flag_system_acqrel);
    t_system_acqrel.join();
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (System, Acq-Rel, Decoupled) | Host : " << ((double) (cpu_time_system_acqrel / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    uint64_t cpu_time_device_acqrel; //-
    std::thread t_device_acqrel(host_ping_function_acqrel_decoupled, (std::atomic<uint16_t> *) flag_device_acqrel, &cpu_time_device_acqrel); //-
    CPU_ZERO(&cpuset); //-
    CPU_SET(0, &cpuset); //-
    pthread_setaffinity_np(t_device_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset); //-
    device_pong_kernel_acqrel_decoupled<<<1,1>>>(flag_device_acqrel); 
    t_device_acqrel.join(); //-
    cudaDeviceSynchronize();

    std::cout << "Host-PING Device-PONG (Device, Acq-Rel, Decoupled) | Host : " << ((double) (cpu_time_device_acqrel / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    // uint64_t cpu_time_thread_acqrel;
    // std::thread t_thread_acqrel(host_ping_function_acqrel_decoupled, (std::atomic<uint16_t> *) flag_thread_acqrel, &cpu_time_thread_acqrel);
    // CPU_ZERO(&cpuset);
    // CPU_SET(0, &cpuset);
    // pthread_setaffinity_np(t_thread_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);
    // device_pong_kernel_acqrel_decoupled<<<1,1>>>(flag_thread_acqrel);
    // t_thread_acqrel.join();
    // cudaDeviceSynchronize();

    // std::cout << "Host-PING Device-PONG (Thread, Acq-Rel, Decoupled) | Host : " << ((double) (cpu_time_thread_acqrel / 10000)) / ((double) get_cpu_freq() / 1000.) * 1000000. << std::endl;

    if (allocator == MALLOC) {
        free(flag_thread_relaxed);
        free(flag_device_relaxed);
        free(flag_system_relaxed);

        free(flag_thread_acqrel);
        free(flag_device_acqrel);
        free(flag_system_acqrel);
    } else if (allocator == CUDA_MALLOC_HOST) {
        cudaFreeHost(flag_thread_relaxed);
        cudaFreeHost(flag_device_relaxed);
        cudaFreeHost(flag_system_relaxed);

        cudaFreeHost(flag_thread_acqrel);
        cudaFreeHost(flag_device_acqrel);
        cudaFreeHost(flag_system_acqrel);
    } else if (allocator == UM) {
        cudaFree(flag_thread_relaxed);
        cudaFree(flag_device_relaxed);
        cudaFree(flag_system_relaxed);

        cudaFree(flag_thread_acqrel);
        cudaFree(flag_device_acqrel);
        cudaFree(flag_system_acqrel);
    }
}

void device_ping_device_pong_decoupled(Allocator allocator) {
    cuda::atomic<uint16_t, cuda::thread_scope_thread> *flag_thread_acqrel;
    cuda::atomic<uint16_t, cuda::thread_scope_device> *flag_device_acqrel;
    cuda::atomic<uint16_t, cuda::thread_scope_system> *flag_system_acqrel;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&flag_thread_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocHost(&flag_device_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocHost(&flag_system_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == MALLOC) {
        flag_thread_acqrel = (cuda::atomic<uint16_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        flag_device_acqrel = (cuda::atomic<uint16_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        flag_system_acqrel = (cuda::atomic<uint16_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == CUDA_MALLOC) {
        cudaMalloc(&flag_thread_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMalloc(&flag_device_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMalloc(&flag_system_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == UM) {
        cudaMallocManaged(&flag_thread_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocManaged(&flag_device_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocManaged(&flag_system_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    }

    cudaStream_t stream_a, stream_b;
    cudaStreamCreate(&stream_a);
    cudaStreamCreate(&stream_b);

    // clock_t time_system_acqrel;
    // device_ping_kernel_acqrel_decoupled<<<1,1,0,stream_a>>>(flag_system_acqrel, &time_system_acqrel);
    // device_pong_kernel_acqrel_decoupled<<<1,1,0,stream_b>>>(flag_system_acqrel);
    // cudaDeviceSynchronize();

    // std::cout << "Device-PING Device-PONG (System, Acq-Rel, Decoupled) | Device : " << ((double) (time_system_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    // clock_t time_device_acqrel;
    // device_ping_kernel_acqrel_decoupled<<<1,1,0,stream_a>>>(flag_device_acqrel, &time_device_acqrel);
    // device_pong_kernel_acqrel_decoupled<<<1,1,0,stream_b>>>(flag_device_acqrel);
    // cudaDeviceSynchronize();

    // std::cout << "Device-PING Device-PONG (Device, Acq-Rel, Decoupled) | Device : " << ((double) (time_device_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    // clock_t time_thread_acqrel;
    // device_ping_kernel_acqrel_decoupled<<<1,1,0,stream_a>>>(flag_thread_acqrel, &time_thread_acqrel);
    // device_pong_kernel_acqrel_decoupled<<<1,1,0,stream_b>>>(flag_thread_acqrel);
    // cudaDeviceSynchronize();

    // std::cout << "Device-PING Device-PONG (Thread, Acq-Rel, Decoupled) | Device : " << ((double) (time_thread_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;
    
    cuda::atomic<uint16_t, cuda::thread_scope_thread> *flag_thread_relaxed;
    cuda::atomic<uint16_t, cuda::thread_scope_device> *flag_device_relaxed;
    cuda::atomic<uint16_t, cuda::thread_scope_system> *flag_system_relaxed;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&flag_thread_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocHost(&flag_device_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocHost(&flag_system_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == MALLOC) {
        flag_thread_relaxed = (cuda::atomic<uint16_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        flag_device_relaxed = (cuda::atomic<uint16_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        flag_system_relaxed = (cuda::atomic<uint16_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == CUDA_MALLOC) {
        cudaMalloc(&flag_thread_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMalloc(&flag_device_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMalloc(&flag_system_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == UM) {
        cudaMallocManaged(&flag_thread_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocManaged(&flag_device_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocManaged(&flag_system_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    }

    clock_t time_system_relaxed;
    device_ping_kernel_relaxed_decoupled<<<1,1,0,stream_a>>>(flag_system_relaxed, &time_system_relaxed);
    device_pong_kernel_relaxed_decoupled<<<1,1,0,stream_b>>>(flag_system_relaxed);
    cudaDeviceSynchronize();

    std::cout << "Device-PING Device-PONG (System, Relaxed, Decoupled) | Device : " << ((double) (time_system_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    clock_t time_device_relaxed;
    device_ping_kernel_relaxed_decoupled<<<1,1,0,stream_a>>>(flag_device_relaxed, &time_device_relaxed);
    device_pong_kernel_relaxed_decoupled<<<1,1,0,stream_b>>>(flag_device_relaxed);
    cudaDeviceSynchronize();

    std::cout << "Device-PING Device-PONG (Device, Relaxed, Decoupled) | Device : " << ((double) (time_device_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    // clock_t time_thread_relaxed;
    // device_ping_kernel_relaxed_decoupled<<<1,1,0,stream_a>>>(flag_thread_relaxed, &time_thread_relaxed);
    // device_pong_kernel_relaxed_decoupled<<<1,1,0,stream_b>>>(flag_thread_relaxed);
    // cudaDeviceSynchronize();

    // std::cout << "Device-PING Device-PONG (Thread, Relaxed, Decoupled) | Device : " << ((double) (time_thread_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;

    if (allocator == MALLOC) {
        free(flag_thread_relaxed);
        free(flag_device_relaxed);
        free(flag_system_relaxed);

        free(flag_thread_acqrel);
        free(flag_device_acqrel);
        free(flag_system_acqrel);
    } else if (allocator == CUDA_MALLOC_HOST) {
        cudaFreeHost(flag_thread_relaxed);
        cudaFreeHost(flag_device_relaxed);
        cudaFreeHost(flag_system_relaxed);

        cudaFreeHost(flag_thread_acqrel);
        cudaFreeHost(flag_device_acqrel);
        cudaFreeHost(flag_system_acqrel);
    } else if (allocator == CUDA_MALLOC || allocator == UM) {
        cudaFree(flag_thread_relaxed);
        cudaFree(flag_device_relaxed);
        cudaFree(flag_system_relaxed);

        cudaFree(flag_thread_acqrel);
        cudaFree(flag_device_acqrel);
        cudaFree(flag_system_acqrel);
    }
}

void device_ping_device_pong_base(Allocator allocator) {
    cuda::atomic<uint16_t, cuda::thread_scope_thread> *flag_thread_relaxed;
    cuda::atomic<uint16_t, cuda::thread_scope_device> *flag_device_relaxed;
    cuda::atomic<uint16_t, cuda::thread_scope_system> *flag_system_relaxed;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&flag_thread_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocHost(&flag_device_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocHost(&flag_system_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == MALLOC) {
        flag_thread_relaxed = (cuda::atomic<uint16_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        flag_device_relaxed = (cuda::atomic<uint16_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        flag_system_relaxed = (cuda::atomic<uint16_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == CUDA_MALLOC) {
        cudaMalloc(&flag_thread_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMalloc(&flag_device_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMalloc(&flag_system_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == UM) {
        cudaMallocManaged(&flag_thread_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocManaged(&flag_device_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocManaged(&flag_system_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    }

    cudaStream_t stream_a, stream_b;
    cudaStreamCreate(&stream_a);
    cudaStreamCreate(&stream_b);

    // clock_t time_system_relaxed;
    clock_t *time_system_relaxed;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&time_system_relaxed, sizeof(clock_t));
    } else if (allocator == MALLOC) {
        time_system_relaxed = (clock_t *) malloc(sizeof(clock_t));
    } else if (allocator == CUDA_MALLOC) {
        cudaMalloc(&time_system_relaxed, sizeof(clock_t));
    } else if (allocator == UM) {
        cudaMallocManaged(&time_system_relaxed, sizeof(clock_t));
    }

    device_ping_kernel_relaxed_base<<<1,1,0,stream_a>>>(flag_system_relaxed, time_system_relaxed);
    device_pong_kernel_relaxed_base<<<1,1,0,stream_b>>>(flag_system_relaxed);

    cudaStreamSynchronize(stream_a);
    cudaStreamSynchronize(stream_b);

    cudaDeviceSynchronize();

    if (allocator == CUDA_MALLOC) {
        clock_t time;

        cudaMemcpy(&time, time_system_relaxed, sizeof(clock_t), cudaMemcpyDeviceToHost);
        std::cout << "Device-PING Device-PONG (System, Relaxed) | Device : " << ((double) (time / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;
    } else {
        std::cout << "Device-PING Device-PONG (System, Relaxed) | Device : " << ((double) (*time_system_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;
    }



    // clock_t time_device_relaxed;
    // device_ping_kernel_relaxed_base<<<1,1,0,stream_a>>>(flag_device_relaxed, &time_device_relaxed);
    // device_pong_kernel_relaxed_base<<<1,1,0,stream_b>>>(flag_device_relaxed);
    // cudaDeviceSynchronize();

    // std::cout << "Device-PING Device-PONG (Device, Relaxed) | Device : " << ((double) (time_device_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    // clock_t time_thread_relaxed;
    // device_ping_kernel_relaxed_base<<<1,1,0,stream_a>>>(flag_thread_relaxed, &time_thread_relaxed);
    // device_pong_kernel_relaxed_base<<<1,1,0,stream_b>>>(flag_thread_relaxed);
    // cudaDeviceSynchronize();

    // std::cout << "Device-PING Device-PONG (Thread, Relaxed) | Device : " << ((double) (time_thread_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    cuda::atomic<uint16_t, cuda::thread_scope_thread> *flag_thread_acqrel;
    cuda::atomic<uint16_t, cuda::thread_scope_device> *flag_device_acqrel;
    cuda::atomic<uint16_t, cuda::thread_scope_system> *flag_system_acqrel;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&flag_thread_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocHost(&flag_device_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocHost(&flag_system_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == MALLOC) {
        flag_thread_acqrel = (cuda::atomic<uint16_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        flag_device_acqrel = (cuda::atomic<uint16_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        flag_system_acqrel = (cuda::atomic<uint16_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == CUDA_MALLOC) {
        cudaMalloc(&flag_thread_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMalloc(&flag_device_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMalloc(&flag_system_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == UM) {
        cudaMallocManaged(&flag_thread_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocManaged(&flag_device_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocManaged(&flag_system_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    }

    // clock_t time_system_acqrel;
    // device_ping_kernel_acqrel_base<<<1,1,0,stream_a>>>(flag_system_acqrel, &time_system_acqrel);
    // device_pong_kernel_acqrel_base<<<1,1,0,stream_b>>>(flag_system_acqrel);
    // cudaDeviceSynchronize();

    // std::cout << "Device-PING Device-PONG (System, Acq-Rel) | Device : " << ((double) (time_system_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    // clock_t time_device_acqrel;
    // device_ping_kernel_acqrel_base<<<1,1,0,stream_a>>>(flag_device_acqrel, &time_device_acqrel);
    // device_pong_kernel_acqrel_base<<<1,1,0,stream_b>>>(flag_device_acqrel);
    // cudaDeviceSynchronize();

    // std::cout << "Device-PING Device-PONG (Device, Acq-Rel) | Device : " << ((double) (time_device_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    // clock_t time_thread_acqrel;
    // device_ping_kernel_acqrel_base<<<1,1,0,stream_a>>>(flag_thread_acqrel, &time_thread_acqrel);
    // device_pong_kernel_acqrel_base<<<1,1,0,stream_b>>>(flag_thread_acqrel);
    // cudaDeviceSynchronize();

    // std::cout << "Device-PING Device-PONG (Thread, Acq-Rel) | Device : " << ((double) (time_thread_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    if (allocator == MALLOC) {
        free(flag_thread_relaxed);
        free(flag_device_relaxed);
        free(flag_system_relaxed);

        free(flag_thread_acqrel);
        free(flag_device_acqrel);
        free(flag_system_acqrel);
    } else if (allocator == CUDA_MALLOC_HOST) {
        cudaFreeHost(flag_thread_relaxed);
        cudaFreeHost(flag_device_relaxed);
        cudaFreeHost(flag_system_relaxed);

        cudaFreeHost(flag_thread_acqrel);
        cudaFreeHost(flag_device_acqrel);
        cudaFreeHost(flag_system_acqrel);
    } else if (allocator == CUDA_MALLOC || allocator == UM) {
        cudaFree(flag_thread_relaxed);
        cudaFree(flag_device_relaxed);
        cudaFree(flag_system_relaxed);

        cudaFree(flag_thread_acqrel);
        cudaFree(flag_device_acqrel);
        cudaFree(flag_system_acqrel);
    }
}

void device_ping_host_pong_base(Allocator allocator) {
    cuda::atomic<uint16_t, cuda::thread_scope_thread> *flag_thread_relaxed;
    cuda::atomic<uint16_t, cuda::thread_scope_device> *flag_device_relaxed;
    cuda::atomic<uint16_t, cuda::thread_scope_system> *flag_system_relaxed;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&flag_thread_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocHost(&flag_device_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocHost(&flag_system_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == MALLOC) {
        flag_thread_relaxed = (cuda::atomic<uint16_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        flag_device_relaxed = (cuda::atomic<uint16_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        flag_system_relaxed = (cuda::atomic<uint16_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == UM) {
        cudaMallocManaged(&flag_thread_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocManaged(&flag_device_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocManaged(&flag_system_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } 

    cpu_set_t cpuset;

    std::thread t_system_relaxed(host_pong_function_relaxed_base, (std::atomic<uint16_t> *) flag_system_relaxed);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_system_relaxed.native_handle(), sizeof(cpu_set_t), &cpuset);

    clock_t time_system_relaxed;
    device_ping_kernel_relaxed_base<<<1,1>>>(flag_system_relaxed, &time_system_relaxed);

    t_system_relaxed.join();
    cudaDeviceSynchronize();

    std::cout << "Device-PING Host-PONG (System, Relaxed) | Device : " << ((double) (time_system_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    std::thread t_device_relaxed(host_pong_function_relaxed_base, (std::atomic<uint16_t> *) flag_device_relaxed);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_device_relaxed.native_handle(), sizeof(cpu_set_t), &cpuset);

    clock_t time_device_relaxed;
    device_ping_kernel_relaxed_base<<<1,1>>>(flag_device_relaxed, &time_device_relaxed);

    t_device_relaxed.join();
    cudaDeviceSynchronize();

    std::cout << "Device-PING Host-PONG (Device, Relaxed) | Device : " << ((double) (time_device_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    std::thread t_thread_relaxed(host_pong_function_relaxed_base, (std::atomic<uint16_t> *) flag_thread_relaxed);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_thread_relaxed.native_handle(), sizeof(cpu_set_t), &cpuset);

    clock_t time_thread_relaxed;
    device_ping_kernel_relaxed_base<<<1,1>>>(flag_thread_relaxed, &time_thread_relaxed);

    t_thread_relaxed.join();
    cudaDeviceSynchronize();

    std::cout << "Device-PING Host-PONG (Thread, Relaxed) | Device : " << ((double) (time_thread_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    cuda::atomic<uint16_t, cuda::thread_scope_thread> *flag_thread_acqrel;
    cuda::atomic<uint16_t, cuda::thread_scope_device> *flag_device_acqrel;
    cuda::atomic<uint16_t, cuda::thread_scope_system> *flag_system_acqrel;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&flag_thread_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocHost(&flag_device_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocHost(&flag_system_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == MALLOC) {
        flag_thread_acqrel = (cuda::atomic<uint16_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        flag_device_acqrel = (cuda::atomic<uint16_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        flag_system_acqrel = (cuda::atomic<uint16_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == UM) {
        cudaMallocManaged(&flag_thread_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocManaged(&flag_device_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocManaged(&flag_system_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    }

    std::thread t_system_acqrel(host_pong_function_acqrel_base, (std::atomic<uint16_t> *) flag_system_acqrel);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset); 
    pthread_setaffinity_np(t_system_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);
    
    clock_t time_system_acqrel;
    device_ping_kernel_acqrel_base<<<1,1>>>(flag_system_acqrel, &time_system_acqrel);

    t_system_acqrel.join();
    cudaDeviceSynchronize();

    std::cout << "Device-PING Host-PONG (System, Acq-Rel) | Device : " << ((double) (time_system_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    std::thread t_device_acqrel(host_pong_function_acqrel_base, (std::atomic<uint16_t> *) flag_device_acqrel);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_device_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);

    clock_t time_device_acqrel;
    device_ping_kernel_acqrel_base<<<1,1>>>(flag_device_acqrel, &time_device_acqrel);

    t_device_acqrel.join();
    cudaDeviceSynchronize();

    std::cout << "Device-PING Host-PONG (Device, Acq-Rel) | Device : " << ((double) (time_device_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    std::thread t_thread_acqrel(host_pong_function_acqrel_base, (std::atomic<uint16_t> *) flag_thread_acqrel);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_thread_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);

    clock_t time_thread_acqrel;
    device_ping_kernel_acqrel_base<<<1,1>>>(flag_thread_acqrel, &time_thread_acqrel);

    t_thread_acqrel.join();
    cudaDeviceSynchronize();

    std::cout << "Device-PING Host-PONG (Thread, Acq-Rel) | Device : " << ((double) (time_thread_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;

    if (allocator == MALLOC) {
        free(flag_thread_relaxed);
        free(flag_device_relaxed);
        free(flag_system_relaxed);

        free(flag_thread_acqrel);
        free(flag_device_acqrel);
        free(flag_system_acqrel);
    } else if (allocator == CUDA_MALLOC_HOST) {
        cudaFreeHost(flag_thread_relaxed);
        cudaFreeHost(flag_device_relaxed);
        cudaFreeHost(flag_system_relaxed);

        cudaFreeHost(flag_thread_acqrel);
        cudaFreeHost(flag_device_acqrel);
        cudaFreeHost(flag_system_acqrel);
    } else if (allocator == UM) {
        cudaFree(flag_thread_relaxed);
        cudaFree(flag_device_relaxed);
        cudaFree(flag_system_relaxed);

        cudaFree(flag_thread_acqrel);
        cudaFree(flag_device_acqrel);
        cudaFree(flag_system_acqrel);
    }
}

void device_ping_host_pong_decoupled(Allocator allocator) {
    cuda::atomic<uint16_t, cuda::thread_scope_thread> *flag_thread_relaxed;
    cuda::atomic<uint16_t, cuda::thread_scope_device> *flag_device_relaxed;
    cuda::atomic<uint16_t, cuda::thread_scope_system> *flag_system_relaxed;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&flag_thread_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocHost(&flag_device_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocHost(&flag_system_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == MALLOC) {
        flag_thread_relaxed = (cuda::atomic<uint16_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        flag_device_relaxed = (cuda::atomic<uint16_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        flag_system_relaxed = (cuda::atomic<uint16_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == UM) {
        cudaMallocManaged(&flag_thread_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocManaged(&flag_device_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocManaged(&flag_system_relaxed, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    }

    cpu_set_t cpuset;

    clock_t time_system_relaxed;
    device_ping_kernel_relaxed_decoupled<<<1,1>>>(flag_system_relaxed, &time_system_relaxed);
    std::thread t_system_relaxed(host_pong_function_relaxed_decoupled, (std::atomic<uint16_t> *) flag_system_relaxed);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_system_relaxed.native_handle(), sizeof(cpu_set_t), &cpuset);


    cudaDeviceSynchronize();
    t_system_relaxed.join();

    std::cout << "Device-PING Host-PONG (System, Relaxed, Decoupled) | Device : " << ((double) (time_system_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    clock_t time_device_relaxed;
    device_ping_kernel_relaxed_decoupled<<<1,1>>>(flag_device_relaxed, &time_device_relaxed);
    std::thread t_device_relaxed(host_pong_function_relaxed_decoupled, (std::atomic<uint16_t> *) flag_device_relaxed);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_device_relaxed.native_handle(), sizeof(cpu_set_t), &cpuset);


    cudaDeviceSynchronize();
    t_device_relaxed.join();

    std::cout << "Device-PING Host-PONG (Device, Relaxed, Decoupled) | Device : " << ((double) (time_device_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    // std::thread t_thread_relaxed(host_pong_function_relaxed_decoupled, (std::atomic<uint16_t> *) flag_thread_relaxed);
    // CPU_ZERO(&cpuset);
    // CPU_SET(0, &cpuset);
    // pthread_setaffinity_np(t_thread_relaxed.native_handle(), sizeof(cpu_set_t), &cpuset);

    // clock_t time_thread_relaxed;
    // device_ping_kernel_relaxed_decoupled<<<1,1>>>(flag_thread_relaxed, &time_thread_relaxed);

    // t_thread_relaxed.join();
    // cudaDeviceSynchronize();

    // std::cout << "Device-PING Host-PONG (Thread, Relaxed, Decoupled) | Device : " << ((double) (time_thread_relaxed / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    cuda::atomic<uint16_t, cuda::thread_scope_thread> *flag_thread_acqrel;
    cuda::atomic<uint16_t, cuda::thread_scope_device> *flag_device_acqrel;
    cuda::atomic<uint16_t, cuda::thread_scope_system> *flag_system_acqrel;

    if (allocator == CUDA_MALLOC_HOST) {
        cudaMallocHost(&flag_thread_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocHost(&flag_device_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocHost(&flag_system_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == MALLOC) {
        flag_thread_acqrel = (cuda::atomic<uint16_t, cuda::thread_scope_thread> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        flag_device_acqrel = (cuda::atomic<uint16_t, cuda::thread_scope_device> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        flag_system_acqrel = (cuda::atomic<uint16_t, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    } else if (allocator == UM) {
        cudaMallocManaged(&flag_thread_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_thread>));
        cudaMallocManaged(&flag_device_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_device>));
        cudaMallocManaged(&flag_system_acqrel, sizeof(cuda::atomic<uint16_t, cuda::thread_scope_system>));
    }

    clock_t time_system_acqrel;
    device_ping_kernel_acqrel_decoupled<<<1,1>>>(flag_system_acqrel, &time_system_acqrel);
    std::thread t_system_acqrel(host_pong_function_acqrel_decoupled, (std::atomic<uint16_t> *) flag_system_acqrel);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset); 
    pthread_setaffinity_np(t_system_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);
    

    cudaDeviceSynchronize();
    t_system_acqrel.join();

    std::cout << "Device-PING Host-PONG (System, Acq-Rel, Decoupled) | Device : " << ((double) (time_system_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    clock_t time_device_acqrel;
    device_ping_kernel_acqrel_decoupled<<<1,1>>>(flag_device_acqrel, &time_device_acqrel);
    std::thread t_device_acqrel(host_pong_function_acqrel_decoupled, (std::atomic<uint16_t> *) flag_device_acqrel);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(t_device_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);


    cudaDeviceSynchronize();
    t_device_acqrel.join();

    std::cout << "Device-PING Host-PONG (Device, Acq-Rel, Decoupled) | Device : " << ((double) (time_device_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;


    // std::thread t_thread_acqrel(host_pong_function_acqrel_decoupled, (std::atomic<uint16_t> *) flag_thread_acqrel);
    // CPU_ZERO(&cpuset);
    // CPU_SET(0, &cpuset);
    // pthread_setaffinity_np(t_thread_acqrel.native_handle(), sizeof(cpu_set_t), &cpuset);

    // clock_t time_thread_acqrel;
    // device_ping_kernel_acqrel_decoupled<<<1,1>>>(flag_thread_acqrel, &time_thread_acqrel);

    // t_thread_acqrel.join();
    // cudaDeviceSynchronize();

    // std::cout << "Device-PING Host-PONG (Thread, Acq-Rel, Decoupled) | Device : " << ((double) (time_thread_acqrel / 10000)) / ((double) get_gpu_freq()) * 1000000. << std::endl;

    if (allocator == MALLOC) {
        free(flag_thread_relaxed);
        free(flag_device_relaxed);
        free(flag_system_relaxed);

        free(flag_thread_acqrel);
        free(flag_device_acqrel);
        free(flag_system_acqrel);
    } else if (allocator == CUDA_MALLOC_HOST) {
        cudaFreeHost(flag_thread_relaxed);
        cudaFreeHost(flag_device_relaxed);
        cudaFreeHost(flag_system_relaxed);

        cudaFreeHost(flag_thread_acqrel);
        cudaFreeHost(flag_device_acqrel);
        cudaFreeHost(flag_system_acqrel);
    } else if (allocator == UM) {
        cudaFree(flag_thread_relaxed);
        cudaFree(flag_device_relaxed);
        cudaFree(flag_system_relaxed);

        cudaFree(flag_thread_acqrel);
        cudaFree(flag_device_acqrel);
        cudaFree(flag_system_acqrel);
    }
}

#endif // CPU_PINGPONG_HPP