//#include <cuda/atomic>

#include <cstdio>
#include <stdio.h>
#include <inttypes.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#if defined(__x86_64__) || defined(_M_X64)
#include <cassert>
#include <immintrin.h>
#endif

//using namespace cuda;
using namespace std;
#if 0

template <typename T, cuda::thread_scope Scope = cuda::thread_scope_system>
__host__ __device__ void atomicStore(T* ptr, T const& val,
                                            std::memory_order memoryOrder) {
  cuda::atomic_ref<T, Scope>{*ptr}.store(val, memoryOrder);
}
#endif

__host__ __device__ __forceinline__ void commit_data(uint64_t  *data ,int val) {
#if __CUDA_ARCH__ || __HIP_DEVICE_COMPILE__
    //if constexpr (Dir == FlowDirection::DeviceToHost) __threadfence_system();
    __threadfence_system();
#else
    //if constexpr (Dir == FlowDirection::DeviceToHost)
     std::atomic_thread_fence(std::memory_order_release);
    //if constexpr (Dir == FlowDirection::HostToHost) HOST_RELEASE();
#endif
    *data= val;
  }
__global__ void consumer(atomic<int>* flag, uint64_t * data, uint64_t * result0, uint64_t *result1) {
     //while (flag->load(std::memory_order_acquire) == 0) {}
    //*result1 = flag->load(std::memory_order_acquire);
    //*result0 = *data;
    //atomicStore(result0,88,std::memory_order_acquire);
    commit_data(result0,88);
}

#define SAFE(x) if (0 != x) { abort(); }


__device__ __forceinline__ uint64_t ld_volatile(uint64_t* ptr) {
#if defined(__CUDA_ARCH__)
  uint64_t ans;
  asm volatile("ld.volatile.global.u64 %0, [%1];"
               : "=l"(ans)
               : "l"(ptr)
               : "memory");
  return ans;
#elif defined(__HIP_DEVICE_COMPILE__)
  return __builtin_nontemporal_load(ptr);
#else
  return *((volatile uint64_t const*)ptr);
#endif
}
  __host__ __device__ __forceinline__ uint64_t load_data(uint64_t* ptr) {
    uint64_t val = 0;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ld_volatile(ptr);
#elif defined(__x86_64__)
    asm volatile("movq %1, %0" : "=r"(val) : "m"(*ptr) : "memory");
#elif defined(__aarch64__)
    asm volatile("ldr %0, [%1]" : "=r"(val) : "r"(ptr) : "memory");
#else
#error "Unsupported architecture"
#endif
    return val;
  }


int main(int argc, char* argv[]) {
    atomic<int>* flag;
    uint64_t * data;
    uint64_t* result0, *result1;

    int data_in_unified_memory = 1;

    ////////////////////////////////////////////////////////////////////////////
#if __CUDA_ARCH__
    printf("cuda arch is usingn");
#endif
    // Flag in unified memory
    SAFE(cudaMallocManaged(&flag, sizeof(atomic<int>)));

    // Data placed as specified
    if (data_in_unified_memory) {
        SAFE(cudaMallocManaged(&data, sizeof(uint64_t)));
    } else {
        SAFE(cudaMalloc(&data, sizeof(uint64_t)));
    }

 #if 0
    // Result array pinned in CPU memory
    SAFE(cudaMallocHost(&result0, sizeof(uint64_t)));
    SAFE(cudaMallocHost(&result1, sizeof(uint64_t)));
#else
    // Result array pinned in CPU memory
    SAFE(cudaMallocManaged(&result0, sizeof(uint64_t)));
    SAFE(cudaMallocManaged(&result1, sizeof(uint64_t)));
#endif

    // Initial values: data = <unknown>, flag = 0
    flag->store(0, std::memory_order_acquire);
    *data = 0;

    ////////////////////////////////////////////////////////////////////////////

    // Launch the consumer asynchronously
    // 1024 consumers and only 1 producer
    consumer<<<2,1024>>>(flag, data, result0, result1);
    
    // Producer sequence
    if (data_in_unified_memory) {
        *data = 42;
    } else {
        uint64_t h_data = 42;
        SAFE(cudaMemcpy(data, &h_data, sizeof(uint64_t), cudaMemcpyHostToDevice));
    }
    flag->store(1, std::memory_order_acquire);


    // Wait for consumer to finish
    SAFE(cudaDeviceSynchronize());

    // Print the result
    printf("data =" "%" PRIu64  "(expected 42) flag = %d \n", *data, flag->load(memory_order_acquire));
    //printf("data = %ul (expected 42) flag = %ul \n", data->load(memory_order_acquire), flag->load(memory_order_acquire));

    load_data(result0);
    printf("result0 =%" PRIu64 " result1 =%" PRIu64 " \n", *result0,*result1);
    return 0;
}
