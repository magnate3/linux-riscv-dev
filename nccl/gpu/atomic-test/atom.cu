#include <cuda_runtime.h>
#include <cuda/atomic>
#include <stdio.h>

constexpr cuda::memory_order memoryOrderRelaxed = cuda::memory_order_relaxed;
//constexpr cuda::memory_order memoryOrderAcquire = cuda::memory_order_acquire;
//constexpr cuda::memory_order memoryOrderRelease = cuda::memory_order_release;
//constexpr cuda::memory_order memoryOrderAcqRel = cuda::memory_order_acq_rel;
//constexpr cuda::memory_order memoryOrderSeqCst = cuda::memory_order_seq_cst;

//constexpr cuda::thread_scope scopeSystem = cuda::thread_scope_system;
//constexpr cuda::thread_scope scopeDevice = cuda::thread_scope_device;
#define MSCCLPP_HOST_DEVICE_INLINE __forceinline__ __host__ __device__

#if 0
template <typename T, int scope = scopeSystem>
MSCCLPP_HOST_DEVICE_INLINE T atomicFetchAdd(T* ptr, const T& val, int memoryOrder) {
  return __atomic_fetch_add(ptr, val, memoryOrder);
}
#endif
template<typename Int>
inline void ncclAtomicRefCountIncrement(Int* refs) {
  __atomic_fetch_add(refs, 1, __ATOMIC_RELAXED);
}

template<typename Int>
inline Int ncclAtomicRefCountDecrement(Int* refs) {
  return __atomic_sub_fetch(refs, 1, __ATOMIC_ACQ_REL);
}
template <typename T>
MSCCLPP_HOST_DEVICE_INLINE T atomicFetchAdd(T* ptr, const T& val, cuda::memory_order memoryOrder) {
  return cuda::atomic_ref<T, cuda::thread_scope_system>{*ptr}.fetch_add(val, memoryOrder);
}

// Device code (GPU kernel)
__global__ void my_kernel(int* d_ptr) {
    atomicFetchAdd<int>(d_ptr, (uint64_t)1, memoryOrderRelaxed);
}


int test1() {
    int* d_ptr;
    size_t size = sizeof(int);
    cudaMalloc((void**)&d_ptr, size);
    int value = 10;
    //cuda::atomic_ref<int, cuda::thread_scope_system> atomic_data(*d_ptr);
    //atomic_data.store(0);
     // Kernel invocation with one block of N * N * 1 threads
    int numBlocks = 1;
    dim3 threadsPerBlock(4,4);
    cudaMemcpy(d_ptr, &value, size, cudaMemcpyHostToDevice);


#if 0
    
    // 在设备上使用 atomic_ref 进行原子操作
    cuda::atomic_ref<int> atomicRef(*devicePtr);
    atomicRef.fetch_add(5); // 原子加5
#else
    printf("init vaule %d \n", value);
    // This is the syntax for calling global functions.
    my_kernel<<<numBlocks, threadsPerBlock>>>(d_ptr);
    cudaDeviceSynchronize();
#endif
    cudaMemcpy(&value,d_ptr, size, cudaMemcpyDeviceToHost);
    printf(" vaule %d \n", value);
    cudaFree(d_ptr);
    return 0;
}
int test2()
{
    void* hostPtr;
    int* devicePtr;
    size_t size = sizeof(int);
    int initialValue = 10;
     // Kernel invocation with one block of N * N * 1 threads
    int numBlocks = 1;
    dim3 threadsPerBlock(4,4);
    cudaHostAlloc(&hostPtr, size, cudaHostAllocMapped); // 分配页面锁定内存
    cudaHostGetDevicePointer((void**)&devicePtr, hostPtr, 0); // 获取设备指针
    memcpy(hostPtr, &initialValue, size); // 初始化数据
#if 0
    
    // 在设备上使用 atomic_ref 进行原子操作
    cuda::atomic_ref<int> atomicRef(*devicePtr);
    atomicRef.fetch_add(5); // 原子加5
#else
    printf("init vaule %d \n",  *(int*)hostPtr);
    // This is the syntax for calling global functions.
    my_kernel<<<numBlocks, threadsPerBlock>>>(devicePtr);
    printf("final vaule %d \n",  *(int*)hostPtr);
    cudaDeviceSynchronize();
#endif
    printf("* hostPtr value %d ", *(int*)hostPtr);
    return 0;
}
int main() {
    test1();
    test2();
    return 0;
}
