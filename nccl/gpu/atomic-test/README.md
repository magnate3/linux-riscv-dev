


#  uccl-ring

```
static __device__ void device_mutex_unlock_system(uint32_t* mutex) {
  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> lock(*mutex);
  lock.store(0, cuda::memory_order_release);
}
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
#if 0
__global__ void consumer(atomic<int>* flag,uint32_t* spinlock, uint64_t * data, uint64_t * result0, uint64_t *result1) {
     //while (flag->load(std::memory_order_acquire) == 0) {}
    //*result1 = flag->load(std::memory_order_acquire);
    //*result0 = *data;
    //atomicStore(result0,88,std::memory_order_acquire);
    commit_data(result0,88);
    // Acquire spinlock
    device_mutex_lock_system(spinlock);
    commit_data(result1,*result1 +1);
    // Release spinlock
    device_mutex_unlock_system(spinlock);
}
#else
__global__ void consumer(atomic<int>* flag,uint32_t* spinlock, uint64_t * data, uint64_t * result0, uint64_t *result1) {
     //while (flag->load(std::memory_order_acquire) == 0) {}
    //*result1 = flag->load(std::memory_order_acquire);
    //*result0 = *data;
    //atomicStore(result0,88,std::memory_order_acquire);
    commit_data(result0,88);
    commit_data(result1,*result1 +1);
}
```


> ## 加上__CUDA_ARCH__

```
nvcc -c cons_pro.cu -o cons_pro.o -O3 -g -G -Xcompiler -Wall -arch=sm_80 -I./ -I/usr/local/cuda/include
nvcc cons_pro.o -D__CUDA_ARCH__ -o cons_pro -L/usr/local/cuda/lib -lcudart -lcuda -lnuma -L/pytorch/thirdparty/gdrcopy/src -lgdrapi -Xlinker -rpath -Xlinker /pytorch/thirdparty/gdrcopy/src -O3 -g -G -Xcompiler -Wall -arch=sm_80
```


```
uccl-ring# ./cons_pro 
data =42(expected 42) flag = 1 
result0 =88 result1 =1 
```