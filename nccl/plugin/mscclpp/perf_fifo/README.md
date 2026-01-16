

# make

```
mscclpp/test/perf# make
/usr/local/cuda/bin/nvcc -ccbin g++ -std=c++17 -I./ -I/usr/local/mpi/include/   -m64      -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o fifo_test fifo_test.cu logger.cc errors.cc env.cpp numa.cc gpu_utils.cc fifo.cc framework.cc  -lnccl -lmpi -L/usr/local/mpi/lib/ -lnuma  -lcuda
In file included from /usr/local/cuda/bin/../targets/x86_64-linux/include/device_types.h:59,
                 from /usr/local/cuda/bin/../targets/x86_64-linux/include/builtin_types.h:56,
                 from /usr/local/cuda/bin/../targets/x86_64-linux/include/cuda_runtime.h:91,
                 from ./mscclpp/gpu.hpp:129,
                 from ./mscclpp/gpu_utils.hpp:12,
                 from fifo.cc:7:
/usr/local/cuda/bin/../targets/x86_64-linux/include/crt/host_defines.h:86: warning: "__forceinline__" redefined
   86 | #define __forceinline__ \
      | 
In file included from /usr/local/cuda/bin/../targets/x86_64-linux/include/cuda/std/cassert:18,
                 from /usr/local/cuda/bin/../targets/x86_64-linux/include/cuda/std/atomic:45,
                 from /usr/local/cuda/bin/../targets/x86_64-linux/include/cuda/atomic:10,
                 from ./mscclpp/atomic_device.hpp:10,
                 from ./mscclpp/fifo_device.hpp:12,
                 from ./mscclpp/fifo.hpp:9,
                 from fifo.cc:6:
/usr/local/cuda/bin/../targets/x86_64-linux/include/cuda/std/detail/__config:38: note: this is the location of the previous definition
   38 |         #define __forceinline__
```

# run


```
/pytorch/mscclpp/test/perf# ./fifo_test 
Running test: AllFifoTests
  FIFO performance tests with multiple configurations
gpu device  numaNode 0 
numaNode -1 totalNumNumaNodes 1 
Running FIFO test with size=1, parallelism_levels=[1]
gpu device  numaNode 0 
numaNode -1 totalNumNumaNodes 1 
Running FIFO test with size=128, parallelism_levels=[1,8,64,128]
gpu device  numaNode 0 
numaNode -1 totalNumNumaNodes 1 
Running FIFO test with size=512, parallelism_levels=[1,8,64,256,512]

=== Test Results ===

Test: FifoTest_Size1_Parallel1 (fifo)
  Metrics:
    p1_num_triggers: 1000
    p1_push_duration_us: 2691.0
    p1_push_sync_duration_us: 2735.0
    p1_push_sync_throughput: 365630.7
    p1_push_throughput: 371609.0
    p1_warmup_triggers: 100

Test: FifoTest_Size128_ParallelCustom (fifo)
  Metrics:
    p128_num_triggers: 163840
    p128_push_duration_us: 86061.0
    p128_push_sync_duration_us: 101840.0
    p128_push_sync_throughput: 1608798.1
    p128_push_throughput: 1903765.9
    p128_warmup_triggers: 32768
    p1_num_triggers: 1280
    p1_push_duration_us: 1780.0
    p1_push_sync_duration_us: 3543.0
    p1_push_sync_throughput: 361275.7
    p1_push_throughput: 719101.1
    p1_warmup_triggers: 256
    p64_num_triggers: 81920
    p64_push_duration_us: 30640.0
    p64_push_sync_duration_us: 41867.0
    p64_push_sync_throughput: 1956672.3
    p64_push_throughput: 2673629.2
    p64_warmup_triggers: 16384
    p8_num_triggers: 10240
    p8_push_duration_us: 3304.0
    p8_push_sync_duration_us: 9124.0
    p8_push_sync_throughput: 1122314.7
    p8_push_throughput: 3099273.6
    p8_warmup_triggers: 2048

Test: FifoTest_Size512_ParallelCustom (fifo)
  Metrics:
    p1_num_triggers: 5120
    p1_push_duration_us: 6910.0
    p1_push_sync_duration_us: 13893.0
    p1_push_sync_throughput: 368530.9
    p1_push_throughput: 740955.1
    p1_warmup_triggers: 1024
    p256_num_triggers: 1310720
    p256_push_duration_us: 441824.0
    p256_push_sync_duration_us: 955608.0
    p256_push_sync_throughput: 1371608.4
    p256_push_throughput: 2966611.1
    p256_warmup_triggers: 262144
    p512_num_triggers: 2621440
    p512_push_duration_us: 1973937.0
    p512_push_sync_duration_us: 2005066.0
    p512_push_sync_throughput: 1307408.3
    p512_push_throughput: 1328026.1
    p512_warmup_triggers: 524288
    p64_num_triggers: 327680
    p64_push_duration_us: 110470.0
    p64_push_sync_duration_us: 165751.0
    p64_push_sync_throughput: 1976941.3
    p64_push_throughput: 2966235.1
    p64_warmup_triggers: 65536
    p8_num_triggers: 40960
    p8_push_duration_us: 13286.0
    p8_push_sync_duration_us: 36631.0
    p8_push_sync_throughput: 1118178.5
    p8_push_throughput: 3082944.4
    p8_warmup_triggers: 8192
```


# 


```
struct Fifo::Impl {
  detail::UniqueGpuHostPtr<ProxyTrigger> triggers;
  detail::UniqueGpuPtr<uint64_t> head;
  detail::UniqueGpuHostPtr<uint64_t> tail;
  detail::UniqueGpuPtr<uint64_t> tailCache;
  const int size;

  Impl(int size)
      : triggers(detail::gpuCallocHostUnique<ProxyTrigger>(size)),
        head(detail::gpuCallocUnique<uint64_t>()),
        tail(detail::gpuCallocHostUnique<uint64_t>()),
        tailCache(detail::gpuCallocUnique<uint64_t>()),
        size(size) {}
};
```
+  gpuCalloc gpuCallocHost也就是cudaMalloc、cudaHostAlloc

```
        // 有 GPU：使用 pinned memory（CPU-GPU 共享内存）
        // cudaHostAllocMapped 让 CPU 和 GPU 可以访问同一块内存
        cudaHostAlloc(&tasks_, kNumTasks_ * sizeof(Task), cudaHostAllocMapped);
        cudaHostGetDevicePointer(&tasks_device_, tasks_, 0);
		cudaHostAlloc((void**)&doorbell, sizeof(int), cudaHostAllocMapped));
```

```
template <class T>
auto gpuCallocHostUnique(size_t nelems = 1, unsigned int flags = cudaHostAllocMapped) {
  return detail::safeAlloc<T, detail::GpuHostDeleter<T>, UniqueGpuHostPtr<T>>(detail::gpuCallocHost, nelems, flags);
}

template <class T>
auto gpuCallocUnique(size_t nelems = 1) {
  return detail::safeAlloc<T, detail::GpuDeleter<T>, UniqueGpuPtr<T>>(detail::gpuCalloc, nelems);
}

```


```
void* gpuCalloc(size_t bytes) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  void* ptr;
  auto stream = gpuStreamPool()->getStream();
  MSCCLPP_CUDATHROW(cudaMalloc(&ptr, bytes));
  MSCCLPP_CUDATHROW(cudaMemsetAsync(ptr, 0, bytes, stream));
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
  return ptr;
}

void* gpuCallocHost(size_t bytes, unsigned int flags) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  void* ptr;
  MSCCLPP_CUDATHROW(cudaHostAlloc(&ptr, bytes, flags));
  ::memset(ptr, 0, bytes);
  return ptr;
}

```


```
cudaHostAlloc st.global.release.sys.v2.u64
```