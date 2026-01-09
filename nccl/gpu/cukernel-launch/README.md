

```
TORCH_CUDA_LIB=`python3 -c 'from torch.utils import cpp_extension; print(f"{cpp_extension.TORCH_LIB_PATH}/libtorch_cuda.so")' 2>/dev/null`
```

```
whereis libnccl
libnccl: /usr/lib/x86_64-linux-gnu/libnccl.so
```

```
nm -DC /usr/lib/x86_64-linux-gnu/libnccl.so |  grep nccl
                 w TLS init function for ncclDebugNoWarn
0000000000093910 T ncclAllGather
00000000000935d0 T ncclAllReduce
```

# 参考


[第17篇 - 集合通信 - NCCL对国产AI芯片设计要求深度分析](https://zhuanlan.zhihu.com/p/1971339020949775717)   


# test1 (cuLaunchKernel kernel.ptx)
```
./kernel_test 
PTX kernel execution time: 0.025568 ms
Runtime kernel execution time: 0.02016 ms
Both kernels produced correct results.
```

# test2
```
root@ubuntu:/pytorch/kernel/test2# ./kernel_test 
Result C (first 10 elements): 0.0 10.0 320.0 30.0 10240.0 50.0 15360.0 70.0 20480.0 90.0
```


#  test3

+ 从libvec.a中cudaGetFuncBySymbol    


```
cudaErr = cudaGetFuncBySymbol(&cuFn, (void*)VecAdd2);
```

```
test3# ./kernel_test 
get nccl kernel function 
Result C (first 10 elements): 0.0 10.0 320.0 30.0 10240.0 50.0 15360.0 70.0 20480.0 90.0 
```


+ 从main2.cu中直接cudaGetFuncBySymbol     


```
 cudaErr = cudaGetFuncBySymbol(&cuFn, (void*)VecAdd3);
```


```
test3# ./kernel_test2
get nccl kernel function 
Result C (first 10 elements): 0.0 10.0 320.0 30.0 10240.0 50.0 15360.0 70.0 20480.0 90.0 
```

+ 从libvec.a中cudaGetFuncBySymbol   

```
 cudaErr = cudaGetFuncBySymbol(&cuFn, (void*)invoke_VecAdd);
```

```
test3# ./kernel_test3 
get nccl kernel function 
Result C (first 10 elements): 0.0 10.0 320.0 30.0 10240.0 50.0 15360.0 70.0 20480.0 90.0 
```

#  ipc

```
/pytorch/kernel/gpu_mmap# ./gpu_mmap 
total sum [master]: 6426637495.977594
total sum [9044]: 6426637495.977592
total sum [9043]: 6426637495.977592
```

参考ncclCuMemHostAlloc
```
static inline ncclResult_t ncclCuMemHostAlloc(void** ptr, CUmemGenericAllocationHandle *handlep, size_t size) {
  ncclResult_t result = ncclSuccess;
  size_t granularity = 0;
  CUdevice currentDev;
  CUmemAllocationProp prop = {};
  CUmemAccessDesc accessDesc = {};
  CUmemGenericAllocationHandle handle;
  int cudaDev;
  int cpuNumaNodeId = -1;
  CUmemAllocationHandleType type = ncclCuMemHandleType;

  CUDACHECK(cudaGetDevice(&cudaDev));
  CUCHECK(cuDeviceGet(&currentDev, cudaDev));
  CUCHECK(cuDeviceGetAttribute(&cpuNumaNodeId, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, currentDev));
  if (cpuNumaNodeId < 0) cpuNumaNodeId = 0;
  prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.requestedHandleTypes = type; // So it can be exported
  prop.location.id = cpuNumaNodeId;
  CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  ALIGN_SIZE(size, granularity);
  /* Allocate the physical memory on the device */
  CUCHECK(cuMemCreate(&handle, size, &prop, 0));
  /* Reserve a virtual address range */
  CUCHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, granularity, 0, 0));
  /* Map the virtual address range to the physical allocation */
  CUCHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
  /* Now allow RW access to the newly mapped memory for local GPU */
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cudaDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));

  /* Now allow RW access to the newly mapped memory from the CPU */
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  accessDesc.location.id = cpuNumaNodeId;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));

  if (handlep) *handlep = handle;
  INFO(NCCL_ALLOC, "CUMEM Host Alloc Size %zi pointer %p handle %llx numa %d dev %d granularity %ld", size, *ptr, handle, cpuNumaNodeId, cudaDev, granularity);
  return result;
}

```


((&map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr, map->mems[NCCL_NET_MAP_HOSTMEM].size)


```
template <typename T>
ncclResult_t ncclCudaHostCallocDebug(T** ptr, size_t nelem, const char *filefunc, int line) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  *ptr = nullptr;
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (nelem > 0) {
    CUDACHECKGOTO(cudaHostAlloc(ptr, nelem*ncclSizeOfT<T>(), cudaHostAllocMapped), result, finish);
    memset(*ptr, 0, nelem*ncclSizeOfT<T>());
  }
finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (*ptr == nullptr && nelem > 0) WARN("Failed to CUDA host alloc %ld bytes", nelem*ncclSizeOfT<T>());
  INFO(NCCL_ALLOC, "%s:%d Cuda Host Alloc Size %ld pointer %p", filefunc, line, nelem*ncclSizeOfT<T>(), *ptr);
  return result;
}
```



# make bug


+  ptxas error   : Undefined reference to 'VecAdd2' in '<input>'    

```
typedef void (*fp)(double *, double *, double *, int);
extern "C" __global__ void VecAdd2(double *a, double *b, double *c, int n);
__device__ fp kernelPtr = VecAdd2;
//__device__ fp kernelPtr = VecAdd2;
```


```
ptxas error   : Undefined reference to 'VecAdd2' in '<input>'
```
加上 __global__ 关键字    




+ undefined reference to `invoke_VecAdd(double*, double*, double*, int)'   


invoke_VecAdd前面加上extern "C" __global__ 

