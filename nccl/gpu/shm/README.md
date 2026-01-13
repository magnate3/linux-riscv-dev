


[第76篇 - NCCL SHM（共享内存）传输层深度分析](https://zhuanlan.zhihu.com/p/1982220126754461512)  

[第103篇 - NCCL Register 机制深度分析](https://zhuanlan.zhihu.com/p/1986077323959751117)   

```
 NCCLCHECK(netMapShm(comm, &send->proxyConn, map->mems + NCCL_NET_MAP_HOSTMEM))
 
static ncclResult_t netMapShm(struct ncclComm *comm, struct ncclProxyConnector* proxyConn, struct connectMapMem* mem) {
  NCCLCHECK(ncclShmImportShareableBuffer(comm, proxyConn->rank, &mem->createDesc, (void**)&mem->cpuPtr, (void**)&mem->gpuPtr, &mem->attachDesc));
  return ncclSuccess;
}
```

```
ncclResult_t ncclShmAllocateShareableBuffer(size_t size, bool legacy, ncclShmIpcDesc_t *desc, void **hptr, void **dptr) {
  if (desc == NULL || hptr == NULL) {
    WARN("Invalid argument desc %p, hptr %p", desc, hptr);
    return ncclInvalidArgument;
  }
#if CUDART_VERSION >= 12020
  if (ncclCuMemEnable() && ncclCuMemHostEnable() && !legacy) {
    // cuMem API support
    CUmemAllocationHandleType type = SHM_HANDLE_TYPE;
    CUmemGenericAllocationHandle handle;

    NCCLCHECK(ncclCuMemHostAlloc(hptr, &handle, size));
    if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
      // Return the native cuMem handle for later Export/Import via UDS
      memcpy(&desc->shmci.data, &handle, sizeof(handle));
    } else {
      CUCHECK(cuMemExportToShareableHandle(&desc->shmci.handle, handle, type, 0));
    }
    desc->shmci.size = size;
    desc->shmci.ptr = *hptr;
    if (dptr) *dptr = *hptr;
    desc->legacy = false;
    INFO(NCCL_SHM, "CUMEM allocated shareable buffer %p size %zi", desc->shmci.ptr, desc->shmci.size);
  } else {
    char shmPath[SHM_PATH_MAX] = { '\0' };
    desc->shmli.shmSize = size;
    NCCLCHECK(ncclShmOpen(shmPath, sizeof(shmPath), size, hptr, dptr, 1, &desc->shmli.handle));
    memcpy(desc->shmli.shmSuffix, shmPath + sizeof("/dev/shm/nccl-") - 1, sizeof(desc->shmli.shmSuffix));
    desc->legacy = true;
    INFO(NCCL_SHM, "MMAP allocated shareable host buffer %s size %zi ptr %p", shmPath, desc->shmli.shmSize, *hptr);
  }
#else /* CUDART_VERSION >= 12020 */
  char shmPath[SHM_PATH_MAX] = { '\0' };
  desc->shmli.shmSize = size;
  NCCLCHECK(ncclShmOpen(shmPath, sizeof(shmPath), size, hptr, dptr, 1, &desc->shmli.handle));
  memcpy(desc->shmli.shmSuffix, shmPath + sizeof("/dev/shm/nccl-") - 1, sizeof(desc->shmli.shmSuffix));
  desc->legacy = true;
  INFO(NCCL_SHM, "MMAP allocated shareable host buffer %s size %zi ptr %p", shmPath, size, *hptr);
#endif /* CUDART_VERSION >= 12020 */
  return ncclSuccess;
}
```

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


```
tcpxResult_t __tcpxNetDeviceQueueNew(void* gpu_dev, bool passive, void** handle, void** d_handle) {
  TCPXCHECK(gpu_push_current(gpu_dev));

  struct tcpxNetDeviceQueue* h;
  struct unpackNetDeviceHandle* d;
  INFO(TCPX_NET, "NetDeviceHandle size %zu", sizeof *h);
  INFO(TCPX_NET, "NetDeviceDevHandle size %zu", sizeof *d);

  // clang-format off

  // host side handle
  CUASSERT(cuMemHostAlloc((void**) &h, sizeof *h, 0));
  memset(h, 0, sizeof *h);
  CUASSERT(cuMemHostAlloc((void**) &(h->meta), sizeof *(h->meta), 
                              CU_MEMHOSTALLOC_DEVICEMAP
                            | CU_MEMHOSTALLOC_PORTABLE));
                            // | CU_MEMHOSTALLOC_WRITECOMBINED));
  h->gpu_dev = gpu_dev;
  h->head = h->tail = 0;

  INFO(TCPX_NET, "handle %p size %zu", h, sizeof *h);
  INFO(TCPX_NET, "h->meta %p size %zu", h->meta, sizeof *(h->meta));

  // cuda side handle
  CUASSERT(cuMemAlloc((CUdeviceptr*) &d, sizeof *d));
  struct unpackNetDeviceHandle h_d;
  CUASSERT(cuMemHostGetDevicePointer((CUdeviceptr*) &(h_d.meta), h->meta, 0));

  if (passive) {
    TCPXASSERT(gpu_get_rxmem(gpu_dev, &(h_d.bounce_buf)));
  }

  // initialize nccl side head, nccl side increments the counter prior to performing copy
  h_d.head = (uint64_t) -1;

  CUASSERT(cuMemcpyHtoD((CUdeviceptr) d, &h_d, sizeof h_d));

  TCPXCHECK(gpu_pop_current(nullptr, nullptr));  // we don't care about output

  *handle = h;
  *d_handle = d;

  return tcpxSuccess;
}
```