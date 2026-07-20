
#  ncclShmOpen

```
(gdb) b cudaHostRegister
Breakpoint 3 at 0x7ffff7ec78a0: file graph/fake_cuda.cc, line 179.
(gdb) b cudaHostGetDevicePointer
Breakpoint 4 at 0x7ffff7ec7d80: file graph/fake_cuda.cc, line 514.
(gdb) 
```
共享内存

```
__shared__ ncclShmemData ncclShmem;
```

```
struct ncclShmemData {
  struct ncclDevKernelArgs args;
  int channelId;
  int aborted;
  alignas(16) struct ncclDevComm comm;
  alignas(16) struct ncclDevChannel channel;

  int batchIx, nextBatchIx;
  enum ncclDevWorkType workType;
  uint8_t directMode;
  uint16_t funcId;
  int nWorks;
  int workSize;
  uint64_t workCounter;
  bool profilerEnabled;
  struct ncclShmemGroup groups[NCCL_MAX_GROUPS];
  uint64_t redOpArgs[NCCL_MAX_NVLS_ARITY+1];

  alignas(16) char workStorage[1024];

  alignas(16) union {
    unpackShmem unpack;
  } devicePlugin;
};

extern __shared__ ncclShmemData ncclShmem;
#if __CUDA_ARCH__ >= 700
  extern __shared__ ulong2 ncclShmemPerWarp[/*ncclShmemDynamicSize()/sizeof(ulong2)*/];
#else
  extern __shared__ ulong2 ncclShmemPerWarp[ncclShmemScratchWarpSize()*(NCCL_MAX_NTHREADS/WARP_SIZE)/sizeof(ulong2)];
#endif
```



```
(gdb) bt
#0  ncclShmOpen (shmPath=0x7fffde7f96a0 "", shmPathSize=128, shmSize=9637888, shmPtr=0x7fffb0004880, devShmPtr=0x7fffb0004888, refcount=1, handle=0x7fffb0004a60) at misc/shmutils.cc:48
#1  0x00007ffff7f25153 in ncclShmAllocateShareableBuffer (size=9637888, legacy=128, desc=0x7fffb0004a58, hptr=0x7fffb0004880, dptr=dptr@entry=0x7fffb0004888) at transport/shm.cc:557
#2  0x00007ffff7f25337 in shmRecvProxySetup (proxyState=<optimized out>, reqSize=16, respSize=112, done=0x7fffde7f97cc, respBuff=0x7fffb0004820, reqBuff=0x7fffb00049c0, 
    connection=0x7fffb0000c88) at transport/shm.cc:491
#3  shmRecvProxySetup (connection=0x7fffb0000c88, proxyState=<optimized out>, reqBuff=0x7fffb00049c0, reqSize=<optimized out>, respBuff=0x7fffb0004820, respSize=<optimized out>, 
    done=0x7fffde7f97cc) at transport/shm.cc:480
#4  0x00007ffff7ebda1f in proxyProgressAsync (op=0x7fffb0000b70, proxyState=proxyState@entry=0x7fffe0001cf0, asyncOpCount=asyncOpCount@entry=0x7fffde7f98b4, peer=peer@entry=0x7fffde7f9b20, 
    connectionPool=connectionPool@entry=0x7fffde7f98c0) at proxy.cc:1458
#5  0x00007ffff7ebf31e in proxyServiceInitOp (asyncOpCount=0x7fffde7f98b4, proxyState=<optimized out>, connectionPool=0x7fffde7f98c0, peer=0x7fffde7f9b20, type=<optimized out>)
    at proxy.cc:1535
#6  ncclProxyService (_args=0x7fffe0001cf0) at proxy.cc:1688
#7  0x00007ffff7c8eb43 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:442
#8  0x00007ffff7d20a00 in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:81
(gdb) 
```

# gpu 加载cpu侧数据


```
// Exit If Abort Barrier across CTA: make sure all threads exit consistently
// Each thread sets a predicate to true if abort == 1
// all CTA's threads enter the barrier and do a popc on their predicates being True
// If any of the thread's predicate was True, all the threads call exit()
static inline __device__ void exitIfAbortBarrier(int abort) {
  uint32_t popc;
  asm ("{");
  asm volatile ("   .reg .pred barr_pred;");
  asm volatile ("   setp.eq.u32 barr_pred,%0,1;" :: "r"(abort));
  asm volatile ("   bar.red.popc.u32 %0, 0, barr_pred;" : "=r"(popc));
  asm ("}");
  if (popc) { asm volatile ("exit;"); }
}

typedef void(*ncclKern_t)(struct ncclWorkElem* args);
extern __device__ ncclKern_t ncclFuncs[];

static __device__ void load_parallel(void* dst, void* src, size_t size, int tid) {
  int* d = (int*)dst;
  int* s = (int*)src;
  for (int o = tid; o < (size/sizeof(int)); o += blockDim.x) d[o] = s[o];
}
static __device__ void load_coll(struct ncclWork* localWork, struct ncclWork* hostWork, int tid, struct ncclDevComm* comm) {
  __syncthreads();
  load_parallel(localWork, hostWork, sizeof(struct ncclWork), tid);
  // Check whether the last operation was aborted and make sure all threads exit
  int abort = tid == 0 ? *(comm->abortFlag) : 0;
  exitIfAbortBarrier(abort);
  if (tid == 0) hostWork->elems[0].active = 0;
}
```


# 网络通信

```
directSend（Send）
directRecvReduceDirectSend（RecvReduceSend）
directRecvReduceCopyDirectSend（RecvReduceCopySend）
directRecvCopyDirectSend（RecvCopySend）
directRecv（Recv）
recvReduceSend
recvReduceCopy
```


> ## 自适应轮询

```
// 源码: src/device/prims_simple.h, line 108-120

template <int DirectRecv, int DirectSend, int Recv, int Send, int Src, int Dst>
__device__ __forceinline__ void waitPeer(
    intptr_t srcIx, intptr_t dstIx, int offset, int nelts) {

  const bool isSendNotRecv = (Send && Recv) 
    ? (flags & RoleWaitSend) 
    : Send;

  if ((flags & (Recv * RoleWaitRecv)) || 
      (flags & (Send * RoleWaitSend))) {

    int spins = 0;
    // 自旋等待直到对端完成
    while (connStepCache + (isSendNotRecv ? NCCL_STEPS : 0) 
           < step + StepPerSlice) {
      connStepCache = loadStepValue(connStepPtr);

      // 检查中止标志 (避免死锁)
      if (checkAbort(flags, Aborted, spins)) break;
    }
  }

  // 更新FIFO大小 (用于流控)
  if ((flags & ConnFifoEnabled) && (flags & (Send * RoleWaitSend)))
    connFifo[step % NCCL_STEPS].size = nelts * sizeof(T);
}
```


> ## 共享内存

[第76篇 - NCCL SHM（共享内存）传输层深度分析](https://zhuanlan.zhihu.com/p/1982220126754461512)   


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

+  conn.connFifo

```
 struct ncclSendMem *sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, sendMem);
  void* gdcMem = map->mems[NCCL_NET_MAP_GDCMEM].gpuPtr;
  send->conn.head = gdcMem ? (uint64_t*)gdcMem : &sendMem->head;

  struct ncclRecvMem *recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, recvMem);
  send->conn.tail = &recvMem->tail;
  send->conn.stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;
  send->conn.connFifo = recvMem->connFifo;
  // Only fuse P2P buffers, continue to allocate dedicated buffers for ring/tree
  for (int i=0; i<NCCL_STEPS; i++) {
    send->conn.connFifo[i].offset = -1;
    recvMem->connFifo[i].mode = map->shared ? NCCL_MODE_OFFSET : NCCL_MODE_NORMAL;
  }
```


+ 同步

```
enqueue.cc:1477:      CUDACHECKGOTO(cudaMemcpyAsync(fifoBufDev, fifoBufHost, workBytes, cudaMemcpyDefault, deviceStream), result, fail);
enqueue.cc.bak:1383:      CUDACHECKGOTO(cudaMemcpyAsync(fifoBufDev, fifoBufHost, workBytes, cudaMemcpyDefault, deviceStream), result, fail);
```


#  GPU_COLLECT_EVENT