


# gdr
```
static ncclResult_t ncclGpuGdrSupport(int* gdrSupport) {
  int netDevs;
  NCCLCHECK(ncclNetDevices(&netDevs));
  *gdrSupport = 0;
  for (int dev=0; dev<netDevs; dev++) {
    // Find a net device which is GDR-capable
    ncclNetProperties_t props;
    NCCLCHECK(ncclNet->getProperties(dev, &props));
    if ((props.ptrSupport & NCCL_PTR_CUDA) == 0) continue;
 
    // Allocate memory on the GPU and try to register it on the NIC.
    void *lComm = NULL, *sComm = NULL, *rComm = NULL;
    ncclNetHandle_t handle;
    void* gpuPtr = NULL;
    void* mHandle = NULL;
    NCCLCHECK(ncclNetListen(dev, &handle, &lComm));
    NCCLCHECK(ncclNetConnect(dev, &handle, &sComm));
    NCCLCHECK(ncclNetAccept(lComm, &rComm));
    CUDACHECK(cudaMalloc(&gpuPtr, GPU_BUF_SIZE));
    ncclDebugNoWarn = NCCL_NET;
    if (ncclNetRegMr(sComm, gpuPtr, GPU_BUF_SIZE, NCCL_PTR_CUDA, &mHandle) == ncclSuccess) {
      NCCLCHECK(ncclNetDeregMr(sComm, mHandle));
      NCCLCHECK(ncclNetRegMr(rComm, gpuPtr, GPU_BUF_SIZE, NCCL_PTR_CUDA, &mHandle));
      NCCLCHECK(ncclNetDeregMr(rComm, mHandle));
      *gdrSupport = 1;
    }
    ncclDebugNoWarn = 0;
    CUDACHECK(cudaFree(gpuPtr));
    NCCLCHECK(ncclNetCloseRecv(rComm));
    NCCLCHECK(ncclNetCloseSend(sComm));
    NCCLCHECK(ncclNetCloseListen(lComm));
    break;
  }
  return ncclSuccess;
}
```