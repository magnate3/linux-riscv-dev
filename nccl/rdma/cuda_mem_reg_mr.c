/*
 * SPDX-License-Identifier: Apache-2.0
 * Copyright(c) 2024 Liu, Changcheng <changcheng.liu@aliyun.com>
 */

#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <infiniband/verbs.h>

#include <cuda_runtime.h>
#include <cuda.h>

/*Build:
 * $ nvcc cuda_mem_reg_mr.c -o reg_mr -lcuda -libverbs
 *
 *Run(use mlx5_2 & GPU0 for example)
 * $ ./reg_mr mlx5_2 0
 *
 *Topology is below:
 *
 *         GPU0    GPU1    GPU2    GPU3    NIC0    NIC1    NIC2    NIC3    CPU Affinity    NUMA Affinity   GPU NUMA ID
 * GPU0     X      NODE    SYS     SYS     NODE    NODE    NODE    SYS     0-31,64-95      0               N/A
 * GPU1    NODE     X      SYS     SYS     NODE    NODE    NODE    SYS     0-31,64-95      0               N/A
 * GPU2    SYS     SYS      X      NODE    SYS     SYS     SYS     NODE    32-63,96-127    1               N/A
 * GPU3    SYS     SYS     NODE     X      SYS     SYS     SYS     NODE    32-63,96-127    1               N/A
 * NIC0    NODE    NODE    SYS     SYS      X      PIX     NODE    SYS
 * NIC1    NODE    NODE    SYS     SYS     PIX      X      NODE    SYS
 * NIC2    NODE    NODE    SYS     SYS     NODE    NODE     X      SYS
 * NIC3    SYS     SYS     NODE    NODE    SYS     SYS     SYS      X
 *
 * Legend:
 *
 *   X    = Self
 *   SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
 *   NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
 *   PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
 *   PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
 *   PIX  = Connection traversing at most a single PCIe bridge
 *   NV#  = Connection traversing a bonded set of # NVLinks
 *
 * NIC Legend:
 *
 *   NIC0: mlx5_0
 *   NIC1: mlx5_1
 *   NIC2: mlx5_2
 *   NIC3: mlx5_3
 */

#define CUDACHECK(cmd) do {                           \
    cudaError_t e = cmd;                              \
    if( e != cudaSuccess ) {                          \
        printf("Failed: Cuda error %s:%d '%s'\n",     \
            __FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                           \
    }                                                 \
} while(0)

#define CUCHECK(cmd) do {                                      \
    CUresult err = cmd;                                        \
    if (err != CUDA_SUCCESS) {                                 \
        const char *errStr;                                    \
        (void) cuGetErrorString(err, &errStr);                 \
        fprintf(stderr, "CUDA Driver error in %s:%d: %d:%s\n", \
                __FILE__, __LINE__, err, errStr);              \
        exit(EXIT_FAILURE);                                    \
    }                                                          \
} while(0)

#define ALIGN_SIZE(size, align)                                \
    size = ((size + (align) - 1) / (align)) * (align)

#define GPU_PAGE_SHIFT   16
#define GPU_PAGE_SIZE    (1UL << GPU_PAGE_SHIFT)

int getMultiCastCap(int deviceIdx)
{
    CUdevice currentDev;
    CUCHECK(cuDeviceGet(&currentDev, deviceIdx));

    int mcSupport = 0;
    CUCHECK(cuDeviceGetAttribute(&mcSupport, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, currentDev));

    return mcSupport;
}

int getDirectRDMACap(int deviceIdx)
{
    CUdevice currentDev;
    CUCHECK(cuDeviceGet(&currentDev, deviceIdx));

    int gdrRDMACap = 0;
    CUCHECK(cuDeviceGetAttribute(&gdrRDMACap, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, currentDev));

    return gdrRDMACap;
}

size_t getMemAllocationGranularity(int deviceIdx, int checkGranularityType)
{
    CUdevice currentDev;
    CUCHECK(cuDeviceGet(&currentDev, deviceIdx));

    CUmemAllocationProp memprop = {};

    memprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    memprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    memprop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    memprop.location.id = currentDev;
    memprop.allocFlags.gpuDirectRDMACapable = getDirectRDMACap(deviceIdx);

    size_t memAllocationGran = 0;
    CUCHECK(cuMemGetAllocationGranularity(&memAllocationGran, &memprop, checkGranularityType));

    return memAllocationGran;
}

size_t getMulticastMemAllocationRecommandGranularity(size_t size)
{
    int deviceCnt = 0;
    CUDACHECK(cudaGetDeviceCount(&deviceCnt));

    CUmulticastObjectProp mcprop = {};
    mcprop.size = size;
    mcprop.numDevices = deviceCnt;
    mcprop.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    mcprop.flags = 0;

    size_t memMultiCastAllocationGran = 0;
    CUCHECK(cuMulticastGetGranularity(&memMultiCastAllocationGran, &mcprop, CU_MULTICAST_GRANULARITY_RECOMMENDED));

    return memMultiCastAllocationGran;
}

void showRunTimeCudaDeviceInfo(int deviceIdx)
{
    struct cudaDeviceProp deviceProp;
    CUDACHECK(cudaGetDeviceProperties(&deviceProp, deviceIdx));

    printf("devices[%d]:\n", deviceIdx);
    printf("    Name: %s\n", deviceProp.name);
    printf("    Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("    Total Global Memory: %.2f GB\n", (float)deviceProp.totalGlobalMem / (1024 * 1024 * 1024));
    printf("    Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("    Multiprocessor Count: %d\n", deviceProp.multiProcessorCount);
    printf("    Clock Rate: %.2f GHz\n", deviceProp.clockRate * 1e-6f);
    printf("    Memory Bus Width: %d-bit\n", deviceProp.memoryBusWidth);
    printf("    L2 Cache Size: %d KB\n", deviceProp.l2CacheSize / 1024);
    printf("    Max Threads Dimensions: (%d, %d, %d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("    Max Grid Size: (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf("    Direct RDMA Capability: %d\n", getDirectRDMACap(deviceIdx));
    printf("    MemAllocationGranularity Min: %zu\n", getMemAllocationGranularity(deviceIdx, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    printf("    MemAllocationGranularity Min: %zu\n", getMemAllocationGranularity(deviceIdx, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

    int multiCastSupport = getMultiCastCap(deviceIdx);
    printf("    Multi Cast Capability: %d\n", multiCastSupport);

    printf("\n");
}

void reqRDMACapableMem(int deviceIdx, void **ptr, size_t size)
{
    int deviceCnt;
    CUDACHECK(cudaGetDeviceCount(&deviceCnt));

    CUdevice currentDev;
    CUCHECK(cuDeviceGet(&currentDev, deviceIdx));

    CUmemAllocationProp memprop = {};
    memprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    memprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    memprop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    memprop.location.id = currentDev;

    int gdrCap = getDirectRDMACap(deviceIdx);
    memprop.allocFlags.gpuDirectRDMACapable = !!gdrCap;

    size_t memAllocationGran = 0;
    CUCHECK(cuMemGetAllocationGranularity(&memAllocationGran, &memprop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

    /* If the device support Multicast, it should meet with below requirement:
     * GPU_PAGE_SIZE >= getMemAllocationGranularity(deviceIdx, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
     */
    ALIGN_SIZE(size, memAllocationGran);
    ALIGN_SIZE(size, GPU_PAGE_SIZE);

    CUmemGenericAllocationHandle handle;
    CUCHECK(cuMemCreate(&handle, size, &memprop, 0));

    CUCHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, memAllocationGran, 0, 0));

    CUCHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
    printf("%s:%d allocated size: 0x%016lx\n", __func__, __LINE__, size);

    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = deviceIdx;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));
}

void reqMultiCastAllocate(int deviceIdx, void **ptr, size_t size)
{
    int deviceCnt;
    CUDACHECK(cudaGetDeviceCount(&deviceCnt));

    CUdevice currentDev;
    CUCHECK(cuDeviceGet(&currentDev, deviceIdx));

    CUmemAllocationProp memprop = {};
    memprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    memprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    memprop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    memprop.location.id = currentDev;

    int gdrCap = getDirectRDMACap(deviceIdx);
    memprop.allocFlags.gpuDirectRDMACapable = !!gdrCap;

    size_t memAllocationGran = 0;
    CUCHECK(cuMemGetAllocationGranularity(&memAllocationGran, &memprop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

    size_t memMultiCastAllocationGran = getMulticastMemAllocationRecommandGranularity(size);
    ALIGN_SIZE(size, memMultiCastAllocationGran);

    CUmemGenericAllocationHandle handle;
    CUCHECK(cuMemCreate(&handle, size, &memprop, 0));

    CUCHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, memAllocationGran, 0, 0));

    CUCHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
    printf("%s:%d allocated size: 0x%016lx\n", __func__, __LINE__, size);

    for (int idx = 0; idx < deviceCnt; idx++) {
        int p2p_access_support = 0;
        if (idx == deviceIdx || ((cudaDeviceCanAccessPeer(&p2p_access_support, deviceIdx, idx) == cudaSuccess) && p2p_access_support)) {
            CUmemAccessDesc accessDesc = {};
            accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            accessDesc.location.id = idx;
            accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));
        }
    }
}

void freeAllocatMem(void* ptr)
{
    CUmemGenericAllocationHandle handle;
    size_t size = 0;

    CUCHECK(cuMemRetainAllocationHandle(&handle, ptr));
    CUCHECK(cuMemRelease(handle));
    CUCHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));
    CUCHECK(cuMemUnmap((CUdeviceptr)ptr, size));
    CUCHECK(cuMemRelease(handle));
    CUCHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
}

struct ibv_context* getDevCtx(const char *devName)
{
    int devCnt = 0;
    struct ibv_device ** deviceList = ibv_get_device_list(&devCnt);
    struct ibv_device *device;

    for (int idx = 0; idx < devCnt; idx++) {
        device = deviceList[idx];
        if (strcmp(devName, ibv_get_device_name(device)) == 0) {
            break;
        }
    }

    struct ibv_context *devCtx = ibv_open_device(device);
    ibv_free_device_list(deviceList);

    return devCtx;
}

int main(int argc, char* argv[])
{
    char *rnicDevName = strdup(argv[1]);
    int cfg_gpu_idx = atoi(argv[2]);

    int send_size = 64 * 1024; //64KB
    int recv_size = 2 * send_size; // suppose 2 ranks in allgather operation

    CUCHECK(cuInit(0));

    // Get the number of CUDA devices
    int deviceCount;
    CUDACHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }
    printf("Found %d CUDA devices:\n\n", deviceCount);

    showRunTimeCudaDeviceInfo(cfg_gpu_idx);

    // pick the GPU dev based on cfg_gpu_idx
    CUDACHECK(cudaSetDevice(cfg_gpu_idx));
    int multiCastSupport = getMultiCastCap(cfg_gpu_idx);
    int directRDMASupport = getDirectRDMACap(cfg_gpu_idx);

    uint32_t *sendbuff, *recvbuff;
    printf("req send_size: 0x%08x, recv_size: 0x%08x\n", send_size, recv_size);
    if (multiCastSupport) {
        uint32_t *hostData = calloc(recv_size / sizeof(uint32_t), sizeof(uint32_t));
        memset(hostData, 0, recv_size);

        reqMultiCastAllocate(cfg_gpu_idx, (void**)&sendbuff, send_size);
        CUDACHECK(cuMemcpyHtoD((CUdeviceptr)sendbuff, hostData, send_size));

        reqMultiCastAllocate(cfg_gpu_idx, (void**)&recvbuff, recv_size);
        CUDACHECK(cuMemcpyHtoD((CUdeviceptr)recvbuff, hostData, recv_size));

        free(hostData);
    } else if(directRDMASupport) {
        uint32_t *hostData = calloc(recv_size / sizeof(uint32_t), sizeof(uint32_t));
        memset(hostData, 0, recv_size);

        reqRDMACapableMem(cfg_gpu_idx, (void**)&sendbuff, send_size);
        CUDACHECK(cuMemcpyHtoD((CUdeviceptr)sendbuff, hostData, send_size));

        reqRDMACapableMem(cfg_gpu_idx, (void**)&recvbuff, recv_size);
        CUDACHECK(cuMemcpyHtoD((CUdeviceptr)recvbuff, hostData, recv_size));

        free(hostData);
    } else {
        CUDACHECK(cudaMalloc((void**)(&sendbuff), send_size));
        CUDACHECK(cudaMemset(sendbuff, 0, send_size));

        CUDACHECK(cudaMalloc((void**)&recvbuff, recv_size));
        CUDACHECK(cudaMemset(recvbuff, 0, recv_size));
    }

    struct ibv_context *devCtx = getDevCtx(rnicDevName);
    struct ibv_pd *pd = ibv_alloc_pd(devCtx);
    unsigned int access_right = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_RELAXED_ORDERING;

    // Why using recv_size instead of send_size?
    struct ibv_mr *mr = ibv_reg_mr_iova2(pd, sendbuff, recv_size, (uint64_t)sendbuff, access_right);

    if (mr == NULL) {
        printf("ibv_reg_mr_iova2 failed with error=%d %s on dev:%s at addr:%p, "
         "length:0x%08x, iova:0x%016lx, access:0x%08x\n", errno,
         strerror(errno), ibv_get_device_name(pd->context->device),
         sendbuff, recv_size, (uint64_t)sendbuff, access_right);
    } else {
       if (ibv_dereg_mr(mr)) {
           printf("ibv_dereg_mr failed\n");
       } else {
           printf("passed test\n");
       }
    }

    ibv_dealloc_pd(pd);
    ibv_close_device(devCtx);

    if (multiCastSupport || directRDMASupport) {
        freeAllocatMem(sendbuff);
        freeAllocatMem(recvbuff);
    } else {
        CUDACHECK(cudaFree(sendbuff));
        CUDACHECK(cudaFree(recvbuff));
    }

    free(rnicDevName);

    return 0;
}
