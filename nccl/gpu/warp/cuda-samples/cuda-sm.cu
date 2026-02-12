#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        fprintf(stderr, "Error getting device count: %s\n", cudaGetErrorString(error));
        return -1;
    }
    printf("Number of CUDA devices: %d\n", deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %zu bytes\n", prop.totalGlobalMem);
        printf("  Multiprocessor count: %d\n", prop.multiProcessorCount);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
        printf("  Warp size: %d\n", prop.warpSize);
        printf("  Memory clock rate: %d kHz\n", prop.memoryClockRate);
        printf("  Memory bus width: %d bits\n", prop.memoryBusWidth);
        printf("  L2 cache size: %zu bytes\n", prop.l2CacheSize);
        printf("  Max texture dimensions: %d x %d x %d\n",
               prop.maxTexture1D, prop.maxTexture2D[0], prop.maxTexture2D[1]);
        printf("  Max surface dimensions: %d x %d x %d\n",
               prop.maxSurface1D, prop.maxSurface2D[0], prop.maxSurface2D[1]);
        printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");
        printf("  ECC enabled: %s\n", prop.ECCEnabled ? "Yes" : "No");
        printf("  PCI bus ID: %d\n", prop.pciBusID);
        printf("  PCI device ID: %d\n", prop.pciDeviceID);
        printf("  TCC driver: %s\n", prop.tccDriver ? "Yes" : "No");
        printf("  Async engine count: %d\n", prop.asyncEngineCount);
        printf("  Unified addressing: %s\n", prop.unifiedAddressing ? "Yes" : "No");
        printf("  Compute mode: %d\n", prop.computeMode);
        printf("  Max grid size: %d x %d x %d\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }

    return 0;
}