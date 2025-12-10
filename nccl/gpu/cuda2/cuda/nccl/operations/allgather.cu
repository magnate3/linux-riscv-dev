#include <cstdio>
#include <cstdlib>
#include <nccl.h>
#include <cuda_runtime.h>

#define N 4  // Number of elements per GPU in the send buffer

// Macro to check for CUDA errors.
#define CHECK_CUDA(call)                                       \
    do {                                                       \
        cudaError_t err = call;                                \
        if (err != cudaSuccess) {                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

// Macro to check for NCCL errors.
#define CHECK_NCCL(call)                                       \
    do {                                                       \
        ncclResult_t res = call;                               \
        if (res != ncclSuccess) {                              \
            fprintf(stderr, "NCCL error at %s:%d: %s\n",       \
                    __FILE__, __LINE__, ncclGetErrorString(res)); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

int main() {
    int numDevices = 0;
    CHECK_CUDA(cudaGetDeviceCount(&numDevices));
    
    // Ensure we have at least 2 GPUs.
    if (numDevices < 2) {
        printf("This example requires at least two GPUs.\n");
        return 0;
    }
    
    // For this example, we use 2 GPUs.
    const int numRanks = 2;
    int devices[numRanks] = {0, 1};
    
    ncclComm_t comms[numRanks];
    float* sendbuffs[numRanks];
    float* recvbuffs[numRanks];
    cudaStream_t streams[numRanks];
    
    // Allocate memory and create streams on each GPU.
    // Each send buffer holds N elements; each receive buffer must hold numRanks * N elements.
    for (int i = 0; i < numRanks; ++i) {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        CHECK_CUDA(cudaMalloc(&sendbuffs[i], N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&recvbuffs[i], numRanks * N * sizeof(float)));
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
        
        // Initialize the send buffer with unique values for each rank.
        // For example: rank 0: [1, 2, 3, 4], rank 1: [11, 12, 13, 14]
        float data[N];
        int base = (i == 0) ? 1 : 11;
        for (int j = 0; j < N; ++j) {
            data[j] = base + j;
        }
        CHECK_CUDA(cudaMemcpy(sendbuffs[i], data, N * sizeof(float), cudaMemcpyHostToDevice));
    }
    
    // Initialize NCCL communicators across the GPUs.
    CHECK_NCCL(ncclCommInitAll(comms, numRanks, devices));
    
    // Perform the AllGather operation.
    // Each GPU gathers its own send buffer and that of the other GPU into its recv buffer.
    // The ordering in recv buffer is: [data from rank0 | data from rank1].
    for (int i = 0; i < numRanks; ++i) {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        CHECK_NCCL(ncclAllGather(
            /* sendbuff */ (const void*)sendbuffs[i],
            /* recvbuff */ (void*)recvbuffs[i],
            /* sendcount */ N,
            /* datatype */ ncclFloat,
            /* communicator */ comms[i],
            /* stream */ streams[i]
        ));
    }
    
    // Wait for all operations to complete.
    for (int i = 0; i < numRanks; ++i) {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
    
    // Copy the gathered data back to the host and print results.
    for (int i = 0; i < numRanks; ++i) {
        float hostData[numRanks * N];
        CHECK_CUDA(cudaMemcpy(hostData, recvbuffs[i], numRanks * N * sizeof(float), cudaMemcpyDeviceToHost));
        printf("Device %d received data: ", devices[i]);
        for (int j = 0; j < numRanks * N; ++j)
            printf("%f ", hostData[j]);
        printf("\n");
    }
    
    // Cleanup: free GPU memory, destroy streams, and destroy NCCL communicators.
    for (int i = 0; i < numRanks; ++i) {
        ncclCommDestroy(comms[i]);
        CHECK_CUDA(cudaFree(sendbuffs[i]));
        CHECK_CUDA(cudaFree(recvbuffs[i]));
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
    
    return 0;
}
