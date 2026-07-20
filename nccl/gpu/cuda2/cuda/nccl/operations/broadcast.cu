#include <cstdio>
#include <cstdlib>
#include <nccl.h>
#include <cuda_runtime.h>

#define N 4  // Number of elements in the buffer

// Macro to check for CUDA errors.
#define CHECK_CUDA(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

// Macro to check for NCCL errors.
#define CHECK_NCCL(call)                                      \
    do {                                                      \
        ncclResult_t res = call;                              \
        if (res != ncclSuccess) {                             \
            fprintf(stderr, "NCCL error at %s:%d: %s\n",      \
                    __FILE__, __LINE__, ncclGetErrorString(res)); \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

int main() {
    int numDevices = 0;
    CHECK_CUDA(cudaGetDeviceCount(&numDevices));
    
    if (numDevices < 2) {
        printf("This example requires at least two GPUs.\n");
        return 0;
    }
    
    // For simplicity, we'll use 2 GPUs.
    const int numRanks = 2;
    int devices[numRanks] = {0, 1};
    
    // Set the root rank for the broadcast.
    int root = 0;
    
    ncclComm_t comms[numRanks];
    float* sendbuffs[numRanks];
    float* recvbuffs[numRanks];
    cudaStream_t streams[numRanks];
    
    // Allocate memory and create streams on each GPU.
    for (int i = 0; i < numRanks; ++i) {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        CHECK_CUDA(cudaMalloc(&sendbuffs[i], N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&recvbuffs[i], N * sizeof(float)));
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
        
        // Only the root rank initializes its send buffer with data.
        if (i == root) {
            float data[N];
            for (int j = 0; j < N; ++j) {
                data[j] = (float)(j + 1);  // For example: [1, 2, 3, 4]
            }
            CHECK_CUDA(cudaMemcpy(sendbuffs[i], data, N * sizeof(float), cudaMemcpyHostToDevice));
        }
    }
    
    // Initialize NCCL communicators.
    CHECK_NCCL(ncclCommInitAll(comms, numRanks, devices));
    
    // Perform the broadcast operation.
    // The root rank's sendbuff is used, and all GPUs receive the broadcast data in their recvbuff.
    for (int i = 0; i < numRanks; ++i) {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        CHECK_NCCL(ncclBroadcast(
            /* sendbuff */ (const void*)sendbuffs[root],
            /* recvbuff */ (void*)recvbuffs[i],
            /* count */ N,
            /* datatype */ ncclFloat,
            /* root */ root,
            /* communicator */ comms[i],
            /* stream */ streams[i]
        ));
    }
    
    // Wait for all broadcast operations to complete.
    for (int i = 0; i < numRanks; ++i) {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
    
    // Copy the data back to the host and print the results.
    for (int i = 0; i < numRanks; ++i) {
        float hostData[N];
        CHECK_CUDA(cudaMemcpy(hostData, recvbuffs[i], N * sizeof(float), cudaMemcpyDeviceToHost));
        printf("Device %d received data: ", devices[i]);
        for (int j = 0; j < N; ++j)
            printf("%f ", hostData[j]);
        printf("\n");
    }
    
    // Cleanup: free GPU memory, destroy streams and NCCL communicators.
    for (int i = 0; i < numRanks; ++i) {
        ncclCommDestroy(comms[i]);
        CHECK_CUDA(cudaFree(sendbuffs[i]));
        CHECK_CUDA(cudaFree(recvbuffs[i]));
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
    
    return 0;
}
