#include <cstdio>
#include <cstdlib>
#include <nccl.h>
#include <cuda_runtime.h>

#define N 4  // Number of elements in each buffer

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

    // For this example, we use 2 GPUs.
    const int numRanks = 2;
    int devices[numRanks] = {0, 1};

    // Define the root rank for the reduction.
    int root = 0;

    ncclComm_t comms[numRanks];
    float* sendbuffs[numRanks];
    float* recvbuffs[numRanks];
    cudaStream_t streams[numRanks];

    // Allocate memory and create streams on each GPU.
    for (int i = 0; i < numRanks; ++i) {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        CHECK_CUDA(cudaMalloc(&sendbuffs[i], N * sizeof(float)));
        // Allocate recvbuff even if it is only used on the root.
        CHECK_CUDA(cudaMalloc(&recvbuffs[i], N * sizeof(float)));
        CHECK_CUDA(cudaStreamCreate(&streams[i]));

        // Initialize the send buffer with sample data:
        // GPU 0 gets [1, 1, 1, 1] and GPU 1 gets [2, 2, 2, 2].
        float data[N];
        for (int j = 0; j < N; ++j)
            data[j] = (i == root) ? 1.0f : 2.0f;
        CHECK_CUDA(cudaMemcpy(sendbuffs[i], data, N * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Initialize NCCL communicators across the GPUs.
    CHECK_NCCL(ncclCommInitAll(comms, numRanks, devices));

    // Perform the Reduce operation.
    // Only the root rank (rank 0) will receive the reduced result.
    for (int i = 0; i < numRanks; ++i) {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        CHECK_NCCL(ncclReduce(
            /* sendbuff */ (const void*)sendbuffs[i],
            /* recvbuff */ (void*)recvbuffs[i],
            /* count */ N,
            /* datatype */ ncclFloat,
            /* op */ ncclSum,
            /* root */ root,
            /* communicator */ comms[i],
            /* stream */ streams[i]
        ));
    }

    // Synchronize streams to ensure the reduce operation is complete.
    for (int i = 0; i < numRanks; ++i) {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    // Only the root rank has the valid reduced result.
    if (root < numRanks) {
        float hostResult[N];
        CHECK_CUDA(cudaMemcpy(hostResult, recvbuffs[root], N * sizeof(float), cudaMemcpyDeviceToHost));
        printf("Reduced result on root (device %d): ", devices[root]);
        for (int j = 0; j < N; ++j)
            printf("%f ", hostResult[j]);
        printf("\n");
    }

    // Cleanup: free allocated GPU memory, destroy streams and NCCL communicators.
    for (int i = 0; i < numRanks; ++i) {
        ncclCommDestroy(comms[i]);
        CHECK_CUDA(cudaFree(sendbuffs[i]));
        CHECK_CUDA(cudaFree(recvbuffs[i]));
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }

    return 0;
}
