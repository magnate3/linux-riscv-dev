#include <cstdio>
#include <cstdlib>
#include <nccl.h>
#include <cuda_runtime.h>

#define N 4 // Number of elements per GPU
#define CHECK_CUDA(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

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

    const int numRanks = 2; // Using 2 GPUs for this example.
    int devices[numRanks] = {0, 1}; // Select which GPUs to use.

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

        // Initialize send buffer with sample data.
        float data[N];
        for (int j = 0; j < N; ++j)
            data[j] = (float)(i + 1); // For example: GPU 0 gets [1,1,1,1], GPU 1 gets [2,2,2,2]
        CHECK_CUDA(cudaMemcpy(sendbuffs[i], data, N * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Initialize NCCL communicators for all GPUs.
    CHECK_NCCL(ncclCommInitAll(comms, numRanks, devices));

    // Perform the AllReduce operation on each GPU.
    for (int i = 0; i < numRanks; ++i) {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        CHECK_NCCL(ncclAllReduce((const void*)sendbuffs[i],
                                 (void*)recvbuffs[i],
                                 N,
                                 ncclFloat,
                                 ncclSum,
                                 comms[i],
                                 streams[i]));
    }

    // Synchronize streams to ensure completion.
    for (int i = 0; i < numRanks; ++i) {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    // Retrieve and print the results from each GPU.
    for (int i = 0; i < numRanks; ++i) {
        float result[N];
        CHECK_CUDA(cudaMemcpy(result, recvbuffs[i], N * sizeof(float), cudaMemcpyDeviceToHost));
        printf("Device %d result: ", devices[i]);
        for (int j = 0; j < N; ++j)
            printf("%f ", result[j]);
        printf("\n");
    }

    // Cleanup: Destroy communicators, free memory, and destroy streams.
    for (int i = 0; i < numRanks; ++i) {
        ncclCommDestroy(comms[i]);
        cudaFree(sendbuffs[i]);
        cudaFree(recvbuffs[i]);
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}
