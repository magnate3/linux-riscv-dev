#include <cstdio>
#include <cstdlib>
#include <nccl.h>
#include <cuda_runtime.h>

#define RECVCOUNT 4       // Each rank will receive 4 elements.
#define NUM_RANKS 2       // Total number of ranks (GPUs).

// Macro to check for CUDA errors.
#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// Macro to check for NCCL errors.
#define CHECK_NCCL(call)                                              \
    do {                                                              \
        ncclResult_t res = call;                                      \
        if (res != ncclSuccess) {                                     \
            fprintf(stderr, "NCCL error at %s:%d: %s\n",              \
                    __FILE__, __LINE__, ncclGetErrorString(res));     \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

int main() {
    // Check available GPUs.
    int numDevices = 0;
    CHECK_CUDA(cudaGetDeviceCount(&numDevices));
    if (numDevices < NUM_RANKS) {
        printf("This example requires at least %d GPUs.\n", NUM_RANKS);
        return 0;
    }

    int devices[NUM_RANKS] = {0, 1};

    // Declare NCCL communicators, buffers, and CUDA streams.
    ncclComm_t comms[NUM_RANKS];
    float* sendbuffs[NUM_RANKS];
    float* recvbuffs[NUM_RANKS];
    cudaStream_t streams[NUM_RANKS];

    // Allocate memory and create streams on each GPU.
    // Each send buffer must have space for NUM_RANKS * RECVCOUNT elements.
    // Each receive buffer will hold RECVCOUNT elements.
    for (int i = 0; i < NUM_RANKS; ++i) {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        CHECK_CUDA(cudaMalloc(&sendbuffs[i], NUM_RANKS * RECVCOUNT * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&recvbuffs[i], RECVCOUNT * sizeof(float)));
        CHECK_CUDA(cudaStreamCreate(&streams[i]));

        // Initialize the send buffer with unique data for each rank.
        // For example, for rank i, fill data as: (i+1)*10 + j, where j = 0 ... (NUM_RANKS*RECVCOUNT-1)
        float data[NUM_RANKS * RECVCOUNT];
        for (int j = 0; j < NUM_RANKS * RECVCOUNT; j++) {
            data[j] = (i + 1) * 10 + j;
        }
        CHECK_CUDA(cudaMemcpy(sendbuffs[i], data, NUM_RANKS * RECVCOUNT * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Initialize NCCL communicators for all GPUs.
    CHECK_NCCL(ncclCommInitAll(comms, NUM_RANKS, devices));

    // Perform the ReduceScatter operation using a sum reduction.
    // Each GPU sends its entire send buffer, and after the reduction each rank
    // receives a block of RECVCOUNT elements corresponding to its rank index.
    for (int i = 0; i < NUM_RANKS; ++i) {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        CHECK_NCCL(ncclReduceScatter(
            (const void*)sendbuffs[i],
            (void*)recvbuffs[i],
            RECVCOUNT,      // Each rank will receive RECVCOUNT elements.
            ncclFloat,
            ncclSum,
            comms[i],
            streams[i]
        ));
    }

    // Synchronize all CUDA streams.
    for (int i = 0; i < NUM_RANKS; ++i) {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    // Copy and print the result from each GPU.
    for (int i = 0; i < NUM_RANKS; ++i) {
        float hostData[RECVCOUNT];
        CHECK_CUDA(cudaMemcpy(hostData, recvbuffs[i], RECVCOUNT * sizeof(float), cudaMemcpyDeviceToHost));
        printf("Device %d received data: ", devices[i]);
        for (int j = 0; j < RECVCOUNT; ++j)
            printf("%f ", hostData[j]);
        printf("\n");
    }

    // Cleanup: free GPU memory, destroy CUDA streams, and NCCL communicators.
    for (int i = 0; i < NUM_RANKS; ++i) {
        ncclCommDestroy(comms[i]);
        CHECK_CUDA(cudaFree(sendbuffs[i]));
        CHECK_CUDA(cudaFree(recvbuffs[i]));
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }

    return 0;
}
