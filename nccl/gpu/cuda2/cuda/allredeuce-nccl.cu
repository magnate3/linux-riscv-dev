#include <stdio.h>
#include <stdlib.h>
#include <nccl.h>
#include <cuda_runtime.h>

#define NUM_ELEMENTS 1024
#define NUM_GPUS 2

// Error checking macros for CUDA and NCCL calls.
#define CHECK_CUDA(call)                                                  \
  do {                                                                    \
    cudaError_t err = call;                                               \
    if (err != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,    \
              cudaGetErrorString(err));                                   \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  } while (0)

#define CHECK_NCCL(call)                                                  \
  do {                                                                    \
    ncclResult_t res = call;                                              \
    if (res != ncclSuccess) {                                             \
      fprintf(stderr, "NCCL Error at %s:%d - %s\n", __FILE__, __LINE__,    \
              ncclGetErrorString(res));                                   \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  } while (0)

int main() {
    int devices[NUM_GPUS] = {0, 1};
    ncclComm_t comms[NUM_GPUS];
    float *sendbuff[NUM_GPUS], *recvbuff[NUM_GPUS];
    cudaStream_t streams[NUM_GPUS];
    float *hostResult = (float*)malloc(NUM_ELEMENTS * sizeof(float));

    // Allocate memory, create streams, and initialize send buffers.
    for (int i = 0; i < NUM_GPUS; i++) {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        CHECK_CUDA(cudaMalloc(&sendbuff[i], NUM_ELEMENTS * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&recvbuff[i], NUM_ELEMENTS * sizeof(float)));
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
        
        // Initialize the send buffer with a simple value: (i+1) for GPU i.
        float *hostData = (float*)malloc(NUM_ELEMENTS * sizeof(float));
        for (int j = 0; j < NUM_ELEMENTS; j++) {
            hostData[j] = (float)(i + 1);
        }
        CHECK_CUDA(cudaMemcpy(sendbuff[i], hostData, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
        free(hostData);
    }

    // Initialize NCCL communicator across the GPUs.
    CHECK_NCCL(ncclCommInitAll(comms, NUM_GPUS, devices));

    // Perform AllReduce: Each GPU contributes its sendbuff and the summed result is stored in recvbuff.
    for (int i = 0; i < NUM_GPUS; i++) {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        CHECK_NCCL(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i],
                                 NUM_ELEMENTS, ncclFloat, ncclSum, comms[i], streams[i]));
    }

    // Synchronize streams to ensure the collective operation has finished.
    for (int i = 0; i < NUM_GPUS; i++) {
        CHECK_CUDA(cudaSetDevice(devices[i]));
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    // Copy result from GPU 0 (all GPUs now hold the same result) back to the host.
    CHECK_CUDA(cudaSetDevice(devices[0]));
    CHECK_CUDA(cudaMemcpy(hostResult, recvbuff[0], NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));

    // Print the first 10 elements of the result.
    printf("Result of AllReduce (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", hostResult[i]);
    }
    printf("\n");

    // Cleanup resources.
    for (int i = 0; i < NUM_GPUS; i++) {
        ncclCommDestroy(comms[i]);
        cudaFree(sendbuff[i]);
        cudaFree(recvbuff[i]);
        cudaStreamDestroy(streams[i]);
    }
    free(hostResult);
    return 0;
}
