#include <stdio.h>
#include <stdlib.h>
#include <nccl.h>
#include <cuda_runtime.h>

// Error checking macro for CUDA calls.
#define CHECK(call)                                                         \
  {                                                                         \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",          \
              __FILE__, __LINE__, cudaGetErrorString(err));                 \
      exit(EXIT_FAILURE);                                                   \
    }                                                                       \
  }

// Error checking macro for NCCL calls.
#define NCCLCHECK(call)                                                     \
  {                                                                         \
    ncclResult_t res = call;                                                \
    if (res != ncclSuccess) {                                               \
      fprintf(stderr, "NCCL error in file '%s' in line %i: %s.\n",          \
              __FILE__, __LINE__, ncclGetErrorString(res));                 \
      exit(EXIT_FAILURE);                                                   \
    }                                                                       \
  }

int main() {
  const int nDevs = 2;  // number of GPUs
  int devs[nDevs] = {0, 1};
  ncclComm_t comms[nDevs];

  // Allocate send and receive buffers on each GPU.
  float *d_send[nDevs];
  float *d_recv[nDevs];
  const int numElements = 1024 * 1024;  // 1 million floats per GPU
  size_t dataSize = numElements * sizeof(float);

  for (int i = 0; i < nDevs; i++) {
    CHECK(cudaSetDevice(devs[i]));
    CHECK(cudaMalloc(&d_send[i], dataSize));
    CHECK(cudaMalloc(&d_recv[i], dataSize));

    // Allocate a host buffer to initialize data.
    float *h_buffer = (float*)malloc(dataSize);
    float initValue = (float)(i + 1);  // GPU0:1.0, GPU1:2.0, etc.
    for (int j = 0; j < numElements; j++) {
      h_buffer[j] = initValue;
    }
    CHECK(cudaMemcpy(d_send[i], h_buffer, dataSize, cudaMemcpyHostToDevice));
    free(h_buffer);
  }

  // Set each GPU as active.
  for (int i = 0; i < nDevs; i++) {
    CHECK(cudaSetDevice(devs[i]));
  }

  // Initialize NCCL communicators for all GPUs.
  NCCLCHECK(ncclCommInitAll(comms, nDevs, devs));

  // Create CUDA events for timing on GPU 0.
  cudaEvent_t start, stop;
  CHECK(cudaSetDevice(devs[0]));
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  // Record the start event.
  CHECK(cudaEventRecord(start, 0));

  // Launch the NCCL AllReduce collective operation.
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDevs; i++) {
    CHECK(cudaSetDevice(devs[i]));
    NCCLCHECK(ncclAllReduce((const void*)d_send[i],
                            (void*)d_recv[i],
                            numElements,
                            ncclFloat,
                            ncclSum,
                            comms[i],
                            0));  // using default CUDA stream
  }
  NCCLCHECK(ncclGroupEnd());

  // Record the stop event and wait for the operation to complete.
  CHECK(cudaSetDevice(devs[0]));
  CHECK(cudaEventRecord(stop, 0));
  CHECK(cudaEventSynchronize(stop));

  // Calculate elapsed time.
  float elapsedTime;
  CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("NCCL AllReduce took %f ms\n", elapsedTime);

  // Cleanup: free device memory and destroy communicators.
  for (int i = 0; i < nDevs; i++) {
    CHECK(cudaSetDevice(devs[i]));
    CHECK(cudaFree(d_send[i]));
    CHECK(cudaFree(d_recv[i]));
    NCCLCHECK(ncclCommDestroy(comms[i]));
  }

  // Destroy CUDA events.
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));

  return 0;
}
