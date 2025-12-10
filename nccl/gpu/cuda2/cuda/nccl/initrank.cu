/*
two gpus recieve 1 float (1 or 2); then, both output 3, through allreduce
*/

#include <stdio.h>
#include <stdlib.h>
#include <nccl.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                         \
  {                                                                         \
    const cudaError_t error = call;                                         \
    if (error != cudaSuccess) {                                             \
      printf("Error: %s:%d, ", __FILE__, __LINE__);                         \
      printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));   \
      exit(1);                                                              \
    }                                                                       \
  }

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

int main() {
  const int nDev = 2;
  int devs[2] = {0, 1};
  ncclComm_t comms[2];

  float* d_send[2];
  float* d_recv[2];

  // Allocate and initialize device memory
  for (int i = 0; i < nDev; ++i) {
    CHECK(cudaSetDevice(devs[i]));
    CHECK(cudaMalloc(&d_send[i], sizeof(float)));
    CHECK(cudaMalloc(&d_recv[i], sizeof(float)));

    float h_val = float(i + 1); // Just some test data
    CHECK(cudaMemcpy(d_send[i], &h_val, sizeof(float), cudaMemcpyHostToDevice));
  }

  // Initialize NCCL
  ncclUniqueId id;
  NCCLCHECK(ncclGetUniqueId(&id));
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i) {
    CHECK(cudaSetDevice(devs[i]));
    NCCLCHECK(ncclCommInitRank(&comms[i], nDev, id, i));
  }
  NCCLCHECK(ncclGroupEnd());

  // Launch NCCL AllReduce
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i) {
    CHECK(cudaSetDevice(devs[i]));
    NCCLCHECK(ncclAllReduce(
      (const void*)d_send[i],
      (void*)d_recv[i],
      1,
      ncclFloat,
      ncclSum,
      comms[i],
      0  // Default stream
    ));
  }
  NCCLCHECK(ncclGroupEnd());

  // Copy results back to host
  for (int i = 0; i < nDev; ++i) {
    float h_result;
    CHECK(cudaSetDevice(devs[i]));
    CHECK(cudaMemcpy(&h_result, d_recv[i], sizeof(float), cudaMemcpyDeviceToHost));
    printf("Device %d result: %f\n", devs[i], h_result);
  }

  // Cleanup
  for (int i = 0; i < nDev; ++i) {
    CHECK(cudaSetDevice(devs[i]));
    CHECK(cudaFree(d_send[i]));
    CHECK(cudaFree(d_recv[i]));
    ncclCommDestroy(comms[i]);
  }

  return 0;
}
