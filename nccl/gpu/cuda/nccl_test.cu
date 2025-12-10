#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"

#define MPICHECK(cmd) do {                           \
  int e = cmd;                                       \
  if(e != MPI_SUCCESS) {                             \
    printf("MPI error %s:%d '%d'\n", __FILE__, __LINE__, e); \
    exit(EXIT_FAILURE);                              \
  }                                                  \
} while(0)

#define CUDACHECK(cmd) do {                          \
  cudaError_t e = cmd;                               \
  if(e != cudaSuccess) {                             \
    printf("Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(EXIT_FAILURE);                              \
  }                                                  \
} while(0)

#define NCCLCHECK(cmd) do {                          \
  ncclResult_t r = cmd;                              \
  if(r != ncclSuccess) {                             \
    printf("NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
    exit(EXIT_FAILURE);                              \
  }                                                  \
} while(0)

// Simple CUDA kernel to initialize a buffer to a constant value.
__global__ void initBuffer(float* buf, int size, float value) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    buf[idx] = value;
  }
}

int main(int argc, char* argv[]) {
  // Initialize MPI.
  MPICHECK(MPI_Init(&argc, &argv));

  int worldRank, worldSize;
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
  printf("worldRank %d , worldSize %d \n",worldRank, worldSize);
  // Require at least 3 processes for this example.
  if(worldSize < 3) {
    if(worldRank == 0) {
      printf("This example requires at least 3 MPI processes.\n");
    }
    MPI_Finalize();
    return 1;
  }

  // For simplicity, assume each MPI process uses a GPU matching its global rank.
  CUDACHECK(cudaSetDevice(worldRank));

  // Buffer size for NCCL collective.
  int size = 1024; // number of float elements
  float *sendBuff, *recvBuff;
  CUDACHECK(cudaMalloc(&sendBuff, size * sizeof(float)));
  CUDACHECK(cudaMalloc(&recvBuff, size * sizeof(float)));

  // Initialize sendBuff with the global rank as a float.
  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  initBuffer<<<blocks, threads>>>(sendBuff, size, (float)worldRank);
  CUDACHECK(cudaDeviceSynchronize());

  // --- Form two MPI groups via MPI_Comm_split ---
  // Group 1: Global ranks 0 and 1.
  MPI_Comm group1Comm;
  int group1Color = (worldRank == 0 || worldRank == 1) ? 0 : MPI_UNDEFINED;
  MPICHECK(MPI_Comm_split(MPI_COMM_WORLD, group1Color, worldRank, &group1Comm));

  // Group 2: Global ranks 0 and 2.
  MPI_Comm group2Comm;
  int group2Color = (worldRank == 0 || worldRank == 2) ? 0 : MPI_UNDEFINED;
  MPICHECK(MPI_Comm_split(MPI_COMM_WORLD, group2Color, worldRank, &group2Comm));

  // --- Create NCCL communicators for each group ---
  ncclUniqueId id1, id2;
  ncclComm_t comm1 = NULL, comm2 = NULL;
  int group1Rank = -1, group1Size = 0;
  int group2Rank = -1, group2Size = 0;

  // For group 1 (ranks 0 and 1):
  if(group1Comm != MPI_COMM_NULL) {
    MPICHECK(MPI_Comm_rank(group1Comm, &group1Rank));
    MPICHECK(MPI_Comm_size(group1Comm, &group1Size));
    if(group1Rank == 0) {
      NCCLCHECK(ncclGetUniqueId(&id1));
    }
    MPICHECK(MPI_Bcast(&id1, sizeof(id1), MPI_BYTE, 0, group1Comm));
    NCCLCHECK(ncclCommInitRank(&comm1, group1Size, id1, group1Rank));
  }

  // For group 2 (ranks 0 and 2):
  if(group2Comm != MPI_COMM_NULL) {
    MPICHECK(MPI_Comm_rank(group2Comm, &group2Rank));
    MPICHECK(MPI_Comm_size(group2Comm, &group2Size));
    if(group2Rank == 0) {
      NCCLCHECK(ncclGetUniqueId(&id2));
    }
    MPICHECK(MPI_Bcast(&id2, sizeof(id2), MPI_BYTE, 0, group2Comm));
    NCCLCHECK(ncclCommInitRank(&comm2, group2Size, id2, group2Rank));
  }

  // Create a CUDA stream for NCCL operations.
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  // --- Perform NCCL AllReduce for each group separately ---
  // Group 1: GPUs in group1 (global ranks 0 and 1) perform an AllReduce.
  if(comm1 != NULL) {
    NCCLCHECK(ncclAllReduce((const void*)sendBuff, (void*)recvBuff, size, ncclFloat, ncclSum, comm1, stream));
    CUDACHECK(cudaStreamSynchronize(stream));

    // For demonstration, have group1 rank 0 print the first 10 elements.
    if(group1Rank == 0) {
      float hostOut[10];
      CUDACHECK(cudaMemcpy(hostOut, recvBuff, 10*sizeof(float), cudaMemcpyDeviceToHost));
      printf("Group1 (ranks 0 & 1) NCCL AllReduce result on global rank %d: ", worldRank);
      for(int i = 0; i < 10; i++) {
        printf("%f ", hostOut[i]);
      }
      printf("\n");
    }
  }

  // Reinitialize sendBuff for the next operation.
  initBuffer<<<blocks, threads>>>(sendBuff, size, (float)worldRank);
  CUDACHECK(cudaDeviceSynchronize());

  // Group 2: GPUs in group2 (global ranks 0 and 2) perform an AllReduce.
  if(comm2 != NULL) {
    NCCLCHECK(ncclAllReduce((const void*)sendBuff, (void*)recvBuff, size, ncclFloat, ncclSum, comm2, stream));
    CUDACHECK(cudaStreamSynchronize(stream));

    // For demonstration, have group2 rank 0 print the first 10 elements.
    if(group2Rank == 0) {
      float hostOut[10];
      CUDACHECK(cudaMemcpy(hostOut, recvBuff, 10*sizeof(float), cudaMemcpyDeviceToHost));
      printf("Group2 (ranks 0 & 2) NCCL AllReduce result on global rank %d: ", worldRank);
      for(int i = 0; i < 10; i++) {
        printf("%f ", hostOut[i]);
      }
      printf("\n");
    }
  }

  // Cleanup NCCL communicators.
  if(comm1) NCCLCHECK(ncclCommDestroy(comm1));
  if(comm2) NCCLCHECK(ncclCommDestroy(comm2));

  // Free the MPI group communicators.
  if(group1Comm != MPI_COMM_NULL) MPICHECK(MPI_Comm_free(&group1Comm));
  if(group2Comm != MPI_COMM_NULL) MPICHECK(MPI_Comm_free(&group2Comm));

  // Cleanup CUDA resources.
  CUDACHECK(cudaStreamDestroy(stream));
  CUDACHECK(cudaFree(sendBuff));
  CUDACHECK(cudaFree(recvBuff));

  MPICHECK(MPI_Finalize());
  return 0;
}
