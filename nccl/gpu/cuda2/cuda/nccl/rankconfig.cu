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

int main(int argc, char* argv[]) {
  // For a multi-process application these values would be determined by your runtime (e.g., MPI)
  // Here we assume a single rank (rank 0) for demonstration purposes.
  int nranks = 1; // Total number of processes/GPUs
  int rank = 0;   // This process's rank

  // Create and distribute NCCL unique ID (in a multi-process setting, rank 0 would generate the id and share it)
  ncclUniqueId id;
  NCCLCHECK(ncclGetUniqueId(&id));

  // Initialize a communicator using the NCCL config structure
  ncclComm_t comm;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = 1;
  // = 0 will outputs operation in progress
  config.minCTAs = 4;
  config.maxCTAs = 16;
  config.cgaClusterSize = 2;
  config.netName = "Socket";

  NCCLCHECK(ncclCommInitRankConfig(&comm, nranks, id, rank, &config));

  // Poll asynchronously for any error events from the communicator.
  ncclResult_t state;
  ncclResult_t res;
    do {
    res = ncclCommGetAsyncError(comm, &state);
    if (res != ncclSuccess && res != ncclInProgress) {
        fprintf(stderr, "Unexpected NCCL error: %s\n", ncclGetErrorString(res));
        exit(EXIT_FAILURE);
    }
    // Optionally, insert a short sleep or yield here
    } while (state == ncclInProgress);


  if (state != ncclSuccess) {
    fprintf(stderr, "Async error encountered: %s\n", ncclGetErrorString(state));
    exit(EXIT_FAILURE);
  } else {
    printf("NCCL communicator initialized successfully with custom configuration.\n");
  }

  // At this point, you can perform your NCCL collective operations...

  // Cleanup
  NCCLCHECK(ncclCommDestroy(comm));
  return 0;
}
