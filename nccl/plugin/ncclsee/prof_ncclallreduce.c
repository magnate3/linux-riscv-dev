#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nccl.h>
#include <time.h>

double gettime(void)
{
  // return __rdtsc() / freq;
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ((ts.tv_sec * 1e6) + (ts.tv_nsec / 1e3));
}


int main(int argc, char *argv[]) {
    // Initialize MPI.
    MPI_Init(&argc, &argv);
    int rank, nRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    // Set the CUDA device based on MPI rank.
    cudaSetDevice(rank);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, rank);
    printf("MPI Rank %d using CUDA Device: %s\n", rank, prop.name);

    // Wait for user input to proceed.
    /* if (rank == 0) { */
    /*     printf("Press Enter to continue...\n"); */
    /*     getchar(); */
    /* } */
    /* MPI_Barrier(MPI_COMM_WORLD); */

    // Initialize CUPTI to record kernel activities.
    /* CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL)); */
    /* CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted)); */

    // Initialize NCCL.
    ncclUniqueId id;
    if (rank == 0) {
        ncclGetUniqueId(&id);
    }
    // Broadcast the NCCL unique ID to all ranks.
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t comm;
    ncclCommInitRank(&comm, nRanks, id, rank);

    // Allocate device memory.
    size_t count = 16777216;
    float *sendbuff, *recvbuff;
    cudaMalloc((void **)&sendbuff, count * sizeof(float));
    cudaMalloc((void **)&recvbuff, count * sizeof(float));

    // Initialize the send buffer with some values on the host.
    float *hostBuffer = (float*) malloc(count * sizeof(float));
    float base = (float)(rank*count);
    for (size_t i = 0; i < count; i++) {
        hostBuffer[i] = (float) (i) + base; // Distinguish data per rank.
    }
    cudaMemcpy(sendbuff, hostBuffer, count * sizeof(float), cudaMemcpyHostToDevice);

    // Optionally clear the receive buffer.
    cudaMemset(recvbuff, 0, count * sizeof(float));

    // Perform NCCL AllReduce (summing across ranks).
    double duration = gettime();
    for(int i = 0; i < 1; i++) {
        ncclAllReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, comm, cudaStreamDefault);
    }
    // Synchronize device to ensure the AllReduce (and its kernels) complete.
    cudaDeviceSynchronize();
    duration = gettime() - duration;
    printf("MPI Rank %d: NCCL AllReduce took %f useconds\n", rank, duration);


    cudaFree(sendbuff);
    cudaFree(recvbuff);

    // Perform NCCL AllGather and Allreduce.
    count = 128;
    cudaMalloc((void **)&sendbuff, count * sizeof(float));
    cudaMalloc((void **)&recvbuff, count * sizeof(float));
    cudaMemcpy(sendbuff, hostBuffer, count * sizeof(float), cudaMemcpyHostToDevice);
    for(int i = 0; i < 0; i++) {
        ncclAllGather(sendbuff, recvbuff, count, ncclFloat, comm, cudaStreamDefault);
        ncclAllReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, comm, cudaStreamDefault);
    }
    // Synchronize device to ensure the AllReduce (and its kernels) complete.
    cudaDeviceSynchronize();
    // Flush CUPTI buffers to process any remaining activity records.
     /* CUPTI_CALL(cuptiActivityFlushAll(0)); */

    // Verify the NCCL AllReduce result.
    float *host_recv = (float*) malloc(count * sizeof(float));
    cudaMemcpy(host_recv, recvbuff, count * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the first 10 elements of the result.
    /* printf("MPI Rank %d: First 5 elements after AllReduce:\n", rank); */
    /* for (int i = 0; i < 5; i++) { */
    /*     printf("%f ", host_recv[i]); */
    /* } */
    /* printf("\n"); */

    // Cleanup resources.
    ncclCommDestroy(comm);
    free(host_recv);
    free(hostBuffer);

    // Finalize MPI.
    MPI_Finalize();

    return 0;
}

