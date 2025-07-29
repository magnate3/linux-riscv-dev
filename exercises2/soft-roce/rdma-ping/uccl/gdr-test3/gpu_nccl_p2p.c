#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N_ITERS 10
#define MSG_SIZE (20ULL * 1024ULL * 1024ULL * 1024ULL) // 2 GB

double get_time_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000.0 + tv.tv_usec;
}

int main(int argc, char **argv) {
    int rank, nprocs;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    if (nprocs != 2) {
        if (rank == 0) fprintf(stderr, "Must run with 2 ranks\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Initialize CUDA
    cudaSetDevice(rank % 8); // Support multi-GPU per node
    
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, rank % 8);
    printf("Rank %d using GPU: %s\n", rank, prop.name);
    
    // Initialize NCCL
    ncclUniqueId ncclId;
    ncclComm_t ncclComm;
    
    if (rank == 0) {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    ncclCommInitRank(&ncclComm, nprocs, ncclId, rank);
    printf("Rank %d: NCCL initialized\n", rank);
    
    // Allocate GPU memory
    void *d_data;
    cudaError_t err = cudaMalloc(&d_data, MSG_SIZE);
    if (err != cudaSuccess) {
        printf("Rank %d: cudaMalloc failed: %s\n", rank, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Initialize data
    cudaMemset(d_data, rank + 1, MSG_SIZE);
    cudaDeviceSynchronize();
    
    printf("Rank %d: Allocated %.2f GB on GPU\n", 
           rank, (double)MSG_SIZE / (1024*1024*1024));
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Starting NCCL GPU-to-GPU benchmark...\n");
    }
    
    double total_time_us = 0;
    
    for (int i = 0; i < N_ITERS; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        
        double t0 = get_time_us();
        
        // NCCL AllReduce - optimized GPU-to-GPU communication
        // This automatically uses best path: NVLink/PCIe/InfiniBand
        // Note: All ranks must participate in collective operations
        ncclAllReduce(d_data, d_data, MSG_SIZE / sizeof(float), 
                     ncclFloat, ncclSum, ncclComm, 0);
        
        cudaDeviceSynchronize();
        
        double t1 = get_time_us();
        total_time_us += (t1 - t0);
        
        if (rank == 0 && (i + 1) % 10 == 0) {
            printf("Iteration %d/%d: %.2f ms\n", i + 1, N_ITERS, (t1 - t0)/1000.0);
        }
    }
    
    if (rank == 0) {
        double avg_time_us = total_time_us / N_ITERS;
        double bandwidth = (MSG_SIZE / (1024.0*1024.0*1024.0)) / (avg_time_us / 1000000.0);
        
        printf("\n=== NCCL GPU Communication Results ===\n");
        printf("Transfer size: %.2f GB\n", (double)MSG_SIZE / (1UL<<30));
        printf("Iterations: %d\n", N_ITERS);
        printf("Average time: %.2f µs (%.3f ms)\n", avg_time_us, avg_time_us/1000.0);
        printf("Bandwidth: %.2f GB/s\n", bandwidth);
        printf("Method: NCCL (topology-aware, GPU-Direct)\n");
        printf("Advantages: Cross-node, optimized, GPU-Direct RDMA\n");
    }
    
    ncclCommDestroy(ncclComm);
    cudaFree(d_data);
    MPI_Finalize();
    return 0;
}