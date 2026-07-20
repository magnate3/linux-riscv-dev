#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdint.h>

#define N_ITERS 10
#define MSG_SIZE (20ULL * 1024ULL * 1024ULL * 1024ULL) // 20 GB for direct testing
#define WARMUP_ITERS 3

double get_time_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000.0 + tv.tv_usec;
}

void print_node_info(int rank) {
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    printf("Rank %d on node %s: Found %d GPU(s)\n", rank, hostname, device_count);
    
    // Print GPU information
    for (int i = 0; i < device_count; i++) {
        struct cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("  GPU %d: %s (Memory: %.2f GB, CC: %d.%d)\n", 
               i, prop.name, 
               (double)prop.totalGlobalMem / (1024*1024*1024),
               prop.major, prop.minor);
    }
}

void check_nccl_error(ncclResult_t result, const char* msg, int rank) {
    if (result != ncclSuccess) {
        printf("Rank %d: NCCL Error at %s: %s\n", rank, msg, ncclGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void check_cuda_error(cudaError_t result, const char* msg, int rank) {
    if (result != cudaSuccess) {
        printf("Rank %d: CUDA Error at %s: %s\n", rank, msg, cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

int main(int argc, char **argv) {
    int rank, nprocs;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    if (nprocs != 2) {
        if (rank == 0) {
            fprintf(stderr, "ERROR: This program requires exactly 2 MPI ranks\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Print node and GPU information
    print_node_info(rank);
    
    // Initialize CUDA - use GPU 0 on each node
    int gpu_id = 0;
    check_cuda_error(cudaSetDevice(gpu_id), "cudaSetDevice", rank);
    
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu_id);
    printf("Rank %d: Using GPU %d: %s\n", rank, gpu_id, prop.name);
    
    // Check available GPU memory
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("Rank %d: GPU Memory - Total: %.2f GB, Free: %.2f GB\n", 
           rank, (double)total_mem / (1UL<<30), (double)free_mem / (1UL<<30));
    
    // Initialize NCCL for cross-node communication
    ncclUniqueId ncclId;
    ncclComm_t ncclComm;
    
    if (rank == 0) {
        check_nccl_error(ncclGetUniqueId(&ncclId), "ncclGetUniqueId", rank);
    }
    
    // Broadcast NCCL ID to all ranks
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    // Initialize NCCL communicator
    check_nccl_error(ncclCommInitRank(&ncclComm, nprocs, ncclId, rank), 
                     "ncclCommInitRank", rank);
    printf("Rank %d: NCCL communicator initialized\n", rank);
    
    // Allocate GPU memory - ONLY one buffer needed for AllReduce
    void *d_data;
    check_cuda_error(cudaMalloc(&d_data, MSG_SIZE), "cudaMalloc", rank);
    
    // Initialize data with rank-specific pattern
    check_cuda_error(cudaMemset(d_data, rank + 1, MSG_SIZE), "cudaMemset", rank);
    check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize init", rank);
    
    printf("Rank %d: Allocated %.2f GB GPU memory\n", 
           rank, (double)MSG_SIZE / (1UL<<30));
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("\n=== Direct GPU-to-GPU RDMA Benchmark (No CPU Data Path) ===\n");
        printf("Transfer size: %.2f GB per operation\n", (double)MSG_SIZE / (1UL<<30));
        printf("Iterations: %d\n", N_ITERS);
        printf("Method: NCCL AllReduce (direct GPU-to-network-to-GPU)\n");
        printf("CPU involvement: MINIMAL (control plane only)\n\n");
    }
    
    // Pre-create CUDA stream to avoid per-iteration overhead
    cudaStream_t stream;
    check_cuda_error(cudaStreamCreate(&stream), "cudaStreamCreate", rank);
    
    // Minimal warmup - just 1 iteration
    ncclAllReduce(d_data, d_data, MSG_SIZE / sizeof(float), 
                 ncclFloat, ncclSum, ncclComm, stream);
    check_cuda_error(cudaStreamSynchronize(stream), "warmup sync", rank);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Benchmark iterations - MINIMAL CPU involvement
    double total_time_us = 0;
    
    for (int i = 0; i < N_ITERS; i++) {
        // CRITICAL: Minimal synchronization to reduce CPU overhead
        double t0 = get_time_us();
        
        // DIRECT GPU-TO-GPU: This bypasses CPU for data transfer
        // Data flows: GPU -> NIC -> Network -> NIC -> GPU
        ncclAllReduce(d_data, d_data, MSG_SIZE / sizeof(float), 
                     ncclFloat, ncclSum, ncclComm, stream);
        
        // Only sync when timing - minimal CPU involvement
        check_cuda_error(cudaStreamSynchronize(stream), "benchmark sync", rank);
        
        double t1 = get_time_us();
        total_time_us += (t1 - t0);
        
        if (rank == 0 && (i + 1) % 10 == 0) {
            printf("Iteration %d/%d: %.2f ms\n", i + 1, N_ITERS, (t1 - t0)/1000.0);
        }
    }
    
    // Calculate and display results
    if (rank == 0) {
        double avg_time_us = total_time_us / N_ITERS;
        double bandwidth = (MSG_SIZE / (1024.0*1024.0*1024.0)) / (avg_time_us / 1000000.0);
        
        printf("\n=== DIRECT GPU-TO-GPU RDMA Results ===\n");
        printf("Transfer size: %.2f GB\n", (double)MSG_SIZE / (1UL<<30));
        printf("Iterations: %d\n", N_ITERS);
        printf("Average time: %.2f µs (%.3f ms)\n", avg_time_us, avg_time_us/1000.0);
        printf("Bandwidth: %.2f GB/s\n", bandwidth);
        printf("Data path: GPU → NIC → Network → NIC → GPU\n");
        printf("CPU involvement: CONTROL PLANE ONLY (no data copying)\n");
        printf("Technology: GPU-Direct RDMA + NCCL optimization\n");
        
        // Performance expectations for direct GPU-to-GPU
        if (bandwidth > 80.0) {
            printf("Status: EXCELLENT - True GPU-Direct RDMA performance\n");
        } else if (bandwidth > 40.0) {
            printf("Status: GOOD - Likely GPU-Direct enabled\n");
        } else if (bandwidth > 10.0) {
            printf("Status: MODERATE - Check GPU-Direct RDMA configuration\n");
        } else {
            printf("Status: POOR - Data likely going through CPU/system memory\n");
        }
    }
    
    // Cleanup
    check_cuda_error(cudaStreamDestroy(stream), "cudaStreamDestroy", rank);
    check_cuda_error(cudaFree(d_data), "cudaFree", rank);
    check_nccl_error(ncclCommDestroy(ncclComm), "ncclCommDestroy", rank);
    
    MPI_Finalize();
    
    if (rank == 0) {
        printf("\nDirect GPU-to-GPU RDMA benchmark completed!\n");
    }
    
    return 0;
}
