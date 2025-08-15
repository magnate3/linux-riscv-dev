#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>

#define N_ITERS 50
#define MSG_SIZE (10ULL * 1024ULL * 1024ULL * 1024ULL) // 10 GB for inter-node testing
#define WARMUP_ITERS 5

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
            fprintf(stderr, "ERROR: This program requires exactly 2 MPI ranks for inter-node communication\n");
            fprintf(stderr, "Usage: mpirun -np 2 -host node1,node2 ./gpu_internode_rdma\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Print node and GPU information
    print_node_info(rank);
    MPI_Barrier(MPI_COMM_WORLD);
    
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
    
    if (free_mem < MSG_SIZE) {
        printf("Rank %d: WARNING: Insufficient GPU memory. Need %.2f GB, have %.2f GB free\n",
               rank, (double)MSG_SIZE / (1UL<<30), (double)free_mem / (1UL<<30));
        // Reduce message size if needed
        // MSG_SIZE = free_mem * 0.8; // Use 80% of available memory
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Initialize NCCL for cross-node communication
    ncclUniqueId ncclId;
    ncclComm_t ncclComm;
    
    if (rank == 0) {
        check_nccl_error(ncclGetUniqueId(&ncclId), "ncclGetUniqueId", rank);
        printf("Rank 0: Generated NCCL unique ID for cross-node communication\n");
    }
    
    // Broadcast NCCL ID to all ranks
    MPI_Bcast(&ncclId, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    // Initialize NCCL communicator
    check_nccl_error(ncclCommInitRank(&ncclComm, nprocs, ncclId, rank), 
                     "ncclCommInitRank", rank);
    printf("Rank %d: NCCL communicator initialized for inter-node communication\n", rank);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Allocate GPU memory
    void *d_send_data, *d_recv_data;
    check_cuda_error(cudaMalloc(&d_send_data, MSG_SIZE), "cudaMalloc send buffer", rank);
    check_cuda_error(cudaMalloc(&d_recv_data, MSG_SIZE), "cudaMalloc recv buffer", rank);
    
    // Initialize data with rank-specific pattern
    check_cuda_error(cudaMemset(d_send_data, rank + 42, MSG_SIZE), "cudaMemset send", rank);
    check_cuda_error(cudaMemset(d_recv_data, 0, MSG_SIZE), "cudaMemset recv", rank);
    check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize init", rank);
    
    printf("Rank %d: Allocated %.2f GB GPU memory for send/recv buffers\n", 
           rank, (double)(MSG_SIZE * 2) / (1UL<<30));
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("\n=== Starting Inter-Node GPU RDMA Benchmark ===\n");
        printf("Transfer size: %.2f GB per operation\n", (double)MSG_SIZE / (1UL<<30));
        printf("Iterations: %d (+ %d warmup)\n", N_ITERS, WARMUP_ITERS);
        printf("Communication pattern: AllReduce (bidirectional GPU-to-GPU)\n");
        printf("Expected topology: Cross-node via InfiniBand/Ethernet\n\n");
    }
    
    // Warmup iterations
    if (rank == 0) printf("Performing %d warmup iterations...\n", WARMUP_ITERS);
    for (int i = 0; i < WARMUP_ITERS; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        
        ncclAllReduce(d_send_data, d_recv_data, MSG_SIZE / sizeof(float), 
                     ncclFloat, ncclSum, ncclComm, 0);
        
        check_cuda_error(cudaDeviceSynchronize(), "warmup sync", rank);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Warmup complete. Starting benchmark...\n\n");
    
    // Benchmark iterations
    double total_time_us = 0;
    double min_time_us = 1e9;
    double max_time_us = 0;
    
    for (int i = 0; i < N_ITERS; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        
        double t0 = get_time_us();
        
        // NCCL AllReduce for bidirectional inter-node GPU communication
        ncclAllReduce(d_send_data, d_recv_data, MSG_SIZE / sizeof(float), 
                     ncclFloat, ncclSum, ncclComm, 0);
        
        check_cuda_error(cudaDeviceSynchronize(), "benchmark sync", rank);
        
        double t1 = get_time_us();
        double iter_time = t1 - t0;
        total_time_us += iter_time;
        
        if (iter_time < min_time_us) min_time_us = iter_time;
        if (iter_time > max_time_us) max_time_us = iter_time;
        
        if (rank == 0 && (i + 1) % 10 == 0) {
            printf("Iteration %d/%d: %.2f ms (%.2f GB/s)\n", 
                   i + 1, N_ITERS, iter_time/1000.0,
                   (MSG_SIZE / (1024.0*1024.0*1024.0)) / (iter_time / 1000000.0));
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Calculate and display results
    if (rank == 0) {
        double avg_time_us = total_time_us / N_ITERS;
        double avg_bandwidth = (MSG_SIZE / (1024.0*1024.0*1024.0)) / (avg_time_us / 1000000.0);
        double min_bandwidth = (MSG_SIZE / (1024.0*1024.0*1024.0)) / (max_time_us / 1000000.0);
        double max_bandwidth = (MSG_SIZE / (1024.0*1024.0*1024.0)) / (min_time_us / 1000000.0);
        
        printf("\n=== Inter-Node GPU RDMA Results ===\n");
        printf("Transfer size: %.2f GB\n", (double)MSG_SIZE / (1UL<<30));
        printf("Iterations: %d\n", N_ITERS);
        printf("Average time: %.2f µs (%.3f ms)\n", avg_time_us, avg_time_us/1000.0);
        printf("Min time: %.2f µs (%.3f ms)\n", min_time_us, min_time_us/1000.0);
        printf("Max time: %.2f µs (%.3f ms)\n", max_time_us, max_time_us/1000.0);
        printf("Average bandwidth: %.2f GB/s\n", avg_bandwidth);
        printf("Min bandwidth: %.2f GB/s\n", min_bandwidth);
        printf("Max bandwidth: %.2f GB/s\n", max_bandwidth);
        printf("Method: NCCL AllReduce (inter-node GPU-Direct RDMA)\n");
        printf("Topology: Cross-node communication via network fabric\n");
        printf("Advantages: GPU-Direct RDMA, topology-aware routing, optimized for HPC\n");
        
        // Performance classification
        if (avg_bandwidth > 50.0) {
            printf("Performance: EXCELLENT (likely InfiniBand with GPU-Direct RDMA)\n");
        } else if (avg_bandwidth > 10.0) {
            printf("Performance: GOOD (high-speed Ethernet or degraded InfiniBand)\n");
        } else if (avg_bandwidth > 1.0) {
            printf("Performance: MODERATE (standard Ethernet)\n");
        } else {
            printf("Performance: LIMITED (check network configuration)\n");
        }
    }
    
    // Verify data integrity (simple check)
    uint8_t *h_recv_data = (uint8_t*)malloc(1024);
    check_cuda_error(cudaMemcpy(h_recv_data, d_recv_data, 1024, cudaMemcpyDeviceToHost), 
                     "data verification copy", rank);
    
    printf("Rank %d: Data verification - First few bytes: %d %d %d %d\n", 
           rank, h_recv_data[0], h_recv_data[1], h_recv_data[2], h_recv_data[3]);
    
    // Cleanup
    free(h_recv_data);
    check_cuda_error(cudaFree(d_send_data), "free send buffer", rank);
    check_cuda_error(cudaFree(d_recv_data), "free recv buffer", rank);
    check_nccl_error(ncclCommDestroy(ncclComm), "ncclCommDestroy", rank);
    
    MPI_Finalize();
    
    if (rank == 0) {
        printf("\nInter-node GPU RDMA benchmark completed successfully!\n");
    }
    
    return 0;
}
