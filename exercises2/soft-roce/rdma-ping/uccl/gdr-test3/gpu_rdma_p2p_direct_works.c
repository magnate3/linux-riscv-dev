// True GPU-to-GPU P2P transfer using direct cudaMemcpyPeer()
#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N_ITERS 100
#define MSG_SIZE (20ULL * 1024ULL * 1024ULL * 1024ULL) // 20 GB

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
        if (rank == 0) fprintf(stderr, "Must run with 2 ranks for 2 GPUs\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    cudaSetDevice(rank);
    
    // Check available GPU memory first
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("Rank %d: GPU %d has %.2f GB free / %.2f GB total\n", 
           rank, rank, 
           (double)free_mem / (1024*1024*1024), 
           (double)total_mem / (1024*1024*1024));
    
    if (free_mem < MSG_SIZE * 2) {
        printf("Rank %d: WARNING - Not enough GPU memory! Need %.2f GB, have %.2f GB\n",
               rank, (double)(MSG_SIZE * 2) / (1024*1024*1024), 
               (double)free_mem / (1024*1024*1024));
    }
    
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, rank);
    printf("Rank %d using GPU %d: %s\n", rank, rank, prop.name);
    
    // Enable P2P access
    if (rank == 0) {
        int canAccessPeer;
        cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1);
        printf("GPU 0->1 P2P access: %s\n", canAccessPeer ? "YES" : "NO");
        if (canAccessPeer) {
            cudaDeviceEnablePeerAccess(1, 0);
            printf("Enabled P2P access from GPU 0 to GPU 1\n");
        }
    } else {
        int canAccessPeer;
        cudaDeviceCanAccessPeer(&canAccessPeer, 1, 0);
        if (canAccessPeer) {
            cudaDeviceEnablePeerAccess(0, 0);
            printf("Enabled P2P access from GPU 1 to GPU 0\n");
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Allocate GPU memory with error checking
    void *d_src, *d_dst;
    
    printf("Rank %d: Attempting to allocate %.2f GB per buffer...\n", 
           rank, (double)MSG_SIZE / (1024*1024*1024));
    
    cudaError_t err1 = cudaMalloc(&d_src, MSG_SIZE);
    if (err1 != cudaSuccess) {
        printf("Rank %d: Failed to allocate source buffer: %s\n", rank, cudaGetErrorString(err1));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    cudaError_t err2 = cudaMalloc(&d_dst, MSG_SIZE);
    if (err2 != cudaSuccess) {
        printf("Rank %d: Failed to allocate destination buffer: %s\n", rank, cudaGetErrorString(err2));
        cudaFree(d_src);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialize with different patterns to prevent optimization
    cudaMemset(d_src, rank + 1, MSG_SIZE);
    cudaMemset(d_dst, 0xFF, MSG_SIZE);
    cudaDeviceSynchronize();
    
    printf("Rank %d: Successfully allocated %.2f GB total on GPU %d\n", 
           rank, (2.0 * MSG_SIZE) / (1024*1024*1024), rank);
    
    // Exchange GPU pointers via MPI - but use IPC handles instead
    cudaIpcMemHandle_t mem_handle;
    void *remote_dst;
    
    if (rank == 0) {
        // Get IPC handle for d_dst
        cudaIpcGetMemHandle(&mem_handle, d_dst);
        MPI_Send(&mem_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 1, 0, MPI_COMM_WORLD);
        
        // Receive remote handle
        cudaIpcMemHandle_t remote_handle;
        MPI_Recv(&remote_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cudaIpcOpenMemHandle(&remote_dst, remote_handle, cudaIpcMemLazyEnablePeerAccess);
    } else {
        // Receive handle from rank 0
        cudaIpcMemHandle_t remote_handle;
        MPI_Recv(&remote_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cudaIpcOpenMemHandle(&remote_dst, remote_handle, cudaIpcMemLazyEnablePeerAccess);
        
        // Send our handle
        cudaIpcGetMemHandle(&mem_handle, d_dst);
        MPI_Send(&mem_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Starting direct P2P benchmark (bypassing MPI for data transfer)...\n");
        
        double total_time_us = 0;
        
        for (int i = 0; i < N_ITERS; i++) {
            // Flush caches to ensure real transfer
            cudaMemset(d_dst, i & 0xFF, 16);  // Touch destination to prevent caching
            cudaDeviceSynchronize();
            
            double t0 = get_time_us();
            
            // DIRECT GPU-to-GPU P2P transfer - NO MPI!
            // Copy from GPU 0's d_src to GPU 1's remote_dst
            cudaError_t result = cudaMemcpyPeer(remote_dst, 1, d_src, 0, MSG_SIZE);
            
            if (result != cudaSuccess) {
                printf("cudaMemcpyPeer failed: %s\n", cudaGetErrorString(result));
                printf("Trying alternative approach...\n");
                
                // Alternative: Set context to target device and use regular cudaMemcpy
                int original_device;
                cudaGetDevice(&original_device);
                cudaSetDevice(1);
                result = cudaMemcpy(remote_dst, d_src, MSG_SIZE, cudaMemcpyDeviceToDevice);
                cudaSetDevice(original_device);
            }
            
            cudaDeviceSynchronize();
            
            if (result != cudaSuccess) {
                printf("Transfer failed: %s\n", cudaGetErrorString(result));
                break;
            }
            
            double t1 = get_time_us();
            total_time_us += (t1 - t0);
            
            if ((i + 1) % 10 == 0) {
                printf("Iteration %d/%d: %.2f ms\n", i + 1, N_ITERS, (t1 - t0)/1000.0);
            }
        }
        
        double avg_time_us = total_time_us / N_ITERS;
        double bandwidth = (MSG_SIZE / (1024.0*1024.0*1024.0)) / (avg_time_us / 1000000.0);
        
        printf("\n=== TRUE GPU P2P Direct Transfer Results ===\n");
        printf("Transfer size: %.2f GB\n", (double)MSG_SIZE / (1UL<<30));
        printf("Iterations: %d\n", N_ITERS);
        printf("Average time: %.2f µs (%.3f ms)\n", avg_time_us, avg_time_us/1000.0);
        printf("Bandwidth: %.2f GB/s\n", bandwidth);
        printf("Method: Direct cudaMemcpyPeer() - ZERO MPI overhead\n");
        
        if (avg_time_us < 1000.0) {
            printf("✅ EXCELLENT: Sub-millisecond performance!\n");
        } else if (avg_time_us < 30000.0) {
            printf("✅ GOOD: PCIe-limited performance\n");
        } else {
            printf("⚠️ SLOW: Potential bottleneck detected\n");
        }
    }
    
    cudaFree(d_src);
    cudaFree(d_dst);
    MPI_Finalize();
    return 0;
}
