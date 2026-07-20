#include <cstdio>
#include <cuda.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// ============================================================================
//                           DEVICE GLOBAL STATE
// ============================================================================
__device__ int d_counter = 0;


// ============================================================================
//                     STRATEGY 0: ATOMIC-BASED GLOBAL BARRIER
// ============================================================================
__device__ __forceinline__
void strategy_atomic_barrier(int total_blocks) {
    // Signal arrival
    atomicAdd(&d_counter, 1);

    // Wait until all blocks arrive
    while (atomicAdd(&d_counter, 0) < total_blocks) {
        // busy-wait
    }
    __syncthreads();
}


// ============================================================================
//                     STRATEGY 1: COOPERATIVE GROUPS BARRIER
// ============================================================================
__device__ __forceinline__
void strategy_cg_barrier() {
    cg::grid_group g = cg::this_grid();
    g.sync();   // true global synchronization
}


// ============================================================================
//                  ABSTRACT DISPATCH FUNCTION FOR STRATEGIES
// ============================================================================
__device__ __forceinline__
void comm_strategy(int strategy_id, int total_blocks) {
    switch (strategy_id) {
        case 0:
            strategy_atomic_barrier(total_blocks);
            break;
        case 1:
            strategy_cg_barrier();
            break;
        default:
            strategy_atomic_barrier(total_blocks);
    }
}


// ============================================================================
//                         COMMUNICATION BENCHMARK KERNEL
// ============================================================================
__global__ void comm_kernel(
    int strategy_id,
    int total_blocks,
    unsigned long long* timing_out)
{
    __shared__ unsigned long long t0, t1;

    if (threadIdx.x == 0)
        t0 = clock64();
    __syncthreads();

    // ---- abstract communication call ----
    comm_strategy(strategy_id, total_blocks);

    __syncthreads();
    if (threadIdx.x == 0) {
        t1 = clock64();
        timing_out[blockIdx.x] = t1 - t0;
    }
}


// ============================================================================
//                 HOST LAUNCHER: ATOMIC BARRIER STRATEGY
// ============================================================================
void launch_atomic(int num_sms) {
    printf("\n=== Atomic Strategy ===\n");

    unsigned long long *t_d, *t_h;
    cudaMalloc(&t_d, num_sms * sizeof(unsigned long long));
    t_h = (unsigned long long*)malloc(num_sms * sizeof(unsigned long long));

    comm_kernel<<<num_sms, 1024>>>(0, num_sms, t_d);
    cudaDeviceSynchronize();

    cudaMemcpy(t_h, t_d, num_sms * sizeof(unsigned long long),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_sms; i++)
        printf("Block %d: %llu cycles\n", i, t_h[i]);

    cudaFree(t_d);
    free(t_h);
}


// ============================================================================
//               HOST LAUNCHER: COOPERATIVE GROUPS GRID SYNC
// ============================================================================
void launch_cg(int num_sms) {
    printf("\n=== Cooperative Groups Strategy ===\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    if (!prop.cooperativeLaunch) {
        printf("This GPU does NOT support cooperative launches.\n");
        return;
    }

    unsigned long long *t_d, *t_h;
    cudaMalloc(&t_d, num_sms * sizeof(unsigned long long));
    t_h = (unsigned long long*)malloc(num_sms * sizeof(unsigned long long));

    int strategy = 1;
    void *args[] = { &strategy, &num_sms, &t_d };

    dim3 grid(num_sms), block(1024);

    cudaLaunchCooperativeKernel(
        (void*)comm_kernel, grid, block, args
    );
    cudaDeviceSynchronize();

    cudaMemcpy(t_h, t_d, num_sms * sizeof(unsigned long long),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_sms; i++)
        printf("Block %d: %llu cycles\n", i, t_h[i]);

    cudaFree(t_d);
    free(t_h);
}


// ============================================================================
//                                   MAIN
// ============================================================================
int main() {
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);

    printf("Detected %d SMs\n", num_sms);
    printf("Launching kernels with <%d blocks, 1024 threads>\n", num_sms);

    // Run both strategies
    launch_atomic(num_sms);
    launch_cg(num_sms);

    return 0;
}
