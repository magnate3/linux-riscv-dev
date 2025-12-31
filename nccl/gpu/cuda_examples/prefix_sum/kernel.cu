#include "kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS) // pad 1 element for every 32 elements
#define NUM_PER_THREAD 2

__global__ void prefix_sum_v0(int* d_in, int* d_prefix_sum, int* d_sum, int n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    const unsigned int NUM_PER_BLOCK = NUM_PER_THREAD * NUM_THREADS_PER_BLOCK;
    // Calculate the size of shared memory needed for conflict-free access
    // conflict-free access requires padding 1 element for every 32 elements
    const unsigned int SHARED_SIZE = NUM_PER_BLOCK + CONFLICT_FREE_OFFSET(NUM_PER_BLOCK - 1);

    __shared__ int shared_data[SHARED_SIZE];
    int leaf_num = NUM_PER_BLOCK;
    
    // load data into shared memory
    #pragma unroll
    for (int i = 0; i < NUM_PER_THREAD; i++) {
        int global_idx = bid * NUM_PER_BLOCK + i * NUM_THREADS_PER_BLOCK + tid;
        int local_idx = i * NUM_THREADS_PER_BLOCK + tid;
        int local_offset = CONFLICT_FREE_OFFSET(local_idx);
        shared_data[local_idx + local_offset] = (global_idx < n) ? d_in[global_idx] : 0;
    }
    __syncthreads();

    // exclusive prefix sum (up-sweep phase)
    int offset = 1;
    for (int d = leaf_num >> 1; d > 0; d >>= 1) {
        if (tid < d) {
            int in_idx = offset * (2 * tid + 1) - 1;
            int out_idx = offset * (2 * tid + 2) - 1;
            int in_offset = CONFLICT_FREE_OFFSET(in_idx);
            int out_offset = CONFLICT_FREE_OFFSET(out_idx);
            shared_data[out_idx + out_offset] += shared_data[in_idx + in_offset];
        }
        offset <<= 1;
        __syncthreads();
    }

    // last element is the sum of this block, store it and then zero it out
    if (tid == 0) {
        int last_idx = leaf_num - 1 + CONFLICT_FREE_OFFSET(leaf_num - 1);
        d_sum[bid] = shared_data[last_idx];
        shared_data[last_idx] = 0;
    }
    __syncthreads();

    // down-sweep phase
    for (int d = 1; d < leaf_num; d <<= 1) {
        offset >>= 1;
        if (tid < d) {
            int in_idx = offset * (2 * tid + 1) - 1;
            int out_idx = offset * (2 * tid + 2) - 1;
            int in_offset = CONFLICT_FREE_OFFSET(in_idx);
            int out_offset = CONFLICT_FREE_OFFSET(out_idx);
            int tmp = shared_data[in_idx + in_offset];
            shared_data[in_idx + in_offset] = shared_data[out_idx + out_offset];
            shared_data[out_idx + out_offset] += tmp;
        }
        __syncthreads();
    }

    // write back the results to global memory
    #pragma unroll
    for (int i = 0; i < NUM_PER_THREAD; i++) {
        int global_idx = bid * NUM_PER_BLOCK + i * NUM_THREADS_PER_BLOCK + tid;
        int local_idx = i * NUM_THREADS_PER_BLOCK + tid;
        int local_offset = CONFLICT_FREE_OFFSET(local_idx);
        if (global_idx < n) {
            d_prefix_sum[global_idx] = shared_data[local_idx + local_offset];
        }
    }
}

__global__ void add_kernel(int* d_prefix_sum, int* d_prefix_sum_for_sum, int n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    const unsigned int NUM_PER_BLOCK = NUM_PER_THREAD * NUM_THREADS_PER_BLOCK;

    int prefix_sum_for_block = d_prefix_sum_for_sum[bid];
    #pragma unroll
    for (int i = 0; i < NUM_PER_THREAD; i++) {
        int idx = bid * NUM_PER_BLOCK + i * NUM_THREADS_PER_BLOCK + tid;
        if (idx < n) {
            d_prefix_sum[idx] += prefix_sum_for_block;
        }
    }
}

void launch_prefix_sum_v0(int* d_in, int* d_prefix_sum, const unsigned int N) {
    unsigned int num_blocks = N / (NUM_PER_THREAD * NUM_THREADS_PER_BLOCK);
    if (N % (NUM_PER_THREAD * NUM_THREADS_PER_BLOCK) != 0) {
        num_blocks++;
    }

    int *d_sum, *d_prefix_sum_for_sum;
    cudaMalloc(&d_sum, num_blocks * sizeof(int));
    cudaMalloc(&d_prefix_sum_for_sum, num_blocks * sizeof(int));
    cudaMemset(d_sum, 0, num_blocks * sizeof(int));

    prefix_sum_v0<<<num_blocks, NUM_THREADS_PER_BLOCK>>>(d_in, d_prefix_sum, d_sum, N);
    cudaDeviceSynchronize();

    // recursively get the prefix sum of the block sums until we have only one block left
    if (num_blocks > 1) {
        launch_prefix_sum_v0(d_sum, d_prefix_sum_for_sum, num_blocks);
        add_kernel<<<num_blocks, NUM_THREADS_PER_BLOCK>>>(d_prefix_sum, d_prefix_sum_for_sum, N);
        cudaDeviceSynchronize();
    }

    cudaFree(d_sum);
    cudaFree(d_prefix_sum_for_sum);
}
