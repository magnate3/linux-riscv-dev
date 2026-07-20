#include "kernel.h"

template <unsigned int block_size>
__device__ float warp_reduce(float sum) {
    if (block_size >= 32) 
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 16); // 0 = 0 + 16, 1 = 1 + 17, 2 = 2 + 18, 3 = 3 + 19, etc.
    if (block_size >= 16)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);  // 0 = 0 + 8, 1 = 1 + 9, 2 = 2 + 10, 3 = 3 + 11, etc.
    if (block_size >= 8)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);  // 0 = 0 + 4, 1 = 1 + 5, 2 = 2 + 6, 3 = 3 + 7, etc.
    if (block_size >= 4)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);  // 0 = 0 + 2, 1 = 1 + 3, 2 = 2 + 4, 3 = 3 + 5, etc.
    if (block_size >= 2)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);  // 0 = 0 + 1, 1 = 1 + 2, 2 = 2 + 3, 3 = 3 + 4, etc.

    return sum; 
}

// based on v1, use warp_shuffle to reduce the number of block-level synchronizations (__syncthreads) and also reduce shared memory accesses
// each warp will use shuffle to reduce the data in shared memory
__global__ void reduce_v2(float *input, float *output, const unsigned int n) {
    int idx = blockIdx.x * (blockDim.x*NUM_PER_THREAD) + threadIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int warp_id = tid / 32;
    int lane_id = tid % 32;
    const unsigned int warp_size = 32;
    const unsigned int num_warps = NUM_THREADS_PER_BLOCK / warp_size;

    // each thread loads two elements from global memory and sum them, store in register
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < NUM_PER_THREAD; ++i) {
        sum += input[idx + i * NUM_THREADS_PER_BLOCK];
    }

    // the data of entire block is stored in register, use warp shuffle to reduce the sum of each warp
    sum = warp_reduce<warp_size>(sum);

    // each thread 0 in a warp will write its warp's partial result to shared memory
    // shared memory is used to store the partial results of each warp
    __shared__ float shared_data[num_warps];
    if (lane_id == 0) {
        shared_data[warp_id] = sum;
    }

    __syncthreads();
    sum = (tid < num_warps) ? shared_data[tid] : 0.0f;

    // call warp_reduce again to reduce the sum of all warps
    sum = warp_reduce<num_warps>(sum);

    // write the final result to global memory
    if (tid == 0) {
        output[bid] = sum;
    }
}

void launch_reduce_2(float *input, float *output, const unsigned int n){
    const unsigned int num_thread_block = n / (NUM_THREADS_PER_BLOCK * NUM_PER_THREAD);
    if (n % (NUM_THREADS_PER_BLOCK * NUM_PER_THREAD) != 0) {
        throw std::runtime_error("Input size must be a multiple of NUM_THREADS_PER_BLOCK * NUM_PER_THREAD");
    }
    dim3 grid_shape = {num_thread_block, 1, 1};
    dim3 block_shape = {NUM_THREADS_PER_BLOCK, 1, 1};

    reduce_v2<<<grid_shape, block_shape>>>(input, output, n);

}