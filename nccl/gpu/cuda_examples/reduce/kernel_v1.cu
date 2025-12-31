#include "kernel.h"

// each thread load two elements from global memory
// like loop unrolling, each thread loads more data
__global__ void reduce_v1(float *input, float *output, const unsigned int n) {
    __shared__ float shared_data[NUM_THREADS_PER_BLOCK];
    int idx = blockIdx.x * (blockDim.x*NUM_PER_THREAD) + threadIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    // load data to shared mem
    shared_data[tid] = 0.0f;
    #pragma unroll
    for (int i = 0; i < NUM_PER_THREAD; i++) {
        shared_data[tid] += input[idx + i * NUM_THREADS_PER_BLOCK];
    }

    // sync
    __syncthreads();

    for (int stride = NUM_THREADS_PER_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
        output[bid] = shared_data[tid];
}

void launch_reduce_1(float *input, float *output, const unsigned int n){
    const unsigned int num_thread_block = n / (NUM_THREADS_PER_BLOCK * NUM_PER_THREAD);
    dim3 grid_shape = {num_thread_block, 1, 1};
    dim3 block_shape = {NUM_THREADS_PER_BLOCK, 1, 1};

    reduce_v1<<<grid_shape, block_shape>>>(input, output, n);

}