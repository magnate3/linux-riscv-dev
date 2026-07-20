#include "kernel.h"

// naive implementation of reduction kernel
// each ghread load one element from global memory
__global__ void reduce_v0(float *input, float *output, const unsigned int n) {
    __shared__ float shared_data[NUM_THREADS_PER_BLOCK];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    // load data to shared mem
    shared_data[tid] = input[idx];

    // sync
    __syncthreads();

    for (int stride = NUM_THREADS_PER_BLOCK / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
        output[bid] = shared_data[tid];
}

void launch_reduce_0(float *input, float *output, const unsigned int n){
    const unsigned int num_thread_block = n / NUM_THREADS_PER_BLOCK;
    dim3 grid_shape = {num_thread_block, 1, 1};
    dim3 block_shape = {NUM_THREADS_PER_BLOCK, 1, 1};

    reduce_v0<<<grid_shape, block_shape>>>(input, output, n);

}