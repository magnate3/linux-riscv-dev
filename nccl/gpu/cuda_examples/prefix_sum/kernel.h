#include <cuda_runtime.h>
#include <iostream>

#define NUM_THREADS_PER_BLOCK 512

typedef void (*launch_kernel_t)(int*, int*, const unsigned int);

void launch_prefix_sum_v0(int* d_input, int* d_output, const unsigned int n);