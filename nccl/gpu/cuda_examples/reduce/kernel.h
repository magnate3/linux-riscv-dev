#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define NUM_THREADS_PER_BLOCK 256
#define NUM_PER_THREAD 8

typedef void (*launch_kernel_t)(float*, float*, const unsigned int);

void launch_reduce_0(float *input, float *output, const unsigned int n);

void launch_reduce_1(float *input, float *output, const unsigned int n);

void launch_reduce_2(float *input, float *output, const unsigned int n);