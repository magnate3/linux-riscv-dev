// vecAdd.cu
#include <cuda_runtime.h>
#include <stdio.h>


extern "C"
{
// CUDA kernel that adds two vectors, each thread handles one element of c
__global__ void vecAdd(double *a, double *b, double *c, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        c[id] = a[id] + b[id];
    }
}
}
