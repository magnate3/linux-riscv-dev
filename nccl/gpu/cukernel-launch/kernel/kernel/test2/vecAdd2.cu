// vecAdd.cu
#include <cuda_runtime.h>
#include <stdio.h>

#if 0
//extern "C"
//{
// CUDA kernel that adds two vectors, each thread handles one element of c
__global__ void vecAdd2(double *a, double *b, double *c, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        c[id] = a[id] + b[id];
    }
}
//}
#else
__global__ void VecAdd(double* A, double* B, double* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
};

extern "C" void invoke_VecAdd(double* d_A, double* d_B, double* d_C, int N) {
    int threadsPerBlock = 256;
	int blocksPerGrid = N / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
};
#endif
