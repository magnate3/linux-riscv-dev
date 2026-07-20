#include <stdio.h>
#include <cuda.h>

#define BLOCK_NUM 1
#define THREAD_NUM 256

__global__ void addVector(int* a, int* b, int* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    c[tid] = a[tid] + b[tid];
}

void init_vector(int* x, int n) {
    for (int i=0; i<n; i++) {
        x[i] = i;
    }
}

int main() {
    int *host_a, *host_b, *host_c;
    int *dev_a, *dev_b, *dev_c; 
    int n = THREAD_NUM * sizeof(int);

    // Allocate memory on host
    host_a = (int*)malloc(n);
    host_b = (int*)malloc(n);
    host_c = (int*)malloc(n);

    // Initialize vectors to sequences
    init_vector(host_a, THREAD_NUM);
    init_vector(host_b, THREAD_NUM);
    init_vector(host_c, THREAD_NUM);

    // Allocate memory on GPU
    cudaMalloc(&dev_a, n);
    cudaMalloc(&dev_b, n);
    cudaMalloc(&dev_c, n);
    
    // Copy data from host to device
    cudaMemcpy(dev_a, host_a, n, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, n, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, host_c, n, cudaMemcpyHostToDevice);
    
    // Run kernel on GPU
    addVector<<<BLOCK_NUM, THREAD_NUM>>>(dev_a, dev_b, dev_c, n);

    // Copy data from GPU to host
    cudaMemcpy(host_c, dev_c, n, cudaMemcpyDeviceToHost);

    // Print result
    for (int i=0; i<THREAD_NUM; i++) {
        printf("%d ", host_c[i]);
    }
    printf("\n");

    // Free up GPU 
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    
    // Free up host
    free(host_a);
    free(host_b);
    free(host_c);

    return 0;
}