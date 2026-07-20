// concurrentKernels.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void kernelMultiply(int *data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) data[idx] *= 2;
}

__global__ void kernelAdd(int *data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) data[idx] += 5;
}

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

int main() {
    int N = 1 << 20;
    size_t size = N * sizeof(int);

    int *h_data1, *h_data2;
    CUDA_CHECK(cudaMallocHost(&h_data1, size));
    CUDA_CHECK(cudaMallocHost(&h_data2, size));
    for (int i = 0; i < N; i++) {
        h_data1[i] = i;
        h_data2[i] = i;
    }

    int *d_data1, *d_data2;
    CUDA_CHECK(cudaMalloc(&d_data1, size));
    CUDA_CHECK(cudaMalloc(&d_data2, size));
    CUDA_CHECK(cudaMemcpy(d_data1, h_data1, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_data2, h_data2, size, cudaMemcpyHostToDevice));

    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    cudaEvent_t start1, stop1, start2, stop2;
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&stop1));
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&stop2));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    CUDA_CHECK(cudaEventRecord(start1, stream1));
    kernelMultiply<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_data1, N);
    CUDA_CHECK(cudaEventRecord(stop1, stream1));

    CUDA_CHECK(cudaEventRecord(start2, stream2));
    kernelAdd<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_data2, N);
    CUDA_CHECK(cudaEventRecord(stop2, stream2));

    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));

    float time1, time2;
    CUDA_CHECK(cudaEventElapsedTime(&time1, start1, stop1));
    CUDA_CHECK(cudaEventElapsedTime(&time2, start2, stop2));
    printf("Multiply Kernel Time: %f ms\n", time1);
    printf("Add Kernel Time: %f ms\n", time2);

    CUDA_CHECK(cudaMemcpy(h_data1, d_data1, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_data2, d_data2, size, cudaMemcpyDeviceToHost));

    printf("Multiply (first 5):\n");
    for (int i = 0; i < 5; i++) printf("%d ", h_data1[i]);
    printf("\nAdd (first 5):\n");
    for (int i = 0; i < 5; i++) printf("%d ", h_data2[i]);
    printf("\n");

    CUDA_CHECK(cudaFree(d_data1)); CUDA_CHECK(cudaFree(d_data2));
    CUDA_CHECK(cudaFreeHost(h_data1)); CUDA_CHECK(cudaFreeHost(h_data2));
    CUDA_CHECK(cudaStreamDestroy(stream1)); CUDA_CHECK(cudaStreamDestroy(stream2));
    CUDA_CHECK(cudaEventDestroy(start1)); CUDA_CHECK(cudaEventDestroy(stop1));
    CUDA_CHECK(cudaEventDestroy(start2)); CUDA_CHECK(cudaEventDestroy(stop2));

    return 0;
}
