#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

int main(void) {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount < 2) {
        printf("This program requires at least 2 GPUs.\n");
        return 1;
    }

    int N = 1 << 20;       // Total elements (1M)
    int halfN = N / 2;     // Elements per GPU
    size_t totalSize = N * sizeof(float);
    size_t halfSize = halfN * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(totalSize);
    float *h_B = (float*)malloc(totalSize);
    float *h_C = (float*)malloc(totalSize);

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Device pointers for GPU 0
    float *d_A0, *d_B0, *d_C0;
    cudaSetDevice(0);
    cudaMalloc((void**)&d_A0, halfSize);
    cudaMalloc((void**)&d_B0, halfSize);
    cudaMalloc((void**)&d_C0, halfSize);

    // Device pointers for GPU 1
    float *d_A1, *d_B1, *d_C1;
    cudaSetDevice(1);
    cudaMalloc((void**)&d_A1, halfSize);
    cudaMalloc((void**)&d_B1, halfSize);
    cudaMalloc((void**)&d_C1, halfSize);

    // Copy data to each GPU
    cudaSetDevice(0);
    cudaMemcpy(d_A0, h_A, halfSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B0, h_B, halfSize, cudaMemcpyHostToDevice);

    cudaSetDevice(1);
    cudaMemcpy(d_A1, h_A + halfN, halfSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B1, h_B + halfN, halfSize, cudaMemcpyHostToDevice);

    // Create CUDA events for timing on each GPU
    cudaEvent_t start0, stop0;
    cudaEventCreate(&start0);
    cudaEventCreate(&stop0);

    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel on GPU 0
    cudaSetDevice(0);
    cudaEventRecord(start0, 0);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A0, d_B0, d_C0, halfN);
    cudaEventRecord(stop0, 0);

    // Launch kernel on GPU 1
    cudaSetDevice(1);
    cudaEventRecord(start1, 0);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A1, d_B1, d_C1, halfN);
    cudaEventRecord(stop1, 0);

    // Synchronize both GPUs
    cudaSetDevice(0);
    cudaEventSynchronize(stop0);
    cudaSetDevice(1);
    cudaEventSynchronize(stop1);

    // Measure elapsed time on each GPU
    float elapsedTime0, elapsedTime1;
    cudaSetDevice(0);
    cudaEventElapsedTime(&elapsedTime0, start0, stop0);
    cudaSetDevice(1);
    cudaEventElapsedTime(&elapsedTime1, start1, stop1);

    printf("GPU 0 vector addition took: %f ms\n", elapsedTime0);
    printf("GPU 1 vector addition took: %f ms\n", elapsedTime1);

    // The overall runtime is approximately the maximum of the two times
    float overallTime = (elapsedTime0 > elapsedTime1) ? elapsedTime0 : elapsedTime1;
    
    printf("Overall runtime (max of both GPUs): %f ms\n", overallTime);

    // Copy results back to host
    cudaSetDevice(0);
    cudaMemcpy(h_C, d_C0, halfSize, cudaMemcpyDeviceToHost);
    cudaSetDevice(1);
    cudaMemcpy(h_C + halfN, d_C1, halfSize, cudaMemcpyDeviceToHost);

    // (Optional) Verify the results
    for (int i = 0; i < N; i++) {
        if (fabs(h_C[i] - 3.0f) > 1e-5) {
            printf("Error at index %d: %f != 3.0\n", i, h_C[i]);
            break;
        }
    }

    // Print h_C array (first 10 elements)
    printf("h_C array (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_C[i]);
    }
    printf("\n");

    // Cleanup GPU 0
    cudaSetDevice(0);
    cudaFree(d_A0);
    cudaFree(d_B0);
    cudaFree(d_C0);
    cudaEventDestroy(start0);
    cudaEventDestroy(stop0);

    // Cleanup GPU 1
    cudaSetDevice(1);
    cudaFree(d_A1);
    cudaFree(d_B1);
    cudaFree(d_C1);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);

    // Cleanup host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
