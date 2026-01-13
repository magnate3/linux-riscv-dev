#include <iostream>
#include <cuda_runtime.h>

// CUDA 核函数，用于执行矩阵乘法
__global__ void matrixMulKernel(float* A, float* B, float* C, long long N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 矩阵乘法的主函数
void matrixMultiply(float* A, float* B, float* C, long long N) {
    long long size = N * N * sizeof(float);

    // 分配 GPU 内存
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // 将矩阵数据从主机内存复制到设备内存
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // 定义 CUDA 线程块和网格大小
    dim3 blockDim(16, 16);  // 每个线程块包含 16x16 个线程
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    // 创建 CUDA 事件用于测量时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 记录开始事件
    cudaEventRecord(start, 0);

    // 调用 CUDA 核函数执行矩阵乘法
    matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    // 记录结束事件
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);  // 等待事件完成

    // 计算执行时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Matrix multiplication took " << milliseconds << " ms." << std::endl;

    // 将结果从设备内存复制回主机内存
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 销毁 CUDA 事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    long long N = 1024 * 20;  // 矩阵大小 N x N
    long long size = N * N * sizeof(float);

    std::cout << "矩阵占用内存：" << size / (1024 * 1024) << " MB  " << size / (1024 * 1024 * 1024) << " GB" << std::endl;

    // 分配主机内存
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // 初始化矩阵 A 和 B
    for (long long i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    // 执行矩阵乘法
    matrixMultiply(h_A, h_B, h_C, N);

    // 打印部分结果
    std::cout << "C[0][0] = " << h_C[0] << std::endl;

    // 释放主机内存
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
