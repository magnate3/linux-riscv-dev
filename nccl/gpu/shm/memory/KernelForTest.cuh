#pragma once
#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>
#include <cstdio>

#define BLOCK_SIZE 16  // CUDA 线程块的大小

// CUDA 核函数，计算矩阵乘法 C = A * B
__global__ void matrixMultiply(const float* A, const float* B, float* C, size_t N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 当前线程负责的行
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 当前线程负责的列

    float sum = 0.0f;

    if (row < N && col < N) {
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * N + col];  // 计算对应元素
        }
        C[row * N + col] = sum;  // 将结果写入矩阵 C
    }
}

// 矩阵乘法的主机端接口函数
void matrixMultiplyHost(const float* A, const float* B, float* C, int N) {
    float *d_A, *d_B, *d_C;

    // 分配设备端内存
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

    // 将数据从主机传输到设备
    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // 定义 CUDA 网格和线程块
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // 启动 CUDA 核函数
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 同步设备
    cudaDeviceSynchronize();

    // 将结果从设备传输回主机
    cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


__global__ void SingleVecKernel_int64_t(int64_t* p_vec, int N, int64_t number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        p_vec[idx] = number;
    }
}

__global__ void SingleVecAddKernel_int64_t(int64_t* p_vec, int N, int64_t number){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        p_vec[idx] += number;
    }
}
