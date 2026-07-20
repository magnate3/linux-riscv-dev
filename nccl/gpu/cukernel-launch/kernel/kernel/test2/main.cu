#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#if 0
typedef void (*fp)(double *, double *, double *, int);
extern __global__ void vecAdd2(double *a, double *b, double *c, int n);
__device__ fp kernelPtr = vecAdd2;
typedef void (*fp)(double *, double *, double *, int);
extern "C" void invoke_VecAdd(double *a, double *b, double *c, int n);
fp kernelPtr = invoke_VecAdd;
#endif
int main() {
    int  N= 16;
    CUdevice dev;
    CUcontext ctx;
    CUmodule module;
    CUfunction kernel;
    CUresult err;

    // 1. 初始化 CUDA Driver
    err = cuInit(0);
    if (err != CUDA_SUCCESS) {
        printf("[ERROR] cuInit failed: %d\n", err);
        return -1;
    }

    err = cuDeviceGet(&dev, 0);
    if (err != CUDA_SUCCESS) {
        printf("[ERROR] cuDeviceGet failed: %d\n", err);
        return -1;
    }

    err = cuCtxCreate(&ctx, 0, dev);
    if (err != CUDA_SUCCESS) {
        printf("[ERROR] cuCtxCreate failed: %d\n", err);
        return -1;
    }
   // 2. 准备数据（CPU 端）
    float h_A[N], h_B[N], h_C[N];
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(i * 10);
    }

    // 3. 分配 GPU 内存
    CUdeviceptr d_A, d_B, d_C;
    err = cuMemAlloc(&d_A, N * sizeof(float));
    err = cuMemAlloc(&d_B, N * sizeof(float));
    err = cuMemAlloc(&d_C, N * sizeof(float));
    if (err != CUDA_SUCCESS) {
        printf("[ERROR] cuMemAlloc failed\n");
        return -1;
    }

    // 4. 拷贝输入数据到 GPU
    err = cuMemcpyHtoD(d_A, h_A, N * sizeof(float));
    err = cuMemcpyHtoD(d_B, h_B, N * sizeof(float));
    if (err != CUDA_SUCCESS) {
        printf("[ERROR] cuMemcpyHtoD failed\n");
        return -1;
    }

    // 5. 加载 PTX 模块（核函数）
    err = cuModuleLoad(&module, "vecAdd.cubin");
    if (err != CUDA_SUCCESS) {
        printf("[ERROR] cuModuleLoad failed: %d\n", err);
        return -1;
    }
    // 6. 获取核函数句柄
    err = cuModuleGetFunction(&kernel, module, "vecAdd");
    if (err != CUDA_SUCCESS) {
        printf("[ERROR] cuModuleGetFunction failed: %d\n", err);
        return -1;
    }
#if 0
    CUfunction cuFn;
    const void* ncclKernel=(void *)kernelPtr;
    //const void* ncclKernel=(void*)0x000000000403ae10;
    //const void* ncclKernel=NULL;
    cudaErr = cudaGetFuncBySymbol(&cuFn, ncclKernel);
    if (cudaErr != cudaSuccess) {
        printf("Failed to get nccl kernel function \n");
        //return -1;
    }
    else
    {
        printf("get nccl kernel function \n");
    }
#endif
    // 7. 设置 Grid 和 Block
    int gridx=(N + 16 - 1) / 16;

    // 8. 启动核函数
    void* args[] = { &d_A, &d_B, &d_C, &N };
    err = cuLaunchKernel(kernel,
                         gridx, 1, 1,     // grid: x 维度
                         16, 1, 1,        // block: x 维度
                         0,               // shared memory
                         0,          // CUDA stream (NULL)
                         args,            // 参数数组
                         NULL);           // extra

    if (err != CUDA_SUCCESS) {
        printf("[ERROR] cuLaunchKernel failed: %d\n", err);
        return -1;
    }

    // 9. 同步并拷贝结果回 CPU
    cuCtxSynchronize();
    err = cuMemcpyDtoH(h_C, d_C, N * sizeof(float));
    if (err != CUDA_SUCCESS) {
        printf("[ERROR] cuMemcpyDtoH failed\n");
        return -1;
    }
    // 10. 打印部分结果（前 10 个）
    printf("Result C (first 10 elements): ");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_C[i]);
    }
    printf("\n");

    // 11. 释放资源
    cuMemFree(d_A);
    cuMemFree(d_B);
    cuMemFree(d_C);
    cuModuleUnload(module);
    cuCtxDestroy(ctx);

    return 0;
}
