#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

__global__ void vectorAddRuntime(const float *A, const float *B, float *C, unsigned int N) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const unsigned int N = 1024 * 1024;
    const size_t memSize = N * sizeof(float);

    std::vector<float> h_A(N), h_B(N), h_C_ptx(N), h_C_runtime(N);
    for (unsigned int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    cudaError_t cudaErr = cudaFree(0);
    if (cudaErr != cudaSuccess) {
        std::cerr << "cudaFree(0) failed: " << cudaGetErrorString(cudaErr) << std::endl;
        return -1;
    }

    CUresult res;
    CUdevice device;
    CUcontext context;
    res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        std::cerr << "cuInit failed" << std::endl;
        return -1;
    }
    res = cuDeviceGet(&device, 0);
    if (res != CUDA_SUCCESS) {
        std::cerr << "cuDeviceGet failed" << std::endl;
        return -1;
    }
    res = cuDevicePrimaryCtxRetain(&context, device);
    if (res != CUDA_SUCCESS) {
        std::cerr << "cuDevicePrimaryCtxRetain failed" << std::endl;
        return -1;
    }

    CUmodule module;
    res = cuModuleLoad(&module, "./kernel.ptx");
    if (res != CUDA_SUCCESS) {
        std::cerr << "Failed to load PTX module" << std::endl;
        return -1;
    }

    CUfunction ptxKernel;
    res = cuModuleGetFunction(&ptxKernel, module, "vectorAdd");
    if (res != CUDA_SUCCESS) {
        std::cerr << "Failed to get kernel function from PTX module" << std::endl;
        return -1;
    }

    CUdeviceptr d_A_driver, d_B_driver, d_C_driver;
    res = cuMemAlloc(&d_A_driver, memSize);
    res = cuMemAlloc(&d_B_driver, memSize);
    res = cuMemAlloc(&d_C_driver, memSize);

    res = cuMemcpyHtoD(d_A_driver, h_A.data(), memSize);
    res = cuMemcpyHtoD(d_B_driver, h_B.data(), memSize);

    unsigned int threadsPerBlock = 256;
    unsigned int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    void *ptxKernelParams[] = { &d_A_driver, &d_B_driver, &d_C_driver, (void *)&N };

    CUevent start_ptx, stop_ptx;
    cuEventCreate(&start_ptx, CU_EVENT_DEFAULT);
    cuEventCreate(&stop_ptx, CU_EVENT_DEFAULT);
    cuEventRecord(start_ptx, 0);

    res = cuLaunchKernel(ptxKernel,
                         blocks, 1, 1,
                         threadsPerBlock, 1, 1,
                         0,
                         0,
                         ptxKernelParams,
                         0);
    if (res != CUDA_SUCCESS) {
        std::cerr << "Failed to launch PTX kernel" << std::endl;
        return -1;
    }

    cuEventRecord(stop_ptx, 0);
    cuEventSynchronize(stop_ptx);
    float time_ptx;
    cuEventElapsedTime(&time_ptx, start_ptx, stop_ptx);
    std::cout << "PTX kernel execution time: " << time_ptx << " ms" << std::endl;

    res = cuMemcpyDtoH(h_C_ptx.data(), d_C_driver, memSize);

    float *d_A_runtime, *d_B_runtime, *d_C_runtime;
    cudaMalloc(&d_A_runtime, memSize);
    cudaMalloc(&d_B_runtime, memSize);
    cudaMalloc(&d_C_runtime, memSize);

    cudaMemcpy(d_A_runtime, h_A.data(), memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_runtime, h_B.data(), memSize, cudaMemcpyHostToDevice);

    cudaEvent_t start_rt, stop_rt;
    cudaEventCreate(&start_rt);
    cudaEventCreate(&stop_rt);
    cudaEventRecord(start_rt, 0);

    vectorAddRuntime<<<blocks, threadsPerBlock>>>(d_A_runtime, d_B_runtime, d_C_runtime, N);

    cudaEventRecord(stop_rt, 0);
    cudaEventSynchronize(stop_rt);
    float time_rt;
    cudaEventElapsedTime(&time_rt, start_rt, stop_rt);
    std::cout << "Runtime kernel execution time: " << time_rt << " ms" << std::endl;

    cudaMemcpy(h_C_runtime.data(), d_C_runtime, memSize, cudaMemcpyDeviceToHost);

    bool success = true;
    for (unsigned int i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C_ptx[i] - expected) > 1e-5 || fabs(h_C_runtime[i] - expected) > 1e-5) {
            std::cerr << "Mismatch at index " << i << std::endl;
            success = false;
            break;
        }
    }
    if (success) {
        std::cout << "Both kernels produced correct results." << std::endl;
    }

    cuMemFree(d_A_driver);
    cuMemFree(d_B_driver);
    cuMemFree(d_C_driver);
    cudaFree(d_A_runtime);
    cudaFree(d_B_runtime);
    cudaFree(d_C_runtime);
    cuEventDestroy(start_ptx);
    cuEventDestroy(stop_ptx);
    cudaEventDestroy(start_rt);
    cudaEventDestroy(stop_rt);
    cuModuleUnload(module);
    cuDevicePrimaryCtxRelease(device);

    return 0;
}
