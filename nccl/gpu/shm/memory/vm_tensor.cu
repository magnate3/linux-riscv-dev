#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>

// 错误检查宏
#define CHECK_CU(call) { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char* errMsg; \
        cuGetErrorString(err, &errMsg); \
        std::cerr << "CUDA error: " << errMsg << " at line " << __LINE__ << std::endl; \
        exit(err); \
    } \
}

void vmmMalloc(void**, long long size, size_t granularity);
void vmmFree(void*);

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
void matrixMultiply(float* A, float* B, float* C, long long N, size_t granularity) {
    int current_device;
    cuCtxGetDevice(&current_device);
    std::cout << "当前设备ID(matrixMultiply)：" << current_device << std::endl;

    long long size = N * N * sizeof(float);

    // 分配 GPU 内存
    float *d_A, *d_B, *d_C;
    vmmMalloc((void**)&d_A, size, granularity );
    vmmMalloc((void**)&d_B, size, granularity );
    vmmMalloc((void**)&d_C, size, granularity );

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
    vmmFree(d_A);
    vmmFree(d_B);
    vmmFree(d_C);

    // 销毁 CUDA 事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/* 获取内存分配所需的最小分配粒度 */
size_t getGranularitySize()
{
    static size_t granularity = 1;

    if(granularity == 1) {
        int current_device = 0;
        CHECK_CU(cuCtxGetDevice(&current_device));  // 获取当前上下文的设备ID

        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;  // 固定内存
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = current_device;  // 设备ID

        CHECK_CU(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    }

    return granularity;
}


void vmmMalloc(void** p_of_ptr, long long size, size_t granularity) {
    size_t device_free, device_total;
    cudaMemGetInfo(&device_free, &device_total);
    std::cout << "设备可用内存：" << device_free / (1024 * 1024) << " MB" << std::endl;

    // reserve一段虚拟内存
    CUdeviceptr ptr;
    CHECK_CU(cuMemAddressReserve(&ptr, size, 0, 0, 0));

    int current_device = 0;
    cuCtxGetDevice(&current_device);
    std::cout << "当前设备ID(vmmMalloc)：" << current_device << std::endl;

    // 循环获取物理块，映射
    long long num_phy_blocks = size / granularity;              // TODO: 这里并不鲁棒
    for(long long i = 0; i < 2 * num_phy_blocks; ++i) { 
        // 获取物理块
        CUmemGenericAllocationHandle alloc_handle;      // 固定内存块句柄
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;      
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;   
        prop.location.id = current_device;               // 设备ID
        CHECK_CU(cuMemCreate(&alloc_handle, granularity, &prop, 0));   // 创建固定内存块

        if (i % 2 == 0){
            CUdeviceptr ppp = ptr + (i / 2) * granularity;
            // 映射到虚拟内存
            CHECK_CU(cuMemMap(ppp, granularity, 0, alloc_handle, 0));
            // 设置权限
            CUmemAccessDesc accessDesc = {};
            accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;     // 常量，用于指示内存位置的类型是设备（即 GPU 设备内存）
            accessDesc.location.id = current_device;
            accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;      // 常量，一个枚举值，表示内存的访问权限是“可读写”的
            CHECK_CU(cuMemSetAccess((CUdeviceptr)ppp, granularity, &accessDesc, 1));   // 设置内存访问权限
        }
    }

    *p_of_ptr = (void*)ptr;  // 返回虚拟内存地址
}

// TODO: 释放虚拟内存
void vmmFree(void* ptr){
    // CHECK_CU(cuMemAddressFree((CUdeviceptr)ptr, 0));  // 释放虚拟内存
}

int main() {
    // 初始化 CUDA
    CUdevice cuDevice;
    CUcontext cuContext;
    CHECK_CU(cuInit(0));  // 初始化 CUDA 驱动 API
    CHECK_CU(cuDeviceGet(&cuDevice, 3));  // 获取设备
    CHECK_CU(cuCtxCreate(&cuContext, 0, cuDevice));  // 创建上下文

    
    const size_t granularity = getGranularitySize();

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
    matrixMultiply(h_A, h_B, h_C, N, granularity);

    // 打印部分结果
    std::cout << "C[0][0] = " << h_C[0] << std::endl;

    // 释放主机内存
    free(h_A);
    free(h_B);
    free(h_C);

    // 销毁上下文
    CHECK_CU(cuCtxDestroy(cuContext));

    return 0;
}

