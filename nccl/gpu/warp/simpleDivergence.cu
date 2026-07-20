#include <cuda_runtime.h>
#include "dbg.h"

/**
* branch efficiency： 83.33%
*/
__global__ void mathKernel1(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    if (tid % 2 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
    //log_info("tid is %d \n", tid);
}

/**
* best branch efficiency： 100%
*/
__global__ void mathKernel2(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    if ((tid / warpSize) % 2 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
    //log_info("tid is %d \n", tid);
}

/**
*  branch efficiency： 71.43%
*/
__global__ void mathKernel3(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    bool ipred = (tid % 2 == 0);

    if (ipred)
    {
        ia = 100.0f;
    }

    if (!ipred)
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
    //log_info("tid is %d \n", tid);
}

__global__ void mathKernel4(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    int itid = tid >> 5;

    if (itid & 0x01 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
    //log_info("tid is %d \n", tid);
}

__global__ void warmingup(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    if ((tid / warpSize) % 2 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
    //log_info("tid is %d \n", tid);
}

int main(int argc, char **argv) {
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    log_info("%s using Device %d: %s", argv[0], dev, deviceProp.name);

    int size = 64;
    int blockSize = 64;

    if (argc > 1) {
        blockSize = atoi(argv[1]);
    }

    if (argc > 2) {
        size = atoi(argv[2]);
    }

    log_info("Data size %d ", size);

    dim3 block(blockSize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    log_info("Execution Configure (block %d grid %d)", block.x, grid.x);

    float *d_C;
    size_t nBytes = size * sizeof(float);
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start, 0));
    warmingup<<<grid, block>>>(d_C);
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    float elapsedTime;
    CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    log_info("warmup <<<%d, %d>>> elapsed %f ms", grid.x, block.x, elapsedTime);

    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(start, 0));
    mathKernel1<<<grid, block>>>(d_C);
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    log_info("mathKernel <<<%d, %d>>> elapsed %f ms", grid.x, block.x, elapsedTime);
    CHECK(cudaGetLastError());

    CHECK(cudaEventRecord(start, 0));
    mathKernel2<<<grid, block>>>(d_C);
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    log_info("mathKernel2 <<<%d, %d>>> elapsed %f ms", grid.x, block.x, elapsedTime);
    CHECK(cudaGetLastError());

    CHECK(cudaEventRecord(start, 0));
    mathKernel3<<<grid, block>>>(d_C);
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    log_info("mathKernel3 <<<%d, %d>>> elapsed %f ms", grid.x, block.x, elapsedTime);
    CHECK(cudaGetLastError());

    CHECK(cudaEventRecord(start, 0));
    mathKernel4<<<grid, block>>>(d_C);
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    log_info("mathKernel4 <<<%d, %d>>> elapsed %f ms", grid.x, block.x, elapsedTime);
    CHECK(cudaGetLastError());

    CHECK(cudaFree(d_C));
    CHECK(cudaDeviceReset());
    return 0;
}
