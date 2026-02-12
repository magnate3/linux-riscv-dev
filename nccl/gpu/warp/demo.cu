// 0_creatingErrors.cu
#include <cuda_runtime.h>
#include <stdio.h>
 
__global__ void kernelA(int * globalArray){
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
 
    // If the problem is small or if printing a subset of the problem 
    // (inside conditional expression, etc...). 
    // Then using printf inside of a kernel can be a viable debugging approach.
    printf("blockIdx.x:%d * blockDim.x:%d + threadIdx.x:%d = globalThreadId:%d\n", blockIdx.x, blockDim.x, threadIdx.x, globalThreadId);
    globalArray[globalThreadId] = globalThreadId;
}
  
int main()
{
    int elementCount = 32;
    int dataSize = elementCount * sizeof(int);
     
    cudaSetDevice(0);
     
    int * managedArray;
    cudaMallocManaged(&managedArray, dataSize);
 
    kernelA <<<4,8>>>(managedArray);
 
    cudaDeviceSynchronize(); 
     
    printf("\n");
 
    // Printing a portion of results can be another good debugging approach
    for(int i = 0; i < elementCount; i++){
        printf("%d%s", managedArray[i], (i < elementCount - 1) ? ", " : "\n");
    }   
     
    cudaFree(managedArray);
 
    cudaDeviceReset();
  
    return 0;
}
