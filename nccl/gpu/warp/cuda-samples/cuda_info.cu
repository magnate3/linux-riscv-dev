// ajee10x

#include <stdio.h>
#include <cuda_runtime_api.h>

int main()
{
  int deviceCount;
  cudaDeviceProp deviceProp;

  // How many CUDA devices are installed on the PC
  cudaGetDeviceCount(&deviceCount);

  printf("Device count: %d\n\n", deviceCount);

  for (int i = 0; i < deviceCount; i++)
  {
   // Get information about the device
    cudaGetDeviceProperties(&deviceProp, i);

 // Displays device information
    printf("Device name: %s\n", deviceProp.name);
    printf("Total global memory: %d\n", deviceProp.totalGlobalMem);
    printf("Shared memory per block: %d\n", deviceProp.sharedMemPerBlock);
    printf("Registers per block: %d\n", deviceProp.regsPerBlock);
    printf("Warp size: %d\n", deviceProp.warpSize);
    printf("Memory pitch: %d\n", deviceProp.memPitch);
    printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    
    printf("Max threads dimensions: x = %d, y = %d, z = %d\n",
      deviceProp.maxThreadsDim[0],
      deviceProp.maxThreadsDim[1],
      deviceProp.maxThreadsDim[2]);
    
    printf("Max grid size: x = %d, y = %d, z = %d\n", 
      deviceProp.maxGridSize[0], 
      deviceProp.maxGridSize[1], 
      deviceProp.maxGridSize[2]); 

    printf("Clock rate: %d\n", deviceProp.clockRate);
    printf("Total constant memory: %d\n", deviceProp.totalConstMem); 
    printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Texture alignment: %d\n", deviceProp.textureAlignment);
    printf("Device overlap: %d\n", deviceProp.deviceOverlap);
    printf("Number of SM: %d\n", deviceProp.multiProcessorCount);
    printf("Kernel execution timeout enabled: %s\n",
      deviceProp.kernelExecTimeoutEnabled ? "true" : "false");
  }

  return 0;
}
