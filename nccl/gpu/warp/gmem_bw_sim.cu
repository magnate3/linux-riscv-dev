/*
  We use this example to simulate slower gmem bandwidth
*/

#define TEST        0

typedef unsigned short int	uint16_t;
typedef unsigned int	    uint32_t;
typedef unsigned long int	uint64_t;

#include <stdio.h>
#include <cooperative_groups.h>
#include "./function.hpp"

const int ROUND = 20000;
const int GRID  = 128;
const int BLOCK = 32;
const int STAGE = 16;
const int SMEM_SIZE = 1024 * 192;
const int COUNT = SMEM_SIZE / STAGE;
const int MBARR_SIZE = 128;
const int MAXBYTES  = 1024 * 227;
const int CLUSTER = 1;
// allocate 10 GB gmem data
const long int GMEM_SIZE = long(10) << 30;
// copy to one CTA in the cluster

__device__ dim3 block_id_in_cluster()
{
  uint32_t x, y, z;
  asm volatile("mov.u32 %0, %cluster_ctaid.x;\n" : "=r"(x) : );
  asm volatile("mov.u32 %0, %cluster_ctaid.y;\n" : "=r"(y) : );
  asm volatile("mov.u32 %0, %cluster_ctaid.z;\n" : "=r"(z) : );
  return {x, y, z};
}

__global__ void
cluster_kernel(uint32_t* gmem_ptr, float real_ratio)
{
  // uint32_t smid = get_smid();
  
  extern __shared__ int data[];
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  dim3 cluster_size = cluster.dim_blocks();
  uint32_t clusterBlockRank = cluster.block_rank();
  uint32_t curr_stage = 0;
  uint32_t phase = 1;
  uint32_t block_id = blockIdx.x * gridDim.y * gridDim.z
                    + blockIdx.y * gridDim.z
                    + blockIdx.z;
  float dummy_ratio = 1 - real_ratio;

  cluster.sync();

  dim3 bid = block_id_in_cluster();
  if (threadIdx.x==0) {
    printf("%d,%d,%d,%d\n", bid.x, bid.y, bid.z, clusterBlockRank);
  }

  if (threadIdx.x==0) {
    char* mbarr_start_ptr = (char*)data + SMEM_SIZE;
    for (int stage = 0; stage < STAGE; stage++) {
      init_mbarr(mbarr_start_ptr + stage*MBARR_SIZE, 1);
    }

    for (int i = 0; i < ROUND; i++) {
      curr_stage = i % STAGE;
      char* mbarr_ptr = mbarr_start_ptr + curr_stage*MBARR_SIZE;
      char* stage_ptr = (char*)data + curr_stage*COUNT;
      uint32_t dummy_data_size = (int(COUNT * dummy_ratio) / 128) * 128;;
      uint32_t real_data_size = (int(COUNT * real_ratio) / 128) * 128;
      producer_acquire(mbarr_ptr, phase, (real_data_size + dummy_data_size));
      char* dummy_gptr = (char*)gmem_ptr + (i*GRID*COUNT + block_id*COUNT) % GMEM_SIZE;
      char* real_gptr  = (char*)gmem_ptr + (i*GRID*COUNT + block_id*COUNT + dummy_data_size) % GMEM_SIZE;
      // Copy dummy data
      gmem2cta_copy_kernel((uint32_t*)dummy_gptr, stage_ptr, mbarr_ptr, dummy_data_size);
      // Copy real data
      gmem2cta_copy_kernel((uint32_t*)real_gptr , stage_ptr, mbarr_ptr, real_data_size);
      if (curr_stage == STAGE - 1) {
        phase ^= 1;
      }
    }
  }

  cluster.sync();
}


int main() 
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  static_assert(GRID % CLUSTER == 0, "invalid cluster config");
  dim3 grid(GRID/8, 4, 1);
  dim3 block(BLOCK, 1, 1);

  uint32_t *d_gptr;

  cudaFuncSetAttribute(cluster_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MAXBYTES);
  cudaFuncSetAttribute(cluster_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  cudaMalloc((void**)&d_gptr, GMEM_SIZE);

  #if TEST
    uint32_t h_gptr[256];
    for (int i=0; i<256; i++)
    {
      h_gptr[i] = 2*i;
    }
    cudaMemcpy((char*)d_gptr + (GRID-1) * COUNT, h_gptr,  256 * sizeof(uint32_t), cudaMemcpyHostToDevice);
  #endif

  uint32_t clk = prop.clockRate / 1000; // in mherz
  uint32_t sm = prop.multiProcessorCount;
  printf("standard clock frequency: %u MHz\n", clk);
  printf("SM: %u\n", sm);

  for (int iter = 0; iter < 10; iter++) {
    float real_ratio = 1 - float(iter) * 0.1;
    printf("Simulating %.2fx gmem bandwidth...\n", real_ratio);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    {
      cudaLaunchConfig_t config = {0};
      // The grid dimension is not affected by cluster launch, and is still enumerated
      // using number of blocks.
      // The grid dimension should be a multiple of cluster size.
      config.gridDim = grid;
      config.blockDim = block;
      config.dynamicSmemBytes = MAXBYTES;

      cudaLaunchAttribute attribute[1];
      attribute[0].id = cudaLaunchAttributeClusterDimension;
      attribute[0].val.clusterDim.x = 2; // Cluster size in X-dimension
      attribute[0].val.clusterDim.y = 4;
      attribute[0].val.clusterDim.z = 1;
      config.attrs = attribute;
      config.numAttrs = 1;

      cudaError_t status = cudaLaunchKernelEx(&config, cluster_kernel, d_gptr, real_ratio);
      if (status != cudaSuccess) {
          const char* errorMessage = cudaGetErrorString(status);
          printf("  CUDA Error: %s\n", errorMessage);
      } else {
          printf("  No CUDA Error\n");
      }
    }
    // cluster_kernel <<<grid, block, MAXBYTES>>> (d_gptr);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    
    // printf("latency: %f ms\n", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    long int total_bytes = long(ROUND) * long(COUNT * real_ratio) * long(GRID);

    printf("  bandwidth: %f GB/s\n", total_bytes/(1024*1024*1024*(time/1000)));
  }
	return 0;
}
