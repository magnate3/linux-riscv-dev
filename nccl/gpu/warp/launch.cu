#include "cuda_runtime.h"
// kernel defintition
// no compile time attribute attached to the kernel 
#include <cooperative_groups.h>
__global__ void __cluster_dims__(2,1,1) cluster_kernel(float *input,float *output)
{

}

int main(){
    flaot *input,*output;
    dim3 threadsPerBlock(16,16); //16x16（256 个线程） 每个线程块里面有256个线程
    dim3 numBlocks(N/threadsIdx.x,N/threadsIdx.y);
    // kernel invocation with runtime cluster size 
    {
        cudaLaunchConfig_t config = {0};
        // 網格維度不受集羣啟動的影響，仍然是枚舉的
        //使用塊的數量。
        //網格維度應該是集羣大小的倍數。
        config.gridDim = numBlocks;
        config.blockDim = threadsPerBlock;

        cudaLaunchAttribute  attribute[1];
        attribute[0].id  = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 2 ; // cluster size in x-dimension
        attribute[0].val.clusterDim.y = 1 ;
        attribute[0].val.clusterDim.z = 1 ;
        config.attrs = attribute;
        config.numAttrs = 1;
        cudaLaunchKernelEx(&config,cluster_kernel,input,output);
    }
    
}
