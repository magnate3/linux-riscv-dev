#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "op128.h"

#define WARP_SIZE 32
#define NCCL_MAX_GROUPS 16
#define NCCL_MAX_DIRECT_ARITY 7
#define NCCL_LL128_MAX_NTHREADS 640
#define NCCL_LL128_LINESIZE 128
#define NCCL_LL128_LINEELEMS (NCCL_LL128_LINESIZE/sizeof(uint64_t))
#define NCCL_LL128_DATAELEMS (NCCL_LL128_LINEELEMS-1)

#define NCCL_LL128_MAX_NTHREADS 640
#define NCCL_LL128_ELEMS_PER_THREAD 120

#define NCCL_LL128_SHMEM_ELEMS_PER_THREAD 8
#define NCCL_LL128_SHMEM_SIZE (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*NCCL_LL128_MAX_NTHREADS)

template<typename X, typename Y, typename Z = decltype(X()+Y())>
__host__ __device__ constexpr Z divUp(X x, Y y) {
  return (x+y-1)/y;
}
#if __CUDA_ARCH__ >= 800
#define COLL_UNROLL 8
#else
#define COLL_UNROLL 4
#endif


#define NCCL_MAX_DEV_ARITY (NCCL_MAX_TREE_ARITY-1)  // Using balanced tree instead of split tree

__device__ inline bool barrierReduceAny(int bit) {
  uint32_t popc;
  asm ("{"
    ".reg .pred barr_pred;"
    "setp.eq.u32 barr_pred, %1, 1;"
    "bar.red.popc.u32 %0, 2, barr_pred;"
  "}" : "=r"(popc) : "r"(bit));
  return popc != 0;
}

// Copy src to dst and fill extra size with zeroes
template<typename Tdst, typename Tsrc>
__device__ void copyToShmem(Tdst *dst, Tsrc const *src, int tid, int nthreads) {
  static_assert(sizeof(Tdst)%(2*sizeof(uint64_t)) == 0 && sizeof(Tsrc)%(2*sizeof(uint64_t)) == 0,
      "copyToShmem needs sizes which are multiple of 16B");
  static_assert(sizeof(Tdst) >= sizeof(Tsrc), "Tdst size is too small");
  static_assert(sizeof(Tdst) <= WARP_SIZE*2*sizeof(uint64_t), "copyToShmem limited to 512B to make sure it can always be done in one cycle");
  uint64_t *d = reinterpret_cast<uint64_t*>(dst);
  uint64_t const *s = reinterpret_cast<uint64_t const*>(src);
  uint64_t *shmemPtr = shmemCvtPtr(d);
  int offset = 2*tid;
  uint64_t v0, v1;
  if (offset >= sizeof(Tsrc)/sizeof(uint64_t)) {
    v0 = v1 = 0ULL;
  } else {
    v0 = s[offset] ; v1 = s[offset+1];
  }
  if (offset < sizeof(Tdst)/sizeof(uint64_t)) storeShmem128(shmemPtr+offset, v0, v1);
}

template<typename T>
__device__ int copyToShmem(T *dst, T const *src, int turn=0) {
  static_assert(sizeof(uint64_t) <= alignof(T), "Uhoh");
  uint64_t *d = reinterpret_cast<uint64_t*>(dst);
  uint64_t const *s = reinterpret_cast<uint64_t const*>(src);
  int t = threadIdx.x - turn;
  if (t < 0) t += blockDim.x;
  int n = sizeof(T)/sizeof(uint64_t);

  int delta = (n + WARP_SIZE-1) & -WARP_SIZE; // round up to warp lane 0
  if (delta < blockDim.x) {
    turn += delta;
    if (turn >= blockDim.x) turn -= blockDim.x;
  }
  else
    turn = 0;

  n -= t;
  d += t;
  s += t;
  #pragma unroll
  for (int i=0; i < divUp(sizeof(T), WARP_SIZE*sizeof(uint64_t)); i++) {
    if (n > 0) {
      *d = *s;
      d += blockDim.x;
      s += blockDim.x;
      n -= blockDim.x;
    }
  }
  return turn;
}

template<typename T>
__device__ void simpleCopy(T *dst, T const *src, int tid, int nthreads) {
  char* d = (char*) dst;
  char* s = (char*) src;
  for (int i = tid; i < sizeof(T); i += nthreads)
    d[i] = s[i];
}
// Copy 16-byte aligned data. You must call with at least `(bytes+15)/16` threads.
inline __device__ void copyToShmem16(int tid, void* dst, void const* src, int bytes) {
  int offset = 16*tid;
  if (offset < bytes) {
    ulong2 *src2, *dst2;
    src2 = (ulong2*)((char const*)src + offset);
    dst2 = (ulong2*)((char*)dst + offset);
    dst2->x = src2->x;
    dst2->y = src2->y;
  }
}
struct ncclShmemData {
  union {
    uint64_t ll128warp[NCCL_LL128_MAX_NTHREADS/WARP_SIZE][NCCL_LL128_SHMEM_ELEMS_PER_THREAD*WARP_SIZE];
    uint64_t  groups[NCCL_MAX_GROUPS];
  };
  uint64_t redOpArgs[NCCL_MAX_DIRECT_ARITY+1];
};
__shared__ ncclShmemData ncclShmem;
__device__ void ncclKernel(ncclShmemData *shd)  {
#if 0
  int tid = threadIdx.x;
  int wid = threadIdx.x/WARP_SIZE;
  int nWarps = blockDim.x/WARP_SIZE;
  int nthreads = blockDim.x;
  int bid = blockIdx.x;
#endif
  int turn = copyToShmem(&ncclShmem, shd);
  turn = copyToShmem(&ncclShmem, shd, turn);
}
__global__ void kernelFunction(ncclShmemData *shd) {
    ncclKernel(shd); 

}

int main() {
    struct ncclShmemData cpudata;
    //printf("sizeof (struct ncclShmemData) : %d \n",sizeof(struct ncclShmemData));
    //ncclKernel(&cpudata);
    //ncclKernel<<<2, 2>>>(&cpudata);
    kernelFunction<<<1, 1024>>>(&cpudata); 
    //ncclKernel<<<1, 1>>>(&cpudata); 
    cudaDeviceSynchronize();
    return 0;
}
