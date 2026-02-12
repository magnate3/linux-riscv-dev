#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
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
  uint64_t comm;
  uint64_t load_data;
  uint64_t store_data;
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

// a copy of the volatile load/store from prims_ll
template<typename U>
__device__ static U load(U *src) {
  union {
    U elt;
    uint8_t u1;
    uint16_t u2;
    uint32_t u4;
    uint64_t u8;
  };

  if(sizeof(U) == 1)
    asm("ld.volatile.global.b8 %0,[%1];" : "=r"(u4) : "l"(src));
  else if(sizeof(U) == 2)
    asm("ld.volatile.global.b16 %0,[%1];" : "=h"(u2) : "l"(src));
  else if(sizeof(U) == 4)
    asm("ld.volatile.global.b32 %0,[%1];" : "=r"(u4) : "l"(src));
  else
    asm("ld.volatile.global.b64 %0,[%1];" : "=l"(u8) : "l"(src));
  return elt;
}

template<typename U>
__device__ static void store(U *dst, U val) {
  union {
    U elt;
    uint8_t u1;
    uint16_t u2;
    uint32_t u4;
    uint64_t u8;
  };

  elt = val;
  if(sizeof(U) == 1)
    asm("st.volatile.global.b8 [%0],%1;" :: "l"(dst), "r"(u4));
  else if(sizeof(U) == 2)
    asm("st.volatile.global.b16 [%0],%1;" :: "l"(dst), "h"(u2));
  else if(sizeof(U) == 4)
    asm("st.volatile.global.b32 [%0],%1;" :: "l"(dst), "r"(u4));
  else
    asm("st.volatile.global.b64 [%0],%1;" :: "l"(dst), "l"(u8));
}
#if 0
inline __device__ static void barrier(int nthreads) {
    asm volatile ("bar.sync %1, %0;" :: "r"(nthreads), "r"(15));
}
#endif

#if 0
__device__ __forceinline__ static void threadBlockCopy(
  uint64_t *dst, uint64_t const *src, uint64_t size, int tid, int nthreads) {
  for (int i = tid; i < size; i += nthreads) {
    dst[i] = src[i];
  }
}
#else
__device__ __forceinline__ static void threadBlockCopy(
  uint64_t *dst, uint64_t const *src, uint64_t size, int tid, int nthreads) {
  for (int i = 0; i < size; ++i) {
    dst[i] = src[i];
  }
}
#endif
__device__ __forceinline__ void mscclRunInterpreter(struct ncclShmemData*cpu) {
  const int tid = threadIdx.x;
  //const int bid = blockIdx.x;
  const int nthreads = blockDim.x;
  //int bytes = 0;
  // bytes = sizeof(uint64_t);
  // initialize mscclShmem.mscclTB
  uint64_t data = 0;
  threadBlockCopy(
    (uint64_t *)&ncclShmem.ll128warp, (uint64_t *)(cpu->ll128warp),
    sizeof(ncclShmem.ll128warp)/sizeof(uint64_t), tid, nthreads);
    __syncthreads(); 
    data = load(&ncclShmem.load_data);
    store(&ncclShmem.store_data,data);
  threadBlockCopy(
    (uint64_t *)&(cpu->store_data), (uint64_t *)(&ncclShmem.store_data),
    sizeof(ncclShmem.store_data)/sizeof(uint64_t), tid, nthreads);
#if 0
   void *dst, *src;
   dst = &ncclShmem.comm;
   src = &cpu->comm;
   copyToShmem16(tid%WARP_SIZE, dst, src, bytes);
    __syncthreads(); // publish shmem
#endif
}

__global__ void kernelFunction2(ncclShmemData *shd) {
    mscclRunInterpreter(shd); 

}
struct ncclShmemData cpudata;
int main() {
    cpudata.load_data = 0x40;
    cpudata.store_data = 0x30;
    //printf("sizeof (struct ncclShmemData) : %d \n",sizeof(struct ncclShmemData));
    //ncclKernel(&cpudata);
    //ncclKernel<<<2, 2>>>(&cpudata);
    kernelFunction<<<1, 1024>>>(&cpudata); 
    printf("load data: %" PRIu64  "    store data: %" PRIu64 "\n", cpudata.load_data,cpudata.store_data);
    kernelFunction2<<<1, 1024>>>(&cpudata); 
    //ncclKernel<<<1, 1>>>(&cpudata); 
    cudaDeviceSynchronize();
    printf("load data: %" PRIu64  "    store data: %" PRIu64 "\n", cpudata.load_data,cpudata.store_data);
    return 0;
}
