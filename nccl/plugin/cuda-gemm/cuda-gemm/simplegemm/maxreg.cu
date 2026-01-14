#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda/barrier>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <ctime>
#include <iostream>
#include <vector>
#include <random>
#include <cuda_bf16.h>
#include <cassert>
#include <unistd.h>

using barrier = cuda::barrier<cuda::thread_scope_block>;

void checkCudaErrors(cudaError_t error, const char* file, int line) {
  if (error != cudaSuccess) {
    fprintf(
        stderr,
        "CUDA error at %s:%d: %s\n",
        file,
        line,
        cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

#define check(err) checkCudaErrors(err, __FILE__, __LINE__)


template <uint32_t RegCount>
__device__ void warpgroup_reg_alloc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

__device__ static void __forceinline__
init_barrier(uint64_t* bar, int thread_count, int transaction_count) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_ptr),
      "r"(thread_count + transaction_count) : "memory");
}

__device__ static void __forceinline__ wait_barrier(uint64_t* bar, int phase) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1 bra.uni DONE;\n"
      "bra.uni LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(mbar_ptr),
      "r"(phase):"memory");
}

__device__ static void __forceinline__
arrive_barrier(uint64_t* bar, int count) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n" ::"r"(
          bar_ptr),
      "r"(count)
      : "memory");
}

__global__ __launch_bounds__(384) void dummy1() {
  __shared__ __align__(8) uint64_t bar, bar2;
  //__shared__ barrier bar;
  int tid = threadIdx.x;
  int wg = tid / 128;
  int wgtid = tid % 128;

  if (tid == 0) {
    init_barrier(&bar, 0, 2);
    init_barrier(&bar2, 0, 1);
  }
  __syncthreads();
  if (wg == 0) {
    int phase = 0;
    if (wgtid == 0) {
      //printf("producer %d\n", wg);
      wait_barrier(&bar, phase);
      //printf("producer %d 1 done\n", wg);
      wait_barrier(&bar, phase ^ 1);
      //arrive_barrier(&bar2, 1);
      //wait_barrier(&bar, phase ^ 1);
      //printf("producer %d 2 done\n", wg);
    }
  } else {
    int phase = 0;
    if (wgtid == 0) {
      //printf("consumer %d\n", wg);
      arrive_barrier(&bar, 1);
    }
    //asm volatile("bar.sync %0, 128;" :: "r"(wg) : "memory");
    if (wgtid < 2) {
      arrive_barrier(&bar, 1);
      //wait_barrier(&bar2, 1);
      //arrive_barrier(&bar, 1);
      //printf("consumer %d done\n", wg);
    }
  }
}

__global__ __launch_bounds__(384) void dummy() {
  __shared__ barrier bar;
  int tid = threadIdx.x;
  int wg = tid / 128;
  int wgtid = tid % 128;

  if (tid == 0) {
    init(&bar, 3);
  }
  __syncthreads();

  if (wg == 0) {
    int phase = 0;
    asm volatile("{\n//test 1\n}\n" ::: "memory");
    if (wgtid == 0) {
      bar.wait(bar.arrive());
      bar.wait(bar.arrive());
    }
    asm volatile("{\n//test 2\n}\n" ::: "memory");
  } else {
    int phase = 0;
    asm volatile("{\n//test 3\n}\n" ::: "memory");
    if (wgtid == 0) {
      bar.arrive();
      bar.arrive();
    }
    asm volatile("{\n//test 4\n}\n" ::: "memory");
  }
}

__global__ __launch_bounds__(384) void dummy() {
  __shared__ barrier bar;
  int tid = threadIdx.x;
  int wg = tid / 128;
  int wgtid = tid % 128;

  if (tid == 0) {
    init(&bar, 3);
  }
  __syncthreads();

  if (wg == 0) {
    int phase = 0;
    asm volatile("{\n//test 1\n}\n" ::: "memory");
    if (wgtid == 0) {
      bar.wait(bar.arrive());
      bar.wait(bar.arrive());
    }
    asm volatile("{\n//test 2\n}\n" ::: "memory");
  } else {
    int phase = 0;
    asm volatile("{\n//test 3\n}\n" ::: "memory");
    if (wgtid == 0) {
      bar.arrive();
      bar.arrive();
    }
    asm volatile("{\n//test 4\n}\n" ::: "memory");
  }
}

int main() {
  fprintf(stderr, "GO!\n");
  dummy<<<1, 384>>>();
  check(cudaDeviceSynchronize());
  fprintf(stderr, "DONE!\n");
  return 0;
}
