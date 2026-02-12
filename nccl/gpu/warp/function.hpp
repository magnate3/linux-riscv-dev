#include <stdio.h>
#include <cooperative_groups.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CHECK_KERNEL_LAUNCH() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Kernel Launch Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

typedef unsigned short int	uint16_t;
typedef unsigned int	    uint32_t;
typedef unsigned long int	uint64_t;

#define CUDA_CHECK(status)                                                    \
{                                                                         \
    cudaError_t error = status;                                           \
    if (error != cudaSuccess)                                             \
    {                                                                     \
        std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                  << " at line: " << __LINE__ << std::endl;               \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
}

#define PRINT_CONDITION(tid) (threadIdx.x == tid && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)

__forceinline__ __device__ uint32_t
cast_smem_ptr_to_uint(void const* const ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__forceinline__ __device__ uint32_t
cast_gmem_ptr_to_uint(void const* const ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_global(ptr));
}

__forceinline__ __device__ uint32_t 
set_block_rank(void const* const smem_ptr, uint32_t rank) {
  uint32_t result;
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
  asm volatile("mapa.shared::cluster.u32  %0, %1, %2;\n"
              : "=r"(result)
              : "r"(smem_int_ptr), "r"(rank));
  return result;
}

__forceinline__ __device__ void
init_mbarr(void const* smem_ptr, uint32_t arrive_count) {
  uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
  asm volatile (
    "mbarrier.init.shared::cta.b64"
    " [%1], %0;"
    :
    : "r"(arrive_count), "r"(smem_addr));
}

__forceinline__ __device__ void 
arrive_and_reset_bytes(void const* smem_ptr, uint32_t transaction_bytes) {
  uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "{\n\t"
      "mbarrier.arrive.expect_tx.shared.b64 _, [%1], %0; \n\t"
      "}"
      :
      : "r"(transaction_bytes), "r"(smem_addr));
}

__forceinline__ __device__ void 
arrive_and_reset_bytes(void const* smem_ptr, uint32_t transaction_bytes, uint32_t cta_id, uint32_t pred=true) {
  uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      ".reg .b32 remAddr32;\n\t"
      "setp.eq.u32 p, %2, 1;\n\t"
      "@p mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
      "@p mbarrier.arrive.expect_tx.shared::cluster.b64  _, [remAddr32], %3;\n\t"
      "}"
      :
      : "r"(smem_addr), "r"(cta_id), "r"(pred), "r"(transaction_bytes));
}

__forceinline__ __device__ void
cta2cluster_copy_kernel(void const* src_ptr, 
                        void const* dst_ptr, 
                        void const* mbarr_ptr, 
                        uint32_t cta_id,
                        uint32_t transaction_bytes
                        )
{
  uint32_t src_addr = cast_smem_ptr_to_uint(src_ptr);
  uint32_t dst_addr = set_block_rank(dst_ptr, cta_id);
  uint32_t mbarr_addr = set_block_rank(mbarr_ptr, cta_id);
  asm volatile (
      "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
      " [%0], [%1], %2, [%3];"
      :
      : "r"(dst_addr), "r"(src_addr), "r"(transaction_bytes), "r"(mbarr_addr)
      : "memory"
  );
}

__forceinline__ __device__ void
gmem2cta_copy_kernel( uint32_t* gmem_ptr, 
                      void const* dst_ptr, 
                      void const* mbarr_ptr,
                      uint32_t transaction_bytes,
                      uint16_t ctaMask = 0
                    )
{
  uint32_t dst_int_addr = cast_smem_ptr_to_uint(dst_ptr);
  uint32_t mbarr_addr = cast_smem_ptr_to_uint(mbarr_ptr);
  #if MULTICAST
    asm volatile (
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
        " [%0], [%1], %2, [%3], %4;"
        :
        : "r"(dst_int_addr), "l"(gmem_ptr), "r"(transaction_bytes), "r"(mbarr_addr), "h"(ctaMask)
        : "memory"
    );
  #else
    asm volatile (
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :
        : "r"(dst_int_addr), "l"(gmem_ptr), "r"(transaction_bytes), "r"(mbarr_addr)
        : "memory"
    );
  #endif
}

__forceinline__ __device__ void
gmem2cta_copy_kernel_with_cache_policy( uint32_t* gmem_ptr, 
                                        void const* dst_ptr, 
                                        void const* mbarr_ptr,
                                        uint32_t transaction_bytes,
                                        uint64_t policy,
                                        uint16_t ctaMask = 0
                                      )
{
  uint32_t dst_int_addr = cast_smem_ptr_to_uint(dst_ptr);
  uint32_t mbarr_addr = cast_smem_ptr_to_uint(mbarr_ptr);
  #if MULTICAST
    asm volatile (
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
        " [%0], [%1], %2, [%3], %4;"
        :
        : "r"(dst_int_addr), "l"(gmem_ptr), "r"(transaction_bytes), "r"(mbarr_addr), "h"(ctaMask), "l"(policy)
        : "memory"
    );
  #else
    asm volatile (
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
        " [%0], [%1], %2, [%3], %4;"
        :
        : "r"(dst_int_addr), "l"(gmem_ptr), "r"(transaction_bytes), "r"(mbarr_addr), "l"(policy)
        : "memory"
    );
  #endif
}

__forceinline__ __device__ void 
wait(void const* smem_ptr, uint32_t phase) {
    uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
    // Arbitrarily large timer value after which try-wait expires and re-tries.
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred       P1; \n\t"
        "LAB_WAIT: \n\t"
        "mbarrier.try_wait.parity.shared.b64 P1, [%0], %1, %2; \n\t"
        "@P1 bra.uni DONE; \n\t"
        "bra.uni     LAB_WAIT; \n\t"
        "DONE: \n\t"
        "}"
        :
        : "r"(smem_addr), "r"(phase), "r"(ticks));
  }

__forceinline__ __device__ uint32_t 
test_wait(void const* smem_ptr, uint32_t phase, uint32_t pred=true) {
  uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
  uint32_t waitComplete;

  asm volatile(
      "{\n\t"
      ".reg .pred P1; \n\t"
      ".reg .pred P2; \n\t"
      "setp.eq.u32 P2, %3, 1;\n\t"
      "@P2 mbarrier.test_wait.parity.shared.b64 P1, [%1], %2; \n\t"
      "selp.b32 %0, 1, 0, P1; \n\t"
      "}"
      : "=r"(waitComplete)
      : "r"(smem_addr), "r"(phase), "r"(pred));

  return waitComplete;
}

__forceinline__ __device__ void 
consumer_wait(void const* smem_ptr, uint32_t phase) {
  uint32_t done = test_wait(smem_ptr, phase);
  if (not done) {
    wait(smem_ptr, phase);
  }
}

__forceinline__ __device__ void 
producer_wait(void const* smem_ptr, uint32_t phase) {
  uint32_t done = test_wait(smem_ptr, phase);
  if (not done) {
    wait(smem_ptr, phase);
  }
}

__forceinline__ __device__ void
producer_acquire(void const* mbarr_ptr, uint32_t phase, uint32_t transaction_bytes)
{
  producer_wait(mbarr_ptr, phase);
  arrive_and_reset_bytes(mbarr_ptr, transaction_bytes);
}

__forceinline__ __device__ void 
arrive(void const* smem_ptr) {
  uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
  uint64_t state = 0;
  asm volatile(
      "{\n\t"
      "mbarrier.arrive.shared.b64 %1, [%0];\n\t"
      "}"
      :
      : "r"(smem_addr), "l"(state));
}

__forceinline__ __device__ void 
arrive(void const* smem_ptr, uint32_t cta_id, uint32_t pred=true) {
  uint32_t smem_addr = cast_smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      ".reg .b32 remAddr32;\n\t"
      "setp.eq.u32 p, %2, 1;\n\t"
      "@p mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
      "@p mbarrier.arrive.shared::cluster.b64  _, [remAddr32];\n\t"
      "}"
      :
      : "r"(smem_addr), "r"(cta_id), "r"(pred));
}


__forceinline__ __device__ uint64_t 
get_clock() {
  uint64_t gpu_clock;
  asm volatile (
      // bar may have bug
      // "bar.sync 0;\n"
      "mov.u64 %0, %%clock64;\n"
      : "=l"(gpu_clock) : : "memory"
  );
  return gpu_clock;
}

__forceinline__ __device__ uint32_t 
get_smid(void) {
  uint ret; 
  asm("mov.u32 %0, %smid;" : "=r"(ret) ); 
  return ret; 
}

__forceinline__ __device__ void
reuduce_kernel_s32( void const* src_ptr, 
                    void const* dst_ptr, 
                    void const* mbarr_ptr, 
                    uint32_t cta_id,
                    uint32_t transaction_bytes
                    )
{
  uint32_t src_addr = cast_smem_ptr_to_uint(src_ptr);
  uint32_t dst_addr = set_block_rank(dst_ptr, cta_id);
  uint32_t mbarr_addr = set_block_rank(mbarr_ptr, cta_id);
  asm volatile (
      "cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.add.s32"
      " [%0], [%1], %2, [%3];"
      :
      : "r"(dst_addr), "r"(src_addr), "r"(transaction_bytes), "r"(mbarr_addr)
      : "memory"
  );
}

__forceinline__ __device__ uint64_t
create_cache_policy(void const* gmem_ptr,
                    uint32_t primary_size,
                    uint32_t total_size
                    )
{
  uint64_t policy;
  asm volatile (
    "createpolicy.range.L2::evict_first.L2::evict_first.b64 %0, [%1], %2, %3;"
    : "=l"(policy)
    : "l"(gmem_ptr), "r"(primary_size), "r"(total_size)
    : "memory"
  );
  return policy;
}