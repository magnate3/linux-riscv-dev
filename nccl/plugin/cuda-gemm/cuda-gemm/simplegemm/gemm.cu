#include <assert.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <iostream>
#include <random>

namespace {

using bf16 = __nv_bfloat16;

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

__host__ __device__ int cdiv(int m, int n) {
  return (m + n - 1) / n;
}


template <typename T>
void tmaPrint(T s[]) {
  for (int i = 0; i < 3; i++) {
    std::cout << "  " << s[i];
  }
  std::cout << "\n";
}

__device__ inline bf16 f2bf(float v) {
  return __float2bfloat16(v);
}

__host__ static inline CUtensorMap create_tma_desc(
    bf16* gmem,
    uint32_t M,
    uint32_t N,
    uint32_t BLOCK_M,
    uint32_t BLOCK_N) {
  CUtensorMap tma_desc;
  // TODO: Check these requirements against the HW spec.
  assert(BLOCK_N >= 64);
  assert(N % 64 == 0);

  // TODO: cdiv?
  // TODO" why the 64 inner dim?
  uint64_t shape[] = {64, M, N / 64};
  uint64_t stride[] = {sizeof(bf16) * N, 64 * sizeof(bf16)};
  uint32_t box_shape[] = {64, BLOCK_M, BLOCK_N / 64};
  uint32_t box_stride[] = {1, 1, 1};

  // tmaPrint(shape);
  // tmaPrint(stride);
  // tmaPrint(box_shape);
  // tmaPrint(box_stride);

  auto result = cuTensorMapEncodeTiled(
      &tma_desc,
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      3,
      gmem,
      shape,
      stride,
      box_shape,
      box_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  if (result != CUDA_SUCCESS) {
    fprintf(stderr, "TMA desc creation failed\n");
    exit(EXIT_FAILURE);
  }

  return tma_desc;
}

__device__ uint64_t matrix_descriptor_encode(uint64_t x) {
  return (x & 0x3ffff) >> 4;
}

__device__ uint64_t make_smem_desc(bf16* ptr) {
  constexpr uint64_t leading_dim_byte_offset = 16;
  constexpr uint64_t stride_dim_byte_offset = 1024;
  constexpr uint64_t swizzle_128b = 1ull;
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  return matrix_descriptor_encode(addr) |
      (matrix_descriptor_encode(leading_dim_byte_offset) << 16) |
      (matrix_descriptor_encode(stride_dim_byte_offset) << 32) |
      (swizzle_128b << 62);
}

template <int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmma256(float d[16][8], bf16* sA, bf16* sB) {
  uint64_t desc_a = make_smem_desc(&sA[0]);
  uint64_t desc_b = make_smem_desc(&sB[0]);
  // if (threadIdx.x == 128) {

  //   printf("%llx\n", desc_a);

  //   printf("%llx\n", desc_b);
  // }

#if 1
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
      " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
      " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
      " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
      " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
      " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
      " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
      " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
      " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
      " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
      " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
      " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
      " %96,  %97,  %98,  %99,  %100, %101, %102, %103,  "
      " %104, %105, %106, %107, %108, %109, %110, %111,  "
      " %112, %113, %114, %115, %116, %117, %118, %119,  "
      " %120, %121, %122, %123, %124, %125, %126, %127},"
      " %128,"
      " %129,"
      " %130,    %131,  %132,  %133,  %134;\n"
      "}\n"
      : "+f"(d[0][0]),
        "+f"(d[0][1]),
        "+f"(d[0][2]),
        "+f"(d[0][3]),
        "+f"(d[0][4]),
        "+f"(d[0][5]),
        "+f"(d[0][6]),
        "+f"(d[0][7]),
        "+f"(d[1][0]),
        "+f"(d[1][1]),
        "+f"(d[1][2]),
        "+f"(d[1][3]),
        "+f"(d[1][4]),
        "+f"(d[1][5]),
        "+f"(d[1][6]),
        "+f"(d[1][7]),
        "+f"(d[2][0]),
        "+f"(d[2][1]),
        "+f"(d[2][2]),
        "+f"(d[2][3]),
        "+f"(d[2][4]),
        "+f"(d[2][5]),
        "+f"(d[2][6]),
        "+f"(d[2][7]),
        "+f"(d[3][0]),
        "+f"(d[3][1]),
        "+f"(d[3][2]),
        "+f"(d[3][3]),
        "+f"(d[3][4]),
        "+f"(d[3][5]),
        "+f"(d[3][6]),
        "+f"(d[3][7]),
        "+f"(d[4][0]),
        "+f"(d[4][1]),
        "+f"(d[4][2]),
        "+f"(d[4][3]),
        "+f"(d[4][4]),
        "+f"(d[4][5]),
        "+f"(d[4][6]),
        "+f"(d[4][7]),
        "+f"(d[5][0]),
        "+f"(d[5][1]),
        "+f"(d[5][2]),
        "+f"(d[5][3]),
        "+f"(d[5][4]),
        "+f"(d[5][5]),
        "+f"(d[5][6]),
        "+f"(d[5][7]),
        "+f"(d[6][0]),
        "+f"(d[6][1]),
        "+f"(d[6][2]),
        "+f"(d[6][3]),
        "+f"(d[6][4]),
        "+f"(d[6][5]),
        "+f"(d[6][6]),
        "+f"(d[6][7]),
        "+f"(d[7][0]),
        "+f"(d[7][1]),
        "+f"(d[7][2]),
        "+f"(d[7][3]),
        "+f"(d[7][4]),
        "+f"(d[7][5]),
        "+f"(d[7][6]),
        "+f"(d[7][7]),
        "+f"(d[8][0]),
        "+f"(d[8][1]),
        "+f"(d[8][2]),
        "+f"(d[8][3]),
        "+f"(d[8][4]),
        "+f"(d[8][5]),
        "+f"(d[8][6]),
        "+f"(d[8][7]),
        "+f"(d[9][0]),
        "+f"(d[9][1]),
        "+f"(d[9][2]),
        "+f"(d[9][3]),
        "+f"(d[9][4]),
        "+f"(d[9][5]),
        "+f"(d[9][6]),
        "+f"(d[9][7]),
        "+f"(d[10][0]),
        "+f"(d[10][1]),
        "+f"(d[10][2]),
        "+f"(d[10][3]),
        "+f"(d[10][4]),
        "+f"(d[10][5]),
        "+f"(d[10][6]),
        "+f"(d[10][7]),
        "+f"(d[11][0]),
        "+f"(d[11][1]),
        "+f"(d[11][2]),
        "+f"(d[11][3]),
        "+f"(d[11][4]),
        "+f"(d[11][5]),
        "+f"(d[11][6]),
        "+f"(d[11][7]),
        "+f"(d[12][0]),
        "+f"(d[12][1]),
        "+f"(d[12][2]),
        "+f"(d[12][3]),
        "+f"(d[12][4]),
        "+f"(d[12][5]),
        "+f"(d[12][6]),
        "+f"(d[12][7]),
        "+f"(d[13][0]),
        "+f"(d[13][1]),
        "+f"(d[13][2]),
        "+f"(d[13][3]),
        "+f"(d[13][4]),
        "+f"(d[13][5]),
        "+f"(d[13][6]),
        "+f"(d[13][7]),
        "+f"(d[14][0]),
        "+f"(d[14][1]),
        "+f"(d[14][2]),
        "+f"(d[14][3]),
        "+f"(d[14][4]),
        "+f"(d[14][5]),
        "+f"(d[14][6]),
        "+f"(d[14][7]),
        "+f"(d[15][0]),
        "+f"(d[15][1]),
        "+f"(d[15][2]),
        "+f"(d[15][3]),
        "+f"(d[15][4]),
        "+f"(d[15][5]),
        "+f"(d[15][6]),
        "+f"(d[15][7])
      : "l"(desc_a),
        "l"(desc_b),
        "n"(int32_t(ScaleD)),
        "n"(int32_t(ScaleA)),
        "n"(int32_t(ScaleB)),
        "n"(int32_t(TransA)),
        "n"(int32_t(TransB)));
#endif
}

__device__ void wgmma_commit_group() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ void wgmma_wait_group() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

__device__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

template <uint32_t REGS>
__device__ static __forceinline__ void setmaxnreg_inc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(REGS));
}

template <uint32_t REGS>
__device__ static void __forceinline__ setmaxnreg_dec() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(REGS));
}

__device__ static void __forceinline__
init_barrier(uint64_t* bar, int thread_count, int transaction_count) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_ptr),
      "r"(thread_count + transaction_count));
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
      "r"(phase));
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

__device__ static void __forceinline__
expect_bytes(uint64_t* bar, uint32_t bytes) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar_ptr),
      "r"(bytes));
}

__device__ static void __forceinline__ tma_load(
    bf16* dst,
    void const* const src_tma_desc,
    uint64_t* bar,
    int n,
    int m) {
  uint64_t tma_ptr = reinterpret_cast<uint64_t>(src_tma_desc);
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5}], [%2];"
      ::
      "r"(dst_ptr),
      "l"(tma_ptr),
      "r"(bar_ptr),
      "n"(0),
      "r"(m),
      "r"(n / 64)
      : "memory");
}

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 256;
constexpr int BLOCK_K = 64;
constexpr int NUM_SMS = 132;
constexpr int STAGES = 3;

constexpr int WARPGROUP_SIZE = 128;
constexpr int WARPGROUPS = 3;
constexpr int NUM_THREADS = WARPGROUPS * WARPGROUP_SIZE;

struct SharedStorage {
  alignas(128) bf16 A[BLOCK_M * BLOCK_K * STAGES];
  alignas(128) bf16 B[BLOCK_K * BLOCK_N * STAGES];
};

__global__ __launch_bounds__(NUM_THREADS) void gemm(
    const __grid_constant__ CUtensorMap A,
    const __grid_constant__ CUtensorMap B,
    bf16* C,
    int M,
    int N,
    int K) {
  // Producer buffers for A and B.
  extern __shared__ __align__(128) uint8_t dynamic_smem[];
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(dynamic_smem);

  // Barriers.
  __shared__ __align__(8) uint64_t prod[STAGES];
  __shared__ __align__(8) uint64_t cons[STAGES];

  int tid = threadIdx.x;
  int wgid = tid / WARPGROUP_SIZE;
  int wg_tid = tid % WARPGROUP_SIZE;

  // Init barriers.
  if (tid == 0) {
    for (int i = 0; i < STAGES; i++) {
      init_barrier(&prod[i], 0, 1);
      init_barrier(&cons[i], 0, WARPGROUPS - 1);
    }
  }
  __syncthreads();

  auto m_blocks = cdiv(M, BLOCK_M);
  auto n_blocks = cdiv(N, BLOCK_N);
  auto k_blocks = cdiv(K, BLOCK_K);

  if (wgid == 0) {
    // Producer warpgroup.
    setmaxnreg_dec<40>();
    // Mainloop.

    //int m = 0, n = 0;
    if (wg_tid == 0) {
      int phase = 0;
      int stage = 0;
      for (auto bid = blockIdx.x; bid < m_blocks * n_blocks; bid += gridDim.x) {
        auto m = bid / n_blocks;
        auto n = bid % n_blocks;

        for (int k = 0; k < k_blocks ; k++) {
          // Wait for consumer.
          // TODO: stage and phase update.
          wait_barrier(&cons[stage], phase);
          // Set expect bytes for TMA.
          expect_bytes(
              &prod[stage], sizeof(bf16) * (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N));
          // Load A.
          // TODO: use proper stage
          tma_load(&smem.A[stage * BLOCK_K * BLOCK_M], &A, &prod[stage], k * BLOCK_K, m * BLOCK_M);
          // Load B.
          tma_load(&smem.B[stage * BLOCK_K * BLOCK_N], &B, &prod[stage], k * BLOCK_K, n * BLOCK_N);
          stage++;
          if (stage == STAGES) {
            stage = 0;
            phase ^= 1;
          }
        }
      }
    }
  } else {
    // Consumer warpgroup.
    setmaxnreg_inc<232>();

    int stage = 0;
    int phase = 0;
    if (wg_tid == 0) {
      for (int i = 0; i < STAGES; i++) {
        arrive_barrier(&cons[i], 1);
      }
    }
    for (auto bid = blockIdx.x; bid < m_blocks * n_blocks; bid += gridDim.x) {
      auto m = bid / n_blocks;
      auto n = bid % n_blocks;

      float acc[16][8];
      memset(acc, 0, sizeof(acc));
      // Mainloop.
      for (int k = 0; k < k_blocks; k++) {
        // Wait for producer.
        wait_barrier(&prod[stage], phase);

        wgmma_fence();

#pragma unroll
        for (int mma_k = 0; mma_k < BLOCK_K; mma_k += 16) {
          wgmma256<1, 1, 1, 0, 0>(
              acc,
              &smem.A[stage * BLOCK_M * BLOCK_K + mma_k + (wgid - 1) * BLOCK_K * (BLOCK_M / 2)],
              &smem.B[stage * BLOCK_N * BLOCK_K + mma_k]);
        }

        wgmma_commit_group();
        wgmma_wait_group<0>();

        // Arrive at consumer.
        if (wg_tid == 0) {
          arrive_barrier(&cons[stage], 1);
        }
        stage++;
        if (stage == STAGES)  {
          stage = 0;
          phase ^= 1;
        }
      }
      // Write back to gmem.
      auto warp = wg_tid / 32;
      auto lane = wg_tid % 32;
      auto row = warp * 16 + lane / 4;
      auto col = (wg_tid % 4) * 2;

      row += (wgid - 1) * 64;
      auto C_BLOCK = &C[m * BLOCK_M + n * BLOCK_N * M];

      //printf("%d %d %d\n", tid - 128, row, col);
      for (int inst_n = 0; inst_n < 256; inst_n += 16) {
#define Cidx(r, c) C_BLOCK[(r) + ((c) * M)]
        // clang-format off
        // printf("%d %d %d %f\n",
        //        tid,
        //        row,
        //        col,
        //        acc[n][0]);
        Cidx(row,     inst_n + col    ) = f2bf(acc[inst_n / 16][0]);
        Cidx(row,     inst_n + col + 1) = f2bf(acc[inst_n / 16][1]);
        Cidx(row + 8, inst_n + col    ) = f2bf(acc[inst_n / 16][2]);
        Cidx(row + 8, inst_n + col + 1) = f2bf(acc[inst_n / 16][3]);
        Cidx(row,     inst_n + col + 8) = f2bf(acc[inst_n / 16][4]);
        Cidx(row,     inst_n + col + 9) = f2bf(acc[inst_n / 16][5]);
        Cidx(row + 8, inst_n + col + 8) = f2bf(acc[inst_n / 16][6]);
        Cidx(row + 8, inst_n + col + 9) = f2bf(acc[inst_n / 16][7]);
        // clang-format on
      }
    }

    // auto row = (wg_tid / 32) * 2 + wg_tid / 4;
    // if (tid == 128) {
    //   for (int i = 0; i < 16; i++) {
    //     for (int j = 0; j < 8; j++) {
    //       printf("  %6.2f", acc[i][j]);
    //     }
    //     printf("\n");
    //   }
    //   printf("\n");
    // }
  }
  // __syncthreads();
  // if (tid == 128) {
  //   printf("smem.A:\n");
  //   for (int i = 0; i < BLOCK_M; i++) {
  //     for (int j = 0; j < BLOCK_K; j++) {
  //       printf("  %6.2f", __bfloat162float(smem.A[i * BLOCK_K + j]));
  //     }
  //     printf("\n");
  //   }
  //   printf("\n");
  //   printf("smem.B:\n");
  //   for (int i = 0; i < BLOCK_K; i++) {
  //     for (int j = 0; j < BLOCK_N; j++) {
  //       printf("  %6.2f", __bfloat162float(smem.B[i + j * BLOCK_K]));
  //     }
  //     printf("\n");
  //   }
  //   printf("\n");
  // }
}
} // namespace

void run_gemm(bf16* A, bf16* B, bf16* C, int M, int N, int K) {
  // Compute necessary shared memory for buffers.
  size_t smem_size = sizeof(SharedStorage);
  check(cudaFuncSetAttribute(
      gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  // Set up TMA descriptors
  auto descA = create_tma_desc(A, M, K, BLOCK_M, BLOCK_K);
  auto descB = create_tma_desc(B, N, K, BLOCK_N, BLOCK_K);

  // Launch kernel!
  gemm<<<NUM_SMS, NUM_THREADS, smem_size>>>(descA, descB, C, M, N, K);
}

void run_gemm(void* A, void* B, void* C, int M, int N, int K) {
  run_gemm((bf16*) A, (bf16*)B, (bf16*)C, M, N, K);
}
