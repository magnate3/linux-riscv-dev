# INT8 TensorCore GEMM 详细教程 - CUDA新手指南

## 概述

本教程将详细分析NVIDIA提供的基于TensorCore的INT8 GEMM（矩阵乘法）实现，特别适合CUDA编程新手。我们将逐块解释代码，并与其他精度实现进行对比。

### 什么是GEMM？
GEMM (General Matrix Multiply) 执行操作：**D = α·A×B + β·C**
- A: M×K 矩阵
- B: K×N 矩阵  
- C: M×N 矩阵（输入）
- D: M×N 矩阵（输出）
- α, β: 标量系数

### 什么是TensorCore？
TensorCore是NVIDIA从Volta架构开始引入的专用AI加速单元，可以极大提升矩阵运算性能。INT8精度特别适合推理场景。

---

## 第一部分：配置参数和宏定义

### 1.1 矩阵维度配置
```cuda
// 每个WMMA操作的基本矩阵块大小
#define WMMA_M 16
#define WMMA_N 16  
#define WMMA_K 16

// 每个warp处理的矩阵块数量
#define WARP_COL_TILES 4
#define WARP_ROW_TILES 2

// 计算得出的矩阵块维度
#define M (WMMA_M * WARP_COL_TILES)  // 16 * 4 = 64
#define N (WMMA_N * WARP_ROW_TILES)  // 16 * 2 = 32
#define K (WMMA_K)                   // 16
```

**CUDA新手解释**：
- WMMA (Warp Matrix Multiply Accumulate) 是TensorCore的编程接口
- 每个TensorCore操作处理16×16×16的矩阵块
- 一个warp（32个线程）可以处理多个这样的块

**与其他精度对比**：
- **BF16/FP16**: 同样使用16×16×16块，但数据类型不同
- **TF32**: 使用16×16×8块（K维度更小）
- **FP64**: 使用8×8×4块（所有维度都更小）

### 1.2 全局矩阵大小
```cuda
#define M_TILES 256    // 全局M方向的块数
#define N_TILES 256    // 全局N方向的块数  
#define K_TILES 256    // 全局K方向的块数

#define M_GLOBAL (M * M_TILES)  // 4096 = 64 * 256
#define N_GLOBAL (N * N_TILES)  // 4096 = 32 * 256  
#define K_GLOBAL (K * K_TILES)  // 4096 = 16 * 256
```

**总矩阵大小**：4096×4096×4096，这是一个相当大的矩阵！

### 1.3 Block和Warp组织
```cuda
#define BLOCK_COL_TILES 2  // Block在列方向的tile数
#define BLOCK_ROW_TILES 4  // Block在行方向的tile数

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)  // 256个线程

// 每个block的工作区域
#define BLOCK_ROW_WARPS 2  // 行方向warp数
#define BLOCK_COL_WARPS 4  // 列方向warp数
```

**Block组织结构**：
```
Block (8 warps = 256 threads)
┌─────────┬─────────┐
│ Warp0   │ Warp1   │  ← 处理2×64矩阵块
├─────────┼─────────┤
│ Warp2   │ Warp3   │
├─────────┼─────────┤
│ Warp4   │ Warp5   │
├─────────┼─────────┤
│ Warp6   │ Warp7   │
└─────────┴─────────┘
```

### 1.4 共享内存优化参数
```cuda
#define CHUNK_K 4      // K维度分块大小
#define CHUNK_COPY_LINES_PER_WARP 2
#define CHUNK_COPY_LINE_LANES 4

#define SKEW_UINT8 32  // 避免bank conflict的偏移量
```

**CUDA新手解释**：
- **CHUNK_K**: 将K维度分成小块处理，减少共享内存使用
- **SKEW_UINT8**: 通过添加偏移避免共享内存的bank冲突，提高访问效率

**与其他精度对比**：
- **所有精度**都使用类似的分块策略，但具体参数根据数据类型调整
- INT8因为数据小，可以在共享内存中存储更多数据

---

## 第二部分：辅助函数

### 2.1 主机端矩阵初始化
```cuda
__host__ void init_host_matrices(uint8_t *a, uint8_t *b, int *c)
{
    for (int i = 0; i < M_GLOBAL; i++) {
        for (int j = 0; j < K_GLOBAL; j++) {
            a[i * K_GLOBAL + j] = (uint8_t)(rand() % 3);  // A矩阵：0,1,2随机值
        }
    }

    for (int i = 0; i < N_GLOBAL; i++) {
        for (int j = 0; j < K_GLOBAL; j++) {
            b[i * K_GLOBAL + j] = (uint8_t)(rand() % 3);  // B矩阵：0,1,2随机值
        }
    }

    for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
        c[t] = (rand() % 3);  // C矩阵：0,1,2随机值
    }
}
```

**CUDA新手解释**：
- 使用小的随机值（0,1,2）避免INT8溢出
- A和B是uint8_t类型，C和D是int类型（因为累加结果可能很大）

**与其他精度对比**：
- **FP16/BF16**: 使用更大范围的随机浮点数
- **TF32**: 类似FP32的初始化方式
- **FP64**: 使用double精度随机数

---

## 第三部分：核心GEMM Kernel分析

### 3.1 Kernel函数签名和共享内存声明
```cuda
__global__ void compute_gemm_imma(const uint8_t *A, const uint8_t *B, 
                                  const int *C, int *D, int alpha, int beta)
{
    extern __shared__ uint8_t shmem[][CHUNK_K * K + SKEW_UINT8];
```

**CUDA新手解释**：
- `extern __shared__`: 动态分配的共享内存
- `shmem[][]`: 二维数组，存储A和B矩阵的分块数据
- `SKEW_UINT8`: 避免bank conflict的padding

### 3.2 线程和Warp识别
```cuda
// Warp和lane识别
const unsigned int warpId = threadIdx.x / WARP_SIZE;  // 当前线程属于哪个warp (0-7)
const unsigned int laneId = threadIdx.x % WARP_SIZE;  // 在warp内的位置 (0-31)

// B矩阵在共享内存中的偏移
const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;
```

**CUDA新手解释**：
- 每个block有256个线程，分成8个warp
- 每个warp有32个线程（lane）
- warpId用于分配不同warp的工作
- laneId用于在warp内部分配具体任务

### 3.3 共享内存指针设置
```cuda
// 访问C和D矩阵tile的指针
int *shmem_warp_tile_ptr = (int *)&shmem[0][0] + 
    (warpId / 2) * SHMEM_STRIDE * K * 2 + (warpId % 2) * SHMEM_OFFSET;

// 流式传输C和D矩阵的指针
int *shmem_warp_stream_ptr = (int *)&shmem[0][0] + warpId * SHMEM_STRIDE * K;
```

**CUDA新手解释**：
- 不同warp访问共享内存的不同区域
- `shmem_warp_tile_ptr`: 用于WMMA操作的精确定位
- `shmem_warp_stream_ptr`: 用于批量数据传输

### 3.4 Block循环：遍历输出矩阵
```cuda
for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int block_tile_i = ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
    const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

    // 如果没有更多tile需要计算，退出
    if (block_tile_i >= M_TILES) {
        break;
    }
```

**CUDA新手解释**：
- 每个block处理输出矩阵的一个区域
- `block_pos`: 当前block的任务编号
- 使用Grid-Stride Loop模式，允许block数量少于总任务数

### 3.5 加载C矩阵到共享内存
```cuda
// 计算全局内存索引
const size_t gmem_idx = (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + block_tile_j * N;
const int *src_gmem_warp_stream_ptr = &C[gmem_idx];

// 流式传输多个C tiles到共享内存
#pragma unroll
for (int i = 0; i < K; i++) {
    typedef int4 copy_t;  // 一次复制16字节（4个int）

    *((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
        *((copy_t *)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId);
}
```

**CUDA新手解释**：
- `int4`: CUDA向量类型，一次传输4个int（16字节）
- `#pragma unroll`: 编译器指令，展开循环提高性能
- 每个warp的32个线程并行加载数据

### 3.6 将C矩阵加载到WMMA Fragment
```cuda
wmma::fragment<wmma::accumulator, M, N, K, int> c[WARP_COL_TILES][WARP_ROW_TILES];

#pragma unroll
for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
    for (int j = 0; j < WARP_ROW_TILES; j++) {
        const int *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;
        wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
    }
}
```

**CUDA新手解释**：
- `wmma::fragment`: TensorCore的数据结构，存储矩阵片段
- `wmma::load_matrix_sync`: 从共享内存加载数据到fragment
- 每个warp处理多个16×16的矩阵块

**与其他精度对比**：
- **FP16**: `wmma::fragment<..., half>`
- **BF16**: `wmma::fragment<..., __nv_bfloat16>`
- **TF32**: `wmma::fragment<..., float>` 但K=8

### 3.7 C矩阵缩放
```cuda
#pragma unroll
for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
    for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
        for (int t = 0; t < c[i][j].num_elements; t++) {
            c[i][j].x[t] *= beta;  // 提前应用beta系数
        }
    }
}
```

**CUDA新手解释**：
- 在计算开始前预先缩放C矩阵
- `num_elements`: fragment中的元素数量
- `c[i][j].x[t]`: 访问fragment中的具体元素

### 3.8 A和B矩阵数据加载设置
```cuda
// 决定哪个warp加载哪个矩阵
// Warp 0-3 加载A矩阵, warp 4-7 加载B矩阵
const uint8_t *warp_ptr = (warpId < 4) ? 
    (&A[block_tile_i * M * K_GLOBAL] + M * K_GLOBAL * (warpId % 4) * 2) :
    (&B[block_tile_j * N * K_GLOBAL] + N * K_GLOBAL * (warpId % 4) * 2);
```

**CUDA新手解释**：
- 工作分工：前4个warp负责A矩阵，后4个warp负责B矩阵
- 每个warp加载矩阵的不同行/列

### 3.9 K维度循环：主要计算部分
```cuda
#pragma unroll
for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
    // 第一步：加载A和B矩阵数据到共享内存
    size_t shmem_idx = warpId < (WARPS_PER_BLOCK / 2)
                        ? (M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
                        : (N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

    // 计算每个lane的数据指针
    int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * K + (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL)
                   + (laneId % CHUNK_COPY_LINE_LANES);

    shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

    // 复制数据到共享内存
#pragma unroll
    for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2; i++) {
        *((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) = *lane_ptr;
        
        lane_ptr = (int4 *)((uint8_t *)lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
        shmem_idx += CHUNK_COPY_LINES_PER_WARP;
    }

    __syncthreads();  // 确保所有线程完成内存加载
```

**CUDA新手解释**：
- K维度按CHUNK_K=4分块处理
- 使用`int4`向量化加载，每次16字节
- `__syncthreads()`: 同步所有线程，确保共享内存数据就绪

### 3.10 TensorCore矩阵乘法计算
```cuda
    // 第二步：使用TensorCore进行矩阵乘法
#pragma unroll
    for (int k_step = 0; k_step < CHUNK_K; k_step++) {
        wmma::fragment<wmma::matrix_a, M, N, K, uint8_t, wmma::row_major> a[WARP_COL_TILES];
        wmma::fragment<wmma::matrix_b, M, N, K, uint8_t, wmma::col_major> b[WARP_ROW_TILES];

#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
            size_t shmem_idx_a = (warpId / 2) * M * 2 + (i * M);
            const uint8_t *tile_ptr = &shmem[shmem_idx_a][k_step * K];

            wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_UINT8);

#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
                if (i == 0) {
                    // B矩阵fragment只需要加载一次，可以重复使用
                    size_t shmem_idx_b = shmem_idx_b_off + (WARP_ROW_TILES * N) * (warpId % 2) + (j * N);
                    const uint8_t *tile_ptr = &shmem[shmem_idx_b][k_step * K];

                    wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_UINT8);
                }

                wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);  // 核心TensorCore运算
            }
        }
    }

    __syncthreads();
}
```

**CUDA新手解释**：
- `wmma::matrix_a/matrix_b`: 指定矩阵类型和存储格式
- `wmma::row_major/col_major`: 指定矩阵的存储顺序
- `wmma::mma_sync`: 真正的TensorCore矩阵乘法运算
- 这就是TensorCore加速的核心！

**与其他精度对比**：
- **FP16**: `wmma::fragment<..., half, ...>`
- **BF16**: `wmma::fragment<..., __nv_bfloat16, ...>`
- **TF32**: 自动将float转换为TF32格式
- **FP64**: 使用double precision fragments

### 3.11 结果存储和输出
```cuda
// 第三步：将结果应用alpha系数并存储到共享内存
#pragma unroll
for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
    for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
        for (int t = 0; t < c[i][j].num_elements; t++)
            c[i][j].x[t] *= alpha;  // 应用alpha系数

        int *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;
        wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
    }
}

__syncthreads();

// 第四步：将结果从共享内存流式传输到全局内存
int *dst_gmem_warp_stream_ptr = &D[gmem_idx];

#pragma unroll
for (int i = 0; i < K; i++) {
    *((int4 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
        *((int4 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
}
```

**CUDA新手解释**：
- `wmma::store_matrix_sync`: 将fragment结果存储到共享内存
- 最后使用向量化传输将结果写回全局内存
- GEMM公式 D = α·A×B + β·C 完整实现！

---

## 第四部分：简化版WMMA Kernel

### 4.1 简化版本功能
```cuda
__global__ void simple_wmma_gemm_imma(const uint8_t *a, const uint8_t *b, 
                                      const int *c, int *d,
                                      int m_ld, int n_ld, int k_ld, 
                                      int alpha, int beta)
```

**CUDA新手解释**：
- 这个kernel不使用共享内存优化
- 直接从全局内存读取数据
- 性能较低，但代码更简单，适合学习

### 4.2 Warp级别的矩阵计算
```cuda
// 使用2D网格分配工作
int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

// 声明fragments
wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, uint8_t, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, uint8_t, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> acc_frag;
wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> c_frag;

wmma::fill_fragment(acc_frag, 0.0f);  // 初始化累加器为0
```

### 4.3 K维度循环
```cuda
for (int i = 0; i < k_ld; i += WMMA_K) {
    int aCol = i;
    int aRow = warpM * WMMA_M;
    int bCol = i;
    int bRow = warpN * WMMA_N;

    if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {
        // 直接从全局内存加载
        wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
        wmma::load_matrix_sync(b_frag, b + bCol + bRow * ldb, ldb);

        // 执行矩阵乘法
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
}
```

**CUDA新手解释**：
- 每次处理K方向的一个16元素块
- 直接从全局内存加载，没有共享内存缓存
- 这就是最基本的WMMA使用方式

---

## 第五部分：主函数和性能测试

### 5.1 设备检查
```cuda
// 检查GPU是否支持TensorCore
if (deviceProp.major < 7 || (deviceProp.major <= 7 && deviceProp.minor < 2)) {
    printf("immaTensorCoreGemm requires SM 7.2 or higher to use Tensor Cores. Exiting...\n");
    exit(EXIT_WAIVED);
}
```

**CUDA新手解释**：
- TensorCore需要SM 7.2及以上（V100, A100等）
- INT8 TensorCore特别需要Turing及以上架构

### 5.2 内存分配和数据传输
```cuda
// 主机内存分配
A_h = (uint8_t *)malloc(sizeof(uint8_t) * M_GLOBAL * K_GLOBAL);
B_h = (uint8_t *)malloc(sizeof(uint8_t) * K_GLOBAL * N_GLOBAL);
C_h = (int *)malloc(sizeof(int) * M_GLOBAL * N_GLOBAL);

// 设备内存分配
checkCudaErrors(cudaMalloc(&A, sizeof(uint8_t) * M_GLOBAL * K_GLOBAL));
checkCudaErrors(cudaMalloc(&B, sizeof(uint8_t) * N_GLOBAL * K_GLOBAL));
checkCudaErrors(cudaMalloc(&C, sizeof(int) * M_GLOBAL * N_GLOBAL));
checkCudaErrors(cudaMalloc(&D, sizeof(int) * M_GLOBAL * N_GLOBAL));

// 数据传输
checkCudaErrors(cudaMemcpy(A, A_h, sizeof(uint8_t) * M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice));
checkCudaErrors(cudaMemcpy(B, B_h, sizeof(uint8_t) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice));
checkCudaErrors(cudaMemcpy(C, C_h, sizeof(int) * M_GLOBAL * N_GLOBAL, cudaMemcpyHostToDevice));
```

### 5.3 共享内存检查和Kernel启动
```cuda
enum {
    SHMEM_SZ = MAX(sizeof(uint8_t) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_UINT8) * 2,
                   M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N * (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(int))
};

if (deviceProp.sharedMemPerMultiprocessor >= SHMEM_SZ) {
    // 使用高性能kernel
    checkCudaErrors(cudaFuncSetAttribute(compute_gemm_imma, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
    checkKernelErrors((compute_gemm_imma<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK, SHMEM_SZ>>>(
        A, B, C, D, alpha, beta)));
}
else {
    // 使用简化kernel
    simple_wmma_gemm_imma<<<gridDim, blockDim>>>(A, B, C, D, M_GLOBAL, N_GLOBAL, K_GLOBAL, alpha, beta);
}
```

**CUDA新手解释**：
- 程序会自动检查GPU的共享内存容量
- 如果足够，使用优化版本；否则使用简化版本
- 这是很好的编程实践：适应不同硬件能力

### 5.4 性能测量
```cuda
cudaEvent_t start, stop;
checkCudaErrors(cudaEventCreate(&start));
checkCudaErrors(cudaEventCreate(&stop));
checkCudaErrors(cudaEventRecord(start));

// ... kernel执行 ...

checkCudaErrors(cudaEventRecord(stop));
checkCudaErrors(cudaEventSynchronize(stop));

float milliseconds = 0;
checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

printf("Time: %f ms\n", milliseconds);
printf("TOPS: %.2f\n", (((double)M_GLOBAL * N_GLOBAL * K_GLOBAL * 2) / (milliseconds / 1000.)) / 1e12);
```

**CUDA新手解释**：
- 使用CUDA Event精确测量GPU时间
- TOPS = 每秒万亿次操作数，衡量计算性能的指标
- 乘以2是因为每次乘加操作包含一次乘法和一次加法

---

## 总结

### 关键技术点
1. **TensorCore编程**：使用WMMA API进行高效矩阵运算
2. **共享内存优化**：减少全局内存访问，提高数据重用
3. **向量化传输**：使用int4等向量类型提高内存带宽
4. **Bank Conflict避免**：通过SKEW技术优化共享内存访问
5. **工作分配**：合理分配warp和block的工作负载

### INT8的特殊优势
1. **内存效率**：相比FP16节省一半内存
2. **计算速度**：TensorCore对INT8有特殊优化
3. **推理友好**：量化模型的理想数据类型

### 与其他精度的主要区别
| 精度 | 数据类型 | 矩阵块大小 | 主要用途 |
|------|----------|------------|----------|
| INT8 | uint8_t/int | 16×16×16 | 推理 |
| FP16 | half | 16×16×16 | 训练/推理 |
| BF16 | __nv_bfloat16 | 16×16×16 | 训练 |
| TF32 | float | 16×16×8 | 训练 |
| FP64 | double | 8×8×4 | 科学计算 |

这个教程展示了现代GPU编程的精髓：充分利用硬件特性（TensorCore）、优化内存访问模式（共享内存）、以及使用适当的数据类型（INT8）来实现极致的性能。对于CUDA新手来说，理解这个实现将为更高级的GPU编程打下坚实基础。
