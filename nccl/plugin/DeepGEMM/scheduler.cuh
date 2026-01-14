#include "utils.cuh"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-member-init"
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t CLUSTER_M,
          uint32_t CLUSTER_N>
struct Scheduler {
    int current_iter = -1;

    uint32_t num_aligned_m_blocks;
    uint32_t num_aligned_n_blocks;
    uint32_t num_blocks;
    uint32_t row_block_in_cluster;
    uint32_t col_block_in_cluster;
    uint32_t idx_in_cluster;

    __device__ __forceinline__ explicit Scheduler(const uint32_t shape_m,
                                                  const uint32_t shape_n) {
        num_aligned_m_blocks = cell_div(shape_m, BLOCK_M);
        num_aligned_n_blocks = cell_div(shape_n, BLOCK_N);
        num_blocks = num_aligned_m_blocks * num_aligned_n_blocks;
    }

    __device__ __forceinline__ void get_swizzled_block_idx(
        int block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx) {
        constexpr uint32_t num_block_per_cluster = CLUSTER_M * CLUSTER_N;
        uint32_t NUM_CLUSTER_COL = num_aligned_n_blocks / CLUSTER_N;
        uint32_t cluster_row =
            block_idx / (num_block_per_cluster * NUM_CLUSTER_COL);
        uint32_t cluster_col =
            (block_idx % (num_block_per_cluster * NUM_CLUSTER_COL)) /
            num_block_per_cluster;
        idx_in_cluster = block_idx % num_block_per_cluster;
        row_block_in_cluster = idx_in_cluster / CLUSTER_N;
        col_block_in_cluster = idx_in_cluster % CLUSTER_N;
        m_block_idx = cluster_row * CLUSTER_M + row_block_in_cluster;
        n_block_idx = cluster_col * CLUSTER_N + col_block_in_cluster;
    }

    __device__ __forceinline__ bool get_next_block(uint32_t& m_block_idx,
                                                   uint32_t& n_block_idx) {
        const auto next_block_idx = (++current_iter) * gridDim.x + blockIdx.x;

        if (next_block_idx >= num_blocks) return false;

        get_swizzled_block_idx(next_block_idx, m_block_idx, n_block_idx);

        return true;
    }
};
#pragma clang diagnostic pop
