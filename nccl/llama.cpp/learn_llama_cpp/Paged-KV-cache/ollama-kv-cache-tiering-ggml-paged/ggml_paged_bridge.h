/**
 * ggml_paged_bridge.h — Bridge between GGML graph execution and paged attention.
 *
 * This file provides the function that GGML's CUDA backend calls when it
 * encounters our custom paged attention op (GGML_OP_FLASH_ATTN_EXT_PAGED).
 *
 * Integration approach:
 *   1. During graph build, ggml_flash_attn_ext_paged() creates a node whose
 *      src[1] (K) and src[2] (V) live on the HOST backend (pinned memory)
 *      while src[0] (Q) and dst are on the CUDA backend.
 *
 *   2. During graph execution, GGML dispatches the op to this bridge,
 *      which invokes our double-buffered paged attention kernel.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef GGML_PAGED_BRIDGE_H
#define GGML_PAGED_BRIDGE_H

#include "paged_attn.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Compute paged attention for a GGML tensor operation.
 *
 * Called by the GGML CUDA backend's op dispatcher.
 *
 * @param Q_data     Device pointer to Q: [head_dim, num_q_heads, seq_q, batch]
 * @param K_data     HOST pointer to K:   [head_dim, num_kv_heads, total_seq, batch]
 * @param V_data     HOST pointer to V:   [head_dim, num_kv_heads, total_seq, batch]  
 * @param dst_data   Device pointer to output: [head_dim_v, num_q_heads, seq_q, batch]
 * @param head_dim   Dimension per head (d_k).
 * @param head_dim_v Dimension per head for V (d_v, usually == head_dim).
 * @param num_q_heads Number of query heads.
 * @param num_kv_heads Number of key/value heads.
 * @param seq_q      Number of query positions (batch dim for current token(s)).
 * @param total_seq  Total sequence length in K/V.
 * @param batch      Batch size (ne[3]).
 * @param scale      Attention scale (1/sqrt(d_k)).
 * @param chunk_size Chunk size for paging (0 = auto-select).
 * @param device     CUDA device ordinal.
 * @param stream     CUDA stream.
 * @return           0 on success.
 */
int ggml_paged_attn_compute(
    const void *Q_data,         /* device */
    const void *K_data,         /* host (pinned) */
    const void *V_data,         /* host (pinned) */
    void       *dst_data,       /* device */
    int head_dim,
    int head_dim_v,
    int num_q_heads,
    int num_kv_heads,
    int seq_q,
    int total_seq,
    int batch,
    float scale,
    int chunk_size,
    int device,
    void *stream
);

/**
 * One-time initialization. Called lazily on first use.
 * Thread-safe (uses pthread_once internally).
 */
void ggml_paged_attn_init(void);

/** Cleanup. Called at program exit. */
void ggml_paged_attn_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* GGML_PAGED_BRIDGE_H */
