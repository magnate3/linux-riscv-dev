/**
 * paged_attn.h — Paged (ring) attention for offloaded KV caches.
 *
 * Extends the effective context window beyond GPU VRAM by streaming KV
 * chunks from host memory (or disk) through a double-buffered pipeline:
 *
 *   disk/NFS  ──read──▶  pinned host  ──cudaMemcpyAsync──▶  GPU buf
 *                           buffer          (stream B)       (ping)
 *                                                              │
 *                                        ┌─── compute ◀───────┘
 *                                        │    (stream A)
 *                                        ▼
 *                              online-softmax accumulator
 *                              (m, l, O) — no full materialization
 *
 * Key properties:
 *   • Only ONE chunk of K and ONE chunk of V live on GPU at a time
 *   • Uses online softmax (Milakov & Gimelshein 2018) to combine chunks
 *     without materializing the full attention matrix
 *   • Supports GQA (grouped query attention)
 *   • Works on CC ≥ 5.2 (Maxwell+) — no tensor cores required
 *   • f16 K/V with f32 accumulation (mixed precision)
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef PAGED_ATTN_H
#define PAGED_ATTN_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ───────────────── data types ───────────────── */

typedef enum {
    PA_DTYPE_F16 = 0,
    PA_DTYPE_F32 = 1,
} pa_dtype_t;

/* Describes one ping-pong GPU buffer pair for K and V. */
typedef struct {
    void *k[2];          /* k[0] = ping, k[1] = pong  — device pointers */
    void *v[2];          /* same for V */
    int   chunk_size;    /* max positions per chunk */
    int   num_kv_heads;
    int   head_dim;
    size_t chunk_bytes;  /* bytes per chunk per tensor (chunk_size * num_kv_heads * head_dim * elem_size) */
} pa_gpu_bufs_t;

/* Per-layer host-side KV storage (pinned or pageable). */
typedef struct {
    void  *data;         /* contiguous [total_pos, num_kv_heads, head_dim] in dtype */
    int    total_pos;    /* how many positions are stored */
} pa_host_kv_t;

/* Full paged-attention context. */
typedef struct pa_ctx pa_ctx_t;

/* ───────────────── lifetime ───────────────── */

/**
 * Allocate and initialize a paged attention context.
 *
 * @param num_kv_heads  Number of KV heads (may differ from Q heads for GQA).
 * @param head_dim      Dimension per head (e.g. 128).
 * @param chunk_size    Positions per chunk (default 2048, power of 2 recommended).
 * @param dtype         Data type for K/V storage (PA_DTYPE_F16 recommended).
 * @param device        CUDA device ordinal.
 * @return              Opaque context, or NULL on error.
 */
pa_ctx_t *pa_ctx_create(int num_kv_heads, int head_dim, int chunk_size,
                        pa_dtype_t dtype, int device);

/** Free all resources. */
void pa_ctx_destroy(pa_ctx_t *ctx);

/* ───────────────── host KV management ───────────────── */

/**
 * Register host-side KV for a layer.
 * The memory MUST be pinned (cudaMallocHost) for async transfer.
 * Layout: [total_pos, num_kv_heads, head_dim], row-major, in ctx->dtype.
 *
 * This does NOT take ownership — caller is responsible for lifetime.
 */
int pa_register_host_kv(pa_ctx_t *ctx, int layer,
                        void *k_host, void *v_host, int total_pos);

/* ───────────────── forward pass ───────────────── */

/**
 * Compute paged attention for one layer.
 *
 * The function handles the double-buffered paging loop internally:
 *   for each chunk c = 0 .. ceil(total_pos / chunk_size) - 1:
 *       async-copy chunk c+1 from host → GPU buffer (copy stream)
 *       kernel: process chunk c on GPU buffer (compute stream)
 *       swap ping/pong
 *   finalize: output = O / l
 *
 * @param ctx           Context.
 * @param layer         Layer index (selects registered host KV).
 * @param Q_dev         Query tensor on GPU: [batch, num_q_heads, head_dim] in f16.
 * @param output_dev    Output tensor on GPU: [batch, num_q_heads, head_dim] in f16.
 * @param batch_size    Number of query positions (typically 1 during generation).
 * @param num_q_heads   Number of query heads (≥ num_kv_heads for GQA).
 * @param seq_len       Total sequence length (positions in KV to attend over).
 * @param scale         Attention scale factor (typically 1/sqrt(head_dim)).
 * @param compute_stream  CUDA stream for kernel launches.
 * @return              0 on success, negative on error.
 */
int pa_forward(pa_ctx_t *ctx, int layer,
               const void *Q_dev, void *output_dev,
               int batch_size, int num_q_heads, int seq_len,
               float scale, void *compute_stream);

/* ───────────────── diagnostics ───────────────── */

typedef struct {
    int64_t chunks_processed;
    int64_t bytes_transferred;   /* host → device */
    double  transfer_time_ms;
    double  compute_time_ms;
} pa_stats_t;

pa_stats_t pa_get_stats(const pa_ctx_t *ctx);
void       pa_reset_stats(pa_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* PAGED_ATTN_H */
