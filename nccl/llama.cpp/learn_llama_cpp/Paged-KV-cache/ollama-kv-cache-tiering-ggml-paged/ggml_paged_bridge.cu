/**
 * ggml_paged_bridge.cu — Bridge implementation.
 *
 * Manages a pool of pa_ctx_t objects (one per unique head_dim)
 * and dispatches GGML paged attention ops through them.
 *
 * SPDX-License-Identifier: MIT
 */

#include "ggml_paged_bridge.h"
#include "paged_attn.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ───────── context pool (keyed by head_dim × num_kv_heads × device) ───────── */

#define MAX_PA_CONTEXTS 16

typedef struct {
    pa_ctx_t *ctx;
    int       head_dim;
    int       num_kv_heads;
    int       chunk_size;
    int       device;
} pa_pool_entry_t;

static pa_pool_entry_t g_pool[MAX_PA_CONTEXTS];
static int             g_pool_count = 0;
static pthread_mutex_t g_pool_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t  g_init_once  = PTHREAD_ONCE_INIT;
static int             g_initialized = 0;

static void do_init(void) {
    memset(g_pool, 0, sizeof(g_pool));
    g_initialized = 1;
}

void ggml_paged_attn_init(void) {
    pthread_once(&g_init_once, do_init);
}

void ggml_paged_attn_cleanup(void) {
    pthread_mutex_lock(&g_pool_mutex);
    for (int i = 0; i < g_pool_count; i++) {
        pa_ctx_destroy(g_pool[i].ctx);
        g_pool[i].ctx = NULL;
    }
    g_pool_count = 0;
    pthread_mutex_unlock(&g_pool_mutex);
}

static pa_ctx_t *get_or_create_ctx(int num_kv_heads, int head_dim,
                                   int chunk_size, int device)
{
    ggml_paged_attn_init();

    pthread_mutex_lock(&g_pool_mutex);

    /* Look for existing */
    for (int i = 0; i < g_pool_count; i++) {
        if (g_pool[i].head_dim     == head_dim &&
            g_pool[i].num_kv_heads == num_kv_heads &&
            g_pool[i].chunk_size   == chunk_size &&
            g_pool[i].device       == device)
        {
            pa_ctx_t *ctx = g_pool[i].ctx;
            pthread_mutex_unlock(&g_pool_mutex);
            return ctx;
        }
    }

    /* Create new */
    if (g_pool_count >= MAX_PA_CONTEXTS) {
        fprintf(stderr, "paged_attn: context pool exhausted\n");
        pthread_mutex_unlock(&g_pool_mutex);
        return NULL;
    }

    pa_ctx_t *ctx = pa_ctx_create(num_kv_heads, head_dim, chunk_size,
                                  PA_DTYPE_F16, device);
    if (ctx) {
        g_pool[g_pool_count] = (pa_pool_entry_t){
            .ctx          = ctx,
            .head_dim     = head_dim,
            .num_kv_heads = num_kv_heads,
            .chunk_size   = chunk_size,
            .device       = device,
        };
        g_pool_count++;
    }

    pthread_mutex_unlock(&g_pool_mutex);
    return ctx;
}

/* ───────── compute dispatch ───────── */

int ggml_paged_attn_compute(
    const void *Q_data,
    const void *K_data,
    const void *V_data,
    void       *dst_data,
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
    void *stream)
{
    if (head_dim != head_dim_v) {
        fprintf(stderr, "paged_attn: head_dim != head_dim_v not yet supported "
                        "(%d vs %d)\n", head_dim, head_dim_v);
        return -1;
    }

    if (chunk_size <= 0) {
        /* Auto-select: use 2048 for long sequences, smaller for short */
        chunk_size = (total_seq > 4096) ? 2048 : 512;
    }

    pa_ctx_t *ctx = get_or_create_ctx(num_kv_heads, head_dim, chunk_size, device);
    if (!ctx) return -1;

    /*
     * GGML tensor layout for KV cache (Causal):
     *   K: [head_dim, num_kv_heads, total_seq]   — contiguous, row-major
     *   V: [head_dim, num_kv_heads, total_seq]
     *
     * Our kernel expects:
     *   K: [total_seq, num_kv_heads, head_dim]
     *
     * But actually, the K/V cache in GGML Causal is stored as:
     *   shape [head_dim, num_kv_heads, numCtx]
     *   stride: [1, head_dim, head_dim * num_kv_heads]
     *
     * This means memory layout is: [pos][kv_head][dim] which IS what we want.
     * (ne[0]=head_dim is the fastest varying dimension)
     *
     * Wait — GGML uses column-major convention for ne[] but row-major storage.
     * ne[0] is innermost = head_dim, ne[1] = num_kv_heads, ne[2] = seq_len.
     * So in memory: data[seq * num_kv_heads * head_dim + kv_head * head_dim + d]
     * This matches our [seq, kv_head, D] layout. 
     */

    /* Register host KV for layer 0 (we re-register each call since
       the host pointer and total_pos may change). */
    pa_register_host_kv(ctx, 0, (void *)K_data, (void *)V_data, total_seq);

    /* For now, process one batch item at a time (batch is typically 1 for generation) */
    int rc = 0;
    size_t kv_row_bytes = (size_t)num_kv_heads * head_dim * sizeof(half);
    size_t q_row_bytes  = (size_t)num_q_heads  * head_dim * sizeof(half);

    for (int b = 0; b < batch; b++) {
        /* Q for this batch: [head_dim, num_q_heads, seq_q] at batch offset */
        const char *Q_b = (const char *)Q_data + b * seq_q * q_row_bytes;
        char *dst_b = (char *)dst_data + b * seq_q * q_row_bytes;

        /* For each query position in the batch */
        for (int sq = 0; sq < seq_q; sq++) {
            const half *Q_pos = (const half *)(Q_b + sq * q_row_bytes);
            half *out_pos = (half *)(dst_b + sq * q_row_bytes);

            rc = pa_forward(ctx, 0, Q_pos, out_pos,
                            1 /* batch=1 per position */, num_q_heads,
                            total_seq, scale, stream);
            if (rc != 0) return rc;
        }
    }

    return rc;
}
