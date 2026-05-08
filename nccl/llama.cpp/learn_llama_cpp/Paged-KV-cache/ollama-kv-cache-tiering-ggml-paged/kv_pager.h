/**
 * kv_pager.h — Host-side KV cache page manager.
 *
 * Manages a three-tier memory hierarchy for KV cache data:
 *
 *   Tier 0 (hot)  : GPU VRAM        — handled by GGML, not by us
 *   Tier 1 (warm) : Pinned host RAM  — fast async transfer to GPU
 *   Tier 2 (cold) : Disk (SSD/NFS)  — unlimited capacity
 *
 * This module owns tiers 1 and 2.  The paged attention kernel
 * (paged_attn.cu) consumes tier 1 data through double-buffered
 * H→D copies.  When tier 1 fills up, older chunks are spilled
 * to tier 2 (disk) and loaded back on demand.
 *
 * Thread safety: all public functions are serialized by an internal mutex.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef KV_PAGER_H
#define KV_PAGER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ───────────────── types ───────────────── */

typedef struct kv_pager kv_pager_t;

typedef struct {
    int    num_layers;
    int    num_kv_heads;
    int    head_dim;
    int    elem_bytes;          /* 2 for f16, 4 for f32 */

    /* Tier 1: pinned host memory budget (bytes). */
    int64_t host_budget_bytes;

    /* Tier 2: disk paths (NULL to disable). */
    const char *local_disk_path;   /* fast SSD tier */
    const char *remote_disk_path;  /* slow NFS/HDD tier */
    int64_t local_disk_budget;     /* bytes */
    int64_t remote_disk_budget;    /* bytes */
} kv_pager_config_t;

typedef struct {
    int64_t host_used_bytes;
    int64_t host_capacity_bytes;
    int64_t disk_local_used_bytes;
    int64_t disk_remote_used_bytes;
    int     total_positions;       /* across all layers */
    int     host_positions;        /* positions in host RAM */
    int     disk_positions;        /* positions on disk */
} kv_pager_stats_t;

/* ───────────────── lifetime ───────────────── */

kv_pager_t *kv_pager_create(const kv_pager_config_t *config);
void        kv_pager_destroy(kv_pager_t *pager);

/* ───────────────── storing KV data ───────────────── */

/**
 * Append a KV pair at the next position for a given layer.
 *
 * @param pager  The pager instance.
 * @param layer  Layer index.
 * @param k_data Pointer to K data: [num_kv_heads, head_dim] in elem_bytes.
 * @param v_data Pointer to V data: same layout.
 * @return       The position index assigned, or -1 on error.
 */
int kv_pager_append(kv_pager_t *pager, int layer,
                    const void *k_data, const void *v_data);

/**
 * Store a KV pair at a specific position.
 * Overwrites any existing data at that position.
 */
int kv_pager_store(kv_pager_t *pager, int layer, int pos,
                   const void *k_data, const void *v_data);

/* ───────────────── retrieving KV data ───────────────── */

/**
 * Get a contiguous range of KV data for a layer, suitable for
 * passing to pa_register_host_kv().
 *
 * Returns pointers to pinned host memory containing:
 *   K: [count, num_kv_heads, head_dim]
 *   V: [count, num_kv_heads, head_dim]
 *
 * If some positions are on disk, they are loaded into the host buffer first.
 * The pointers remain valid until the next call to kv_pager_get_range()
 * or kv_pager_destroy().
 *
 * @param pager   The pager instance.
 * @param layer   Layer index.
 * @param start   Starting position (inclusive).
 * @param count   Number of positions.
 * @param k_out   Output: pointer to K data (pinned host memory).
 * @param v_out   Output: pointer to V data (pinned host memory).
 * @return        Number of positions actually available, or -1 on error.
 */
int kv_pager_get_range(kv_pager_t *pager, int layer, int start, int count,
                       void **k_out, void **v_out);

/**
 * Get the full KV for a layer (convenience wrapper).
 * Equivalent to kv_pager_get_range(pager, layer, 0, total_pos, k_out, v_out).
 */
int kv_pager_get_layer(kv_pager_t *pager, int layer,
                       void **k_out, void **v_out);

/* ───────────────── eviction ───────────────── */

/**
 * Remove all positions in range [start, start+count) for all layers.
 * Frees host memory and deletes disk files.
 */
int kv_pager_remove_range(kv_pager_t *pager, int start, int count);

/**
 * Remove all data for a specific sequence.
 */
int kv_pager_clear(kv_pager_t *pager);

/* ───────────────── stats ───────────────── */

kv_pager_stats_t kv_pager_get_stats(const kv_pager_t *pager);

#ifdef __cplusplus
}
#endif

#endif /* KV_PAGER_H */
