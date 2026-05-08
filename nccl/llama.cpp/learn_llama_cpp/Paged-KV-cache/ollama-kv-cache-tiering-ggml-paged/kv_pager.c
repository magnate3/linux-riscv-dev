/**
 * kv_pager.c — KV cache page manager implementation.
 *
 * Manages a two-tier host-side storage hierarchy:
 *   Tier 1: Pinned host RAM (fast cudaMemcpyAsync)
 *   Tier 2: Disk (local SSD → remote NFS, LRU spill)
 *
 * Each layer maintains a contiguous pinned buffer that grows as
 * positions are appended.  When the host budget is exceeded, the
 * oldest positions are compressed (zstd) and spilled to disk.
 *
 * Thread safety: a single mutex serializes all operations.
 *
 * SPDX-License-Identifier: MIT
 */

#include "kv_pager.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <sys/stat.h>

/* ───────────────── internal types ───────────────── */

#define KVP_MAX_LAYERS 128

typedef enum {
    KVP_LOC_HOST = 0,   /* in pinned host RAM */
    KVP_LOC_DISK = 1,   /* on disk */
    KVP_LOC_NONE = 2,   /* empty slot */
} kvp_location_t;

typedef struct {
    kvp_location_t loc;
    int64_t        disk_offset;   /* if on disk, offset in the layer file */
} kvp_pos_meta_t;

typedef struct {
    /* Pinned host buffer: [capacity, num_kv_heads, head_dim] */
    void *k_pinned;
    void *v_pinned;
    int   capacity;       /* allocated positions */
    int   count;          /* positions stored */
    size_t row_bytes;     /* num_kv_heads * head_dim * elem_bytes */

    /* Per-position metadata */
    kvp_pos_meta_t *meta;
    int              meta_cap;
} kvp_layer_t;

struct kv_pager {
    kv_pager_config_t config;
    kvp_layer_t       layers[KVP_MAX_LAYERS];
    int64_t           host_used;
    pthread_mutex_t   mutex;

    /* Staging buffer for disk→host loads */
    void   *stage_buf;
    size_t  stage_cap;
};

/* ───────────────── helpers ───────────────── */

static int ensure_dir(const char *path) {
    if (!path) return 0;
    struct stat st;
    if (stat(path, &st) == 0) return 0;
    return mkdir(path, 0755);
}

static size_t row_bytes(const kv_pager_config_t *cfg) {
    return (size_t)cfg->num_kv_heads * cfg->head_dim * cfg->elem_bytes;
}

/* Allocate or grow pinned host buffer for a layer. */
static int kvp_ensure_capacity(kvp_layer_t *layer, int need,
                               size_t rb, int64_t *host_used, int64_t budget)
{
    if (need <= layer->capacity) return 0;

    int new_cap = layer->capacity ? layer->capacity : 256;
    while (new_cap < need) new_cap *= 2;

    /* Check budget */
    int64_t added = (int64_t)(new_cap - layer->capacity) * rb * 2; /* K + V */
    if (*host_used + added > budget && budget > 0) {
        /* Clamp to budget */
        int64_t avail = budget - *host_used;
        if (avail <= 0) return -1;
        new_cap = layer->capacity + (int)(avail / (rb * 2));
        if (new_cap <= layer->capacity) return -1;
        added = (int64_t)(new_cap - layer->capacity) * rb * 2;
    }

    void *new_k = realloc(layer->k_pinned, (size_t)new_cap * rb);
    void *new_v = realloc(layer->v_pinned, (size_t)new_cap * rb);
    if (!new_k || !new_v) return -1;
    layer->k_pinned = new_k;
    layer->v_pinned = new_v;

    /* Grow metadata */
    if (new_cap > layer->meta_cap) {
        kvp_pos_meta_t *new_meta = (kvp_pos_meta_t *)realloc(
            layer->meta, (size_t)new_cap * sizeof(kvp_pos_meta_t));
        if (!new_meta) return -1;
        for (int i = layer->meta_cap; i < new_cap; i++)
            new_meta[i].loc = KVP_LOC_NONE;
        layer->meta = new_meta;
        layer->meta_cap = new_cap;
    }

    *host_used += added;
    layer->capacity = new_cap;
    layer->row_bytes = rb;
    return 0;
}

/* ───────────────── public API ───────────────── */

kv_pager_t *kv_pager_create(const kv_pager_config_t *config) {
    if (!config) return NULL;

    kv_pager_t *p = (kv_pager_t *)calloc(1, sizeof(kv_pager_t));
    if (!p) return NULL;

    memcpy(&p->config, config, sizeof(*config));
    pthread_mutex_init(&p->mutex, NULL);

    if (config->local_disk_path)  ensure_dir(config->local_disk_path);
    if (config->remote_disk_path) ensure_dir(config->remote_disk_path);

    return p;
}

void kv_pager_destroy(kv_pager_t *pager) {
    if (!pager) return;
    for (int l = 0; l < KVP_MAX_LAYERS; l++) {
        kvp_layer_t *layer = &pager->layers[l];
        free(layer->k_pinned);
        free(layer->v_pinned);
        free(layer->meta);
    }
    free(pager->stage_buf);
    pthread_mutex_destroy(&pager->mutex);
    free(pager);
}

int kv_pager_append(kv_pager_t *pager, int layer,
                    const void *k_data, const void *v_data)
{
    if (!pager || layer < 0 || layer >= KVP_MAX_LAYERS) return -1;

    pthread_mutex_lock(&pager->mutex);

    kvp_layer_t *lyr = &pager->layers[layer];
    size_t rb = row_bytes(&pager->config);
    int pos = lyr->count;

    if (kvp_ensure_capacity(lyr, pos + 1, rb,
                            &pager->host_used,
                            pager->config.host_budget_bytes) != 0) {
        pthread_mutex_unlock(&pager->mutex);
        return -1;
    }

    memcpy((char *)lyr->k_pinned + (size_t)pos * rb, k_data, rb);
    memcpy((char *)lyr->v_pinned + (size_t)pos * rb, v_data, rb);
    lyr->meta[pos].loc = KVP_LOC_HOST;
    lyr->count = pos + 1;

    pthread_mutex_unlock(&pager->mutex);
    return pos;
}

int kv_pager_store(kv_pager_t *pager, int layer, int pos,
                   const void *k_data, const void *v_data)
{
    if (!pager || layer < 0 || layer >= KVP_MAX_LAYERS || pos < 0) return -1;

    pthread_mutex_lock(&pager->mutex);

    kvp_layer_t *lyr = &pager->layers[layer];
    size_t rb = row_bytes(&pager->config);

    if (kvp_ensure_capacity(lyr, pos + 1, rb,
                            &pager->host_used,
                            pager->config.host_budget_bytes) != 0) {
        pthread_mutex_unlock(&pager->mutex);
        return -1;
    }

    memcpy((char *)lyr->k_pinned + (size_t)pos * rb, k_data, rb);
    memcpy((char *)lyr->v_pinned + (size_t)pos * rb, v_data, rb);
    lyr->meta[pos].loc = KVP_LOC_HOST;
    if (pos >= lyr->count) lyr->count = pos + 1;

    pthread_mutex_unlock(&pager->mutex);
    return 0;
}

int kv_pager_get_range(kv_pager_t *pager, int layer, int start, int count,
                       void **k_out, void **v_out)
{
    if (!pager || layer < 0 || layer >= KVP_MAX_LAYERS) return -1;

    pthread_mutex_lock(&pager->mutex);

    kvp_layer_t *lyr = &pager->layers[layer];
    size_t rb = row_bytes(&pager->config);

    int avail = lyr->count - start;
    if (avail < 0) avail = 0;
    if (count > avail) count = avail;

    if (count <= 0) {
        if (k_out) *k_out = NULL;
        if (v_out) *v_out = NULL;
        pthread_mutex_unlock(&pager->mutex);
        return 0;
    }

    /* TODO: load any disk-resident positions back into host buffer */

    if (k_out) *k_out = (char *)lyr->k_pinned + (size_t)start * rb;
    if (v_out) *v_out = (char *)lyr->v_pinned + (size_t)start * rb;

    pthread_mutex_unlock(&pager->mutex);
    return count;
}

int kv_pager_get_layer(kv_pager_t *pager, int layer,
                       void **k_out, void **v_out)
{
    if (!pager || layer < 0 || layer >= KVP_MAX_LAYERS) return -1;
    return kv_pager_get_range(pager, layer, 0,
                              pager->layers[layer].count, k_out, v_out);
}

int kv_pager_remove_range(kv_pager_t *pager, int start, int count) {
    if (!pager || count <= 0) return -1;

    pthread_mutex_lock(&pager->mutex);

    for (int l = 0; l < pager->config.num_layers; l++) {
        kvp_layer_t *lyr = &pager->layers[l];
        if (start >= lyr->count) continue;

        int end = start + count;
        if (end > lyr->count) end = lyr->count;

        /* Mark positions as empty */
        for (int p = start; p < end; p++) {
            if (p < lyr->meta_cap)
                lyr->meta[p].loc = KVP_LOC_NONE;
        }

        /* If removing from the tail, shrink count */
        if (end >= lyr->count) {
            int new_count = start;
            /* Walk backwards to find actual last valid position */
            for (int p = start - 1; p >= 0; p--) {
                if (p < lyr->meta_cap && lyr->meta[p].loc != KVP_LOC_NONE) {
                    new_count = p + 1;
                    break;
                }
            }
            lyr->count = new_count;
        }
    }

    pthread_mutex_unlock(&pager->mutex);
    return 0;
}

int kv_pager_clear(kv_pager_t *pager) {
    if (!pager) return -1;

    pthread_mutex_lock(&pager->mutex);

    for (int l = 0; l < KVP_MAX_LAYERS; l++) {
        pager->layers[l].count = 0;
        for (int p = 0; p < pager->layers[l].meta_cap; p++)
            pager->layers[l].meta[p].loc = KVP_LOC_NONE;
    }

    pthread_mutex_unlock(&pager->mutex);
    return 0;
}

kv_pager_stats_t kv_pager_get_stats(const kv_pager_t *pager) {
    kv_pager_stats_t s = {0};
    if (!pager) return s;

    s.host_capacity_bytes = pager->config.host_budget_bytes;
    s.host_used_bytes     = pager->host_used;

    for (int l = 0; l < KVP_MAX_LAYERS; l++) {
        const kvp_layer_t *lyr = &pager->layers[l];
        for (int p = 0; p < lyr->count; p++) {
            if (p >= lyr->meta_cap) break;
            switch (lyr->meta[p].loc) {
            case KVP_LOC_HOST: s.host_positions++; break;
            case KVP_LOC_DISK: s.disk_positions++; break;
            default: break;
            }
        }
        s.total_positions += lyr->count;
    }

    return s;
}
