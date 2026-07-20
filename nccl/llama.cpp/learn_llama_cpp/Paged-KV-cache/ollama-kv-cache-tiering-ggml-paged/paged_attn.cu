/**
 * paged_attn.cu — CUDA kernels for paged ring attention.
 *
 * Core algorithm: online softmax (Milakov & Gimelshein 2018) applied
 * chunk-by-chunk as K/V pages stream from host memory through a
 * double-buffered pipeline.
 *
 * For each query head, one thread block maintains running state:
 *   m  = running maximum of attention logits
 *   l  = running sum of exp(logit - m)
 *   O  = running weighted sum of V rows
 *
 * After all chunks: output = O / l
 *
 * Target: CC ≥ 5.2 (Maxwell, Pascal, Volta, …)
 * Precision: f16 K/V loads, f32 accumulation.
 *
 * SPDX-License-Identifier: MIT
 */

#include "paged_attn.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

/* ───────────────── helpers ───────────────── */

#define PA_CHECK_CUDA(call)                                         \
    do {                                                            \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            return -1;                                              \
        }                                                           \
    } while (0)

#define PA_CHECK_CUDA_VOID(call)                                    \
    do {                                                            \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            return;                                                 \
        }                                                           \
    } while (0)

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

/* Warp-level sum reduction using shuffle. */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

/* Warp-level max reduction using shuffle. */
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    return val;
}

/* ───────────────── kernel: process one KV chunk ───────────────── */

/**
 * Each block handles one (batch, q_head) pair.
 * Threads are mapped 1:1 to head_dim elements (blockDim.x == D).
 * For D > 32 we have multiple warps — cross-warp reduction uses smem.
 *
 * Grid:  (num_q_heads, batch_size)
 * Block: (D)   where D = head_dim (must be ≤ 1024, typically 64–256)
 *
 * Shared memory layout:
 *   float scores[chunk_len]   — attention logits for this chunk
 *   float warp_scratch[num_warps]  — for cross-warp reductions
 *
 * Template parameter D avoids dynamic indexing overhead.
 */
template <int D>
__global__ void __launch_bounds__(D)
paged_attn_chunk_kernel(
    const half * __restrict__ Q,          /* [batch, num_q_heads, D]                 */
    const half * __restrict__ K_chunk,    /* [chunk_len, num_kv_heads, D]            */
    const half * __restrict__ V_chunk,    /* [chunk_len, num_kv_heads, D]            */
    float      * __restrict__ m_state,    /* [batch, num_q_heads]        — running max     */
    float      * __restrict__ l_state,    /* [batch, num_q_heads]        — running sum_exp */
    float      * __restrict__ O_state,    /* [batch, num_q_heads, D]     — running output  */
    const int   chunk_len,                /* actual positions in this chunk (≤ chunk_size)  */
    const int   num_q_heads,
    const int   num_kv_heads,
    const float scale,
    const int   is_first_chunk            /* 1 → initialize state, 0 → accumulate */
) {
    const int q_head  = blockIdx.x;
    const int batch   = blockIdx.y;
    const int tid     = threadIdx.x;              /* 0 .. D-1 */

    const int kv_head = q_head * num_kv_heads / num_q_heads;  /* GQA mapping */

    /* ---- indices into global arrays ---- */
    const int state_idx = batch * num_q_heads + q_head;
    const int out_base  = state_idx * D;
    const int q_offset  = (batch * num_q_heads + q_head) * D;

    /* ---- load Q into register ---- */
    const float q_val = __half2float(Q[q_offset + tid]);

    /* ---- shared memory ---- */
    constexpr int NUM_WARPS = (D + 31) / 32;
    extern __shared__ char smem_raw[];
    float *scores = reinterpret_cast<float *>(smem_raw);
    float *warp_scratch = scores + chunk_len;
    /* Also store m and l in shared so all threads can read them. */
    float *sm_ml = warp_scratch + NUM_WARPS;   /* sm_ml[0]=m, sm_ml[1]=l */

    const int warp_id = tid / 32;
    const int lane    = tid % 32;

    /* ---- phase 1: compute Q·K scores for all positions in chunk ---- */
    for (int pos = 0; pos < chunk_len; pos++) {
        /* K layout: [chunk_len, num_kv_heads, D] */
        const int k_idx = (pos * num_kv_heads + kv_head) * D + tid;
        float partial = q_val * __half2float(K_chunk[k_idx]);

        /* Intra-warp reduction */
        partial = warp_reduce_sum(partial);

        /* Cross-warp reduction via shared memory */
        if (NUM_WARPS > 1) {
            if (lane == 0) warp_scratch[warp_id] = partial;
            __syncthreads();
            if (warp_id == 0) {
                float v = (lane < NUM_WARPS) ? warp_scratch[lane] : 0.0f;
                v = warp_reduce_sum(v);
                if (lane == 0) scores[pos] = v * scale;
            }
            __syncthreads();
        } else {
            if (lane == 0) scores[pos] = partial * scale;
            __syncthreads();
        }
    }

    /* ---- phase 2: find chunk-local max ---- */
    float m_chunk = -FLT_MAX;
    for (int pos = tid; pos < chunk_len; pos += D)
        m_chunk = fmaxf(m_chunk, scores[pos]);
    /* Reduce across all threads */
    m_chunk = warp_reduce_max(m_chunk);
    if (NUM_WARPS > 1) {
        if (lane == 0) warp_scratch[warp_id] = m_chunk;
        __syncthreads();
        if (warp_id == 0) {
            float v = (lane < NUM_WARPS) ? warp_scratch[lane] : -FLT_MAX;
            v = warp_reduce_max(v);
            if (lane == 0) sm_ml[0] = v;  /* chunk max → shared */
        }
        __syncthreads();
        m_chunk = sm_ml[0];
    }

    /* ---- phase 3: compute correction and accumulate ---- */
    float m_old, l_old, o_val;
    if (is_first_chunk) {
        m_old = -FLT_MAX;
        l_old = 0.0f;
        o_val = 0.0f;
    } else {
        m_old = m_state[state_idx];
        l_old = l_state[state_idx];
        o_val = O_state[out_base + tid];
    }

    float m_new = fmaxf(m_old, m_chunk);
    float correction = (m_old > -FLT_MAX) ? expf(m_old - m_new) : 0.0f;

    /* Rescale old accumulator */
    o_val *= correction;
    l_old *= correction;

    /* Accumulate this chunk's contribution */
    float l_local = 0.0f;
    for (int pos = 0; pos < chunk_len; pos++) {
        float w = expf(scores[pos] - m_new);
        l_local += w;

        /* V layout: [chunk_len, num_kv_heads, D] */
        const int v_idx = (pos * num_kv_heads + kv_head) * D + tid;
        o_val += w * __half2float(V_chunk[v_idx]);
    }

    /* Reduce l_local across threads (all threads computed same scores but
       we need only one copy of l — use thread 0's copy which is correct
       since scores[] is in shared memory and all threads read the same values). */
    /* Actually all threads compute the same l_local since scores[] is shared.
       So any thread's value is correct. No reduction needed for l. */

    /* ---- phase 4: write state back ---- */
    O_state[out_base + tid] = o_val;
    if (tid == 0) {
        m_state[state_idx] = m_new;
        l_state[state_idx] = l_old + l_local;
    }
}

/* ───────────────── kernel: normalize final output ───────────────── */

template <int D>
__global__ void __launch_bounds__(D)
paged_attn_normalize_kernel(
    const float * __restrict__ O_state,   /* [batch, num_q_heads, D] */
    const float * __restrict__ l_state,   /* [batch, num_q_heads]    */
    half        * __restrict__ output,    /* [batch, num_q_heads, D] */
    const int num_q_heads
) {
    const int q_head = blockIdx.x;
    const int batch  = blockIdx.y;
    const int tid    = threadIdx.x;

    const int state_idx = batch * num_q_heads + q_head;
    const float l = l_state[state_idx];
    const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;

    const int base = state_idx * D;
    output[base + tid] = __float2half(O_state[base + tid] * inv_l);
}

/* ───────────────── context implementation ───────────────── */

#define PA_MAX_LAYERS 128

struct pa_ctx {
    int          num_kv_heads;
    int          head_dim;
    int          chunk_size;
    pa_dtype_t   dtype;
    int          device;

    /* Double-buffer on GPU (ping-pong) */
    void        *k_buf[2];
    void        *v_buf[2];
    size_t       chunk_bytes;     /* per buffer: chunk_size * num_kv_heads * head_dim * elem */

    /* Attention state on GPU: m, l, O — allocated per forward call */
    float       *m_dev;
    float       *l_dev;
    float       *o_dev;
    int          state_capacity;  /* batch * num_q_heads — current allocation */

    /* Copy stream for async H→D transfers */
    cudaStream_t copy_stream;

    /* Per-layer host KV registration */
    struct {
        void *k_host;
        void *v_host;
        int   total_pos;
    } layers[PA_MAX_LAYERS];
    int          num_layers_registered;

    /* Stats */
    pa_stats_t   stats;
};

static size_t pa_elem_size(pa_dtype_t dt) {
    return (dt == PA_DTYPE_F16) ? sizeof(half) : sizeof(float);
}

pa_ctx_t *pa_ctx_create(int num_kv_heads, int head_dim, int chunk_size,
                        pa_dtype_t dtype, int device)
{
    pa_ctx_t *ctx = (pa_ctx_t *)calloc(1, sizeof(pa_ctx_t));
    if (!ctx) return NULL;

    ctx->num_kv_heads = num_kv_heads;
    ctx->head_dim     = head_dim;
    ctx->chunk_size   = chunk_size;
    ctx->dtype        = dtype;
    ctx->device       = device;

    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) { free(ctx); return NULL; }

    /* Allocate double-buffer on GPU */
    size_t elem = pa_elem_size(dtype);
    ctx->chunk_bytes = (size_t)chunk_size * num_kv_heads * head_dim * elem;

    for (int i = 0; i < 2; i++) {
        err = cudaMalloc(&ctx->k_buf[i], ctx->chunk_bytes);
        if (err != cudaSuccess) goto fail;
        err = cudaMalloc(&ctx->v_buf[i], ctx->chunk_bytes);
        if (err != cudaSuccess) goto fail;
    }

    err = cudaStreamCreateWithFlags(&ctx->copy_stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) goto fail;

    ctx->state_capacity = 0;
    ctx->m_dev = NULL;
    ctx->l_dev = NULL;
    ctx->o_dev = NULL;

    return ctx;

fail:
    pa_ctx_destroy(ctx);
    return NULL;
}

void pa_ctx_destroy(pa_ctx_t *ctx) {
    if (!ctx) return;
    cudaSetDevice(ctx->device);
    for (int i = 0; i < 2; i++) {
        if (ctx->k_buf[i]) cudaFree(ctx->k_buf[i]);
        if (ctx->v_buf[i]) cudaFree(ctx->v_buf[i]);
    }
    if (ctx->m_dev) cudaFree(ctx->m_dev);
    if (ctx->l_dev) cudaFree(ctx->l_dev);
    if (ctx->o_dev) cudaFree(ctx->o_dev);
    if (ctx->copy_stream) cudaStreamDestroy(ctx->copy_stream);
    free(ctx);
}

int pa_register_host_kv(pa_ctx_t *ctx, int layer,
                        void *k_host, void *v_host, int total_pos)
{
    if (!ctx || layer < 0 || layer >= PA_MAX_LAYERS) return -1;
    ctx->layers[layer].k_host    = k_host;
    ctx->layers[layer].v_host    = v_host;
    ctx->layers[layer].total_pos = total_pos;
    if (layer >= ctx->num_layers_registered)
        ctx->num_layers_registered = layer + 1;
    return 0;
}

/* Ensure state buffers are large enough. */
static int pa_ensure_state(pa_ctx_t *ctx, int capacity, int head_dim) {
    if (capacity <= ctx->state_capacity) return 0;

    cudaSetDevice(ctx->device);
    if (ctx->m_dev) cudaFree(ctx->m_dev);
    if (ctx->l_dev) cudaFree(ctx->l_dev);
    if (ctx->o_dev) cudaFree(ctx->o_dev);

    PA_CHECK_CUDA(cudaMalloc(&ctx->m_dev, capacity * sizeof(float)));
    PA_CHECK_CUDA(cudaMalloc(&ctx->l_dev, capacity * sizeof(float)));
    PA_CHECK_CUDA(cudaMalloc(&ctx->o_dev, capacity * head_dim * sizeof(float)));
    ctx->state_capacity = capacity;
    return 0;
}

/* ───────────────── dispatch: select kernel by head_dim ───────────────── */

typedef void (*chunk_launcher_t)(
    const half*, const half*, const half*,
    float*, float*, float*,
    int, int, int, float, int,
    int, cudaStream_t);

template <int D>
static void launch_chunk_kernel(
    const half *Q, const half *K, const half *V,
    float *m, float *l, float *O,
    int chunk_len, int num_q_heads, int num_kv_heads,
    float scale, int is_first,
    int batch_size, cudaStream_t stream)
{
    dim3 grid(num_q_heads, batch_size);
    dim3 block(D);
    constexpr int NUM_WARPS = (D + 31) / 32;
    /* smem: scores[chunk_len] + warp_scratch[NUM_WARPS] + ml[2] */
    size_t smem = (size_t)chunk_len * sizeof(float)
                + NUM_WARPS * sizeof(float)
                + 2 * sizeof(float);

    paged_attn_chunk_kernel<D><<<grid, block, smem, stream>>>(
        Q, K, V, m, l, O,
        chunk_len, num_q_heads, num_kv_heads, scale, is_first);
}

template <int D>
static void launch_normalize_kernel(
    const float *O, const float *l, half *output,
    int num_q_heads, int batch_size, cudaStream_t stream)
{
    dim3 grid(num_q_heads, batch_size);
    dim3 block(D);
    paged_attn_normalize_kernel<D><<<grid, block, 0, stream>>>(
        O, l, output, num_q_heads);
}

/* ───────────────── forward pass ───────────────── */

int pa_forward(pa_ctx_t *ctx, int layer,
               const void *Q_dev, void *output_dev,
               int batch_size, int num_q_heads, int seq_len,
               float scale, void *compute_stream_ptr)
{
    if (!ctx) return -1;
    if (layer < 0 || layer >= PA_MAX_LAYERS) return -1;
    if (!ctx->layers[layer].k_host || !ctx->layers[layer].v_host) return -1;
    if (seq_len <= 0) return -1;

    cudaStream_t compute_stream = (cudaStream_t)compute_stream_ptr;
    const int D = ctx->head_dim;
    const int num_kv_heads = ctx->num_kv_heads;
    const size_t elem = pa_elem_size(ctx->dtype);
    const int chunk_size = ctx->chunk_size;
    const int num_chunks = ceil_div(seq_len, chunk_size);

    /* Ensure state buffers */
    int state_cap = batch_size * num_q_heads;
    if (pa_ensure_state(ctx, state_cap, D) != 0) return -1;

    const char *k_host = (const char *)ctx->layers[layer].k_host;
    const char *v_host = (const char *)ctx->layers[layer].v_host;
    const size_t row_bytes = (size_t)num_kv_heads * D * elem;

    /* Prefetch first chunk into buf[0] */
    int first_len = (seq_len < chunk_size) ? seq_len : chunk_size;
    size_t first_bytes = (size_t)first_len * row_bytes;
    PA_CHECK_CUDA(cudaMemcpyAsync(ctx->k_buf[0], k_host, first_bytes,
                                   cudaMemcpyHostToDevice, ctx->copy_stream));
    PA_CHECK_CUDA(cudaMemcpyAsync(ctx->v_buf[0], v_host, first_bytes,
                                   cudaMemcpyHostToDevice, ctx->copy_stream));
    PA_CHECK_CUDA(cudaStreamSynchronize(ctx->copy_stream));

    int ping = 0;

    for (int c = 0; c < num_chunks; c++) {
        int chunk_start = c * chunk_size;
        int chunk_len   = ((chunk_start + chunk_size) <= seq_len)
                            ? chunk_size
                            : (seq_len - chunk_start);

        /* Start async copy of NEXT chunk into the other buffer */
        if (c + 1 < num_chunks) {
            int next_start = (c + 1) * chunk_size;
            int next_len   = ((next_start + chunk_size) <= seq_len)
                               ? chunk_size
                               : (seq_len - next_start);
            size_t next_bytes = (size_t)next_len * row_bytes;
            size_t next_off   = (size_t)next_start * row_bytes;

            PA_CHECK_CUDA(cudaMemcpyAsync(
                ctx->k_buf[1 - ping], k_host + next_off, next_bytes,
                cudaMemcpyHostToDevice, ctx->copy_stream));
            PA_CHECK_CUDA(cudaMemcpyAsync(
                ctx->v_buf[1 - ping], v_host + next_off, next_bytes,
                cudaMemcpyHostToDevice, ctx->copy_stream));
        }

        /* Make compute stream wait for copy stream (for current chunk) */
        cudaEvent_t event;
        PA_CHECK_CUDA(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        PA_CHECK_CUDA(cudaEventRecord(event, ctx->copy_stream));
        PA_CHECK_CUDA(cudaStreamWaitEvent(compute_stream, event, 0));
        PA_CHECK_CUDA(cudaEventDestroy(event));

        /* Launch chunk kernel */
        const half *K_cur = (const half *)ctx->k_buf[ping];
        const half *V_cur = (const half *)ctx->v_buf[ping];
        const half *Q_ptr = (const half *)Q_dev;
        int is_first = (c == 0) ? 1 : 0;

        /* Dispatch by head_dim */
        switch (D) {
        case 64:
            launch_chunk_kernel<64>(Q_ptr, K_cur, V_cur,
                ctx->m_dev, ctx->l_dev, ctx->o_dev,
                chunk_len, num_q_heads, num_kv_heads,
                scale, is_first, batch_size, compute_stream);
            break;
        case 80:
            launch_chunk_kernel<80>(Q_ptr, K_cur, V_cur,
                ctx->m_dev, ctx->l_dev, ctx->o_dev,
                chunk_len, num_q_heads, num_kv_heads,
                scale, is_first, batch_size, compute_stream);
            break;
        case 96:
            launch_chunk_kernel<96>(Q_ptr, K_cur, V_cur,
                ctx->m_dev, ctx->l_dev, ctx->o_dev,
                chunk_len, num_q_heads, num_kv_heads,
                scale, is_first, batch_size, compute_stream);
            break;
        case 128:
            launch_chunk_kernel<128>(Q_ptr, K_cur, V_cur,
                ctx->m_dev, ctx->l_dev, ctx->o_dev,
                chunk_len, num_q_heads, num_kv_heads,
                scale, is_first, batch_size, compute_stream);
            break;
        case 256:
            launch_chunk_kernel<256>(Q_ptr, K_cur, V_cur,
                ctx->m_dev, ctx->l_dev, ctx->o_dev,
                chunk_len, num_q_heads, num_kv_heads,
                scale, is_first, batch_size, compute_stream);
            break;
        default:
            fprintf(stderr, "paged_attn: unsupported head_dim=%d\n", D);
            return -1;
        }

        ctx->stats.chunks_processed++;
        ctx->stats.bytes_transferred += (int64_t)chunk_len * row_bytes * 2; /* K + V */

        /* Make copy stream wait for compute stream before overwriting the buffer */
        PA_CHECK_CUDA(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        PA_CHECK_CUDA(cudaEventRecord(event, compute_stream));
        PA_CHECK_CUDA(cudaStreamWaitEvent(ctx->copy_stream, event, 0));
        PA_CHECK_CUDA(cudaEventDestroy(event));

        ping = 1 - ping;
    }

    /* Normalize: output = O / l */
    half *out_ptr = (half *)output_dev;
    switch (D) {
    case 64:  launch_normalize_kernel<64> (ctx->o_dev, ctx->l_dev, out_ptr, num_q_heads, batch_size, compute_stream); break;
    case 80:  launch_normalize_kernel<80> (ctx->o_dev, ctx->l_dev, out_ptr, num_q_heads, batch_size, compute_stream); break;
    case 96:  launch_normalize_kernel<96> (ctx->o_dev, ctx->l_dev, out_ptr, num_q_heads, batch_size, compute_stream); break;
    case 128: launch_normalize_kernel<128>(ctx->o_dev, ctx->l_dev, out_ptr, num_q_heads, batch_size, compute_stream); break;
    case 256: launch_normalize_kernel<256>(ctx->o_dev, ctx->l_dev, out_ptr, num_q_heads, batch_size, compute_stream); break;
    default:  return -1;
    }

    return 0;
}

/* ───────────────── stats ───────────────── */

pa_stats_t pa_get_stats(const pa_ctx_t *ctx) {
    return ctx ? ctx->stats : pa_stats_t{};
}

void pa_reset_stats(pa_ctx_t *ctx) {
    if (ctx) memset(&ctx->stats, 0, sizeof(ctx->stats));
}
