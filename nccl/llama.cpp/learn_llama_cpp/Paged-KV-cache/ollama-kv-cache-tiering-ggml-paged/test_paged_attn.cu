/**
 * test_paged_attn.cu — Tests for paged ring attention kernel.
 *
 * Validates that the chunked online-softmax approach produces the same
 * output as a reference full-attention implementation.
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

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t e = (call);                                     \
        if (e != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA ERROR %s:%d: %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(e));     \
            exit(1);                                                \
        }                                                           \
    } while (0)

/* ───────── reference: standard softmax attention on CPU ───────── */
static void ref_attention_f32(
    const float *Q,   /* [num_q_heads, D]          */
    const float *K,   /* [seq_len, num_kv_heads, D] */
    const float *V,   /* [seq_len, num_kv_heads, D] */
    float *output,    /* [num_q_heads, D]           */
    int num_q_heads, int num_kv_heads, int D, int seq_len, float scale)
{
    int gqa_ratio = num_q_heads / num_kv_heads;
    for (int qh = 0; qh < num_q_heads; qh++) {
        int kv_h = qh / gqa_ratio;

        /* Compute scores */
        float *scores = (float *)calloc(seq_len, sizeof(float));
        float m = -FLT_MAX;
        for (int s = 0; s < seq_len; s++) {
            float dot = 0;
            for (int d = 0; d < D; d++)
                dot += Q[qh * D + d] * K[s * num_kv_heads * D + kv_h * D + d];
            scores[s] = dot * scale;
            if (scores[s] > m) m = scores[s];
        }

        /* Softmax */
        float sum = 0;
        for (int s = 0; s < seq_len; s++) {
            scores[s] = expf(scores[s] - m);
            sum += scores[s];
        }
        for (int s = 0; s < seq_len; s++)
            scores[s] /= sum;

        /* Weighted sum of V */
        for (int d = 0; d < D; d++) {
            float val = 0;
            for (int s = 0; s < seq_len; s++)
                val += scores[s] * V[s * num_kv_heads * D + kv_h * D + d];
            output[qh * D + d] = val;
        }

        free(scores);
    }
}

/* ───────── test harness ───────── */

static int test_basic_attention(int num_q_heads, int num_kv_heads,
                                int D, int seq_len, int chunk_size)
{
    printf("  test: Q_heads=%d KV_heads=%d D=%d seq=%d chunk=%d ... ",
           num_q_heads, num_kv_heads, D, seq_len, chunk_size);
    fflush(stdout);

    const int batch_size = 1;
    float scale = 1.0f / sqrtf((float)D);

    /* Allocate and fill random data (f32 for ref, f16 for kernel) */
    size_t q_elems  = (size_t)batch_size * num_q_heads * D;
    size_t kv_elems = (size_t)seq_len * num_kv_heads * D;

    float *Q_f32 = (float *)malloc(q_elems * sizeof(float));
    float *K_f32 = (float *)malloc(kv_elems * sizeof(float));
    float *V_f32 = (float *)malloc(kv_elems * sizeof(float));
    float *ref_out = (float *)calloc(q_elems, sizeof(float));

    srand(42);
    for (size_t i = 0; i < q_elems; i++)
        Q_f32[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (size_t i = 0; i < kv_elems; i++) {
        K_f32[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        V_f32[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }

    /* Convert to f16 first, then back to f32 for reference (apples-to-apples) */
    half *Q_f16 = (half *)malloc(q_elems * sizeof(half));
    half *K_f16 = (half *)malloc(kv_elems * sizeof(half));
    half *V_f16 = (half *)malloc(kv_elems * sizeof(half));
    for (size_t i = 0; i < q_elems; i++)  Q_f16[i] = __float2half(Q_f32[i]);
    for (size_t i = 0; i < kv_elems; i++) K_f16[i] = __float2half(K_f32[i]);
    for (size_t i = 0; i < kv_elems; i++) V_f16[i] = __float2half(V_f32[i]);

    /* Convert f16 back to f32 for reference — this ensures the reference sees
       the same quantized values the kernel sees. */
    float *Q_ref = (float *)malloc(q_elems * sizeof(float));
    float *K_ref = (float *)malloc(kv_elems * sizeof(float));
    float *V_ref = (float *)malloc(kv_elems * sizeof(float));
    for (size_t i = 0; i < q_elems; i++)  Q_ref[i] = __half2float(Q_f16[i]);
    for (size_t i = 0; i < kv_elems; i++) K_ref[i] = __half2float(K_f16[i]);
    for (size_t i = 0; i < kv_elems; i++) V_ref[i] = __half2float(V_f16[i]);

    /* Reference output (using f16-quantized values, computed in f32) */
    ref_attention_f32(Q_ref, K_ref, V_ref, ref_out,
                      num_q_heads, num_kv_heads, D, seq_len, scale);

    /* Allocate pinned host buffers for K, V (required for async copy) */
    half *K_host, *V_host;
    CHECK_CUDA(cudaMallocHost(&K_host, kv_elems * sizeof(half)));
    CHECK_CUDA(cudaMallocHost(&V_host, kv_elems * sizeof(half)));
    memcpy(K_host, K_f16, kv_elems * sizeof(half));
    memcpy(V_host, V_f16, kv_elems * sizeof(half));

    /* GPU: Q input and output */
    half *Q_dev, *out_dev;
    CHECK_CUDA(cudaMalloc(&Q_dev, q_elems * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&out_dev, q_elems * sizeof(half)));
    CHECK_CUDA(cudaMemcpy(Q_dev, Q_f16, q_elems * sizeof(half),
                          cudaMemcpyHostToDevice));

    /* Create paged attention context */
    pa_ctx_t *ctx = pa_ctx_create(num_kv_heads, D, chunk_size,
                                  PA_DTYPE_F16, 0);
    if (!ctx) {
        printf("FAIL (ctx_create)\n");
        return 1;
    }

    /* Register host KV */
    pa_register_host_kv(ctx, 0, K_host, V_host, seq_len);

    /* Run paged attention */
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    int rc = pa_forward(ctx, 0, Q_dev, out_dev,
                        batch_size, num_q_heads, seq_len, scale, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    if (rc != 0) {
        printf("FAIL (pa_forward returned %d)\n", rc);
        pa_ctx_destroy(ctx);
        return 1;
    }

    /* Read back result */
    half *out_f16 = (half *)malloc(q_elems * sizeof(half));
    CHECK_CUDA(cudaMemcpy(out_f16, out_dev, q_elems * sizeof(half),
                          cudaMemcpyDeviceToHost));

    /* Compare */
    float max_err = 0;
    float max_abs_err = 0;
    float avg_err = 0;
    int outliers = 0;
    for (size_t i = 0; i < q_elems; i++) {
        float got = __half2float(out_f16[i]);
        float ref = ref_out[i];
        float abs_err = fabsf(got - ref);
        /* Use absolute error threshold to avoid noise on near-zero values */
        float rel = (fabsf(ref) > 1e-4f) ? (abs_err / fabsf(ref)) : abs_err;
        if (rel > max_err) max_err = rel;
        if (abs_err > max_abs_err) max_abs_err = abs_err;
        if (rel > 0.01f) outliers++;
        avg_err += rel;
    }
    avg_err /= (float)q_elems;

    /* Tolerance: average must be < 0.5%, max must be < 5%
       (accounts for f16→f32 accumulation rounding across many positions) */
    float avg_tol = 0.005f;
    float max_tol = 0.05f;
    int pass = (avg_err < avg_tol && max_err < max_tol);

    if (pass) {
        printf("PASS (max_rel=%.3f%% avg=%.4f%% max_abs=%.6f)\n",
               max_err * 100, avg_err * 100, max_abs_err);
    } else {
        printf("FAIL (max_rel=%.3f%% avg=%.4f%% max_abs=%.6f outliers=%d)\n",
               max_err * 100, avg_err * 100, max_abs_err, outliers);
    }

    /* Print stats */
    pa_stats_t stats = pa_get_stats(ctx);
    printf("         chunks=%ld  bytes_xfer=%ld\n",
           (long)stats.chunks_processed, (long)stats.bytes_transferred);

    /* Cleanup */
    pa_ctx_destroy(ctx);
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(Q_dev));
    CHECK_CUDA(cudaFree(out_dev));
    CHECK_CUDA(cudaFreeHost(K_host));
    CHECK_CUDA(cudaFreeHost(V_host));
    free(Q_f32); free(K_f32); free(V_f32);
    free(Q_ref); free(K_ref); free(V_ref);
    free(Q_f16); free(K_f16); free(V_f16);
    free(ref_out); free(out_f16);

    return pass ? 0 : 1;
}

int main() {
    int device;
    cudaError_t e = cudaGetDevice(&device);
    if (e != cudaSuccess) {
        fprintf(stderr, "No CUDA device available: %s\n", cudaGetErrorString(e));
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Paged Attention Tests — %s (CC %d.%d)\n\n", prop.name, prop.major, prop.minor);

    int failures = 0;

    /* Basic tests: single KV head, varying seq_len and chunk_size */
    printf("1. Single-head tests:\n");
    failures += test_basic_attention(1, 1, 128, 64,   64);    /* 1 chunk */
    failures += test_basic_attention(1, 1, 128, 128,  64);    /* 2 chunks */
    failures += test_basic_attention(1, 1, 128, 300,  128);   /* 3 chunks (last partial) */
    failures += test_basic_attention(1, 1, 128, 2048, 256);   /* 8 chunks */

    /* GQA tests */
    printf("\n2. GQA tests:\n");
    failures += test_basic_attention(8,  2, 128, 256, 128);   /* GQA ratio 4 */
    failures += test_basic_attention(40, 8, 128, 512, 256);   /* GQA ratio 5 (like qwen2.5) */

    /* Different head dims */
    printf("\n3. Head dimension tests:\n");
    failures += test_basic_attention(4, 4, 64,  256, 128);
    failures += test_basic_attention(4, 4, 96,  256, 128);
    failures += test_basic_attention(4, 4, 128, 256, 128);

    /* Longer sequences (stress test) */
    printf("\n4. Long sequence tests:\n");
    failures += test_basic_attention(4, 4, 128, 4096, 1024);
    failures += test_basic_attention(8, 2, 128, 8192, 2048);

    printf("\n%s: %d/%d tests passed\n",
           failures ? "FAILURE" : "SUCCESS",
           11 - failures, 11);

    return failures;
}
