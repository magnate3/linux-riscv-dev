// KV Cache Compaction via Attention Matching - Math Utilities
//
// Pure CPU float32 linear algebra routines used by the compaction algorithm.
// Extracted for testability.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

// ============================================================================
// Linear algebra utilities (CPU-side, float32)
// ============================================================================

// Compute C = A * B^T  where A is (m x k), B is (n x k), result is (m x n)
static void mat_mul_ABt(const float * A, const float * B, float * C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[j * k + l];
            }
            C[i * n + j] = sum;
        }
    }
}

// Compute C = A^T * B  where A is (m x k), B is (m x n), result is (k x n)
static void mat_mul_AtB(const float * A, const float * B, float * C, int m, int k, int n) {
    // zero out C
    memset(C, 0, k * n * sizeof(float));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            for (int l = 0; l < n; l++) {
                C[j * n + l] += A[i * k + j] * B[i * n + l];
            }
        }
    }
}

// Softmax over rows: input (m x n), output (m x n), in-place safe
static void softmax_rows(float * data, int m, int n) {
    for (int i = 0; i < m; i++) {
        float * row = data + i * n;
        float max_val = row[0];
        for (int j = 1; j < n; j++) {
            if (row[j] > max_val) max_val = row[j];
        }
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            row[j] = expf(row[j] - max_val);
            sum += row[j];
        }
        float inv_sum = 1.0f / (sum + 1e-12f);
        for (int j = 0; j < n; j++) {
            row[j] *= inv_sum;
        }
    }
}

// Row-wise exp with max-shift for numerical stability: input (m x n)
// Returns exp(data - max_per_row) and stores the sum per row in row_sums
static void exp_rows_stable(float * data, float * row_sums, int m, int n) {
    for (int i = 0; i < m; i++) {
        float * row = data + i * n;
        float max_val = row[0];
        for (int j = 1; j < n; j++) {
            if (row[j] > max_val) max_val = row[j];
        }
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            row[j] = expf(row[j] - max_val);
            sum += row[j];
        }
        row_sums[i] = sum;
    }
}

// Solve non-negative least squares via projected gradient descent:
//   min_{w >= 0} ||A*w - b||^2
// A is (m x n), b is (m), w is (n)
// Returns solution in w
static void nnls_solve(const float * A, const float * b, float * w, int m, int n, int max_iter = 200) {
    // Precompute A^T * A and A^T * b
    std::vector<float> AtA(n * n);
    std::vector<float> Atb(n);

    // AtA = A^T * A
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < m; k++) {
                sum += A[k * n + i] * A[k * n + j];
            }
            AtA[i * n + j] = sum;
        }
    }

    // Atb = A^T * b
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int k = 0; k < m; k++) {
            sum += A[k * n + i] * b[k];
        }
        Atb[i] = sum;
    }

    // Initialize w to unconstrained least squares, clamped to >= 0
    // Simple init: w = max(0, (A^T A)^{-1} A^T b) via gradient descent from w=1
    for (int i = 0; i < n; i++) {
        w[i] = 1.0f;
    }

    // Compute step size: 1 / (max eigenvalue of AtA) ≈ 1 / (trace(AtA))
    float trace = 0.0f;
    for (int i = 0; i < n; i++) {
        trace += AtA[i * n + i];
    }
    float step = 1.0f / (trace + 1e-8f);

    // Projected gradient descent
    std::vector<float> grad(n);
    for (int iter = 0; iter < max_iter; iter++) {
        // grad = AtA * w - Atb
        for (int i = 0; i < n; i++) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                sum += AtA[i * n + j] * w[j];
            }
            grad[i] = sum - Atb[i];
        }

        // w = max(0, w - step * grad)
        for (int i = 0; i < n; i++) {
            w[i] = std::max(1e-12f, w[i] - step * grad[i]);
        }
    }
}

// Solve least squares: min ||A*x - b||^2 via normal equations
// A is (m x n), b is (m x p), x is (n x p)
// Uses Cholesky-like approach: x = (A^T A)^{-1} A^T b
// For simplicity, uses pseudo-inverse via regularized normal equations
static void least_squares_solve(const float * A, const float * b, float * x,
                                int m, int n, int p, float ridge = 1e-6f) {
    // Compute AtA = A^T * A  (n x n)
    std::vector<float> AtA(n * n, 0.0f);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < m; k++) {
                sum += A[k * n + i] * A[k * n + j];
            }
            AtA[i * n + j] = sum;
        }
    }

    // Add ridge regularization
    for (int i = 0; i < n; i++) {
        AtA[i * n + i] += ridge;
    }

    // Compute Atb = A^T * b  (n x p)
    std::vector<float> Atb(n * p, 0.0f);
    for (int i = 0; i < n; i++) {
        for (int l = 0; l < p; l++) {
            float sum = 0.0f;
            for (int k = 0; k < m; k++) {
                sum += A[k * n + i] * b[k * p + l];
            }
            Atb[i * p + l] = sum;
        }
    }

    // Solve AtA * x = Atb via Gaussian elimination with partial pivoting
    // Augmented matrix [AtA | Atb]
    std::vector<float> aug(n * (n + p));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aug[i * (n + p) + j] = AtA[i * n + j];
        }
        for (int j = 0; j < p; j++) {
            aug[i * (n + p) + n + j] = Atb[i * p + j];
        }
    }

    // Forward elimination with partial pivoting
    for (int col = 0; col < n; col++) {
        // Find pivot
        int max_row = col;
        float max_val = fabsf(aug[col * (n + p) + col]);
        for (int row = col + 1; row < n; row++) {
            float val = fabsf(aug[row * (n + p) + col]);
            if (val > max_val) {
                max_val = val;
                max_row = row;
            }
        }

        // Swap rows
        if (max_row != col) {
            for (int j = 0; j < n + p; j++) {
                std::swap(aug[col * (n + p) + j], aug[max_row * (n + p) + j]);
            }
        }

        float pivot = aug[col * (n + p) + col];
        if (fabsf(pivot) < 1e-12f) {
            continue; // skip singular column
        }

        // Eliminate below
        for (int row = col + 1; row < n; row++) {
            float factor = aug[row * (n + p) + col] / pivot;
            for (int j = col; j < n + p; j++) {
                aug[row * (n + p) + j] -= factor * aug[col * (n + p) + j];
            }
        }
    }

    // Back substitution
    for (int col = n - 1; col >= 0; col--) {
        float pivot = aug[col * (n + p) + col];
        if (fabsf(pivot) < 1e-12f) {
            for (int j = 0; j < p; j++) {
                x[col * p + j] = 0.0f;
            }
            continue;
        }
        for (int j = 0; j < p; j++) {
            float val = aug[col * (n + p) + n + j];
            for (int row = col + 1; row < n; row++) {
                val -= aug[col * (n + p) + row] * x[row * p + j];
            }
            x[col * p + j] = val / pivot;
        }
    }
}

// ============================================================================
// Compaction algorithm types and implementation
// ============================================================================

struct compacted_head {
    std::vector<int>   selected_indices;  // which original tokens were selected
    std::vector<float> beta;              // attention mass biases [t]
    std::vector<float> C_v;               // refit values [t * d_v]
};

// Result of compacting all heads within a single layer
struct compacted_layer {
    std::vector<int>   selected_indices;  // [t] shared token selection across all heads
    int                n_head_kv;         // number of KV heads
    int                t;                 // compacted size
    int                d_k;               // key dimension per head
    int                d_v;               // value dimension per head

    // Per-head results: beta[h] is [t], C_v[h] is [t * d_v]
    std::vector<std::vector<float>> beta;  // [n_head_kv][t]
    std::vector<std::vector<float>> C_v;   // [n_head_kv][t * d_v]
};

// Compact a single KV head using the Highest Attention Keys method
//
//   K:       [T, d_k] original keys for this head
//   V:       [T, d_v] original values for this head
//   Q_ref:   [n_q, d_k] reference queries
//   t:       target compacted size
//   d_k:     key dimension
//   d_v:     value dimension
//
// Returns compacted_head with selected indices, beta, and C_v
static compacted_head compact_head_highest_attn(
        const float * K, const float * V, const float * Q_ref,
        int T, int n_q, int d_k, int d_v, int t) {

    compacted_head result;
    result.selected_indices.resize(t);
    result.beta.resize(t);
    result.C_v.resize(t * d_v);

    if (t >= T) {
        // No compaction needed
        for (int i = 0; i < T; i++) result.selected_indices[i] = i;
        std::fill(result.beta.begin(), result.beta.end(), 0.0f);
        memcpy(result.C_v.data(), V, T * d_v * sizeof(float));
        return result;
    }

    // Step 1: Compute attention scores Q_ref @ K^T / sqrt(d_k)
    //   scores: [n_q, T]
    const float inv_sqrt_dk = 1.0f / sqrtf((float) d_k);
    std::vector<float> scores(n_q * T);
    mat_mul_ABt(Q_ref, K, scores.data(), n_q, T, d_k);
    for (int i = 0; i < n_q * T; i++) {
        scores[i] *= inv_sqrt_dk;
    }

    // Compute exp(scores) with max-shift for mass computation
    std::vector<float> exp_scores(scores); // copy
    std::vector<float> row_sums(n_q);
    exp_rows_stable(exp_scores.data(), row_sums.data(), n_q, T);

    // Compute softmax attention weights for key scoring
    std::vector<float> attn_weights(scores);
    softmax_rows(attn_weights.data(), n_q, T);

    // Score each key: max attention weight across queries
    std::vector<float> key_scores(T, 0.0f);
    for (int j = 0; j < T; j++) {
        float max_score = 0.0f;
        for (int i = 0; i < n_q; i++) {
            float w = attn_weights[i * T + j];
            if (w > max_score) max_score = w;
        }
        key_scores[j] = max_score;
    }

    // Select top-t keys by score
    std::vector<int> indices(T);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + t, indices.end(),
                      [&](int a, int b) { return key_scores[a] > key_scores[b]; });

    // Sort selected indices for cache locality
    std::vector<int> selected(indices.begin(), indices.begin() + t);
    std::sort(selected.begin(), selected.end());
    result.selected_indices = selected;

    // Step 2: Solve NNLS for beta (mass matching)
    //   We want: sum_j exp(q_i * C_k_j / sqrt(d)) * w_j ≈ sum_j exp(q_i * K_j / sqrt(d))
    //   where C_k are the selected keys and w_j = exp(beta_j)
    //
    //   Design matrix M: M_ij = exp(q_i * K_{selected[j]} / sqrt(d))
    //   Target: m_i = sum_j exp(q_i * K_j / sqrt(d)) = row_sums[i] (already computed)

    std::vector<float> M(n_q * t);
    for (int i = 0; i < n_q; i++) {
        for (int j = 0; j < t; j++) {
            M[i * t + j] = exp_scores[i * T + selected[j]];
        }
    }

    // Target mass: already computed as row_sums
    std::vector<float> w(t);
    nnls_solve(M.data(), row_sums.data(), w.data(), n_q, t);

    // beta = log(w)
    for (int j = 0; j < t; j++) {
        result.beta[j] = logf(std::max(1e-12f, w[j]));
    }

    // Step 3: Solve least squares for C_v (value fitting)
    //   We want: softmax(q * C_k^T + beta) * C_v ≈ softmax(q * K^T) * V
    //
    //   X_ij = softmax(q_i * C_k_j + beta_j) (compacted attention weights)
    //   Y_i  = softmax(q_i * K^T) * V         (original attention output)
    //   Solve: X * C_v = Y

    // Compute X: attention weights with compacted keys + bias
    std::vector<float> X(n_q * t);
    for (int i = 0; i < n_q; i++) {
        for (int j = 0; j < t; j++) {
            X[i * t + j] = scores[i * T + selected[j]] * inv_sqrt_dk + result.beta[j];
            // Wait, scores was already scaled. Let me recompute properly.
            // Actually scores[i*T + selected[j]] is already q*K/sqrt(d)
            // But we already scaled scores by inv_sqrt_dk above, so:
            X[i * t + j] = scores[i * T + selected[j]] + result.beta[j];
        }
    }
    softmax_rows(X.data(), n_q, t);

    // Compute Y: original attention output = attn_weights @ V  [n_q, d_v]
    std::vector<float> Y(n_q * d_v, 0.0f);
    for (int i = 0; i < n_q; i++) {
        for (int j = 0; j < T; j++) {
            float w_ij = attn_weights[i * T + j];
            for (int d = 0; d < d_v; d++) {
                Y[i * d_v + d] += w_ij * V[j * d_v + d];
            }
        }
    }

    // Solve: X * C_v = Y  =>  C_v = (X^T X)^{-1} X^T Y
    least_squares_solve(X.data(), Y.data(), result.C_v.data(), n_q, t, d_v);

    return result;
}

// Compact all KV heads within a single layer using shared key selection
//
//   K_all:     [T, n_embd_k_gqa] all heads concatenated, row-major
//   V_all:     [T, n_embd_v_gqa] all heads concatenated, row-major
//   Q_ref_all: [n_q, n_embd_k_gqa] reference queries (all heads concatenated)
//   T:         number of tokens (cache positions)
//   n_q:       number of reference queries
//   n_head_kv: number of KV heads
//   d_k:       key dimension per head
//   d_v:       value dimension per head
//   t:         target compacted size
//
// Algorithm:
//   1. For each head, compute attention scores and per-key importance
//   2. Global key selection: max importance across heads for each position
//   3. Per-head NNLS (beta) and least-squares (C_v) on shared selection
//
static compacted_layer compact_layer_all_heads(
        const float * K_all, const float * V_all, const float * Q_ref_all,
        int T, int n_q, int n_head_kv, int d_k, int d_v, int t) {

    compacted_layer result;
    result.n_head_kv = n_head_kv;
    result.t = t;
    result.d_k = d_k;
    result.d_v = d_v;
    result.beta.resize(n_head_kv);
    result.C_v.resize(n_head_kv);

    const int n_embd_k_gqa = n_head_kv * d_k;
    const int n_embd_v_gqa = n_head_kv * d_v;

    if (t >= T) {
        // No compaction needed
        result.selected_indices.resize(T);
        for (int i = 0; i < T; i++) result.selected_indices[i] = i;
        for (int h = 0; h < n_head_kv; h++) {
            result.beta[h].assign(T, 0.0f);
            result.C_v[h].resize(T * d_v);
            for (int i = 0; i < T; i++) {
                memcpy(result.C_v[h].data() + i * d_v,
                       V_all + i * n_embd_v_gqa + h * d_v,
                       d_v * sizeof(float));
            }
        }
        return result;
    }

    // ---- Step 1: Global key selection via max importance across heads ----

    // Compute per-head key importance scores, then take max across heads
    std::vector<float> global_scores(T, 0.0f);

    // Per-head temporary data for reuse in steps 2-3
    struct head_data {
        std::vector<float> scores;      // [n_q, T] scaled attention logits
        std::vector<float> exp_scores;  // [n_q, T] exp with max-shift
        std::vector<float> row_sums;    // [n_q] sum of exp per query
        std::vector<float> attn_weights;// [n_q, T] softmax attention
    };
    std::vector<head_data> hdata(n_head_kv);

    const float inv_sqrt_dk = 1.0f / sqrtf((float) d_k);

    for (int h = 0; h < n_head_kv; h++) {
        auto & hd = hdata[h];
        hd.scores.resize(n_q * T);
        hd.exp_scores.resize(n_q * T);
        hd.row_sums.resize(n_q);
        hd.attn_weights.resize(n_q * T);

        // Extract per-head K and Q_ref slices
        // K_head[i] = K_all[i * n_embd_k_gqa + h * d_k ... + (h+1)*d_k]
        // Instead of extracting, compute Q_ref_h @ K_h^T directly

        // Compute scores: Q_ref_h @ K_h^T / sqrt(d_k)
        for (int qi = 0; qi < n_q; qi++) {
            const float * q_row = Q_ref_all + qi * n_embd_k_gqa + h * d_k;
            for (int ki = 0; ki < T; ki++) {
                const float * k_row = K_all + ki * n_embd_k_gqa + h * d_k;
                float dot = 0.0f;
                for (int d = 0; d < d_k; d++) {
                    dot += q_row[d] * k_row[d];
                }
                hd.scores[qi * T + ki] = dot * inv_sqrt_dk;
            }
        }

        // Compute exp(scores) for mass computation
        memcpy(hd.exp_scores.data(), hd.scores.data(), n_q * T * sizeof(float));
        exp_rows_stable(hd.exp_scores.data(), hd.row_sums.data(), n_q, T);

        // Compute softmax for key scoring
        memcpy(hd.attn_weights.data(), hd.scores.data(), n_q * T * sizeof(float));
        softmax_rows(hd.attn_weights.data(), n_q, T);

        // Per-key max attention weight across queries
        for (int j = 0; j < T; j++) {
            float max_w = 0.0f;
            for (int qi = 0; qi < n_q; qi++) {
                float w = hd.attn_weights[qi * T + j];
                if (w > max_w) max_w = w;
            }
            // Global score = max across heads
            if (max_w > global_scores[j]) {
                global_scores[j] = max_w;
            }
        }
    }

    // Select top-t positions globally
    std::vector<int> indices(T);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + t, indices.end(),
                      [&](int a, int b) { return global_scores[a] > global_scores[b]; });

    std::vector<int> selected(indices.begin(), indices.begin() + t);
    std::sort(selected.begin(), selected.end());
    result.selected_indices = selected;

    // ---- Steps 2-3: Per-head NNLS (beta) and least squares (C_v) ----

    for (int h = 0; h < n_head_kv; h++) {
        const auto & hd = hdata[h];

        result.beta[h].resize(t);
        result.C_v[h].resize(t * d_v);

        // Step 2: NNLS for beta
        // M_ij = exp(q_i * K_{selected[j]} / sqrt(d)) (from precomputed exp_scores)
        std::vector<float> M(n_q * t);
        for (int qi = 0; qi < n_q; qi++) {
            for (int j = 0; j < t; j++) {
                M[qi * t + j] = hd.exp_scores[qi * T + selected[j]];
            }
        }

        std::vector<float> w(t);
        nnls_solve(M.data(), hd.row_sums.data(), w.data(), n_q, t);

        for (int j = 0; j < t; j++) {
            result.beta[h][j] = logf(std::max(1e-12f, w[j]));
        }

        // Step 3: Least squares for C_v
        // X_ij = softmax(score[qi, selected[j]] + beta[j])
        std::vector<float> X(n_q * t);
        for (int qi = 0; qi < n_q; qi++) {
            for (int j = 0; j < t; j++) {
                X[qi * t + j] = hd.scores[qi * T + selected[j]] + result.beta[h][j];
            }
        }
        softmax_rows(X.data(), n_q, t);

        // Y = original attention output: attn_weights @ V_head  [n_q, d_v]
        std::vector<float> Y(n_q * d_v, 0.0f);
        for (int qi = 0; qi < n_q; qi++) {
            for (int ki = 0; ki < T; ki++) {
                float w_ij = hd.attn_weights[qi * T + ki];
                const float * v_row = V_all + ki * n_embd_v_gqa + h * d_v;
                for (int d = 0; d < d_v; d++) {
                    Y[qi * d_v + d] += w_ij * v_row[d];
                }
            }
        }

        least_squares_solve(X.data(), Y.data(), result.C_v[h].data(), n_q, t, d_v);
    }

    return result;
}
