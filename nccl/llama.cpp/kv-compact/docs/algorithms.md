# KV Cache Compaction — Algorithms & Techniques Reference

Technical reference for all algorithms, numerical methods, and data structures
used in the KV cache compaction feature. Based on "Fast KV Compaction via
Attention Matching" (Zweiger et al., 2026, [arXiv:2602.16284](https://arxiv.org/abs/2602.16284)).

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Algorithm Overview](#2-algorithm-overview)
3. [Step 1: Key Selection — Highest Attention Keys](#3-step-1-key-selection--highest-attention-keys)
4. [Step 2: Bias Fitting — Non-Negative Least Squares](#4-step-2-bias-fitting--non-negative-least-squares)
5. [Step 3: Value Fitting — Regularized Least Squares](#5-step-3-value-fitting--regularized-least-squares)
6. [Supporting Algorithms](#6-supporting-algorithms)
7. [Numerical Stability Techniques](#7-numerical-stability-techniques)
8. [Constants & Convergence Parameters](#8-constants--convergence-parameters)
9. [Data Structures & Memory Layout](#9-data-structures--memory-layout)
10. [Quality Metrics](#10-quality-metrics)
11. [Token Eviction Baseline](#11-token-eviction-baseline)
12. [Limitations & Future Work](#12-limitations--future-work)

---

## 1. Problem Statement

Given a KV cache of T tokens with keys K in R^(T x d_k) and values
V in R^(T x d_v), find a compacted representation of t < T entries that
preserves attention output quality for future queries.

**Objective — match the attention output for any query q:**

```
Original:   y = softmax(q K^T / sqrt(d_k)) V
Compacted:  y_hat = softmax(q C_k^T / sqrt(d_k) + beta) C_v
```

where C_k in R^(t x d_k) are selected keys, beta in R^t are attention biases,
and C_v in R^(t x d_v) are refitted values.

The algorithm minimizes ||y - y_hat||^2 across a set of n_q reference queries.

---

## 2. Algorithm Overview

Compaction is decomposed into three independent subproblems, solved
sequentially per attention head:

```
Input:  K [T, d_k], V [T, d_v], Q_ref [n_q, d_k], target size t
                          |
                  Step 1: Key Selection
                  Select top-t keys by max attention score
                          |
                  Step 2: Bias Fitting (NNLS)
                  Find beta so attention mass is preserved
                          |
                  Step 3: Value Fitting (Least Squares)
                  Find C_v so attention output is preserved
                          |
Output: selected_indices [t], beta [t], C_v [t, d_v]
```

Each step only depends on the output of prior steps, enabling clear
correctness testing of each component independently.

---

## 3. Step 1: Key Selection — Highest Attention Keys

**Goal:** Choose the t keys that contribute most to attention across reference
queries.

### 3.1 Scaled dot-product attention scores

```
scores = Q_ref @ K^T / sqrt(d_k)         [n_q x T]
```

Implementation: `mat_mul_ABt(Q_ref, K, scores, n_q, T, d_k)` followed by
element-wise multiplication by `1/sqrt(d_k)`.

### 3.2 Softmax attention weights

```
attn_weights[i, j] = exp(scores[i, j]) / sum_k exp(scores[i, k])
```

Uses max-shift for numerical stability (see Section 7).

### 3.3 Per-key importance scoring

Each key is scored by its maximum attention weight across all queries:

```
key_score[j] = max_i attn_weights[i, j]
```

This captures whether *any* reference query strongly attends to key j. A key
that is critical for even one query pattern will be retained.

### 3.4 Top-t selection

```
indices = argsort(key_scores, descending)[:t]
selected = sort(indices)     // re-sort for cache locality
```

Implementation uses `std::partial_sort` (O(T log t)) followed by `std::sort`
on the t selected indices.

### 3.5 Exponential scores and mass (precomputed for Step 2)

In parallel with softmax, the algorithm computes:

```
exp_scores[i, j] = exp(scores[i, j] - max_j scores[i, j])    [n_q x T]
row_sums[i] = sum_j exp_scores[i, j]                          [n_q]
```

These unnormalized exponentials and their row sums become the target for mass
matching in Step 2.

---

## 4. Step 2: Bias Fitting — Non-Negative Least Squares

**Goal:** Find bias vector beta such that the compacted keys preserve the
total attention mass of the original cache.

### 4.1 Why mass preservation matters

When generating new tokens after compaction, attention to old (compacted)
cache competes with attention to new tokens:

```
w_old = m_compacted / (m_compacted + m_new)
```

If m_compacted != m_original, the model over- or under-attends to the cached
context, degrading generation quality.

### 4.2 Mass matching formulation

The unnormalized attention mass for query i over the full cache is:

```
m_i = sum_j exp(q_i @ K_j^T / sqrt(d_k))
```

For the compacted cache with biases:

```
m_hat_i = sum_j exp(q_i @ C_k_j^T / sqrt(d_k) + beta_j)
        = sum_j exp(q_i @ C_k_j^T / sqrt(d_k)) * exp(beta_j)
        = sum_j exp(q_i @ C_k_j^T / sqrt(d_k)) * w_j
```

where w_j = exp(beta_j). Setting M[i,j] = exp(q_i @ C_k_j^T / sqrt(d_k)),
this becomes the linear system:

```
M @ w ≈ m      subject to w >= 0
```

### 4.3 Design matrix construction

```
M[i, j] = exp_scores[i, selected[j]]     [n_q x t]
```

These are the precomputed max-shifted exponentials from Step 1, subselected to
the chosen key positions.

### 4.4 NNLS solver: Projected Gradient Descent

```
Precompute:
    AtA = M^T @ M           [t x t]
    Atb = M^T @ m            [t]
    step = 1 / (trace(AtA) + 1e-8)

Initialize:
    w = [1, 1, ..., 1]       (uniform start)

For iter = 0 to max_iter-1:
    grad = AtA @ w - Atb     (gradient of ||Mw - m||^2)
    w = max(1e-12, w - step * grad)   (gradient step + projection)
```

**Convergence:** The step size 1/trace(AtA) is a conservative upper bound on
1/lambda_max(AtA), guaranteeing descent. 200 iterations typically suffice for
the small t values used in practice (t ~ 50-500).

### 4.5 Conversion to log-space bias

```
beta[j] = log(max(w[j], 1e-12))
```

The floor at 1e-12 prevents -inf. During attention computation:

```
logit[i, j] = q_i @ C_k_j^T / sqrt(d_k) + beta[j]
```

The additive bias in log-space is equivalent to the multiplicative weight w_j
in probability space.

---

## 5. Step 3: Value Fitting — Regularized Least Squares

**Goal:** Find C_v so that the compacted attention output matches the
original.

### 5.1 Compacted attention weights (X matrix)

```
X[i, j] = softmax_j(scores[i, selected[j]] + beta[j])     [n_q x t]
```

These are the attention weights a future query would produce when attending to
the compacted cache with biases applied.

### 5.2 Original attention output (Y matrix)

```
Y[i, d] = sum_j attn_weights[i, j] * V[j, d]              [n_q x d_v]
```

This is the ground truth — what the full cache would produce.

### 5.3 Least squares problem

```
min_{C_v} ||X @ C_v - Y||^2
```

Closed-form solution via normal equations:

```
C_v = (X^T X + ridge * I)^{-1} X^T Y
```

### 5.4 Solver: Gaussian elimination with partial pivoting

The normal equations are solved directly rather than via iterative methods.

```
Form augmented matrix:
    aug = [X^T X + ridge*I | X^T Y]        [t x (t + d_v)]

Forward elimination with partial pivoting:
    For each column col:
        Find row with max |value| in column (partial pivot)
        Swap rows if needed
        Skip if |pivot| < 1e-12 (rank deficiency)
        Eliminate all rows below pivot

Back substitution:
    Solve from bottom row upward
    Set zero for rank-deficient columns
```

**Why direct solve:** The matrix X^T X is small (t x t, typically 50-500), so
O(t^3) direct solve is faster than iterative methods and gives exact results
up to floating-point precision.

### 5.5 Ridge regularization (Tikhonov)

Adding ridge * I to X^T X:
- Ensures invertibility even when X is rank-deficient
- Shrinks C_v toward zero, preventing extreme values
- Default lambda = 1e-6 — small enough to not distort the solution
  significantly, large enough to stabilize ill-conditioned systems

---

## 6. Supporting Algorithms

### 6.1 Matrix multiply A * B^T

```
C[i, j] = sum_l A[i, l] * B[j, l]
```

Used for: Q @ K^T attention score computation. B is stored in row-major order,
so B^T access pattern reads B row-by-row (cache-friendly).

Source: `mat_mul_ABt()` in kv-compact-math.h

### 6.2 Matrix multiply A^T * B

```
C[j, l] = sum_i A[i, j] * B[i, l]
```

Used for: Gram matrix computation (A^T A), cross terms (A^T b).

Source: `mat_mul_AtB()` in kv-compact-math.h

### 6.3 Row-wise softmax

```
softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
```

Applied independently to each row. In-place operation.

Source: `softmax_rows()` in kv-compact-math.h

### 6.4 Stable row-wise exponentiation

```
exp_stable(x_i) = exp(x_i - max(x))
row_sum = sum_i exp_stable(x_i)
```

Returns both the shifted exponentials and per-row sums. Shares the max-shift
stability technique with softmax but does not normalize.

Source: `exp_rows_stable()` in kv-compact-math.h

---

## 7. Numerical Stability Techniques

### 7.1 Max-shift in softmax and exp

**Problem:** exp(x) overflows for x > ~88 in float32.

**Solution:** Subtract the row maximum before exponentiating:

```
exp(x_i - max_j x_j) <= exp(0) = 1
```

This is mathematically equivalent (cancels in softmax normalization) and keeps
all intermediate values in [0, 1].

### 7.2 Division-by-zero guards

All divisions include epsilon guards:

| Location | Guard | Value |
|----------|-------|-------|
| Softmax normalization | 1/(sum + eps) | 1e-12 |
| NNLS step size | 1/(trace + eps) | 1e-8 |
| Cosine similarity | dot/(norm + eps) | 1e-8 |
| Beta computation | log(max(w, eps)) | 1e-12 |

### 7.3 Partial pivoting in Gaussian elimination

Selects the row with the largest absolute value as pivot at each elimination
step. This minimizes the growth factor in the elimination, reducing
accumulated rounding error.

### 7.4 Ridge regularization

Adds lambda * I to the Gram matrix X^T X, ensuring its condition number is
bounded by (lambda_max + lambda) / lambda. With lambda = 1e-6, this prevents
numerical instability when reference queries are nearly collinear.

### 7.5 Non-negativity floor in NNLS

The projection step uses max(1e-12, ...) instead of max(0, ...) to prevent
exact zeros. This is critical because:
- beta = log(w) would produce -inf for w = 0
- Zero weights make the corresponding key invisible, wasting a budget slot

---

## 8. Constants & Convergence Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| NNLS max iterations | 200 | Conservative; convergence typically by ~50 |
| NNLS initial weights | 1.0 (uniform) | Neutral starting point; avoids zero bias |
| NNLS step size | 1/trace(A^T A) | Upper bound on 1/lambda_max; guarantees descent |
| NNLS projection floor | 1e-12 | Prevents log(0) downstream |
| LS ridge (lambda) | 1e-6 | Stabilizes without distortion |
| LS pivot threshold | 1e-12 | Detects rank deficiency |
| Softmax epsilon | 1e-12 | Prevents division by zero |
| Score scaling | 1/sqrt(d_k) | Standard scaled dot-product attention |
| Attention sink count | 4 | First few tokens (eviction baseline only) |

---

## 9. Data Structures & Memory Layout

### 9.1 Compaction result

```cpp
struct compacted_head {
    std::vector<int>   selected_indices;  // [t] ascending token positions
    std::vector<float> beta;              // [t] attention biases
    std::vector<float> C_v;              // [t * d_v] refitted values (row-major)
};
```

### 9.2 KV cache tensor layout

**K tensor:** `[n_embd_k_gqa, kv_size]`
- Each row = one token, all heads concatenated
- Head h, token i, dimension d: `K[i * n_embd_k_gqa + h * d_k + d]`
- Row stride: `ggml_row_size(type, n_embd_k_gqa)` bytes

**V tensor (non-transposed):** Same layout as K with `n_embd_v_gqa`.

**V tensor (transposed, `v_trans=true`):** `[d_v, n_head_kv, kv_size]`
- Head h, token i, dimension d: byte offset = `((h * d_v + d) * kv_size + i) * elem_size`
- Used in some backend configurations for memory access efficiency

### 9.3 Serialized state format

```
Header:
    n_stream                           (u32)

Per stream:
    cell_count                         (u32)
    cells[cell_count]:
        pos                            (i32)
        n_seq_id                       (u32)
        seq_ids[n_seq_id]              (i32 each)
    v_trans                            (u32, 0 or 1)
    n_layer                            (u32)

    K data (per layer):
        k_type                         (i32, ggml_type enum)
        k_size_row                     (u64, bytes per row)
        k_data[k_size_row * cell_count] (raw bytes)

    V data (per layer):
        v_type                         (i32)
        v_size_row                     (u64)
        v_data[v_size_row * cell_count]
```

### 9.4 Type conversion

| KV type | Read | Write |
|---------|------|-------|
| F32 | Direct float* cast | Direct write |
| F16 | `ggml_fp16_to_fp32()` | `ggml_fp32_to_fp16()` |
| Quantized | Skipped in POC | Skipped in POC |

---

## 10. Quality Metrics

### 10.1 Mean Squared Error (MSE)

```
MSE = (1/d_v) sum_d (y_hat[d] - y[d])^2
```

Measures per-dimension average squared deviation of the compacted output from
the original.

### 10.2 Relative L2 Error

```
rel_err = sqrt(MSE) / (sqrt(sum_d y[d]^2 / d_v) + 1e-8)
```

Normalizes error by the magnitude of the original output. Scale-invariant.

### 10.3 Cosine Similarity

```
cos_sim = (y_hat . y) / (||y_hat|| * ||y|| + 1e-8)
```

Measures directional agreement. cos_sim = 1.0 means the output vectors point
in exactly the same direction (perfect preservation of relative values).

### 10.4 Observed quality (from unit tests)

| Compression | Metric | Value |
|-------------|--------|-------|
| 4x (T=32, t=8) | MSE improvement (refit vs eviction) | ~4,000,000x |
| 2x (T=24, t=12) | Cosine similarity | 0.999999 |

---

## 11. Token Eviction Baseline

The POC includes a simple eviction heuristic for quality comparison:

```
Budget: t tokens total

Allocation:
    n_sink   = 4              (attention sink — first few tokens)
    n_recent = t / 2          (most recent tokens)
    n_middle = t - n_sink - n_recent

Selection:
    Keep tokens [0, n_sink)                     (attention sinks)
    Keep tokens [T - n_recent, T)               (recent context)
    Uniformly sample n_middle from [n_sink, T - n_recent)

Eviction:
    Remove all non-selected positions via llama_memory_seq_rm()
```

**Attention sinks:** The first few tokens in transformer models accumulate
disproportionate attention regardless of content (Xiao et al., 2023). Evicting
them causes quality collapse.

This baseline does not use beta or C_v — it keeps original K and V values for
selected tokens. The Attention Matching algorithm should strictly dominate
this approach on output quality.

---

## 12. Limitations & Future Work

### Current implementation limitations

1. **No beta injection during generation:** The POC computes beta but cannot
   apply it during `llama_decode`. Requires attention graph modification or a
   bias hook in `ggml_flash_attn_ext`.

2. **C_v not written back:** Compacted values are computed but not written to
   the KV cache tensors. Requires `llama_state_seq_set_data` or direct tensor
   modification.

3. **Single-head demonstration:** The POC runs compaction on layer 0, head 0
   only. Production requires iterating all layers and heads.

4. **Simplified reference queries:** Uses K vectors from the last positions as
   query proxies instead of true repeat-prefill (running the context through
   the model twice).

5. **Quantized KV types skipped:** Only F32 and F16 are supported. Quantized
   types need dequantize-compact-requantize.

### Algorithmic extensions from the paper

1. **OMP key selection:** Orthogonal Matching Pursuit selects keys iteratively
   by greedy residual reduction. Higher quality than Highest Attention but
   ~100x slower (minutes vs seconds).

2. **Non-uniform per-head budgets:** Sensitivity varies across heads.
   Pre-computed sensitivity curves allow allocating more budget to critical
   heads and less to redundant ones.

3. **Iterative compaction:** The algorithm can be applied multiple times as
   context grows. Paper demonstrates 6 consecutive compressions on AIME
   without quality loss.

4. **Direct key optimization:** Removing the constraint that C_k must be a
   subset of original keys could yield better results but makes the
   optimization non-convex.
