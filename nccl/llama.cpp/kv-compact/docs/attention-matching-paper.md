# Fast KV Compaction via Attention Matching — Paper Breakdown

**Paper:** [Fast KV Compaction via Attention Matching](https://arxiv.org/abs/2602.16284) (arXiv:2602.16284)
**Authors:** Adam Zweiger, Xinghong Fu, Han Guo, Yoon Kim — MIT
**Published:** February 18, 2026
**Code:** https://github.com/adamzweiger/compaction

---

## 1. The Problem: KV Cache as a Memory Bottleneck

Transformer-based LLMs generate tokens one at a time. To avoid recomputing attention over the entire history at every step, the model stores **key** and **value** vectors for all previously-processed tokens. This is the **KV cache**.

- KV cache size grows **linearly** with context length.
- For long sequences (60k+ tokens), it can consume **many gigabytes per request**.
- This limits batch size, maximum context length, and overall throughput.

### Prior workarounds and their limitations

| Approach | How it works | Downside |
|----------|-------------|----------|
| **Summarization** | Condense context into fewer tokens in text space | Highly lossy — on dense data (medical records), accuracy drops to no-context baseline |
| **Token eviction** (H2O, SnapKV, PyramidKV) | Drop low-attention tokens from the cache | Degrades significantly at high compression ratios (20x+) |
| **KVzip** | Query-agnostic compression via context reconstruction | Better than naive eviction, but still token-space selection |
| **Cartridges** (Eyuboglu et al., 2025) | Train compact KV caches via end-to-end gradient optimization in latent space | Achieves high quality but costs **hours of GPU time per context** |

---

## 2. The Core Idea: Attention Matching

Attention Matching operates in **latent space** rather than token space. Instead of selecting which tokens to keep, it constructs compact key and value vectors that **reproduce the original attention behavior**.

### Notation

- Original keys and values: **K**, **V** ∈ ℝ^(T×d) where T = sequence length, d = head dimension
- Compacted keys and values: **C_k**, **C_v** ∈ ℝ^(t×d) where t ≪ T
- Per-token bias: **β** ∈ ℝ^t (a critical addition — explained below)

### What the method preserves

For any future query vector **q**, the method ensures two properties hold:

**Property 1 — Attention Output Matching:**

The information extracted from the compacted cache should match what the full cache would produce:

```
softmax(qK^T)V  ≈  softmax(qC_k^T + β)C_v
```

**Property 2 — Mass Preservation:**

The total unnormalized attention mass should be preserved:

```
Σ_j exp(qK_j^T)  ≈  Σ_j exp(qC_k_j^T + β_j)
```

### Why mass preservation matters

When the model later appends new tokens to the cache (i.e., during generation), attention is computed over the concatenation `[compacted_cache ; new_tokens]`. The attention output decomposes as a **weighted mixture**, where the weights are determined by relative mass:

```
Attn(q; [C_k; K_new], [C_v; V_new]) =
    w_old × Attn(q; C_k, C_v) + w_new × Attn(q; K_new, V_new)
```

where `w_old = Mass(C_k) / (Mass(C_k) + Mass(K_new))`.

If mass is wrong, the balance between old context and new tokens is distorted — even if the compacted attention output itself is perfect.

### Why the bias term β is necessary

Without biases, mass matching is impossible when t < T. Consider q = 0:
- Original mass = T (sum of T ones)
- Compacted mass without bias = t (sum of t ones)

These can never be equal. The bias `exp(β_j)` lets each compacted key represent the aggregate mass of multiple original keys.

The memory overhead of storing β is negligible: a factor of (2d+1)/(2d) additional storage.

---

## 3. The Three-Step Decomposition

The joint optimization of (C_k, β, C_v) **decomposes** into three sequential subproblems, some with closed-form solutions. This is the key insight that makes the method fast.

### Step 1: Key Selection

Choose which t keys from the original K to retain as C_k. Two approaches:

#### Approach A — Highest Attention Keys (fast, ~3 seconds for 60k tokens)

1. Compute attention weights between reference queries and all keys.
2. Score each key via RMS aggregation across queries.
3. Select the top-t keys by score.

#### Approach B — Orthogonal Matching Pursuit / OMP (higher quality, ~104–565 seconds)

A greedy algorithm that selects keys to best reconstruct the original attention mass:

```
Input: Keys K, Reference Queries Q, budget t

Compute mass feature matrix:  Φ_ij = exp(q_i · K_j^T / √d)
Compute target mass vector:   m_i  = Σ_j Φ_ij

Initialize: residual r ← m, selected set S ← ∅

For k = 1 to t:
    j* ← argmax_{j∉S} (r^T · Φ_{:,j})      // greedily pick best key
    S  ← S ∪ {j*}
    w  ← argmin_{w≥0} ||Φ_{:,S} · w - m||²  // refit via NNLS
    r  ← m - Φ_{:,S} · w                     // update residual

Return: selected indices S, weights w, biases β = log(w)
```

**Speedup variant (OMP-fast):** Select k=4 keys per iteration, refit NNLS only every τ=2 iterations → 4–8x speedup with little quality loss.

### Step 2: Bias Fitting via Non-Negative Least Squares (~2.2 seconds)

Given fixed C_k (from Step 1), solve for optimal biases:

```
min_{w ≥ 0}  ||A · w - m||²

where  A_ij = exp(q_i · C_k_j^T / √d)
       m_i  = Σ_k exp(q_i · K_k^T / √d)   (original mass)
       β_j  = log(w_j)
```

Intuition: w_j represents how many original keys the j-th compacted key "covers" in terms of attention mass.

### Step 3: Value Fitting via Least Squares (~1.8 seconds)

Given C_k and β, solve for optimal C_v in closed form:

```
min_{C_v}  ||X · C_v - Y||²

where  x_i = softmax(q_i · C_k^T + β)    (compacted attention weights)
       y_i = softmax(q_i · K^T) · V       (original attention output)

Solution: C_v* = (X^T X)^{-1} X^T Y
```

This is ordinary least squares — a single matrix inversion.

---

## 4. Reference Query Generation

The method requires reference queries Q_ref to optimize against. Three strategies:

| Method | Description | Time (60k tokens) |
|--------|-------------|-------------------|
| **Repeat-Prefill** | Feed `{context} Repeat the previous context. {context}` and extract queries from the second pass | ~8s |
| **Self-Study** | Generate synthetic Q&A interactions about the context using 4 fixed prompts | ~139s |
| **On-Policy** | Compact layers sequentially; extract Q_ref at layer ℓ with layers <ℓ already compacted | Slight overhead |

Repeat-prefill alone nearly matches the best performance while being much faster. Random vectors perform significantly worse — the queries need semantic structure.

---

## 5. Nonuniform Compaction Across Heads

Not all attention heads are equally sensitive to compression.

### Key finding: head sensitivity is input-invariant

Although absolute loss varies by input, the **relative ranking** of which heads are most sensitive to compaction remains stable across different contexts and datasets.

### Consequence: precomputed per-head budgets

A one-time greedy exchange algorithm determines how to allocate the total budget across heads:

1. Start with uniform allocation across all heads.
2. Iteratively swap budget units from insensitive heads to sensitive heads.
3. Use single-head sensitivity curves (measured once) to predict impact.
4. Stop when no swap improves predicted loss.

This is computed **once per model** and reused for all contexts and compression ratios.

---

## 6. Experimental Results

### Benchmarks

| Dataset | Domain | Context Length | Density |
|---------|--------|---------------|---------|
| **QuALITY** | Long-document reading comprehension | 5–8k tokens | Moderate |
| **LongHealth** | Medical patient records QA | 60k tokens | Very high |
| **AIME** | Mathematical reasoning | Variable | N/A (tests online compaction) |

### Models tested

Qwen3-4B, Llama3.1-8B, Gemma3-12B

### Key results

**At 50x compression on QuALITY:**
- AM-OMP **matches Cartridges** performance while being **2 orders of magnitude faster** (seconds vs. hours).
- AM variants form the **Pareto frontier** of the speed-quality tradeoff.
- All token-selection baselines (H2O, SnapKV, KVzip) degrade significantly.

**On LongHealth (dense medical data):**
- Standard summarization **collapses to no-context baseline** (55.2% vs. 71.5% full context).
- AM-OMP significantly outperforms summarization, though requires lower compression ratios than on QuALITY.
- Token-pruning baselines also degrade badly.

**Ultra-high compression (Summarize + AM combined):**
- Summarize first (20x reduction), then apply AM-OMP at 0.2x → **200x total compression**.
- Accuracy: 55.7% (comparable to summarization alone at 55.2%), showing AM preserves what summarization retains.

### Computational cost breakdown (60k tokens, Gemma-3-12B, single H200)

| Component | Time |
|-----------|------|
| Context prefill | 7s |
| Repeat-prefill (reference queries) | 8s |
| Highest-attention key selection | 3s |
| OMP key selection (unoptimized) | 565s |
| OMP-fast (k=4, refit interval=2) | 104s |
| Bias fitting (NNLS) | 2.2s |
| Value fitting (least squares) | 1.8s |

**Total for AM-HighestAttn + repeat-prefill: ~20 seconds**
**Total for AM-OMP-fast + repeat-prefill: ~115 seconds**

### Ablation: what matters most

Ranked by impact on quality:

1. **Nonuniform head budgets** — most critical component
2. **Bias (β) fitting** — removing degrades quality but still reasonable
3. **Value (C_v) refitting** — consistent improvement
4. **Self-study queries** — least critical; repeat-prefill alone is nearly as good
5. **On-policy queries** — slight but consistent improvements

---

## 7. Online Compaction for Reasoning

For long-horizon tasks (e.g., multi-step math reasoning), the KV cache can be compacted **during generation**:

1. Set a fixed memory budget.
2. When the cache fills up, pause generation.
3. Apply Attention Matching to compress the cache by 50%.
4. Resume generation.

**Result on AIME:** After hitting the memory wall and being compressed up to **6 consecutive times**, the model's math reasoning performance **matched the unlimited-memory baseline**.

This is particularly promising for agentic workflows where tool outputs consume significant context.

---

## 8. Technical Details

### Numerical stability

Mass computations use per-query max-shift: `s = max_j(ℓ_j)`, then evaluate `Σ exp(ℓ_j) = exp(s) · Σ exp(ℓ_j - s)`. OMP operates on these shifted features.

### Position encoding compatibility

- Compacted cache stores t KV entries physically but retains logical length T for position IDs.
- New tokens receive the same position IDs as under the uncompacted prefix.
- RoPE phase alignment ensures correctness when chunking.

### Sliding window attention (Gemma-3-12B)

Gemma-3 uses a 5:1 ratio of sliding-window to global-attention layers. Only global-attention layers are compacted. This requires only slightly more conservative compression ratios.

### Attention bias support

The bias term β is compatible with standard attention implementations (PyTorch SDPA, FlexAttention) without modification.

---

## 9. Limitations

1. **OMP is still slow** — minutes rather than seconds at 60k tokens (though orders of magnitude better than Cartridges).
2. **Extreme compression (100x+)** — Cartridges outperforms AM at these ratios, since gradient-based search explores a wider representation space.
3. **Key subset restriction** — current implementation restricts C_k to be a subset of original keys; direct optimization could yield better results.
4. **Reference query dependence** — quality depends on how well Q_ref represents future queries; random vectors perform poorly.

---

## 10. Scaling Note

KV cache size grows modestly across model scales: Qwen3-235B has a KV cache only ~30% larger than Qwen3-4B for the same sequence length (due to grouped-query attention). This suggests the method's computational efficiency should scale well to much larger models, including mixture-of-experts architectures.

---

## 11. Summary

Attention Matching is a **post-hoc, training-free** KV cache compaction method that:

- Achieves **50x compression** with minimal accuracy loss
- Runs in **seconds to minutes** (vs. hours for prior latent-space methods)
- Works by preserving **attention outputs** and **attention mass** through a decomposed optimization with closed-form solutions
- Supports **online compaction** during generation for reasoning tasks
- Uses **precomputed per-head budgets** that transfer across contexts

It bridges the gap between fast-but-lossy token eviction and slow-but-high-quality gradient optimization, forming the Pareto frontier of the speed-quality tradeoff for KV cache compression.
