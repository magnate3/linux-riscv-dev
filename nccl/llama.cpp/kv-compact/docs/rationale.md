# KV Cache Compaction — Researcher Rationale & Improvement Opportunities

Why the researchers made each design choice, what trade-offs they accepted,
and where the real leverage is for doing better.

---

## 1. The Fundamental Insight: Work in Latent Space, Not Token Space

**Prior art (H2O, SnapKV, PyramidKV, KVzip):** All operate in *token space* —
they decide which tokens to keep or drop. This has a hard ceiling: at 50x
compression, you're keeping 2% of tokens. No matter how smart your selection
is, you're throwing away 98% of the information.

**The researchers' realization:** The KV cache doesn't store "tokens" — it
stores *vectors in a continuous latent space*. There's no reason a compacted
key or value vector must correspond to any single original token. A single
compacted value vector can encode a *weighted combination* of many original
values.

**This is why the method works at 50x compression where token eviction
collapses.** Token eviction is a special case of attention matching where
beta = 0 and C_v = V_selected. The optimization has strictly more degrees of
freedom.

**Improvement angle:** The current implementation still restricts C_k (keys)
to be a subset of originals. Lifting this constraint — optimizing keys in
continuous space — would give even more freedom. The researchers explicitly
note this as future work but avoided it because the optimization becomes
non-convex.

---

## 2. The Decomposition: Why Three Steps Instead of Joint Optimization

The "obvious" approach is to jointly optimize (C_k, beta, C_v) to minimize
attention output error. The researchers rejected this because:

1. **Joint optimization is non-convex.** The attention output depends on
   softmax(q @ C_k^T + beta) @ C_v — the product of C_k inside a softmax
   multiplied by C_v creates a non-convex landscape with many local minima.

2. **Cartridges (Eyuboglu 2025) tried the joint approach.** It works — but
   requires hours of GPU-based gradient optimization per context. The
   researchers wanted seconds, not hours.

3. **The decomposition makes each step convex (or combinatorial + convex).**
   - Step 1 (key selection): combinatorial, but greedy/top-k is effective
   - Step 2 (bias fitting): convex quadratic program with linear constraints
   - Step 3 (value fitting): ordinary least squares — closed-form solution

**The trade-off they accepted:** Solving three convex subproblems sequentially
is suboptimal compared to the joint optimum. Errors in Step 1 (bad key
selection) propagate to Steps 2 and 3 with no way to recover. But the speedup
is 2-3 orders of magnitude.

**Improvement angle:** Any method that couples the steps more tightly without
losing convexity would help. For example:
- Iterate: run Steps 1-2-3, then re-score keys using the refitted values,
  re-select, and repeat. The paper doesn't explore this.
- Use Step 2/3 residuals to guide key selection refinement (swap out poor keys
  for better candidates).

---

## 3. Key Selection: Why "Max Attention" and Not Something Smarter

The Highest Attention method scores each key by:

```
key_score[j] = max_i softmax(q_i @ K^T)[i, j]
```

**Why max, not sum or mean?** The researchers' reasoning:

- A key that is *critical for even one query* must be retained. Sum/mean would
  dilute this signal — a key could score low on average but be the *only* key
  that answers a specific question.
- Max is a worst-case metric: "what's the most damage we'd do by dropping
  this key?" This is more robust than average-case reasoning for downstream
  task quality.

**The paper notes this is actually described as "RMS aggregation"** across
queries, but the POC implements max. The difference matters when n_q is large.

**Why the researchers also developed OMP:** Highest Attention is greedy and
independent — it doesn't consider *redundancy* between selected keys. If two
keys are nearly identical, Highest Attention might select both, wasting a
budget slot. OMP avoids this by iteratively selecting keys that reduce the
*residual* — each new key must contribute something the already-selected keys
don't.

**The cost:** OMP is 35-190x slower (104-565s vs 3s at 60k tokens).

**Improvement angle:**
- Diversity-aware fast selection: cluster keys, select top-1 per cluster.
  This captures the "avoid redundancy" benefit of OMP at near-Highest-Attention
  speed.
- Importance + coverage: score = max_attention * (1 - max_similarity_to_already_selected).
  A simple greedy diversification that's O(T * t) instead of OMP's O(T * t^2).

---

## 4. Why the Bias Term beta Is the Key Innovation

**The insight that makes everything work:**

Without beta, mass matching is *provably impossible* when t < T. Consider the
trivial case q = 0:
- Original mass = sum of T ones = T
- Compacted mass = sum of t ones = t
- These can never be equal.

This isn't just a theoretical edge case. In practice, some queries attend
broadly across many tokens. Without beta, the compacted cache systematically
*underestimates* the total attention mass, causing the model to over-attend
to new tokens and under-attend to cached context.

**beta_j = log(w_j) where w_j >= 1 means "this key covers multiple original
keys."** It's an attention multiplier in log-space. A key with w_j = 5
effectively represents 5 original keys' worth of attention mass.

**The memory overhead is negligible:** storing beta costs 1 float per
compacted token, vs 2*d floats for K+V. For d=128, that's 0.4% overhead.

**Improvement angle:**
- The current NNLS solver (projected gradient descent, 200 iterations) is a
  generic solver. For this specific problem structure (mass matching with
  exponential features), specialized solvers may converge faster or find
  better solutions.
- The paper uses a *single* beta per key. A query-dependent bias
  beta(q) would be more expressive but requires runtime computation.

---

## 5. Why NNLS and Not Unconstrained Least Squares for Bias

The mass matching problem is:

```
M @ w ≈ m,  where M_ij = exp(q_i @ C_k_j / sqrt(d)),  w >= 0
```

**Why the non-negativity constraint matters:**

w_j = exp(beta_j) represents attention mass contribution. Negative w would
mean a key *absorbs* mass — physically meaningless and numerically dangerous.
In the softmax, a large negative beta would make the key invisible, wasting a
budget slot. A very large negative w could create cancellation errors where
large positive and negative terms nearly cancel.

**The researchers chose projected gradient descent over active-set NNLS
(Lawson-Hanson) because:**

1. The problem is small (n = t, typically 50-500) — even a naive solver runs
   in milliseconds.
2. Projected gradient is trivially parallel (each w_i updates independently).
3. The step size heuristic (1/trace) is conservative but safe — no line
   search needed.

**Improvement angle:**
- Active-set NNLS (Lawson-Hanson) gives exact solutions in finite steps and
  is well-suited for small dense problems. Would likely converge in fewer
  iterations.
- The 200-iteration budget may be excessive or insufficient depending on
  problem conditioning. Adaptive convergence criteria (gradient norm < eps)
  would be more principled.

---

## 6. Value Fitting: Why Least Squares Is (Almost) Optimal

Step 3 solves:

```
min_{C_v} ||X @ C_v - Y||^2
```

where X = softmax(q @ C_k^T + beta) and Y = softmax(q @ K^T) @ V.

**Why this is the right formulation:**

The attention output for a query q is sum_j attn_weight_j * v_j. The
researchers observed that once keys and biases are fixed (Steps 1-2), the
attention weights X are *fixed linear coefficients*. Finding optimal C_v is
then a linear regression problem — the textbook case where least squares
gives the globally optimal solution.

**The closed form C_v = (X^T X)^{-1} X^T Y is exact up to floating point.**
No iterative optimization, no hyperparameters (besides the ridge), no
convergence concerns.

**Why ridge regularization (lambda = 1e-6):**

When reference queries are near-collinear (attend to similar keys with similar
weights), X^T X becomes ill-conditioned. Ridge pulls the solution toward zero,
which is a safe default — an overfitted C_v that's wildly large would produce
correct outputs for reference queries but terrible outputs for new queries.

**Improvement angle:**
- The ridge lambda = 1e-6 is fixed. Cross-validation or GCV (generalized
  cross-validation) could select an optimal lambda per head.
- The formulation assumes fixed X. But X depends on beta, and beta was fit in
  Step 2 without knowledge of C_v. Alternating optimization (fit C_v, update
  beta given C_v, repeat) could improve the joint solution.

---

## 7. Reference Queries: The Hidden Critical Dependency

**The entire method optimizes against Q_ref.** If Q_ref doesn't represent
future queries well, the compacted cache will be tuned for the wrong use
case.

**The researchers' key finding:** Random query vectors perform poorly.
Queries need semantic structure — they need to "ask questions" that the
cached context can answer.

**Why repeat-prefill works:** Feeding the context twice (with a "repeat"
prompt) generates queries that systematically probe every part of the cached
content. The model's own attention patterns during the second pass reveal
which parts of the context are semantically important.

**The researchers accepted a major limitation here:** repeat-prefill costs a
full forward pass (~8s for 60k tokens). This doubles the prefill cost.

**Improvement angle — this is likely the highest-leverage area:**
- Can you generate good reference queries *without* a second forward pass?
  Using the K vectors themselves as query proxies (as the POC does) is cheap
  but loses the semantic structure.
- Synthetic query generation: use the model to generate a few targeted
  questions about the context, then use those as Q_ref. The paper's
  "Self-Study" does this but at 139s — far too slow.
- Cached sensitivity: if head sensitivity is input-invariant (as the paper
  shows), perhaps *query quality requirements* are also somewhat invariant.
  A set of generic "probing" queries might work across contexts.

---

## 8. Per-Head Budget Allocation: The Most Impactful Component

**The ablation results are striking:** nonuniform head budgets matter more
than bias fitting, more than value refitting, more than query strategy.

**The researchers' insight:** Different attention heads serve different
functions. Some heads are highly compressible (e.g., broad attention heads
that attend uniformly), while others are critical (e.g., "retrieval" heads
that do precise lookups). Giving every head the same budget wastes slots on
easy heads and starves critical ones.

**Why this is input-invariant:** The function a head performs is determined by
its learned weights (W_Q, W_K, W_V), not by the input. A head that does
precise retrieval does it on all inputs. So the sensitivity measured on one
input transfers to others.

**The precomputation is done once per model:** measure single-head sensitivity
curves, then run a greedy exchange algorithm to find the optimal allocation.

**Improvement angle:**
- The current allocation is uniform across compression ratios (same
  proportions regardless of total budget). But head importance likely shifts
  at extreme compression — at 100x, different heads may become the
  bottleneck than at 10x.
- The greedy exchange is a local search. For small numbers of heads (32-128),
  global search via dynamic programming might find better allocations.
- Can we make allocation input-*aware* cheaply? A quick metric (entropy of
  attention distribution per head) could adjust allocations at runtime.

---

## 9. The Researchers' Implicit Assumptions

Several assumptions underpin the method. Questioning them may reveal
improvement paths:

### Assumption 1: The softmax structure must be preserved

The method preserves softmax(q @ C_k^T + beta) @ C_v. But maybe the right
abstraction is to directly approximate the *output* f(q) = sum_j alpha_j v_j
without constraining the intermediate form. This would allow non-softmax
mixing weights.

**Counter-argument:** The softmax structure must be preserved for the
compacted cache to work with the existing attention implementation without
modification.

### Assumption 2: Reference queries are representative

The method optimizes for Q_ref and hopes this generalizes. If the actual
future queries are very different from Q_ref (e.g., the user asks about a
minor detail the repeat-prefill didn't probe), quality degrades.

**This is analogous to the train/test distribution mismatch in ML.** The
paper partially addresses this with diverse query strategies but doesn't
bound the worst case.

### Assumption 3: Per-head independence

Each head is compacted independently. But heads interact — what one head
misses, another might compensate for. Joint multi-head compaction could
exploit this redundancy.

**The cost:** joint optimization over all heads simultaneously would be far
more expensive and lose the clean decomposition.

### Assumption 4: Linear value fitting is sufficient

Step 3 assumes C_v lives in a linear subspace spanned by the attention weight
rows. For high compression (t << T), this subspace may be too constrained.

---

## 10. Ranked Improvement Opportunities

Based on the researchers' own ablation data and the structural analysis above,
ranked by expected impact and feasibility:

### Tier 1 — High impact, clearly feasible

1. **Input-aware head budget allocation.** Use per-head attention entropy as a
   fast signal to adjust budgets at runtime. The infrastructure for nonuniform
   budgets already exists; this just makes the allocation dynamic.

2. **Iterative refinement (Steps 1-2-3 -> re-score -> repeat).** After value
   fitting, re-evaluate which keys are actually contributing. Swap out keys
   whose beta is near-zero (contributing nothing) for unchosen keys. 2-3
   iterations may significantly improve key selection quality at modest cost.

3. **Better NNLS solver.** Replace projected gradient descent with
   Lawson-Hanson active-set NNLS. Exact convergence, likely fewer iterations,
   and the problem size (t ~ 50-500) is well within the method's sweet spot.

### Tier 2 — High impact, requires more work

4. **Cheap reference query generation.** Find a way to generate semantically
   meaningful Q_ref without a second forward pass. Candidates: attention
   patterns from the original prefill (already computed), principal components
   of K, or a small learned "probing" matrix.

5. **Diversity-aware key selection.** Add a redundancy penalty to the
   Highest Attention scoring: down-weight keys that are similar to
   already-selected keys. Captures OMP's main benefit at Highest Attention's
   speed.

6. **Alternating optimization of beta and C_v.** After Step 3, update beta
   to account for the fitted C_v, then re-solve C_v. This couples the two
   fitting steps without joint non-convex optimization.

### Tier 3 — Potentially transformative, high research risk

7. **Continuous key optimization.** Remove the constraint C_k ⊆ {K_1,...,K_T}.
   Optimize C_k in continuous space via gradient descent on the attention
   matching objective. The researchers avoided this due to non-convexity, but
   with good initialization (start from selected keys) it might work.

8. **Multi-head joint compaction.** Exploit cross-head redundancy by solving a
   joint optimization where heads can "share" budget dynamically.

9. **Learned compaction.** Train a small network that takes (K, V) and
   produces (C_k, beta, C_v) directly. Amortizes the optimization cost across
   many contexts. Similar spirit to Cartridges but with an explicit
   attention-matching loss and lightweight architecture.

---

## 11. What the Timing Data Tells Us

From the paper's breakdown (60k tokens, Gemma-3-12B, H200):

| Component | Time | % of total |
|-----------|------|-----------|
| Context prefill | 7s | 35% |
| Repeat-prefill | 8s | 40% |
| Key selection (fast) | 3s | 15% |
| Bias + value fitting | 4s | 20% |

**The bottleneck is reference query generation (40% of total time).** Key
selection and fitting are already fast. This confirms that Tier 2 item #4
(cheap Q_ref generation) would have the largest speed impact.

**For OMP, key selection dominates** (104-565s vs 4s for fitting). This is
why the researchers developed OMP-fast (batch selection + sparse refitting) —
but it's still 50x slower than Highest Attention.

---

## 12. Summary: The Core Trade-off Chain

```
Joint optimization (Cartridges)
    |  sacrifice: speed (hours -> seconds)
    v
Decomposed 3-step pipeline
    |  sacrifice: optimality (sequential subproblems can't recover from step 1 errors)
    v
Highest Attention key selection
    |  sacrifice: key diversity (no redundancy awareness)
    v
Repeat-prefill for reference queries
    |  sacrifice: extra forward pass cost
    v
Fixed per-head budgets
    |  sacrifice: input-awareness
    v
Current implementation
```

Each arrow is a trade-off the researchers made for speed. **Every arrow is a
potential improvement point** — you can trade back some speed for quality at
any level, or find a way to get the quality without the speed cost.

The most promising: items 1, 2, and 5 from the ranked list, because they
improve quality with minimal speed cost by operating within the existing
pipeline structure.
