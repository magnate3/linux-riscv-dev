# KV Cache Compaction — Adjacent Concepts & Cross-Pollination Map

A discovery-driven exploration of techniques from adjacent fields that map
onto specific components of the KV cache compaction pipeline. Each concept
includes: what it is, the structural analogy, the concrete improvement it
suggests, and relevant papers.

---

## How to Read This Document

The compaction pipeline has four components. Each adjacent concept maps to
one or more:

```
[A] Key Selection    — choosing which t keys to keep from T
[B] Bias Fitting     — finding beta so attention mass is preserved
[C] Value Fitting    — finding C_v so attention output is preserved
[D] Budget Alloc     — distributing budget across heads/layers
```

---

## 1. CUR Decomposition & Interpolative Decomposition

**Maps to: [A] Key Selection + [C] Value Fitting**

### What it is

CUR decomposition factorizes a matrix A ≈ C U R where C is a subset of
columns of A, R is a subset of rows, and U is a small linking matrix. The
Interpolative Decomposition (ID) is the related factorization A ≈ C D where
D is an interpolation matrix containing the identity.

### The structural analogy

The attention output matrix is softmax(Q K^T / sqrt(d)) @ V. This is
*exactly* a matrix that CUR was designed to approximate:
- C = selected key columns (key selection)
- R = corresponding value rows (value selection)
- U = the linking/interpolation matrix (plays the role of beta + C_v fitting)

The compaction pipeline's three steps are literally a CUR decomposition of
the attention output matrix, with the added constraint that the "C" selection
must work through the softmax nonlinearity.

### What it suggests concretely

**Leverage score sampling** for key selection. Instead of max-attention
scoring, compute leverage scores of the attention matrix (proportional to
diagonal of the projection onto the top-k singular subspace). Leverage scores
capture *structural importance* — how much removing a column degrades the
low-rank approximation — rather than just *magnitude*.

### Direct application exists

**CurDKV** (Shi et al., NeurIPS 2025) already applies CUR decomposition to
KV cache compression. Key finding: *attention score approximation does not
guarantee output preservation*, but CUR-based selection minimizes end-to-end
reconstruction loss. Achieves up to 9.6% higher accuracy than SnapKV and
ChunkKV under aggressive compression.

### Surprising empirical finding

LevAttention (Kannan et al., 2024) computes leverage scores of the key
matrix in O(nd) time and selects top-scoring keys. However, a follow-up
(2025) found that **k-means clustering of keys outperforms leverage score
selection** (84.46% vs 77.17% accuracy on ViT-Large when selecting 128 of
197 keys). This suggests geometric structure matters more than algebraic
importance for pre-trained models — a direct argument for centroid-based
approaches (Section 16).

### Papers
- [CurDKV: Value-Guided KV Compression via CUR Decomposition](https://arxiv.org/abs/2509.15038) (NeurIPS 2025)
- [LevAttention: Leverage Scores for Attention](https://arxiv.org/abs/2410.05462) (2024)
- [Efficient Attention via Pre-Scoring](https://arxiv.org/abs/2505.11040) (2025)
- [Column and Row Subset Selection Using Nuclear Scores](https://arxiv.org/abs/2407.01698) (2024)
- Voronin & Martinsson, "Efficient algorithms for CUR and interpolative matrix decompositions" (2017)

---

## 2. Coresets

**Maps to: [A] Key Selection + [B] Bias Fitting**

### What it is

A coreset is a small weighted subset of a dataset that provably approximates
some function (loss, density, etc.) computed over the full dataset. The key
idea: each coreset point carries a *weight* representing how many original
points it "stands for."

### The structural analogy

The KV cache compaction problem IS coreset construction:
- "Dataset" = the T original KV pairs
- "Function to preserve" = the attention output for any query
- "Coreset" = the t selected KV pairs
- **"Coreset weights" = exp(beta)** — this is exactly the bias term!

The beta term in Attention Matching is a *coreset weight*. The connection is
not metaphorical — it's algebraically identical.

### What it suggests concretely

**Sensitivity-based sampling.** Coreset theory says the optimal sampling
probability for point j is proportional to its "sensitivity" — the maximum
influence it has on the objective across all possible queries:

```
s_j = sup_q  |contribution of key j to attention output for query q|
           / |total attention output for query q|
```

This is theoretically superior to max-attention scoring because it accounts
for the *relative* contribution, not just the absolute attention weight. The
total coreset size needed for (1+eps) approximation is O(s_total / eps^2)
where s_total = sum of all sensitivities.

**Importance: this gives a theoretical lower bound on t.** For a given error
tolerance eps, coreset theory tells you the minimum number of keys you need.
The current method has no such guarantee.

### Papers
- [Improved Coresets for Kernel Density Estimates](https://dl.acm.org/doi/abs/10.5555/3174304.3175477) (SODA 2018) — O(1/eps^2) coreset size, dimension-independent
- Feldman & Langberg, "A Unified Framework for Approximating and Clustering Data" (STOC 2011)
- [Efficient Coreset Constructions via Sensitivity Sampling](https://proceedings.mlr.press/v157/braverman21a.html) (ACML 2021)

---

## 3. Nyström Approximation

**Maps to: [A] Key Selection + [B] Bias Fitting**

### What it is

The Nyström method approximates a large kernel matrix K ≈ K_{nm} K_{mm}^{-1}
K_{mn} by selecting m "landmark" points and using the kernel evaluations
between all points and the landmarks.

### The structural analogy

The attention weight matrix softmax(Q K^T / sqrt(d)) is a kernel matrix
(specifically, a softmax kernel). The Nyström approximation selects a subset
of "landmark keys" and reconstructs the full attention matrix from them. The
linking matrix K_{mm}^{-1} plays the same role as the NNLS-fitted weights —
it corrects for how the landmarks represent the full set.

### What it suggests concretely

**Nyström landmark selection methods** as alternatives to max-attention key
scoring:
- **K-means Nyström:** cluster keys, use centroids as landmarks. O(T*k*iters).
  This naturally produces *diverse* landmarks without redundancy.
- **Leverage score Nyström:** sample keys proportional to their statistical
  leverage in the kernel matrix. Gives provable (1+eps) multiplicative error
  bounds.
- **Greedy Nyström:** iteratively add the landmark that most reduces the
  approximation error. This is essentially what OMP does, confirming that OMP
  is the right greedy strategy.

**Key difference from current approach:** Nyström theory says the
approximation quality depends on the *spectral decay* of the kernel matrix.
If the attention matrix has fast spectral decay (few dominant eigenvalues),
aggressive compression is possible. This suggests a *diagnostic*: compute the
top-k eigenvalues of the attention matrix to predict achievable compression
before attempting it.

**Nystromformer** (Xiong et al., AAAI 2021) made this connection explicit for
transformers, applying the Nyström formula directly to the softmax kernel
matrix with strided average pooling for landmark selection.

### Papers
- [Nystromformer: A Nyström-Based Algorithm for Approximating Self-Attention](https://arxiv.org/abs/2102.03902) (AAAI 2021)
- Williams & Seeger, "Using the Nyström Method to Speed Up Kernel Machines" (NeurIPS 2001)
- [Improving CUR and Nyström Using QR](https://www.jmlr.org/papers/volume14/wang13c/wang13c.pdf) (JMLR 2013)
- Kumar et al., "Sampling Methods for the Nyström Method" (JMLR 2012)

---

## 4. Determinantal Point Processes (DPPs)

**Maps to: [A] Key Selection**

### What it is

A DPP is a probability distribution over subsets that favors *diverse*
selections. The probability of selecting a subset S is proportional to
det(L_S), where L_S is the principal submatrix of a kernel matrix L. Items
that are similar (high kernel value) are unlikely to be co-selected.

### The structural analogy

The key selection problem needs both *importance* (keep high-attention keys)
and *diversity* (don't keep redundant keys). Max-attention scoring gets
importance but ignores diversity. OMP gets both but is slow.

DPPs naturally balance quality and diversity in a single framework. The DPP
kernel L decomposes as:

```
L_ij = q_i * phi_i^T phi_j * q_j
```

where q_i is a scalar "quality" score (attention importance) and phi_i is a
unit diversity feature vector (normalized key vector). The determinant
simultaneously rewards high individual quality AND mutual orthogonality.
**Items with parallel feature vectors are selected together with probability
zero** — the parallelepiped they span is degenerate. This is exactly the
diversity property that max-attention scoring lacks.

The log-determinant of the DPP kernel is a *submodular* function, so greedy
maximization gives a (1 - 1/e) approximation guarantee (Nemhauser-Wolsey-
Fisher theorem). No such guarantee exists for the current max-attention
scoring.

### What it suggests concretely

**k-DPP sampling** to select exactly t keys. Standard complexity is O(T^3)
for eigendecomposition, but recent work brings this down:
- Coreset-based k-DPP (Li et al., 2016): linear-time construction
- Alpha-DPP (Calandriello et al., NeurIPS 2020): sublinear sampling

For typical T ~ 1000-60000, even O(T^2) may be acceptable given the rest of
the pipeline is O(T * t).

**Practical simplification:** Greedy DPP maximization (select item maximizing
det increase) is O(T * t^2) and often within a constant factor of optimal.
This is simpler than OMP while capturing the diversity benefit.

### Papers
- [Kulesza & Taskar, "Determinantal Point Processes for ML"](http://www.alexkulesza.com/pubs/dpps_fnt12.pdf) (FnT ML 2012) — comprehensive reference
- [k-DPPs: Fixed-Size DPPs](https://icml.cc/2011/papers/611_icmlpaper.pdf) (ICML 2011)
- [Efficient k-DPP Sampling](https://arxiv.org/abs/1509.01618) (2016)
- [Sampling from k-DPP Without Looking at All Items](https://proceedings.neurips.cc/paper/2020/file/4d410063822cd9be28f86701c0bc3a31-Paper.pdf) (NeurIPS 2020)

---

## 5. Column Subset Selection Problem (CSSP) & Leverage Scores

**Maps to: [A] Key Selection**

### What it is

Given a matrix A, select t columns that minimize ||A - A_S A_S^+ A|| (the
error of projecting A onto the selected column span). The gold standard is
leverage score sampling — sample columns proportional to their "leverage,"
which measures how much each column contributes to the row space.

### The structural analogy

Key selection IS column subset selection on the attention matrix. Each key
defines a column of the attention matrix; selecting t keys = selecting t
columns. The CSSP literature provides both algorithms and provable
approximation guarantees.

### What it suggests concretely

**Leverage scores can be computed in O(T * d_k) time** from the key matrix
alone (via a randomized SVD), without computing the full attention matrix.
This is cheaper than the current approach which requires the full Q_ref @ K^T
product.

The leverage score of column j is l_j = ||V_k^T e_j||^2 — the squared norm
of the j-th row of the top-k right singular vectors. This measures how much
column j contributes to the dominant subspace. Approximate leverage scores
can be computed in O(nnz(K) * log T + poly(t)) time using CountSketch or
SRHT random projections.

The approximation guarantee: with O(t * log(t) / eps^2) columns sampled by
leverage scores, the reconstruction error is within (1 + eps) of the best
rank-t approximation. This is an *existential guarantee* — the current
pipeline has none.

**Practical suggestion:** Replace max-attention scoring with leverage score
sampling. If the result is worse (because leverage scores don't account for
the softmax nonlinearity), use a hybrid: pre-filter by leverage scores, then
re-rank by attention.

**Existing application:** LevAttention (2024) already applies this to
transformer attention — computing key leverage scores in O(nd) and selecting
top-scoring keys. Achieves >90% accuracy retention on ViT when keeping only
32 of 197 keys.

### Papers
- Drineas et al., "CUR Matrix Decompositions for Improved Data Analysis" (PNAS 2009)
- [Provably Correct Column Subset Selection](https://jmlr.org/papers/volume18/15-233/15-233.pdf) (JMLR 2017)
- Mahoney & Drineas, "CUR Factorization and Leverage Scores" ([lecture notes](https://www.cs.cornell.edu/courses/cs6220/2017fa/CS6220_Lecture14.pdf))

---

## 6. Optimal Transport & Sinkhorn Iterations

**Maps to: [B] Bias Fitting**

### What it is

Optimal transport (OT) finds the minimum-cost way to transform one
probability distribution into another. Sinkhorn iterations solve the
entropy-regularized OT problem via alternating row/column normalization of
a cost matrix.

### The structural analogy — and a deep theoretical connection

A 2025 paper ("Scaled-Dot-Product Attention as One-Sided Entropic Optimal
Transport," arXiv:2508.08369) **proves that standard softmax attention IS a
one-sided entropic OT problem**. It minimizes transport cost (negative
similarity) from queries to keys with entropy regularization (temperature),
subject to per-query mass conservation. The unique solution is exactly the
softmax function.

This means the mass matching problem (Step 2) is literally an optimal
transport problem: redistribute the original attention mass (distributed
across T keys) onto t keys via non-negative weights, preserving per-query
mass marginals.

### What it suggests concretely

**Replace projected gradient NNLS with Sinkhorn-like multiplicative updates.**
Recent work on Optimal Transport Linear Models (arXiv:2504.04609) shows that
Sinkhorn-like iterations can solve non-negative linear regression with an OT
loss. The updates are multiplicative, which *automatically ensures
non-negativity* — no projection step needed. This is potentially faster and
more numerically stable than the current projected gradient approach.

**Entropic regularization** naturally prevents degenerate solutions (all mass
on one key), acting as a smoother version of the 1e-12 floor hack.

### Related: doubly-stochastic attention

**Sinkformers** (Sander et al., AISTATS 2022) replace softmax (row-stochastic)
with Sinkhorn normalization (doubly-stochastic). Key finding: trained
transformers' attention matrices naturally converge toward doubly-stochastic
matrices over epochs, suggesting full mass conservation is a natural inductive
bias. For compaction, this implies the mass-matching step should consider not
just per-query totals (row sums) but also per-key participation balance
(column sums).

### Papers
- [Attention as One-Sided Entropic OT](https://arxiv.org/pdf/2508.08369) (2025)
- [Sinkformers: Doubly-Stochastic Attention](https://proceedings.mlr.press/v151/sander22a/sander22a.pdf) (AISTATS 2022)
- Cuturi, "Sinkhorn Distances: Lightspeed Computation of Optimal Transport" (NeurIPS 2013)
- [Scalable Approximate Algorithms for OT Linear Models](https://arxiv.org/html/2504.04609v1) (2025)
- [Near-Linear Time Approximation Algorithms for OT via Sinkhorn](https://arxiv.org/pdf/1705.09634) (NeurIPS 2017)

---

## 7. Kernel Herding & Maximum Mean Discrepancy (MMD)

**Maps to: [A] Key Selection + [B] Bias Fitting**

### What it is

Kernel herding is a deterministic algorithm that greedily selects points to
minimize the Maximum Mean Discrepancy (MMD) between the selected subset and
the full distribution. It is equivalent to the Frank-Wolfe algorithm applied
to MMD minimization. Convergence rate: O(1/n) for n selected points.

### The structural analogy

The key selection problem is: choose t keys such that the attention
distribution over selected keys approximates the attention distribution over
all keys. MMD is a natural metric for this — it measures the worst-case
difference in expectations over functions in a reproducing kernel Hilbert
space.

Kernel herding applied to key selection would:
1. Start with empty selection
2. At each step, add the key that most reduces MMD between the selected
   subset's attention distribution and the full distribution
3. Each selected key gets a weight (equivalent to beta)

### What it suggests concretely

**Kernel herding as a principled replacement for OMP.** OMP minimizes mass
residual; herding minimizes MMD. The convergence rate O(1/n) means the
approximation error halves every time you double the budget — a clean
theoretical guarantee.

**The connection to Frank-Wolfe is key:** kernel herding = Frank-Wolfe on
MMD = conditional gradient on a convex objective. This means it inherits
Frank-Wolfe's convergence guarantees while naturally producing sparse
(subset) solutions. The current OMP has no such convergence rate guarantee.

### Papers
- Chen, Welling, Smola, "Super-Samples from Kernel Herding" (UAI 2010)
- [Performance Analysis of Greedy MMD Minimization](https://link.springer.com/article/10.1007/s11222-022-10184-1) (Statistics & Computing 2022)
- [Improved Coresets for Kernel Density Estimates](https://dl.acm.org/doi/abs/10.5555/3174304.3175477) (SODA 2018)

---

## 8. Rate-Distortion Theory

**Maps to: [D] Budget Allocation + overall compression limits**

### What it is

Rate-distortion theory establishes the theoretical minimum "rate" (bits or,
here, number of retained keys) needed to represent a source at a given
distortion level. The rate-distortion function R(D) is the fundamental limit
of lossy compression.

### The structural analogy

For a given model, context, and quality tolerance D (measured as attention
output MSE), there exists a *minimum* number of KV entries t* below which no
compaction method can succeed. This is the rate-distortion limit for the
specific "source" (the attention function).

### What it suggests concretely

**Compute an empirical R(D) curve per head** by measuring reconstruction error
at various compression levels. This directly tells you:
- Which heads have fast-decaying R(D) → compress aggressively
- Which heads have slow-decaying R(D) → need more budget
- The overall achievable compression ratio for a given quality target

This is a more principled version of the paper's "sensitivity curves" for
per-head budget allocation. Instead of measuring sensitivity heuristically,
measure the actual rate-distortion function.

**Vector quantization (VQ):** Rate-distortion theory says VQ is always better
than scalar quantization. Applied to KV compaction: representing C_v as
codebook entries (k-means centroids of original V vectors) rather than
arbitrary vectors should approach the rate-distortion limit more efficiently.

### Papers
- Shannon, "Coding Theorems for a Discrete Source with a Fidelity Criterion" (1959)
- [Stanford — Lossy Compression Basics & Quantization](https://stanforddatacompressionclass.github.io/notes/lossy/quant.html)
- [Gray & Neuhoff — Quantization](https://www.math.ucdavis.edu/~saito/data/quantization/44it06-gray.pdf) (IEEE 1998)

---

## 9. Frank-Wolfe (Conditional Gradient) Method

**Maps to: [A] Key Selection + [C] Value Fitting jointly**

### What it is

Frank-Wolfe optimizes a smooth function over a convex set by iteratively
adding the element from the constraint set that most improves the objective.
It naturally produces *sparse* solutions — after t iterations, the solution
is a combination of at most t extreme points.

### The structural analogy

If we constrain C_v to be a weighted combination of at most t original value
vectors, Frank-Wolfe jointly solves key selection AND value fitting:
- Each iteration selects one key (the "extreme point" that most reduces
  the attention output error)
- The weights on selected keys are jointly optimized
- After t iterations: t selected keys with optimal combination weights

This is equivalent to OMP but with a cleaner convergence theory.

### What it suggests concretely

**Frank-Wolfe as a unified framework for Steps 1+3.** Instead of the current
pipeline (select keys, then fit values separately), Frank-Wolfe selects keys
and fits values in a single optimization loop. Each iteration:
1. Compute gradient of attention output error w.r.t. the current C_v
2. Find the original value vector most aligned with this gradient (= select
   next key)
3. Update the combination weights (line search or fixed step)

**Convergence:** O(1/t) rate for smooth objectives, with the sparsity of the
solution directly matching the budget constraint.

### Deep theoretical connection: attention IS Frank-Wolfe

A remarkable 2025 paper ("Attention's Forward Pass and Frank-Wolfe,"
arXiv:2508.09628) proves that in the hardmax (zero-temperature) limit,
**the self-attention update rule IS a Frank-Wolfe step** over the convex hull
of token embeddings. Key results:
- With positive semidefinite key-query matrix, the dynamics induce a Voronoi
  diagram over tokens, with super-exponential convergence to vertices.
- This provides geometric justification for why attention naturally
  concentrates on ~1-2% of keys — the same observation that makes KV cache
  compaction possible.

This means FW-style greedy selection isn't just a useful algorithm — it's
*aligned with the attention mechanism's own dynamics*.

### Papers
- [Attention's Forward Pass and Frank-Wolfe](https://arxiv.org/abs/2508.09628) (2025)
- Frank & Wolfe, "An Algorithm for Quadratic Programming" (1956)
- Jaggi, "Revisiting Frank-Wolfe" (ICML 2013)
- [Clarkson, "Coresets, Sparse Greedy Approximation, and Frank-Wolfe"](https://kenclarkson.org/sga/p.pdf) (2010)

---

## 10. Alternating Minimization

**Maps to: [B] Bias Fitting + [C] Value Fitting jointly**

### What it is

Alternating minimization optimizes a function of two variables (x, y) by
alternately fixing one and optimizing the other. Despite overall
non-convexity, it converges to a global optimum for many structured problems
(matrix factorization, phase retrieval) given a good initialization.

### The structural analogy

The current pipeline fits beta (Step 2) and C_v (Step 3) independently. But
they interact: the optimal C_v depends on beta (through the softmax weights),
and the "best" beta depends on what C_v can achieve. Alternating minimization
would:
1. Fit beta with C_v fixed (NNLS — existing Step 2)
2. Fit C_v with beta fixed (LS — existing Step 3)
3. Repeat until convergence

### What it suggests concretely

**2-3 alternation rounds between Steps 2 and 3.** Theory says: if the
initialization is good (which it is — single-pass Steps 2+3), alternating
minimization converges *linearly* to the joint optimum. The per-iteration
cost is just repeating the existing NNLS + LS solves, which together take ~4s.

**This is the single cheapest improvement to implement** — it requires zero
new algorithms, just a loop around existing Steps 2 and 3.

**Convergence guarantee:** For bilinear problems (which the beta/C_v
interaction approximates), alternating minimization achieves linear
convergence to the global optimum from a constant-factor initialization
(Jain et al., 2013).

### Papers
- Jain, Netrapalli, Sanghavi, "Low-rank Matrix Completion using Alternating Minimization" (STOC 2013)
- [Nonconvex Optimization Meets Low-Rank Matrix Factorization: An Overview](https://yuxinchen2020.github.io/publications/NcxOverview_Arxiv.pdf)
- Bolte et al., "Proximal Alternating Linearized Minimization" (Math. Programming 2014)

---

## 11. Attention Head Pruning & the Lottery Ticket Hypothesis

**Maps to: [D] Budget Allocation**

### What it is

Research on which attention heads can be removed without hurting performance.
The lottery ticket hypothesis says sparse subnetworks exist that match full
network performance. Applied to heads: most heads are redundant; a small
"winning ticket" subset does the heavy lifting.

### The structural analogy

Per-head budget allocation in KV compaction asks: which heads are sensitive
to compression? Head pruning research asks: which heads matter at all? These
are the same question at different compression levels.

### What it suggests concretely

**Known head importance patterns transfer directly:**
- Voita et al. (ACL 2019): only a small subset of heads have interpretable
  functions (positional, syntactic, rare-word). These are the heads that need
  the most budget.
- Behnke & Heafield (EMNLP 2020): up to 75% of heads can be removed in
  transformer-big with negligible BLEU loss. This implies 75% of heads could
  be maximally compressed.
- Differentiable subset pruning (Li et al., 2021): learns per-head importance
  scores differentiably. These scores could directly inform budget allocation.

**Concrete application:** Use a pre-computed head importance ranking (from
pruning literature) as the prior for budget allocation. Heads ranked as
"prunable" get minimal budget; heads ranked as "essential" get maximum budget.

### Papers
- [Voita et al., "Analyzing Multi-Head Self-Attention"](https://lena-voita.github.io/posts/acl19_heads.html) (ACL 2019)
- [Behnke & Heafield, "Losing Heads in the Lottery"](https://aclanthology.org/2020.emnlp-main.211/) (EMNLP 2020)
- [Differentiable Subset Pruning of Transformer Heads](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00436/108868/) (TACL 2021)

---

## 12. Mixture of Experts Merging

**Maps to: [C] Value Fitting**

### What it is

MoE compression combines multiple expert networks into fewer ones while
preserving output quality. Recent methods (MergeMoE, Sub-MoE, PuzzleMoE)
merge expert weights via learned combinations, SVD-based subspace alignment,
or output-matching optimization.

### The structural analogy

Each original value vector V_j is an "expert" that contributes to the
attention output. Compaction merges T "experts" into t by finding C_v vectors
that preserve the combined output — exactly what MoE merging does.

### What it suggests concretely

**Output-matching merging (MergeMoE):** rather than fitting C_v to minimize
||X @ C_v - Y|| directly, use the MergeMoE insight that *merging should
target output matching, not parameter averaging*. For KV compaction, this
means optimizing C_v to match the final model output (after all subsequent
layers), not just the attention output of the current layer.

**Subspace merging (Sub-MoE):** joint SVD on concatenated value vectors to
find shared structure, then merge in the shared subspace. This could identify
which value vectors are naturally groupable, informing both key selection and
value fitting.

### Papers
- [MergeMoE: Efficient Compression via Expert Output Merging](https://arxiv.org/html/2510.14436v1) (2025)
- [Sub-MoE: Compression via Subspace Expert Merging](https://arxiv.org/abs/2506.23266) (2025)
- [PuzzleMoE: Sparse Expert Merging](https://arxiv.org/html/2511.04805v1) (2025)

---

## 13. Matrix Sketching (Frequent Directions)

**Maps to: [A]+[C] — an alternative to subset selection entirely**

### What it is

Matrix sketching maintains a low-rank approximation of a stream of vectors.
Frequent Directions (Liberty, 2013) maintains a t-row sketch B such that
||A - A B^+ B|| ≤ ||A - A_t|| + ||A||_F / sqrt(t), where A_t is the best
rank-t approximation.

### The structural analogy

Instead of selecting t keys from T, project *all* T keys into a
t-dimensional sketch. The sketch is a t × d matrix that approximates the
full key matrix. This is fundamentally different from subset selection — the
sketched keys don't correspond to any original token.

### What it suggests concretely

**Streaming compaction.** Frequent Directions processes keys one at a time,
maintaining a fixed-size sketch. This enables *online* compaction during
generation — every new token's KV pair is absorbed into the sketch — without
needing to batch-process the full cache.

**The trade-off:** sketching produces better low-rank approximations than
subset selection (provably optimal bounds), but the sketched keys lose
interpretability and correspondence to original tokens. This makes it
incompatible with the current beta framework (which assumes C_k ⊆ K).

**Hybrid approach:** Use sketching for the *value* side (where correspondence
doesn't matter) while keeping subset selection for keys (where the softmax
structure requires real key vectors).

**Critical challenge:** sketching K into B_K means the softmax applies to
Q @ B_K^T rather than Q @ K^T. The covariance guarantee (B^T B ≈ A^T A) does
not transfer through the softmax nonlinearity. This is why subset selection
dominates for keys, but sketching may work for values.

**Already deployed variant:** PALU (ICLR 2025) decomposes K/V projection
weight matrices via SVD into y = xAB, caches only the compressed intermediate
xA, and reconstructs on the fly. Achieves 91%+ compression — but requires
model-specific decomposition rather than post-hoc compaction.

### Papers
- Liberty, "Simple and Deterministic Matrix Sketching" (KDD 2013)
- Ghashami et al., "Frequent Directions: Simple and Deterministic Matrix Sketching" (SIAM 2016)
- [PALU: KV-Cache Compression with Low-Rank Projection](https://arxiv.org/abs/2407.21118) (ICLR 2025)

---

## 14. Knowledge Distillation (Feature-Level)

**Maps to: [C] Value Fitting — alternative loss functions**

### What it is

Feature-level distillation trains a student to match intermediate
representations of a teacher, not just final outputs. CKA (Centered Kernel
Alignment) measures representational similarity between layers, invariant
to rotation and scaling.

### The structural analogy

Step 3 (value fitting) minimizes ||X @ C_v - Y|| — MSE between compacted and
original attention outputs. But does matching attention output at layer L
guarantee matching the final model output? Feature distillation research
says: not necessarily. Intermediate representation errors can amplify or
cancel through subsequent layers.

### What it suggests concretely

**End-to-end distillation loss.** Instead of matching attention output at the
current layer, propagate the compacted output through subsequent layers and
minimize final output divergence. This is more expensive but directly
optimizes what we care about.

**CKA as a fitting objective.** Instead of minimizing ||X @ C_v - Y||_F,
minimize a CKA-based loss that matches the *relational structure* of
attention outputs rather than pointwise values. Saha et al. (BMVC 2022)
showed "it is better to teach students the shape of the similarity
distribution rather than raw values." Recent work (IJCAI 2024) proves
maximizing CKA is equivalent to minimizing an upper bound on MMD — linking
back to kernel herding (Section 7).

**Layer-wise importance weighting:** distillation research shows some layers
are more important to match than others. Apply more aggressive compression
to layers where errors don't propagate, less to layers where they do.

**Does matching attention output guarantee matching final output?** No —
errors compound through subsequent layers, bounded by epsilon_l *
prod_{j=l+1}^{L} ||W_j||. But in practice, spectral norms are close to 1
and the network has inherent robustness. A 2025 study found that even
*reverse* layer matching (student layer 1 to teacher layer L) works
surprisingly well, suggesting intermediate matching is more of a
regularization effect than strict functional equivalence.

### Papers
- [CKA for Knowledge Distillation](https://bmvc2022.mpi-inf.mpg.de/535/) (BMVC 2022)
- [Rethinking CKA in KD](https://arxiv.org/abs/2401.11824) (IJCAI 2024) — proves CKA ≈ MMD
- [PALU: KV-Cache Compression with Low-Rank Projection](https://arxiv.org/abs/2407.21118) (ICLR 2025)
- Romero et al., "FitNets: Hints for Thin Deep Nets" (ICLR 2015)
- Kornblith et al., "Similarity of Neural Network Representations Revisited" (ICML 2019)

---

## 15. Compressed Sensing & Sparse Recovery

**Maps to: [B] Bias Fitting**

### What it is

Compressed sensing recovers sparse signals from underdetermined linear
measurements. The mass matching equation M @ w = m is an underdetermined
system (n_q equations, t unknowns, n_q < t typically) with a non-negativity
constraint.

### The structural analogy

The NNLS problem in Step 2 is a non-negative sparse recovery problem. The
measurement matrix M has a specific structure — its entries are exponentials
of dot products — which may satisfy restricted isometry-like properties.

### What it suggests concretely

**Basis pursuit (L1 minimization) instead of NNLS.** If we want *sparse*
biases (most beta_j = 0, a few are large), L1 regularization naturally
produces this. Sparse beta means most keys contribute their "natural" mass
while a few keys are boosted to compensate for removed keys.

**The sparsity pattern is informative:** if the NNLS solution has many w_j ≈ 1
(beta_j ≈ 0), those keys are "self-sufficient" — they naturally carry the
right mass. Keys with large w_j are "load-bearing" — they compensate for
many removed keys. This pattern could guide key selection: prefer keys whose
natural mass is close to what's needed (w ≈ 1) and avoid keys that require
extreme beta corrections.

**Surprising finding:** NNLS is already near-optimal for this structure.
Research on non-negative sparse recovery shows NNLS reliably recovers sparse
non-negative vectors *without* any explicit L1 regularization — the
non-negativity constraint alone provides implicit sparsity promotion. This
validates the current approach's choice of NNLS over basis pursuit.

**Direct application:** CS-VLM (2025) explicitly frames attention as
compressed sensing, projecting K/V into lower dimensions and using sparse
recovery (ISTA, OMP, or learned LISTA) to reconstruct attention outputs.

### Papers
- Candès & Tao, "Near-Optimal Signal Recovery from Random Projections" (IEEE IT 2006)
- [Perfect Recovery Conditions for Non-Negative Sparse Modeling](https://arxiv.org/pdf/1512.02743) (2015)
- [CS-VLM: Compressed Sensing Attention](https://arxiv.org/html/2507.02957) (2025)
- Slawski & Hein, "Non-negative Least Squares for High-Dimensional Linear Models" (2013)

---

## 16. K-Means Clustering of Keys

**Maps to: [A] Key Selection + [C] Value Fitting**

### What it is

Lloyd's algorithm partitions T keys into t clusters and represents each
cluster by its centroid. This is the classic vector quantization approach.

### The structural analogy

Instead of selecting t keys (subset selection), cluster all T keys into t
groups and use centroids. Each centroid is a weighted average of the keys in
its cluster — a natural "merged" key that represents multiple original keys.

### What it suggests concretely

**Centroid keys lift the C_k ⊆ K constraint** (identified as a key limitation
in the paper) while remaining computationally cheap. K-means on T keys with
d_k dimensions runs in O(T * d_k * t * iters) — comparable to existing key
scoring.

**The cluster assignment naturally defines beta:** if cluster j contains n_j
keys, then beta_j ≈ log(n_j) (the cluster mass).

**The cluster centroid naturally defines C_v:** the centroid value is the
attention-weighted average of values in the cluster.

**Trade-off:** centroid keys are not real key vectors from the cache, so they
may not be compatible with inference engines that expect integer token
indices. But at the tensor level, you just overwrite the key/value data.

---

## 17. Expectation-Maximization (EM)

**Maps to: [A]+[B]+[C] — an alternative joint framework**

### What it is

EM iterates between assigning data points to clusters (E-step) and
recomputing cluster parameters (M-step). It maximizes a lower bound on the
log-likelihood.

### The structural analogy

View compaction as a latent variable model:
- Latent variable z_j ∈ {1,...,t}: which compacted key does original key j
  map to?
- E-step: compute soft assignment probabilities (attention-weighted)
- M-step: recompute C_k, beta, C_v given assignments

### What it suggests concretely

**Soft assignment instead of hard selection.** The current approach makes a
hard decision in Step 1 (select or discard each key). EM would maintain soft
assignments throughout, allowing keys to "partially contribute" to multiple
compacted entries. This is more expressive but harder to implement in the
existing attention framework.

**Practical version:** Run hard selection (Step 1) but then do 2-3 EM
iterations: reassign borderline keys, update beta and C_v. This captures
most of the benefit of soft assignment with minimal implementation cost.

---

## 18. WildCat & Randomly Pivoted Cholesky (RPCholesky)

**Maps to: [A]+[B]+[C] — the unified framework that ties everything together**

### What it is

**WildCat** (Feb 2025, arXiv:2602.10056) explicitly treats attention as a
kernel matrix and applies Nyström approximation with RPCholesky landmark
selection. RPCholesky is an adaptive column selection algorithm that samples
pivots proportional to the *residual diagonal* of the kernel matrix:

```
For r = 1, 2, ..., t:
    p_j^r = h_residual(k_j, k_j) / sum_l h_residual(k_l, k_l)
    sample pivot j* ~ p^r
    update residual via rank-one deflation
```

This is an *adaptive leverage score* method — the residual diagonal entries
serve as proxies for current leverage scores, updated after each selection.
Total cost: O(T * t^2) — same as greedy DPP but simpler.

### Why this is the most important discovery in this document

WildCat **unifies** coresets, Nyström, and KV cache compression into a single
framework with *provable super-polynomial error decay*:

```
||output - output_approx||_max <= 3 * ||V||_max * T^{-a}
```

where a is controlled by the coreset size t. The error decays faster than any
polynomial in T — meaning moderate increases in t give dramatic quality gains.

**The three-way equivalence:**

| Coreset view | Nyström view | KV cache view |
|-------------|-------------|---------------|
| Sensitivity sampling | Landmark selection | Key selection (Step 1) |
| Coreset weights 1/(|S|*p_i) | Pseudoinverse K_{mm}^{-1} K_{mn} | NNLS bias (Step 2) |
| Weighted function eval | Nyström reconstruction | Attention output (Step 3) |

The KV compaction pipeline independently reinvented this structure. Connecting
to the established theory gives access to:
- **Provable error bounds** (the current pipeline has none)
- **Adaptive selection** (RPCholesky adapts based on what's already selected)
- **Optimal weights** via pseudoinverse (vs. the 200-iteration NNLS)

### What it suggests concretely

**Replace max-attention key scoring with RPCholesky.** Instead of scoring all
keys independently then taking top-t, adaptively select keys one at a time,
updating the residual after each. Keys that are redundant with already-
selected keys will have small residual diagonal entries and won't be selected.
This captures diversity automatically.

**Replace NNLS with pseudoinverse weights.** The Nyström formula
W = K_{mm}^{-1} K_{mn} gives optimal unconstrained weights in O(t^2 * T)
time. If non-negativity is needed, project the result or use the
Caratheodory-based recombination that guarantees positive weights.

**The RPCholesky algorithm is simpler than OMP** (no NNLS sub-solve per
iteration) while providing stronger theoretical guarantees.

### Comparison to current pipeline

| Property | Current (HighestAttn) | RPCholesky/WildCat |
|----------|---------------------|-------------------|
| Diversity-aware | No | Yes (adaptive) |
| Error guarantee | None | Super-polynomial decay |
| Needs Q_ref | Yes (repeat-prefill) | No (kernel diagonal only) |
| Complexity | O(T * n_q * d_k) | O(T * t^2) |
| Weight computation | NNLS (200 iters) | Pseudoinverse (closed-form) |

### Papers
- [WildCat: Near-Linear Attention in Theory and Practice](https://arxiv.org/abs/2602.10056) (Feb 2025)
- [Randomly Pivoted Cholesky](https://arxiv.org/abs/2207.06503) (Chen, Epperly, Tropp, Webber, 2023)
- [Accelerated RPCholesky](https://arxiv.org/abs/2410.03969) (2024) — 40x speedup via block computation
- [Adaptive Randomized Pivoting for CSSP](https://doi.org/10.1137/24M1719189) (SIAM 2024)

---

## 19. Sparse Gaussian Processes & Inducing Points

**Maps to: [A] Key Selection + [B] Bias Fitting + [C] Value Fitting**

### What it is

A full Gaussian Process has O(N^3) cost. Sparse GPs introduce M << N
**inducing points** — pseudo-inputs that summarize the entire dataset. The
posterior is approximated by conditioning on these M points instead of all N
data points, reducing cost to O(NM^2). The inducing point locations and the
variational distribution over their function values are jointly optimized.

### The structural analogy

The analogy is exact:

| Sparse GP | KV Cache Compaction |
|---|---|
| Full dataset (N points) | Full KV cache (T tokens) |
| M inducing points | t compacted KV entries |
| Inducing points summarize the function | Compacted KV entries summarize the context |
| Posterior conditioned on inducing points | Attention computed over reduced KV set |
| Variational bound optimizes inducing locations | Steps 1-3 optimize C_k, beta, C_v |
| Nyström approximation of kernel matrix | Low-rank approximation of attention matrix |

The compacted KV entries ARE inducing points for the attention kernel.

### What it suggests concretely

**Variational inference for compaction.** Titsias (2009) showed that the
optimal inducing point locations maximize a variational lower bound on the
marginal likelihood. Applied to KV compaction: instead of the heuristic
three-step pipeline, maximize a variational lower bound on the attention
log-likelihood. This gives a principled objective that jointly optimizes
key selection, bias, and values.

**Decoupled inducing points.** SGPA (arXiv:2303.02444) introduces "decoupled
inducing points" — separate per-sample (amortized) and global (shared)
inducing variables. For KV compaction, this suggests a hybrid: keep some
globally important keys (high-attention across all queries) plus
context-specific keys selected per reference query batch.

### Direct application exists

**KEP-SVGP** (Chen et al., ICML 2024, arXiv:2402.01476) recasts self-attention
as a sparse variational GP. It handles the asymmetric attention kernel (Q*K^T)
via Kernel SVD, producing two sets of singular vectors that induce paired SVGPs.
This makes the structural analogy between attention and GPs mathematically
precise and provides calibrated uncertainty estimates.

### Papers
- [KEP-SVGP: Self-Attention through Kernel-Eigen Pair Sparse Variational GPs](https://arxiv.org/abs/2402.01476) (ICML 2024)
- [SGPA: Calibrating Transformers via Sparse Gaussian Processes](https://arxiv.org/abs/2303.02444) (2023)
- Titsias, "Variational Learning of Inducing Variables in Sparse GPs" (PMLR 2009)
- [Hensman et al., "GPs for Big Data"](https://arxiv.org/abs/1309.6835) (2013)

---

## 20. Token Merging (ToMe)

**Maps to: [A] Key Selection + [C] Value Fitting**

### What it is

Token Merging (Bolya et al., ICLR 2023) reduces token count by **merging
redundant tokens** rather than dropping them. The core algorithm:

1. Partition tokens into two sets A, B (e.g., even/odd indices)
2. Compute cosine similarity between A and B using key vectors (already
   computed for attention — nearly free)
3. Find bipartite soft matching: each token in A finds its most similar
   token in B. Select top-r most similar pairs.
4. Merge matched pairs by (weighted) averaging, reducing token count by r
5. Apply progressively at each transformer block

### The structural analogy

Token merging is a form of KV cache compaction where:
- Key selection = which tokens to merge vs. keep distinct
- Value fitting = the merged value is a weighted average of original values
- The similarity metric (key cosine similarity) replaces attention scoring

The critical difference from eviction: **information is preserved, not
discarded**. A merged token carries the combined signal of both originals.

### What it suggests concretely

**Merge-based compaction as an alternative to subset selection.** Instead of
picking t of T keys, iteratively merge the most similar pairs until t remain.
Each merge:
- Averages the key vectors (new key = centroid of merged cluster)
- Averages the value vectors (weighted by attention mass)
- Accumulates beta as log(cluster_size)

This naturally lifts the C_k ⊆ K constraint (merged keys are centroids) while
preserving information that eviction destroys.

**Graceful degradation.** Merging degrades much more gracefully than eviction at
high compression ratios — merged tokens still carry aggregate signal. D2O
(arXiv:2406.13035) confirms this with a hybrid approach: evict clearly
unimportant tokens, merge borderline ones.

### Papers
- [Token Merging: Your ViT But Faster](https://arxiv.org/abs/2210.09461) (ICLR 2023)
- [Token Fusion: Bridging Pruning and Merging](https://arxiv.org/abs/2312.01026) (WACV 2024)
- [PiToMe: Parallel Informed Token Merging](https://arxiv.org/abs/2405.13828) (2024)
- [D2O: Dynamic Discriminative Operations](https://arxiv.org/abs/2406.13035) (2024) — eviction + merging hybrid

---

## 21. Submodular Optimization (BumbleBee)

**Maps to: [A] Key Selection + [D] Budget Allocation**

### What it is

A set function f is **submodular** if it satisfies diminishing returns:
adding an element to a smaller set gives at least as much marginal gain as
adding it to a larger set. Greedy maximization of monotone submodular functions
achieves a (1 - 1/e) ≈ 0.63 approximation to the optimum (Nemhauser-Wolsey-
Fisher theorem, 1978).

### The structural analogy

Key selection IS subset selection maximizing a quality function. If that quality
function is submodular (or approximately so), greedy selection comes with
approximation guarantees. The key scoring function needs two properties:
- **Coverage**: selected keys should collectively explain the attention mass
- **Diversity**: selecting redundant keys has diminishing returns

A mixture of facility location (coverage) and graph cut (diversity) submodular
functions naturally captures both.

### Direct application exists

**BumbleBee** (Rao et al., COLM 2024) formulates KV cache selection as
streaming submodular summarization. Key contributions:

1. Uses a **mixture of submodular functions** balancing:
   - Diversity among keys in embedding space (graph cut / determinantal)
   - Importance via accumulated attention scores (facility location)
2. Works in both **offline** (prefill) and **online** (streaming/decoding) modes
3. Captures both **long-range coarse-grained** and **short-term fine-grained**
   dependencies
4. Validated across 13 datasets on LLaMA-7B and 13B, outperforming H2O and
   SnapKV at comparable compression ratios

### What it suggests concretely

**Replace max-attention scoring with submodular key selection.** The greedy
submodular algorithm:
1. Start with S = empty set
2. For each step: add key j* = argmax_{j ∉ S} f(S ∪ {j}) - f(S)
3. The (1-1/e) guarantee holds for any monotone submodular f

**Streaming variant for online compaction.** BumbleBee's online mode maintains
a summary as tokens arrive — directly applicable to the paper's "online
compaction during generation" use case (Section 7 of the paper).

**The connection to DPPs (Section 4):** the log-determinant of a DPP kernel is
submodular, so DPP-based key selection is a special case of submodular
optimization with the specific diversity kernel being the key similarity matrix.

### Papers
- [BumbleBee: Dynamic KV-Cache Streaming Submodular Summarization](https://openreview.net/forum?id=8w0RApM5yG) (COLM 2024)
- Nemhauser, Wolsey, Fisher, "Analysis of Approximations for Maximizing Submodular Set Functions" (Math. Programming 1978)
- [Streaming Submodular Optimization](https://arxiv.org/abs/1612.02712) (NeurIPS 2016)

---

## 22. Carathéodory's Theorem

**Maps to: theoretical minimum for compacted cache size**

### What it is

**Carathéodory's theorem (1907):** If a point x lies in the convex hull of a
set S in R^d, then x can be expressed as a convex combination of at most
**d + 1** points from S.

### The structural analogy

The attention output for a query q is:

```
Attn(q) = Σ_i w_i · v_i
```

where w_i are softmax weights (non-negative, sum to 1) and v_i ∈ R^{d_v}.
This is a **convex combination** of value vectors.

By Carathéodory's theorem, this can be exactly represented using at most
**d_v + 1** value vectors with recomputed weights, regardless of T.

### What it suggests concretely

**Hard theoretical floor: t_min = d_v + 1 per head.** For typical d_v = 64
or 128, this means ~65-129 entries per head suffice to exactly reproduce any
single query's attention output. This gives a fundamental compression limit
that the current pipeline lacks.

**Carathéodory recombination** is the constructive algorithm: iteratively find
linear dependencies among value vectors (which must exist when n > d+1), shift
weight off one vector, eliminate it. Repeat until d+1 remain. Runs in
O(n · d^2) per elimination step.

**The key limitation:** attention weights are query-dependent. A single static
Carathéodory reduction preserves the output for one query but not all future
queries. This is why the compaction pipeline needs reference queries Q_ref —
to approximate the universal reduction.

**Practical implication for budget allocation:** if a head's effective value
rank (number of significant singular values of V) is r << d_v, then
Carathéodory says only r+1 entries are needed. Heads with low-rank value
matrices can be compressed far more aggressively. This gives a tighter,
per-head compression floor than the current sensitivity heuristic.

### Papers
- Carathéodory, "Über den Variabilitätsbereich der Koeffizienten" (Math. Annalen, 1907)
- [Braverman, Feldman, Lang — "Coreset Constructions"](https://arxiv.org/abs/1612.00889) (2016) — formalizes Carathéodory-based coreset reduction
- [Feldman, "Core-Sets: An Updated Survey"](https://arxiv.org/abs/2011.09384) (2020)

---

## 23. Information Bottleneck

**Maps to: [D] Budget Allocation + principled compression-quality tradeoff**

### What it is

The Information Bottleneck (Tishby et al., 1999) finds a compressed
representation T of input X that retains maximum information about a relevant
target Y. The objective:

```
minimize  I(X; T) - β · I(T; Y)
```

where β controls the compression-quality tradeoff. The rate-distortion curve
R(D) from Section 8 is the Lagrangian dual of this formulation.

### The structural analogy

For KV cache compaction:
- **X** = full KV cache (all T tokens' keys and values)
- **T** = compacted cache (t entries with beta and C_v)
- **Y** = model's next-token prediction quality

The compacted cache should be a lossy compression of the full cache that
preserves maximum information about generating correct outputs. The IB
Lagrangian formalizes this: tokens carrying high mutual information with the
output should be retained; tokens carrying redundant or irrelevant information
should be compressed away.

### What it suggests concretely

**IB-optimal eviction policy.** Attention weights serve as a proxy for mutual
information: tokens receiving consistently high attention across heads and
layers carry more information about future predictions. The IB framework
formalizes *why* attention-based scoring works and suggests improvements:

1. **Cross-layer information flow.** A token may have low attention at layer L
   but high information content for layer L+5 (because it feeds through
   residual connections). IB says the eviction decision should account for
   information flow through all subsequent layers, not just current attention.

2. **β as a principled compression dial.** Instead of setting a fixed budget t,
   set a target IB loss β and let the optimal t emerge from the tradeoff curve.
   Different contexts have different information densities — medical records
   need more budget than conversational text.

3. **Connection to variational inference.** The IB objective can be optimized
   via variational bounds (the "Deep Variational Information Bottleneck" of
   Alemi et al., 2016). This connects to the sparse GP framework (Section 19)
   — both use variational lower bounds on information-theoretic objectives.

### Papers
- [Tishby et al., "The Information Bottleneck Method"](https://arxiv.org/abs/physics/0004057) (1999)
- [Shwartz-Ziv & Tishby, "Opening the Black Box of DNNs via Information"](https://arxiv.org/abs/1703.00810) (2017)
- [Ge et al., "Model Tells You What to Discard: Adaptive KV Cache Compression"](https://arxiv.org/abs/2310.01801) (ICLR 2024) — applies IB intuition to KV cache
- Alemi et al., "Deep Variational Information Bottleneck" (ICLR 2017)

---

## 24. Fast Multipole Method & Hierarchical Attention (MuSe)

**Maps to: [A] Key Selection — multi-resolution alternative**

### What it is

The Fast Multipole Method (Greengard & Rokhlin, 1987) computes N-body
interactions in O(N) by organizing points hierarchically:
- **Nearby** interactions: computed at full resolution
- **Distant** interactions: approximated by compact "multipole" summaries
  (total mass, dipole direction, etc.)

The key insight: **full pairwise detail is unnecessary for distant interactions
— a compact summary suffices.**

### The structural analogy

KV cache compaction is an N-body problem where "force" = attention weight:

| FMM Concept | KV Cache Analogy |
|---|---|
| Nearby particles (full resolution) | Recent/high-attention KV pairs at full fidelity |
| Distant particles (multipole summary) | Old/low-attention KV pairs as compressed summaries |
| Multipole expansion (monopole + dipole) | Cluster centroid + covariance correction |
| Hierarchical tree levels | Multiple resolution tiers of cached KV pairs |
| Adaptive refinement | Dynamic eviction/merging based on attention scores |

### Direct application exists

**MuSe** (Multipole Semantic Attention, arXiv:2509.10406, 2025) extends
multipole ideas into representation space:

1. **Semantic clustering:** keys and queries are clustered in representation
   space (not positional space), yielding query-specific key summaries
2. **Dipole corrections:** beyond centroid (monopole) representations, adds
   covariance-based dipole terms capturing directional variance within
   clusters — reduces approximation error
3. **Drop-in compatible** with pretrained models (validated on Llama 3.1-8B
   without retraining). Accelerates 64k-context pretraining by 36% while
   matching baseline loss.

### What it suggests concretely

**Multi-resolution compaction.** Instead of a single flat compacted cache,
maintain a hierarchy:
- **Tier 1** (full resolution): last N tokens, always kept
- **Tier 2** (moderate compression): keys compressed 4x via merging
- **Tier 3** (aggressive compression): keys compressed 16x via centroids

As tokens age, they cascade down tiers. This naturally matches the attention
decay pattern: recent tokens get high attention (need full resolution), distant
tokens get low attention (coarse summary suffices).

**The dipole correction idea applies directly to value fitting.** When merging
t_cluster keys into a centroid, store not just the centroid but also its
principal direction of variance. This captures within-cluster structure that
simple averaging misses, at minimal memory cost (1 extra vector per cluster).

### Papers
- [MuSe: Multipole Semantic Attention](https://arxiv.org/abs/2509.10406) (2025)
- [Fast Multipole Attention for Transformers](https://arxiv.org/abs/2310.11960) (2023)
- [HERMES: KV Cache as Hierarchical Memory](https://arxiv.org/abs/2601.14724) (2025)
- Greengard & Rokhlin, "A Fast Algorithm for Particle Simulations" (J. Comp. Physics 1987)

---

## Summary: Concept-to-Component Map

| Concept | Key Select [A] | Bias [B] | Value [C] | Budget [D] | New idea? |
|---------|:-:|:-:|:-:|:-:|-----------|
| CUR / ID decomposition | x | | x | | Leverage scores for selection |
| Coresets | x | x | | | Sensitivity sampling + theoretical bounds |
| Nyström approximation | x | x | | | Spectral decay diagnostic |
| DPPs | x | | | | Diversity-aware selection |
| CSSP / Leverage scores | x | | | | Cheap O(Td) score computation |
| Optimal transport / Sinkhorn | | x | | | Multiplicative NNLS updates |
| Kernel herding / MMD | x | x | | | Frank-Wolfe with convergence rate |
| Rate-distortion theory | | | | x | Theoretical compression limits |
| Frank-Wolfe | x | | x | | Joint key+value optimization |
| Alternating minimization | | x | x | | Iterate Steps 2+3 (cheapest win) |
| Head pruning / lottery | | | | x | Pre-computed head importance |
| MoE merging | | | x | | Output-matching loss |
| Matrix sketching | x | | x | | Streaming online compaction |
| Feature distillation | | | x | | End-to-end loss |
| Compressed sensing | | x | | | Sparse beta via L1 |
| K-means clustering | x | | x | | Centroid keys (lift C_k ⊆ K) |
| EM algorithm | x | x | x | | Soft assignment framework |
| **WildCat / RPCholesky** | **x** | **x** | **x** | | **Unified framework with provable bounds** |
| Sparse GP / Inducing pts | x | x | x | | Variational objective for joint optimization |
| Token Merging (ToMe) | x | | x | | Merge-based compaction preserves info |
| Submodular (BumbleBee) | x | | | x | (1-1/e) guarantee + streaming mode |
| Carathéodory's theorem | | | | | Theoretical floor: t_min = d_v + 1 |
| Information Bottleneck | | | | x | Principled compression-quality tradeoff |
| Fast Multipole / MuSe | x | | | | Multi-resolution hierarchical caching |

### The Grand Unification

Three independent results converge on the same conclusion:
- **Clarkson (2010):** Frank-Wolfe = Matching Pursuit = Coreset construction
- **Alcalde et al. (2025):** Attention's forward pass IS a Frank-Wolfe step
- **WildCat (2025):** KV compaction IS Nyström approximation of a kernel matrix

**The compaction pipeline is not an ad-hoc engineering solution.** It is a
specific instance of well-studied mathematical structures with decades of
optimality theory. The researchers independently reinvented these structures.
Connecting to the established theory unlocks provable bounds, faster
algorithms, and principled design choices.

### Top 7 Most Actionable (effort vs. impact)

1. **RPCholesky key selection** (Sec 18) — adaptive, diversity-aware,
   provable error bounds, no Q_ref needed. Replaces both key selection AND
   NNLS in a unified framework. **The single most impactful change.**
2. **Alternating minimization** (Sec 10) — loop existing Steps 2+3. Zero new
   code, just a for loop. Cheapest quality improvement to implement.
3. **Submodular key selection** (Sec 21) — BumbleBee's mixture of facility
   location + diversity gives (1-1/e) guarantee with streaming online mode.
   Directly applicable to the paper's online compaction use case.
4. **Token merging** (Sec 20) — merge similar keys instead of evicting.
   Preserves information, degrades gracefully at high compression, lifts
   C_k ⊆ K constraint. Hybrid with eviction (D2O) is most promising.
5. **Sinkhorn for mass matching** (Sec 6) — drop-in NNLS replacement with
   automatic non-negativity and theoretical OT connection.
6. **K-means centroid keys** (Sec 16) — lifts the C_k ⊆ K restriction at
   low cost. Empirically outperforms leverage scores (84% vs 77% on ViT).
7. **Carathéodory-informed budgets** (Sec 22) — compute per-head value rank
   to determine theoretical compression floor (t_min = rank(V) + 1). Heads
   with low-rank values can be compressed far more aggressively than the
   current sensitivity heuristic suggests.
