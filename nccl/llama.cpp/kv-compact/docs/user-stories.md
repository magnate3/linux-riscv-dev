# User Stories: KV Cache Compaction via Attention Matching

## Overview

These user stories describe the incremental path from the current POC to a
production-ready KV cache compaction feature in llama.cpp, based on the
"Fast KV Compaction via Attention Matching" paper (Zweiger et al., 2026).

---

## Epic 1: Core Compaction Integration

### US-1: Inject attention biases (beta) into generation

**As a** developer integrating KV compaction into the inference pipeline,
**I want** the compacted cache's beta biases to be applied during attention
computation at generation time,
**so that** the compacted keys correctly approximate the original attention mass
distribution.

**Acceptance criteria:**
- Beta values are stored alongside compacted KV entries (per layer, per head)
- During `llama_decode`, attention scores for compacted positions have beta
  added before softmax: `score_ij = q_i @ k_j / sqrt(d) + beta_j`
- Generation output with beta injection matches or improves upon token-eviction
  baseline quality on a reference prompt
- No regression in inference speed for non-compacted contexts (beta = 0 path)

**Notes:**
- The POC currently computes beta but cannot inject it — the attention graph
  would need modification or a bias hook
- Investigate `llama_kv_cache_*` or `ggml_flash_attn_ext` for bias support

---

### US-2: Write refitted values (C_v) back into the KV cache

**As a** developer integrating KV compaction into the inference pipeline,
**I want** the least-squares-optimized values (C_v) to replace the original
values in the KV cache after compaction,
**so that** the attention output with the compacted cache closely approximates
the original uncompressed output.

**Acceptance criteria:**
- After compaction, C_v values are written to the V tensor for selected
  positions via `llama_state_seq_set_data` or direct tensor writes
- Supports F32 and F16 KV cache types (quantized types deferred)
- Round-trip test: compact, write C_v, read back, verify values match
- Quality test: MSE of attention output with C_v < MSE with original V
  (already demonstrated in unit tests — needs integration-level verification)

---

### US-3: Compact all layers and heads (not just layer 0)

**As a** user running compaction on a real model,
**I want** the algorithm to compact every layer and every KV head independently,
**so that** the full model benefits from cache reduction rather than just a
single demo layer.

**Acceptance criteria:**
- Compaction loop iterates over all `n_layer` layers and `n_head_kv` heads
- Each head gets its own selected indices, beta, and C_v
- Per-head compaction is independent (no cross-head data dependencies)
- Progress reporting shows layer/head progress
- Total wall-clock time is reported

---

## Epic 2: Reference Query Generation

### US-4: Implement true repeat-prefill for reference query extraction

**As a** developer improving compaction quality,
**I want** to generate reference queries by running a "repeat-prefill" pass
(feeding the context twice as described in the paper),
**so that** the reference queries reflect actual model behavior rather than
using K vectors as a proxy.

**Acceptance criteria:**
- After initial prefill, a second pass processes the same context
- Query activations are captured from the second pass for each layer/head
- Configurable number of reference queries (`--n-ref-queries`)
- Quality comparison: repeat-prefill queries vs. K-vector proxy, measured by
  MSE of compacted attention output vs. original

---

## Epic 3: Advanced Key Selection

### US-5: Support per-head non-uniform compression budgets

**As a** user seeking optimal quality at a given compression ratio,
**I want** the compaction algorithm to allocate more budget (keep more keys)
for attention heads that are more sensitive to compression,
**so that** quality-critical heads retain more information while less important
heads are compressed more aggressively.

**Acceptance criteria:**
- Sensitivity metric computed per head (e.g., max attention entropy, attention
  spread, or reconstruction error with uniform budget)
- Budget allocation redistributes the total `t` budget across heads
- Total tokens kept across all heads equals the global target
- Quality improves over uniform budget allocation on at least one benchmark

---

### US-6: Implement Orthogonal Matching Pursuit (OMP) key selection

**As a** researcher comparing compaction strategies,
**I want** an OMP-based key selection method as an alternative to "Highest
Attention Keys",
**so that** I can evaluate the quality/speed tradeoff described in the paper
(OMP is slower but can yield better key subsets).

**Acceptance criteria:**
- OMP iteratively selects keys that maximize residual reduction
- Selectable via `--key-selection omp` vs `--key-selection highest-attn`
- OMP produces equal or better quality than Highest Attention at same budget
- Wall-clock time reported for comparison

---

## Epic 4: Production Readiness

### US-7: Support quantized KV cache types

**As a** user running models with quantized KV caches (Q8_0, Q4_0, etc.),
**I want** compaction to work with quantized K and V tensors,
**so that** I can benefit from both quantization and compaction simultaneously.

**Acceptance criteria:**
- K/V data is dequantized to F32 for compaction math
- Compacted C_v is re-quantized to match the original KV type before writing
- Round-trip quantization error is measured and reported
- Quality degrades gracefully compared to F32/F16 compaction

---

### US-8: Expose compaction as a library API (not just a CLI tool)

**As a** developer building applications with llama.cpp,
**I want** a C API for KV cache compaction that I can call programmatically,
**so that** I can trigger compaction at runtime when context grows too large
without spawning a separate tool.

**Acceptance criteria:**
- New API functions in `llama.h`:
  - `llama_kv_compact(ctx, seq_id, target_ratio, params)` — compact a sequence
  - `llama_kv_compact_params_default()` — sensible defaults
- Thread-safe: compaction can run while other sequences are being decoded
- Returns compaction statistics (tokens before/after, quality metrics)
- Documented in header with usage example

---

### US-9: Iterative (multi-round) compaction support

**As a** user with very long conversations,
**I want** to apply compaction multiple times as the context grows,
**so that** the cache stays within budget over extended interactions without
catastrophic quality loss.

**Acceptance criteria:**
- Compaction can be applied to an already-compacted cache
- Quality after N rounds of compaction is measured and reported
- Paper claims 6 consecutive compressions on AIME maintain quality —
  verify this with at least 3 rounds on a reference task
- Beta values from previous compactions are preserved or re-optimized

---

## Epic 5: Benchmarking & Validation

### US-10: Automated quality benchmarks

**As a** developer validating compaction quality,
**I want** automated benchmark scripts that compare compacted vs. uncompressed
generation across standard tasks,
**so that** quality regressions are caught before merging changes.

**Acceptance criteria:**
- Benchmark script runs perplexity evaluation with and without compaction
- Reports: perplexity delta, token-level agreement rate, cosine similarity
- Tests at multiple compression ratios (20%, 50%, 80% retention)
- Runs on at least one small model (e.g., 1B parameter) in CI

---

### US-11: Memory and latency profiling

**As a** user evaluating whether compaction is worthwhile for my use case,
**I want** the tool to report memory savings and compaction latency,
**so that** I can make informed decisions about the memory/quality/speed
tradeoff.

**Acceptance criteria:**
- Reports peak memory before and after compaction
- Reports wall-clock time for each compaction phase (key selection, NNLS, LS)
- Reports amortized cost: compaction time vs. time saved from smaller cache
  during subsequent generation
- Output format is machine-parseable (JSON option)
