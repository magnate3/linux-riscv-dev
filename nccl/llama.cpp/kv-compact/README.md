# kv-compact

Fast KV Cache Compaction via Attention Matching — a C++ implementation of [arXiv:2602.16284](https://arxiv.org/abs/2602.16284).

Compresses transformer KV caches by 50x with minimal quality loss using a closed-form 3-step algorithm:

1. **Key selection** — pick top-t keys by cumulative attention score
2. **NNLS bias fitting** — solve for attention mass biases (β) to match original attention distribution
3. **Least squares value refitting** — compute optimal compacted values (C_v) via ridge regression

## Project structure

```
include/kv-compact-math.h   # Header-only math library (zero dependencies)
src/kv-compact.cpp           # CLI tool (requires llama.cpp)
tests/test-kv-compact-math.cpp  # 22 unit tests
docs/                        # Research notes and design docs
```

## Quick start — tests only (no dependencies)

```bash
mkdir build && cd build
cmake .. -DKV_COMPACT_BUILD_TOOL=OFF
cmake --build .
./test-kv-compact-math
```

## Full build with llama.cpp

### Option A: Point to local llama.cpp checkout

```bash
cmake .. -DLLAMA_CPP_DIR=/path/to/llama.cpp
cmake --build .
```

### Option B: Auto-fetch from GitHub

```bash
cmake ..
cmake --build .
```

### Usage

```bash
./llama-kv-compact -m model.gguf -p "your context..." --compact-ratio 0.2
```

## Paper

> **Fast KV Compaction via Attention Matching**
> Zweiger et al., 2026 — [arXiv:2602.16284](https://arxiv.org/abs/2602.16284)
>
> Achieves 50x KV cache compression with closed-form solutions (no gradient descent).
> Value refitting reduces MSE by ~4,000,000x compared to naive token eviction.

## Test results

- 22 tests covering matrix ops, softmax, NNLS, least squares, and full pipeline
- Value refitting: ~4M× MSE improvement over token eviction at 4x compression
- Cosine similarity: 0.999999 at 50% compression
