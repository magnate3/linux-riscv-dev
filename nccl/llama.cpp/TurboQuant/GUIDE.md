# TurboQuant Fork: Development & Testing Guide

## What This Is

This is a development guide for improving the turbo-tan/llama.cpp-tq3 fork.
We are adding features from the Google TurboQuant paper (arXiv:2504.19874) and
novel improvements. See PLAN.md for the full improvement roadmap.

This is NOT a "download model and run" guide. It's a build-test-iterate workflow.

---

## Quick Reference: Our New Types

| Type | bpw | Block size | Use | Status |
|------|-----|-----------|-----|--------|
| TQ3_0 | 3.5 | 14B/32elem | KV cache (existing fork) | Baseline |
| TQ3_1S | 4.0 | 16B/32elem | Weights (existing fork) | Baseline |
| **TQ3_QJL** | **5.0** | **20B/32elem** | **KV cache + QJL** | **Phase 3a done** |
| **TQ3_1S_QJL** | **5.5** | **22B/32elem** | **Weights + QJL** | **Phase 3b pending** |

---

## Build on exile (primary)

```bash
ssh exile
cd /opt/tq/llama.cpp-tq3
export PATH=/usr/local/cuda/bin:$PATH

# Apply patches from local
# (scp patches from D:\projects\tq\ then git apply)

# Build
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j8

# Verify
./build/bin/llama-quantize --help | grep TQ
```

## Build locally (Windows, editing only)

```bash
# PowerShell approach (from bash):
powershell.exe -NoProfile -Command "
& 'C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\Launch-VsDevShell.ps1' -Arch amd64 -SkipAutomaticLocation 2>`$null
Set-Location D:\projects\tq\llama.cpp-tq3
cmake -B build -G Ninja -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DCMAKE_BUILD_TYPE=Release '-DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler'
cmake --build build --config Release -j
"
```

## Push patches to exile

```bash
# From local (D:\projects\tq\llama.cpp-tq3):
cd /d/projects/tq/llama.cpp-tq3
git diff HEAD > /tmp/tq_patches.diff
scp /tmp/tq_patches.diff exile:/opt/tq/

# On exile:
ssh exile "cd /opt/tq/llama.cpp-tq3 && git checkout -- . && git apply /opt/tq/tq_patches.diff"
```

---

## Testing Workflow

### Prerequisites
```bash
ssh exile
systemctl stop wallpaper-worker   # frees 16 GB VRAM
nvidia-smi                         # should show ~24 GB free
```

### 1. Weight quantization comparison

Quantize Qwen3-14B to different formats and compare perplexity:

```bash
cd /opt/tq/llama.cpp-tq3
BIN=./build/bin
MODEL=/opt/tq/models/Qwen3-14B-BF16.gguf
OUT=/opt/tq/models

# Baseline: Q4_0 (4.5 bpw)
$BIN/llama-quantize $MODEL $OUT/Qwen3-14B-Q4_0.gguf Q4_0 8

# Fork baseline: TQ3_1S (4.0 bpw)
$BIN/llama-quantize $MODEL $OUT/Qwen3-14B-TQ3_1S.gguf TQ3_1S 8

# Our improvement: TQ3_1S_QJL (5.5 bpw) — after Phase 3b
$BIN/llama-quantize $MODEL $OUT/Qwen3-14B-TQ3_1S_QJL.gguf TQ3_1S_QJL 8

# Reference: Q5_K_S (5.5 bpw)
$BIN/llama-quantize $MODEL $OUT/Qwen3-14B-Q5_K_S.gguf Q5_K_S 8
```

### 2. Perplexity tests (weights)

```bash
# Download wikitext if not present
# wget https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
# unzip -j wikitext-2-raw-v1.zip -d /opt/tq/data/

for QUANT in Q4_0 TQ3_1S TQ3_1S_QJL Q5_K_S; do
  echo "=== $QUANT ==="
  $BIN/llama-perplexity \
    -m $OUT/Qwen3-14B-${QUANT}.gguf \
    -f /opt/tq/data/wiki.test.raw \
    -c 512 -ngl 99 -fa 1 -t 8 --no-warmup
done
```

### 3. KV cache comparison

Using the same weight model, compare KV cache types:

```bash
WEIGHTS=$OUT/Qwen3-14B-TQ3_1S.gguf

# Baseline: FP16 KV
$BIN/llama-perplexity -m $WEIGHTS -f /opt/tq/data/wiki.test.raw \
  -c 512 -ngl 99 -fa 1 -t 8 --no-warmup

# Fork: TQ3_0 KV (3.5 bpw)
$BIN/llama-perplexity -m $WEIGHTS -f /opt/tq/data/wiki.test.raw \
  -c 512 -ngl 99 -fa 1 -t 8 --no-warmup \
  --cache-type-k tq3_0 --cache-type-v f16

# Ours: TQ3_QJL KV (5.0 bpw)
$BIN/llama-perplexity -m $WEIGHTS -f /opt/tq/data/wiki.test.raw \
  -c 512 -ngl 99 -fa 1 -t 8 --no-warmup \
  --cache-type-k tq3_qjl --cache-type-v f16
```

### 4. Interactive test

```bash
$BIN/llama-cli \
  -m $OUT/Qwen3-14B-TQ3_1S.gguf \
  -ngl 99 -fa on -c 4096 \
  --cache-type-k tq3_qjl --cache-type-v f16 \
  -cnv -p "You are a helpful assistant"
```

### 5. Cleanup — restart wallpaper

```bash
systemctl start wallpaper-worker
```

---

## Results Table (fill in as we test)

### Weight quantization perplexity (wiki.test.raw, c=512, 5 chunks)

| Format | bpw | Size (14B) | PPL | Speed (s/pass) |
|--------|-----|-----------|-----|----------------|
| BF16 (baseline) | 16.0 | 28 GB | — | — |
| Q5_K_S | 5.54 | ~9.7 GB | | |
| TQ3_1S_QJL (ours) | 5.5 | ~9.6 GB | — | — (not implemented yet) |
| **Q4_0** | **4.50** | **5.9 GB** | **8.23** | **1.3** |
| **TQ3_1S (fork)** | **4.00** | **7.2 GB** | **8.99** | **4.2** |

### KV cache perplexity impact (TQ3_1S weights, c=512, 5 chunks)

| Cache K type | Cache V type | bpw (K) | PPL | Speed (s/pass) | Slowdown |
|-------------|-------------|---------|-----|----------------|----------|
| **FP16** | **FP16** | **16.0** | **8.99** | **4.2** | **1x** |
| **TQ3_0** | **F16** | **3.5** | **9.71** | **92** | **22x** |
| TQ3_QJL | F16 | 5.0 | SEGFAULT | — | needs GET_ROWS kernel |

**Key finding:** TQ3_0 KV cache is **22x slower** on RTX 3090 and adds +8% PPL.
The massive slowdown makes Phase A (residual window with FP16 recent tokens) critical.
| TQ3_QJL (ours) | 5.0 | |
| TQ3_0 (fork) | 3.5 | |

---

## Adding a New Phase Implementation

When implementing a new phase from PLAN.md:

1. Read the phase description in PLAN.md
2. Create GitHub issue: `gh issue create --repo smurz/turboquant-llama`
3. Edit code locally in `D:\projects\tq\llama.cpp-tq3\`
4. Build locally to check compilation
5. Push patches to exile: `git diff HEAD > /tmp/patch.diff && scp ... && ssh exile git apply`
6. Rebuild on exile: `ssh exile "cd /opt/tq/llama.cpp-tq3 && cmake --build build -j8"`
7. Test on exile (24 GB VRAM)
8. Record results in this guide's results table
9. Commit and push to smurz/turboquant-llama
10. Close GitHub issue with results
