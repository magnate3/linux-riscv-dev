# SYNTHESIS v2: The Uncached Memory Fabric

## The Clean Cut

**The memory fabric is a software abstraction that routes memory access to cached or uncached paths based on access pattern, enabling cross-core cooperation without coherency overhead.**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE UNCACHED MEMORY FABRIC                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                         ┌─────────────────────┐                             │
│                         │   Access Pattern    │                             │
│                         │     Classifier      │                             │
│                         └──────────┬──────────┘                             │
│                                    │                                         │
│                    ┌───────────────┼───────────────┐                        │
│                    ▼               ▼               ▼                        │
│           ┌────────────┐   ┌────────────┐   ┌────────────┐                 │
│           │ Sequential │   │   Random   │   │  Repeated  │                 │
│           │   < 16MB   │   │   > 16MB   │   │   < 16MB   │                 │
│           └─────┬──────┘   └─────┬──────┘   └─────┬──────┘                 │
│                 │                 │                 │                        │
│                 ▼                 ▼                 ▼                        │
│           ┌────────────┐   ┌────────────┐   ┌────────────┐                 │
│           │   CACHED   │   │  UNCACHED  │   │   CACHED   │                 │
│           │   malloc   │   │  dma_heap  │   │   malloc   │                 │
│           │            │   │            │   │            │                 │
│           │ HW prefetch│   │ + Prefetch │   │ Cache hits │                 │
│           │  handles   │   │   Thread   │   │  handle    │                 │
│           └────────────┘   └────────────┘   └────────────┘                 │
│                                    │                                         │
│                                    ▼                                         │
│                         ┌─────────────────────┐                             │
│                         │  LITTLE Core Pool   │                             │
│                         │  (Prefetch Workers) │                             │
│                         └─────────────────────┘                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Empirical Foundation

### Probe Results Summary

| Probe | Test | Result | Insight |
|-------|------|--------|---------|
| B v7 | Cached baseline | 172 µs, CV 64.5% | Cached has high variance |
| B v7 | Uncached baseline | 137 µs, CV 1.9% | Uncached faster + consistent |
| B v8 | Sequential cached vs uncached | Cached 2-4x faster | HW prefetch wins |
| B v8 | Random >16MB cached vs uncached | Uncached 1.2-1.4x faster | No coherency overhead |
| C | Uncached + prefetch (random) | **1.09x, CV 1.6%** | **Fabric works!** |
| D | LITTLE prefetch (sequential) | 0.91-0.99x | Contention for sequential |

### The Key Finding

**Probe C proves the fabric concept works:**
- Uncached CPU + Uncached "GPU" (LITTLE core) prefetch
- Random access: 1.09x speedup, variance drops from 13.2% to 1.6%
- No cache coherency overhead because neither caches the data

---

## Architecture

### Memory Classes

```c
typedef enum {
    MEM_CACHED,     // malloc/mmap - for sequential, repeated, small
    MEM_UNCACHED,   // dma_heap - for large random access
} mem_class_t;

typedef struct {
    void *ptr;
    size_t size;
    mem_class_t class;
    int dma_fd;     // Only for UNCACHED
    int heap_fd;    // Only for UNCACHED
} fabric_alloc_t;
```

### Allocation Strategy

```c
fabric_alloc_t* fabric_alloc(size_t size, access_pattern_t pattern) {
    fabric_alloc_t *alloc = calloc(1, sizeof(fabric_alloc_t));
    alloc->size = size;
    
    // Decision tree
    if (size < 16 * 1024 * 1024) {
        // Small: always cached
        alloc->class = MEM_CACHED;
        alloc->ptr = aligned_alloc(4096, size);
    } else if (pattern == PATTERN_SEQUENTIAL) {
        // Large sequential: cached (HW prefetch)
        alloc->class = MEM_CACHED;
        alloc->ptr = aligned_alloc(4096, size);
    } else if (pattern == PATTERN_RANDOM) {
        // Large random: uncached
        alloc->class = MEM_UNCACHED;
        alloc->ptr = dma_heap_alloc(size, &alloc->heap_fd, &alloc->dma_fd);
    } else {
        // Default: cached
        alloc->class = MEM_CACHED;
        alloc->ptr = aligned_alloc(4096, size);
    }
    
    return alloc;
}
```

### LLM Component Mapping

```c
// Model loading
weights = fabric_alloc(model_size, PATTERN_SEQUENTIAL);     // CACHED
kv_cache = fabric_alloc(kv_size, PATTERN_RANDOM);           // UNCACHED
embeddings = fabric_alloc(embed_size, PATTERN_RANDOM);      // UNCACHED
activations = fabric_alloc(act_size, PATTERN_REPEATED);     // CACHED
```

### Prefetch Thread Pool

```c
#define NUM_PREFETCH_THREADS 2  // Use 2 LITTLE cores

typedef struct {
    void *region;           // Uncached region to prefetch
    size_t offset;          // Start offset
    size_t length;          // Bytes to prefetch
    volatile int ready;     // Signal to prefetch
    volatile int done;      // Signal prefetch complete
} prefetch_request_t;

prefetch_request_t prefetch_queue[NUM_PREFETCH_THREADS];
pthread_t prefetch_threads[NUM_PREFETCH_THREADS];

void *prefetch_worker(void *arg) {
    int id = *(int*)arg;
    prefetch_request_t *req = &prefetch_queue[id];
    
    // Pin to LITTLE core
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(id, &set);  // Cores 0-5 are LITTLE
    sched_setaffinity(0, sizeof(set), &set);
    
    while (1) {
        while (!req->ready) usleep(10);
        
        // Prefetch by reading (uncached - warms DRAM rows)
        volatile float sum = 0;
        float *ptr = (float*)req->region;
        size_t start = req->offset / sizeof(float);
        size_t count = req->length / sizeof(float);
        for (size_t i = 0; i < count; i += 16) {
            sum += ptr[start + i];
        }
        
        req->done = 1;
        req->ready = 0;
    }
}

void fabric_prefetch(fabric_alloc_t *alloc, size_t offset, size_t length) {
    if (alloc->class != MEM_UNCACHED) return;  // Only for uncached
    
    // Find available prefetch thread
    for (int i = 0; i < NUM_PREFETCH_THREADS; i++) {
        if (!prefetch_queue[i].ready) {
            prefetch_queue[i].region = alloc->ptr;
            prefetch_queue[i].offset = offset;
            prefetch_queue[i].length = length;
            prefetch_queue[i].done = 0;
            prefetch_queue[i].ready = 1;
            return;
        }
    }
}
```

---

## Integration with llama.cpp

### Option 1: Custom Backend

Create a new backend that uses the fabric allocator:

```cpp
// ggml-fabric.cpp

struct ggml_backend_fabric_buffer_context {
    fabric_alloc_t *alloc;
};

static void * ggml_backend_fabric_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto ctx = (ggml_backend_fabric_buffer_context *)buffer->context;
    return ctx->alloc->ptr;
}

// Register KV cache as UNCACHED
ggml_backend_buffer_t ggml_backend_fabric_alloc_kv_cache(size_t size) {
    fabric_alloc_t *alloc = fabric_alloc(size, PATTERN_RANDOM);
    // ... create buffer with alloc ...
}
```

### Option 2: Environment Variable Hack

Quick test without code changes:

```bash
# Set LD_PRELOAD to intercept malloc for large allocations
# Route >16MB to uncached dma_heap
# This is hacky but could validate the concept
```

### Option 3: Direct KV Cache Modification

Modify llama.cpp's KV cache allocation specifically:

```cpp
// In llama.cpp, find kv_cache allocation
// Replace with:

#ifdef ANDROID
    // Use uncached memory for KV cache
    kv_cache.cells = (llama_kv_cell *)fabric_alloc_uncached(kv_cache_size);
#else
    kv_cache.cells = (llama_kv_cell *)malloc(kv_cache_size);
#endif
```

---

## Expected Impact

Based on probe measurements:

### Latency
- KV cache random access: ~1.3x faster with uncached
- KV cache with prefetch: additional 1.09x
- Combined: **~1.4x faster KV access**

### Variance
- Cached KV: CV 10-65%
- Uncached KV: CV 1-2%
- **10-30x more consistent latency**

### Overall Inference
- KV cache is ~20-30% of attention time
- Attention is ~40-50% of total inference
- KV improvement: 1.4x on 20-30% of 40-50%
- **Expected: 5-15% overall tok/s improvement**
- **Expected: Much smoother token generation**

---

## Implementation Roadmap

### Phase 1: Validation (1 day)
1. Modify llama.cpp to use dma_heap for KV cache
2. Measure tok/s before/after
3. Measure latency variance before/after

### Phase 2: Fabric Library (3 days)
1. Create `libfabric.so` with:
   - `fabric_alloc()` / `fabric_free()`
   - `fabric_prefetch()` / `fabric_wait()`
   - Prefetch thread pool
2. Integrate with llama.cpp

### Phase 3: Advanced Features (1 week)
1. Automatic access pattern detection
2. LITTLE core role assignment
3. UFS swap integration for overflow KV
4. Power management (sleep idle prefetch threads)

---

## The Funky Stuff

### LITTLE Core Fabric Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FULL FABRIC ARCHITECTURE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                         BIG CORES (A78)                               │  │
│   │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│   │  │  Core 6: Inference Thread 1                                     │ │  │
│   │  │  - Process layers 0, 2, 4, ...                                  │ │  │
│   │  │  - Uses CACHED weights (HW prefetch)                            │ │  │
│   │  │  - Uses UNCACHED KV cache (fabric prefetched)                   │ │  │
│   │  └─────────────────────────────────────────────────────────────────┘ │  │
│   │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│   │  │  Core 7: Inference Thread 2                                     │ │  │
│   │  │  - Process layers 1, 3, 5, ...                                  │ │  │
│   │  │  - Parallel with Thread 1                                       │ │  │
│   │  └─────────────────────────────────────────────────────────────────┘ │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│                                    │ Prefetch requests                       │
│                                    ▼                                         │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                       LITTLE CORES (A55)                              │  │
│   │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │  │
│   │  │   Core 0    │ │   Core 1    │ │   Core 2    │ │   Core 3    │    │  │
│   │  │  KV Prefetch│ │  KV Prefetch│ │ Swap Writer │ │ Swap Reader │    │  │
│   │  │  (Layer N+1)│ │  (Layer N+2)│ │ (Evict old) │ │ (Load next) │    │  │
│   │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘    │  │
│   │  ┌─────────────┐ ┌─────────────┐                                     │  │
│   │  │   Core 4    │ │   Core 5    │                                     │  │
│   │  │   Monitor   │ │   Spare     │                                     │  │
│   │  │(Track stats)│ │ (Overflow)  │                                     │  │
│   │  └─────────────┘ └─────────────┘                                     │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                         MEMORY REGIONS                                │  │
│   │  ┌─────────────────────┐  ┌─────────────────────┐                    │  │
│   │  │    CACHED (malloc)  │  │  UNCACHED (dma_heap)│                    │  │
│   │  │                     │  │                     │                    │  │
│   │  │  - Model weights    │  │  - KV cache         │                    │  │
│   │  │  - Activations      │  │  - Embeddings       │                    │  │
│   │  │  - Scratch buffers  │  │  - Overflow region  │                    │  │
│   │  │                     │  │                     │                    │  │
│   │  │  HW prefetch active │  │  Fabric prefetch    │                    │  │
│   │  └─────────────────────┘  └─────────────────────┘                    │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why This is "Funky"

1. **We repurpose LITTLE cores** - Instead of idle or running system tasks, they become dedicated memory management workers.

2. **Uncached is the key** - The entire fabric depends on using uncached memory, which seems counterintuitive but eliminates coherency overhead.

3. **We're not fighting the hardware** - We're using existing paths (cached/uncached) and existing resources (LITTLE cores) in a coordinated way.

4. **Software-defined fabric** - The "fabric" is a software abstraction, not hardware modification. It can be deployed without kernel changes.

---

## Success Criteria

### Phase 1 Success
- [ ] tok/s improves ≥5% with uncached KV cache
- [ ] Latency variance drops ≥50%

### Phase 2 Success
- [ ] Fabric library works standalone
- [ ] llama.cpp integration is clean
- [ ] No memory leaks or crashes

### Phase 3 Success
- [ ] All 6 LITTLE cores utilized
- [ ] Swap integration works for large context
- [ ] Power consumption reasonable

---

## The Wood Cuts Itself

The original question was: *"Can we create a fabric that unifies the paths in a way that eliminates cache coherency overhead?"*

The answer: **Yes. Use uncached memory.**

The insight: The fabric isn't about hardware interconnects. It's about choosing the right memory type for each access pattern and coordinating software threads to warm memory for each other.

The hardware already has everything we need:
- Uncached path via dma_heap
- Multiple cores (BIG + LITTLE)
- Parallel memory bandwidth (18 GB/s combined)

We just needed to understand the grain.

**The wood cuts itself.**
