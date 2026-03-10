# llama.cpp Offload Strategy Detailed Explanation

This document details how llama.cpp applies offload strategies at runtime, distributing different parts of the model to different compute devices (CPU/GPU).

## I. Offload Configuration Parameters

### 1. Model Parameters (`llama_model_params`)

```cpp
struct llama_model_params {
    // Device list
    ggml_backend_dev_t * devices;                                // NULL = use all available devices
    
    // Tensor buffer type overrides
    const llama_model_tensor_buft_override * tensor_buft_overrides;
    
    // GPU layer count control
    int32_t n_gpu_layers;                                        // Number of layers to offload to VRAM, -1 for all
    enum llama_split_mode split_mode;                            // How to split model across multiple GPUs
    
    // GPU split control
    int32_t main_gpu;                                            // Main GPU (used when split_mode=NONE)
    const float * tensor_split;                                  // Proportion for each GPU (by layer or by row)
    
    ...
};
```

### 2. Context Parameters (`llama_context_params`)

```cpp
struct llama_context_params {
    bool offload_kqv;     // Offload KQV operations (including KV cache) to GPU
    bool op_offload;      // Offload host tensor operations to device
    ...
};
```

## II. Three Levels of Offload Strategy

### Layer 1: Device Allocation for Model Weights (at Load Time)

Completed during model loading, determines which device each weight tensor is stored on.

#### 1.1 Device Initialization

```cpp
// src/llama.cpp: llama_model_load_from_file_impl()
// Build list of available devices
for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
    ggml_backend_dev_t dev = ggml_backend_dev_get(i);
    switch (ggml_backend_dev_type(dev)) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:
        case GGML_BACKEND_DEVICE_TYPE_GPU:      // CUDA/Metal/Vulkan etc.
        case GGML_BACKEND_DEVICE_TYPE_GPU_FULL: // Dedicated GPU
        case GGML_BACKEND_DEVICE_TYPE_ACCEL:    // Accelerator
        // Add to model->devices
    }
}
```

#### 1.2 Device Allocation by Layer

```cpp
// src/llama-model.cpp
pimpl->dev_layer.resize(n_layer);
for (uint32_t il = 0; il < n_layer; il++) {
    pimpl->dev_layer[il] = get_layer_buft_list(il);
}

// get_layer_buft_list() decides device based on n_gpu_layers:
//   - il < (n_layer - n_gpu_layers): CPU
//   - il >= (n_layer - n_gpu_layers): GPU
```

**Allocation Strategy**:
- **n_gpu_layers = 0**: All layers on CPU
- **n_gpu_layers = 10**: Last 10 layers on GPU, rest on CPU
- **n_gpu_layers = -1**: All layers on GPU

#### 1.3 Tensor Buffer Type Allocation

```cpp
// src/llama-model-loader.cpp
// Select buffer type for each tensor
auto * buft = get_buft_list(tn.idx, tensor_name);

// Select based on layer ID and tensor type:
if (il < gpu_layer_start) {
    buft = CPU_buffer_type;
} else {
    buft = GPU_buffer_type;
}
```

**Tensor Classification**:
- **Input tensors** (token_embd, pos_embd): Usually on CPU
- **Layer weights** (attn_q, attn_k, ffn_gate etc.): Allocated based on layer ID
- **Output tensors** (output_norm, output): Allocated based on n_gpu_layers

#### 1.4 Actual Weight Loading

```cpp
// src/llama-model-loader.cpp: load_all_data()
// Allocate memory for each buffer type
for (auto & [buft, buffers] : bufs_grouped) {
    buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
}

// Read weight data from file to corresponding buffer
for (auto & tensor : tensors) {
    ggml_backend_tensor_set(tensor, data, 0, size);
}
```

### Layer 2: Device Allocation for KV Cache (at Context Creation)

KV cache allocation strategy is controlled by the `offload_kqv` parameter.

```cpp
// src/llama-kv-cache.cpp: llama_kv_cache constructor
for (uint32_t il = 0; il < n_layer; il++) {
    ggml_backend_buffer_type_t buft;
    
    if (offload) {
        // Get device where this layer's weights are located
        auto * dev = model.dev_layer(il);
        buft = ggml_backend_dev_buffer_type(dev);
    } else {
        // Keep on CPU
        buft = ggml_backend_cpu_buffer_type();
    }
    
    // Create K/V tensors on corresponding device
    ggml_tensor * k = ggml_new_tensor_3d(ctx, type_k, n_embd_k, kv_size, n_stream);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, type_v, n_embd_v, kv_size, n_stream);
}
```

**Allocation Results**:
- `offload_kqv = false`: All KV cache in CPU memory
- `offload_kqv = true`: KV cache follows corresponding layer's weight device
  - Layers 0-9 on CPU → KV cache on CPU
  - Layers 10-31 on GPU → KV cache on GPU VRAM

### Layer 3: Device Allocation for Compute Graph Operations (at Inference Time)

During each inference, Backend Scheduler dynamically allocates each operation to appropriate devices.

#### 3.1 Backend Scheduler Initialization

```cpp
// src/llama-context.cpp: llama_context constructor
// Create backend list
std::vector<ggml_backend_t> backends;
backends.push_back(gpu_backend);   // GPU backend (CUDA/Metal/...)
backends.push_back(cpu_backend);   // CPU backend

// Create scheduler
sched = ggml_backend_sched_new(
    backends.data(),
    buffer_types.data(),
    n_backends,
    max_nodes,
    pipeline_parallel,
    cparams.op_offload     // ← Enable op_offload
);
```

#### 3.2 Compute Graph Construction

```cpp
// src/llama-graph.cpp: llm_build_*()
// Build transformer compute graph
ggml_cgraph * gf = model.build_graph(gparams);

// Example: attention operations
cur = ggml_mul_mat(ctx0, k_cache, q);     // Q·K^T
cur = ggml_soft_max(ctx0, cur);           // softmax
cur = ggml_mul_mat(ctx0, v_cache, cur);   // attn·V
```

#### 3.3 Automatic Backend Assignment (Pass 1-4)

Backend Scheduler automatically assigns execution device for each operation through multiple passes:

```cpp
// ggml/src/ggml-backend.cpp: ggml_backend_sched_split_graph()

// Pass 1: Assign backend based on input tensor location
for (node : graph->nodes) {
    int backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);
    // Rules:
    // - If input in GPU weights buffer → assign to GPU
    // - If operation supports offload → assign to GPU
    // - Otherwise → CPU
}

// Pass 2: Extend backend assignment to neighboring nodes
// Merge operations with same backend to reduce data transfers

// Pass 3: Upgrade nodes to higher priority backend
// If node's inputs already on GPU and GPU supports the operation, upgrade to GPU

// Pass 4: Assign backend for remaining nodes and handle data copying
for (node : unassigned_nodes) {
    // Insert necessary data copy operations
    if (input_backend != node_backend) {
        insert_copy_node(input, node_backend);
    }
}
```

**Assignment Rule Examples**:

| Operation | Input Location | Supported Backends | Final Assignment |
|-----------|----------------|-------------------|------------------|
| `token_embd` (GET_ROWS) | Weights:CPU | CPU, GPU | CPU (input on CPU) |
| `attn_q` (MUL_MAT) | Weights:GPU | CPU, GPU | GPU (weights on GPU) |
| `rope` | Input:GPU | CPU, GPU | GPU (input on GPU) |
| `soft_max` | Input:GPU | CPU, GPU | GPU (input on GPU) |
| `add` | Input:GPU | CPU, GPU | GPU (input on GPU) |

#### 3.4 Op Offload Mechanism

The `op_offload` parameter enables more aggressive offload strategy:

```cpp
// ggml/src/ggml-backend.cpp
if (op_offload && ggml_backend_offload_op(gpu_backend, node)) {
    // Even if weights on CPU, if operation is expensive (e.g., large matrix multiplication)
    // offload it to GPU for execution
    backend_id = gpu_backend_id;
}
```

**Applicable Operations**:
- `MUL_MAT`, `MUL_MAT_ID`: Large matrix multiplications
- `CONV_TRANSPOSE_1D`: Convolution operations
- Compute-intensive operations

#### 3.5 Split Graph into Subgraphs

Scheduler splits compute graph into multiple subgraphs, each executed on a single backend:

```cpp
// ggml/src/ggml-backend.cpp: ggml_backend_sched_split_graph()
// Split rules:
// - Create new split when backend switches
// - Insert cross-backend data copy nodes

splits[0]: backend=GPU, nodes=[0-15]     // GPU operations
  copy: GPU→CPU                          // Data transfer
splits[1]: backend=CPU, nodes=[16-20]    // CPU operations
  copy: CPU→GPU                          // Data transfer
splits[2]: backend=GPU, nodes=[21-50]    // GPU operations
```

#### 3.6 Execute Computation

```cpp
// src/llama-context.cpp: graph_compute()
ggml_backend_sched_graph_compute_async(sched, gf);

// Internal flow:
// 1. 为每个 split allocation compute buffer
// 2. 复制输入数据到目标 backend
// 3. 在对应 backend 上执行子图
// 4. 复制输出数据（如果需要）
```

## III.  in Complete Inference Flow Offload  Application

```
┌─────────────────────────────────────────────────────────────────┐
│                    1. 模型loading阶段                                │
│  Based on n_gpu_layers allocationweights到 CPU/GPU                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Layer 0-19   │  │ Layer 20-31  │  │ Output Layer │          │
│  │   Weights    │  │   Weights    │  │   Weights    │          │
│  │   [CPU]      │  │   [GPU]      │  │   [GPU]      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                  2. Context Creation Phase                                 │
│  Based on offload_kqv allocation KV Cache                                  │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │ KV Cache     │  │ KV Cache     │                             │
│  │ Layer 0-19   │  │ Layer 20-31  │                             │
│  │   [CPU]      │  │   [GPU]      │                             │
│  └──────────────┘  └──────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                   3. Inference Execution Phase                                 │
│  Backend Scheduler 动态allocationoperations                                  │
│                                                                   │
│  Input Tokens [CPU]                                              │
│       ↓                                                           │
│  Token Embedding (GET_ROWS) [CPU]  ← Weights on CPU             │
│       ↓                                                           │
│  Position Embedding [CPU]                                        │
│       ↓ [copy CPU→GPU]                                           │
│  ┌─────────────────────────────────────┐                        │
│  │     Layer 0-19 (CPU Weights)        │                        │
│  │  · Attention Q/K/V [GPU] ← op_offload enabled                   │
│  │  · Rope [GPU]                         │                        │
│  │  · Attention [CPU] ← KV cache on CPU │                        │
│  │  · FFN [GPU] ← op_offload enabled        │                        │
│  └─────────────────────────────────────┘                        │
│       ↓                                                           │
│  ┌─────────────────────────────────────┐                        │
│  │     Layer 20-31 (GPU Weights)       │                        │
│  │  · Attention Q/K/V [GPU]             │                        │
│  │  · Rope [GPU]                         │                        │
│  │  · Attention [GPU] ← KV cache on GPU │                        │
│  │  · FFN [GPU]                          │                        │
│  └─────────────────────────────────────┘                        │
│       ↓                                                           │
│  Output Norm [GPU]                                               │
│       ↓                                                           │
│  LM Head (MUL_MAT) [GPU]                                         │
│       ↓ [copy GPU→CPU]                                           │
│  Logits [CPU]                                                    │
└─────────────────────────────────────────────────────────────────┘
```

## IV. Key Data Flow

### 示例：n_gpu_layers=20, offload_kqv=true

```
Token → [CPU]
  ↓
Token Embedding → [CPU] (weights on CPU)
  ↓ [copy→GPU]
Layer 0 Q/K/V Proj → [GPU] (op_offload)
  ↓ [copy→CPU]
Layer 0 Attention → [CPU] (KV cache on CPU)
  ↓ [copy→GPU]
Layer 0 FFN → [GPU] (op_offload)
  ↓ [copy→CPU]
...
Layer 19 → [CPU/GPU混合]
  ↓ [copy→GPU]
Layer 20 Q/K/V Proj → [GPU] (weights on GPU)
  ↓
Layer 20 Attention → [GPU] (KV cache on GPU)
  ↓
Layer 20 FFN → [GPU] (weights on GPU)
  ↓
...
Layer 31 → [GPU]
  ↓
Output → [GPU]
  ↓ [copy→CPU]
Logits → [CPU]
```

## V. Performance Optimization Recommendations

### 1. Reduce Data Transfer

**Problem**: CPU-GPU 频繁数据传输成为瓶颈

**Optimization**:
- 设置 `n_gpu_layers` 为连续的layer（避免交错）
- 如果内存充足，设置 `offload_kqv=true`
- 启用 `op_offload` 让更多operationson GPU 执行

### 2. Balance Memory Usage

**VRAM  is Insufficient**:
```cpp
// Reduce GPU layer数
params.n_gpu_layers = 10;           // 只offload 10 layer
params.offload_kqv = false;          // KV cache 留on CPU
```

**VRAM  is Sufficient**:
```cpp
// Maximize GPU usage
params.n_gpu_layers = -1;            // 所有layer到 GPU
params.offload_kqv = true;           // KV cache also to GPU
params.op_offload = true;            // Aggressiveoperationsoffload
```

### 3. Multi-GPU Split

**Row Split** (`split_mode = LLAMA_SPLIT_MODE_ROW`):
```cpp
// Split large matrices by row
params.split_mode = LLAMA_SPLIT_MODE_ROW;
params.tensor_split = {0.6, 0.4};    // GPU0:60%, GPU1:40%
```

**Layer Split** (`split_mode = LLAMA_SPLIT_MODE_LAYER`):
```cpp
// 按layer分割
params.split_mode = LLAMA_SPLIT_MODE_LAYER;
// Layer 0-15 → GPU0, Layer 16-31 → GPU1
```

## VI. Debugging Offload  Behavior

### View Allocation Results

```cpp
// 1. Enable logging
setenv("LLAMA_LOG_LEVEL", "DEBUG", 1);

// 2. 查看layerallocation
for (int il = 0; il < n_layer; il++) {
    auto * dev = model.dev_layer(il);
    printf("Layer %d: %s\n", il, ggml_backend_dev_name(dev));
}

// 3. 查看 backend scheduler  splits
int n_splits = ggml_backend_sched_get_n_splits(sched);
printf("Graph splits: %d\n", n_splits);
```

### GDB  Breakpoint Recommendations

```gdb
# weightsloading
break llama_model_loader::load_all_data

# 设备allocation
break llama_model::dev_layer

# Backend allocation
break ggml_backend_sched_split_graph
break ggml_backend_sched_backend_id_from_cur

# 数据传输
break ggml_backend_tensor_copy
```

## VII. Summary

llama.cpp 的 offload 策略是一个**三layermechanism**：

1. **静态layer（模型loading）**: Based on `n_gpu_layers` Determinesweights存储 location
2. **准静态layer（上下文创建）**: Based on `offload_kqv` Determines KV cache  location
3. **动态layer（推理执行）**: Backend Scheduler Based on数据 location and operations类型动态allocation

This design balances **flexibility**  and  **performance** ：
- Users control overall strategy through simple parameters
- 系统自动Optimization具体执行细节
- Supports CPU/GPU/多GPU  and other complex configurations
