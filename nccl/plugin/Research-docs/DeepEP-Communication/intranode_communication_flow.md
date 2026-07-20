# DeepEP Intranode Communication Flow

This document provides a visual representation of the intranode communication flow in DeepEP, showing how tokens are dispatched and combined across multiple GPUs within a single node using NVLink.

## Overview

Intranode communication handles expert-parallel (EP) communication between GPUs within the same node, leveraging high-bandwidth NVLink connections for efficient token dispatch and combination.

## Data Flow Architecture

```mermaid
graph TD
    A[test_intranode.py] --> B["buffer.py dispatch()"]
    B --> C{num_rdma_ranks > 1?}
    C -->|No| D[Intranode Path]
    C -->|Yes| E[Internode Path]
    
    D --> F["deep_ep.cpp intranode_dispatch()"]
    F --> G{handle exists?}
    G -->|Yes| H[Cached Mode]
    G -->|No| I[Fresh Mode]
    
    H --> J["intranode.cu cached_notify_dispatch()"]
    I --> K["intranode.cu notify_dispatch()"]
    I --> L[CPU Synchronization]
    
    J --> M["intranode.cu dispatch kernel"]
    K --> M
    L --> M
    
    M --> N[Output Tensors]
```

## Detailed Communication Flow

```mermaid
sequenceDiagram
    participant P as Python Test
    participant B as Buffer.py
    participant C as deep_ep.cpp
    participant K as intranode.cu
    participant NVL as NVLink
    
    P->>B: dispatch(x, topk_idx, ...)
    B->>C: intranode_dispatch()
    C->>K: notify_dispatch() - Layout calculation
    K->>NVL: Barrier synchronization
    K->>NVL: Token counting & prefix sums
    K->>C: Layout metadata
    C->>K: dispatch() - Main communication
    K->>NVL: Send tokens via channels
    K->>NVL: Receive tokens via channels
    K->>C: Received tensors
    C->>B: Output tensors
    B->>P: recv_x, recv_topk_idx, handle
```

## Kernel Architecture

```mermaid
graph LR
    subgraph "GPU Kernel Grid"
        subgraph "Even Blocks (Senders)"
            S1[SM 0<br/>Channel 0<br/>Sender]
            S2[SM 2<br/>Channel 1<br/>Sender]
            S3[SM 4<br/>Channel 2<br/>Sender]
        end
        
        subgraph "Odd Blocks (Receivers)"
            R1[SM 1<br/>Channel 0<br/>Receiver]
            R2[SM 3<br/>Channel 1<br/>Receiver]
            R3[SM 5<br/>Channel 2<br/>Receiver]
        end
    end
    
    subgraph "Shared Memory Layout"
        SM[Buffer 0<br/>Buffer 1<br/>Buffer 2<br/>...<br/>Buffer N]
    end
    
    S1 --> SM
    S2 --> SM
    S3 --> SM
    SM --> R1
    SM --> R2
    SM --> R3
```

## Memory Layout Structure

```mermaid
graph TD
    subgraph "Per-Rank Buffer Layout"
        A[rank_prefix_matrix<br/>ranks × ranks × int] --> B[expert_prefix_matrix<br/>ranks × local_experts × int]
        B --> C[channel_start_offset<br/>channels × ranks × int]
        C --> D[channel_end_offset<br/>channels × ranks × int]
        D --> E[channel_head_idx<br/>channels × ranks × int]
        E --> F[channel_tail_idx<br/>channels × ranks × int]
        F --> G[channel_x_buffers<br/>channels × ranks × tokens × hidden × int4]
        G --> H[channel_src_idx_buffers<br/>channels × ranks × tokens × int]
        H --> I[channel_topk_idx_buffers<br/>channels × ranks × tokens × topk × int64]
        I --> J[channel_topk_weights_buffers<br/>channels × ranks × tokens × topk × float]
        J --> K[channel_x_scales_buffers<br/>channels × ranks × tokens × scales × float]
    end
```

## Channel-Based Communication Pattern

```mermaid
graph TB
    subgraph "Channel 0"
        S0[Sender Block 0] --> Q0[Queue 0] --> R0[Receiver Block 1]
    end
    
    subgraph "Channel 1"
        S2[Sender Block 2] --> Q1[Queue 1] --> R2[Receiver Block 3]
    end
    
    subgraph "Channel 2"
        S4[Sender Block 4] --> Q2[Queue 2] --> R4[Receiver Block 5]
    end
    
    subgraph "Shared Buffers"
        Q0
        Q1
        Q2
    end
```

## Data Transformation Pipeline

```mermaid
graph LR
    subgraph "Input"
        A[x: num_tokens × hidden<br/>BF16/FP8]
        B[topk_idx: num_tokens × num_topk<br/>int64]
        C[topk_weights: num_tokens × num_topk<br/>float32]
    end
    
    subgraph "Processing"
        D[Token Counting]
        E[Channel Distribution]
        F[Queue Management]
    end
    
    subgraph "Output"
        G[recv_x: num_recv_tokens × hidden<br/>BF16/FP8]
        H[recv_topk_idx: num_recv_tokens × num_topk<br/>int64]
        I[recv_topk_weights: num_recv_tokens × num_topk<br/>float32]
        J[recv_src_idx: num_recv_tokens<br/>int32]
    end
    
    A --> D
    B --> D
    C --> D
    D --> E
    E --> F
    F --> G
    F --> H
    F --> I
    F --> J
```

## Performance Optimizations

```mermaid
graph TD
    subgraph "SM90 Features"
        A[TMA - Tensor Memory Accelerator]
        B[FP8 Support]
        C[Async Memory Operations]
    end
    
    subgraph "Parallelism"
        D[Channel-based Parallelism<br/>num_sms / 2 channels]
        E[Even/Odd Block Split<br/>Half send, half receive]
        F[Work Distribution<br/>Tokens across channels]
    end
    
    subgraph "Memory Access"
        G[Coalesced Access<br/>Warp-level operations]
        H[Shared Memory<br/>Temporary storage]
        I[Non-blocking Queues<br/>Head/tail pointers]
    end
```

## Key Functions and Their Roles

| Function | Location | Purpose |
|----------|----------|---------|
| `dispatch()` | buffer.py | Python interface, routes to intranode |
| `intranode_dispatch()` | deep_ep.cpp | C++ interface, handles tensor allocation |
| `notify_dispatch()` | intranode.cu | Layout calculation and barrier sync |
| `dispatch()` | intranode.cu | Main communication kernel |
| `get_dispatch_layout()` | layout.cu | Token counting and distribution |

## Communication Phases

1. **Layout Phase**: Calculate token distribution and routing information
2. **Barrier Phase**: Synchronize all ranks before communication
3. **Dispatch Phase**: Send tokens through NVLink channels
4. **Receive Phase**: Collect tokens and update output tensors
5. **Cleanup Phase**: Prepare buffers for next iteration
