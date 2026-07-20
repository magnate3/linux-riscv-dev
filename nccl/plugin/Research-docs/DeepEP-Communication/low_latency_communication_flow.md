# DeepEP Low Latency Communication Flow

This document provides a visual representation of the low latency communication flow in DeepEP, showing how tokens are dispatched and combined across multiple GPUs with minimal latency using IBGDA (InfiniBand GPU Direct Async) and specialized buffer management.

## Overview

Low latency communication is designed for scenarios requiring minimal communication latency, such as online inference or real-time applications. It uses a dual-buffer architecture with IBGDA for direct GPU-to-GPU RDMA communication, bypassing CPU involvement and reducing latency to microseconds.

## Key Features

- **IBGDA Communication**: Direct GPU-to-GPU RDMA without CPU involvement
- **Dual Buffer Architecture**: Ping-pong buffers for continuous operation
- **Hook-based Overlapping**: Separate send and receive phases for computation overlap
- **FP8 Support**: 8-bit floating point for reduced bandwidth
- **LogFMT Format**: Specialized format for weight reduction
- **Statistics Collection**: Built-in performance monitoring

## Data Flow Architecture

```mermaid
graph TD
    A[test_low_latency.py] --> B["buffer.py low_latency_dispatch()"]
    B --> C["deep_ep.cpp low_latency_dispatch()"]
    C --> D{return_recv_hook?}
    D -->|No| E[Direct Execution]
    D -->|Yes| F[Hook-based Execution]
    
    E --> G["internode_ll.cu dispatch() - Both Phases"]
    F --> H["internode_ll.cu dispatch() - Send Phase"]
    F --> I[Hook Function - Receive Phase]
    
    G --> J[Output Tensors]
    H --> K[Send Hook]
    I --> J
    K --> I
    
    J --> L["buffer.py low_latency_combine()"]
    L --> M["internode_ll.cu combine() - Both Phases"]
    M --> N[Combined Output]
```

## Dual Buffer Architecture

```mermaid
graph TB
    subgraph "Low Latency Buffer Layout"
        subgraph "Buffer 0 (Even)"
            A1[Dispatch RDMA Send Buffer]
            A2[Dispatch RDMA Recv Data Buffer]
            A3[Dispatch RDMA Recv Count Buffer]
            A4[Combine RDMA Send Buffer]
            A5[Combine RDMA Recv Data Buffer]
            A6[Combine RDMA Recv Flag Buffer]
        end
        
        subgraph "Buffer 1 (Odd)"
            B1[Dispatch RDMA Send Buffer]
            B2[Dispatch RDMA Recv Data Buffer]
            B3[Dispatch RDMA Recv Count Buffer]
            B4[Combine RDMA Send Buffer]
            B5[Combine RDMA Recv Data Buffer]
            B6[Combine RDMA Recv Flag Buffer]
        end
    end
    
    subgraph "Buffer Management"
        C[low_latency_buffer_idx<br/>Alternates between 0 and 1]
        D[Buffer Cleanup<br/>Zero-initialize next buffer]
        E[Atomic Operations<br/>Thread-safe buffer switching]
    end
    
    C --> A1
    C --> B1
    D --> A1
    D --> B1
    E --> C
```

## Detailed Communication Sequence

```mermaid
sequenceDiagram
    participant P as Python Test
    participant B as Buffer.py
    participant C as deep_ep.cpp
    participant K as internode_ll.cu
    participant RDMA as IBGDA RDMA
    participant H as Hook Function
    
    P->>B: low_latency_dispatch(x, topk_idx, ...)
    B->>C: low_latency_dispatch()
    C->>C: Buffer layout calculation
    C->>K: dispatch() kernel launch
    
    alt Direct Execution (async_finish=False)
        K->>RDMA: Send tokens via IBGDA
        K->>RDMA: Receive tokens via IBGDA
        K->>C: Packed tensors ready
        C->>B: Output tensors
        B->>P: recv_x, recv_count, handle, event
    else Hook-based Execution (return_recv_hook=True)
        K->>RDMA: Send tokens via IBGDA
        K->>C: Send phase complete
        C->>B: Hook function returned
        B->>P: recv_x, recv_count, handle, event, hook
        P->>H: hook() - Execute receive phase
        H->>K: dispatch() - Receive phase only
        K->>RDMA: Receive tokens via IBGDA
        K->>H: Receive complete
        H->>P: Receive phase finished
    end
    
    P->>B: low_latency_combine(x, topk_idx, topk_weights, handle, ...)
    B->>C: low_latency_combine()
    C->>K: combine() kernel launch
    K->>RDMA: Send combined tokens via IBGDA
    K->>RDMA: Receive combined tokens via IBGDA
    K->>C: Combined output ready
    C->>B: combined_x, event
    B->>P: Final combined result
```

## Message Format and Layout

```mermaid
graph TB
    subgraph "Dispatch Message Format"
        A[Source Index: int4<br/>4 bytes] --> B[Reserved Fields: int4 × 3<br/>12 bytes]
        B --> C[Token Data: BF16/FP8<br/>hidden × element_size]
        C --> D[FP8 Scales: float<br/>hidden/128 × 4 bytes]
    end
    
    subgraph "Combine Message Format"
        E[LogFMT Meta: nv_bfloat162<br/>hidden/128 × 4 bytes] --> F[Token Data: BF16<br/>hidden × 2 bytes]
    end
    
    subgraph "Buffer Layout Per Expert"
        G[Expert 0 Buffer<br/>num_max_tokens × msg_size] --> H[Expert 1 Buffer<br/>num_max_tokens × msg_size]
        H --> I[Expert 2 Buffer<br/>num_max_tokens × msg_size]
        I --> J[Expert N Buffer<br/>num_max_tokens × msg_size]
    end
```

## Kernel Warp Organization

```mermaid
graph TB
    subgraph "SM Block Organization"
        subgraph "Warp Groups"
            W1[Warp Group 0<br/>Expert 0-7]
            W2[Warp Group 1<br/>Expert 8-15]
            W3[Warp Group 2<br/>Expert 16-23]
            W4[Warp Group N<br/>Expert N-N+7]
        end
        
        subgraph "Warp Roles per Group"
            R1[Sub-warp 0: Send Phase]
            R2[Sub-warp 1: Receive Phase]
            R3[Sub-warp 2: Buffer Management]
            R4[Sub-warp 3: Statistics Collection]
        end
    end
    
    subgraph "Expert Responsibility"
        E1[Expert 0: Warp Group 0, Sub-warp 0]
        E2[Expert 1: Warp Group 0, Sub-warp 1]
        E3[Expert 2: Warp Group 0, Sub-warp 2]
        E4[Expert 8: Warp Group 1, Sub-warp 0]
    end
```

## IBGDA Communication Pattern

```mermaid
graph TB
    subgraph "GPU 0"
        G0[Expert 0-7<br/>Local Experts]
    end
    
    subgraph "GPU 1"
        G1[Expert 8-15<br/>Local Experts]
    end
    
    subgraph "GPU 2"
        G2[Expert 16-23<br/>Local Experts]
    end
    
    subgraph "GPU N"
        GN[Expert N-N+7<br/>Local Experts]
    end
    
    G0 ==>|IBGDA Put| G1
    G0 ==>|IBGDA Put| G2
    G0 ==>|IBGDA Put| GN
    
    G1 ==>|IBGDA Put| G0
    G1 ==>|IBGDA Put| G2
    G1 ==>|IBGDA Put| GN
    
    G2 ==>|IBGDA Put| G0
    G2 ==>|IBGDA Put| G1
    G2 ==>|IBGDA Put| GN
    
    GN ==>|IBGDA Put| G0
    GN ==>|IBGDA Put| G1
    GN ==>|IBGDA Put| G2
```

## Hook-based Overlapping

```mermaid
graph LR
    subgraph "Traditional Execution"
        A1[Send Phase] --> A2[Receive Phase] --> A3[Combine Phase]
    end
    
    subgraph "Hook-based Execution"
        B1[Send Phase] --> B2[Compute Phase]
        B2 --> B3[Hook: Receive Phase]
        B3 --> B4[Combine Phase]
    end
    
    subgraph "Timeline"
        C1[Time 0: Send starts]
        C2[Time 1: Send completes, Compute starts]
        C3[Time 2: Hook called, Receive starts]
        C4[Time 3: Receive completes, Combine starts]
        C5[Time 4: Combine completes]
    end
```

## Data Transformation Pipeline

```mermaid
graph LR
    subgraph "Input Processing"
        A[x: num_tokens × hidden<br/>BF16]
        B[topk_idx: num_tokens × num_topk<br/>int64]
        C[FP8 Casting<br/>Optional]
    end
    
    subgraph "Dispatch Phase"
        D[Expert Routing<br/>Calculate target experts]
        E[Message Packing<br/>Add metadata + scales]
        F[IBGDA Send<br/>Direct GPU-to-GPU]
        G[IBGDA Receive<br/>Collect from all ranks]
        H[Message Unpacking<br/>Extract tokens + metadata]
    end
    
    subgraph "Compute Phase"
        I[Expert Computation<br/>Simulated GEMM]
        J[Hook Execution<br/>Receive phase trigger]
    end
    
    subgraph "Combine Phase"
        K[Weight Application<br/>Apply expert weights]
        L[Message Packing<br/>Combine format]
        M[IBGDA Send<br/>Send combined results]
        N[IBGDA Receive<br/>Collect from all ranks]
        O[Reduction<br/>Sum weighted results]
    end
    
    subgraph "Output"
        P[combined_x: num_tokens × hidden<br/>BF16]
    end
    
    A --> D
    B --> D
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
    N --> O
    O --> P
```

## Performance Optimizations

```mermaid
graph TD
    subgraph "IBGDA Features"
        A[Hardware RDMA<br/>Direct GPU-to-GPU]
        B[Multiple QPs<br/>Parallel channels]
        C[Async Operations<br/>Non-blocking RDMA]
        D[Zero-copy<br/>Direct memory access]
    end
    
    subgraph "Buffer Management"
        E[Dual Buffers<br/>Ping-pong operation]
        F[Atomic Operations<br/>Lock-free switching]
        G[Buffer Cleanup<br/>Zero-initialization]
        H[Memory Alignment<br/>Cache-friendly access]
    end
    
    subgraph "Computation Overlap"
        I[Hook Functions<br/>Separate send/receive]
        J[Async Execution<br/>Non-blocking operations]
        K[Statistics Collection<br/>Performance monitoring]
        L[FP8 Support<br/>Reduced bandwidth]
    end
```

## Statistics and Monitoring

```mermaid
graph TB
    subgraph "Dispatch Statistics"
        A[cumulative_local_expert_recv_stats<br/>Expert-level token counts]
        B[dispatch_wait_recv_cost_stats<br/>Wait time per rank]
    end
    
    subgraph "Combine Statistics"
        C[combine_wait_recv_cost_stats<br/>Wait time per rank]
    end
    
    subgraph "Performance Metrics"
        D[Bandwidth Measurement<br/>GB/s calculation]
        E[Latency Measurement<br/>Microsecond timing]
        F[Throughput Analysis<br/>Tokens per second]
    end
    
    A --> D
    B --> E
    C --> E
    D --> F
    E --> F
```

## Key Functions and Their Roles

| Function | Location | Purpose |
|----------|----------|---------|
| `low_latency_dispatch()` | buffer.py | Python interface for low latency dispatch |
| `low_latency_dispatch()` | deep_ep.cpp | C++ interface, buffer management |
| `dispatch()` | internode_ll.cu | Main low latency dispatch kernel |
| `low_latency_combine()` | buffer.py | Python interface for low latency combine |
| `low_latency_combine()` | deep_ep.cpp | C++ interface, buffer management |
| `combine()` | internode_ll.cu | Main low latency combine kernel |
| `clean_low_latency_buffer()` | buffer.py | Buffer cleanup between operations |
| `get_next_low_latency_combine_buffer()` | buffer.py | Zero-copy buffer access |

## Communication Phases

1. **Buffer Setup Phase**: Initialize dual buffers and calculate layout
2. **Dispatch Send Phase**: Send tokens via IBGDA to target experts
3. **Dispatch Receive Phase**: Receive tokens from all ranks (via hook or direct)
4. **Compute Phase**: Process tokens through expert networks (simulated)
5. **Combine Send Phase**: Send weighted results via IBGDA
6. **Combine Receive Phase**: Receive and reduce weighted results
7. **Buffer Cleanup Phase**: Prepare buffers for next iteration

## Configuration Parameters

```mermaid
graph TD
    subgraph "Buffer Configuration"
        A[num_max_dispatch_tokens_per_rank<br/>Maximum tokens per rank]
        B[num_rdma_bytes<br/>RDMA buffer size]
        C[num_qps_per_rank<br/>Queue pairs per rank]
    end
    
    subgraph "Performance Configuration"
        D[use_fp8<br/>Enable FP8 casting]
        E[round_scale<br/>Round FP8 scales to power of 2]
        F[use_ue8m0<br/>Use UE8M0 scale format]
        G[use_logfmt<br/>Use LogFMT for combine]
    end
    
    subgraph "Execution Configuration"
        H[return_recv_hook<br/>Enable hook-based execution]
        I[async_finish<br/>Async completion]
        J[zero_copy<br/>Zero-copy buffer access]
    end
```

## Memory Layout Structure

```mermaid
graph TD
    subgraph "LowLatencyLayout"
        A["total_bytes<br/>Total buffer size"]
        B["buffers[2]<br/>Dual buffer array"]
    end
    
    subgraph "LowLatencyBuffer"
        C["num_clean_int<br/>Cleanup metadata count"]
        D["dispatch_rdma_send_buffer<br/>Dispatch send buffer"]
        E["dispatch_rdma_recv_data_buffer<br/>Dispatch receive data buffer"]
        F["dispatch_rdma_recv_count_buffer<br/>Dispatch receive count buffer"]
        G["combine_rdma_send_buffer<br/>Combine send buffer"]
        H["combine_rdma_recv_data_buffer<br/>Combine receive data buffer"]
        I["combine_rdma_recv_flag_buffer<br/>Combine receive flag buffer"]
    end
    
    A --> B
    B --> C
    B --> D
    B --> E
    B --> F
    B --> G
    B --> H
    B --> I
```

This low latency communication system enables DeepEP to achieve microsecond-level communication latency, making it suitable for real-time applications and online inference scenarios where traditional high-throughput communication would introduce unacceptable delays.
