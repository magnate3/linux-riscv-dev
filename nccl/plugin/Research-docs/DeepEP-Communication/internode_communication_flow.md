# DeepEP Internode Communication Flow

This document provides a visual representation of the internode communication flow in DeepEP, showing how tokens are dispatched and combined across multiple nodes using both RDMA and NVLink.

## Overview

Internode communication handles expert-parallel (EP) communication between GPUs across multiple nodes, using a dual-level architecture that combines RDMA for inter-node communication and NVLink for intra-node communication.

## Data Flow Architecture

```mermaid
graph TD
    A[test_internode.py] --> B["buffer.py dispatch()"]
    B --> C{num_rdma_ranks > 1?}
    C -->|Yes| D[Internode Path]
    C -->|No| E[Intranode Path]
    
    D --> F["buffer.py internode_dispatch()"]
    F --> G["deep_ep.cpp internode_dispatch()"]
    G --> H{handle exists?}
    H -->|Yes| I[Cached Mode]
    H -->|No| J[Fresh Mode]
    
    I --> K["internode.cu cached_notify()"]
    J --> L["internode.cu notify_dispatch()"]
    J --> M[CPU Synchronization]
    
    K --> N["internode.cu dispatch kernel"]
    L --> N
    M --> N
    
    N --> O[Output Tensors]
```

## Dual-Level Communication Architecture

```mermaid
graph TB
    subgraph "Node 0"
        N0G0[GPU 0] 
        N0G1[GPU 1]
        N0G2[GPU 2]
        N0G3[GPU 3]
        N0G4[GPU 4]
        N0G5[GPU 5]
        N0G6[GPU 6]
        N0G7[GPU 7]
    end
    
    subgraph "Node 1"
        N1G0[GPU 0]
        N1G1[GPU 1]
        N1G2[GPU 2]
        N1G3[GPU 3]
        N1G4[GPU 4]
        N1G5[GPU 5]
        N1G6[GPU 6]
        N1G7[GPU 7]
    end
    
    subgraph "Node N"
        NNG0[GPU 0]
        NNG1[GPU 1]
        NNG2[GPU 2]
        NNG3[GPU 3]
        NNG4[GPU 4]
        NNG5[GPU 5]
        NNG6[GPU 6]
        NNG7[GPU 7]
    end
    
    N0G0 -.->|NVLink| N0G1
    N0G1 -.->|NVLink| N0G2
    N0G2 -.->|NVLink| N0G3
    N0G4 -.->|NVLink| N0G5
    N0G5 -.->|NVLink| N0G6
    N0G6 -.->|NVLink| N0G7
    
    N1G0 -.->|NVLink| N1G1
    N1G1 -.->|NVLink| N1G2
    N1G2 -.->|NVLink| N1G3
    N1G4 -.->|NVLink| N1G5
    N1G5 -.->|NVLink| N1G6
    N1G6 -.->|NVLink| N1G7
    
    N0G0 ==>|RDMA| N1G0
    N0G1 ==>|RDMA| N1G1
    N0G2 ==>|RDMA| N1G2
    N0G3 ==>|RDMA| N1G3
    N0G4 ==>|RDMA| N1G4
    N0G5 ==>|RDMA| N1G5
    N0G6 ==>|RDMA| N1G6
    N0G7 ==>|RDMA| N1G7
    
    N0G0 ==>|RDMA| NNG0
    N1G0 ==>|RDMA| NNG0
```

## Detailed Communication Sequence

```mermaid
sequenceDiagram
    participant P as Python Test
    participant B as Buffer.py
    participant C as deep_ep.cpp
    participant K as internode.cu
    participant RDMA as RDMA/NVSHMEM
    participant NVL as NVLink
    
    P->>B: dispatch(x, topk_idx, num_tokens_per_rdma_rank, ...)
    B->>C: internode_dispatch()
    C->>K: notify_dispatch() - Dual-level layout
    K->>RDMA: RDMA barrier & size exchange
    K->>NVL: NVL barrier & size exchange
    K->>K: Calculate dual-level prefix sums
    K->>C: Layout metadata (RDMA + NVL)
    C->>K: dispatch() - Main communication
    K->>RDMA: Send tokens via RDMA channels
    K->>RDMA: Forward RDMA→NVL within nodes
    K->>NVL: Receive tokens via NVL channels
    K->>C: Received tensors
    C->>B: Output tensors
    B->>P: recv_x, recv_topk_idx, handle
```

## Kernel Warp Role Architecture

```mermaid
graph TB
    subgraph "SM Block (Forwarder)"
        subgraph "Warp Roles"
            W1[Warp 0: RDMA→NVL Forwarder<br/>Target Rank 0]
            W2[Warp 1: RDMA→NVL Forwarder<br/>Target Rank 1]
            W3[Warp 2: RDMA→NVL Forwarder<br/>Target Rank 2]
            W4[Warp 3: RDMA→NVL Forwarder<br/>Target Rank 3]
            W5[Warp 4: RDMA→NVL Forwarder<br/>Target Rank 4]
            W6[Warp 5: RDMA→NVL Forwarder<br/>Target Rank 5]
            W7[Warp 6: RDMA→NVL Forwarder<br/>Target Rank 6]
            W8[Warp 7: RDMA→NVL Forwarder<br/>Target Rank 7]
            W9[Warp 8: Forwarder Coordinator]
        end
    end
    
    subgraph "SM Block (RDMA Sender)"
        subgraph "Warp Roles"
            R1[Warp 0: RDMA Sender]
            R2[Warp 1: RDMA Sender]
            R3[Warp 2: RDMA Sender]
            R4[Warp 3: RDMA Sender]
            R5[Warp 4: RDMA Sender]
            R6[Warp 5: RDMA Sender]
            R7[Warp 6: RDMA Sender]
            R8[Warp 7: RDMA Sender Coordinator]
        end
    end
    
    subgraph "SM Block (NVL Receiver)"
        subgraph "Warp Roles"
            N1[Warp 0: NVL Receiver<br/>Target Rank 0]
            N2[Warp 1: NVL Receiver<br/>Target Rank 1]
            N3[Warp 2: NVL Receiver<br/>Target Rank 2]
            N4[Warp 3: NVL Receiver<br/>Target Rank 3]
            N5[Warp 4: NVL Receiver<br/>Target Rank 4]
            N6[Warp 5: NVL Receiver<br/>Target Rank 5]
            N7[Warp 6: NVL Receiver<br/>Target Rank 6]
            N8[Warp 7: NVL Receiver<br/>Target Rank 7]
        end
    end
```

## Dual-Level Buffer Layout

```mermaid
graph TD
    subgraph "RDMA Buffer (Shared via NVSHMEM)"
        A["rdma_recv_num_tokens_mixed<br/>rdma_ranks x (nvl_ranks + experts + 1) x int"] --> B["rdma_channel_data<br/>channels x rdma_ranks x tokens x bytes_per_token x uint8"]
        B --> C["rdma_channel_metadata<br/>channels x rdma_ranks x metadata x int"]
    end
    
    subgraph "NVL Buffer (Per Node via IPC)"
        D["nvl_reduced_num_tokens_per_expert<br/>experts x int"] --> E["nvl_send_num_tokens_per_rank<br/>rdma_ranks x nvl_ranks x int"]
        E --> F["nvl_channel_data<br/>channels x nvl_ranks x tokens x hidden x int4"]
        F --> G["nvl_channel_metadata<br/>channels x nvl_ranks x metadata x int"]
    end
```

## SourceMeta Structure

```mermaid
graph LR
    subgraph "SourceMeta (8 bytes)"
        A["src_rdma_rank: int<br/>Source RDMA rank (node)"]
        B["is_token_in_nvl_rank_bits: int<br/>Bitmask for NVL ranks within source node"]
    end
    
    subgraph "Bitmask Example"
        C["Bit 0: NVL Rank 0"]
        D["Bit 1: NVL Rank 1"]
        E["Bit 2: NVL Rank 2"]
        F["Bit 3: NVL Rank 3"]
        G["Bit 4: NVL Rank 4"]
        H["Bit 5: NVL Rank 5"]
        I["Bit 6: NVL Rank 6"]
        J["Bit 7: NVL Rank 7"]
    end
    
    B --> C
    B --> D
    B --> E
    B --> F
    B --> G
    B --> H
    B --> I
    B --> J
```

## Communication Pipeline

```mermaid
graph LR
    subgraph "Input Data"
        A[x: num_tokens × hidden<br/>BF16/FP8]
        B[topk_idx: num_tokens × num_topk<br/>int64]
        C[num_tokens_per_rdma_rank: num_nodes<br/>int32]
    end
    
    subgraph "RDMA Level"
        D[RDMA Token Counting]
        E[RDMA Channel Distribution]
        F[RDMA Send/Receive]
    end
    
    subgraph "NVL Level"
        G[NVL Token Counting]
        H[NVL Channel Distribution]
        I[NVL Send/Receive]
    end
    
    subgraph "Output Data"
        J[recv_x: num_recv_tokens × hidden<br/>BF16/FP8]
        K[recv_topk_idx: num_recv_tokens × num_topk<br/>int64]
        L[recv_src_meta: num_recv_tokens × SourceMeta<br/>uint8]
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
    I --> K
    I --> L
```

## NVSHMEM IBGDA Integration

```mermaid
graph TB
    subgraph "NVSHMEM Teams"
        A[CPU RDMA Team<br/>Inter-node coordination]
        B[GPU RDMA Team<br/>GPU-to-GPU RDMA]
    end
    
    subgraph "IBGDA Features"
        C[Hardware RDMA<br/>Direct GPU-to-GPU]
        D[Multiple QPs<br/>Parallel channels]
        E[Async Operations<br/>Non-blocking RDMA]
    end
    
    subgraph "Buffer Management"
        F[Symmetric Buffers<br/>Same virtual address space]
        G[Remote Memory Access<br/>Direct peer memory access]
        H[Atomic Operations<br/>Lock-free synchronization]
    end
    
    A --> C
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
```

## Performance Optimizations

```mermaid
graph TD
    subgraph "Dual-Level Parallelism"
        A[RDMA Channels<br/>Parallel inter-node communication]
        B[NVL Channels<br/>Parallel intra-node communication]
        C[Warp Specialization<br/>Different roles per warp]
    end
    
    subgraph "Memory Access"
        D[Coalesced RDMA<br/>Optimized RDMA patterns]
        E[Shared Memory<br/>Efficient intra-node movement]
        F[TMA Acceleration<br/>Hardware-accelerated transfers]
    end
    
    subgraph "Buffer Management"
        G[Dual Buffers<br/>RDMA + NVL separation]
        H[Queue-based<br/>Producer-consumer pattern]
        I[Chunked Processing<br/>Better parallelism]
    end
```

## Key Functions and Their Roles

| Function | Location | Purpose |
|----------|----------|---------|
| `dispatch()` | buffer.py | Python interface, routes to internode |
| `internode_dispatch()` | buffer.py | Python internode wrapper |
| `internode_dispatch()` | deep_ep.cpp | C++ interface, handles tensor allocation |
| `notify_dispatch()` | internode.cu | Dual-level layout calculation |
| `dispatch()` | internode.cu | Main dual-level communication kernel |
| `get_dispatch_layout()` | layout.cu | Token counting and distribution |

## Communication Phases

1. **RDMA Layout Phase**: Calculate inter-node token distribution
2. **NVL Layout Phase**: Calculate intra-node token distribution  
3. **Dual Barrier Phase**: Synchronize RDMA and NVL ranks
4. **RDMA Dispatch Phase**: Send tokens between nodes via RDMA
5. **RDMA→NVL Forward Phase**: Forward RDMA data to local NVL ranks
6. **NVL Receive Phase**: Collect tokens via NVLink within nodes
7. **Cleanup Phase**: Prepare dual buffers for next iteration

## Routing Logic

```mermaid
graph TD
    A[Input Token] --> B{Expert Assignment}
    B --> C[Calculate RDMA Rank<br/>target_node = expert_idx // experts_per_node]
    B --> D[Calculate NVL Rank<br/>target_gpu = expert_idx % gpus_per_node]
    
    C --> E[RDMA Routing<br/>Send to target node]
    D --> F[NVL Routing<br/>Send to target GPU within node]
    
    E --> G[RDMA Buffer<br/>Inter-node communication]
    F --> H[NVL Buffer<br/>Intra-node communication]
    
    G --> I[RDMA→NVL Forwarder<br/>Bridge between levels]
    H --> I
    I --> J[Final Output<br/>Token at correct GPU]
```
