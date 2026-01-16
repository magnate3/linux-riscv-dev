# CUDA Programming Guide

A comprehensive, hands-on guide to CUDA programming covering everything from basic concepts to advanced optimization techniques. This guide is structured as a practical reference for developers at all levels.

## Table of Contents

### Quick Start - [`00_quick_start/`](00_quick_start/)
- **[CUDA Cheat Sheet](00_quick_start/0_cuda_cheat_sheet.md)** - Essential commands, concepts, and quick references

### Execution Model - [`01_execution_model/`](01_execution_model/)
- **[Execution Model](01_execution_model/1_cuda_execution_model.md)** - High-level concepts and navigation
- **[Thread Hierarchy](01_execution_model/2_thread_hierarchy.md)** - Thread/Block/Grid organization and indexing
- **[Warp Execution](01_execution_model/3_warp_execution.md)** - SIMT model, divergence, and optimization
- **[Streaming Multiprocessors Deep Dive](01_execution_model/4_streaming_multiprocessors_deep.md)** - SM architecture and occupancy
- **[Execution Constraints Guide](01_execution_model/5_execution_constraints_guide.md)** - Hardware limits and best practices

### Memory Hierarchy - [`02_memory_hierarchy/`](02_memory_hierarchy/)
- **[Memory Hierarchy](02_memory_hierarchy/1_cuda_memory_hierarchy.md)** - Memory types and access patterns
- **[Global Memory](02_memory_hierarchy/2_global_memory.md)** - Coalescing and optimization strategies
- **[Shared Memory](02_memory_hierarchy/3_shared_memory.md)** - Bank conflicts and performance tuning
- **[Constant Memory](02_memory_hierarchy/4_constant_memory.md)** - Broadcast patterns and use cases
- **[Unified Memory](02_memory_hierarchy/5_unified_memory.md)** - Advanced techniques and multi-GPU
- **[Memory Debugging](02_memory_hierarchy/6_memory_debugging.md)** - Profiling and troubleshooting

### Synchronization - [`03_synchronization/`](03_synchronization/)
- **[Synchronization Fundamentals](03_synchronization/1_synchronization_basics.md)** - Basics of thread coordination
- **[Block-Level Synchronization](03_synchronization/2_block_synchronization.md)** - Barriers and shared memory
- **[Warp-Level Synchronization](03_synchronization/3_warp_synchronization.md)** - SIMT and warp primitives
- **[Grid-Level Coordination](03_synchronization/4_grid_coordination.md)** - Multi-kernel and global patterns
- **[Atomic Operations](03_synchronization/5_atomic_operations.md)** - Thread-safe memory access
- **[Memory Consistency](03_synchronization/6_memory_consistency.md)** - Memory ordering and fences
- **[Debugging Synchronization](03_synchronization/7_synchronization_debugging.md)** - Tools and techniques
- **[Advanced Patterns](03_synchronization/8_advanced_synchronization.md)** - Wave and pipeline synchronization

### Streams & Concurrency - [`04_streams_concurrency/`](04_streams_concurrency/)
- **[CUDA Streams Concurrency](04_streams_concurrency/1_cuda_streams_concurrency.md)** - Index and advanced patterns
- **[Stream Fundamentals](04_streams_concurrency/1_stream_fundamentals.md)** - Stream types, properties, and debugging
- **[Asynchronous Operations](04_streams_concurrency/2_asynchronous_operations.md)** - Overlap, pipelines, and synchronization
- **[Memory Transfer Optimization](04_streams_concurrency/3_memory_transfer.md)** - Pinned memory and bandwidth
- **[Event-Driven Programming](04_streams_concurrency/4_event_driven_programming.md)** - Events, timing, and coordination
- **[CUDA Graphs Deep Dive](04_streams_concurrency/5_cuda_graphs.md)** - Graphs, updates, and optimization
- **[Advanced Stream Patterns](04_streams_concurrency/6_advanced_patterns.md)** - Producers, pipelines, and load balancing

### Performance & Profiling - [`05_performance_profiling/`](05_performance_profiling/)
- **[Profiling Overview](05_performance_profiling/1_profiling_overview.md)** - Key tools and concepts
- **[Nsight Systems](05_performance_profiling/2_nsight_systems.md)** - Timeline debugging
- **[Nsight Compute](05_performance_profiling/3_nsight_compute.md)** - Kernel performance analysis
- **[Roofline Model](05_performance_profiling/4_roofline_model.md)** - Bottleneck visualization
- **[Optimization Strategies](05_performance_profiling/5_optimization_strategies.md)** - Memory and compute tuning
- **[Occupancy Analysis](05_performance_profiling/6_occupancy_analysis.md)** - Maximizing GPU utilization
- **[Kernel Launch Tuning](05_performance_profiling/7_kernel_launch_tuning.md)** - Grid and block configuration
- **[Detecting Bottlenecks](05_performance_profiling/8_detecting_bottlenecks.md)** - Root cause analysis

### Advanced Features - [`06_advanced_features/`](06_advanced_features/)
- **[Tensor Cores](06_advanced_features/1_tensor_cores.md)** - Mixed-precision matrix operations with WMMA
- **[Dynamic Parallelism](06_advanced_features/2_dynamic_parallelism.md)** - Launching kernels from the device

## Learning Path Recommendations

### **Beginner Path**
1. Start with [CUDA Cheat Sheet](00_quick_start/0_cuda_cheat_sheet.md) for essential concepts
2. Read [Execution Model](01_execution_model/1_cuda_execution_model.md)
3. Study [Thread Hierarchy](01_execution_model/2_thread_hierarchy.md)
4. Learn [Memory Hierarchy](02_memory_hierarchy/1_cuda_memory_hierarchy.md)
5. Practice with [Global Memory](02_memory_hierarchy/2_global_memory.md)

### **Intermediate Path**
1. Deep dive into [Warp Execution](01_execution_model/3_warp_execution.md)
2. Understand [Streaming Multiprocessors Deep Dive](01_execution_model/4_streaming_multiprocessors_deep.md)
3. Learn [Shared Memory](02_memory_hierarchy/3_shared_memory.md)
4. Master [Synchronization Fundamentals](03_synchronization/1_synchronization_basics.md) and [Block-Level Sync](03_synchronization/2_block_synchronization.md)
5. Apply [Profiling Overview](05_performance_profiling/1_profiling_overview.md) techniques

### **Advanced Path**
1. Optimize with [Execution Constraints Guide](01_execution_model/5_execution_constraints_guide.md)
2. Master [Unified Memory](02_memory_hierarchy/5_unified_memory.md)
3. Implement [CUDA Streams Concurrency](04_streams_concurrency/1_stream_fundamentals.md)
4. Debug with [Memory Debugging](02_memory_hierarchy/6_memory_debugging.md)
5. Apply all concepts in real projects

## Prerequisites

- Basic understanding of C/C++ programming
- NVIDIA GPU with CUDA Compute Capability 3.0 or higher
- CUDA Toolkit installed (version 11.0 or later recommended)
- Basic familiarity with parallel programming concepts

## Development Environment Setup

```bash
# Check CUDA installation
nvcc --version

# Verify GPU availability
nvidia-smi

# Compile a simple CUDA program
nvcc -o hello hello.cu
```

## How to Use This Guide

### **Folder Structure**
The guide is organized into logical sections with numbered folders:
- **`00_quick_start/`** - Essential references and cheat sheets
- **`01_execution_model/`** - Thread hierarchy, warps, and execution concepts
- **`02_memory_hierarchy/`** - Memory types, optimization, and debugging
- **`03_synchronization/`** - Thread coordination and cooperation patterns
- **`04_streams_concurrency/`** - Asynchronous execution and concurrency
- **`05_performance_profiling/`** - Performance analysis and optimization tools

Each file is designed to be:
- **Self-contained** - Can be read independently
- **Practical** - Includes code examples and benchmarks
- **Reference-friendly** - Quick lookup tables and summaries
- **Progressive** - Builds complexity gradually

### **Navigation Tips**
- Use the overview files (ending in `.md`) for quick orientation
- Detailed guides (ending in `.md`) provide comprehensive coverage
- Cross-references between files help you find related concepts
- Code examples are optimized and ready to run

## Key Features

- **Comprehensive Coverage** - From basics to advanced optimization
- **Practical Examples** - Real-world code patterns and benchmarks
- **Performance Focus** - Optimization strategies and profiling techniques
- **Best Practices** - Industry-proven approaches and common pitfalls
- **Quick Reference** - Cheat sheets and lookup tables
- **Progressive Learning** - Multiple learning paths for different levels

## Contributing

This guide is designed to be a living reference. If you find errors, have suggestions, or want to add content:

1. Focus on practical, tested examples
2. Include performance analysis where relevant
3. Maintain the cross-reference structure
4. Follow the formatting conventions

## License

This guide is provided for educational purposes. Code examples are free to use and modify.

---

**Happy CUDA Programming!**

*Last updated: August 2025*
