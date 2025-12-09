# NCCL Multi-Node Multi-GPU Communication Example (MPI-free)

This project demonstrates NCCL-based communication across multiple nodes and GPUs without requiring MPI. Each node runs a single process that manages multiple GPUs through multi-threading, with one dedicated thread per GPU.

## Building and Running

```bash
# Create and enter build directory
mkdir -p build && cd build

# Configure CMake project
cmake ..

# Compile
cmake --build .

# Run on the master node
./nccl_multi_node_demo --rank 0 --nproc 2 --port [communication_port] --size [data_size]

# Run on worker nodes
./nccl_multi_node_demo --rank 1 --nproc 2 --master [master_node_IP] --port [communication_port] --size [data_size]
```

## Results

### Master Node Output

![Master Node Results](https://github.com/whitelok/nccl_allreduce_demo_without_mpi/blob/master/imgs/master.png?raw=true "Master Node Results")

### Worker Node Output

![Worker Node Results](https://github.com/whitelok/nccl_allreduce_demo_without_mpi/blob/master/imgs/worker.png?raw=true "Worker Node Results")