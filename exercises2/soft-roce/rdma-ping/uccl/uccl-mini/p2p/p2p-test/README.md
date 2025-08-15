# UCCL P2P Engine - High-Performance RDMA P2P Transfer

UCCL P2P Engine is a high-performance, RDMA-based P2P transfer system designed for distributed machine learning and high-throughput data processing applications. It provides a Python API for seamless integration with PyTorch tensors, NumPy arrays, and other data structures while leveraging the performance of InfiniBand/RoCE RDMA for ultra-low latency communication.

UCCL has an experimental GPU-driven P2P engine, see [gpu_driven](../gpu_driven/) folder.

## Project Structure

```
p2p/
├── engine.h          # C++ Endpoint class header with RDMA functionality
├── engine.cc         # C++ Endpoint implementation
├── pybind_engine.cc  # pybind11 wrapper for Python integration
├── Makefile          # Build configuration
├── test_engine.py    # Comprehensive test suite
├── demo.py           # Usage demonstration
└── README.md         # This file
```

## Prerequisites

### System Requirements
- Linux with RDMA support (optional for development)
- Python 3.7+ with development headers
- C++17 compatible compiler (GCC 7+ or Clang 5+)
- pybind11 library
- PyTorch or NumPy (for tensor/array operations)

### Optional Dependencies
- RDMA drivers and libraries (`libibverbs-dev`)
- RDMA-capable network hardware (InfiniBand, RoCE)
- CUDA (for GPU tensor operations)

## Installation

1. **Install Python dependencies:**
   ```bash
   make install-deps
   ```

2. **Build the module:**
   ```bash
   make
   ```

3. **Run tests:**
   ```bash
   make test
   ```

## Performance Benchmarks

### Running UCCL P2P

On client:
```bash
NCCL_IB_HCA="mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1" \
NCCL_SOCKET_IFNAME="ds-eap-1,ds-eap-2,ds-eap-3" \
python benchmark.py \
    --role client --remote-ip 192.168.0.100 --device gpu \
    --local-gpu-idx 0 --remote-gpu-idx 0 --num-cpus 4
```
On server: 
```bash
NCCL_IB_HCA="mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1" \
NCCL_SOCKET_IFNAME="ds-eap-1,ds-eap-2,ds-eap-3" \
python benchmark.py --role server --local-gpu-idx 0 --num-cpus 4
```

### Running NCCL
```bash
NCCL_NCHANNELS_PER_NET_PEER=4 python benchmark_nccl.py \
    --role client --remote-ip 192.168.0.100 --device gpu \
    --local-gpu-idx 0

NCCL_NCHANNELS_PER_NET_PEER=4 \
python benchmark_nccl.py --role server --local-gpu-idx 0
```

## Usage Examples

### Basic Endpoint Setup

```python
import uccl_p2p
import torch

# Create endpoint with local GPU index and number of CPUs
endpoint = uccl_p2p.Endpoint(local_gpu_idx=0, num_cpus=4)
```

### Client-Server Communication

```python
# Server side - accept incoming connections
success, remote_ip_addr, remote_gpu_idx, conn_id = endpoint.accept()
if success:
    print(f"Connected to {remote_ip_addr}, GPU {remote_gpu_idx}, conn_id={conn_id}")

# Client side - connect to remote server  
success, conn_id = endpoint.connect("192.168.1.100", remote_gpu_idx=1)
if success:
    print(f"Connected with conn_id={conn_id}")
```

### PyTorch Tensor Transfer

```python
# Sender side
send_tensor = torch.ones(1024, dtype=torch.float32)
assert send_tensor.is_contiguous()  # Ensure tensor is contiguous

# Register tensor for RDMA
success, mr_id = endpoint.reg(send_tensor.data_ptr(), send_tensor.numel() * 4)
assert success

# Send the tensor
success = endpoint.send(conn_id, mr_id, send_tensor.data_ptr(), send_tensor.numel() * 4)
assert success

# Receiver side
recv_tensor = torch.zeros(1024, dtype=torch.float32)
assert recv_tensor.is_contiguous()

# Register receive buffer
success, mr_id = endpoint.reg(recv_tensor.data_ptr(), recv_tensor.numel() * 4)
assert success

# Receive the tensor
success, recv_size = endpoint.recv(conn_id, mr_id, recv_tensor.data_ptr(), recv_tensor.numel() * 4)
assert success and recv_size == recv_tensor.numel() * 4
```

### NumPy Array Transfer

```python
import numpy as np

# Create and prepare NumPy array
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
assert data.flags['C_CONTIGUOUS']  # Ensure array is contiguous

# Register for RDMA
ptr = data.ctypes.data
size = data.nbytes
success, mr_id = endpoint.reg(ptr, size)

# Send array
if success:
    success = endpoint.send(conn_id, mr_id, ptr, size)
    
# Receive array
recv_data = np.zeros_like(data)
recv_ptr = recv_data.ctypes.data
success, recv_mr_id = endpoint.reg(recv_ptr, recv_data.nbytes)
success, recv_size = endpoint.recv(conn_id, recv_mr_id, recv_ptr, recv_data.nbytes)
```


## API Reference

### Endpoint Class

#### Constructor
```python
Endpoint(local_gpu_idx, num_cpus)
```
Create a new RDMA endpoint instance.

**Parameters:**
- `local_gpu_idx` (int): GPU index for this endpoint
- `num_cpus` (int): Number of CPU threads to use for RDMA operations

#### Connection Management

```python
connect(remote_ip_addr, remote_gpu_idx) -> (success, conn_id)
```
Connect to a remote endpoint.

**Parameters:**
- `remote_ip_addr` (str): IP address of remote server
- `remote_gpu_idx` (int): GPU index of remote endpoint

**Returns:**
- `success` (bool): Whether connection succeeded
- `conn_id` (int): Connection ID for subsequent operations

```python
accept() -> (success, remote_ip_addr, remote_gpu_idx, conn_id)
```
Accept an incoming connection (blocking).

**Returns:**
- `success` (bool): Whether connection was accepted
- `remote_ip_addr` (str): IP address of connecting client
- `remote_gpu_idx` (int): GPU index of connecting client
- `conn_id` (int): Connection ID for subsequent operations

#### Memory Registration

```python
reg(ptr, size) -> (success, mr_id)
```
Register a memory region for RDMA operations.

**Parameters:**
- `ptr` (int): Memory pointer (use `tensor.data_ptr()` for PyTorch)
- `size` (int): Size in bytes

**Returns:**
- `success` (bool): Whether registration succeeded
- `mr_id` (int): Memory region ID for transfer operations

#### Data Transfer

```python
send(conn_id, mr_id, ptr, size) -> success
```
Send data to remote endpoint (blocking).

**Parameters:**
- `conn_id` (int): Connection ID from connect/accept
- `mr_id` (int): Memory region ID from register
- `ptr` (int): Pointer to data to send
- `size` (int): Number of bytes to send

**Returns:**
- `success` (bool): Whether send completed successfully

```python
recv(conn_id, mr_id, ptr, max_size) -> (success, recv_size)
```
Receive data from remote endpoint (blocking).

**Parameters:**
- `conn_id` (int): Connection ID from connect/accept
- `mr_id` (int): Memory region ID from register
- `ptr` (int): Pointer to buffer for received data
- `max_size` (int): Maximum number of bytes to receive

**Returns:**
- `success` (bool): Whether receive completed successfully
- `recv_size` (int): Number of bytes actually received


## Development and Testing

### Build Targets
```bash
make all          # Build the module
make clean        # Clean build artifacts  
make test         # Run test suite
make install-deps # Install Python dependencies
make help         # Show available targets
```

### Testing Your Setup
```bash
# Run the included test suite
NCCL_IB_HCA="mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1" \
NCCL_SOCKET_IFNAME="ds-eap-1,ds-eap-2,ds-eap-3" \
python3 test_engine.py

# Check if RDMA hardware is available
# (This will work even without RDMA hardware for testing)
```