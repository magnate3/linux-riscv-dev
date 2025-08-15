# use cpu memory test

```
pip3 install  pybind11
apt-get install libelf-dev -y
```

```
int const kMaxNumGPUs = 1;
```

```
#ifndef CPU_MEMORY
  DCHECK(local_gpu_idx_ < gpu_cards.size() && gpu_cards.size() <= kMaxNumGPUs)
      << "Local GPU index out of range";
  auto ib_nics = uccl::get_rdma_nics();
  // Find the RDMA NIC that is closest to each of the GPUs.
  for (int i = 0; i < kMaxNumGPUs; i++) {
    auto gpu_device_path = gpu_cards[i];
    auto ib_nic_it = std::min_element(
        ib_nics.begin(), ib_nics.end(), [&](auto const& a, auto const& b) {
          return uccl::cal_pcie_distance(gpu_device_path, a.second) <
                 uccl::cal_pcie_distance(gpu_device_path, b.second);
        });
    gpu_to_dev[i] = ib_nic_it - ib_nics.begin();
  }
  std::cout << "Detected best GPU-NIC mapping: " << std::endl;
  for (int i = 0; i < kMaxNumGPUs; i++) {
    std::cout << "\tGPU " << i << " -> NIC " << gpu_to_dev[i] << " ("
              << ib_nics[gpu_to_dev[i]].first << ")" << std::endl;
  }
  std::cout << std::endl;
#endif
```

+ server


```
python3  benchmark.py --role server --local-gpu-idx 0 --num-cpus 4
UCCL P2P Benchmark — role: server
Message sizes: 256 B, 1.0 KB, 4.0 KB, 16.0 KB, 64.0 KB, 256.0 KB, 1.0 MB, 10.0 MB, 100.0 MB
Device: cpu | Local GPU idx: 0 | Iterations: 1000
Creating Engine with GPU index: 0, CPUs: 4
Initialized mlx5_1
Initialized 4 engines for 1 devices totally, with 4 engines per device
Creating Engine GPU num: 0
Endpoint initialized successfully
[Server] Waiting for connection …
Waiting to accept incoming connection...
[Server] Connected to 10.22.116.220 (GPU 0) conn_id=0
[Server]    256 B :   0.25 Gbps |   0.03 GB/s
[Server]   1.0 KB :   0.98 Gbps |   0.12 GB/s
[Server]   4.0 KB :   3.62 Gbps |   0.45 GB/s
[Server]  16.0 KB :  12.30 Gbps |   1.54 GB/s
[Server]  64.0 KB :  35.55 Gbps |   4.44 GB/s
[Server] 256.0 KB :  68.28 Gbps |   8.54 GB/s
[Server]   1.0 MB :  86.73 Gbps |  10.84 GB/s
[Server]  10.0 MB :  95.50 Gbps |  11.94 GB/s
[Server] 100.0 MB :  96.71 Gbps |  12.09 GB/s
[Server] Benchmark complete
Destroying Engine...
Engine destroyed
```

+ client   
```
python3  benchmark.py --role client --remote-ip 10.22.116.221  --local-gpu-idx 0 --num-cpus 4
UCCL P2P Benchmark — role: client
Message sizes: 256 B, 1.0 KB, 4.0 KB, 16.0 KB, 64.0 KB, 256.0 KB, 1.0 MB, 10.0 MB, 100.0 MB
Device: cpu | Local GPU idx: 0 | Iterations: 1000
Creating Engine with GPU index: 0, CPUs: 4
Initialized mlx5_1
Initialized 4 engines for 1 devices totally, with 4 engines per device
Creating Engine GPU num: 0
Endpoint initialized successfully
Attempting to connect to 10.22.116.221:0
[Client] Connected to 10.22.116.221 conn_id=0
[Client]    256 B :   0.25 Gbps |   0.03 GB/s
[Client]   1.0 KB :   0.98 Gbps |   0.12 GB/s
[Client]   4.0 KB :   3.62 Gbps |   0.45 GB/s
[Client]  16.0 KB :  12.30 Gbps |   1.54 GB/s
[Client]  64.0 KB :  35.55 Gbps |   4.44 GB/s
[Client] 256.0 KB :  68.28 Gbps |   8.54 GB/s
[Client]   1.0 MB :  86.73 Gbps |  10.84 GB/s
[Client]  10.0 MB :  95.50 Gbps |  11.94 GB/s
[Client] 100.0 MB :  96.71 Gbps |  12.09 GB/s
[Client] Benchmark complete
Destroying Engine...
Engine destroyed
```