

```
ubuntu@ubuntu:~$ sudo  docker run -v ~/pytorch:/workspace --gpus all -it  local/llama.cpp:server-cuda   bash

==========
== CUDA ==
==========

CUDA Version 11.6.2

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

root@82e6803f67ad:/app# cd /workspace/proj1/ggml-paged/
root@82e6803f67ad:/workspace/proj1/ggml-paged# rm -rf build/
root@82e6803f67ad:/workspace/proj1/ggml-paged# cmake -S . -B build
-- The C compiler identification is GNU 9.4.0
-- The CUDA compiler identification is NVIDIA 11.6.124 with host compiler GNU 9.4.0
```

```

 cmake -S . -B build
```

```
root@82e6803f67ad:/workspace/proj1/ggml-paged# ./build/test_paged_attn 
Paged Attention Tests — NVIDIA A10 (CC 8.6)

1. Single-head tests:
  test: Q_heads=1 KV_heads=1 D=128 seq=64 chunk=64 ... PASS (max_rel=0.044% avg=0.0163% max_abs=0.000004)
         chunks=1  bytes_xfer=32768
  test: Q_heads=1 KV_heads=1 D=128 seq=128 chunk=64 ... PASS (max_rel=0.043% avg=0.0172% max_abs=0.000002)
         chunks=2  bytes_xfer=65536
  test: Q_heads=1 KV_heads=1 D=128 seq=300 chunk=128 ... PASS (max_rel=0.039% avg=0.0163% max_abs=0.000002)
         chunks=3  bytes_xfer=153600
  test: Q_heads=1 KV_heads=1 D=128 seq=2048 chunk=256 ... PASS (max_rel=0.042% avg=0.0142% max_abs=0.000000)
         chunks=8  bytes_xfer=1048576

2. GQA tests:
  test: Q_heads=8 KV_heads=2 D=128 seq=256 chunk=128 ... PASS (max_rel=0.046% avg=0.0158% max_abs=0.000002)
         chunks=2  bytes_xfer=262144
  test: Q_heads=40 KV_heads=8 D=128 seq=512 chunk=256 ... PASS (max_rel=0.048% avg=0.0164% max_abs=0.000002)
         chunks=2  bytes_xfer=2097152

3. Head dimension tests:
  test: Q_heads=4 KV_heads=4 D=64 seq=256 chunk=128 ... PASS (max_rel=0.046% avg=0.0165% max_abs=0.000002)
         chunks=2  bytes_xfer=262144
  test: Q_heads=4 KV_heads=4 D=96 seq=256 chunk=128 ... PASS (max_rel=0.045% avg=0.0170% max_abs=0.000002)
         chunks=2  bytes_xfer=393216
  test: Q_heads=4 KV_heads=4 D=128 seq=256 chunk=128 ... PASS (max_rel=0.046% avg=0.0170% max_abs=0.000002)
         chunks=2  bytes_xfer=524288

4. Long sequence tests:
  test: Q_heads=4 KV_heads=4 D=128 seq=4096 chunk=1024 ... PASS (max_rel=0.045% avg=0.0142% max_abs=0.000000)
         chunks=4  bytes_xfer=8388608
  test: Q_heads=8 KV_heads=2 D=128 seq=8192 chunk=2048 ... PASS (max_rel=0.048% avg=0.0129% max_abs=0.000000)
         chunks=4  bytes_xfer=8388608

SUCCESS: 11/11 tests passed
```