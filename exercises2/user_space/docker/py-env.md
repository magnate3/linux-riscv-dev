

```
root@ubuntu:/home/ubuntu/pytorch/nano-vllm-dev# docker build -t py10 -f Dockerfile-py  .
[+] Building 1.3s (8/8) FINISHED                                                                                                                                                  docker:default
 => [internal] load build definition from Dockerfile-py                                                                                                                                     0.0s
 => => transferring dockerfile: 2.20kB                                                                                                                                                      0.0s
 => [internal] load metadata for nvcr.io/nvidia/cuda:11.6.2-devel-ubuntu20.04                                                                                                               0.0s
 => [internal] load .dockerignore                                                                                                                                                           0.0s
 => => transferring context: 481B                                                                                                                                                           0.0s
 => [1/4] FROM nvcr.io/nvidia/cuda:11.6.2-devel-ubuntu20.04                                                                                                                                 0.0s
 => CACHED [2/4] RUN mkdir -p /opt/env                                                                                                                                                      0.0s
 => CACHED [3/4] RUN echo 'deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy main restricted universe multiverse' > /etc/apt/sources.list.d/jammy-temp.list     && apt-get upda  0.0s
 => CACHED [4/4] RUN apt-get update && apt-get install -y     git     wget     curl     vim     build-essential     && rm -rf /var/lib/apt/lists/*                                          0.0s
 => exporting to image                                                                                                                                                                      1.3s
 => => exporting layers                                                                                                                                                                     1.3s
 => => writing image sha256:d13ead4980c634eedd48b36aa507473949e13427d95198f5ee7884f7318dc398                                                                                                0.0s
 => => naming to docker.io/library/py10                                                                                                                                                     0.0s
root@ubuntu:/home/ubuntu/pytorch/nano-vllm-dev# ls
assets  bench.py  docker-compose.yml  Dockerfile  Dockerfile-py  example.py  LICENSE  nanovllm  pyproject.toml  README.md
root@ubuntu:/home/ubuntu/pytorch/nano-vllm-dev# docker images
REPOSITORY            TAG                          IMAGE ID       CREATED          SIZE
py10                  latest                       d13ead4980c6   44 minutes ago   5.14GB
local/llama.cpp       server-cuda                  581cc262c133   7 days ago       8.38GB
llama.cpp             v1                           8352e141192e   8 days ago       6.84GB
nvcr.io/nvidia/cuda   12.4.0-devel-ubuntu22.04     4606c06b593e   2 years ago      5.87GB
nvcr.io/nvidia/cuda   12.4.0-runtime-ubuntu22.04   13ba2538fe6d   2 years ago      2.08GB
nvcr.io/nvidia/cuda   11.6.2-devel-ubuntu20.04     4304e869f051   2 years ago      4.81GB
nvcr.io/nvidia/cuda   11.6.2-runtime-ubuntu20.04   ea9227c77de1   2 years ago      1.95GB
nvcr.io/nvidia/cuda   11.6.2-base-ubuntu20.04      ca86ee401ec1   2 years ago      84.5MB
```


```
root@ubuntu:/home/ubuntu/pytorch/nano-vllm-dev# docker run  --net=host    -it py10:latest bash 

==========
== CUDA ==
==========

CUDA Version 11.6.2

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
   Use the NVIDIA Container Toolkit to start this container with GPU support; see
   https://docs.nvidia.com/datacenter/cloud-native/ .

root@ubuntu:/# python3 --version
Python 3.10.4
root@ubuntu:/# 
```