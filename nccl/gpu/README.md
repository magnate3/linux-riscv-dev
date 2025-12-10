

```
nvidia-smi
lspci | grep -i nvidia
nvidia-smi nvlink -R
```

```
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

#  NVIDIA Container Toolkit(docker支持gpu)

Configure the repository:   
```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
&& sudo apt-get update
```
Install the NVIDIA Container Toolkit packages:   

```
apt-get install -y nvidia-container-toolkit
```

Configure the container runtime by using the nvidia-ctk command:    
```
nvidia-ctk runtime configure --runtime=docker
 
```
Restart the Docker daemon:   
```
systemctl restart docker
```

#  /nvidia/pytorch:24.07-py3
```
 
docker run  --net=host    --gpus=all -it    -e UID=root    --ipc host --shm-size="32g"  -v /pytorch:/pytorch -u 0 --name=nccl docker.io/nvcr.io/nvidia/pytorch:24.07-py3 bash
 
```


```
python3 --version
Python 3.10.12
nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Jun__6_02:18:23_PDT_2024
Cuda compilation tools, release 12.5, V12.5.82
Build cuda_12.5.r12.5/compiler.34385749_0
```


```
root@ubuntu:/pytorch# make
make: Warning: File 'Makefile' has modification time 28751 s in the future
/usr/local/cuda/bin/nvcc -ccbin g++ -m64 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o nonblocking_double_streams nonblocking_double_streams.cu -lnccl
/usr/local/cuda/bin/nvcc -ccbin g++ -m64 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o node_server node_server.cu -lnccl
/usr/local/cuda/bin/nvcc -ccbin g++ -m64 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o node_client node_client.cu -lnccl
/usr/local/cuda/bin/nvcc -ccbin g++ -m64 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o one_device_per_thread one_device_per_thread.cu -lnccl
/usr/local/cuda/bin/nvcc -ccbin g++ -m64 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o multi_devices_per_thread multi_devices_per_thread.cu -lnccl
/usr/local/cuda/bin/nvcc -ccbin g++ -m64 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o alltoall_test alltoall.cu -lnccl
make: warning:  Clock skew detected.  Your build may be incomplete.
```

```

root@ubuntu/pytorch# gcc --version
gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Copyright (C) 2021 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

root@ubuntu/pytorch# ./one_device_per_thread  --nranks 2
Local rank size: 2
NCCL version 2.22.3+cuda12.5

ubuntu1583:1585 [0] init.cc:738 NCCL WARN Duplicate GPU detected : rank 0 and rank 1 both on CUDA device 1000

ubuntu1583:1586 [0] init.cc:738 NCCL WARN Duplicate GPU detected : rank 1 and rank 0 both on CUDA device 1000
Failed, NCCL error one_device_per_thread.cu:17 'invalid usage (run with NCCL_DEBUG=WARN for details)'
Failed, NCCL error one_device_per_thread.cu:17 'invalid usage (run with NCCL_DEBUG=WARN for details)'
```
nranks太多   


```
 root@ubuntu/pytorch# ./one_device_per_thread  --nranks 1
Local rank size: 1
NCCL version 2.22.3+cuda12.5
GPU:0 data: 0.000000.
Finished successfully.
```
 
```
root@ubuntu:/pytorch# g++ --version
g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Copyright (C) 2021 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

root@ubuntu:/pytorch# gcc --version
gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Copyright (C) 2021 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

root@ubuntu:/pytorch# 
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
GLIBCXX_3.4
GLIBCXX_3.4.1
GLIBCXX_3.4.2
GLIBCXX_3.4.3
GLIBCXX_3.4.4
GLIBCXX_3.4.5
GLIBCXX_3.4.6
GLIBCXX_3.4.7
GLIBCXX_3.4.8
GLIBCXX_3.4.9
GLIBCXX_3.4.10
GLIBCXX_3.4.11
GLIBCXX_3.4.12
GLIBCXX_3.4.13
GLIBCXX_3.4.14
GLIBCXX_3.4.15
GLIBCXX_3.4.16
GLIBCXX_3.4.17
GLIBCXX_3.4.18
GLIBCXX_3.4.19
GLIBCXX_3.4.20
GLIBCXX_3.4.21
GLIBCXX_3.4.22
GLIBCXX_3.4.23
GLIBCXX_3.4.24
GLIBCXX_3.4.25
GLIBCXX_3.4.26
GLIBCXX_3.4.27
GLIBCXX_3.4.28
GLIBCXX_3.4.29
GLIBCXX_3.4.30
GLIBCXX_DEBUG_MESSAGE_LENGTH
```

```
 strings /usr/lib/x86_64-linux-gnu/libc.so.6 | grep GLIBC
GLIBC_2.2.5
GLIBC_2.2.6
GLIBC_2.3
GLIBC_2.3.2
GLIBC_2.3.3
GLIBC_2.3.4
GLIBC_2.4
GLIBC_2.5
GLIBC_2.6
GLIBC_2.7
GLIBC_2.8
GLIBC_2.9
GLIBC_2.10
GLIBC_2.11
GLIBC_2.12
GLIBC_2.13
GLIBC_2.14
GLIBC_2.15
GLIBC_2.16
GLIBC_2.17
GLIBC_2.18
GLIBC_2.22
GLIBC_2.23
GLIBC_2.24
GLIBC_2.25
GLIBC_2.26
GLIBC_2.27
GLIBC_2.28
GLIBC_2.29
GLIBC_2.30
GLIBC_2.31
GLIBC_2.32
GLIBC_2.33
GLIBC_2.34
GLIBC_2.35
GLIBC_PRIVATE
```

> ## gdb

```
root@ubuntu:/pytorch# export dbg=1
root@ubuntu:/pytorch# make
```

> ## mpi


```
root@ubuntu:/pytorch# vim Makefile 
root@ubuntu:/pytorch# make mpi
/usr/local/cuda/bin/nvcc -ccbin g++ -m64 -I./ -I/usr/local/mpi/include/ -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o nccl_with_mpi nccl_with_mpi.cu -lnccl -lmpi -L/usr/local/mpi/lib/
/usr/local/cuda/bin/nvcc -ccbin g++ -m64 -I./ -I/usr/local/mpi/include/ -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o allreduce_2comms example_allreduce_2comms.cu -lnccl -lmpi -L/usr/local/mpi/lib/
root@ubuntu:/pytorch# 
root@ubuntu:/pytorch# mpirun -np 1 --allow-run-as-root   allreduce_2comms 
The local rank is: 0
[MPI Rank 0] Success 
[Rank MPI 0] Success 
```

```
root@ubuntu:/pytorch# export NCCL_DEBUG=INFO
root@ubuntu:/pytorch# mpirun -np 2 --allow-run-as-root   allreduce_2comms 
The local rank is: 0
The local rank is: 1
ubuntu:1482:1482 [0] NCCL INFO Bootstrap : Using eno1:172.22.116.89<0>
Failed: Cuda error example_allreduce_2comms.cu:107 'invalid device ordinal'
--------------------------------------------------------------------------
Primary job  terminated normally, but 1 process returned
a non-zero exit code. Per user-direction, the job has been aborted.
--------------------------------------------------------------------------
ubuntu:1482:1482 [0] NCCL INFO cudaDriverVersion 12080
ubuntu:1482:1482 [0] NCCL INFO NCCL version 2.22.3+cuda12.5
ubuntu:1482:1482 [0] NCCL INFO Plugin Path : /opt/hpcx/nccl_rdma_sharp_plugin/lib/libnccl-net.so
ubuntu:1482:1482 [0] NCCL INFO P2P plugin IBext_v8
ubuntu:1482:1482 [0] NCCL INFO NET/IB : No device found.
ubuntu:1482:1482 [0] NCCL INFO NET/IB : No device found.
ubuntu:1482:1482 [0] NCCL INFO NET/Socket : Using [0]eno1:172.22.116.89<0> [1]ztyou3pbk2:192.168.193.155<0>
ubuntu:1482:1482 [0] NCCL INFO Using network Socket
--------------------------------------------------------------------------
mpirun detected that one or more processes exited with non-zero status, thus causing
the job to be terminated. The first process to do so was:

  Process name: [[34415,1],1]
  Exit code:    1
--------------------------------------------------------------------------
```

> ## nonblocking_double_streams
```
./nonblocking_double_streams --nranks 1
Local rank size: 1
GPU:0 data: 0.000000.
GPU:0 data: 0.000000.
All streams finished successfully.
```
+  DEBUG   
```
DEBUG=1 ./nonblocking_double_streams --nranks 1
Local rank size: 1
Id diff internal idx_2: id1:2:d diff internal idx_3: id1:2:S
Group: 1 GPU idx: 0. The start time: 2025-12-09 11:00:30.602
Group: 1 GPU idx: 0. The first iter end time: 2025-12-09 11:00:30.602
Group: 1 GPU idx: 0. The end time: 2025-12-09 11:00:30.603
GPU:0 data: 0.000000.
Group: 2 GPU idx: 0. The start time: 2025-12-09 11:00:30.604
Group: 2 GPU idx: 0. The first iter end time: 2025-12-09 11:00:30.604
Group: 2 GPU idx: 0. The end time: 2025-12-09 11:00:30.604
GPU:0 data: 0.000000.
All streams finished successfully.
```


> ##  libnccl
 
[NCCL Release](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2234/release-notes/rel_2-22-3.html) 
 
```
./one_device_per_thread 
./one_device_per_thread: error while loading shared libraries: libnccl.so.2: cannot open shared object file: No such file or directory
```
+ ubuntu20.04
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

sudo apt install libnccl2 libnccl-dev

ldconfig -p | grep libnccl
```

```
ldconfig -p | grep libnccl
        libnccl.so.2 (libc6,x86-64) => /lib/x86_64-linux-gnu/libnccl.so.2
        libnccl.so (libc6,x86-64) => /lib/x86_64-linux-gnu/libnccl.so
```

> ## glibc(需要gcc11)

安装g++-11
```
add-apt-repository -y ppa:ubuntu-toolchain-r/test
apt install -y g++-11
```

```
/pytorch# ./one_device_per_thread 
./one_device_per_thread: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by ./one_device_per_thread)
./one_device_per_thread: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.34' not found (required by ./one_device_per_thread)
```




```
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
GLIBCXX_3.4
GLIBCXX_3.4.1
GLIBCXX_3.4.2
GLIBCXX_3.4.3
GLIBCXX_3.4.4
GLIBCXX_3.4.5
GLIBCXX_3.4.6
GLIBCXX_3.4.7
GLIBCXX_3.4.8
GLIBCXX_3.4.9
GLIBCXX_3.4.10
GLIBCXX_3.4.11
GLIBCXX_3.4.12
GLIBCXX_3.4.13
GLIBCXX_3.4.14
GLIBCXX_3.4.15
GLIBCXX_3.4.16
GLIBCXX_3.4.17
GLIBCXX_3.4.18
GLIBCXX_3.4.19
GLIBCXX_3.4.20
GLIBCXX_3.4.21
GLIBCXX_3.4.22
GLIBCXX_3.4.23
GLIBCXX_3.4.24
GLIBCXX_3.4.25
GLIBCXX_3.4.26
GLIBCXX_3.4.27
GLIBCXX_3.4.28
GLIBCXX_DEBUG_MESSAGE_LENGTH
```
查看gcc   

```
gcc --version
gcc (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
```





搜索glibc

```
 find / -name "libstdc++.so.6"*
```
发现/snap/gnome-42-2204/226/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30    

```
strings /snap/gnome-42-2204/226/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30 | grep GLIBCXX
GLIBCXX_3.4
GLIBCXX_3.4.1
GLIBCXX_3.4.2
GLIBCXX_3.4.3
GLIBCXX_3.4.4
GLIBCXX_3.4.5
GLIBCXX_3.4.6
GLIBCXX_3.4.7
GLIBCXX_3.4.8
GLIBCXX_3.4.9
GLIBCXX_3.4.10
GLIBCXX_3.4.11
GLIBCXX_3.4.12
GLIBCXX_3.4.13
GLIBCXX_3.4.14
GLIBCXX_3.4.15
GLIBCXX_3.4.16
GLIBCXX_3.4.17
GLIBCXX_3.4.18
GLIBCXX_3.4.19
GLIBCXX_3.4.20
GLIBCXX_3.4.21
GLIBCXX_3.4.22
GLIBCXX_3.4.23
GLIBCXX_3.4.24
GLIBCXX_3.4.25
GLIBCXX_3.4.26
GLIBCXX_3.4.27
GLIBCXX_3.4.28
GLIBCXX_3.4.29
GLIBCXX_3.4.30
GLIBCXX_DEBUG_MESSAGE_LENGTH
```


> ## nonblocking_double_streams

```
root@ubuntu/pytorch# DEBUG=1 ./nonblocking_double_streams --nranks 1
Local rank size: 1
Id diff internal idx_0: id1: id2:n
Id diff internal idx_1: id1: id2:d diff internal idx_2: id1:r id2:
Id diff internal idx_3: id1:2:d diff internal idx_4: id1:V id2:d diff internal idx_5: id1:2:d diff internal idx_6: id1:I id2:d diff internal idx_7: id1: id2:
d diff internal idx_10: id1:2:d diff internal idx_11: id1:2:a
Id diff internal idx_40: id1: id2:
Id diff internal idx_41: id1: id2:d diff internal idx_42: id1: id2:Q
Id diff internal idx_43: id1: id2:d diff internal idx_44: id1: id2:d diff internal idx_45: id1: id2:
Id diff internal idx_48: id1: id2:8
Id diff internal idx_49: id1: id2:d diff internal idx_50: id1: id2:Q
Id diff internal idx_51: id1: id2:d diff internal idx_52: id1: id2:d diff internal idx_53: id1: id2:
Id diff internal idx_56: id1:| id2:d diff internal idx_57: id1: id2:I
Id diff internal idx_58: id1: id2:{
Id diff internal idx_59: id1: id2:d diff internal idx_60: id1:w id2:d diff internal idx_61: id1: id2:
Id diff internal idx_64: id1: id2:d diff internal idx_65: id1: id2:d diff internal idx_66: id1: id2:d diff internal idx_67: id1: id2:4
Id diff internal idx_68: id1: id2:d diff internal idx_69: id1: id2:U
Id diff internal idx_72: id1: id2:d diff internal idx_73: id1: id2:d diff internal idx_74: id1: id2:d diff internal idx_75: id1: id2:4
Id diff internal idx_76: id1: id2:d diff internal idx_77: id1: id2:U
Id diff internal idx_79: id1:2:
Id diff internal idx_80: id1: id2:X
Id diff internal idx_81: id1: id2:d diff internal idx_82: id1: id2:d diff internal idx_83: id1: id2:d diff internal idx_84: id1: id2:d diff internal idx_85: id1: id2:
Id diff internal idx_88: id1: id2:d diff internal idx_89: id1: id2:d diff internal idx_90: id1: id2:d diff internal idx_91: id1: id2:4
Id diff internal idx_92: id1: id2:d diff internal idx_93: id1: id2:U
Id diff internal idx_96: id1: id2:0
Id diff internal idx_97: id1: id2:d diff internal idx_98: id1: id2:d diff internal idx_99: id1: id2:4
Id diff internal idx_100: id1: id2:d diff internal idx_101: id1: id2:U
Id diff internal idx_104: id1: id2:d diff internal idx_105: id1: id2:d diff internal idx_106: id1: id2:d diff internal idx_107: id1: id2:4
Id diff internal idx_108: id1: id2:d diff internal idx_109: id1: id2:U
Id diff internal idx_112: id1: id2:d diff internal idx_121: id1: id2:'
Id diff internal idx_122: id1: id2:z
Id diff internal idx_123: id1: id2:d diff internal idx_124: id1: id2:d diff internal idx_125: id1: id2:d diff internal idx_126: id1: id2:
Id diff internal idx_127: id1: id2:g
NCCL version 2.22.3+cuda12.5
Group: 2 GPU idx: 0. The start time: 2025-12-03 09:13:28.774
Group: 2 GPU idx: 0. The first iter end time: 2025-12-03 09:13:28.774
Failed: Cuda error nonblocking_double_streams.cu:145 'out of memory'
```

> ## torch
```
python3 torch_hello.py 
2.4.0a0+3bcc3cddb5.nv24.07
True
hello, world.
cpu 0.04186892509460449 tensor(141178.4375)
cuda:0 0.05131340026855469 tensor(141321., device='cuda:0')
cuda:0 0.0001766681671142578 tensor(141321., device='cuda:0')
```