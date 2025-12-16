



```
export NCCL_ROOT_DIR=/workspace/nccl-latest
export LD_LIBRARY_PATH=./libcuda/:$NCCL_ROOT_DIR/build/lib
```

```
g++ -D_GLIBCXX_USE_CXX11_ABI=0  -g -I/workspace/nccl-latest/build/include  -I/workspace/nccl-latest/fake_cuda/include  -I/workspace/nccl-latest/src/include -I/workspace/nccl-latest/src/graph  -I/workspace/nccl-latest/src/include/plugin -I./ -I/usr/local/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/ -I /usr/local/lib/python3.9/site-packages/torch/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/  -ldl -L/workspace/nccl-latest/build/lib -lnccl -lpthread -Wl,-rpath=/usr/local/lib/python3.9/site-packages/torch/lib -L/usr/local/lib/python3.9/site-packages/torch/lib -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch -Wl,-rpath=/usr/lib/x86_64-linux-gnu/libmpi.so -L/usr/lib/x86_64-linux-gnu/libmpi.so -lmpi -lmpi_cxx -Wl,-rpath=./libcuda  -L./libcuda  -lcuda  -c c10d/test/ProcessGroupNCCLTest.cpp -o c10d/test/ProcessGroupNCCLTest.o
g++  -o main c10d/ProcessGroup.o c10d/NCCLUtils.o c10d/Utils.o c10d/Store.o c10d/ProcessGroupMPI.o c10d/ProcessGroupRoundRobin.o c10d/TCPStore.o c10d/HashStore.o c10d/PrefixStore.o c10d/FileStore.o c10d/ProcessGroupNCCL.o c10d/test/ProcessGroupNCCLTest.o -D_GLIBCXX_USE_CXX11_ABI=0  -g -I/workspace/nccl-latest/build/include  -I/workspace/nccl-latest/fake_cuda/include  -I/workspace/nccl-latest/src/include -I/workspace/nccl-latest/src/graph  -I/workspace/nccl-latest/src/include/plugin -I./ -I/usr/local/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/ -I /usr/local/lib/python3.9/site-packages/torch/include -I/usr/lib/x86_64-linux-gnu/openmpi/include/  -ldl -L/workspace/nccl-latest/build/lib -lnccl -lpthread -Wl,-rpath=/usr/local/lib/python3.9/site-packages/torch/lib -L/usr/local/lib/python3.9/site-packages/torch/lib -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch -Wl,-rpath=/usr/lib/x86_64-linux-gnu/libmpi.so -L/usr/lib/x86_64-linux-gnu/libmpi.so -lmpi -lmpi_cxx -Wl,-rpath=./libcuda  -L./libcuda  -lcuda
```

```
root@6bc5e2b9e885:/workspace/dist-opt# ./main 
root@6bc5e2b9e885:/workspace/dist-opt# 
```