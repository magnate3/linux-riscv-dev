 
 ```
  sudo  docker run --rm --net=host    --gpus=all -it    -e UID=root    --ipc host --shm-size="32g"  --privileged   -u 0   -v /pytorch:/pytorch  ghcr.io/microsoft/mscclpp/mscclpp:base-dev-cuda12.8 bash
 ```
 
#  make
 
 ```
 g++ --version
g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Copyright (C) 2021 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 ```
 
安装nlohmann-json3    
 
```
apt-get install nlohmann-json3-dev -y
```

注释CMakeLists.txt中nlohmann json源码编译
```

#include(FetchContent)
#FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
#FetchContent_MakeAvailable(json)
```
 
 
```
  cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=/pytorch/mscclpp/install ..
```

```
make -j$(nproc)
make install
```


```
/pytorch/mscclpp/install# ls
_mscclpp.cpython-310-x86_64-linux-gnu.so  include  lib
```

> ##  tutorials/02-bootstrap

```
mscclpp/examples/tutorials/02-bootstrap
```
```
/usr/local/cuda/bin/nvcc -arch=native -o gpu_ping_pong_mp gpu_ping_pong_mp.cu  -I/pytorch/mscclpp/install/include -L /pytorch/mscclpp/install/lib -lmscclpp
```



```
 export LD_LIBRARY_PATH=/pytorch/mscclpp/install/lib:$LD_LIBRARY_PATH
```

```
./gpu_ping_pong_mp 
Error: At least two GPUs are required.
Error: At least two GPUs are required.
One of the processes failed.
```