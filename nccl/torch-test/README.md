


#  环境


```
root@6bc5e2b9e885:/workspace/torch-test# python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'
/usr/local/lib/python3.9/site-packages/torch/share/cmake
root@6bc5e2b9e885:/workspace/torch-test# 
```


```
root@6bc5e2b9e885:/workspace/torch-test# python3 -c "import torch; print(torch.__version__)"
2.0.1+cu118
root@6bc5e2b9e885:/workspace/torch-test# 
```
+ g++
```
root@6bc5e2b9e885:/workspace/torch-test# g++ --version
g++ (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0
Copyright (C) 2019 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

root@6bc5e2b9e885:/workspace/torch-test# 
```

# run

## run with cpu    


```
 export LD_LIBRARY_PATH=/usr/local/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH
```

```
root@6bc5e2b9e885:/workspace/torch-test# ./main
bad W
bad X
bad B
bad Y
-nan
[ CPUFloatType{1} ]
bad TWO
root@6bc5e2b9e885:/workspace/torch-test# 
```

```
root@6bc5e2b9e885:/workspace/torch-test# make
g++ -D_GLIBCXX_USE_CXX11_ABI=0  -g -I/workspace/nccl-latest/build/include  -I/workspace/nccl-latest/fake_cuda/include  -I/workspace/nccl-latest/src/include -I/workspace/nccl-latest/src/graph  -I/workspace/nccl-latest/src/include/plugin -I/usr/local/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/ -I /usr/local/lib/python3.9/site-packages/torch/include   -ldl -L/workspace/nccl-latest/build/lib -lnccl -lpthread -Wl,-rpath=/usr/local/lib/python3.9/site-packages/torch/lib -L/usr/local/lib/python3.9/site-packages/torch/lib -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch   -c pytorch_test.cpp -o pytorch_test.o
g++  -o main pytorch_test.o -D_GLIBCXX_USE_CXX11_ABI=0  -g -I/workspace/nccl-latest/build/include  -I/workspace/nccl-latest/fake_cuda/include  -I/workspace/nccl-latest/src/include -I/workspace/nccl-latest/src/graph  -I/workspace/nccl-latest/src/include/plugin -I/usr/local/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/ -I /usr/local/lib/python3.9/site-packages/torch/include   -ldl -L/workspace/nccl-latest/build/lib -lnccl -lpthread -Wl,-rpath=/usr/local/lib/python3.9/site-packages/torch/lib -L/usr/local/lib/python3.9/site-packages/torch/lib -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch 
```

## run with gpu

[Installing C++ Distributions of PyTorch](https://docs.pytorch.org/cppdocs/installing.html)

```
root@4b5a67132e81:/example-app# mkdir build
root@4b5a67132e81:/example-app# cd build
```


```
cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
-- The C compiler identification is GNU 9.4.0
-- The CXX compiler identification is GNU 9.4.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - found
-- Found Threads: TRUE
-- Found CUDA: /usr/local/cuda (found version "11.6") 
-- The CUDA compiler identification is NVIDIA 11.6.124 with host compiler GNU 9.4.0
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: /usr/local/cuda/bin/nvcc - skipped
-- Detecting CUDA compile features
-- Detecting CUDA compile features - done
-- Caffe2: CUDA detected: 11.6
-- Caffe2: CUDA nvcc is: /usr/local/cuda/bin/nvcc
-- Caffe2: CUDA toolkit directory: /usr/local/cuda
-- Caffe2: Header version is: 11.6
-- Found CUDNN: /usr/lib/x86_64-linux-gnu/libcudnn.so
-- Found cuDNN: v8.4.0  (include: /usr/include, library: /usr/lib/x86_64-linux-gnu/libcudnn.so)
-- /usr/local/cuda/lib64/libnvrtc.so shorthash is 4dd39364
-- Autodetected CUDA architecture(s):  8.6
-- Added CUDA NVCC flags for: -gencode;arch=compute_86,code=sm_86
CMake Warning at /usr/local/lib/python3.8/dist-packages/torch/share/cmake/Torch/TorchConfig.cmake:22 (message):
  static library kineto_LIBRARY-NOTFOUND not found.
Call Stack (most recent call first):
  /usr/local/lib/python3.8/dist-packages/torch/share/cmake/Torch/TorchConfig.cmake:127 (append_torchlib_if_found)
  CMakeLists.txt:4 (find_package)


-- Found Torch: /usr/local/lib/python3.8/dist-packages/torch/lib/libtorch.so
-- Configuring done (2.0s)
CMake Warning at CMakeLists.txt:7 (add_executable):
  Cannot generate a safe runtime search path for target pytorch_test because
  files in some directories may conflict with libraries in implicit
  directories:

    runtime library [libcudnn.so.8] in /usr/lib/x86_64-linux-gnu may be hidden by files in:
      /usr/local/lib/python3.8/dist-packages/torch/lib

  Some of these libraries may not be found correctly.


-- Generating done (0.0s)
-- Build files have been written to: /pytorch/torch-test/build
root@ubuntu:/pytorch/torch-test/build# 
```

直接make
```
root@ubuntu:/pytorch/torch-test/build# VERBOSE=1 make
/usr/local/lib/python3.8/dist-packages/cmake/data/bin/cmake -S/pytorch/torch-test -B/pytorch/torch-test/build --check-build-system CMakeFiles/Makefile.cmake 0
/usr/local/lib/python3.8/dist-packages/cmake/data/bin/cmake -E cmake_progress_start /pytorch/torch-test/build/CMakeFiles /pytorch/torch-test/build//CMakeFiles/progress.marks
make  -f CMakeFiles/Makefile2 all
make[1]: Entering directory '/pytorch/torch-test/build'
make  -f CMakeFiles/pytorch_test.dir/build.make CMakeFiles/pytorch_test.dir/depend
make[2]: Entering directory '/pytorch/torch-test/build'
cd /pytorch/torch-test/build && /usr/local/lib/python3.8/dist-packages/cmake/data/bin/cmake -E cmake_depends "Unix Makefiles" /pytorch/torch-test /pytorch/torch-test /pytorch/torch-test/build /pytorch/torch-test/build /pytorch/torch-test/build/CMakeFiles/pytorch_test.dir/DependInfo.cmake "--color=" pytorch_test
make[2]: Leaving directory '/pytorch/torch-test/build'
make  -f CMakeFiles/pytorch_test.dir/build.make CMakeFiles/pytorch_test.dir/build
make[2]: Entering directory '/pytorch/torch-test/build'
[ 50%] Building CXX object CMakeFiles/pytorch_test.dir/pytorch_test.cpp.o
/usr/bin/c++ -DUSE_C10D_GLOO -DUSE_C10D_NCCL -DUSE_DISTRIBUTED -DUSE_RPC -DUSE_TENSORPIPE -isystem /usr/local/lib/python3.8/dist-packages/torch/include -isystem /usr/local/lib/python3.8/dist-packages/torch/include/torch/csrc/api/include -isystem /usr/local/cuda/include -D_GLIBCXX_USE_CXX11_ABI=0 -std=gnu++17 -D_GLIBCXX_USE_CXX11_ABI=0 -MD -MT CMakeFiles/pytorch_test.dir/pytorch_test.cpp.o -MF CMakeFiles/pytorch_test.dir/pytorch_test.cpp.o.d -o CMakeFiles/pytorch_test.dir/pytorch_test.cpp.o -c /pytorch/torch-test/pytorch_test.cpp
[100%] Linking CXX executable pytorch_test
/usr/local/lib/python3.8/dist-packages/cmake/data/bin/cmake -E cmake_link_script CMakeFiles/pytorch_test.dir/link.txt --verbose=1
/usr/bin/c++  -D_GLIBCXX_USE_CXX11_ABI=0 CMakeFiles/pytorch_test.dir/pytorch_test.cpp.o -o pytorch_test  -Wl,-rpath,/usr/local/cuda/lib64:/usr/local/lib/python3.8/dist-packages/torch/lib /usr/local/lib/python3.8/dist-packages/torch/lib/libtorch.so /usr/local/lib/python3.8/dist-packages/torch/lib/libc10.so /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libnvrtc.so /usr/local/cuda/lib64/libnvToolsExt.so /usr/local/cuda/lib64/libcudart.so /usr/local/lib/python3.8/dist-packages/torch/lib/libc10_cuda.so -Wl,--no-as-needed,"/usr/local/lib/python3.8/dist-packages/torch/lib/libtorch_cuda.so" -Wl,--as-needed -Wl,--no-as-needed,"/usr/local/lib/python3.8/dist-packages/torch/lib/libtorch_cuda_cpp.so" -Wl,--as-needed -Wl,--no-as-needed,"/usr/local/lib/python3.8/dist-packages/torch/lib/libtorch_cpu.so" -Wl,--as-needed -lpthread /usr/local/lib/python3.8/dist-packages/torch/lib/libc10_cuda.so /usr/local/lib/python3.8/dist-packages/torch/lib/libc10.so /usr/local/cuda/lib64/libcufft.so /usr/local/cuda/lib64/libcurand.so /usr/local/cuda/lib64/libcublas.so /usr/lib/x86_64-linux-gnu/libcudnn.so -Wl,--no-as-needed,"/usr/local/lib/python3.8/dist-packages/torch/lib/libtorch_cuda_cu.so" -Wl,--as-needed -Wl,--no-as-needed,"/usr/local/lib/python3.8/dist-packages/torch/lib/libtorch.so" -Wl,--as-needed /usr/local/cuda/lib64/libnvToolsExt.so /usr/local/cuda/lib64/libcudart.so
make[2]: Leaving directory '/pytorch/torch-test/build'
[100%] Built target pytorch_test
make[1]: Leaving directory '/pytorch/torch-test/build'
/usr/local/lib/python3.8/dist-packages/cmake/data/bin/cmake -E cmake_progress_start /pytorch/torch-test/build/CMakeFiles 0
```


```
root@ubuntu:/pytorch/torch-test/build#  export LD_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/torch/lib:$LD_LIBRARY_PATH
root@ubuntu:/pytorch/torch-test/build# ./main
bash: ./main: No such file or directory
root@ubuntu:/pytorch/torch-test/build# ls
CMakeCache.txt  CMakeFiles  Makefile  cmake_install.cmake  detect_cuda_compute_capabilities.cu  detect_cuda_version.cc  pytorch_test
root@ubuntu:/pytorch/torch-test/build# ./pytorch_test 
bad W
bad X
bad B
bad Y
-nan
[ CPUFloatType{1} ]
bad TWO
```