# `baidu-allreduce`

`baidu-allreduce` is a small C++ library, demonstrating the ring allreduce and
ring allgather techniques. The goal is to provide a template for deep learning
framework authors to use when implementing these communication algorithms
within their respective frameworks.

A description of the ring allreduce with its application to deep learning is
available on the [Baidu SVAIL blog](http://research.baidu.com/bringing-hpc-techniques-deep-learning/).

[baidu-allreduce算法的实现解析](https://zhuanlan.zhihu.com/p/1944848399892460554)   

## Installation

**Prerequisites:** Before compiling `baidu-allreduce`, make sure you have
installed CUDA (7.5 or greater) and an MPI implementation.

`baidu-allreduce` has been tested with [OpenMPI](https://www.open-mpi.org/),
but should work with any CUDA-aware MPI implementation, such as MVAPICH.

To compile `baidu-allreduce`, run

```bash
# Modify MPI_ROOT to point to your installation of MPI.
# You should see $MPI_ROOT/include/mpi.h and $MPI_ROOT/lib/libmpi.so.
# Modify CUDA_ROOT to point to your installation of CUDA.
make MPI_ROOT=/usr/lib/openmpi CUDA_ROOT=/path/to/cuda/lib64
```

```
root@ubuntu:/pytorch/baidu-allreduce# ls /usr/local/mpi
bin  etc  include  lib  share
root@ubuntu:/pytorch/baidu-allreduce# ls /usr/local/cuda
bin  compat  compute-sanitizer  doc  extras  gds  include  lib64  nvml  nvvm  share  src  targets
root@ubuntu:/pytorch/baidu-allreduce# make MPI_ROOT=/usr/local/mpi CUDA_ROOT=/usr/local/cuda
make: Warning: File 'Makefile' has modification time 28730 s in the future
mpic++ -c -std=c++11 -I/usr/local/mpi/include -I. -I/usr/local/cuda/include -DOMPI_SKIP_MPICXX= timer.cpp -o timer.o
mpic++ -c -std=c++11 -I/usr/local/mpi/include -I. -I/usr/local/cuda/include -DOMPI_SKIP_MPICXX= test/test.cpp -o test/test.o
nvcc -c -std=c++11 -I/usr/local/mpi/include -I. -I/usr/local/cuda/include -DOMPI_SKIP_MPICXX= collectives.cu -o collectives.o
mpic++ -o allreduce-test -L/usr/local/cuda/lib64 -L/usr/local/mpi/lib -lcudart -lmpi -DOMPI_SKIP_MPICXX= timer.o test/test.o collectives.o -L/usr/local/cuda/lib64 -L/usr/local/mpi/lib -lcudart -lmpi -DOMPI_SKIP_MPICXX=
make: warning:  Clock skew detected.  Your build may be incomplete.
```

```bash
# On CPU.
mpirun --np 3 allreduce-test cpu

# On GPU. Requires a CUDA-aware MPI implementation.
mpirun --np 3 allreduce-test gpu
```


```
mpirun --np 1 allreduce-test gpu
Verified allreduce for size 0 (6.90844e-07 per iteration)
Verified allreduce for size 32 (8.87529e-06 per iteration)
```

## Interface

The `baidu-allreduce` library provides the following C++ functions:

```c++
// Initialize the library, including MPI and if necessary the CUDA device.
// If device == NO_DEVICE, no GPU is used; otherwise, the device specifies which CUDA
// device should be used. All data passed to other functions must be on that device.
#define NO_DEVICE -1
void InitCollectives(int device);

// The ring allreduce. The lengths of the data chunks passed to this function
// must be the same across all MPI processes. The output memory will be
// allocated and written into `output`.
void RingAllreduce(float* data, size_t length, float** output);

// The ring allgather. The lengths of the data chunks passed to this function
// may differ across different devices. The output memory will be allocated and
// written into `output`.
void RingAllgather(float* data, size_t length, float** output);
```

The interface is simple and inflexible and is meant as a demonstration. The
code is fairly straightforward and the same technique can be integrated into
existing codebases in a variety of ways.
