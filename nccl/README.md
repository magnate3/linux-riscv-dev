
# pytorch cpp

[使用Cpp扩展自定义进程组后端](https://github.com/pytorch/tutorials/blob/main/intermediate_source/process_group_cpp_extension_tutorial.rst)      
[rstorch](https://github.com/Kylejeong2/rstorch/tree/c0958b9593d23e647806cefddd7876d82b5bbe42)


```
 python3 -c 'import torch;print(torch.utils.cmake_prefix_path())' 
root@ubuntu:/# find ./ -name torch.h
./usr/local/lib/python3.8/dist-packages/torch/include/torch/csrc/api/include/torch/torch.h
root@ubuntu:/# find ./ -name ProcessGroupMPI.hpp
./usr/local/lib/python3.8/dist-packages/torch/include/torch/csrc/distributed/c10d/ProcessGroupMPI.hpp
root@ubuntu:/# 
```

```

 
 
 c++  -std=c++17 -lm -ldl dist-mnist.cpp -o dist-mnist -pthread -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -g -fwrapv -O2 -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -fPIC -I/usr/local/lib/python3.8/dist-packages/torch/include -I/usr/local/lib/python3.8/dist-packages/torch/include/torch/csrc/api/include -I/usr/local/lib/python3.8/dist-packages/torch/include/TH -I/usr/local/lib/python3.8/dist-packages/torch/include/THC -I/usr/include/python3.8 -I/usr/local/lib/python3.8/dist-packages/torch/include/torch/csrc/distributed

```

```

 
 c++  -std=c++17 -lm -ldl dist-mnist.cpp -o dist-mnist -pthread -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -g -fwrapv -O2 -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -fPIC -I/usr/local/lib/python3.8/dist-packages/torch/include -I/usr/local/lib/python3.8/dist-packages/torch/include/torch/csrc/api/include  -I/usr/include/python3.8 -I/usr/local/lib/python3.8/dist-packages/torch/include/torch/csrc/distributed

```

# cuda   
```
 root@ubuntu:/pytorch/cuda2/cuda/nccl# ls /usr/local/cuda/lib64/libcudart.so
/usr/local/cuda/lib64/libcudart.so
root@ubuntu:/pytorch/cuda2/cuda/nccl# ls /usr/local/cuda/include
CL                           cuda_occupancy.h            curand_lognormal.h                   host_defines.h                            nvml.h
 
```

```
export LD_LIBRARY_PATH="/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH"
export PATH="/usr/local/cuda-11.6/bin:$PATH"
 export CUDA_INCLUDE_DIRS=/usr/local/cuda-11.6/include
  export CUDA_CUDART_LIBRARY=/usr/local/cuda-11.6/lib64  
```

#  rankhasroot 
```
cat rankhasroot.cu 
/*
no gpu was used. if it was, itll be need 4
*/
#include <iostream>

// This function determines if a given rank is the root of its group.
// It divides nRanks among nIds groups. The groups may have different sizes
// if nRanks isn't exactly divisible by nIds.
bool rankHasRoot(const int rank, const int nRanks, const int nIds) {
  // rmr: remainder of total ranks divided by the number of IDs.
  const int rmr = nRanks % nIds;
  // rpr: quotient, representing the base number of ranks per ID.
  const int rpr = nRanks / nIds;
  // rlim: threshold rank that separates groups with one extra rank.
  const int rlim = rmr * (rpr + 1);

  // For the first rmr groups, each group has (rpr+1) ranks.
  if (rank < rlim) {
    // The first rank in the group (i.e., a "root") will be at an index that's a multiple of (rpr+1)
    return (rank % (rpr + 1)) == 0;
  } else {
    // For the remaining groups, each group has rpr ranks.
    return ((rank - rlim) % rpr) == 0;
  }
}

int main() {
  // Example configuration:
  // Total number of ranks and the number of groups (IDs)
  const int nRanks = 16;
  const int nIds = 4;

  std::cout << "Total Ranks: " << nRanks << ", Number of IDs (groups): " << nIds << "\n";
  std::cout << "Determining which rank is the group root:\n";

  // Loop through each rank and use the function to check if it's a root.
  for (int rank = 0; rank < nRanks; ++rank) {
    bool hasRoot = rankHasRoot(rank, nRanks, nIds);
    std::cout << "Rank " << rank << (hasRoot ? " is a root." : " is NOT a root.") << "\n";
  }

  return 0;
}
```

```
/pytorch/cuda2/cuda/nccl# ./rankhasroot 
Total Ranks: 16, Number of IDs (groups): 4
Determining which rank is the group root:
Rank 0 is a root.
Rank 1 is NOT a root.
Rank 2 is NOT a root.
Rank 3 is NOT a root.
Rank 4 is a root.
Rank 5 is NOT a root.
Rank 6 is NOT a root.
Rank 7 is NOT a root.
Rank 8 is a root.
Rank 9 is NOT a root.
Rank 10 is NOT a root.
Rank 11 is NOT a root.
Rank 12 is a root.
Rank 13 is NOT a root.
Rank 14 is NOT a root.
Rank 15 is NOT a root.
```


#  learnnccl
```
root@ubuntu:/pytorch/cuda2/learnnccl# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/lib
root@ubuntu:/pytorch/cuda2/learnnccl# make CUDA_HOME=/usr/local/cuda clean
rm -rf build
root@ubuntu:/pytorch/cuda2/learnnccl# make CUDA_HOME=/usr/local/cuda test 
```

```
./build/test/all_reduce_test 10 1 
# Using devices
Segmentation fault (core dumped)
```

# ring


构建prims对象

```cpp
__device__ Primitives(
    const int tid, const int nthreads, int const *recvPeers, int const *sendPeers,
    void const *inputBuf, void *outputBuf, uint64_t redOpArg, int group=0
):

Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0> prims
      (tid, nthreads, &ring->prev, &ring->next, args->sendbuff, args->recvbuff, args->redOpArg);
```

几步走

```cpp
// step 0: push data to next GPU
chunk = modRanks(ringIx + nranks-1);
offset = calcOffset(chunk);
nelem = min(realChunkSize, size-offset);
prims.send(offset, nelem);

// k-2 steps: reduce and copy to next GPU
for (int j=2; j<nranks; ++j) {
  chunk = modRanks(ringIx + nranks-j);
  offset = calcOffset(chunk);
  nelem = min(realChunkSize, size-offset);
  prims.recvReduceSend(offset, nelem);
}

// step k-1: reduce this buffer and data, which will produce the final
// result that we store in this data and push to the next GPU
chunk = ringIx + 0;
offset = calcOffset(chunk);
nelem = min(realChunkSize, size-offset);
prims.directRecvReduceCopySend(offset, offset, offset, nelem, /*postOp=*/true);

// k-2 steps: copy to next GPU
for (int j=1; j<nranks-1; ++j) {
  chunk = modRanks(ringIx + nranks-j);
  offset = calcOffset(chunk);
  nelem = min(realChunkSize, size-offset);
  prims.directRecvCopySend(offset, offset, nelem);
}

// Make final copy from buffer to dest.
chunk = modRanks(ringIx + 1);
offset = calcOffset(chunk);
nelem = min(realChunkSize, size-offset);
prims.directRecv(offset, nelem);
```
