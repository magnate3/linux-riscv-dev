
# pytorch cpp

[使用Cpp扩展自定义进程组后端](https://github.com/pytorch/tutorials/blob/main/intermediate_source/process_group_cpp_extension_tutorial.rst)      
[rstorch](https://github.com/Kylejeong2/rstorch/tree/c0958b9593d23e647806cefddd7876d82b5bbe42)

# cuda   
```
 root@ubuntu:/pytorch/cuda2/cuda/nccl# ls /usr/local/cuda/lib64/libcudart.so
/usr/local/cuda/lib64/libcudart.so
root@ubuntu:/pytorch/cuda2/cuda/nccl# ls /usr/local/cuda/include
CL                           cuda_occupancy.h            curand_lognormal.h                   host_defines.h                            nvml.h
 
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