
```
 mpic++ -std=c++11 -I/usr/local/mpi/include -I. -I/usr/local/cuda/include mpi-allreduce-ring.c -o allreduce
```


```
mpirun --np 4 ./allreduce 
Allreduce benchmark
max size=4, size=4
rank=2 src index=0 has value  8
rank=2 src index=1 has value  9
rank=2 src index=2 has value 10
rank=2 src index=3 has value 11
Allreduce benchmark
max size=4, size=4
rank=3 src index=0 has value 12
rank=3 src index=1 has value 13
rank=3 src index=2 has value 14
rank=3 src index=3 has value 15
Allreduce benchmark
max size=4, size=4
rank=0 src index=0 has value  0
rank=0 src index=1 has value  1
rank=0 src index=2 has value  2
rank=0 src index=3 has value  3
rank=3 dst index=0 has value 24
rank=3 dst index=1 has value 28
rank=3 dst index=2 has value 32
rank=3 dst index=3 has value 36
Ring Algo allreduce successful
Allreduce benchmark
max size=4, size=4
rank=1 src index=0 has value  4
rank=1 src index=1 has value  5
rank=1 src index=2 has value  6
rank=1 src index=3 has value  7
rank=1 dst index=0 has value 24
rank=1 dst index=1 has value 28
rank=1 dst index=2 has value 32
rank=1 dst index=3 has value 36
Ring Algo allreduce successful
rank=2 dst index=0 has value 24
rank=2 dst index=1 has value 28
rank=2 dst index=2 has value 32
rank=2 dst index=3 has value 36
Ring Algo allreduce successful
rank=0 dst index=0 has value 24
rank=0 dst index=1 has value 28
rank=0 dst index=2 has value 32
rank=0 dst index=3 has value 36
Ring Algo allreduce successful
```