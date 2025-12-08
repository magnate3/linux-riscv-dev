 
```
export NCCL_TOPO_FILE=$NCCL_ROOT_DIR/topo/nvlink_h100.xml
export NCCL_GRAPH_DUMP_FILE=$NCCL_ROOT_DIR/topo/graph_dump.xml
export GPU_DEV_NUM=5
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=ALL
```
 
 
```
root@ubuntu:/pytorch# make mpi
/usr/local/cuda/bin/nvcc -ccbin g++ -m64 -I./ -I/usr/local/mpi/include/ -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o allreduce_2comms example_allreduce_2comms.cu -lnccl -lmpi -L/usr/local/mpi/lib/
root@ubuntu:/pytorch# mpirun -np 1 --allow-run-as-root   allreduce_2comms 
The local rank is: 0
ubuntu:2061:2061 [0] NCCL INFO Bootstrap : Using eno1:172.22.116.89<0>
ubuntu:2061:2061 [0] NCCL INFO cudaDriverVersion 12080
ubuntu:2061:2061 [0] NCCL INFO NCCL version 2.22.3+cuda12.5
ubuntu:2061:2061 [0] NCCL INFO Plugin Path : /opt/hpcx/nccl_rdma_sharp_plugin/lib/libnccl-net.so
ubuntu:2061:2061 [0] NCCL INFO P2P plugin IBext_v8
ubuntu:2061:2061 [0] NCCL INFO NET/IB : No device found.
ubuntu:2061:2061 [0] NCCL INFO NET/IB : No device found.
ubuntu:2061:2061 [0] NCCL INFO NET/Socket : Using [0]eno1:172.22.116.89<0> [1]ztyou3pbk2:192.168.193.155<0>
ubuntu:2061:2061 [0] NCCL INFO Using network Socket
ubuntu:2061:2061 [0] NCCL INFO ncclCommInitRank comm 0x564300ddbb50 rank 0 nranks 1 cudaDev 0 nvmlDev 0 busId 1000 commId 0x506d690a4886b5c5 - Init START
ubuntu:2061:2061 [0] NCCL INFO comm 0x564300ddbb50 rank 0 nRanks 1 nNodes 1 localRanks 1 localRank 0 MNNVL 0
ubuntu:2061:2061 [0] NCCL INFO Channel 00/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 01/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 02/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 03/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 04/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 05/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 06/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 07/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 08/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 09/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 10/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 11/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 12/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 13/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 14/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 15/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 16/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 17/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 18/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 19/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 20/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 21/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 22/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 23/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 24/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 25/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 26/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 27/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 28/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 29/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 30/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 31/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Trees [0] -1/-1/-1->0->-1 [1] -1/-1/-1->0->-1 [2] -1/-1/-1->0->-1 [3] -1/-1/-1->0->-1 [4] -1/-1/-1->0->-1 [5] -1/-1/-1->0->-1 [6] -1/-1/-1->0->-1 [7] -1/-1/-1->0->-1 [8] -1/-1/-1->0->-1 [9] -1/-1/-1->0->-1 [10] -1/-1/-1->0->-1 [11] -1/-1/-1->0->-1 [12] -1/-1/-1->0->-1 [13] -1/-1/-1->0->-1 [14] -1/-1/-1->0->-1 [15] -1/-1/-1->0->-1 [16] -1/-1/-1->0->-1 [17] -1/-1/-1->0->-1 [18] -1/-1/-1->0->-1 [19] -1/-1/-1->0->-1 [20] -1/-1/-1->0->-1 [21] -1/-1/-1->0->-1 [22] -1/-1/-1->0->-1 [23] -1/-1/-1->0->-1 [24] -1/-1/-1->0->-1 [25] -1/-1/-1->0->-1 [26] -1/-1/-1->0->-1 [27] -1/-1/-1->0->-1 [28] -1/-1/-1->0->-1 [29] -1/-1/-1->0->-1 [30] -1/-1/-1->0->-1 [31] -1/-1/-1->0->-1
ubuntu:2061:2061 [0] NCCL INFO P2P Chunksize set to 131072
ubuntu:2061:2061 [0] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 32 p2p channels per peer
ubuntu:2061:2061 [0] NCCL INFO CC Off, Multi-GPU CC Off, workFifoBytes 1048576
ubuntu:2061:2061 [0] NCCL INFO TUNER/Plugin: Failed to find ncclTunerPlugin_v3 symbol.
ubuntu:2061:2061 [0] NCCL INFO TUNER/Plugin: Failed to find ncclTunerPlugin_v2 symbol, using internal tuner instead.
ubuntu:2061:2061 [0] NCCL INFO ncclCommInitRank comm 0x564300ddbb50 rank 0 nranks 1 cudaDev 0 nvmlDev 0 busId 1000 commId 0x506d690a4886b5c5 - Init COMPLETE
ubuntu:2061:2061 [0] NCCL INFO Init timings: rank 0 nranks 1 total 0.11 (kernels 0.08, bootstrap 0.03, allgathers 0.00, topo 0.00, graphs 0.00, connections 0.00, rest 0.00)
ubuntu:2061:2061 [0] NCCL INFO Using network Socket
ubuntu:2061:2061 [0] NCCL INFO ncclCommInitRank comm 0x5643049c18b0 rank 0 nranks 1 cudaDev 0 nvmlDev 0 busId 1000 commId 0x3afd41e9de60b575 - Init START
ubuntu:2061:2061 [0] NCCL INFO comm 0x5643049c18b0 rank 0 nRanks 1 nNodes 1 localRanks 1 localRank 0 MNNVL 0
ubuntu:2061:2061 [0] NCCL INFO Channel 00/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 01/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 02/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 03/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 04/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 05/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 06/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 07/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 08/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 09/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 10/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 11/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 12/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 13/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 14/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 15/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 16/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 17/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 18/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 19/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 20/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 21/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 22/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 23/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 24/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 25/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 26/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 27/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 28/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 29/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 30/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Channel 31/32 :    0
ubuntu:2061:2061 [0] NCCL INFO Trees [0] -1/-1/-1->0->-1 [1] -1/-1/-1->0->-1 [2] -1/-1/-1->0->-1 [3] -1/-1/-1->0->-1 [4] -1/-1/-1->0->-1 [5] -1/-1/-1->0->-1 [6] -1/-1/-1->0->-1 [7] -1/-1/-1->0->-1 [8] -1/-1/-1->0->-1 [9] -1/-1/-1->0->-1 [10] -1/-1/-1->0->-1 [11] -1/-1/-1->0->-1 [12] -1/-1/-1->0->-1 [13] -1/-1/-1->0->-1 [14] -1/-1/-1->0->-1 [15] -1/-1/-1->0->-1 [16] -1/-1/-1->0->-1 [17] -1/-1/-1->0->-1 [18] -1/-1/-1->0->-1 [19] -1/-1/-1->0->-1 [20] -1/-1/-1->0->-1 [21] -1/-1/-1->0->-1 [22] -1/-1/-1->0->-1 [23] -1/-1/-1->0->-1 [24] -1/-1/-1->0->-1 [25] -1/-1/-1->0->-1 [26] -1/-1/-1->0->-1 [27] -1/-1/-1->0->-1 [28] -1/-1/-1->0->-1 [29] -1/-1/-1->0->-1 [30] -1/-1/-1->0->-1 [31] -1/-1/-1->0->-1
ubuntu:2061:2061 [0] NCCL INFO P2P Chunksize set to 131072
ubuntu:2061:2061 [0] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 32 p2p channels per peer
ubuntu:2061:2061 [0] NCCL INFO CC Off, Multi-GPU CC Off, workFifoBytes 1048576
ubuntu:2061:2061 [0] NCCL INFO ncclCommInitRank comm 0x5643049c18b0 rank 0 nranks 1 cudaDev 0 nvmlDev 0 busId 1000 commId 0x3afd41e9de60b575 - Init COMPLETE
ubuntu:2061:2061 [0] NCCL INFO Init timings: rank 0 nranks 1 total 0.01 (kernels 0.00, bootstrap 0.00, allgathers 0.00, topo 0.00, graphs 0.00, connections 0.00, rest 0.00)
comm0 socket addr 172.22.116.89<45725> 
comm1 socket addr 172.22.116.89<60517> 
[MPI Rank 0] Success 
[Rank MPI 0] Success 
ubuntu:2061:2061 [0] NCCL INFO comm 0x564300ddbb50 rank 0 nranks 1 cudaDev 0 busId 1000 - Destroy COMPLETE
ubuntu:2061:2061 [0] NCCL INFO comm 0x5643049c18b0 rank 0 nranks 1 cudaDev 0 busId 1000 - Destroy COMPLETE
```

两个listen socket

```
comm0 socket addr 172.22.116.89<45725> 
comm1 socket addr 172.22.116.89<60517> 
```

```
NCCL INFO comm 0x564300ddbb50 rank 0 nRanks 1 nNodes 1 localRanks 1 localRank 0 MNNVL 0
```