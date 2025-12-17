# NCCL Multi-Node Multi-GPU Communication Example (MPI-free)

This project demonstrates NCCL-based communication across multiple nodes and GPUs without requiring MPI. Each node runs a single process that manages multiple GPUs through multi-threading, with one dedicated thread per GPU.

## Building and Running

```bash
# Create and enter build directory
mkdir -p build && cd build

# Configure CMake project
cmake ..

# Compile
cmake --build .

# Run on the master node
./nccl_multi_node_demo --rank 0 --nproc 2 --port [communication_port] --size [data_size]

# Run on worker nodes
./nccl_multi_node_demo --rank 1 --nproc 2 --master [master_node_IP] --port [communication_port] --size [data_size]
```



## Results

### Master Node Output

![Master Node Results](https://github.com/whitelok/nccl_allreduce_demo_without_mpi/blob/master/imgs/master.png?raw=true "Master Node Results")

### Worker Node Output

![Worker Node Results](https://github.com/whitelok/nccl_allreduce_demo_without_mpi/blob/master/imgs/worker.png?raw=true "Worker Node Results")


## my

```
export NCCL_ROOT_DIR=/workspace/nccl-latest
export LD_LIBRARY_PATH=$NCCL_ROOT_DIR/build/lib
export NCCL_TOPO_FILE=$NCCL_ROOT_DIR/topo/h100_topo_rdma.xml
export NCCL_GRAPH_DUMP_FILE=$NCCL_ROOT_DIR/topo/graph_dump.xml
export NCCL_DEBUG=TRACE
```

```
./main --rank 0 --nproc 2 --port 3333 --size 256
./main  --rank 1 --nproc 2 --master  172.20.0.20 --port 3333 --size 256
```


## rdma or tcp 传输

nccl通过**ncclTransportP2pSetup**完成数据通信链路的建立，还是以上节两机十六卡的环为例：

- 第一台机器的环
```sh
graph->intra: GPU/0 GPU/7 GPU/6 GPU/3 GPU/2 GPU/5 GPU/4 GPU/1
graph->inter: NET/0 NET/0
```

- 第二台机器的环
```sh
graph->intra: GPU/10 GPU/9 GPU/8 GPU/13 GPU/12 GPU/15 GPU/14 GPU/11
graph->inter: NET/0 NET/0
```

 

- ncclChannelPeer
首先介绍一下ncclPeer，ncclPeer保存了两个connector，对于rank 10，send负责和rank 9通信，recv负责和rank 1通信。后续为了方便表述，假设rank 10叫接收端，rank 1叫发送端。

```c++
struct ncclChannelPeer {
  struct ncclConnector send[NCCL_MAX_CONNS]; // send负责和rank 9通信
  struct ncclConnector recv[NCCL_MAX_CONNS]; // recv负责和rank 1通信
  int refCount;
};
```

```

/* Determine if we will use this transport for this peer and return connect
 * information for this peer */
static ncclResult_t sendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex) {
  struct setupReq req = { 0 };

  send->conn.shared = req.shared = graph || connIndex == 0 ? 0 : ncclParamNetSharedBuffers() != -2 ? ncclParamNetSharedBuffers() : 1;
  req.channelId = channelId;
  req.connIndex = connIndex;

  int proxyRank;
  int64_t netId;
  NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, peerInfo->rank, &netId, &req.netDev, &proxyRank));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->rank, netId, 1, &req.useGdr));
  send->conn.flags |= req.useGdr ? NCCL_DIRECT_NIC : 0;
  if (!req.useGdr && connIndex == 0) comm->useGdr = 0;
  if (proxyRank != myInfo->rank && connIndex == 0) comm->useNetPXN = true;

  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_NET, 1, proxyRank, &send->proxyConn));
  req.tpLocalRank = comm->topParentLocalRanks[comm->localRank];
  req.tpRank = comm->topParentRanks[myInfo->rank];
  req.tpRemoteRank = comm->topParentRanks[peerInfo->rank];
  NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgSetup, &req, sizeof(req), NULL, 0));

  if (proxyRank == myInfo->rank) {
    INFO(NCCL_INIT|NCCL_NET,"Channel %02d/%d : %d[%d] -> %d[%d] [send] via NET/%s/%d%s%s%s", channelId, connIndex, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, comm->ncclNet->name, req.netDev,
        req.useGdr ? "/GDRDMA" : "", req.useGdr==ncclTopoGdrModePci ? "(PCI)" : "",
        req.shared ? "/Shared" : "");
  } else {
    INFO(NCCL_INIT|NCCL_NET,"Channel %02d/%d : %d[%d] -> %d[%d] [send] via NET/%s/%d(%d)%s%s%s", channelId, connIndex, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, comm->ncclNet->name, req.netDev,
        proxyRank,
        req.useGdr ? "/GDRDMA" : "", req.useGdr==ncclTopoGdrModePci ? "(PCI)" : "",
        req.shared ? "/Shared" : "");
  }
  *((int*)connectInfo) = comm->topParentRanks[proxyRank];
  memcpy((uint8_t*)connectInfo + sizeof(ncclNetHandle_t), &req.useGdr, sizeof(int));
  return ncclSuccess;
}
```

# socket rdma netDev


```
ncclNet_t ncclNetIb = {
  "IB",
  ncclIbInit,
  ncclIbDevices,
  ncclIbGetProperties,
  ncclIbListen,
  ncclIbConnect,
  ncclIbAccept,
  ncclIbRegMr,
  ncclIbRegMrDmaBuf,
  ncclIbDeregMr,
  ncclIbIsend,
  ncclIbIrecv,
  ncclIbIflush,
  ncclIbTest,
  ncclIbCloseSend,
  ncclIbCloseRecv,
  ncclIbCloseListen,
  NULL /* getDeviceMr */,
  NULL /* irecvConsumed */,
  ncclIbMakeVDevice
};
```


```
ncclNet_t ncclNetSocket = {
  "Socket",
  ncclNetSocketInit,
  ncclNetSocketDevices,
  ncclNetSocketGetProperties,
  ncclNetSocketListen,
  ncclNetSocketConnect,
  ncclNetSocketAccept,
  ncclNetSocketRegMr,
  NULL, // No DMA-BUF support
  ncclNetSocketDeregMr,
  ncclNetSocketIsend,
  ncclNetSocketIrecv,
  ncclNetSocketIflush,
  ncclNetSocketTest,
  ncclNetSocketClose,
  ncclNetSocketClose,
  ncclNetSocketCloseListen,
  NULL /* getDeviceMr */,
  NULL /* irecvConsumed */,
  NULL /* mergeDevices */
};
```
+  ncclNetSocketIrecv
```
(gdb) bt
#0  ncclNetSocketIrecv (recvComm=0x0, n=0, data=0x7ffff7f00b98, sizes=0x1, tags=0x8000000000000000, mhandles=0x7ffff7f871a0 <ncclParamNetOptionalRecvCompletion()::cache>, 
    phandles=0x7fffb0f4ab50, request=0x7fff83f870d0) at transport/net_socket.cc:655
#1  0x00007ffff7e62b5e in recvProxyProgress (proxyState=0x7ffff010b1b0, args=0x7fff83f87018) at transport/net.cc:1367
#2  0x00007ffff7dd2f3c in progressOps (proxyState=0x7ffff010b1b0, state=0x7ffff010b2f8, opStart=0x7fff83f87018, idle=0x7fffb0f4add4) at proxy.cc:750
#3  0x00007ffff7dd3c10 in ncclProxyProgress (proxyState_=0x7ffff010b1b0) at proxy.cc:931
#4  0x00007ffff7cc9609 in start_thread (arg=<optimized out>) at pthread_create.c:477
#5  0x00007ffff79f1353 in clone () at ../sysdeps/unix/sysv/linux/x86_64/clone.S:95
```
```
gdb) bt
#0  sendProxyProgress (proxyState=0x0, args=0x0) at transport/net.cc:1108
#1  0x00007ffff7dd2f3c in progressOps (proxyState=0x7ffff010aa20, state=0x7ffff010ab68, opStart=0x7fff83f87018, idle=0x7fffb174bdd4) at proxy.cc:750
#2  0x00007ffff7dd3c10 in ncclProxyProgress (proxyState_=0x7ffff010aa20) at proxy.cc:931
```

## 集合通信原语的实现

在完成设备的Communicator初始化后，就可以调用集合通信的相关原语。在这里我们以Allreduce为例，分析集合通信原语的实现逻辑。

```C++
NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
    sendbuff, recvbuff, count, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}
```

可以看到，调用`ncclAllReduce`时，所有的变量被存到一个`ncclInfo` 结构体中，然后通过`ncclEnqueueCheck` 将这个结构体插入到队列中。

**`ncclEnqueueCheck`** 执行了以下操作：

- `ncclSaveKernel `：

  对将要执行的操作进行一些准备工作。主要是调用`computeColl()`计算`ncclProxyArgs` 这个结构体中的信息。这个结构体将被用于初始化Proxy。并且这个函数也计算了将要launch的 CUDA kernel的参数。

- `ncclBarrierEnqueue` `ncclBarrierEnqueueWait`：

  launch所有的 CUDA kernel，通过插入barrier的方式设置kernel之间的依赖关系。最后，通过`ncclProxyStart()` 启动Proxy线程。

AllReduce的具体执行逻辑：`all_reduce.h` 中，有9种实现。

```
三种算法：NCCL_ALGO_RING, NCCL_ALGO_TREE, NCCL_ALGO_COLLNET
三种协议：NCCL_PROTO_SIMPLE, NCCL_PROTO_LL, NCCL_PROTO_LL128
```

```C++
// 以NCCL_ALGO_RING, NCCL_PROTO_SIMPLE的实现为例
// 首先从args这个结构体中获取当前操作所需要的参数，例如当前的channel，数据传输的chunk size等。
// 实例化一个ncclPrimitives类
ncclPrimitives<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, T, 1, 1, 1, FUNC> prims(tid, nthreads, &ring->prev, &ring->next, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);
```

- 一个非常重要的类：`ncclPrimitives` 。这个类实现了各类通信原语。


```
#12 0x00007fb6d246723d in ?? () from /usr/local/cuda/lib64/libcudart.so.10.0
#13 0x00007fb6d24672c7 in ?? () from /usr/local/cuda/lib64/libcudart.so.10.0
#14 0x00007fb6d249b3c5 in cudaLaunchKernel () from /usr/local/cuda/lib64/libcudart.so.10.0
#15 0x00007fb60cb15695 in ncclBarrierEnqueueWait (comm=0x7fb60a43a1a0) at enqueue.cc:195
#16 0x00007fb60cb15b0f in ncclEnqueueCheck (info=info@entry=0x7fb67c313420) at enqueue.cc:460
#17 0x00007fb60cb25f90 in ncclAllReduce (sendbuff=0x7faee6cc9e00, recvbuff=<optimized out>,
    count=<optimized out>, datatype=ncclFloat16, op=<optimized out>, comm=<optimized out>,
    stream=0x7fb60a422fe0) at collectives/all_reduce.cc:1
```

单个rank

```
NCCLCHECK(ncclLaunchOneRank(info->recvbuff, info->sendbuff, info->count, opDev, info->datatype, info->stream));
```
多个rank

```
 ncclTaskCollSorterInsert(&planner->collSorter, t, t->trafficBytes);
```
```
(gdb) bt
#0  0x00007ffff7d78150 in ncclTaskCollSorterInsert(ncclTaskCollSorter*, ncclTaskColl*, unsigned long)@plt () from /workspace/nccl-latest/build/lib/libnccl.so.2
#1  0x00007ffff7db0baa in taskAppend (comm=0x7ffff00019f0, info=0x7ffff6ce97b0) at enqueue.cc:2544
#2  0x00007ffff7db0faa in ncclEnqueueCheck (info=0x7ffff6ce97b0) at enqueue.cc:2596
```


```
Thread 7 "main" hit Breakpoint 8, ncclGroupEndInternal (simInfo=0x0) at group.cc:654
654       if (hasCommHead || !ncclIntruQueueEmpty(&groupJob->asyncJobs) || ncclGroupCommPreconnectHead != nullptr) {
(gdb) n
```
#   ncclLaunchKernel
```
(gdb) bt
#0  groupLaunch (job_=0x320000000b, simInfo=0x2) at group.cc:447
#1  0x00007ffff7d9fd5e in ncclGroupEndInternal (simInfo=0x0) at group.cc:694
#2  0x00007ffff7db0d7f in ncclEnqueueCheck (info=0x7ffff5ce77b0) at enqueue.cc:2603
```

```
 ncclLaunchKernel(struct ncclComm* comm, struct ncclKernelPlan* plan)
  CUCHECKGOTO(cuLaunchKernel(fn, grid.x, grid.y, grid.z, block.x, block.y, block.z, smem, launchStream, nullptr, extra), ret, do_return);
  CUCHECKGOTO(cuLaunchKernelEx(&launchConfig, fn, nullptr, extra), ret, do_return);
```