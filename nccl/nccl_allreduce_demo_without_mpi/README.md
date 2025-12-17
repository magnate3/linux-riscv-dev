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