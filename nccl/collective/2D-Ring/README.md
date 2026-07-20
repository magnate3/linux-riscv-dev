
#  RingAllReduce

[分布式训练 – 第4篇 - 分布式训练常用的网络结构及集合通信拓扑算法](https://github.com/jianye0428/jianye0428.github.io/blob/3324cdbd2ea239e35d806a6dbc4cc83510bd54db/posts/distributedtraining_4/index.md)   

  - Ring AllReduce 的最佳组合是 ScatterReduce + AllGather；   
  - 2D-Ring AllReduce = 主机内 ringAllReduce/Ring Reduce +主机间 RingAllReduce + 主机内Broadcast；   
  - 2D-Torus AllReduce = 主机内 Ring ReduceScatter + 主机间N个Ring AllReduce + 主机内Ring AllGather；   
  - 2D-Mesh AllReduce = 主机内Ring AllReduce + 主机间N个Ring AllReduce;    
Ring  AllReduce适合主机内互联Ring的情况使用，2D-RiAllReduce适合一台服务器配置了一张网卡的异构网络场景，2D-Torus AllReduce与2D-Mesh AllReduce适合一台服务器配置了2/4/8张网卡的异构网络场景。


+ 经典RingAllReduce    
```
class RingAllReduce:
    def
    _init_(self, rank, world_size):
     self.rank = rank
    self.world_size = world_size
    self.left_neighbor = (rank - 1) % world_size
     self.right_neighbor = (rank + l) % world size 
    def execute(self, local_data) :
    #阶段l:Scatter-Reduce
    chunks = self.split_data(local _data)
    result = [None] * self.world_size
    for step in range(self.world_size - 1):
        send_chunk_idx = (self.rank( - step)% self.world_size
        recv_chunk_idx = (self.rank - step 0 - 1) % self.world_size
        #发送指定分块，接收相邻分块
        send_data = chunks [send_chunk_idx]
        recv_data = self.send_recv(send_data, self.right_neighbor.self.left_neighbor)
        #累加接收到的分块
        result[recv_chunk_idx] += recv_data
        if result[recv_chunk_idx] is None:
            result[recv_chunk_idx] = recv_data
        else:
	        result[recv_chunk_idx] += recv_data
    # 阶段2: All-Gather
    for step in range(self.world_size - 1):
        send_chunk_idx=(self.rank- step+ 1) % self.world_size
        recv_chunk_idx = (self.rank - step) % self.world_size
        send_data = result[send_chunk_idx]
        recv_data = self.send_recv(send_data, self.right_neighbor,self.left_neighbor)
        result[recv_chunk_idx] = recv_data
    return self.concat_results(result)


```

+  TwoDRingAllReduce
```
class TwoDRingAllReduce:
    def _init_（self,rank,world_size）:
        self.rank=rank
        self.world_size= world_size
        self.grid_rows=int（math.sqrt（world_size))
        self.grid_cols=world_size//self.grid_rows
        #计算二维坐标
        self.row_idx=rank//self.grid_cols
        self.col_idx=rankself.grid_cols
    def get_row_ring_ranks(self):
        “获取行内环拓扑节点
        start=self.row_idx* self.grid_cols
        return list（range(start,start+self.grid_cols))
    def get_col_ring_ranks(self):
        …获取列内环拓扑节点“
        return [self.col_idx+i*self.grid_cols for i in range(self.grid_rows）]
    def execute(self,local_data):
        #阶段1：行内Reduce-Scatter
        row_ranks=self.get_row_ring_ranks()
        row_result=self.row_ring_all_reduce(local_data,row_ranks)
        #阶段2：列内ALL-Reduce
        col_ranks= self.get_col_ring_ranks（)
        col_result=self.col_ring_all_reduce(row_result,col_ranks)
        #阶段3：行内ALl-Gather
        final_result= self.row_ring_all _gather(col_result,row_ranks)
        return final_result

```

> ## Ring算法使用的NCCL通信原语

src/device/all_reduce.h:
```
template<typename T, typename RedOp, typename Proto>
__device__ void runRing(int tid, int nthreads, ncclDevWorkColl* work) {
  ncclRing *ring = &ncclShmem.channel.ring;
  const int nranks = ncclShmem.comm.nRanks;

  // Reduce-Scatter阶段
  for (int j = 2; j < nranks; ++j) {
    prims.directRecvReduceDirectSend(offset, offset, nelem);
  }

  // AllGather阶段
  for (int j = 1; j < nranks - 1; ++j) {
    prims.directRecvCopyDirectSend(offset, offset, nelem);
  }
}
```

原语组合流程:    
```
// 1. 创建对称通信原语（1个输入peer，1个输出peer）
Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0> prims(
  tid, nthreads, &ring->prev, &ring->next, ...  // prev和next形成环
);

// 2. Reduce-Scatter阶段
prims.directSend(offset, offset, nelem);  // 步骤0: 发送
for (int j = 2; j < nranks; ++j) {
  prims.directRecvReduceDirectSend(offset, offset, nelem);  // 接收+归约+发送
}
prims.directRecvReduceCopyDirectSend(offset, offset, nelem, true);  // 最后步含postOp

// 3. AllGather阶段
for (int j = 1; j < nranks - 1; ++j) {
  prims.directRecvCopyDirectSend(offset, offset, nelem);  // 接收+复制+发送
}
prims.directRecv(offset, nelem);  // 最后接收
```

> ## ring setupChannel

从当前rank为起点，将环写到userRanks。

```c++
/**
 * @brief 配置NCCL通道的环形拓扑结构（依赖initChannel初始化基础资源）
 *
 * 该函数是通道初始化的后续配置步骤，在initChannel完成基础资源分配后，
 * 基于全局环形rank列表（ringRanks），为当前rank个性化配置环形拓扑信息：
 * 1. 计算当前rank在环形中的相对索引（相对于rank 0的位置）；
 * 2. 生成以当前rank为起点的环形rank序列，便于后续环形通信时快速定位上下游节点。
 * NCCL的核心集合通信（如allreduce）依赖环形拓扑实现高效数据传输，此函数是环形通信的关键配置步骤。
 *
 * @param comm 已初始化的NCCL通信器指针（需确保通道已通过initChannel完成基础初始化）
 * @param channelId 要配置的通道ID（需与initChannel的目标通道一致）
 * @param rank 当前进程/设备的本地rank（在当前通信器中的标识）
 * @param nranks 通信器中的总rank数（环形拓扑的节点总数）
 * @param ringRanks 全局环形拓扑的rank列表（预定义的环形节点顺序，所有rank共享同一列表）
 * @return ncclResult_t NCCL状态码，ncclSuccess表示配置成功，其他码表示错误
 */
static ncclResult_t setupChannel(struct ncclComm* comm, int channelId, int rank, int nranks, int* ringRanks) {
  // 打印初始化日志（TRACE是NCCL的日志宏，记录当前rank和总rank数，用于调试）
  TRACE(NCCL_INIT, "rank %d nranks %d", rank, nranks);

  // 第一步：确保通道已完成基础初始化（调用之前分析的initChannel函数，分配peers、devPeers等资源）
  // 若通道未初始化则触发初始化，已初始化则直接返回成功（initChannel有幂等性）
  NCCLCHECK(initChannel(comm, channelId));

  // 获取当前通道的环形拓扑结构体（环形拓扑是NCCL数据传输的核心拓扑模式）
  struct ncclRing* ring = &comm->channels[channelId].ring;

  // 查找两个关键索引：
  // ixZero：全局环形列表（ringRanks）中rank 0的位置（环形拓扑的逻辑起点）
  // ixRank：全局环形列表（ringRanks）中当前rank的位置
  int ixZero = 0, ixRank = 0;
  for (int i = 0; i < nranks; i++) {
    if (ringRanks[i] == 0) ixZero = i;   // 定位rank 0在全局环形中的索引
    if (ringRanks[i] == rank) ixRank = i;// 定位当前rank在全局环形中的索引
  }

  /**
   * 计算当前rank在环形中的相对索引（ring->index）
   * 逻辑：以rank 0为环形起点，计算当前rank相对于起点的“环形距离”
   * - (ixRank - ixZero)：当前rank与rank 0的索引差（可能为负）
   * - +nranks：避免负数（确保计算结果非负）
   * - %nranks：确保索引落在[0, nranks-1]范围内（符合环形拓扑的循环特性）
   * 示例：nranks=4，ixZero=1（rank0在全局列表索引1），ixRank=3（当前rank在全局列表索引3）
   * 计算：(3-1 +4) %4 = 2 → 当前rank在环形中的相对索引为2
   */
  ring->index = (ixRank - ixZero + nranks) % nranks;

  /**
   * 生成以当前rank为起点的环形rank序列（ring->userRanks）
   * 逻辑：将全局环形列表（ringRanks）按“当前rank的位置”重新排序，使序列起点为当前rank
   * 目的：后续通信时，可通过索引直接定位上下游节点（如index+1=下一个节点，index-1=上一个节点）
   * 示例：全局ringRanks=[2,0,3,1]，当前rank=3（ixRank=2），nranks=4
   * 循环i=0→3：
   * i=0：(0+2)%4=2 → ringRanks[2]=3（当前rank自身）
   * i=1：(1+2)%4=3 → ringRanks[3]=1（下一个节点）
   * i=2：(2+2)%4=0 → ringRanks[0]=2（下下个节点）
   * i=3：(3+2)%4=1 → ringRanks[1]=0（上一个节点）
   * 最终userRanks=[3,1,2,0]（以当前rank为起点的环形序列）
   */
  for (int i = 0; i < nranks; i++) {
    ring->userRanks[i] = ringRanks[(i + ixRank) % nranks];
  }

  // 环形拓扑配置完成
  return ncclSuccess;
}
```

> ## 快慢节点通信问题

单边通信能较好解决“慢节点”问题：它将 同步的“通信握手” 变成了 异步的“内存操作” 。在双边通信中，一个“快节点”的Send操作必须等待“慢节点”准备好Recv，从而被强制阻塞。而在单边通信中，“快节点”只需知道目标地址，就可以通过Put操作直接将数据写入“慢节点”的内存，然后立即返回执行后续任务。数据写入后，“慢节点”何时去处理它，并不会反过来阻塞“快节点”的进度。


> ## 通信和计算重叠

  
```
cudaStream_t computeStream, commStream;
cudaStreamCreate(&computeStream);
cudaStreamCreate(&commStream);

// 分块处理
for (int chunk = 0; chunk < numChunks; chunk++) {
    // 异步通信
    ncclAllReduce(gradients[chunk], gradients[chunk], chunkSize,
                 ncclFloat, ncclSum, comm, commStream);
    
    // 重叠的计算
    if (chunk > 0) {
        updateWeights<<<blocks, threads, 0, computeStream>>>(
            weights[chunk-1], gradients[chunk-1], chunkSize);
    }
}
```

# 2D ring  all reduce 参考

[第94篇 - NCCL通过环境变量指定拓扑算法修改指南](https://zhuanlan.zhihu.com/p/1983600291682223282)  

[第61篇 - NCCL集合通信常用拓扑：2D Mesh详解](https://zhuanlan.zhihu.com/p/1976054948421715577)   

 

+  在topoGetAlgoInfo()中强制算法选择    


```
┌─────────────────────────────────────────┐
│  getAlgoInfo() 算法选择                 │
│  ├─ initCollCostTable()                 │
│  ├─ updateCollCostTable()               │
│  ├─ topoGetAlgoInfo()  ← 核心选择逻辑   │
│  └─ 根据cost table选择最优algo/proto    │
└──────────────┬──────────────────────────┘
```

```
static ncclResult_t topoGetAlgoInfo(
    struct ncclComm* comm, struct ncclTaskColl* info, size_t nBytes,
    float** collCostTable, ncclSimInfo_t* simInfo
  ) {
  float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;

  // ===== 新增：检查是否强制使用2D-Ring =====
  int nodesPerDim = force2DRingTopo(comm);
  if (nodesPerDim > 0 && info->func == ncclFuncAllReduce) {
    // 强制使用2D-Ring算法
    info->algorithm = NCCL_ALGO_2DRING;
    info->protocol = NCCL_PROTO_SIMPLE;  // 2D-Ring默认使用SIMPLE协议
    
    INFO(NCCL_COLL, "Forcing 2D-Ring AllReduce (%dx%d grid) for %ld bytes", 
         nodesPerDim, nodesPerDim, nBytes);
    
    // 跳过cost table选择，直接设置channel/thread参数
    int nc = comm->nChannels;
    int nt = comm->maxThreads[NCCL_ALGO_RING][info->protocol];  // 复用Ring的线程配置
    int threadThreshold = comm->threadThresholds[NCCL_ALGO_RING][info->protocol];
    
    // Channel调优（简化版，可根据需要调整）
    while (nBytes < nc * nt * threadThreshold && nc >= 2) {
      nc--;
    }
    
    info->nMaxChannels = nc;
    info->nWarps = nt / WARP_SIZE;
    
    if (simInfo) simInfo->estimatedTime = 0.0f;  // 无法估算时间，设为0
    
    return ncclSuccess;
  }
  // ===== 新增结束 =====

  // 原有算法选择逻辑保持不变
  float minTime = 3600000000.0;
  int algorithm = info->algorithm = NCCL_ALGO_UNDEF;
  // ... 后续代码不变
}
```   

+ 跳过拓扑搜索 nccl/src/init.cc    
在ncclCommInitRankDev()中添加条件跳过：    

```
ncclCommInitAll/ncclCommInitRank()
  └─> ncclCommInitRankDev()
      └─> ncclTopoGetGraphs()
          └─> ncclTopoComputeSearch()  ← 这里执行拓扑搜索
```  

```
ncclResult_t ncclCommInitRankDev(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, 
                                  int myrank, int cudaDev) {
  // ... 前面的初始化代码 ...

  // ===== 新增：检查是否跳过拓扑搜索 =====
  int skip2DTopoSearch = ncclParamTopo2DRing();
  if (skip2DTopoSearch) {
    INFO(NCCL_INIT, "Skipping topology search due to NCCL_TOPO_2DRING=1");
    
    // 手动构建简化的2D拓扑图
    int nodesPerDim = ncclParamTopo2DNodesPerDim();
    if (nodesPerDim > 0) {
      NCCLCHECK(ncclTopoManual2DRingSetup(comm, nodesPerDim));
    } else {
      WARN("NCCL_TOPO_2DRING enabled but NCCL_TOPO_2D_NODES_PER_DIM not set");
      return ncclInvalidUsage;
    }
  } else {
    // 原有拓扑搜索逻辑
    NCCLCHECK(ncclTopoGetGraphs(comm));
    NCCLCHECK(ncclTopoComputeSearch(comm));
  }
  // ===== 新增结束 =====
  
  // ... 后续初始化代码 ...
}
```
+ 实现2D拓扑手动设置,在 nccl/src/graph/topo.cc新增函数手动构建2D-Ring拓扑：    

```
ncclResult_t ncclTopoManual2DRingSetup(struct ncclComm* comm, int nodesPerDim) {
  int nRanks = comm->nRanks;
  int nNodes = comm->nNodes;
  int ranksPerNode = nRanks / nNodes;
  
  if (nNodes != nodesPerDim * nodesPerDim) {
    WARN("2D-Ring: nodes %d != %dx%d", nNodes, nodesPerDim, nodesPerDim);
    return ncclInvalidUsage;
  }
  
  // 计算当前节点在2D网格中的位置
  int myNode = comm->node;
  int myLocalRank = comm->localRank;
  int gridX = myNode % nodesPerDim;
  int gridY = myNode / nodesPerDim;
  
  for (int c = 0; c < comm->nChannels; c++) {
    struct ncclChannel* channel = &comm->channels[c];
    NCCLCHECK(ncclCalloc(&channel->ring.userRanks, nRanks));
    
    // 构建2D-Ring的rank映射
    // 策略：先节点内排列，再按网格顺序排列节点
    int idx = 0;
    for (int ny = 0; ny < nodesPerDim; ny++) {
      for (int nx = 0; nx < nodesPerDim; nx++) {
        int nodeId = ny * nodesPerDim + nx;
        for (int lr = 0; lr < ranksPerNode; lr++) {
          channel->ring.userRanks[idx++] = nodeId * ranksPerNode + lr;
        }
      }
    }
    
    // 设置prev/next：2D-Ring特殊规则
    // 节点内：线性连接
    // 节点间：仅头GPU(localRank=0)参与跨节点ring
    int myGlobalRank = comm->rank;
    
    if (myLocalRank == 0) {
      // 头GPU：参与节点内+节点间通信
      // 节点内prev/next
      int nodeBasePrev = (myNode - 1 + nNodes) % nNodes;
      int nodeBaseNext = (myNode + 1) % nNodes;
      channel->ring.prev = nodeBasePrev * ranksPerNode + (ranksPerNode - 1);
      channel->ring.next = myNode * ranksPerNode + 1;
    } else if (myLocalRank == ranksPerNode - 1) {
      // 尾GPU：仅节点内通信
      channel->ring.prev = myNode * ranksPerNode + myLocalRank - 1;
      int nodeBaseNext = (myNode + 1) % nNodes;
      channel->ring.next = nodeBaseNext * ranksPerNode + 0;
    } else {
      // 中间GPU：仅节点内通信
      channel->ring.prev = myGlobalRank - 1;
      channel->ring.next = myGlobalRank + 1;
    }
    
    channel->ring.index = myGlobalRank;
    
    INFO(NCCL_INIT, 
         "Rank %d: 2D-Ring Ch%d setup (node %d[%d,%d], local %d, prev %d, next %d)",
         myGlobalRank, c, myNode, gridX, gridY, myLocalRank, 
         channel->ring.prev, channel->ring.next);
  }
  
  comm->topo2DRing = 1;
  comm->topo2DNodesPerDim = nodesPerDim;
  
  return ncclSuccess;
}
```


> ##  完整的2D拓扑映射实现
```
 完整的2D拓扑映射实现
// src/graph/topo.cc - 更完整的2D-Ring拓扑构建

ncclResult_t ncclTopoManual2DRingSetup(struct ncclComm* comm, int nodesPerDim) {
  int nRanks = comm->nRanks;
  int nNodes = comm->nNodes;
  int ranksPerNode = nRanks / nNodes;
  
  if (nNodes != nodesPerDim * nodesPerDim) {
    WARN("2D-Ring: nodes %d != %dx%d", nNodes, nodesPerDim, nodesPerDim);
    return ncclInvalidUsage;
  }
  
  // 计算当前节点在2D网格中的位置
  int myNode = comm->node;
  int myLocalRank = comm->localRank;
  int gridX = myNode % nodesPerDim;
  int gridY = myNode / nodesPerDim;
  
  for (int c = 0; c < comm->nChannels; c++) {
    struct ncclChannel* channel = &comm->channels[c];
    NCCLCHECK(ncclCalloc(&channel->ring.userRanks, nRanks));
    
    // 构建2D-Ring的rank映射
    // 策略：先节点内排列，再按网格顺序排列节点
    int idx = 0;
    for (int ny = 0; ny < nodesPerDim; ny++) {
      for (int nx = 0; nx < nodesPerDim; nx++) {
        int nodeId = ny * nodesPerDim + nx;
        for (int lr = 0; lr < ranksPerNode; lr++) {
          channel->ring.userRanks[idx++] = nodeId * ranksPerNode + lr;
        }
      }
    }
    
    // 设置prev/next：2D-Ring特殊规则
    // 节点内：线性连接
    // 节点间：仅头GPU(localRank=0)参与跨节点ring
    int myGlobalRank = comm->rank;
    
    if (myLocalRank == 0) {
      // 头GPU：参与节点内+节点间通信
      // 节点内prev/next
      int nodeBasePrev = (myNode - 1 + nNodes) % nNodes;
      int nodeBaseNext = (myNode + 1) % nNodes;
      channel->ring.prev = nodeBasePrev * ranksPerNode + (ranksPerNode - 1);
      channel->ring.next = myNode * ranksPerNode + 1;
    } else if (myLocalRank == ranksPerNode - 1) {
      // 尾GPU：仅节点内通信
      channel->ring.prev = myNode * ranksPerNode + myLocalRank - 1;
      int nodeBaseNext = (myNode + 1) % nNodes;
      channel->ring.next = nodeBaseNext * ranksPerNode + 0;
    } else {
      // 中间GPU：仅节点内通信
      channel->ring.prev = myGlobalRank - 1;
      channel->ring.next = myGlobalRank + 1;
    }
    
    channel->ring.index = myGlobalRank;
    
    INFO(NCCL_INIT, 
         "Rank %d: 2D-Ring Ch%d setup (node %d[%d,%d], local %d, prev %d, next %d)",
         myGlobalRank, c, myNode, gridX, gridY, myLocalRank, 
         channel->ring.prev, channel->ring.next);
  }
  
  comm->topo2DRing = 1;
  comm->topo2DNodesPerDim = nodesPerDim;
  
  return ncclSuccess;
}
```