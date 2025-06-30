

# make


```
apt install libgtest-dev
```

+ 使用cpu memeory    
```
-UUSE_CUDA -DCPU_MEMORY
```

+ 屏蔽rdma_test.cc编译（因为其依赖cudaMallocManaged or  cudaMalloc）      

```
rm rdma_test.cc 
```

#  export (不需要export)
```
export NCCL_IB_HCA=mlx5_1
export NCCL_SOCKET_IFNAME="enp61s0f1np1"
export NCCL_IB_GID_INDEX=3 
export NCCL_IB_PCI_RELAXED_ORDERING=1
CHUNK_SIZE_KB
```


#  config 

```
ibv_devices
    device                 node GUID
    ------              ----------------
    mlx5_0              c470bd0300aa1fc8
    mlx5_1              c470bd0300aa1fc9
```
mlx5_1处于工作状态，mlx5_0 出于down     

+ transport_config.h     
```
UCCL_PARAM(ROCE_GID_IDX, "ROCE_GID_IDX", 3);
UCCL_PARAM(RCMode, "RCMODE", true);
```

+ azure_perm_traffic/transport_config.h   

```
static constexpr bool ROCE_NET = true;
static char const* IB_DEVICE_NAME_PREFIX = "mlx5_";
static std::string SINGLE_CTRL_NIC("enp61s0f1np1");
static constexpr uint8_t DEVNAME_SUFFIX_LIST[8] = {0, 1, 0, 0, 0, 0, 0, 0};
static constexpr uint8_t NUM_DEVICES = 1;
static constexpr double LINK_BANDWIDTH = 100.0 * 1e9 / 8;  // 1m00Gbps
```
IB_DEVICE_NAME_PREFIX +  DEVNAME_SUFFIX_LIST 组成

#  test

```
ibv_devices
    device                 node GUID
    ------              ----------------
    mlx5_0              c470bd0300aa1fc8
    mlx5_1              c470bd0300aa1fc9
```
mlx5_1处于工作状态，mlx5_0 出于down   




+ server

```
./transport_test --logtostderr   --server=true
```

+ client    
```
 ./transport_test --logtostderr   --serverip=10.22.116.221
devices name mlx5_1  num_ifs 0 match succ 
Initialized mlx5_1
Initialized 4 engines for 1 devices totally, with 4 engines per device
num_devices_ 1 
Client connected to 10.22.116.221 (flow#0)
Client connected to 10.22.116.221 (flow#1)
Client connected to 10.22.116.221 (flow#2)
Client connected to 10.22.116.221 (flow#3)
E0630 06:37:52.695745 3514092 transport.cc:3360] RTO retransmission threshold reached.1
*** Aborted at 1751265473 (unix time) try "date -d @1751265473" if you are using GNU date ***
PC: @                0x0 (unknown)
*** SIGSEGV (@0x7f34df945000) received by PID 3514082 (TID 0x7f9e099f1640) from PID 18446744073165623296; stack trace: ***
    @     0x7f9e56969046 (unknown)
    @     0x7f9e563e4520 (unknown)
    @     0x55a1d8b38e29 uccl::RDMAContext::try_retransmit_chunk()
    @     0x55a1d8b42954 uccl::RDMAContext::__retransmit_for_flow()
    @     0x55a1d8b42b6e uccl::UcclRDMAEngine::handle_rto()
    @     0x55a1d8b461a6 uccl::UcclRDMAEngine::periodic_process()
    @     0x55a1d8b46498 uccl::UcclRDMAEngine::run()
    @     0x7f9e567b0253 (unknown)
    @     0x7f9e56436ac3 (unknown)
    @     0x7f9e564c8850 (unknown)
    @                0x0 (unknown)
Segmentation fault
```

将UCCL_PARAM(RCMode, "RCMODE", false);改为UCCL_PARAM(RCMode, "RCMODE", true);


![images](uccl.png)    