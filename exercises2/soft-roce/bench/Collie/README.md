# Collie
Collie is for uncovering RDMA NIC performance anomalies. 

# Overview

* [Prerequisite](#Prerequisite) 
* [Quick Start](#Quick-start)
* [Content](#Content)
* [Publication](#Publications)
* [Copyright](#Copyright)

# Prerequisite
- Two hosts with RDMA NICs.
  - Connected to the same switch is recommended since Collie currently does not take network(fabric) effect into consideration. But Collie should work once two hosts are connected and RDMA communication enabled.   

- Set up passwordless SSH login (e.g., ssh public/private keys login).
  - Collie currently uses passwordless SSH login to run traffic_engine on different hosts.

- Google gflags and glog library installed. 
  - Collie uses glog for logging and gflags for commandline flags processing.

- Collie should supports all types of RDMA NICs and drivers that follow IB verbs specification, but currently we've only tested with Mellanox and Broadcom RNICs. 

# Quick Start

## Environment Setup
- Install prerequisites.

```
apt-get install -y libgflags-dev libgoogle-glog-dev
```

- Setup passwordless SSH login.

## Build Traffic Engine

- Build the traffic engine without GPU and CUDA:

``` 
cd traffic_engine && make -j8
```

- OR buidl the traffic engine that supports GPU Direct RDMA:

``` 
cd traffic_engine && GDR=1 make -j8
``` 

NOTICE: GDR is supported only for Tesla or Quadro GPUs according to [GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html).

Please refer to `traffic_engine/README` for more details.

## How to Run: Arguments and Examples

Collie uses JSON configuration file to set parameters for a given RDMA subsystem. 
- Configuration Example: see `./example.json`
  - **username** -- Collie uses SSH to run engines on different hosts, so it needs the username for login.
  - **iplist** --  the client IP and the server IP, given in a list.
  - **logpath** --  the logging path for Collie. Users can get detailed results of anomalies and the reproduce scripts for Collie here.
  - **engine** -- the path for traffic engine.
  - **iters** -- at most `iters` tests that Collie would run.
  - **bars** --  user's expected performance. 
    - tx_pfc_bar -- TX (sent) PFC pause duration in us per second. 
    - rx_pfc_bar -- RX (received) PFC pause duration in us per second.
    - bps_bar -- bits per second of the entire NIC.
    - pps_bar -- packets per second of the entire NIC.
  
- Quick Run Example

``` 
python3 search/collie.py --config  ./example.json
```


## test run

help    
```
./collie_engine  --help
```

 log  
```
LOG(ERROR) << "dbg batch_size is " << batch_size;
```
 
 
+ 修复bug

```
pici_client->CloseDeive();
```

```

  if(listen_thread.joinable())
      listen_thread.join();
  if(server_thread.joinable())
      server_thread.join();
```



```
./collie_engine --server --dev=mlx5_1 --gid=3 --qp_type=2 --mtu=5 --qp_num=8 --buf_num=4 --mr_num=4 --buf_size=4096 --receive=1_4096 --send_batch 1  --send_sge_batch_size 1 --recv_batch 1 --recv_sge_batch_size 1
I0717 01:58:33.530675 1498836 main.cpp:18] Grfwork starts
I0717 01:58:33.571673 1498836 main.cpp:30] Collie server has started.
I0717 01:58:33.571770 1498838 context.cpp:369] About to listen on port 12000
I0717 01:58:33.571789 1498838 context.cpp:375] Server listen thread starts
I0717 01:58:39.716312 1498840 context.cpp:571] Endpoint 0 has started
I0717 02:02:15.486127 1498841 context.cpp:571] Endpoint 1 has started
I0717 02:02:21.397862 1498842 context.cpp:571] Endpoint 2 has started
I0717 02:09:00.185498 1498846 context.cpp:571] Endpoint 3 has started
I0717 02:09:41.089587 1498847 context.cpp:571] Endpoint 4 has started
I0717 02:09:46.272518 1498848 context.cpp:571] Endpoint 5 has started
I0717 02:12:31.673686 1498849 context.cpp:571] Endpoint 6 has started

```

```
./collie_engine --connect=10.22.116.221 --dev=mlx5_1 --gid=3 --qp_type=2 --mtu=5 --qp_num=1 --buf_num=4 --mr_num=4 --buf_size=4096  --request=s_1_4096 --send_batch 1  --send_sge_batch_size 1 --recv_batch 1 --recv_sge_batch_size 1 --iters 1
I0717 02:12:31.642817 3587718 main.cpp:18] Grfwork starts
E0717 02:12:31.672823 3587718 context.cpp:82] debug request : s_1_4096 req.opcode 2 req.sge_num  1
E0717 02:12:31.672849 3587718 main.cpp:49] Collie client run over...
```



# Content
Collie consists of two components, the traffic engine and the search algorithms (the monitor is included as a part of search algorithm).
- Traffic Engine (`./traffic_engine`)
  
  Traffic engine is an independent part that implemented in C/C++. Users can use the engine to generate flexible traffic of different patterns. See `./traffic_engine/README` for more details and examples of complex traffic patterns.It is recommended to reproduce the anomalies (see Appendix of our NSDI paper) with the tool. 

- Search Algorithms (`./search`)
  
  Our simulated-annealing (SA) based algorithm and minimal feature set (MFS) are implemented in python scripts. 
  - `space.py` -- the search space. `Space` defines the search space (upper/lower bounds, granularity for each parameter). Each `Point` has several `Traffics` (e.g., one A->B and one B->A). Each `Traffic` has two `Endhost`, one server and one client, as well as many other attributes that describe this traffic (e.g., QP type).
  - `engine.py` -- given a point, running `collie_engine ` to set up the corresponding traffic described in the `Point`. If users need to set up traffics in different ways (rather than SSH), please modify the `Engine` class.
  - `anneal.py` -- the simulated-annealing based algorithm and minimal feature set algorithm are implemented here. If users need to modify the temperature and mutation logics, please modify here.
  - `logger.py` -- logging assistant functions for logging results and reproduce scripts. 
  - `bone.py` -- monitor performance counters and collect statistic results based on vendor's tools.
  - `hardware.py` -- monitor diagnostic counters and collect statistic results based on vendor's tools.  (Unfortunately currently diagnostic counters tools like [NeoHost](https://support.mellanox.com/s/productdetails/a2v50000000N2OlAAK/mellanox-neohost) is not publicly available and open-sourced, so we only provide performance counter based code for NDA reasons.)
  - `collie.py` -- read user parameters and call SA to search. 
  

# Copyright

Collie is provided under the MIT license. See LICENSE for more details.
