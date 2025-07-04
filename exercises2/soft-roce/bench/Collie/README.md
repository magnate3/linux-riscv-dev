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

```
./collie_engine --server --dev=mlx5_1 --gid=3 --qp_type=2 --mtu=3 --qp_num=1 --buf_num=4 --mr_num=4 --buf_size=65536 --receive=1_65536,2_4096_4096
I0611 06:40:41.073875 278602 main.cpp:18] Grfwork starts
I0611 06:40:41.106448 278602 main.cpp:30] Collie server has started.
I0611 06:40:41.106570 278603 context.cpp:369] About to listen on port 12000
I0611 06:40:41.106587 278603 context.cpp:375] Server listen thread starts
E0611 06:41:42.277411 278605 endpoint.cpp:179] Failed to modify QP to RTR: Invalid argument [22]
E0611 06:41:42.277446 278605 context.cpp:553] Activate Recv Endpoint 0 failed
E0611 06:41:55.957463 278607 context.cpp:474] QP Overflow, request rejected
```

```
 ./collie_engine --connect=10.22.116.221 --dev=mlx5_1 --gid=3 --qp_type=2 --mtu=3 --qp_num=1 --buf_num=4 --mr_num=4 --buf_size=65536 --request=s_1_5120,w_2_4096_512,r_1_65536
I0611 06:41:55.883708 3200525 main.cpp:18] Grfwork starts
E0611 06:41:55.917788 3200525 context.cpp:676] Receiver does not support 1 senders
E0611 06:41:55.917811 3200525 main.cpp:44] Collie client connect to 10.22.116.221 failed
terminate called after throwing an instance of 'std::system_error'
  what():  Invalid argument
Aborted (core dumped)
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
