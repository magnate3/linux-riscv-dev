-------------------------------------------------------------------------------

# RPerf: Accurate Latency Measurement Framework for RDMA #

-------------------------------------------------------------------------------

**Table of Contents**

- [RPerf](#rperf)
- [Prerequisites](#prerequisites)
- [Install](#install)
- [Configuration](#configuration)
- [Running Tests](#running-tests)
- [Contacts](#contacts)


## dependcy
```
 apt-get -y install libncurses5-dev
```

## undefined reference to `RDTSC'

```
gcc -E -dM - < /dev/null | grep 86_64
#define __x86_64 1
#define __x86_64__ 1
```
rperf/include/clock.h添加defined(__x86_64)    

```
#if defined(__X86_64__) || defined(__x86_64)  || defined(__i386__) || defined(__i386) || defined(_M_IX86)
```

## RPerf ##

This package provides an accurate benchmark tool for **RDMA**-based networks.

+ client

```
cat rdmarc 
device_name=mlx5_1
msg_size=64
rx_depth=8000
tx_depth=8000
qps_number=1
num_concurrent_msgs=1
iterations=5000000
duration=15
burst_size=1
rate_limit=15000
num_client_post_send_threads=1
num_client_poll_recv_threads=1
num_client_poll_send_threads=1
num_server_post_recv_threads=4
num_server_poll_recv_threads=3
num_server_poll_send_threads=3
test_type=AckRtt
socket_port=9999
server_name=10.22.116.221
output_filename=histogram
is_server=false
bw_limiter=false
sampling=true
show_result=true
verbose=true
sampling_ratio=0.01
```
+ client

```
device_name=mlx5_1
msg_size=64
rx_depth=8000
tx_depth=8000
qps_number=1
num_concurrent_msgs=1
iterations=5000000
duration=15
burst_size=1
rate_limit=15000
num_client_post_send_threads=1
num_client_poll_recv_threads=1
num_client_poll_send_threads=1
num_server_post_recv_threads=4
num_server_poll_recv_threads=3
num_server_poll_send_threads=3
test_type=AckRtt
socket_port=9999
server_name=10.22.116.221
output_filename=histogram
is_server=true
bw_limiter=false
sampling=true
show_result=true
verbose=true
sampling_ratio=0.01
```

## Prerequisites ##

Before you install RPerf, you must have the following libraries:

- cmake
- libncurses5-dev
- rdma-core libibverbs1 librdmacm1 libibmad5 libibumad3 librdmacm1 ibverbs-providers rdmacm-utils infiniband-diags libfabric1 ibverbs-utils libibverbs-dev

## Install ##

Clone the repository:
```
git clone https://github.com/ease-lab/rperf.git
```
Then you can simply make the package:
```
cd rperf
cmake -H. -Bbuild -G "Unix Makefiles"
cmake --build build -j4
```

## Configuration ##

RPerf by default locates *rdmarc* file in the working directory. This file contains test parameters. Change the parameters according to what you desire. 

## Running Tests ##
The simplest way to run with default settings, on the server and clients:
```
./build/rperf_c 
```
Make sure *rdmarc* file on each node has the proper values for __is_server__ and __server_name__ parameters.


## Contacts ##

This implementation is a research prototype that shows the feasibility of accurate latency measurement and has been tested on a cluster equipped with _Mellanox MT27700 ConnectX-4_ HCAs and a _Mellanox SX6012_ IB switch. It is NOT production quality code. The technical details can be found [here](https://ease-lab.github.io/ease_website/pubs/RPERF_ISPASS20.pdf). If you have any questions, please raise issues on Github or contact the authors below.

[M.R. Siavash Katebzadeh](http://mr.katebzadeh.xyz) (m.r.katebzadeh@ed.ac.uk)
<!-- markdown-toc end -->

 
