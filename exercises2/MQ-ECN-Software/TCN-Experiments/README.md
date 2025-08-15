# TCN-Experiments
## 1. Software Requirements
To reproduce Figure 6 to 9 of [TCN paper](http://sing.cse.ust.hk/~wei/papers/tcn-conext2016.pdf), you need following software:
  - [TCN software prototype](https://github.com/HKUST-SING/TCN-Software)
  - [A simple traffic generator](https://github.com/HKUST-SING/TrafficGenerator)
  - [PIAS software prototype](https://github.com/HKUST-SING/PIAS-Software)
  
In addition, you also need to install Linux 3.18x kernel which supports DCTCP. We used 3.18.11 kernel in our experiments. We also provide a [patch](https://github.com/baiwei0427/Latency-Measurement/blob/master/kernel_measurement3.patch) for Linux 3.18.11 kernel which allows users to adjust TCP RTOmin using sysctl.  

## 2. Testbed Setup
In the testbed, 9 servers are connected to a 9-port server-emulated switch. One server acts as the receiver while the rest 8 servers act as the sender. The IP address of the receiver is 192.168.101.1(/24). The IP addresses of senders are 192.168.102.1(/24), 192.168.103.1(/24), ... 192.168.109.1(/24). The server-emulated switch has 9 NICs whose IP addresses are 192.168.101.2(/24), 192.168.102.2(/24), ... 192.168.109.2(/24). To avoid large segments, please disable offloading techniques (e.g., GSO, TSO and GRO) on the server-emulated switch.   

To enable packet forwarding on the server-emulated switch:
```
$ sysctl -w net.ipv4.ip_forward=1
```
To ensure servers on different subnets (e.g., 192.168.101.0/24 and 192.168.102.0/24) can access each other, you need to add some routing entries on servers. For example, at the receiver (192.168.101.1), I add a new routing entry as follows:
```
$ route add -net 192.168.0.0/16 gw 192.168.101.2
```
To enable DCTCP on servers:
```
$ sysctl -w net.ipv4.tcp_ecn=1
$ sysctl -w net.ipv4.tcp_congestion_control=dctcp
```
If you have applied our [patch](https://github.com/baiwei0427/Latency-Measurement/blob/master/kernel_measurement3.patch), you can adjust TCP RTOmin (in milliseconds) as follows:
```
$ sysctl -w net.ipv4.tcp_rto_min=10
```

## 3. Installation
#### 3.1 [TCN Software](https://github.com/HKUST-SING/TCN-Software) 
To install TCN qdisc kernel module, please follow the [guidance](https://github.com/HKUST-SING/TCN-Software). By default:
  - All DWRR queues have the same quantum (1.5KB). All WFQ queues have the equal weight (1). 
  - All the queues belonging to a switch port share the per-port buffer space in a first-in-first-serve bias.
  - Packets with DSCP value **i** are classified to queue **i** of the scheduler.

To configure SP (1 queue) / DWRR (4 queues) (Figure 8):
```
$ sysctl -w dwrr.queue_prio_0=0
$ sysctl -w dwrr.queue_prio_1=1
$ sysctl -w dwrr.queue_prio_2=1
$ sysctl -w dwrr.queue_prio_3=1
$ sysctl -w dwrr.queue_prio_4=1
```
To configure SP (1 queue) / WFQ (4 queues) (Figure 9):
```
$ sysctl -w wfq.queue_prio_0=0
$ sysctl -w wfq.queue_prio_1=1
$ sysctl -w wfq.queue_prio_2=1
$ sysctl -w wfq.queue_prio_3=1
$ sysctl -w wfq.queue_prio_4=1
```
In this way, queue 0 will have a strict higher priority than queue 1, 2, 3 and 4.

To set per-port buffer size to 96KB (SP/DWRR):
```
$ sysctl -w dwrr.shared_buffer=96000
```
To set per-port buffer size to 96KB (SP/WFQ):
```
$ sysctl -w wfq.shared_buffer=96000
```
To enable per-queue ECN/RED with the standard threshold (32KB) (SP/DWRR):
```
$ sysctl -w dwrr.ecn_scheme=1
$ sysctl -w dwrr.queue_thresh_0=32000
$ sysctl -w dwrr.queue_thresh_1=32000
$ sysctl -w dwrr.queue_thresh_2=32000
$ sysctl -w dwrr.queue_thresh_3=32000
$ sysctl -w dwrr.queue_thresh_4=32000
```
To enable per-queue ECN/RED with the standard threshold (32KB) (SP/WFQ):
```
$ sysctl -w wfq.ecn_scheme=1
$ sysctl -w wfq.queue_thresh_0=32000
$ sysctl -w wfq.queue_thresh_1=32000
$ sysctl -w wfq.queue_thresh_2=32000
$ sysctl -w wfq.queue_thresh_3=32000
$ sysctl -w wfq.queue_thresh_4=32000
```
To enable MQ-ECN (SP/DWRR):
```
$ sysctl -w dwrr.ecn_scheme=3
$ sysctl -w dwrr.port_thresh=32000
```
To enable MQ-ECN (SP/WFQ):
```
$ sysctl -w wfq.ecn_scheme=3
$ sysctl -w wfq.port_thresh=32000
```
To enable TCN (SP/DWRR):
```
$ sysctl -w dwrr.ecn_scheme=4
$ sysctl -w dwrr.tcn_thresh=250
```
To enable TCN (SP/WFQ):
```
$ sysctl -w wfq.ecn_scheme=4
$ sysctl -w wfq.tcn_thresh=250
```
To enable CoDel (SP/DWRR):
```
$ sysctl -w dwrr.ecn_scheme=5
$ sysctl -w dwrr.codel_target=50
$ sysctl -w dwrr.codel_interval=1000
```
To enable CoDel (SP/WFQ):
```
$ sysctl -w wfq.ecn_scheme=5
$ sysctl -w wfq.codel_target=50
$ sysctl -w wfq.codel_interval=1000
```
For more parameter settings, please see following two files:
  - [SP/DWRR params.h](https://github.com/HKUST-SING/TCN-Software/blob/master/sch_dwrr/params.h)
  - [SP/WFQ params.h](https://github.com/HKUST-SING/TCN-Software/blob/master/sch_wfq/params.h)

#### 3.2 [Traffic Generator](https://github.com/HKUST-SING/TrafficGenerator)
To install the traffic generator, please follow the [guidance](https://github.com/HKUST-SING/TrafficGenerator). After installation, you should start *server* on senders (192.168.102.1 to 192.168.109.1). 

#### 3.3 [PIAS Software](https://github.com/HKUST-SING/PIAS-Software)
To install the traffic generator, please follow the [guidance](https://github.com/HKUST-SING/PIAS-Software). By default, PIAS kernel module supports two priorities. The only demotion threshold is [100KB](https://github.com/HKUST-SING/PIAS-Software/blob/master/pias4/params.c#L23). Packets in the high priority will be marked with DSCP 0. Packets in the low priority will not get any DSCP marked.     


