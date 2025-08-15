
TSO 全称为 TCP Segment Offload ，简单的讲，就是靠网卡硬件来分段 TCP ，计算 checksum ，从而解放 CPU 周期。

我们知道通常以太网的 MTU 是 1500 ，除去 TCP/IP 的包头，TCP 的 MSS(Max Segment Size)大小是 1460 ；通常情况下协议栈会对超过 1460 的 TCP payload 进行 segmentation 以保证生成的 IP 包不超过 MTU 的大小；但是对于支持 TSO/GSO 的网卡而言，就没这个必要了；我们可以把最多 64K 大小的 TCP payload 直接往下传给协议栈，此时 IP 层也不会进行 segmentation ，而是会一直传给网卡驱动，支持 TSO/GSO 的网卡会自己生成 TCP/IP 包头和帧头，这样可以 offload 很多协议栈上的内存操作，checksum 计算等原本靠 CPU 来做的工作都移给了网卡；

而 GSO 可以看作是 TSO 的增强，但 GSO 不只针对 TCP ，而是对任意协议；其会尽可能把 segmentation 推后到交给网卡的那一刻，才会判断网卡是否支持 SG(scatter-gather) 和 GSO ；如果不支持，则在协议栈里做 segmentation ，如果支持，则把 payload 直接发给网卡；

1）抓包查看数据size,报文长度是否超过MSS

TCP Segmentation Offload is supported in Linux by the network device layer.
 A driver that wants to offer TSO needs to set the NETIF_F_TSO bit in the network device structure. 
 In order for a device to support TSO, it needs to also support Net:TCP checksum offloading and Net:Scatter Gather.

The driver will then receive super-sized skb's. 
These are indicated to the driver by skb_shinfo(skb)→gso_size being non-zero. 
**The gso_size is the size the hardware should fragment the TCP data. **
TSO may change how and when TCP decides to send data.

# skb模型
1） IP 数据包分片（fragment）时用到的 frag_list 模型：

分片的数据有各自的 skb 结构体，它们通过 skb->next 链接成一个单链表，表头是第一个 skb 的 shared_info 中的 frag_list。

2） GSO 进行分段（segmentation）用到的一种模型: 

当一个大的 TCP 数据包被切割成几个 MTU 大小的数据时，它们也是通过 skb->next 链接到一起的：

与分片不同的是，相关的信息是记录在最前面一个 skb 的 skb_shared_info 里面的 gso_segs 和 gso_size 

# MAX_TCP_HEADER

```
#define MAX_TCP_HEADER  L1_CACHE_ALIGN(128 + MAX_HEADER)
skb = alloc_skb(MAX_TCP_HEADER,
sk_gfp_mask(sk, GFP_ATOMIC | __GFP_NOWARN))
```
This function allocates a network buffer. It takes 2 arguments the size in bytes of the data area requested
and the set of flags that tell the memory allocator how to behave. For the most part the network code calls
the memory allocator with the GFP_ATOMIC set of flags: do not return without completing the task (for the
moment this is equivalent to the __GFP_HIGH flag : can use emergency pools). The tcp code for instance
uses its own skb allocator tcp_alloc_skb to request additional MAX_TCP_HEADER bytes to accomodate a
 headroom sufficient for the header ( usually this is 128+32=160 bytes ).
 
 ![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/tcp-segmentation-offload/tcp_alloc.png)

# ethtool -k  enp0s31f6 | grep tcp-segmentation-offload

```
root@ubuntux86:/work/e1000e# ethtool -k  enp0s31f6 | grep tcp-segmentation-offload
tcp-segmentation-offload: off
root@ubuntux86:/work/e1000e# ethtool -K enp0s31f6  tso on
root@ubuntux86:/work/e1000e# ethtool -k  enp0s31f6 | grep tcp-segmentation-offload
tcp-segmentation-offload: on
root@ubuntux86:/work/e1000e# 
```

# iperf3

## client

```
iperf3 -c 10.11.12.80    -p 5201 -i 1  -l 6556
```

## server

**iperf3  -s  -B 192.168.6.1 -4**

```
iperf3 -s  10.11.12.80 
-----------------------------------------------------------
Server listening on 5201
-----------------------------------------------------------
Accepted connection from 10.11.12.82, port 38498
[  5] local 10.11.12.80 port 5201 connected to 10.11.12.82 port 38500
```

# e1000_tx_map

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/tcp-segmentation-offload/tx-map.png)


# tcp-segmentation-offload: on

```
root@ubuntux86:/home/ubuntu#  ethtool -k  enp0s31f6 | grep tcp-segmentation-offload
tcp-segmentation-offload: on
root@ubuntux86:/home/ubuntu# 
```

## tcpdump -i enp0s31f6 tcp and dst host  10.11.11.80 -eennvv

** iperf3 -c 10.11.11.81 -p 5201 -i 1 -l 6556 -t 1**

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/tcp-segmentation-offload/tcp_df.png)

```
>>> 261526977 -  261526625
352
>>> 261526625 - 261523729
2896  大于MTU
>>> 
>>> 261396657 - 261424169
-27512
>>> 260614737 - 260635009
-20272
>>> 
```

## tcpdump -i enp0s31f6 icmp and dst host  10.11.11.80 -eennvv

**ping 10.11.11.80 -s 6500**

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/tcp-segmentation-offload/icmp_f.png)


#  skb_shinfo(skb)->gso_size

## tcp-segmentation-offload: on

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/tcp-segmentation-offload/tcp_ssh.png)


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/tcp-segmentation-offload/tcp_ssh2.jpg)

** iperf3 -c 10.11.11.81 -p 5201 -i 1 -l 6556 -t 1**

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/tcp-segmentation-offload/tcp_iperf.png)


## tcp-segmentation-offload: off

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/tcp-segmentation-offload/tcp_ssh3.png)

** iperf3 -c 10.11.11.81 -p 5201 -i 1 -l 6556 -t 1**

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/tcp-segmentation-offload/tcp_iperf2.png)

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/tcp-segmentation-offload/tcp_iperf3.png)


***mss 都是0***


#  e1000_tso

## skb_is_gso(skb)

### tcp-segmentation-offload: on

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/tcp-segmentation-offload/tcp_ssh4.png)


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/tcp-segmentation-offload/tcp_ssh5.png)

*dmesg | grep 'tcp proto and mms' |   grep 'src port 22' | grep 'skb_is_gso(skb): 1'*

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/tcp-segmentation-offload/tcp_ssh6.png)

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/tcp-segmentation-offload/tcp_iperf4.png)


## tcp-segmentation-offload: off



![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/tcp-segmentation-offload/tcp_ssh7.png)



![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/tcp-segmentation-offload/tcp_iperf5.png)


***skb_is_gso(skb)都是0***


# skb_shinfo(skb)->gso_segs

## tcp-segmentation-offload: on

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/tcp-segmentation-offload/gso_seg1.png)


```
[21772.672227] e1000e: tcp proto and mms : 0, src port 38140, dst pot 5201, skb_is_gso: 0, gso_segs: 1 , skb_shinfo(skb)->nr_frags:1, skb_is_nonlinear:1, dma count: 2 
[21772.672247] e1000e: tcp proto and mms : 0, src port 38140, dst pot 5201, skb_is_gso: 0, gso_segs: 1 , skb_shinfo(skb)->nr_frags:1, skb_is_nonlinear:1, dma count: 2 
[21772.673734] e1000e: tcp proto and mms : 0, src port 38140, dst pot 5201, skb_is_gso: 0, gso_segs: 1 , skb_shinfo(skb)->nr_frags:1, skb_is_nonlinear:1, dma count: 2 
[21772.673835] e1000e: tcp proto and mms : 0, src port 38140, dst pot 5201, skb_is_gso: 0, gso_segs: 1 , skb_shinfo(skb)->nr_frags:1, skb_is_nonlinear:1, dma count: 2 
[21772.673845] e1000e: tcp proto and mms : 0, src port 38140, dst pot 5201, skb_is_gso: 0, gso_segs: 1 , skb_shinfo(skb)->nr_frags:1, skb_is_nonlinear:1, dma count: 2 
[21772.673854] e1000e: tcp proto and mms : 0, src port 38140, dst pot 5201, skb_is_gso: 0, gso_segs: 1 , skb_shinfo(skb)->nr_frags:1, skb_is_nonlinear:1, dma count: 2 
[21772.673865] e1000e: tcp proto and mms : 0, src port 38140, dst pot 5201, skb_is_gso: 0, gso_segs: 1 , skb_shinfo(skb)->nr_frags:1, skb_is_nonlinear:1, dma count: 2 
[21772.673882] e1000e: tcp proto and mms : 1448, src port 38140, dst pot 5201, skb_is_gso: 1, gso_segs: 3 , skb_shinfo(skb)->nr_frags:1, skb_is_nonlinear:1, dma count: 2 
[21772.673934] e1000e: tcp proto and mms : 1448, src port 38140, dst pot 5201, skb_is_gso: 1, gso_segs: 9 , skb_shinfo(skb)->nr_frags:1, skb_is_nonlinear:1, dma count: 4 
[21772.675595] e1000e: tcp proto and mms : 1448, src port 38140, dst pot 5201, skb_is_gso: 1, gso_segs: 9 , skb_shinfo(skb)->nr_frags:2, skb_is_nonlinear:1, dma count: 5 
[21772.677456] e1000e: tcp proto and mms : 1448, src port 38140, dst pot 5201, skb_is_gso: 1, gso_segs: 9 , skb_shinfo(skb)->nr_frags:1, skb_is_nonlinear:1, dma count: 4 
[21772.677464] e1000e: tcp proto and mms : 1448, src port 38140, dst pot 5201, skb_is_gso: 1, gso_segs: 8 , skb_shinfo(skb)->nr_frags:2, skb_is_nonlinear:1, dma count: 4 
[21772.679268] e1000e: tcp proto and mms : 1448, src port 38140, dst pot 5201, skb_is_gso: 1, gso_segs: 9 , skb_shinfo(skb)->nr_frags:1, skb_is_nonlinear:1, dma count: 4 
```

***skb_is_gso: 0, gso_segs: 1***

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/tcp-segmentation-offload/gso_seg2.png)

## tcp-segmentation-offload: off

```
skb_is_gso: 0, gso_segs: 0 , skb_shinfo(skb)->nr_frags:1
skb_is_gso: 0, gso_segs: 1 , skb_shinfo(skb)->nr_frags:0
```

skb_is_gso , gso_segs , skb_shinfo(skb)->nr_frags 这三个参数

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/tcp-segmentation-offload/gso_seg3.png)

#  skb_is_gso and  skb_is_nonlinear

## tcp-segmentation-offload: on

*** skb_is_gso: 1 ,skb_shinfo(skb)->nr_frags:1***
*** skb_is_gso: 0 ,skb_shinfo(skb)->nr_frags:1***


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/tcp-segmentation-offload/tcp_ssh8.png)

<!--
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/tcp-segmentation-offload/tcp_ssh82.png)
-->

```
[19830.692410] e1000e: tcp proto and mms : 0, src port 22, dst pot 60145, skb_is_gso: 0 ,skb_shinfo(skb)->nr_frags:1, skb_is_nonlinear:1, sk_buff: len:194  skb->data_len:128  
[19830.724911] e1000e: tcp proto and mms : 0, src port 22, dst pot 60145, skb_is_gso: 0 ,skb_shinfo(skb)->nr_frags:1, skb_is_nonlinear:1, sk_buff: len:178  skb->data_len:112  
[19830.725028] e1000e: tcp proto and mms : 0, src port 22, dst pot 60145, skb_is_gso: 0 ,skb_shinfo(skb)->nr_frags:1, skb_is_nonlinear:1, sk_buff: len:194  skb->data_len:128  
[19830.758382] e1000e: tcp proto and mms : 0, src port 22, dst pot 60145, skb_is_gso: 0 ,skb_shinfo(skb)->nr_frags:1, skb_is_nonlinear:1, sk_buff: len:178  skb->data_len:112  
[19830.758547] e1000e: tcp proto and mms : 0, src port 22, dst pot 60145, skb_is_gso: 0 ,skb_shinfo(skb)->nr_frags:1, skb_is_nonlinear:1, sk_buff: len:194  skb->data_len:128  
[19830.787441] e1000e: tcp proto and mms : 0, src port 22, dst pot 60145, skb_is_gso: 0 ,skb_shinfo(skb)->nr_frags:1, skb_is_nonlinear:1, sk_buff: len:178  skb->data_len:112  
[19830.787592] e1000e: tcp proto and mms : 0, src port 22, dst pot 60145, skb_is_gso: 0 ,skb_shinfo(skb)->nr_frags:1, skb_is_nonlinear:1, sk_buff: len:194  skb->data_len:128  
[19830.818598] e1000e: tcp proto and mms : 0, src port 22, dst pot 60145, skb_is_gso: 0 ,skb_shinfo(skb)->nr_frags:1, skb_is_nonlinear:1, sk_buff: len:178  skb->data_len:112  
[19830.818768] e1000e: tcp proto and mms : 0, src port 22, dst pot 60145, skb_is_gso: 0 ,skb_shinfo(skb)->nr_frags:1, skb_is_nonlinear:1, sk_buff: len:194  skb->data_len:128 
```
*** sk_buff->len + skb->data_len < mtu竟然也有skb_shinfo(skb)->nr_frags>1 ***

###  why  skb_is_gso: 0 ,skb_shinfo(skb)->nr_frags:1

*** sk_buff->len + skb->data_len < mtu竟然也有skb_shinfo(skb)->nr_frags>1 ***

## tcp-segmentation-offload: off

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/tcp-segmentation-offload/tcp_ssh83.png)

# frag_list

there is no frag_list in e1000e

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/tcp-segmentation-offload/frag_list.png)

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/tcp-segmentation-offload/tcp_ssh9.png)