# dpdk-hping

## Name
hping3 - send ICMP packets that can calculate minimum/maximum and average RTTs for the number of packets sent. 

## Synopsis
dpdk-hping [ -p port id ] [ -c client mode] [ -s server mode ] [ -l number of packets ] [ -W timeout ] [ -a server ip address ] [ -b client ip address ]

## Description
dpdk-hping is a network tool able to send ICMP packets, packets containing only ethernet header address, and displays the statistics of the received packets from clients. It is able to calculate RTT between server and client hosts. 

## Base Options

-p: Port Id: <br>
    The index of the port to be used by the DPDK driver for server or client. <br>
    <br>
-c: client mode: <br>
    Run the host in client mode and give the MAC address of host as argument.<br>
    <br>
-s: server mode:<br>
    Run the host in server mode. <br>
    <br>
-l: number of packets:<br>
    Number of packets to be sent. Default = flood.<br>
    <br>
-W: timeout:<br>
    Change timeout using this setting. Default = 2 secs<br>
    <br>
-a: server IP address:<br>
    Pass the server IP address as argument.<br>
    <br>
-b: client IP address:<br>
    Pass the client IP address as argument. <br>
    <br>
## Installation Guide
Setup in MAC:<br>
Download the 23.07 version of DPDK from https://core.dpdk.org/doc/quick-start/. <br>
Follow the steps given in quick start guide and [https://doc.dpdk.org/guides/linux_gsg/sys_reqs.html#compilation-of-the-dpdk](https://doc.dpdk.org/guides/linux_gsg/index.html). <br>
For Apple Silicon M1/M2 chips, follow the following commands:
```
sudo modprobe vfio-pci enable_sriov=1
sudo mount -t hugetlbfs pagesize=1GB /mnt/huge
for i in {1..50}; do sudo sh -c 'echo "1024" > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages'; done
cat /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages
sudo dpdk-hugepages.py --reserve 1G
sudo modprobe vfio enable_unsafe_noiommu_mode=1
sudo sh -c 'echo "1" > /sys/module/vfio/parameters/enable_unsafe_noiommu_mode'
sudo chmod 666 /sys/bus/pci/drivers/vfio-pci/bind
sudo ./<dpdk-23.07folder>/usertools/dpdk-devbind.py --bind=vfio-pci <network_interface>
```

## How to setup?
Configure DPDK.
Bind your machine with DPDK-compatible driver. If using the same machine, bind two network interfaces to each machine. 
Run the following commands on each host: 
```
 sudo ./dpdk-hping --file-prefix server -a <network_interface> -- -s -p 0
 sudo ./dpdk-hping --file-prefix client -a <network_interface> -- -c <server MAC address>  -p 0
```
Few other examples: 
```
sudo ./dpdk-hping --file-prefix client -a <network_interface> -- -c <server MAC address>  -p 0 -l 150
sudo ./dpdk-hping --file-prefix client -a <network_interface> -- -c <server MAC address>  -p 0 -W 5
sudo ./dpdk-hping --file-prefix client -a <network_interface> -- -c <server MAC address>  -p 0 -a 10.0.0.1
```
## Authors
Dhruv Gupta, Hirva Patel, Srujan Shetty, Dhruv Khandelwal

## References
https://github.com/JiakunYan/dpu-pingpong
