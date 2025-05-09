

# make


```
CONFIG_RTE_LIBRTE_IEEE1588=y
```

```
export RTE_SDK=/root/dpdk-stable-19.11.1
export RTE_TARGET=arm64-armv8a-linuxapp-gcc
make install T=$RTE_TARGET -j 48
```

```
[root@centos7 ptpclient]# ls /root/dpdk-stable-19.11.1
ABI_VERSION  arm64-armv8a-linuxapp-gcc  config    doc      examples     kernel  license      Makefile     meson_options.txt  README     VERSION
app          buildtools
```


```
 ./build/ptpclient -c0x1   -n 4 -- -p 0x1 -T 0
EAL: Detected 128 lcore(s)
EAL: Detected 4 NUMA nodes
EAL: Multi-process socket /var/run/dpdk/rte/mp_socket
EAL: Selected IOVA mode 'VA'
EAL: No available hugepages reported in hugepages-2048kB
EAL: Probing VFIO support...
EAL: VFIO support initialized
EAL: PCI device 0000:05:00.0 on NUMA socket 0
EAL:   probe driver: 19e5:200 net_hinic
EAL: PCI device 0000:06:00.0 on NUMA socket 0
EAL:   probe driver: 19e5:200 net_hinic
EAL:   using IOMMU type 1 (Type 1)
net_hinic: Initializing pf hinic-0000:06:00.0 in primary process
net_hinic: Device 0000:06:00.0 hwif attribute:
net_hinic: func_idx:1, p2p_idx:1, pciintf_idx:0, vf_in_pf:0, ppf_idx:0, global_vf_id:135, func_type:0
net_hinic: num_aeqs:4, num_ceqs:4, num_irqs:32, dma_attr:2
net_hinic: Get public resource capability:
net_hinic: host_id: 0x0, ep_id: 0x1, intr_type: 0x0, max_cos_id: 0x7, er_id: 0x1, port_id: 0x1
net_hinic: host_total_function: 0xf2, host_oq_id_mask_val: 0x8, max_vf: 0x78
net_hinic: pf_num: 0x2, pf_id_start: 0x0, vf_num: 0xf0, vf_id_start: 0x10
net_hinic: Get l2nic resource capability:
net_hinic: max_sqs: 0x10, max_rqs: 0x10, vf_max_sqs: 0x4, vf_max_rqs: 0x4
net_hinic: Initialize 0000:06:00.0 in primary successfully
EAL: PCI device 0000:7d:00.0 on NUMA socket 0
EAL:   probe driver: 19e5:a222 net_hns3
EAL: PCI device 0000:7d:00.1 on NUMA socket 0
EAL:   probe driver: 19e5:a221 net_hns3
EAL: PCI device 0000:7d:00.2 on NUMA socket 0
EAL:   probe driver: 19e5:a222 net_hns3
EAL: PCI device 0000:7d:00.3 on NUMA socket 0
EAL:   probe driver: 19e5:a221 net_hns3
net_hinic: Disable vlan filter succeed, device: hinic-0000:06:00.0, port_id: 0
net_hinic: Disable vlan strip succeed, device: hinic-0000:06:00.0, port_id: 0
net_hinic: Set new mac address 44:a1:91:a4:9c:0c

net_hinic: Disable promiscuous, nic_dev: hinic-0000:06:00.0, port_id: 0, promisc: 0
net_hinic: Disable allmulticast succeed, nic_dev: hinic-0000:06:00.0, port_id: 0
Timesync enable failed: -95
EAL: Error - exiting with code: 1
  Cause: Cannot init port 0
```

