# Selected IOVA mode 'VA' 

```
[root@centos7 dpdk-19.11]# export RTE_SDK=`pwd`
[root@centos7 dpdk-19.11]# export RTE_TARGET=arm64-armv8a-linuxapp-gcc
[root@centos7 dpdk-19.11]#  cd examples/helloworld
[root@centos7 helloworld]# make
```

```
[root@centos7 helloworld]# ./build/helloworld --no-huge
EAL: Detected 128 lcore(s)
EAL: Detected 4 NUMA nodes
EAL: Static memory layout is selected, amount of reserved memory can be adjusted with -m or --socket-mem
EAL: Multi-process socket /var/run/dpdk/rte/mp_socket
EAL: Selected IOVA mode 'VA'  
EAL: Probing VFIO support...
EAL: VFIO support initialized
EAL: PCI device 0000:05:00.0 on NUMA socket 0
EAL:   probe driver: 19e5:200 net_hinic
EAL:   using IOMMU type 1 (Type 1)
```

 


# rte_eal_iova_mode

***iommu enalbe ***

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/users/dpdk-iova/pic/iova.png)

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/users/dpdk-iova/pic/iommu.png)

# rte_malloc_virt2iova
```
rte_iova_t
rte_mem_virt2iova(const void *virtaddr)
{
        if (rte_eal_iova_mode() == RTE_IOVA_VA)
                return (uintptr_t)virtaddr;
        return rte_mem_virt2phy(virtaddr);
}
```

# dpdk ixgbe_alloc_rx_queue_mbufs 

```
static int __attribute__((cold))
ixgbe_alloc_rx_queue_mbufs(struct ixgbe_rx_queue *rxq)
{
  struct ixgbe_rx_entry *rxe = rxq->sw_ring;
  uint64_t dma_addr;
  unsigned int i;

  /* Initialize software ring entries */
  for (i = 0; i < rxq->nb_rx_desc; i++) {
    volatile union ixgbe_adv_rx_desc *rxd;
    struct rte_mbuf *mbuf = rte_mbuf_raw_alloc(rxq->mb_pool); /* 分配mbuf */

    if (mbuf == NULL) {
      PMD_INIT_LOG(ERR, "RX mbuf alloc failed queue_id=%u",
             (unsigned) rxq->queue_id);
      return -ENOMEM;
    }

    mbuf->data_off = RTE_PKTMBUF_HEADROOM;
    mbuf->port = rxq->port_id;

    dma_addr =
      rte_cpu_to_le_64(rte_mbuf_data_dma_addr_default(mbuf)); /* mbuf的总线地址 */
    rxd = &rxq->rx_ring[i];
    rxd->read.hdr_addr = 0;
    rxd->read.pkt_addr = dma_addr;  /* 总线地址赋给 rxd->read.pkt_addr */
    rxe[i].mbuf = mbuf;       /* 将 mbuf 挂载到 rxe */
  }

  return 0;
}
```


# reference
[DPDK内存管理——iova地址模式（虚拟/ 物理 地址)](https://blog.csdn.net/leiyanjie8995/article/details/121227740)