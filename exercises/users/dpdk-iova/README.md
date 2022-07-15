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

# reference
[DPDK内存管理——iova地址模式（虚拟/ 物理 地址)](https://blog.csdn.net/leiyanjie8995/article/details/121227740)