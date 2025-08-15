

# 21.11.版本移植到19.11
```
MBUF: error setting mempool handler
```


```
EAL: Detected 128 lcore(s)
EAL: Detected 4 NUMA nodes
EAL: Multi-process socket /var/run/dpdk/rte/mp_socket
EAL: Selected IOVA mode 'VA'
EAL: No available hugepages reported in hugepages-2048kB
EAL: Probing VFIO support...
EAL: VFIO support initialized
MBUF: error setting mempool handler
EAL: Error - exiting with code: 1
Cause: Cannot init mbuf pool on socket 0
```
在Makefile中加上include $(RTE_SDK)/mk/rte.extapp.mk   