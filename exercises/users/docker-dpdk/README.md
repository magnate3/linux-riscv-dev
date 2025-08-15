

# make

 

```
export RTE_SDK=/ovs/dpdk
export RTE_TARGET=x86_64-native-linuxapp-gcc
make -j 2 install T=x86_64-native-linuxapp-gcc
```

1) x86_64-native-linuxapp-gcc/.config 将igb_uio、kni等内核模块设置为n   

```
CONFIG_RTE_LIBRTE_IGB_PMD=n
CONFIG_RTE_EAL_IGB_UIO=n
```

CONFIG_RTE_KNI_KMOD   

```
CONFIG_RTE_LIBRTE_KNI=n
CONFIG_RTE_KNI_KMOD=n
CONFIG_RTE_KNI_PREEMPT_DEFAULT=y
CONFIG_RTE_KNI_KO_DEBUG=n
CONFIG_RTE_KNI_VHOST=n
CONFIG_RTE_KNI_VHOST_MAX_CACHE_SIZE=1024
CONFIG_RTE_KNI_VHOST_VNET_HDR_EN=n
CONFIG_RTE_KNI_VHOST_DEBUG_RX=n
CONFIG_RTE_KNI_VHOST_DEBUG_TX=n
```

2) 设置ovs依赖的dpdk路径    

```
OVS_CFLAGS =  -I/ovs/dpdk/x86_64-native-linuxapp-gcc/include -mssse3   
OVS_LDFLAGS =  -L/ovs/dpdk/x86_64-native-linuxapp-gcc/lib   
```

# run

1) host上设置大页  mount -t hugetlbfs nodev /mnt/huge   

```
echo 1024 > /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages
```

在创建容器时，除了共享如上面测试helloworld程序时的/sys/bus/pci/devices、/sys/kernel/mm/hugepages、
/sys/devices/system/node、/dev外，还需要共享/var/run/dpdk/目录

即

```
-v /sys/bus/pci/devices:/sys/bus/pci/devices \
-v /sys/kernel/mm/hugepages:/sys/kernel/mm/hugepages \
-v /sys/devices/system/node:/sys/devices/system/node \
-v /dev:/dev \
-v /var/run/dpdk:/var/run/dpdk
```
/var/run/dpdk/目录存放了dpdk创建的ring mempool等其他的一些信息   

注意   
本例使用的是dpdk-18.11版本，共享/var/run/dpdk/目录就可以了   

在dpdk-stable-16.11.1版本中，配置信息是存放在/var/run/.rte_config和/var/run/.rte_hugepage_info目录的
因此在dpdk-stable-16.11.1版本中需要共享   

```
-v /sys/bus/pci/devices:/sys/bus/pci/devices \
-v /sys/kernel/mm/hugepages:/sys/kernel/mm/hugepages \
-v /sys/devices/system/node:/sys/devices/system/node \
-v /dev:/dev \
-v /var/run/.rte_config:/var/run/.rte_config \
-v /var/run/.rte_hugepage_info:/var/run/.rte_hugepage_info
```

# run dpdk2.2

```
docker run -it -d  --net=host --cap-add=NET_ADMIN --name ovs2  \
        -v /sys/bus/pci/devices:/sys/bus/pci/devices \
        -v /mnt/huge:/mnt/huge \
        -v /sys/devices/system/node:/sys/devices/system/node \
        -v /dev:/dev \
        -v /var/run/dpdk:/var/run/dpdk \
         817d4c0c2a1f   /bin/bash
```

```
docker exec -it 1441e31e00b0   /bin/bash
docker rm -f d8ae8f12e31f
```

```
docker export -o  ovs-dpdk-img.tar 1e22992f69ce 
docker import  ovs-dpdk-img.tar   ovs-dpdk-img
```

# bug

```
降低CONFIG_RTE_LOG_HISTORY，设置CONFIG_RTE_LOG_HISTORY=32
CONFIG_RTE_MAX_MEMSEG=1024
echo 1024 > /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages
```




运行成功   
```
./build/helloworld  -c0x1 -n4
EAL: Requesting 1024 pages of size 2MB from socket 0
EAL: TSC frequency is ~3696008 KHz
EAL: Master lcore 0 is ready (tid=31ed2940;cpuset=[0])
hello from core 0
```
## bug1
```
CONFIG_RTE_MAX_MEMSEG=256
echo 256 > /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages
```
```
EAL: Requesting 256 pages of size 2MB from socket 0
EAL: rte_eal_common_log_init(): cannot create log_history mempool
PANIC in rte_eal_init():
Cannot init logs
6: [./build/helloworld() [0x4290d3]]
5: [/lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0xf5) [0x7fed6b012f45]]
4: [./build/helloworld(main+0x6) [0x427d16]]
3: [./build/helloworld(rte_eal_init+0xeae) [0x461b5e]]
2: [./build/helloworld(__rte_panic+0xc9) [0x42225f]]
1: [./build/helloworld(rte_dump_stack+0x18) [0x467bc8]]
Aborted (core dumped
```

## bug2
```
CONFIG_RTE_MAX_MEMSEG=256
echo 512 > /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages
```

```
EAL: Requesting 512 pages of size 2MB from socket 0
EAL: Can only reserve 256 pages from 512 requested
Current CONFIG_RTE_MAX_MEMSEG=256 is not enough
Please either increase it or request less amount of memory.
PANIC in rte_eal_init():
Cannot init memory
6: [./build/helloworld() [0x4290d3]]
5: [/lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0xf5) [0x7f0dcea2af45]]
4: [./build/helloworld(main+0x6) [0x427d16]]
3: [./build/helloworld(rte_eal_init+0xfa7) [0x461c57]]
2: [./build/helloworld(__rte_panic+0xc9) [0x42225f]]
1: [./build/helloworld(rte_dump_stack+0x18) [0x467bc8]]
Aborted (core dumped)
```