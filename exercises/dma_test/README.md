# insmod cma_example.ko 
```
 [root@centos7 dma_test]# insmod cma_example.ko 
[root@centos7 dma_test]# dmsg | tail -n 10
-bash: dmsg: command not found
[root@centos7 dma_test]# dmesg | tail -n 10
[   44.298531] virbr0: port 1(virbr0-nic) entered blocking state
[   44.304255] virbr0: port 1(virbr0-nic) entered disabled state
[   44.310072] device virbr0-nic entered promiscuous mode
[   44.464692] virbr0: port 1(virbr0-nic) entered blocking state
[   44.470423] virbr0: port 1(virbr0-nic) entered listening state
[   44.503122] virbr0: port 1(virbr0-nic) entered disabled state
[   44.608264] IPv6: ADDRCONF(NETDEV_UP): docker0: link is not ready
[ 1783.933457] cma_example: loading out-of-tree module taints kernel.
[ 1783.939681] cma_example: module verification failed: signature and/or required key missing - tainting kernel
[ 1783.950024] misc cma_test: registered.
[root@centos7 dma_test]# ls /dev/cma_test  
/dev/cma_test
[root@centos7 dma_test]# echo 1024 /dev/cma_test  
1024 /dev/cma_test
[root@centos7 dma_test]# dmesg | grep cma
[    0.000000] Memory: 535396160K/536866624K available (8316K kernel code, 1886K rwdata, 3392K rodata, 1472K init, 9784K bss, 1470464K reserved, 0K cma-reserved)
[ 1783.933457] cma_example: loading out-of-tree module taints kernel.
[ 1783.939681] cma_example: module verification failed: signature and/or required key missing - tainting kernel
[ 1783.950024] misc cma_test: registered.
[root@centos7 dma_test]# cat /dev/cma_test
[root@centos7 dma_test]# 
```

 
