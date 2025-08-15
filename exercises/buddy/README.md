
# insmod  buddy_test.ko 
```
[root@centos7 buddy]# insmod  buddy_test.ko 
[root@centos7 buddy]# dmesg | tail -n 10
[   45.082054] virbr0: port 1(virbr0-nic) entered blocking state
[   45.087786] virbr0: port 1(virbr0-nic) entered listening state
[   45.118649] virbr0: port 1(virbr0-nic) entered disabled state
[   45.241805] IPv6: ADDRCONF(NETDEV_UP): docker0: link is not ready
[ 1434.116406] buddy_test: loading out-of-tree module taints kernel.
[ 1434.122537] buddy_test: module verification failed: signature and/or required key missing - tainting kernel
[ 1434.132729] alloc_p
```