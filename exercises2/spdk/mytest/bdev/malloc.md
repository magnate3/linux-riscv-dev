

# run


> ## target
```
root@target:~# ./spdk/scripts/setup.sh 
0000:00:03.0 (1b36 0010): nvme -> uio_pci_generic
root@target:~# spdk/build/bin/nvmf_tgt 
```

> ## bdev

```
root@target:~/spdk# 
root@target:~/spdk# scripts/rpc.py nvmf_create_transport -t TCP -u 16384 -p 8 -c 8192
WARNING: max_qpairs_per_ctrlr is deprecated, please use max_io_qpairs_per_ctrlr.
root@target:~/spdk# scripts/rpc.py bdev_malloc_create -b Malloc0 512 512
Malloc0
root@target:~/spdk# scripts/rpc.py bdev_malloc_create -b Malloc1 512 512
Malloc1
root@target:~/spdk# scripts/rpc.py nvmf_create_subsystem nqn.2016-06.io.spdk:cnode1 -a -s SPDK00000000000001 -d SPDK_Controller1
root@target:~/spdk# scripts/rpc.py nvmf_subsystem_add_ns nqn.2016-06.io.spdk:cnode1 Malloc0
root@target:~/spdk# scripts/rpc.py nvmf_subsystem_add_ns nqn.2016-06.io.spdk:cnode1 Malloc1
root@target:~/spdk# scripts/rpc.py nvmf_subsystem_add_listener nqn.2016-06.io.spdk:cnode1 -t tcp -a 192.168.11.22 -s 4420
root@target:~/spdk# iptables -F
root@target:~/spdk# 
```

> ## client

```
root@ubuntux86:# ls /dev/nvme-fabrics
ls: cannot access '/dev/nvme-fabrics': No such file or directory
root@ubuntux86:# modprobe nvme-fabrics
root@ubuntux86:# ls /dev/nvme-fabrics
/dev/nvme-fabrics
root@ubuntux86:# nvme connect -t tcp -n "nqn.2016-06.io.spdk:cnode1" -a 192.168.11.22 -s 4420
```

```
root@ubuntux86:# lsblk | grep nvme1
nvme1n1     259:4    0   512M  0 disk 
nvme1n2     259:6    0   512M  0 disk 
root@ubuntux86:# 
```