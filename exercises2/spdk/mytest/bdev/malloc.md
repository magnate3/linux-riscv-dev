

# run


> ## target
```
root@target:~# ./spdk/scripts/setup.sh 
0000:00:03.0 (1b36 0010): nvme -> uio_pci_generic
root@target:~# spdk/build/bin/nvmf_tgt 
```

> ## bdev (一个controller 一个subsytem 两个namespace)

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

```
root@ubuntux86:# nvme disconnect-all
```


#  两个Controller

```
root@target:~# spdk/scripts/rpc.py nvmf_create_transport -t TCP -u 16384 -p 8 -c 8192
WARNING: max_qpairs_per_ctrlr is deprecated, please use max_io_qpairs_per_ctrlr.
root@target:~# cd spdk
root@target:~/spdk# scripts/rpc.py bdev_malloc_create -b Malloc0 512 512
Malloc0
root@target:~/spdk# scripts/rpc.py bdev_malloc_create -b Malloc1 512 512
Malloc1
root@target:~/spdk# scripts/rpc.py nvmf_create_subsystem nqn.2016-06.io.spdk:cnode1 -a -s SPDK00000000000001 -d SPDK_Controller1
root@target:~/spdk# scripts/rpc.py nvmf_create_subsystem nqn.2016-06.io.spdk:cnode2 -a -s SPDK00000000000002 -d SPDK_Controller2
root@target:~/spdk#  scripts/rpc.py nvmf_subsystem_add_ns nqn.2016-06.io.spdk:cnode1 Malloc0
root@target:~/spdk#  scripts/rpc.py nvmf_subsystem_add_ns nqn.2016-06.io.spdk:cnode2 Malloc1
root@target:~/spdk# scripts/rpc.py nvmf_subsystem_add_listener nqn.2016-06.io.spdk:cnode1 -t tcp -a 192.168.11.22 -s 4420
root@target:~/spdk# iptables -F
root@target:~/spdk#  scripts/rpc.py nvmf_subsystem_add_listener nqn.2016-06.io.spdk:cnode2 -t tcp -a 192.168.11.22 -s 4422
root@target:~/spdk# 
```

> ## client

```
root@ubuntux86:# nvme connect -t tcp -n "nqn.2016-06.io.spdk:cnode1" -a 192.168.11.22 -s 4420
root@ubuntux86:# lsblk | grep nvme1
nvme1n1     259:4    0   512M  0 disk 
```

```
root@ubuntux86:# nvme connect -t tcp -n "nqn.2016-06.io.spdk:cnode2" -a 192.168.11.22 -s 4420
Failed to write to /dev/nvme-fabrics: Input/output error
root@ubuntux86:#  modprobe nvme-fabrics
root@ubuntux86:# nvme connect -t tcp -n "nqn.2016-06.io.spdk:cnode2" -a 192.168.11.22 -s 4422
root@ubuntux86:# lsblk | grep nvme1
nvme1n1     259:4    0   512M  0 disk
```
nvme connect -t tcp -n "nqn.2016-06.io.spdk:cnode2" -a 192.168.11.22 -s 4422 在target端报错了  

```
[2024-05-10 02:25:26.200050] ctrlr.c: 625:nvmf_qpair_access_allowed: *ERROR*: Subsystem 'nqn.2016-06.io.spdk:cnode2' does not allow host 'nqn.2014-08.or.
[2024-05-10 02:26:43.522994] tcp.c: 748:nvmf_tcp_listen: *NOTICE*: *** NVMe/TCP Target Listening on 192.168.11.22 port 4422 ***
[2024-05-10 02:26:53.725734] ctrlr.c:2389:nvmf_ctrlr_identify: *ERROR*: Identify command with unsupported CNS 0x06
```


> ## clear



```
root@target:~/spdk# scripts/rpc.py nvmf_subsystem_remove_listener nqn.2016-06.io.spdk:cnode1 -t tcp -a 192.168.11.22 -s 4420
root@target:~/spdk#
```

```
root@target:~/spdk# scripts/rpc.py get_nvmf_subsystems
[
  {
    "nqn": "nqn.2014-08.org.nvmexpress.discovery",
    "subtype": "Discovery",
    "listen_addresses": [],
    "allow_any_host": true,
    "hosts": []
  },
  {
    "nqn": "nqn.2016-06.io.spdk:cnode1",
    "subtype": "NVMe",
    "listen_addresses": [],
    "allow_any_host": true,
    "hosts": [],
    "serial_number": "SPDK00000000000001",
    "model_number": "SPDK_Controller1",
    "max_namespaces": 32,
    "namespaces": [
      {
        "nsid": 1,
        "bdev_name": "Malloc0",
        "name": "Malloc0",
        "uuid": "607dbf77-53f8-48a4-ad57-5899b952f53b"
      }
    ]
  },
  {
    "nqn": "nqn.2016-06.io.spdk:cnode2",
    "subtype": "NVMe",
    "listen_addresses": [
      {
        "transport": "TCP",
        "trtype": "TCP",
        "adrfam": "IPv4",
        "traddr": "192.168.11.22",
        "trsvcid": "4422"
      }
    ],
    "allow_any_host": true,
    "hosts": [],
    "serial_number": "SPDK00000000000002",
    "model_number": "SPDK_Controller2",
    "max_namespaces": 32,
    "namespaces": [
      {
        "nsid": 1,
        "bdev_name": "Malloc1",
        "name": "Malloc1",
        "uuid": "c23b8450-47d7-4970-b365-04f6d2d0f097"
      }
    ]
  }
]
```


```
root@target:~/spdk# scripts/rpc.py nvmf_subsystem_remove_ns nqn.2016-06.io.spdk:cnode2 1
root@target:~/spdk# 
root@target:~/spdk# scripts/rpc.py  delete_nvmf_subsystem nqn.2016-06.io.spdk:cnode2
delete_nvmf_subsystem is deprecated, use nvmf_delete_subsystem instead.
root@target:~/spdk# 
```

# 一个controller 两个nvmf_create_subsystem

```
root@target:~# spdk/scripts/rpc.py nvmf_create_transport -t TCP -u 16384 -p 8 -c 8192
WARNING: max_qpairs_per_ctrlr is deprecated, please use max_io_qpairs_per_ctrlr.
root@target:~# cd spdk
root@target:~/spdk# scripts/rpc.py bdev_malloc_create -b Malloc0 512 512
Malloc0
root@target:~/spdk# scripts/rpc.py bdev_malloc_create -b Malloc1 512 512
Malloc1

```

```
root@target:~/spdk# scripts/rpc.py get_nvmf_subsystems
[
  {
    "nqn": "nqn.2014-08.org.nvmexpress.discovery",
    "subtype": "Discovery",
    "listen_addresses": [],
    "allow_any_host": true,
    "hosts": []
  }
]
get_nvmf_subsystems is deprecated, use nvmf_get_subsystems instead.
root@target:~/spdk# 
```

<nqn.2016-06.io.spdk:cnode1,SPDK00000000000001>

<nqn.2016-06.io.spdk:cnode2,SPDK00000000000002>   
```
root@target:~/spdk# scripts/rpc.py nvmf_create_subsystem nqn.2016-06.io.spdk:cnode1 -a -s SPDK00000000000001 -d SPDK_Controller1
root@target:~/spdk# scripts/rpc.py nvmf_create_subsystem nqn.2016-06.io.spdk:cnode2 -a -s SPDK00000000000002 -d SPDK_Controller1
root@target:~/spdk# scripts/rpc.py nvmf_subsystem_add_ns nqn.2016-06.io.spdk:cnode1 Malloc0
root@target:~/spdk# scripts/rpc.py nvmf_subsystem_add_ns nqn.2016-06.io.spdk:cnode2 Malloc1
```

```
root@target:~/spdk# scripts/rpc.py nvmf_subsystem_add_listener nqn.2016-06.io.spdk:cnode1 -t tcp -a 192.168.11.22 -s 4420
root@target:~/spdk# scripts/rpc.py nvmf_subsystem_add_listener nqn.2016-06.io.spdk:cnode2 -t tcp -a 192.168.11.22 -s 4422
root@target:~/spdk# 
```
<-a 192.168.11.22 -s 4420>   
<-a 192.168.11.22 -s 4422>   


```
root@target:~/spdk# scripts/rpc.py get_nvmf_subsystems
[
  {
    "nqn": "nqn.2014-08.org.nvmexpress.discovery",
    "subtype": "Discovery",
    "listen_addresses": [],
    "allow_any_host": true,
    "hosts": []
  },
  {
    "nqn": "nqn.2016-06.io.spdk:cnode1",
    "subtype": "NVMe",
    "listen_addresses": [
      {
        "transport": "TCP",
        "trtype": "TCP",
        "adrfam": "IPv4",
        "traddr": "192.168.11.22",
        "trsvcid": "4420"
      }
    ],
    "allow_any_host": true,
    "hosts": [],
    "serial_number": "SPDK00000000000001",
    "model_number": "SPDK_Controller1",
    "max_namespaces": 32,
    "namespaces": [
      {
        "nsid": 1,
        "bdev_name": "Malloc0",
        "name": "Malloc0",
        "uuid": "607dbf77-53f8-48a4-ad57-5899b952f53b"
      }
    ]
  },
  {
    "nqn": "nqn.2016-06.io.spdk:cnode2",
    "subtype": "NVMe",
    "listen_addresses": [
      {
        "transport": "TCP",
        "trtype": "TCP",
        "adrfam": "IPv4",
        "traddr": "192.168.11.22",
        "trsvcid": "4422"
      }
    ],
    "allow_any_host": true,
    "hosts": [],
    "serial_number": "SPDK00000000000002",
    "model_number": "SPDK_Controller1",
    "max_namespaces": 32,
    "namespaces": [
      {
        "nsid": 1,
        "bdev_name": "Malloc1",
        "name": "Malloc1",
        "uuid": "c23b8450-47d7-4970-b365-04f6d2d0f097"
      }
    ]
  }
]
get_nvmf_subsystems is deprecated, use nvmf_get_subsystems instead.
root@target:~/spdk# 
```


> ## client
```
root@ubuntux86:# nvme connect -t tcp -n "nqn.2016-06.io.spdk:cnode2" -a 192.168.11.22 -s 4422
root@ubuntux86:# lsblk | grep nvme
nvme0n1     259:0    0 238.5G  0 disk 
├─nvme0n1p1 259:1    0   512M  0 part /boot/efi
└─nvme0n1p2 259:2    0   238G  0 part /
nvme1n1     259:6    0   512M  0 disk 
root@ubuntux86:# nvme connect -t tcp -n "nqn.2016-06.io.spdk:cnode2" -a 192.168.11.22 -s 4420
Failed to write to /dev/nvme-fabrics: Input/output error
root@ubuntux86:# nvme connect -t tcp -n "nqn.2016-06.io.spdk:cnode1" -a 192.168.11.22 -s 4420
root@ubuntux86:# lsblk | grep nvme
nvme0n1     259:0    0 238.5G  0 disk 
├─nvme0n1p1 259:1    0   512M  0 part /boot/efi
└─nvme0n1p2 259:2    0   238G  0 part /
nvme1n1     259:6    0   512M  0 disk 
nvme2n1     259:8    0   512M  0 disk 
root@ubuntux86:# 

```


> ## 两个subsystem通用一个tcp port


+  server

```
root@target:~/spdk# scripts/rpc.py nvmf_subsystem_remove_listener nqn.2016-06.io.spdk:cnode2 -t tcp -a 192.168.11.22 -s 4422
root@target:~/spdk# scripts/rpc.py nvmf_subsystem_add_listener nqn.2016-06.io.spdk:cnode2 -t tcp -a 192.168.11.22 -s 4420
root@target:~/spdk# 
```


+  client    
```
root@ubuntux86:# nvme disconnect-all
root@ubuntux86:# nvme connect -t tcp -n "nqn.2016-06.io.spdk:cnode1" -a 192.168.11.22 -s 4420
root@ubuntux86:# lsblk | grep nvme
nvme0n1     259:0    0 238.5G  0 disk 
├─nvme0n1p1 259:1    0   512M  0 part /boot/efi
└─nvme0n1p2 259:2    0   238G  0 part /
nvme1n1     259:8    0   512M  0 disk 
root@ubuntux86:# nvme connect -t tcp -n "nqn.2016-06.io.spdk:cnode2" -a 192.168.11.22 -s 4420
root@ubuntux86:# lsblk | grep nvme
nvme0n1     259:0    0 238.5G  0 disk 
├─nvme0n1p1 259:1    0   512M  0 part /boot/efi
└─nvme0n1p2 259:2    0   238G  0 part /
nvme1n1     259:8    0   512M  0 disk 
nvme2n1     259:10   0   512M  0 disk 
root@ubuntux86:# 
```

# references

[一个controller,两个subsystem通用](https://cloud.tencent.com/developer/article/1694515)    



 
