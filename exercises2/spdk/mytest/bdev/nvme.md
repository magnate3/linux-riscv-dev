
# run


+ 1) 启动nvmf_target

```
root@target:~/spdk# ./build/bin/nvmf_tgt &
[1] 1048
```

+ 2) nvmf 设备配置

+ a) 准备工作，把nvme从内核解绑  

```
HUGEMEM=4096 PCI_ALLOWED="0000:00:03.0" ./scripts/setup.sh
```

+ b) 把nvme 盘映射为一个 bdev  

```
root@target:~/spdk# ./scripts/rpc.py bdev_nvme_attach_controller -b Nvme0 -t PCIe -a 0000:00:03.0
Nvme0n1
root@target:~/spdk# ./scripts/rpc.py bdev_nvme_get_controllers
[
  {
    "name": "Nvme0",
    "trid": {
      "trtype": "PCIe",
      "traddr": "0000:00:03.0"
    }
  }
]
root@target:~/spdk# 
```

+ c) 创建一个NVM subsystem   

```
root@target:~/spdk#  ./scripts/rpc.py nvmf_create_subsystem nqn.2022-03.io.spdk:cnode1 -a -s SPDK00000000000002 -d SPDK_Controller1
root@target:~/spdk# 
```

+ d) 用nvme bdev给NVMf subsystem 增加一个namespace,即把nvme bdev和NVMf subsystem 关联   
名字 Nvme0 + n1   
```
root@target:~/spdk# ./scripts/rpc.py nvmf_subsystem_add_ns nqn.2022-03.io.spdk:cnode1 Nvme0n1
root@target:~/spdk# 
```

+ e)  创建相应的tcp transport

```
root@target:~/spdk# ./scripts/rpc.py nvmf_create_transport -t tcp -u 8192 -p 4 -c 0
WARNING: max_qpairs_per_ctrlr is deprecated, please use max_io_qpairs_per_ctrlr.
[2024-05-07 07:41:24.872580] nvmf_rpc.c:1755:nvmf_rpc_decode_max_qpairs: *WARNING*: Parameter max_qpairs_per_ctrlr is deprecated, use max_io_qpairs_per_.
[2024-05-07 07:41:24.873647] nvmf_rpc.c:1755:nvmf_rpc_decode_max_qpairs: *WARNING*: Parameter max_qpairs_per_ctrlr is deprecated, use max_io_qpairs_per_.
[2024-05-07 07:41:24.874953] tcp.c: 554:nvmf_tcp_create: *NOTICE*: *** TCP Transport Init ***
root@target:~/spdk# ./scripts/rpc.py nvmf_get_transports
[
  {
    "trtype": "TCP",
    "max_queue_depth": 128,
    "max_io_qpairs_per_ctrlr": 3,
    "in_capsule_data_size": 0,
    "max_io_size": 131072,
    "io_unit_size": 8192,
    "max_aq_depth": 128,
    "num_shared_buffers": 511,
    "buf_cache_size": 32,
    "dif_insert_or_strip": false,
    "c2h_success": true,
    "sock_priority": 0,
    "abort_timeout_sec": 1
  }
]
```

+ f) 监听nvmf subsystem 端口   
让nvmf subsystem监听对应的端口，至此一块nvmf 盘已经建立成功，可以成功的被远端host discover到。
```
root@target:~/spdk# ./scripts/rpc.py nvmf_subsystem_add_listener nqn.2022-03.io.spdk:cnode1 -t tcp -a 192.168.11.22 -s 4420
[2024-05-07 07:42:37.136954] tcp.c: 748:nvmf_tcp_listen: *NOTICE*: *** NVMe/TCP Target Listening on 192.168.11.22 port 4420 ***
root@target:~/spdk# 
```
+ 3) perf 下发io   
./build/examples/perf   
```
root@target:~/spdk# ./scripts/rpc.py nvmf_subsystem_add_listener nqn.2022-03.io.spdk:cnode1 -t tcp -a 192.168.11.22 -s 4420
[2024-05-07 07:42:37.136954] tcp.c: 748:nvmf_tcp_listen: *NOTICE*: *** NVMe/TCP Target Listening on 192.168.11.22 port 4420 ***
root@target:~/spdk# ./build/examples/perf -i 0 -q 128 -o 4096 -w rw -M 50 -t 60 -r 'trtype:TCP adrfam:IPv4 traddr:192.168.11.22 trsvcid:4420'
[2024-05-07 07:43:16.386180] Starting SPDK v21.01.2-pre git sha1 752ceb0c1 / DPDK 20.11.0 initialization...
[2024-05-07 07:43:16.386234] [ DPDK EAL parameters: [2024-05-07 07:43:16.386304] perf [2024-05-07 07:43:16.386394] -c 0x1 [2024-05-07 07:43:16.386483] -]
EAL: No available hugepages reported in hugepages-1048576kB
EAL: No legacy callbacks, legacy socket not created
Initializing NVMe Controllers
Attached to NVMe over Fabrics controller at 192.168.11.22:4420: nqn.2022-03.io.spdk:cnode1
controller IO queue size 128 less than required
Consider using lower queue depth or small IO size because IO requests may be queued at the NVMe driver.
Associating TCP (addr:192.168.11.22 subnqn:nqn.2022-03.io.spdk:cnode1) NSID 1 with lcore 0
Initialization complete. Launching workers.
========================================================
                                                                                                                  Latency(us)
Device Information                                                            :       IOPS      MiB/s    Average        min        max
TCP (addr:192.168.11.22 subnqn:nqn.2022-03.io.spdk:cnode1) NSID 1 from core  0:    4287.03      16.75   29869.93   10975.25   58942.59
========================================================
Total                                                                         :    4287.03      16.75   29869.93   10975.25   58942.59

root@target:~/spdk# 
```

+ 4) iostat.py 观察
```
root@target:~# ./spdk/scripts/iostat.py -m -i 1 -t 180
cpu_stat:  user_stat  nice_stat  system_stat  iowait_stat  steal_stat  idle_stat
           5.47%      0.00%      0.10%        0.07%        0.00%       94.36%   

Device   tps    MB_read/s  MB_wrtn/s  MB_dscd/s  MB_read  MB_wrtn  MB_dscd
Nvme0n1  22.84  0.04       0.04       0.00       119.51   119.55   0.00   

cpu_stat:  user_stat  nice_stat  system_stat  iowait_stat  steal_stat  idle_stat
           10.36%     0.00%      6.25%        0.00%        0.00%       83.39%   

Device   tps      MB_read/s  MB_wrtn/s  MB_dscd/s  MB_read  MB_wrtn  MB_dscd
Nvme0n1  4209.42  8.26       8.19       0.00       8.37     8.30     0.00   

cpu_stat:  user_stat  nice_stat  system_stat  iowait_stat  steal_stat  idle_stat
           10.26%     0.00%      6.29%        0.00%        0.00%       83.44%   

Device   tps      MB_read/s  MB_wrtn/s  MB_dscd/s  MB_read  MB_wrtn  MB_dscd
Nvme0n1  4329.15  8.50       8.41       0.00       8.56     8.46     0.00   

^CTraceback (most recent call last):
  File "./spdk/scripts/iostat.py", line 356, in <module>
    io_stat_display_loop(args)
  File "./spdk/scripts/iostat.py", line 286, in io_stat_display_loop
    time.sleep(interval)
KeyboardInterrupt

root@target:~# 
```



# 两个nvme

```
root@target:~# lsblk
NAME    MAJ:MIN RM  SIZE RO TYPE MOUNTPOINT
fd0       2:0    1    4K  0 disk 
loop0     7:0    0 63.2M  1 loop /snap/core20/1634
loop1     7:1    0 63.3M  1 loop /snap/core20/1879
loop2     7:2    0 67.8M  1 loop /snap/lxd/22753
loop3     7:3    0 53.2M  1 loop /snap/snapd/19122
loop4     7:4    0 91.9M  1 loop /snap/lxd/24061
sda       8:0    0 22.2G  0 disk 
├─sda1    8:1    0 22.1G  0 part /
├─sda14   8:14   0    4M  0 part 
└─sda15   8:15   0  106M  0 part /boot/efi
sr0      11:0    1 1024M  0 rom  
nvme1n1 259:0    0  512M  0 disk 
nvme0n1 259:1    0  512M  0 disk 
root@target:~#
```
+ 1) setup.sh
```
root@target:~# PCI_ALLOWED="0000:00:03.0 0000:00:04.0" ./spdk/scripts/setup.sh
0000:00:03.0 (1b36 0010): nvme -> uio_pci_generic
0000:00:04.0 (1b36 0010): nvme -> uio_pci_generic
root@target:~# 
```

+ 2) bdev_nvme_attach_controller  

```
root@target:~/spdk# ./scripts/rpc.py bdev_nvme_attach_controller -b Nvme0 -t PCIe -a 0000:00:03.0
Nvme0n1
root@target:~/spdk# ./scripts/rpc.py bdev_nvme_attach_controller -b Nvme1 -t PCIe -a 0000:00:04.0
Nvme1n1
root@target:~/spdk# 
```

```
root@target:~/spdk# ./scripts/rpc.py bdev_nvme_get_controllers
[
  {
    "name": "Nvme0",
    "trid": {
      "trtype": "PCIe",
      "traddr": "0000:00:03.0"
    }
  },
  {
    "name": "Nvme1",
    "trid": {
      "trtype": "PCIe",
      "traddr": "0000:00:04.0"
    }
  }
]
```

+ 3 nvmf_create_subsystem   
```
root@target:~/spdk# ./scripts/rpc.py nvmf_create_subsystem nqn.2022-03.io.spdk:cnode1 -a -s SPDK00000000000002 -d SPDK_Controller1
root@target:~/spdk# 
```

+ 4 nvmf_subsystem_add_ns 2个 bdv   
Nvme2 不存在   
```
root@target:~/spdk# ./scripts/rpc.py nvmf_subsystem_add_ns nqn.2022-03.io.spdk:cnode1 Nvme0n1
root@target:~/spdk# ./scripts/rpc.py nvmf_subsystem_add_ns nqn.2022-03.io.spdk:cnode1 Nvme1n1
root@target:~/spdk# ./scripts/rpc.py nvmf_subsystem_add_ns nqn.2022-03.io.spdk:cnode1 Nvme2n1
request:
{
  "nqn": "nqn.2022-03.io.spdk:cnode1",
  "namespace": {
    "bdev_name": "Nvme2n1"
  },
  "method": "nvmf_subsystem_add_ns",
  "req_id": 1
}
Got JSON-RPC error response
response:
{
  "code": -32602,
  "message": "Invalid parameters"
}
root@target:~/spdk# 
```

```
root@target:~/spdk# ./scripts/rpc.py nvmf_get_subsystems
[
  {
    "nqn": "nqn.2014-08.org.nvmexpress.discovery",
    "subtype": "Discovery",
    "listen_addresses": [],
    "allow_any_host": true,
    "hosts": []
  },
  {
    "nqn": "nqn.2022-03.io.spdk:cnode1",
    "subtype": "NVMe",
    "listen_addresses": [],
    "allow_any_host": true,
    "hosts": [],
    "serial_number": "SPDK00000000000002",
    "model_number": "SPDK_Controller1",
    "max_namespaces": 32,
    "namespaces": [
      {
        "nsid": 1,
        "bdev_name": "Nvme0n1",
        "name": "Nvme0n1",
        "uuid": "422f13af-d767-40b1-9412-cafa9448a0a9"
      },
      {
        "nsid": 2,
        "bdev_name": "Nvme1n1",
        "name": "Nvme1n1",
        "uuid": "5b06fd94-0a1a-4f40-8d29-dc8051f8b908"
      }
    ]
  }
]
```

+ 5  nvmf_create_transport and nvmf_subsystem_add_listener   
```
root@target:~/spdk# ./scripts/rpc.py nvmf_create_transport -t tcp -u 8192 -p 4 -c 0
WARNING: max_qpairs_per_ctrlr is deprecated, please use max_io_qpairs_per_ctrlr.
root@target:~/spdk# ./scripts/rpc.py nvmf_subsystem_add_listener nqn.2022-03.io.spdk:cnode1 -t tcp -a 192.168.11.22 -s 4420
root@target:~/spdk# 
```

+ 6 perf

```
Associating TCP (addr:192.168.11.22 subnqn:nqn.2022-03.io.spdk:cnode1) NSID 1 with lcore 0
Associating TCP (addr:192.168.11.22 subnqn:nqn.2022-03.io.spdk:cnode1) NSID 2 with lcore 0
```
两个namespaces   

```
root@target:~/spdk# ./build/examples/perf -i 0 -q 128 -o 4096 -w rw -M 50 -t 60 -r 'trtype:TCP adrfam:IPv4 traddr:192.168.11.22 trsvcid:4420'
[2024-05-07 08:06:11.708098] Starting SPDK v21.01.2-pre git sha1 752ceb0c1 / DPDK 20.11.0 initialization...
[2024-05-07 08:06:11.708151] [ DPDK EAL parameters: [2024-05-07 08:06:11.708159] perf [2024-05-07 08:06:11.708166] -c 0x1 [2024-05-07 08:06:11.708171] --no-pci [2024-05-07 08:06:11.708176] --log-level=lib.eal:6 [2024-05-07 08:06:11.708181] --log-level=lib.cryptodev:5 [2024-05-07 08:06:11.708187] --log-level=user1:6 [2024-05-07 08:06:11.708190] --iova-mode=pa [2024-05-07 08:06:11.708195] --base-virtaddr=0x200000000000 [2024-05-07 08:06:11.708201] --match-allocations [2024-05-07 08:06:11.708206] --file-prefix=spdk0 [2024-05-07 08:06:11.708212] --proc-type=auto [2024-05-07 08:06:11.708216] ]
EAL: No available hugepages reported in hugepages-1048576kB
EAL: No legacy callbacks, legacy socket not created
Initializing NVMe Controllers
Attached to NVMe over Fabrics controller at 192.168.11.22:4420: nqn.2022-03.io.spdk:cnode1
controller IO queue size 128 less than required
Consider using lower queue depth or small IO size because IO requests may be queued at the NVMe driver.
controller IO queue size 128 less than required
Consider using lower queue depth or small IO size because IO requests may be queued at the NVMe driver.
Associating TCP (addr:192.168.11.22 subnqn:nqn.2022-03.io.spdk:cnode1) NSID 1 with lcore 0
Associating TCP (addr:192.168.11.22 subnqn:nqn.2022-03.io.spdk:cnode1) NSID 2 with lcore 0
Initialization complete. Launching workers.
========================================================
                                                                                                                  Latency(us)
Device Information                                                            :       IOPS      MiB/s    Average        min        max
TCP (addr:192.168.11.22 subnqn:nqn.2022-03.io.spdk:cnode1) NSID 1 from core  0:    4294.90      16.78   29806.63    9737.95   51880.46
TCP (addr:192.168.11.22 subnqn:nqn.2022-03.io.spdk:cnode1) NSID 2 from core  0:    4293.80      16.77   29816.63    9709.88   51884.40
========================================================
Total                                                                         :    8588.70      33.55   29811.63    9709.88   51884.40

root@target:~/spdk# 
```

> ## host

```
root@ubuntux86:# nvme connect -t tcp -n "nqn.2022-03.io.spdk:cnode1" -a 192.168.11.22 -s 4420
```
```
root@ubuntux86:# lsblk | grep nvme
nvme0n1     259:0    0 238.5G  0 disk 
├─nvme0n1p1 259:1    0   512M  0 part /boot/efi
└─nvme0n1p2 259:2    0   238G  0 part /
nvme1n1     259:4    0   512M  0 disk 
nvme1n2     259:6    0   512M  0 disk 
root@ubuntux86:# nvme disconnect-all
```

# hellorld world
bdev_get_bdevs   
```
scripts/rpc.py bdev_get_bdevs
[
  {
    "name": "Nvme0n1",
    "aliases": [],
    "product_name": "NVMe disk",
    "block_size": 512,
    "num_blocks": 1048576,
    "uuid": "422f13af-d767-40b1-9412-cafa9448a0a9",
    "assigned_rate_limits": {
      "rw_ios_per_sec": 0,
      "rw_mbytes_per_sec": 0,
      "r_mbytes_per_sec": 0,
      "w_mbytes_per_sec": 0
    },
    "claimed": true,
    "zoned": false,
    "supported_io_types": {
      "read": true,
      "write": true,
      "unmap": true,
      "write_zeroes": true,
      "flush": true,
      "reset": true,
      "nvme_admin": true,
      "nvme_io": true
    },
    "driver_specific": {
      "nvme": {
        "pci_address": "0000:00:03.0",
        "trid": {
          "trtype": "PCIe",
          "traddr": "0000:00:03.0"
        },
        "ctrlr_data": {
          "vendor_id": "0x1b36",
          "model_number": "QEMU NVMe Ctrl",
          "serial_number": "nvme-dev",
          "firmware_revision": "1.0",
          "subnqn": "nqn.2019-08.org.qemu:nvme-dev",
          "oacs": {
            "security": 0,
            "format": 1,
            "firmware": 0,
            "ns_manage": 1
          }
        },
        "vs": {
          "nvme_version": "1.4"
        },
        "csts": {
          "rdy": 1,
          "cfs": 0
        },
        "ns_data": {
          "id": 1
        }
      }
    }
  },
  {
    "name": "Nvme1n1",
    "aliases": [],
    "product_name": "NVMe disk",
    "block_size": 512,
    "num_blocks": 1048576,
    "uuid": "5b06fd94-0a1a-4f40-8d29-dc8051f8b908",
    "assigned_rate_limits": {
      "rw_ios_per_sec": 0,
      "rw_mbytes_per_sec": 0,
      "r_mbytes_per_sec": 0,
      "w_mbytes_per_sec": 0
    },
    "claimed": true,
    "zoned": false,
    "supported_io_types": {
      "read": true,
      "write": true,
      "unmap": true,
      "write_zeroes": true,
      "flush": true,
      "reset": true,
      "nvme_admin": true,
      "nvme_io": true
    },
    "driver_specific": {
      "nvme": {
        "pci_address": "0000:00:04.0",
        "trid": {
          "trtype": "PCIe",
          "traddr": "0000:00:04.0"
        },
        "ctrlr_data": {
          "vendor_id": "0x1b36",
          "model_number": "QEMU NVMe Ctrl",
          "serial_number": "nvme-de2v",
          "firmware_revision": "1.0",
          "subnqn": "nqn.2019-08.org.qemu:nvme-de2v",
          "oacs": {
            "security": 0,
            "format": 1,
            "firmware": 0,
            "ns_manage": 1
          }
        },
        "vs": {
          "nvme_version": "1.4"
        },
        "csts": {
          "rdy": 1,
          "cfs": 0
        },
        "ns_data": {
          "id": 1
        }
      }
    }
  }
]
```


```
root@target:~/spdk/examples/hello_nvme_bdev# ../../build/examples/my_hello_nvme_bdev -r "trtype:tcp adrfam:IPv4 traddr:192.168.11.22 trsvcid:4420 subnqn:nqn.2022-03.io.spdk:cnode1"
[2024-05-07 09:12:16.563700] Starting SPDK v21.01.2-pre git sha1 752ceb0c1 / DPDK 20.11.0 initialization...
[2024-05-07 09:12:16.564261] [ DPDK EAL parameters: [2024-05-07 09:12:16.564285] hello_bdev [2024-05-07 09:12:16.564290] --no-shconf [2024-05-07 09:12:16.564294] -c 0x1 [2024-05-07 09:12:16.564297] --log-level=lib.eal:6 [2024-05-07 09:12:16.564300] --log-level=lib.cryptodev:5 [2024-05-07 09:12:16.564304] --log-level=user1:6 [2024-05-07 09:12:16.564308] --iova-mode=pa [2024-05-07 09:12:16.564311] --base-virtaddr=0x200000000000 [2024-05-07 09:12:16.564317] --match-allocations [2024-05-07 09:12:16.564323] --file-prefix=spdk_pid1482 [2024-05-07 09:12:16.564328] ]
EAL: No available hugepages reported in hugepages-1048576kB
EAL: No legacy callbacks, legacy socket not created
[2024-05-07 09:12:16.669741] app.c: 538:spdk_app_start: *NOTICE*: Total cores available: 1
[2024-05-07 09:12:16.795007] reactor.c: 915:reactor_run: *NOTICE*: Reactor started on core 0
[2024-05-07 09:12:16.795083] accel_engine.c: 692:spdk_accel_engine_initialize: *NOTICE*: Accel engine initialized to use software engine.
[2024-05-07 09:12:16.831215] hello_nvme_bdev.c: 169:hello_start: *NOTICE*: Successfully started the application
[2024-05-07 09:12:16.831261] hello_nvme_bdev.c: 190:hello_start: *ERROR*: Could not find the bdev: Nvme0n1
[2024-05-07 09:12:16.831267] app.c: 629:spdk_app_stop: *WARNING*: spdk_app_stop'd on non-zero
[2024-05-07 09:12:16.889567] hello_nvme_bdev.c: 306:main: *ERROR*: ERROR starting application
root@target:~/spdk/examples/hello_nvme_bdev# 
```

# gen_nvme.sh --json-with-subsystems


```
root@target:~/spdk# ./scripts/gen_nvme.sh --json-with-subsystems > bdev.json
root@target:~/spdk# cat bdev.json 
{
"subsystems": [
{
"subsystem": "bdev",
"config": [
{
"method": "bdev_nvme_attach_controller",
"params": {
"trtype": "PCIe",
"name":"Nvme0",
"traddr":"0000:00:03.0"
}
},{
"method": "bdev_nvme_attach_controller",
"params": {
"trtype": "PCIe",
"name":"Nvme1",
"traddr":"0000:00:04.0"
}
}
]
}
]
}
```

# nvme/hello_world

```
./build/examples/hello_world 
```

# raid 



The first step is to connect the drives:    
```
rpc.py bdev_nvme_attach_controller -b nvme0 -t PCIe -a 0000:02:00.0
rpc.py bdev_nvme_attach_controller -b nvme1 -t PCIe -a 0000:45:00.0
rpc.py bdev_nvme_attach_controller -b nvme2 -t PCIe -a 0000:03:00.0
rpc.py bdev_nvme_attach_controller -b nvme3 -t PCIe -a 0000:81:00.0
rpc.py bdev_nvme_attach_controller -b nvme4 -t PCIe -a 0000:84:00.0
rpc.py bdev_nvme_attach_controller -b nvme5 -t PCIe -a 0000:41:00.0
rpc.py bdev_nvme_attach_controller -b nvme6 -t PCIe -a 0000:46:00.0
rpc.py bdev_nvme_attach_controller -b nvme7 -t PCIe -a 0000:44:00.0
rpc.py bdev_nvme_attach_controller -b nvme8 -t PCIe -a 0000:43:00.0
rpc.py bdev_nvme_attach_controller -b nvme9 -t PCIe -a 0000:82:00.0
rpc.py bdev_nvme_attach_controller -b nvme10 -t PCIe -a 0000:48:00.0
rpc.py bdev_nvme_attach_controller -b nvme11 -t PCIe -a 0000:47:00.0
rpc.py bdev_nvme_attach_controller -b nvme12 -t PCIe -a 0000:83:00.0
rpc.py bdev_nvme_attach_controller -b nvme13 -t PCIe -a 0000:42:00.0
rpc.py bdev_nvme_attach_controller -b nvme14 -t PCIe -a 0000:01:00.0
rpc.py bdev_nvme_attach_controller -b nvme15 -t PCIe -a 0000:04:00.0
```
After that we can create an array:   
```
rpc.py bdev_raid_create -n raid5 -z 64 -r raid5f -b "nvme0n1 nvme1n1 nvme2n1 nvme3n1 nvme4n1 nvme5n1 nvme6n1 nvme7n1 nvme8n1 nvme9n1 nvme10n1 nvme11n1 nvme12n1 nvme13n1 nvme14n1 nvme15n1"
```

# references

[SPDK RAID EVALUATION AND IMPROVEMENT](https://xinnor.io/blog/spdk-raid-evaluation-and-improvement/)  

[通过spdk 脚本iostat.py 观测bdev 的IO统计](https://zhuanlan.zhihu.com/p/593455010)   


[spdk bdev layer](https://hackmd.io/@Hyam/spdk_test)   