
# run

+ 1) blobcli.json 
```
root@target:~# ./spdk/scripts/setup.sh 
0000:00:03.0 (1b36 0010): nvme -> uio_pci_generic
root@target:~# mkdir blob
root@target:~# ./spdk/scripts/gen_nvme.sh --json-with-subsystems > blob/blobcli.json  
root@target:~# cat blob/blobcli.json  
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
}
]
}
]
}
root@target:~# 
```

#  blobcli

 -b 一定是 name+n1 这样子格式   
 
 ```
 root@target:~# ./spdk/build/examples/blobcli -b Nvme0n1 -i blob/blobcli.json
Your entire blobstore will be destroyed. Are you sure? (y/n) y
Error: No config file found.
To create a config file named 'blobcli.json' for your NVMe device:
   <path to spdk>/scripts/gen_nvme.sh --json-with-subsystems > blobcli.json
and then re-run the cli tool.
 ```
 
 进入blob目录   
 
 
 ```
 root@target:~/blob# pwd
/root/blob
root@target:~/blob# ls
blobcli.json
root@target:~/blob# ../spdk/build/examples/blobcli -b Nvme0n1 -i blob/blobcli.json
Your entire blobstore will be destroyed. Are you sure? (y/n) y
[2024-05-09 02:07:02.765224] Starting SPDK v21.01.2-pre git sha1 752ceb0c1 / DPDK 20.11.0 initialization...
[2024-05-09 02:07:02.765623] [ DPDK EAL parameters: [2024-05-09 02:07:02.765651] blobcli [2024-05-09 02:07:02.765667] --no-shconf [2024-05-09 02:07:02.765682] -c 0x1 [2024-05-09 02:07:02.765698] --log-level=lib.eal:6 [2024-05-09 02:07:02.765753] --log-level=lib.cryptodev:5 [2024-05-09 02:07:02.765771] --log-level=user1:6 [2024-05-09 02:07:02.765787] --iova-mode=pa [2024-05-09 02:07:02.765803] --base-virtaddr=0x200000000000 [2024-05-09 02:07:02.765818] --match-allocations [2024-05-09 02:07:02.765834] --file-prefix=spdk_pid1051 [2024-05-09 02:07:02.765849] ]
EAL: No available hugepages reported in hugepages-1048576kB
EAL: No legacy callbacks, legacy socket not created
[2024-05-09 02:07:02.879508] app.c: 538:spdk_app_start: *NOTICE*: Total cores available: 1
[2024-05-09 02:07:02.955271] reactor.c: 915:reactor_run: *NOTICE*: Reactor started on core 0
[2024-05-09 02:07:02.955661] accel_engine.c: 692:spdk_accel_engine_initialize: *NOTICE*: Accel engine initialized to use software engine.
Init blobstore using bdev Name: Nvme0n1
Error in bs init callback (err -28)
[2024-05-09 02:07:03.004168] app.c: 629:spdk_app_stop: *WARNING*: spdk_app_stop'd on non-zero
ERROR!
root@target:~/blob# 
 ```