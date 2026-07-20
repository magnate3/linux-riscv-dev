
# 编译

 ln -sf ../spdk spdk 指向开发使用的的spdk  
```
 rm -rf spdk
 ln -sf ../spdk spdk
```
创建目录    
```
 mkdir build/bin -p
 mkdir build/test -p
```


spdk_nvme_detach_poll改为spdk_nvme_detach_poll_async
```
 if (detach_ctx) {
        spdk_nvme_detach_poll_async(detach_ctx);
    }
```


# spdk 测试


```
root@target:~/spfs# cd ..
root@target:~# cd spdk
root@target:~/spdk# ./scripts/setup.sh
0000:00:03.0 (1b36 0010): nvme -> uio_pci_generic
0000:00:04.0 (1b36 0010): nvme -> uio_pci_generic
root@target:~/spdk# ./build/examples/hello_world 
[2024-05-08 12:41:54.238911] Starting SPDK v21.01.2-pre git sha1 752ceb0c1 / DPDK 20.11.0 initialization...
[2024-05-08 12:41:54.239150] [ DPDK EAL parameters: [2024-05-08 12:41:54.239188] hello_world [2024-05-08 12:41:54.239217] -c 0x1 [2024-05-08 12:41:54.239233] --log-level=lib.eal:6 [2024-05-08 12:41:54.239251] --log-level=lib.cryptodev:5 [2024-05-08 12:41:54.239268] --log-level=user1:6 [2024-05-08 12:41:54.239294] --iova-mode=pa [2024-05-08 12:41:54.239318] --base-virtaddr=0x200000000000 [2024-05-08 12:41:54.239342] --match-allocations [2024-05-08 12:41:54.239386] --file-prefix=spdk0 [2024-05-08 12:41:54.239411] --proc-type=auto [2024-05-08 12:41:54.239438] ]
EAL: No available hugepages reported in hugepages-1048576kB
EAL: No legacy callbacks, legacy socket not created
Initializing NVMe Controllers
Attaching to 0000:00:03.0
Attaching to 0000:00:04.0
Attached to 0000:00:03.0
Using controller QEMU NVMe Ctrl       (nvme-dev            ) with 256 namespaces.
  Namespace ID: 1 size: 0GB
Attached to 0000:00:04.0
Using controller QEMU NVMe Ctrl       (nvme-de2v           ) with 256 namespaces.
  Namespace ID: 1 size: 0GB
Initialization complete.
INFO: using host memory buffer for IO
Hello world!
INFO: using host memory buffer for IO
Hello world!
root@target:~/spdk# 
```

# spfs（只能采用一个nvme测试spfs）

1)  一个nvme    
```
root@target:~# cd spdk
root@target:~/spdk# ./scripts/setup.sh 
0000:00:03.0 (1b36 0010): nvme -> uio_pci_generic
```
2) nvme大小

```
qemu-img create -f qcow2 nvme.qcow2 128M
```
不是128M

```
root@target:~/spdk# cd ../spfs/build/bin/
root@target:~/spfs/build/bin# ./touch /a/b.txt
[2024-05-08 12:46:11.652491] Starting SPDK v21.01.2-pre git sha1 752ceb0c1 / DPDK 20.11.0 initialization...
[2024-05-08 12:46:11.652657] [ DPDK EAL parameters: [2024-05-08 12:46:11.652680] spfs [2024-05-08 12:46:11.652727] --no-shconf [2024-05-08 12:46:11.652754] -c 0x1 [2024-05-08 12:46:11.652777] --log-level=lib.eal:6 [2024-05-08 12:46:11.652799] --log-level=lib.cryptodev:5 [2024-05-08 12:46:11.652821] --log-level=user1:6 [2024-05-08 12:46:11.652846] --iova-mode=pa [2024-05-08 12:46:11.652868] --base-virtaddr=0x200000000000 [2024-05-08 12:46:11.652891] --match-allocations [2024-05-08 12:46:11.652913] --file-prefix=spdk_pid1019 [2024-05-08 12:46:11.652935] ]
EAL: No available hugepages reported in hugepages-1048576kB
EAL: No legacy callbacks, legacy socket not created
ERROR: invalid path.
```
不是128M

```
root@target:~/spfs/build/bin#  ./ls /
[2024-05-08 12:47:00.649828] Starting SPDK v21.01.2-pre git sha1 752ceb0c1 / DPDK 20.11.0 initialization...
[2024-05-08 12:47:00.650032] [ DPDK EAL parameters: [2024-05-08 12:47:00.650103] spfs [2024-05-08 12:47:00.650149] --no-shconf [2024-05-08 12:47:00.650186] -c 0x1 [2024-05-08 12:47:00.650214] --log-level=lib.eal:6 [2024-05-08 12:47:00.650238] --log-level=lib.cryptodev:5 [2024-05-08 12:47:00.650277] --log-level=user1:6 [2024-05-08 12:47:00.650303] --iova-mode=pa [2024-05-08 12:47:00.650318] --base-virtaddr=0x200000000000 [2024-05-08 12:47:00.650332] --match-allocations [2024-05-08 12:47:00.650353] --file-prefix=spdk_pid1022 [2024-05-08 12:47:00.650369] ]
EAL: No available hugepages reported in hugepages-1048576kB
EAL: No legacy callbacks, legacy socket not created
---------------- ls 1 start ----------------
<FILE>          Blkno: 1799
<FILE>          Blkno: 1799
<FILE>          Blkno: 1799
<FILE>          Blkno: 1799
<FILE>          Blkno: 1799
<FILE>          Blkno: 1799
<FILE>          Blkno: 1799
<FILE>          Blkno: 1799
<FILE>          Blkno: 1799
<FILE>          Blkno: 1799
<FILE>          Blkno: 1799
<FILE>          Blkno: 1799
<FILE>          Blkno: 1799
<FILE>          Blkno: 1799
<FILE>          Blkno: 1799
<FILE>          Blkno: 1799
<FILE>          Blkno: 1799
<FILE>          Blkno: 1799
<FILE>          Blkno: 1799
<FILE>          Blkno: 1799
<FILE>          Blkno: 1799
<FILE>          Blkno: 1799
<FILE>          Blkno: 1799
<FILE>          Blkno: 1799
<FILE>          Blkno: 1799
```

> ## 改成128M

```
root@target:~# ./spdk/scripts/setup.sh 
0000:00:03.0 (1b36 0010): nvme -> uio_pci_generic
root@target:~# cd spfs/build/bin/
root@target:~/spfs/build/bin# ls
cat  ls  mkdir  mkfs  rm  touch
root@target:~/spfs/build/bin# ./ls /
[2024-05-08 12:51:50.849240] Starting SPDK v21.01.2-pre git sha1 752ceb0c1 / DPDK 20.11.0 initialization...
[2024-05-08 12:51:50.850643] [ DPDK EAL parameters: [2024-05-08 12:51:50.851217] spfs [2024-05-08 12:51:50.853180] --no-shconf [2024-05-08 12:51:50.8533]
EAL: No available hugepages reported in hugepages-1048576kB
EAL: No legacy callbacks, legacy socket not created
---------------- ls 1 start ----------------
---------------- ls 1 end ----------------
root@target:~/spfs/build/bin# 
```
+ ./mkfs /    
```
root@target:~/spfs/build/bin# ls
cat  ls  mkdir  mkfs  rm  touch
root@target:~/spfs/build/bin# ./mkfs /
[2024-05-08 12:54:35.773811] Starting SPDK v21.01.2-pre git sha1 752ceb0c1 / DPDK 20.11.0 initialization...
[2024-05-08 12:54:35.775273] [ DPDK EAL parameters: [2024-05-08 12:54:35.775854] spfs [2024-05-08 12:54:35.776143] --no-shconf [2024-05-08 12:54:35.7764]
EAL: No available hugepages reported in hugepages-1048576kB
EAL: No legacy callbacks, legacy socket not created
File system created successfully.
```

+ ./mkdir /a    
```
root@target:~/spfs/build/bin# ./mkdir /a
[2024-05-08 12:54:50.629734] Starting SPDK v21.01.2-pre git sha1 752ceb0c1 / DPDK 20.11.0 initialization...
[2024-05-08 12:54:50.629998] [ DPDK EAL parameters: [2024-05-08 12:54:50.630377] spfs [2024-05-08 12:54:50.631103] --no-shconf [2024-05-08 12:54:50.6314]
EAL: No available hugepages reported in hugepages-1048576kB
EAL: No legacy callbacks, legacy socket not created
Directory created successfully.
```
+ ./touch  /a/b.txt
```
root@target:~/spfs/build/bin# ./touch  /a/b.txt
[2024-05-08 12:54:55.999504] Starting SPDK v21.01.2-pre git sha1 752ceb0c1 / DPDK 20.11.0 initialization...
[2024-05-08 12:54:55.999611] [ DPDK EAL parameters: [2024-05-08 12:54:55.999774] spfs [2024-05-08 12:54:55.999918] --no-shconf [2024-05-08 12:54:56.0000]
EAL: No available hugepages reported in hugepages-1048576kB
EAL: No legacy callbacks, legacy socket not created
File created successfully.
```
+ ../test/test_file   
```
root@target:~/spfs/build/bin# ../test/test_file 
[2024-05-08 12:55:00.872531] Starting SPDK v21.01.2-pre git sha1 752ceb0c1 / DPDK 20.11.0 initialization...
[2024-05-08 12:55:00.872636] [ DPDK EAL parameters: [2024-05-08 12:55:00.872800] spfs [2024-05-08 12:55:00.872945] --no-shconf [2024-05-08 12:55:00.8730]
EAL: No available hugepages reported in hugepages-1048576kB
EAL: No legacy callbacks, legacy socket not created
0123
89ab
Test passed.
root@target:~/spfs/build/bin# 
```

# reference

[Leohh123/spfs](https://github.com/Leohh123/spfs/tree/main)         

[深入理解SPDK读写数据的过程，从应用到NVMe驱动](https://cloud.tencent.com/developer/news/1136961)    

