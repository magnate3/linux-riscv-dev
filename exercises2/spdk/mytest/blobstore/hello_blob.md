

# make
改成"Nvme0n1"

```
       rc = spdk_bdev_create_bs_dev_ext("Nvme0n1", base_bdev_event_cb, NULL, &bs_dev);
        if (rc != 0) {
                SPDK_ERRLOG("Could not create blob bdev, %s!!\n",
                            spdk_strerror(-rc));
                spdk_app_stop(-1);
                return;
        }
```

# run


```
root@target:~/spdk/examples/blob/hello_blob# ~/spdk/scripts/gen_nvme.sh --json-with-subsystems >  blobcli.json  
root@target:~/spdk/examples/blob/hello_blob# ls
Makefile  blobcli.json  hello_blob.c  hello_blob.d  hello_blob.json  hello_blob.o
```

```
root@target:~/spdk/examples/blob/hello_blob# ~/spdk/build/examples/my_hello_blob  ./blobcli.json 
[2024-05-09 02:24:35.245526] hello_blob.c: 451:main: *NOTICE*: entry
[2024-05-09 02:24:35.245657] Starting SPDK v21.01.2-pre git sha1 752ceb0c1 / DPDK 20.11.0 initialization...
[2024-05-09 02:24:35.245677] [ DPDK EAL parameters: [2024-05-09 02:24:35.245684] hello_blob [2024-05-09 02:24:35.245690] --no-shconf [2024-05-09 02:24:35.245695] -c 0x1 [2024-05-09 02:24:35.245700] --log-level=lib.eal:6 [2024-05-09 02:24:35.245716] --log-level=lib.cryptodev:5 [2024-05-09 02:24:35.245723] --log-level=user1:6 [2024-05-09 02:24:35.245729] --iova-mode=pa [2024-05-09 02:24:35.245734] --base-virtaddr=0x200000000000 [2024-05-09 02:24:35.245739] --match-allocations [2024-05-09 02:24:35.245744] --file-prefix=spdk_pid1203 [2024-05-09 02:24:35.245749] ]
EAL: No available hugepages reported in hugepages-1048576kB
EAL: No legacy callbacks, legacy socket not created
[2024-05-09 02:24:35.358808] app.c: 538:spdk_app_start: *NOTICE*: Total cores available: 1
[2024-05-09 02:24:35.455854] reactor.c: 915:reactor_run: *NOTICE*: Reactor started on core 0
[2024-05-09 02:24:35.456295] accel_engine.c: 692:spdk_accel_engine_initialize: *NOTICE*: Accel engine initialized to use software engine.
[2024-05-09 02:24:35.505559] hello_blob.c: 414:hello_start: *NOTICE*: entry
[2024-05-09 02:24:35.505604] hello_blob.c: 373:bs_init_complete: *NOTICE*: entry
[2024-05-09 02:24:35.505621] hello_blob.c:  94:unload_bs: *ERROR*: Error init'ing the blobstore (err -28)
[2024-05-09 02:24:35.505634] app.c: 629:spdk_app_stop: *WARNING*: spdk_app_stop'd on non-zero
[2024-05-09 02:24:35.527509] hello_blob.c: 485:main: *NOTICE*: ERROR!
root@target:~/spdk/examples/blob/hello_blob# 
```

This is a bug in hello_blob (not blobstore itself). Blobstore used to only support read/write in 4KB units. It now supports 512B units if the backing device has 512B LBAs. hello_blob wasn't updated when those changes were made.