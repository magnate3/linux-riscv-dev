# krdma

RDMA easy to use in kernel.

`struct krdma_cb`: control block that supports both RDMA send/recv and read/write

## RDMA SEND/RECV APIs
```c
int krdma_send(struct krdma_cb *cb, const char *buffer, size_t length);

int krdma_receive(struct krdma_cb *cb, char *buffer);

/* Called with remote host & port */
int krdma_connect(const char *host, const char *port, struct krdma_cb **conn_cb);

/* Called with local host & port */
int krdma_listen(const char *host, const char *port, struct krdma_cb **listen_cb);

int krdma_accept(struct krdma_cb *listen_cb, struct krdma_cb **accept_cb);
```

KRDMA send receive example:

**node0**
```c
krdma_connect()
krdma_send()
krdma_receive()
krdma_send()
krdma_receive()
...
```

**node1**
```c
krdma_listen()
while (1) {
    krdma_accept()
    krdma_receive()
    krdma_send()
    krdma_receive()
    krdma_send()
}
```

## RDMA READ/WRITE APIs
```c
/* Called with remote host & port */
int krdma_rw_init_client(const char *host, const char *port, struct krdma_cb **cbp);

/* Called with local host & port */
int krdma_rw_init_server(const char *host, const char *port, struct krdma_cb **cbp);

int krdma_read(struct krdma_cb *cb, char *buffer, size_t length);

int krdma_write(struct krdma_cb *cb, const char *buffer, size_t length);

/* RDMA release API */
int krdma_release_cb(struct krdma_cb *cb);
```

KRDMA read write example:

**node0**
```c
krdma_rw_init_client()
krdma_write()
krdma_read()
krdma_write()
krdma_read()
...
```

**node1**
```c
krdma_rw_init_server()
while (1) {
    ...
}
```

# test


##   sr 
**server**
```
[root@centos7 krdma]# insmod  krdma_test.ko server=1 rw=0
[root@centos7 krdma]# 

[4320580.365522] Node 0 hugepages_total=4096 hugepages_free=4096 hugepages_surp=0 hugepages_size=2048kB
[4320580.374615] Node 0 hugepages_total=64 hugepages_free=63 hugepages_surp=0 hugepages_size=524288kB
[4320580.383536] Node 1 hugepages_total=0 hugepages_free=0 hugepages_surp=0 hugepages_size=2048kB
[4320580.392110] Node 1 hugepages_total=64 hugepages_free=64 hugepages_surp=0 hugepages_size=524288kB
[4320580.401029] Node 2 hugepages_total=0 hugepages_free=0 hugepages_surp=0 hugepages_size=2048kB
[4320580.409599] Node 2 hugepages_total=64 hugepages_free=64 hugepages_surp=0 hugepages_size=524288kB
[4320580.418517] Node 3 hugepages_total=0 hugepages_free=0 hugepages_surp=0 hugepages_size=2048kB
[4320580.427092] Node 3 hugepages_total=64 hugepages_free=64 hugepages_surp=0 hugepages_size=524288kB
[4320580.436012] 225070 total pagecache pages
[4320580.440095] 0 pages in swap cache
[4320580.443568] Swap cache stats: add 0, delete 0, find 0/0
[4320580.448941] Free swap  = 0kB
[4320580.451984] Total swap = 0kB
[4320580.455025] 8388541 pages RAM
[4320580.458151] 0 pages HighMem/MovableOnly
[4320580.462144] 22676 pages reserved
[4320580.465531] 0 pages hwpoisoned
[4320580.468744] __krdma_setup_mr_sr(): 199 ib_dma_alloc_coherent recv_buf failed
[4320580.476010] krdma_init_cb(): 771 krdma_setup_mr failed, ret -12
[4320580.482102] sr_server(): 1682 krdma_accept failed.
```
**client**
```
insmod krdma_test.ko server=0 rw=0
dmesg | tail -n 20
[  605.446175]  krdma_connect+0x8c/0x178 [krdma_test]
[  605.446178]  sr_client+0x74/0x120 [krdma_test]
[  605.446181]  kthread+0x134/0x138
[  605.446183]  ret_from_fork+0x10/0x18
[  726.277768] INFO: task sr_client:8967 blocked for more than 120 seconds.
[  726.277888]       Tainted: G           OE     5.0.0-23-generic #24~18.04.1-Ubuntu
[  726.278010] "echo 0 > /proc/sys/kernel/hung_task_timeout_secs" disables this message.
[  726.278139] sr_client       D    0  8967      2 0x00000028
[  726.278143] Call trace:
[  726.278152]  __switch_to+0xb4/0x1b8
[  726.278157]  __schedule+0x344/0x968
[  726.278160]  schedule+0x2c/0x78
[  726.278162]  schedule_timeout+0x224/0x448
[  726.278164]  wait_for_common+0xfc/0x1f8
[  726.278167]  wait_for_completion+0x28/0x38
[  726.278180]  krdma_connect_single+0x194/0x2f8 [krdma_test]
[  726.278183]  krdma_connect+0x8c/0x178 [krdma_test]
[  726.278186]  sr_client+0x74/0x120 [krdma_test]
[  726.278189]  kthread+0x134/0x138
```
