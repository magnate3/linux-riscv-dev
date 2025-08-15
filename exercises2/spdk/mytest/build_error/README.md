

# "spdk/bdev_module.h"  pk "spdk/bdev_zone.h"

## 采用  "spdk/bdev_module.h"

```C
root@ubuntu:~/spdk/examples/hello_nvme_bdev# make
  CC hello_nvme_bdev/hello_nvme_bdev.o
  LINK my_hello_nvme_bdev
root@ubuntu:~/spdk/examples/hello_nvme_bdev# 
```

## 采用 "spdk/bdev_zone.h"

```C
root@ubuntu:~/spdk/examples/hello_nvme_bdev# make
  CC hello_nvme_bdev/hello_nvme_bdev.o
In file included from /root/spdk/include/spdk/event.h:48,
                 from hello_nvme_bdev.c:31:
hello_nvme_bdev.c: In function ‘hello_start’:
hello_nvme_bdev.c:177:45: error: dereferencing pointer to incomplete type ‘struct spdk_bdev’
  177 |         SPDK_ERRLOG("bdev name: %s\n", first->name);
      |                                             ^~
/root/spdk/include/spdk/log.h:132:57: note: in definition of macro ‘SPDK_ERRLOG’
  132 |  spdk_log(SPDK_LOG_ERROR, __FILE__, __LINE__, __func__, __VA_ARGS__)
      |                                                         ^~~~~~~~~~~
make: *** [/root/spdk/mk/spdk.common.mk:399: hello_nvme_bdev.o] Error 1
root@ubuntu:~/spdk/examples/hello_nvme_bdev# 
```