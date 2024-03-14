## 说明
* exporter-fd.c - exporter 内核驱动
* importer-fd.c - importer 内核驱动
* share_fd.c - userspace应用程序

## test
```
[root@centos7 06]# insmod  exporter-test.ko 
[root@centos7 06]# insmod  importer-test.ko 
[root@centos7 06]# ./dmabuf-test/share_fd 
[root@centos7 06]# dmesg | tail -n 4
[  490.931699] exporter_test: module verification failed: signature and/or required key missing - tainting kernel
[  500.957483] ************ dmabuf release
[ 1015.636335] read from dmabuf kmap: hello world!
[ 1015.640853] read from dmabuf vmap: hello world!
[root@centos7 06]# 
```
