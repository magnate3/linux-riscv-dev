

```
[root@centos7 drm_test]# make
make -C /lib/modules/4.14.0-115.el7a.0.1.aarch64/build \
M=/root/programming/kernel/dma-buf/drm_test modules
make[1]: Entering directory '/usr/src/kernels/4.14.0-115.el7a.0.1.aarch64'
  Building modules, stage 2.
  MODPOST 1 modules
make[1]: Leaving directory '/usr/src/kernels/4.14.0-115.el7a.0.1.aarch64'
[root@centos7 drm_test]# 
```



```
 yum install libdrm libdrm-devel
 apt-get install libdrm-dev
```


```
[root@centos7 drm_test]# ls /usr/include/libdrm/drm.h
/usr/include/libdrm/drm.h
[root@centos7 drm_test]#
```


```
[root@centos7 drm_test]# gcc test.c  -o test -I /usr/include/libdrm -ldrm
[root@centos7 drm_test]# 
```


```
[root@centos7 drm_test]# ./test 
create dumb: handle = 1, pitch = 960, size = 307200
get mmap offset 0x1000f0000
read from mmap: This is a dumb buffer!
```

# references

[drm 驱动系列 - 第三章 gem 内存管理](https://blog.csdn.net/sty01z/article/details/134694799)      
[my-gem/dumb](https://github.com/hexiaolong2008/sample-code/blob/master/drm/driver/my-gem/dumb.c)   