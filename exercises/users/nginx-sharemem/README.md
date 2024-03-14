

# comapre src and src-tiny


in src , config.h  uses  ngx_atomic.h  and  ngx_config.h 、 ngx_posix_config.h

in src-tiny , config.h   not  uses  ngx_atomic.h  and  ngx_config.h 、 ngx_posix_config.h

# make sharemem

```
 cmake .
 make
```

```
[root@centos7 nginx-sharemem]# 
[root@centos7 nginx-sharemem]# ls bin/
sharemem_test  tree
[root@centos7 nginx-sharemem]#
```



# ngx_cpu_pause

ngx_cpu_pause will be called by  ngx_spinlock and defined in  ngx_atomic.h
