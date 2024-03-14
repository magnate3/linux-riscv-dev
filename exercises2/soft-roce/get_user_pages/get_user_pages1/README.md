
#  pagesize 对齐

```
posix_memalign((void **)&ptr, pagesize, 4096);
```


```
[root@centos7 get_user_pages]# ./test 
data is Mohan
[root@centos7 get_user_pages]# dmesg | tail -n 10
[22037.309348] sample_open
[22037.311828] sample_write
[22037.314351] Got mmaped.
[22037.316787] krishna
[22037.318894] sample_release
[22202.782160] sample_open
[22202.784635] sample_write
[22202.787158] Got mmaped.
[22202.789603] krishna
[22202.791711] sample_release
```

#  非pagesize 对齐

```
posix_memalign((void **)&ptr, 4096, 4096);
```

```
[root@centos7 get_user_pages]# ./test 
data is krishna
[root@centos7 get_user_pages]# dmesg | tail -n 10
[22202.782160] sample_open
[22202.784635] sample_write
[22202.787158] Got mmaped.
[22202.789603] krishna
[22202.791711] sample_release
[22258.004120] sample_open
[22258.006599] sample_write
[22258.009122] Got mmaped.

[22258.013061] sample_release
```

# 参考

[lab2_usermap.c and lab2_write_aligned.c](https://github.com/bgmerrell/driver-samples/blob/f017f3bf9232d81b49fb447461c34e0d88b66185/s_16/lab2_usermap.c)