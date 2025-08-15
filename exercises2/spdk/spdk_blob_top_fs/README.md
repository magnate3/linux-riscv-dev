
# 编译simple_fs

设置软连接   
```
root@target:~/spdk_blob_top_fs/simple_fs# ln -sf ../spdk_fs  spdkfs
```


# lib 库
libspdk_spdkXX在spdk/build/lib/目录中    

```
root@target:~/spdk# pwd
/root/spdk
```


```
root@target:~/spdk# find ./ -name "libspdk_spdk_fs_top.a"  
./build/lib/libspdk_spdk_fs_top.a
root@target:~/spdk# find ./ -name libspdk_spdk_simple_fs.a  
./build/lib/libspdk_spdk_simple_fs.a
root@target:~/spdk# 
```

# 编译 posix_test

```
root@target:~/spdk_blob_top_fs/test/posix_test# ln -sf ../../spdk_fs  spdkfs
```