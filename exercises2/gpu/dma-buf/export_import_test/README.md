
```
insmod  exporter-page.ko 
insmod  importer-page.ko 
```


```
[root@centos7 dmabuf-test]# ./dmabuf_sync 
ion alloc success: size = 65536, dmabuf_fd = 4
dma buf fd 4 
read from dmabuf mmap: hello world!
```

```
[root@centos7 dmabuf-test]# ./mmap_dmabuf 
ion alloc success: size = 65536, dmabuf_fd = 4
dma buf fd 4 
```

```
read from dmabuf mmap: hello world!
[root@centos7 dmabuf-test]# ./share_fd 
[root@centos7 dmabuf-test]# 
```