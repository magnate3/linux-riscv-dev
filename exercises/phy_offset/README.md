在mmap的手册中有一段示例代码，其中有一行：

offset = atoi(argv[2]);
pa_offset = offset & ~(sysconf(_SC_PAGE_SIZE) - 1);
/* offset for mmap() must be page aligned */
 
 
```
getconf -a | grep PAGE
PAGESIZE                           65536
PAGE_SIZE                          65536
_AVPHYS_PAGES                      8174145
_PHYS_PAGES                        8365865
[root@centos7 mem_map]# 
```