# mount huge
```
mount -t hugetlbfs nodev /mnt/huge
```

```
#define TEST_HUGEPAGE_PATH      "/mnt/huge/test"
```

# make
```
gcc migrate_test.c  -o migrate_test -lnuma
```

# 大页内存

```

#if 1
        page_base = (char *)mmap(NULL,
                                TEST_MAP_SIZE,
                                PROT_READ | PROT_WRITE,
                                MAP_SHARED,
                                fd,
                                0);
#else
        page_base = (char *)mmap(NULL,
                                TEST_MAP_SIZE,
                                PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANONYMOUS,
                                -1,
                                0);
#endif
```

```
[root@centos7 numa]# gcc migrate_test.c  -o migrate_test -lnuma
[root@centos7 numa]# ./migrate_test 
******** Before Migration
before migrate, Physical address is 208842784768
  Page vaddr: 0x400020000000 node: 0
Migrating the current processes pages ...
  Page vaddr: 0x400020000000 node: 0
after migrate, Physical address is 208842784768
[root@centos7 numa]# 
```
实际上没有迁移到了 node1,物理地址没有变化   
# 非大页内存


```
[root@centos7 numa]# ./migrate_test 
******** Before Migration
before migrate, Physical address is 205132857344
  Page vaddr: 0xffff60000000 node: 0
Migrating the current processes pages ...
  Page vaddr: 0xffff60000000 node: 1
after migrate, Physical address is 407981916160
[root@centos7 numa]# 
```
迁移到了 node1，物理地址变化   