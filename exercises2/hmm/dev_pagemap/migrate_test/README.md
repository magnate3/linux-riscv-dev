
# test1

```
#if 1
        ret = posix_memalign((void **)&ptr, pagesize, pagesize);
        if(!ret)
        {
            memcpy(ptr, "krishna", strlen("krishna"));
            printf("phy addr of ptr  0x%lx \n",mem_virt2phy(ptr));
            //read(fd, ptr, pagesize);

            write(fd, ptr, pagesize);
            printf("after migrate, phy addr of ptr 0x%lx \n",mem_virt2phy(addr));
            //read(fd, ptr, pagesize);
#if 0
            memcpy(ptr, "krishna2", strlen("krishna"));
            printf("after migrate and memecpy again , phy addr of ptr 0x%lx \n", mem_virt2phy(addr));
            //read(fd, ptr, pagesize);
#endif
            free(ptr);
        }
        else
        {
            fprintf(stderr, "posix_memalign: %s\n", strerror (ret));
        }
#endif
```

```
root@ubuntux86:# ./mmap_test 
addr: 0x7fa88136e000 

Write/Read test ...
0x66616365
0x66616365
0x66616365
phy addr of ptr  0x13a36d000 
Zero page frame number
after migrate, phy addr of ptr 0x0 
root@ubuntux86:# 
```


# test2

```
#if 1
        ret = posix_memalign((void **)&ptr, pagesize, pagesize);
        if(!ret)
        {
            memcpy(ptr, "krishna", strlen("krishna"));
            printf("phy addr of ptr  0x%lx \n",mem_virt2phy(ptr));
            //read(fd, ptr, pagesize);

            write(fd, ptr, pagesize);
            printf("after migrate, phy addr of ptr 0x%lx \n",mem_virt2phy(addr));
            //read(fd, ptr, pagesize);
#if 1
            memcpy(ptr, "krishna2", strlen("krishna"));
            printf("after migrate and memecpy again , phy addr of ptr 0x%lx \n", mem_virt2phy(addr));
            //read(fd, ptr, pagesize);
#endif
            free(ptr);
        }
        else
        {
            fprintf(stderr, "posix_memalign: %s\n", strerror (ret));
        }
#endif
```
第二次memcpy(ptr, "krishna2", strlen("krishna"));导致程序coredump   

```
root@ubuntux86:# ./mmap_test 
addr: 0x7faa2b798000 

Write/Read test ...
0x66616365
0x66616365
0x66616365
phy addr of ptr  0x14742a000 
Zero page frame number
after migrate, phy addr of ptr 0x0 
Segmentation fault
root@ubuntux86:# dmesg | tail -n 60
```

# do_swap_page
```
[   43.984917] PKRU: 55555554
[   43.984920] Call Trace:
[   43.984924]  <TASK>
[   43.984929]  migration_entry_wait+0xa1/0xb0
[   43.984936]  do_swap_page+0x657/0x730
[   43.984948]  __handle_mm_fault+0x882/0x8e0
[   43.984956]  handle_mm_fault+0xda/0x2b0
[   43.984962]  do_user_addr_fault+0x1bb/0x650
[   43.984969]  exc_page_fault+0x7d/0x170
[   43.984977]  ? asm_exc_page_fault+0x8/0x30
[   43.984988]  asm_exc_page_fault+0x1e/0x30
[   43.984996] RIP: 0033:0x7f54783829ae
```