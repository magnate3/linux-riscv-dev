
# 不分配物理内存

```
#if 0
        // allow read and write
        ret = mprotect(buffer->ptr, self->page_size, PROT_READ | PROT_WRITE);
        ASSERT_EQ(ret, 0);
        memcpy(buffer->ptr,str1,strlen(str1)+1);
#endif
```

```
root@ubuntux86:# ./user 
not equal 
```
dmirror migrate return val: -22    
```
[ 8994.867923] memmap_init_zone_device initialised 65536 pages in 0ms
[ 8994.867935] added new 256 MB chunk (total 1 chunks, 256 MB) PFNs [0x3ff0000 0x4000000)
[ 8994.871147] memmap_init_zone_device initialised 65536 pages in 0ms
[ 8994.871152] added new 256 MB chunk (total 1 chunks, 256 MB) PFNs [0x3fe0000 0x3ff0000)
[ 8994.872437] HMM test module loaded. This is only for testing HMM.
[ 8998.790981] dmirror migrate return val: -22 
root@ubuntux86:# 
```

# 分配内存

```
#if 1
        // allow read and write
        ret = mprotect(buffer->ptr, self->page_size, PROT_READ | PROT_WRITE);
        ASSERT_EQ(ret, 0);
        memcpy(buffer->ptr,str1,strlen(str1)+1);
#endif
```
dmirror migrate return val: 0    
```
root@ubuntux86:# ./user 
run over 
root@ubuntux86:# dmesg | tail -n 10
[ 8994.867935] added new 256 MB chunk (total 1 chunks, 256 MB) PFNs [0x3ff0000 0x4000000)
[ 8994.871147] memmap_init_zone_device initialised 65536 pages in 0ms
[ 8994.871152] added new 256 MB chunk (total 1 chunks, 256 MB) PFNs [0x3fe0000 0x3ff0000)
[ 8994.872437] HMM test module loaded. This is only for testing HMM.
[ 8998.790981] dmirror migrate return val: -22 
[ 9111.958992] cmp page begin 
[ 9111.958998] g_addr 139723440275456 , page start 18446613406063329280, page end 18446613406063333376
[ 9111.959007] src and dts page are equal 
[ 9111.959009] buf is hello world 
[ 9111.959037] dmirror migrate return val: 0 
root@ubuntux86:# 
```


# HMM_DMIRROR_MIGRATE pk swap
swap缺页中断会调用migrate_to_ram    
```
else if (is_device_private_entry(entry)) {
            vmf->page = device_private_entry_to_page(entry);
            ret = vmf->page->pgmap->ops->migrate_to_ram(vmf);
        } 
```
![images](./swap.png)