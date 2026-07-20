


# make 
```
root@ubuntux86:# make
cat nv.symvers >> Module.symvers
make -C /lib/modules/5.13.0-39-generic/build M=/work/kernel_learn/gpu_usermap NOSTDINC_FLAGS="-I./" modules
make[1]: Entering directory '/usr/src/linux-headers-5.13.0-39-generic'
  CC [M]  /work/kernel_learn/gpu_usermap/gpu_usermap.o
  MODPOST /work/kernel_learn/gpu_usermap/Module.symvers
ERROR: modpost: "nvidia_p2p_put_pages" [/work/kernel_learn/gpu_usermap/gpu_usermap.ko] undefined!
ERROR: modpost: "nvidia_p2p_get_pages" [/work/kernel_learn/gpu_usermap/gpu_usermap.ko] undefined!
ERROR: modpost: "nvidia_p2p_free_page_table" [/work/kernel_learn/gpu_usermap/gpu_usermap.ko] undefined!
make[2]: *** [scripts/Makefile.modpost:150: /work/kernel_learn/gpu_usermap/Module.symvers] Error 1
make[2]: *** Deleting file '/work/kernel_learn/gpu_usermap/Module.symvers'
make[1]: *** [Makefile:1794: modules] Error 2
make[1]: Leaving directory '/usr/src/linux-headers-5.13.0-39-generic'
make: *** [Makefile:14: all] Error 2
root@ubuntux86:# 
```

#  nv-p2p-dummy.c

[nv-p2p-dummy.c](https://github.com/drossetti/gdrcopy/blob/0d78e6ecc64b97baf79c12ae0c62975ef3f92a9a/src/gdrdrv/nv-p2p-dummy.c)

```
int nvidia_p2p_get_pages(uint64_t p2p_token, uint32_t va_space,
                         uint64_t virtual_address,
                         uint64_t length,
                         struct nvidia_p2p_page_table **page_table,
                         void (*free_callback)(void *data),
                         void *data)
{
    return -EINVAL;
}
EXPORT_SYMBOL(nvidia_p2p_get_pages);

int nvidia_p2p_put_pages(uint64_t p2p_token, uint32_t va_space,
                         uint64_t virtual_address,
                         struct nvidia_p2p_page_table *page_table)
{
    return -EINVAL;
}
EXPORT_SYMBOL(nvidia_p2p_put_pages);

int nvidia_p2p_free_page_table(struct nvidia_p2p_page_table *page_table)
{
    return -EINVAL;
}
EXPORT_SYMBOL(nvidia_p2p_free_page_table);
```