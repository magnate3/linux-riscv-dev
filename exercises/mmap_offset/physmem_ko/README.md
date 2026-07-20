## /dev/physmem

An unrestricted device to map physical memory to userland. I created this to load on machines that were compiled with `STRICT_DEVMEM`, which restricts mapping physical memory beyond 1mb. Also exposes an IOCTL for allocating and freeing physical memory

## Build and Load

```
make
sudo insmod physmem.ko
```

## Tests

```
# Shows example of physical memory mmap differences between /dev/mem and /dev/physmem
sudo ./mmap_test

# Shows example of calling IOCTL to allocate and free physical memory
sudo ./ioctl_test
```

## Unload

```
sudo rmmod physmem
```
