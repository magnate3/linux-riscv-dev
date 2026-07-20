# arm64

## pagesize

```
[root@centos7 physmem_ko]# dmesg | grep 'offset :'
[1374410.844015] offset : 0x0 
[1374410.850409] offset : 0x20600000 //0x206000000000
[root@centos7 physmem_ko]# cat /proc/iomem  | grep -i ram
00010000-29ecffff : System RAM
2ac80000-2c09ffff : System RAM
2c0c0000-2e3bffff : System RAM
2e3d0000-2f32ffff : System RAM
2f6a0000-2f6dffff : System RAM
2f780000-2f79ffff : System RAM
2f8e0000-2f96ffff : System RAM
2fc00000-2fc1ffff : System RAM
2fc80000-3f02ffff : System RAM
3f060000-3fbfffff : System RAM
50000000-7fffffff : System RAM
2080000000-3fffffffff : System RAM
4000000000-5fffffffff : System RAM
202000000000-203fffffffff : System RAM
204000000000-205fffffffff : System RAM  //0x206000000000
[root@centos7 physmem_ko]# getconf -a | grep -i page
PAGESIZE                           65536  // 16bit
PAGE_SIZE                          65536
_AVPHYS_PAGES                      8133864
_PHYS_PAGES                        8365865
```
## ./mmap_test

```
}

int device_mmap(struct file *filp, struct vm_area_struct *vma)
{
    unsigned long offset = vma->vm_pgoff;
    pr_info("offset : 0x%lx \n", offset); //*********************************
    if (offset >= __pa(high_memory) || (filp->f_flags & O_SYNC))
        vma->vm_flags |= VM_IO;
    vma->vm_flags |= (VM_DONTEXPAND | VM_DONTDUMP);

    if (io_remap_pfn_range(vma, vma->vm_start, offset,
        vma->vm_end-vma->vm_start, vma->vm_page_prot))
        return -EAGAIN;
    return 0;
}

```

```
[root@centos7 physmem_ko]# rmmod  physmem.ko 
[root@centos7 physmem_ko]# insmod physmem.ko 
[root@centos7 physmem_ko]# ./mmap_test 
Trying at offset 0x00

/dev/physmem offset: 0
00 00 00 00 00 00 00 00   00 00 00 00 00 00 00 00   
00 00 00 00 00 00 00 00   00 00 00 00 00 00 00 00   
00 00 00 00 00 00 00 00   00 00 00 00 00 00 00 00   
00 00 00 00 00 00 00 00   00 00 00 00 00 00 00 00   

Trying beyond 1mb...

/dev/physmem offset: 206000000000
ff ff ff ff ff ff ff ff   ff ff ff ff ff ff ff ff   
ff ff ff ff ff ff ff ff   ff ff ff ff ff ff ff ff   
ff ff ff ff ff ff ff ff   ff ff ff ff ff ff ff ff   
ff ff ff ff ff ff ff ff   ff ff ff ff ff ff ff ff   
[root@centos7 physmem_ko]# 
```
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/mmap_offset/arm64.png)


# x86

```
root@ubuntux86:/work/test/physmem_ko# getconf -a | grep -i page
PAGESIZE                           4096
PAGE_SIZE                          4096
_AVPHYS_PAGES                      7818918
_PHYS_PAGES                        8150133
root@ubuntux86:/work/test/physmem_ko# dmesg | tail -n 10
[ 3986.341268] offset : 0x0 
[ 3986.341396] offset : 0x890000 
[ 5796.694281] pcieport 0000:00:1d.0: AER: Corrected error received: 0000:02:00.0
[ 5796.694378] nvme 0000:02:00.0: PCIe Bus Error: severity=Corrected, type=Physical Layer, (Receiver ID)
[ 5796.694384] nvme 0000:02:00.0:   device [15b7:5006] error status/mask=00000001/0000e000
[ 5796.694390] nvme 0000:02:00.0:    [ 0] RxErr                 
[ 6772.578652] pcieport 0000:00:1d.0: AER: Corrected error received: 0000:02:00.0
[ 6772.578749] nvme 0000:02:00.0: PCIe Bus Error: severity=Corrected, type=Physical Layer, (Receiver ID)
[ 6772.578754] nvme 0000:02:00.0:   device [15b7:5006] error status/mask=00000001/0000e000
[ 6772.578761] nvme 0000:02:00.0:    [ 0] RxErr                 
root@ubuntux86:/work/test/physmem_ko# cat /proc/iomem  | grep -i ram
00001000-0009efff : System RAM
00100000-5caf8017 : System RAM
5caf8018-5cb17c57 : System RAM
5cb17c58-5cb38fff : System RAM
5cc7e000-5cd49fff : System RAM
5cd4b000-61c71fff : System RAM
666ff000-666fffff : System RAM
100000000-88d7fffff : System RAM
88d800000-88fffffff : RAM buffer
```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/mmap_offset/x86.png)