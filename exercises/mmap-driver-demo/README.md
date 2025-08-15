# mmap-driver-demo

## insmod mydemodev.ko
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/mmap-driver-demo/addr.png)


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/mmap-driver-demo/mmap.png)

```
root@ubuntux86:/work/kernel_learn/mmap-driver-demo# insmod mydemodev.ko
root@ubuntux86:/work/kernel_learn/mmap-driver-demo# ls /dev | grep demo
demo_dev
root@ubuntux86:/work/kernel_learn/mmap-driver-demo# 
```


```
static int demodev_mmap(struct file *file, struct vm_area_struct *vma)
{
    struct mm_struct *mm;
    unsigned long size;
    unsigned long pfn_start;
    void *virt_start;
    int ret;

    mm = current->mm;
    pfn_start = page_to_pfn(page) + vma->vm_pgoff;
    virt_start = page_address(page) + (vma->vm_pgoff << PAGE_SHIFT);

    /* 映射大小不超过实际物理页 */
    size = min(((1 << PAGE_ORDER) - vma->vm_pgoff) << PAGE_SHIFT,
               vma->vm_end - vma->vm_start);

    printk("phys_start: 0x%lx, offset: 0x%lx, vma_size: 0x%lx, map size:0x%lx\n",
           pfn_start << PAGE_SHIFT, vma->vm_pgoff << PAGE_SHIFT,
           vma->vm_end - vma->vm_start, size);

    if (size <= 0) {
        printk("%s: offset 0x%lx too large, max size is 0x%lx\n", __func__,
               vma->vm_pgoff << PAGE_SHIFT, MAX_SIZE);
        return -EINVAL;
    }

    // 外层vm_mmap_pgoff已经用信号量保护了 
    // down_read(&mm->mmap_sem);
    ret = remap_pfn_range(vma, vma->vm_start, pfn_start, size, vma->vm_page_prot);
    // up_read(&mm->mmap_sem);

    if (ret) {
        printk("remap_pfn_range failed, vm_start: 0x%lx\n", vma->vm_start);
    }
    else {
        printk("map kernel 0x%px to user 0x%lx, size: 0x%lx\n",
               virt_start, vma->vm_start, size);
    }

    return ret;
}

[ 4284.452760] demodrv_open: major=10, minor=122
[ 4284.452769] client: test1 (2503)
[ 4284.452772] code  section: [0x5606f286c000   0x5606f286c615]
[ 4284.452776] data  section: [0x5606f286ed48   0x5606f286f010]
[ 4284.452778] brk   section: s: 0x5606f3efa000, c: 0x5606f3efa000
[ 4284.452781] mmap  section: s: 0x7f93e52e1000
[ 4284.452783] stack section: s: 0x7ffd093ad8d0
[ 4284.452785] arg   section: [0x7ffd093af78b   0x7ffd093af793]
[ 4284.452787] env   section: [0x7ffd093af793   0x7ffd093afff0]
[ 4289.453213] phys_start: 0x1517ec000, offset: 0x0, vma_size: 0x1000, map size:0x1000
[ 4289.453237] map kernel 0xffff8b5c517ec000 to user 0x7f93e52dd000, size: 0x1000
[ 4294.453722] demodrv_read actual_readed=64, pos=64
[ 4294.453742] demodrv_write actual_written=13, pos=77
```


# arm64

```
[root@centos7 mmap-driver-demo]# insmod vm_file_test.ko test_name=test1
```

```
[root@centos7 mmap-driver-demo]# gcc test1.c -o test1
[root@centos7 mmap-driver-demo]# ./test1 
addr is : 0xffffa13b0000 

[root@centos7 mmap-driver-demo]# dmesg | tail -n 20
[315968.536531] code  section: [0x400000   0x400f5c]
[315968.541213] data  section: [0x41fdf0   0x42008c]
[315968.545902] brk   section: s: 0x6e50000, c: 0x6e50000
[315968.551017] mmap  section: s: 0xffffa1420000
[315968.555360] stack section: s: 0xffffc2706820
[315968.559696] arg   section: [0xffffc270f7e5   0xffffc270f7ed]
[315968.565422] env   section: [0xffffc270f7ed   0xffffc270fff0]
[315973.571178] phys_start: 0x203ee0340000, offset: 0x0, vma_size: 0x10000, map size:0x10000
[315973.579323] map kernel 0xffffa03ee0340000 to user 0xffffa13b0000, size: 0x10000
[315978.586753] demodrv_read actual_readed=64, pos=64
[315978.591525] demodrv_write actual_written=13, pos=77
[315979.279910] vm_file->f_path.dentry->d_iname:  test1 
[315979.284947] vm_file->f_path.dentry->d_iname:  test1 
[315979.289975] vm_file->f_path.dentry->d_iname:  libc-2.17.so 
[315979.295616] vm_file->f_path.dentry->d_iname:  libc-2.17.so 
[315979.301248] vm_file->f_path.dentry->d_iname:  libc-2.17.so 
[315979.306884] vm_file->f_path.dentry->d_iname:  demo_dev 
[315979.312171] vm_file->f_path.dentry->d_iname:  ld-2.17.so 
[315979.317637] vm_file->f_path.dentry->d_iname:  ld-2.17.so 
[315979.323101] vm_file->f_path.dentry->d_iname:  ld-2.17.so 
```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/mmap-driver-demo/file.png)

[see blog](https://catbro666.github.io/posts/5ec4fb12/)
