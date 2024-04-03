

```
root@ubuntux86:# insmod simplefs.ko
root@ubuntux86:# mkdir -p test
root@ubuntux86:# dd if=/dev/zero of=test.img bs=1M count=50
50+0 records in
50+0 records out
52428800 bytes (52 MB, 50 MiB) copied, 0.0278448 s, 1.9 GB/s
root@ubuntux86:# ./mkfs.simplefs test.img
Superblock: (4096)
        magic=0xdeadce
        nr_blocks=12800
        nr_inodes=12824 (istore=229 blocks)
        nr_ifree_blocks=1
        nr_bfree_blocks=1
        nr_free_inodes=12823
        nr_free_blocks=12568
Inode store: wrote 229 blocks
        inode size = 72 B
Ifree blocks: wrote 1 blocks
Bfree blocks: wrote 1 blocks
root@ubuntux86:# mount -o loop -t simplefs test.img test
root@ubuntux86:# echo "Hello World" > test/hello
root@ubuntux86:# cat test/hello
Hello World
root@ubuntux86:# 
root@ubuntux86:# umount test
root@ubuntux86:# rmmod simplefs
root@ubuntux86:# 
```

# sync

```
const struct file_operations simplefs_file_ops = {
    .llseek = generic_file_llseek,
    .owner = THIS_MODULE,
    .read_iter = generic_file_read_iter,
    .write_iter = generic_file_write_iter,
    .mmap       = generic_file_mmap,
    .fsync = generic_file_fsync,
};
    
```


```
[  922.182092] [<ffff000000da2c08>] simplefs_writepage+0x20/0x48 [simplefs]
[  922.188764] [<ffff000008219fc0>] __writepage+0x34/0x80
[  922.193877] [<ffff00000821a974>] write_cache_pages+0x1ec/0x4c4
[  922.199682] [<ffff00000821aca0>] generic_writepages+0x54/0x84
[  922.205401] [<ffff00000821c4ac>] do_writepages+0x74/0x98
[  922.210688] [<ffff00000820c818>] __filemap_fdatawrite_range+0xe0/0x144
[  922.217183] [<ffff00000820cad8>] file_write_and_wait_range+0x74/0xd8
[  922.223508] [<ffff0000082e1eb0>] __generic_file_fsync+0x44/0xe4
[  922.229399] [<ffff0000082e1f90>] generic_file_fsync+0x40/0x68
[  922.235118] [<ffff0000082eddcc>] vfs_fsync_range+0xc8/0xf8
[  922.240580] [<ffff00000825a7dc>] SyS_msync+0x170/0x1d8
```