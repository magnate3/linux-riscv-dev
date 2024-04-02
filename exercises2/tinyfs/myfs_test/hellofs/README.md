

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