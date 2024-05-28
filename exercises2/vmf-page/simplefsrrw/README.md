
```
root@ubuntux86:# insmod  simplefsrrw.ko 
root@ubuntux86:# dd if=/dev/zero of=test.img bs=1M count=50
50+0 records in
50+0 records out
52428800 bytes (52 MB, 50 MiB) copied, 0.0260284 s, 2.0 GB/s
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
root@ubuntux86:# mkdir test
root@ubuntux86:# mount -o loop -t simplefs test.img test
mount: /work/kernel_learn/simplefs-rrw/test: unknown filesystem type 'simplefs'.
root@ubuntux86:# dmesg | tail -n 10
[21806.965398] pcieport 0000:00:1d.0: AER: Corrected error received: 0000:02:00.0
[21806.966615] nvme 0000:02:00.0: PCIe Bus Error: severity=Corrected, type=Physical Layer, (Receiver ID)
[21806.966621] nvme 0000:02:00.0:   device [15b7:5006] error status/mask=00000001/0000e000
[21806.966628] nvme 0000:02:00.0:    [ 0] RxErr                 
[23437.435621] pcieport 0000:00:1d.0: AER: Corrected error received: 0000:02:00.0
[23437.436874] nvme 0000:02:00.0: PCIe Bus Error: severity=Corrected, type=Physical Layer, (Receiver ID)
[23437.436880] nvme 0000:02:00.0:   device [15b7:5006] error status/mask=00000001/0000e000
[23437.436886] nvme 0000:02:00.0:    [ 0] RxErr                 
[24692.674494] simplefsrrw: module loaded
[24739.089438] loop15: detected capacity change from 0 to 102400
```


```
root@ubuntux86:# mount -o loop -t simplefsrrw test.img test
root@ubuntux86:# echo "Hello World" > test/hello
root@ubuntux86:# cat test/hello
Hello World
root@ubuntux86:# ls -lR test
test:
total 1
-rw-r--r-- 1 root root 12 5月  28 17:08 hello
root@ubuntux86:# umount test
root@ubuntux86:# rmmod simplefs
simplefs.h         simplefsrrw.ko     simplefsrrw.mod    simplefsrrw.mod.c  simplefsrrw.mod.o  simplefsrrw.o      
root@ubuntux86:# rmmod simplefsrrw.ko
root@ubuntux86:# 
```