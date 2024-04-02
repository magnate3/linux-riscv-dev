
# minif

[minfs](https://linux-kernel-labs.github.io/refs/heads/master/so2/lab9-filesystems-part2.html)   

```
[root@centos7 user]# dd if=/dev/zero of=diska.img bs=512 count=2
2+0 records in
2+0 records out
1024 bytes (1.0 kB) copied, 0.000127252 s, 8.0 MB/s
[root@centos7 user]# insmod ../kernel/minfs.ko 
[root@centos7 user]# ./mkfs.minfs diska.img
[root@centos7 user]# mount -t minfs diska.img /mnt/myfs
mount: /root/programming/kernel/myfs/minfs/user/diska.img is not a block device (maybe try `-o loop'?)
[root@centos7 user]# 
```

解决  
```
[root@centos7 user]# losetup /dev/loop0 diska.img 
[root@centos7 user]# ./mkfs.minfs diska.img
[root@centos7 user]# mount -t minfs /dev/loop0 /mnt/myfs
[root@centos7 user]# 
```


```
[root@centos7 user]# umount /mnt/myfs
[root@centos7 user]# losetup -d /dev/loop0
[root@centos7 user]# 
```

## bad superblock


```
[root@centos7 user]# dd if=/dev/zero of=diska.img bs=4096 count=1
1+0 records in
1+0 records out
4096 bytes (4.1 kB) copied, 0.000112653 s, 36.4 MB/s
[root@centos7 user]# losetup /dev/loop0 diska.img 
[root@centos7 user]# ./mkfs.minfs diska.img
[root@centos7 user]# mount -t minfs /dev/loop0 /mnt/myfs
mount: wrong fs type, bad option, bad superblock on /dev/loop0,
       missing codepage or helper program, or other error

       In some cases useful info is found in syslog - try
       dmesg | tail or so.
[root@centos7 user]# mount -t minfs /dev/loop0 /mnt/myfs
```


```
[root@centos7 user]# dd if=/dev/zero of=diska.img bs=4096 count=2
2+0 records in
2+0 records out
8192 bytes (8.2 kB) copied, 0.000124582 s, 65.8 MB/s
[root@centos7 user]# losetup /dev/loop0 diska.img 
[root@centos7 user]# mke2fs /dev/loopp0
mke2fs 1.42.9 (28-Dec-2013)
Could not stat /dev/loopp0 --- No such file or directory

The device apparently does not exist; did you specify it correctly?
[root@centos7 user]# mke2fs /dev/loop0 
mke2fs 1.42.9 (28-Dec-2013)
mke2fs: inode_size (128) * inodes_count (0) too big for a
        filesystem with 0 blocks, specify higher inode_ratio (-i)
        or lower inode count (-N).

[root@centos7 user]# ./mkfs.minfs diska.img
[root@centos7 user]# mount /dev/loop0 /mnt/myfs/
[root@centos7 user]# ls /mnt/myfs/
ls: reading directory /mnt/myfs/: Cannot allocate memory
[root@centos7 user]# ls /mnt/myfs
ls: reading directory /mnt/myfs: Cannot allocate memory
[root@centos7 user]# dmesg | tail -n 10
[ 1919.903270] Read a.txt from folder /, ctx->pos: 0
[ 1933.782193] mode is 40755; data_block is 2
[ 1933.782201] wrote inode 0
[ 1951.431913] writing at sector 8, 8 sectors
[ 1951.436017] released superblock resources
[ 1951.436021] writing at sector 0, 8 sectors
[ 2050.734379] module successfully unloaded, Major No. = 253
[ 2290.596102] isofs_fill_super: bread failed, dev=loop0, iso_blknum=16, block=32
[ 2295.292225] could not read block
[ 2302.231169] could not read block
[root@centos7 user]# 
```

## run successfully


```
[root@centos7 ramdisk-lab]#  insmod   ramdisk_test.ko
[root@centos7 ramdisk-lab]# cd ..
[root@centos7 user]# ls
diska.img  Makefile  mkfs.minfs  mkfs.minfs.c  ramdisk-lab  ramdisk-lab.zip  sb.txt  test-minfs-0.sh  test-minfs-1.sh  test-minfs-2.sh  test-minfs.sh
[root@centos7 user]# ./mkfs.minfs /dev/myramdisk 
[root@centos7 user]# mount /dev/myramdisk /mnt/myfs/
[root@centos7 user]# 
root@centos7 user]# ls /mnt/myfs/
ls: cannot access /mnt/myfs/a.txt: No such file or directory
a.txt
```

# reference

