

# amp

[昉·惊鸿-7110上运行异构AMP双系统](https://doc.rvspace.org/VisionFive2/Application_Notes/RT-Thread/])     


## 编译步骤

[编译步骤](https://doc.rvspace.org/VisionFive2/Application_Notes/RT-Thread/VisionFive_2/RT_Thread/configuration.html)   


### Generate Booting SD Card   

[Generate Booting SD Card](https://github.com/starfive-tech/VisionFive2/tree/JH7110_VisionFive2_6.6.y_devel)  

### Build  AMP Image of  SD Card 
```
$ cd buildroot && git checkout --track origin/JH7110_VisionFive2_devel && cd ..
$ cd u-boot && git checkout --track origin/JH7110_VisionFive2_devel && cd ..
$ cd linux && git checkout --track origin/JH7110_VisionFive2_6.6.y_devel && cd ..
$ cd opensbi && git checkout master && cd ..
$ cd soft_3rdpart && git checkout JH7110_VisionFive2_devel && cd 
```
First need to download RT-Thread code:  

```
$ git clone -b amp-5.0.2-devel git@github.com:starfive-tech/rt-thread.git rtthread
```

Then download and prepare the toolchain needed for RT-Thread code:   
```
# For Ubuntu 18.04:
$ wget https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2022.04.12/riscv64-elf-ubuntu-18.04-nightly-2022.04.12-nightly.tar.gz
$ sudo tar xf riscv64-elf-ubuntu-18.04-nightly-2022.04.12-nightly.tar.gz -C /opt/
$ /opt/riscv/bin/riscv64-unknown-elf-gcc --version
riscv64-unknown-elf-gcc (g5964b5cd727) 11.1.0
```






Generate rtthread amp sdcard image:  

```
$ make -j$(nproc)
$ make ampuboot_fit  # build amp uboot image
$ make buildroot_rootfs -j$(nproc)
$ make img
$ make amp_img       # generate sdcard img
```

## details:     
```
root@ubuntux86:# pwd
/proj/VisionFive2
root@ubuntux86:# make -j$(nproc)

To completely erase, reformat, and program a disk sdX, run:
  make DISK=/dev/sdX format-boot-loader
  ... you will need gdisk and e2fsprogs installed
  Please note this will not currently format the SDcard ext4 partition
  This can be done manually if needed

```

```
root@ubuntux86:# ls work/
amp                  buildroot_initramfs_sysroot  initramfs.cpio.gz    linux                opensbi   starfive-visionfive2-vfat.part  u-boot-spl.bin.normal.out  visionfive2_fw_payload.img
buildroot_initramfs  image.fit                    initramfs.cpio.gz.d  module_install_path  spl_tool  u-boot                          version                    vmlinux.bin
root@ubuntux86:# 
```

+ make ampuboot_fit   
```
make[1]: *** No rule to make target '/proj/VisionFive2/work/amp/u-boot.bin', needed by '/proj/VisionFive2/work/amp/opensbi/platform/generic/firmware/fw_payload.o'.  Stop.
make[1]: *** Waiting for unfinished jobs....
 CPP       platform/generic/firmware/fw_jump.elf.ld
make[1]: Leaving directory '/proj/VisionFive2/opensbi'
make: *** [Makefile:415: /proj/VisionFive2/work/amp/opensbi/platform/generic/firmware/fw_payload.bin] Error 2
root@ubuntux86:# ls /proj/VisionFive2/work/amp/u-boot.bin
```

```
git describe --tags --abbrev=0
JH7110_VF2_6.6_v5.13.1
```
执行下述拷贝    
```
cp /proj/VisionFive2/work/amp/u-boot/u-boot.bin /proj/VisionFive2/work/amp/
```

```
/proj/VisionFive2/work/spl_tool/spl_tool -c -f /proj/VisionFive2/work/amp/u-boot/spl/u-boot-spl.bin
ubsplhdr.sofs:0x240, ubsplhdr.bofs:0x200000, ubsplhdr.vers:0x1010101 name:/proj/VisionFive2/work/amp/u-boot/spl/u-boot-spl.bin
SPL written to /proj/VisionFive2/work/amp/u-boot/spl/u-boot-spl.bin.normal.out successfully.
cp /proj/VisionFive2/work/amp/u-boot/spl/u-boot-spl.bin.normal.out /proj/VisionFive2/work/u-boot-amp-spl.bin.normal.out
/proj/VisionFive2/work/amp/u-boot/tools/mkimage -f /proj/VisionFive2/conf/visionfive2-uboot-amp-fit-image.its -A riscv -O u-boot -T firmware /proj/VisionFive2/proj/VisionFive2_fw_payload_amp.img
```
visionfive2_fw_payload_amp.img    
+ make buildroot_rootfs -j$(nproc)   
```
root@ubuntux86:# find ./ -name rootfs.ext4
./work/buildroot_rootfs/images/rootfs.ext4
root@ubuntux86:# find ./ -name rootfs.ext2
./work/buildroot_rootfs/images/rootfs.ext2
root@ubuntux86:#
```

```
root@ubuntux86:# gcc --version
gcc (Ubuntu 8.4.0-3ubuntu2) 8.4.0
Copyright (C) 2018 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

root@ubuntux86:# 
```

```
gcc: error: unrecognized command line option ‘-Waddress-of-packed-member’
gcc: error: unrecognized command line option ‘-Waddress-of-packed-member’
```

```
root@ubuntux86:# update-alternatives --config gcc
There are 2 choices for the alternative gcc (providing /usr/bin/gcc).

  Selection    Path            Priority   Status
------------------------------------------------------------
* 0            /usr/bin/gcc-8   100       auto mode
  1            /usr/bin/gcc-8   100       manual mode
  2            /usr/bin/gcc-9   30        manual mode

Press <enter> to keep the current choice[*], or type selection number: ^C
root@ubuntux86:# 
```
将gcc选择gcc-9解决   


+  make img   

```
/proj/VisionFive2/work/spl_tool/spl_tool -i -f /proj/VisionFive2/work/sdcard.img
IMG  /proj/VisionFive2/work/sdcard.img fixed hdr successfully.
```
+  make amp_img   
```
work/VisionFive2/work/spl_tool/spl_tool -i -f /proj/VisionFive2/work/sdcard_amp.img
IMG  /proj/VisionFive2/work/sdcard_amp.img fixed hdr successfully.
```

## sdcard
+ 删除分区  
```
sdb           8:16   1  59.5G  0 disk 
├─sdb1        8:17   1    16M  0 part 
├─sdb2        8:18   1   100M  0 part 
└─sdb3        8:19   1  59.3G  0 part
```

```
root@ubuntux86:# fdisk /dev/sdb

Welcome to fdisk (util-linux 2.34).
Changes will remain in memory only, until you decide to write them.
Be careful before using the write command.


Command (m for help): d
Partition number (1-3, default 3): 3

Partition 3 has been deleted.

Command (m for help): d
Partition number (1,2, default 2): 2

Partition 2 has been deleted.

Command (m for help): d

Selected partition 1
Partition 1 has been deleted.

Command (m for help): 1
1: unknown command

Command (m for help): w

The partition table has been altered.
Calling ioctl() to re-read partition table.
Syncing disks.

root@ubuntux86:# 
```

```
sda           8:0    0   1.8T  0 disk 
├─sda1        8:1    0   128M  0 part 
└─sda2        8:2    0   1.8T  0 part /work
sdb           8:16   1  59.5G  0 disk
```

+ dd  


```
root@ubuntux86:# dd if=work/sdcard_amp.img of=/dev/sdb bs=4096
135685+0 records in
135685+0 records out
555765760 bytes (556 MB, 530 MiB) copied, 71.1261 s, 7.8 MB/s
```
dd后     
```
sdb           8:16   1  59.5G  0 disk 
├─sdb1        8:17   1     2M  0 part 
├─sdb2        8:18   1     4M  0 part 
├─sdb3        8:19   1    22M  0 part 
└─sdb4        8:20   1   500M  0 part
```
extend   sdb4    
```
growpart /dev/sdb 4
e2fsck -f /dev/sdb4
resize2fs /dev/sdb4
fsck.ext4 /dev/sdb4
```

```
sdb           8:16   1  59.5G  0 disk 
├─sdb1        8:17   1     2M  0 part 
├─sdb2        8:18   1     4M  0 part 
├─sdb3        8:19   1    22M  0 part 
└─sdb4        8:20   1  59.4G  0 part
```

## test


```
# ./rpmsg_echo 
Sending message #0: hello there 0!
Receiving message #0: test this time 1029250 ns, avg time 1029250 ns, maxtime 1029250 ns
Sending message #1: hello there 1!
Receiving message #1: test this time 173500 ns, avg time 601375 ns, maxtime 1029250 ns
Sending message #2: hello there 2!
Receiving message #2: test this time 166500 ns, avg time 456416 ns, maxtime 1029250 ns
Sending message #3: hello there 3!
Receiving message #3: test this time 109000 ns, avg time 369562 ns, maxtime 1029250 ns
Sending message #4: hello there 4!
Receiving message #4: test this time 162500 ns, avg time 328150 ns, maxtime 1029250 ns
Sending message #5: hello there 5!
Receiving message #5: test this time 191000 ns, avg time 305291 ns, maxtime 1029250 ns
^C
# 
```