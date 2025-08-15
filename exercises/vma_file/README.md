

# ./mmap_fork

***after fork child***

#  insmod  vma_file_test.ko pid=42959

```
[root@centos7 vma_file]# ps -elf | grep  mmap_fork
0 S root      42959  41007  0  80   0 -  4138 n_tty_ 22:09 pts/1    00:00:00 ./mmap_fork
1 S root      42973  42959  0  80   0 -  4138 wait_w 22:11 pts/1    00:00:00 ./mmap_fork
0 S root      43314  39424  0  80   0 -  1729 pipe_w 22:17 pts/0    00:00:00 grep --color=auto mmap_fork
[root@centos7 vma_file]# insmod  vma_file_test.ko pid=42959
[root@centos7 vma_file]# dmesg | tail -n 30
[139839.850150] ffff77240000 - ffff77250000 rw-p ffff77240000 
[139839.855701] ffff77250000 - ffff87250000 rw-s 000000000000 00:05   03059576   /dev/zero (deleted)
[139839.864542] ffff87250000 - ffff873c0000 r-xp 000000000000 08:03   00050677   /usr/lib64/libc-2.17.so
[139839.873730] ffff873c0000 - ffff873d0000 r--p 000000160000 08:03   00050677   /usr/lib64/libc-2.17.so
[139839.882915] ffff873d0000 - ffff873e0000 rw-p 000000170000 08:03   00050677   /usr/lib64/libc-2.17.so
[139839.892103] ffff873e0000 - ffff873f0000 rw-p ffff873e0000 
[139839.897653] ffff873f0000 - ffff87400000 r--p 000000000000 
[139839.903207] ffff87400000 - ffff87410000 r-xp 000000000000 
[139839.908757] ffff87410000 - ffff87430000 r-xp 000000000000 08:03   00050670   /usr/lib64/ld-2.17.so
[139839.917767] ffff87430000 - ffff87440000 r--p 000000010000 08:03   00050670   /usr/lib64/ld-2.17.so
[139839.926780] ffff87440000 - ffff87450000 rw-p 000000020000 08:03   00050670   /usr/lib64/ld-2.17.so
[139839.935793] ffffe2930000 - ffffe2960000 rw-p fffffffd0000                     [ stack ]
[140224.123927] View Process [mmap_fork]
[140224.127579] Address                     Mode Offset       dev_t   inode       Mapping/Type
[140224.135903] 00400000 - 00410000 vm_flags: r-xp virtual address offset: 000000000000 <MAJOR:MINOR>08:03   <inode no>289718553   file path: /root/programming/kernel/vma_file/mmap_fork
[140224.152094] 00410000 - 00420000 vm_flags: r--p virtual address offset: 000000000000 <MAJOR:MINOR>08:03   <inode no>289718553   file path: /root/programming/kernel/vma_file/mmap_fork
[140224.168284] 00420000 - 00430000 vm_flags: rw-p virtual address offset: 000000010000 <MAJOR:MINOR>08:03   <inode no>289718553   file path: /root/programming/kernel/vma_file/mmap_fork
[140224.184469] 137c0000 - 137f0000 vm_flags: rw-p virtual address offset: 0000137c0000                     [ heap ]
[140224.194689] ffff77240000 - ffff77250000 vm_flags: rw-p virtual address offset: ffff77240000 
[140224.203180] ffff77250000 - ffff87250000 vm_flags: rw-s virtual address offset: 000000000000 <MAJOR:MINOR>00:05   <inode no>03059576   file path: /dev/zero (deleted)
[140224.217896] ffff87250000 - ffff873c0000 vm_flags: r-xp virtual address offset: 000000000000 <MAJOR:MINOR>08:03   <inode no>00050677   file path: /usr/lib64/libc-2.17.so
[140224.232961] ffff873c0000 - ffff873d0000 vm_flags: r--p virtual address offset: 000000160000 <MAJOR:MINOR>08:03   <inode no>00050677   file path: /usr/lib64/libc-2.17.so
[140224.248022] ffff873d0000 - ffff873e0000 vm_flags: rw-p virtual address offset: 000000170000 <MAJOR:MINOR>08:03   <inode no>00050677   file path: /usr/lib64/libc-2.17.so
[140224.263086] ffff873e0000 - ffff873f0000 vm_flags: rw-p virtual address offset: ffff873e0000 
[140224.271573] ffff873f0000 - ffff87400000 vm_flags: r--p virtual address offset: 000000000000 
[140224.280063] ffff87400000 - ffff87410000 vm_flags: r-xp virtual address offset: 000000000000 
[140224.288556] ffff87410000 - ffff87430000 vm_flags: r-xp virtual address offset: 000000000000 <MAJOR:MINOR>08:03   <inode no>00050670   file path: /usr/lib64/ld-2.17.so
[140224.303443] ffff87430000 - ffff87440000 vm_flags: r--p virtual address offset: 000000010000 <MAJOR:MINOR>08:03   <inode no>00050670   file path: /usr/lib64/ld-2.17.so
[140224.318333] ffff87440000 - ffff87450000 vm_flags: rw-p virtual address offset: 000000020000 <MAJOR:MINOR>08:03   <inode no>00050670   file path: /usr/lib64/ld-2.17.so
[140224.333221] ffffe2930000 - ffffe2960000 vm_flags: rw-p virtual address offset: fffffffd0000                     [ stack ]
[root@centos7 vma_file]# ls -al mmap_fork
-rwxr-xr-x 1 root root 71512 Aug  2 22:09 mmap_fork
```