
# 生成 initrd.img
[向虚拟文件系统initrd.img中添加驱动](https://www.cnblogs.com/ssslinppp/p/5945537.html)   
```
sudo dracut -v -f image/initrd.img
sudo chmod 777 image/initrd.img
dracut --print-cmdline "root=UUID=437a4991-d1ba-4577-b624-48a51910a100 ro rootfstype=btrfs  rootflags=subvol=root no_timer_check net.ifnames=0 console=tty1 console=ttyS0,115200n8" initrd.img
lsinitrd initramfs-5.14.10-300.fc35.x86_64.img | grep aufs.ko   


```

```
root@ubuntux86:# file initramfs-5.14.10-300.fc35.x86_64.img
initramfs-5.14.10-300.fc35.x86_64.img: gzip compressed data, max compression, from Unix, original size modulo 2^32 38782976
root@ubuntux86:# 
```


```
root@ubuntux86:# file initramfs-5.14.10-300.fc35.x86_64.gz
initramfs-5.14.10-300.fc35.x86_64.gz: ASCII cpio archive (SVR4 with no CRC)
root@ubuntux86:# cpio -ivmd < initramfs-5.14.10-300.fc35.x86_64.gz
root@ubuntux86:# ls
bin  dev  etc  init  initrd  lib  lib32  lib64  libx32  proc  root  run  sbin  shutdown  sys  sysroot  tmp  usr  var
root@ubuntux86:# 
root@ubuntux86:#  find ./* | cpio -o -H newc | gzip > ../initramfs.cpio.gz
```

#  ASCII cpio archive (SVR4 with no CRC)

```
root@ubuntux86:# file initramfs-4.18.0-305.3.1.el8.x86_64.img 
initramfs-4.18.0-305.3.1.el8.x86_64.img: ASCII cpio archive (SVR4 with no CRC)
```

```
root@ubuntux86:# ls
initramfs-4.18.0-305.3.1.el8.x86_64.img  initramfs-5.14.10-300.fc35.x86_64.img
root@ubuntux86:# 
root@ubuntux86:# cpio -ivmd < initramfs-4.18.0-305.3.1.el8.x86_64.img 
.
early_cpio
kernel
kernel/x86
kernel/x86/microcode
kernel/x86/microcode/AuthenticAMD.bin
kernel/x86/microcode/GenuineIntel.bin
9092 blocks
```