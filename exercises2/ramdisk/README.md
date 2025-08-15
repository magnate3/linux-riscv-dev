编译内核的最后一步执行make install，然后执行mkinitramfs


```
root@ubuntux86:# ls k_install/
config-6.3.9  System.map-6.3.9
root@ubuntux86:# mkinitramfs -o ramdisk.img
root@ubuntux86:# du -sh ramdisk.img 
62M     ramdisk.img
root@ubuntux86:#
```