ramfs-loadable-module
=====================

This is a modification of the ramfs code in the Linux kernel to run as a loadable
module. Additionally, it adds some explanatory comments.

# How to use this module

Build the module by running `make`.
Like all kernel modules, you will need to have the appropriate kernel headers installed.
Insert the module with `sudo insmod ramfs.ko`
and mount a directory with `sudo mount -t myramfs none /path/to/dir`. Note that ramfs doesn't use an actual block device. 
To unload the module use `sudo rmmod ramfs.ko`. Keep in mind to unmount any ramfs directories
before unloading.