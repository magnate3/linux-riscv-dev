## ktaskcmdline

### Note
> ktaskcmdline is a simple kernel module to get program cmd-line arguments when program start in kernel.

### Usage
1. compile
> make
2. add kernel-module ktaskcmdline.ko:
> /sbin/insmod ktaskcmdline.ko
3. show result
> dmesg
4. remove kernel-module
> /sbin/rmmod ktaskcmdline.ko
