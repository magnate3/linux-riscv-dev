# ReserveLinuxMemory
Reserve memory in linux at boot time and then map it to user level program.

# Dependencies
* Linux Kernel 4.0+
* CONFIG_STRICT_DEVMEM=n (if on x86: CONFIG_x86_PAT=n)

# Getting Started 
* Reserve memory at boot time in the kernel. Pass mem=2G memmap=30M\$2G.  mem= sets the kernel to run only within 2G and memmap= requests the kernel to reserve 30M starting at 2G

* Check if memory has been reserved after kernel boots up using cat /proc/iomem. This shows you the physical map and you shoud see a "reserved" section.

* Set the parameters for the driver in simple.c to match the boot parameters.
  The parameters of interest are in simple.c to match the memmap and mem.
  ```
  #define RAW_DATA_SIZE 31457280
  #define RAW_DATA_OFFSET 0x80000000UL
  ```


# Build
* In makefile modify the following lines to set the kernel source to build against and where the module file should be copied.
```
export KERNEL_SOURCE := /Kernel/linux-4.4.0/
PUBLIC_DRIVER_PWD=/modules
```
```
$ make.
````
* If thing went well you will find a .ko file under PUBLIC_DRIVER_PWD/

* To install module
```
$ sudo insmod simple.ko
```
* Check if module is installed. dmesg should return a line with "RawdataStart..".
```
$ dmesg | tail -1
```

# User level program and mmap.
* The driver has been mapped to /sys/kernel/debug/mmap_example. When
mmaping the RAW_DATA_SIZE has to match the parameter set at boot time.
``` C
#define RAW_DATA_SIZE 31457280
open("/sys/kernel/debug/mmap_example", O_RDWR);
 address = (unsigned char*) mmap(NULL, RAW_DATA_SIZE, PROT_READ|PROT_WRITE, MAP_PRIVATE, configfd, 0);
```

# Thanks
The code here was derived from this example on stackoverflow.
http://stackoverflow.com/questions/37919223/map-reserver-memory-at-boot-to-user-space-using-remap-pfn-range
