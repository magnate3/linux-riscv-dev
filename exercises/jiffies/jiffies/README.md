# Jiffies Linux Kernal Modules
## Jiffies module 
This module can be loaded to the Linux kernel and work also as a Linux process. After loading the module into the kernel if you call the process it shows the number of interrupts that had been occurred into the CPU since the start-up of the system!
## Second one
This is basically like the first one but a little different in runtime, when you call the process it shows the time (as seconds) since the module was loaded into the kernel.
## How to use 
You have to run the below command to execute ```Makefile```:
```
make
```
If it executes correctly then you can see the filename.ko beside the main C file like ```jiffies.c``` and some other files. 
Then you have to load the module to the kernel by:
```
sudo insmod filename.ko
```
It's important to use the correct format (.ko not .c).
It's possible to see what will happen when you call the process by:
```
cat /proc/module  
``` 
