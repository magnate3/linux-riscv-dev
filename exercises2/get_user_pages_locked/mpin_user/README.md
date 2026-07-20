# mpin_ko
Kernel driver for pinning user-mode memory buffers to physical addresses


```
root@ubuntux86:# make
root@ubuntux86:# make test
root@ubuntux86:# insmod  mpin_user.ko 
root@ubuntux86:# ./mpin_user-test 
mpin_user test all done
root@ubuntux86:# 
```
