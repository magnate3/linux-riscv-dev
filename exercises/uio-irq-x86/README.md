 # insmod uio_test.ko 
 
 ```
 root@ubuntux86:/work/uio# modprobe uio
root@ubuntux86:/work/uio# insmod uio_test.ko 
root@ubuntux86:/work/uio# 
 ```
 
 ```
 root@ubuntux86:/work/uio# dmesg | grep 'Registered UIO'
[31970.997406] Registered UIO handler for IRQ=2
root@ubuntux86:/work/uio# 
 ```
 
 
 # gcc uio_user.c  -o uio_user
 ```
root@ubuntux86:/work/uio# gcc uio_user.c  -o uio_user
root@ubuntux86:/work/uio# ./uio_user 
Started uio test driver.
root@ubuntux86:/work/uio# 
 ```