 # Operation not permitted
 ```
 [root@centos7 uio]# insmod  uio_test.ko 
insmod: ERROR: could not insert module uio_test.ko: Operation not permitted
 ```
 interrupt 
 ```
  info->irq = UIO_IRQ_CUSTOM;
 ```
 
 
 ```
         info->irq = UIO_IRQ_CUSTOM;
        //info->irq = irq;
        //info->irq_flags = IRQF_SHARED;
 ```
 
 # /dev/uio
 
 ```

[root@centos7 uio]# ls /dev/my_uio_device
ls: cannot access /dev/my_uio_device: No such file or directory
[root@centos7 uio]# ls /dev/uio0
/dev/uio0
[root@centos7 uio]#
 ```
 
 
 #insmod uio_test.ko
 
 ```
 [root@centos7 uio]# rmmod uio_test.ko
[root@centos7 uio]# insmod uio_test.ko
[root@centos7 uio]# dmesg | tail -n 5
 ```
 
 ```
 [root@centos7 uio]# dmesg | tail -n 5
[894259.473417] Failing to register uio device
[894454.250553] Registered UIO handler for IRQ=1
[894981.202589] releasing my uio device
[894981.206149] Un-Registered UIO handler for IRQ=1
[894989.739153] Registered UIO handler for IRQ=1
[root@centos7 uio]# 
 ```